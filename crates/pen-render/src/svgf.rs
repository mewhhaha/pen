//! Spatiotemporal Variance-Guided Filter (SVGF, Schied et al., HPG 2017)
//! for real-time denoising of ray-traced shadows, RTAO, reflections, and diffuse GI.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "GPU dispatch and texture dimensions")]

use anyhow::{Result, ensure};

pub const SVGF_WGSL: &str = crate::wgsl!("svgf.wgsl");

/// Tuning parameters for the SVGF temporal accumulation and À-Trous edge-stopping passes.
#[derive(Clone, Copy, Debug)]
pub struct SvgfParams {
    /// Minimum temporal blend alpha once history matures (default `0.05`).
    pub min_alpha: f32,
    /// Maximum accumulated frame history count (default `32.0`).
    pub max_history: f32,
    /// Relative depth disocclusion rejection threshold (default `0.05`).
    pub depth_threshold: f32,
    /// Minimum normal dot product for temporal acceptance (default `0.90`).
    pub normal_threshold: f32,
    /// Luminance variance edge-stopping factor `phi_l` (default `4.0`).
    pub phi_color: f32,
    /// Normal exponent edge-stopping factor `phi_n` (default `128.0`).
    pub phi_normal: f32,
    /// Depth gradient edge-stopping factor `phi_z` (default `1.0`).
    pub phi_depth: f32,
    /// Variance boost multiplier for newly disoccluded pixels (default `4.0`).
    pub variance_boost: f32,
    /// Number of À-Trous wavelet iterations (`1..=5`, default `4`).
    pub atrous_iterations: u32,
    /// Force-discard temporal history on camera cuts / teleports.
    pub reset_history: bool,
    /// Specular reflection denoising mode: reprojects via virtual hit-point parallax motion (`motion.zw`),
    /// applies 3x3 YCoCg temporal neighborhood clamping to suppress mirror ghosting, and scales
    /// A-Trous edge-stopping weights by surface roughness (`normal.w`).
    pub specular_mode: bool,
    /// In specular mode, use noisy signal alpha (hit distance in meters) to guide filtering.
    pub hit_distance_adaptive: bool,
}

impl Default for SvgfParams {
    fn default() -> Self {
        Self {
            min_alpha: 0.05,
            max_history: 32.0,
            depth_threshold: 0.05,
            normal_threshold: 0.90,
            phi_color: 4.0,
            phi_normal: 128.0,
            phi_depth: 1.0,
            variance_boost: 4.0,
            atrous_iterations: 4,
            reset_history: false,
            specular_mode: false,
            hit_distance_adaptive: false,
        }
    }
}

/// Input texture views for one frame of SVGF denoising.
pub struct SvgfInputs<'a> {
    pub noisy_signal: &'a wgpu::TextureView,
    pub depth: &'a wgpu::TextureView,
    pub prev_depth: &'a wgpu::TextureView,
    pub normal: &'a wgpu::TextureView,
    pub prev_normal: &'a wgpu::TextureView,
    pub motion: &'a wgpu::TextureView,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SvgfUniform {
    dims: [u32; 4],
    temporal_params: [f32; 4],
    atrous_params: [f32; 4],
}

struct Target {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Target {
    fn new(device: &wgpu::Device, label: &str, size: (u32, u32)) -> Self {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        Self { tex, view }
    }
}

struct SvgfState {
    size: (u32, u32),
    color_bufs: [Target; 3],
    moments: [Target; 2],
    history_color_idx: usize,
    output_idx: usize,
    moments_idx: usize,
    has_history: bool,
    /// Cached temporal bind groups indexed by `hist_c * 2 + prev_m`, keyed on external input views.
    temporal_bgs: [Option<([wgpu::TextureView; 6], wgpu::BindGroup)>; 6],
    /// Cached A-Trous bind groups indexed by `(src_idx * 3 + dst_idx) * 2 + curr_m`, keyed on `(depth, normal)`.
    atrous_bgs: [Option<([wgpu::TextureView; 2], wgpu::BindGroup)>; 18],
}

/// Stateful GPU SVGF spatiotemporal denoiser.
pub struct SvgfDenoiser {
    temporal_pipe: wgpu::ComputePipeline,
    temporal_bgl: wgpu::BindGroupLayout,
    atrous_pipe: wgpu::ComputePipeline,
    atrous_bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    uniform_stride: u32,
    staged: Vec<u8>,
    state: Option<Box<SvgfState>>,
}

fn tex_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_wo(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn encode_copy(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::Buffer, bytes: &[u8]) {
    crate::staging::encode_copy(device, encoder, dst, 0, bytes);
}

impl SvgfDenoiser {
    pub fn new(device: &wgpu::Device) -> Self {
        let uniform_size = std::mem::size_of::<SvgfUniform>() as u32;
        let align = device.limits().min_uniform_buffer_offset_alignment.max(16);
        let uniform_stride = uniform_size.next_multiple_of(align);
        let u_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(u64::from(uniform_size)),
            },
            count: None,
        };
        let temporal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("svgf_temporal_bgl"),
            entries: &[
                u_entry,
                tex_ro(1),
                tex_ro(2),
                tex_ro(3),
                tex_ro(4),
                tex_ro(5),
                tex_ro(6),
                tex_ro(7),
                tex_ro(8),
                storage_wo(9),
                storage_wo(10),
            ],
        });
        let atrous_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("svgf_atrous_bgl"),
            entries: &[u_entry, tex_ro(1), tex_ro(2), tex_ro(3), storage_wo(4), tex_ro(5)],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("svgf_wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::deferred_wgsl!("svgf.wgsl").into()),
        });
        let temporal_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("svgf_temporal_pl"),
            bind_group_layouts: &[Some(&temporal_bgl)],
            immediate_size: 0,
        });
        let atrous_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("svgf_atrous_pl"),
            bind_group_layouts: &[Some(&atrous_bgl)],
            immediate_size: 0,
        });
        let temporal_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("svgf_temporal_pipe"),
            layout: Some(&temporal_pl),
            module: &module,
            entry_point: Some("svgf_temporal"),
            compilation_options: Default::default(),
            cache: crate::pipeline_cache::active().as_ref(),
        });
        let atrous_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("svgf_atrous_pipe"),
            layout: Some(&atrous_pl),
            module: &module,
            entry_point: Some("svgf_atrous"),
            compilation_options: Default::default(),
            cache: crate::pipeline_cache::active().as_ref(),
        });
        // 6 dynamic uniform slots: 1 temporal + up to 5 A-Trous iterations uploaded in one copy.
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("svgf_u"),
            size: u64::from(uniform_stride) * 6,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            temporal_pipe,
            temporal_bgl,
            atrous_pipe,
            atrous_bgl,
            uniform,
            uniform_stride,
            staged: Vec::new(),
            state: None,
        }
    }

    /// Invalidates temporal history on the next call to [`denoise`].
    pub fn reset_history(&mut self) {
        if let Some(st) = &mut self.state {
            st.has_history = false;
        }
    }

    pub fn output_texture(&self) -> Option<&wgpu::Texture> {
        self.state.as_ref().map(|s| &s.color_bufs[s.output_idx].tex)
    }

    pub fn output_view(&self) -> Option<&wgpu::TextureView> {
        self.state.as_ref().map(|s| &s.color_bufs[s.output_idx].view)
    }

    /// Runs temporal accumulation + multi-pass À-Trous edge-stopping filtering.
    pub fn denoise(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: &SvgfInputs<'_>,
        size: (u32, u32),
        params: &SvgfParams,
    ) -> Result<()> {
        let max_dim = device.limits().max_texture_dimension_2d;
        ensure!(size.0 >= 1 && size.1 >= 1 && size.0 <= max_dim && size.1 <= max_dim, "invalid SVGF size {size:?}");
        for (name, view) in [
            ("noisy_signal", inputs.noisy_signal),
            ("depth", inputs.depth),
            ("prev_depth", inputs.prev_depth),
            ("normal", inputs.normal),
            ("prev_normal", inputs.prev_normal),
            ("motion", inputs.motion),
        ] {
            let sz = view.texture().size();
            ensure!((sz.width, sz.height) == size, "SVGF {name} extent {}x{} != {size:?}", sz.width, sz.height);
        }

        let need_alloc = !matches!(&self.state, Some(st) if st.size == size);
        if need_alloc {
            self.state = Some(Box::new(SvgfState {
                size,
                color_bufs: [
                    Target::new(device, "svgf_color_0", size),
                    Target::new(device, "svgf_color_1", size),
                    Target::new(device, "svgf_color_2", size),
                ],
                moments: [Target::new(device, "svgf_moments_0", size), Target::new(device, "svgf_moments_1", size)],
                history_color_idx: 0,
                output_idx: 1,
                moments_idx: 0,
                has_history: false,
                temporal_bgs: std::array::from_fn(|_| None),
                atrous_bgs: std::array::from_fn(|_| None),
            }));
        }
        let st = self.state.as_mut().unwrap();
        let reset = params.reset_history || !st.has_history;
        let spec_flags = (u32::from(params.specular_mode) << 1)
            | (u32::from(params.specular_mode && params.hit_distance_adaptive) << 2);
        let iters = params.atrous_iterations.clamp(1, 5);

        let u_temporal = SvgfUniform {
            dims: [size.0, size.1, 1, u32::from(reset) | spec_flags],
            temporal_params: [
                params.min_alpha.clamp(0.01, 1.0),
                params.max_history.clamp(1.0, 256.0),
                params.depth_threshold.max(1e-4),
                params.normal_threshold.clamp(-1.0, 1.0),
            ],
            atrous_params: [
                params.phi_color.max(0.05),
                params.phi_normal.max(1.0),
                params.phi_depth.max(1e-4),
                params.variance_boost.max(1.0),
            ],
        };
        let stride = self.uniform_stride as usize;
        let u_size = std::mem::size_of::<SvgfUniform>();
        let slots = 1 + iters as usize;
        self.staged.clear();
        self.staged.resize(slots * stride, 0);
        self.staged[..u_size].copy_from_slice(bytemuck::bytes_of(&u_temporal));
        for iter in 0..iters {
            let step = 1u32 << iter;
            let u_iter = SvgfUniform { dims: [size.0, size.1, step, spec_flags], ..u_temporal };
            let off = (1 + iter as usize) * stride;
            self.staged[off..off + u_size].copy_from_slice(bytemuck::bytes_of(&u_iter));
        }
        encode_copy(device, encoder, &self.uniform, &self.staged);

        let hist_c = st.history_color_idx;
        let slot_a = (hist_c + 1) % 3;
        let slot_b = (hist_c + 2) % 3;
        let prev_m = st.moments_idx;
        let curr_m = 1 - prev_m;

        let u_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.uniform,
            offset: 0,
            size: wgpu::BufferSize::new(u_size as u64),
        });
        let t_slot = hist_c * 2 + prev_m;
        let t_hit = st.temporal_bgs[t_slot].as_ref().is_some_and(|(keys, _)| {
            &keys[0] == inputs.noisy_signal
                && &keys[1] == inputs.depth
                && &keys[2] == inputs.prev_depth
                && &keys[3] == inputs.normal
                && &keys[4] == inputs.prev_normal
                && &keys[5] == inputs.motion
        });
        if !t_hit {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("svgf_temporal_bg"),
                layout: &self.temporal_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u_binding.clone() },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(inputs.noisy_signal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&st.color_bufs[hist_c].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&st.moments[prev_m].view),
                    },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(inputs.depth) },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(inputs.prev_depth),
                    },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(inputs.normal) },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(inputs.prev_normal),
                    },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(inputs.motion) },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::TextureView(&st.color_bufs[slot_a].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::TextureView(&st.moments[curr_m].view),
                    },
                ],
            });
            st.temporal_bgs[t_slot] = Some((
                [
                    inputs.noisy_signal.clone(),
                    inputs.depth.clone(),
                    inputs.prev_depth.clone(),
                    inputs.normal.clone(),
                    inputs.prev_normal.clone(),
                    inputs.motion.clone(),
                ],
                bg,
            ));
        }
        let tbg = &st.temporal_bgs[t_slot].as_ref().unwrap().1;
        let (gx, gy) = (size.0.div_ceil(8), size.1.div_ceil(8));
        {
            let ts = crate::renderer::timing::scope(crate::renderer::timing::Span::Scene);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("svgf_temporal"),
                timestamp_writes: ts.compute(),
            });
            pass.set_pipeline(&self.temporal_pipe);
            pass.set_bind_group(0, tbg, &[0]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Iter 0 writes `slot_b` which stays preserved as next frame's temporal history color
        // (SVGF Section 4.1) with zero texture copy; subsequent iterations ping-pong `slot_a` <-> `hist_c`.
        let mut src_idx = slot_a;
        let mut dst_idx = slot_b;
        for iter in 0..iters {
            let a_slot = (src_idx * 3 + dst_idx) * 2 + curr_m;
            let a_hit = st.atrous_bgs[a_slot]
                .as_ref()
                .is_some_and(|(keys, _)| &keys[0] == inputs.depth && &keys[1] == inputs.normal);
            if !a_hit {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("svgf_atrous_bg"),
                    layout: &self.atrous_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: u_binding.clone() },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&st.color_bufs[src_idx].view),
                        },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(inputs.depth) },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(inputs.normal),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&st.color_bufs[dst_idx].view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&st.moments[curr_m].view),
                        },
                    ],
                });
                st.atrous_bgs[a_slot] = Some(([inputs.depth.clone(), inputs.normal.clone()], bg));
            }
            let abg = &st.atrous_bgs[a_slot].as_ref().unwrap().1;
            let dyn_off = (1 + iter) * self.uniform_stride;
            {
                let ts = crate::renderer::timing::scope(crate::renderer::timing::Span::Scene);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("svgf_atrous"),
                    timestamp_writes: ts.compute(),
                });
                pass.set_pipeline(&self.atrous_pipe);
                pass.set_bind_group(0, abg, &[dyn_off]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            if iter + 1 < iters {
                if iter == 0 {
                    // Keep `slot_b` untouched for next frame's history; read `slot_b` -> write `slot_a`.
                    src_idx = slot_b;
                    dst_idx = slot_a;
                } else {
                    // Ping-pong between `slot_a` and `hist_c` (`slot_b` stays untouched).
                    src_idx = dst_idx;
                    dst_idx = if dst_idx == slot_a { hist_c } else { slot_a };
                }
            }
        }
        st.output_idx = dst_idx;
        st.history_color_idx = slot_b;
        st.moments_idx = curr_m;
        st.has_history = true;
        Ok(())
    }
}

/// Per-channel configuration for [`SvgfMultiDenoiser`] (up to 3 diffuse channels, e.g. SAO/GI/DI).
#[derive(Clone, Copy)]
pub struct SvgfMultiChannel<'a> {
    pub noisy_signal: &'a wgpu::TextureView,
    pub min_alpha: f32,
    pub phi_color: f32,
    pub atrous_iterations: u32,
    pub reset_history: bool,
}

/// Shared geometric inputs and up to 3 diffuse channels for [`SvgfMultiDenoiser::denoise`].
pub struct SvgfMultiInputs<'a> {
    pub channels: [Option<SvgfMultiChannel<'a>>; 3],
    pub depth: &'a wgpu::TextureView,
    pub prev_depth: &'a wgpu::TextureView,
    pub normal: &'a wgpu::TextureView,
    pub prev_normal: &'a wgpu::TextureView,
    pub motion: &'a wgpu::TextureView,
    pub max_history: f32,
    pub depth_threshold: f32,
    pub normal_threshold: f32,
    pub phi_normal: f32,
    pub phi_depth: f32,
    pub variance_boost: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SvgfMultiUniform {
    dims: [u32; 4],
    temporal_params: [f32; 4],
    min_alphas: [f32; 4],
    phi_colors: [f32; 4],
    edge_params: [f32; 4],
}

struct ArrayTarget {
    _tex: wgpu::Texture,
    array_view: wgpu::TextureView,
    layer_views: [wgpu::TextureView; 3],
}

impl ArrayTarget {
    fn new(device: &wgpu::Device, label: &str, size: (u32, u32)) -> Self {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 3 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let array_view = tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_array_layer: 0,
            array_layer_count: Some(3),
            ..Default::default()
        });
        let layer_views = std::array::from_fn(|layer| {
            tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });
        Self { _tex: tex, array_view, layer_views }
    }
}

struct SvgfMultiState {
    size: (u32, u32),
    color_bufs: [ArrayTarget; 3],
    moments: [ArrayTarget; 2],
    history_color_idx: usize,
    output_idx: [usize; 3],
    moments_idx: usize,
    channel_has_history: [bool; 3],
    temporal_bgs: [Option<([wgpu::TextureView; 8], wgpu::BindGroup)>; 6],
    atrous_bgs: [Option<([wgpu::TextureView; 2], wgpu::BindGroup)>; 9],
}

fn tex_array_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_array_wo(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: wgpu::TextureViewDimension::D2Array,
        },
        count: None,
    }
}

/// Fused multi-channel GPU SVGF denoiser for up to 3 diffuse/non-specular signals
/// (SAO, GI, DI) sharing geometry fetches, disocclusion checks, and A-Trous edge weights.
pub struct SvgfMultiDenoiser {
    temporal_pipe: wgpu::ComputePipeline,
    temporal_bgl: wgpu::BindGroupLayout,
    atrous_pipe: wgpu::ComputePipeline,
    atrous_bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    uniform_stride: u32,
    staged: Vec<u8>,
    state: Option<Box<SvgfMultiState>>,
}

impl SvgfMultiDenoiser {
    pub fn new(device: &wgpu::Device) -> Self {
        let uniform_size = std::mem::size_of::<SvgfMultiUniform>() as u32;
        let align = device.limits().min_uniform_buffer_offset_alignment.max(16);
        let uniform_stride = uniform_size.next_multiple_of(align);
        let u_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(u64::from(uniform_size)),
            },
            count: None,
        };
        let temporal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("svgf_multi_temporal_bgl"),
            entries: &[
                u_entry,
                tex_ro(1),
                tex_ro(2),
                tex_ro(3),
                tex_array_ro(4),
                tex_array_ro(5),
                tex_ro(6),
                tex_ro(7),
                tex_ro(8),
                tex_ro(9),
                tex_ro(10),
                storage_array_wo(11),
                storage_array_wo(12),
            ],
        });
        let atrous_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("svgf_multi_atrous_bgl"),
            entries: &[u_entry, tex_array_ro(1), tex_ro(2), tex_ro(3), storage_array_wo(4)],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("svgf_multi_wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::deferred_wgsl!("svgf.wgsl").into()),
        });
        let temporal_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("svgf_multi_temporal_pl"),
            bind_group_layouts: &[Some(&temporal_bgl)],
            immediate_size: 0,
        });
        let atrous_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("svgf_multi_atrous_pl"),
            bind_group_layouts: &[Some(&atrous_bgl)],
            immediate_size: 0,
        });
        let temporal_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("svgf_multi_temporal_pipe"),
            layout: Some(&temporal_pl),
            module: &module,
            entry_point: Some("svgf_temporal_multi"),
            compilation_options: Default::default(),
            cache: crate::pipeline_cache::active().as_ref(),
        });
        let atrous_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("svgf_multi_atrous_pipe"),
            layout: Some(&atrous_pl),
            module: &module,
            entry_point: Some("svgf_atrous_multi"),
            compilation_options: Default::default(),
            cache: crate::pipeline_cache::active().as_ref(),
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("svgf_multi_u"),
            size: u64::from(uniform_stride) * 6,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            temporal_pipe,
            temporal_bgl,
            atrous_pipe,
            atrous_bgl,
            uniform,
            uniform_stride,
            staged: Vec::new(),
            state: None,
        }
    }

    pub fn reset_history(&mut self) {
        if let Some(st) = &mut self.state {
            st.channel_has_history = [false; 3];
        }
    }

    pub fn output_view(&self, channel: usize) -> Option<&wgpu::TextureView> {
        let st = self.state.as_ref()?;
        let buf_idx = *st.output_idx.get(channel)?;
        st.color_bufs[buf_idx].layer_views.get(channel)
    }

    #[cfg(test)]
    fn output_texture(&self, channel: usize) -> Option<(&wgpu::Texture, u32)> {
        let st = self.state.as_ref()?;
        let buf_idx = *st.output_idx.get(channel)?;
        Some((&st.color_bufs[buf_idx]._tex, channel as u32))
    }

    pub fn denoise(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: &SvgfMultiInputs<'_>,
        size: (u32, u32),
    ) -> Result<()> {
        let max_dim = device.limits().max_texture_dimension_2d;
        ensure!(size.0 >= 1 && size.1 >= 1 && size.0 <= max_dim && size.1 <= max_dim, "invalid SVGF size {size:?}");
        for (name, view) in [
            ("depth", inputs.depth),
            ("prev_depth", inputs.prev_depth),
            ("normal", inputs.normal),
            ("prev_normal", inputs.prev_normal),
            ("motion", inputs.motion),
        ] {
            let sz = view.texture().size();
            ensure!((sz.width, sz.height) == size, "SVGF {name} extent {}x{} != {size:?}", sz.width, sz.height);
        }
        let mut active_mask = 0u32;
        let mut max_iters = 0u32;
        let mut first_noisy: Option<&wgpu::TextureView> = None;
        for (ch, opt) in inputs.channels.iter().enumerate() {
            if let Some(cfg) = opt {
                let sz = cfg.noisy_signal.texture().size();
                ensure!(
                    (sz.width, sz.height) == size,
                    "SVGF channel {ch} extent {}x{} != {size:?}",
                    sz.width,
                    sz.height
                );
                active_mask |= 1u32 << ch;
                max_iters = max_iters.max(cfg.atrous_iterations.clamp(1, 5));
                if first_noisy.is_none() {
                    first_noisy = Some(cfg.noisy_signal);
                }
            }
        }
        let Some(fallback_noisy) = first_noisy else { return Ok(()) };

        if !matches!(&self.state, Some(st) if st.size == size) {
            self.state = Some(Box::new(SvgfMultiState {
                size,
                color_bufs: [
                    ArrayTarget::new(device, "svgf_multi_color_0", size),
                    ArrayTarget::new(device, "svgf_multi_color_1", size),
                    ArrayTarget::new(device, "svgf_multi_color_2", size),
                ],
                moments: [
                    ArrayTarget::new(device, "svgf_multi_moments_0", size),
                    ArrayTarget::new(device, "svgf_multi_moments_1", size),
                ],
                history_color_idx: 0,
                output_idx: [1, 1, 1],
                moments_idx: 0,
                channel_has_history: [false; 3],
                temporal_bgs: std::array::from_fn(|_| None),
                atrous_bgs: std::array::from_fn(|_| None),
            }));
        }
        let st = self.state.as_mut().unwrap();

        let mut reset_mask = 0u32;
        let mut min_alphas = [0.05f32; 4];
        let mut phi_colors = [4.0f32; 4];
        let mut ch_iters = [0u32; 3];
        for ch in 0..3 {
            if let Some(cfg) = inputs.channels[ch] {
                if cfg.reset_history || !st.channel_has_history[ch] {
                    reset_mask |= 1u32 << ch;
                }
                min_alphas[ch] = cfg.min_alpha.clamp(0.01, 1.0);
                phi_colors[ch] = cfg.phi_color.max(0.05);
                ch_iters[ch] = cfg.atrous_iterations.clamp(1, 5);
            } else {
                st.channel_has_history[ch] = false;
            }
        }

        let u_base = SvgfMultiUniform {
            dims: [size.0, size.1, 1, reset_mask | (active_mask << 4)],
            temporal_params: [
                inputs.max_history.clamp(1.0, 256.0),
                inputs.depth_threshold.max(1e-4),
                inputs.normal_threshold.clamp(-1.0, 1.0),
                inputs.variance_boost.max(1.0),
            ],
            min_alphas,
            phi_colors,
            edge_params: [inputs.phi_normal.max(1.0), inputs.phi_depth.max(1e-4), 0.0, 0.0],
        };
        let stride = self.uniform_stride as usize;
        let u_size = std::mem::size_of::<SvgfMultiUniform>();
        let slots = 1 + max_iters as usize;
        self.staged.clear();
        self.staged.resize(slots * stride, 0);
        self.staged[..u_size].copy_from_slice(bytemuck::bytes_of(&u_base));
        for iter in 0..max_iters {
            let mut iter_mask = 0u32;
            for (ch, &iters) in ch_iters.iter().enumerate() {
                if iter < iters {
                    iter_mask |= 1u32 << ch;
                }
            }
            let u_iter = SvgfMultiUniform { dims: [size.0, size.1, 1u32 << iter, iter_mask << 4], ..u_base };
            let off = (1 + iter as usize) * stride;
            self.staged[off..off + u_size].copy_from_slice(bytemuck::bytes_of(&u_iter));
        }
        encode_copy(device, encoder, &self.uniform, &self.staged);

        let hist_c = st.history_color_idx;
        let slot_a = (hist_c + 1) % 3;
        let slot_b = (hist_c + 2) % 3;
        let prev_m = st.moments_idx;
        let curr_m = 1 - prev_m;

        let u_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.uniform,
            offset: 0,
            size: wgpu::BufferSize::new(u_size as u64),
        });
        let noisy0 = inputs.channels[0].map_or(fallback_noisy, |c| c.noisy_signal);
        let noisy1 = inputs.channels[1].map_or(fallback_noisy, |c| c.noisy_signal);
        let noisy2 = inputs.channels[2].map_or(fallback_noisy, |c| c.noisy_signal);
        let t_slot = hist_c * 2 + prev_m;
        let t_hit = st.temporal_bgs[t_slot].as_ref().is_some_and(|(keys, _)| {
            &keys[0] == noisy0
                && &keys[1] == noisy1
                && &keys[2] == noisy2
                && &keys[3] == inputs.depth
                && &keys[4] == inputs.prev_depth
                && &keys[5] == inputs.normal
                && &keys[6] == inputs.prev_normal
                && &keys[7] == inputs.motion
        });
        if !t_hit {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("svgf_multi_temporal_bg"),
                layout: &self.temporal_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u_binding.clone() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(noisy0) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(noisy1) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(noisy2) },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&st.color_bufs[hist_c].array_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&st.moments[prev_m].array_view),
                    },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(inputs.depth) },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(inputs.prev_depth),
                    },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(inputs.normal) },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::TextureView(inputs.prev_normal),
                    },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(inputs.motion) },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wgpu::BindingResource::TextureView(&st.color_bufs[slot_a].array_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: wgpu::BindingResource::TextureView(&st.moments[curr_m].array_view),
                    },
                ],
            });
            st.temporal_bgs[t_slot] = Some((
                [
                    noisy0.clone(),
                    noisy1.clone(),
                    noisy2.clone(),
                    inputs.depth.clone(),
                    inputs.prev_depth.clone(),
                    inputs.normal.clone(),
                    inputs.prev_normal.clone(),
                    inputs.motion.clone(),
                ],
                bg,
            ));
        }
        let tbg = &st.temporal_bgs[t_slot].as_ref().unwrap().1;
        let (gx, gy) = (size.0.div_ceil(8), size.1.div_ceil(8));
        {
            let ts = crate::renderer::timing::scope(crate::renderer::timing::Span::Scene);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("svgf_multi_temporal"),
                timestamp_writes: ts.compute(),
            });
            pass.set_pipeline(&self.temporal_pipe);
            pass.set_bind_group(0, tbg, &[0]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        let mut src_idx = slot_a;
        let mut dst_idx = slot_b;
        for iter in 0..max_iters {
            let a_slot = src_idx * 3 + dst_idx;
            let a_hit = st.atrous_bgs[a_slot]
                .as_ref()
                .is_some_and(|(keys, _)| &keys[0] == inputs.depth && &keys[1] == inputs.normal);
            if !a_hit {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("svgf_multi_atrous_bg"),
                    layout: &self.atrous_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: u_binding.clone() },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&st.color_bufs[src_idx].array_view),
                        },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(inputs.depth) },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(inputs.normal),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&st.color_bufs[dst_idx].array_view),
                        },
                    ],
                });
                st.atrous_bgs[a_slot] = Some(([inputs.depth.clone(), inputs.normal.clone()], bg));
            }
            let abg = &st.atrous_bgs[a_slot].as_ref().unwrap().1;
            let dyn_off = (1 + iter) * self.uniform_stride;
            {
                let ts = crate::renderer::timing::scope(crate::renderer::timing::Span::Scene);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("svgf_multi_atrous"),
                    timestamp_writes: ts.compute(),
                });
                pass.set_pipeline(&self.atrous_pipe);
                pass.set_bind_group(0, abg, &[dyn_off]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            for (ch, &iters) in ch_iters.iter().enumerate() {
                if iters == iter + 1 {
                    st.output_idx[ch] = dst_idx;
                }
            }
            if iter + 1 < max_iters {
                if iter == 0 {
                    src_idx = slot_b;
                    dst_idx = slot_a;
                } else {
                    src_idx = dst_idx;
                    dst_idx = if dst_idx == slot_a { hist_c } else { slot_a };
                }
            }
        }
        for (ch, opt) in inputs.channels.iter().enumerate() {
            if opt.is_some() {
                st.channel_has_history[ch] = true;
            }
        }
        st.history_color_idx = slot_b;
        st.moments_idx = curr_m;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_to_f16_bits(v: f32) -> u16 {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff).cast_signed() - 127;
        let frac = b & 0x7fffff;
        if exp < -14 {
            sign
        } else if exp > 15 {
            sign | 0x7c00
        } else {
            sign | (((exp + 15) as u16) << 10) | ((frac >> 13) as u16)
        }
    }

    fn f16_to_f32(bits: u16) -> f32 {
        let sign = u32::from((bits >> 15) & 1) << 31;
        let exp = u32::from((bits >> 10) & 0x1f);
        let frac = u32::from(bits & 0x3ff);
        if exp == 0 {
            if frac == 0 {
                f32::from_bits(sign)
            } else {
                let mut f = frac;
                let mut e = -14i32;
                while (f & 0x400) == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= 0x3ff;
                f32::from_bits(sign | (((e + 127) as u32) << 23) | (f << 13))
            }
        } else if exp == 31 {
            f32::from_bits(sign | (0xff << 23) | (frac << 13))
        } else {
            f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13))
        }
    }

    fn upload_rgba16f(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        pixels: &[[f32; 4]],
    ) -> wgpu::Texture {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("svgf_test_tex"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let raw: Vec<u16> = pixels.iter().flat_map(|p| p.map(f32_to_f16_bits)).collect();
        queue.write_texture(
            tex.as_image_copy(),
            bytemuck::cast_slice(&raw),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        tex
    }

    #[test]
    fn gpu_svgf_reduces_variance_while_preserving_sharp_depth_and_shadow_edges() {
        let (device, queue) = crate::testing::device().expect("GPU adapter required");
        let (w, h) = (32u32, 32u32);

        // Two distinct planes split at x = 16: left = dark plane at z = 0.2, right = bright plane at z = 0.8.
        let mut depth = vec![0.0f32; (w * h) as usize];
        let mut normal = vec![[0.0f32; 4]; (w * h) as usize];
        let motion = vec![[0.0f32; 4]; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                depth[idx] = if x < 16 { 0.2 } else { 0.8 };
                normal[idx] = if x < 16 { [0.0, 1.0, 0.0, 0.0] } else { [0.0, 0.0, 1.0, 0.0] };
            }
        }
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("svgf_d"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            depth_tex.as_image_copy(),
            bytemuck::cast_slice(&depth),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let normal_tex = upload_rgba16f(&device, &queue, w, h, &normal);
        let motion_tex = upload_rgba16f(&device, &queue, w, h, &motion);
        let depth_view = depth_tex.create_view(&Default::default());
        let normal_view = normal_tex.create_view(&Default::default());
        let motion_view = motion_tex.create_view(&Default::default());

        let mut denoiser = SvgfDenoiser::new(&device);
        let mut input_var_sum = 0.0f64;

        // Feed 8 noisy frames with strong checkerboard + temporal jitter noise.
        for frame in 0u32..8 {
            let mut noisy = vec![[0.0f32; 4]; (w * h) as usize];
            for y in 0..h {
                for x in 0..w {
                    let sign = if ((x + y + frame) & 1) == 0 { 1.0 } else { -1.0 };
                    let base: f32 = if x < 16 { 0.15 } else { 0.85 };
                    let val: f32 = (base + 0.12 * sign).clamp(0.0, 1.0);
                    noisy[(y * w + x) as usize] = [val, val, val, 1.0];
                    if frame == 7 && (20..28).contains(&x) && (8..24).contains(&y) {
                        let diff = f64::from(val) - 0.85;
                        input_var_sum += diff * diff;
                    }
                }
            }
            let noisy_tex = upload_rgba16f(&device, &queue, w, h, &noisy);
            let noisy_view = noisy_tex.create_view(&Default::default());
            let mut enc = device.create_command_encoder(&Default::default());
            denoiser
                .denoise(
                    &device,
                    &mut enc,
                    &SvgfInputs {
                        noisy_signal: &noisy_view,
                        depth: &depth_view,
                        prev_depth: &depth_view,
                        normal: &normal_view,
                        prev_normal: &normal_view,
                        motion: &motion_view,
                    },
                    (w, h),
                    &SvgfParams::default(),
                )
                .unwrap();
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }

        // Read back denoised frame 7.
        let bpr = (w * 8).next_multiple_of(256);
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("svgf_rb"),
            size: u64::from(bpr * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            denoiser.output_texture().unwrap().as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &rb,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().unwrap();
        let read_r = |x: u32, y: u32| -> f32 {
            let o = (y * bpr + x * 8) as usize;
            f16_to_f32(u16::from_le_bytes([data[o], data[o + 1]]))
        };

        let mut out_var_sum = 0.0f64;
        for y in 8..24 {
            for x in 20..28 {
                let diff = f64::from(read_r(x, y)) - 0.85;
                out_var_sum += diff * diff;
            }
        }
        let in_var = input_var_sum / 128.0;
        let out_var = out_var_sum / 128.0;
        assert!(
            out_var < in_var * 0.05,
            "SVGF must cut flat-surface noise variance by >20x: in_var={in_var:.6}, out_var={out_var:.6}"
        );

        // Edge sharpness across x = 15 | x = 16 discontinuity (different depth + normal).
        let left_edge = read_r(15, 16);
        let right_edge = read_r(16, 16);
        assert!(
            left_edge < 0.25 && right_edge > 0.75,
            "SVGF must preserve sharp geometry edge across depth/normal discontinuity: left={left_edge}, right={right_edge}"
        );
        drop(data);
        rb.unmap();
    }

    fn read_rgba16f_tex(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tex: &wgpu::Texture,
        w: u32,
        h: u32,
    ) -> Vec<[f32; 4]> {
        let bpr = (w * 8).next_multiple_of(256);
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("svgf_spec_rb"),
            size: u64::from(bpr * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &rb,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().unwrap();
        let mut out = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let o = (y * bpr + x * 8) as usize;
                out.push([
                    f16_to_f32(u16::from_le_bytes([data[o], data[o + 1]])),
                    f16_to_f32(u16::from_le_bytes([data[o + 2], data[o + 3]])),
                    f16_to_f32(u16::from_le_bytes([data[o + 4], data[o + 5]])),
                    f16_to_f32(u16::from_le_bytes([data[o + 6], data[o + 7]])),
                ]);
            }
        }
        drop(data);
        rb.unmap();
        out
    }

    #[test]
    fn gpu_svgf_hdr_moments_stay_finite_across_frames() {
        let (device, queue) = crate::testing::device().expect("GPU adapter required");
        let upload = |pixels| upload_rgba16f(&device, &queue, 2, 1, pixels).create_view(&Default::default());
        let noisy = upload(&[[1024.0; 4], [f32::INFINITY; 4]]);
        let depth = upload(&[[0.5; 4]; 2]);
        let normal = upload(&[[0.0, 0.0, 1.0, 0.5]; 2]);
        let motion = upload(&[[0.0; 4]; 2]);
        let inputs = SvgfInputs {
            noisy_signal: &noisy,
            depth: &depth,
            prev_depth: &depth,
            normal: &normal,
            prev_normal: &normal,
            motion: &motion,
        };
        let mut svgf = SvgfDenoiser::new(&device);
        for frame in 0..2 {
            let mut encoder = device.create_command_encoder(&Default::default());
            let params = SvgfParams { reset_history: frame == 0, specular_mode: true, ..Default::default() };
            svgf.denoise(&device, &mut encoder, &inputs, (2, 1), &params).unwrap();
            queue.submit([encoder.finish()]);
            let st = svgf.state.as_ref().unwrap();
            let history = &st.moments[st.moments_idx].tex;
            let moments = read_rgba16f_tex(&device, &queue, history, 2, 1);
            for [mean, second, len, hit] in moments {
                assert_eq!(mean, 240.0);
                assert_eq!(second, 57_600.0);
                assert!(len.is_finite() && hit.is_finite());
            }
        }
    }

    #[test]
    fn gpu_svgf_specular_ycocg_clamp_parallax_reproj_and_roughness_atrous_witness() {
        let (device, queue) = crate::testing::device().expect("GPU adapter required");
        let (w, h) = (32u32, 32u32);
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("svgf_spec_d"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            depth_tex.as_image_copy(),
            bytemuck::cast_slice(&vec![0.5f32; (w * h) as usize]),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let depth_v = depth_tex.create_view(&Default::default());
        let mirror_n = upload_rgba16f(&device, &queue, w, h, &vec![[0.0, 1.0, 0.0, 0.06]; (w * h) as usize]);
        let rough_n = upload_rgba16f(&device, &queue, w, h, &vec![[0.0, 1.0, 0.0, 0.70]; (w * h) as usize]);
        let zero_m = upload_rgba16f(&device, &queue, w, h, &vec![[0.0; 4]; (w * h) as usize]);
        let mirror_nv = mirror_n.create_view(&Default::default());
        let rough_nv = rough_n.create_view(&Default::default());
        let zero_mv = zero_m.create_view(&Default::default());

        // 1. YCoCg neighborhood clamping suppresses stale mirror reflection ghosting immediately on color flip.
        let green = upload_rgba16f(&device, &queue, w, h, &vec![[0.0, 1.0, 0.0, 1.0]; (w * h) as usize]);
        let red = upload_rgba16f(&device, &queue, w, h, &vec![[1.0, 0.0, 0.0, 1.0]; (w * h) as usize]);
        let green_v = green.create_view(&Default::default());
        let red_v = red.create_view(&Default::default());

        let mut spec = SvgfDenoiser::new(&device);
        let mut diff = SvgfDenoiser::new(&device);
        let spec_p = SvgfParams { min_alpha: 0.05, atrous_iterations: 2, specular_mode: true, ..Default::default() };
        let diff_p = SvgfParams { min_alpha: 0.05, atrous_iterations: 2, specular_mode: false, ..Default::default() };
        for _ in 0..8 {
            let mut enc = device.create_command_encoder(&Default::default());
            let inp = SvgfInputs {
                noisy_signal: &green_v,
                depth: &depth_v,
                prev_depth: &depth_v,
                normal: &mirror_nv,
                prev_normal: &mirror_nv,
                motion: &zero_mv,
            };
            spec.denoise(&device, &mut enc, &inp, (w, h), &spec_p).unwrap();
            diff.denoise(&device, &mut enc, &inp, (w, h), &diff_p).unwrap();
            queue.submit(Some(enc.finish()));
        }
        let mut enc_flip = device.create_command_encoder(&Default::default());
        let flip_in = SvgfInputs {
            noisy_signal: &red_v,
            depth: &depth_v,
            prev_depth: &depth_v,
            normal: &mirror_nv,
            prev_normal: &mirror_nv,
            motion: &zero_mv,
        };
        spec.denoise(&device, &mut enc_flip, &flip_in, (w, h), &spec_p).unwrap();
        diff.denoise(&device, &mut enc_flip, &flip_in, (w, h), &diff_p).unwrap();
        queue.submit(Some(enc_flip.finish()));
        let s_px = read_rgba16f_tex(&device, &queue, spec.output_texture().unwrap(), w, h)[(16 * w + 16) as usize];
        let d_px = read_rgba16f_tex(&device, &queue, diff.output_texture().unwrap(), w, h)[(16 * w + 16) as usize];
        assert!(
            s_px[0] > 0.85 && s_px[1] < 0.05 && d_px[1] > 0.80,
            "specular YCoCg clamp must wipe stale green reflection immediately: spec={s_px:?}, diff={d_px:?}"
        );

        // 2. Virtual parallax motion (`motion.zw`) reprojects shifted mirror highlights while `motion.xy == 0`.
        let mut sq0 = vec![[0.02f32, 0.02, 0.02, 1.0]; (w * h) as usize];
        let mut sq1 = vec![[0.02f32, 0.02, 0.02, 1.0]; (w * h) as usize];
        for y in 12..20 {
            for x in 8..16 {
                sq0[(y * w + x) as usize] = [0.1, 0.9, 1.0, 1.0];
                sq1[(y * w + (x + 4)) as usize] = [0.1, 0.9, 1.0, 1.0];
            }
        }
        let sq0_t = upload_rgba16f(&device, &queue, w, h, &sq0);
        let sq1_t = upload_rgba16f(&device, &queue, w, h, &sq1);
        let sq0_v = sq0_t.create_view(&Default::default());
        let sq1_v = sq1_t.create_view(&Default::default());
        let parallax_m =
            upload_rgba16f(&device, &queue, w, h, &vec![[0.0, 0.0, 4.0 / w as f32, 0.0]; (w * h) as usize]);
        let parallax_mv = parallax_m.create_view(&Default::default());
        let mut para = SvgfDenoiser::new(&device);
        let mut enc_p = device.create_command_encoder(&Default::default());
        para.denoise(
            &device,
            &mut enc_p,
            &SvgfInputs {
                noisy_signal: &sq0_v,
                depth: &depth_v,
                prev_depth: &depth_v,
                normal: &mirror_nv,
                prev_normal: &mirror_nv,
                motion: &zero_mv,
            },
            (w, h),
            &SvgfParams { reset_history: true, atrous_iterations: 1, specular_mode: true, ..Default::default() },
        )
        .unwrap();
        para.denoise(
            &device,
            &mut enc_p,
            &SvgfInputs {
                noisy_signal: &sq1_v,
                depth: &depth_v,
                prev_depth: &depth_v,
                normal: &mirror_nv,
                prev_normal: &mirror_nv,
                motion: &parallax_mv,
            },
            (w, h),
            &SvgfParams { reset_history: false, atrous_iterations: 1, specular_mode: true, ..Default::default() },
        )
        .unwrap();
        queue.submit(Some(enc_p.finish()));
        let p_hit = read_rgba16f_tex(&device, &queue, para.output_texture().unwrap(), w, h)[(16 * w + 16) as usize];
        assert!(
            p_hit[2] > 0.85,
            "specular parallax reprojection via motion.zw must track shifted highlight: {p_hit:?}"
        );

        // 3. Roughness-scaled A-Trous keeps mirror reflection peaks much sharper than rough glossy lobes.
        let mut impulse = vec![[0.05f32, 0.05, 0.05, 1.0]; (w * h) as usize];
        impulse[(16 * w + 16) as usize] = [1.0, 1.0, 1.0, 1.0];
        let imp_t = upload_rgba16f(&device, &queue, w, h, &impulse);
        let imp_v = imp_t.create_view(&Default::default());
        let mut d_mirror = SvgfDenoiser::new(&device);
        let mut d_rough = SvgfDenoiser::new(&device);
        let rp = SvgfParams {
            atrous_iterations: 3,
            reset_history: true,
            specular_mode: true,
            phi_color: 4.0,
            ..Default::default()
        };
        let mut enc_r = device.create_command_encoder(&Default::default());
        d_mirror
            .denoise(
                &device,
                &mut enc_r,
                &SvgfInputs {
                    noisy_signal: &imp_v,
                    depth: &depth_v,
                    prev_depth: &depth_v,
                    normal: &mirror_nv,
                    prev_normal: &mirror_nv,
                    motion: &zero_mv,
                },
                (w, h),
                &rp,
            )
            .unwrap();
        d_rough
            .denoise(
                &device,
                &mut enc_r,
                &SvgfInputs {
                    noisy_signal: &imp_v,
                    depth: &depth_v,
                    prev_depth: &depth_v,
                    normal: &rough_nv,
                    prev_normal: &rough_nv,
                    motion: &zero_mv,
                },
                (w, h),
                &rp,
            )
            .unwrap();
        queue.submit(Some(enc_r.finish()));
        let mirror_peak =
            read_rgba16f_tex(&device, &queue, d_mirror.output_texture().unwrap(), w, h)[(16 * w + 16) as usize][0];
        let rough_peak =
            read_rgba16f_tex(&device, &queue, d_rough.output_texture().unwrap(), w, h)[(16 * w + 16) as usize][0];
        assert!(
            mirror_peak > rough_peak + 0.08,
            "roughness-scaled A-Trous must preserve mirror peak sharper than rough glossy: mirror={mirror_peak:.4}, rough={rough_peak:.4}"
        );
    }

    #[test]
    fn gpu_svgf_hit_distance_adaptive_preserves_contact_reflection_sharpness() {
        let (device, queue) = crate::testing::device().expect("headless GPU adapter required");
        let (w, h) = (64u32, 32u32);
        let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SVGF hit distance witness depth"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let count = (w * h) as usize;
        queue.write_texture(
            depth.as_image_copy(),
            bytemuck::cast_slice(&vec![0.5f32; count]),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: None },
            size,
        );
        let depth = depth.create_view(&Default::default());
        let normal =
            upload_rgba16f(&device, &queue, w, h, &vec![[0.0, 1.0, 0.0, 0.35]; count]).create_view(&Default::default());
        let motion = upload_rgba16f(&device, &queue, w, h, &vec![[0.0; 4]; count]).create_view(&Default::default());
        let make_signal = |distance| {
            (0..w * h)
                .map(|i| {
                    let hash = (i ^ (i >> 4)).wrapping_mul(0x9e3779b9).rotate_left(11).wrapping_mul(0x85ebca6b);
                    let noise = ((hash >> 24) as f32 / 255.0 - 0.5) * 0.4;
                    let value = if i % w < w / 2 { 0.28 } else { 0.80 } + noise;
                    [value, value, value, distance]
                })
                .collect::<Vec<_>>()
        };
        let params = SvgfParams {
            specular_mode: true,
            hit_distance_adaptive: true,
            atrous_iterations: 3,
            phi_color: 8.0,
            ..Default::default()
        };
        let denoise = |distance: f32| {
            let mut denoiser = SvgfDenoiser::new(&device);
            let mut output = Vec::new();
            for (frame, sample_distance) in [distance, distance * 1.2].into_iter().enumerate() {
                let mut encoder = device.create_command_encoder(&Default::default());
                let noisy = upload_rgba16f(&device, &queue, w, h, &make_signal(sample_distance))
                    .create_view(&Default::default());
                let inputs = SvgfInputs {
                    noisy_signal: &noisy,
                    depth: &depth,
                    prev_depth: &depth,
                    normal: &normal,
                    prev_normal: &normal,
                    motion: &motion,
                };
                denoiser.denoise(&device, &mut encoder, &inputs, (w, h), &params).unwrap();
                queue.submit([encoder.finish()]);
                if frame == 0 {
                    output = read_rgba16f_tex(&device, &queue, denoiser.output_texture().unwrap(), w, h);
                }
            }
            let st = denoiser.state.as_ref().unwrap();
            let moments = &st.moments[st.moments_idx].tex;
            let filtered = read_rgba16f_tex(&device, &queue, moments, w, h);
            let expected = distance * 1.1;
            assert!((filtered[0][3] - expected).abs() < expected * 0.002, "filtered hit distance={}", filtered[0][3]);
            output
        };

        let contact = denoise(0.1);
        let distant = denoise(15.0);
        let mean = |values: &[f32]| values.iter().sum::<f32>() / values.len() as f32;
        let gradient = |image: &[[f32; 4]]| {
            let differences: Vec<_> = (4..h - 4)
                .map(|y| image[(y * w + w / 2) as usize][0] - image[(y * w + w / 2 - 1) as usize][0])
                .collect();
            mean(&differences)
        };
        let flat_variance = |image: &[[f32; 4]]| {
            let samples: Vec<_> =
                (4..h - 4).flat_map(|y| (4..20).map(move |x| image[(y * w + x) as usize][0])).collect();
            let avg = mean(&samples);
            mean(&samples.iter().map(|v| (v - avg).powi(2)).collect::<Vec<_>>())
        };
        let noisy_var = flat_variance(&make_signal(0.1));
        let (contact_gradient, distant_gradient) = (gradient(&contact), gradient(&distant));
        let (contact_var, distant_var) = (flat_variance(&contact), flat_variance(&distant));
        eprintln!(
            "SVGF edge gradient: contact={contact_gradient:.4}, distant={distant_gradient:.4}; variance: input={noisy_var:.5}, contact={contact_var:.5}, distant={distant_var:.5}"
        );
        assert!(contact_gradient > distant_gradient + 0.03, "contact={contact_gradient}, distant={distant_gradient}");
        assert!(contact_var < noisy_var * 0.8 && distant_var < noisy_var * 0.8);
    }

    fn read_rgba16f_layer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tex: &wgpu::Texture,
        layer: u32,
        w: u32,
        h: u32,
    ) -> Vec<[f32; 4]> {
        let bpp = 8u32;
        let bpr = (w * bpp).next_multiple_of(256);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("svgf_multi_readback"),
            size: u64::from(bpr * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range().unwrap();
        let mut out = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            let row = (y * bpr) as usize;
            for x in 0..w {
                let off = row + (x * bpp) as usize;
                let mut px = [0.0f32; 4];
                for (c, lane) in px.iter_mut().enumerate() {
                    let bits = u16::from_le_bytes([data[off + c * 2], data[off + c * 2 + 1]]);
                    *lane = f16_to_f32(bits);
                }
                out.push(px);
            }
        }
        out
    }

    #[test]
    fn gpu_svgf_multi_matches_single_channel_denoiser() {
        let Some((device, queue)) = crate::testing::device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let (w, h) = (32u32, 32u32);
        let n = (w * h) as usize;
        let depth = upload_rgba16f(
            &device,
            &queue,
            w,
            h,
            &(0..n).map(|i| [if (i as u32 % w) < 16 { 5.0 } else { 12.0 }, 0.0, 0.0, 1.0]).collect::<Vec<_>>(),
        )
        .create_view(&Default::default());
        let normal =
            upload_rgba16f(&device, &queue, w, h, &vec![[0.0, 1.0, 0.0, 0.5]; n]).create_view(&Default::default());
        let motion = upload_rgba16f(&device, &queue, w, h, &vec![[0.0; 4]; n]).create_view(&Default::default());

        let make_noisy = |seed: u32, base: [f32; 3]| {
            let px: Vec<[f32; 4]> = (0..n)
                .map(|i| {
                    let h = ((i as u32) ^ seed).wrapping_mul(0x9e37_79b9).rotate_left(7);
                    let jitter = ((h & 255) as f32 / 255.0 - 0.5) * 0.25;
                    [(base[0] + jitter).max(0.0), (base[1] + jitter).max(0.0), (base[2] + jitter).max(0.0), 1.0]
                })
                .collect();
            upload_rgba16f(&device, &queue, w, h, &px).create_view(&Default::default())
        };

        let mut single_sao = SvgfDenoiser::new(&device);
        let mut single_gi = SvgfDenoiser::new(&device);
        let mut single_di = SvgfDenoiser::new(&device);
        let mut multi = SvgfMultiDenoiser::new(&device);
        let defaults = SvgfParams::default();
        let sao_p = SvgfParams { min_alpha: 0.1, phi_color: 4.0, atrous_iterations: 3, ..defaults };
        let gi_p = SvgfParams { min_alpha: 0.1, phi_color: 4.5, atrous_iterations: 3, ..defaults };
        let di_p = SvgfParams { min_alpha: 0.12, phi_color: 4.0, atrous_iterations: 2, ..defaults };

        for frame in 0..3u32 {
            let n0 = make_noisy(11 + frame * 7, [0.8, 0.7, 0.2]);
            let n1 = make_noisy(37 + frame * 13, [0.3, 0.5, 0.9]);
            let n2 = make_noisy(91 + frame * 19, [0.9, 0.4, 0.2]);
            let mut enc = device.create_command_encoder(&Default::default());
            single_sao
                .denoise(
                    &device,
                    &mut enc,
                    &SvgfInputs {
                        noisy_signal: &n0,
                        depth: &depth,
                        prev_depth: &depth,
                        normal: &normal,
                        prev_normal: &normal,
                        motion: &motion,
                    },
                    (w, h),
                    &sao_p,
                )
                .unwrap();
            single_gi
                .denoise(
                    &device,
                    &mut enc,
                    &SvgfInputs {
                        noisy_signal: &n1,
                        depth: &depth,
                        prev_depth: &depth,
                        normal: &normal,
                        prev_normal: &normal,
                        motion: &motion,
                    },
                    (w, h),
                    &gi_p,
                )
                .unwrap();
            single_di
                .denoise(
                    &device,
                    &mut enc,
                    &SvgfInputs {
                        noisy_signal: &n2,
                        depth: &depth,
                        prev_depth: &depth,
                        normal: &normal,
                        prev_normal: &normal,
                        motion: &motion,
                    },
                    (w, h),
                    &di_p,
                )
                .unwrap();
            multi
                .denoise(
                    &device,
                    &mut enc,
                    &SvgfMultiInputs {
                        channels: [
                            Some(SvgfMultiChannel {
                                noisy_signal: &n0,
                                min_alpha: 0.1,
                                phi_color: 4.0,
                                atrous_iterations: 3,
                                reset_history: false,
                            }),
                            Some(SvgfMultiChannel {
                                noisy_signal: &n1,
                                min_alpha: 0.1,
                                phi_color: 4.5,
                                atrous_iterations: 3,
                                reset_history: false,
                            }),
                            Some(SvgfMultiChannel {
                                noisy_signal: &n2,
                                min_alpha: 0.12,
                                phi_color: 4.0,
                                atrous_iterations: 2,
                                reset_history: false,
                            }),
                        ],
                        depth: &depth,
                        prev_depth: &depth,
                        normal: &normal,
                        prev_normal: &normal,
                        motion: &motion,
                        max_history: defaults.max_history,
                        depth_threshold: defaults.depth_threshold,
                        normal_threshold: defaults.normal_threshold,
                        phi_normal: defaults.phi_normal,
                        phi_depth: defaults.phi_depth,
                        variance_boost: defaults.variance_boost,
                    },
                    (w, h),
                )
                .unwrap();
            queue.submit([enc.finish()]);
        }

        for (ch, single) in [(0usize, &single_sao), (1, &single_gi), (2, &single_di)] {
            let ref_px = read_rgba16f_tex(&device, &queue, single.output_texture().unwrap(), w, h);
            let (mtex, layer) = multi.output_texture(ch).unwrap();
            let got_px = read_rgba16f_layer(&device, &queue, mtex, layer, w, h);
            let max_diff = ref_px
                .iter()
                .zip(got_px.iter())
                .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs()))
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 2e-3,
                "fused SVGF channel {ch} diverged from single-channel SVGF: max_diff={max_diff:.6}"
            );
        }
    }
}
