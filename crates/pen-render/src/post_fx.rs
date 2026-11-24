//! Native-Resolution Temporal Upscaling (TAAU), AMD FidelityFX RCAS Sharpening,
//! and Post-FX Polish (Film Grain, Anamorphic Lens Flare, Screen-Space Outlines).
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "GPU texture dimensions")]

use anyhow::{Result, ensure};

pub const UPSCALE_RCAS_WGSL: &str = include_str!("shaders/upscale_rcas.wgsl");
pub const POST_FX_WGSL: &str = include_str!("shaders/post_fx.wgsl");

/// Low-discrepancy Halton(2, 3) sub-pixel jitter sequence in `[-0.5, 0.5]` low-res pixels for TAAU.
/// Pair directly with [`apply_projection_jitter`] and [`TaauUpscaler::upscale`] so jitter sign aligns.
pub fn halton_jitter(frame_index: u32) -> [f32; 2] {
    fn halton(mut i: u32, base: u32) -> f32 {
        let mut f = 1.0f32;
        let mut r = 0.0f32;
        while i > 0 {
            f /= base as f32;
            r += f * (i % base) as f32;
            i /= base;
        }
        r
    }
    let idx = (frame_index % 16) + 1;
    [halton(idx, 2) - 0.5, halton(idx, 3) - 0.5]
}

/// Applies sub-pixel `jitter_px` (in low-res pixel units across and down) to a column-major
/// projection matrix so rasterized features shift by `+jitter_px`, matching [`TaauUpscaler::upscale`].
pub fn apply_projection_jitter(projection: &[f32; 16], jitter_px: [f32; 2], in_size: (u32, u32)) -> [f32; 16] {
    let mut out = *projection;
    let w = in_size.0.max(1) as f32;
    let h = in_size.1.max(1) as f32;
    // Pixel +X -> NDC +2/w; Pixel +Y (down) -> NDC -2/h.
    let dx_ndc = (2.0 * jitter_px[0]) / w;
    let dy_ndc = -(2.0 * jitter_px[1]) / h;
    // Perspective matrices store clip.w = -view.z in column 2 row 3 (-1); ortho stores 1 in col 3 row 3.
    if out[11].abs() > 1e-5 {
        out[8] += dx_ndc * out[11];
        out[9] += dy_ndc * out[11];
    } else {
        out[12] += dx_ndc;
        out[13] += dy_ndc;
    }
    out
}

fn encode_stage_copy(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::Buffer, bytes: &[u8]) {
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("post_fx_stage"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    stage.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytes);
    stage.unmap();
    encoder.copy_buffer_to_buffer(&stage, 0, dst, 0, bytes.len() as u64);
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TaauUniform {
    dims: [f32; 4],
    jitter_blend: [f32; 4],
    flags: [f32; 4],
}

/// Native-resolution temporal upscaler (TAAU) with ping-pong `Rgba16Float` history buffers.
pub struct TaauUpscaler {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    history: Option<(u32, u32, [wgpu::Texture; 2], [wgpu::TextureView; 2])>,
    parity: usize,
    has_history: bool,
}

impl TaauUpscaler {
    pub fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taau_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<TaauUniform>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("upscale_rcas"),
            source: wgpu::ShaderSource::Wgsl(UPSCALE_RCAS_WGSL.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("taau_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("taau_upscale"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("taau_upscale"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("taau_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taau_u"),
            size: std::mem::size_of::<TaauUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, bgl, sampler, uniform, history: None, parity: 0, has_history: false }
    }

    pub fn reset(&mut self) {
        self.has_history = false;
    }

    pub fn output_texture(&self) -> Option<&wgpu::Texture> {
        self.history.as_ref().map(|h| &h.2[self.parity])
    }

    pub fn output_view(&self) -> Option<&wgpu::TextureView> {
        self.history.as_ref().map(|h| &h.3[self.parity])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upscale(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input_color: &wgpu::TextureView,
        input_depth: &wgpu::TextureView,
        input_velocity: &wgpu::TextureView,
        in_size: (u32, u32),
        out_size: (u32, u32),
        jitter_px: [f32; 2],
        base_alpha: f32,
        reversed_z: bool,
    ) -> Result<()> {
        let max_dim = device.limits().max_texture_dimension_2d;
        ensure!(
            in_size.0 >= 1 && in_size.1 >= 1 && out_size.0 >= 1 && out_size.1 >= 1 && out_size.0 <= max_dim && out_size.1 <= max_dim,
            "invalid TAAU sizes in={in_size:?} out={out_size:?}"
        );
        let c_sz = input_color.texture().size();
        let d_sz = input_depth.texture().size();
        let v_sz = input_velocity.texture().size();
        ensure!(
            (c_sz.width, c_sz.height) == in_size && (d_sz.width, d_sz.height) == in_size && (v_sz.width, v_sz.height) == in_size,
            "TAAU input view extents color={:?} depth={:?} vel={:?} must match in_size {in_size:?}",
            (c_sz.width, c_sz.height),
            (d_sz.width, d_sz.height),
            (v_sz.width, v_sz.height),
        );
        let need_alloc = !matches!(&self.history, Some((w, h, _, _)) if *w == out_size.0 && *h == out_size.1);
        if need_alloc {
            let make = |label| {
                let t = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: out_size.0, height: out_size.1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let v = t.create_view(&wgpu::TextureViewDescriptor::default());
                (t, v)
            };
            let (t0, v0) = make("taau_hist_0");
            let (t1, v1) = make("taau_hist_1");
            self.history = Some((out_size.0, out_size.1, [t0, t1], [v0, v1]));
            self.parity = 0;
            self.has_history = false;
        }
        let read_idx = self.parity;
        let write_idx = 1 - self.parity;
        let (_, _, texs, views) = self.history.as_ref().unwrap();
        ensure!(
            input_color.texture() != &texs[write_idx],
            "TAAU input_color cannot alias internal write target"
        );
        let u = TaauUniform {
            dims: [in_size.0 as f32, in_size.1 as f32, out_size.0 as f32, out_size.1 as f32],
            jitter_blend: [
                if jitter_px[0].is_finite() { jitter_px[0] } else { 0.0 },
                if jitter_px[1].is_finite() { jitter_px[1] } else { 0.0 },
                base_alpha.clamp(0.02, 0.5),
                if self.has_history { 0.0 } else { 1.0 },
            ],
            flags: [1.25, if reversed_z { 1.0 } else { 0.0 }, 0.0, 0.0],
        };
        encode_stage_copy(device, encoder, &self.uniform, bytemuck::bytes_of(&u));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taau_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input_color) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(input_depth) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(input_velocity) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&views[read_idx]) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&views[write_idx]) },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("taau_upscale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(out_size.0.div_ceil(8), out_size.1.div_ceil(8), 1);
        }
        self.parity = write_idx;
        self.has_history = true;
        Ok(())
    }
}

/// AMD FidelityFX RCAS contrast-adaptive sharpening pass (`Rgba16Float` HDR-safe output).
pub struct RcasSharpener {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    output: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
}

impl RcasSharpener {
    pub fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rcas_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rcas_shader"),
            source: wgpu::ShaderSource::Wgsl(UPSCALE_RCAS_WGSL.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rcas_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rcas_sharpen"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("rcas_sharpen"),
            compilation_options: Default::default(),
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rcas_u"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, bgl, uniform, output: None }
    }

    pub fn output_texture(&self) -> Option<&wgpu::Texture> {
        self.output.as_ref().map(|o| &o.2)
    }

    pub fn output_view(&self) -> Option<&wgpu::TextureView> {
        self.output.as_ref().map(|o| &o.3)
    }

    pub fn sharpen(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        size: (u32, u32),
        sharpness: f32,
    ) -> Result<()> {
        let max_dim = device.limits().max_texture_dimension_2d;
        ensure!(size.0 >= 1 && size.1 >= 1 && size.0 <= max_dim && size.1 <= max_dim, "invalid RCAS size {size:?}");
        let in_sz = input.texture().size();
        ensure!(
            (in_sz.width, in_sz.height) == size,
            "RCAS input texture size {}x{} does not match requested {size:?}",
            in_sz.width,
            in_sz.height
        );
        if !matches!(&self.output, Some((w, h, _, _)) if *w == size.0 && *h == size.1) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("rcas_output"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.output = Some((size.0, size.1, tex, view));
        }
        let (_, _, out_tex, out_view) = self.output.as_ref().unwrap();
        ensure!(input.texture() != out_tex, "RCAS input cannot alias internal output texture");
        let s = if sharpness.is_finite() { sharpness.clamp(0.0, 1.0) } else { 0.0 };
        encode_stage_copy(device, encoder, &self.uniform, bytemuck::cast_slice(&[s, 1.0f32, 0.0, 0.0]));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rcas_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(out_view) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rcas_sharpen"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        Ok(())
    }
}

/// Configuration for film grain, anamorphic lens flare, and screen-space depth/normal outlines.
#[derive(Clone, Copy, Debug)]
pub struct PostFxParams {
    pub grain_strength: f32,
    pub frame_seed: f32,
    pub flare_strength: f32,
    pub flare_threshold: f32,
    pub flare_tint: [f32; 3],
    pub flare_spread_px: f32,
    pub outline_color: [f32; 3],
    pub outline_strength: f32,
    pub depth_threshold: f32,
    pub normal_threshold: f32,
    pub outline_thickness_px: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub reversed_z: bool,
    pub orthographic: bool,
    /// Optional inverse projection matrix for exact view-space normal reconstruction in outlines.
    pub inverse_projection: Option<[f32; 16]>,
}

impl Default for PostFxParams {
    fn default() -> Self {
        Self {
            grain_strength: 0.03,
            frame_seed: 0.0,
            flare_strength: 0.35,
            flare_threshold: 1.2,
            flare_tint: [0.65, 0.85, 1.2],
            flare_spread_px: 3.0,
            outline_color: [0.02, 0.02, 0.04],
            outline_strength: 0.0,
            depth_threshold: 0.15,
            normal_threshold: 0.45,
            outline_thickness_px: 1.0,
            near_plane: 0.1,
            far_plane: 200.0,
            reversed_z: false,
            orthographic: false,
            inverse_projection: None,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PostFxUniform {
    inverse_projection: [f32; 16],
    grain_flare: [f32; 4],
    flare_tint_spread: [f32; 4],
    outline_color_str: [f32; 4],
    outline_params: [f32; 4],
    depth_params: [f32; 4],
}

/// GPU compute pass combining film grain, anamorphic lens flare, and screen-space silhouette outlines.
pub struct PostFxPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    output: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
}

impl PostFxPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post_fx_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<PostFxUniform>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post_fx_shader"),
            source: wgpu::ShaderSource::Wgsl(POST_FX_WGSL.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post_fx_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("post_fx"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("post_fx"),
            compilation_options: Default::default(),
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post_fx_u"),
            size: std::mem::size_of::<PostFxUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, bgl, uniform, output: None }
    }

    pub fn output_texture(&self) -> Option<&wgpu::Texture> {
        self.output.as_ref().map(|o| &o.2)
    }

    pub fn output_view(&self) -> Option<&wgpu::TextureView> {
        self.output.as_ref().map(|o| &o.3)
    }

    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        color_in: &wgpu::TextureView,
        depth_in: &wgpu::TextureView,
        size: (u32, u32),
        params: &PostFxParams,
    ) -> Result<()> {
        let max_dim = device.limits().max_texture_dimension_2d;
        ensure!(size.0 >= 1 && size.1 >= 1 && size.0 <= max_dim && size.1 <= max_dim, "invalid post-fx size {size:?}");
        let c_sz = color_in.texture().size();
        ensure!(
            (c_sz.width, c_sz.height) == size,
            "PostFxPass color_in extent {}x{} does not match requested {size:?}",
            c_sz.width,
            c_sz.height
        );
        ensure!(params.near_plane > 0.0 && params.far_plane > params.near_plane, "invalid near/far planes");
        if !matches!(&self.output, Some((w, h, _, _)) if *w == size.0 && *h == size.1) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("post_fx_output"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.output = Some((size.0, size.1, tex, view));
        }
        let (_, _, out_tex, out_view) = self.output.as_ref().unwrap();
        ensure!(color_in.texture() != out_tex, "PostFxPass color_in cannot alias internal output texture");
        let (inv_proj, has_inv_proj) = match params.inverse_projection {
            Some(m) => {
                ensure!(m.iter().all(|v| v.is_finite()), "non-finite inverse_projection");
                (m, 1.0f32)
            }
            None => (crate::matrix::IDENTITY, 0.0f32),
        };
        let u = PostFxUniform {
            inverse_projection: inv_proj,
            grain_flare: [params.grain_strength.max(0.0), params.frame_seed, params.flare_strength.max(0.0), params.flare_threshold.max(0.0)],
            flare_tint_spread: [params.flare_tint[0], params.flare_tint[1], params.flare_tint[2], params.flare_spread_px.clamp(0.5, 8.0)],
            outline_color_str: [params.outline_color[0], params.outline_color[1], params.outline_color[2], params.outline_strength.clamp(0.0, 1.0)],
            outline_params: [
                params.depth_threshold.max(1e-4),
                params.normal_threshold.max(1e-4),
                params.outline_thickness_px.clamp(1.0, 4.0),
                params.near_plane,
            ],
            depth_params: [
                params.far_plane,
                if params.reversed_z { 1.0 } else { 0.0 },
                if params.orthographic { 1.0 } else { 0.0 },
                has_inv_proj,
            ],
        };
        encode_stage_copy(device, encoder, &self.uniform, bytemuck::bytes_of(&u));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post_fx_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(color_in) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_in) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(out_view) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("post_fx"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_to_f16(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xff).cast_signed() - 112;
        let mant = bits & 0x7f_ffff;
        if exp >= 31 {
            return (sign | 0x7c00) as u16;
        }
        if exp > 0 {
            return (sign | (((exp as u32) << 10) + ((mant + 0x1000) >> 13))) as u16;
        }
        if exp < -10 {
            return sign as u16;
        }
        (sign | ((((mant | 0x80_0000) >> (1 - exp)) + 0x1000) >> 13)) as u16
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

    fn upload_rgba16f(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32, pixels: &[[f32; 4]]) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_rgba16f"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let halfs: Vec<u16> = pixels.iter().flatten().map(|&v| f32_to_f16(v)).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&halfs),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn upload_r32f(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32, vals: &[f32]) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_r32f"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(vals),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn read_rgba16f(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture) -> Vec<[f32; 4]> {
        let (w, h) = (tex.width(), tex.height());
        let bpr = (w * 8).next_multiple_of(256);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post_fx_rb"),
            size: u64::from(bpr * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let slice = buf.slice(..);
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
        buf.unmap();
        out
    }

    #[test]
    fn gpu_taau_rcas_hdr_anamorphic_flare_outlines_and_black_safe_grain_witness() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter required");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

        // 1. TAAU 2x upscale (16x16 -> 32x32) with matching `+jitter_px` convention across 4 Halton frames,
        //    plus verifying HDR linear preservation on reset (no clamping at high radiance!).
        let mut taau = TaauUpscaler::new(&device);
        let (_dt, depth_low) = upload_r32f(&device, &queue, 16, 16, &vec![0.5f32; 256]);
        let (_vt, vel_low) = upload_rgba16f(&device, &queue, 16, 16, &vec![[0.0; 4]; 256]);
        for frame in 0..4u32 {
            let jitter = halton_jitter(frame);
            let mut low_px = vec![[0.05f32, 0.05, 0.05, 1.0]; 256];
            for y in 0..16 {
                for x in 0..16 {
                    // Feature at unjittered low-res x = 8.0 shifts to `8.0 + jitter[0]` in jittered raster!
                    if (x as f32 + 0.5) >= 8.0 + jitter[0] {
                        low_px[y * 16 + x] = [0.95, 0.95, 0.95, 1.0];
                    }
                }
            }
            let (_ct, col_low) = upload_rgba16f(&device, &queue, 16, 16, &low_px);
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            taau.upscale(&device, &mut enc, &col_low, &depth_low, &vel_low, (16, 16), (32, 32), jitter, 0.25, false)
                .unwrap();
            queue.submit(Some(enc.finish()));
        }
        let up = read_rgba16f(&device, &queue, taau.output_texture().unwrap());
        assert!(up.iter().flatten().all(|v| v.is_finite() && *v >= 0.0));
        assert!(up[16 * 32 + 10][0] < 0.15 && up[16 * 32 + 22][0] > 0.85, "TAAU edge reconstruction failed");

        // 2. RCAS sharpening on an HDR ridge (> 2.0 radiance!) + flat white patch -> sharpens HDR without shutting off!
        let mut ridge = vec![[1.2f32, 1.2, 1.2, 1.0]; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                let v = if y < 6 {
                    1.0
                } else if x == 16 {
                    3.2 // HDR peak > 1.0!
                } else if x == 15 || x == 17 {
                    2.0
                } else {
                    1.2
                };
                ridge[y * 32 + x] = [v, v, v, 1.0];
            }
        }
        let (_rt, ridge_v) = upload_rgba16f(&device, &queue, 32, 32, &ridge);
        let mut rcas = RcasSharpener::new(&device);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        rcas.sharpen(&device, &mut enc, &ridge_v, (32, 32), 1.0).unwrap();
        queue.submit(Some(enc.finish()));
        let sharp = read_rgba16f(&device, &queue, rcas.output_texture().unwrap());
        assert!(sharp.iter().flatten().all(|v| v.is_finite() && *v >= 0.0));
        let orig_hdr_diff = 3.2 - 2.0;
        let sharp_hdr_diff = sharp[16 * 32 + 16][0] - sharp[16 * 32 + 15][0];
        assert!(
            sharp_hdr_diff > orig_hdr_diff + 0.15,
            "RCAS must actively sharpen HDR peaks > 1.0: orig={orig_hdr_diff} sharp={sharp_hdr_diff}"
        );
        assert!((sharp[2 * 32 + 2][0] - 1.0).abs() < 1e-2, "flat white must stay ~1.0: {:?}", sharp[2 * 32 + 2]);

        // 3. Post-FX: cross-res depth (16x16 depth vs 32x32 color), anamorphic flare, and pure-black grain safety!
        let mut hdr = vec![[0.4f32, 0.4, 0.4, 1.0]; 32 * 32];
        hdr[..32].fill([0.0, 0.0, 0.0, 1.0]); // Pure black top row must stay pure black under film grain!
        hdr[16 * 32 + 16] = [8.0, 8.0, 8.0, 1.0];
        let mut depth_low_vals = vec![0.8f32; 16 * 16];
        for y in 0..16 {
            for x in 0..5 {
                depth_low_vals[y * 16 + x] = 0.2; // Silhouette step at half-res x=5 -> native x=10!
            }
        }
        let (_ht, hdr_v) = upload_rgba16f(&device, &queue, 32, 32, &hdr);
        let (_dh, depth_v) = upload_r32f(&device, &queue, 16, 16, &depth_low_vals);
        let mut pfx = PostFxPass::new(&device);
        let params = PostFxParams {
            grain_strength: 0.08,
            frame_seed: 3.0,
            flare_strength: 1.0,
            flare_threshold: 1.5,
            flare_tint: [0.6, 0.9, 1.3],
            flare_spread_px: 2.0,
            outline_color: [0.0, 0.0, 0.0],
            outline_strength: 1.0,
            depth_threshold: 0.1,
            normal_threshold: 10.0,
            outline_thickness_px: 1.0,
            near_plane: 0.1,
            far_plane: 0.9, // Short-range projection (< near + 1.0) must be accepted cleanly!
            reversed_z: false,
            orthographic: true,
            inverse_projection: None,
        };
        let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        pfx.dispatch(&device, &mut enc2, &hdr_v, &depth_v, (32, 32), &params).unwrap();
        queue.submit(Some(enc2.finish()));
        let fx = read_rgba16f(&device, &queue, pfx.output_texture().unwrap());
        assert!(fx[16 * 32 + 20][2] > fx[20 * 32 + 16][2] + 0.15, "anamorphic horizontal streak failed");
        assert!(fx[16 * 32 + 10][0] < fx[16 * 32 + 4][0] - 0.2, "cross-res depth silhouette outline failed");
        assert_eq!(fx[2][0], 0.0, "film grain must not lift pure black pixels: {:?}", fx[2]);
    }
}
