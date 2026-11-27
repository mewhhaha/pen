//! Native-Resolution Temporal Upscaling (TAAU), AMD FidelityFX RCAS Sharpening,
//! and Post-FX Polish (Film Grain, Anamorphic Lens Flare, Screen-Space Outlines).
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "GPU texture dimensions")]

use anyhow::{Result, ensure};

pub const UPSCALE_RCAS_WGSL: &str = crate::wgsl!("upscale_rcas.wgsl");
pub const POST_FX_WGSL: &str = crate::wgsl!("post_fx.wgsl");

/// Standard AMD FSR 2.2 / 3.1 quality presets mapping output displays to internal render scale,
/// jitter phase count, negative texture LOD mip bias, and recommended RCAS sharpening.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FsrQualityPreset {
    /// 1.0x native resolution temporal anti-aliasing (`scale = 1.0`, 8 jitter phases).
    NativeAA,
    /// 1.5x upscale (`scale = 0.6667`, ~44% rendered pixels, 18 jitter phases).
    Quality,
    /// 1.7x upscale (`scale = 0.5882`, ~35% rendered pixels, 24 jitter phases).
    Balanced,
    /// 2.0x upscale (`scale = 0.50`, 25% rendered pixels, 32 jitter phases).
    Performance,
    /// 3.0x upscale (`scale = 0.3333`, ~11% rendered pixels, 72 jitter phases).
    UltraPerformance,
}

impl FsrQualityPreset {
    /// Upscale factor (`display_dim / render_dim`).
    pub const fn ratio(self) -> f32 {
        match self {
            Self::NativeAA => 1.0,
            Self::Quality => 1.5,
            Self::Balanced => 1.7,
            Self::Performance => 2.0,
            Self::UltraPerformance => 3.0,
        }
    }

    /// Internal render scale factor (`Quality::scale`, `render_dim / display_dim`).
    pub const fn scale(self) -> f32 {
        1.0 / self.ratio()
    }

    /// Recommended Halton(2, 3) jitter phase count (`ceil(8 * ratio^2)`).
    pub const fn jitter_phases(self) -> u32 {
        match self {
            Self::NativeAA => 8,
            Self::Quality => 18,
            Self::Balanced => 24,
            Self::Performance => 32,
            Self::UltraPerformance => 72,
        }
    }

    /// Recommended negative texture LOD mip bias (`-log2(ratio)`).
    pub fn mip_bias(self) -> f32 {
        match self {
            Self::NativeAA => 0.0,
            _ => -self.ratio().log2(),
        }
    }

    /// Recommended AMD FidelityFX RCAS sharpening strength when `grade.sharpness` is not overridden.
    pub const fn default_sharpness(self) -> f32 {
        match self {
            Self::NativeAA => 0.20,
            Self::Quality => 0.30,
            Self::Balanced => 0.35,
            Self::Performance => 0.45,
            Self::UltraPerformance => 0.55,
        }
    }

    /// Classifies an arbitrary `Quality::scale` into the closest FSR preset.
    pub fn from_scale(scale: f32) -> Self {
        if scale >= 0.90 {
            Self::NativeAA
        } else if scale >= 0.625 {
            Self::Quality
        } else if scale >= 0.54 {
            Self::Balanced
        } else if scale >= 0.40 {
            Self::Performance
        } else {
            Self::UltraPerformance
        }
    }
}

/// Ratio-scaled Halton(2, 3) sub-pixel jitter sequence in `[-0.5, 0.5]` low-res pixels for FSR 2.2/3.1.
pub fn fsr_halton_jitter(frame_index: u32, in_width: u32, out_width: u32) -> [f32; 2] {
    let phases = crate::taa::fsr_jitter_phase_count(in_width, out_width);
    let (jx, jy) = crate::taa::jitter_for(u64::from(frame_index), phases);
    [jx as f32, jy as f32]
}

/// Low-discrepancy Halton(2, 3) sub-pixel jitter sequence in `[-0.5, 0.5]` low-res pixels for TAAU.
/// Pair directly with [`apply_projection_jitter`] and [`TaauUpscaler::upscale`] so jitter sign aligns.
pub fn halton_jitter(frame_index: u32) -> [f32; 2] {
    let (jx, jy) = crate::taa::jitter_for(u64::from(frame_index), 16);
    [jx as f32, jy as f32]
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

/// FSR 2.2/3.1-class native-resolution temporal upscaler (TAAU) with Lanczos-2 accumulation,
/// optional reactive mask, motion divergence rejection, luma-stability tracking, and ping-pong
/// `Rgba16Float` history buffers.
pub struct TaauUpscaler {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    dummy_reactive: (wgpu::Texture, wgpu::TextureView),
    history: Option<(u32, u32, [wgpu::Texture; 2], [wgpu::TextureView; 2])>,
    cached_bgs: [Option<([wgpu::TextureView; 4], wgpu::BindGroup)>; 2],
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
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("upscale_rcas"),
            source: wgpu::ShaderSource::Wgsl(crate::deferred_wgsl!("upscale_rcas.wgsl").into()),
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
        let dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("taau_dummy_reactive"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy_tex.create_view(&Default::default());
        Self {
            pipeline,
            bgl,
            sampler,
            uniform,
            dummy_reactive: (dummy_tex, dummy_view),
            history: None,
            cached_bgs: [None, None],
            parity: 0,
            has_history: false,
        }
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
        self.upscale_with_reactive(
            device,
            encoder,
            input_color,
            input_depth,
            input_velocity,
            None,
            in_size,
            out_size,
            jitter_px,
            base_alpha,
            reversed_z,
        )
    }

    /// Same as [`TaauUpscaler::upscale`], accepting an optional low-resolution FSR reactive mask
    /// texture (`[0, 1]` in `.r`) to prevent temporal ghosting on particles, translucents, and emissives.
    #[allow(clippy::too_many_arguments)]
    pub fn upscale_with_reactive(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input_color: &wgpu::TextureView,
        input_depth: &wgpu::TextureView,
        input_velocity: &wgpu::TextureView,
        reactive_mask: Option<&wgpu::TextureView>,
        in_size: (u32, u32),
        out_size: (u32, u32),
        jitter_px: [f32; 2],
        base_alpha: f32,
        reversed_z: bool,
    ) -> Result<()> {
        let max_dim = device.limits().max_texture_dimension_2d;
        ensure!(
            in_size.0 >= 1
                && in_size.1 >= 1
                && out_size.0 >= 1
                && out_size.1 >= 1
                && out_size.0 <= max_dim
                && out_size.1 <= max_dim,
            "invalid TAAU sizes in={in_size:?} out={out_size:?}"
        );
        let c_sz = input_color.texture().size();
        let d_sz = input_depth.texture().size();
        let v_sz = input_velocity.texture().size();
        ensure!(
            (c_sz.width, c_sz.height) == in_size
                && (d_sz.width, d_sz.height) == in_size
                && (v_sz.width, v_sz.height) == in_size,
            "TAAU input view extents color={:?} depth={:?} vel={:?} must match in_size {in_size:?}",
            (c_sz.width, c_sz.height),
            (d_sz.width, d_sz.height),
            (v_sz.width, v_sz.height),
        );
        if let Some(rm) = reactive_mask {
            let r_sz = rm.texture().size();
            ensure!(
                (r_sz.width, r_sz.height) == in_size,
                "TAAU reactive_mask extent {:?} must match in_size {in_size:?}",
                (r_sz.width, r_sz.height)
            );
        }
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
            self.cached_bgs = [None, None];
            self.parity = 0;
            self.has_history = false;
        }
        let read_idx = self.parity;
        let write_idx = 1 - self.parity;
        let (_, _, texs, views) = self.history.as_ref().unwrap();
        ensure!(input_color.texture() != &texs[write_idx], "TAAU input_color cannot alias internal write target");
        let phases = crate::taa::fsr_jitter_phase_count(in_size.0, out_size.0);
        let u = TaauUniform {
            dims: [in_size.0 as f32, in_size.1 as f32, out_size.0 as f32, out_size.1 as f32],
            jitter_blend: [
                if jitter_px[0].is_finite() { jitter_px[0] } else { 0.0 },
                if jitter_px[1].is_finite() { jitter_px[1] } else { 0.0 },
                base_alpha.clamp(0.02, 0.5),
                if self.has_history { 0.0 } else { 1.0 },
            ],
            flags: [
                1.25,
                if reversed_z { 1.0 } else { 0.0 },
                phases as f32,
                if reactive_mask.is_some() { 1.0 } else { 0.0 },
            ],
        };
        encode_stage_copy(device, encoder, &self.uniform, bytemuck::bytes_of(&u));
        let reactive_view = reactive_mask.unwrap_or(&self.dummy_reactive.1);
        let hit = self.cached_bgs[read_idx].as_ref().is_some_and(|(keys, _)| {
            &keys[0] == input_color
                && &keys[1] == input_depth
                && &keys[2] == input_velocity
                && &keys[3] == reactive_view
        });
        if !hit {
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
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&views[write_idx]),
                    },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(reactive_view) },
                ],
            });
            self.cached_bgs[read_idx] =
                Some(([input_color.clone(), input_depth.clone(), input_velocity.clone(), reactive_view.clone()], bg));
        }
        let bg = &self.cached_bgs[read_idx].as_ref().unwrap().1;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("taau_upscale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(out_size.0.div_ceil(8), out_size.1.div_ceil(8), 1);
        }
        self.parity = write_idx;
        self.has_history = true;
        Ok(())
    }
}

/// AMD FidelityFX FSR 1 EASU (12-tap Edge-Adaptive Spatial Upsampling, HDR-safe `Rgba16Float` output).
pub struct FsrEasuUpscaler {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    output: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
    cached_bg: Option<(wgpu::TextureView, wgpu::BindGroup)>,
}

impl FsrEasuUpscaler {
    pub fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("easu_bgl"),
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
            label: Some("easu_shader"),
            source: wgpu::ShaderSource::Wgsl(crate::deferred_wgsl!("upscale_rcas.wgsl").into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("easu_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("easu_upscale"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("easu_upscale"),
            compilation_options: Default::default(),
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("easu_u"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, bgl, uniform, output: None, cached_bg: None }
    }

    pub fn output_texture(&self) -> Option<&wgpu::Texture> {
        self.output.as_ref().map(|o| &o.2)
    }

    pub fn output_view(&self) -> Option<&wgpu::TextureView> {
        self.output.as_ref().map(|o| &o.3)
    }

    pub fn upscale(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        in_size: (u32, u32),
        out_size: (u32, u32),
    ) -> Result<()> {
        let max_dim = device.limits().max_texture_dimension_2d;
        ensure!(
            in_size.0 >= 1
                && in_size.1 >= 1
                && out_size.0 >= 1
                && out_size.1 >= 1
                && out_size.0 <= max_dim
                && out_size.1 <= max_dim,
            "invalid EASU sizes in={in_size:?} out={out_size:?}"
        );
        let src_sz = input.texture().size();
        ensure!(
            (src_sz.width, src_sz.height) == in_size,
            "EASU input extent {:?} must match in_size {in_size:?}",
            (src_sz.width, src_sz.height)
        );
        if !matches!(&self.output, Some((w, h, _, _)) if *w == out_size.0 && *h == out_size.1) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("easu_output"),
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
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.output = Some((out_size.0, out_size.1, tex, view));
            self.cached_bg = None;
        }
        let dims: [f32; 4] = [in_size.0 as f32, in_size.1 as f32, out_size.0 as f32, out_size.1 as f32];
        encode_stage_copy(device, encoder, &self.uniform, bytemuck::bytes_of(&dims));
        let (_, _, _, out_view) = self.output.as_ref().unwrap();
        if self.cached_bg.as_ref().is_none_or(|(key, _)| key != input) {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("easu_bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(out_view) },
                ],
            });
            self.cached_bg = Some((input.clone(), bg));
        }
        let bg = &self.cached_bg.as_ref().unwrap().1;
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("easu_upscale"), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(out_size.0.div_ceil(8), out_size.1.div_ceil(8), 1);
        Ok(())
    }
}

/// AMD FidelityFX RCAS contrast-adaptive sharpening pass (`Rgba16Float` HDR-safe output).
pub struct RcasSharpener {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    output: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
    cached_bg: Option<(wgpu::TextureView, wgpu::BindGroup)>,
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
            source: wgpu::ShaderSource::Wgsl(crate::deferred_wgsl!("upscale_rcas.wgsl").into()),
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
        Self { pipeline, bgl, uniform, output: None, cached_bg: None }
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
        self.sharpen_with_knee(device, encoder, input, size, sharpness, 1.0)
    }

    /// Runs RCAS sharpening with explicit tone-compression control:
    /// - `hdr_knee > 0.0`: HDR-safe Reinhard compression (`c / (c + knee)`) preserving highlights `> 1.0`.
    /// - `hdr_knee <= 0.0`: direct SDR linear sharpening on `[0, 1]` without Reinhard midtone distortion.
    pub fn sharpen_with_knee(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        size: (u32, u32),
        sharpness: f32,
        hdr_knee: f32,
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
            self.cached_bg = None;
        }
        let (_, _, out_tex, out_view) = self.output.as_ref().unwrap();
        ensure!(input.texture() != out_tex, "RCAS input cannot alias internal output texture");
        let s = if sharpness.is_finite() { sharpness.clamp(0.0, 1.0) } else { 0.0 };
        let knee = if hdr_knee.is_finite() && hdr_knee > 0.0 { hdr_knee.max(0.25) } else { 0.0 };
        encode_stage_copy(device, encoder, &self.uniform, bytemuck::cast_slice(&[s, knee, 0.0, 0.0]));
        if self.cached_bg.as_ref().is_none_or(|(key, _)| key != input) {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rcas_bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(out_view) },
                ],
            });
            self.cached_bg = Some((input.clone(), bg));
        }
        let bg = &self.cached_bg.as_ref().unwrap().1;
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rcas_sharpen"), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
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
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    output: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
    cached_bg: Option<([wgpu::TextureView; 2], wgpu::BindGroup)>,
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
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post_fx_shader"),
            source: wgpu::ShaderSource::Wgsl(crate::deferred_wgsl!("post_fx.wgsl").into()),
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post_fx_linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post_fx_u"),
            size: std::mem::size_of::<PostFxUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, bgl, sampler, uniform, output: None, cached_bg: None }
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
            self.cached_bg = None;
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
            grain_flare: [
                params.grain_strength.max(0.0),
                params.frame_seed,
                params.flare_strength.max(0.0),
                params.flare_threshold.max(0.0),
            ],
            flare_tint_spread: [
                params.flare_tint[0],
                params.flare_tint[1],
                params.flare_tint[2],
                params.flare_spread_px.clamp(0.5, 8.0),
            ],
            outline_color_str: [
                params.outline_color[0],
                params.outline_color[1],
                params.outline_color[2],
                params.outline_strength.clamp(0.0, 1.0),
            ],
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
        if self.cached_bg.as_ref().is_none_or(|(keys, _)| &keys[0] != color_in || &keys[1] != depth_in) {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post_fx_bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(color_in) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_in) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(out_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            self.cached_bg = Some(([color_in.clone(), depth_in.clone()], bg));
        }
        let bg = &self.cached_bg.as_ref().unwrap().1;
        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("post_fx"), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
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

    fn upload_rgba16f(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        pixels: &[[f32; 4]],
    ) -> (wgpu::Texture, wgpu::TextureView) {
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
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&halfs),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn upload_r32f(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        vals: &[f32],
    ) -> (wgpu::Texture, wgpu::TextureView) {
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
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
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
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
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

    #[test]
    fn gpu_fsr_easu_spatial_upscale_reactive_taau_and_presets_witness() {
        // 1. Standard FSR quality presets: monotone scale, ratio-scaled Halton phases, negative mip-bias.
        let presets = [
            FsrQualityPreset::NativeAA,
            FsrQualityPreset::Quality,
            FsrQualityPreset::Balanced,
            FsrQualityPreset::Performance,
            FsrQualityPreset::UltraPerformance,
        ];
        assert_eq!(FsrQualityPreset::NativeAA.mip_bias(), 0.0);
        assert_eq!(FsrQualityPreset::NativeAA.jitter_phases(), 8);
        assert_eq!(FsrQualityPreset::Performance.jitter_phases(), 32);
        assert!((FsrQualityPreset::Performance.mip_bias() - (-1.0)).abs() < 1e-4);
        for pair in presets.windows(2) {
            assert!(pair[0].scale() > pair[1].scale());
            assert!(pair[0].ratio() < pair[1].ratio());
            assert!(pair[0].jitter_phases() < pair[1].jitter_phases());
            assert!(pair[0].mip_bias() > pair[1].mip_bias());
            assert!(pair[0].default_sharpness() > 0.0 && pair[1].default_sharpness() >= pair[0].default_sharpness());
        }
        assert_eq!(FsrQualityPreset::from_scale(1.0), FsrQualityPreset::NativeAA);
        assert_eq!(FsrQualityPreset::from_scale(0.67), FsrQualityPreset::Quality);
        assert_eq!(FsrQualityPreset::from_scale(0.59), FsrQualityPreset::Balanced);
        assert_eq!(FsrQualityPreset::from_scale(0.50), FsrQualityPreset::Performance);
        assert_eq!(FsrQualityPreset::from_scale(0.33), FsrQualityPreset::UltraPerformance);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter required");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

        // 2. AMD FidelityFX FSR 1 EASU: HDR uniform preservation + crisp anti-ringing diagonal edge.
        let mut easu = FsrEasuUpscaler::new(&device);
        let (_st, solid_v) = upload_rgba16f(&device, &queue, 16, 16, &vec![[0.4f32, 1.8, 0.2, 1.0]; 256]);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        easu.upscale(&device, &mut enc, &solid_v, (16, 16), (32, 32)).unwrap();
        queue.submit(Some(enc.finish()));
        let solid_up = read_rgba16f(&device, &queue, easu.output_texture().unwrap());
        let mid = solid_up[16 * 32 + 16];
        assert!(
            (mid[0] - 0.4).abs() < 0.02 && (mid[1] - 1.8).abs() < 0.05 && (mid[2] - 0.2).abs() < 0.02,
            "FSR 1 EASU must preserve uniform HDR color accurately: {mid:?}"
        );

        let mut diag = vec![[0.05f32, 0.05, 0.05, 1.0]; 256];
        for y in 0..16 {
            for x in 0..16 {
                if x + y >= 16 {
                    diag[y * 16 + x] = [1.0, 1.0, 1.0, 1.0];
                }
            }
        }
        let (_dt, diag_v) = upload_rgba16f(&device, &queue, 16, 16, &diag);
        let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        easu.upscale(&device, &mut enc2, &diag_v, (16, 16), (32, 32)).unwrap();
        queue.submit(Some(enc2.finish()));
        let diag_up = read_rgba16f(&device, &queue, easu.output_texture().unwrap());
        assert!(
            diag_up.iter().all(|p| p[0].is_finite() && p[0] >= 0.04 && p[0] <= 1.02),
            "FSR 1 EASU anti-ringing clamp must stay within source bounds"
        );
        assert!(
            diag_up[8 * 32 + 8][0] < 0.08 && diag_up[24 * 32 + 24][0] > 0.95,
            "FSR 1 EASU must keep both sides of diagonal step crisp: dark={:?} bright={:?}",
            diag_up[8 * 32 + 8],
            diag_up[24 * 32 + 24]
        );

        // 3. FSR 2.2/3.1-class reactive mask anti-ghosting witness on TaauUpscaler::upscale_with_reactive.
        //    Alternating horizontal bands keep both 0.05 and 1.0 inside every pixel's 3x3 neighborhood so
        //    variance clipping alone cannot discard stale history; reactive=1.0 on the left half must drop
        //    stale history immediately while reactive=0.0 on the right half retains temporal inertia.
        let mut taau = TaauUpscaler::new(&device);
        let (_dep_t, depth_low) = upload_r32f(&device, &queue, 16, 16, &vec![0.5f32; 256]);
        let (_vel_t, vel_low) = upload_rgba16f(&device, &queue, 16, 16, &vec![[0.0; 4]; 256]);
        let mut stripes_a = vec![[0.05f32, 0.05, 0.05, 1.0]; 256];
        let mut stripes_b = vec![[1.0f32, 1.0, 1.0, 1.0]; 256];
        for y in 0..16 {
            for x in 0..16 {
                if y % 2 == 1 {
                    stripes_a[y * 16 + x] = [1.0, 1.0, 1.0, 1.0];
                    stripes_b[y * 16 + x] = [0.05, 0.05, 0.05, 1.0];
                }
            }
        }
        let (_sa_t, sa_v) = upload_rgba16f(&device, &queue, 16, 16, &stripes_a);
        let (_sb_t, sb_v) = upload_rgba16f(&device, &queue, 16, 16, &stripes_b);
        let mut reactive_vals = vec![0.0f32; 256];
        for y in 0..16 {
            for x in 0..8 {
                reactive_vals[y * 16 + x] = 1.0; // Left half reactive (translucent/VFX), right half opaque
            }
        }
        let (_rt, reactive_v) = upload_r32f(&device, &queue, 16, 16, &reactive_vals);
        for _ in 0..8 {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            taau.upscale_with_reactive(
                &device,
                &mut enc,
                &sa_v,
                &depth_low,
                &vel_low,
                Some(&reactive_v),
                (16, 16),
                (32, 32),
                [0.0, 0.0],
                0.08,
                false,
            )
            .unwrap();
            queue.submit(Some(enc.finish()));
        }
        let mut enc_flip = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        taau.upscale_with_reactive(
            &device,
            &mut enc_flip,
            &sb_v,
            &depth_low,
            &vel_low,
            Some(&reactive_v),
            (16, 16),
            (32, 32),
            [0.0, 0.0],
            0.08,
            false,
        )
        .unwrap();
        queue.submit(Some(enc_flip.finish()));
        let reacted = read_rgba16f(&device, &queue, taau.output_texture().unwrap());
        // Sample center of low-res row y=8 (switched 0.05 -> 1.0) at output y=16.
        let left_reactive = reacted[16 * 32 + 6][0];
        let right_inert = reacted[16 * 32 + 24][0];
        assert!(
            left_reactive > right_inert + 0.25,
            "FSR reactive mask must suppress ghosting on sudden color changes: reactive={left_reactive}, opaque={right_inert}"
        );
    }

    #[test]
    fn gpu_rcas_sdr_linear_vs_hdr_reinhard_witness() {
        let (device, queue) = crate::testing::device().expect("GPU adapter required");
        let (w, h) = (32u32, 32u32);
        // Top half: flat midtone 0.5 + SDR ridge (0.2 -> 0.5 -> 0.8); bottom row: HDR peak 4.0.
        let mut img = vec![[0.5f32, 0.5, 0.5, 1.0]; (w * h) as usize];
        for y in 8..20 {
            for x in 0..w {
                let v = if x == 16 {
                    0.8
                } else if x == 15 || x == 17 {
                    0.5
                } else {
                    0.2
                };
                img[(y * w + x) as usize] = [v, v, v, 1.0];
            }
        }
        img[(26 * w + 16) as usize] = [4.0, 4.0, 4.0, 1.0];
        let (_t, in_v) = upload_rgba16f(&device, &queue, w, h, &img);

        let mut rcas = RcasSharpener::new(&device);
        let mut enc_sdr = device.create_command_encoder(&Default::default());
        rcas.sharpen_with_knee(&device, &mut enc_sdr, &in_v, (w, h), 1.0, 0.0).unwrap();
        queue.submit(Some(enc_sdr.finish()));
        let sdr = read_rgba16f(&device, &queue, rcas.output_texture().unwrap());

        assert!(
            (sdr[(3 * w + 3) as usize][0] - 0.5).abs() < 2e-3,
            "SDR flat midtone distorted: {:?}",
            sdr[(3 * w + 3) as usize]
        );
        let sdr_diff = sdr[(14 * w + 16) as usize][0] - sdr[(14 * w + 15) as usize][0];
        assert!(sdr_diff > 0.30 + 0.05, "SDR linear RCAS must sharpen edge contrast: diff={sdr_diff}");
        assert!(sdr.iter().all(|p| p[0] >= 0.0 && p[0] <= 1.001), "SDR linear RCAS must stay within [0, 1]");

        let mut enc_hdr = device.create_command_encoder(&Default::default());
        rcas.sharpen_with_knee(&device, &mut enc_hdr, &in_v, (w, h), 1.0, 1.0).unwrap();
        queue.submit(Some(enc_hdr.finish()));
        let hdr = read_rgba16f(&device, &queue, rcas.output_texture().unwrap());
        assert!(
            hdr[(26 * w + 16) as usize][0] > 3.5,
            "HDR RCAS must preserve peaks > 1.0: {:?}",
            hdr[(26 * w + 16) as usize]
        );
    }
}
