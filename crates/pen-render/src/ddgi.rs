//! Dynamic Diffuse Global Illumination (DDGI, Majercik et al., JCGT 2019).
//!
//! Maintains a 3D grid of octahedral irradiance (`8x8 + 1px` border) and Chebyshev
//! visibility (`16x16 + 1px` border) probes updated via GPU ray tracing against
//! [`TlasSceneGpu`], supporting multi-bounce indirect diffuse color bleeding and
//! leak-free thin-wall separation.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "GPU dispatch and probe grid indexing")]

use anyhow::{Result, ensure};

use crate::ray_tracing::{TLAS_TRAVERSAL_WGSL, TlasSceneGpu};

pub const DDGI_WGSL: &str = include_str!("shaders/ddgi.wgsl");

pub const DDGI_IRRADIANCE_INNER: u32 = 8;
pub const DDGI_IRRADIANCE_TILE: u32 = 10;
pub const DDGI_VISIBILITY_INNER: u32 = 16;
pub const DDGI_VISIBILITY_TILE: u32 = 18;

/// Static layout & filtering configuration for a 3D [`DdgiVolume`].
#[derive(Clone, Copy, Debug)]
pub struct DdgiGridConfig {
    pub origin: [f32; 3],
    pub spacing: [f32; 3],
    pub dims: [u32; 3],
    pub rays_per_probe: u32,
    pub hysteresis: f32,
    pub max_ray_distance: f32,
    pub normal_bias: f32,
    pub view_bias: f32,
    pub depth_sharpness: f32,
    pub irradiance_gamma: f32,
    pub multi_bounce_scale: f32,
}

impl Default for DdgiGridConfig {
    fn default() -> Self {
        Self {
            origin: [-4.0, 0.5, -4.0],
            spacing: [2.0, 2.0, 2.0],
            dims: [5, 3, 5],
            rays_per_probe: 128,
            hysteresis: 0.90,
            max_ray_distance: 40.0,
            normal_bias: 0.10,
            view_bias: 0.20,
            depth_sharpness: 50.0,
            irradiance_gamma: 5.0,
            multi_bounce_scale: 0.75,
        }
    }
}

/// Per-frame lighting inputs for [`DdgiVolume::update`].
#[derive(Clone, Copy, Debug)]
pub struct DdgiFrameParams {
    pub sun_direction: [f32; 3],
    pub sun_color: [f32; 3],
    pub sky_zenith: [f32; 3],
    pub sky_horizon: [f32; 3],
    pub visibility_mask: u32,
    pub reset_history: bool,
}

impl Default for DdgiFrameParams {
    fn default() -> Self {
        Self {
            sun_direction: [0.4, 0.8, 0.45],
            sun_color: [3.0, 2.9, 2.7],
            sky_zenith: [0.15, 0.25, 0.5],
            sky_horizon: [0.35, 0.42, 0.55],
            visibility_mask: u32::MAX,
            reset_history: false,
        }
    }
}

/// Query point for GPU evaluation via [`DdgiVolume::evaluate_points`].
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DdgiQueryPoint {
    pub pos: [f32; 4],
    pub normal: [f32; 4],
    pub view: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DdgiUniform {
    origin_and_hysteresis: [f32; 4],
    spacing_and_max_dist: [f32; 4],
    sun_dir_and_gamma: [f32; 4],
    sun_color_and_sharp: [f32; 4],
    sky_zenith_and_nbias: [f32; 4],
    sky_horizon_and_vbias: [f32; 4],
    rot_col0: [f32; 4],
    rot_col1: [f32; 4],
    rot_col2: [f32; 4],
    grid_dims: [u32; 4],
    tlas_counts: [u32; 4],
}

struct AtlasTex {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

impl AtlasTex {
    fn new(device: &wgpu::Device, label: &str, w: u32, h: u32) -> Self {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
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
        let view = tex.create_view(&Default::default());
        Self { tex, view }
    }
}

/// Deterministic SO(3) rotation matrix for frame `n` using a low-discrepancy Halton/golden quaternion.
fn frame_rotation_matrix(frame: u32) -> [[f32; 3]; 3] {
    let f = f64::from(frame + 1);
    let u1 = (f * 0.754_877_666_246_692_7).fract();
    let u2 = (f * 0.569_840_290_998_053_2).fract() * std::f64::consts::TAU;
    let u3 = (f * 0.430_159_709_001_946_8).fract() * std::f64::consts::TAU;
    let s1 = (1.0 - u1).sqrt();
    let s2 = u1.sqrt();
    let x = (s1 * u2.sin()) as f32;
    let y = (s1 * u2.cos()) as f32;
    let z = (s2 * u3.sin()) as f32;
    let w = (s2 * u3.cos()) as f32;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)],
        [2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)],
        [2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

fn encode_copy(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::Buffer, bytes: &[u8]) {
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ddgi_stage"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    stage.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytes);
    stage.unmap();
    encoder.copy_buffer_to_buffer(&stage, 0, dst, 0, bytes.len() as u64);
}

/// GPU 3D DDGI probe volume with octahedral irradiance and Chebyshev visibility atlases.
pub struct DdgiVolume {
    config: DdgiGridConfig,
    trace_pipe: wgpu::ComputePipeline,
    trace_bgl: wgpu::BindGroupLayout,
    blend_irr_pipe: wgpu::ComputePipeline,
    blend_vis_pipe: wgpu::ComputePipeline,
    blend_bgl: wgpu::BindGroupLayout,
    border_irr_pipe: wgpu::ComputePipeline,
    border_vis_pipe: wgpu::ComputePipeline,
    border_bgl: wgpu::BindGroupLayout,
    eval_pipe: wgpu::ComputePipeline,
    eval_bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    ray_buf: wgpu::Buffer,
    sampler: wgpu::Sampler,
    irr_curr: AtlasTex,
    irr_tmp: AtlasTex,
    vis_curr: AtlasTex,
    vis_tmp: AtlasTex,
    frame_index: u32,
}

impl DdgiVolume {
    pub fn new(device: &wgpu::Device, config: DdgiGridConfig) -> Result<Self> {
        let [nx, ny, nz] = config.dims;
        ensure!(nx >= 1 && ny >= 1 && nz >= 1, "DDGI grid dims must be >= 1");
        ensure!(config.rays_per_probe >= 16 && config.rays_per_probe <= 512, "DDGI rays_per_probe must be in 16..=512");
        let max_dim = device.limits().max_texture_dimension_2d;
        let irr_w = nx * nz * DDGI_IRRADIANCE_TILE;
        let irr_h = ny * DDGI_IRRADIANCE_TILE;
        let vis_w = nx * nz * DDGI_VISIBILITY_TILE;
        let vis_h = ny * DDGI_VISIBILITY_TILE;
        ensure!(vis_w <= max_dim && vis_h <= max_dim, "DDGI atlas {vis_w}x{vis_h} exceeds device 2D limit {max_dim}");

        let u_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<DdgiUniform>() as u64),
            },
            count: None,
        };
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let tex_ro = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let storage_tex = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let samp_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };

        let trace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ddgi_trace_bgl"),
            entries: &[
                u_entry,
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_ro(4),
                tex_ro(5, true),
                tex_ro(6, true),
                samp_entry(7),
                storage_rw(8),
            ],
        });
        let blend_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ddgi_blend_bgl"),
            entries: &[u_entry, storage_ro(1), tex_ro(2, false), tex_ro(3, false), storage_tex(4), storage_tex(5)],
        });
        let border_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ddgi_border_bgl"),
            entries: &[tex_ro(1, false), storage_tex(2)],
        });
        let eval_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ddgi_eval_bgl"),
            entries: &[u_entry, tex_ro(1, true), tex_ro(2, true), samp_entry(3), storage_ro(4), storage_rw(5)],
        });

        let source = format!("{TLAS_TRAVERSAL_WGSL}\n{DDGI_WGSL}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ddgi_wgsl"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let make_pipe = |label: &str, bgl: &wgpu::BindGroupLayout, entry: &str| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let trace_pipe = make_pipe("ddgi_trace", &trace_bgl, "ddgi_trace_probes");
        let blend_irr_pipe = make_pipe("ddgi_blend_irr", &blend_bgl, "ddgi_blend_irradiance");
        let blend_vis_pipe = make_pipe("ddgi_blend_vis", &blend_bgl, "ddgi_blend_visibility");
        let border_irr_pipe = make_pipe("ddgi_border_irr", &border_bgl, "ddgi_border_irradiance");
        let border_vis_pipe = make_pipe("ddgi_border_vis", &border_bgl, "ddgi_border_visibility");
        let eval_pipe = make_pipe("ddgi_eval", &eval_bgl, "ddgi_eval_points");

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ddgi_u"),
            size: std::mem::size_of::<DdgiUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let total_rays = u64::from(nx) * u64::from(ny) * u64::from(nz) * u64::from(config.rays_per_probe);
        let ray_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ddgi_rays"),
            size: total_rays * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ddgi_linear_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Ok(Self {
            config,
            trace_pipe,
            trace_bgl,
            blend_irr_pipe,
            blend_vis_pipe,
            blend_bgl,
            border_irr_pipe,
            border_vis_pipe,
            border_bgl,
            eval_pipe,
            eval_bgl,
            uniform,
            ray_buf,
            sampler,
            irr_curr: AtlasTex::new(device, "ddgi_irr_curr", irr_w, irr_h),
            irr_tmp: AtlasTex::new(device, "ddgi_irr_tmp", irr_w, irr_h),
            vis_curr: AtlasTex::new(device, "ddgi_vis_curr", vis_w, vis_h),
            vis_tmp: AtlasTex::new(device, "ddgi_vis_tmp", vis_w, vis_h),
            frame_index: 0,
        })
    }

    pub fn config(&self) -> DdgiGridConfig {
        self.config
    }

    pub fn irradiance_view(&self) -> &wgpu::TextureView {
        &self.irr_curr.view
    }

    pub fn visibility_view(&self) -> &wgpu::TextureView {
        &self.vis_curr.view
    }

    pub fn irradiance_texture(&self) -> &wgpu::Texture {
        &self.irr_curr.tex
    }

    pub fn visibility_texture(&self) -> &wgpu::Texture {
        &self.vis_curr.tex
    }

    pub fn uniform_buffer(&self) -> &wgpu::Buffer {
        &self.uniform
    }

    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    pub fn reset(&mut self) {
        self.frame_index = 0;
    }

    /// Traces rotated Spherical Fibonacci rays per probe, updates irradiance and Chebyshev
    /// visibility atlases with hysteresis, and refreshes octahedral wrap borders.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &TlasSceneGpu,
        params: &DdgiFrameParams,
    ) {
        if params.reset_history {
            self.frame_index = 0;
        }
        let rot = frame_rotation_matrix(self.frame_index);
        let h = if self.frame_index == 0 { 0.0 } else { self.config.hysteresis.clamp(0.0, 0.99) };
        let u = DdgiUniform {
            origin_and_hysteresis: [self.config.origin[0], self.config.origin[1], self.config.origin[2], h],
            spacing_and_max_dist: [
                self.config.spacing[0].max(1e-3),
                self.config.spacing[1].max(1e-3),
                self.config.spacing[2].max(1e-3),
                self.config.max_ray_distance.max(1.0),
            ],
            sun_dir_and_gamma: [
                params.sun_direction[0],
                params.sun_direction[1],
                params.sun_direction[2],
                self.config.irradiance_gamma.max(1.0),
            ],
            sun_color_and_sharp: [
                params.sun_color[0].max(0.0),
                params.sun_color[1].max(0.0),
                params.sun_color[2].max(0.0),
                self.config.depth_sharpness.max(1.0),
            ],
            sky_zenith_and_nbias: [
                params.sky_zenith[0].max(0.0),
                params.sky_zenith[1].max(0.0),
                params.sky_zenith[2].max(0.0),
                self.config.normal_bias.max(0.0),
            ],
            sky_horizon_and_vbias: [
                params.sky_horizon[0].max(0.0),
                params.sky_horizon[1].max(0.0),
                params.sky_horizon[2].max(0.0),
                self.config.view_bias.max(0.0),
            ],
            rot_col0: [rot[0][0], rot[0][1], rot[0][2], self.config.multi_bounce_scale.clamp(0.0, 1.0)],
            rot_col1: [rot[1][0], rot[1][1], rot[1][2], self.frame_index as f32],
            rot_col2: [rot[2][0], rot[2][1], rot[2][2], 0.0],
            grid_dims: [self.config.dims[0], self.config.dims[1], self.config.dims[2], self.config.rays_per_probe],
            tlas_counts: [
                scene.tlas_node_count,
                scene.tlas_instance_count,
                scene.blas_tri_count,
                params.visibility_mask,
            ],
        };
        encode_copy(device, encoder, &self.uniform, bytemuck::bytes_of(&u));

        let [nx, ny, nz] = self.config.dims;
        let total_rays = nx * ny * nz * self.config.rays_per_probe;

        // 1. Trace probe rays.
        let trace_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ddgi_trace_bg"),
            layout: &self.trace_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scene.tlas_nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scene.tlas_instances_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: scene.blas_nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: scene.blas_tris_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.irr_curr.view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.vis_curr.view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 8, resource: self.ray_buf.as_entire_binding() },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ddgi_trace"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.trace_pipe);
            pass.set_bind_group(0, &trace_bg, &[]);
            pass.dispatch_workgroups(total_rays.div_ceil(64), 1, 1);
        }

        // 2. Blend interior octahedral texels into tmp atlases.
        let blend_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ddgi_blend_bg"),
            layout: &self.blend_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.ray_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.irr_curr.view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.vis_curr.view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.irr_tmp.view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.vis_tmp.view) },
            ],
        });
        let cols = nx * nz;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ddgi_blend"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blend_irr_pipe);
            pass.set_bind_group(0, &blend_bg, &[]);
            pass.dispatch_workgroups((cols * DDGI_IRRADIANCE_INNER).div_ceil(8), (ny * DDGI_IRRADIANCE_INNER).div_ceil(8), 1);

            pass.set_pipeline(&self.blend_vis_pipe);
            pass.set_bind_group(0, &blend_bg, &[]);
            pass.dispatch_workgroups((cols * DDGI_VISIBILITY_INNER).div_ceil(8), (ny * DDGI_VISIBILITY_INNER).div_ceil(8), 1);
        }

        // 3. Copy interior + wrap borders from tmp -> curr so curr is always sampling-ready!
        let border_irr_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ddgi_border_irr_bg"),
            layout: &self.border_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.irr_tmp.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.irr_curr.view) },
            ],
        });
        let border_vis_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ddgi_border_vis_bg"),
            layout: &self.border_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.vis_tmp.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.vis_curr.view) },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ddgi_borders"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.border_irr_pipe);
            pass.set_bind_group(0, &border_irr_bg, &[]);
            pass.dispatch_workgroups((cols * DDGI_IRRADIANCE_TILE).div_ceil(8), (ny * DDGI_IRRADIANCE_TILE).div_ceil(8), 1);

            pass.set_pipeline(&self.border_vis_pipe);
            pass.set_bind_group(0, &border_vis_bg, &[]);
            pass.dispatch_workgroups((cols * DDGI_VISIBILITY_TILE).div_ceil(8), (ny * DDGI_VISIBILITY_TILE).div_ceil(8), 1);
        }

        self.frame_index = self.frame_index.saturating_add(1);
    }

    /// Evaluates indirect diffuse irradiance on the GPU at a batch of world-space points.
    pub fn evaluate_points(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        points: &[DdgiQueryPoint],
    ) -> Result<Vec<[f32; 3]>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let in_bytes = bytemuck::cast_slice(points);
        let in_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ddgi_eval_in"),
            size: in_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        in_buf.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(in_bytes);
        in_buf.unmap();

        let out_size = (points.len() * 16) as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ddgi_eval_out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ddgi_eval_rb"),
            size: out_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ddgi_eval_bg"),
            layout: &self.eval_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.irr_curr.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.vis_curr.view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: in_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ddgi_eval_points"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.eval_pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((points.len() as u32).div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &rb, 0, out_size);
        queue.submit(Some(enc.finish()));

        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap()?;
        let data = slice.get_mapped_range()?;
        let vec4s: &[[f32; 4]] = bytemuck::cast_slice(&data);
        let res = vec4s.iter().map(|v| [v[0], v[1], v[2]]).collect();
        drop(data);
        rb.unmap();
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_tracing::{Blas, BlasTriangle, TlasInstance, TlasScene};

    fn quad_blas(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]) -> Blas {
        Blas::build(&[BlasTriangle::flat(p0, p1, p2), BlasTriangle::flat(p0, p2, p3)]).unwrap()
    }

    const ID: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    #[test]
    fn gpu_ddgi_cornell_color_bleeding_and_chebyshev_thin_wall_leak_suppression() {
        let (device, queue) = crate::testing::device().expect("GPU adapter required");

        // Two-chamber scene divided by an opaque vertical wall at x = 0:
        // Left chamber (x in [-4, 0], y in [0, 4], z in [-4, 4]):
        //   - Bright emissive red wall at x = -4
        //   - Closed white floor/ceiling/back/front
        // Right chamber (x in [0, 4], y in [0, 4], z in [-4, 4]):
        //   - Completely dark closed box separated by thin dividing wall at x = 0!
        let blases = vec![
            // 0: Floor y = 0, x in [-4, 4], z in [-4, 4]
            quad_blas([-4.0, 0.0, -4.0], [4.0, 0.0, -4.0], [4.0, 0.0, 4.0], [-4.0, 0.0, 4.0]),
            // 1: Ceiling y = 4
            quad_blas([-4.0, 4.0, -4.0], [-4.0, 4.0, 4.0], [4.0, 4.0, 4.0], [4.0, 4.0, -4.0]),
            // 2: Left emissive RED wall x = -4
            quad_blas([-4.0, 0.0, -4.0], [-4.0, 0.0, 4.0], [-4.0, 4.0, 4.0], [-4.0, 4.0, -4.0]),
            // 3: Dividing opaque wall at x = 0 (separating lit red room from dark room!)
            quad_blas([0.0, 0.0, -4.0], [0.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 4.0, -4.0]),
            // 4: Right dark wall x = 4
            quad_blas([4.0, 0.0, -4.0], [4.0, 4.0, -4.0], [4.0, 4.0, 4.0], [4.0, 0.0, 4.0]),
            // 5: Back wall z = -4
            quad_blas([-4.0, 0.0, -4.0], [-4.0, 4.0, -4.0], [4.0, 4.0, -4.0], [4.0, 0.0, -4.0]),
            // 6: Front wall z = 4
            quad_blas([-4.0, 0.0, 4.0], [4.0, 0.0, 4.0], [4.0, 4.0, 4.0], [-4.0, 4.0, 4.0]),
        ];
        let mut red_wall = TlasInstance::new(2, ID);
        red_wall.albedo = [0.9, 0.05, 0.05];
        red_wall.emissive = [6.0, 0.1, 0.1];

        let dark_wall = |bi: u32| {
            let mut inst = TlasInstance::new(bi, ID);
            inst.albedo = [0.75, 0.75, 0.75];
            inst
        };

        let instances = vec![
            dark_wall(0),
            dark_wall(1),
            red_wall,
            dark_wall(3),
            dark_wall(4),
            dark_wall(5),
            dark_wall(6),
        ];
        let scene = TlasScene::build(&blases, &instances).unwrap();
        let scene_gpu = TlasSceneGpu::upload(&device, &scene).unwrap();

        let cfg = DdgiGridConfig {
            origin: [-3.0, 1.0, -2.0],
            spacing: [2.0, 2.0, 2.0],
            dims: [4, 2, 3], // x probes at -3, -1, +1, +3 (straddling wall at x = 0!)
            rays_per_probe: 192,
            hysteresis: 0.7,
            max_ray_distance: 20.0,
            normal_bias: 0.08,
            view_bias: 0.12,
            depth_sharpness: 50.0,
            irradiance_gamma: 5.0,
            multi_bounce_scale: 0.8,
        };
        let mut ddgi = DdgiVolume::new(&device, cfg).unwrap();
        let frame_params = DdgiFrameParams {
            sun_direction: [0.0, 1.0, 0.0],
            sun_color: [0.0, 0.0, 0.0],
            sky_zenith: [0.0, 0.0, 0.0],
            sky_horizon: [0.0, 0.0, 0.0],
            visibility_mask: u32::MAX,
            reset_history: false,
        };

        // Run 6 multi-bounce updates so irradiance and Chebyshev visibility settle.
        for _ in 0..6 {
            let mut enc = device.create_command_encoder(&Default::default());
            ddgi.update(&device, &mut enc, &scene_gpu, &frame_params);
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }

        let results = ddgi
            .evaluate_points(
                &device,
                &queue,
                &[
                    // Point 0: Floor inside the red-lit left chamber (x = -1.5, y = 0.05, z = 0).
                    DdgiQueryPoint {
                        pos: [-1.5, 0.05, 0.0, 0.0],
                        normal: [0.0, 1.0, 0.0, 0.0],
                        view: [0.0, 1.0, 0.0, 0.0],
                    },
                    // Point 1: Floor inside the dark right chamber right behind the dividing wall (x = +0.4, y = 0.05, z = 0).
                    DdgiQueryPoint {
                        pos: [0.4, 0.05, 0.0, 0.0],
                        normal: [0.0, 1.0, 0.0, 0.0],
                        view: [0.0, 1.0, 0.0, 0.0],
                    },
                ],
            )
            .unwrap();

        let lit_room = results[0];
        let dark_room = results[1];
        assert!(
            lit_room[0] > 0.35 && lit_room[0] > lit_room[2] * 3.0,
            "DDGI left chamber must exhibit strong red indirect diffuse color bleed: {lit_room:?}"
        );
        assert!(
            dark_room[0] < lit_room[0] * 0.15,
            "Chebyshev visibility must suppress red light leaking across the dividing wall: lit={lit_room:?}, dark={dark_room:?}"
        );
    }
}
