//! Dynamic Diffuse Global Illumination (DDGI, Majercik et al., JCGT 2019).
//!
//! Maintains a 3D grid of octahedral irradiance (`8x8 + 1px` border) and Chebyshev
//! visibility (`16x16 + 1px` border) probes updated via GPU ray tracing against
//! [`TlasSceneGpu`], supporting multi-bounce indirect diffuse color bleeding and
//! leak-free thin-wall separation.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "GPU dispatch and probe grid indexing")]

use anyhow::{Result, ensure};

use crate::ray_tracing::{MAX_RT_LOCAL_LIGHTS, RtLocalLight, TLAS_TRAVERSAL_WGSL, TlasSceneGpu};
#[cfg(not(target_arch = "wasm32"))]
use crate::ray_tracing::{TLAS_TRAVERSAL_HW_WGSL, device_supports_hardware_ray_query};

pub const DDGI_WGSL: &str = include_str!("shaders/ddgi.wgsl");

/// Number of `vec4f` header slots reserved at the start of the packed DDGI storage buffer (1 global + 2 cascades * 6).
pub const DDGI_PACKED_HEADER_VEC4S: u32 = 16;

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
    scroll_and_lights: [i32; 4],
    scroll_delta: [i32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DdgiPackUniform {
    meta: [u32; 4],
    strength: [f32; 4],
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

/// GPU 3D DDGI probe volume with octahedral irradiance and Chebyshev visibility atlases,
/// dynamic local-light injection, toroidal camera scrolling, and optional native hardware ray-query tracing.
pub struct DdgiVolume {
    config: DdgiGridConfig,
    trace_pipe: wgpu::ComputePipeline,
    trace_bgl: wgpu::BindGroupLayout,
    #[cfg(not(target_arch = "wasm32"))]
    trace_pipe_hw: Option<wgpu::ComputePipeline>,
    #[cfg(not(target_arch = "wasm32"))]
    trace_bgl_hw: Option<wgpu::BindGroupLayout>,
    blend_irr_pipe: wgpu::ComputePipeline,
    blend_vis_pipe: wgpu::ComputePipeline,
    blend_bg: wgpu::BindGroup,
    relocate_pipe: wgpu::ComputePipeline,
    reloc_bg: wgpu::BindGroup,
    border_irr_pipe: wgpu::ComputePipeline,
    border_vis_pipe: wgpu::ComputePipeline,
    border_irr_bg: wgpu::BindGroup,
    border_vis_bg: wgpu::BindGroup,
    eval_pipe: wgpu::ComputePipeline,
    eval_bgl: wgpu::BindGroupLayout,
    pack_pipe: wgpu::ComputePipeline,
    pack_bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    pack_uniform: wgpu::Buffer,
    local_lights_buf: wgpu::Buffer,
    ray_buf: wgpu::Buffer,
    probe_offsets: wgpu::Buffer,
    sampler: wgpu::Sampler,
    irr_curr: AtlasTex,
    _irr_tmp: AtlasTex,
    vis_curr: AtlasTex,
    _vis_tmp: AtlasTex,
    scroll_offset: [i32; 3],
    pending_scroll_delta: [i32; 3],
    frame_index: u32,
    trace_bg: Option<(Option<usize>, [wgpu::Buffer; 4], wgpu::BindGroup)>,
}

impl DdgiVolume {
    pub(crate) fn supported(device: &wgpu::Device) -> bool {
        device.limits().max_storage_buffers_per_shader_stage >= 7
    }

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
        let pack_u_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<DdgiPackUniform>() as u64),
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
                storage_ro(9),
                storage_ro(10),
            ],
        });
        let blend_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ddgi_blend_bgl"),
            entries: &[u_entry, storage_ro(1), tex_ro(2, false), tex_ro(3, false), storage_tex(4), storage_tex(5)],
        });
        let relocate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ddgi_relocate_bgl"),
            entries: &[u_entry, storage_ro(1), storage_rw(11)],
        });
        let border_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ddgi_border_bgl"),
            entries: &[tex_ro(1, false), storage_tex(2)],
        });
        let eval_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ddgi_eval_bgl"),
            entries: &[
                u_entry,
                tex_ro(1, true),
                tex_ro(2, true),
                samp_entry(3),
                storage_ro(4),
                storage_rw(5),
                storage_ro(10),
            ],
        });
        let pack_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ddgi_pack_bgl"),
            entries: &[u_entry, pack_u_entry, tex_ro(2, false), tex_ro(3, false), storage_rw(4), storage_ro(10)],
        });

        let sw_body = DDGI_WGSL
            .replace("// __TLAS_BINDING_1__", "@group(0) @binding(1) var<storage, read> tlas_nodes: array<BvhNode>;");
        let source = format!("{TLAS_TRAVERSAL_WGSL}\n{sw_body}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ddgi_wgsl"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let make_pipe = |label: &str, m: &wgpu::ShaderModule, bgl: &wgpu::BindGroupLayout, entry: &str| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: m,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let trace_pipe = make_pipe("ddgi_trace", &module, &trace_bgl, "ddgi_trace_probes");
        let blend_irr_pipe = make_pipe("ddgi_blend_irr", &module, &blend_bgl, "ddgi_blend_irradiance");
        let blend_vis_pipe = make_pipe("ddgi_blend_vis", &module, &blend_bgl, "ddgi_blend_visibility");
        let relocate_pipe = make_pipe("ddgi_relocate", &module, &relocate_bgl, "ddgi_relocate_probes");
        let border_irr_pipe = make_pipe("ddgi_border_irr", &module, &border_bgl, "ddgi_border_irradiance");
        let border_vis_pipe = make_pipe("ddgi_border_vis", &module, &border_bgl, "ddgi_border_visibility");
        let eval_pipe = make_pipe("ddgi_eval", &module, &eval_bgl, "ddgi_eval_points");
        let pack_pipe = make_pipe("ddgi_pack", &module, &pack_bgl, "ddgi_pack_cascade");

        #[cfg(not(target_arch = "wasm32"))]
        let (trace_pipe_hw, trace_bgl_hw) = if device_supports_hardware_ray_query(device) {
            let bgl_hw = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ddgi_trace_bgl_hw"),
                entries: &[
                    u_entry,
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                        count: None,
                    },
                    storage_ro(2),
                    storage_ro(3),
                    storage_ro(4),
                    tex_ro(5, true),
                    tex_ro(6, true),
                    samp_entry(7),
                    storage_rw(8),
                    storage_ro(9),
                    storage_ro(10),
                ],
            });
            let hw_body = DDGI_WGSL
                .replace("// __TLAS_BINDING_1__", "@group(0) @binding(1) var hw_tlas: acceleration_structure;");
            let hw_src = format!("{TLAS_TRAVERSAL_HW_WGSL}\n{hw_body}");
            let hw_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ddgi_wgsl_hw"),
                source: wgpu::ShaderSource::Wgsl(hw_src.into()),
            });
            let p_hw = make_pipe("ddgi_trace_hw", &hw_mod, &bgl_hw, "ddgi_trace_probes");
            (Some(p_hw), Some(bgl_hw))
        } else {
            (None, None)
        };

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ddgi_u"),
            size: std::mem::size_of::<DdgiUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pack_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ddgi_pack_u"),
            size: std::mem::size_of::<DdgiPackUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let local_lights_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ddgi_local_lights"),
            size: (MAX_RT_LOCAL_LIGHTS * std::mem::size_of::<RtLocalLight>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let total_probes = u64::from(nx) * u64::from(ny) * u64::from(nz);
        let total_rays = total_probes * u64::from(config.rays_per_probe);
        let ray_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ddgi_rays"),
            size: total_rays * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let probe_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ddgi_probe_offsets"),
            size: total_probes * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut mapped = probe_offsets.slice(..).get_mapped_range_mut().unwrap();
            let init: Vec<[f32; 4]> = vec![[0.0, 0.0, 0.0, 1.0]; total_probes as usize];
            mapped.copy_from_slice(bytemuck::cast_slice(&init));
        }
        probe_offsets.unmap();
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ddgi_linear_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let irr_curr = AtlasTex::new(device, "ddgi_irr_curr", irr_w, irr_h);
        let irr_tmp = AtlasTex::new(device, "ddgi_irr_tmp", irr_w, irr_h);
        let vis_curr = AtlasTex::new(device, "ddgi_vis_curr", vis_w, vis_h);
        let vis_tmp = AtlasTex::new(device, "ddgi_vis_tmp", vis_w, vis_h);

        let blend_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ddgi_blend_bg"),
            layout: &blend_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ray_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&irr_curr.view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&vis_curr.view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&irr_tmp.view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&vis_tmp.view) },
            ],
        });
        let border_irr_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ddgi_border_irr_bg"),
            layout: &border_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&irr_tmp.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&irr_curr.view) },
            ],
        });
        let border_vis_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ddgi_border_vis_bg"),
            layout: &border_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&vis_tmp.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&vis_curr.view) },
            ],
        });
        let reloc_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ddgi_relocate_bg"),
            layout: &relocate_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ray_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: probe_offsets.as_entire_binding() },
            ],
        });

        Ok(Self {
            config,
            trace_pipe,
            trace_bgl,
            #[cfg(not(target_arch = "wasm32"))]
            trace_pipe_hw,
            #[cfg(not(target_arch = "wasm32"))]
            trace_bgl_hw,
            blend_irr_pipe,
            blend_vis_pipe,
            blend_bg,
            relocate_pipe,
            reloc_bg,
            border_irr_pipe,
            border_vis_pipe,
            border_irr_bg,
            border_vis_bg,
            eval_pipe,
            eval_bgl,
            pack_pipe,
            pack_bgl,
            uniform,
            pack_uniform,
            local_lights_buf,
            ray_buf,
            probe_offsets,
            sampler,
            irr_curr,
            _irr_tmp: irr_tmp,
            vis_curr,
            _vis_tmp: vis_tmp,
            scroll_offset: [0; 3],
            pending_scroll_delta: [0; 3],
            frame_index: 0,
            trace_bg: None,
        })
    }

    pub fn config(&self) -> DdgiGridConfig {
        self.config
    }

    pub fn scroll_offset(&self) -> [i32; 3] {
        self.scroll_offset
    }

    pub fn uses_hardware_ray_query(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.trace_pipe_hw.is_some()
        }
        #[cfg(target_arch = "wasm32")]
        {
            false
        }
    }

    /// Toroidally scrolls the probe grid so its center tracks `target_center` in discrete cell steps,
    /// preserving temporal history for all probes that remain inside the volume and resetting only
    /// newly entered boundary slices.
    pub fn scroll_to_center(&mut self, target_center: [f32; 3]) {
        for (axis, &tc) in target_center.iter().enumerate() {
            let sp = self.config.spacing[axis].max(1e-3);
            let half_ext = 0.5 * (self.config.dims[axis].saturating_sub(1) as f32) * sp;
            let curr_center = self.config.origin[axis] + half_ext;
            let steps = ((tc - curr_center) / sp).round() as i32;
            if steps != 0 {
                self.config.origin[axis] += steps as f32 * sp;
                self.scroll_offset[axis] = self.scroll_offset[axis].wrapping_add(steps);
                self.pending_scroll_delta[axis] = self.pending_scroll_delta[axis].saturating_add(steps);
            }
        }
    }

    /// Number of `vec4f` texel entries this cascade occupies in a flat packed DDGI storage buffer
    /// (irradiance `10x10` + paired visibility `18x18 / 2` per probe).
    pub fn packed_data_vec4_counts(&self) -> (u32, u32) {
        let [nx, ny, nz] = self.config.dims;
        let irr_w = nx * nz * DDGI_IRRADIANCE_TILE;
        let irr_h = ny * DDGI_IRRADIANCE_TILE;
        let vis_w = nx * nz * DDGI_VISIBILITY_TILE;
        let vis_h = ny * DDGI_VISIBILITY_TILE;
        let irr_vec4s = irr_w * irr_h;
        let vis_vec4s = (vis_w * vis_h).div_ceil(2);
        (irr_vec4s, vis_vec4s)
    }

    /// Number of `vec4f` probe-offset entries appended immediately after visibility data in packed storage.
    pub fn packed_probe_offset_vec4_count(&self) -> u32 {
        let [nx, ny, nz] = self.config.dims;
        nx * ny * nz
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
        self.pending_scroll_delta = [0; 3];
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
        self.update_with_lights(device, encoder, scene, params, &[]);
    }

    /// Same as [`DdgiVolume::update`], additionally injecting dynamic point, spot, and area
    /// lights into secondary bounce hit shading.
    pub fn update_with_lights(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &TlasSceneGpu,
        params: &DdgiFrameParams,
        local_lights: &[RtLocalLight],
    ) {
        if params.reset_history {
            self.frame_index = 0;
        }
        let light_count = local_lights.len().min(MAX_RT_LOCAL_LIGHTS);
        let scroll_delta = self.pending_scroll_delta;
        self.pending_scroll_delta = [0; 3];
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
            scroll_and_lights: [
                self.scroll_offset[0],
                self.scroll_offset[1],
                self.scroll_offset[2],
                i32::try_from(light_count).unwrap_or(0),
            ],
            scroll_delta: [scroll_delta[0], scroll_delta[1], scroll_delta[2], 0],
        };
        let u_bytes = bytemuck::bytes_of(&u);
        if light_count > 0 {
            let l_bytes: &[u8] = bytemuck::cast_slice(&local_lights[..light_count]);
            let u_len = u_bytes.len() as u64;
            let l_len = l_bytes.len() as u64;
            let stage = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ddgi_frame_stage"),
                size: u_len + l_len,
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            stage.slice(0..u_len).get_mapped_range_mut().unwrap().copy_from_slice(u_bytes);
            stage.slice(u_len..u_len + l_len).get_mapped_range_mut().unwrap().copy_from_slice(l_bytes);
            stage.unmap();
            encoder.copy_buffer_to_buffer(&stage, 0, &self.uniform, 0, u_len);
            encoder.copy_buffer_to_buffer(&stage, u_len, &self.local_lights_buf, 0, l_len);
        } else {
            encode_copy(device, encoder, &self.uniform, u_bytes);
        }

        let [nx, ny, nz] = self.config.dims;
        let total_rays = nx * ny * nz * self.config.rays_per_probe;

        // 1. Trace probe rays (hardware ray-query on native RT GPUs when available, else portable compute BVH).
        #[allow(unused_mut)]
        let mut dispatched_hw = false;
        #[cfg(not(target_arch = "wasm32"))]
        if let (Some(pipe_hw), Some(bgl_hw), Some(hw_tlas)) = (&self.trace_pipe_hw, &self.trace_bgl_hw, &scene.hw_tlas)
        {
            let hw_slots = Some(hw_tlas.get().len());
            let hit = matches!(
                &self.trace_bg,
                Some((slots, bufs, _))
                    if *slots == hw_slots
                        && bufs[0] == scene.tlas_nodes_buf
                        && bufs[1] == scene.tlas_instances_buf
                        && bufs[2] == scene.blas_nodes_buf
                        && bufs[3] == scene.blas_tris_buf
            );
            if !hit {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ddgi_trace_bg_hw"),
                    layout: bgl_hw,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: hw_tlas.as_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: scene.tlas_instances_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: scene.blas_nodes_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: scene.blas_tris_buf.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&self.irr_curr.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&self.vis_curr.view),
                        },
                        wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                        wgpu::BindGroupEntry { binding: 8, resource: self.ray_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 9, resource: self.local_lights_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 10, resource: self.probe_offsets.as_entire_binding() },
                    ],
                });
                self.trace_bg = Some((
                    hw_slots,
                    [
                        scene.tlas_nodes_buf.clone(),
                        scene.tlas_instances_buf.clone(),
                        scene.blas_nodes_buf.clone(),
                        scene.blas_tris_buf.clone(),
                    ],
                    bg,
                ));
            }
            let (_, _, trace_bg) = self.trace_bg.as_ref().unwrap();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ddgi_trace_hw"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe_hw);
            pass.set_bind_group(0, trace_bg, &[]);
            pass.dispatch_workgroups(total_rays.div_ceil(64), 1, 1);
            dispatched_hw = true;
        }
        if !dispatched_hw {
            let hit = matches!(
                &self.trace_bg,
                Some((slots, bufs, _))
                    if slots.is_none()
                        && bufs[0] == scene.tlas_nodes_buf
                        && bufs[1] == scene.tlas_instances_buf
                        && bufs[2] == scene.blas_nodes_buf
                        && bufs[3] == scene.blas_tris_buf
            );
            if !hit {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ddgi_trace_bg"),
                    layout: &self.trace_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: scene.tlas_nodes_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: scene.tlas_instances_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: scene.blas_nodes_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: scene.blas_tris_buf.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&self.irr_curr.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&self.vis_curr.view),
                        },
                        wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                        wgpu::BindGroupEntry { binding: 8, resource: self.ray_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 9, resource: self.local_lights_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 10, resource: self.probe_offsets.as_entire_binding() },
                    ],
                });
                self.trace_bg = Some((
                    None,
                    [
                        scene.tlas_nodes_buf.clone(),
                        scene.tlas_instances_buf.clone(),
                        scene.blas_nodes_buf.clone(),
                        scene.blas_tris_buf.clone(),
                    ],
                    bg,
                ));
            }
            let (_, _, trace_bg) = self.trace_bg.as_ref().unwrap();
            let mut pass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ddgi_trace"), timestamp_writes: None });
            pass.set_pipeline(&self.trace_pipe);
            pass.set_bind_group(0, trace_bg, &[]);
            pass.dispatch_workgroups(total_rays.div_ceil(64), 1, 1);
        }

        // 2. Blend interior octahedral texels into tmp atlases.
        let cols = nx * nz;
        {
            let mut pass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ddgi_blend"), timestamp_writes: None });
            pass.set_pipeline(&self.blend_irr_pipe);
            pass.set_bind_group(0, &self.blend_bg, &[]);
            pass.dispatch_workgroups(
                (cols * DDGI_IRRADIANCE_INNER).div_ceil(8),
                (ny * DDGI_IRRADIANCE_INNER).div_ceil(8),
                1,
            );

            pass.set_pipeline(&self.blend_vis_pipe);
            pass.set_bind_group(0, &self.blend_bg, &[]);
            pass.dispatch_workgroups(
                (cols * DDGI_VISIBILITY_INNER).div_ceil(8),
                (ny * DDGI_VISIBILITY_INNER).div_ceil(8),
                1,
            );
        }

        // 3. Copy interior + wrap borders from tmp -> curr so curr is always sampling-ready!
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ddgi_borders"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.border_irr_pipe);
            pass.set_bind_group(0, &self.border_irr_bg, &[]);
            pass.dispatch_workgroups(
                (cols * DDGI_IRRADIANCE_TILE).div_ceil(8),
                (ny * DDGI_IRRADIANCE_TILE).div_ceil(8),
                1,
            );

            pass.set_pipeline(&self.border_vis_pipe);
            pass.set_bind_group(0, &self.border_vis_bg, &[]);
            pass.dispatch_workgroups(
                (cols * DDGI_VISIBILITY_TILE).div_ceil(8),
                (ny * DDGI_VISIBILITY_TILE).div_ceil(8),
                1,
            );
        }

        // 4. Relocate probes trapped inside solid geometry or hugging surfaces for subsequent frames.
        let total_probes = nx * ny * nz;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ddgi_relocate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.relocate_pipe);
            pass.set_bind_group(0, &self.reloc_bg, &[]);
            pass.dispatch_workgroups(total_probes.div_ceil(64), 1, 1);
        }

        self.frame_index = self.frame_index.saturating_add(1);
    }

    /// Reads back the current GPU probe relocation offsets (`[dx, dy, dz, active]`) in storage index order.
    pub fn read_probe_offsets(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<[f32; 4]>> {
        let [nx, ny, nz] = self.config.dims;
        let total_probes = (nx * ny * nz) as usize;
        let size = (total_probes * 16) as u64;
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ddgi_offsets_rb"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.probe_offsets, 0, &rb, 0, size);
        queue.submit(Some(enc.finish()));
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap()?;
        let data = slice.get_mapped_range()?;
        let out = bytemuck::cast_slice::<u8, [f32; 4]>(&data).to_vec();
        drop(data);
        rb.unmap();
        Ok(out)
    }

    /// Packs this cascade's metadata, irradiance atlas, visibility atlas, and probe offsets into a
    /// flat `array<vec4f>` storage buffer so forward PBR materials (`environment.wgsl`) and RT
    /// reflections can sample DDGI directly without consuming any extra sampled-texture slots.
    #[allow(clippy::too_many_arguments)]
    pub fn pack_cascade(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dst_buf: &wgpu::Buffer,
        cascade_idx: u32,
        cascade_count: u32,
        irr_offset_vec4s: u32,
        vis_offset_vec4s: u32,
        strength: f32,
    ) {
        let pu = DdgiPackUniform {
            meta: [cascade_idx, cascade_count, irr_offset_vec4s, vis_offset_vec4s],
            strength: [strength.max(0.0), 0.0, 0.0, 0.0],
        };
        encode_copy(device, encoder, &self.pack_uniform, bytemuck::bytes_of(&pu));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ddgi_pack_bg"),
            layout: &self.pack_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.pack_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.irr_curr.view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.vis_curr.view) },
                wgpu::BindGroupEntry { binding: 4, resource: dst_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.probe_offsets.as_entire_binding() },
            ],
        });
        let (irr_v4, vis_v4) = self.packed_data_vec4_counts();
        let off_v4 = self.packed_probe_offset_vec4_count();
        let work = irr_v4.max(vis_v4).max(off_v4).max(1);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ddgi_pack_cascade"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pack_pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(work.div_ceil(64), 1, 1);
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
                wgpu::BindGroupEntry { binding: 10, resource: self.probe_offsets.as_entire_binding() },
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

        // Two-chamber scene divided by an opaque vertical wall at x = 0 (all outer walls face INWARD):
        // Left chamber (x in [-4, 0], y in [0, 4], z in [-4, 4]):
        //   - Bright emissive red wall at x = -4 (normal +X)
        //   - Closed white floor/ceiling/back/front
        // Right chamber (x in [0, 4], y in [0, 4], z in [-4, 4]):
        //   - Completely dark closed box separated by thin dividing wall at x = 0!
        let blases = vec![
            // 0: Floor y = 0, normal +Y
            quad_blas([-4.0, 0.0, -4.0], [-4.0, 0.0, 4.0], [4.0, 0.0, 4.0], [4.0, 0.0, -4.0]),
            // 1: Ceiling y = 4, normal -Y
            quad_blas([-4.0, 4.0, -4.0], [4.0, 4.0, -4.0], [4.0, 4.0, 4.0], [-4.0, 4.0, 4.0]),
            // 2: Left emissive RED wall x = -4, normal +X
            quad_blas([-4.0, 0.0, -4.0], [-4.0, 4.0, -4.0], [-4.0, 4.0, 4.0], [-4.0, 0.0, 4.0]),
            // 3: Dividing opaque wall at x = 0 (two opposing faces so both chambers hit front faces)
            Blas::build(&[
                BlasTriangle::flat([-0.02, 0.0, -4.0], [-0.02, 0.0, 4.0], [-0.02, 4.0, 4.0]),
                BlasTriangle::flat([-0.02, 0.0, -4.0], [-0.02, 4.0, 4.0], [-0.02, 4.0, -4.0]),
                BlasTriangle::flat([0.02, 0.0, -4.0], [0.02, 4.0, -4.0], [0.02, 4.0, 4.0]),
                BlasTriangle::flat([0.02, 0.0, -4.0], [0.02, 4.0, 4.0], [0.02, 0.0, 4.0]),
            ])
            .unwrap(),
            // 4: Right dark wall x = 4, normal -X
            quad_blas([4.0, 0.0, -4.0], [4.0, 0.0, 4.0], [4.0, 4.0, 4.0], [4.0, 4.0, -4.0]),
            // 5: Back wall z = -4, normal +Z
            quad_blas([-4.0, 0.0, -4.0], [4.0, 0.0, -4.0], [4.0, 4.0, -4.0], [-4.0, 4.0, -4.0]),
            // 6: Front wall z = 4, normal -Z
            quad_blas([-4.0, 0.0, 4.0], [-4.0, 4.0, 4.0], [4.0, 4.0, 4.0], [4.0, 0.0, 4.0]),
        ];
        let mut red_wall = TlasInstance::new(2, ID);
        red_wall.albedo = [0.9, 0.05, 0.05];
        red_wall.emissive = [6.0, 0.1, 0.1];

        let dark_wall = |bi: u32| {
            let mut inst = TlasInstance::new(bi, ID);
            inst.albedo = [0.75, 0.75, 0.75];
            inst
        };

        let instances =
            vec![dark_wall(0), dark_wall(1), red_wall, dark_wall(3), dark_wall(4), dark_wall(5), dark_wall(6)];
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

    #[test]
    fn gpu_ddgi_local_light_bounce_toroidal_scroll_and_inside_solid_classification() {
        let (device, queue) = crate::testing::device_with_optional().expect("GPU adapter required");
        use crate::ray_tracing::GpuBlasPool;

        // Open-ceiling white room with a bright green local point light (zero sun, zero emissive geometry!)
        // PLUS a closed solid box surrounding probe (x=2, y=1, z=0) to verify backface inside-solid classification.
        let floor_tris = [
            BlasTriangle::flat([-8.0, 0.0, -8.0], [-8.0, 0.0, 8.0], [8.0, 0.0, 8.0]),
            BlasTriangle::flat([-8.0, 0.0, -8.0], [8.0, 0.0, 8.0], [8.0, 0.0, -8.0]),
        ];
        // Closed solid cube centered at origin with outward normals (half-extent 0.75).
        let s = 0.75f32;
        let cube_tris = [
            // +X
            BlasTriangle::flat([s, -s, -s], [s, s, -s], [s, s, s]),
            BlasTriangle::flat([s, -s, -s], [s, s, s], [s, -s, s]),
            // -X
            BlasTriangle::flat([-s, -s, s], [-s, s, s], [-s, s, -s]),
            BlasTriangle::flat([-s, -s, s], [-s, s, -s], [-s, -s, -s]),
            // +Y
            BlasTriangle::flat([-s, s, -s], [-s, s, s], [s, s, s]),
            BlasTriangle::flat([-s, s, -s], [s, s, s], [s, s, -s]),
            // -Y
            BlasTriangle::flat([-s, -s, s], [-s, -s, -s], [s, -s, -s]),
            BlasTriangle::flat([-s, -s, s], [s, -s, -s], [s, -s, s]),
            // +Z
            BlasTriangle::flat([-s, -s, s], [s, -s, s], [s, s, s]),
            BlasTriangle::flat([-s, -s, s], [s, s, s], [-s, s, s]),
            // -Z
            BlasTriangle::flat([s, -s, -s], [-s, -s, -s], [-s, s, -s]),
            BlasTriangle::flat([s, -s, -s], [-s, s, -s], [s, s, -s]),
        ];

        let mut pool = GpuBlasPool::new(&device);
        let mut enc0 = device.create_command_encoder(&Default::default());
        let bi_floor = pool.ensure_blas(&device, &mut enc0, 1, &floor_tris).unwrap().unwrap();
        let bi_cube = pool.ensure_blas(&device, &mut enc0, 2, &cube_tris).unwrap().unwrap();
        let mut floor_inst = TlasInstance::new(bi_floor, ID);
        floor_inst.albedo = [0.9, 0.9, 0.9];
        // Solid cube centered right on probe (2.0, 1.0, 0.0).
        let cube_xform = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0];
        let mut cube_inst = TlasInstance::new(bi_cube, cube_xform);
        cube_inst.albedo = [0.1, 0.1, 0.1];
        pool.update_tlas(&device, &mut enc0, &[floor_inst, cube_inst]).unwrap();
        queue.submit(Some(enc0.finish()));
        let scene_gpu = pool.snapshot_scene();

        let cfg = DdgiGridConfig {
            origin: [-2.0, 1.0, -2.0],
            spacing: [2.0, 2.0, 2.0],
            dims: [4, 2, 3],
            rays_per_probe: 128,
            hysteresis: 0.7,
            max_ray_distance: 20.0,
            ..Default::default()
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
        // Green point light hovering above the white floor at (0, 2, 0).
        let green_point = RtLocalLight {
            position_range: [0.0, 2.0, 0.0, 12.0],
            color_outer_cos: [0.0, 24.0, 0.0, -1.0],
            direction_inner_cos: [0.0, -1.0, 0.0, -1.0],
        };

        for _ in 0..4 {
            let mut enc = device.create_command_encoder(&Default::default());
            ddgi.update_with_lights(&device, &mut enc, &scene_gpu, &frame_params, &[green_point]);
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }

        // Query a downward-facing ceiling surface above the floor: receives bounced green light from the floor!
        let bounce = ddgi
            .evaluate_points(
                &device,
                &queue,
                &[DdgiQueryPoint {
                    pos: [0.0, 2.5, 0.0, 0.0],
                    normal: [0.0, -1.0, 0.0, 0.0],
                    view: [0.0, -1.0, 0.0, 0.0],
                }],
            )
            .unwrap()[0];
        assert!(
            bounce[1] > 0.2 && bounce[1] > bounce[0] * 3.0 && bounce[1] > bounce[2] * 3.0,
            "local point light must inject colored indirect bounce into DDGI (hw_rt={}): {bounce:?}",
            ddgi.uses_hardware_ray_query()
        );

        // Toroidally scroll the probe volume by +2m on X without recreating the volume!
        let prev_origin = ddgi.config().origin;
        ddgi.scroll_to_center([prev_origin[0] + 5.0, prev_origin[1] + 1.0, prev_origin[2] + 2.0]);
        assert!(
            ddgi.scroll_offset()[0] > 0 && ddgi.config().origin[0] > prev_origin[0],
            "toroidal scroll must advance origin and scroll_offset in discrete steps"
        );
        for _ in 0..3 {
            let mut enc = device.create_command_encoder(&Default::default());
            ddgi.update_with_lights(&device, &mut enc, &scene_gpu, &frame_params, &[green_point]);
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let scrolled_bounce = ddgi
            .evaluate_points(
                &device,
                &queue,
                &[DdgiQueryPoint {
                    pos: [1.0, 2.5, 0.0, 0.0],
                    normal: [0.0, -1.0, 0.0, 0.0],
                    view: [0.0, -1.0, 0.0, 0.0],
                }],
            )
            .unwrap()[0];
        assert!(
            scrolled_bounce[1] > 0.15 && scrolled_bounce[1] > scrolled_bounce[0] * 2.5,
            "toroidally scrolled DDGI volume must maintain accurate green bounce irradiance: {scrolled_bounce:?}"
        );
    }

    #[test]
    fn gpu_ddgi_probe_relocation_and_adaptive_hysteresis_witness() {
        let (device, queue) = crate::testing::device().expect("GPU adapter required");

        // Closed solid wall slab spanning x in [-0.25, +0.25], y in [0, 3], z in [-3, 3] with outward normals.
        let (hx, hy0, hy1, hz) = (0.25f32, 0.0f32, 3.0f32, 3.0f32);
        let wall_tris = [
            // +X
            BlasTriangle::flat([hx, hy0, -hz], [hx, hy1, -hz], [hx, hy1, hz]),
            BlasTriangle::flat([hx, hy0, -hz], [hx, hy1, hz], [hx, hy0, hz]),
            // -X
            BlasTriangle::flat([-hx, hy0, hz], [-hx, hy1, hz], [-hx, hy1, -hz]),
            BlasTriangle::flat([-hx, hy0, hz], [-hx, hy1, -hz], [-hx, hy0, -hz]),
            // +Y
            BlasTriangle::flat([-hx, hy1, -hz], [-hx, hy1, hz], [hx, hy1, hz]),
            BlasTriangle::flat([-hx, hy1, -hz], [hx, hy1, hz], [hx, hy1, -hz]),
            // -Y
            BlasTriangle::flat([-hx, hy0, hz], [-hx, hy0, -hz], [hx, hy0, -hz]),
            BlasTriangle::flat([-hx, hy0, hz], [hx, hy0, -hz], [hx, hy0, hz]),
            // +Z
            BlasTriangle::flat([-hx, hy0, hz], [hx, hy0, hz], [hx, hy1, hz]),
            BlasTriangle::flat([-hx, hy0, hz], [hx, hy1, hz], [-hx, hy1, hz]),
            // -Z
            BlasTriangle::flat([hx, hy0, -hz], [-hx, hy0, -hz], [-hx, hy1, -hz]),
            BlasTriangle::flat([hx, hy0, -hz], [-hx, hy1, -hz], [hx, hy1, -hz]),
        ];
        let floor_blas = quad_blas([-6.0, 0.0, -6.0], [-6.0, 0.0, 6.0], [6.0, 0.0, 6.0], [6.0, 0.0, -6.0]);
        let wall_blas = Blas::build(&wall_tris).unwrap();
        let mut floor_inst = TlasInstance::new(0, ID);
        floor_inst.albedo = [0.85, 0.85, 0.85];
        let mut wall_inst = TlasInstance::new(1, ID);
        wall_inst.albedo = [0.5, 0.5, 0.5];
        let scene = TlasScene::build(&[floor_blas, wall_blas], &[floor_inst, wall_inst]).unwrap();
        let scene_gpu = TlasSceneGpu::upload(&device, &scene).unwrap();

        // Probe (1, 0, 1) sits at (0.0, 1.0, 0.0) -- dead center inside the solid wall slab!
        let cfg = DdgiGridConfig {
            origin: [-2.0, 1.0, -2.0],
            spacing: [2.0, 2.0, 2.0],
            dims: [3, 2, 3],
            rays_per_probe: 128,
            hysteresis: 0.95,
            max_ray_distance: 20.0,
            ..Default::default()
        };
        let mut ddgi = DdgiVolume::new(&device, cfg).unwrap();
        let dark_params = DdgiFrameParams {
            sun_direction: [0.0, 1.0, 0.0],
            sun_color: [0.0, 0.0, 0.0],
            sky_zenith: [0.02, 0.02, 0.02],
            sky_horizon: [0.02, 0.02, 0.02],
            visibility_mask: u32::MAX,
            reset_history: false,
        };
        for _ in 0..4 {
            let mut enc = device.create_command_encoder(&Default::default());
            ddgi.update(&device, &mut enc, &scene_gpu, &dark_params);
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let offsets = ddgi.read_probe_offsets(&device, &queue).unwrap();
        let trapped_idx = 3 + 1; // (ix=1, iy=0, iz=1) -> (0*3 + 1)*3 + 1 = 4
        let trapped_off = offsets[trapped_idx];
        assert!(
            trapped_off[0].abs() > hx && trapped_off[0].abs() <= 0.45 * 2.0 + 1e-3,
            "DDGI relocation must push probe outside thin wall [-{hx}, +{hx}] within cell bound: {trapped_off:?}"
        );
        assert!(
            trapped_off[3] > 0.5,
            "relocated probe must become active after escaping solid geometry: {trapped_off:?}"
        );

        // Adaptive perceptual hysteresis witness: flip sun ON brightly with hysteresis=0.95!
        let bright_params = DdgiFrameParams {
            sun_direction: [0.2, 0.95, 0.2],
            sun_color: [12.0, 12.0, 12.0],
            sky_zenith: [0.6, 0.8, 1.2],
            sky_horizon: [0.8, 0.9, 1.0],
            visibility_mask: u32::MAX,
            reset_history: false,
        };
        for _ in 0..2 {
            let mut enc = device.create_command_encoder(&Default::default());
            ddgi.update(&device, &mut enc, &scene_gpu, &bright_params);
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let q =
            [DdgiQueryPoint { pos: [-2.0, 2.0, 0.0, 0.0], normal: [0.0, -1.0, 0.0, 0.0], view: [0.0, -1.0, 0.0, 0.0] }];
        let after_2 = ddgi.evaluate_points(&device, &queue, &q).unwrap()[0];
        for _ in 0..14 {
            let mut enc = device.create_command_encoder(&Default::default());
            ddgi.update(&device, &mut enc, &scene_gpu, &bright_params);
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let steady = ddgi.evaluate_points(&device, &queue, &q).unwrap()[0];
        assert!(
            steady[0] > 0.5 && after_2[0] > steady[0] * 0.40,
            "adaptive perceptual hysteresis must converge rapidly after large lighting jump (h=0.95): after_2={after_2:?}, steady={steady:?}"
        );
    }
}
