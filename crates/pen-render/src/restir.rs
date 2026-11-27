//! GPU ReSTIR direct and diffuse indirect lighting from Pen's thin G-buffer.

use std::collections::HashMap;

use anyhow::{Result, ensure};
#[cfg(test)]
use wgpu::util::DeviceExt;

#[cfg(not(target_arch = "wasm32"))]
use crate::ray_tracing::TLAS_TRAVERSAL_HW_WGSL;
use crate::ray_tracing::{RtLocalLight, TlasSceneGpu, device_supports_hardware_ray_query};
use crate::sharc::{SHARC_COMMON_WGSL, SharcCache};

pub const RESTIR_WGSL: &str = crate::wgsl!("restir.wgsl");
pub const RESTIR_MAX_LIGHTS: usize = 256;

const RESTIR_SHARC_DECLS_WGSL: &str = r"
@group(0) @binding(21) var<uniform> sharc_u: SharcUniform;
@group(0) @binding(22) var<storage, read_write> sharc_keys: array<atomic<u32>>;
@group(0) @binding(23) var<storage, read_write> sharc_accum: array<atomic<u32>>;
@group(0) @binding(24) var<storage, read> sharc_resolved: array<vec4f>;
";

const RESTIR_SHARC_BOUNCE_SNIPPET: &str = r"
      if (u.sky_zenith_and_bounce.w > 0.0 && shaded.a < 1.0) {
        let cached = sharc_query(pos, nrm);
        if (cached.a > 0.5) {
          radiance = safe_rgb(radiance + diff_albedo * cached.rgb * ((1.0 - shaded.a) * u.sky_zenith_and_bounce.w));
        }
      }
";

const RESTIR_SHARC_DEPOSIT_SNIPPET: &str = r"
  if (g.valid && (u.flags.w & 2u) != 0u) {
    sharc_accumulate(g.pos, g.nrm, rgb);
  }
";

const RESTIR_SHARC_WGSL: &str = r"
struct RestirUniform {
  inverse_view_projection: mat4x4f,
  prev_view_projection: mat4x4f,
  camera_and_clear: vec4f,
  previous_camera: vec4f,
  sun_and_max_distance: vec4f,
  sun_color_and_bias: vec4f,
  sky_zenith_and_bounce: vec4f,
  sky_horizon: vec4f,
  scene_counts_and_mask: vec4u,
  size_lights_frame: vec4u,
  flags: vec4u,
};

struct GiReservoir {
  sample_pos_m: vec4f,
  sample_nrm_w: vec4f,
  radiance_wsum: vec4f,
  vis_pad: vec4f,
};

@group(0) @binding(0) var<uniform> u: RestirUniform;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_roughness: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> gi_reservoirs: array<GiReservoir>;
@group(0) @binding(4) var<uniform> sharc_u: SharcUniform;
@group(0) @binding(5) var<storage, read_write> sharc_keys: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> sharc_accum: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> sharc_resolved: array<vec4f>;
@group(0) @binding(8) var out_gi_tex: texture_2d<f32>;

const EPS: f32 = 1e-6;

fn rs_finite_nonnegative(x: f32, ceiling: f32) -> f32 {
  return select(clamp(x, 0.0, ceiling), 0.0, (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u);
}

fn rs_safe_rgb(rgb: vec3f) -> vec3f {
  return vec3f(rs_finite_nonnegative(rgb.x, 16384.0), rs_finite_nonnegative(rgb.y, 16384.0),
               rs_finite_nonnegative(rgb.z, 16384.0));
}

fn rs_luma(rgb: vec3f) -> f32 {
  return dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
}

fn rs_input_pixel(pixel: vec2i, dims: vec2i) -> vec2i {
  let dispatch_dims = vec2i(u.size_lights_frame.xy);
  let scaled = vec2i((vec2f(pixel) + 0.5) * vec2f(dims) / vec2f(dispatch_dims));
  return select(clamp(scaled, vec2i(0), dims - 1), pixel, all(dims == dispatch_dims));
}

struct RsGeom {
  pos: vec3f,
  nrm: vec3f,
  valid: bool,
};

fn rs_load_geometry(pixel: vec2i) -> RsGeom {
  let dims = vec2i(textureDimensions(normal_roughness, 0));
  let g_px = rs_input_pixel(pixel, dims);
  let depth = textureLoad(depth_tex, g_px, 0).x;
  let nr = textureLoad(normal_roughness, g_px, 0);
  if (!(dot(nr.xyz, nr.xyz) >= 0.01) || !(abs(depth - u.camera_and_clear.w) > 1e-6)) {
    return RsGeom(vec3f(0.0), vec3f(0.0), false);
  }
  let uv = (vec2f(g_px) + 0.5) / vec2f(dims);
  let clip = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), depth, 1.0);
  let world = u.inverse_view_projection * clip;
  if (!(abs(world.w) > 1e-8)) {
    return RsGeom(vec3f(0.0), vec3f(0.0), false);
  }
  return RsGeom(world.xyz / world.w, normalize(nr.xyz), true);
}

@compute @workgroup_size(8, 8, 1)
fn restir_gi_sharc_bounce(@builtin(global_invocation_id) id: vec3u) {
  let ext = u.size_lights_frame.xy;
  if (any(id.xy >= ext) || u.sky_zenith_and_bounce.w <= 0.0) {
    return;
  }
  let pixel = vec2i(id.xy);
  let index = id.y * ext.x + id.x;
  var r = gi_reservoirs[index];
  if (r.vis_pad.x <= 0.5 || r.vis_pad.y >= u.sun_and_max_distance.w) {
    return;
  }
  let cached = sharc_query(r.sample_pos_m.xyz, r.sample_nrm_w.xyz);
  if (cached.a <= 0.5) {
    return;
  }
  let packed = unpack4x8unorm(bitcast<u32>(r.vis_pad.z));
  let add = packed.rgb * cached.rgb * ((1.0 - packed.a) * u.sky_zenith_and_bounce.w);
  let new_rad = rs_safe_rgb(r.radiance_wsum.rgb + add);
  let g = rs_load_geometry(pixel);
  if (!g.valid) {
    return;
  }
  let delta = r.sample_pos_m.xyz - g.pos;
  let dir = delta / max(length(delta), EPS);
  var cos_term = max(dot(g.nrm, dir), 0.0);
  if (dot(r.sample_nrm_w.xyz, -dir) <= 0.0) {
    cos_term = 0.0;
  }
  r.radiance_wsum = vec4f(new_rad, rs_luma(new_rad) * cos_term * r.sample_nrm_w.w);
  gi_reservoirs[index] = r;
}

@compute @workgroup_size(8, 8, 1)
fn restir_gi_sharc_deposit(@builtin(global_invocation_id) id: vec3u) {
  let ext = u.size_lights_frame.xy;
  if (any(id.xy >= ext)) {
    return;
  }
  let pixel = vec2i(id.xy);
  let gi_val = textureLoad(out_gi_tex, pixel, 0);
  if (gi_val.a <= 0.5) {
    return;
  }
  let g = rs_load_geometry(pixel);
  if (!g.valid) {
    return;
  }
  sharc_accumulate(g.pos, g.nrm, gi_val.rgb);
}
";

const OUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DI_STRIDE: u64 = 32;
const GI_STRIDE: u64 = 64;
const MIN_RAY_DISTANCE: f32 = 0.001;

#[derive(Clone, Copy, Debug)]
pub struct RestirParams {
    /// Current and previous column-major WebGPU view-projection matrices.
    pub view_projection: [f32; 16],
    pub prev_view_projection: [f32; 16],
    pub camera_pos: [f32; 3],
    pub depth_clear: f32,
    /// World-space direction toward the sun.
    pub sun_direction: [f32; 3],
    pub sun_color: [f32; 3],
    pub sky_zenith: [f32; 3],
    pub sky_horizon: [f32; 3],
    pub max_ray_distance: f32,
    pub normal_bias: f32,
    pub visibility_mask: u32,
    pub frame_index: u32,
    pub enable_di: bool,
    pub enable_gi: bool,
    pub reset_history: bool,
    /// Strength of optional DDGI irradiance at secondary hits.
    pub gi_bounce_scale: f32,
}

pub struct RestirInputs<'a> {
    pub depth: &'a wgpu::TextureView,
    pub normal_roughness: &'a wgpu::TextureView,
    pub albedo_metallic: &'a wgpu::TextureView,
    /// Rg16Float or Rgba16Float, with current UV minus previous UV in xy.
    /// Missing motion or abs(x) >= 100 uses matrix reprojection.
    pub motion: Option<&'a wgpu::TextureView>,
    /// Packed cascades from DDGI, with STORAGE usage.
    pub ddgi_packed: Option<&'a wgpu::Buffer>,
    /// Optional world-space SHaRC cache for unbounded multi-bounce GI lookup & deposit.
    pub sharc: Option<&'a SharcCache>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    inverse_view_projection: [f32; 16],
    prev_view_projection: [f32; 16],
    camera_and_clear: [f32; 4],
    previous_camera: [f32; 4],
    sun_and_max_distance: [f32; 4],
    sun_color_and_bias: [f32; 4],
    sky_zenith_and_bounce: [f32; 4],
    sky_horizon: [f32; 4],
    scene_counts_and_mask: [u32; 4],
    size_lights_frame: [u32; 4],
    flags: [u32; 4],
}

struct Image {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Image {
    fn new(device: &wgpu::Device, size: (u32, u32), label: &str, extra_usage: wgpu::TextureUsages) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OUT_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | extra_usage,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        Self { texture, view }
    }
}

struct Reservoirs {
    scratch: wgpu::Buffer,
    history: [wgpu::Buffer; 2],
}

impl Reservoirs {
    fn new(device: &wgpu::Device, bytes: u64, labels: [&str; 3]) -> Self {
        let make = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        Self { scratch: make(labels[0]), history: [make(labels[1]), make(labels[2])] }
    }
}

struct Frame {
    size: (u32, u32),
    di: Image,
    gi: Image,
    depth_nrm: [Image; 2],
    di_reservoirs: Option<Reservoirs>,
    gi_reservoirs: Option<Reservoirs>,
    previous: usize,
    previous_camera: [f32; 3],
}

impl Frame {
    fn new(device: &wgpu::Device, size: (u32, u32)) -> Self {
        let out = |label| Image::new(device, size, label, wgpu::TextureUsages::COPY_SRC);
        let history = |label| Image::new(device, size, label, wgpu::TextureUsages::empty());
        Self {
            size,
            di: out("restir out_di"),
            gi: out("restir out_gi"),
            depth_nrm: [history("restir geometry A"), history("restir geometry B")],
            di_reservoirs: None,
            gi_reservoirs: None,
            previous: 0,
            previous_camera: [0.0; 3],
        }
    }
}

struct Pass {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    entry: &'static str,
    bindings: &'static [u32],
}

impl Pass {
    fn new(
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        entry: &'static str,
        bindings: &'static [u32],
        hardware: bool,
        native_depth: bool,
    ) -> Self {
        let buffer = |ty| wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None };
        let storage = |read_only| buffer(wgpu::BufferBindingType::Storage { read_only });
        let entries: Vec<_> = bindings
            .iter()
            .map(|&binding| {
                let ty = match binding {
                    0 | 21 => buffer(wgpu::BufferBindingType::Uniform),
                    #[cfg(not(target_arch = "wasm32"))]
                    1 if hardware => wgpu::BindingType::AccelerationStructure { vertex_return: false },
                    1..=4 | 9..=10 | 13..=14 | 17..=18 | 24 => storage(true),
                    15 | 19 | 22 | 23 => storage(false),
                    5..=8 | 11 => wgpu::BindingType::Texture {
                        sample_type: if binding == 5 && native_depth {
                            wgpu::TextureSampleType::Depth
                        } else {
                            wgpu::TextureSampleType::Float { filterable: false }
                        },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    12 | 16 | 20 => wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: OUT_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    _ => unreachable!("unknown ReSTIR binding {binding}"),
                };
                wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None }
            })
            .collect();
        #[cfg(target_arch = "wasm32")]
        let _ = hardware;
        let layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(entry), entries: &entries });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { layout, pipeline, entry, bindings }
    }

    fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &[Option<wgpu::BindingResource<'_>>; 25],
        size: (u32, u32),
    ) {
        let entries: Vec<_> = self
            .bindings
            .iter()
            .map(|&binding| wgpu::BindGroupEntry { binding, resource: resources[binding as usize].clone().unwrap() })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(self.entry),
            layout: &self.layout,
            entries: &entries,
        });
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(self.entry), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
    }
}

struct SharcAuxPipelines {
    bounce_bgl: wgpu::BindGroupLayout,
    bounce_pipe: wgpu::ComputePipeline,
    deposit_bgl: wgpu::BindGroupLayout,
    deposit_pipe: wgpu::ComputePipeline,
}

impl SharcAuxPipelines {
    fn new(device: &wgpu::Device, native_depth: bool) -> Self {
        let mut body = RESTIR_SHARC_WGSL.to_string();
        if native_depth {
            body = body
                .replace("var depth_tex: texture_2d<f32>;", "var depth_tex: texture_depth_2d;")
                .replace("textureLoad(depth_tex, g_px, 0).x", "textureLoad(depth_tex, g_px, 0)");
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("restir sharc"),
            source: wgpu::ShaderSource::Wgsl(format!("{SHARC_COMMON_WGSL}\n{body}").into()),
        });
        let buf = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let tex = |binding, depth| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: if depth {
                    wgpu::TextureSampleType::Depth
                } else {
                    wgpu::TextureSampleType::Float { filterable: false }
                },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bounce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("restir_gi_sharc_bounce"),
            entries: &[
                buf(0, wgpu::BufferBindingType::Uniform),
                tex(1, native_depth),
                tex(2, false),
                buf(3, wgpu::BufferBindingType::Storage { read_only: false }),
                buf(4, wgpu::BufferBindingType::Uniform),
                buf(5, wgpu::BufferBindingType::Storage { read_only: false }),
                buf(7, wgpu::BufferBindingType::Storage { read_only: true }),
            ],
        });
        let deposit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("restir_gi_sharc_deposit"),
            entries: &[
                buf(0, wgpu::BufferBindingType::Uniform),
                tex(1, native_depth),
                tex(2, false),
                buf(4, wgpu::BufferBindingType::Uniform),
                buf(5, wgpu::BufferBindingType::Storage { read_only: false }),
                buf(6, wgpu::BufferBindingType::Storage { read_only: false }),
                tex(8, false),
            ],
        });
        let make = |label, layout: &wgpu::BindGroupLayout| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(label),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            bounce_pipe: make("restir_gi_sharc_bounce", &bounce_bgl),
            deposit_pipe: make("restir_gi_sharc_deposit", &deposit_bgl),
            bounce_bgl,
            deposit_bgl,
        }
    }
}

struct Pipelines {
    module: wgpu::ShaderModule,
    di: Option<[Pass; 3]>,
    gi: Option<[Pass; 3]>,
    sharc_aux: Option<SharcAuxPipelines>,
    geometry: Pass,
    fuse_bounce: bool,
    fuse_deposit: bool,
}

impl Pipelines {
    fn new(device: &wgpu::Device, hardware: bool, native_depth: bool, fuse_bounce: bool, fuse_deposit: bool) -> Self {
        let (traversal, declaration) = (
            crate::deferred_wgsl!("tlas_traversal.wgsl"),
            "@group(0) @binding(1) var<storage, read> tlas_nodes: array<BvhNode>;",
        );
        #[cfg(not(target_arch = "wasm32"))]
        let (traversal, declaration) = if hardware {
            (TLAS_TRAVERSAL_HW_WGSL, "@group(0) @binding(1) var hw_tlas: acceleration_structure;")
        } else {
            (traversal, declaration)
        };
        let mut body = crate::deferred_wgsl!("restir.wgsl").replace("// __TLAS_BINDING_1__", declaration);
        let has_fused_sharc = fuse_bounce || fuse_deposit;
        if has_fused_sharc {
            body = body.replace("// __RESTIR_SHARC_DECLS__", RESTIR_SHARC_DECLS_WGSL);
        }
        if fuse_bounce {
            body = body.replace("// __RESTIR_SHARC_BOUNCE__", RESTIR_SHARC_BOUNCE_SNIPPET);
        }
        if fuse_deposit {
            body = body.replace("// __RESTIR_SHARC_DEPOSIT__", RESTIR_SHARC_DEPOSIT_SNIPPET);
        }
        if native_depth {
            body = body
                .replace("var depth_tex: texture_2d<f32>;", "var depth_tex: texture_depth_2d;")
                .replace("textureLoad(depth_tex, g_px, 0).x", "textureLoad(depth_tex, g_px, 0)");
        }
        let source = if has_fused_sharc {
            format!("{traversal}\n{SHARC_COMMON_WGSL}\n{body}")
        } else {
            format!("{traversal}\n{body}")
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("restir"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let geometry =
            Pass::new(device, &module, "restir_store_geometry", &[0, 5, 6, 12, 16, 20], hardware, native_depth);
        Self { module, di: None, gi: None, sharc_aux: None, geometry, fuse_bounce, fuse_deposit }
    }

    fn prepare(&mut self, device: &wgpu::Device, hardware: bool, native_depth: bool, di: bool, gi: bool, sharc: bool) {
        let make = |entry, bindings| Pass::new(device, &self.module, entry, bindings, hardware, native_depth);
        if di && self.di.is_none() {
            let di_init: &'static [u32] =
                if hardware { &[0, 1, 2, 4, 5, 6, 7, 9, 15] } else { &[0, 1, 2, 3, 4, 5, 6, 7, 9, 15] };
            let di_spat: &'static [u32] =
                if hardware { &[0, 1, 2, 4, 5, 6, 7, 9, 13, 15, 16] } else { &[0, 1, 2, 3, 4, 5, 6, 7, 9, 13, 15, 16] };
            self.di = Some([
                make("restir_di_initial", di_init),
                make("restir_di_temporal", &[0, 5, 6, 7, 8, 9, 11, 13, 14, 15]),
                make("restir_di_spatial", di_spat),
            ]);
        }
        if gi && self.gi.is_none() {
            let gi_init: &'static [u32] = match (hardware, self.fuse_bounce) {
                (true, true) => &[0, 1, 2, 4, 5, 6, 9, 10, 19, 21, 22, 24],
                (true, false) => &[0, 1, 2, 4, 5, 6, 9, 10, 19],
                (false, true) => &[0, 1, 2, 3, 4, 5, 6, 9, 10, 19, 21, 22, 24],
                (false, false) => &[0, 1, 2, 3, 4, 5, 6, 9, 10, 19],
            };
            let gi_spat: &'static [u32] = match (hardware, self.fuse_deposit) {
                (true, true) => &[0, 1, 2, 4, 5, 6, 17, 20, 21, 22, 23],
                (true, false) => &[0, 1, 2, 4, 5, 6, 17, 20],
                (false, true) => &[0, 1, 2, 3, 4, 5, 6, 17, 20, 21, 22, 23],
                (false, false) => &[0, 1, 2, 3, 4, 5, 6, 17, 20],
            };
            self.gi = Some([
                make("restir_gi_initial", gi_init),
                make("restir_gi_temporal", &[0, 5, 6, 8, 11, 17, 18, 19]),
                make("restir_gi_spatial", gi_spat),
            ]);
        }
        if gi && sharc && (!self.fuse_bounce || !self.fuse_deposit) && self.sharc_aux.is_none() {
            self.sharc_aux = Some(SharcAuxPipelines::new(device, native_depth));
        }
    }
}

pub struct RestirRenderer {
    pipelines: HashMap<(bool, bool, bool, bool), Pipelines>,
    hardware_supported: bool,
    dummy: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    lights_buf: wgpu::Buffer,
    frame: Option<Frame>,
    di_history_light_count: Option<u32>,
    gi_history_valid: bool,
}

impl RestirRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("restir frame uniforms"),
            size: std::mem::size_of::<Uniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("restir local lights"),
            size: (RESTIR_MAX_LIGHTS * std::mem::size_of::<RtLocalLight>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipelines: HashMap::new(),
            hardware_supported: device_supports_hardware_ray_query(device),
            dummy: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("restir empty DDGI and local lights"),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            uniform_buf,
            lights_buf,
            frame: None,
            di_history_light_count: None,
            gi_history_valid: false,
        }
    }

    /// Whether native ray queries are enabled on the device. Snapshots without a hardware TLAS use compute traversal.
    pub fn uses_hardware_ray_query(&self) -> bool {
        self.hardware_supported
    }

    pub fn reset_history(&mut self) {
        self.di_history_light_count = None;
        self.gi_history_valid = false;
    }

    /// Linear direct lighting with the primary surface BRDF applied.
    pub fn di_view(&self) -> Option<&wgpu::TextureView> {
        self.frame.as_ref().map(|frame| &frame.di.view)
    }

    pub fn di_texture(&self) -> Option<&wgpu::Texture> {
        self.frame.as_ref().map(|frame| &frame.di.texture)
    }

    /// Linear diffuse bounce lighting, without the primary surface albedo or metallic.
    pub fn gi_view(&self) -> Option<&wgpu::TextureView> {
        self.frame.as_ref().map(|frame| &frame.gi.view)
    }

    pub fn gi_texture(&self) -> Option<&wgpu::Texture> {
        self.frame.as_ref().map(|frame| &frame.gi.texture)
    }

    /// Encode one frame. Submit frames in dispatch order. Reset history on camera cuts,
    /// light-slot reassignment, or discontinuous changes to GI lighting or scene geometry.
    /// Inputs are single-sampled 2D views of mip level zero. The G-buffer shares
    /// one extent, and it and the motion texture may be larger than the dispatch.
    #[allow(clippy::too_many_arguments, reason = "caller supplies frame inputs and command encoder")]
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &TlasSceneGpu,
        inputs: RestirInputs<'_>,
        size: (u32, u32),
        params: &RestirParams,
        local_lights: &[RtLocalLight],
    ) -> Result<()> {
        let limits = device.limits();
        ensure!(
            size.0 > 0 && size.1 > 0 && size.0.max(size.1) <= limits.max_texture_dimension_2d,
            "invalid ReSTIR extent {size:?}"
        );
        ensure!(local_lights.len() <= RESTIR_MAX_LIGHTS, "ReSTIR supports at most {RESTIR_MAX_LIGHTS} local lights");
        let count = u32::try_from(local_lights.len())?;
        let depth_format = inputs.depth.texture().format();
        let native_depth = depth_format == wgpu::TextureFormat::Depth32Float;
        ensure!(native_depth || depth_format == wgpu::TextureFormat::R32Float, "unsupported ReSTIR depth format");
        let gbuffer_size = validate_input(inputs.depth, &[depth_format])?;
        ensure!(
            gbuffer_size.0 >= size.0
                && gbuffer_size.1 >= size.1
                && validate_input(inputs.normal_roughness, &[OUT_FORMAT])? == gbuffer_size
                && validate_input(inputs.albedo_metallic, &[OUT_FORMAT])? == gbuffer_size,
            "ReSTIR G-buffer textures must share an extent of at least {size:?}"
        );
        if let Some(motion) = inputs.motion {
            let motion_size = validate_input(motion, &[wgpu::TextureFormat::Rg16Float, OUT_FORMAT])?;
            ensure!(
                motion_size.0 >= size.0 && motion_size.1 >= size.1,
                "ReSTIR motion extent must be at least {size:?}"
            );
        }
        if let Some(ddgi) = inputs.ddgi_packed {
            ensure!(
                ddgi.size() >= 16 && ddgi.usage().contains(wgpu::BufferUsages::STORAGE),
                "ReSTIR DDGI must bind a packed storage buffer"
            );
        }
        ensure!(
            params.max_ray_distance.is_finite() && params.max_ray_distance >= MIN_RAY_DISTANCE,
            "invalid ReSTIR ray distance"
        );
        ensure!(params.normal_bias.is_finite() && params.normal_bias >= 0.0, "invalid ReSTIR normal bias");
        let pixels = u64::from(size.0) * u64::from(size.1);
        for (enabled, stride) in [(params.enable_di, DI_STRIDE), (params.enable_gi, GI_STRIDE)] {
            if enabled {
                let bytes = pixels * stride;
                ensure!(
                    bytes <= limits.max_storage_buffer_binding_size && bytes <= limits.max_buffer_size,
                    "ReSTIR reservoir buffer of {bytes} bytes exceeds device limits"
                );
            }
        }
        let hardware = self.hardware_supported && scene.has_hw_tlas();
        let storage_needed = if params.enable_di || params.enable_gi { if hardware { 6 } else { 7 } } else { 0 };
        ensure!(
            limits.max_storage_buffers_per_shader_stage >= storage_needed,
            "ReSTIR requires {storage_needed} storage buffers per compute stage"
        );

        if params.reset_history || self.frame.as_ref().is_none_or(|frame| frame.size != size) {
            self.reset_history();
        }
        let frame = self.frame.get_or_insert_with(|| Frame::new(device, size));
        if frame.size != size {
            *frame = Frame::new(device, size);
        }
        let sun = params.sun_direction;
        let sun_length = sun.iter().map(|v| v * v).sum::<f32>().sqrt();
        let sun = if sun_length.is_finite() && sun_length > 0.0 { sun.map(|v| v / sun_length) } else { [0.0; 3] };
        let pack = |[x, y, z]: [f32; 3], w| [x, y, z, w];
        let uniform = Uniform {
            inverse_view_projection: crate::matrix::invert(&params.view_projection),
            prev_view_projection: params.prev_view_projection,
            camera_and_clear: pack(params.camera_pos, params.depth_clear),
            previous_camera: pack(frame.previous_camera, 0.0),
            sun_and_max_distance: pack(sun, params.max_ray_distance),
            sun_color_and_bias: pack(params.sun_color, params.normal_bias),
            sky_zenith_and_bounce: pack(params.sky_zenith, params.gi_bounce_scale.max(0.0)),
            sky_horizon: pack(params.sky_horizon, 0.0),
            scene_counts_and_mask: [
                scene.tlas_node_count,
                scene.tlas_instance_count,
                scene.blas_tri_count,
                params.visibility_mask,
            ],
            size_lights_frame: [size.0, size.1, count, params.frame_index],
            flags: [
                u32::from(self.di_history_light_count == Some(count)),
                u32::from(self.gi_history_valid),
                u32::from(inputs.motion.is_some()),
                u32::from(params.enable_di) | (u32::from(params.enable_gi) << 1),
            ],
        };
        // Encoder-ordered upload into persistent buffers preserves ordering when several frames
        // are recorded before submission while avoiding multiple per-frame GPU buffer allocations.
        let uniform_bytes = bytemuck::bytes_of(&uniform);
        let live_lights: &[u8] = bytemuck::cast_slice(local_lights);
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("restir upload stage"),
            size: (uniform_bytes.len() + live_lights.len()) as u64,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        let off_l = uniform_bytes.len() as u64;
        stage.slice(0..off_l).get_mapped_range_mut().unwrap().copy_from_slice(uniform_bytes);
        if !live_lights.is_empty() {
            stage.slice(off_l..).get_mapped_range_mut().unwrap().copy_from_slice(live_lights);
        }
        stage.unmap();
        encoder.copy_buffer_to_buffer(&stage, 0, &self.uniform_buf, 0, off_l);
        if !live_lights.is_empty() {
            encoder.copy_buffer_to_buffer(&stage, off_l, &self.lights_buf, 0, live_lights.len() as u64);
        }

        let has_sharc = params.enable_gi && inputs.sharc.is_some();
        let fuse_bounce = has_sharc && limits.max_storage_buffers_per_shader_stage >= if hardware { 7 } else { 9 };
        let fuse_deposit = has_sharc;

        let pipelines = self
            .pipelines
            .entry((hardware, native_depth, fuse_bounce, fuse_deposit))
            .or_insert_with(|| Pipelines::new(device, hardware, native_depth, fuse_bounce, fuse_deposit));
        pipelines.prepare(device, hardware, native_depth, params.enable_di, params.enable_gi, has_sharc);
        let prev = frame.previous;
        let next = 1 - prev;
        let view = |view| Some(wgpu::BindingResource::TextureView(view));
        let tlas = scene.tlas_nodes_buf.as_entire_binding();
        #[cfg(not(target_arch = "wasm32"))]
        let tlas = if hardware { scene.hw_tlas.as_ref().unwrap().as_binding() } else { tlas };
        let mut resources: [Option<wgpu::BindingResource<'_>>; 25] = [
            Some(buffer_binding(&self.uniform_buf)),
            Some(tlas),
            Some(buffer_binding(&scene.tlas_instances_buf)),
            Some(buffer_binding(&scene.blas_nodes_buf)),
            Some(buffer_binding(&scene.blas_tris_buf)),
            view(inputs.depth),
            view(inputs.normal_roughness),
            view(inputs.albedo_metallic),
            view(inputs.motion.unwrap_or(inputs.normal_roughness)),
            Some(buffer_binding(if local_lights.is_empty() { &self.dummy } else { &self.lights_buf })),
            Some(buffer_binding(inputs.ddgi_packed.unwrap_or(&self.dummy))),
            view(&frame.depth_nrm[prev].view),
            view(&frame.depth_nrm[next].view),
            None,
            None,
            None,
            view(&frame.di.view),
            None,
            None,
            None,
            view(&frame.gi.view),
            inputs.sharc.map(|s| buffer_binding(s.uniform_buffer())),
            inputs.sharc.map(|s| buffer_binding(s.keys_buffer())),
            inputs.sharc.map(|s| buffer_binding(s.accum_buffer())),
            inputs.sharc.map(|s| buffer_binding(s.resolved_buffer())),
        ];

        if params.enable_di {
            let buffers = frame.di_reservoirs.get_or_insert_with(|| {
                Reservoirs::new(
                    device,
                    pixels * DI_STRIDE,
                    ["restir DI scratch", "restir DI history A", "restir DI history B"],
                )
            });
            let [initial, temporal, spatial] = pipelines.di.as_ref().unwrap();
            resources[15] = Some(buffer_binding(&buffers.scratch));
            initial.dispatch(device, encoder, &resources, size);
            resources[13] = Some(buffer_binding(&buffers.scratch));
            resources[14] = Some(buffer_binding(&buffers.history[prev]));
            resources[15] = Some(buffer_binding(&buffers.history[next]));
            temporal.dispatch(device, encoder, &resources, size);
            resources[13] = Some(buffer_binding(&buffers.history[next]));
            // The initial reservoir is dead here. Reuse its allocation for the spatial result.
            resources[15] = Some(buffer_binding(&buffers.scratch));
            spatial.dispatch(device, encoder, &resources, size);
        }
        if params.enable_gi {
            let buffers = frame.gi_reservoirs.get_or_insert_with(|| {
                Reservoirs::new(
                    device,
                    pixels * GI_STRIDE,
                    ["restir GI scratch", "restir GI history A", "restir GI history B"],
                )
            });
            let [initial, temporal, spatial] = pipelines.gi.as_ref().unwrap();
            resources[19] = Some(buffer_binding(&buffers.scratch));
            initial.dispatch(device, encoder, &resources, size);
            if !fuse_bounce && let (Some(sharc), Some(aux)) = (inputs.sharc, pipelines.sharc_aux.as_ref()) {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("restir_gi_sharc_bounce"),
                    layout: &aux.bounce_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(inputs.depth) },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(inputs.normal_roughness),
                        },
                        wgpu::BindGroupEntry { binding: 3, resource: buffers.scratch.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: sharc.uniform_buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: sharc.keys_buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 7, resource: sharc.resolved_buffer().as_entire_binding() },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("restir_gi_sharc_bounce"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&aux.bounce_pipe);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
            }
            resources[17] = Some(buffer_binding(&buffers.scratch));
            resources[18] = Some(buffer_binding(&buffers.history[prev]));
            resources[19] = Some(buffer_binding(&buffers.history[next]));
            temporal.dispatch(device, encoder, &resources, size);
            resources[17] = Some(buffer_binding(&buffers.history[next]));
            spatial.dispatch(device, encoder, &resources, size);
            if !fuse_deposit && let (Some(sharc), Some(aux)) = (inputs.sharc, pipelines.sharc_aux.as_ref()) {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("restir_gi_sharc_deposit"),
                    layout: &aux.deposit_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(inputs.depth) },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(inputs.normal_roughness),
                        },
                        wgpu::BindGroupEntry { binding: 4, resource: sharc.uniform_buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: sharc.keys_buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: sharc.accum_buffer().as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::TextureView(&frame.gi.view),
                        },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("restir_gi_sharc_deposit"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&aux.deposit_pipe);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
            }
        }
        pipelines.geometry.dispatch(device, encoder, &resources, size);
        frame.previous = next;
        frame.previous_camera = params.camera_pos;
        self.di_history_light_count = params.enable_di.then_some(count);
        self.gi_history_valid = params.enable_gi;
        Ok(())
    }
}

fn validate_input(view: &wgpu::TextureView, formats: &[wgpu::TextureFormat]) -> Result<(u32, u32)> {
    let texture = view.texture();
    ensure!(
        formats.contains(&texture.format())
            && texture.dimension() == wgpu::TextureDimension::D2
            && texture.sample_count() == 1
            && texture.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING),
        "ReSTIR input must be a single-sampled 2D texture with format in {formats:?} and TEXTURE_BINDING usage"
    );
    Ok((texture.width(), texture.height()))
}

fn buffer_binding(buffer: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    buffer.as_entire_binding()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::IDENTITY;
    use crate::ray_tracing::{Blas, BlasTriangle, GpuBlasPool, TlasInstance, TlasScene};
    use crate::testing;

    const SIZE: (u32, u32) = (64, 32);
    const HALF_SIZE: (u32, u32) = (SIZE.0 / 2, SIZE.1 / 2);
    // x covers [-2, 2], z covers [-0.5, 0.5]. The floor at y = 0 has clip depth 0.5.
    const FLOOR_VP: [f32; 16] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0];

    fn params() -> RestirParams {
        RestirParams {
            view_projection: FLOOR_VP,
            prev_view_projection: FLOOR_VP,
            camera_pos: [0.0, 12.0, 0.0],
            depth_clear: 0.0,
            sun_direction: [0.0, 1.0, 0.0],
            sun_color: [0.0; 3],
            sky_zenith: [0.0; 3],
            sky_horizon: [0.0; 3],
            max_ray_distance: 20.0,
            normal_bias: 0.015,
            visibility_mask: u32::MAX,
            frame_index: 0,
            enable_di: false,
            enable_gi: false,
            reset_history: false,
            gi_bounce_scale: 1.0,
        }
    }

    struct GBuffer {
        depth: wgpu::TextureView,
        normal: wgpu::TextureView,
        albedo: wgpu::TextureView,
        motion: wgpu::TextureView,
    }

    impl GBuffer {
        fn new(device: &wgpu::Device, queue: &wgpu::Queue, native_depth: bool, half_res: bool) -> Self {
            let count = (SIZE.0 * SIZE.1) as usize;
            let desc = wgpu::TextureDescriptor {
                label: Some("ReSTIR witness depth"),
                size: wgpu::Extent3d { width: SIZE.0, height: SIZE.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: if native_depth { wgpu::TextureFormat::Depth32Float } else { wgpu::TextureFormat::R32Float },
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | if native_depth { wgpu::TextureUsages::RENDER_ATTACHMENT } else { wgpu::TextureUsages::COPY_DST },
                view_formats: &[],
            };
            let depth = if native_depth {
                let texture = device.create_texture(&desc);
                let view = texture.create_view(&Default::default());
                let mut encoder = device.create_command_encoder(&Default::default());
                let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.5),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                queue.submit([encoder.finish()]);
                view
            } else {
                device
                    .create_texture_with_data(
                        queue,
                        &desc,
                        wgpu::util::TextureDataOrder::LayerMajor,
                        bytemuck::cast_slice(&vec![0.5f32; count]),
                    )
                    .create_view(&Default::default())
            };
            let full_pixels = (0..SIZE.1).flat_map(|y| (0..SIZE.0).map(move |x| (x, y)));
            let sampled = |x, y| !half_res || (x % 2 == 1 && y % 2 == 1);
            let picture = |pixel| {
                let data: Vec<_> =
                    full_pixels.clone().map(|(x, y)| if sampled(x, y) { pixel } else { [0.0; 4] }).collect();
                testing::picture(device, queue, SIZE.0, SIZE.1, &data).create_view(&Default::default())
            };
            let motion: Vec<_> = full_pixels
                .clone()
                .map(|(x, y)| {
                    if !sampled(x, y) {
                        [2.0, 0.0]
                    } else if (x / 2 + y / 2) % 2 == 0 {
                        [0.0, 0.0]
                    } else {
                        [100.0, 0.0]
                    }
                })
                .collect();
            let motion = if half_res {
                let bits: Vec<_> = motion.iter().flatten().copied().map(testing::half).collect();
                device.create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some("ReSTIR witness velocity"),
                        format: wgpu::TextureFormat::Rg16Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        ..desc
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    bytemuck::cast_slice(&bits),
                )
            } else {
                let rgba: Vec<_> = motion.iter().map(|&[x, y]| [x, y, 0.0, 0.0]).collect();
                testing::picture(device, queue, SIZE.0, SIZE.1, &rgba)
            };
            Self {
                depth,
                normal: picture([0.0, 1.0, 0.0, 0.8]),
                albedo: picture([0.8, 0.8, 0.8, 0.0]),
                motion: motion.create_view(&Default::default()),
            }
        }

        fn inputs(&self, motion: bool) -> RestirInputs<'_> {
            RestirInputs {
                depth: &self.depth,
                normal_roughness: &self.normal,
                albedo_metallic: &self.albedo,
                motion: motion.then_some(&self.motion),
                ddgi_packed: None,
                sharc: None,
            }
        }
    }

    #[allow(clippy::too_many_arguments, reason = "witness varies both inputs and output extent")]
    fn render(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut RestirRenderer,
        scene: &TlasSceneGpu,
        gbuffer: &GBuffer,
        size: (u32, u32),
        params: &RestirParams,
        lights: &[RtLocalLight],
    ) -> Vec<[f32; 4]> {
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer.dispatch(device, &mut encoder, scene, gbuffer.inputs(params.enable_gi), size, params, lights).unwrap();
        let output = if params.enable_di { renderer.di_texture() } else { renderer.gi_texture() }.unwrap();
        assert_eq!((output.width(), output.height()), size);
        let row_bytes = size.0 * 8;
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let aligned_row_bytes = row_bytes.div_ceil(alignment) * alignment;
        let bytes = u64::from(aligned_row_bytes * size.1);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReSTIR witness"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_row_bytes),
                    rows_per_image: None,
                },
            },
            output.size(),
        );
        queue.submit([encoder.finish()]);
        let raw = testing::read(device, queue, &readback, bytes);
        raw.chunks_exact(aligned_row_bytes as usize)
            .flat_map(|row| row[..row_bytes as usize].as_chunks::<8>().0)
            .map(|pixel| {
                std::array::from_fn(|i| {
                    let bits = u16::from_le_bytes([pixel[2 * i], pixel[2 * i + 1]]);
                    assert_ne!(bits & 0x7c00, 0x7c00, "nonfinite ReSTIR output");
                    testing::unhalf(bits)
                })
            })
            .collect()
    }

    fn quad(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> [BlasTriangle; 2] {
        [BlasTriangle::flat(a, b, c), BlasTriangle::flat(a, c, d)]
    }

    fn divider(top: f32) -> [BlasTriangle; 2] {
        quad([0.0, 0.0, -2.0], [0.0, top, -2.0], [0.0, top, 2.0], [0.0, 0.0, 2.0])
    }

    fn region(size: (u32, u32), columns: std::ops::Range<usize>) -> impl Iterator<Item = usize> {
        let (width, height) = (size.0 as usize, size.1 as usize);
        let y_margin = height / 8;
        (y_margin..height - y_margin).flat_map(move |y| columns.clone().map(move |x| y * width + x))
    }

    #[allow(clippy::cast_precision_loss, reason = "witness averages contain at most 2048 finite pixels")]
    fn mean(values: impl Iterator<Item = f32>) -> f32 {
        let (sum, count) = values.fold((0.0, 0u32), |(sum, n), val| (sum + val, n + 1));
        sum / count as f32
    }

    #[test]
    fn gpu_restir_di_many_lights_shadow_and_spatiotemporal_convergence_witness() {
        let (device, queue) = testing::device_with_optional().expect("headless GPU adapter required");
        let wall = Blas::build(&divider(4.0)).unwrap();
        let scene = TlasScene::build(&[wall], &[TlasInstance::new(0, IDENTITY)]).unwrap();
        // Static uploads intentionally exercise software fallback even on a hardware-capable device.
        let scene = TlasSceneGpu::upload(&device, &scene).unwrap();
        let gbuffer = GBuffer::new(&device, &queue, false, false);
        let mut renderer = RestirRenderer::new(&device);
        let mut params = params();
        params.enable_di = true;

        // Coincident emitters make the exact answer available using one equivalent light.
        // One bright emitter makes the eight-candidate starting estimate deliberately noisy.
        let point = |intensity| RtLocalLight {
            position_range: [-1.4, 2.0, 0.0, 14.0],
            color_outer_cos: [intensity, intensity, intensity, -1.0],
            direction_inner_cos: [0.0, -1.0, 0.0, -1.0],
        };
        let mut lights = [point(0.4); 32];
        lights[0] = point(26.0);
        let mut first = Vec::new();
        let mut last = Vec::new();
        for frame in 0..=8 {
            params.frame_index = frame;
            let pixels = render(&device, &queue, &mut renderer, &scene, &gbuffer, SIZE, &params, &lights);
            if frame == 0 {
                first = pixels;
            } else if frame == 8 {
                last = pixels;
            }
        }
        renderer.reset_history();
        let reference =
            render(&device, &queue, &mut renderer, &scene, &gbuffer, SIZE, &params, &[point(26.0 + 31.0 * 0.4)]);
        let mse =
            |pixels: &[[f32; 4]]| mean(region(SIZE, 4..25).map(|i| (pixels[i][0] / reference[i][0] - 1.0).powi(2)));
        let (first_error, last_error) = (mse(&first), mse(&last));
        let lit = mean(region(SIZE, 4..25).map(|i| last[i][0]));
        let expected = mean(region(SIZE, 4..25).map(|i| reference[i][0]));
        let shadow = mean(region(SIZE, 32..48).map(|i| last[i][0]));
        eprintln!("ReSTIR DI: lit={lit}, expected={expected}, shadow={shadow}, MSE ratio={}", first_error / last_error);
        assert!(lit > 0.75 && (lit / expected - 1.0).abs() < 0.2, "lit={lit}, exact={expected}");
        assert!(shadow < lit * 0.01, "occluded region={shadow}, lit={lit}");
        assert!(last_error * 3.0 < first_error, "normalized DI MSE: frame 0={first_error}, frame 8={last_error}");

        // A single light with no blockers gives the same BRDF value at the
        // selected full-resolution texel and its corresponding half-size pixel.
        params.visibility_mask = 0;
        renderer.reset_history();
        let single_light = [point(26.0 + 31.0 * 0.4)];
        let full = render(&device, &queue, &mut renderer, &scene, &gbuffer, SIZE, &params, &single_light);
        let patterned = GBuffer::new(&device, &queue, false, true);
        let half = render(&device, &queue, &mut renderer, &scene, &patterned, HALF_SIZE, &params, &single_light);
        for (i, pixel) in half.iter().enumerate() {
            let x = i % HALF_SIZE.0 as usize;
            let y = i / HALF_SIZE.0 as usize;
            let texel = full[(2 * y + 1) * SIZE.0 as usize + 2 * x + 1];
            assert_eq!(pixel[3], 1.0, "invalid geometry at half-size pixel ({x}, {y})");
            assert!((pixel[0] - texel[0]).abs() <= 0.002, "DI mismatch at ({x}, {y}): {pixel:?} vs {texel:?}");
        }
    }

    #[test]
    fn gpu_restir_gi_cornell_color_bleed_and_thin_wall_visibility_witness() {
        let (device, queue) = testing::device_with_optional().expect("headless GPU adapter required");
        let quads = [
            quad([-2.2, 0.0, -2.0], [-2.2, 0.0, 2.0], [2.2, 0.0, 2.0], [2.2, 0.0, -2.0]),
            quad([-2.2, 2.4, -2.0], [2.2, 2.4, -2.0], [2.2, 2.4, 2.0], [-2.2, 2.4, 2.0]),
            quad([-2.2, 0.0, -2.0], [-2.2, 2.4, -2.0], [-2.2, 2.4, 2.0], [-2.2, 0.0, 2.0]),
            divider(2.4),
            quad([2.2, 0.0, -2.0], [2.2, 0.0, 2.0], [2.2, 2.4, 2.0], [2.2, 2.4, -2.0]),
            quad([-2.2, 0.0, -2.0], [2.2, 0.0, -2.0], [2.2, 2.4, -2.0], [-2.2, 2.4, -2.0]),
            quad([-2.2, 0.0, 2.0], [-2.2, 2.4, 2.0], [2.2, 2.4, 2.0], [2.2, 0.0, 2.0]),
        ];
        let mut pool = GpuBlasPool::new(&device);
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut instances = Vec::new();
        for (i, tris) in quads.iter().enumerate() {
            let blas = pool.ensure_blas(&device, &mut encoder, i as u64 + 1, tris).unwrap().unwrap();
            let mut instance = TlasInstance::new(blas, IDENTITY);
            if i == 2 {
                instance.emissive = [10.0, 0.08, 0.06];
            }
            instances.push(instance);
        }
        pool.update_tlas(&device, &mut encoder, &instances).unwrap();
        queue.submit([encoder.finish()]);
        let scene = pool.snapshot_scene();
        let mut renderer = RestirRenderer::new(&device);
        let mut params = params();
        params.enable_gi = true;
        params.depth_clear = 1.0;
        for (size, half_res) in [(SIZE, false), (HALF_SIZE, true)] {
            let gbuffer = GBuffer::new(&device, &queue, true, half_res);
            let frames: Vec<_> = (0..=8)
                .map(|frame| {
                    params.frame_index = frame;
                    render(&device, &queue, &mut renderer, &scene, &gbuffer, size, &params, &[])
                })
                .collect();
            let last = &frames[8];
            assert!(last.iter().all(|pixel| pixel[3] == 1.0), "dispatch must load valid G-buffer texels");
            let divisor = if half_res { 2 } else { 1 };
            let lit_columns = 5 / divisor..22 / divisor;
            let lit_red = mean(region(size, lit_columns.clone()).map(|i| last[i][0]));
            let lit_green = mean(region(size, lit_columns.clone()).map(|i| last[i][1]));
            // The tested dark strip is directly across the wall, within the reuse disk.
            let dark_red = mean(region(size, 32 / divisor..36 / divisor).map(|i| last[i][0]));
            assert!(lit_red > 0.45 && lit_red > 10.0 * lit_green, "lit red={lit_red}, green={lit_green}");
            assert!(dark_red < lit_red * 0.15, "red behind thin wall={dark_red}, lit red={lit_red}");

            let frame_delta = |a: usize, b: usize| {
                mean(region(size, lit_columns.clone()).map(|i| (frames[a][i][0] - frames[b][i][0]).powi(2)))
            };
            let (early, late) = (frame_delta(0, 1), frame_delta(7, 8));
            eprintln!(
                "ReSTIR GI {size:?} (hardware TLAS={}): lit red={lit_red}, lit green={lit_green}, dark red={dark_red}, temporal delta ratio={}",
                scene.has_hw_tlas(),
                early / late
            );
            assert!(late < early * 0.65, "GI temporal variance: first delta={early}, final delta={late}");
        }
    }
}
