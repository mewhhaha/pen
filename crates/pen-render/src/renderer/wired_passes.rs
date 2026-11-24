//! End-to-end wiring of high-fidelity rendering passes into `Renderer::draw`:
//! - Hillaire physical atmosphere LUTs & volumetric clouds (`quality.physical_sky`)
//! - Full PBR deferred decals (`frame.pbr_decals`)
//! - Weighted Blended Order-Independent Transparency (`quality.oit`)
//! - Portable 2-level TLAS/BLAS ray tracing (`quality.ray_tracing` 1 = shadows+RTAO, 2 = +reflections+GI)
//! - Dynamic Diffuse Global Illumination probe grid (`quality.ddgi`)
//! - AMD FidelityFX RCAS sharpening & Post-FX polish (`grade.{sharpness, grain, flare, outline_strength}`)
//! - Render-to-texture & split-screen secondary cameras (`frame.camera_targets`)

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use anyhow::Result;
use pen_protocol::{Catalog, Matrix, RenderFrame};

use super::{GpuAsset, Scratch, Shape, State, TextureRef, ViewTemporalState};
use crate::asset_store::AssetKind;
use crate::atmosphere_lut::{
    AtmosphereParams, BASE_CAMERA_ALTITUDE_KM, PhysicalAtmosphere, WORLD_METERS_TO_KM, physical_camera_km,
};
use crate::built_in::{BuiltIn, BuiltIns};
use crate::camera_precision::DepthMode;
use crate::ddgi::{DDGI_WGSL, DdgiFrameParams, DdgiGridConfig, DdgiVolume};
use crate::decals::{PbrDecal, PbrDecalPass, PbrDecalView};
use crate::environment_passes::{
    FoliageInstance, FoliagePipeline, FoliageUniform, PipelineTarget, TerrainHeightmap, TerrainPatch, TerrainPipeline,
    TerrainUniform, WaterPipeline, WaterUniform,
};
use crate::lights::MAX_AREA_COOKIES;
use crate::matrix::{IDENTITY, invert, multiply};
use crate::meshlets::{MeshletCullParams, MeshletCuller, MeshletIndirectDraw};
use crate::oit::OitPass;
use crate::post::{PhysicalSkyBinding, SCENE_FORMAT};
use crate::post_fx::{PostFxParams, PostFxPass, RcasSharpener};
use crate::ray_tracing::{Blas, TLAS_TRAVERSAL_WGSL, TlasInstance, TlasRayTraceParams, TlasRayTracer, TlasScene, TlasSceneGpu};
use crate::render_targets::{ColorTransfer, OffscreenRenderTarget, SplitScreenCompositor, ViewportRect};
use crate::shader::{self, Stage};
use crate::svgf::{SvgfDenoiser, SvgfInputs, SvgfParams};
use crate::RenderTarget;

/// Built-in optional `RenderFrame::extensions` tags driving outdoor GPU environment passes from Blot.
pub const ENV_TERRAIN_EXT_TAG: u32 = 65_001;
pub const ENV_WATER_EXT_TAG: u32 = 65_002;
pub const ENV_FOLIAGE_EXT_TAG: u32 = 65_003;

/// CDLOD heightfield terrain descriptor attached to [`EnvironmentScene`].
pub struct EnvironmentTerrain {
    pub heightmap: Arc<TerrainHeightmap>,
    pub patches: Vec<TerrainPatch>,
    pub uniform: TerrainUniform,
}

/// GPU-instanced wind-swayed foliage descriptor attached to [`EnvironmentScene`].
pub struct EnvironmentFoliage {
    pub uniform: FoliageUniform,
    pub instances: Vec<FoliageInstance>,
}

/// Outdoor GPU environment passes (CDLOD terrain, Gerstner water, instanced foliage) rendered inside
/// `Renderer::draw`'s opaque HDR scene pass.
#[derive(Default)]
pub struct EnvironmentScene {
    pub terrain: Option<EnvironmentTerrain>,
    pub water: Vec<WaterUniform>,
    pub foliage: Option<EnvironmentFoliage>,
}

type CookieUpload = (u32, wgpu::TextureView, u64);
type PreparedCookies = (Vec<Option<u32>>, Vec<CookieUpload>);

pub(super) struct PreparedPbrDecalBatch {
    pub albedo: wgpu::TextureView,
    pub normal: wgpu::TextureView,
    pub orm: wgpu::TextureView,
    pub decals: Vec<PbrDecal>,
}

const COPY_BLIT_WGSL: &str = r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;

@vertex fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
  let y = 1.0 - f32(idx & 2u) * 2.0;
  return vec4f(x, y, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(src_tex, 0));
  let uv = pos.xy / max(dims, vec2f(1.0));
  return textureSampleLevel(src_tex, src_samp, uv, 0.0);
}
"#;

struct BlitShared {
    module: wgpu::ShaderModule,
    bgl: wgpu::BindGroupLayout,
    pl: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
}

pub(super) struct SplitViewSlot {
    pub target: OffscreenRenderTarget,
    pub temporal: ViewTemporalState,
}

const RT_COMPOSITE_WGSL: &str = r#"
struct RtCompUniform {
  depth_clear: f32,
  _p0: f32,
  _p1: f32,
  _p2: f32,
};
@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var shadow_ao_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> cfg: RtCompUniform;

@vertex fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
  let y = 1.0 - f32(idx & 2u) * 2.0;
  return vec4f(x, y, 0.0, 1.0);
}

@fragment fn fs_shadow_ao(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let dims = vec2i(textureDimensions(depth_tex, 0));
  let p = clamp(vec2i(pos.xy), vec2i(0), dims - vec2i(1));
  let d = textureLoad(depth_tex, p, 0).r;
  if (abs(d - cfg.depth_clear) <= 0.00001) {
    return vec4f(1.0, 1.0, 1.0, 1.0);
  }
  let sao = textureLoad(shadow_ao_tex, p, 0);
  let shadow = mix(0.25, 1.0, clamp(sao.r, 0.0, 1.0));
  let ao = mix(0.35, 1.0, clamp(sao.g, 0.0, 1.0));
  let atten = shadow * ao;
  return vec4f(atten, atten, atten, 1.0);
}
"#;

const RT_BOUNCE_WGSL: &str = r#"
struct RtCompUniform {
  depth_clear: f32,
  _p0: f32,
  _p1: f32,
  _p2: f32,
};
@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var refl_tex: texture_2d<f32>;
@group(0) @binding(2) var gi_tex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> cfg: RtCompUniform;

@vertex fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
  let y = 1.0 - f32(idx & 2u) * 2.0;
  return vec4f(x, y, 0.0, 1.0);
}

@fragment fn fs_bounce(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let dims = vec2i(textureDimensions(depth_tex, 0));
  let p = clamp(vec2i(pos.xy), vec2i(0), dims - vec2i(1));
  let d = textureLoad(depth_tex, p, 0).r;
  if (abs(d - cfg.depth_clear) <= 0.00001) {
    discard;
  }
  let refl = max(textureLoad(refl_tex, p, 0).rgb, vec3f(0.0));
  let gi = max(textureLoad(gi_tex, p, 0).rgb, vec3f(0.0));
  let bounce = refl * 0.35 + gi * 0.65;
  return vec4f(bounce, 0.0);
}
"#;

const SVGF_PREP_WGSL: &str = r#"
struct SvgfPrepUniform {
  inv_vp: mat4x4f,
  prev_vp: mat4x4f,
  cam_and_clear: vec4f, // xyz = camera world pos, w = depth_clear (1.0 ForwardZ, 0.0 ReversedZ)
  dims: vec4u,          // x = width, y = height, z = has_prev_vp (0 or 1), w = 0
};

@group(0) @binding(0) var<uniform> u: SvgfPrepUniform;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var out_depth: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var out_normal: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var out_motion: texture_storage_2d<rgba16float, write>;

struct WorldPt {
  pos: vec3f,
  valid: bool,
};

fn load_world(px: vec2i, dims: vec2i) -> WorldPt {
  if (px.x < 0 || px.y < 0 || px.x >= dims.x || px.y >= dims.y) {
    return WorldPt(vec3f(0.0), false);
  }
  let d = textureLoad(depth_tex, px, 0).r;
  if (abs(d - u.cam_and_clear.w) <= 0.00001) {
    return WorldPt(vec3f(0.0), false);
  }
  let ndc = vec2f(
    (f32(px.x) + 0.5) / f32(dims.x) * 2.0 - 1.0,
    1.0 - (f32(px.y) + 0.5) / f32(dims.y) * 2.0,
  );
  let clip = u.inv_vp * vec4f(ndc, d, 1.0);
  if (abs(clip.w) <= 1e-6) {
    return WorldPt(vec3f(0.0), false);
  }
  return WorldPt(clip.xyz / clip.w, true);
}

@compute @workgroup_size(8, 8, 1)
fn prep(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(i32(u.dims.x), i32(u.dims.y));
  let px = vec2i(id.xy);
  if (px.x >= dims.x || px.y >= dims.y) {
    return;
  }
  let c = load_world(px, dims);
  if (!c.valid) {
    textureStore(out_depth, px, vec4f(10000.0, 0.0, 0.0, 1.0));
    textureStore(out_normal, px, vec4f(0.0, 1.0, 0.0, 1.0));
    textureStore(out_motion, px, vec4f(0.0, 0.0, 0.0, 0.0));
    return;
  }
  let sr = load_world(px + vec2i(1, 0), dims);
  let sl = load_world(px - vec2i(1, 0), dims);
  let sd = load_world(px + vec2i(0, 1), dims);
  let su = load_world(px - vec2i(0, 1), dims);
  var dx = vec3f(1.0, 0.0, 0.0);
  var has_dx = true;
  if (sr.valid && sl.valid) {
    let dr = sr.pos - c.pos;
    let dl = c.pos - sl.pos;
    dx = select(dl, dr, dot(dr, dr) < dot(dl, dl));
  } else if (sr.valid) {
    dx = sr.pos - c.pos;
  } else if (sl.valid) {
    dx = c.pos - sl.pos;
  } else {
    has_dx = false;
  }
  var dy = vec3f(0.0, 0.0, 1.0);
  var has_dy = true;
  if (sd.valid && su.valid) {
    let dd = sd.pos - c.pos;
    let du = c.pos - su.pos;
    dy = select(du, dd, dot(dd, dd) < dot(du, du));
  } else if (sd.valid) {
    dy = sd.pos - c.pos;
  } else if (su.valid) {
    dy = c.pos - su.pos;
  } else {
    has_dy = false;
  }
  var n = vec3f(0.0, 1.0, 0.0);
  if (has_dx && has_dy) {
    let raw_n = cross(dx, dy);
    let area2 = dot(raw_n, raw_n);
    let ref2 = dot(dx, dx) * dot(dy, dy);
    if (area2 > 1e-12 * ref2 && ref2 > 0.0) {
      n = raw_n * inverseSqrt(area2);
    }
  }
  let to_cam = u.cam_and_clear.xyz - c.pos;
  let dist = length(to_cam);
  let view_dir = select(vec3f(0.0, 1.0, 0.0), to_cam / dist, dist > 1e-6);
  if (dot(n, view_dir) < 0.0) {
    n = -n;
  }
  var mv = vec2f(0.0);
  if (u.dims.z != 0u) {
    let curr_uv = (vec2f(px) + vec2f(0.5)) / vec2f(f32(dims.x), f32(dims.y));
    let pclip = u.prev_vp * vec4f(c.pos, 1.0);
    if (abs(pclip.w) > 1e-6) {
      let pndc = pclip.xy / pclip.w;
      let prev_uv = vec2f(pndc.x * 0.5 + 0.5, 0.5 - pndc.y * 0.5);
      mv = curr_uv - prev_uv;
    }
  }
  textureStore(out_depth, px, vec4f(min(dist, 65000.0), 0.0, 0.0, 1.0));
  textureStore(out_normal, px, vec4f(n, 1.0));
  textureStore(out_motion, px, vec4f(mv, 0.0, 0.0));
}
"#;

const DDGI_COMPOSITE_SUFFIX_WGSL: &str = r#"
struct DdgiCompCam {
  inverse_view_projection: mat4x4f,
  camera_pos_and_strength: vec4f, // xyz = camera world pos, w = indirect strength
  depth_cfg: vec4f,               // x = depth_clear (1.0 ForwardZ, 0.0 ReversedZ)
};

@group(1) @binding(0) var<uniform> ddgi_cam: DdgiCompCam;
@group(1) @binding(1) var ddgi_depth_tex: texture_2d<f32>;
@group(1) @binding(2) var ddgi_irr_tex: texture_2d<f32>;
@group(1) @binding(3) var ddgi_vis_tex: texture_2d<f32>;
@group(1) @binding(4) var ddgi_samp: sampler;

struct DdgiDepthPt {
  world: vec3f,
  valid: bool,
};

fn ddgi_load_world(px: vec2i, dims: vec2i) -> DdgiDepthPt {
  if (px.x < 0 || px.y < 0 || px.x >= dims.x || px.y >= dims.y) {
    return DdgiDepthPt(vec3f(0.0), false);
  }
  let d = textureLoad(ddgi_depth_tex, px, 0).r;
  if (abs(d - ddgi_cam.depth_cfg.x) <= 0.00001) {
    return DdgiDepthPt(vec3f(0.0), false);
  }
  let ndc = vec2f(
    (f32(px.x) + 0.5) / f32(dims.x) * 2.0 - 1.0,
    1.0 - (f32(px.y) + 0.5) / f32(dims.y) * 2.0,
  );
  let clip = ddgi_cam.inverse_view_projection * vec4f(ndc, d, 1.0);
  if (abs(clip.w) <= 1e-6) {
    return DdgiDepthPt(vec3f(0.0), false);
  }
  return DdgiDepthPt(clip.xyz / clip.w, true);
}

@vertex fn ddgi_comp_vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
  let y = 1.0 - f32(idx & 2u) * 2.0;
  return vec4f(x, y, 0.0, 1.0);
}

@fragment fn ddgi_comp_fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let dims = vec2i(textureDimensions(ddgi_depth_tex, 0));
  let px = clamp(vec2i(pos.xy), vec2i(0), dims - vec2i(1));
  let center = ddgi_load_world(px, dims);
  if (!center.valid) {
    discard;
  }
  let sr = ddgi_load_world(px + vec2i(1, 0), dims);
  let sl = ddgi_load_world(px - vec2i(1, 0), dims);
  let sd = ddgi_load_world(px + vec2i(0, 1), dims);
  let su = ddgi_load_world(px - vec2i(0, 1), dims);

  var dx = vec3f(1.0, 0.0, 0.0);
  var has_dx = true;
  if (sr.valid && sl.valid) {
    let dr = sr.world - center.world;
    let dl = center.world - sl.world;
    dx = select(dl, dr, dot(dr, dr) < dot(dl, dl));
  } else if (sr.valid) {
    dx = sr.world - center.world;
  } else if (sl.valid) {
    dx = center.world - sl.world;
  } else {
    has_dx = false;
  }

  var dy = vec3f(0.0, 0.0, 1.0);
  var has_dy = true;
  if (sd.valid && su.valid) {
    let dd = sd.world - center.world;
    let du = center.world - su.world;
    dy = select(du, dd, dot(dd, dd) < dot(du, du));
  } else if (sd.valid) {
    dy = sd.world - center.world;
  } else if (su.valid) {
    dy = center.world - su.world;
  } else {
    has_dy = false;
  }

  var n = vec3f(0.0, 1.0, 0.0);
  if (has_dx && has_dy) {
    let raw_n = cross(dx, dy);
    let area2 = dot(raw_n, raw_n);
    let ref2 = dot(dx, dx) * dot(dy, dy);
    if (area2 > 1e-12 * ref2 && ref2 > 0.0) {
      n = raw_n * inverseSqrt(area2);
    }
  }
  let to_cam = ddgi_cam.camera_pos_and_strength.xyz - center.world;
  let cam_dist = length(to_cam);
  let view_dir = select(vec3f(0.0, 1.0, 0.0), to_cam / cam_dist, cam_dist > 1e-6);
  if (dot(n, view_dir) < 0.0) {
    n = -n;
  }
  let irr = ddgi_sample_irradiance(
    center.world,
    n,
    view_dir,
    u,
    ddgi_irr_tex,
    ddgi_vis_tex,
    ddgi_samp,
  );
  let indirect = max(irr, vec3f(0.0)) * ddgi_cam.camera_pos_and_strength.w;
  return vec4f(indirect, 0.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RtCompUniform {
    depth_clear: f32,
    _p: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SvgfPrepUniform {
    inv_vp: [f32; 16],
    prev_vp: [f32; 16],
    cam_and_clear: [f32; 4],
    dims: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DdgiCompCamUniform {
    inverse_view_projection: [f32; 16],
    camera_pos_and_strength: [f32; 4],
    depth_cfg: [f32; 4],
}

struct RtCompositePipelines {
    sao_bgl: wgpu::BindGroupLayout,
    sao_pipeline: wgpu::RenderPipeline,
    bounce_bgl: wgpu::BindGroupLayout,
    bounce_pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
}

struct SvgfTex {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

impl SvgfTex {
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

struct SvgfRtBuffers {
    size: (u32, u32),
    curr_depth: SvgfTex,
    prev_depth: SvgfTex,
    curr_normal: SvgfTex,
    prev_normal: SvgfTex,
    motion: SvgfTex,
}

struct SvgfRtState {
    prep_pipe: wgpu::ComputePipeline,
    prep_bgl: wgpu::BindGroupLayout,
    prep_u: wgpu::Buffer,
    sao: SvgfDenoiser,
    refl: SvgfDenoiser,
    gi: SvgfDenoiser,
    buffers: Option<SvgfRtBuffers>,
    prev_vp: Option<[f32; 16]>,
}

impl SvgfRtState {
    fn new(device: &wgpu::Device) -> Self {
        let ro_tex = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let wo_tex = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let prep_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("svgf_rt_prep_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(size_of::<SvgfPrepUniform>() as u64),
                    },
                    count: None,
                },
                ro_tex(1),
                wo_tex(2),
                wo_tex(3),
                wo_tex(4),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("svgf_rt_prep_wgsl"),
            source: wgpu::ShaderSource::Wgsl(SVGF_PREP_WGSL.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("svgf_rt_prep_pl"),
            bind_group_layouts: &[Some(&prep_bgl)],
            immediate_size: 0,
        });
        let prep_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("svgf_rt_prep_pipe"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("prep"),
            compilation_options: Default::default(),
            cache: None,
        });
        let prep_u = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("svgf_rt_prep_u"),
            size: size_of::<SvgfPrepUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            prep_pipe,
            prep_bgl,
            prep_u,
            sao: SvgfDenoiser::new(device),
            refl: SvgfDenoiser::new(device),
            gi: SvgfDenoiser::new(device),
            buffers: None,
            prev_vp: None,
        }
    }

    fn reset_history(&mut self) {
        self.sao.reset_history();
        self.refl.reset_history();
        self.gi.reset_history();
        self.prev_vp = None;
    }
}

pub(super) struct EnvPipelines {
    pub terrain: TerrainPipeline,
    pub water: WaterPipeline,
    pub foliage: FoliagePipeline,
}

impl EnvPipelines {
    fn new(device: &wgpu::Device, samples: u32, depth_mode: DepthMode) -> Self {
        let target = match depth_mode {
            DepthMode::ForwardZ => PipelineTarget {
                color_format: SCENE_FORMAT,
                depth_format: Some(wgpu::TextureFormat::Depth24PlusStencil8),
                depth_compare: wgpu::CompareFunction::LessEqual,
                sample_count: samples.max(1),
            },
            DepthMode::ReversedZ => PipelineTarget::reversed_z(
                SCENE_FORMAT,
                Some(wgpu::TextureFormat::Depth24PlusStencil8),
                samples.max(1),
            ),
        };
        Self {
            terrain: TerrainPipeline::new(device, target),
            water: WaterPipeline::new(device, target),
            foliage: FoliagePipeline::new(device, target),
        }
    }
}

struct DdgiCompositePipeline {
    u_bgl: wgpu::BindGroupLayout,
    cam_bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    cam_uniform: wgpu::Buffer,
}

#[derive(Default)]
pub(super) struct WiredPasses {
    pub physical_atmosphere: Option<PhysicalAtmosphere>,
    pub physical_sky_serial: u64,
    pub physical_sky_last_key: (bool, u64),
    pub pbr_decals: HashMap<u32, PbrDecalPass>,
    pub oit_pass: Option<OitPass>,
    pub oit_pipelines: HashMap<(usize, u64, u32, bool), Option<wgpu::RenderPipeline>>,
    pub blas_cache: HashMap<u64, Arc<Blas>>,
    pub tlas_hash: u64,
    pub tlas_gpu: Option<TlasSceneGpu>,
    pub tlas_bounds: Option<([f32; 3], [f32; 3])>,
    pub ray_tracer: Option<TlasRayTracer>,
    svgf_rt: Option<SvgfRtState>,
    rt_composite: Option<RtCompositePipelines>,
    pub ddgi: Option<DdgiVolume>,
    ddgi_composite: Option<DdgiCompositePipeline>,
    pub meshlet_culler: Option<MeshletCuller>,
    pub last_meshlet_cull_draws: u32,
    pub env_pipelines: HashMap<(u32, bool), EnvPipelines>,
    pub env_scene: Option<EnvironmentScene>,
    pub ext_terrain_hm: Option<(u64, TerrainHeightmap)>,
    pub rcas: Option<RcasSharpener>,
    pub post_fx: Option<PostFxPass>,
    pub split_screen: Option<(wgpu::TextureFormat, SplitScreenCompositor)>,
    pub split_views: Vec<SplitViewSlot>,
    pub polish_input: Option<(u32, u32, wgpu::TextureFormat, wgpu::Texture, wgpu::TextureView)>,
    pub msaa_depth_resolve: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
    pub dummy_depth: Option<(wgpu::Texture, wgpu::TextureView)>,
    blit_shared: Option<BlitShared>,
    blit_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    pub in_secondary_camera: bool,
    pub bypass_display_output: bool,
}

/// Builds a lightweight secondary-camera `RenderFrame` cloning only 3D scene fields and
/// omitting audio, UI, services, haptics, video streams, camera targets, sprites, and insets.
pub(super) fn secondary_frame(frame: &RenderFrame, view: Matrix, projection: Matrix) -> RenderFrame {
    RenderFrame {
        view,
        projection,
        clear: frame.clear,
        ambient: frame.ambient,
        light_direction: frame.light_direction,
        light_color: frame.light_color,
        aberration: frame.aberration,
        quality: frame.quality,
        sky: frame.sky,
        probe_volume: frame.probe_volume,
        fog: frame.fog,
        shadow: frame.shadow,
        blur: frame.blur,
        clouds: frame.clouds,
        exposure: frame.exposure,
        grade: frame.grade,
        reflection: frame.reflection,
        motion_blur: frame.motion_blur,
        dof: frame.dof,
        draws: frame.draws.clone(),
        solids: frame.solids.clone(),
        decals: frame.decals.clone(),
        poses: frame.poses.clone(),
        emitters: frame.emitters.clone(),
        lights: frame.lights.clone(),
        cascades: frame.cascades.clone(),
        probes: frame.probes.clone(),
        reflection_probes: frame.reflection_probes.clone(),
        fog_volumes: frame.fog_volumes.clone(),
        bone_matrices: frame.bone_matrices.clone(),
        morph_weights: frame.morph_weights.clone(),
        motions: frame.motions.clone(),
        ribbons: frame.ribbons.clone(),
        area_lights: frame.area_lights.clone(),
        pbr_decals: frame.pbr_decals.clone(),
        ..RenderFrame::default()
    }
}

impl WiredPasses {
    pub fn reset_svgf_history(&mut self) {
        if let Some(svgf) = &mut self.svgf_rt {
            svgf.reset_history();
        }
    }

    pub fn wants_post_polish(frame: &RenderFrame) -> bool {
        frame.grade.sharpness > 0.0
            || frame.quality.upscale
            || frame.grade.grain > 0.0
            || frame.grade.flare > 0.0
            || frame.grade.outline_strength > 0.0
    }

    pub fn polish_target(
        &mut self,
        device: &wgpu::Device,
        size: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        let size = (size.0.max(1), size.1.max(1));
        if !matches!(&self.polish_input, Some((w, h, fmt, _, _)) if *w == size.0 && *h == size.1 && *fmt == format) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("post_polish_input"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.polish_input = Some((size.0, size.1, format, tex, view));
        }
        self.polish_input.as_ref().unwrap().4.clone()
    }

    pub fn dummy_depth_view(&mut self, device: &wgpu::Device) -> wgpu::TextureView {
        if self.dummy_depth.is_none() {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("wired_dummy_depth"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.dummy_depth = Some((tex, view));
        }
        self.dummy_depth.as_ref().unwrap().1.clone()
    }

    /// Resolves a multisampled depth view into a cached 1x `R32Float` texture via `BuiltIn::OpaqueDepth`
    /// so screen-space outlines work identically under MSAA.
    pub fn resolve_msaa_depth(
        &mut self,
        device: &wgpu::Device,
        built_ins: &mut BuiltIns,
        encoder: &mut wgpu::CommandEncoder,
        depth_only: &wgpu::TextureView,
        size: (u32, u32),
        samples: u32,
    ) -> wgpu::TextureView {
        let size = (size.0.max(1), size.1.max(1));
        if !matches!(&self.msaa_depth_resolve, Some((w, h, _, _)) if *w == size.0 && *h == size.1) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("wired_msaa_depth_resolve"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.msaa_depth_resolve = Some((size.0, size.1, tex, view));
        }
        let dst = self.msaa_depth_resolve.as_ref().unwrap().3.clone();
        let pipeline = built_ins.get(BuiltIn::OpaqueDepth(samples));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wired_msaa_depth_resolve_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_only) }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("wired_msaa_depth_resolve"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
        dst
    }

    fn ensure_blit(&mut self, device: &wgpu::Device, target_format: wgpu::TextureFormat) {
        let shared = self.blit_shared.get_or_insert_with(|| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("wired_copy_blit"),
                source: wgpu::ShaderSource::Wgsl(COPY_BLIT_WGSL.into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("wired_copy_blit_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("wired_copy_blit_pl"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("wired_copy_blit_samp"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            BlitShared { module, bgl, pl, sampler }
        });
        self.blit_pipelines.entry(target_format).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("wired_copy_blit"),
                layout: Some(&shared.pl),
                vertex: wgpu::VertexState {
                    module: &shared.module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shared.module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
    }

    pub fn blit(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
    ) {
        self.ensure_blit(device, target_format);
        let shared = self.blit_shared.as_ref().unwrap();
        let pipeline = self.blit_pipelines.get(&target_format).unwrap();
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wired_copy_blit_bg"),
            layout: &shared.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shared.sampler) },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("wired_copy_blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    fn ensure_rt_composite(&mut self, device: &wgpu::Device) -> &RtCompositePipelines {
        self.rt_composite.get_or_insert_with(|| {
            let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            };
            let u_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(size_of::<RtCompUniform>() as u64),
                },
                count: None,
            };
            let sao_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rt_sao_comp_bgl"),
                entries: &[tex_entry(0), tex_entry(1), u_entry(2)],
            });
            let sao_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rt_sao_comp_pl"),
                bind_group_layouts: &[Some(&sao_bgl)],
                immediate_size: 0,
            });
            let sao_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rt_sao_comp"),
                source: wgpu::ShaderSource::Wgsl(RT_COMPOSITE_WGSL.into()),
            });
            let sao_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("rt_sao_comp"),
                layout: Some(&sao_pl),
                vertex: wgpu::VertexState {
                    module: &sao_mod,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &sao_mod,
                    entry_point: Some("fs_shadow_ao"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: SCENE_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Dst,
                                dst_factor: wgpu::BlendFactor::Zero,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

            let bounce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rt_bounce_comp_bgl"),
                entries: &[tex_entry(0), tex_entry(1), tex_entry(2), u_entry(3)],
            });
            let bounce_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rt_bounce_comp_pl"),
                bind_group_layouts: &[Some(&bounce_bgl)],
                immediate_size: 0,
            });
            let bounce_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rt_bounce_comp"),
                source: wgpu::ShaderSource::Wgsl(RT_BOUNCE_WGSL.into()),
            });
            let bounce_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("rt_bounce_comp"),
                layout: Some(&bounce_pl),
                vertex: wgpu::VertexState {
                    module: &bounce_mod,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &bounce_mod,
                    entry_point: Some("fs_bounce"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: SCENE_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
            let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rt_comp_u"),
                size: size_of::<RtCompUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            RtCompositePipelines { sao_bgl, sao_pipeline, bounce_bgl, bounce_pipeline, uniform }
        })
    }

    fn ensure_ddgi_composite(&mut self, device: &wgpu::Device) -> &DdgiCompositePipeline {
        self.ddgi_composite.get_or_insert_with(|| {
            let u_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ddgi_comp_u_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
            let tex_entry = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            };
            let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ddgi_comp_cam_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(size_of::<DdgiCompCamUniform>() as u64),
                        },
                        count: None,
                    },
                    tex_entry(1, false),
                    tex_entry(2, true),
                    tex_entry(3, true),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ddgi_comp_pl"),
                bind_group_layouts: &[Some(&u_bgl), Some(&cam_bgl)],
                immediate_size: 0,
            });
            let source = format!("{TLAS_TRAVERSAL_WGSL}\n{DDGI_WGSL}\n{DDGI_COMPOSITE_SUFFIX_WGSL}");
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ddgi_composite"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ddgi_composite"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("ddgi_comp_vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("ddgi_comp_fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: SCENE_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
            let cam_uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ddgi_comp_cam_u"),
                size: size_of::<DdgiCompCamUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            DdgiCompositePipeline { u_bgl, cam_bgl, pipeline, cam_uniform }
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn composite_ray_tracing(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene_view: &wgpu::TextureView,
        depth_r32f: &wgpu::TextureView,
        sao_view: &wgpu::TextureView,
        refl_view: Option<&wgpu::TextureView>,
        gi_view: Option<&wgpu::TextureView>,
        tier: u32,
        depth_clear: f32,
    ) {
        let comp = self.ensure_rt_composite(device);
        let cfg = RtCompUniform { depth_clear, _p: [0.0; 3] };
        queue.write_buffer(&comp.uniform, 0, bytemuck::bytes_of(&cfg));
        let sao_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_sao_comp_bg"),
            layout: &comp.sao_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_r32f) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(sao_view) },
                wgpu::BindGroupEntry { binding: 2, resource: comp.uniform.as_entire_binding() },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rt_shadow_ao_composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&comp.sao_pipeline);
            pass.set_bind_group(0, &sao_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        if tier >= 2
            && let (Some(rv), Some(gv)) = (refl_view, gi_view)
        {
            let bounce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rt_bounce_comp_bg"),
                layout: &comp.bounce_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_r32f) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(rv) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(gv) },
                    wgpu::BindGroupEntry { binding: 3, resource: comp.uniform.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rt_bounce_composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&comp.bounce_pipeline);
            pass.set_bind_group(0, &bounce_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn composite_ddgi(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene_view: &wgpu::TextureView,
        depth_r32f: &wgpu::TextureView,
        inv_vp: [f32; 16],
        camera_pos: [f32; 3],
        depth_clear: f32,
    ) {
        if self.ddgi.is_none() {
            return;
        }
        let _ = self.ensure_ddgi_composite(device);
        let comp = self.ddgi_composite.as_ref().unwrap();
        let ddgi = self.ddgi.as_ref().unwrap();
        let cam_u = DdgiCompCamUniform {
            inverse_view_projection: inv_vp,
            camera_pos_and_strength: [camera_pos[0], camera_pos[1], camera_pos[2], 0.55],
            depth_cfg: [depth_clear, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&comp.cam_uniform, 0, bytemuck::bytes_of(&cam_u));
        let u_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ddgi_comp_u_bg"),
            layout: &comp.u_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: ddgi.uniform_buffer().as_entire_binding() }],
        });
        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ddgi_comp_cam_bg"),
            layout: &comp.cam_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: comp.cam_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_r32f) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(ddgi.irradiance_view()) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(ddgi.visibility_view()) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(ddgi.sampler()) },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ddgi_composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: scene_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&comp.pipeline);
        pass.set_bind_group(0, &u_bg, &[]);
        pass.set_bind_group(1, &cam_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

impl State {
    /// Executes any non-split-screen RTT `frame.camera_targets` before the main pass so materials
    /// referencing those dynamic textures read fresh images without recursion.
    pub(super) fn run_rtt_cameras(&mut self, frame: &RenderFrame, catalog: &Catalog) -> Result<()> {
        if self.wired.in_secondary_camera || frame.camera_targets.is_empty() {
            return Ok(());
        }
        self.wired.in_secondary_camera = true;
        let result = (|| -> Result<()> {
            for cam in &frame.camera_targets {
                if cam.split_screen {
                    continue;
                }
                let Some(mat) = self.material(catalog, cam.material)? else { continue };
                let Some(key) = mat.material.texture.clone() else { continue };
                let w = if cam.width == 0 { 256 } else { cam.width.clamp(16, 4096) };
                let h = if cam.height == 0 { 256 } else { cam.height.clamp(16, 4096) };
                let sub = secondary_frame(frame, cam.view, cam.projection);
                self.render_camera_to_texture(&key, &sub, catalog, (w, h))?;
            }
            Ok(())
        })();
        self.wired.in_secondary_camera = false;
        result
    }

    /// Ensures physical atmosphere LUTs & optional cloud target exist before `Post::begin`
    /// builds the pass graph, returning their views and generation key for `BuiltIn::Atmosphere`.
    pub(super) fn prepare_physical_sky(
        &mut self,
        frame: &RenderFrame,
        scene_size: (u32, u32),
    ) -> Result<Option<PhysicalSkyBinding>> {
        if !frame.quality.physical_sky {
            return Ok(None);
        }
        let device = self.device.clone();
        let atmo = self.wired.physical_atmosphere.get_or_insert_with(|| PhysicalAtmosphere::new(&device));
        let wants_clouds = frame.clouds.cover > 0.001;
        let cloud_serial = if wants_clouds {
            let cw = (scene_size.0 / 2).clamp(16, 1024);
            let ch = (scene_size.1 / 2).clamp(16, 1024);
            atmo.ensure_clouds(&device, (cw, ch))?
        } else {
            0
        };
        let key = (wants_clouds, cloud_serial);
        if self.wired.physical_sky_serial == 0 || self.wired.physical_sky_last_key != key {
            self.wired.physical_sky_serial += 1;
            self.wired.physical_sky_last_key = key;
        }
        Ok(Some(PhysicalSkyBinding {
            serial: self.wired.physical_sky_serial,
            sky_view: atmo.sky_view_view().clone(),
            transmittance: atmo.transmittance_view().clone(),
            clouds: if wants_clouds { atmo.cloud_view().cloned() } else { None },
        }))
    }

    /// Updates Hillaire physical atmosphere LUTs & volumetric clouds when `quality.physical_sky` is on.
    /// Uses the active (potentially TAA-jittered) projection and unified world-meters-to-km camera coordinates.
    pub(super) fn encode_physical_sky(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        active_projection: &Matrix,
        scene_size: (u32, u32),
    ) -> Result<()> {
        if !frame.quality.physical_sky {
            return Ok(());
        }
        let device = self.device.clone();
        let atmo = self.wired.physical_atmosphere.get_or_insert_with(|| PhysicalAtmosphere::new(&device));
        let eye = invert(&frame.view);
        let world_eye = [eye[12], eye[13], eye[14]];
        let cam_km = physical_camera_km(world_eye);
        let mut params = AtmosphereParams {
            camera_altitude_km: cam_km[1],
            sun_direction: [-frame.light_direction.x, -frame.light_direction.y, -frame.light_direction.z],
            sun_irradiance: [frame.light_color.x, frame.light_color.y, frame.light_color.z],
            cloud_coverage: frame.clouds.cover.clamp(0.0, 1.0),
            cloud_wind_km: [frame.clouds.drift.x * 0.01, 0.0, frame.clouds.drift.z * 0.01],
            ..Default::default()
        };
        if frame.clouds.cover <= 0.001 {
            params.cloud_density = 0.0;
        }
        atmo.update_luts(&device, encoder, &params);
        if frame.clouds.cover > 0.001 {
            let vp = multiply(active_projection, &frame.view);
            let inv_vp = invert(&vp);
            let cw = (scene_size.0 / 2).clamp(16, 1024);
            let ch = (scene_size.1 / 2).clamp(16, 1024);
            atmo.render_clouds_world(
                &device,
                encoder,
                &params,
                inv_vp,
                world_eye,
                WORLD_METERS_TO_KM,
                BASE_CAMERA_ALTITUDE_KM,
                (cw, ch),
            )?;
        }
        Ok(())
    }

    /// Resolves cookie textures named by `frame.area_lights[i].cookie_material` (1-based catalog material
    /// index) into packed storage slots `0..MAX_AREA_COOKIES`.
    pub(super) fn prepare_area_cookies(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
    ) -> Result<PreparedCookies> {
        if frame.area_lights.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut slots = Vec::with_capacity(frame.area_lights.len().min(64));
        let mut uploads: Vec<CookieUpload> = Vec::new();
        let mut view_to_slot: HashMap<u64, u32> = HashMap::new();
        for spec in frame.area_lights.iter().take(64) {
            if spec.kind != 3 || spec.cookie_material == 0 {
                slots.push(None);
                continue;
            }
            let mat_idx = spec.cookie_material - 1;
            let Some(mat) = self.material(catalog, mat_idx)? else {
                slots.push(None);
                continue;
            };
            let Some(tex_key) = &mat.material.texture else {
                slots.push(None);
                continue;
            };
            let tex_ref = if let Some(&vid) = self.video_ids.get(tex_key.as_str()) {
                Some(TextureRef::Video(vid))
            } else if let Some(&cid) = self.camera_target_ids.get(tex_key.as_str()) {
                Some(TextureRef::CameraTarget(cid))
            } else if !self.reserved_videos.contains(tex_key.as_str()) {
                self.loaded_texture(&mat.material.texture)?.map(TextureRef::Slot)
            } else {
                None
            };
            let Some(tref) = tex_ref else {
                slots.push(None);
                continue;
            };
            let tex = self.texture(tref);
            let view_id = tex.view_id;
            if let Some(&existing) = view_to_slot.get(&view_id) {
                slots.push(Some(existing));
            } else if uploads.len() < MAX_AREA_COOKIES {
                let slot = u32::try_from(uploads.len()).unwrap_or(0);
                view_to_slot.insert(view_id, slot);
                uploads.push((slot, tex.view.clone(), view_id));
                slots.push(Some(slot));
            } else {
                slots.push(None);
            }
        }
        Ok((slots, uploads))
    }

    /// Resolves PBR decal materials/textures ahead of constructing `Drawing` so the render pass
    /// can borrow `self.wired` disjointly from `self.assets`.
    pub(super) fn prepare_pbr_decals(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
    ) -> Result<Vec<PreparedPbrDecalBatch>> {
        if frame.pbr_decals.is_empty() {
            return Ok(Vec::new());
        }
        let mut by_mat: HashMap<u32, Vec<PbrDecal>> = HashMap::new();
        for spec in &frame.pbr_decals {
            let mut d = PbrDecal::from_transform(
                spec.transform,
                [spec.albedo.r, spec.albedo.g, spec.albedo.b, spec.albedo.a],
                [spec.emissive_strength.max(0.0); 3],
                spec.normal_strength.clamp(0.0, 4.0),
                spec.roughness.clamp(0.04, 1.0),
                spec.metallic.clamp(0.0, 1.0),
            );
            d.pbr_params[3] = spec.fade.clamp(0.0, 1.0);
            by_mat.entry(spec.material).or_default().push(d);
        }

        let mut prepared = Vec::with_capacity(by_mat.len());
        for (mat_idx, decals) in by_mat {
            let Some(mat) = self.material(catalog, mat_idx)? else { continue };
            let tex_ref = |state: &mut State, key: &Option<String>, fallback: TextureRef| -> Result<TextureRef> {
                match key {
                    Some(k) => {
                        let slot = state.resolve(AssetKind::Texture, k)?;
                        Ok(if matches!(state.assets.current(slot), Some(GpuAsset::Texture(_))) {
                            TextureRef::Slot(slot)
                        } else {
                            fallback
                        })
                    }
                    None => Ok(fallback),
                }
            };
            let a_ref = tex_ref(self, &mat.material.texture, TextureRef::White)?;
            let n_ref = tex_ref(self, &mat.material.normal_texture, TextureRef::FlatNormal)?;
            let o_ref = tex_ref(self, &mat.material.orm_texture, TextureRef::White)?;
            prepared.push(PreparedPbrDecalBatch {
                albedo: self.texture(a_ref).view.clone(),
                normal: self.texture(n_ref).linear.clone(),
                orm: self.texture(o_ref).linear.clone(),
                decals,
            });
        }
        Ok(prepared)
    }

    /// Encodes prepared PBR decal batches via `PbrDecalPass` onto the opaque HDR scene target.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_pbr_decals_prepared(
        wired: &mut WiredPasses,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        batches: &[PreparedPbrDecalBatch],
        color_attachment: wgpu::RenderPassColorAttachment<'_>,
        depth_view: &wgpu::TextureView,
        depth_only: &wgpu::TextureView,
        samples: u32,
        scene_size: (u32, u32),
        depth_mode: DepthMode,
    ) -> Result<()> {
        if batches.is_empty() {
            return Ok(());
        }
        if let std::collections::hash_map::Entry::Vacant(e) = wired.pbr_decals.entry(samples) {
            let pass = PbrDecalPass::new(
                device,
                samples,
                SCENE_FORMAT,
                wgpu::TextureFormat::Depth24PlusStencil8,
                true,
            )?;
            e.insert(pass);
        }
        let pass_obj = wired.pbr_decals.get_mut(&samples).unwrap();
        pass_obj.begin_frame();

        let vp = multiply(&frame.projection, &frame.view);
        let eye = invert(&frame.view);
        let decal_view = PbrDecalView {
            view_projection: vp,
            camera_pos: [eye[12], eye[13], eye[14]],
            sun_direction: [-frame.light_direction.x, -frame.light_direction.y, -frame.light_direction.z],
            sun_color: [frame.light_color.x, frame.light_color.y, frame.light_color.z],
            ambient_intensity: 1.0,
            sky_ambient: [frame.sky.zenith.x, frame.sky.zenith.y, frame.sky.zenith.z],
            ground_ambient: [frame.sky.ground.x, frame.sky.ground.y, frame.sky.ground.z],
            depth_clear: depth_mode.clear_value(),
            viewport: Some([0.0, 0.0, scene_size.0 as f32, scene_size.1 as f32]),
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pbr_decals"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: None,
                stencil_ops: None,
            }),
            ..Default::default()
        });
        for batch in batches {
            pass_obj.draw(
                device,
                queue,
                &mut render_pass,
                &decal_view,
                &batch.decals,
                depth_only,
                &batch.albedo,
                &batch.normal,
                &batch.orm,
            )?;
        }
        Ok(())
    }

    /// Runs AMD FidelityFX RCAS sharpening and/or Post-FX polish (film grain, anamorphic lens flare,
    /// screen-space outlines) on `polish_src` and writes the final result into `dst_output`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_post_polish_wired(
        wired: &mut WiredPasses,
        built_ins: &mut BuiltIns,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        frames_count: u64,
        target_format: wgpu::TextureFormat,
        polish_src: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        depth_size: (u32, u32),
        dst_output: &wgpu::TextureView,
        size: (u32, u32),
        samples: u32,
    ) -> Result<()> {
        let mut current = polish_src.clone();

        let sharp = if frame.grade.sharpness > 0.0 {
            frame.grade.sharpness.clamp(0.0, 1.0)
        } else if frame.quality.upscale {
            0.35
        } else {
            0.0
        };
        if sharp > 0.0 {
            let rcas = wired.rcas.get_or_insert_with(|| RcasSharpener::new(device));
            rcas.sharpen(device, encoder, &current, size, sharp)?;
            if let Some(view) = rcas.output_view() {
                current = view.clone();
            }
        }

        if frame.grade.grain > 0.0 || frame.grade.flare > 0.0 || frame.grade.outline_strength > 0.0 {
            let outline = frame.grade.outline_strength.clamp(0.0, 1.0);
            let depth_for_fx = if samples > 1 {
                if outline > 0.0 {
                    wired.resolve_msaa_depth(device, built_ins, encoder, depth_view, depth_size, samples)
                } else {
                    wired.dummy_depth_view(device)
                }
            } else {
                depth_view.clone()
            };
            let fx = wired.post_fx.get_or_insert_with(|| PostFxPass::new(device));
            let params = PostFxParams {
                grain_strength: frame.grade.grain.clamp(0.0, 1.0),
                frame_seed: (frames_count % 4096) as f32 * 0.618_034,
                flare_strength: frame.grade.flare.clamp(0.0, 2.0),
                outline_strength: outline,
                inverse_projection: Some(invert(&frame.projection)),
                ..Default::default()
            };
            fx.dispatch(device, encoder, &current, &depth_for_fx, size, &params)?;
            if let Some(view) = fx.output_view() {
                current = view.clone();
            }
        }

        wired.blit(device, encoder, &current, dst_output, target_format);
        Ok(())
    }

    /// Pre-validates split-screen pixel rectangles and renders each `split_screen == true` camera
    /// into its own temporally-isolated offscreen target in presentation format (`self.context.format`),
    /// bypassing display decode and global accessibility filters so those run once on the composite.
    pub(super) fn render_split_screen_views(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        output_size: (u32, u32),
    ) -> Result<(Vec<ViewportRect>, bool, crate::RenderStats)> {
        let split_cams: Vec<_> = frame.camera_targets.iter().filter(|c| c.split_screen).cloned().collect();
        anyhow::ensure!(
            output_size.0 > 0 && output_size.1 > 0,
            "split-screen output target must be non-empty"
        );
        let mut rects: Vec<ViewportRect> = Vec::with_capacity(split_cams.len());
        for cam in &split_cams {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "validated normalized [0, 1] viewport"
            )]
            let (x0, y0, x1, y1) = {
                let ow = output_size.0 as f32;
                let oh = output_size.1 as f32;
                let x0 = (cam.viewport_x * ow).round().clamp(0.0, ow) as u32;
                let y0 = (cam.viewport_y * oh).round().clamp(0.0, oh) as u32;
                let x1 = ((cam.viewport_x + cam.viewport_w) * ow).round().clamp(0.0, ow) as u32;
                let y1 = ((cam.viewport_y + cam.viewport_h) * oh).round().clamp(0.0, oh) as u32;
                (x0, y0, x1, y1)
            };
            anyhow::ensure!(
                x0 < output_size.0 && y0 < output_size.1 && x1 > x0 && y1 > y0,
                "split-screen viewport ({}, {}, {}, {}) produces an empty or out-of-bounds rectangle on {}x{}",
                cam.viewport_x,
                cam.viewport_y,
                cam.viewport_w,
                cam.viewport_h,
                output_size.0,
                output_size.1
            );
            rects.push(ViewportRect { x: x0, y: y0, width: x1 - x0, height: y1 - y0 });
        }

        let format = self.context.format;
        let mut views = std::mem::take(&mut self.wired.split_views);
        let mut any_waiting = false;
        let mut combined = crate::RenderStats::default();
        let mut scale_sum = 0.0f32;
        let mut rendered_count = 0u32;

        let res = (|| -> Result<()> {
            for (idx, (cam, rect)) in split_cams.iter().zip(&rects).enumerate() {
                let size = (rect.width, rect.height);
                if idx >= views.len() {
                    let target = OffscreenRenderTarget::new(&self.device, size.0, size.1, format)?;
                    let temporal = ViewTemporalState::new(&self.device, &self.queue);
                    views.push(SplitViewSlot { target, temporal });
                } else if views[idx].target.size() != size || views[idx].target.format() != format {
                    views[idx].target = OffscreenRenderTarget::new(&self.device, size.0, size.1, format)?;
                }
                let sub = secondary_frame(frame, cam.view, cam.projection);
                let slot = &mut views[idx];
                let (child_waiting, child_stats) =
                    self.submit_secondary_view(&sub, catalog, &mut slot.target, &mut slot.temporal)?;
                any_waiting |= child_waiting;
                combined.draw_calls += child_stats.draw_calls;
                combined.instanced_draw_calls += child_stats.instanced_draw_calls;
                combined.instances += child_stats.instances;
                combined.instance_upload_bytes += child_stats.instance_upload_bytes;
                combined.instance_upload_calls += child_stats.instance_upload_calls;
                combined.triangles += child_stats.triangles;
                combined.culled += child_stats.culled;
                combined.shadow_culled += child_stats.shadow_culled;
                combined.local_shadow_casters += child_stats.local_shadow_casters;
                combined.local_shadow_faces += child_stats.local_shadow_faces;
                combined.local_shadow_kept += child_stats.local_shadow_kept;
                combined.volume_froxels += child_stats.volume_froxels;
                combined.volume_history |= child_stats.volume_history;
                if combined.exposure.is_none() {
                    combined.exposure = child_stats.exposure;
                }
                scale_sum += child_stats.scale;
                rendered_count += 1;
            }
            views.truncate(split_cams.len());
            Ok(())
        })();
        self.wired.split_views = views;
        res?;
        if rendered_count > 0 {
            combined.scale = scale_sum / rendered_count as f32;
        }
        Ok((rects, any_waiting, combined))
    }

    /// Composites already-rendered `split_views` onto `output`.
    pub(super) fn composite_split_screen(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        output_size: (u32, u32),
        rects: Vec<ViewportRect>,
    ) -> Result<()> {
        let format = self.context.format;
        let device = self.device.clone();
        if !matches!(&self.wired.split_screen, Some((fmt, _)) if *fmt == format) {
            self.wired.split_screen =
                Some((format, SplitScreenCompositor::new(&device, format, ColorTransfer::Passthrough)));
        }
        let (_, comp) = self.wired.split_screen.as_mut().unwrap();
        let views: Vec<&wgpu::TextureView> =
            self.wired.split_views.iter().take(rects.len()).map(|s| s.target.render_view()).collect();
        let layout = crate::render_targets::SplitScreenLayout {
            output_size,
            viewports: rects,
            divider_px: 0,
        };
        comp.composite(&device, encoder, output, &layout, &views, wgpu::Color::BLACK)?;
        Ok(())
    }

    /// Prepares (lazily compiling and caching) WBOIT render pipelines for every translucent entry
    /// when `frame.quality.oit` is enabled.
    pub(super) fn prepare_oit_pipelines(
        &mut self,
        frame: &RenderFrame,
        scratch: &Scratch,
        samples: u32,
    ) -> Vec<Option<wgpu::RenderPipeline>> {
        if !frame.quality.oit || scratch.translucent.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(scratch.translucent.len());
        for &ready_idx in &scratch.translucent {
            let entry = &scratch.ready[ready_idx];
            let skinned = entry.skin.is_some();
            let Some(GpuAsset::Shader(sh)) = self.assets.current(entry.shader) else {
                out.push(None);
                continue;
            };
            let key = (entry.shader, sh.revision, samples, skinned);
            if let Some(cached) = self.wired.oit_pipelines.get(&key) {
                out.push(cached.clone());
                continue;
            }
            let code = sh.code.clone();
            let stage = if skinned { Stage::Skinned } else { Stage::Plain };
            let compiled = crate::oit::derive_oit_wgsl(&code).and_then(|oit_wgsl| {
                let (pipe, mut check) = shader::scoped(&self.device, || {
                    let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("oit_derived_shader"),
                        source: wgpu::ShaderSource::Wgsl(oit_wgsl.into()),
                    });
                    shader::oit_pipeline(&self.context, &module, samples, stage)
                });
                match shader::settled(&mut check) {
                    Some(Ok(())) | None => Some(pipe),
                    Some(Err(_)) => None,
                }
            });
            self.wired.oit_pipelines.insert(key, compiled.clone());
            out.push(compiled);
        }
        out
    }

    /// Lazily builds per-mesh BLASes and updates the cached 2-level `TlasSceneGpu` whenever
    /// `frame.quality.ray_tracing > 0 || frame.quality.ddgi` and opaque geometry changed.
    pub(super) fn prepare_tlas_scene(&mut self, frame: &RenderFrame, scratch: &Scratch) -> Result<bool> {
        if frame.quality.ray_tracing == 0 && !frame.quality.ddgi {
            return Ok(false);
        }
        let mut blas_id_to_idx: HashMap<u64, u32> = HashMap::new();
        let mut blas_ids: Vec<u64> = Vec::new();
        let mut instances: Vec<TlasInstance> = Vec::new();
        let mut world_min = [f32::INFINITY; 3];
        let mut world_max = [f32::NEG_INFINITY; 3];

        let mut push_inst = |blas_cache: &mut HashMap<u64, Arc<Blas>>,
                             source: &Arc<super::RtMeshSource>,
                             transform: [f32; 16],
                             m: &crate::asset_store::Material|
         -> Result<()> {
            let blas = match blas_cache.entry(source.id) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => entry.insert(Arc::new(Blas::build(&source.triangles)?)),
            };
            if blas.nodes().is_empty() {
                return Ok(());
            }
            let inv = invert(&transform);
            if !inv.iter().all(|v| v.is_finite()) {
                return Ok(());
            }
            let bi = *blas_id_to_idx.entry(source.id).or_insert_with(|| {
                let idx = u32::try_from(blas_ids.len()).unwrap_or(u32::MAX);
                blas_ids.push(source.id);
                idx
            });
            let (bmin, bmax) = blas.bounds();
            for &x in &[bmin[0], bmax[0]] {
                for &y in &[bmin[1], bmax[1]] {
                    for &z in &[bmin[2], bmax[2]] {
                        let wx = transform[0] * x + transform[4] * y + transform[8] * z + transform[12];
                        let wy = transform[1] * x + transform[5] * y + transform[9] * z + transform[13];
                        let wz = transform[2] * x + transform[6] * y + transform[10] * z + transform[14];
                        world_min[0] = world_min[0].min(wx);
                        world_min[1] = world_min[1].min(wy);
                        world_min[2] = world_min[2].min(wz);
                        world_max[0] = world_max[0].max(wx);
                        world_max[1] = world_max[1].max(wy);
                        world_max[2] = world_max[2].max(wz);
                    }
                }
            }
            #[allow(clippy::cast_possible_truncation, reason = "clamped finite material color")]
            let albedo = [
                m.color[0].clamp(0.0, 10.0) as f32,
                m.color[1].clamp(0.0, 10.0) as f32,
                m.color[2].clamp(0.0, 10.0) as f32,
            ];
            let emissive = if !m.lit {
                let s = m.surface.emissive_strength.max(1.0);
                [albedo[0] * s, albedo[1] * s, albedo[2] * s]
            } else if m.surface.emissive_strength > 1.0 {
                let s = m.surface.emissive_strength - 1.0;
                [albedo[0] * s, albedo[1] * s, albedo[2] * s]
            } else {
                [0.0; 3]
            };
            let mut inst = TlasInstance::new(bi, transform);
            inst.albedo = albedo;
            inst.emissive = emissive;
            inst.roughness = if m.surface.clearcoat > 0.0 { 0.2 } else { 0.45 };
            inst.metallic = 0.0;
            instances.push(inst);
            Ok(())
        };

        for &ready_idx in &scratch.opaque {
            let entry = &scratch.ready[ready_idx];
            let item = &scratch.items[entry.item];
            let mat = &entry.material.material;
            match item.shape {
                Shape::Mesh(_) => {
                    if let Some(GpuAsset::Mesh(mesh)) = entry.mesh.and_then(|slot| self.assets.current(slot))
                        && let Some(src) = &mesh.rt_source
                    {
                        push_inst(&mut self.wired.blas_cache, src, *item.transform(frame), mat)?;
                    }
                }
                Shape::Solid(idx) => {
                    if let Some(solid) = self.solid_meshes.get(idx)
                        && let Some(src) = &solid.rt_source
                    {
                        push_inst(&mut self.wired.blas_cache, src, IDENTITY, mat)?;
                    }
                }
                Shape::Batch(batch_idx) => {
                    let batch = &scratch.batches[batch_idx];
                    if let Some(GpuAsset::Mesh(mesh)) = self.assets.current(batch.mesh)
                        && let Some(src) = &mesh.rt_source
                    {
                        for (draw_idx, placement) in scratch.placements.iter().enumerate() {
                            if placement.batch as usize == batch_idx
                                && let Some(draw) = frame.draws.get(draw_idx)
                            {
                                push_inst(&mut self.wired.blas_cache, src, draw.transform, mat)?;
                            }
                        }
                    }
                }
            }
        }

        if instances.is_empty() {
            self.wired.tlas_gpu = None;
            self.wired.tlas_bounds = None;
            return Ok(false);
        }

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        blas_ids.hash(&mut hasher);
        instances.len().hash(&mut hasher);
        for inst in &instances {
            inst.blas_index.hash(&mut hasher);
            for v in inst.transform {
                v.to_bits().hash(&mut hasher);
            }
            for v in inst.albedo {
                v.to_bits().hash(&mut hasher);
            }
            for v in inst.emissive {
                v.to_bits().hash(&mut hasher);
            }
        }
        let scene_hash = hasher.finish();
        if self.wired.tlas_gpu.is_none() || self.wired.tlas_hash != scene_hash {
            let blases: Vec<Blas> = blas_ids
                .iter()
                .map(|id| (**self.wired.blas_cache.get(id).expect("cached BLAS")).clone())
                .collect();
            let tlas = TlasScene::build(&blases, &instances)?;
            self.wired.tlas_gpu = Some(TlasSceneGpu::upload(&self.device, &tlas)?);
            self.wired.tlas_hash = scene_hash;
        }
        self.wired.tlas_bounds = Some((world_min, world_max));
        Ok(true)
    }

    /// Dispatches GPU meshlet cluster culling (local frustum + shear-safe normal-cone backface +
    /// optional prior-frame Hi-Z) for opaque LOD0 mesh draws with `MeshAsset::meshlets`.
    pub(super) fn encode_meshlet_culling(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scratch: &mut Scratch,
        active_proj: &Matrix,
        depth_mode: DepthMode,
    ) -> Result<()> {
        self.wired.last_meshlet_cull_draws = 0;
        let active_vp = multiply(active_proj, &frame.view);
        let planes = frustum_planes_from_vp(&active_vp);
        let eye = invert(&frame.view);
        let camera_pos = [eye[12], eye[13], eye[14]];
        let reversed_z = depth_mode == DepthMode::ReversedZ;
        let (hiz_view_ref, _, hiz_vp, _, hiz_mips, hiz_active) = self.hiz.source();
        let use_hiz = hiz_active && !reversed_z;
        let hiz_opt = use_hiz.then_some(hiz_view_ref);

        let mut slot_cursor = 0usize;
        for &ready_idx in &scratch.opaque {
            let entry = &scratch.ready[ready_idx];
            if entry.skin.is_some() || entry.morph.is_some() {
                continue;
            }
            let item = &scratch.items[entry.item];
            if item.lod != 0 || !matches!(item.shape, Shape::Mesh(_)) {
                continue;
            }
            let Some(GpuAsset::Mesh(mesh)) = entry.mesh.and_then(|s| self.assets.current(s)) else {
                continue;
            };
            if mesh.instances.is_some() {
                continue;
            }
            let Some(ml) = &mesh.meshlets else {
                continue;
            };
            if ml.count <= 1 {
                continue;
            }
            let need_bytes = u64::from(ml.count) * size_of::<MeshletIndirectDraw>() as u64;
            if slot_cursor >= self.meshlet_indirect.len() {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("meshlet_indirect_draws"),
                    size: need_bytes.max(1280),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                    mapped_at_creation: false,
                });
                self.meshlet_indirect.push((ml.count, buf));
            } else if self.meshlet_indirect[slot_cursor].1.size() < need_bytes {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("meshlet_indirect_draws"),
                    size: need_bytes.next_power_of_two(),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                    mapped_at_creation: false,
                });
                self.meshlet_indirect[slot_cursor] = (ml.count, buf);
            } else {
                self.meshlet_indirect[slot_cursor].0 = ml.count;
            }
            let culler = self.wired.meshlet_culler.get_or_insert_with(|| MeshletCuller::new(device));
            let model = *item.transform(frame);
            let det3 = model[0] * (model[5] * model[10] - model[6] * model[9])
                - model[4] * (model[1] * model[10] - model[2] * model[9])
                + model[8] * (model[1] * model[6] - model[2] * model[5]);
            let params = MeshletCullParams {
                model,
                hiz_vp: if use_hiz { hiz_vp } else { active_vp },
                frustum_planes: planes,
                camera_pos,
                hiz_enabled: use_hiz,
                hiz_mips,
                backface_cull: det3 > 0.0,
                reversed_z,
            };
            culler.cull(
                device,
                encoder,
                &ml.descriptors,
                &self.meshlet_indirect[slot_cursor].1,
                ml.count,
                hiz_opt,
                &params,
            )?;
            scratch.ready[ready_idx].meshlet_slot = Some(slot_cursor);
            slot_cursor += 1;
        }
        self.wired.last_meshlet_cull_draws = u32::try_from(slot_cursor).unwrap_or(u32::MAX);
        Ok(())
    }

    /// Encodes outdoor GPU environment passes (CDLOD terrain, Gerstner water, GPU foliage)
    /// driven by either [`Renderer::set_environment_scene`] or optional Blot `RenderFrame::extensions`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_environment_passes(
        wired: &mut WiredPasses,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        frame: &RenderFrame,
        active_proj: &Matrix,
        samples: u32,
        depth_mode: DepthMode,
    ) -> Result<()> {
        let has_ext_env = frame.extensions.iter().any(|e| {
            matches!(e.tag, ENV_TERRAIN_EXT_TAG | ENV_WATER_EXT_TAG | ENV_FOLIAGE_EXT_TAG) && !e.values.is_empty()
        });
        if wired.env_scene.is_none() && !has_ext_env {
            return Ok(());
        }
        let key = (samples.max(1), depth_mode == DepthMode::ReversedZ);
        let pipes = wired.env_pipelines.entry(key).or_insert_with(|| EnvPipelines::new(device, samples, depth_mode));
        pipes.terrain.begin_frame();
        pipes.water.begin_frame();
        pipes.foliage.begin_frame();

        let active_vp = multiply(active_proj, &frame.view);
        let eye = invert(&frame.view);
        let cam_pos = [eye[12], eye[13], eye[14]];
        let sun_raw = [-frame.light_direction.x, -frame.light_direction.y, -frame.light_direction.z];
        let sl = (sun_raw[0] * sun_raw[0] + sun_raw[1] * sun_raw[1] + sun_raw[2] * sun_raw[2]).sqrt();
        let sun_dir = if sl > 1e-6 { [sun_raw[0] / sl, sun_raw[1] / sl, sun_raw[2] / sl] } else { [0.0, 1.0, 0.0] };
        let sun_col = [frame.light_color.x.max(0.0), frame.light_color.y.max(0.0), frame.light_color.z.max(0.0)];

        if let Some(scene) = &wired.env_scene {
            if let Some(terrain) = &scene.terrain {
                let mut u = terrain.uniform;
                if u.view_projection.iter().all(|v| *v == 0.0) {
                    u.view_projection = active_vp;
                    u.camera_pos_and_fog = [cam_pos[0], cam_pos[1], cam_pos[2], u.camera_pos_and_fog[3]];
                }
                pipes.terrain.draw(device, queue, pass, &u, &terrain.patches, &terrain.heightmap)?;
            }
            if let Some(fol) = &scene.foliage {
                let mut u = fol.uniform;
                if u.view_projection.iter().all(|v| *v == 0.0) {
                    u.view_projection = active_vp;
                    u.camera_pos_and_time = [cam_pos[0], cam_pos[1], cam_pos[2], u.camera_pos_and_time[3]];
                }
                pipes.foliage.draw(device, queue, pass, &u, &fol.instances)?;
            }
            for &wu in &scene.water {
                let mut u = wu;
                if u.view_projection.iter().all(|v| *v == 0.0) {
                    u.view_projection = active_vp;
                    u.camera_pos_and_time = [cam_pos[0], cam_pos[1], cam_pos[2], u.camera_pos_and_time[3]];
                }
                pipes.water.draw(queue, pass, &u)?;
            }
        }

        // Decode optional Blot environment extension sections.
        for ext in &frame.extensions {
            match ext.tag {
                ENV_TERRAIN_EXT_TAG if ext.values.len() >= 12 => {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let gw = (ext.values[0].round().clamp(2.0, 512.0)) as u32;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let gh = (ext.values[1].round().clamp(2.0, 512.0)) as u32;
                    let expected = (gw as usize) * (gh as usize);
                    if ext.values.len() >= 8 + expected {
                        let heights = &ext.values[8..8 + expected];
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        gw.hash(&mut hasher);
                        gh.hash(&mut hasher);
                        for &h in heights {
                            h.to_bits().hash(&mut hasher);
                        }
                        let h_hash = hasher.finish();
                        if !matches!(&wired.ext_terrain_hm, Some((prev, _)) if *prev == h_hash) {
                            let hm = TerrainPipeline::upload_heightmap(device, queue, gw, gh, heights)?;
                            wired.ext_terrain_hm = Some((h_hash, hm));
                        }
                        let (_, hm) = wired.ext_terrain_hm.as_ref().unwrap();
                        let wsx = ext.values[2].max(1.0);
                        let wsz = ext.values[3].max(1.0);
                        let ox = ext.values[4];
                        let oz = ext.values[5];
                        let hscale = ext.values[6].max(0.01);
                        let grid = (ext.values[7].round().clamp(2.0, 128.0) / 2.0).round() * 2.0;
                        let tu = TerrainUniform {
                            view_projection: active_vp,
                            camera_pos_and_fog: [cam_pos[0], cam_pos[1], cam_pos[2], frame.fog.density.clamp(0.0, 0.05)],
                            sun_dir_and_height_scale: [sun_dir[0], sun_dir[1], sun_dir[2], hscale],
                            sun_color_and_ambient: [sun_col[0], sun_col[1], sun_col[2], 0.35],
                            world_bounds: [wsx, wsz, wsx * 0.6, wsx * 1.6],
                            layer_heights: [hscale * 0.12, hscale * 0.55, hscale * 0.85, 0.2],
                            fog_color: [frame.sky.horizon.x, frame.sky.horizon.y, frame.sky.horizon.z, 0.45],
                        };
                        let patch = TerrainPatch {
                            origin_scale_morph: [ox, oz, wsx.max(wsz), 0.0],
                            uv_rect_and_grid: [0.0, 0.0, 1.0, grid.max(2.0)],
                        };
                        pipes.terrain.draw(device, queue, pass, &tu, &[patch], hm)?;
                    }
                }
                ENV_WATER_EXT_TAG if ext.values.len() >= 16 => {
                    let v = &ext.values;
                    let amp = v[15].clamp(0.0, 2.0);
                    let half_extent = if v.len() >= 17 && v[16].is_finite() && v[16] > 0.1 { v[16] } else { 12.0 };
                    let wu = WaterUniform {
                        view_projection: active_vp,
                        camera_pos_and_time: [cam_pos[0], cam_pos[1], cam_pos[2], v[4]],
                        sun_dir_and_level: [sun_dir[0], sun_dir[1], sun_dir[2], v[0]],
                        sun_color_and_bed: [sun_col[0], sun_col[1], sun_col[2], v[1]],
                        absorption_and_grid: [v[5].max(0.0), v[6].max(0.0), v[7].max(0.0), v[3].clamp(2.0, 128.0)],
                        deep_and_foam: [v[8].max(0.0), v[9].max(0.0), v[10].max(0.0), v[11].clamp(0.0, 1.0)],
                        sky_horizon: [frame.sky.horizon.x, frame.sky.horizon.y, frame.sky.horizon.z, half_extent],
                        sky_zenith: [frame.sky.zenith.x, frame.sky.zenith.y, frame.sky.zenith.z, 0.65],
                        bed_color_and_slope: [v[12].max(0.0), v[13].max(0.0), v[14].max(0.0), v[2]],
                        waves: [
                            [1.0, 0.2, 0.12 * amp, 5.0],
                            [0.4, 0.9, 0.08 * amp, 3.1],
                            [-0.6, 0.8, 0.05 * amp, 1.8],
                            [0.8, -0.5, 0.03 * amp, 1.1],
                        ],
                    };
                    pipes.water.draw(queue, pass, &wu)?;
                }
                ENV_FOLIAGE_EXT_TAG if ext.values.len() >= 26 => {
                    let v = &ext.values;
                    let count = (v.len() - 18) / 8;
                    if count > 0 {
                        let mut insts = Vec::with_capacity(count);
                        for i in 0..count {
                            let o = 18 + i * 8;
                            insts.push(FoliageInstance {
                                pos_and_scale: [v[o], v[o + 1], v[o + 2], v[o + 3]],
                                rot_phase_tint: [v[o + 4], v[o + 5], v[o + 6], v[o + 7]],
                            });
                        }
                        let fu = FoliageUniform {
                            view_projection: active_vp,
                            camera_pos_and_time: [cam_pos[0], cam_pos[1], cam_pos[2], v[0]],
                            sun_dir_and_wind_str: [sun_dir[0], sun_dir[1], sun_dir[2], v[3]],
                            sun_color_and_gust: [sun_col[0], sun_col[1], sun_col[2], v[5]],
                            wind_dir_and_speed: [v[1], 0.05, v[2], v[4]],
                            root_color_and_width: [v[8], v[9], v[10], v[6]],
                            tip_color_and_height: [v[11], v[12], v[13], v[7]],
                            sss_color_and_power: [v[14], v[15], v[16], v[17]],
                        };
                        pipes.foliage.draw(device, queue, pass, &fu, &insts)?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Runs `TlasRayTracer` (`quality.ray_tracing >= 1`) with SVGF spatiotemporal denoising and/or
    /// `DdgiVolume` (`quality.ddgi`) against the resolved 1x opaque HDR scene buffer and 1x R32Float depth.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_rt_and_ddgi_wired(
        wired: &mut WiredPasses,
        built_ins: &mut BuiltIns,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        frames_count: u64,
        active_proj: &Matrix,
        scene_view: &wgpu::TextureView,
        depth_only: &wgpu::TextureView,
        scene_size: (u32, u32),
        samples: u32,
        depth_mode: DepthMode,
    ) -> Result<()> {
        let wants_rt = frame.quality.ray_tracing > 0;
        let wants_ddgi = frame.quality.ddgi;
        if (!wants_rt && !wants_ddgi) || wired.tlas_gpu.is_none() {
            return Ok(());
        }
        let depth_clear = depth_mode.clear_value();
        let depth_r32f = wired.resolve_msaa_depth(device, built_ins, encoder, depth_only, scene_size, samples);
        let active_vp = multiply(active_proj, &frame.view);
        let inv_vp = invert(&active_vp);
        let eye = invert(&frame.view);
        let camera_pos = [eye[12], eye[13], eye[14]];
        let sun_raw = [-frame.light_direction.x, -frame.light_direction.y, -frame.light_direction.z];
        let sun_len = (sun_raw[0] * sun_raw[0] + sun_raw[1] * sun_raw[1] + sun_raw[2] * sun_raw[2]).sqrt();
        let sun_direction =
            if sun_len > 1e-6 { [sun_raw[0] / sun_len, sun_raw[1] / sun_len, sun_raw[2] / sun_len] } else { [0.0, 1.0, 0.0] };
        let sun_color = [frame.light_color.x.max(0.0), frame.light_color.y.max(0.0), frame.light_color.z.max(0.0)];
        let sky_zenith = [frame.sky.zenith.x.max(0.0), frame.sky.zenith.y.max(0.0), frame.sky.zenith.z.max(0.0)];
        let sky_horizon = [frame.sky.horizon.x.max(0.0), frame.sky.horizon.y.max(0.0), frame.sky.horizon.z.max(0.0)];

        if wants_rt {
            let tier = frame.quality.ray_tracing;
            let params = TlasRayTraceParams {
                view_projection: active_vp,
                camera_pos,
                depth_clear,
                sun_direction,
                sun_angular_radius: 0.015,
                sun_color,
                sky_zenith,
                sky_horizon,
                surface_roughness: if tier >= 2 { 0.12 } else { 0.45 },
                ao_radius: 2.0,
                max_ray_distance: 64.0,
                shadow_samples: 2,
                ao_samples: 4,
                reflection_samples: if tier >= 2 { 2 } else { 0 },
                gi_samples: if tier >= 2 { 2 } else { 0 },
                gi_bounce_scale: 1.0,
                frame_seed: (frames_count % 4096) as f32 * 0.618_034,
                normal_bias: 0.015,
                visibility_mask: u32::MAX,
            };
            let tlas = wired.tlas_gpu.as_ref().unwrap();
            let rt = wired.ray_tracer.get_or_insert_with(|| TlasRayTracer::new(device));
            rt.trace_with_scene(device, encoder, tlas, &depth_r32f, scene_size, &params)?;
            let raw_sao = rt.shadow_ao_view().cloned();
            let raw_refl = rt.reflection_view().cloned();
            let raw_gi = rt.gi_view().cloned();

            if let Some(sao_raw_view) = raw_sao {
                // Prepare linear depth, world normals and screen motion vectors, then run SVGF
                // spatiotemporal denoising on shadow/AO (and reflections + GI at tier 2).
                let svgf = wired.svgf_rt.get_or_insert_with(|| SvgfRtState::new(device));
                let resized = !matches!(&svgf.buffers, Some(b) if b.size == scene_size);
                if resized {
                    svgf.buffers = Some(SvgfRtBuffers {
                        size: scene_size,
                        curr_depth: SvgfTex::new(device, "svgf_rt_depth", scene_size),
                        prev_depth: SvgfTex::new(device, "svgf_rt_prev_depth", scene_size),
                        curr_normal: SvgfTex::new(device, "svgf_rt_normal", scene_size),
                        prev_normal: SvgfTex::new(device, "svgf_rt_prev_normal", scene_size),
                        motion: SvgfTex::new(device, "svgf_rt_motion", scene_size),
                    });
                    svgf.reset_history();
                }
                let has_prev = svgf.prev_vp.is_some() && !frame.exposure.cut;
                let prep_u = SvgfPrepUniform {
                    inv_vp,
                    prev_vp: svgf.prev_vp.unwrap_or(active_vp),
                    cam_and_clear: [camera_pos[0], camera_pos[1], camera_pos[2], depth_clear],
                    dims: [scene_size.0, scene_size.1, u32::from(has_prev), 0],
                };
                queue.write_buffer(&svgf.prep_u, 0, bytemuck::bytes_of(&prep_u));
                let bufs = svgf.buffers.as_ref().unwrap();
                let prep_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("svgf_rt_prep_bg"),
                    layout: &svgf.prep_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: svgf.prep_u.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&depth_r32f) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&bufs.curr_depth.view) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&bufs.curr_normal.view) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&bufs.motion.view) },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("svgf_rt_prep"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&svgf.prep_pipe);
                    pass.set_bind_group(0, &prep_bg, &[]);
                    pass.dispatch_workgroups(scene_size.0.div_ceil(8), scene_size.1.div_ceil(8), 1);
                }
                let sao_params = SvgfParams {
                    min_alpha: 0.1,
                    atrous_iterations: 2,
                    reset_history: !has_prev,
                    ..Default::default()
                };
                let sao_in = SvgfInputs {
                    noisy_signal: &sao_raw_view,
                    depth: &bufs.curr_depth.view,
                    prev_depth: &bufs.prev_depth.view,
                    normal: &bufs.curr_normal.view,
                    prev_normal: &bufs.prev_normal.view,
                    motion: &bufs.motion.view,
                };
                svgf.sao.denoise(device, encoder, &sao_in, scene_size, &sao_params)?;
                let mut denoised_refl = None;
                let mut denoised_gi = None;
                if tier >= 2
                    && let (Some(rv), Some(gv)) = (raw_refl.as_ref(), raw_gi.as_ref())
                {
                    let bounce_params = SvgfParams {
                        min_alpha: 0.1,
                        phi_color: 4.5,
                        atrous_iterations: 3,
                        reset_history: !has_prev,
                        ..Default::default()
                    };
                    let refl_in = SvgfInputs { noisy_signal: rv, ..sao_in };
                    svgf.refl.denoise(device, encoder, &refl_in, scene_size, &bounce_params)?;
                    let gi_in = SvgfInputs { noisy_signal: gv, ..sao_in };
                    svgf.gi.denoise(device, encoder, &gi_in, scene_size, &bounce_params)?;
                    denoised_refl = svgf.refl.output_view().cloned();
                    denoised_gi = svgf.gi.output_view().cloned();
                }
                let copy_ext = wgpu::Extent3d { width: scene_size.0, height: scene_size.1, depth_or_array_layers: 1 };
                encoder.copy_texture_to_texture(
                    bufs.curr_depth.tex.as_image_copy(),
                    bufs.prev_depth.tex.as_image_copy(),
                    copy_ext,
                );
                encoder.copy_texture_to_texture(
                    bufs.curr_normal.tex.as_image_copy(),
                    bufs.prev_normal.tex.as_image_copy(),
                    copy_ext,
                );
                svgf.prev_vp = Some(active_vp);
                let denoised_sao = svgf.sao.output_view().cloned().unwrap_or(sao_raw_view);
                wired.composite_ray_tracing(
                    device,
                    queue,
                    encoder,
                    scene_view,
                    &depth_r32f,
                    &denoised_sao,
                    denoised_refl.as_ref(),
                    denoised_gi.as_ref(),
                    tier,
                    depth_clear,
                );
            }
        }

        if wants_ddgi {
            let pv = frame.probe_volume;
            let desired_cfg = if pv.nx > 0 && pv.ny > 0 && pv.nz > 0 && pv.spacing > 0.0 {
                let dims = [pv.nx.clamp(2, 8), pv.ny.clamp(2, 4), pv.nz.clamp(2, 8)];
                let origin = [pv.origin.x, pv.origin.y, pv.origin.z];
                let sp = pv.spacing.max(0.25);
                DdgiGridConfig { origin, spacing: [sp, sp, sp], dims, rays_per_probe: 96, hysteresis: 0.85, ..Default::default() }
            } else if let Some((bmin, bmax)) = wired.tlas_bounds {
                let dims = [5, 3, 5];
                let pad = [
                    ((bmax[0] - bmin[0]) * 0.1).max(0.35),
                    ((bmax[1] - bmin[1]) * 0.1).max(0.35),
                    ((bmax[2] - bmin[2]) * 0.1).max(0.35),
                ];
                let lo = [bmin[0] - pad[0], bmin[1] - pad[1], bmin[2] - pad[2]];
                let hi = [bmax[0] + pad[0], bmax[1] + pad[1], bmax[2] + pad[2]];
                let spacing = [
                    ((hi[0] - lo[0]) / (dims[0] - 1) as f32).max(0.5),
                    ((hi[1] - lo[1]) / (dims[1] - 1) as f32).max(0.5),
                    ((hi[2] - lo[2]) / (dims[2] - 1) as f32).max(0.5),
                ];
                DdgiGridConfig { origin: lo, spacing, dims, rays_per_probe: 96, hysteresis: 0.85, ..Default::default() }
            } else {
                DdgiGridConfig { rays_per_probe: 96, hysteresis: 0.85, ..Default::default() }
            };
            let need_recreate = match &wired.ddgi {
                None => true,
                Some(existing) => {
                    let c = existing.config();
                    c.dims != desired_cfg.dims
                        || (0..3).any(|k| (c.origin[k] - desired_cfg.origin[k]).abs() > 1e-2)
                        || (0..3).any(|k| (c.spacing[k] - desired_cfg.spacing[k]).abs() > 1e-2)
                }
            };
            if need_recreate {
                wired.ddgi = Some(DdgiVolume::new(device, desired_cfg)?);
            }
            let ddgi_params = DdgiFrameParams {
                sun_direction,
                sun_color,
                sky_zenith,
                sky_horizon,
                visibility_mask: u32::MAX,
                reset_history: frame.exposure.cut,
            };
            let tlas = wired.tlas_gpu.as_ref().unwrap();
            wired.ddgi.as_mut().unwrap().update(device, encoder, tlas, &ddgi_params);
            wired.composite_ddgi(device, queue, encoder, scene_view, &depth_r32f, inv_vp, camera_pos, depth_clear);
        }

        Ok(())
    }
}

/// Extracts six normalized inward-facing frustum planes from a column-major WebGPU `[0, 1]`
/// view-projection matrix (valid for both Forward-Z and Reversed-Z projections).
fn frustum_planes_from_vp(vp: &Matrix) -> [[f32; 4]; 6] {
    let row = |r: usize| [vp[r], vp[4 + r], vp[8 + r], vp[12 + r]];
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);
    let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
    let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
    let norm = |p: [f32; 4]| {
        let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if l > 1e-7 { [p[0] / l, p[1] / l, p[2] / l, p[3] / l] } else { [0.0; 4] }
    };
    [
        norm(add(r3, r0)), // Left:   x + w >= 0
        norm(sub(r3, r0)), // Right:  w - x >= 0
        norm(add(r3, r1)), // Bottom: y + w >= 0
        norm(sub(r3, r1)), // Top:    w - y >= 0
        norm(r2),          // Near/Far z >= 0
        norm(sub(r3, r2)), // Far/Near w - z >= 0
    ]
}
