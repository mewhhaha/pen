use super::super::State;
use super::WiredPasses;
use super::ddgi::{
    has_active_point_spot_lights, pack_rt_local_lights, pack_rt_point_spot_lights, resolve_rt_sky_colors,
};
use crate::built_in::BuiltIns;
use crate::camera_precision::DepthMode;
use crate::matrix::{invert, multiply};
use crate::post::SCENE_FORMAT;
use crate::ray_tracing::{RtLocalLight, TlasRayTraceParams, TlasRayTracer};
use crate::restir::{RESTIR_MAX_LIGHTS, RestirInputs, RestirParams, RestirRenderer};
use crate::restir_specular::{RestirSpecularInputs, RestirSpecularParams, RestirSpecularRenderer};
use crate::sharc::{SharcCache, SharcConfig, SharcParams};
use crate::svgf::{SvgfDenoiser, SvgfInputs, SvgfParams};
use anyhow::Result;
use pen_protocol::{Matrix, RenderFrame};

const RT_COMBINED_WGSL: &str = r#"
struct RtCompUniform {
  depth_clear: f32,
  refl_weight: f32,
  gi_weight: f32,
  shadow_floor: f32,
  gi_demodulated: f32,
  di_weight: f32,
  _pad1: f32,
  _pad2: f32,
};
@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var refl_tex: texture_2d<f32>;
@group(0) @binding(2) var gi_tex: texture_2d<f32>;
@group(0) @binding(3) var sao_tex: texture_2d<f32>;
@group(0) @binding(4) var<uniform> cfg: RtCompUniform;
@group(0) @binding(5) var gbuf_albedo_metallic: texture_2d<f32>;
@group(0) @binding(6) var di_tex: texture_2d<f32>;

@vertex fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
  let y = 1.0 - f32(idx & 2u) * 2.0;
  return vec4f(x, y, 0.0, 1.0);
}

@fragment fn fs_combined(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let dims = vec2i(textureDimensions(depth_tex, 0));
  let p = clamp(vec2i(pos.xy), vec2i(0), dims - vec2i(1));
  let d = textureLoad(depth_tex, p, 0).r;
  if (abs(d - cfg.depth_clear) <= 0.00001) {
    discard;
  }
  let rt_dims = vec2i(textureDimensions(sao_tex, 0));
  var sao = vec4f(1.0, 1.0, 0.0, 1.0);
  var refl = vec3f(0.0);
  var gi = vec3f(0.0);
  var di = vec3f(0.0);
  if (rt_dims.x == dims.x && rt_dims.y == dims.y) {
    sao = textureLoad(sao_tex, p, 0);
    if (cfg.refl_weight > 0.0) {
      refl = max(textureLoad(refl_tex, p, 0).rgb, vec3f(0.0));
    }
    if (cfg.gi_weight > 0.0) {
      gi = max(textureLoad(gi_tex, p, 0).rgb, vec3f(0.0));
    }
    if (cfg.di_weight > 0.0) {
      di = max(textureLoad(di_tex, p, 0).rgb, vec3f(0.0));
    }
  } else {
    let f_coord = (vec2f(p) + vec2f(0.5)) * vec2f(rt_dims) / vec2f(dims) - vec2f(0.5);
    let base_h = vec2i(floor(f_coord));
    let frac = clamp(f_coord - vec2f(base_h), vec2f(0.0), vec2f(1.0));
    var s_sum = vec4f(0.0);
    var r_sum = vec3f(0.0);
    var g_sum = vec3f(0.0);
    var d_sum = vec3f(0.0);
    var w_sum = 0.0;
    for (var oy = 0; oy < 2; oy++) {
      for (var ox = 0; ox < 2; ox++) {
        let hp = clamp(base_h + vec2i(ox, oy), vec2i(0), rt_dims - vec2i(1));
        let fp = clamp(vec2i((vec2f(hp) + vec2f(0.5)) * vec2f(dims) / vec2f(rt_dims)), vec2i(0), dims - vec2i(1));
        let dh = textureLoad(depth_tex, fp, 0).r;
        let bw = select(1.0 - frac.x, frac.x, ox == 1) * select(1.0 - frac.y, frac.y, oy == 1);
        let dz = abs(d - dh);
        let dw = 1.0 / (1.0 + dz * 400.0);
        let w = bw * dw + 1e-5;
        s_sum += textureLoad(sao_tex, hp, 0) * w;
        if (cfg.refl_weight > 0.0) {
          r_sum += max(textureLoad(refl_tex, hp, 0).rgb, vec3f(0.0)) * w;
        }
        if (cfg.gi_weight > 0.0) {
          g_sum += max(textureLoad(gi_tex, hp, 0).rgb, vec3f(0.0)) * w;
        }
        if (cfg.di_weight > 0.0) {
          d_sum += max(textureLoad(di_tex, hp, 0).rgb, vec3f(0.0)) * w;
        }
        w_sum += w;
      }
    }
    let inv_w = 1.0 / max(w_sum, 1e-5);
    sao = s_sum * inv_w;
    refl = r_sum * inv_w;
    gi = g_sum * inv_w;
    di = d_sum * inv_w;
  }
  let shadow = mix(cfg.shadow_floor, 1.0, clamp(sao.r, 0.0, 1.0));
  let ao = mix(0.58, 1.0, clamp(sao.g, 0.0, 1.0));
  let atten = shadow * ao;
  let occlude = clamp(clamp(sao.b, 0.0, 0.75) * cfg.refl_weight, 0.0, 0.75);
  if (cfg.gi_weight > 0.0 && cfg.gi_demodulated > 0.5) {
    let g_dims = vec2i(textureDimensions(gbuf_albedo_metallic, 0));
    let g_px = clamp(vec2i((vec2f(p) + vec2f(0.5)) * vec2f(g_dims) / vec2f(dims)), vec2i(0), g_dims - vec2i(1));
    let g_am = textureLoad(gbuf_albedo_metallic, g_px, 0);
    gi *= max(g_am.rgb, vec3f(0.0)) * (1.0 - clamp(g_am.w, 0.0, 1.0));
  }
  let bounce = refl * cfg.refl_weight + gi * cfg.gi_weight + di * cfg.di_weight;
  // Single-pass blend (`One + OneMinusSrcAlpha`) combines shadow/AO attenuation and bounce:
  // dst' = bounce + dst * (atten * (1 - occlude)) = bounce + dst * (1 - alpha).
  return vec4f(bounce, 1.0 - atten * (1.0 - occlude));
}
"#;

const SVGF_PREP_WGSL: &str = r#"
struct SvgfPrepUniform {
  inv_vp: mat4x4f,
  prev_vp: mat4x4f,
  cam_and_clear: vec4f, // xyz = camera world pos, w = depth_clear (1.0 ForwardZ, 0.0 ReversedZ)
  dims: vec4u,          // x = out_width, y = out_height, z = has_prev_vp (0 or 1), w = flags (bit0: velocity, bit1: thin_gbuf)
};

@group(0) @binding(0) var<uniform> u: SvgfPrepUniform;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var out_depth: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var out_normal: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var out_motion: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var velocity_tex: texture_2d<f32>;
@group(0) @binding(6) var rt_sao_tex: texture_2d<f32>;
@group(0) @binding(7) var rt_refl_tex: texture_2d<f32>;
@group(0) @binding(8) var gbuf_normal_roughness: texture_2d<f32>;

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
  let out_dims = vec2i(i32(u.dims.x), i32(u.dims.y));
  let depth_dims = vec2i(textureDimensions(depth_tex, 0));
  let px = vec2i(id.xy);
  if (px.x >= out_dims.x || px.y >= out_dims.y) {
    return;
  }
  let d_px = select(
    clamp(vec2i((vec2f(px) + vec2f(0.5)) * vec2f(depth_dims) / vec2f(out_dims)), vec2i(0), depth_dims - vec2i(1)),
    px,
    out_dims.x == depth_dims.x && out_dims.y == depth_dims.y,
  );
  let c = load_world(d_px, depth_dims);
  if (!c.valid) {
    textureStore(out_depth, px, vec4f(10000.0, 0.0, 0.0, 1.0));
    textureStore(out_normal, px, vec4f(0.0, 1.0, 0.0, 1.0));
    textureStore(out_motion, px, vec4f(0.0, 0.0, 0.0, 0.0));
    return;
  }
  let to_cam = u.cam_and_clear.xyz - c.pos;
  let dist = length(to_cam);
  let view_dir = select(vec3f(0.0, 1.0, 0.0), to_cam / dist, dist > 1e-6);
  var n = vec3f(0.0, 1.0, 0.0);
  var mat_rough = clamp(textureLoad(rt_sao_tex, px, 0).a, 0.04, 1.0);
  var has_gbuf_n = false;
  if ((u.dims.w & 2u) != 0u) {
    let g_dims = vec2i(textureDimensions(gbuf_normal_roughness, 0));
    let g_px = clamp(vec2i((vec2f(px) + vec2f(0.5)) * vec2f(g_dims) / vec2f(out_dims)), vec2i(0), g_dims - vec2i(1));
    let g_nr = textureLoad(gbuf_normal_roughness, g_px, 0);
    if (dot(g_nr.xyz, g_nr.xyz) >= 0.01) {
      n = normalize(g_nr.xyz);
      mat_rough = clamp(g_nr.w, 0.04, 1.0);
      has_gbuf_n = true;
    }
  }
  if (!has_gbuf_n) {
    let sr = load_world(d_px + vec2i(1, 0), depth_dims);
    let sl = load_world(d_px - vec2i(1, 0), depth_dims);
    let sd = load_world(d_px + vec2i(0, 1), depth_dims);
    let su = load_world(d_px - vec2i(0, 1), depth_dims);
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
    if (has_dx && has_dy) {
      let raw_n = cross(dx, dy);
      let area2 = dot(raw_n, raw_n);
      let ref2 = dot(dx, dx) * dot(dy, dy);
      if (area2 > 1e-12 * ref2 && ref2 > 0.0) {
        n = raw_n * inverseSqrt(area2);
      }
    }
  }
  if (dot(n, view_dir) < 0.0) {
    n = -n;
  }
  var mv = vec2f(0.0);
  var used_obj_vel = false;
  if ((u.dims.w & 1u) != 0u) {
    let v_dims = vec2i(textureDimensions(velocity_tex, 0));
    let v_px = clamp(vec2i((vec2f(px) + vec2f(0.5)) * vec2f(v_dims) / vec2f(out_dims)), vec2i(0), v_dims - vec2i(1));
    let obj_v = textureLoad(velocity_tex, v_px, 0).xy;
    if (abs(obj_v.x) < 100.0 && abs(obj_v.y) < 100.0) {
      mv = obj_v;
      used_obj_vel = true;
    }
  }
  if (!used_obj_vel && u.dims.z != 0u) {
    let curr_uv = (vec2f(d_px) + vec2f(0.5)) / vec2f(f32(depth_dims.x), f32(depth_dims.y));
    let pclip = u.prev_vp * vec4f(c.pos, 1.0);
    if (abs(pclip.w) > 1e-6) {
      let pndc = pclip.xy / pclip.w;
      let prev_uv = vec2f(pndc.x * 0.5 + 0.5, 0.5 - pndc.y * 0.5);
      mv = curr_uv - prev_uv;
    }
  }
  let hit_dist = max(textureLoad(rt_refl_tex, px, 0).a, 0.0);
  // Specular virtual hit-point parallax reprojection: virtual reflected image sits at (cam_dist + hit_dist)
  // along the incoming view ray; blend toward surface motion as surface roughness increases.
  var spec_mv = mv;
  if (!used_obj_vel && u.dims.z != 0u && hit_dist > 1e-3 && hit_dist < 50.0) {
    let curr_uv = (vec2f(d_px) + vec2f(0.5)) / vec2f(f32(depth_dims.x), f32(depth_dims.y));
    let virt_pos = u.cam_and_clear.xyz - view_dir * (dist + hit_dist);
    let vclip = u.prev_vp * vec4f(virt_pos, 1.0);
    if (abs(vclip.w) > 1e-6) {
      let vndc = vclip.xy / vclip.w;
      let prev_vuv = vec2f(vndc.x * 0.5 + 0.5, 0.5 - vndc.y * 0.5);
      let virt_mv = curr_uv - prev_vuv;
      let mirror_w = clamp((1.0 - mat_rough) * (1.0 - mat_rough), 0.0, 1.0);
      spec_mv = mix(mv, virt_mv, mirror_w);
    }
  }
  textureStore(out_depth, px, vec4f(min(dist, 65000.0), 0.0, 0.0, 1.0));
  textureStore(out_normal, px, vec4f(n, mat_rough));
  textureStore(out_motion, px, vec4f(mv, spec_mv));
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RtCompUniform {
    depth_clear: f32,
    refl_weight: f32,
    gi_weight: f32,
    shadow_floor: f32,
    gi_demodulated: f32,
    di_weight: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SvgfPrepUniform {
    inv_vp: [f32; 16],
    prev_vp: [f32; 16],
    cam_and_clear: [f32; 4],
    dims: [u32; 4],
}

pub(super) struct RtCompositePipelines {
    combined_bgl: wgpu::BindGroupLayout,
    combined_pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    combined_bg: Option<([wgpu::TextureView; 6], wgpu::BindGroup)>,
}

struct SvgfTex {
    _tex: wgpu::Texture,
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
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        Self { _tex: tex, view }
    }
}

struct SvgfRtBuffers {
    size: (u32, u32),
    depth: [SvgfTex; 2],
    normal: [SvgfTex; 2],
    motion: SvgfTex,
    parity: usize,
    prep_bgs: [Option<([wgpu::TextureView; 5], wgpu::BindGroup)>; 2],
}

pub(super) struct SvgfRtState {
    prep_pipe: wgpu::ComputePipeline,
    prep_bgl: wgpu::BindGroupLayout,
    prep_u: wgpu::Buffer,
    sao: SvgfDenoiser,
    refl: SvgfDenoiser,
    gi: SvgfDenoiser,
    di: SvgfDenoiser,
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
                ro_tex(5),
                ro_tex(6),
                ro_tex(7),
                ro_tex(8),
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
            di: SvgfDenoiser::new(device),
            buffers: None,
            prev_vp: None,
        }
    }

    pub(super) fn reset_history(&mut self) {
        self.sao.reset_history();
        self.refl.reset_history();
        self.gi.reset_history();
        self.di.reset_history();
        self.prev_vp = None;
    }
}

impl WiredPasses {
    fn ensure_rt_composite(&mut self, device: &wgpu::Device) -> &mut RtCompositePipelines {
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
            let combined_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rt_combined_comp_bgl"),
                entries: &[
                    tex_entry(0),
                    tex_entry(1),
                    tex_entry(2),
                    tex_entry(3),
                    u_entry(4),
                    tex_entry(5),
                    tex_entry(6),
                ],
            });
            let combined_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rt_combined_comp_pl"),
                bind_group_layouts: &[Some(&combined_bgl)],
                immediate_size: 0,
            });
            let combined_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rt_combined_comp"),
                source: wgpu::ShaderSource::Wgsl(RT_COMBINED_WGSL.into()),
            });
            let combined_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("rt_combined_comp"),
                layout: Some(&combined_pl),
                vertex: wgpu::VertexState {
                    module: &combined_mod,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &combined_mod,
                    entry_point: Some("fs_combined"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: SCENE_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
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
            RtCompositePipelines { combined_bgl, combined_pipeline, uniform, combined_bg: None }
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
        di_view: Option<&wgpu::TextureView>,
        gbuf_albedo_metallic: Option<&wgpu::TextureView>,
        gi_demodulated: bool,
        tier: u32,
        depth_clear: f32,
        ddgi_active: bool,
        has_csm: bool,
    ) {
        let dummy_gbuf = self.dummy_rgba16f_view(device);
        let gbuf_am_bind = gbuf_albedo_metallic.unwrap_or(&dummy_gbuf);
        let has_bounce = tier >= 2 && refl_view.is_some();
        let rv = if has_bounce { refl_view.unwrap() } else { &dummy_gbuf };
        let gv = if has_bounce { gi_view.unwrap_or(rv) } else { &dummy_gbuf };
        let di_bind = if has_bounce { di_view.unwrap_or(&dummy_gbuf) } else { &dummy_gbuf };
        let comp = self.ensure_rt_composite(device);
        let cfg = RtCompUniform {
            depth_clear,
            refl_weight: if has_bounce { 1.0 } else { 0.0 },
            gi_weight: if has_bounce && !ddgi_active { 0.85 } else { 0.0 },
            shadow_floor: if has_csm { 0.84 } else { 0.25 },
            gi_demodulated: if has_bounce && gi_demodulated && gbuf_albedo_metallic.is_some() { 1.0 } else { 0.0 },
            // Pen's forward PBR (`surface_lobes.wgsl`) scales direct lights by PI relative to physical 1/PI BRDFs.
            di_weight: if has_bounce && di_view.is_some() { std::f32::consts::PI } else { 0.0 },
            _pad: [0.0; 2],
        };
        queue.write_buffer(&comp.uniform, 0, bytemuck::bytes_of(&cfg));
        let key_views = [depth_r32f, rv, gv, sao_view, gbuf_am_bind, di_bind];
        let hit = matches!(
            &comp.combined_bg,
            Some((cached, _)) if cached.iter().zip(key_views.iter()).all(|(a, b)| a == *b)
        );
        if !hit {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rt_combined_comp_bg"),
                layout: &comp.combined_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_r32f) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(rv) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(gv) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(sao_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: comp.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(gbuf_am_bind) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(di_bind) },
                ],
            });
            comp.combined_bg = Some((key_views.map(Clone::clone), bg));
        }
        let (_, combined_bg) = comp.combined_bg.as_ref().unwrap();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("rt_combined_composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: scene_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&comp.combined_pipeline);
        pass.set_bind_group(0, combined_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn rt_grid_size(size: (u32, u32), binding_bytes: u64) -> (u32, u32) {
    let max_pixels = (binding_bytes / crate::restir_specular::RESERVOIR_STRIDE).max(1);
    let mut divisor = if size.0 > 128 || size.1 > 128 { 2 } else { 1 };
    loop {
        let grid = (size.0.div_ceil(divisor), size.1.div_ceil(divisor));
        if u64::from(grid.0) * u64::from(grid.1) <= max_pixels {
            return grid;
        }
        divisor += 1;
    }
}

impl State {
    /// Runs half-resolution `TlasRayTracer` (`quality.ray_tracing >= 1`) fed directly by the
    /// rasterized Thin G-Buffer (zero primary TLAS material-lookup rays, exact skinned/instanced/sheared
    /// normals & per-instance material overrides), optional Tier-3 `RestirRenderer` ReSTIR GI
    /// (spatiotemporal Jacobian Pairwise MIS + thin-wall visibility validation + demodulated SVGF
    /// denoising re-modulated at full resolution), live DDGI multi-bounce at secondary hits, and
    /// depth-aware bilateral upsample onto the resolved 1x opaque HDR scene buffer.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_rt_wired(
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
        velocity_view: Option<&wgpu::TextureView>,
        thin_gbuffer: Option<(&wgpu::TextureView, &wgpu::TextureView, &wgpu::TextureView)>,
        ddgi_packed_buf: Option<&wgpu::Buffer>,
        scene_size: (u32, u32),
        samples: u32,
        depth_mode: DepthMode,
    ) -> Result<()> {
        if frame.quality.ray_tracing == 0 || !TlasRayTracer::supported(device) || wired.tlas_gpu.is_none() {
            return Ok(());
        }
        let ddgi_active = ddgi_packed_buf.is_some();
        let depth_clear = depth_mode.clear_value();
        let depth_r32f = wired.resolve_msaa_depth(device, built_ins, encoder, depth_only, scene_size, samples);
        let limits = device.limits();
        let rt_size = rt_grid_size(scene_size, limits.max_storage_buffer_binding_size.min(limits.max_buffer_size));
        let active_vp = multiply(active_proj, &frame.view);
        let inv_vp = invert(&active_vp);
        let eye = invert(&frame.view);
        let camera_pos = [eye[12], eye[13], eye[14]];
        let sun_raw = [-frame.light_direction.x, -frame.light_direction.y, -frame.light_direction.z];
        let sun_len = (sun_raw[0] * sun_raw[0] + sun_raw[1] * sun_raw[1] + sun_raw[2] * sun_raw[2]).sqrt();
        let sun_direction = if sun_len > 1e-6 {
            [sun_raw[0] / sun_len, sun_raw[1] / sun_len, sun_raw[2] / sun_len]
        } else {
            [0.0, 1.0, 0.0]
        };
        let sun_color = [frame.light_color.x.max(0.0), frame.light_color.y.max(0.0), frame.light_color.z.max(0.0)];
        let (sky_zenith, sky_horizon) = resolve_rt_sky_colors(frame);
        let local_lights = pack_rt_local_lights(frame);

        let tier = frame.quality.ray_tracing;
        let use_restir_di = tier >= 3 && thin_gbuffer.is_some() && has_active_point_spot_lights(frame);
        let use_restir_gi = tier >= 3 && !ddgi_active && thin_gbuffer.is_some();
        let use_restir_specular = tier >= 3 && thin_gbuffer.is_some();
        let use_sharc = use_restir_gi;
        if use_sharc {
            if wired.sharc.is_none() {
                wired.sharc = Some(SharcCache::new(device, SharcConfig::default())?);
                wired.sharc_needs_clear = true;
            }
            if let Some(sharc) = wired.sharc.as_ref()
                && (wired.sharc_frame != Some(frames_count) || wired.sharc_needs_clear)
            {
                let sharc_reset = wired.sharc_needs_clear;
                let sharc_params = SharcParams {
                    camera_pos,
                    frame_index: u32::try_from(frames_count & 0xffff_ffff).unwrap_or(0),
                    reset: sharc_reset,
                    enabled: true,
                };
                sharc.update_uniform(queue, &sharc_params);
                if sharc_reset {
                    sharc.clear(encoder);
                    wired.sharc_needs_clear = false;
                } else {
                    sharc.resolve_uploaded(encoder, true);
                }
                wired.sharc_frame = Some(frames_count);
            }
        }
        let sharc_ref = if use_sharc { wired.sharc.as_ref() } else { None };

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
            shadow_samples: 1,
            ao_samples: 4,
            reflection_samples: if tier >= 2 && !use_restir_specular { 2 } else { 0 },
            gi_samples: if tier >= 2 && !ddgi_active && !use_restir_gi { 1 } else { 0 },
            gi_bounce_scale: 1.0,
            frame_seed: (frames_count % 4096) as f32 * 0.618_034,
            normal_bias: 0.015,
            visibility_mask: u32::MAX,
        };
        let tlas = wired.tlas_gpu.as_ref().unwrap();
        let rt = wired.ray_tracer.get_or_insert_with(|| TlasRayTracer::new(device));
        rt.trace_with_scene_gbuffer_sharc(
            device,
            encoder,
            tlas,
            &depth_r32f,
            rt_size,
            &params,
            &local_lights,
            ddgi_packed_buf,
            true,
            thin_gbuffer.map(|(nr, am, _)| (nr, am)),
            sharc_ref,
        )?;
        let raw_sao = rt.shadow_ao_view().cloned();
        let mut raw_refl = rt.reflection_view().cloned();
        let mut raw_refl_motion: Option<wgpu::TextureView> = None;
        let mut raw_gi = rt.gi_view().cloned();
        let mut raw_di: Option<wgpu::TextureView> = None;
        let mut gi_demodulated = false;

        if (use_restir_di || use_restir_gi || use_restir_specular)
            && let Some((gbuf_nr, gbuf_am, gbuf_depth)) = thin_gbuffer
        {
            let restir_has_prev =
                wired.rt_history.svgf_rt.as_ref().is_some_and(|s| s.prev_vp.is_some()) && !frame.exposure.cut;
            let restir_prev_vp = wired.rt_history.svgf_rt.as_ref().and_then(|s| s.prev_vp).unwrap_or(active_vp);
            let frame_idx = u32::try_from(frames_count & 0xffff_ffff).unwrap_or(0);
            if use_restir_di || use_restir_gi {
                let restir_params = RestirParams {
                    view_projection: active_vp,
                    prev_view_projection: restir_prev_vp,
                    camera_pos,
                    depth_clear,
                    sun_direction,
                    sun_color,
                    sky_zenith,
                    sky_horizon,
                    max_ray_distance: 64.0,
                    normal_bias: 0.015,
                    visibility_mask: u32::MAX,
                    frame_index: frame_idx,
                    enable_di: use_restir_di,
                    enable_gi: use_restir_gi,
                    reset_history: !restir_has_prev,
                    gi_bounce_scale: 1.0,
                };
                // When ReSTIR DI is active, supply ONLY point/spot lights so LTC area lights (still shaded
                // analytically on opaque pixels via `Clusters::area_only_group()`) are never double-counted.
                let point_spot_lights;
                let restir_lights: &[RtLocalLight] = if use_restir_di {
                    point_spot_lights = pack_rt_point_spot_lights(frame);
                    &point_spot_lights
                } else {
                    &local_lights[..local_lights.len().min(RESTIR_MAX_LIGHTS)]
                };
                let restir_inputs = RestirInputs {
                    depth: gbuf_depth,
                    normal_roughness: gbuf_nr,
                    albedo_metallic: gbuf_am,
                    motion: velocity_view,
                    ddgi_packed: ddgi_packed_buf,
                    sharc: sharc_ref,
                };
                let restir = wired.rt_history.restir.get_or_insert_with(|| RestirRenderer::new(device));
                restir.dispatch(device, encoder, tlas, restir_inputs, rt_size, &restir_params, restir_lights)?;
                if use_restir_di {
                    raw_di = restir.di_view().cloned();
                }
                if use_restir_gi && let Some(gv) = restir.gi_view().cloned() {
                    raw_gi = Some(gv);
                    gi_demodulated = true;
                }
            }
            if use_restir_specular {
                // Zero sky on additive composite so forward sky IBL is never double-counted while
                // local glossy reflections, sun/local-light specular hits, and SHaRC/DDGI bounce shine.
                let spec_params = RestirSpecularParams {
                    view_projection: active_vp,
                    prev_view_projection: restir_prev_vp,
                    camera_pos,
                    depth_clear,
                    sun_direction,
                    sun_color,
                    sky_zenith: [0.0; 3],
                    sky_horizon: [0.0; 3],
                    max_ray_distance: 64.0,
                    normal_bias: 0.015,
                    visibility_mask: u32::MAX,
                    frame_index: frame_idx,
                    max_roughness: 0.72,
                    reset_history: !restir_has_prev,
                    gi_bounce_scale: 1.0,
                };
                let spec_inputs = RestirSpecularInputs {
                    depth: gbuf_depth,
                    normal_roughness: gbuf_nr,
                    albedo_metallic: gbuf_am,
                    motion: velocity_view,
                    ddgi_buf: ddgi_packed_buf,
                    sharc: sharc_ref,
                };
                let restir_spec =
                    wired.rt_history.restir_specular.get_or_insert_with(|| RestirSpecularRenderer::new(device));
                let spec_lights = &local_lights[..local_lights.len().min(RESTIR_MAX_LIGHTS)];
                let out =
                    restir_spec.dispatch(device, encoder, tlas, spec_inputs, rt_size, &spec_params, spec_lights)?;
                raw_refl = Some(out.specular_view.clone());
                raw_refl_motion = Some(out.motion_view.clone());
            }
        }

        if let Some(sao_raw_view) = raw_sao {
            let dummy_vel = wired.dummy_velocity_view(device);
            let dummy_gbuf = wired.dummy_rgba16f_view(device);
            let vel_bind = velocity_view.unwrap_or(&dummy_vel);
            let gbuf_nr_bind = thin_gbuffer.map_or(&dummy_gbuf, |(nr, _, _)| nr);
            let svgf = wired.rt_history.svgf_rt.get_or_insert_with(|| Box::new(SvgfRtState::new(device)));
            let resized = !matches!(&svgf.buffers, Some(b) if b.size == rt_size);
            if resized {
                svgf.buffers = Some(SvgfRtBuffers {
                    size: rt_size,
                    depth: [
                        SvgfTex::new(device, "svgf_rt_depth_0", rt_size),
                        SvgfTex::new(device, "svgf_rt_depth_1", rt_size),
                    ],
                    normal: [
                        SvgfTex::new(device, "svgf_rt_normal_0", rt_size),
                        SvgfTex::new(device, "svgf_rt_normal_1", rt_size),
                    ],
                    motion: SvgfTex::new(device, "svgf_rt_motion", rt_size),
                    parity: 0,
                    prep_bgs: [None, None],
                });
                svgf.reset_history();
            }
            let has_prev = svgf.prev_vp.is_some() && !frame.exposure.cut;
            let prep_flags = u32::from(velocity_view.is_some()) | (u32::from(thin_gbuffer.is_some()) << 1);
            let prep_u = SvgfPrepUniform {
                inv_vp,
                prev_vp: svgf.prev_vp.unwrap_or(active_vp),
                cam_and_clear: [camera_pos[0], camera_pos[1], camera_pos[2], depth_clear],
                dims: [rt_size.0, rt_size.1, u32::from(has_prev), prep_flags],
            };
            queue.write_buffer(&svgf.prep_u, 0, bytemuck::bytes_of(&prep_u));
            let bufs = svgf.buffers.as_mut().unwrap();
            let curr = bufs.parity;
            let prev = 1 - curr;
            let refl_bind = raw_refl.as_ref().unwrap_or(&sao_raw_view);
            let prep_views = [&depth_r32f, vel_bind, &sao_raw_view, refl_bind, gbuf_nr_bind];
            let prep_hit = matches!(
                &bufs.prep_bgs[curr],
                Some((cached, _)) if cached.iter().zip(prep_views.iter()).all(|(a, b)| a == *b)
            );
            if !prep_hit {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("svgf_rt_prep_bg"),
                    layout: &svgf.prep_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: svgf.prep_u.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&depth_r32f) },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&bufs.depth[curr].view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&bufs.normal[curr].view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&bufs.motion.view),
                        },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(vel_bind) },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&sao_raw_view),
                        },
                        wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(refl_bind) },
                        wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(gbuf_nr_bind) },
                    ],
                });
                bufs.prep_bgs[curr] = Some((prep_views.map(Clone::clone), bg));
            }
            let (_, prep_bg) = bufs.prep_bgs[curr].as_ref().unwrap();
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("svgf_rt_prep"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&svgf.prep_pipe);
                pass.set_bind_group(0, prep_bg, &[]);
                pass.dispatch_workgroups(rt_size.0.div_ceil(8), rt_size.1.div_ceil(8), 1);
            }
            let sao_params =
                SvgfParams { min_alpha: 0.1, atrous_iterations: 3, reset_history: !has_prev, ..Default::default() };
            let sao_in = SvgfInputs {
                noisy_signal: &sao_raw_view,
                depth: &bufs.depth[curr].view,
                prev_depth: &bufs.depth[prev].view,
                normal: &bufs.normal[curr].view,
                prev_normal: &bufs.normal[prev].view,
                motion: &bufs.motion.view,
            };
            svgf.sao.denoise(device, encoder, &sao_in, rt_size, &sao_params)?;
            let mut denoised_refl = None;
            let mut denoised_gi = None;
            let mut denoised_di = None;
            if tier >= 2 {
                let refl_params = SvgfParams {
                    min_alpha: 0.12,
                    phi_color: 4.0,
                    atrous_iterations: 3,
                    reset_history: !has_prev,
                    specular_mode: true,
                    hit_distance_adaptive: use_restir_specular,
                    ..Default::default()
                };
                let gi_params = SvgfParams {
                    min_alpha: 0.1,
                    phi_color: 4.5,
                    atrous_iterations: 3,
                    reset_history: !has_prev,
                    specular_mode: false,
                    ..Default::default()
                };
                if let Some(rv) = raw_refl.as_ref() {
                    let refl_motion = raw_refl_motion.as_ref().unwrap_or(sao_in.motion);
                    let refl_in = SvgfInputs { noisy_signal: rv, motion: refl_motion, ..sao_in };
                    svgf.refl.denoise(device, encoder, &refl_in, rt_size, &refl_params)?;
                    denoised_refl = svgf.refl.output_view().cloned();
                }
                if !ddgi_active && let Some(gv) = raw_gi.as_ref() {
                    let gi_in = SvgfInputs { noisy_signal: gv, ..sao_in };
                    svgf.gi.denoise(device, encoder, &gi_in, rt_size, &gi_params)?;
                    denoised_gi = svgf.gi.output_view().cloned();
                }
                if let Some(dv) = raw_di.as_ref() {
                    let di_params = SvgfParams {
                        min_alpha: 0.12,
                        phi_color: 4.0,
                        atrous_iterations: 2,
                        reset_history: !has_prev,
                        specular_mode: false,
                        ..Default::default()
                    };
                    let di_in = SvgfInputs { noisy_signal: dv, ..sao_in };
                    svgf.di.denoise(device, encoder, &di_in, rt_size, &di_params)?;
                    denoised_di = svgf.di.output_view().cloned();
                }
            }
            bufs.parity = prev;
            svgf.prev_vp = Some(active_vp);
            let denoised_sao = svgf.sao.output_view().cloned().unwrap_or(sao_raw_view);
            let has_csm = !frame.cascades.is_empty() && frame.shadow.texels > 0;
            wired.composite_ray_tracing(
                device,
                queue,
                encoder,
                scene_view,
                &depth_r32f,
                &denoised_sao,
                denoised_refl.as_ref(),
                denoised_gi.as_ref(),
                denoised_di.as_ref(),
                thin_gbuffer.map(|(_, am, _)| am),
                gi_demodulated,
                tier,
                depth_clear,
                ddgi_active,
                has_csm,
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::rt_grid_size;

    #[test]
    fn restir_grid_fits_4k_and_larger_devices() {
        let limit = 128 * 1024 * 1024;
        assert_eq!(rt_grid_size((3840, 2160), limit), (1920, 1080));
        assert_eq!(rt_grid_size((4096, 2160), limit), (1366, 720));
        let (width, height) = rt_grid_size((7680, 4320), 16 * 1024 * 1024);
        assert!(u64::from(width) * u64::from(height) * 64 <= 16 * 1024 * 1024);
    }
}
