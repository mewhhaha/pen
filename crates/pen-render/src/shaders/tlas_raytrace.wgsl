// 2-Level TLAS/BLAS GPU Ray Tracer (native hardware `wgpu_ray_query` or portable compute BVH):
// Supports full-res (1:1) and half-res (2:1) tracing, primary TLAS material lookup (smooth normal,
// albedo, roughness, metallic + Schlick Fresnel), dynamic local lights, and live DDGI multi-bounce
// shading inside secondary reflection/GI hits.
//
// Outputs:
//  - out_shadow_ao  (Rgba16Float): (rt_shadow, rtao, hit_dist, 1.0)
//  - out_reflection (Rgba16Float): (fresnel_ggx_refl.rgb, hit_dist)
//  - out_gi         (Rgba16Float): (one_bounce_diffuse_gi.rgb, 1.0)
//
// Prefixed at compile time with tlas_traversal.wgsl or tlas_traversal_hw.wgsl.

struct TlasRayTraceUniform {
  inverse_view_projection: mat4x4f,
  camera_pos_and_clear: vec4f, // xyz = camera pos, w = depth clear (1.0 or 0.0)
  sun_dir_and_radius: vec4f,   // xyz = normalized toward sun, w = sun angular radius
  sun_color_and_rough: vec4f,  // rgb = sun radiance, a = default surface roughness
  sky_zenith: vec4f,           // rgb = sky zenith radiance, a = ao_radius
  sky_horizon: vec4f,          // rgb = sky horizon radiance, a = max_ray_dist
  samples_seed: vec4f,         // x = shadow_samples, y = ao_samples, z = gi_samples, w = frame_seed
  bias_and_refl: vec4f,        // x = normal_bias, y = refl_samples, z = gi_bounce_scale, w = 0
  counts: vec4u,               // x = tlas_nodes, y = tlas_instances, z = blas_tris, w = visibility_mask
  extra_counts: vec4u,         // x = local_light_count, y = out_w, z = out_h, w = flags (bit0: mat_gbuf, bit1: ddgi_bounce, bit2: thin_gbuf)
};

struct RtLocalLight {
  position_range: vec4f,
  color_outer_cos: vec4f,
  direction_inner_cos: vec4f,
};

@group(0) @binding(0) var<uniform> u: TlasRayTraceUniform;
// __TLAS_BINDING_1__
@group(0) @binding(2) var<storage, read> tlas_instances: array<TlasInstance>;
@group(0) @binding(3) var<storage, read> blas_nodes: array<BvhNode>;
@group(0) @binding(4) var<storage, read> blas_tris: array<BlasTri>;
@group(0) @binding(5) var depth_tex: texture_2d<f32>;
@group(0) @binding(6) var out_shadow_ao: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var out_reflection: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var out_gi: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var<storage, read> local_lights: array<RtLocalLight>;
@group(0) @binding(10) var<storage, read> ddgi_packed: array<vec4f>;
@group(0) @binding(11) var gbuf_normal_roughness: texture_2d<f32>;
@group(0) @binding(12) var gbuf_albedo_metallic: texture_2d<f32>;
@group(0) @binding(13) var<storage, read> sharc_keys: array<u32>;
@group(0) @binding(14) var<storage, read> sharc_resolved: array<vec4f>;
@group(0) @binding(15) var<uniform> sharc_u: SharcUniform;

const PI: f32 = 3.14159265359;

fn hash23(p3: vec3f) -> vec2f {
  var p = fract(p3 * vec3f(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yzx + 33.33);
  return fract((p.xx + p.yz) * p.zy);
}

fn basis_from_normal(n: vec3f) -> mat3x3f {
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t = normalize(cross(up, n));
  let b = cross(n, t);
  return mat3x3f(t, b, n);
}

fn eval_sky(dir: vec3f) -> vec3f {
  let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
  let ground = u.sky_horizon.rgb * 0.35;
  let upper = mix(u.sky_horizon.rgb, u.sky_zenith.rgb, pow(clamp(dir.y, 0.0, 1.0), 0.6));
  return select(mix(ground, u.sky_horizon.rgb, t * 2.0), upper, dir.y >= 0.0);
}

// --- Packed DDGI Multi-Bounce Sampling at RT Hit Points ---
fn rt_ddgi_sign_nz(v: vec2f) -> vec2f {
  return vec2f(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
}

fn rt_ddgi_oct_encode(v: vec3f) -> vec2f {
  let l1 = abs(v.x) + abs(v.y) + abs(v.z);
  var p = v.xy / max(l1, 1e-8);
  if (v.z < 0.0) {
    p = (1.0 - abs(p.yx)) * rt_ddgi_sign_nz(p);
  }
  return p;
}

fn rt_ddgi_wrap(v: i32, n: u32) -> u32 {
  let ni = max(i32(n), 1);
  return u32(((v % ni) + ni) % ni);
}

fn rt_ddgi_sample_irr(irr_off: u32, w: u32, h: u32, fpx: vec2f) -> vec4f {
  let p = fpx - vec2f(0.5);
  let x0 = u32(clamp(floor(p.x), 0.0, f32(w - 1u)));
  let y0 = u32(clamp(floor(p.y), 0.0, f32(h - 1u)));
  let x1 = min(x0 + 1u, w - 1u);
  let y1 = min(y0 + 1u, h - 1u);
  let fx = clamp(p.x - f32(x0), 0.0, 1.0);
  let fy = clamp(p.y - f32(y0), 0.0, 1.0);
  let c00 = ddgi_packed[irr_off + y0 * w + x0];
  let c10 = ddgi_packed[irr_off + y0 * w + x1];
  let c01 = ddgi_packed[irr_off + y1 * w + x0];
  let c11 = ddgi_packed[irr_off + y1 * w + x1];
  return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

fn rt_ddgi_vis_at(vis_off: u32, w: u32, x: u32, y: u32) -> vec2f {
  let idx = y * w + x;
  let packed = ddgi_packed[vis_off + (idx >> 1u)];
  return select(packed.xy, packed.zw, (idx & 1u) != 0u);
}

fn rt_ddgi_sample_vis(vis_off: u32, w: u32, h: u32, fpx: vec2f) -> vec2f {
  let p = fpx - vec2f(0.5);
  let x0 = u32(clamp(floor(p.x), 0.0, f32(w - 1u)));
  let y0 = u32(clamp(floor(p.y), 0.0, f32(h - 1u)));
  let x1 = min(x0 + 1u, w - 1u);
  let y1 = min(y0 + 1u, h - 1u);
  let fx = clamp(p.x - f32(x0), 0.0, 1.0);
  let fy = clamp(p.y - f32(y0), 0.0, 1.0);
  let c00 = rt_ddgi_vis_at(vis_off, w, x0, y0);
  let c10 = rt_ddgi_vis_at(vis_off, w, x1, y0);
  let c01 = rt_ddgi_vis_at(vis_off, w, x0, y1);
  let c11 = rt_ddgi_vis_at(vis_off, w, x1, y1);
  return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

fn rt_ddgi_cascade(cascade_idx: u32, world: vec3f, normal: vec3f) -> vec4f {
  let base = 1u + cascade_idx * 6u;
  let h0 = ddgi_packed[base + 0u];
  let h1 = ddgi_packed[base + 1u];
  let h2 = ddgi_packed[base + 2u];
  let h3 = ddgi_packed[base + 3u];
  let h4 = ddgi_packed[base + 4u];
  let h5 = ddgi_packed[base + 5u];
  let dims = vec3u(u32(max(h2.x, 0.0)), u32(max(h2.y, 0.0)), u32(max(h2.z, 0.0)));
  if (dims.x < 2u || dims.y < 2u || dims.z < 2u) {
    return vec4f(0.0);
  }
  let origin = h0.xyz;
  let gamma = max(h0.w, 1.0);
  let spacing = max(h1.xyz, vec3f(1e-4));
  let min_sp = min(spacing.x, min(spacing.y, spacing.z));
  let biased_pos = world + normal * (h1.w * min_sp);
  let rel = (biased_pos - origin) / spacing;
  let max_c = vec3f(f32(dims.x - 1u), f32(dims.y - 1u), f32(dims.z - 1u));
  let edge = min(min(min(rel.x, max_c.x - rel.x), min(rel.y, max_c.y - rel.y)), min(rel.z, max_c.z - rel.z));
  let reach = clamp(edge + 0.5, 0.0, 1.0);
  if (reach <= 0.0) {
    return vec4f(0.0);
  }
  let scroll = vec3i(i32(round(h3.x)), i32(round(h3.y)), i32(round(h3.z)));
  let irr_off = u32(max(h3.w, 0.0));
  let vis_off = u32(max(h4.x, 0.0));
  let strength = max(h4.y, 0.0);
  let irr_w = u32(max(h4.z, 1.0));
  let irr_h = u32(max(h4.w, 1.0));
  let vis_w = u32(max(h5.x, 1.0));
  let vis_h = u32(max(h5.y, 1.0));
  let offsets_off = u32(max(h5.z, 0.0));

  let clamped = clamp(rel, vec3f(0.0), max_c);
  let base_i = vec3u(floor(clamped));
  let alpha = clamped - vec3f(base_i);
  let oct_n = rt_ddgi_oct_encode(normalize(normal)) * 0.5 + vec2f(0.5);

  var sum_irr = vec3f(0.0);
  var sum_w = 0.0;
  for (var i = 0u; i < 8u; i++) {
    let off = vec3u(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
    let coord = min(base_i + off, dims - vec3u(1u));
    let sc = vec3u(
      rt_ddgi_wrap(i32(coord.x) + scroll.x, dims.x),
      rt_ddgi_wrap(i32(coord.y) + scroll.y, dims.y),
      rt_ddgi_wrap(i32(coord.z) + scroll.z, dims.z),
    );
    let tri_v = mix(vec3f(1.0) - alpha, alpha, vec3f(off));
    var w = max(tri_v.x * tri_v.y * tri_v.z, 1e-4);
    let s_idx = (sc.y * dims.z + sc.z) * dims.x + sc.x;
    let probe_off = select(vec3f(0.0), ddgi_packed[offsets_off + s_idx].xyz, offsets_off > 0u);
    let probe_pos = origin + vec3f(coord) * spacing + probe_off;
    let to_probe = probe_pos - world;
    let d_u = length(to_probe);
    let dir_p = select(normal, to_probe / d_u, d_u > 1e-5);
    let wrap = (dot(dir_p, normal) + 1.0) * 0.5;
    w *= wrap * wrap + 0.05;

    let s_col = sc.z * dims.x + sc.x;
    let s_row = sc.y;
    // Chebyshev visibility test from probe toward biased surface point (prevents thin-wall GI leaks in reflections).
    if (vis_off > 0u) {
      let p2pt = biased_pos - probe_pos;
      let d_pt = length(p2pt);
      let dir_from_probe = select(-normal, p2pt / d_pt, d_pt > 1e-5);
      let oct_v = rt_ddgi_oct_encode(dir_from_probe) * 0.5 + vec2f(0.5);
      let vis_px = vec2f(f32(s_col * 18u + 1u), f32(s_row * 18u + 1u)) + oct_v * 16.0;
      let moments = rt_ddgi_sample_vis(vis_off, vis_w, vis_h, vis_px);
      if (d_pt > moments.x) {
        let variance = max(abs(moments.y - moments.x * moments.x), 1e-4);
        let d = d_pt - moments.x;
        var cheb = variance / (variance + d * d);
        cheb = max(cheb * cheb * cheb, 0.0);
        w *= cheb;
      }
    }

    let irr_px = vec2f(f32(s_col * 10u + 1u), f32(s_row * 10u + 1u)) + oct_n * 8.0;
    let texel = rt_ddgi_sample_irr(irr_off, irr_w, irr_h, irr_px);
    w *= clamp(texel.a, 0.0, 1.0);
    if (w <= 1e-6) {
      continue;
    }
    let irr = pow(max(texel.rgb, vec3f(0.0)), vec3f(gamma * 0.5));
    sum_irr += irr * w;
    sum_w += w;
  }
  if (sum_w <= 1e-6) {
    return vec4f(0.0);
  }
  let avg = sum_irr / sum_w;
  return vec4f(avg * avg * strength, reach);
}

fn rt_sample_ddgi(world: vec3f, normal: vec3f) -> vec4f {
  let count = u32(clamp(round(ddgi_packed[0u].x), 0.0, 2.0));
  if (count == 0u) {
    return vec4f(0.0);
  }
  let c0 = rt_ddgi_cascade(0u, world, normal);
  if (count == 1u || c0.a >= 0.999) {
    return c0;
  }
  let c1 = rt_ddgi_cascade(1u, world, normal);
  return vec4f(mix(c1.rgb, c0.rgb, c0.a), max(c0.a, c1.a));
}

fn shade_hit(
  hit_pos: vec3f,
  hit: TlasHit,
  sun_dir: vec3f,
  sun_color: vec3f,
  bias: f32,
  mask: u32,
) -> vec3f {
  let diffuse_albedo = hit.albedo * (1.0 - hit.metallic);
  var rad = hit.emissive;

  // 1. Direct sun + TLAS shadow ray.
  let ndl = max(0.0, dot(hit.world_normal, sun_dir));
  if (ndl > 0.0) {
    let sh = trace_tlas(
      hit_pos + hit.world_normal * bias,
      sun_dir,
      1e-4,
      u.sky_horizon.a,
      mask,
      true,
      u.counts.x,
      u.counts.y,
      u.counts.z,
    );
    if (!sh.hit) {
      rad += (diffuse_albedo / PI) * sun_color * ndl;
    }
  }

  // 2. Dynamic point, spot, and area lights with TLAS shadows.
  let n_lights = u.extra_counts.x;
  for (var li = 0u; li < n_lights; li++) {
    let lt = local_lights[li];
    let to_light = lt.position_range.xyz - hit_pos;
    let dist = length(to_light);
    let reach = lt.position_range.w;
    if (dist <= 1e-4 || dist >= reach) {
      continue;
    }
    let ldir = to_light / dist;
    let l_ndl = max(0.0, dot(hit.world_normal, ldir));
    if (l_ndl <= 0.0) {
      continue;
    }
    let ratio = dist / reach;
    let r2 = ratio * ratio;
    let win = clamp(1.0 - r2 * r2, 0.0, 1.0);
    var atten = (win * win) / (dist * dist + 1.0);
    if (lt.color_outer_cos.w > -0.999) {
      let cone = dot(-ldir, lt.direction_inner_cos.xyz);
      atten *= smoothstep(lt.color_outer_cos.w, max(lt.direction_inner_cos.w, lt.color_outer_cos.w + 1e-4), cone);
    }
    if (atten <= 1e-5) {
      continue;
    }
    let lsh = trace_tlas(
      hit_pos + hit.world_normal * bias,
      ldir,
      1e-4,
      max(dist - 0.03, 2e-3),
      mask,
      true,
      u.counts.x,
      u.counts.y,
      u.counts.z,
    );
    if (!lsh.hit) {
      rad += (diffuse_albedo / PI) * lt.color_outer_cos.rgb * (atten * l_ndl);
    }
  }

  // 3. Multi-bounce indirect diffuse at hit point: DDGI where covered, else unbounded SHaRC cache, else sky ambient.
  let amb_sky = mix(u.sky_horizon.rgb * 0.4, u.sky_zenith.rgb, clamp(hit.world_normal.y * 0.5 + 0.5, 0.0, 1.0)) * 0.25;
  let sharc_hit = sharc_query_ro(hit_pos, hit.world_normal);
  let fallback_gi = select(amb_sky, sharc_hit.rgb * u.bias_and_refl.z, sharc_hit.a > 0.5);
  if ((u.extra_counts.w & 2u) != 0u) {
    let ddgi = rt_sample_ddgi(hit_pos, hit.world_normal);
    rad += diffuse_albedo * mix(fallback_gi, ddgi.rgb, ddgi.a);
  } else {
    rad += diffuse_albedo * fallback_gi;
  }
  return rad;
}

struct DepthPt {
  world: vec3f,
  valid: bool,
};

fn load_world(px: vec2i, dims: vec2i) -> DepthPt {
  if (px.x < 0 || px.y < 0 || px.x >= dims.x || px.y >= dims.y) {
    return DepthPt(vec3f(0.0), false);
  }
  let d = textureLoad(depth_tex, px, 0).r;
  if (abs(d - u.camera_pos_and_clear.w) <= 1e-6) {
    return DepthPt(vec3f(0.0), false);
  }
  let ndc = vec2f(
    (f32(px.x) + 0.5) / f32(dims.x) * 2.0 - 1.0,
    1.0 - (f32(px.y) + 0.5) / f32(dims.y) * 2.0,
  );
  let clip = u.inverse_view_projection * vec4f(ndc, d, 1.0);
  if (abs(clip.w) <= 1e-6) {
    return DepthPt(vec3f(0.0), false);
  }
  return DepthPt(clip.xyz / clip.w, true);
}

// Eric Heitz (JCGT 2018): Sampling the GGX Distribution of Visible Normals (VNDF).
// Dramatically reduces grazing-angle fireflies and eliminates invalid below-horizon samples.
fn sample_ggx_vndf(v_local: vec3f, roughness: f32, rnd: vec2f) -> vec3f {
  let alpha = max(roughness * roughness, 0.002);
  let vh = normalize(vec3f(alpha * v_local.x, alpha * v_local.y, max(v_local.z, 1e-4)));
  let lensq = vh.x * vh.x + vh.y * vh.y;
  let t1 = select(vec3f(1.0, 0.0, 0.0), vec3f(-vh.y, vh.x, 0.0) * inverseSqrt(lensq), lensq > 1e-8);
  let t2 = cross(vh, t1);
  let r = sqrt(clamp(rnd.x, 0.0, 0.99999));
  let phi = 2.0 * PI * rnd.y;
  let t1_c = r * cos(phi);
  var t2_c = r * sin(phi);
  let s = 0.5 * (1.0 + vh.z);
  t2_c = (1.0 - s) * sqrt(max(0.0, 1.0 - t1_c * t1_c)) + s * t2_c;
  let nh = t1_c * t1 + t2_c * t2 + sqrt(max(0.0, 1.0 - t1_c * t1_c - t2_c * t2_c)) * vh;
  return normalize(vec3f(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
}

@compute @workgroup_size(8, 8, 1)
fn trace_scene(@builtin(global_invocation_id) id: vec3u) {
  let out_dims = vec2i(i32(u.extra_counts.y), i32(u.extra_counts.z));
  let depth_dims = vec2i(textureDimensions(depth_tex, 0));
  if (i32(id.x) >= out_dims.x || i32(id.y) >= out_dims.y) {
    return;
  }
  let px = vec2i(id.xy);
  let d_px = select(
    clamp(vec2i((vec2f(px) + vec2f(0.5)) * vec2f(depth_dims) / vec2f(out_dims)), vec2i(0), depth_dims - vec2i(1)),
    px,
    out_dims.x == depth_dims.x && out_dims.y == depth_dims.y,
  );
  let center = load_world(d_px, depth_dims);
  if (!center.valid) {
    textureStore(out_shadow_ao, px, vec4f(1.0, 1.0, 1.0, 1.0));
    textureStore(out_reflection, px, vec4f(0.0, 0.0, 0.0, 0.0));
    textureStore(out_gi, px, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }

  let pos = center.world;
  let to_cam = u.camera_pos_and_clear.xyz - pos;
  let cam_dist = length(to_cam);
  let view_dir = select(vec3f(0.0, 1.0, 0.0), to_cam / cam_dist, cam_dist > 1e-6);

  // Prefer rasterized Thin G-Buffer (zero primary rays, exact skinned/instanced/sheared vertex normals
  // and per-instance material overrides); skip 4-neighbor depth unprojection when Thin G-Buffer hits.
  let use_mat_gbuf = (u.extra_counts.w & 1u) != 0u;
  let use_thin_gbuf = (u.extra_counts.w & 4u) != 0u;
  var n = vec3f(0.0, 1.0, 0.0);
  var mat_albedo = vec3f(1.0);
  var mat_roughness = clamp(u.sun_color_and_rough.a, 0.04, 1.0);
  var mat_metallic = 0.0;
  var has_mat_hit = false;
  var pixel_scale = cam_dist / f32(max(depth_dims.x, depth_dims.y)) * 0.1;
  let mask = u.counts.w;
  if (use_thin_gbuf) {
    let g_dims = vec2i(textureDimensions(gbuf_normal_roughness, 0));
    let g_px = select(
      clamp(vec2i((vec2f(px) + vec2f(0.5)) * vec2f(g_dims) / vec2f(out_dims)), vec2i(0), g_dims - vec2i(1)),
      px,
      out_dims.x == g_dims.x && out_dims.y == g_dims.y,
    );
    let g_nr = textureLoad(gbuf_normal_roughness, g_px, 0);
    if (dot(g_nr.xyz, g_nr.xyz) >= 0.01) {
      let g_am = textureLoad(gbuf_albedo_metallic, g_px, 0);
      has_mat_hit = true;
      n = normalize(g_nr.xyz);
      if (dot(n, view_dir) < 0.0) {
        n = -n;
      }
      mat_albedo = max(g_am.rgb, vec3f(0.0));
      mat_roughness = clamp(g_nr.w, 0.04, 1.0);
      mat_metallic = clamp(g_am.w, 0.0, 1.0);
    }
  }
  if (!has_mat_hit) {
    let sr = load_world(d_px + vec2i(1, 0), depth_dims);
    let sl = load_world(d_px - vec2i(1, 0), depth_dims);
    let sd = load_world(d_px + vec2i(0, 1), depth_dims);
    let su = load_world(d_px - vec2i(0, 1), depth_dims);

    var dx = vec3f(1.0, 0.0, 0.0);
    var has_dx = true;
    if (sr.valid && sl.valid) {
      let dr = sr.world - pos;
      let dl = pos - sl.world;
      dx = select(dl, dr, dot(dr, dr) < dot(dl, dl));
    } else if (sr.valid) {
      dx = sr.world - pos;
    } else if (sl.valid) {
      dx = pos - sl.world;
    } else {
      has_dx = false;
    }

    var dy = vec3f(0.0, 0.0, 1.0);
    var has_dy = true;
    if (sd.valid && su.valid) {
      let dd = sd.world - pos;
      let du = pos - su.world;
      dy = select(du, dd, dot(dd, dd) < dot(du, du));
    } else if (sd.valid) {
      dy = sd.world - pos;
    } else if (su.valid) {
      dy = pos - su.world;
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
    if (dot(n, view_dir) < 0.0) {
      n = -n;
    }
    pixel_scale = max(length(dx), length(dy)) * 0.05;
  }
  if (!has_mat_hit && use_mat_gbuf && cam_dist > 1e-3) {
    let prim_dir = -view_dir;
    let t_lo = max(cam_dist * 0.92, 1e-3);
    let t_hi = cam_dist * 1.08 + 0.05;
    let ph = trace_tlas(
      u.camera_pos_and_clear.xyz,
      prim_dir,
      t_lo,
      t_hi,
      mask,
      false,
      u.counts.x,
      u.counts.y,
      u.counts.z,
    );
    if (ph.hit) {
      has_mat_hit = true;
      n = ph.world_normal;
      if (dot(n, view_dir) < 0.0) {
        n = -n;
      }
      mat_albedo = ph.albedo;
      mat_roughness = clamp(ph.roughness, 0.04, 1.0);
      mat_metallic = clamp(ph.metallic, 0.0, 1.0);
    }
  }

  let min_bias = max(u.bias_and_refl.x, 1e-5);
  let bias = max(min_bias, min(pixel_scale, cam_dist * 2e-4));
  let ray_origin = pos + n * bias;
  let max_dist = max(u.sky_horizon.a, 1.0);

  let sun_raw = u.sun_dir_and_radius.xyz;
  let sun_dir = select(vec3f(0.0, 1.0, 0.0), normalize(sun_raw), length(sun_raw) > 1e-6);
  let sun_radius = max(u.sun_dir_and_radius.w, 0.0);
  let sun_color = u.sun_color_and_rough.rgb;

  // 1. Ray-traced soft sun shadows.
  let sh_samples = clamp(u32(round(u.samples_seed.x)), 1u, 8u);
  let sun_basis = basis_from_normal(sun_dir);
  var sun_vis = 0.0;
  if (dot(n, sun_dir) > 0.0) {
    for (var s = 0u; s < sh_samples; s++) {
      var d_ray = sun_dir;
      if (sun_radius > 1e-5) {
        let rnd = hash23(vec3f(vec2f(id.xy), u.samples_seed.w * 13.7 + f32(s) * 5.3));
        let r_disk = sqrt(rnd.x) * sun_radius;
        let phi = rnd.y * 2.0 * PI;
        d_ray = normalize(sun_basis * vec3f(r_disk * cos(phi), r_disk * sin(phi), 1.0));
      }
      let h = trace_tlas(ray_origin, d_ray, 1e-4, max_dist, mask, true, u.counts.x, u.counts.y, u.counts.z);
      if (!h.hit) {
        sun_vis += 1.0;
      }
    }
    sun_vis /= f32(sh_samples);
  }

  // 2. Cosine-weighted RTAO.
  let ao_samples = clamp(u32(round(u.samples_seed.y)), 0u, 16u);
  let ao_radius = max(u.sky_zenith.a, 0.05);
  let n_basis = basis_from_normal(n);
  var ao_vis = 1.0;
  var dist_sum = 0.0;
  if (ao_samples > 0u) {
    var open = 0.0;
    for (var s = 0u; s < ao_samples; s++) {
      let rnd = hash23(vec3f(vec2f(id.xy) + vec2f(19.1, 7.3), u.samples_seed.w * 17.3 + f32(s) * 11.1));
      let phi = 2.0 * PI * ((f32(s) + rnd.x) / f32(ao_samples));
      let cos_t = sqrt(clamp(1.0 - rnd.y, 0.0, 1.0));
      let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
      let d_ray = normalize(n_basis * vec3f(cos(phi) * sin_t, sin(phi) * sin_t, cos_t));
      let h = trace_tlas(ray_origin, d_ray, 1e-4, ao_radius, mask, false, u.counts.x, u.counts.y, u.counts.z);
      if (!h.hit) {
        open += 1.0;
        dist_sum += 1.0;
      } else {
        let frac = clamp(h.t / ao_radius, 0.0, 1.0);
        open += smoothstep(0.0, 1.0, frac);
        dist_sum += frac;
      }
    }
    ao_vis = clamp(open / f32(ao_samples), 0.0, 1.0);
    dist_sum /= f32(ao_samples);
  }
  // 3. Heitz VNDF GGX Specular Ray-Traced Reflections with matte roughness cutoff (> 0.75)
  //    and energy-conserving local-hit coverage for sky IBL attenuation.
  let refl_samples = clamp(u32(round(u.bias_and_refl.y)), 0u, 8u);
  let roughness = select(clamp(u.sun_color_and_rough.a, 0.0, 1.0), mat_roughness, use_mat_gbuf);
  let skip_matte = use_mat_gbuf && (roughness > 0.75);
  var refl_acc = vec3f(0.0);
  var refl_dist = max_dist;
  var refl_valid = 0.0;
  var refl_hits = 0.0;
  var refl_occlude = 0.0;
  if (refl_samples > 0u && !skip_matte) {
    refl_dist = 0.0;
    let v_local = transpose(n_basis) * view_dir;
    for (var s = 0u; s < refl_samples; s++) {
      var r_dir: vec3f;
      if (roughness < 0.02) {
        r_dir = reflect(-view_dir, n);
      } else {
        let rnd = hash23(vec3f(vec2f(id.xy) + vec2f(41.7, 3.1), u.samples_seed.w * 23.1 + f32(s) * 7.9));
        let h_local = sample_ggx_vndf(v_local, roughness, rnd);
        let h_world = normalize(n_basis * h_local);
        r_dir = reflect(-view_dir, h_world);
      }
      if (dot(r_dir, n) <= 0.0) {
        continue;
      }
      let h = trace_tlas(ray_origin, r_dir, 1e-4, max_dist, mask, false, u.counts.x, u.counts.y, u.counts.z);
      if (h.hit) {
        let hp = ray_origin + r_dir * h.t;
        refl_acc += shade_hit(hp, h, sun_dir, sun_color, bias, mask);
        refl_dist += h.t;
        refl_hits += 1.0;
      } else if (!use_mat_gbuf) {
        refl_acc += eval_sky(r_dir);
        refl_dist += max_dist;
      } else {
        // Forward PBR (`surface_sky`) already supplies sky IBL reflection; add local RT hits only.
        refl_dist += max_dist;
      }
      refl_valid += 1.0;
    }
    if (refl_valid > 0.0) {
      refl_acc /= refl_valid;
      refl_dist /= refl_valid;
    } else {
      refl_dist = max_dist;
    }
    if (use_mat_gbuf) {
      let f0 = select(vec3f(0.04), mix(vec3f(0.04), mat_albedo, mat_metallic), has_mat_hit);
      let ndv = clamp(dot(n, view_dir), 0.0, 1.0);
      let fresnel = f0 + (vec3f(1.0) - f0) * pow(1.0 - ndv, 5.0);
      let rough_fade = smoothstep(0.75, 0.58, roughness);
      let rough_scale = clamp(1.0 - roughness * 0.65, 0.15, 1.0) * rough_fade;
      refl_acc *= fresnel * rough_scale;
      let hit_cov = select(0.0, refl_hits / refl_valid, refl_valid > 0.0);
      let fresnel_luma = dot(fresnel, vec3f(0.2126, 0.7152, 0.0722));
      refl_occlude = clamp(hit_cov * fresnel_luma * rough_fade, 0.0, 0.75);
    }
  }
  // When primary material lookup is active, pack denoised reflection sky-occlusion into `out_shadow_ao.b`
  // and per-pixel material roughness into `out_shadow_ao.a` for specular SVGF + energy-conserving composite.
  let sao_out = select(
    vec4f(sun_vis, ao_vis, dist_sum, 1.0),
    vec4f(sun_vis, ao_vis, refl_occlude, roughness),
    use_mat_gbuf,
  );
  textureStore(out_shadow_ao, px, sao_out);
  textureStore(out_reflection, px, vec4f(refl_acc, refl_dist));

  // 4. 1-Bounce Cosine-Weighted Diffuse GI (skipped when DDGI supplies multi-bounce diffuse in-shader).
  let gi_samples = clamp(u32(round(u.samples_seed.z)), 0u, 8u);
  var gi_acc = vec3f(0.0);
  if (gi_samples > 0u) {
    for (var s = 0u; s < gi_samples; s++) {
      let rnd = hash23(vec3f(vec2f(id.xy) + vec2f(73.3, 29.9), u.samples_seed.w * 31.7 + f32(s) * 13.3));
      let phi = 2.0 * PI * ((f32(s) + rnd.x) / f32(gi_samples));
      let cos_t = sqrt(clamp(1.0 - rnd.y, 0.0, 1.0));
      let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
      let d_ray = normalize(n_basis * vec3f(cos(phi) * sin_t, sin(phi) * sin_t, cos_t));
      let h = trace_tlas(ray_origin, d_ray, 1e-4, max_dist, mask, false, u.counts.x, u.counts.y, u.counts.z);
      if (h.hit) {
        let hp = ray_origin + d_ray * h.t;
        gi_acc += shade_hit(hp, h, sun_dir, sun_color, bias, mask);
      } else {
        gi_acc += eval_sky(d_ray);
      }
    }
    // Cosine-weighted hemisphere PDF = cos(theta)/PI; multiplying by PI cancels 1/PI.
    gi_acc = (gi_acc / f32(gi_samples)) * max(u.bias_and_refl.z, 0.0);
    if (use_mat_gbuf && has_mat_hit) {
      gi_acc *= mat_albedo * (1.0 - mat_metallic);
    }
  }
  textureStore(out_gi, px, vec4f(gi_acc, 1.0));
}
