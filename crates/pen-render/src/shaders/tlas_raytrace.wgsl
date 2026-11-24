// 2-Level TLAS/BLAS GPU Compute Ray Tracer:
// Outputs:
//  - out_shadow_ao  (Rgba16Float): (rt_shadow, rtao, hit_dist, 1.0)
//  - out_reflection (Rgba16Float): (ggx_refl.rgb, hit_dist)
//  - out_gi         (Rgba16Float): (one_bounce_diffuse_gi.rgb, 1.0)
//
// Prefixed at compile time with tlas_traversal.wgsl.

struct TlasRayTraceUniform {
  inverse_view_projection: mat4x4f,
  camera_pos_and_clear: vec4f, // xyz = camera pos, w = depth clear (1.0 or 0.0)
  sun_dir_and_radius: vec4f,   // xyz = normalized toward sun, w = sun angular radius
  sun_color_and_rough: vec4f,  // rgb = sun radiance, a = default surface roughness
  sky_zenith: vec4f,           // rgb = sky zenith radiance, a = ao_radius
  sky_horizon: vec4f,          // rgb = sky horizon radiance, a = max_ray_dist
  samples_seed: vec4f,         // x = shadow_samples, y = ao_samples, z = gi_samples, w = frame_seed
  bias_and_refl: vec4f,        // x = normal_bias, y = refl_samples, z = gi_bounce_scale, w = visibility_mask_f32
  counts: vec4u,               // x = tlas_nodes, y = tlas_instances, z = blas_tris, w = visibility_mask
};

@group(0) @binding(0) var<uniform> u: TlasRayTraceUniform;
@group(0) @binding(1) var<storage, read> tlas_nodes: array<BvhNode>;
@group(0) @binding(2) var<storage, read> tlas_instances: array<TlasInstance>;
@group(0) @binding(3) var<storage, read> blas_nodes: array<BvhNode>;
@group(0) @binding(4) var<storage, read> blas_tris: array<BlasTri>;
@group(0) @binding(5) var depth_tex: texture_2d<f32>;
@group(0) @binding(6) var out_shadow_ao: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var out_reflection: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var out_gi: texture_storage_2d<rgba16float, write>;

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

fn shade_hit(
  hit_pos: vec3f,
  hit: TlasHit,
  sun_dir: vec3f,
  sun_color: vec3f,
  bias: f32,
  mask: u32,
) -> vec3f {
  var rad = hit.emissive;
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
      let diffuse_albedo = hit.albedo * (1.0 - hit.metallic);
      rad += (diffuse_albedo / PI) * sun_color * ndl;
    }
  }
  let amb_sky = mix(u.sky_horizon.rgb * 0.4, u.sky_zenith.rgb, clamp(hit.world_normal.y * 0.5 + 0.5, 0.0, 1.0));
  rad += hit.albedo * (1.0 - hit.metallic) * amb_sky * 0.25;
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

fn sample_ggx_half(rnd: vec2f, roughness: f32) -> vec3f {
  let a = max(roughness * roughness, 0.002);
  let phi = 2.0 * PI * rnd.x;
  let cos_t = sqrt(clamp((1.0 - rnd.y) / (1.0 + (a * a - 1.0) * rnd.y), 0.0, 1.0));
  let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
  return vec3f(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);
}

@compute @workgroup_size(8, 8, 1)
fn trace_scene(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(depth_tex, 0));
  if (i32(id.x) >= dims.x || i32(id.y) >= dims.y) {
    return;
  }
  let px = vec2i(id.xy);
  let center = load_world(px, dims);
  if (!center.valid) {
    textureStore(out_shadow_ao, px, vec4f(1.0, 1.0, 1.0, 1.0));
    textureStore(out_reflection, px, vec4f(0.0, 0.0, 0.0, 0.0));
    textureStore(out_gi, px, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }

  let pos = center.world;
  let sr = load_world(px + vec2i(1, 0), dims);
  let sl = load_world(px - vec2i(1, 0), dims);
  let sd = load_world(px + vec2i(0, 1), dims);
  let su = load_world(px - vec2i(0, 1), dims);

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

  var n = vec3f(0.0, 1.0, 0.0);
  if (has_dx && has_dy) {
    let raw_n = cross(dx, dy);
    let area2 = dot(raw_n, raw_n);
    let ref2 = dot(dx, dx) * dot(dy, dy);
    if (area2 > 1e-12 * ref2 && ref2 > 0.0) {
      n = raw_n * inverseSqrt(area2);
    }
  }
  let to_cam = u.camera_pos_and_clear.xyz - pos;
  let cam_dist = length(to_cam);
  let view_dir = select(vec3f(0.0, 1.0, 0.0), to_cam / cam_dist, cam_dist > 1e-6);
  if (dot(n, view_dir) < 0.0) {
    n = -n;
  }

  let min_bias = max(u.bias_and_refl.x, 1e-5);
  let pixel_scale = max(length(dx), length(dy)) * 0.05;
  let bias = max(min_bias, min(pixel_scale, cam_dist * 2e-4));
  let ray_origin = pos + n * bias;
  let mask = u.counts.w;
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
      let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
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
  textureStore(out_shadow_ao, px, vec4f(sun_vis, ao_vis, dist_sum, 1.0));

  // 3. GGX Specular Ray-Traced Reflections.
  let refl_samples = clamp(u32(round(u.bias_and_refl.y)), 0u, 8u);
  let roughness = clamp(u.sun_color_and_rough.a, 0.0, 1.0);
  var refl_acc = vec3f(0.0);
  var refl_dist = 0.0;
  var refl_valid = 0.0;
  if (refl_samples > 0u) {
    for (var s = 0u; s < refl_samples; s++) {
      var r_dir: vec3f;
      if (roughness < 0.02) {
        r_dir = reflect(-view_dir, n);
      } else {
        let rnd = hash23(vec3f(vec2f(id.xy) + vec2f(41.7, 3.1), u.samples_seed.w * 23.1 + f32(s) * 7.9));
        let h_local = sample_ggx_half(rnd, roughness);
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
      } else {
        refl_acc += eval_sky(r_dir);
        refl_dist += max_dist;
      }
      refl_valid += 1.0;
    }
    if (refl_valid > 0.0) {
      refl_acc /= refl_valid;
      refl_dist /= refl_valid;
    }
  }
  textureStore(out_reflection, px, vec4f(refl_acc, refl_dist));

  // 4. 1-Bounce Cosine-Weighted Diffuse GI.
  let gi_samples = clamp(u32(round(u.samples_seed.z)), 0u, 8u);
  var gi_acc = vec3f(0.0);
  if (gi_samples > 0u) {
    for (var s = 0u; s < gi_samples; s++) {
      let rnd = hash23(vec3f(vec2f(id.xy) + vec2f(73.3, 29.9), u.samples_seed.w * 31.7 + f32(s) * 13.3));
      let phi = 2.0 * PI * ((f32(s) + rnd.x) / f32(gi_samples));
      let cos_t = sqrt(clamp(1.0 - rnd.y, 0.0, 1.0));
      let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
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
  }
  textureStore(out_gi, px, vec4f(gi_acc, 1.0));
}
