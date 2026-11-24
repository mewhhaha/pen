// Multi-Bounce GPU Path-Traced Lightmap Baker, Gutter Dilator, and Bilateral Denoiser.
// Prefixed at compile time with tlas_traversal.wgsl.

struct BakeUniform {
  sun_dir_and_radius: vec4f, // xyz = normalized sun dir, w = sun angular radius
  sun_color_and_bias: vec4f, // rgb = sun radiance, a = normal bias
  sky_zenith_and_max: vec4f, // rgb = sky zenith, a = max ray dist
  sky_horizon: vec4f,        // rgb = sky horizon, a = 0
  counts: vec4u,             // x = tlas_nodes, y = tlas_insts, z = blas_tris, w = visibility_mask
  bake_cfg: vec4u,           // x = width, y = height, z = samples_per_texel, w = max_bounces
};

const LM_PI: f32 = 3.14159265359;

fn lm_hash23(p3: vec3f) -> vec2f {
  var p = fract(p3 * vec3f(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yzx + 33.33);
  return fract((p.xx + p.yz) * p.zy);
}

fn lm_basis(n: vec3f) -> mat3x3f {
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t = normalize(cross(up, n));
  let b = cross(n, t);
  return mat3x3f(t, b, n);
}

fn lm_eval_sky(dir: vec3f, u: BakeUniform) -> vec3f {
  let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
  let ground = u.sky_horizon.rgb * 0.25;
  let upper = mix(u.sky_horizon.rgb, u.sky_zenith_and_max.rgb, pow(clamp(dir.y, 0.0, 1.0), 0.6));
  return select(mix(ground, u.sky_horizon.rgb, t * 2.0), upper, dir.y >= 0.0);
}

fn lm_luma(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// ---------------- Pass 1: Multi-Bounce Path-Traced Lightmap Bake ----------------
@group(0) @binding(0) var<uniform> u: BakeUniform;
@group(0) @binding(1) var<storage, read> tlas_nodes: array<BvhNode>;
@group(0) @binding(2) var<storage, read> tlas_instances: array<TlasInstance>;
@group(0) @binding(3) var<storage, read> blas_nodes: array<BvhNode>;
@group(0) @binding(4) var<storage, read> blas_tris: array<BlasTri>;
@group(0) @binding(5) var gbuf_pos: texture_2d<f32>;
@group(0) @binding(6) var gbuf_norm: texture_2d<f32>;
@group(0) @binding(7) var out_irr: texture_storage_2d<rgba32float, write>;
@group(0) @binding(8) var out_dir: texture_storage_2d<rgba32float, write>;
@group(0) @binding(9) var out_shadow: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn bake_lightmap(@builtin(global_invocation_id) id: vec3u) {
  let w = u.bake_cfg.x;
  let h = u.bake_cfg.y;
  if (id.x >= w || id.y >= h) {
    return;
  }
  let px = vec2i(id.xy);
  let pos_w = textureLoad(gbuf_pos, px, 0);
  if (pos_w.w < 0.5) {
    textureStore(out_irr, px, vec4f(0.0, 0.0, 0.0, 0.0));
    textureStore(out_dir, px, vec4f(0.5, 1.0, 0.5, 0.0));
    textureStore(out_shadow, px, vec4f(1.0, 0.0, 0.0, 0.0));
    return;
  }

  let surf_pos = pos_w.xyz;
  let n_raw = textureLoad(gbuf_norm, px, 0).xyz;
  let surf_n = select(vec3f(0.0, 1.0, 0.0), normalize(n_raw), length(n_raw) > 1e-6);
  let bias = max(u.sun_color_and_bias.w, 1e-4);
  let max_dist = max(u.sky_zenith_and_max.w, 1.0);
  let mask = u.counts.w;
  let sun_raw = u.sun_dir_and_radius.xyz;
  let sun_dir = select(vec3f(0.0, 1.0, 0.0), normalize(sun_raw), length(sun_raw) > 1e-6);
  let sun_rad = max(u.sun_dir_and_radius.w, 0.0);
  let sun_col = u.sun_color_and_bias.rgb;

  let samples = clamp(u.bake_cfg.z, 1u, 256u);
  let max_bounces = clamp(u.bake_cfg.w, 1u, 4u);
  let surf_basis = lm_basis(surf_n);
  let sun_basis = lm_basis(sun_dir);

  var irr_sum = vec3f(0.0);
  var dir_vec_sum = vec3f(0.0);
  var dir_weight_sum = 0.0;
  var shadow_vis_sum = 0.0;

  let strata = max(u32(sqrt(f32(samples))), 1u);

  for (var s = 0u; s < samples; s++) {
    // 1. Direct soft sun shadowmask visibility.
    var s_vis = 0.0;
    let ndl0 = max(0.0, dot(surf_n, sun_dir));
    if (ndl0 > 0.0) {
      var d_sun = sun_dir;
      if (sun_rad > 1e-5) {
        let r_sun = lm_hash23(vec3f(vec2f(id.xy), f32(s) * 7.13 + 1.0));
        let rd = sqrt(r_sun.x) * sun_rad;
        let phi = r_sun.y * 2.0 * LM_PI;
        d_sun = normalize(sun_basis * vec3f(rd * cos(phi), rd * sin(phi), 1.0));
      }
      let sh = trace_tlas(surf_pos + surf_n * bias, d_sun, 1e-4, max_dist, mask, true, u.counts.x, u.counts.y, u.counts.z);
      if (!sh.hit) {
        s_vis = 1.0;
      }
    }
    shadow_vis_sum += s_vis;

    // 2. Multi-bounce indirect path trace (cosine-weighted hemisphere sampling).
    let sx = s % strata;
    let sy = (s / strata) % strata;
    let jitter = lm_hash23(vec3f(vec2f(id.xy) + vec2f(13.7, 5.1), f32(s) * 19.3));
    let u1 = (f32(sx) + jitter.x) / f32(strata);
    let u2 = (f32(sy) + jitter.y) / f32(strata);

    let phi0 = 2.0 * LM_PI * u1;
    let cos0 = sqrt(clamp(1.0 - u2, 0.0, 1.0));
    let sin0 = sqrt(max(0.0, 1.0 - cos0 * cos0));
    let first_dir = normalize(surf_basis * vec3f(cos(phi0) * sin0, sin(phi0) * sin0, cos0));

    var ray_orig = surf_pos + surf_n * bias;
    var ray_dir = first_dir;
    var throughput = vec3f(1.0);
    var sample_irr = vec3f(0.0);

    for (var b = 0u; b < max_bounces; b++) {
      let hit = trace_tlas(ray_orig, ray_dir, 1e-4, max_dist, mask, false, u.counts.x, u.counts.y, u.counts.z);
      if (!hit.hit) {
        sample_irr += throughput * lm_eval_sky(ray_dir, u);
        break;
      }
      let hp = ray_orig + ray_dir * hit.t;
      let diff_albedo = hit.albedo * (1.0 - hit.metallic);

      // Emissive contribution at hit.
      sample_irr += throughput * hit.emissive;

      // Direct sun next-event estimation at hit.
      let ndl = max(0.0, dot(hit.world_normal, sun_dir));
      if (ndl > 0.0 && lm_luma(sun_col) > 1e-6) {
        let sh = trace_tlas(hp + hit.world_normal * bias, sun_dir, 1e-4, max_dist, mask, true, u.counts.x, u.counts.y, u.counts.z);
        if (!sh.hit) {
          sample_irr += throughput * (diff_albedo / LM_PI) * sun_col * ndl;
        }
      }

      // Continue path along cosine-weighted next bounce.
      throughput *= diff_albedo;
      if (max(throughput.x, max(throughput.y, throughput.z)) < 1e-3) {
        break;
      }
      let rb = lm_hash23(vec3f(vec2f(id.xy) + vec2f(f32(b) * 41.0, 17.0), f32(s) * 29.7 + f32(b) * 3.1));
      let phib = 2.0 * LM_PI * rb.x;
      let cosb = sqrt(clamp(1.0 - rb.y, 0.0, 1.0));
      let sinb = sqrt(max(0.0, 1.0 - cosb * cosb));
      let h_basis = lm_basis(hit.world_normal);
      ray_orig = hp + hit.world_normal * bias;
      ray_dir = normalize(h_basis * vec3f(cos(phib) * sinb, sin(phib) * sinb, cosb));
    }

    irr_sum += sample_irr;
    let w_dir = lm_luma(sample_irr);
    dir_vec_sum += first_dir * w_dir;
    dir_weight_sum += w_dir;
  }

  let inv_s = 1.0 / f32(samples);
  let irr = irr_sum * inv_s;
  let shadow_vis = shadow_vis_sum * inv_s;

  // Include direct sun in directional lightmap dominant vector when visible.
  let direct_luma = lm_luma(sun_col) * max(0.0, dot(surf_n, sun_dir)) * shadow_vis;
  let total_dir_vec = dir_vec_sum * inv_s + sun_dir * direct_luma;
  let total_dir_w = dir_weight_sum * inv_s + direct_luma;
  let dv_len = length(total_dir_vec);
  let dom_dir = select(surf_n, total_dir_vec / dv_len, dv_len > 1e-5);
  let directionality = select(0.0, clamp(dv_len / max(total_dir_w, 1e-5), 0.0, 1.0), total_dir_w > 1e-5);

  textureStore(out_irr, px, vec4f(irr, 1.0));
  textureStore(out_dir, px, vec4f(dom_dir * 0.5 + vec3f(0.5), directionality));
  textureStore(out_shadow, px, vec4f(shadow_vis, 1.0, 0.0, 1.0));
}

// ---------------- Pass 2: Gutter Dilation (prevents UV2 seam bleeding) ----------------
@group(0) @binding(0) var dilate_src: texture_2d<f32>;
@group(0) @binding(1) var dilate_dst: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn dilate_lightmap(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(dilate_src, 0));
  let px = vec2i(id.xy);
  if (px.x >= dims.x || px.y >= dims.y) {
    return;
  }
  let center = textureLoad(dilate_src, px, 0);
  if (center.a > 0.5) {
    textureStore(dilate_dst, px, center);
    return;
  }
  var acc = vec3f(0.0);
  var cnt = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      let q = px + vec2i(dx, dy);
      if (q.x < 0 || q.y < 0 || q.x >= dims.x || q.y >= dims.y) {
        continue;
      }
      let s = textureLoad(dilate_src, q, 0);
      if (s.a > 0.5) {
        acc += s.rgb;
        cnt += 1.0;
      }
    }
  }
  if (cnt > 0.0) {
    textureStore(dilate_dst, px, vec4f(acc / cnt, 1.0));
  } else {
    textureStore(dilate_dst, px, center);
  }
}

// ---------------- Pass 3: Normal-Aware Bilateral Lightmap Denoise ----------------
@group(0) @binding(0) var dn_src: texture_2d<f32>;
@group(0) @binding(1) var dn_norm: texture_2d<f32>;
@group(0) @binding(2) var dn_pos: texture_2d<f32>;
@group(0) @binding(3) var dn_dst: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn denoise_lightmap(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(dn_src, 0));
  let px = vec2i(id.xy);
  if (px.x >= dims.x || px.y >= dims.y) {
    return;
  }
  let center = textureLoad(dn_src, px, 0);
  let cp = textureLoad(dn_pos, px, 0);
  if (cp.w < 0.5) {
    textureStore(dn_dst, px, center);
    return;
  }
  let cn = normalize(textureLoad(dn_norm, px, 0).xyz);
  var sum_c = vec3f(0.0);
  var sum_w = 0.0;
  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let q = px + vec2i(dx, dy);
      if (q.x < 0 || q.y < 0 || q.x >= dims.x || q.y >= dims.y) {
        continue;
      }
      let qp = textureLoad(dn_pos, q, 0);
      if (qp.w < 0.5) {
        continue;
      }
      let qn = normalize(textureLoad(dn_norm, q, 0).xyz);
      let wn = pow(max(0.0, dot(cn, qn)), 32.0);
      let d2 = f32(dx * dx + dy * dy);
      let ws = exp(-d2 * 0.25);
      let wp = exp(-dot(qp.xyz - cp.xyz, qp.xyz - cp.xyz) * 2.0);
      let w = wn * ws * wp;
      sum_c += textureLoad(dn_src, q, 0).rgb * w;
      sum_w += w;
    }
  }
  let out_rgb = select(center.rgb, sum_c / sum_w, sum_w > 1e-5);
  textureStore(dn_dst, px, vec4f(out_rgb, center.a));
}
