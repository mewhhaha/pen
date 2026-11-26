// Spatiotemporal Variance-Guided Filter (SVGF, Schied et al., HPG 2017) with Specular Reflection mode.
// Pass 1 (svgf_temporal): motion-reprojected temporal accumulation (using virtual hit-point parallax
//                         motion `motion.zw` + 3x3 YCoCg neighborhood clamping in specular mode)
//                         + 1st/2nd luminance moments + spatial fallback variance on disocclusion.
// Pass 2 (svgf_atrous):   multi-scale 5x5 B3-spline A-Trous edge-stopping wavelet filter
//                         guided by depth, normal, roughness (`normal.w` in specular mode), and variance.

struct SvgfUniform {
  dims: vec4u,           // x = width, y = height, z = step_size, w = flags (bit0: reset, bit1: specular_mode, bit2: hit_distance_adaptive)
  temporal_params: vec4f, // x = min_alpha, y = max_history, z = depth_threshold, w = normal_threshold
  atrous_params: vec4f,  // x = phi_color, y = phi_normal, z = phi_depth, w = variance_boost
};

@group(0) @binding(0) var<uniform> u: SvgfUniform;

// Temporal pass bindings.
@group(0) @binding(1) var curr_noisy: texture_2d<f32>;
@group(0) @binding(2) var prev_color: texture_2d<f32>;
@group(0) @binding(3) var prev_moments: texture_2d<f32>;
@group(0) @binding(4) var curr_depth: texture_2d<f32>;
@group(0) @binding(5) var prev_depth: texture_2d<f32>;
@group(0) @binding(6) var curr_normal: texture_2d<f32>;
@group(0) @binding(7) var prev_normal: texture_2d<f32>;
@group(0) @binding(8) var motion_tex: texture_2d<f32>;
@group(0) @binding(9) var out_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var out_moments: texture_storage_2d<rgba16float, write>;

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn finite_clamp(value: f32, limit: f32) -> f32 {
  let bits = bitcast<u32>(value) & 0x7fffffffu;
  return select(clamp(value, 0.0, limit), 0.0, bits > 0x7f800000u);
}

fn finite_rgb(c: vec3f) -> vec3f {
  return vec3f(finite_clamp(c.x, 65504.0), finite_clamp(c.y, 65504.0), finite_clamp(c.z, 65504.0));
}

fn rgb_to_ycocg(c: vec3f) -> vec3f {
  return vec3f(
    dot(c, vec3f(0.25, 0.5, 0.25)),
    dot(c, vec3f(0.5, 0.0, -0.5)),
    dot(c, vec3f(-0.25, 0.5, -0.25)),
  );
}

fn ycocg_to_rgb(y: vec3f) -> vec3f {
  return max(vec3f(y.x + y.y - y.z, y.x + y.z, y.x - y.y - y.z), vec3f(0.0));
}

fn load_normal(tex: texture_2d<f32>, px: vec2i) -> vec3f {
  let raw = textureLoad(tex, px, 0).xyz;
  let l2 = dot(raw, raw);
  return select(vec3f(0.0, 1.0, 0.0), raw * inverseSqrt(l2), l2 > 1e-8);
}

fn is_consistent(z_c: f32, z_p: f32, n_c: vec3f, n_p: vec3f) -> bool {
  let depth_tol = max(u.temporal_params.z, 1e-4) * max(abs(z_c), 1e-3);
  if (abs(z_c - z_p) > depth_tol) {
    return false;
  }
  return dot(n_c, n_p) >= u.temporal_params.w;
}

@compute @workgroup_size(8, 8, 1)
fn svgf_temporal(@builtin(global_invocation_id) id: vec3u) {
  let w = i32(u.dims.x);
  let h = i32(u.dims.y);
  let px = vec2i(id.xy);
  if (px.x >= w || px.y >= h) {
    return;
  }

  let reset_hist = (u.dims.w & 1u) != 0u;
  let spec_mode = (u.dims.w & 2u) != 0u;

  let adaptive_hit_dist = (u.dims.w & 4u) != 0u;
  let noisy = textureLoad(curr_noisy, px, 0);
  let sample_c = finite_rgb(noisy.rgb);
  let hit_dist = select(0.0, finite_clamp(noisy.a, 65504.0), adaptive_hit_dist);
  let z_c = textureLoad(curr_depth, px, 0).r;
  let n_raw = textureLoad(curr_normal, px, 0);
  let n_l2 = dot(n_raw.xyz, n_raw.xyz);
  let n_c = select(vec3f(0.0, 1.0, 0.0), n_raw.xyz * inverseSqrt(n_l2), n_l2 > 1e-8);
  let rough_c = clamp(n_raw.w, 0.04, 1.0);
  let lum_c = finite_clamp(luminance(sample_c), 240.0);
  let moments_c = vec2f(lum_c, lum_c * lum_c);

  var acc_color = sample_c;
  var acc_moments = moments_c;
  var hist_len = 1.0;
  var filtered_hit_dist = hit_dist;

  if (!reset_hist) {
    let raw_mv = textureLoad(motion_tex, px, 0);
    let mv = select(raw_mv.xy, raw_mv.zw, spec_mode);
    let prev_uv = (vec2f(px) + vec2f(0.5)) / vec2f(f32(w), f32(h)) - mv;
    let prev_pos = prev_uv * vec2f(f32(w), f32(h)) - vec2f(0.5);
    let base = vec2i(floor(prev_pos));
    let frac = prev_pos - vec2f(base);

    let bilinear_w = array<f32, 4>(
      (1.0 - frac.x) * (1.0 - frac.y),
      frac.x * (1.0 - frac.y),
      (1.0 - frac.x) * frac.y,
      frac.x * frac.y,
    );
    let offsets = array<vec2i, 4>(
      vec2i(0, 0),
      vec2i(1, 0),
      vec2i(0, 1),
      vec2i(1, 1),
    );

    var sum_w = 0.0;
    var hist_c = vec3f(0.0);
    var hist_m = vec2f(0.0);
    var hist_l = 0.0;
    var hist_hit_dist = 0.0;
    var hit_dist_w = 0.0;

    for (var i = 0; i < 4; i++) {
      let q = base + offsets[i];
      if (q.x < 0 || q.y < 0 || q.x >= w || q.y >= h) {
        continue;
      }
      let z_p = textureLoad(prev_depth, q, 0).r;
      let n_p = load_normal(prev_normal, q);
      if (is_consistent(z_c, z_p, n_c, n_p)) {
        let bw = bilinear_w[i];
        hist_c += finite_rgb(textureLoad(prev_color, q, 0).rgb) * bw;
        let pm = textureLoad(prev_moments, q, 0);
        hist_m += vec2f(finite_clamp(pm.x, 240.0), finite_clamp(pm.y, 57600.0)) * bw;
        hist_l += pm.z * bw;
        // Zero encodes sky or an invalid sample. Keep it out of surface distance history.
        if (adaptive_hit_dist && hit_dist > 0.0 && pm.w > 0.0) {
          hist_hit_dist += pm.w * bw;
          hit_dist_w += bw;
        }
        sum_w += bw;
      }
    }

    if (sum_w > 1e-4) {
      hist_c /= sum_w;
      hist_m /= sum_w;
      hist_l /= sum_w;

      // Specular mode: 3x3 YCoCg variance neighborhood clamping eliminates mirror ghosting
      // when reflected objects or dynamic lights move across reflective surfaces.
      if (spec_mode) {
        let yc_center = rgb_to_ycocg(max(sample_c, vec3f(0.0)));
        var yc_m1 = yc_center;
        var yc_m2 = yc_center * yc_center;
        var yc_w = 1.0;
        for (var dy = -1; dy <= 1; dy++) {
          for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) {
              continue;
            }
            let q = clamp(px + vec2i(dx, dy), vec2i(0), vec2i(w - 1, h - 1));
            let z_q = textureLoad(curr_depth, q, 0).r;
            let n_q = load_normal(curr_normal, q);
            if (is_consistent(z_c, z_q, n_c, n_q)) {
              let yc = rgb_to_ycocg(max(finite_rgb(textureLoad(curr_noisy, q, 0).rgb), vec3f(0.0)));
              yc_m1 += yc;
              yc_m2 += yc * yc;
              yc_w += 1.0;
            }
          }
        }
        let mu = yc_m1 / yc_w;
        let sigma = sqrt(max(yc_m2 / yc_w - mu * mu, vec3f(0.0)));
        let gamma = mix(1.15, 2.25, rough_c);
        let box_lo = mu - sigma * gamma;
        let box_hi = mu + sigma * gamma;
        let hist_yc = rgb_to_ycocg(max(hist_c, vec3f(0.0)));
        let clamped_yc = clamp(hist_yc, box_lo, box_hi);
        let clip_err = length(hist_yc - clamped_yc);
        hist_c = ycocg_to_rgb(clamped_yc);
        // Drop stale accumulated history count proportionally when history was heavily clipped.
        hist_l *= clamp(1.0 - clip_err * 2.5, 0.15, 1.0);
      }

      hist_len = min(hist_l + 1.0, max(u.temporal_params.y, 1.0));
      let min_a = select(
        u.temporal_params.x,
        mix(max(u.temporal_params.x, 0.14), u.temporal_params.x, rough_c),
        spec_mode,
      );
      let alpha = max(min_a, 1.0 / hist_len);
      acc_color = mix(hist_c, sample_c, alpha);
      acc_moments = mix(hist_m, moments_c, alpha);
      if (hit_dist_w > 1e-4) {
        filtered_hit_dist = mix(hist_hit_dist / hit_dist_w, hit_dist, alpha);
      }
    }
  }

  var variance = max(0.0, acc_moments.y - acc_moments.x * acc_moments.x);

  // Spatial 3x3 variance estimation for newly disoccluded pixels (< 4 frames of history).
  if (hist_len < 4.0) {
    let inv_z_denom = 1.0 / (max(u.atrous_params.z, 1e-4) * max(abs(z_c), 1e-3) + 1e-5);
    let phi_n = max(u.atrous_params.y, 1.0);
    var sw = 1.0;
    var sm = moments_c;
    var sc = sample_c;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        let q = clamp(px + vec2i(dx, dy), vec2i(0), vec2i(w - 1, h - 1));
        let n_q = load_normal(curr_normal, q);
        let ndot = dot(n_c, n_q);
        if (ndot <= 0.0) {
          continue;
        }
        let z_q = textureLoad(curr_depth, q, 0).r;
        let wz = exp(-abs(z_c - z_q) * inv_z_denom);
        let wn = pow(ndot, phi_n);
        let wt = wz * wn;
        let cq = finite_rgb(textureLoad(curr_noisy, q, 0).rgb);
        let lq = finite_clamp(luminance(cq), 240.0);
        sc += cq * wt;
        sm += vec2f(lq, lq * lq) * wt;
        sw += wt;
      }
    }
    if (sw > 1e-5) {
      sm /= sw;
      let spatial_var = max(0.0, sm.y - sm.x * sm.x);
      variance = max(variance, spatial_var * max(u.atrous_params.w, 1.0) / hist_len);
      if (hist_len <= 1.01) {
        let fallback_blend = select(0.35, mix(0.10, 0.35, rough_c), spec_mode);
        acc_color = mix(acc_color, sc / sw, fallback_blend);
      }
    }
  }

  textureStore(out_color, px, vec4f(finite_rgb(acc_color), finite_clamp(variance, 65504.0)));
  textureStore(out_moments, px, vec4f(
    finite_clamp(acc_moments.x, 240.0), finite_clamp(acc_moments.y, 57600.0),
    finite_clamp(hist_len, 65504.0), finite_clamp(filtered_hit_dist, 65504.0),
  ));
}

// A-Trous pass bindings (separate bind group layout for minimal bindings).
@group(0) @binding(1) var atrous_in: texture_2d<f32>;
@group(0) @binding(2) var atrous_depth: texture_2d<f32>;
@group(0) @binding(3) var atrous_normal: texture_2d<f32>;
@group(0) @binding(4) var atrous_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var hit_dist_tex: texture_2d<f32>;

// Shared 10x10 tile (8x8 + 1px border) for 3x3 variance pre-filter and 4-neighbor depth gradient.
var<workgroup> wg_var_z: array<vec2f, 100>;

@compute @workgroup_size(8, 8, 1)
fn svgf_atrous(
  @builtin(global_invocation_id) id: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
) {
  let w = i32(u.dims.x);
  let h = i32(u.dims.y);
  let max_coord = vec2i(max(w - 1, 0), max(h - 1, 0));
  let tile_origin = vec2i(wid.xy * 8u) - vec2i(1);
  let lidx = lid.y * 8u + lid.x;
  for (var slot = lidx; slot < 100u; slot += 64u) {
    let lx = i32(slot % 10u);
    let ly = i32(slot / 10u);
    let gp = clamp(tile_origin + vec2i(lx, ly), vec2i(0), max_coord);
    wg_var_z[slot] = vec2f(textureLoad(atrous_in, gp, 0).a, textureLoad(atrous_depth, gp, 0).r);
  }
  workgroupBarrier();

  let px = vec2i(id.xy);
  if (px.x >= w || px.y >= h) {
    return;
  }

  let spec_mode = (u.dims.w & 2u) != 0u;
  let step_size = max(i32(u.dims.z), 1);
  let step_f = f32(step_size);
  let center = textureLoad(atrous_in, px, 0);
  let c_p = center.rgb;
  let l_p = luminance(c_p);
  let tile_cx = lid.x + 1u;
  let tile_cy = lid.y + 1u;
  let z_p = wg_var_z[tile_cy * 10u + tile_cx].y;
  let n_raw_p = textureLoad(atrous_normal, px, 0);
  let n_l2_p = dot(n_raw_p.xyz, n_raw_p.xyz);
  let n_p = select(vec3f(0.0, 1.0, 0.0), n_raw_p.xyz * inverseSqrt(n_l2_p), n_l2_p > 1e-8);
  let rough_p = clamp(n_raw_p.w, 0.04, 1.0);

  // 3x3 Gaussian variance pre-filter from shared tile (SVGF Section 4.3).
  let r0 = (tile_cy - 1u) * 10u + tile_cx;
  let r1 = tile_cy * 10u + tile_cx;
  let r2 = (tile_cy + 1u) * 10u + tile_cx;
  let filtered_var =
    0.0625 * (wg_var_z[r0 - 1u].x + wg_var_z[r0 + 1u].x + wg_var_z[r2 - 1u].x + wg_var_z[r2 + 1u].x) +
    0.125 * (wg_var_z[r0].x + wg_var_z[r1 - 1u].x + wg_var_z[r1 + 1u].x + wg_var_z[r2].x) +
    0.25 * wg_var_z[r1].x;

  // Depth gradient estimate from shared tile immediate screen neighbors.
  let z_rx = wg_var_z[r1 + 1u].y;
  let z_lx = wg_var_z[r1 - 1u].y;
  let z_dy = wg_var_z[r2].y;
  let z_uy = wg_var_z[r0].y;
  let grad_z = max(max(abs(z_rx - z_p), abs(z_p - z_lx)), max(abs(z_dy - z_p), abs(z_p - z_uy)));

  // Roughness-scaled specular edge stopping: keeps mirror reflections sharp while smoothing glossy lobes.
  let rough_l_scale = select(1.0, mix(0.40, 1.0, smoothstep(0.04, 0.45, rough_p)), spec_mode);
  let rough_n_scale = select(1.0, mix(2.5, 1.0, smoothstep(0.05, 0.50, rough_p)), spec_mode);
  var contact = 0.0;
  if ((u.dims.w & 4u) != 0u) {
    let hit_dist = textureLoad(hit_dist_tex, px, 0).w;
    if (hit_dist > 0.0) {
      let cone_footprint = rough_p * rough_p * clamp(hit_dist, 0.02, 20.0);
      contact = 1.0 - smoothstep(0.01, 0.8, cone_footprint);
    }
  }
  let sigma_l = max(u.atrous_params.x, 0.05) * rough_l_scale * mix(1.0, 0.45, contact)
      * sqrt(max(filtered_var, 0.0)) + 1e-4;
  let inv_sigma_l = 1.0 / sigma_l;
  let phi_n = max(u.atrous_params.y, 1.0) * rough_n_scale;
  let phi_z = max(u.atrous_params.z, 1e-4);
  let z_floor = max(abs(z_p) * 1e-3, 1e-5);
  let mirror_taper = select(0.0, (1.0 - rough_p) * (1.0 - rough_p) * 0.45, spec_mode) + contact * 0.65;
  let phi_rough = mix(10.0, 24.0, contact);

  // 5x5 B3-spline wavelet weights; seed with center tap (0.375 * 0.375 = 0.140625) to skip center reload.
  let k5 = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);

  var sum_w = 0.140625;
  var sum_c = c_p * 0.140625;
  var sum_v = center.a * 0.019775390625;

  for (var ky = -2; ky <= 2; ky++) {
    let qy = px.y + ky * step_size;
    if (qy < 0 || qy >= h) {
      continue;
    }
    let kwy = k5[ky + 2];
    for (var kx = -2; kx <= 2; kx++) {
      if (kx == 0 && ky == 0) {
        continue;
      }
      let qx = px.x + kx * step_size;
      if (qx < 0 || qx >= w) {
        continue;
      }
      let q = vec2i(qx, qy);
      let n_raw_q = textureLoad(atrous_normal, q, 0);
      let n_l2_q = dot(n_raw_q.xyz, n_raw_q.xyz);
      let n_q = select(vec3f(0.0, 1.0, 0.0), n_raw_q.xyz * inverseSqrt(n_l2_q), n_l2_q > 1e-8);
      let ndot = dot(n_p, n_q);
      if (ndot <= 0.0) {
        continue;
      }
      let z_q = textureLoad(atrous_depth, q, 0).r;
      let sample_q = textureLoad(atrous_in, q, 0);
      let l_q = luminance(sample_q.rgb);

      let dist_steps = sqrt(f32(kx * kx + ky * ky)) * step_f;
      let denom_z = phi_z * max(grad_z * dist_steps, z_floor);
      var expo = -abs(z_p - z_q) / denom_z - abs(l_p - l_q) * inv_sigma_l;
      if (spec_mode) {
        let rough_q = clamp(n_raw_q.w, 0.04, 1.0);
        expo -= abs(rough_p - rough_q) * phi_rough + dist_steps * mirror_taper;
      }
      let wt = k5[kx + 2] * kwy * pow(ndot, phi_n) * exp(expo);
      sum_c += sample_q.rgb * wt;
      sum_v += sample_q.a * wt * wt;
      sum_w += wt;
    }
  }

  let inv_w = select(1.0, 1.0 / sum_w, sum_w > 1e-6);
  let out_rgb = select(c_p, sum_c * inv_w, sum_w > 1e-6);
  let out_var = select(center.a, sum_v * inv_w * inv_w, sum_w > 1e-6);
  textureStore(atrous_out, px, vec4f(out_rgb, out_var));
}
