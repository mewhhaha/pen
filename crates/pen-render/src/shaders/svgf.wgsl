// Spatiotemporal Variance-Guided Filter (SVGF, Schied et al., HPG 2017).
// Pass 1 (svgf_temporal): motion-reprojected temporal accumulation + 1st/2nd luminance moments
//                         + spatial fallback variance initialization on disocclusion.
// Pass 2 (svgf_atrous):   multi-scale 5x5 B3-spline A-Trous edge-stopping wavelet filter
//                         guided by depth, normal, and 3x3-filtered luminance variance.

struct SvgfUniform {
  dims: vec4u,           // x = width, y = height, z = step_size, w = reset_history (0 or 1)
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

  let sample_c = textureLoad(curr_noisy, px, 0).rgb;
  let z_c = textureLoad(curr_depth, px, 0).r;
  let n_c = load_normal(curr_normal, px);
  let lum_c = luminance(sample_c);
  let moments_c = vec2f(lum_c, lum_c * lum_c);

  var acc_color = sample_c;
  var acc_moments = moments_c;
  var hist_len = 1.0;

  if (u.dims.w == 0u) {
    let mv = textureLoad(motion_tex, px, 0).xy;
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

    for (var i = 0; i < 4; i++) {
      let q = base + offsets[i];
      if (q.x < 0 || q.y < 0 || q.x >= w || q.y >= h) {
        continue;
      }
      let z_p = textureLoad(prev_depth, q, 0).r;
      let n_p = load_normal(prev_normal, q);
      if (is_consistent(z_c, z_p, n_c, n_p)) {
        let bw = bilinear_w[i];
        hist_c += textureLoad(prev_color, q, 0).rgb * bw;
        let pm = textureLoad(prev_moments, q, 0);
        hist_m += pm.xy * bw;
        hist_l += pm.z * bw;
        sum_w += bw;
      }
    }

    if (sum_w > 1e-4) {
      hist_c /= sum_w;
      hist_m /= sum_w;
      hist_l /= sum_w;
      hist_len = min(hist_l + 1.0, max(u.temporal_params.y, 1.0));
      let alpha = max(u.temporal_params.x, 1.0 / hist_len);
      acc_color = mix(hist_c, sample_c, alpha);
      acc_moments = mix(hist_m, moments_c, alpha);
    }
  }

  var variance = max(0.0, acc_moments.y - acc_moments.x * acc_moments.x);

  // Spatial 3x3 variance estimation for newly disoccluded pixels (< 4 frames of history).
  if (hist_len < 4.0) {
    var sw = 0.0;
    var sm = vec2f(0.0);
    var sc = vec3f(0.0);
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let q = clamp(px + vec2i(dx, dy), vec2i(0), vec2i(w - 1, h - 1));
        let z_q = textureLoad(curr_depth, q, 0).r;
        let n_q = load_normal(curr_normal, q);
        let wz = exp(-abs(z_c - z_q) / (max(u.atrous_params.z, 1e-4) * max(abs(z_c), 1e-3) + 1e-5));
        let wn = pow(max(0.0, dot(n_c, n_q)), max(u.atrous_params.y, 1.0));
        let wt = wz * wn;
        let cq = textureLoad(curr_noisy, q, 0).rgb;
        let lq = luminance(cq);
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
        acc_color = mix(acc_color, sc / sw, 0.35);
      }
    }
  }

  textureStore(out_color, px, vec4f(acc_color, variance));
  textureStore(out_moments, px, vec4f(acc_moments.x, acc_moments.y, hist_len, 0.0));
}

// A-Trous pass bindings (separate bind group layout for minimal bindings).
@group(0) @binding(1) var atrous_in: texture_2d<f32>;
@group(0) @binding(2) var atrous_depth: texture_2d<f32>;
@group(0) @binding(3) var atrous_normal: texture_2d<f32>;
@group(0) @binding(4) var atrous_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn svgf_atrous(@builtin(global_invocation_id) id: vec3u) {
  let w = i32(u.dims.x);
  let h = i32(u.dims.y);
  let px = vec2i(id.xy);
  if (px.x >= w || px.y >= h) {
    return;
  }

  let step_size = max(i32(u.dims.z), 1);
  let center = textureLoad(atrous_in, px, 0);
  let c_p = center.rgb;
  let l_p = luminance(c_p);
  let z_p = textureLoad(atrous_depth, px, 0).r;
  let n_p = load_normal(atrous_normal, px);

  // 3x3 Gaussian variance pre-filter (SVGF Section 4.3).
  let g3 = array<f32, 3>(0.25, 0.5, 0.25);
  var filtered_var = 0.0;
  for (var gy = -1; gy <= 1; gy++) {
    for (var gx = -1; gx <= 1; gx++) {
      let q = clamp(px + vec2i(gx, gy), vec2i(0), vec2i(w - 1, h - 1));
      filtered_var += textureLoad(atrous_in, q, 0).a * g3[gx + 1] * g3[gy + 1];
    }
  }

  // Depth gradient estimate from immediate screen neighbors.
  let z_rx = textureLoad(atrous_depth, clamp(px + vec2i(1, 0), vec2i(0), vec2i(w - 1, h - 1)), 0).r;
  let z_lx = textureLoad(atrous_depth, clamp(px - vec2i(1, 0), vec2i(0), vec2i(w - 1, h - 1)), 0).r;
  let z_dy = textureLoad(atrous_depth, clamp(px + vec2i(0, 1), vec2i(0), vec2i(w - 1, h - 1)), 0).r;
  let z_uy = textureLoad(atrous_depth, clamp(px - vec2i(0, 1), vec2i(0), vec2i(w - 1, h - 1)), 0).r;
  let grad_z = max(max(abs(z_rx - z_p), abs(z_p - z_lx)), max(abs(z_dy - z_p), abs(z_p - z_uy)));

  let sigma_l = max(u.atrous_params.x, 0.05) * sqrt(max(filtered_var, 0.0)) + 1e-4;
  let phi_n = max(u.atrous_params.y, 1.0);
  let phi_z = max(u.atrous_params.z, 1e-4);

  // 5x5 B3-spline wavelet weights.
  let k5 = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);

  var sum_w = 0.0;
  var sum_c = vec3f(0.0);
  var sum_v = 0.0;

  for (var ky = -2; ky <= 2; ky++) {
    for (var kx = -2; kx <= 2; kx++) {
      let q = px + vec2i(kx * step_size, ky * step_size);
      if (q.x < 0 || q.y < 0 || q.x >= w || q.y >= h) {
        continue;
      }
      let kw = k5[kx + 2] * k5[ky + 2];
      let sample_q = textureLoad(atrous_in, q, 0);
      var wt = kw;
      if (kx != 0 || ky != 0) {
        let z_q = textureLoad(atrous_depth, q, 0).r;
        let n_q = load_normal(atrous_normal, q);
        let l_q = luminance(sample_q.rgb);

        let dist_steps = length(vec2f(f32(kx * step_size), f32(ky * step_size)));
        let denom_z = phi_z * max(grad_z * dist_steps, max(abs(z_p) * 1e-3, 1e-5));
        let wz = exp(-abs(z_p - z_q) / denom_z);
        let wn = pow(max(0.0, dot(n_p, n_q)), phi_n);
        let wl = exp(-abs(l_p - l_q) / sigma_l);
        wt = kw * wz * wn * wl;
      }
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
