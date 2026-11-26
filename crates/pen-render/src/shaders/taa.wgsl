// FSR 2.2 / 3.1-class Portable Temporal Super Resolution & TAA.
// Features:
// - Ratio-scaled Halton(2, 3) jitter sequence (`ceil(8 * ratio^2)` phases, up to 72)
// - Exposure-invariant pre-exposure before Karis tone compression & YCoCg variance clipping
// - Forward-Z & Reversed-Z 3x3 nearest-depth velocity dilation + motion-divergence detection
// - 3x3 dilated auto-reactive mask from opaque backdrop vs. translucent/particle/ribbon delta
// - Geometric previous-depth disocclusion via 2x2 footprint check against `prev_lock.g`
// - Dedicated thin-feature lock status (`lock_out` MRT) with immediate release on disocclusion/luma shift
// - 3x3 Lanczos-2 rational sub-pixel reconstruction with anti-ringing clamp in upscale mode
// - 5-tap Catmull-Rom history resampling + sample-count convergence up to `phase_count` (72) in `history.a`
struct Taa {
  // This frame's jittered clip space back into the world.
  inverse: mat4x4f,
  // The world into the last frame's clip space, unjittered.
  previous: mat4x4f,
  // xy: this frame's jitter in texture coordinates, across and down; z: its
  // baseline share of the blend; w: 1 when there is no history yet.
  blend: vec4f,
  // xy: output history size in pixels; zw: rendered scene size in pixels.
  size: vec4f,
  // x: reversed_z (>0.5), y: jitter phase count, z: has_opaque_reactive (>0.5), w: depth_clear.
  params: vec4f,
};
struct TaaOut {
  @location(0) color: vec4f,
  // r: thin-feature lock strength [0..1], g: NDC depth [0..1], b: pre-exposed Karis luma, a: valid (>0.5).
  @location(1) lock: vec4f,
};
@group(0) @binding(0) var<uniform> taa: Taa;
@group(0) @binding(1) var current: texture_2d<f32>;
@group(0) @binding(2) var history: texture_2d<f32>;
@group(0) @binding(3) var depth: ${depth_texture};
@group(0) @binding(4) var filtering: sampler;
@group(0) @binding(5) var velocity: texture_2d<f32>;
@group(0) @binding(6) var opaque_color: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> exposure: array<f32, 4>;
@group(0) @binding(8) var prev_lock: texture_2d<f32>;
${fullscreen}

fn luma(color: vec3f) -> f32 {
  return dot(max(color, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722));
}

// Light squeezed below 1 (and back), so a blazing highlight weighs no more
// in a blend than a white one and does not flicker.
fn squeezed(color: vec3f) -> vec3f {
  let c = max(color, vec3f(0.0));
  return c / (1.0 + luma(c));
}
fn unsqueezed(color: vec3f) -> vec3f {
  let c = max(color, vec3f(0.0));
  return c / max(1.0 - luma(c), 0.0001);
}

// Luma and two chroma axes, where a neighbourhood's colours make a tighter box.
fn ycocg(color: vec3f) -> vec3f {
  return vec3f(dot(color, vec3f(0.25, 0.5, 0.25)), dot(color, vec3f(0.5, 0.0, -0.5)), dot(color, vec3f(-0.25, 0.5, -0.25)));
}
fn rgb(color: vec3f) -> vec3f {
  return vec3f(color.x + color.y - color.z, color.x + color.z, color.x - color.y - color.z);
}

// Exact AMD FSR 2 rational Lanczos-2 window approximation (zero trig, sharp sub-pixel energy).
fn lanczos2_approx(x: f32) -> f32 {
  let x2 = min(x * x, 4.0);
  let a = 0.4 * x2 - 1.0;
  let b = 0.25 * x2 - 1.0;
  return (1.5625 * a * a - 0.5625) * (b * b);
}

// Scale-invariant geometric depth disocclusion test against the current 3x3 [lo_d, hi_d] depth
// bracket (works for both Forward-Z and Reversed-Z). Checking against the 3x3 bracket preserves
// full temporal AA on static jittered silhouette edges while immediately firing on true disocclusions.
fn depth_disocc_bracket(prev_d: f32, lo_d: f32, hi_d: f32, rev_z: bool, tol_scale: f32) -> f32 {
  let m_prev = select(1.0 - prev_d, prev_d, rev_z);
  let m_a = select(1.0 - lo_d, lo_d, rev_z);
  let m_b = select(1.0 - hi_d, hi_d, rev_z);
  let m_min = min(m_a, m_b);
  let m_max = max(m_a, m_b);
  let clamped = clamp(m_prev, m_min, m_max);
  let diff = abs(m_prev - clamped);
  let tol = max(tol_scale * max(m_prev, m_max), 0.004);
  return smoothstep(tol, tol * 3.0, diff);
}

// The history RGB at uv through a 5-tap Catmull-Rom filter plus accumulated confidence in .a.
fn history_at_rgba(uv: vec2f) -> vec4f {
  let size = taa.size.xy;
  let position = uv * size;
  let center = floor(position - 0.5) + 0.5;
  let f = position - center;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = -0.5 * f3 + f2 - 0.5 * f;
  let w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
  let w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
  let w3 = 0.5 * f3 - 0.5 * f2;
  let w12 = w1 + w2;
  let inv_size = 1.0 / size;
  let half_px = vec2f(0.5) * inv_size;
  let max_uv = vec2f(1.0) - half_px;
  let at0 = clamp((center - 1.0) * inv_size, half_px, max_uv);
  let at3 = clamp((center + 2.0) * inv_size, half_px, max_uv);
  let at12 = clamp((center + w2 / max(w12, vec2f(1e-5))) * inv_size, half_px, max_uv);
  let sum = textureSampleLevel(history, filtering, vec2f(at12.x, at0.y), 0.0).rgb * (w12.x * w0.y)
    + textureSampleLevel(history, filtering, vec2f(at0.x, at12.y), 0.0).rgb * (w0.x * w12.y)
    + textureSampleLevel(history, filtering, at12, 0.0).rgb * (w12.x * w12.y)
    + textureSampleLevel(history, filtering, vec2f(at3.x, at12.y), 0.0).rgb * (w3.x * w12.y)
    + textureSampleLevel(history, filtering, vec2f(at12.x, at3.y), 0.0).rgb * (w12.x * w3.y);
  let weight = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  let accum = textureSampleLevel(history, filtering, clamp(uv, half_px, max_uv), 0.0).a;
  return vec4f(max(sum / max(weight, 1e-5), vec3f(0.0)), clamp(accum, 0.0, 1.0));
}

// The history pulled toward the middle of the box until it is inside it.
fn clipped(color: vec3f, low: vec3f, high: vec3f) -> vec3f {
  let middle = 0.5 * (high + low);
  let reach = 0.5 * (high - low) + vec3f(0.0001);
  let offset = color - middle;
  let units = abs(offset / reach);
  let most = max(units.x, max(units.y, units.z));
  return select(color, middle + offset / most, most > 1.0);
}

@fragment fn fragment(@builtin(position) position: vec4f) -> TaaOut {
  let out_size = taa.size.xy;
  let scene_size = select(out_size, taa.size.zw, taa.size.z > 0.5 && taa.size.w > 0.5);
  let is_upscale = any(abs(out_size - scene_size) > vec2f(0.5));
  let rev_z = taa.params.x > 0.5;
  let phase_count = max(taa.params.y, 8.0);
  let has_reactive = taa.params.z > 0.5;
  let uv = position.xy / out_size;
  let sample_uv = clamp(uv + taa.blend.xy, vec2f(0.0), vec2f(1.0));
  let scene_last = vec2i(scene_size) - vec2i(1);
  let in_pos = sample_uv * scene_size;
  let pixel = select(vec2i(position.xy), clamp(vec2i(floor(in_pos)), vec2i(0), scene_last), is_upscale);

  // FSR 2.2/3.1 Pre-exposure: normalize scene & history radiance into display-scale brightness
  // before Karis tone compression and YCoCg variance clipping, then undo after blending.
  let exp_stops = clamp(exposure[0], -10.0, 10.0);
  let pre_exp = clamp(exp2(exp_stops), 1.0 / 1024.0, 1024.0);
  let inv_pre_exp = 1.0 / pre_exp;

  // 3x3 neighbourhood statistics, nearest-depth velocity dilation (Forward-Z & Reversed-Z),
  // 3x3 dilated auto-reactive mask, motion divergence check, and Lanczos-2 sub-pixel accumulation.
  var low = vec3f(1e9);
  var high = vec3f(-1e9);
  var mean = vec3f(0.0);
  var square = vec3f(0.0);
  var middle = vec3f(0.0);
  var lanczos_yc = vec3f(0.0);
  var lanczos_w = 0.0;
  let center_depth = textureLoad(depth, pixel, 0);
  var nearest = center_depth;
  var nearest_at = pixel;
  var farthest = nearest;
  var vel_min = vec2f(1e9);
  var vel_max = vec2f(-1e9);
  var valid_vels = 0u;
  var reactive = 0.0;

  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let at = clamp(pixel + vec2i(x, y), vec2i(0), scene_last);
      let raw_pre = max(textureLoad(current, at, 0).rgb, vec3f(0.0)) * pre_exp;
      let color = ycocg(squeezed(raw_pre));
      if (x == 0 && y == 0) {
        middle = color;
      }
      low = min(low, color);
      high = max(high, color);
      mean += color;
      square += color * color;

      if (has_reactive) {
        let opq_pre = max(textureLoad(opaque_color, at, 0).rgb, vec3f(0.0)) * pre_exp;
        let diff = abs(raw_pre - opq_pre);
        let delta_max = max(diff.r, max(diff.g, diff.b));
        let r_tap = clamp((delta_max - 0.04) / max(0.08 + luma(raw_pre) * 0.45, 0.14), 0.0, 1.0);
        let r_weight = select(0.85, 1.0, (x == 0 || y == 0));
        reactive = max(reactive, r_tap * r_weight);
      }

      if (is_upscale) {
        let delta = (vec2f(at) + vec2f(0.5)) - in_pos;
        let w = lanczos2_approx(delta.x) * lanczos2_approx(delta.y);
        lanczos_yc += color * w;
        lanczos_w += w;
      }

      let d = select(textureLoad(depth, at, 0), center_depth, x == 0 && y == 0);
      let is_nearer = select((d < nearest), (d > nearest), rev_z);
      if (is_nearer) {
        nearest = d;
        nearest_at = at;
      }
      let is_farther = select((d > farthest), (d < farthest), rev_z);
      if (is_farther) {
        farthest = d;
      }

      let v = textureLoad(velocity, at, 0).xy;
      if (abs(v.x) < 500.0) {
        vel_min = min(vel_min, v);
        vel_max = max(vel_max, v);
        valid_vels += 1u;
      }
    }
  }
  if (is_upscale && abs(lanczos_w) > 1e-4) {
    middle = clamp(lanczos_yc / lanczos_w, low, high);
  } else if (is_upscale) {
    let fallback_pre = max(textureSampleLevel(current, filtering, sample_uv, 0.0).rgb, vec3f(0.0)) * pre_exp;
    middle = clamp(ycocg(squeezed(fallback_pre)), low, high);
  }
  mean /= 9.0;

  // Motion-vector divergence across 3x3 neighborhood signals disoccluded moving silhouettes.
  var vel_div = 0.0;
  if (valid_vels >= 2u) {
    let span_px = length((vel_max - vel_min) * out_size);
    vel_div = clamp((span_px - 1.0) * 0.25, 0.0, 1.0);
  }

  // Reproject via dilated nearest-depth object velocity (rigid/skinned/morphed) or analytical depth.
  let obj_vel = textureLoad(velocity, nearest_at, 0).xy;
  var was = uv;
  var gone = false;
  var expected_near = nearest;
  var expected_far = farthest;
  var depth_tol_scale = 0.02;
  if (abs(obj_vel.x) < 500.0) {
    was = uv - obj_vel;
    gone = any(was < vec2f(0.0)) || any(was > vec2f(1.0));
    depth_tol_scale = 0.06;
  } else {
    let reproj_uv = select(uv, sample_uv, is_upscale);
    let world = taa.inverse * vec4f(reproj_uv.x * 2.0 - 1.0, 1.0 - reproj_uv.y * 2.0, nearest, 1.0);
    let before = taa.previous * vec4f(world.xyz / world.w, 1.0);
    was = vec2f(before.x, -before.y) / before.w * 0.5 + 0.5 + select(taa.blend.xy, vec2f(0.0), is_upscale);
    gone = before.w <= 0.0 || any(was < vec2f(0.0)) || any(was > vec2f(1.0));
    expected_near = clamp(before.z / max(before.w, 1e-6), 0.0, 1.0);
    let world_f = taa.inverse * vec4f(reproj_uv.x * 2.0 - 1.0, 1.0 - reproj_uv.y * 2.0, farthest, 1.0);
    let before_f = taa.previous * vec4f(world_f.xyz / world_f.w, 1.0);
    expected_far = select(farthest, clamp(before_f.z / max(before_f.w, 1e-6), 0.0, 1.0), before_f.w > 1e-6);
  }

  if (taa.blend.w > 0.5 || gone) {
    return TaaOut(
      vec4f(unsqueezed(rgb(middle)) * inv_pre_exp, 1.0 / phase_count),
      vec4f(0.0, center_depth, middle.x, 1.0),
    );
  }

  // Geometric previous-depth disocclusion & previous lock status via 2x2 footprint load.
  let out_last = vec2i(out_size) - vec2i(1);
  let prev_f = was * out_size - vec2f(0.5);
  let p00 = clamp(vec2i(floor(prev_f)), vec2i(0), out_last);
  let p10 = clamp(p00 + vec2i(1, 0), vec2i(0), out_last);
  let p01 = clamp(p00 + vec2i(0, 1), vec2i(0), out_last);
  let p11 = clamp(p00 + vec2i(1, 1), vec2i(0), out_last);
  let l00 = textureLoad(prev_lock, p00, 0);
  let l10 = textureLoad(prev_lock, p10, 0);
  let l01 = textureLoad(prev_lock, p01, 0);
  let l11 = textureLoad(prev_lock, p11, 0);
  let lf = clamp(prev_f - vec2f(p00), vec2f(0.0), vec2f(1.0));
  let prev_lock_val = mix(mix(l00, l10, lf.x), mix(l01, l11, lf.x), lf.y);

  var depth_disocc = 0.0;
  if (l00.a > 0.5 && l10.a > 0.5 && l01.a > 0.5 && l11.a > 0.5) {
    let d00 = depth_disocc_bracket(l00.g, expected_near, expected_far, rev_z, depth_tol_scale);
    let d10 = depth_disocc_bracket(l10.g, expected_near, expected_far, rev_z, depth_tol_scale);
    let d01 = depth_disocc_bracket(l01.g, expected_near, expected_far, rev_z, depth_tol_scale);
    let d11 = depth_disocc_bracket(l11.g, expected_near, expected_far, rev_z, depth_tol_scale);
    depth_disocc = min(min(d00, d10), min(d01, d11));
  }

  // Tighten variance box around reactive particles/glass, diverging silhouettes, and depth disocclusions.
  let gamma = mix(1.25, 0.80, max(max(reactive, vel_div), depth_disocc));
  let spread = sqrt(max(square / 9.0 - mean * mean, vec3f(0.0))) * gamma;

  let hist_rgba = history_at_rgba(was);
  let hist_yc_raw = ycocg(squeezed(hist_rgba.rgb * pre_exp));
  let clip_lo = max(low, mean - spread);
  let clip_hi = min(high, mean + spread);
  let remembered = clipped(hist_yc_raw, clip_lo, clip_hi);

  // Luma-stability, clip-distance, and geometric depth disocclusion detection.
  let clip_dist = length(hist_yc_raw - remembered);
  let luma_instability = clamp(abs(hist_yc_raw.x - middle.x) / max(max(hist_yc_raw.x, middle.x), 0.12), 0.0, 1.0);
  let disocc = clamp(max(depth_disocc, clip_dist * 4.0 + vel_div * 0.5), 0.0, 1.0);

  // Dedicated thin-feature lock status: locks high-contrast static micro-edges to stop sub-pixel
  // shimmer, and releases immediately on disocclusion, reactive VFX, motion divergence, or light jumps.
  let local_contrast = high.x - low.x;
  let prev_lock_str = select(0.0, clamp(prev_lock_val.r, 0.0, 1.0), prev_lock_val.a > 0.5);
  let lock_luma_jump = select(0.0, abs(prev_lock_val.b - middle.x), prev_lock_val.a > 0.5);
  let lock_break = max(max(depth_disocc, reactive), max(vel_div, smoothstep(0.08, 0.22, lock_luma_jump)));
  let eff_lock = prev_lock_str * (1.0 - lock_break);
  let thin_feature = smoothstep(0.10, 0.30, local_contrast)
    * (1.0 - reactive)
    * (1.0 - depth_disocc)
    * (1.0 - vel_div)
    * (1.0 - luma_instability);
  let next_lock = clamp(eff_lock * 0.92 + thin_feature * 0.28, 0.0, 1.0) * (1.0 - lock_break);

  let prev_accum = hist_rgba.a * phase_count * (1.0 - 0.85 * max(disocc, reactive));
  let next_accum = min(prev_accum + 1.0, phase_count);
  var alpha = taa.blend.z;
  if (is_upscale) {
    // Sub-pixel jitter proximity weighting + FSR-style sample-count convergence curve.
    let center_delta = (vec2f(pixel) + vec2f(0.5)) - in_pos;
    let prox = exp(-2.8 * dot(center_delta, center_delta));
    let steady = clamp((1.0 / phase_count) * (0.65 + 1.35 * prox), 1.0 / 72.0, 0.45);
    alpha = max(1.0 / max(next_accum, 1.0), steady);
    alpha = mix(alpha, 0.65, luma_instability * 0.45);
  } else if (reactive > 0.01 || vel_div > 0.01 || depth_disocc > 0.01) {
    alpha = max(alpha, 1.0 / max(next_accum, 1.0));
  }
  // Locked static thin features hold temporal history tighter; depth disocclusions drop stale history fast.
  alpha = mix(alpha, max(1.0 / phase_count, 0.02), eff_lock * 0.55);
  alpha = max(alpha, depth_disocc * 0.90);
  alpha = clamp(mix(alpha, 0.85, reactive * 0.82), 1.0 / 72.0, 0.95);

  let out_rgb = unsqueezed(rgb(mix(remembered, middle, alpha))) * inv_pre_exp;
  return TaaOut(
    vec4f(out_rgb, clamp(next_accum / phase_count, 1.0 / phase_count, 1.0)),
    vec4f(next_lock, center_depth, middle.x, 1.0),
  );
}
