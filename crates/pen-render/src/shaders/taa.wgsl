// FSR 2.2 / 3.1-class Portable Temporal Super Resolution & TAA.
// Features:
// - Ratio-scaled Halton(2, 3) jitter sequence (`ceil(8 * ratio^2)` phases)
// - Forward-Z & Reversed-Z 3x3 nearest-depth velocity dilation + motion-divergence detection
// - Auto-reactive mask from opaque backdrop vs. translucent/particle/ribbon delta (`opaque_color`)
// - 3x3 Lanczos-2 rational sub-pixel reconstruction with anti-ringing clamp in upscale mode
// - 5-tap Catmull-Rom history resampling + Karis tone-compressed YCoCg variance clipping
// - Per-pixel luma-stability locking & sample-count convergence packed in `history.a`
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
@group(0) @binding(0) var<uniform> taa: Taa;
@group(0) @binding(1) var current: texture_2d<f32>;
@group(0) @binding(2) var history: texture_2d<f32>;
@group(0) @binding(3) var depth: ${depth_texture};
@group(0) @binding(4) var filtering: sampler;
@group(0) @binding(5) var velocity: texture_2d<f32>;
@group(0) @binding(6) var opaque_color: texture_2d<f32>;
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
  let half_px = vec2f(0.5) / size;
  let max_uv = vec2f(1.0) - half_px;
  let at0 = clamp((center - 1.0) / size, half_px, max_uv);
  let at3 = clamp((center + 2.0) / size, half_px, max_uv);
  let at12 = clamp((center + w2 / max(w12, vec2f(1e-5))) / size, half_px, max_uv);
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

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
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

  // 3x3 neighbourhood statistics, nearest-depth velocity dilation (Forward-Z & Reversed-Z),
  // motion divergence check, and Lanczos-2 sub-pixel accumulation for FSR-class upscaling.
  var low = vec3f(1e9);
  var high = vec3f(-1e9);
  var mean = vec3f(0.0);
  var square = vec3f(0.0);
  var middle = vec3f(0.0);
  var middle_raw = vec3f(0.0);
  var lanczos_yc = vec3f(0.0);
  var lanczos_w = 0.0;
  var nearest = textureLoad(depth, pixel, 0);
  var nearest_at = pixel;
  var farthest = nearest;
  var vel_min = vec2f(1e9);
  var vel_max = vec2f(-1e9);
  var valid_vels = 0u;

  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let at = clamp(pixel + vec2i(x, y), vec2i(0), scene_last);
      let raw = max(textureLoad(current, at, 0).rgb, vec3f(0.0));
      let color = ycocg(squeezed(raw));
      if (x == 0 && y == 0) {
        middle = color;
        middle_raw = raw;
      }
      low = min(low, color);
      high = max(high, color);
      mean += color;
      square += color * color;

      if (is_upscale) {
        let delta = (vec2f(at) + vec2f(0.5)) - in_pos;
        let w = lanczos2_approx(delta.x) * lanczos2_approx(delta.y);
        lanczos_yc += color * w;
        lanczos_w += w;
      }

      let d = textureLoad(depth, at, 0);
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
    middle = clamp(ycocg(squeezed(textureSampleLevel(current, filtering, sample_uv, 0.0).rgb)), low, high);
  }
  mean /= 9.0;

  // Auto-reactive mask: delta between translucent/particle-composited frame and opaque backdrop.
  var reactive = 0.0;
  if (has_reactive) {
    let opq = max(textureLoad(opaque_color, pixel, 0).rgb, vec3f(0.0));
    let diff = abs(middle_raw - opq);
    let delta_max = max(diff.r, max(diff.g, diff.b));
    reactive = clamp((delta_max - 0.04) / max(0.08 + luma(middle_raw) * 0.45, 0.14), 0.0, 1.0);
  }

  // Motion-vector divergence across 3x3 neighborhood signals disoccluded moving silhouettes.
  var vel_div = 0.0;
  if (valid_vels >= 2u) {
    let span_px = length((vel_max - vel_min) * out_size);
    vel_div = clamp((span_px - 1.0) * 0.25, 0.0, 1.0);
  }

  // Tighten variance box around reactive particles/glass and diverging moving silhouettes.
  let gamma = mix(1.25, 0.85, max(reactive, vel_div));
  let spread = sqrt(max(square / 9.0 - mean * mean, vec3f(0.0))) * gamma;

  // Reproject via dilated nearest-depth object velocity (rigid/skinned/morphed) or analytical depth.
  let obj_vel = textureLoad(velocity, nearest_at, 0).xy;
  var was = uv;
  var gone = false;
  if (abs(obj_vel.x) < 500.0) {
    was = uv - obj_vel;
    gone = any(was < vec2f(0.0)) || any(was > vec2f(1.0));
  } else {
    let reproj_uv = select(uv, sample_uv, is_upscale);
    let world = taa.inverse * vec4f(reproj_uv.x * 2.0 - 1.0, 1.0 - reproj_uv.y * 2.0, nearest, 1.0);
    let before = taa.previous * vec4f(world.xyz / world.w, 1.0);
    was = vec2f(before.x, -before.y) / before.w * 0.5 + 0.5 + select(taa.blend.xy, vec2f(0.0), is_upscale);
    gone = before.w <= 0.0 || any(was < vec2f(0.0)) || any(was > vec2f(1.0));
  }

  if (taa.blend.w > 0.5 || gone) {
    return vec4f(unsqueezed(rgb(middle)), 1.0 / 32.0);
  }

  let hist_rgba = history_at_rgba(was);
  let hist_yc_raw = ycocg(squeezed(hist_rgba.rgb));
  let clip_lo = max(low, mean - spread);
  let clip_hi = min(high, mean + spread);
  let remembered = clipped(hist_yc_raw, clip_lo, clip_hi);

  // Luma-stability & clip-distance disocclusion detection (drops stale lock on sudden light changes).
  let clip_dist = length(hist_yc_raw - remembered);
  let luma_instability = clamp(abs(hist_yc_raw.x - middle.x) / max(max(hist_yc_raw.x, middle.x), 0.12), 0.0, 1.0);
  let disocc = clamp(clip_dist * 4.0 + vel_div * 0.5, 0.0, 1.0);

  let prev_accum = hist_rgba.a * 32.0 * (1.0 - 0.75 * max(disocc, reactive));
  let next_accum = min(prev_accum + 1.0, phase_count);
  var alpha = taa.blend.z;
  if (is_upscale) {
    // Sub-pixel jitter proximity weighting + FSR-style sample-count convergence curve.
    let center_delta = (vec2f(pixel) + vec2f(0.5)) - in_pos;
    let prox = exp(-2.8 * dot(center_delta, center_delta));
    let steady = clamp((1.0 / phase_count) * (0.65 + 1.35 * prox), 0.03, 0.45);
    alpha = max(1.0 / max(next_accum, 1.0), steady);
    alpha = mix(alpha, 0.65, luma_instability * 0.45);
  } else if (reactive > 0.01 || vel_div > 0.01) {
    alpha = max(alpha, 1.0 / max(next_accum, 1.0));
  }
  alpha = clamp(mix(alpha, 0.85, reactive * 0.82), 0.03, 0.92);

  let out_rgb = unsqueezed(rgb(mix(remembered, middle, alpha)));
  return vec4f(out_rgb, clamp(next_accum / 32.0, 1.0 / 32.0, 1.0));
}
