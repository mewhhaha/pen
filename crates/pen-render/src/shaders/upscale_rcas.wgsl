// Native-Resolution Temporal Upscaling (TAAU) + AMD FidelityFX RCAS Sharpening.
// - `taau_upscale`: upsamples jittered low-resolution HDR frames into a native-resolution
//   history buffer using center-tie-breaking 3x3 nearest-depth velocity dilation, 5-tap
//   Catmull-Rom history resampling, high-headroom Karis tone-compressed YCoCg variance clipping,
//   and jitter-aware sub-pixel blending.
//   Jitter convention: `jitter_blend.xy` is the camera sample displacement in low-res pixels
//   (positive +x/+y shifts rasterized features down/right, matching `apply_projection_jitter`).
// - `rcas_sharpen`: AMD FidelityFX RCAS operating in per-channel Reinhard working domain so HDR
//   values > 1.0 still sharpen smoothly while enforcing per-channel anti-ringing bounds.

struct TaauUniform {
  dims: vec4f,         // xy = low-res input dims, zw = native output dims
  jitter_blend: vec4f, // xy = low-res jitter_px [-0.5, 0.5], z = base_alpha, w = reset (>0.5)
  flags: vec4f,        // x = variance_gamma, y = reversed_z (>0.5), zw = reserved
};

@group(0) @binding(0) var<uniform> u: TaauUniform;
@group(0) @binding(1) var input_color: texture_2d<f32>;
@group(0) @binding(2) var input_depth: texture_2d<f32>;
@group(0) @binding(3) var input_vel: texture_2d<f32>;
@group(0) @binding(4) var history_tex: texture_2d<f32>;
@group(0) @binding(5) var linear_samp: sampler;
@group(0) @binding(6) var out_tex: texture_storage_2d<rgba16float, write>;

fn luma(c: vec3f) -> f32 {
  return dot(max(c, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722));
}

// High-headroom reversible Karis tone compression (lossless well past Rgba16Float 65504 max).
fn tonemap(c: vec3f) -> vec3f {
  let clamped = max(c, vec3f(0.0));
  return clamped / (1.0 + luma(clamped));
}

fn untonemap(c: vec3f) -> vec3f {
  let clamped = max(c, vec3f(0.0));
  return clamped / max(1.0 - luma(clamped), 1e-6);
}

fn rgb_to_ycocg(c: vec3f) -> vec3f {
  return vec3f(
    0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
    0.5 * c.r - 0.5 * c.b,
    -0.25 * c.r + 0.5 * c.g - 0.25 * c.b,
  );
}

fn ycocg_to_rgb(y: vec3f) -> vec3f {
  return vec3f(y.x + y.y - y.z, y.x + y.z, y.x - y.y - y.z);
}

fn sample_history_catmull_rom(uv: vec2f, size: vec2f) -> vec3f {
  let pos = uv * size;
  let center = floor(pos - 0.5) + 0.5;
  let f = pos - center;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = -0.5 * f3 + f2 - 0.5 * f;
  let w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
  let w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
  let w3 = 0.5 * f3 - 0.5 * f2;
  let w12 = w1 + w2;
  let t0 = clamp((center - 1.0) / size, vec2f(0.5) / size, (size - 0.5) / size);
  let t3 = clamp((center + 2.0) / size, vec2f(0.5) / size, (size - 0.5) / size);
  let t12 = clamp((center + w2 / max(w12, vec2f(1e-5))) / size, vec2f(0.5) / size, (size - 0.5) / size);
  let sum = textureSampleLevel(history_tex, linear_samp, vec2f(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y)
    + textureSampleLevel(history_tex, linear_samp, vec2f(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y)
    + textureSampleLevel(history_tex, linear_samp, t12, 0.0).rgb * (w12.x * w12.y)
    + textureSampleLevel(history_tex, linear_samp, vec2f(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y)
    + textureSampleLevel(history_tex, linear_samp, vec2f(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
  let weight = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  return max(sum / max(weight, 1e-5), vec3f(0.0));
}

fn clip_aabb(color: vec3f, low: vec3f, high: vec3f) -> vec3f {
  let mid = 0.5 * (low + high);
  let ext = 0.5 * (high - low) + vec3f(1e-5);
  let off = color - mid;
  let units = abs(off / ext);
  let m = max(units.x, max(units.y, units.z));
  return select(color, mid + off / m, m > 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn taau_upscale(@builtin(global_invocation_id) id: vec3u) {
  let out_dims = vec2u(u.dims.zw);
  if (id.x >= out_dims.x || id.y >= out_dims.y) {
    return;
  }
  let in_dims = vec2f(textureDimensions(input_color, 0));
  let in_last = vec2i(in_dims) - vec2i(1);
  let out_size = vec2f(out_dims);
  let uv = (vec2f(id.xy) + 0.5) / out_size;

  // Map native pixel into jittered low-resolution pixel space (matches `apply_projection_jitter`).
  let in_pos = uv * in_dims + u.jitter_blend.xy;
  let base_px = clamp(vec2i(floor(in_pos)), vec2i(0), in_last);
  let rev_z = u.flags.y > 0.5;

  // Initialize nearest-depth dilation at center `base_px` so equal-depth ties never bias toward top-left.
  var best_d = textureLoad(input_depth, base_px, 0).r;
  var best_px = base_px;

  var m1 = vec3f(0.0);
  var m2 = vec3f(0.0);
  var c_min = vec3f(1e9);
  var c_max = vec3f(-1e9);
  var splat_lin = vec3f(0.0);
  var splat_yc = vec3f(0.0);
  var splat_w = 0.0;

  for (var oy = -1; oy <= 1; oy++) {
    for (var ox = -1; ox <= 1; ox++) {
      let px = clamp(base_px + vec2i(ox, oy), vec2i(0), in_last);
      let c_lin = max(textureLoad(input_color, px, 0).rgb, vec3f(0.0));
      let c_yc = rgb_to_ycocg(tonemap(c_lin));
      m1 += c_yc;
      m2 += c_yc * c_yc;
      c_min = min(c_min, c_yc);
      c_max = max(c_max, c_yc);

      let delta = (vec2f(px) + 0.5) - in_pos;
      let dist2 = dot(delta, delta);
      let w = exp(-2.2 * dist2);
      splat_lin += c_lin * w;
      splat_yc += c_yc * w;
      splat_w += w;

      let d = textureLoad(input_depth, px, 0).r;
      let nearer = select((d < best_d), (d > best_d), rev_z);
      if (nearer) {
        best_d = d;
        best_px = px;
      }
    }
  }
  let mean = m1 / 9.0;
  let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3f(0.0)));
  let gamma = clamp(u.flags.x, 0.5, 2.5);
  let box_min = max(c_min, mean - sigma * gamma);
  let box_max = min(c_max, mean + sigma * gamma);
  let inv_splat = 1.0 / max(splat_w, 1e-5);
  let curr_lin = splat_lin * inv_splat;
  let curr_yc = splat_yc * inv_splat;

  // On reset or disocclusion, output exact linear HDR reconstruction with zero tone-curve clamping.
  if (u.jitter_blend.w > 0.5) {
    textureStore(out_tex, vec2i(id.xy), vec4f(curr_lin, 1.0));
    return;
  }

  let vel = textureLoad(input_vel, best_px, 0).xy;
  let prev_uv = uv - vel;
  if (any(prev_uv < vec2f(0.0)) || any(prev_uv > vec2f(1.0))) {
    textureStore(out_tex, vec2i(id.xy), vec4f(curr_lin, 1.0));
    return;
  }

  let hist_lin = sample_history_catmull_rom(prev_uv, out_size);
  let hist_yc = clip_aabb(rgb_to_ycocg(tonemap(hist_lin)), box_min, box_max);

  let center_delta = (vec2f(base_px) + 0.5) - in_pos;
  let prox = exp(-3.0 * dot(center_delta, center_delta));
  let base_alpha = clamp(u.jitter_blend.z, 0.02, 0.5);
  let alpha = clamp(base_alpha * (0.4 + 1.2 * prox), 0.02, 0.85);

  let blended_yc = mix(hist_yc, curr_yc, alpha);
  let out_rgb = untonemap(max(ycocg_to_rgb(blended_yc), vec3f(0.0)));
  textureStore(out_tex, vec2i(id.xy), vec4f(out_rgb, 1.0));
}

// --- AMD FidelityFX RCAS (Per-Channel HDR-Safe Robust Contrast Adaptive Sharpening) ---
struct RcasUniform {
  params: vec4f, // x = sharpness [0..1], y = hdr_knee (default 1.0 -> Reinhard c/(c+knee))
};

@group(0) @binding(0) var<uniform> rcas_u: RcasUniform;
@group(0) @binding(1) var rcas_in: texture_2d<f32>;
@group(0) @binding(2) var rcas_out: texture_storage_2d<rgba16float, write>;

fn rcas_luma(c: vec3f) -> f32 {
  return 0.25 * c.r + 0.5 * c.g + 0.25 * c.b;
}

fn rcas_compress(c: vec3f, knee: f32) -> vec3f {
  let p = max(c, vec3f(0.0));
  return p / (p + vec3f(knee));
}

fn rcas_decompress(c: vec3f, knee: f32) -> vec3f {
  let clamped = clamp(c, vec3f(0.0), vec3f(0.9999));
  return (clamped * knee) / max(vec3f(1.0) - clamped, vec3f(1e-4));
}

@compute @workgroup_size(8, 8, 1)
fn rcas_sharpen(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(rcas_in, 0);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }
  let p = vec2i(id.xy);
  let last = vec2i(dims) - vec2i(1);
  let e_raw = max(textureLoad(rcas_in, p, 0).rgb, vec3f(0.0));
  let sharp = clamp(rcas_u.params.x, 0.0, 1.0);
  if (sharp <= 1e-4) {
    textureStore(rcas_out, p, vec4f(e_raw, 1.0));
    return;
  }
  // Map HDR [0, +inf) cleanly into [0, 1) per-channel so HDR peaks > 1.0 still sharpen with anti-ringing.
  let knee = max(rcas_u.params.y, 0.25);
  let e = rcas_compress(e_raw, knee);
  let b = rcas_compress(textureLoad(rcas_in, clamp(p + vec2i(0, -1), vec2i(0), last), 0).rgb, knee);
  let d = rcas_compress(textureLoad(rcas_in, clamp(p + vec2i(-1, 0), vec2i(0), last), 0).rgb, knee);
  let f = rcas_compress(textureLoad(rcas_in, clamp(p + vec2i(1, 0), vec2i(0), last), 0).rgb, knee);
  let h = rcas_compress(textureLoad(rcas_in, clamp(p + vec2i(0, 1), vec2i(0), last), 0).rgb, knee);

  // Per-channel min/max anti-ringing limits (exact AMD FSR1 RCAS per-channel formulation).
  let mn4 = min(min(b, d), min(f, h));
  let mx4 = max(max(b, d), max(f, h));
  let mn_all = min(mn4, e);
  let mx_all = max(mx4, e);

  let hit_min = mn_all / max(4.0 * mx_all, vec3f(1e-5));
  let hit_max = (vec3f(1.0) - mx_all) / min(4.0 * mn_all - vec3f(4.0), vec3f(-1e-5));
  let lobe_rgb = max(-hit_min, hit_max);
  // Tightest anti-ringing lobe across all 3 channels prevents chroma halos on saturated edges.
  let lobe = max(max(lobe_rgb.r, max(lobe_rgb.g, lobe_rgb.b)), -0.1875);

  let e_l = rcas_luma(e);
  let b_l = rcas_luma(b);
  let d_l = rcas_luma(d);
  let f_l = rcas_luma(f);
  let h_l = rcas_luma(h);
  let nz = 0.25 * (b_l + d_l + f_l + h_l) - e_l;
  let l_range = max(max(max(b_l, d_l), max(f_l, h_l)), e_l) - min(min(min(b_l, d_l), min(f_l, h_l)), e_l);
  let noise_damp = 1.0 - 0.5 * clamp(abs(nz) / max(l_range, 1e-4), 0.0, 1.0);

  let w = clamp(lobe, -0.1875, 0.0) * sharp * noise_damp;
  let inv_norm = 1.0 / (1.0 + 4.0 * w);
  let sharp_comp = clamp((e + (b + d + f + h) * w) * inv_norm, vec3f(0.0), vec3f(0.9999));
  let out_hdr = rcas_decompress(sharp_comp, knee);
  textureStore(rcas_out, p, vec4f(max(out_hdr, vec3f(0.0)), 1.0));
}
