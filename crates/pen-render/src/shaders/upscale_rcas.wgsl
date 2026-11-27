// FSR 2.2/3.1-Class Compute Temporal Super Resolution (TAAU),
// AMD FidelityFX FSR 1 EASU (Edge-Adaptive Spatial Upsampling),
// and AMD FidelityFX RCAS (Per-Channel HDR-Safe Robust Contrast Adaptive Sharpening).

struct TaauUniform {
  dims: vec4f,         // xy = low-res input dims, zw = native output dims
  jitter_blend: vec4f, // xy = low-res jitter_px [-0.5, 0.5], z = base_alpha, w = reset (>0.5)
  flags: vec4f,        // x = variance_gamma, y = reversed_z (>0.5), z = phase_count, w = has_reactive (>0.5)
};

@group(0) @binding(0) var<uniform> u: TaauUniform;
@group(0) @binding(1) var input_color: texture_2d<f32>;
@group(0) @binding(2) var input_depth: texture_2d<f32>;
@group(0) @binding(3) var input_vel: texture_2d<f32>;
@group(0) @binding(4) var history_tex: texture_2d<f32>;
@group(0) @binding(5) var linear_samp: sampler;
@group(0) @binding(6) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var reactive_tex: texture_2d<f32>;

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

fn lanczos2_approx(x: f32) -> f32 {
  let x2 = min(x * x, 4.0);
  let a = 0.4 * x2 - 1.0;
  let b = 0.25 * x2 - 1.0;
  return (1.5625 * a * a - 0.5625) * (b * b);
}

fn sample_history_catmull_rom_rgba(uv: vec2f, size: vec2f) -> vec4f {
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
  let half_px = vec2f(0.5) / size;
  let max_uv = (size - 0.5) / size;
  let t0 = clamp((center - 1.0) / size, half_px, max_uv);
  let t3 = clamp((center + 2.0) / size, half_px, max_uv);
  let t12 = clamp((center + w2 / max(w12, vec2f(1e-5))) / size, half_px, max_uv);
  let sum = textureSampleLevel(history_tex, linear_samp, vec2f(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y)
    + textureSampleLevel(history_tex, linear_samp, vec2f(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y)
    + textureSampleLevel(history_tex, linear_samp, t12, 0.0).rgb * (w12.x * w12.y)
    + textureSampleLevel(history_tex, linear_samp, vec2f(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y)
    + textureSampleLevel(history_tex, linear_samp, vec2f(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
  let weight = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  let accum = textureSampleLevel(history_tex, linear_samp, clamp(uv, half_px, max_uv), 0.0).a;
  return vec4f(max(sum / max(weight, 1e-5), vec3f(0.0)), clamp(accum, 0.0, 1.0));
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
  let phase_count = max(u.flags.z, 8.0);
  let has_reactive = u.flags.w > 0.5;

  // Initialize nearest-depth dilation at center `base_px` so equal-depth ties never bias toward top-left.
  var best_d = textureLoad(input_depth, base_px, 0).r;
  var best_px = base_px;

  var m1 = vec3f(0.0);
  var m2 = vec3f(0.0);
  var c_min = vec3f(1e9);
  var c_max = vec3f(-1e9);
  var lin_min = vec3f(1e9);
  var lin_max = vec3f(-1e9);
  var splat_lin = vec3f(0.0);
  var splat_yc = vec3f(0.0);
  var splat_w = 0.0;
  var vel_min = vec2f(1e9);
  var vel_max = vec2f(-1e9);

  for (var oy = -1; oy <= 1; oy++) {
    for (var ox = -1; ox <= 1; ox++) {
      let px = clamp(base_px + vec2i(ox, oy), vec2i(0), in_last);
      let c_lin = max(textureLoad(input_color, px, 0).rgb, vec3f(0.0));
      let c_yc = rgb_to_ycocg(tonemap(c_lin));
      m1 += c_yc;
      m2 += c_yc * c_yc;
      c_min = min(c_min, c_yc);
      c_max = max(c_max, c_yc);
      lin_min = min(lin_min, c_lin);
      lin_max = max(lin_max, c_lin);

      let delta = (vec2f(px) + 0.5) - in_pos;
      let w = lanczos2_approx(delta.x) * lanczos2_approx(delta.y);
      splat_lin += c_lin * w;
      splat_yc += c_yc * w;
      splat_w += w;

      let d = textureLoad(input_depth, px, 0).r;
      let nearer = select((d < best_d), (d > best_d), rev_z);
      if (nearer) {
        best_d = d;
        best_px = px;
      }

      let v = textureLoad(input_vel, px, 0).xy;
      vel_min = min(vel_min, v);
      vel_max = max(vel_max, v);
    }
  }
  let mean = m1 / 9.0;
  let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3f(0.0)));
  var reactive = 0.0;
  if (has_reactive) {
    reactive = clamp(textureLoad(reactive_tex, base_px, 0).r, 0.0, 1.0);
  }
  let vel_div = clamp((length((vel_max - vel_min) * out_size) - 1.0) * 0.25, 0.0, 1.0);
  let gamma = mix(clamp(u.flags.x, 0.5, 2.5), 0.85, max(reactive, vel_div));
  let box_min = max(c_min, mean - sigma * gamma);
  let box_max = min(c_max, mean + sigma * gamma);

  let valid_w = abs(splat_w) > 1e-4;
  let center_lin = max(textureLoad(input_color, base_px, 0).rgb, vec3f(0.0));
  let curr_lin = select(center_lin, clamp(splat_lin / splat_w, lin_min, lin_max), valid_w);
  let curr_yc = select(rgb_to_ycocg(tonemap(center_lin)), clamp(splat_yc / splat_w, c_min, c_max), valid_w);

  // On reset or disocclusion, output exact linear HDR reconstruction with zero tone-curve clamping.
  if (u.jitter_blend.w > 0.5) {
    textureStore(out_tex, vec2i(id.xy), vec4f(curr_lin, 1.0 / 32.0));
    return;
  }

  let vel = textureLoad(input_vel, best_px, 0).xy;
  let prev_uv = uv - vel;
  if (any(prev_uv < vec2f(0.0)) || any(prev_uv > vec2f(1.0))) {
    textureStore(out_tex, vec2i(id.xy), vec4f(curr_lin, 1.0 / 32.0));
    return;
  }

  let hist_rgba = sample_history_catmull_rom_rgba(prev_uv, out_size);
  let hist_yc_raw = rgb_to_ycocg(tonemap(hist_rgba.rgb));
  let hist_yc = clip_aabb(hist_yc_raw, box_min, box_max);

  let clip_dist = length(hist_yc_raw - hist_yc);
  let luma_instab = clamp(abs(hist_yc_raw.x - curr_yc.x) / max(max(hist_yc_raw.x, curr_yc.x), 0.12), 0.0, 1.0);
  let disocc = clamp(clip_dist * 4.0 + vel_div * 0.5, 0.0, 1.0);
  let prev_accum = hist_rgba.a * 32.0 * (1.0 - 0.75 * max(disocc, reactive));
  let next_accum = min(prev_accum + 1.0, phase_count);

  let center_delta = (vec2f(base_px) + 0.5) - in_pos;
  let prox = exp(-3.0 * dot(center_delta, center_delta));
  let base_alpha = clamp(u.jitter_blend.z, 0.02, 0.5);
  var alpha = clamp(base_alpha * (0.4 + 1.2 * prox), 0.02, 0.85);
  alpha = max(alpha, min(1.0 / max(next_accum, 1.0), 0.5));
  alpha = mix(alpha, 0.65, luma_instab * 0.35);
  alpha = clamp(mix(alpha, 0.90, reactive * 0.85), 0.02, 0.94);

  let blended_yc = mix(hist_yc, curr_yc, alpha);
  let out_rgb = untonemap(max(ycocg_to_rgb(blended_yc), vec3f(0.0)));
  textureStore(out_tex, vec2i(id.xy), vec4f(out_rgb, clamp(next_accum / 32.0, 1.0 / 32.0, 1.0)));
}

// --- AMD FidelityFX FSR 1 EASU (12-Tap Edge-Adaptive Spatial Upsampling) ---
struct EasuUniform {
  dims: vec4f, // xy = input dims, zw = output dims
};

@group(0) @binding(0) var<uniform> easu_u: EasuUniform;
@group(0) @binding(1) var easu_in: texture_2d<f32>;
@group(0) @binding(2) var easu_out: texture_storage_2d<rgba16float, write>;

fn easu_compress(c: vec3f) -> vec3f {
  let p = max(c, vec3f(0.0));
  return p / (p + vec3f(1.0));
}

fn easu_decompress(c: vec3f) -> vec3f {
  let clamped = clamp(c, vec3f(0.0), vec3f(0.9999));
  return clamped / max(vec3f(1.0) - clamped, vec3f(1e-4));
}

fn easu_luma(c: vec3f) -> f32 {
  return 0.25 * c.r + 0.5 * c.g + 0.25 * c.b;
}

fn easu_load(p: vec2i, last: vec2i) -> vec3f {
  return easu_compress(textureLoad(easu_in, clamp(p, vec2i(0), last), 0).rgb);
}

fn easu_tap(
  ac: ptr<function, vec3f>,
  aw: ptr<function, f32>,
  off: vec2f,
  dir: vec2f,
  len_scale: vec2f,
  lob: f32,
  clmp: f32,
  col: vec3f,
) {
  let vx = off.x * dir.x + off.y * dir.y;
  let vy = -off.x * dir.y + off.y * dir.x;
  let sx = vx * len_scale.x;
  let sy = vy * len_scale.y;
  let d2 = min(sx * sx + sy * sy, clmp);
  var wb = 0.4 * d2 - 1.0;
  var wa = lob * d2 - 1.0;
  wb = wb * wb;
  wa = wa * wa;
  wb = 1.5625 * wb - 0.5625;
  let w = wb * wa;
  *ac += col * w;
  *aw += w;
}

fn easu_dir_set(
  dir: ptr<function, vec2f>,
  len_acc: ptr<function, f32>,
  w: f32,
  lb: f32,
  le: f32,
  lf: f32,
  lg: f32,
  lj: f32,
) {
  let dcx = lg - le;
  let dcy = lj - lb;
  var lx = max(abs(lg - lf), abs(lf - le));
  var ly = max(abs(lj - lf), abs(lf - lb));
  lx = clamp(abs(dcx) / max(lx, 1e-4), 0.0, 1.0);
  ly = clamp(abs(dcy) / max(ly, 1e-4), 0.0, 1.0);
  *dir += vec2f(dcx, dcy) * w;
  *len_acc += (lx * lx + ly * ly) * w;
}

@compute @workgroup_size(8, 8, 1)
fn easu_upscale(@builtin(global_invocation_id) id: vec3u) {
  let out_dims = vec2u(easu_u.dims.zw);
  if (id.x >= out_dims.x || id.y >= out_dims.y) {
    return;
  }
  let in_dims = easu_u.dims.xy;
  let last = vec2i(in_dims) - vec2i(1);
  let pp_full = (vec2f(id.xy) + 0.5) * (in_dims / vec2f(out_dims)) - 0.5;
  let fp = vec2i(floor(pp_full));
  let pp = pp_full - vec2f(fp);

  // 12-tap cross/ring footprint:
  //      b c
  //    e f g h
  //    i j k l
  //      n o
  let b = easu_load(fp + vec2i(0, -1), last);
  let c = easu_load(fp + vec2i(1, -1), last);
  let e = easu_load(fp + vec2i(-1, 0), last);
  let f = easu_load(fp + vec2i(0, 0), last);
  let g = easu_load(fp + vec2i(1, 0), last);
  let h = easu_load(fp + vec2i(2, 0), last);
  let i = easu_load(fp + vec2i(-1, 1), last);
  let j = easu_load(fp + vec2i(0, 1), last);
  let k = easu_load(fp + vec2i(1, 1), last);
  let l = easu_load(fp + vec2i(2, 1), last);
  let n = easu_load(fp + vec2i(0, 2), last);
  let o = easu_load(fp + vec2i(1, 2), last);

  let lb = easu_luma(b);
  let lc = easu_luma(c);
  let le = easu_luma(e);
  let lf = easu_luma(f);
  let lg = easu_luma(g);
  let lh = easu_luma(h);
  let li = easu_luma(i);
  let lj = easu_luma(j);
  let lk = easu_luma(k);
  let ll = easu_luma(l);
  let ln = easu_luma(n);
  let lo = easu_luma(o);

  var dir = vec2f(0.0);
  var aniso = 0.0;
  easu_dir_set(&dir, &aniso, (1.0 - pp.x) * (1.0 - pp.y), lb, le, lf, lg, lj);
  easu_dir_set(&dir, &aniso, pp.x * (1.0 - pp.y), lc, lf, lg, lh, lk);
  easu_dir_set(&dir, &aniso, (1.0 - pp.x) * pp.y, lf, li, lj, lk, ln);
  easu_dir_set(&dir, &aniso, pp.x * pp.y, lg, lj, lk, ll, lo);

  let dir2 = dot(dir, dir);
  let zro = dir2 < 1e-6;
  dir = select(dir * inverseSqrt(max(dir2, 1e-6)), vec2f(1.0, 0.0), zro);
  aniso = clamp(aniso * 0.5, 0.0, 1.0);
  aniso = aniso * aniso;

  let stretch = 1.0 / max(abs(dir.x), abs(dir.y));
  let len_scale = vec2f(1.0 + (stretch - 1.0) * aniso, 1.0 - 0.5 * aniso);
  let lob = 0.5 + (0.25 - 0.04 - 0.5) * aniso;
  let clmp = 1.0 / max(lob, 0.1);

  var ac = vec3f(0.0);
  var aw = 0.0;
  easu_tap(&ac, &aw, vec2f(0.0, -1.0) - pp, dir, len_scale, lob, clmp, b);
  easu_tap(&ac, &aw, vec2f(1.0, -1.0) - pp, dir, len_scale, lob, clmp, c);
  easu_tap(&ac, &aw, vec2f(-1.0, 0.0) - pp, dir, len_scale, lob, clmp, e);
  easu_tap(&ac, &aw, vec2f(0.0, 0.0) - pp, dir, len_scale, lob, clmp, f);
  easu_tap(&ac, &aw, vec2f(1.0, 0.0) - pp, dir, len_scale, lob, clmp, g);
  easu_tap(&ac, &aw, vec2f(2.0, 0.0) - pp, dir, len_scale, lob, clmp, h);
  easu_tap(&ac, &aw, vec2f(-1.0, 1.0) - pp, dir, len_scale, lob, clmp, i);
  easu_tap(&ac, &aw, vec2f(0.0, 1.0) - pp, dir, len_scale, lob, clmp, j);
  easu_tap(&ac, &aw, vec2f(1.0, 1.0) - pp, dir, len_scale, lob, clmp, k);
  easu_tap(&ac, &aw, vec2f(2.0, 1.0) - pp, dir, len_scale, lob, clmp, l);
  easu_tap(&ac, &aw, vec2f(0.0, 2.0) - pp, dir, len_scale, lob, clmp, n);
  easu_tap(&ac, &aw, vec2f(1.0, 2.0) - pp, dir, len_scale, lob, clmp, o);

  let mn4 = min(min(f, g), min(j, k));
  let mx4 = max(max(f, g), max(j, k));
  let res_comp = clamp(ac / max(aw, 1e-5), mn4, mx4);
  textureStore(easu_out, vec2i(id.xy), vec4f(easu_decompress(res_comp), 1.0));
}

// --- AMD FidelityFX RCAS (Per-Channel SDR Linear & HDR-Safe Robust Contrast Adaptive Sharpening) ---
struct RcasUniform {
  params: vec4f, // x = sharpness [0..1], y = hdr_knee (>0 -> Reinhard c/(c+knee); <=0 -> linear SDR [0, 1])
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

var<workgroup> wg_rcas: array<vec3f, 100>;

@compute @workgroup_size(8, 8, 1)
fn rcas_sharpen(
  @builtin(global_invocation_id) id: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
) {
  let dims = textureDimensions(rcas_in, 0);
  let sharp = clamp(rcas_u.params.x, 0.0, 1.0);
  // When hdr_knee > 0, map HDR [0, +inf) into [0, 1) per-channel so HDR peaks > 1.0 sharpen with anti-ringing.
  // When hdr_knee <= 0, operate directly in SDR [0, 1] without nonlinear Reinhard midtone distortion.
  let use_hdr = rcas_u.params.y > 0.0;
  let knee = max(rcas_u.params.y, 0.25);
  let last = vec2i(dims) - vec2i(1);
  let tile_origin = vec2i(wid.xy * 8u) - vec2i(1);
  let lidx = lid.y * 8u + lid.x;
  if (sharp > 1e-4) {
    for (var slot = lidx; slot < 100u; slot += 64u) {
      let lx = i32(slot % 10u);
      let ly = i32(slot / 10u);
      let gp = clamp(tile_origin + vec2i(lx, ly), vec2i(0), last);
      let raw = max(textureLoad(rcas_in, gp, 0).rgb, vec3f(0.0));
      wg_rcas[slot] = select(clamp(raw, vec3f(0.0), vec3f(1.0)), rcas_compress(raw, knee), use_hdr);
    }
  }
  workgroupBarrier();

  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }
  let p = vec2i(id.xy);
  if (sharp <= 1e-4) {
    let e_raw = max(textureLoad(rcas_in, p, 0).rgb, vec3f(0.0));
    textureStore(rcas_out, p, vec4f(e_raw, 1.0));
    return;
  }
  let c_idx = (lid.y + 1u) * 10u + (lid.x + 1u);
  let e = wg_rcas[c_idx];
  let b = wg_rcas[c_idx - 10u];
  let d = wg_rcas[c_idx - 1u];
  let f = wg_rcas[c_idx + 1u];
  let h = wg_rcas[c_idx + 10u];

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
  let sharp_val = (e + (b + d + f + h) * w) * inv_norm;
  let out_rgb = select(
    clamp(sharp_val, vec3f(0.0), vec3f(1.0)),
    rcas_decompress(clamp(sharp_val, vec3f(0.0), vec3f(0.9999)), knee),
    use_hdr,
  );
  textureStore(rcas_out, p, vec4f(max(out_rgb, vec3f(0.0)), 1.0));
}
