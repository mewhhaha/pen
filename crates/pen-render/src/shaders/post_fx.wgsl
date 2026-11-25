// Post-FX Polish Compute Pass:
// 1. Anamorphic Horizontal Lens Flare & Chromatic Streaks (zero-padded borders, no edge weighting bias)
// 2. Screen-Space Depth + View-Space Normal Silhouette Outlines (actual inverse_projection + cross-res depth mapping)
// 3. Luminance-Aware Animated Perceptual Film Grain (tapers strictly to 0 at pure black).

struct PostFxUniform {
  inverse_projection: mat4x4f,
  // x = grain_strength, y = frame_seed, z = flare_strength, w = flare_threshold
  grain_flare: vec4f,
  // rgb = flare_tint, w = flare_spread_px
  flare_tint_spread: vec4f,
  // rgb = outline_color, w = outline_strength (0 = off)
  outline_color_str: vec4f,
  // x = depth_threshold, y = normal_threshold, z = outline_thickness_px, w = near_plane
  outline_params: vec4f,
  // x = far_plane, y = reversed_z (>0.5), z = ortho_flag (>0.5), w = has_inv_proj (>0.5)
  depth_params: vec4f,
};

@group(0) @binding(0) var<uniform> u: PostFxUniform;
@group(0) @binding(1) var color_in: texture_2d<f32>;
@group(0) @binding(2) var depth_in: texture_2d<f32>;
@group(0) @binding(3) var color_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var color_sampler: sampler;

// Precomputed exp(-0.28 * k) for k = 0..=10.
const FLARE_WEIGHTS = array<f32, 11>(
  1.0,
  0.75578374,
  0.57120906,
  0.43171052,
  0.32627979,
  0.24659696,
  0.18637398,
  0.14085842,
  0.10645850,
  0.08045961,
  0.06081006,
);

fn hash13(p3: vec3f) -> f32 {
  var p = fract(p3 * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}

fn linearize_depth(raw_d: f32) -> f32 {
  let d = select(raw_d, 1.0 - raw_d, u.depth_params.y > 0.5);
  let near = max(u.outline_params.w, 1e-5);
  let far = max(u.depth_params.x, near + 1e-4);
  if (u.depth_params.z > 0.5) {
    return mix(near, far, clamp(d, 0.0, 1.0));
  }
  return (near * far) / max(far - clamp(d, 0.0, 1.0) * (far - near), 1e-5);
}

fn sample_color_zero_border(p: vec2i, dims: vec2i) -> vec3f {
  if (p.x < 0 || p.y < 0 || p.x >= dims.x || p.y >= dims.y) {
    return vec3f(0.0);
  }
  return max(textureLoad(color_in, p, 0).rgb, vec3f(0.0));
}

fn sample_color_linear_x(x_f: f32, y: i32, dims: vec2i, inv_dims: vec2f) -> vec3f {
  let max_x = f32(dims.x - 1);
  if (x_f < -1.0 || x_f >= f32(dims.x)) {
    return vec3f(0.0);
  }
  if (x_f >= 0.0 && x_f <= max_x) {
    let uv = vec2f((x_f + 0.5) * inv_dims.x, (f32(y) + 0.5) * inv_dims.y);
    return max(textureSampleLevel(color_in, color_sampler, uv, 0.0).rgb, vec3f(0.0));
  }
  let x0 = i32(floor(x_f));
  let frac = x_f - f32(x0);
  let c0 = sample_color_zero_border(vec2i(x0, y), dims);
  let c1 = sample_color_zero_border(vec2i(x0 + 1, y), dims);
  return mix(c0, c1, frac);
}

fn bright_pass(c: vec3f, threshold: f32) -> vec3f {
  let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  let excess = max(l - threshold, 0.0);
  return c * (excess / max(l, 1e-4));
}

fn view_pos_from_raw(c: vec2i, raw_d: f32, d_dims: vec2i) -> vec3f {
  let ndc = vec2f(
    (f32(c.x) + 0.5) / f32(d_dims.x) * 2.0 - 1.0,
    1.0 - (f32(c.y) + 0.5) / f32(d_dims.y) * 2.0,
  );
  if (u.depth_params.w > 0.5) {
    let h = u.inverse_projection * vec4f(ndc, raw_d, 1.0);
    if (abs(h.w) > 1e-6) {
      return h.xyz / h.w;
    }
  }
  let z = linearize_depth(raw_d);
  return vec3f(ndc.x * z, ndc.y * z, -z);
}

fn load_view_pos(d_px: vec2i, d_dims: vec2i) -> vec3f {
  let c = clamp(d_px, vec2i(0), d_dims - vec2i(1));
  let raw_d = textureLoad(depth_in, c, 0).r;
  return view_pos_from_raw(c, raw_d, d_dims);
}

fn view_normal_from_neighbors(d_px: vec2i, d_dims: vec2i, pc: vec3f, pr: vec3f, pl: vec3f, pd: vec3f, pu: vec3f) -> vec3f {
  let dx_r = pr - pc;
  let dx_l = pc - pl;
  let dy_d = pd - pc;
  let dy_u = pc - pu;
  let dx = select(dx_l, dx_r, d_px.x + 1 < d_dims.x && (d_px.x == 0 || dot(dx_r, dx_r) < dot(dx_l, dx_l)));
  let dy = select(dy_u, dy_d, d_px.y + 1 < d_dims.y && (d_px.y == 0 || dot(dy_d, dy_d) < dot(dy_u, dy_u)));
  let raw = cross(dx, dy);
  let len2 = dot(raw, raw);
  var n = select(vec3f(0.0, 0.0, 1.0), raw * inverseSqrt(len2), len2 > 1e-12 * dot(dx, dx) * dot(dy, dy));
  if (n.z < 0.0) {
    n = -n;
  }
  return n;
}

fn view_normal_at_cached(d_px: vec2i, pc: vec3f, d_dims: vec2i) -> vec3f {
  let pr = load_view_pos(d_px + vec2i(1, 0), d_dims);
  let pl = load_view_pos(d_px - vec2i(1, 0), d_dims);
  let pd = load_view_pos(d_px + vec2i(0, 1), d_dims);
  let pu = load_view_pos(d_px - vec2i(0, 1), d_dims);
  return view_normal_from_neighbors(d_px, d_dims, pc, pr, pl, pd, pu);
}

@compute @workgroup_size(8, 8, 1)
fn post_fx(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(color_in, 0));
  if (i32(id.x) >= dims.x || i32(id.y) >= dims.y) {
    return;
  }
  let p = vec2i(id.xy);
  let last = dims - vec2i(1);
  var color = max(textureLoad(color_in, p, 0).rgb, vec3f(0.0));

  // 1. Anamorphic horizontal lens flare streaks (off-image taps contribute 0 so edges aren't overweighted).
  let flare_str = max(u.grain_flare.z, 0.0);
  if (flare_str > 1e-4) {
    let thresh = max(u.grain_flare.w, 0.0);
    let spread = clamp(u.flare_tint_spread.w, 0.5, 8.0);
    let inv_dims = vec2f(1.0) / vec2f(dims);
    let base = f32(p.x);
    let step = spread * 0.5;
    var streak = vec3f(0.0);
    for (var i = -10; i <= 10; i++) {
      let w = FLARE_WEIGHTS[abs(i)];
      let dx = f32(i) * step;
      let cr = bright_pass(sample_color_linear_x(base + dx * 1.08, p.y, dims, inv_dims), thresh).r;
      let cg = bright_pass(sample_color_linear_x(base + dx, p.y, dims, inv_dims), thresh).g;
      let cb = bright_pass(sample_color_linear_x(base + dx * 0.92, p.y, dims, inv_dims), thresh).b;
      streak += vec3f(cr, cg, cb) * w;
    }
    let mirror_p = last - p;
    let ghost = bright_pass(sample_color_zero_border(mirror_p, dims), thresh * 1.25) * 0.22;
    color += (streak * 0.14 + ghost) * u.flare_tint_spread.rgb * flare_str;
  }

  // 2. Screen-space Roberts-cross Depth + View-Space Normal Silhouette Outlines (mapped across color/depth sizes).
  let outline_str = clamp(u.outline_color_str.w, 0.0, 1.0);
  if (outline_str > 1e-4) {
    let d_dims = vec2i(textureDimensions(depth_in, 0));
    let d_last = d_dims - vec2i(1);
    let uv = (vec2f(p) + 0.5) / vec2f(dims);
    let d_p = clamp(vec2i(floor(uv * vec2f(d_dims))), vec2i(0), d_last);
    let r = clamp(i32(round(u.outline_params.z)), 1, 4);
    let p00 = clamp(d_p + vec2i(-r, -r), vec2i(0), d_last);
    let p11 = clamp(d_p + vec2i(r, r), vec2i(0), d_last);
    let p10 = clamp(d_p + vec2i(r, -r), vec2i(0), d_last);
    let p01 = clamp(d_p + vec2i(-r, r), vec2i(0), d_last);
    let raw00 = textureLoad(depth_in, p00, 0).r;
    let raw11 = textureLoad(depth_in, p11, 0).r;
    let raw10 = textureLoad(depth_in, p10, 0).r;
    let raw01 = textureLoad(depth_in, p01, 0).r;
    let d00 = linearize_depth(raw00);
    let d11 = linearize_depth(raw11);
    let d10 = linearize_depth(raw10);
    let d01 = linearize_depth(raw01);
    let d_ref = max(min(min(d00, d11), min(d10, d01)), 0.05);
    let depth_grad = sqrt((d00 - d11) * (d00 - d11) + (d10 - d01) * (d10 - d01)) / d_ref;
    let d_thresh = max(u.outline_params.x, 1e-4);
    let depth_edge = smoothstep(d_thresh, d_thresh * 2.0, depth_grad);

    let vp00 = view_pos_from_raw(p00, raw00, d_dims);
    let vp11 = view_pos_from_raw(p11, raw11, d_dims);
    let vp10 = view_pos_from_raw(p10, raw10, d_dims);
    let vp01 = view_pos_from_raw(p01, raw01, d_dims);
    var n00: vec3f;
    var n11: vec3f;
    var n10: vec3f;
    var n01: vec3f;
    if (r == 1 && d_p.x >= 1 && d_p.x < d_last.x && d_p.y >= 1 && d_p.y < d_last.y) {
      let pt = load_view_pos(d_p + vec2i(0, -1), d_dims);
      let pb = load_view_pos(d_p + vec2i(0, 1), d_dims);
      let pl_m = load_view_pos(d_p + vec2i(-1, 0), d_dims);
      let pr_m = load_view_pos(d_p + vec2i(1, 0), d_dims);
      n00 = view_normal_from_neighbors(p00, d_dims, vp00, pt, load_view_pos(p00 - vec2i(1, 0), d_dims), pl_m, load_view_pos(p00 - vec2i(0, 1), d_dims));
      n10 = view_normal_from_neighbors(p10, d_dims, vp10, load_view_pos(p10 + vec2i(1, 0), d_dims), pt, pr_m, load_view_pos(p10 - vec2i(0, 1), d_dims));
      n01 = view_normal_from_neighbors(p01, d_dims, vp01, pb, load_view_pos(p01 - vec2i(1, 0), d_dims), load_view_pos(p01 + vec2i(0, 1), d_dims), pl_m);
      n11 = view_normal_from_neighbors(p11, d_dims, vp11, load_view_pos(p11 + vec2i(1, 0), d_dims), pb, load_view_pos(p11 + vec2i(0, 1), d_dims), pr_m);
    } else {
      n00 = view_normal_at_cached(p00, vp00, d_dims);
      n11 = view_normal_at_cached(p11, vp11, d_dims);
      n10 = view_normal_at_cached(p10, vp10, d_dims);
      n01 = view_normal_at_cached(p01, vp01, d_dims);
    }
    let n_diff0 = n00 - n11;
    let n_diff1 = n10 - n01;
    let norm_grad = sqrt(dot(n_diff0, n_diff0) + dot(n_diff1, n_diff1));
    let n_thresh = max(u.outline_params.y, 1e-4);
    let norm_edge = smoothstep(n_thresh, n_thresh * 1.8, norm_grad);

    let edge = clamp(max(depth_edge, norm_edge), 0.0, 1.0) * outline_str;
    color = mix(color, u.outline_color_str.rgb, edge);
  }

  // 3. Luminance-aware animated perceptual film grain (tapers strictly to 0 at black so blacks stay true 0).
  let grain_str = max(u.grain_flare.x, 0.0);
  if (grain_str > 1e-4) {
    let seed = u.grain_flare.y;
    let n1 = hash13(vec3f(vec2f(id.xy), seed));
    let n2 = hash13(vec3f(vec2f(id.xy) + vec2f(17.3, 31.7), seed * 1.37 + 5.1));
    let tri_noise = n1 - n2;
    let l = dot(color, vec3f(0.2126, 0.7152, 0.0722));
    let l_norm = l / (1.0 + l);
    let envelope = 4.0 * l_norm * (1.0 - l_norm);
    color = max(color + vec3f(tri_noise * grain_str * envelope), vec3f(0.0));
  }

  textureStore(color_out, p, vec4f(max(color, vec3f(0.0)), 1.0));
}
