// Camera and per-object velocity motion blur (motion_blur.rs).
// Reads screen-space velocity from the velocity buffer (falling back to
// depth-reprojected camera velocity where velocity holds the sentinel >= 500),
// finds the dominant neighborhood velocity to smear fast foreground edges over
// static backgrounds, and integrates depth-weighted taps along the motion line.
struct MotionBlur {
  inverse: mat4x4f,
  previous: mat4x4f,
  // x: strength, y: max_radius, z: samples, w: 1 if previous camera exists
  params: vec4f,
  // xy: size in pixels, zw: jitter in UV coordinates
  size: vec4f,
};
@group(0) @binding(0) var<uniform> mb: MotionBlur;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var velocity: texture_2d<f32>;
@group(0) @binding(3) var depth: ${depth_texture};
@group(0) @binding(4) var filtering: sampler;
${fullscreen}

fn depth_at_uv(uv: vec2f, dsize: vec2i) -> f32 {
  let p = clamp(vec2i(uv * vec2f(dsize)), vec2i(0), dsize - 1);
  return textureLoad(depth, p, 0);
}

fn clamp_len(v: vec2f, max_len: f32) -> vec2f {
  let len2 = dot(v, v);
  if (len2 > max_len * max_len && len2 > 1e-12) {
    return v * (max_len * inverseSqrt(len2));
  }
  return v;
}

fn velocity_at_uv_with_depth(uv: vec2f, d: f32, vsize: vec2i) -> vec2f {
  let vp = clamp(vec2i(uv * vec2f(vsize)), vec2i(0), vsize - 1);
  let obj = textureLoad(velocity, vp, 0).xy;
  if (abs(obj.x) < 500.0) {
    return obj;
  }
  if (mb.params.w < 0.5) {
    return vec2f(0.0);
  }
  let world = mb.inverse * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
  let before = mb.previous * vec4f(world.xyz / world.w, 1.0);
  if (before.w <= 1e-5) {
    return vec2f(0.0);
  }
  let was = vec2f(before.x, -before.y) / before.w * 0.5 + 0.5 + mb.size.zw;
  return uv - was;
}

fn velocity_at_uv(uv: vec2f, vsize: vec2i, dsize: vec2i) -> vec2f {
  let vp = clamp(vec2i(uv * vec2f(vsize)), vec2i(0), vsize - 1);
  let obj = textureLoad(velocity, vp, 0).xy;
  if (abs(obj.x) < 500.0) {
    return obj;
  }
  if (mb.params.w < 0.5) {
    return vec2f(0.0);
  }
  let d = depth_at_uv(uv, dsize);
  let world = mb.inverse * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
  let before = mb.previous * vec4f(world.xyz / world.w, 1.0);
  if (before.w <= 1e-5) {
    return vec2f(0.0);
  }
  let was = vec2f(before.x, -before.y) / before.w * 0.5 + 0.5 + mb.size.zw;
  return uv - was;
}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let size = mb.size.xy;
  let uv = position.xy / size;
  let pixel = vec2i(position.xy);
  let last = vec2i(size) - 1;
  let center_color = textureLoad(source, clamp(pixel, vec2i(0), last), 0);
  let strength = mb.params.x;
  let max_radius = mb.params.y;
  if (strength <= 1e-5 || max_radius <= 1e-5) {
    return center_color;
  }

  let vsize = vec2i(textureDimensions(velocity));
  let dsize = vec2i(textureDimensions(depth));
  let z_c = depth_at_uv(uv, dsize);
  let v_c = clamp_len(velocity_at_uv_with_depth(uv, z_c, vsize) * strength, max_radius);
  let v_c_len = length(v_c);
  var v_max = v_c;
  var max_speed = v_c_len;

  // Neighborhood max-velocity search across an 8-direction ring (and half-ring)
  // so fast-moving foreground objects smear smoothly over static backgrounds.
  let dirs = array<vec2f, 8>(
    vec2f(1.0, 0.0), vec2f(-1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, -1.0),
    vec2f(0.7071, 0.7071), vec2f(-0.7071, 0.7071), vec2f(0.7071, -0.7071), vec2f(-0.7071, -0.7071),
  );
  for (var ring = 1; ring <= 2; ring++) {
    let step_uv = max_radius * 0.25 * f32(ring);
    for (var i = 0; i < 8; i++) {
      let nuv = clamp(uv + dirs[i] * step_uv, vec2f(0.0), vec2f(1.0));
      let vn = clamp_len(velocity_at_uv(nuv, vsize, dsize) * strength, max_radius);
      let s = length(vn);
      if (s > max_speed) {
        max_speed = s;
        v_max = vn;
      }
    }
  }

  let min_speed = 0.5 / max(size.x, size.y);
  if (max_speed < min_speed) {
    return center_color;
  }

  let count = i32(clamp(mb.params.z, 4.0, 32.0));
  let inv_count = 1.0 / f32(count);
  let inv_v_c_half = 1.0 / max(v_c_len * 0.5, 1e-5);
  // Interleaved gradient noise breaks stepping bands across fast silhouettes.
  let jitter = fract(52.9829189 * fract(dot(floor(position.xy), vec2f(0.06711056, 0.00583715)))) - 0.5;

  var accum = center_color.rgb;
  var total = 1.0;
  for (var i = 0; i < count; i++) {
    let t = (f32(i) + 0.5 + jitter * 0.5) * inv_count - 0.5;
    let sample_uv = clamp(uv + v_max * t, vec2f(0.0), vec2f(1.0));
    let z_q = depth_at_uv(sample_uv, dsize);
    let v_q = clamp_len(velocity_at_uv_with_depth(sample_uv, z_q, vsize) * strength, max_radius);
    let v_q_len = length(v_q);
    let dist = abs(t) * max_speed;
    // Soft depth comparison (hardware depth 0..1, smaller = nearer).
    let fg = clamp(1.0 - (z_q - z_c) * 120.0, 0.0, 1.0);
    let bg = clamp(1.0 - (z_c - z_q) * 120.0, 0.0, 1.0);
    let cone_q = clamp(1.0 - dist / max(v_q_len * 0.5, 1e-5), 0.0, 1.0);
    let cone_c = clamp(1.0 - dist * inv_v_c_half, 0.0, 1.0);
    let both = select(0.0, 0.5, dist <= min(v_q_len, v_c_len) * 0.5);
    let w = fg * cone_q + bg * cone_c + both;
    if (w > 1e-4) {
      let col = textureSampleLevel(source, filtering, sample_uv, 0.0).rgb;
      accum += col * w;
      total += w;
    }
  }
  return vec4f(accum / total, center_color.a);
}
