// Half-resolution Horizon-based Ground-Truth Ambient Occlusion (GTAO): integrates analytical
// cosine-weighted hemisphere visibility across screen-space horizon slices using fast linear-Z
// view-space reconstruction.
struct Occlusion {
  projection: mat4x4f,
  inverse_projection: mat4x4f,
  // radius, power, bias, reversed_z (>0.5)
  settings: vec4f,
  // full_width, full_height, scale_x (full_w / ao_w), scale_y (full_h / ao_h)
  size: vec4f,
};
@group(0) @binding(0) var<uniform> occlusion: Occlusion;
@group(0) @binding(1) var scene_depth: ${depth_texture};
${fullscreen}

const PI: f32 = 3.14159265359;
const HALF_PI: f32 = 1.57079632679;

struct LinearUnproj {
  // x_view = (pixel.x * px_scale.x + px_bias.x) * inv_w
  // y_view = (pixel.y * px_scale.y + px_bias.y) * inv_w
  // z_view = (depth * z_params.x + z_params.y) * inv_w
  // w_clip = depth * z_params.z + z_params.w
  px_scale: vec2f,
  px_bias: vec2f,
  z_params: vec4f,
};

fn make_unproj(inv_size: vec2f) -> LinearUnproj {
  let ip = occlusion.inverse_projection;
  let ip0x = ip[0].x;
  let ip1y = ip[1].y;
  let ip3 = ip[3];
  return LinearUnproj(
    vec2f(2.0 * inv_size.x * ip0x, -2.0 * inv_size.y * ip1y),
    vec2f(ip3.x - ip0x, ip3.y + ip1y),
    vec4f(ip[2].z, ip3.z, ip[2].w, ip3.w),
  );
}

fn depth_at(pixel: vec2i, last: vec2i) -> f32 {
  return textureLoad(scene_depth, clamp(pixel, vec2i(0), last), 0);
}

fn is_sky(depth: f32, reversed_z: bool) -> bool {
  return select(depth >= 1.0, depth <= 0.0, reversed_z);
}

fn view_at(pixel: vec2f, depth: f32, u: LinearUnproj) -> vec3f {
  let inv_w = 1.0 / (depth * u.z_params.z + u.z_params.w);
  let xy = pixel * u.px_scale + u.px_bias;
  let z = depth * u.z_params.x + u.z_params.y;
  return vec3f(xy.x, xy.y, z) * inv_w;
}

fn neighbour(pixel: vec2f, step: vec2i, last: vec2i, u: LinearUnproj) -> vec3f {
  return view_at(pixel + vec2f(step), depth_at(vec2i(pixel) + step, last), u);
}

fn gtao_arc_cs(theta: f32, gamma: f32, cos_g: f32, sin_g: f32) -> f32 {
  return 0.25 * (-cos(2.0 * theta - gamma) + cos_g + 2.0 * theta * sin_g);
}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let last = vec2i(occlusion.size.xy) - 1;
  let inv_size = 1.0 / occlusion.size.xy;
  let scale = select(vec2f(1.0), occlusion.size.zw, occlusion.size.z > 0.0 && occlusion.size.w > 0.0);
  let full_pos = position.xy * scale;
  let reversed_z = occlusion.settings.w > 0.5;
  let depth = depth_at(vec2i(full_pos), last);
  if (is_sky(depth, reversed_z)) {
    return vec4f(1.0);
  }
  let unproj = make_unproj(inv_size);
  let middle = view_at(full_pos, depth, unproj);
  let view_dir = normalize(-middle);

  // Edge-preserving view-space normal from the closer neighbour on each axis.
  let right = neighbour(full_pos, vec2i(1, 0), last, unproj) - middle;
  let left = middle - neighbour(full_pos, vec2i(-1, 0), last, unproj);
  let down = neighbour(full_pos, vec2i(0, 1), last, unproj) - middle;
  let up = middle - neighbour(full_pos, vec2i(0, -1), last, unproj);
  let across = select(left, right, abs(right.z) < abs(left.z));
  let along = select(up, down, abs(down.z) < abs(up.z));
  var normal = normalize(cross(across, along));
  if (dot(normal, view_dir) < 0.0) {
    normal = -normal;
  }

  let radius = occlusion.settings.x;
  let inv_radius = 1.0 / max(radius, 1e-6);
  let bias = occlusion.settings.z;
  let bias2 = bias * bias;
  let max_r = radius * 1.5;
  let max_r2 = max_r * max_r;
  let proj_scale = abs(occlusion.projection[1][1]) * 0.5 * occlusion.size.y;
  let screen_radius = clamp(radius * proj_scale / max(-middle.z, 0.05), 2.0, 64.0);

  // Interleaved gradient noise rotates the 3 horizon slices and jitters step starts.
  let ign = fract(52.9829189 * fract(dot(floor(position.xy), vec2f(0.06711056, 0.00583715))));
  let step_jitter = fract(ign * 7.319);

  let slices = 3;
  let steps = 4;
  var visibility = 0.0;
  for (var s = 0; s < slices; s++) {
    let phi = (f32(s) + ign) * (PI / f32(slices));
    let dir_px = vec2f(cos(phi), sin(phi));
    // Screen +Y maps to view -Y.
    let slice_view = vec3f(dir_px.x, -dir_px.y, 0.0);
    let plane_raw = cross(slice_view, view_dir);
    let plane_len = length(plane_raw);
    if (plane_len < 1e-4) {
      visibility += 1.0;
      continue;
    }
    let plane_n = plane_raw / plane_len;
    let tangent = cross(view_dir, plane_n);
    let n_proj = normal - plane_n * dot(normal, plane_n);
    let n_proj_len = length(n_proj);
    if (n_proj_len < 1e-4) {
      visibility += 1.0;
      continue;
    }
    let cos_gamma = clamp(dot(n_proj, view_dir) / n_proj_len, -1.0, 1.0);
    let gamma_sign = select(-1.0, 1.0, dot(n_proj, tangent) >= 0.0);
    let gamma = acos(cos_gamma) * gamma_sign;
    let sin_gamma = sqrt(max(1.0 - cos_gamma * cos_gamma, 0.0)) * gamma_sign;

    var max_cos_pos = -1.0;
    var max_cos_neg = -1.0;
    for (var step = 0; step < steps; step++) {
      let frac = (f32(step) + 0.35 + 0.65 * step_jitter) / f32(steps);
      let offset_px = dir_px * max(screen_radius * frac * frac, f32(step + 1));

      // +side horizon search
      let p_pos = full_pos + offset_px;
      let d_pos = depth_at(vec2i(p_pos), last);
      if (!is_sky(d_pos, reversed_z)) {
        let delta = view_at(p_pos, d_pos, unproj) - middle;
        let dist2 = dot(delta, delta);
        if (dist2 > bias2 && dist2 < max_r2) {
          let inv_dist = inverseSqrt(dist2);
          let dist = dist2 * inv_dist;
          let falloff = clamp((radius - dist) * inv_radius, 0.0, 1.0);
          let h = dot(delta, view_dir) * inv_dist;
          max_cos_pos = max(max_cos_pos, mix(-1.0, h, falloff));
        }
      }

      // -side horizon search
      let p_neg = full_pos - offset_px;
      let d_neg = depth_at(vec2i(p_neg), last);
      if (!is_sky(d_neg, reversed_z)) {
        let delta = view_at(p_neg, d_neg, unproj) - middle;
        let dist2 = dot(delta, delta);
        if (dist2 > bias2 && dist2 < max_r2) {
          let inv_dist = inverseSqrt(dist2);
          let dist = dist2 * inv_dist;
          let falloff = clamp((radius - dist) * inv_radius, 0.0, 1.0);
          let h = dot(delta, view_dir) * inv_dist;
          max_cos_neg = max(max_cos_neg, mix(-1.0, h, falloff));
        }
      }
    }

    let h_lo = gamma - HALF_PI;
    let h_hi = gamma + HALF_PI;
    let theta2 = clamp(acos(clamp(max_cos_pos, -1.0, 1.0)), h_lo, h_hi);
    let theta1 = clamp(-acos(clamp(max_cos_neg, -1.0, 1.0)), h_lo, h_hi);
    let full_slice = max(cos_gamma + gamma * sin_gamma, 1e-4);
    let slice_vis = clamp((gtao_arc_cs(theta1, gamma, cos_gamma, sin_gamma) + gtao_arc_cs(theta2, gamma, cos_gamma, sin_gamma)) / full_slice, 0.0, 1.0);
    visibility += slice_vis;
  }

  let ao = clamp(visibility / f32(slices), 0.0, 1.0);
  return vec4f(pow(ao, occlusion.settings.y));
}
