// Horizon-based Ground-Truth Ambient Occlusion (GTAO): integrates analytical
// cosine-weighted hemisphere visibility across screen-space horizon slices.
struct Occlusion {
  projection: mat4x4f,
  inverse_projection: mat4x4f,
  // radius, power, bias
  settings: vec4f,
  size: vec4f,
};
@group(0) @binding(0) var<uniform> occlusion: Occlusion;
@group(0) @binding(1) var scene_depth: ${depth_texture};
${fullscreen}

const PI: f32 = 3.14159265359;
const HALF_PI: f32 = 1.57079632679;

fn depth_at(pixel: vec2i) -> f32 {
  let size = vec2i(occlusion.size.xy);
  return textureLoad(scene_depth, clamp(pixel, vec2i(0), size - 1), 0);
}

fn view_at(pixel: vec2f, depth: f32) -> vec3f {
  let ndc = vec2f(pixel.x / occlusion.size.x * 2.0 - 1.0, 1.0 - pixel.y / occlusion.size.y * 2.0);
  let point = occlusion.inverse_projection * vec4f(ndc, depth, 1.0);
  return point.xyz / point.w;
}

fn neighbour(pixel: vec2f, step: vec2i) -> vec3f {
  return view_at(pixel + vec2f(step), depth_at(vec2i(pixel) + step));
}

fn gtao_arc(theta: f32, gamma: f32) -> f32 {
  return 0.25 * (-cos(2.0 * theta - gamma) + cos(gamma) + 2.0 * theta * sin(gamma));
}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let depth = depth_at(vec2i(position.xy));
  if (depth >= 1.0) {
    return vec4f(1.0);
  }
  let middle = view_at(position.xy, depth);
  let view_dir = normalize(-middle);

  // Edge-preserving view-space normal from the closer neighbour on each axis.
  let right = neighbour(position.xy, vec2i(1, 0)) - middle;
  let left = middle - neighbour(position.xy, vec2i(-1, 0));
  let down = neighbour(position.xy, vec2i(0, 1)) - middle;
  let up = middle - neighbour(position.xy, vec2i(0, -1));
  let across = select(left, right, abs(right.z) < abs(left.z));
  let along = select(up, down, abs(down.z) < abs(up.z));
  var normal = normalize(cross(across, along));
  if (dot(normal, view_dir) < 0.0) {
    normal = -normal;
  }

  let radius = occlusion.settings.x;
  let bias = occlusion.settings.z;
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
    let gamma = acos(cos_gamma) * select(-1.0, 1.0, dot(n_proj, tangent) >= 0.0);

    var max_cos_pos = -1.0;
    var max_cos_neg = -1.0;
    for (var step = 0; step < steps; step++) {
      let frac = (f32(step) + 0.35 + 0.65 * step_jitter) / f32(steps);
      let offset_px = dir_px * max(screen_radius * frac * frac, f32(step + 1));

      // +side horizon search
      let p_pos = position.xy + offset_px;
      let d_pos = depth_at(vec2i(p_pos));
      if (d_pos < 1.0) {
        let delta = view_at(p_pos, d_pos) - middle;
        let dist = length(delta);
        if (dist > bias && dist < radius * 1.5) {
          let falloff = clamp((radius - dist) / radius, 0.0, 1.0);
          let h = dot(delta / dist, view_dir);
          max_cos_pos = max(max_cos_pos, mix(-1.0, h, falloff));
        }
      }

      // -side horizon search
      let p_neg = position.xy - offset_px;
      let d_neg = depth_at(vec2i(p_neg));
      if (d_neg < 1.0) {
        let delta = view_at(p_neg, d_neg) - middle;
        let dist = length(delta);
        if (dist > bias && dist < radius * 1.5) {
          let falloff = clamp((radius - dist) / radius, 0.0, 1.0);
          let h = dot(delta / dist, view_dir);
          max_cos_neg = max(max_cos_neg, mix(-1.0, h, falloff));
        }
      }
    }

    let h_lo = gamma - HALF_PI;
    let h_hi = gamma + HALF_PI;
    let theta2 = clamp(acos(clamp(max_cos_pos, -1.0, 1.0)), h_lo, h_hi);
    let theta1 = clamp(-acos(clamp(max_cos_neg, -1.0, 1.0)), h_lo, h_hi);
    let full_slice = max(cos(gamma) + gamma * sin(gamma), 1e-4);
    let slice_vis = clamp((gtao_arc(theta1, gamma) + gtao_arc(theta2, gamma)) / full_slice, 0.0, 1.0);
    visibility += slice_vis;
  }

  let ao = clamp(visibility / f32(slices), 0.0, 1.0);
  return vec4f(pow(ao, occlusion.settings.y));
}
