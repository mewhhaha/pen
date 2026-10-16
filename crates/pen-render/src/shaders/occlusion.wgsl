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

const kernel = array<vec3f, 12>(${kernel});

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

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let depth = depth_at(vec2i(position.xy));
  if (depth >= 1.0) {
    return vec4f(1.0);
  }
  let middle = view_at(position.xy, depth);
  // The normal from the nearer neighbour on each axis, so edges keep theirs.
  let right = neighbour(position.xy, vec2i(1, 0)) - middle;
  let left = middle - neighbour(position.xy, vec2i(-1, 0));
  let down = neighbour(position.xy, vec2i(0, 1)) - middle;
  let up = middle - neighbour(position.xy, vec2i(0, -1));
  let across = select(left, right, abs(right.z) < abs(left.z));
  let along = select(up, down, abs(down.z) < abs(up.z));
  var normal = normalize(cross(across, along));
  if (dot(normal, middle) > 0.0) {
    normal = -normal;
  }
  // A different turn of the kernel at each pixel of a 4 x 4 tile.
  let turn = fract(52.9829189 * fract(dot(floor(position.xy), vec2f(0.06711056, 0.00583715)))) * 6.2831853;
  let spin = vec3f(cos(turn), sin(turn), 0.0);
  let tangent = normalize(spin - normal * dot(spin, normal) + vec3f(0.0, 0.0, 1e-4));
  let bitangent = cross(normal, tangent);
  let radius = occlusion.settings.x;
  var filled = 0.0;
  for (var index = 0; index < 12; index++) {
    let offset = kernel[index];
    let probe = middle + (tangent * offset.x + bitangent * offset.y + normal * offset.z) * radius;
    let clip = occlusion.projection * vec4f(probe, 1.0);
    let ndc = clip.xy / clip.w;
    let at = vec2f((ndc.x * 0.5 + 0.5) * occlusion.size.x, (0.5 - ndc.y * 0.5) * occlusion.size.y);
    let found = view_at(at, depth_at(vec2i(at))).z;
    // Filled where the scene lies in front of the probe, nearby.
    let near = smoothstep(0.0, 1.0, radius / max(abs(middle.z - found), 1e-4));
    filled += select(0.0, near, found >= probe.z + occlusion.settings.z);
  }
  return vec4f(pow(1.0 - filled / 12.0, occlusion.settings.y));
}
