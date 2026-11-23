// One face of one level of the sky's reflection cube. The sky is the game's
// three-color gradient (the horizon to the zenith or the ground by the square
// root of the height, as ink/environment.sky has it, without the sun that
// the directional light brings); level 0 is the sky itself and each coarser
// level the GGX lobe of a rougher surface over it, summed by importance
// sampling with the view along the normal.
struct Pass {
  zenith: vec4f,
  horizon: vec4f,
  ground: vec4f,
  // x: the cube face (+x, -x, +y, -y, +z, -z), y: roughness 0 to 1, z: the
  // samples to sum, w: the face's size in texels.
  settings: vec4f,
};
@group(0) @binding(0) var<uniform> pass_data: Pass;

@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
}

fn sky(direction: vec3f) -> vec3f {
  let up = clamp(direction.y, -1.0, 1.0);
  if (up >= 0.0) {
    return mix(pass_data.horizon.rgb, pass_data.zenith.rgb, sqrt(up));
  }
  return mix(pass_data.horizon.rgb, pass_data.ground.rgb, sqrt(-up));
}

// The direction a cube face's texel center looks along (WebGPU's face
// orientation: u across, v down the face).
fn face_direction(face: u32, uv: vec2f) -> vec3f {
  let sc = uv.x * 2.0 - 1.0;
  let tc = uv.y * 2.0 - 1.0;
  switch (face) {
    case 0u: { return vec3f(1.0, -tc, -sc); }
    case 1u: { return vec3f(-1.0, -tc, sc); }
    case 2u: { return vec3f(sc, 1.0, tc); }
    case 3u: { return vec3f(sc, -1.0, -tc); }
    case 4u: { return vec3f(sc, -tc, 1.0); }
    default: { return vec3f(-sc, -tc, -1.0); }
  }
}

fn radical_inverse(bits_in: u32) -> f32 {
  var bits = bits_in;
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return f32(bits) * 2.3283064365386963e-10;
}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let size = pass_data.settings.w;
  let normal = normalize(face_direction(u32(pass_data.settings.x), position.xy / size));
  let roughness = pass_data.settings.y;
  if (roughness <= 0.0) {
    return vec4f(sky(normal), 1.0);
  }
  // GGX with alpha = roughness squared, as the lights have it.
  let a = roughness * roughness;
  let a2 = a * a;
  let tangent = normalize(cross(select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.999), normal));
  let bitangent = cross(normal, tangent);
  let samples = u32(pass_data.settings.z);
  var total = vec3f(0.0);
  var weight = 0.0;
  for (var i = 0u; i < samples; i++) {
    let u = vec2f((f32(i) + 0.5) / f32(samples), radical_inverse(i));
    let phi = 2.0 * 3.14159265359 * u.x;
    let cos_theta = sqrt((1.0 - u.y) / (1.0 + (a2 - 1.0) * u.y));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let half_way = tangent * (sin_theta * cos(phi)) + bitangent * (sin_theta * sin(phi)) + normal * cos_theta;
    // The view is the normal, so the light is its reflection in the half way.
    let light = 2.0 * dot(normal, half_way) * half_way - normal;
    let facing = dot(normal, light);
    if (facing > 0.0) {
      total += sky(light) * facing;
      weight += facing;
    }
  }
  return vec4f(total / max(weight, 0.0001), 1.0);
}
