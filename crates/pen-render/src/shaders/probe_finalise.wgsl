// The last step of a reflection probe's face: the scene drawn from the probe
// (the frame's opaque surfaces, lit as the camera's are, its alpha the share of
// each pixel left to the sky) laid over the sky, mirrored, as the cube's texels
// have it. Its alpha is the mean light of the diffuse probe lattice where the
// probe sits (`ambient_mean`): the shaders that read the cube scale it by how
// far that has moved since. Added after ENVIRONMENT_WGSL, which declares
// group 0.
struct Face {
  // x: the cube face (+x, -x, +y, -y, +z, -z), y: its size in texels.
  settings: vec4f,
  // xyz: where the probe is.
  position: vec4f,
};
@group(1) @binding(0) var<uniform> face_data: Face;
@group(1) @binding(1) var captured: texture_2d<f32>;

@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
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

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let uv = position.xy / face_data.settings.y;
  let direction = normalize(face_direction(u32(face_data.settings.x), uv));
  // A right-handed camera looking down a face sees the cube's left and right swapped.
  let seen = textureSampleLevel(captured, sky_sampler, vec2f(1.0 - uv.x, uv.y), 0.0);
  var sky = vec3f(0.0);
  if (sky_light.tint.w > 0.5) {
    sky = textureSampleLevel(sky_cube, sky_sampler, direction, 0.0).rgb * sky_light.tint.rgb;
  }
  return vec4f(seen.rgb + seen.a * sky, ambient_mean(face_data.position.xyz));
}
