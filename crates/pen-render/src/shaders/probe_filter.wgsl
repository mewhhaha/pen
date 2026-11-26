// A reflection probe's cube after it is drawn: `downsample` halves a face of
// the scratch cube into the next level, and `prefilter` sums GGX lobes of
// growing roughness over the scratch cube into a slot's levels (level n holds
// roughness n / (levels - 1)), each sample read from the scratch level whose
// texels span the solid angle the sample stands for (filtered importance
// sampling), so a sparse sum of a bright small thing does not speckle.
struct Pass {
  // x: the cube face, y: roughness 0 to 1, z: the samples to sum, w: the
  // face's size in texels.
  settings: vec4f,
  // x: the scratch cube's level 0 width in texels.
  source: vec4f,
};
@group(0) @binding(0) var<uniform> pass_data: Pass;
@group(0) @binding(1) var source_cube: texture_cube<f32>;
@group(0) @binding(2) var source_face: texture_2d<f32>;
@group(0) @binding(3) var source_sampler: sampler;

@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
}

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
  return f32(reverseBits(bits_in)) * 2.3283064365386963e-10;
}

@fragment fn downsample(@builtin(position) position: vec4f) -> @location(0) vec4f {
  // Halfway between four texels of the level above: their mean.
  return textureSampleLevel(source_face, source_sampler, position.xy / pass_data.settings.w, 0.0);
}

@fragment fn prefilter(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let size = pass_data.settings.w;
  let normal = normalize(face_direction(u32(pass_data.settings.x), position.xy / size));
  let roughness = pass_data.settings.y;
  // GGX with alpha = roughness squared, as the lights have it.
  let a = roughness * roughness;
  let a2 = a * a;
  let tangent = normalize(cross(select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.999), normal));
  let bitangent = cross(normal, tangent);
  let samples = u32(pass_data.settings.z);
  let texel_angle = 4.0 * 3.14159265359 / (6.0 * pass_data.source.x * pass_data.source.x);
  var total = vec4f(0.0);
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
      // The density of this sample over the sphere, D / 4 with the view on the normal.
      let denominator = cos_theta * cos_theta * (a2 - 1.0) + 1.0;
      let density = a2 / (3.14159265359 * denominator * denominator) / 4.0;
      let level = max(0.5 * log2(1.0 / (f32(samples) * density * texel_angle)) + 1.0, 0.0);
      total += textureSampleLevel(source_cube, source_sampler, light, level) * facing;
      weight += facing;
    }
  }
  return total / max(weight, 0.0001);
}
