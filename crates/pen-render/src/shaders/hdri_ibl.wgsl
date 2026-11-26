// Equirectangular HDRI -> prefiltered GGX specular cubemap mip chain.
// Dispatched per (face, mip) using a compute shader over Rgba16Float storage layers.

struct PrefilterParams {
  face: u32,
  size: u32,
  samples: u32,
  _pad: u32,
  roughness: f32,
  intensity: f32,
  rotation_rad: f32,
  _pad2: f32,
};

@group(0) @binding(0) var<uniform> params: PrefilterParams;
@group(0) @binding(1) var equirect: texture_2d<f32>;
@group(0) @binding(2) var equirect_sampler: sampler;
@group(0) @binding(3) var out_face: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

fn face_direction(face: u32, uv: vec2f) -> vec3f {
  // WebGPU cubemap face orientation (+X, -X, +Y, -Y, +Z, -Z), uv in [-1, 1].
  let u = uv.x;
  let v = uv.y;
  switch (face) {
    case 0u: { return normalize(vec3f( 1.0, -v, -u)); }
    case 1u: { return normalize(vec3f(-1.0, -v,  u)); }
    case 2u: { return normalize(vec3f( u,  1.0,  v)); }
    case 3u: { return normalize(vec3f( u, -1.0, -v)); }
    case 4u: { return normalize(vec3f( u, -v,  1.0)); }
    default: { return normalize(vec3f(-u, -v, -1.0)); }
  }
}

fn sample_equirect_cs(dir: vec3f, rot_c: f32, rot_s: f32) -> vec3f {
  let d = vec3f(dir.x * rot_c + dir.z * rot_s, dir.y, -dir.x * rot_s + dir.z * rot_c);
  let phi = atan2(d.z, d.x);
  let u = fract(phi * (1.0 / TWO_PI) + 0.5);
  let v = acos(clamp(d.y, -1.0, 1.0)) * (1.0 / PI);
  let rgb = textureSampleLevel(equirect, equirect_sampler, vec2f(u, v), 0.0).rgb;
  // Clamp extreme fireflies to finite f16 headroom.
  return clamp(rgb * params.intensity, vec3f(0.0), vec3f(65000.0));
}

fn radical_inverse_vdc(bits_in: u32) -> f32 {
  return f32(reverseBits(bits_in)) * 2.3283064365386963e-10;
}

@compute @workgroup_size(8, 8, 1)
fn prefilter_main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.size || gid.y >= params.size) {
    return;
  }
  let uv = (vec2f(f32(gid.x) + 0.5, f32(gid.y) + 0.5) / f32(params.size)) * 2.0 - vec2f(1.0);
  let n = face_direction(params.face, uv);
  let r = n;
  let v = r;
  let rot_c = cos(params.rotation_rad);
  let rot_s = sin(params.rotation_rad);

  if (params.roughness <= 0.001 || params.samples <= 1u) {
    let color = sample_equirect_cs(r, rot_c, rot_s);
    textureStore(out_face, vec2i(gid.xy), vec4f(color, 1.0));
    return;
  }

  let a = params.roughness * params.roughness;
  let a2_m1 = a * a - 1.0;
  let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);
  let tangent = normalize(cross(up, n));
  let bitangent = cross(n, tangent);

  var prefiltered = vec3f(0.0);
  var total_weight = 0.0;
  let n_samples = max(params.samples, 1u);
  let inv_samples = 1.0 / f32(n_samples);
  for (var i = 0u; i < n_samples; i = i + 1u) {
    let xi = vec2f(f32(i) * inv_samples, radical_inverse_vdc(i));
    let phi = TWO_PI * xi.x;
    let cos_theta = sqrt(max((1.0 - xi.y) / (1.0 + a2_m1 * xi.y), 0.0));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let h = normalize(tangent * (cos(phi) * sin_theta) + bitangent * (sin(phi) * sin_theta) + n * cos_theta);
    let l = normalize(2.0 * dot(v, h) * h - v);
    let nl = max(dot(n, l), 0.0);
    if (nl > 0.0) {
      prefiltered += sample_equirect_cs(l, rot_c, rot_s) * nl;
      total_weight += nl;
    }
  }
  let result = select(sample_equirect_cs(n, rot_c, rot_s), prefiltered / total_weight, total_weight > 1e-5);
  textureStore(out_face, vec2i(gid.xy), vec4f(result, 1.0));
}
