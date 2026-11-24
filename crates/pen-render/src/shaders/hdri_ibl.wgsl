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

fn rotate_y(d: vec3f, angle: f32) -> vec3f {
  let c = cos(angle);
  let s = sin(angle);
  return vec3f(d.x * c + d.z * s, d.y, -d.x * s + d.z * c);
}

fn sample_equirect(dir: vec3f) -> vec3f {
  let d = rotate_y(normalize(dir), params.rotation_rad);
  let phi = atan2(d.z, d.x);
  let u = fract(phi / TWO_PI + 0.5);
  let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
  let rgb = textureSampleLevel(equirect, equirect_sampler, vec2f(u, v), 0.0).rgb;
  // Clamp extreme fireflies to finite f16 headroom.
  return clamp(rgb * params.intensity, vec3f(0.0), vec3f(65000.0));
}

fn radical_inverse_vdc(bits_in: u32) -> f32 {
  var bits = bits_in;
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, n: u32) -> vec2f {
  return vec2f(f32(i) / f32(max(n, 1u)), radical_inverse_vdc(i));
}

fn importance_sample_ggx(xi: vec2f, n: vec3f, roughness: f32) -> vec3f {
  let a = roughness * roughness;
  let phi = TWO_PI * xi.x;
  let cos_theta = sqrt(max((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y), 0.0));
  let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
  let h_local = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
  let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);
  let tangent = normalize(cross(up, n));
  let bitangent = cross(n, tangent);
  return normalize(tangent * h_local.x + bitangent * h_local.y + n * h_local.z);
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

  if (params.roughness <= 0.001 || params.samples <= 1u) {
    let color = sample_equirect(r);
    textureStore(out_face, vec2i(gid.xy), vec4f(color, 1.0));
    return;
  }

  var prefiltered = vec3f(0.0);
  var total_weight = 0.0;
  let n_samples = max(params.samples, 1u);
  for (var i = 0u; i < n_samples; i = i + 1u) {
    let xi = hammersley(i, n_samples);
    let h = importance_sample_ggx(xi, n, params.roughness);
    let l = normalize(2.0 * dot(v, h) * h - v);
    let nl = max(dot(n, l), 0.0);
    if (nl > 0.0) {
      prefiltered += sample_equirect(l) * nl;
      total_weight += nl;
    }
  }
  let result = select(sample_equirect(n), prefiltered / total_weight, total_weight > 1e-5);
  textureStore(out_face, vec2i(gid.xy), vec4f(result, 1.0));
}
