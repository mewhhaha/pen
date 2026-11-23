// The split-sum BRDF table, computed once: for a surface of roughness v
// (down) seen at cosine u (across) from its normal, the share of white light
// a GGX lobe with height-correlated Smith visibility sends back, as
// scale * f0 + bias for Schlick's Fresnel. The lights use the same lobe.
@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
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

const TABLE_SIZE = 32.0;
const SAMPLES = 512u;

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let nv = max((position.x) / TABLE_SIZE, 0.0001);
  let roughness = clamp(position.y / TABLE_SIZE, 0.04, 1.0);
  let a = roughness * roughness;
  let a2 = a * a;
  let view = vec3f(sqrt(1.0 - nv * nv), 0.0, nv);
  var scale = 0.0;
  var bias = 0.0;
  for (var i = 0u; i < SAMPLES; i++) {
    let u = vec2f((f32(i) + 0.5) / f32(SAMPLES), radical_inverse(i));
    let phi = 2.0 * 3.14159265359 * u.x;
    let cos_theta = sqrt((1.0 - u.y) / (1.0 + (a2 - 1.0) * u.y));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let half_way = vec3f(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let light = 2.0 * dot(view, half_way) * half_way - view;
    let nl = clamp(light.z, 0.0, 1.0);
    let nh = clamp(half_way.z, 0.0, 1.0);
    let vh = clamp(dot(view, half_way), 0.0, 1.0);
    if (nl > 0.0) {
      let visibility = 0.5 / max(0.000001, nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2));
      // D * NoL / (4 * pdf) with pdf = D * NoH / (4 * VoH).
      let weight = visibility * nl * 4.0 * vh / max(nh, 0.0001);
      let fresnel = pow(1.0 - vh, 5.0);
      scale += (1.0 - fresnel) * weight;
      bias += fresnel * weight;
    }
  }
  return vec4f(scale / f32(SAMPLES), bias / f32(SAMPLES), 0.0, 1.0);
}
