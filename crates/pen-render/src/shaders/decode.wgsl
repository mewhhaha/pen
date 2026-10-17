// The picture frames drew, sRGB-encoded with highlights past 1, decoded for
// an HDR surface: to linear light in nits over scRGB's 80 (scRGB), or turned
// into BT.2020's primaries and encoded with the PQ curve over 10,000 nits
// (BT.2100 PQ, HDR10).
struct Decode {
  // x: SDR white in nits; y: 1 for PQ.
  settings: vec4f,
};
@group(0) @binding(0) var<uniform> decode: Decode;
@group(0) @binding(1) var picture: texture_2d<f32>;
${fullscreen}

fn linear(encoded: vec3f) -> vec3f {
  let magnitude = abs(encoded);
  let curve = select(pow((magnitude + 0.055) / 1.055, vec3f(2.4)), magnitude / 12.92, magnitude <= vec3f(0.04045));
  return sign(encoded) * curve;
}

fn pq(nits: vec3f) -> vec3f {
  let m1 = 0.1593017578125;
  let m2 = 78.84375;
  let c1 = 0.8359375;
  let c2 = 18.8515625;
  let c3 = 18.6875;
  let y = pow(clamp(nits / 10000.0, vec3f(0.0), vec3f(1.0)), vec3f(m1));
  return pow((c1 + c2 * y) / (1.0 + c3 * y), vec3f(m2));
}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let light = linear(textureLoad(picture, vec2i(position.xy), 0).rgb);
  let white = decode.settings.x;
  if (decode.settings.y > 0.5) {
    // BT.709 primaries into BT.2020's (BT.2087).
    let wide = mat3x3f(
      vec3f(0.6274, 0.0691, 0.0164),
      vec3f(0.3293, 0.9195, 0.0880),
      vec3f(0.0433, 0.0114, 0.8956),
    ) * light;
    return vec4f(pq(max(wide, vec3f(0.0)) * white), 1.0);
  }
  return vec4f(light * white / 80.0, 1.0);
}
