// The color grade: what the present pass does to the exposed light before it is
// toned. It works in linear light and in a log encoding of it (ACEScc: the
// Academy's S-2014-003 curve on the frame's own primaries, so no gamut change),
// where equal steps are equal stops, so contrast turns about a stop rather than a
// number and a look-up table covers 17 stops in 64 texels at most. The same
// formula is Ink's `color.grade` (Blot) and the f64 reference in this crate's
// tests, term for term.
//
// Order: vignette, exposure offset and white balance, contrast about the pivot in
// log, ASC CDL (slope, offset, power) in linear, saturation and vibrance, then the
// look-up tables in log. With nothing set it is the identity.
struct Grade {
  // x temperature, y tint, z exposure offset in stops, w contrast.
  basics: vec4f,
  // x contrast pivot (linear), y saturation, z vibrance, w the tables' weight.
  tone: vec4f,
  // ASC CDL: the offset, the gamma (the power is its reciprocal), the slope.
  lift: vec4f,
  gamma: vec4f,
  gain: vec4f,
  // x the first table's size (0 for none), y the second's, z their blend, w 1 when
  // the grade applies at all.
  luts: vec4f,
  // x the corners' darkening, y where it starts.
  vignette: vec4f,
}

const CC_LOW = -0.358447488584; // (-16 + 9.72) / 17.52, which black encodes to
const CC_HIGH = 1.46799631204; // (log2(65504) + 9.72) / 17.52
const CC_BREAK = -0.301369863; // (9.72 - 15) / 17.52, where the curve's foot ends

fn cc_encode_one(x: f32) -> f32 {
  if (x <= 0.0) { return CC_LOW; }
  if (x < 3.0517578125e-05) { return (log2(1.52587890625e-05 + x * 0.5) + 9.72) / 17.52; }
  return (log2(x) + 9.72) / 17.52;
}

fn cc_decode_one(v: f32) -> f32 {
  if (v < CC_BREAK) { return max((exp2(v * 17.52 - 9.72) - 1.52587890625e-05) * 2.0, 0.0); }
  return min(exp2(v * 17.52 - 9.72), 65504.0);
}

fn cc_encode(light: vec3f) -> vec3f {
  return vec3f(cc_encode_one(light.x), cc_encode_one(light.y), cc_encode_one(light.z));
}

fn cc_decode(v: vec3f) -> vec3f {
  return vec3f(cc_decode_one(v.x), cc_decode_one(v.y), cc_decode_one(v.z));
}

// How much of the light the vignette leaves at `uv`: 1 inside `radius` of the way
// from the middle to a corner, falling smoothly to 1 - strength at the corner.
fn vignette_factor(strength: f32, radius: f32, uv: vec2f) -> f32 {
  let d = length(uv - vec2f(0.5)) * 1.41421356237;
  let t = clamp((d - radius) / (1.0 - radius), 0.0, 1.0);
  return 1.0 - strength * t * t * (3.0 - 2.0 * t);
}

// The table `lut`, a strip of `size` slices (blue picks the slice, red runs along
// it, green down), at `v` in 0..1 on each axis, trilinearly between its texels
// (their centres at i / (size - 1)).
fn lut_lookup(lut: texture_2d<f32>, size: i32, v: vec3f) -> vec3f {
  let top = size - 1;
  let p = clamp(v, vec3f(0.0), vec3f(1.0)) * f32(top);
  let base = floor(p);
  let f = p - base;
  let i0 = min(vec3i(base), vec3i(top));
  let i1 = min(i0 + 1, vec3i(top));
  let c000 = textureLoad(lut, vec2i(i0.z * size + i0.x, i0.y), 0).rgb;
  let c100 = textureLoad(lut, vec2i(i0.z * size + i1.x, i0.y), 0).rgb;
  let c010 = textureLoad(lut, vec2i(i0.z * size + i0.x, i1.y), 0).rgb;
  let c110 = textureLoad(lut, vec2i(i0.z * size + i1.x, i1.y), 0).rgb;
  let c001 = textureLoad(lut, vec2i(i1.z * size + i0.x, i0.y), 0).rgb;
  let c101 = textureLoad(lut, vec2i(i1.z * size + i1.x, i0.y), 0).rgb;
  let c011 = textureLoad(lut, vec2i(i1.z * size + i0.x, i1.y), 0).rgb;
  let c111 = textureLoad(lut, vec2i(i1.z * size + i1.x, i1.y), 0).rgb;
  let x00 = mix(c000, c100, f.x);
  let x10 = mix(c010, c110, f.x);
  let x01 = mix(c001, c101, f.x);
  let x11 = mix(c011, c111, f.x);
  return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

// Linear light `light` at `uv` of the picture, graded.
fn grade_apply(g: Grade, lut_a: texture_2d<f32>, lut_b: texture_2d<f32>, light: vec3f, uv: vec2f) -> vec3f {
  var c = max(light, vec3f(0.0));
  c = c * vignette_factor(g.vignette.x, min(g.vignette.y, 0.999), uv);
  // White balance moves red and blue against each other along temperature, and
  // green against both along tint, summing to no change in stops.
  let t = g.basics.x;
  let s = g.basics.y;
  c = c * exp2(vec3f(0.5 * t + 0.25 * s, -0.5 * s, -0.5 * t + 0.25 * s) + g.basics.z);
  // Contrast turns the log encoding about the pivot's.
  let pivot = cc_encode_one(g.tone.x);
  c = cc_decode(clamp((cc_encode(c) - pivot) * g.basics.w + pivot, vec3f(CC_LOW), vec3f(CC_HIGH)));
  // ASC CDL, per channel.
  c = pow(max(c * g.gain.xyz + g.lift.xyz, vec3f(0.0)), 1.0 / g.gamma.xyz);
  // Saturation about the luminance, and vibrance, which weighs it by how dull
  // (positive) or how vivid (negative) the color is.
  let y = luminance(c);
  let high = max(c.x, max(c.y, c.z));
  let low = min(c.x, min(c.y, c.z));
  let chroma = select(0.0, (high - low) / high, high > 1e-6);
  let weight = select(chroma, 1.0 - chroma, g.tone.z > 0.0);
  let k = g.tone.y * (1.0 + g.tone.z * weight);
  c = max(c * k + vec3f(y * (1.0 - k)), vec3f(0.0));
  // The tables, in log: what they do to the part of the encoding within 0..1, the
  // rest kept as it is, so black and the brightest highlights pass through.
  if (g.luts.x > 0.0 && g.tone.w > 0.0) {
    let v = cc_encode(c);
    let inside = clamp(v, vec3f(0.0), vec3f(1.0));
    var looked = lut_lookup(lut_a, i32(g.luts.x), inside);
    if (g.luts.y > 0.0 && g.luts.z > 0.0) {
      looked = mix(looked, lut_lookup(lut_b, i32(g.luts.y), inside), g.luts.z);
    }
    c = cc_decode(v + (looked - inside) * g.tone.w);
  }
  return c;
}
