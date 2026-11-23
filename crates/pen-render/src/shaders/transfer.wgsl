// The sRGB transfer function, which the frame's light is in (frames draw as for an
// SDR screen, in sRGB-encoded light, see display.rs), and exposure as light: a
// multiplication of the decoded light by 2^stops. Shared by the bloom's first pass,
// the metering and the present pass. Negative values are black; values past 1
// follow the curve on, as HDR keeps them.
fn srgb_decode(color: vec3f) -> vec3f {
  let c = max(color, vec3f(0.0));
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

fn srgb_encode(light: vec3f) -> vec3f {
  let l = max(light, vec3f(0.0));
  return select(1.055 * pow(l, vec3f(1.0 / 2.4)) - 0.055, l * 12.92, l <= vec3f(0.0031308));
}

// `color` seen with `stops` more exposure, in the same encoding; 0 stops leave it
// exactly as it is.
fn exposed(color: vec3f, stops: f32) -> vec3f {
  if (stops == 0.0) { return color; }
  return srgb_encode(srgb_decode(color) * exp2(stops));
}

// The luminance of linear light, in the units the metering and the grade share.
fn luminance(light: vec3f) -> f32 {
  return dot(light, vec3f(0.2126, 0.7152, 0.0722));
}
