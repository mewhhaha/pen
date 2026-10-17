struct Taa {
  // This frame's jittered clip space back into the world.
  inverse: mat4x4f,
  // The world into the last frame's clip space, unjittered.
  previous: mat4x4f,
  // xy: this frame's jitter in texture coordinates, across and down; z: its
  // share of the blend; w: 1 when there is no history yet.
  blend: vec4f,
  // xy: the picture's size in pixels.
  size: vec4f,
};
@group(0) @binding(0) var<uniform> taa: Taa;
@group(0) @binding(1) var current: texture_2d<f32>;
@group(0) @binding(2) var history: texture_2d<f32>;
@group(0) @binding(3) var depth: ${depth_texture};
@group(0) @binding(4) var filtering: sampler;
${fullscreen}

fn luma(color: vec3f) -> f32 {
  return dot(color, vec3f(0.2126, 0.7152, 0.0722));
}

// Light squeezed below 1 (and back), so a blazing highlight weighs no more
// in a blend than a white one and does not flicker.
fn squeezed(color: vec3f) -> vec3f {
  return color / (1.0 + luma(color));
}
fn unsqueezed(color: vec3f) -> vec3f {
  return color / max(1.0 - luma(color), 0.0001);
}

// Luma and two chroma axes, where a neighbourhood's colours make a tighter box.
fn ycocg(color: vec3f) -> vec3f {
  return vec3f(dot(color, vec3f(0.25, 0.5, 0.25)), dot(color, vec3f(0.5, 0.0, -0.5)), dot(color, vec3f(-0.25, 0.5, -0.25)));
}
fn rgb(color: vec3f) -> vec3f {
  return vec3f(color.x + color.y - color.z, color.x + color.z, color.x - color.y - color.z);
}

// The history at uv through a Catmull-Rom filter in five bilinear taps, which
// keeps it sharp where a bilinear tap would blur it a little every frame.
fn history_at(uv: vec2f) -> vec3f {
  let size = taa.size.xy;
  let position = uv * size;
  let center = floor(position - 0.5) + 0.5;
  let f = position - center;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = -0.5 * f3 + f2 - 0.5 * f;
  let w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
  let w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
  let w3 = 0.5 * f3 - 0.5 * f2;
  let w12 = w1 + w2;
  let at0 = (center - 1.0) / size;
  let at3 = (center + 2.0) / size;
  let at12 = (center + w2 / w12) / size;
  let sum = textureSampleLevel(history, filtering, vec2f(at12.x, at0.y), 0.0).rgb * (w12.x * w0.y)
    + textureSampleLevel(history, filtering, vec2f(at0.x, at12.y), 0.0).rgb * (w0.x * w12.y)
    + textureSampleLevel(history, filtering, at12, 0.0).rgb * (w12.x * w12.y)
    + textureSampleLevel(history, filtering, vec2f(at3.x, at12.y), 0.0).rgb * (w3.x * w12.y)
    + textureSampleLevel(history, filtering, vec2f(at12.x, at3.y), 0.0).rgb * (w12.x * w3.y);
  let weight = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  return max(sum / weight, vec3f(0.0));
}

// The history pulled toward the middle of the box until it is inside it.
fn clipped(color: vec3f, low: vec3f, high: vec3f) -> vec3f {
  let middle = 0.5 * (high + low);
  let reach = 0.5 * (high - low) + vec3f(0.0001);
  let offset = color - middle;
  let units = abs(offset / reach);
  let most = max(units.x, max(units.y, units.z));
  return select(color, middle + offset / most, most > 1.0);
}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let size = taa.size.xy;
  let pixel = vec2i(position.xy);
  let last = vec2i(size) - 1;
  // The neighbourhood of this frame: its colours' range and spread, and the
  // nearest depth in it, so an edge follows the surface in front.
  var low = vec3f(1e9);
  var high = vec3f(-1e9);
  var mean = vec3f(0.0);
  var square = vec3f(0.0);
  var middle = vec3f(0.0);
  var nearest = 1.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let at = clamp(pixel + vec2i(x, y), vec2i(0), last);
      let color = ycocg(squeezed(textureLoad(current, at, 0).rgb));
      if (x == 0 && y == 0) {
        middle = color;
      }
      low = min(low, color);
      high = max(high, color);
      mean += color;
      square += color * color;
      nearest = min(nearest, textureLoad(depth, at, 0));
    }
  }
  mean /= 9.0;
  let spread = sqrt(max(square / 9.0 - mean * mean, vec3f(0.0))) * 1.25;
  // Where that surface was in the last frame. The pixel's sample sat the
  // jitter off its middle, so the surface's place is shifted back by as
  // much: a still picture reads its history where it is, not resampled.
  let uv = position.xy / size;
  let world = taa.inverse * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, nearest, 1.0);
  let before = taa.previous * vec4f(world.xyz / world.w, 1.0);
  let was = vec2f(before.x, -before.y) / before.w * 0.5 + 0.5 + taa.blend.xy;
  let gone = before.w <= 0.0 || any(was < vec2f(0.0)) || any(was > vec2f(1.0));
  if (taa.blend.w > 0.5 || gone) {
    return vec4f(unsqueezed(rgb(middle)), 1.0);
  }
  let remembered = clipped(ycocg(squeezed(history_at(was))), max(low, mean - spread), min(high, mean + spread));
  return vec4f(unsqueezed(rgb(mix(remembered, middle, taa.blend.z))), 1.0);
}
