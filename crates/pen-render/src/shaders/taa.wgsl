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
@group(0) @binding(5) var velocity: texture_2d<f32>;
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
  let out_size = taa.size.xy;
  let scene_size = select(out_size, taa.size.zw, taa.size.z > 0.5 && taa.size.w > 0.5);
  let is_upscale = any(abs(out_size - scene_size) > vec2f(0.5));
  let uv = position.xy / out_size;
  let sample_uv = clamp(uv + taa.blend.xy, vec2f(0.0), vec2f(1.0));
  let scene_last = vec2i(scene_size) - 1;
  let pixel = select(vec2i(position.xy), clamp(vec2i(floor(sample_uv * scene_size)), vec2i(0), scene_last), is_upscale);
  // The neighbourhood of this frame: its colours' range and spread, and the
  // nearest depth in it, so an edge follows the surface in front.
  var low = vec3f(1e9);
  var high = vec3f(-1e9);
  var mean = vec3f(0.0);
  var square = vec3f(0.0);
  var middle = vec3f(0.0);
  var nearest = 1.0;
  var nearest_at = pixel;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let at = clamp(pixel + vec2i(x, y), vec2i(0), scene_last);
      let color = ycocg(squeezed(textureLoad(current, at, 0).rgb));
      if (x == 0 && y == 0) {
        middle = color;
      }
      low = min(low, color);
      high = max(high, color);
      mean += color;
      square += color * color;
      let d = textureLoad(depth, at, 0);
      if (d < nearest) {
        nearest = d;
        nearest_at = at;
      }
    }
  }
  if (is_upscale) {
    middle = clamp(ycocg(squeezed(textureSampleLevel(current, filtering, sample_uv, 0.0).rgb)), low, high);
  }
  mean /= 9.0;
  let spread = sqrt(max(square / 9.0 - mean * mean, vec3f(0.0))) * 1.25;
  // Where that surface was in the last frame. When a velocity vector is present
  // at the nearest depth in the 3x3 neighbourhood, follow it directly (covering
  // rigid, skinned and morphed motion); otherwise fall back to camera depth
  // reprojection. In temporal-upscaling mode (`is_upscale`), `sample_uv` maps
  // the unjittered output pixel to its jittered scene coordinate so `before`
  // lands directly on the unjittered full-res history UV.
  let obj_vel = textureLoad(velocity, nearest_at, 0).xy;
  var was = uv;
  var gone = false;
  if (abs(obj_vel.x) < 500.0) {
    was = uv - obj_vel;
    gone = any(was < vec2f(0.0)) || any(was > vec2f(1.0));
  } else {
    let reproj_uv = select(uv, sample_uv, is_upscale);
    let world = taa.inverse * vec4f(reproj_uv.x * 2.0 - 1.0, 1.0 - reproj_uv.y * 2.0, nearest, 1.0);
    let before = taa.previous * vec4f(world.xyz / world.w, 1.0);
    was = vec2f(before.x, -before.y) / before.w * 0.5 + 0.5 + select(taa.blend.xy, vec2f(0.0), is_upscale);
    gone = before.w <= 0.0 || any(was < vec2f(0.0)) || any(was > vec2f(1.0));
  }
  if (taa.blend.w > 0.5 || gone) {
    return vec4f(unsqueezed(rgb(middle)), 1.0);
  }
  let remembered = clipped(ycocg(squeezed(history_at(was))), max(low, mean - spread), min(high, mean + spread));
  return vec4f(unsqueezed(rgb(mix(remembered, middle, taa.blend.z))), 1.0);
}
