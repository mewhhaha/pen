struct Present {
  output: vec4f,
  source: vec4f,
  split: vec4f,
  // x: how much of the blur's sum of levels to add, y: how much of the picture
  // to keep.
  bloom: vec4f,
};
@group(0) @binding(0) var<uniform> present: Present;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var source_sampler: sampler;
// The scene's stencil: bit 1 where a surface asks for the frame's aberration.
@group(0) @binding(3) var marks: ${marks_texture};
@group(0) @binding(4) var glow: texture_2d<f32>;
// The color grade, and the exposure in stops that the frame's metering or its
// fixed setting left in the first lane.
@group(0) @binding(5) var<uniform> grade: Grade;
@group(0) @binding(6) var<storage, read> exposure: array<f32, 4>;
// The grade's look-up tables, or stand-ins while it has none.
@group(1) @binding(0) var lut_a: texture_2d<f32>;
@group(1) @binding(1) var lut_b: texture_2d<f32>;
${fullscreen}
${transfer}
${grade}
${tonedWgsl}

fn marked(uv: vec2f) -> bool {
  let size = present.split.zw;
  let pixel = clamp(vec2i(uv * size), vec2i(0), vec2i(size) - 1);
  return (textureLoad(marks, pixel, 0).x & 1u) != 0u;
}

// The picture at uv as the screen shows it: the scene's light as the camera
// exposes it and its bloom (drawn from the exposed light), graded, then toned.
// A picture already shown (a scene filter's) is kept as it is.
fn fetch(uv: vec2f) -> vec3f {
  let scene = textureSampleLevel(source, source_sampler, uv, 0.0).rgb;
  let glow_light = textureSampleLevel(glow, source_sampler, uv, 0.0).rgb * present.bloom.x;
  if (present.source.w > 0.5) { return scene; }
  var light = exposed(scene, exposure[0]);
  if (present.bloom.x > 0.0) {
    // Glare is light scattered, so the picture and its blur mix in linear light:
    // the blur takes `bloom.x` of the picture's light in all, and what it takes,
    // the picture gives up (`bloom.y`).
    light = srgb_encode(srgb_decode(light) * present.bloom.y + glow_light);
  }
  if (grade.luts.w > 0.5) {
    light = srgb_encode(grade_apply(grade, lut_a, lut_b, srgb_decode(light), uv));
  }
  let tonemap_mode = u32(max(grade.vignette.z, 0.0) + 0.5);
  let hdr_peak = max(grade.vignette.w, 1.0);
  if (tonemap_mode == 0u) {
    return toned(light);
  }
  return toned_with_mode(light, tonemap_mode, hdr_peak);
}

fn luma(color: vec3f) -> f32 {
  return dot(color, vec3f(0.299, 0.587, 0.114));
}

${fxaa}

fn shade(uv: vec2f) -> vec3f {
  if (present.source.z > 0.5) {
    return fxaa(uv);
  }
  return fetch(uv);
}

// The picture smears toward the middle of the screen, over the share of the
// way there present.output.z says, as speed does. The marked surfaces' red
// moves one way and their blue the other, over whatever lies beside them;
// green stays in place.
@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let uv = position.xy * present.output.xy;
  var base = shade(uv);
  let blur = present.output.z;
  if (blur > 0.0) {
    var sum = base;
    for (var sample = 1; sample < 12; sample++) {
      sum += fetch(mix(uv, vec2f(0.5), f32(sample) / 11.0 * blur));
    }
    base = sum / 12.0;
  }
  let split = present.split.xy;
  if (all(split == vec2f(0.0))) {
    return vec4f(base, 1.0);
  }
  let red = select(base.r, shade(uv - split).r, marked(uv - split));
  let blue = select(base.b, shade(uv + split).b, marked(uv + split));
  return vec4f(red, base.g, blue, 1.0);
}
