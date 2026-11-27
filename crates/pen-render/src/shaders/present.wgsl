struct Present {
  // xy: 1/output_size, z: radial blur, w: >0.5 enables AMD FidelityFX FSR 1 EASU spatial upscaling.
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

// --- AMD FidelityFX FSR 1 EASU (12-Tap Edge-Adaptive Spatial Upsampling) ---
fn easu_compress(c: vec3f) -> vec3f {
  let p = max(c, vec3f(0.0));
  return p / (p + vec3f(1.0));
}

fn easu_decompress(c: vec3f) -> vec3f {
  let clamped = clamp(c, vec3f(0.0), vec3f(0.9999));
  return clamped / max(vec3f(1.0) - clamped, vec3f(1e-4));
}

fn easu_luma(c: vec3f) -> f32 {
  return 0.25 * c.r + 0.5 * c.g + 0.25 * c.b;
}

fn easu_load(p: vec2i, last: vec2i) -> vec3f {
  return easu_compress(textureLoad(source, clamp(p, vec2i(0), last), 0).rgb);
}

fn easu_tap(
  ac: ptr<function, vec3f>,
  aw: ptr<function, f32>,
  off: vec2f,
  dir: vec2f,
  len_scale: vec2f,
  lob: f32,
  clmp: f32,
  col: vec3f,
) {
  let vx = off.x * dir.x + off.y * dir.y;
  let vy = -off.x * dir.y + off.y * dir.x;
  let sx = vx * len_scale.x;
  let sy = vy * len_scale.y;
  let d2 = min(sx * sx + sy * sy, clmp);
  var wb = 0.4 * d2 - 1.0;
  var wa = lob * d2 - 1.0;
  wb = wb * wb;
  wa = wa * wa;
  wb = 1.5625 * wb - 0.5625;
  let w = wb * wa;
  *ac += col * w;
  *aw += w;
}

fn easu_dir_set(
  dir: ptr<function, vec2f>,
  aniso_acc: ptr<function, f32>,
  w: f32,
  lb: f32,
  le: f32,
  lf: f32,
  lg: f32,
  lj: f32,
) {
  let dcx = lg - le;
  let dcy = lj - lb;
  var lx = max(abs(lg - lf), abs(lf - le));
  var ly = max(abs(lj - lf), abs(lf - lb));
  lx = clamp(abs(dcx) / max(lx, 1e-4), 0.0, 1.0);
  ly = clamp(abs(dcy) / max(ly, 1e-4), 0.0, 1.0);
  *dir += vec2f(dcx, dcy) * w;
  *aniso_acc += (lx * lx + ly * ly) * w;
}

fn easu_sample_source(uv: vec2f) -> vec3f {
  let dims = vec2f(textureDimensions(source, 0));
  let last = vec2i(dims) - vec2i(1);
  let pp_full = uv * dims - 0.5;
  let fp = vec2i(floor(pp_full));
  let pp = pp_full - vec2f(fp);

  var offsets = array<vec2i, 12>(
    vec2i(0, -1), vec2i(1, -1),
    vec2i(-1, 0), vec2i(0, 0), vec2i(1, 0), vec2i(2, 0),
    vec2i(-1, 1), vec2i(0, 1), vec2i(1, 1), vec2i(2, 1),
    vec2i(0, 2), vec2i(1, 2),
  );
  var c: array<vec3f, 12>;
  for (var k = 0u; k < 12u; k++) {
    c[k] = easu_load(fp + offsets[k], last);
  }

  var dir = vec2f(0.0);
  var aniso = 0.0;
  easu_dir_set(&dir, &aniso, (1.0 - pp.x) * (1.0 - pp.y), easu_luma(c[0]), easu_luma(c[2]), easu_luma(c[3]), easu_luma(c[4]), easu_luma(c[7]));
  easu_dir_set(&dir, &aniso, pp.x * (1.0 - pp.y), easu_luma(c[1]), easu_luma(c[3]), easu_luma(c[4]), easu_luma(c[5]), easu_luma(c[8]));
  easu_dir_set(&dir, &aniso, (1.0 - pp.x) * pp.y, easu_luma(c[3]), easu_luma(c[6]), easu_luma(c[7]), easu_luma(c[8]), easu_luma(c[10]));
  easu_dir_set(&dir, &aniso, pp.x * pp.y, easu_luma(c[4]), easu_luma(c[7]), easu_luma(c[8]), easu_luma(c[9]), easu_luma(c[11]));

  let dir2 = dot(dir, dir);
  dir = select(dir * inverseSqrt(max(dir2, 1e-6)), vec2f(1.0, 0.0), dir2 < 1e-6);
  aniso = clamp(aniso * 0.5, 0.0, 1.0);
  aniso = aniso * aniso;

  let stretch = 1.0 / max(abs(dir.x), abs(dir.y));
  let len_scale = vec2f(1.0 + (stretch - 1.0) * aniso, 1.0 - 0.5 * aniso);
  let lob = 0.5 + (0.25 - 0.04 - 0.5) * aniso;
  let clmp = 1.0 / max(lob, 0.1);

  var ac = vec3f(0.0);
  var aw = 0.0;
  for (var k = 0u; k < 12u; k++) {
    easu_tap(&ac, &aw, vec2f(offsets[k]) - pp, dir, len_scale, lob, clmp, c[k]);
  }

  let mn4 = min(min(c[3], c[4]), min(c[7], c[8]));
  let mx4 = max(max(c[3], c[4]), max(c[7], c[8]));
  return easu_decompress(clamp(ac / max(aw, 1e-5), mn4, mx4));
}

// The picture at uv as the screen shows it: the scene's light as the camera
// exposes it and its bloom (drawn from the exposed light), graded, then toned.
// A picture already shown (a scene filter's) is kept as it is.
fn grade_and_tone(uv: vec2f, scene: vec3f) -> vec3f {
  if (present.source.w > 0.5) { return scene; }
  var light = exposed(scene, exposure[0]);
  if (present.bloom.x > 0.0) {
    // Glare is light scattered, so the picture and its blur mix in linear light:
    // the blur takes `bloom.x` of the picture's light in all, and what it takes,
    // the picture gives up (`bloom.y`).
    let glow_light = textureSampleLevel(glow, source_sampler, uv, 0.0).rgb * present.bloom.x;
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

fn fetch(uv: vec2f) -> vec3f {
  var scene: vec3f;
  if (present.output.w > 0.5) {
    scene = easu_sample_source(uv);
  } else {
    scene = textureSampleLevel(source, source_sampler, uv, 0.0).rgb;
  }
  return grade_and_tone(uv, scene);
}

// Fast bilinear probe used for FXAA edge-detection neighborhood taps so EASU
// 12-tap spatial upscaling and FXAA coexist without multiplying EASU taps.
fn fetch_fast(uv: vec2f) -> vec3f {
  let scene = textureSampleLevel(source, source_sampler, uv, 0.0).rgb;
  return grade_and_tone(uv, scene);
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
  var red = base.r;
  var blue = base.b;
  for (var channel = 0u; channel < 2u; channel++) {
    let offset = select(split, -split, channel == 0u);
    let sample_uv = uv + offset;
    if (marked(sample_uv)) {
      let sample_rgb = fetch(sample_uv);
      if (channel == 0u) {
        red = sample_rgb.r;
      } else {
        blue = sample_rgb.b;
      }
    }
  }
  return vec4f(red, base.g, blue, 1.0);
}
