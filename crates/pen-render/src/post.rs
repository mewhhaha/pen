// The passes around the scene. The scene draws offscreen in floating point
// (`SCENE_FORMAT`), so light can go past white: multisampled and resolved for
// MSAA, at the render scale's size. Ambient occlusion is then worked out from
// its depth; a sky fills what the scene left empty, the occlusion darkens the
// rest and fog mists everything beyond the arena. With bloom, what is
// brighter than white spills light round it (`bloom`). A last pass adds the
// bloom, rolls the highlights off into white (`toned`), applies FXAA, splits
// the colors of the surfaces whose material asks for the frame's aberration,
// and scales the image to the screen.
//
// Most WGSL originated in gdev's post.ts; `${...}` marks where the TS template
// put a value in.
use std::collections::HashMap;

use pen_protocol::{Quality, RenderFrame};

use crate::matrix::{invert, multiply};

/// The scene's format: floating point, so its light can go past white.
pub(crate) const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// WGSL: the scene's light as the screen shows it. Up to `shoulder` each
/// channel shows as it is; above, it rolls off smoothly toward 1 without
/// reaching it, so bright surfaces go white as they did when the screen
/// clipped them, keeping their shading, and nothing hard-clips.
pub(crate) const TONED_WGSL: &str = r#"
fn toned(color: vec3f) -> vec3f {
  let shoulder = 0.9;
  let d = 1.0 - shoulder;
  let rolled = 1.0 - d * d / (max(color, vec3f(shoulder)) + d - shoulder);
  return select(color, rolled, color > vec3f(shoulder));
}"#;

const FULLSCREEN: &str = r#"
@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
}"#;

fn depth_texture(samples: u32) -> &'static str {
    if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" }
}

// The sky's colors come from the frame's lighting. Each pixel blends the sky
// and fog by how many of its samples the scene left empty, so MSAA keeps its
// smooth edges against the sky.
const ATMOSPHERE_SHADER: &str = r#"
struct Atmosphere {
  inverse_view_projection: mat4x4f,
  camera: vec4f,
  zenith: vec4f,
  horizon: vec4f,
  ground: vec4f,
  sun: vec4f,
  sun_color: vec4f,
  clear: vec4f,
  fog: vec4f,
  size: vec4f,
  // x: how much of the sky the clouds cover (0 none), yz: how far the wind
  // has carried them along x and z.
  clouds: vec4f,
};
@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var scene_color: texture_2d<f32>;
@group(0) @binding(2) var scene_depth: ${depth_texture};
@group(0) @binding(3) var occlusion_map: texture_2d<f32>;
@group(0) @binding(4) var occlusion_sampler: sampler;
${fullscreen}

// The ambient occlusion here, blurred over the 4 x 4 pixels its noise repeats
// in: four filtered reads, each averaging 2 x 2.
fn occlusion(pixel: vec2f) -> f32 {
  let texel = 1.0 / atmosphere.size.xy;
  let at = pixel * texel;
  var total = 0.0;
  for (var corner = 0; corner < 4; corner++) {
    let offset = vec2f(select(-0.5, 1.5, (corner & 1) == 1), select(-0.5, 1.5, corner > 1));
    total += textureSampleLevel(occlusion_map, occlusion_sampler, at + offset * texel, 0.0).r;
  }
  return total * 0.25;
}

fn unproject(pixel: vec2f, depth: f32) -> vec3f {
  let ndc = vec2f(pixel.x / atmosphere.size.x * 2.0 - 1.0, 1.0 - pixel.y / atmosphere.size.y * 2.0);
  let world = atmosphere.inverse_view_projection * vec4f(ndc, depth, 1.0);
  return world.xyz / world.w;
}

fn hash(p: vec3f) -> f32 {
  return fract(sin(dot(p, vec3f(127.1, 311.7, 74.7))) * 43758.5453);
}

// A few cells of a grid around the viewer hold a star.
fn stars(direction: vec3f) -> f32 {
  let p = direction * 70.0;
  let cell = floor(p);
  let chance = hash(cell);
  if (chance < 0.955) {
    return 0.0;
  }
  let star = normalize(cell + vec3f(hash(cell + 1.3), hash(cell + 2.9), hash(cell + 4.1))) * 70.0;
  let brightness = (chance - 0.955) / 0.045;
  return smoothstep(0.12, 0.0, length(p - star)) * (0.3 + 0.7 * brightness);
}

// Smooth noise over the plane, 0..1, a lattice cell a unit, and its sum over
// five octaves.
fn lattice(cell: vec2f) -> f32 {
  let c = vec2u(vec2i(cell) + vec2i(65536));
  var h = (c.x * 0x8da6b343u) ^ (c.y * 0xd8163841u);
  h ^= h >> 16u;
  h *= 0x7feb352du;
  h ^= h >> 15u;
  h *= 0x846ca68bu;
  h ^= h >> 16u;
  return f32(h >> 8u) / 16777216.0;
}
fn smooth_noise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = p - i;
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(lattice(i), lattice(i + vec2f(1.0, 0.0)), u.x),
    mix(lattice(i + vec2f(0.0, 1.0)), lattice(i + vec2f(1.0, 1.0)), u.x),
    u.y,
  );
}
fn billows(p: vec2f) -> f32 {
  var sum = 0.0;
  var weight = 0.5;
  var q = p;
  for (var octave = 0; octave < 5; octave++) {
    sum += weight * smooth_noise(q);
    q = q * 2.03 + vec2f(1.7, -3.1);
    weight *= 0.5;
  }
  return sum / 0.96875;
}

// How thick the clouds are at a point on their layer (in its own units), for
// how much of the sky they cover: none where the billows are thin, soft at
// their edges.
fn thickness(at: vec2f, cover: f32) -> f32 {
  let edge = 1.0 - cover;
  return smoothstep(edge - 0.04, edge + 0.2, billows(at));
}

// Project moving cloud cover onto world surfaces. No cloud geometry or sky
// layer is needed, and the field stays attached to the world as the camera moves.
fn cloud_shadow(point: vec3f) -> f32 {
  if (atmosphere.clouds.x<=0.0) { return 1.0; }
  let sun=atmosphere.sun.xyz;
  let at=(point.xz-point.y*sun.xz/max(sun.y,.25)+atmosphere.clouds.yz)/18.0;
  let density=thickness(at,atmosphere.clouds.x);
  let daylight=smoothstep(.15,.65,max(atmosphere.sun_color.r,atmosphere.sun_color.g));
  return 1.0-density*.42*daylight;
}

// Paler at the horizon, hazy below it, glowing toward the sun or moon. With
// lights, it shows their discs and stars. Clouds only cast shadows below.
fn sky(direction: vec3f, lights: bool) -> vec3f {
  let up = direction.y;
  var color = mix(atmosphere.horizon.rgb, atmosphere.zenith.rgb, sqrt(clamp(up, 0.0, 1.0)));
  color = mix(color, atmosphere.ground.rgb, smoothstep(0.0, 0.3, -up));
  let toward = max(dot(direction, atmosphere.sun.xyz), 0.0);
  let light = atmosphere.sun_color.rgb;
  color += light * (0.1 * pow(toward, 4.0) + 0.25 * pow(toward, 32.0));
  if (lights) {
    let above = smoothstep(-0.02, 0.02, up);
    color += light * 4.0 * smoothstep(0.99945, 0.9997, toward) * above;
    color += vec3f(stars(direction) * atmosphere.zenith.w * smoothstep(0.0, 0.15, up));
  }
  return color;
}

// The mist along the segment from a to b, whose density falls off
// exponentially with height.
fn mist_along(a: vec3f, b: vec3f) -> f32 {
  let height = atmosphere.fog.z;
  let low = max(a.y, 0.0);
  let high = max(b.y, 0.0);
  let length = distance(a, b);
  if (abs(high - low) < 0.01) {
    return length * exp(-low / height);
  }
  return length * height * (exp(-low / height) - exp(-high / height)) / (high - low);
}

// The mist between two points outside the arena: the vertical cylinder of the
// fog's radius about the origin, which stays clear.
fn mist_beyond_arena(start: vec3f, end: vec3f) -> f32 {
  let radius = atmosphere.fog.x;
  let along = end.xz - start.xz;
  let a = dot(along, along);
  let b = 2.0 * dot(start.xz, along);
  let c = dot(start.xz, start.xz) - radius * radius;
  let discriminant = b * b - 4.0 * a * c;
  if (a < 1e-6 || discriminant <= 0.0) {
    return select(0.0, mist_along(start, end), c > 0.0);
  }
  let root = sqrt(discriminant);
  let enter = clamp((-b - root) / (2.0 * a), 0.0, 1.0);
  let leave = clamp((-b + root) / (2.0 * a), 0.0, 1.0);
  return mist_along(start, mix(start, end, enter)) + mist_along(mix(start, end, leave), end);
}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(position.xy);
  let color = textureLoad(scene_color, pixel, 0).rgb;
  var empty = 0.0;
  var nearest = 1.0;
  for (var sample = 0; sample < ${samples}; sample++) {
    let depth = textureLoad(scene_depth, pixel, sample);
    if (depth >= 1.0) {
      empty += ${share};
    } else {
      nearest = min(nearest, depth);
    }
  }
  let camera = atmosphere.camera.xyz;
  let direction = normalize(unproject(position.xy, 0.5) - camera);
  let with_sky = atmosphere.camera.w > 0.5;
  let background = select(atmosphere.clear.rgb, sky(direction, true), with_sky);
  if (empty > 0.999) {
    return vec4f(background, 1.0);
  }
  // The resolved color mixes the scene with the clear color it left; take
  // the scene's part, mist it, then add the sky where the clear color was.
  let surface = (color - empty * atmosphere.clear.rgb) / (1.0 - empty) *
    mix(1.0, occlusion(position.xy), atmosphere.fog.w);
  let point = unproject(position.xy, nearest);
  let mist = select(atmosphere.clear.rgb, sky(direction, false), with_sky);
  let thickness = mist_beyond_arena(camera, point) * atmosphere.fog.y;
  let misted = mix(surface * cloud_shadow(point), mist, 1.0 - exp(-thickness));
  return vec4f(mix(misted, background, empty), 1.0);
}
"#;

fn atmosphere_shader(samples: u32) -> String {
    ATMOSPHERE_SHADER
        .replace("${depth_texture}", depth_texture(samples))
        .replace("${fullscreen}", FULLSCREEN)
        .replace("${samples}", &samples.to_string())
        .replace("${share}", &format!("{:.4}", 1.0 / f64::from(samples)))
}

// Directions around a surface's normal (z) to test for occluders, spread on a
// golden-angle spiral and gathered toward the surface.
fn kernel() -> String {
    (0..12)
        .map(|index| {
            let index = f64::from(index);
            let z = 1.0 - (index + 0.5) / 12.0 * 0.9;
            let around = (1.0 - z * z).sqrt();
            let angle = index * std::f64::consts::PI * (3.0 - 5f64.sqrt());
            let reach = 0.25 + 0.75 * ((index + 1.0) / 12.0).powi(2);
            let values = [angle.cos() * around, angle.sin() * around, z].map(|value| format!("{:.5}", value * reach));
            format!("vec3f({})", values.join(", "))
        })
        .collect::<Vec<_>>()
        .join(", ")
}

// Screen-space ambient occlusion: how much of the hemisphere above each
// surface the depth buffer shows to be filled, within a world-space radius.
const OCCLUSION_SHADER: &str = r#"
struct Occlusion {
  projection: mat4x4f,
  inverse_projection: mat4x4f,
  // radius, power, bias
  settings: vec4f,
  size: vec4f,
};
@group(0) @binding(0) var<uniform> occlusion: Occlusion;
@group(0) @binding(1) var scene_depth: ${depth_texture};
${fullscreen}

const kernel = array<vec3f, 12>(${kernel});

fn depth_at(pixel: vec2i) -> f32 {
  let size = vec2i(occlusion.size.xy);
  return textureLoad(scene_depth, clamp(pixel, vec2i(0), size - 1), 0);
}

fn view_at(pixel: vec2f, depth: f32) -> vec3f {
  let ndc = vec2f(pixel.x / occlusion.size.x * 2.0 - 1.0, 1.0 - pixel.y / occlusion.size.y * 2.0);
  let point = occlusion.inverse_projection * vec4f(ndc, depth, 1.0);
  return point.xyz / point.w;
}

fn neighbour(pixel: vec2f, step: vec2i) -> vec3f {
  return view_at(pixel + vec2f(step), depth_at(vec2i(pixel) + step));
}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let depth = depth_at(vec2i(position.xy));
  if (depth >= 1.0) {
    return vec4f(1.0);
  }
  let middle = view_at(position.xy, depth);
  // The normal from the nearer neighbour on each axis, so edges keep theirs.
  let right = neighbour(position.xy, vec2i(1, 0)) - middle;
  let left = middle - neighbour(position.xy, vec2i(-1, 0));
  let down = neighbour(position.xy, vec2i(0, 1)) - middle;
  let up = middle - neighbour(position.xy, vec2i(0, -1));
  let across = select(left, right, abs(right.z) < abs(left.z));
  let along = select(up, down, abs(down.z) < abs(up.z));
  var normal = normalize(cross(across, along));
  if (dot(normal, middle) > 0.0) {
    normal = -normal;
  }
  // A different turn of the kernel at each pixel of a 4 x 4 tile.
  let turn = fract(52.9829189 * fract(dot(floor(position.xy), vec2f(0.06711056, 0.00583715)))) * 6.2831853;
  let spin = vec3f(cos(turn), sin(turn), 0.0);
  let tangent = normalize(spin - normal * dot(spin, normal) + vec3f(0.0, 0.0, 1e-4));
  let bitangent = cross(normal, tangent);
  let radius = occlusion.settings.x;
  var filled = 0.0;
  for (var index = 0; index < 12; index++) {
    let offset = kernel[index];
    let probe = middle + (tangent * offset.x + bitangent * offset.y + normal * offset.z) * radius;
    let clip = occlusion.projection * vec4f(probe, 1.0);
    let ndc = clip.xy / clip.w;
    let at = vec2f((ndc.x * 0.5 + 0.5) * occlusion.size.x, (0.5 - ndc.y * 0.5) * occlusion.size.y);
    let found = view_at(at, depth_at(vec2i(at))).z;
    // Filled where the scene lies in front of the probe, nearby.
    let near = smoothstep(0.0, 1.0, radius / max(abs(middle.z - found), 1e-4));
    filled += select(0.0, near, found >= probe.z + occlusion.settings.z);
  }
  return vec4f(pow(1.0 - filled / 12.0, occlusion.settings.y));
}
"#;

fn occlusion_shader(samples: u32) -> String {
    OCCLUSION_SHADER
        .replace("${depth_texture}", depth_texture(samples))
        .replace("${fullscreen}", FULLSCREEN)
        .replace("${kernel}", &kernel())
}

const PRESENT_SHADER: &str = r#"
struct Present {
  output: vec4f,
  source: vec4f,
  split: vec4f,
  // x: how much of the bloom to add.
  bloom: vec4f,
};
@group(0) @binding(0) var<uniform> present: Present;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var source_sampler: sampler;
// The scene's stencil: bit 1 where a surface asks for the frame's aberration.
@group(0) @binding(3) var marks: ${marks_texture};
@group(0) @binding(4) var glow: texture_2d<f32>;
${fullscreen}
${tonedWgsl}

fn marked(uv: vec2f) -> bool {
  let size = present.split.zw;
  let pixel = clamp(vec2i(uv * size), vec2i(0), vec2i(size) - 1);
  return (textureLoad(marks, pixel, 0).x & 1u) != 0u;
}

// The picture at uv as the screen shows it: the scene's light and its
// bloom, toned.
fn fetch(uv: vec2f) -> vec3f {
  let light = textureSampleLevel(source, source_sampler, uv, 0.0).rgb +
    textureSampleLevel(glow, source_sampler, uv, 0.0).rgb * present.bloom.x;
  return toned(light);
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
"#;

fn present_shader(samples: u32) -> String {
    PRESENT_SHADER
        .replace("${marks_texture}", if samples > 1 { "texture_multisampled_2d<u32>" } else { "texture_2d<u32>" })
        .replace("${fullscreen}", FULLSCREEN)
        .replace("${tonedWgsl}", TONED_WGSL)
        .replace("${fxaa}", include_str!("fxaa.wgsl"))
}

// Bloom: the light past white, spilling round where it is. The first pass
// halves the scene, keeping only what is brighter than `threshold` (eased
// in over `knee` below it) and weighting each group of taps against the
// brightest pixels, so lone sparkles do not flicker; each next pass halves
// the last, a mip down. Then each level, blurred with a tent, is added onto
// the one above, back up to half the scene's size, which the last pass adds
// over the scene (after Jorge Jimenez's bloom for Call of Duty: Advanced
// Warfare: a 13-tap filter down, a 9-tap tent up).
const BLOOM_VERTEX: &str = r#"
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};
@vertex fn vertex(@builtin(vertex_index) index: u32) -> Corner {
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  var output: Corner;
  output.position = vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
  output.uv = vec2f(corner.x, 1.0 - corner.y);
  return output;
}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
fn tap(uv: vec2f, x: f32, y: f32) -> vec3f {
  let texel = 1.0 / vec2f(textureDimensions(source));
  return textureSampleLevel(source, source_sampler, uv + vec2f(x, y) * texel, 0.0).rgb;
}"#;

const BLOOM_DOWN_SHADER: &str = r#"
${bloomVertex}
const threshold = 1.3;
const knee = 0.6;

fn brightness(color: vec3f) -> f32 {
  return max(color.r, max(color.g, color.b));
}

// A group of four taps, averaged; first, each weighted the less the brighter
// it is (Karis), so one blazing pixel cannot outshine its neighbours.
fn weight(color: vec3f) -> f32 {
  return ${weight};
}
fn group(a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f {
  let wa = weight(a);
  let wb = weight(b);
  let wc = weight(c);
  let wd = weight(d);
  return (a * wa + b * wb + c * wc + d * wd) / (wa + wb + wc + wd);
}

fn past_white(color: vec3f) -> vec3f {
  let bright = brightness(color);
  let eased = clamp(bright - threshold + knee, 0.0, 2.0 * knee);
  let kept = max(eased * eased / (4.0 * knee), bright - threshold);
  return color * kept / max(bright, 1e-4);
}

@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let uv = corner.uv;
  let a = tap(uv, -2.0, -2.0); let b = tap(uv, 0.0, -2.0); let c = tap(uv, 2.0, -2.0);
  let d = tap(uv, -2.0, 0.0); let e = tap(uv, 0.0, 0.0); let f = tap(uv, 2.0, 0.0);
  let g = tap(uv, -2.0, 2.0); let h = tap(uv, 0.0, 2.0); let i = tap(uv, 2.0, 2.0);
  let j = tap(uv, -1.0, -1.0); let k = tap(uv, 1.0, -1.0);
  let l = tap(uv, -1.0, 1.0); let m = tap(uv, 1.0, 1.0);
  var color = group(j, k, l, m) * 0.5 +
    (group(a, b, d, e) + group(b, c, e, f) + group(d, e, g, h) + group(e, f, h, i)) * 0.125;
  ${past_white}
  return vec4f(color, 1.0);
}
"#;

fn bloom_down_shader(first: bool) -> String {
    BLOOM_DOWN_SHADER
        .replace("${bloomVertex}", BLOOM_VERTEX)
        .replace("${weight}", if first { "1.0 / (1.0 + brightness(color))" } else { "1.0" })
        .replace("${past_white}", if first { "color = past_white(color);" } else { "" })
}

const BLOOM_UP_SHADER: &str = r#"
${bloomVertex}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let uv = corner.uv;
  let color = tap(uv, 0.0, 0.0) * 4.0 +
    (tap(uv, 0.0, -1.0) + tap(uv, -1.0, 0.0) + tap(uv, 1.0, 0.0) + tap(uv, 0.0, 1.0)) * 2.0 +
    tap(uv, -1.0, -1.0) + tap(uv, 1.0, -1.0) + tap(uv, -1.0, 1.0) + tap(uv, 1.0, 1.0);
  return vec4f(color / 16.0, 1.0);
}
"#;

fn bloom_up_shader() -> String {
    BLOOM_UP_SHADER.replace("${bloomVertex}", BLOOM_VERTEX)
}

// How much of the bloom the last pass adds, and how many times at most the
// scene is halved for it.
const BLOOM_STRENGTH: f32 = 0.35;
const BLOOM_LEVELS: u32 = 6;

const MAX_RENDER_PIXELS: f64 = 3840.0 * 2160.0;

// How far ambient occlusion darkens, and its reach, curve and bias.
const OCCLUSION_STRENGTH: f32 = 1.0;
const OCCLUSION_RADIUS: f32 = 1.8;
const OCCLUSION_POWER: f32 = 2.2;
const OCCLUSION_BIAS: f32 = 0.04;

// The atmosphere pass's uniform: the camera, the sky the game worked out, the
// fog and how far ambient occlusion darkens.
fn atmosphere_of(frame: &RenderFrame, quality: &Quality, width: u32, height: u32) -> Vec<f32> {
    let camera = invert(&frame.view);
    let sky = &frame.sky;
    let clear = &frame.clear;
    let light = &frame.light_color;
    let mut data = invert(&multiply(&frame.projection, &frame.view)).to_vec();
    data.extend([
        camera[12],
        camera[13],
        camera[14],
        if quality.sky { 1.0 } else { 0.0 },
        sky.zenith.x,
        sky.zenith.y,
        sky.zenith.z,
        sky.night,
        sky.horizon.x,
        sky.horizon.y,
        sky.horizon.z,
        0.0,
        sky.ground.x,
        sky.ground.y,
        sky.ground.z,
        0.0,
        sky.sun.x,
        sky.sun.y,
        sky.sun.z,
        0.0,
        light.x,
        light.y,
        light.z,
        0.0,
        clear.r,
        clear.g,
        clear.b,
        1.0,
        frame.fog.radius,
        frame.fog.density,
        frame.fog.height,
        if quality.occlusion { OCCLUSION_STRENGTH } else { 0.0 },
        width as f32,
        height as f32,
        0.0,
        0.0,
        frame.clouds.cover,
        frame.clouds.drift.x,
        frame.clouds.drift.z,
        0.0,
    ]);
    data
}

struct BloomLevel {
    view: wgpu::TextureView,
    down: wgpu::BindGroup,
    up: Option<wgpu::BindGroup>,
}

struct Groups {
    occlusion: Option<wgpu::BindGroup>,
    atmosphere: Option<wgpu::BindGroup>,
    /// Each bloom level's pass down from the one above, then up into it.
    bloom: Vec<BloomLevel>,
    present: wgpu::BindGroup,
}

#[derive(PartialEq)]
struct TargetsKey {
    width: u32,
    height: u32,
    samples: u32,
    occluding: bool,
    atmosphere: bool,
    levels: u32,
}

struct Targets {
    key: TargetsKey,
    /// Depth, and the stencil marks of surfaces the aberration splits.
    depth: wgpu::Texture,
    /// The scene's samples, resolved into `scene`.
    multisampled: Option<wgpu::Texture>,
    /// The resolved scene.
    scene: wgpu::Texture,
    /// Ambient occlusion, when it is on.
    occlusion: Option<wgpu::Texture>,
    /// Sky and fog over the scene, when there are any.
    composed: Option<wgpu::Texture>,
    groups: Groups,
}

/// The attachments of one frame's scene passes, from `Post::begin`.
pub(crate) struct SceneTargets {
    pub samples: u32,
    pub size: (u32, u32),
    pub depth: wgpu::TextureView,
    /// The depth alone, for passes that read it while it stays attached.
    pub depth_only: wgpu::TextureView,
    scene: wgpu::TextureView,
    multisampled: Option<wgpu::TextureView>,
}

impl SceneTargets {
    /// The color attachment of a scene pass: the first clears, the last
    /// resolves samples, and one pass can be both.
    pub fn color(&self, clear: wgpu::Color, first: bool, last: bool) -> wgpu::RenderPassColorAttachment<'_> {
        let load = if first { wgpu::LoadOp::Clear(clear) } else { wgpu::LoadOp::Load };
        match &self.multisampled {
            None => wgpu::RenderPassColorAttachment {
                view: &self.scene,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
            },
            Some(multisampled) => wgpu::RenderPassColorAttachment {
                view: multisampled,
                depth_slice: None,
                resolve_target: last.then_some(&self.scene),
                ops: wgpu::Operations {
                    load,
                    store: if last { wgpu::StoreOp::Discard } else { wgpu::StoreOp::Store },
                },
            },
        }
    }
}

pub(crate) struct Post {
    device: wgpu::Device,
    queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    pipelines: HashMap<String, wgpu::RenderPipeline>,
    uniforms: HashMap<&'static str, wgpu::Buffer>,
    sampler: wgpu::Sampler,
    // Bound in place of the occlusion when it is off, and of the bloom.
    unoccluded: wgpu::Texture,
    unbloomed: wgpu::Texture,
    targets: Option<Targets>,
    /// The begun frame's quality and output size.
    quality: Option<Quality>,
    size: (u32, u32),
}

fn texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    usage: wgpu::TextureUsages,
    sample_count: u32,
    format: wgpu::TextureFormat,
    mip_level_count: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

impl Post {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        // A 1 x 1 texture holding one value, bound in place of a pass that is off.
        let single = |format: wgpu::TextureFormat, value: u8| {
            let made = texture(device, 1, 1, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, 1, format, 1);
            if format == wgpu::TextureFormat::R8Unorm {
                queue.write_texture(
                    made.as_image_copy(),
                    &[value],
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(1), rows_per_image: None },
                    wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                );
            }
            made
        };
        Self {
            device: device.clone(),
            queue: queue.clone(),
            format,
            pipelines: HashMap::new(),
            uniforms: HashMap::new(),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            unoccluded: single(wgpu::TextureFormat::R8Unorm, 255),
            unbloomed: single(SCENE_FORMAT, 0),
            targets: None,
            quality: None,
            size: (1, 1),
        }
    }

    fn pipeline(
        &mut self,
        name: String,
        code: impl FnOnce() -> String,
        target: wgpu::TextureFormat,
        blend: Option<wgpu::BlendState>,
    ) -> wgpu::RenderPipeline {
        let device = &self.device;
        self.pipelines
            .entry(name)
            .or_insert_with_key(|name| {
                let clock = crate::platform::Clock::new();
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(code().into()),
                });
                let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(name),
                    layout: None,
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: None,
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: None,
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState { format: target, blend, write_mask: wgpu::ColorWrites::ALL })],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
                #[cfg(not(target_arch = "wasm32"))]
                if std::env::var_os("PEN_FRAME_TIMES").is_some() {
                    eprintln!("post pipeline {name}: {:.2} ms", clock.now() * 1000.0);
                }
                #[cfg(target_arch = "wasm32")]
                let _ = clock;
                pipeline
            })
            .clone()
    }

    fn atmosphere_pipeline(&mut self, samples: u32) -> wgpu::RenderPipeline {
        self.pipeline(format!("atmosphere {samples}"), || atmosphere_shader(samples), SCENE_FORMAT, None)
    }
    fn occlusion_pipeline(&mut self, samples: u32) -> wgpu::RenderPipeline {
        self.pipeline(format!("occlusion {samples}"), || occlusion_shader(samples), wgpu::TextureFormat::R8Unorm, None)
    }
    fn present_pipeline(&mut self, samples: u32) -> wgpu::RenderPipeline {
        let format = self.format;
        self.pipeline(format!("present {samples}"), || present_shader(samples), format, None)
    }
    fn bloom_down_pipeline(&mut self, first: bool) -> wgpu::RenderPipeline {
        self.pipeline(format!("bloom down {first}"), || bloom_down_shader(first), SCENE_FORMAT, None)
    }
    // Each level up is added onto the one above.
    fn bloom_up_pipeline(&mut self) -> wgpu::RenderPipeline {
        let add = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        self.pipeline("bloom up".into(), bloom_up_shader, SCENE_FORMAT, Some(wgpu::BlendState { color: add, alpha: add }))
    }

    fn uniform(&mut self, name: &'static str, size: u64) -> wgpu::Buffer {
        let device = &self.device;
        self.uniforms
            .entry(name)
            .or_insert_with(|| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(name),
                    size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .clone()
    }

    fn group(&self, from: &wgpu::RenderPipeline, resources: &[wgpu::BindingResource<'_>]) -> wgpu::BindGroup {
        let entries: Vec<_> = resources
            .iter()
            .enumerate()
            .map(|(binding, resource)| wgpu::BindGroupEntry { binding: binding as u32, resource: resource.clone() })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &from.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    /// Prepares the attachments for `frame` drawn with `quality` on an
    /// output of `size`, and returns the scene pass's color and depth
    /// attachments.
    pub fn begin(&mut self, frame: &RenderFrame, quality: Quality, size: (u32, u32)) -> SceneTargets {
        self.quality = Some(quality);
        self.size = size;
        let occluding = quality.occlusion;
        let atmosphere = quality.sky || frame.fog.density > 0.0 || occluding || frame.clouds.cover > 0.0;
        // At most the pixels of a 4K screen, whatever the scale and display:
        // the multisampled targets of more would take gigabytes.
        let limit = self.device.limits().max_texture_dimension_2d;
        let budget = f64::from(quality.scale)
            .min((MAX_RENDER_PIXELS / (f64::from(size.0) * f64::from(size.1)).max(1.0)).sqrt());
        let scaled = |value: u32| ((f64::from(value) * budget).round().max(1.0) as u32).min(limit);
        let (width, height) = (scaled(size.0), scaled(size.1));
        let samples = quality.msaa;
        // Halving down to no smaller than a few pixels.
        let half = (width.div_ceil(2), height.div_ceil(2));
        let levels = if quality.bloom {
            let fit = (f64::from(half.0.min(half.1)) / 4.0).log2().floor() as i64 + 1;
            fit.min(i64::from(BLOOM_LEVELS)).max(1) as u32
        } else {
            0
        };
        let key = TargetsKey { width, height, samples, occluding, atmosphere, levels };
        if self.targets.as_ref().is_none_or(|targets| targets.key != key) {
            self.targets = None;
            let attachment = wgpu::TextureUsages::RENDER_ATTACHMENT;
            let sampled = attachment | wgpu::TextureUsages::TEXTURE_BINDING;
            let device = &self.device;
            // Decals read the depth.
            let depth = texture(device, width, height, sampled, samples, wgpu::TextureFormat::Depth24PlusStencil8, 1);
            let depth_only = || depth.create_view(&wgpu::TextureViewDescriptor { aspect: wgpu::TextureAspect::DepthOnly, ..Default::default() });
            let multisampled = (samples > 1).then(|| texture(device, width, height, attachment, samples, SCENE_FORMAT, 1));
            let scene = texture(device, width, height, sampled, 1, SCENE_FORMAT, 1);
            let occlusion = occluding.then(|| texture(device, width, height, sampled, 1, wgpu::TextureFormat::R8Unorm, 1));
            let composed = atmosphere.then(|| texture(device, width, height, sampled, 1, SCENE_FORMAT, 1));
            let bloom = (levels > 0).then(|| texture(device, half.0, half.1, sampled, 1, SCENE_FORMAT, levels));
            let lit = composed.as_ref().unwrap_or(&scene).create_view(&Default::default());
            let level = |index: u32| {
                bloom.as_ref().unwrap().create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: index,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            };
            let occlusion_group = if occluding {
                let pipeline = self.occlusion_pipeline(samples);
                let uniform = self.uniform("occlusion", 160);
                Some(self.group(&pipeline, &[uniform.as_entire_binding(), wgpu::BindingResource::TextureView(&depth_only())]))
            } else {
                None
            };
            let atmosphere_group = if atmosphere {
                let pipeline = self.atmosphere_pipeline(samples);
                let uniform = self.uniform("atmosphere", 224);
                let occlusion_view = occlusion.as_ref().unwrap_or(&self.unoccluded).create_view(&Default::default());
                Some(self.group(
                    &pipeline,
                    &[
                        uniform.as_entire_binding(),
                        wgpu::BindingResource::TextureView(&scene.create_view(&Default::default())),
                        wgpu::BindingResource::TextureView(&depth_only()),
                        wgpu::BindingResource::TextureView(&occlusion_view),
                        wgpu::BindingResource::Sampler(&self.sampler),
                    ],
                ))
            } else {
                None
            };
            let mut bloom_levels = Vec::new();
            for index in 0..levels {
                let down_pipeline = self.bloom_down_pipeline(index == 0);
                let from = if index == 0 { lit.clone() } else { level(index - 1) };
                let down = self.group(
                    &down_pipeline,
                    &[wgpu::BindingResource::TextureView(&from), wgpu::BindingResource::Sampler(&self.sampler)],
                );
                let up = if index + 1 < levels {
                    let up_pipeline = self.bloom_up_pipeline();
                    Some(self.group(
                        &up_pipeline,
                        &[
                            wgpu::BindingResource::TextureView(&level(index + 1)),
                            wgpu::BindingResource::Sampler(&self.sampler),
                        ],
                    ))
                } else {
                    None
                };
                bloom_levels.push(BloomLevel { view: level(index), down, up });
            }
            let present_pipeline = self.present_pipeline(samples);
            let present_uniform = self.uniform("present", 64);
            let marks = depth.create_view(&wgpu::TextureViewDescriptor { aspect: wgpu::TextureAspect::StencilOnly, ..Default::default() });
            let glow = if bloom.is_some() { level(0) } else { self.unbloomed.create_view(&Default::default()) };
            let present = self.group(
                &present_pipeline,
                &[
                    present_uniform.as_entire_binding(),
                    wgpu::BindingResource::TextureView(&lit),
                    wgpu::BindingResource::Sampler(&self.sampler),
                    wgpu::BindingResource::TextureView(&marks),
                    wgpu::BindingResource::TextureView(&glow),
                ],
            );
            self.targets = Some(Targets {
                key,
                depth,
                multisampled,
                scene,
                occlusion,
                composed,
                groups: Groups { occlusion: occlusion_group, atmosphere: atmosphere_group, bloom: bloom_levels, present },
            });
        }
        let current = self.targets.as_ref().unwrap();
        SceneTargets {
            samples,
            size: (current.key.width, current.key.height),
            depth: current.depth.create_view(&Default::default()),
            depth_only: current.depth.create_view(&wgpu::TextureViewDescriptor { aspect: wgpu::TextureAspect::DepthOnly, ..Default::default() }),
            scene: current.scene.create_view(&Default::default()),
            multisampled: current.multisampled.as_ref().map(|texture| texture.create_view(&Default::default())),
        }
    }

    /// Encodes the passes after the scene, ending on `output`.
    pub fn finish(&mut self, encoder: &mut wgpu::CommandEncoder, frame: &RenderFrame, output: &wgpu::TextureView) {
        let quality = self.quality.expect("Post::begin comes first");
        let size = self.size;
        let samples = self.targets.as_ref().unwrap().key.samples;
        let (width, height) = {
            let key = &self.targets.as_ref().unwrap().key;
            (key.width, key.height)
        };
        let draw = |encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::RenderPipeline, group: &wgpu::BindGroup, view: &wgpu::TextureView, add: bool| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if add { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(wgpu::Color::BLACK) },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        };
        if self.targets.as_ref().unwrap().groups.occlusion.is_some() {
            let uniform = self.uniform("occlusion", 160);
            let mut data = frame.projection.to_vec();
            data.extend(invert(&frame.projection));
            data.extend([OCCLUSION_RADIUS, OCCLUSION_POWER, OCCLUSION_BIAS, 0.0, width as f32, height as f32, 0.0, 0.0]);
            self.queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&data));
            let pipeline = self.occlusion_pipeline(samples);
            let targets = self.targets.as_ref().unwrap();
            let view = targets.occlusion.as_ref().unwrap().create_view(&Default::default());
            draw(encoder, &pipeline, targets.groups.occlusion.as_ref().unwrap(), &view, false);
        }
        if self.targets.as_ref().unwrap().groups.atmosphere.is_some() {
            let uniform = self.uniform("atmosphere", 224);
            self.queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&atmosphere_of(frame, &quality, width, height)));
            let pipeline = self.atmosphere_pipeline(samples);
            let targets = self.targets.as_ref().unwrap();
            let view = targets.composed.as_ref().unwrap().create_view(&Default::default());
            draw(encoder, &pipeline, targets.groups.atmosphere.as_ref().unwrap(), &view, false);
        }
        let levels = self.targets.as_ref().unwrap().groups.bloom.len();
        for index in 0..levels {
            let pipeline = self.bloom_down_pipeline(index == 0);
            let level = &self.targets.as_ref().unwrap().groups.bloom[index];
            draw(encoder, &pipeline, &level.down, &level.view, false);
        }
        if levels > 1 {
            let pipeline = self.bloom_up_pipeline();
            for index in (0..levels - 1).rev() {
                let level = &self.targets.as_ref().unwrap().groups.bloom[index];
                draw(encoder, &pipeline, level.up.as_ref().unwrap(), &level.view, true);
            }
        }
        let uniform = self.uniform("present", 64);
        let (output_width, output_height) = (size.0 as f32, size.1 as f32);
        self.queue.write_buffer(
            &uniform,
            0,
            bytemuck::cast_slice(&[
                1.0 / output_width,
                1.0 / output_height,
                if quality.effects { frame.blur } else { 0.0 },
                0.0,
                1.0 / width as f32,
                1.0 / height as f32,
                if quality.fxaa { 1.0 } else { 0.0 },
                0.0,
                // Screen heights, as texture coordinates.
                if quality.effects { frame.aberration.x * output_height / output_width } else { 0.0 },
                if quality.effects { frame.aberration.y } else { 0.0 },
                width as f32,
                height as f32,
                if levels > 0 { BLOOM_STRENGTH } else { 0.0 },
                0.0,
                0.0,
                0.0f32,
            ]),
        );
        let pipeline = self.present_pipeline(samples);
        draw(encoder, &pipeline, &self.targets.as_ref().unwrap().groups.present, output, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_fill_every_hole() {
        for code in [
            atmosphere_shader(1),
            atmosphere_shader(4),
            occlusion_shader(4),
            present_shader(1),
            bloom_down_shader(true),
            bloom_down_shader(false),
            bloom_up_shader(),
        ] {
            assert!(!code.contains("${"), "{code}");
        }
        assert!(atmosphere_shader(4).contains("empty += 0.2500;"));
        // As the TS's toFixed(5) prints it.
        assert_eq!(
            kernel(),
            "vec3f(0.06923, 0.00000, 0.24564), vec3f(-0.09203, 0.08430, 0.24036), vec3f(0.01513, -0.17240, 0.24121), \
             vec3f(0.13697, 0.17865, 0.24583), vec3f(-0.28045, -0.04961, 0.25189), vec3f(0.29872, -0.19002, 0.25703), \
             vec3f(-0.11262, 0.41894, 0.25892), vec3f(-0.24177, -0.46551, 0.25521), vec3f(0.58818, 0.21480, 0.24355), \
             vec3f(-0.68243, 0.28170, 0.22161), vec3f(0.36455, -0.77903, 0.18704), vec3f(0.29644, 0.94510, 0.13750)"
        );
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "post/fxaa_tests.rs"]
mod fxaa_tests;
