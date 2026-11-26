// Screen-space subsurface scattering for skin-like materials, after Jimenez et
// al., "Separable Subsurface Scattering" (2015). One pass blurs the scene's
// picture along one screen axis and the next along the other, so that the two
// make the kernel's separable approximation of a diffusion profile: taps
// spread over a sum of Gaussians (the profile of skin, a wider one for red than
// for blue), their reach worked out in world units from the pixel's own depth,
// so a surface farther away is blurred over fewer pixels.
//
// Only pixels whose samples all carry the subsurface stencil mark are blurred;
// every other pixel is copied as it is, bit for bit. A tap on an unmarked or
// empty pixel, or one whose surface lies more than a few widths nearer or
// farther along the view, gives the pixel's own colour in its place, so light
// does not cross an edge between surfaces.
struct Subsurface {
  // The inverse of the projection the scene was drawn with: pixels and depth
  // buffer values to view space.
  inverse_projection: mat4x4f,
  // xy: the step to the next pixel along the blur's axis; z: the world distance
  // from a pixel to its outermost tap; w: unused.
  axis: vec4f,
  // The picture's size in pixels, then their reciprocals.
  size: vec4f,
};
@group(0) @binding(0) var<uniform> subsurface: Subsurface;
@group(0) @binding(1) var picture: texture_2d<f32>;
@group(0) @binding(2) var scene_depth: ${depth_texture};
@group(0) @binding(3) var marks: ${marks_texture};
${fullscreen}

// The kernel (subsurface.rs): the centre's weights, then each of seven pairs of
// taps' weights by channel and, in w, how far out of the reach they lie.
${kernel}

const MARK = ${mark}u;
// A tap whose surface is this many widths nearer or farther than the pixel's
// counts in full, one more than `DROP_FROM` widths away not at all, in between
// by what is left.
const KEEP_UNTIL = 2.0;
const DROP_FROM = 4.0;
// The most pixels the outermost tap lies from a pixel, however close the surface.
const MOST_REACH = 64.0;

// The pixel's nearest depth buffer value if each of its samples is a surface
// of the subsurface mark, else 2: it is not one.
fn marked_depth(pixel: vec2i) -> f32 {
  var nearest = 1.0;
  for (var sample = 0; sample < ${samples}; sample++) {
    if ((textureLoad(marks, pixel, sample).x & MARK) == 0u) { return 2.0; }
    nearest = min(nearest, textureLoad(scene_depth, pixel, sample));
  }
  return nearest;
}

// View-space z for a picture point and depth buffer value (only rows 2 and 3).
fn view_point_z(at: vec2f, depth: f32) -> f32 {
  let ndc = vec2f(at.x * subsurface.size.z * 2.0 - 1.0, 1.0 - at.y * subsurface.size.w * 2.0);
  let m = subsurface.inverse_projection;
  let z = m[0].z * ndc.x + m[1].z * ndc.y + m[2].z * depth + m[3].z;
  let w = m[0].w * ndc.x + m[1].w * ndc.y + m[2].w * depth + m[3].w;
  return z / w;
}

// How far a step of one pixel along the blur's axis is in view space, at the
// depth buffer value `depth` of the pixel `at`: the difference between the two
// points, worked out as a difference (of the projection's inverse's terms)
// rather than between them, which f32 cannot do for a pixel a hundredth of the
// way across the view.
fn pixel_world_size(at: vec2f, depth: f32) -> f32 {
  let ndc = vec2f(at.x * subsurface.size.z * 2.0 - 1.0, 1.0 - at.y * subsurface.size.w * 2.0);
  let here = subsurface.inverse_projection * vec4f(ndc, depth, 1.0);
  let across = subsurface.inverse_projection *
    vec4f(subsurface.axis.x * subsurface.size.z * 2.0, -subsurface.axis.y * subsurface.size.w * 2.0, 0.0, 0.0);
  let moved = (here.w * across.xyz - here.xyz * across.w) / (here.w * (here.w + across.w));
  return length(moved);
}

// The picture at a point between pixels, read bilinearly (2-tap separable fast path on axis-aligned passes).
fn picture_at(at: vec2f) -> vec3f {
  let corner = at - 0.5;
  let base = floor(corner);
  let share = corner - base;
  let first = vec2i(base);
  let last = vec2i(subsurface.size.xy) - 1;
  let a = textureLoad(picture, clamp(first, vec2i(0), last), 0).rgb;
  if (share.y <= 1e-4) {
    let b = textureLoad(picture, clamp(first + vec2i(1, 0), vec2i(0), last), 0).rgb;
    return mix(a, b, share.x);
  }
  if (share.x <= 1e-4) {
    let c = textureLoad(picture, clamp(first + vec2i(0, 1), vec2i(0), last), 0).rgb;
    return mix(a, c, share.y);
  }
  let b = textureLoad(picture, clamp(first + vec2i(1, 0), vec2i(0), last), 0).rgb;
  let c = textureLoad(picture, clamp(first + vec2i(0, 1), vec2i(0), last), 0).rgb;
  let d = textureLoad(picture, clamp(first + vec2i(1, 1), vec2i(0), last), 0).rgb;
  return mix(mix(a, b, share.x), mix(c, d, share.x), share.y);
}

// What a tap at `at` brings to a pixel whose surface is `here` (view space z)
// ahead of the eye and whose own colour is `centre`.
fn tap(at: vec2f, here: f32, inv_width: f32, centre: vec3f) -> vec3f {
  let pixel = vec2i(floor(at));
  if (any(pixel < vec2i(0)) || any(pixel >= vec2i(subsurface.size.xy))) { return centre; }
  let depth = marked_depth(pixel);
  if (depth > 1.0) { return centre; }
  let gap = abs(view_point_z(at, depth) - here) * inv_width;
  let keep = 1.0 - clamp((gap - KEEP_UNTIL) / (DROP_FROM - KEEP_UNTIL), 0.0, 1.0);
  if (keep <= 0.0) { return centre; }
  return mix(centre, picture_at(at), keep);
}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(position.xy);
  let centre = textureLoad(picture, pixel, 0);
  let depth = marked_depth(pixel);
  if (depth > 1.0) { return centre; }
  let here_z = view_point_z(position.xy, depth);
  let inv_width = 1.0 / subsurface.axis.z;
  let reach = min(subsurface.axis.z / max(pixel_world_size(position.xy, depth), 1e-9), MOST_REACH);
  var sum = centre.rgb * KERNEL[0].rgb;
  for (var index = 1; index < 8; index++) {
    let offset = subsurface.axis.xy * (KERNEL[index].w * reach);
    sum += KERNEL[index].rgb * (tap(position.xy + offset, here_z, inv_width, centre.rgb) + tap(position.xy - offset, here_z, inv_width, centre.rgb));
  }
  return vec4f(sum, centre.a);
}
