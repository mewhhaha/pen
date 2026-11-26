// The sun's shadows for a material shader: its cascaded maps, filtered, and
// the screen-space contact shadows that catch what the maps are too coarse
// for. Declare `Scene` through `shadow_params` (see materials.md), as `scene`
// at group 0 binding 0, and add this after it; it declares bindings 2, 3 and 9
// of the same group. Call `sun_shadow(world, normal, position)` where the sun's
// light is multiplied by its visibility: 1 in the sun, 0 in shadow.
//
// What the game sets, per frame, in the scene's fields:
//   cascades.x   how many maps there are, nearest first (0: no shadows);
//   cascades.y   the share of a slice over which one map blends into the next;
//   cascades.z   the filter: 0 one hardware-filtered tap, 1 percentage-closer
//                filtering over a Poisson disk, 2 percentage-closer soft
//                shadows, whose penumbra follows the blocker's distance;
//   cascades.w   steps of the contact shadow march, 0 for none;
//   cascade_view_projection[i]   world to the map's clip space (orthographic);
//   cascade_texels[i]            x: a texel in texture coordinates, y: a texel
//                in world units, z: how far ahead of the camera the slice
//                ends, w: the depth range in world units;
//   shadow_params.x   the penumbra's width per unit of blocker distance;
//   shadow_params.y   how far toward the sun contact shadows look.
//
// Every map's depth is linear (orthographic), so a receiver's own plane gives
// the depth a filter tap should compare against: the bias follows the
// surface's slope instead of a constant that either lets acne through at
// grazing angles or lifts shadows off their casters. The normal offset
// handles what the plane cannot: texels coarser than the surface's detail.
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
@group(0) @binding(3) var shadow_cascades: texture_depth_2d_array;
@group(0) @binding(9) var contact_depth: texture_depth_2d;

// Sixteen points of a Poisson disk of radius 1 (a best-candidate sample,
// kept in sync with the reference in pen-render's tests).
var<private> shadow_taps = array<vec2f, 16>(
  vec2f(-0.94201624, -0.39906216), vec2f(0.94558609, -0.76890725),
  vec2f(-0.09418410, -0.92938870), vec2f(0.34495938, 0.29387760),
  vec2f(-0.91588581, 0.45771432), vec2f(-0.81544232, -0.87912464),
  vec2f(-0.38277543, 0.27676845), vec2f(0.97484398, 0.75648379),
  vec2f(0.44323325, -0.97511554), vec2f(0.53742981, -0.47373420),
  vec2f(-0.26496911, -0.41893023), vec2f(0.79197514, 0.19090188),
  vec2f(-0.24188840, 0.99706507), vec2f(-0.81409955, 0.91437590),
  vec2f(0.19984126, 0.78641367), vec2f(0.14383161, -0.14100790));

// The widest filter, in texels, and how far past a map's edge a tap may reach.
const sun_max_radius: f32 = 12.0;
// How far a surface is moved off along its normal (in texels, at a right
// angle to the light) before it reads the map, and how much depth a texel of
// the receiver's slope adds on top of what the plane already gives.
const sun_normal_offset: f32 = 1.5;
const sun_bias_base: f32 = 1.0;
const sun_bias_slope: f32 = 0.75;
const sun_max_slope: f32 = 6.0;
// The widest filter, in texels, hard shadows and plain filtering use.
const sun_filter_radius: f32 = 1.5;
// The farthest a blocker is searched for, in world units from the receiver.
const sun_blocker_reach: f32 = 8.0;

// A pseudo-random number in [0, 1) from a pixel's position: interleaved
// gradient noise, fixed per pixel, so that the filter's pattern holds still
// from frame to frame and temporal antialiasing averages it away.
fn shadow_noise(pixel: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(pixel, vec2f(0.06711056, 0.00583715))));
}

// (cos, sin) of a turn of the filter's disk, from a pixel's position.
fn shadow_spin(pixel: vec2f) -> vec2f {
  let angle = 6.2831853 * shadow_noise(pixel);
  return vec2f(cos(angle), sin(angle));
}

// A map's view of one surface: where it reads, which way depth runs.
struct SunMap {
  // Texture coordinates, and depth, of the surface moved off its normal.
  uv: vec2f,
  depth: f32,
  // How far inside the map's edge the filter's reach still is, in texture
  // coordinates: negative when it reaches past it.
  inside: f32,
  // Pre-scaled receiver plane depth change per texel offset, and biased base depth.
  slope_texel: vec2f,
  base_depth: f32,
  // A world unit's depth, and a texel's width in world units.
  unit: f32,
  texel: f32,
};

fn sun_map_of(map: u32, world: vec3f, normal: vec3f, reach: f32) -> SunMap {
  let texel = scene.cascade_texels[map];
  let matrix = scene.cascade_view_projection[map];
  let toward = -normalize(scene.light_direction.xyz);
  let facing = dot(normal, toward);
  let sine = sqrt(clamp(1.0 - facing * facing, 0.0, 1.0));
  let shifted = world + normal * (texel.y * sun_normal_offset * sine);
  let ndc = (matrix * vec4f(shifted, 1.0)).xyz;
  var result: SunMap;
  result.uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  result.depth = ndc.z;
  let edge = min(min(result.uv.x, result.uv.y), min(1.0 - result.uv.x, 1.0 - result.uv.y));
  result.inside = edge - (reach + 1.0) * texel.x;
  // The map's axes in the world: its rows are scaled unit vectors. Depth
  // grows away from the light.
  let row_x = vec3f(matrix[0].x, matrix[1].x, matrix[2].x);
  let row_y = vec3f(matrix[0].y, matrix[1].y, matrix[2].y);
  let row_z = vec3f(matrix[0].z, matrix[1].z, matrix[2].z);
  let unit_z = length(row_z);
  let deeper = row_z / unit_z;
  let across = dot(normal, deeper);
  let flat_ = min(across, -0.1);
  let sx = clamp(-dot(normal, row_x / length(row_x)) / flat_, -sun_max_slope, sun_max_slope);
  let sy = clamp(-dot(normal, row_y / length(row_y)) / flat_, -sun_max_slope, sun_max_slope);
  result.unit = unit_z;
  // The plane's depth change per texel across, and receiver bias pre-subtracted once.
  let slope = vec2f(sx, sy) * unit_z;
  result.slope_texel = slope * texel.y;
  let bias = (sun_bias_base * unit_z + sun_bias_slope * (abs(slope.x) + abs(slope.y))) * texel.y;
  result.base_depth = ndc.z - bias;
  result.texel = texel.y;
  return result;
}

// The depth a tap offset by `offset` texels from the receiver compares
// against: the receiver's plane there, and a bias that grows with its slope.
fn sun_tap_depth(m: SunMap, offset: vec2f) -> f32 {
  return m.base_depth + dot(m.slope_texel, offset);
}

// The share of sun a filter of `radius` texels, turned by `spin`, lets through.
fn sun_filtered(map: u32, m: SunMap, radius: f32, spin: vec2f, taps: u32) -> f32 {
  let texel = scene.cascade_texels[map].x;
  var light = 0.0;
  for (var i = 0u; i < taps; i++) {
    let tap = shadow_taps[i];
    let offset = vec2f(tap.x * spin.x - tap.y * spin.y, tap.x * spin.y + tap.y * spin.x) * radius;
    let depth = sun_tap_depth(m, vec2f(offset.x, -offset.y));
    light += textureSampleCompareLevel(shadow_cascades, shadow_sampler, m.uv + offset * texel, map, depth);
  }
  return light / f32(taps);
}

// How much sun the map `map` lets through to `world`, in x, with in y how far
// inside the map's edge the surface is (negative: it reads past it and the
// next map should answer).
fn sun_in(map: u32, world: vec3f, normal: vec3f, spin: vec2f) -> vec2f {
  let mode = u32(scene.cascades.z);
  var reach = sun_filter_radius;
  if (mode == 2u) {
    reach = sun_max_radius;
  }
  let m = sun_map_of(map, world, normal, reach);
  if (m.inside < 0.0 || m.depth > 1.0 || m.depth < 0.0) {
    return vec2f(1.0, -1.0);
  }
  if (mode == 0u) {
    return vec2f(textureSampleCompareLevel(shadow_cascades, shadow_sampler, m.uv, map, m.base_depth), m.inside);
  }
  if (mode == 1u) {
    return vec2f(sun_filtered(map, m, sun_filter_radius, spin, 16u), m.inside);
  }
  // Soft shadows: look for blockers where the penumbra could reach from,
  // average how far they are, and filter as wide as the sun's size makes
  // the penumbra there.
  let softness = scene.shadow_params.x;
  let size = vec2f(textureDimensions(shadow_cascades));
  let texel_uv = scene.cascade_texels[map].x;
  let search = clamp(softness * sun_blocker_reach * 0.5 / m.texel, 1.0, sun_max_radius - 1.0);
  var gap = 0.0;
  var blockers = 0.0;
  for (var i = 0u; i < 12u; i++) {
    let tap = shadow_taps[i];
    let offset = vec2f(tap.x * spin.x - tap.y * spin.y, tap.x * spin.y + tap.y * spin.x) * search;
    let at = vec2i(floor((m.uv + offset * texel_uv) * size));
    let seen = textureLoad(shadow_cascades, at, map, 0);
    let wanted = sun_tap_depth(m, vec2f(offset.x, -offset.y));
    if (seen < wanted) {
      gap += wanted - seen;
      blockers += 1.0;
    }
  }
  if (blockers == 0.0) {
    return vec2f(1.0, m.inside);
  }
  let blocked = gap / (blockers * m.unit);
  let radius = clamp(softness * blocked * 0.5 / m.texel, sun_filter_radius, sun_max_radius);
  return vec2f(sun_filtered(map, m, radius, spin, 16u), m.inside);
}

// How much sunlight reaches `world` (a surface facing `normal`, unit length)
// from the sun's cascaded maps alone, the kernel of the filter turned by
// `spin` (cos, sin). Past the last map's slice, all of it.
fn sun_visibility(world: vec3f, normal: vec3f, spin: vec2f) -> f32 {
  let count = u32(scene.cascades.x);
  if (scene.shadow.x < 0.5 || count == 0u) {
    return 1.0;
  }
  let depth = -(scene.view * vec4f(world, 1.0)).z;
  var map = 0u;
  while (map + 1u < count && depth >= scene.cascade_texels[map].z) {
    map += 1u;
  }
  // The slice this map covers, and the share of its far end it blends over.
  let far = scene.cascade_texels[map].z;
  let near = select(0.0, scene.cascade_texels[map - 1u].z, map > 0u);
  let band = max((far - near) * scene.cascades.y, 0.0001);
  let blend = smoothstep(0.0, 1.0, (depth - (far - band)) / band);
  var here = sun_in(map, world, normal, spin);
  // Reading past a map's edge: the next, wider map answers.
  while (here.y < 0.0 && map + 1u < count) {
    map += 1u;
    here = sun_in(map, world, normal, spin);
  }
  if (here.y < 0.0) {
    return 1.0;
  }
  var light = here.x;
  if (blend > 0.0) {
    if (map + 1u < count) {
      let next = sun_in(map + 1u, world, normal, spin);
      if (next.y >= 0.0) {
        light = mix(light, next.x, blend);
      }
    } else {
      light = mix(light, 1.0, blend);
    }
  }
  return light;
}

// How much sunlight the depth buffer lets through, marching from `world`
// toward the sun in screen space for as far as `shadow_params.y` and
// `cascades.w` say: a ray that dips behind what the camera sees, but not far
// behind it, is shadowed. This catches the small things the maps' texels
// cannot, a crack, a foot on the floor. `jitter` (0 to 1) shifts the steps
// along the ray, to be averaged away by temporal antialiasing.
fn contact_shadow(world: vec3f, normal: vec3f, jitter: f32) -> f32 {
  let steps = u32(scene.cascades.w);
  if (steps == 0u || scene.shadow.x < 0.5 || scene.projection[2][3] > -0.5) {
    return 1.0;
  }
  let toward = -normalize(scene.light_direction.xyz);
  let length = scene.shadow_params.y;
  let size = vec2f(textureDimensions(contact_depth));
  let a = scene.projection[2][2];
  let b = scene.projection[3][2];
  let eye_depth = -(scene.view * vec4f(world, 1.0)).z;
  // Start off the surface by a little along its normal, so it does not hit itself.
  let origin = world + normal * (0.01 + 0.003 * eye_depth);
  let clip_origin = scene.projection * (scene.view * vec4f(origin, 1.0));
  let clip_delta = scene.projection * (scene.view * vec4f(toward * length, 0.0));
  let inv_steps = 1.0 / f32(steps);
  var shadowed = 0.0;
  for (var i = 0u; i < steps; i++) {
    let along = (f32(i) + 0.25 + 0.75 * jitter) * inv_steps;
    let clip = clip_origin + clip_delta * along;
    if (clip.w <= 0.0) {
      break;
    }
    let ndc = clip.xyz / clip.w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (min(uv.x, uv.y) < 0.0 || max(uv.x, uv.y) >= 1.0 || ndc.z < 0.0) {
      break;
    }
    let seen = textureLoad(contact_depth, vec2i(uv * size), 0);
    // The depth ahead of the eye of what the camera sees there.
    let surface = b / (seen + a);
    let gap = clip.w - surface;
    let least = 0.01 + 0.004 * clip.w;
    let most = least + 0.2 + 0.4 * along * length;
    if (gap > least && gap < most) {
      // Nearer the surface, darker: a shadow tapers away from its caster.
      shadowed = max(shadowed, 1.0 - along * along);
    }
  }
  return 1.0 - shadowed;
}

// The sun's visibility at a fragment: `position` is its @builtin(position).
// The cascades' filter and the contact shadows, whichever the frame asks for.
fn sun_shadow(world: vec3f, normal: vec3f, position: vec4f) -> f32 {
  if (scene.shadow.x < 0.5) {
    return 1.0;
  }
  let noise = shadow_noise(position.xy);
  var spin = vec2f(1.0, 0.0);
  if (u32(scene.cascades.x) > 0u && u32(scene.cascades.z) > 0u) {
    let angle = 6.2831853 * noise;
    spin = vec2f(cos(angle), sin(angle));
  }
  let cascaded = sun_visibility(world, normal, spin);
  if (cascaded <= 0.0) {
    return 0.0;
  }
  return min(cascaded, contact_shadow(world, normal, fract(noise * 7.0 + 0.5)));
}
