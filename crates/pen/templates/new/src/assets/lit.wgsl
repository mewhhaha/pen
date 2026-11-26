// A material shader: what pen-render binds for every draw (group 0, the
// frame's scene, its sky light and the sun's shadows; group 1, the draw and its
// material; group 2, the frame's point and spot lights) and the mesh's vertex
// layout. This one lights the material's color as a standard surface: the
// sun, shadowed (`sun_shadow`: the cascaded maps, filtered, and contact
// shadows, as the game asks), the sky's light all round (the game's probe
// volume where it has one, else its sky) and its reflections, and the frame's
// lights, through a GGX base lobe with a clear coat, sheen, anisotropy,
// transmission and subsurface lobes to turn on (see shader_lobes below, and
// materials.md). The leading fields of each struct are fixed; a shader
// declares as many of them as it reads, but the shadows need the whole Scene
// as it is here. The blocks between the BEGIN and END lines are pen-render's
// SHADOWS_WGSL, LIGHTS_WGSL, ENVIRONMENT_WGSL, SURFACE_LOBES_WGSL and
// SURFACE_WGSL, which declare their bindings; a test keeps them the same.
struct Scene {
  view: mat4x4f,
  projection: mat4x4f,
  ambient: vec4f,
  light_direction: vec4f,
  light_color: vec4f,
  light_view_projection: mat4x4f,
  shadow: vec4f,
  zenith: vec4f,
  horizon: vec4f,
  ground: vec4f,
  eye: vec4f,
  cascades: vec4f,
  cascade_view_projection: array<mat4x4f, 4>,
  cascade_texels: array<vec4f, 4>,
  shadow_params: vec4f,
};
struct Draw {
  model: mat4x4f,
  normal: mat4x4f,
  color: vec4f,
  settings: vec4f,
  parameters: vec4f,
  // The draw's `extra` vectors, whatever the game sends (zero past them).
  extra: array<vec4f, 32>,
  // The material's `surface` block: clearcoat, clearcoat roughness, sheen
  // roughness, anisotropy; sheen color, anisotropy rotation; transmission,
  // ior, thickness, attenuation distance; attenuation color, specular;
  // specular color, emissive strength (see `surface_apply_block`).
  surface: array<vec4f, 5>,
  // KHR_texture_transform (offset.xy, scale.xy), (cos(rot), sin(rot), orm_map, emissive_map),
  // (atlas_tile_mask, has_base_texture, 0, 0), and (iridescence, iridescence_ior, thickness_min, thickness_max).
  uv_transform: vec4f,
  maps: vec4f,
  atlas: vec4f,
  iridescence: vec4f,
};
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@group(1) @binding(1) var base_texture: texture_2d<f32>;
@group(1) @binding(2) var base_sampler: sampler;
@group(1) @binding(4) var layer_texture: texture_2d<f32>;
// BEGIN shadows
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
// END shadows
// BEGIN lights
// The frame's point and spot lights, sorted into clusters of the view: the
// picture cut into tiles across and down, and the depth ahead into slices,
// closer together near the eye. A fragment reads only the lights whose reach
// overlaps its cluster. Material shaders that light with them declare these
// bindings (group 2 of every mesh pipeline) and call `clustered_light`.
struct ClusterGrid {
  // x, y, z: tiles across, down and slices deep; w: the frame's lights.
  counts: vec4u,
  // xy: tiles per pixel across and down; z, w: a depth's slice is
  // log(depth) * z + w, or depth * z + w when options.x is 1.
  scale: vec4f,
  options: vec4f,
};
struct ClusterLight {
  // xyz: where it is; w: how far it reaches.
  position: vec4f,
  // rgb: its color times its intensity; w: the cosine of a spot's cone, -1 for a point light.
  color: vec4f,
  // xyz: the way a spot shines; w: the cosine within which it is at full strength.
  direction: vec4f,
  // x: the first layer of its shadow map in `local_shadows`, -1 for none (a
  // point light's six faces follow it); y: 1 for a cube, 0 for a spot's one
  // face; z: a face texel's width at unit depth, in world units.
  shadow: vec4f,
};
@group(2) @binding(0) var<uniform> cluster_grid: ClusterGrid;
@group(2) @binding(1) var<storage, read> cluster_lights: array<ClusterLight>;
@group(2) @binding(2) var<storage, read> cluster_ranges: array<vec2u>;
@group(2) @binding(3) var<storage, read> cluster_indices: array<u32>;
// The shadow maps of the lights that cast (a layer a face), their comparison
// sampler and each layer's world-to-clip matrix.
@group(2) @binding(4) var local_shadows: texture_depth_2d_array;
@group(2) @binding(5) var local_shadow_sampler: sampler_comparison;
@group(2) @binding(6) var<storage, read> local_shadow_faces: array<mat4x4f>;
// Packed LTC area lights (up to 64 lights * 5 vec4f at offset 0) followed by
// 64x64 LTC inverse-M matrix LUT (offset 320) and 64x64 LTC amplitude LUT (offset 4416).
@group(2) @binding(7) var<storage, read> area_light_data: array<vec4f>;

// Eight points of a Poisson disk of radius 1.
const local_shadow_taps = array<vec2f, 8>(
  vec2f(-0.7071, -0.7071), vec2f(0.0, -0.1), vec2f(0.7071, -0.7071), vec2f(-0.4, 0.2),
  vec2f(0.45, 0.25), vec2f(-0.85, 0.5), vec2f(0.1, 0.9), vec2f(0.9, 0.35));
// How far a surface is moved along its normal (texels, at a right angle to
// the light), and toward the light (texels: a base, and more with the slope
// of the surface to the light), before it reads its map; and the filter's reach.
const local_normal_offset: f32 = 1.5;
const local_bias_base: f32 = 1.5;
const local_bias_slope: f32 = 0.75;
const local_filter_radius: f32 = 1.5;

// How much of `light` reaches `world`, a surface facing `normal`, where `way`
// is the unit vector toward the light and `distance` how far it is: 1 where
// nothing the light's map holds lies between, 0 behind a caster, filtered over
// a small disk turned by a number from the surface's position. A point
// light's face is the one its direction from the light leans on most; each
// face's window reaches a few texels past it, so the filter never leaves it.
fn local_shadow(light: ClusterLight, world: vec3f, normal: vec3f, way: vec3f, distance: f32) -> f32 {
  if (light.shadow.x < 0.0) {
    return 1.0;
  }
  var layer = u32(light.shadow.x);
  if (light.shadow.y > 0.5) {
    let out = -way * distance;
    let a = abs(out);
    var face = 0u;
    if (a.x >= a.y && a.x >= a.z) {
      face = select(0u, 1u, out.x < 0.0);
    } else if (a.y >= a.z) {
      face = select(2u, 3u, out.y < 0.0);
    } else {
      face = select(4u, 5u, out.z < 0.0);
    }
    layer += face;
  }
  let matrix = local_shadow_faces[layer];
  let ahead = (matrix * vec4f(world, 1.0)).w;
  let texel = light.shadow.z * max(ahead, 0.0001);
  let facing = clamp(dot(normal, way), 0.0, 1.0);
  let sine = sqrt(clamp(1.0 - facing * facing, 0.0, 1.0));
  let tangent = min(sine / max(facing, 0.05), 4.0);
  let shifted = world + normal * (texel * local_normal_offset * sine) + way * (texel * (local_bias_base + local_bias_slope * tangent));
  let clip = matrix * vec4f(shifted, 1.0);
  if (clip.w <= 0.0) {
    return 1.0;
  }
  let ndc = clip.xyz / clip.w;
  if (ndc.z >= 1.0) {
    return 1.0;
  }
  let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let angle = 6.2831853 * fract(sin(dot(world, vec3f(12.9898, 78.233, 37.719))) * 43758.5453);
  let spin = vec2f(cos(angle), sin(angle));
  let step = local_filter_radius / f32(textureDimensions(local_shadows).x);
  var lit = 0.0;
  for (var i = 0u; i < 8u; i++) {
    let tap = local_shadow_taps[i];
    let offset = vec2f(tap.x * spin.x - tap.y * spin.y, tap.x * spin.y + tap.y * spin.x) * step;
    lit += textureSampleCompareLevel(local_shadows, local_shadow_sampler, uv + offset, layer, ndc.z);
  }
  return lit * 0.125;
}

// How much of `light` reaches `world` on a surface facing `normal`: its color
// times the cosine of its angle to the surface, falling smoothly to nothing
// at its reach (and, for a spot, outside its cone).
fn cluster_light(light: ClusterLight, world: vec3f, normal: vec3f) -> vec3f {
  let toward = light.position.xyz - world;
  let reach = light.position.w;
  let dist2 = dot(toward, toward);
  if (reach <= 0.0 || dist2 >= reach * reach) {
    return vec3f(0.0);
  }
  if (dot(normal, toward) <= 0.0) {
    return vec3f(0.0);
  }
  let distance = sqrt(dist2);
  let way = toward / max(distance, 0.0001);
  let ratio = distance / reach;
  let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
  var strength = window * window / (distance * distance + 1.0);
  if (light.color.w > -1.0) {
    strength *= smoothstep(light.color.w, max(light.direction.w, light.color.w + 0.0001), dot(-way, light.direction.xyz));
  }
  let nl = max(dot(normal, way), 0.0);
  return light.color.rgb * strength * nl * local_shadow(light, world, normal, way, distance);
}

// The cluster of the fragment at `position` (its @builtin(position)), whose
// distance ahead of the eye is `depth`.
fn cluster_of(position: vec4f, depth: f32) -> u32 {
  let counts = cluster_grid.counts;
  let tile = min(vec2u(max(position.xy * cluster_grid.scale.xy, vec2f(0.0))), counts.xy - 1u);
  let along = select(log(max(depth, 0.000001)), depth, cluster_grid.options.x > 0.5);
  let slice = u32(clamp(along * cluster_grid.scale.z + cluster_grid.scale.w, 0.0, f32(counts.z - 1u)));
  return (slice * counts.y + tile.y) * counts.x + tile.x;
}

// The light the frame's point and spot lights bring to `world`, on a surface
// facing `normal` (unit length), seen at the fragment `position` (its
// @builtin(position)) whose view-space z is `view_z` (negative ahead of the
// eye): their colors summed, each times the cosine of its angle to the
// surface, as a diffuse surface's irradiance.
fn clustered_light(world: vec3f, normal: vec3f, position: vec4f, view_z: f32) -> vec3f {
  if (cluster_grid.counts.w == 0u) {
    return vec3f(0.0);
  }
  let range = cluster_ranges[cluster_of(position, -view_z)];
  var sum = vec3f(0.0);
  for (var index = range.x; index < range.x + range.y; index++) {
    sum += cluster_light(cluster_lights[cluster_indices[index]], world, normal);
  }
  return sum;
}

// One light as a surface sees it: the unit direction toward it and the
// radiance it brings there (its color times its strength and its visibility,
// the frame's light scale, with no cosine), zero where it does not reach or
// the surface faces away from it.
struct ClusterSample {
  way: vec3f,
  radiance: vec3f,
};

// `light` at `world` for a surface facing `normal`. A surface that faces away
// from it takes nothing (and the shadow map is not read), unless `behind`: a
// subsurface that lets light through from the back.
fn cluster_light_sample(light: ClusterLight, world: vec3f, normal: vec3f, behind: bool) -> ClusterSample {
  var taken = ClusterSample(vec3f(0.0), vec3f(0.0));
  let toward = light.position.xyz - world;
  let reach = light.position.w;
  let dist2 = dot(toward, toward);
  if (reach <= 0.0 || dist2 >= reach * reach) {
    return taken;
  }
  if (!behind && dot(normal, toward) <= 0.0) {
    return taken;
  }
  let distance = sqrt(dist2);
  let way = toward / max(distance, 0.0001);
  let ratio = distance / reach;
  let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
  var strength = window * window / (distance * distance + 1.0);
  if (light.color.w > -1.0) {
    strength *= smoothstep(light.color.w, max(light.direction.w, light.color.w + 0.0001), dot(-way, light.direction.xyz));
  }
  taken.way = way;
  taken.radiance = light.color.rgb * strength * local_shadow(light, world, normal, way, distance);
  return taken;
}

// ---- Linearly Transformed Cosines (LTC) Area Lights via @group(2) @binding(7).
const area_max_lights: u32 = 64u;
const area_mat_offset: u32 = 320u;
const area_amp_offset: u32 = 4416u;
const area_pi: f32 = 3.14159265359;
const area_octagon_scale: f32 = 1.0539073;

fn area_sample_lut(base_offset: u32, rough: f32, q: f32) -> vec4f {
  let fx = clamp(rough, 0.0, 1.0) * 63.0;
  let fy = clamp(q, 0.0, 1.0) * 63.0;
  let x0 = min(u32(floor(fx)), 62u);
  let y0 = min(u32(floor(fy)), 62u);
  let tx = fx - f32(x0);
  let ty = fy - f32(y0);
  let i00 = base_offset + y0 * 64u + x0;
  let i10 = i00 + 1u;
  let i01 = i00 + 64u;
  let i11 = i01 + 1u;
  let row0 = mix(area_light_data[i00], area_light_data[i10], tx);
  let row1 = mix(area_light_data[i01], area_light_data[i11], tx);
  return mix(row0, row1, ty);
}

fn area_integrate_edge(v1: vec3f, v2: vec3f) -> vec3f {
  let c = cross(v1, v2);
  let sin_t = length(c);
  if (sin_t <= 1e-6) {
    return vec3f(0.0);
  }
  let cos_t = clamp(dot(v1, v2), -1.0, 1.0);
  let theta = atan2(sin_t, cos_t);
  return c * (theta / (2.0 * area_pi * sin_t));
}

fn area_eval_quad(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f) -> f32 {
  let poly = array<vec3f, 4>(p0, p1, p2, p3);
  var clipped = array<vec3f, 5>(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
  var out_n = 0u;
  for (var i = 0u; i < 4u; i++) {
    let cur = poly[i];
    let nxt = poly[(i + 1u) % 4u];
    let cur_in = cur.z >= 0.0;
    let nxt_in = nxt.z >= 0.0;
    if (cur_in && out_n < 5u) {
      clipped[out_n] = cur;
      out_n += 1u;
    }
    if (cur_in != nxt_in && out_n < 5u) {
      let denom = cur.z - nxt.z;
      let t = select(0.5, cur.z / denom, abs(denom) > 1e-7);
      clipped[out_n] = mix(cur, nxt, clamp(t, 0.0, 1.0));
      out_n += 1u;
    }
  }
  if (out_n < 3u) {
    return 0.0;
  }
  var dirs = array<vec3f, 5>(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
  var valid_n = 0u;
  for (var i = 0u; i < out_n; i++) {
    let len2 = dot(clipped[i], clipped[i]);
    if (len2 > 1e-10) {
      let d = clipped[i] * inverseSqrt(len2);
      if (valid_n == 0u || dot(d - dirs[valid_n - 1u], d - dirs[valid_n - 1u]) > 1e-10) {
        dirs[valid_n] = d;
        valid_n += 1u;
      }
    }
  }
  if (valid_n < 3u) {
    return 0.0;
  }
  var sum = vec3f(0.0);
  for (var i = 0u; i < valid_n; i++) {
    sum += area_integrate_edge(dirs[i], dirs[(i + 1u) % valid_n]);
  }
  return clamp(abs(sum.z), 0.0, 1.0);
}

fn area_basis(n: vec3f, v: vec3f) -> mat3x3f {
  let v_proj = v - n * dot(v, n);
  let len2 = dot(v_proj, v_proj);
  var t1: vec3f;
  if (len2 > 1e-6) {
    t1 = v_proj * inverseSqrt(len2);
  } else {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    t1 = normalize(cross(up, n));
  }
  let t2 = cross(n, t1);
  return transpose(mat3x3f(t1, t2, n));
}

const area_cookie_offset: u32 = 8512u;
const area_cookie_dim: u32 = 32u;
const area_cookie_texels: u32 = 1024u;
const area_max_cookies: u32 = 8u;

fn area_sample_cookie(slot: u32, uv: vec2f) -> vec3f {
  let c = clamp(uv, vec2f(0.0), vec2f(1.0)) * f32(area_cookie_dim - 1u);
  let x0 = u32(floor(c.x));
  let y0 = u32(floor(c.y));
  let x1 = min(x0 + 1u, area_cookie_dim - 1u);
  let y1 = min(y0 + 1u, area_cookie_dim - 1u);
  let fx = c.x - f32(x0);
  let fy = c.y - f32(y0);
  let base = area_cookie_offset + min(slot, area_max_cookies - 1u) * area_cookie_texels;
  let t00 = area_light_data[base + y0 * area_cookie_dim + x0].rgb;
  let t10 = area_light_data[base + y0 * area_cookie_dim + x1].rgb;
  let t01 = area_light_data[base + y1 * area_cookie_dim + x0].rgb;
  let t11 = area_light_data[base + y1 * area_cookie_dim + x1].rgb;
  return mix(mix(t00, t10, fx), mix(t01, t11, fx), fy);
}

fn area_range_window(dist: f32, reach: f32) -> f32 {
  if (reach <= 0.0) {
    return 1.0;
  }
  if (dist >= reach) {
    return 0.0;
  }
  let r = dist / reach;
  let w = clamp(1.0 - r * r * r * r, 0.0, 1.0);
  return w * w;
}

// Evaluates all active LTC area lights (Rect, Disk, Tube, CookieSpot) in `area_light_data`
// with full Fresnel endpoints (`f0`, `f90`), lobe energy sharing (transmission/sheen/coat),
// packed RGBA cookies, and directional GGX BRDF for cookie projectors.
// Emitter `col_reach.rgb` is radiance in scene-linear units (a unit-radiance full hemisphere
// produces irradiance 1.0 in Pen's pi-scaled Lambert convention).
fn clustered_area_lights_surface(
  world: vec3f,
  normal: vec3f,
  view: vec3f,
  base_color: vec3f,
  metallic: f32,
  f0: vec3f,
  f90: f32,
  roughness: f32,
  diff_scale: f32,
  spec_scale: f32,
  coat_weight: f32,
  coat_roughness: f32,
  coat_normal: vec3f,
  sheen_tint: vec3f,
) -> vec3f {
  let count = min(u32(max(cluster_grid.options.y, 0.0) + 0.5), area_max_lights);
  if (count == 0u) {
    return vec3f(0.0);
  }
  let n = normalize(normal);
  let v = normalize(view);
  let ndotv = clamp(dot(n, v), 0.0, 1.0);
  let rough = clamp(roughness, 0.04, 1.0);
  let metal = clamp(metallic, 0.0, 1.0);
  let has_sheen = dot(sheen_tint, sheen_tint) > 0.0;

  var ltc_ready = false;
  var basis = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
  var spec_xf = basis;
  var spec_fresnel = vec3f(0.0);
  var diff_albedo = vec3f(0.0);
  var coat_xf = basis;
  var coat_fresnel = 0.0;
  var total = vec3f(0.0);

  for (var i = 0u; i < count; i++) {
    let base = i * 5u;
    let pos_kind = area_light_data[base];
    let col_reach = area_light_data[base + 3u];
    let center = pos_kind.xyz;
    let dist_center = length(center - world);
    let atten = area_range_window(dist_center, col_reach.w);
    if (atten <= 0.0) {
      continue;
    }
    let right_ext = area_light_data[base + 1u];
    let up_ext = area_light_data[base + 2u];
    let flags = area_light_data[base + 4u];

    let kind = u32(round(pos_kind.w));
    let right_len = length(right_ext.xyz);
    let up_len = length(up_ext.xyz);
    if (right_len <= 1e-5 || up_len <= 1e-5) {
      continue;
    }
    let right = right_ext.xyz / right_len;
    let up = up_ext.xyz / up_len;
    let two_sided = flags.x > 0.5;

    if (kind == 0u || kind == 1u) {
      let emit_raw = cross(right, up);
      let emit_len = length(emit_raw);
      if (emit_len <= 1e-5) {
        continue;
      }
      let emit_n = emit_raw / emit_len;
      if (!two_sided && dot(emit_n, world - center) <= 0.0) {
        continue;
      }
    }

    if (!ltc_ready && kind != 3u) {
      let q = sqrt(max(1.0 - ndotv, 0.0));
      let t1 = area_sample_lut(area_mat_offset, rough, q);
      let t2 = area_sample_lut(area_amp_offset, rough, q);
      let minv = mat3x3f(
        vec3f(t1.x, 0.0, t1.y),
        vec3f(0.0, 1.0, 0.0),
        vec3f(t1.z, 0.0, t1.w),
      );
      basis = area_basis(n, v);
      spec_xf = minv * basis;
      spec_fresnel = f0 * t2.x + vec3f(f90 * t2.y);
      // Conserve energy against the LTC directional specular albedo so a unit-radiance hemisphere stays <= 1.
      diff_albedo = base_color * (1.0 - metal) * max(vec3f(1.0) - spec_fresnel, vec3f(0.0)) * diff_scale;
      coat_xf = spec_xf;
      if (coat_weight > 0.0) {
        let cn = normalize(coat_normal);
        let cndv = clamp(dot(cn, v), 0.0, 1.0);
        let cq = sqrt(max(1.0 - cndv, 0.0));
        let cr = clamp(coat_roughness, 0.04, 1.0);
        let ct1 = area_sample_lut(area_mat_offset, cr, cq);
        let ct2 = area_sample_lut(area_amp_offset, cr, cq);
        let cminv = mat3x3f(
          vec3f(ct1.x, 0.0, ct1.y),
          vec3f(0.0, 1.0, 0.0),
          vec3f(ct1.z, 0.0, ct1.w),
        );
        coat_xf = cminv * area_basis(cn, v);
        coat_fresnel = coat_weight * (0.04 * ct2.x + ct2.y);
      }
      ltc_ready = true;
    }

    if (kind == 3u) {
      // Cookie spotlight cone projector evaluated with directional GGX microfacet BRDF.
      let forward = normalize(cross(right, up));
      let rel = world - center;
      let lz = dot(rel, forward);
      if (lz <= 1e-4) {
        continue;
      }
      let tan_fov = max(flags.w, 0.05);
      let aspect = max(up_ext.w, 0.1);
      let u = 0.5 + dot(rel, right) / (2.0 * lz * tan_fov * aspect);
      let v_coord = 0.5 - dot(rel, up) / (2.0 * lz * tan_fov);
      if (u < 0.0 || u > 1.0 || v_coord < 0.0 || v_coord > 1.0) {
        continue;
      }
      let edge = max(flags.z, 0.005);
      let mask = smoothstep(0.0, edge, u) * smoothstep(0.0, edge, 1.0 - u)
        * smoothstep(0.0, edge, v_coord) * smoothstep(0.0, edge, 1.0 - v_coord);
      let cookie_rgb = select(vec3f(1.0), area_sample_cookie(u32(flags.y + 0.5), vec2f(u, v_coord)), flags.y >= 0.0);
      let to_l = -rel / max(dist_center, 1e-4);
      let ndl = max(dot(n, to_l), 0.0);
      if (ndl <= 0.0) {
        continue;
      }
      let falloff = mask / (dist_center * dist_center + 1.0);
      let radiance = col_reach.rgb * cookie_rgb * (falloff * atten);
      let h = normalize(v + to_l);
      let nh = clamp(dot(n, h), 0.0, 1.0);
      let vh = clamp(dot(v, h), 0.0, 1.0);
      let nv = max(ndotv, 1e-5);
      let a = rough * rough;
      let a2 = a * a;
      let denom = nh * nh * (a2 - 1.0) + 1.0;
      let ggx_d = a2 / (3.14159265359 * denom * denom);
      let vis = 0.5 / max(1e-6, ndl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(ndl * ndl * (1.0 - a2) + a2));
      let fade = 1.0 - vh;
      let schlick = f0 + (vec3f(f90) - f0) * (fade * fade * fade * fade * fade);
      let dir_diff = base_color * (1.0 - metal) * max(vec3f(1.0) - schlick, vec3f(0.0)) * (ndl * diff_scale);
      let dir_spec = schlick * (ggx_d * vis * ndl * 3.14159265359 * spec_scale);
      var dir_coat = vec3f(0.0);
      if (coat_weight > 0.0) {
        let cn = normalize(coat_normal);
        let nlc = max(dot(cn, to_l), 0.0);
        let nhc = clamp(dot(cn, h), 0.0, 1.0);
        let nvc = max(dot(cn, v), 1e-5);
        let ca = coat_roughness * coat_roughness;
        let ca2 = ca * ca;
        let cd = nhc * nhc * (ca2 - 1.0) + 1.0;
        let c_ggx = ca2 / (3.14159265359 * cd * cd);
        let c_vis = 0.5 / max(1e-6, nlc * sqrt(nvc * nvc * (1.0 - ca2) + ca2) + nvc * sqrt(nlc * nlc * (1.0 - ca2) + ca2));
        let fc = coat_weight * (0.04 + 0.96 * fade * fade * fade * fade * fade);
        dir_coat = vec3f(fc * c_ggx * c_vis * nlc * 3.14159265359);
      }
      var dir_sheen = vec3f(0.0);
      if (has_sheen) {
        dir_sheen = sheen_tint * ndl;
      }
      total += (dir_diff + dir_spec + dir_sheen + dir_coat) * radiance;
      continue;
    }

    var diff_form = 0.0;
    var spec_form = 0.0;
    var coat_form = 0.0;

    if (kind == 0u) {
      let ex = right * right_ext.w;
      let ey = up * up_ext.w;
      let q0 = (center - ex - ey) - world;
      let q1 = (center + ex - ey) - world;
      let q2 = (center + ex + ey) - world;
      let q3 = (center - ex + ey) - world;
      diff_form = area_eval_quad(basis * q0, basis * q1, basis * q2, basis * q3);
      spec_form = area_eval_quad(spec_xf * q0, spec_xf * q1, spec_xf * q2, spec_xf * q3);
      if (coat_weight > 0.0) {
        coat_form = area_eval_quad(coat_xf * q0, coat_xf * q1, coat_xf * q2, coat_xf * q3);
      }
    } else if (kind == 1u) {
      let rad = max(right_ext.w, 1e-4) * area_octagon_scale;
      let d = rad * 0.70710678;
      let c0 = center + right * rad - world;
      let c1 = center + right * d + up * d - world;
      let c2 = center + up * rad - world;
      let c3 = center - right * d + up * d - world;
      let c4 = center - right * rad - world;
      let c5 = center - right * d - up * d - world;
      let c6 = center - up * rad - world;
      let c7 = center + right * d - up * d - world;
      diff_form = area_eval_quad(basis * c0, basis * c1, basis * c2, basis * c3)
        + area_eval_quad(basis * c4, basis * c5, basis * c6, basis * c7)
        + area_eval_quad(basis * c0, basis * c3, basis * c4, basis * c7);
      spec_form = area_eval_quad(spec_xf * c0, spec_xf * c1, spec_xf * c2, spec_xf * c3)
        + area_eval_quad(spec_xf * c4, spec_xf * c5, spec_xf * c6, spec_xf * c7)
        + area_eval_quad(spec_xf * c0, spec_xf * c3, spec_xf * c4, spec_xf * c7);
      if (coat_weight > 0.0) {
        coat_form = area_eval_quad(coat_xf * c0, coat_xf * c1, coat_xf * c2, coat_xf * c3)
          + area_eval_quad(coat_xf * c4, coat_xf * c5, coat_xf * c6, coat_xf * c7)
          + area_eval_quad(coat_xf * c0, coat_xf * c3, coat_xf * c4, coat_xf * c7);
      }
    } else {
      let half_len = max(right_ext.w, 1e-4);
      let tube_r = max(up_ext.w, 0.01);
      let to_p = world - center;
      let perp_raw = to_p - right * dot(to_p, right);
      let perp_dir = select(up, normalize(perp_raw), dot(perp_raw, perp_raw) > 1e-6);
      let side = normalize(cross(right, perp_dir));
      let ex = right * half_len;
      let ey = side * tube_r;
      let q0 = (center - ex - ey) - world;
      let q1 = (center + ex - ey) - world;
      let q2 = (center + ex + ey) - world;
      let q3 = (center - ex + ey) - world;
      diff_form = area_eval_quad(basis * q0, basis * q1, basis * q2, basis * q3);
      spec_form = area_eval_quad(spec_xf * q0, spec_xf * q1, spec_xf * q2, spec_xf * q3);
      if (coat_weight > 0.0) {
        coat_form = area_eval_quad(coat_xf * q0, coat_xf * q1, coat_xf * q2, coat_xf * q3);
      }
    }

    var contrib = diff_albedo * diff_form + spec_fresnel * (spec_form * spec_scale);
    if (has_sheen) {
      contrib += sheen_tint * diff_form;
    }
    if (coat_weight > 0.0) {
      contrib += vec3f(coat_fresnel * coat_form);
    }
    total += contrib * col_reach.rgb * atten;
  }
  return total;
}

// Convenience wrapper for shaders calling the 7-argument LTC area-light entry point.
fn clustered_area_lights(
  world: vec3f,
  normal: vec3f,
  view: vec3f,
  base_color: vec3f,
  metallic: f32,
  f0: vec3f,
  roughness: f32,
) -> vec3f {
  return clustered_area_lights_surface(
    world, normal, view, base_color, metallic, f0, 1.0, roughness, 1.0, 1.0, 0.0, 0.04, normal, vec3f(0.0),
  );
}
// END lights
// BEGIN environment
// The sky as light for a material shader: diffuse irradiance from nine
// spherical-harmonic coefficients the game works out, and glossy reflections
// from a cube map of the same sky prefiltered by roughness (environment.rs)
// with the split-sum BRDF table. Declare these bindings (group 0 of every
// mesh pipeline, after the scene's) and call `ambient_light` where the flat
// ambient color went, and `sky_reflection` for gloss (`surface_ambient` and
// the rest of the standard surface are in surface.wgsl).
struct SkyLight {
  // The sky's diffuse light, real SH terms of bands 0 to 2 in the order
  // y, z, x (band 1), xy, yz, 3zz - 1, xz, xx - yy (band 2): what a white
  // surface facing a normal shows, cosine lobe applied and divided by pi.
  sh: array<vec4f, 9>,
  // rgb: scale from the cube's sky colors to the light the sky gives all
  // round; w: 1 when the cube holds the frame's sky, 0 for none (an inset's).
  tint: vec4f,
  // x: the sky cube's last level; y: the reflection probes' cube's last level.
  reflection: vec4f,
  // xyz: where probe 0 of the baked probe volume is; w: the spacing between probes.
  probe_origin: vec4f,
  // xyz: probes along x, y and z; w: 1 when the frame has a probe volume.
  probe_counts: vec4f,
  // Screen-space reflections. x: steps of the march, 0 for none; y: the
  // frame's number, which turns the noise; z: how thick a surface is taken to
  // be (world units, at the eye); w: how far a ray reaches.
  ssr: vec4f,
  // x: the layer of `ssr_depth` this frame drew; y: last frame's; z: 1 when
  // `ssr_history` holds last frame's picture; w: refinement steps.
  ssr_layers: vec4f,
  // x, y: elements 10 and 14 of the projection, which give a depth buffer
  // value's distance ahead of the eye.
  ssr_projection: vec4f,
  // The world into the clip space the depth was drawn through (jittered), its
  // inverse, and the world into last frame's clip space (unjittered).
  ssr_clip: mat4x4f,
  ssr_unclip: mat4x4f,
  ssr_previous: mat4x4f,
};
// One reflection probe, as the host holds it after capture.
struct ReflectionProbe {
  // xyz: where the cube was drawn from; w: its slot in the cube array.
  position: vec4f,
  // xyz: the influence box's middle; w: how far inside its faces it fades in.
  centre: vec4f,
  // xyz: the influence box's half size; w: priority.
  extent: vec4f,
  // xyz: the middle of the box the cube is projected onto; w: 1 when it is.
  parallax_centre: vec4f,
  // xyz: that box's half size.
  parallax_extent: vec4f,
};
struct ReflectionProbes {
  // x: how many of `probes` hold a cube. They are ordered by priority, highest first.
  counts: vec4f,
  probes: array<ReflectionProbe, 8>,
};
@group(0) @binding(4) var sky_cube: texture_cube<f32>;
@group(0) @binding(5) var sky_table: texture_2d<f32>;
@group(0) @binding(6) var sky_sampler: sampler;
@group(0) @binding(7) var<uniform> sky_light: SkyLight;
// Seven vec4s a probe: nine RGB terms (27 floats, order as `sky_ambient`) then
// a validity weight of 1 (open) or 0 (inside geometry); probe (z * ny + y) * nx + x.
@group(0) @binding(8) var<storage, read> probe_data: array<vec4f>;
// The reflection probes' cubes, six layers a slot, with the light of the
// diffuse probes' lattice when they were drawn in the alpha (see `ambient_mean`).
@group(0) @binding(10) var reflection_cubes: texture_cube_array<f32>;
@group(0) @binding(11) var<uniform> reflection_probes: ReflectionProbes;
// The camera's depth (layer `ssr_layers.x`) and last frame's (`.y`), and last
// frame's picture, as screen-space reflections read them.
@group(0) @binding(12) var ssr_depth: texture_depth_2d_array;
@group(0) @binding(13) var ssr_history: texture_2d<f32>;

// The sky's diffuse light toward `normal` (unit length): the SH sum, never
// below zero. Multiply it by albedo.
fn sky_ambient(normal: vec3f) -> vec3f {
  let x = normal.x;
  let y = normal.y;
  let z = normal.z;
  var light = sky_light.sh[0].rgb * 0.2820947918;
  light += sky_light.sh[1].rgb * (0.4886025119 * y);
  light += sky_light.sh[2].rgb * (0.4886025119 * z);
  light += sky_light.sh[3].rgb * (0.4886025119 * x);
  light += sky_light.sh[4].rgb * (1.0925484306 * x * y);
  light += sky_light.sh[5].rgb * (1.0925484306 * y * z);
  light += sky_light.sh[6].rgb * (0.3153915653 * (3.0 * z * z - 1.0));
  light += sky_light.sh[7].rgb * (1.0925484306 * x * z);
  light += sky_light.sh[8].rgb * (0.5462742153 * (x * x - y * y));
  return max(light, vec3f(0.0));
}

fn probe_float(at: u32) -> f32 {
  return probe_data[at >> 2u][at & 3u];
}

// What probe `index` shows a surface with SH `terms` of its normal.
fn probe_light(index: u32, terms: array<f32, 9>) -> vec3f {
  let base = index * 28u;
  var light = vec3f(0.0);
  for (var i = 0u; i < 9u; i++) {
    light += vec3f(probe_float(base + i * 3u), probe_float(base + i * 3u + 1u), probe_float(base + i * 3u + 2u)) * terms[i];
  }
  return max(light, vec3f(0.0));
}

// The probe volume's light at `world` facing `normal` (unit length): rgb, and
// in w how much of it to trust (0 outside the lattice, fading in over a cell
// beyond its edge; the share of valid corners inside). Trilinear between the
// eight probes round the point lifted a quarter spacing off the surface,
// each weighed by its validity and by how much it faces the surface, so a
// wall does not hand the light of its other side through. A probe's own
// position gives exactly that probe's light.
fn probe_ambient(world: vec3f, normal: vec3f) -> vec4f {
  let counts = vec3u(sky_light.probe_counts.xyz);
  let spacing = sky_light.probe_origin.w;
  let origin = sky_light.probe_origin.xyz;
  let lifted = world + normal * (0.25 * spacing);
  let g = (lifted - origin) / spacing;
  let last = vec3f(counts) - vec3f(1.0);
  let edge = min(min(min(g.x, last.x - g.x), min(g.y, last.y - g.y)), min(g.z, last.z - g.z));
  let reach = clamp(edge + 1.0, 0.0, 1.0);
  if (reach <= 0.0) {
    return vec4f(0.0);
  }
  let c = clamp(g, vec3f(0.0), last);
  let low = vec3u(min(floor(c), max(last - vec3f(1.0), vec3f(0.0))));
  let f = c - vec3f(low);
  let x = normal.x;
  let y = normal.y;
  let z = normal.z;
  var terms = array<f32, 9>(
    0.2820947918, 0.4886025119 * y, 0.4886025119 * z, 0.4886025119 * x,
    1.0925484306 * x * y, 1.0925484306 * y * z, 0.3153915653 * (3.0 * z * z - 1.0),
    1.0925484306 * x * z, 0.5462742153 * (x * x - y * y));
  var sum = vec3f(0.0);
  var weight = 0.0;
  var share = 0.0;
  var total = 0.0;
  for (var corner = 0u; corner < 8u; corner++) {
    let d = vec3u(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
    let at = min(low + d, counts - vec3u(1u));
    let w3 = select(vec3f(1.0) - f, f, d == vec3u(1u));
    let tri = w3.x * w3.y * w3.z;
    if (tri <= 0.0) {
      continue;
    }
    let index = (at.z * counts.y + at.y) * counts.x + at.x;
    let valid = probe_float(index * 28u + 27u);
    total += tri;
    share += tri * valid;
    if (valid > 0.0) {
      let toward = origin + vec3f(at) * spacing - world;
      let gap = length(toward);
      let facing = select(0.5 + 0.5 * dot(toward, normal) / gap, 1.0, gap < 0.0001);
      let w = tri * valid * max(0.02, facing * facing);
      sum += w * probe_light(index, terms);
      weight += w;
    }
  }
  if (weight <= 0.000001 || total <= 0.0) {
    return vec4f(0.0);
  }
  return vec4f(sum / weight, reach * reach * (3.0 - 2.0 * reach) * (share / total));
}

// Live GPU DDGI dual-cascade lookup packed into `probe_data` (activated when `sky_light.probe_counts.w >= 1.5`).
fn ddgi_env_sign_nz(v: vec2f) -> vec2f {
  return vec2f(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
}

fn ddgi_env_oct_encode(v: vec3f) -> vec2f {
  let l1 = abs(v.x) + abs(v.y) + abs(v.z);
  var p = v.xy / max(l1, 1e-8);
  if (v.z < 0.0) {
    p = (1.0 - abs(p.yx)) * ddgi_env_sign_nz(p);
  }
  return p;
}

fn ddgi_env_wrap(v: i32, n: u32) -> u32 {
  let ni = max(i32(n), 1);
  return u32(((v % ni) + ni) % ni);
}

fn ddgi_env_sample_irr(irr_off: u32, w: u32, h: u32, fpx: vec2f) -> vec4f {
  let p = fpx - vec2f(0.5);
  let x0 = u32(clamp(floor(p.x), 0.0, f32(w - 1u)));
  let y0 = u32(clamp(floor(p.y), 0.0, f32(h - 1u)));
  let x1 = min(x0 + 1u, w - 1u);
  let y1 = min(y0 + 1u, h - 1u);
  let fx = clamp(p.x - f32(x0), 0.0, 1.0);
  let fy = clamp(p.y - f32(y0), 0.0, 1.0);
  let c00 = probe_data[irr_off + y0 * w + x0];
  let c10 = probe_data[irr_off + y0 * w + x1];
  let c01 = probe_data[irr_off + y1 * w + x0];
  let c11 = probe_data[irr_off + y1 * w + x1];
  return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

fn ddgi_env_vis_at(vis_off: u32, w: u32, x: u32, y: u32) -> vec2f {
  let idx = y * w + x;
  let packed = probe_data[vis_off + (idx >> 1u)];
  return select(packed.xy, packed.zw, (idx & 1u) != 0u);
}

fn ddgi_env_sample_vis(vis_off: u32, w: u32, h: u32, fpx: vec2f) -> vec2f {
  let p = fpx - vec2f(0.5);
  let x0 = u32(clamp(floor(p.x), 0.0, f32(w - 1u)));
  let y0 = u32(clamp(floor(p.y), 0.0, f32(h - 1u)));
  let x1 = min(x0 + 1u, w - 1u);
  let y1 = min(y0 + 1u, h - 1u);
  let fx = clamp(p.x - f32(x0), 0.0, 1.0);
  let fy = clamp(p.y - f32(y0), 0.0, 1.0);
  let c00 = ddgi_env_vis_at(vis_off, w, x0, y0);
  let c10 = ddgi_env_vis_at(vis_off, w, x1, y0);
  let c01 = ddgi_env_vis_at(vis_off, w, x0, y1);
  let c11 = ddgi_env_vis_at(vis_off, w, x1, y1);
  return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

fn ddgi_env_cascade(cascade_idx: u32, world: vec3f, normal: vec3f, view_dir: vec3f) -> vec4f {
  let base = 1u + cascade_idx * 6u;
  let h0 = probe_data[base + 0u];
  let h1 = probe_data[base + 1u];
  let h2 = probe_data[base + 2u];
  let h3 = probe_data[base + 3u];
  let h4 = probe_data[base + 4u];
  let h5 = probe_data[base + 5u];
  let dims = vec3u(u32(max(h2.x, 0.0)), u32(max(h2.y, 0.0)), u32(max(h2.z, 0.0)));
  if (dims.x < 2u || dims.y < 2u || dims.z < 2u) {
    return vec4f(0.0);
  }
  let origin = h0.xyz;
  let gamma = max(h0.w, 1.0);
  let spacing = max(h1.xyz, vec3f(1e-4));
  let nbias = h1.w;
  let vbias = h2.w;
  let scroll = vec3i(i32(round(h3.x)), i32(round(h3.y)), i32(round(h3.z)));
  let irr_off = u32(max(h3.w, 0.0));
  let vis_off = u32(max(h4.x, 0.0));
  let strength = max(h4.y, 0.0);
  let irr_w = u32(max(h4.z, 1.0));
  let irr_h = u32(max(h4.w, 1.0));
  let vis_w = u32(max(h5.x, 1.0));
  let vis_h = u32(max(h5.y, 1.0));
  let offsets_off = u32(max(h5.z, 0.0));

  let min_sp = min(spacing.x, min(spacing.y, spacing.z));
  let biased_pos = world + (normal * nbias + view_dir * vbias) * min_sp;
  let rel = (biased_pos - origin) / spacing;
  let max_c = vec3f(f32(dims.x - 1u), f32(dims.y - 1u), f32(dims.z - 1u));
  let edge = min(min(min(rel.x, max_c.x - rel.x), min(rel.y, max_c.y - rel.y)), min(rel.z, max_c.z - rel.z));
  let reach = clamp(edge + 1.0, 0.0, 1.0);
  if (reach <= 0.0) {
    return vec4f(0.0);
  }
  let clamped = clamp(rel, vec3f(0.0), max_c);
  let base_i = vec3u(floor(clamped));
  let alpha = clamped - vec3f(base_i);
  let oct_n = ddgi_env_oct_encode(normalize(normal)) * 0.5 + vec2f(0.5);

  var sum_irr = vec3f(0.0);
  var sum_w = 0.0;
  var share = 0.0;
  var total = 0.0;
  for (var i = 0u; i < 8u; i++) {
    let off = vec3u(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
    let coord = min(base_i + off, dims - vec3u(1u));
    let sc = vec3u(
      ddgi_env_wrap(i32(coord.x) + scroll.x, dims.x),
      ddgi_env_wrap(i32(coord.y) + scroll.y, dims.y),
      ddgi_env_wrap(i32(coord.z) + scroll.z, dims.z),
    );
    let trilinear = mix(vec3f(1.0) - alpha, alpha, vec3f(off));
    let tri = max(trilinear.x * trilinear.y * trilinear.z, 1e-4);
    var w = tri;

    let s_idx = (sc.y * dims.z + sc.z) * dims.x + sc.x;
    let probe_off = select(vec3f(0.0), probe_data[offsets_off + s_idx].xyz, offsets_off > 0u);
    let probe_pos = origin + vec3f(coord) * spacing + probe_off;
    let to_probe = probe_pos - world;
    let dist_u = length(to_probe);
    let dir_to_probe = select(normal, to_probe / dist_u, dist_u > 1e-5);
    let wrap = (dot(dir_to_probe, normal) + 1.0) * 0.5;
    w *= wrap * wrap + 0.05;

    let p2pt = biased_pos - probe_pos;
    let d_pt = length(p2pt);
    let dir_from_probe = select(-normal, p2pt / d_pt, d_pt > 1e-5);
    let oct_v = ddgi_env_oct_encode(dir_from_probe) * 0.5 + vec2f(0.5);
    let s_col = sc.z * dims.x + sc.x;
    let s_row = sc.y;
    let vis_px = vec2f(f32(s_col * 18u + 1u), f32(s_row * 18u + 1u)) + oct_v * 16.0;
    let moments = ddgi_env_sample_vis(vis_off, vis_w, vis_h, vis_px);
    if (d_pt > moments.x) {
      let variance = max(abs(moments.y - moments.x * moments.x), 1e-4);
      let d = d_pt - moments.x;
      var cheb = variance / (variance + d * d);
      cheb = max(cheb * cheb * cheb, 0.0);
      w *= cheb;
    }

    let irr_px = vec2f(f32(s_col * 10u + 1u), f32(s_row * 10u + 1u)) + oct_n * 8.0;
    let texel = ddgi_env_sample_irr(irr_off, irr_w, irr_h, irr_px);
    let valid = clamp(texel.a, 0.0, 1.0);
    total += tri;
    share += tri * valid;
    w *= valid;
    if (w <= 1e-6) {
      continue;
    }
    let encoded = max(texel.rgb, vec3f(0.0));
    var irr: vec3f;
    if (abs(gamma - 5.0) < 1e-4) {
      irr = encoded * encoded * sqrt(encoded);
    } else {
      irr = pow(encoded, vec3f(gamma * 0.5));
    }
    sum_irr += irr * w;
    sum_w += w;
  }
  if (sum_w <= 1e-6 || total <= 0.0) {
    return vec4f(0.0);
  }
  let avg = sum_irr / sum_w;
  let trust = reach * reach * (3.0 - 2.0 * reach) * clamp(share / total, 0.0, 1.0);
  return vec4f(avg * avg * strength, trust);
}

fn ddgi_packed_ambient(world: vec3f, normal: vec3f, view_dir: vec3f) -> vec4f {
  let count = u32(clamp(round(probe_data[0u].x), 0.0, 2.0));
  if (count == 0u) {
    return vec4f(0.0);
  }
  let c0 = ddgi_env_cascade(0u, world, normal, view_dir);
  if (count == 1u || c0.a >= 0.999) {
    return c0;
  }
  let c1 = ddgi_env_cascade(1u, world, normal, view_dir);
  let blended = mix(c1.rgb, c0.rgb, c0.a);
  return vec4f(blended, max(c0.a, c1.a));
}

// The light all round a surface at `world` facing `normal` (unit length), seen
// from `view` (unit length toward eye), to multiply by diffuse albedo:
// live GPU DDGI cascades when enabled (`probe_counts.w >= 1.5`), else baked SH
// probe volume (`>= 0.5`), blending smoothly to sky SH past the volume's edge.
fn ambient_light_view(world: vec3f, normal: vec3f, view: vec3f) -> vec3f {
  let sky = sky_ambient(normal);
  if (sky_light.probe_counts.w >= 1.5) {
    let ddgi = ddgi_packed_ambient(world, normal, view);
    return mix(sky, ddgi.rgb, ddgi.a);
  }
  if (sky_light.probe_counts.w < 0.5) {
    return sky;
  }
  let probes = probe_ambient(world, normal);
  return mix(sky, probes.rgb, probes.a);
}

// The light all round a surface at `world` facing `normal` (unit length), to
// multiply by albedo: live DDGI cascades, baked SH probe volume, or sky SH.
fn ambient_light(world: vec3f, normal: vec3f) -> vec3f {
  return ambient_light_view(world, normal, normal);
}

// The mean of the light all round `world`, as a luminance: the average of
// `ambient_light` over every direction, which is its constant SH term alone.
// A reflection probe keeps it from when it was drawn, and scales what it
// shows by how the light has changed since (`probe_cube_light`).
fn ambient_mean(world: vec3f) -> f32 {
  let weights = vec3f(0.2126, 0.7152, 0.0722);
  let sky = dot(sky_light.sh[0].rgb, weights) * 0.2820947918;
  if (sky_light.probe_counts.w >= 1.5) {
    let up = ddgi_packed_ambient(world, vec3f(0.0, 1.0, 0.0), vec3f(0.0, 1.0, 0.0));
    return mix(sky, dot(up.rgb, weights), up.a);
  }
  if (sky_light.probe_counts.w < 0.5) {
    return sky;
  }
  let counts = vec3u(sky_light.probe_counts.xyz);
  let spacing = sky_light.probe_origin.w;
  let g = (world - sky_light.probe_origin.xyz) / spacing;
  let last = vec3f(counts) - vec3f(1.0);
  let edge = min(min(min(g.x, last.x - g.x), min(g.y, last.y - g.y)), min(g.z, last.z - g.z));
  let reach = clamp(edge + 1.0, 0.0, 1.0);
  if (reach <= 0.0) {
    return sky;
  }
  let c = clamp(g, vec3f(0.0), last);
  let low = vec3u(min(floor(c), max(last - vec3f(1.0), vec3f(0.0))));
  let f = c - vec3f(low);
  var sum = 0.0;
  var weight = 0.0;
  var share = 0.0;
  var total = 0.0;
  for (var corner = 0u; corner < 8u; corner++) {
    let d = vec3u(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
    let at = min(low + d, counts - vec3u(1u));
    let w3 = select(vec3f(1.0) - f, f, d == vec3u(1u));
    let tri = w3.x * w3.y * w3.z;
    if (tri <= 0.0) {
      continue;
    }
    let index = (at.z * counts.y + at.y) * counts.x + at.x;
    let valid = probe_float(index * 28u + 27u);
    total += tri;
    share += tri * valid;
    if (valid > 0.0) {
      let term = vec3f(probe_float(index * 28u), probe_float(index * 28u + 1u), probe_float(index * 28u + 2u));
      sum += tri * valid * dot(term, weights) * 0.2820947918;
      weight += tri * valid;
    }
  }
  if (weight <= 0.000001 || total <= 0.0) {
    return sky;
  }
  let trust = reach * reach * (3.0 - 2.0 * reach) * (share / total);
  return mix(sky, sum / weight, trust);
}

// The split-sum table at `normal_dot_view` and `roughness`: x scales f0 and y
// adds to it. The same GGX, height-correlated Smith and Schlick Fresnel as the
// point and spot lights'.
fn sky_brdf(normal_dot_view: f32, roughness: f32) -> vec2f {
  let uv = vec2f(clamp(normal_dot_view, 0.0, 1.0), clamp(roughness, 0.0, 1.0));
  return textureSampleLevel(sky_table, sky_sampler, uv, 0.0).rg;
}

// How much of a reflection probe's box a point at `world` is in: 0 outside it,
// and inside it rising from 0 at a face to 1 at `blend` in from every face
// (smoothly: the smoothstep of the distance to the nearest face over `blend`),
// or at once to 1 for a blend of 0.
fn reflection_weight(probe: ReflectionProbe, world: vec3f) -> f32 {
  let inside = probe.extent.xyz - abs(world - probe.centre.xyz);
  let edge = min(inside.x, min(inside.y, inside.z));
  if (edge <= 0.0) {
    return 0.0;
  }
  let blend = probe.centre.w;
  let t = select(1.0, clamp(edge / blend, 0.0, 1.0), blend > 0.0);
  return t * t * (3.0 - 2.0 * t);
}

// The direction from a probe's capture point to where the ray from `world`
// along `direction` leaves the probe's parallax box: looking the cube up
// there puts a wall of the room where the room's wall is, not at infinity.
// A probe without a box takes `direction` as it is.
fn box_projected(probe: ReflectionProbe, world: vec3f, direction: vec3f) -> vec3f {
  if (probe.parallax_centre.w < 0.5) {
    return direction;
  }
  let low = probe.parallax_centre.xyz - probe.parallax_extent.xyz;
  let high = probe.parallax_centre.xyz + probe.parallax_extent.xyz;
  let way = select(direction, vec3f(1e-8), abs(direction) < vec3f(1e-8));
  let leaves = (select(low, high, way > vec3f(0.0)) - world) / way;
  let along = max(min(leaves.x, min(leaves.y, leaves.z)), 0.0);
  return world + direction * along - probe.position.xyz;
}

// What reflection probe `probe` shows toward `direction` from `world` at
// `roughness`: its cube at the box-projected direction, in the level of that
// roughness, scaled by how much the light of the lattice at the probe has
// changed since it was drawn (alpha holds its mean then), within a factor of
// four either way, so a probe drawn under other light does not glow.
fn probe_cube_light(probe: ReflectionProbe, world: vec3f, direction: vec3f, roughness: f32) -> vec3f {
  let lookup = box_projected(probe, world, direction);
  let level = clamp(roughness, 0.0, 1.0) * sky_light.reflection.y;
  let texel = textureSampleLevel(reflection_cubes, sky_sampler, lookup, i32(probe.position.w), level);
  var scale = 1.0;
  if (texel.a > 0.0001) {
    scale = clamp(ambient_mean(probe.position.xyz) / texel.a, 0.25, 4.0);
  }
  return texel.rgb * scale;
}

// The light of the reflection probes toward `direction` from `world`, and how
// much of the reflection they cover (the sky shows through the rest). Probes
// take the surface highest priority first: each priority's probes blend by
// their weights into one light, covering what the best of them does
// (1 - the product of the misses), and that takes its share of what the higher
// priorities left. At most four probes with a weight count. rgb comes
// already multiplied by the coverage.
fn local_reflection(world: vec3f, direction: vec3f, roughness: f32) -> vec4f {
  let count = min(u32(reflection_probes.counts.x), 8u);
  var total = vec3f(0.0);
  var remaining = 1.0;
  var sum = vec3f(0.0);
  var weight = 0.0;
  var missed = 1.0;
  var priority = -1.0;
  var used = 0u;
  for (var i = 0u; i < count; i++) {
    let probe = reflection_probes.probes[i];
    let w = reflection_weight(probe, world);
    if (w <= 0.0) {
      continue;
    }
    if (used >= 4u) {
      break;
    }
    used += 1u;
    if (probe.extent.w != priority) {
      if (weight > 0.0) {
        total += remaining * (1.0 - missed) * sum / weight;
        remaining *= missed;
      }
      sum = vec3f(0.0);
      weight = 0.0;
      missed = 1.0;
      priority = probe.extent.w;
    }
    sum += w * probe_cube_light(probe, world, direction, roughness);
    weight += w;
    missed *= 1.0 - w;
  }
  if (weight > 0.0) {
    total += remaining * (1.0 - missed) * sum / weight;
    remaining *= missed;
  }
  return vec4f(total, 1.0 - remaining);
}

// Interleaved gradient noise of a pixel and the frame, in [0, 1): fixed per
// pixel within a frame, and moved from frame to frame so that temporal
// antialiasing averages it.
fn ssr_noise(pixel: vec2f, seed: f32) -> f32 {
  let shifted = pixel + vec2f(5.588238) * (sky_light.ssr.y + seed);
  return fract(52.9829189 * fract(dot(shifted, vec2f(0.06711056, 0.00583715))));
}

// A depth buffer value's distance ahead of the eye.
fn ssr_distance(depth: f32) -> f32 {
  return sky_light.ssr_projection.y / (depth + sky_light.ssr_projection.x);
}

// The world position of the surface the depth buffer has at a pixel position
// (in pixels, a texel's middle at +0.5).
fn ssr_position(pixel: vec2f, depth: f32, size: vec2f) -> vec3f {
  let ndc = vec2f(pixel.x / size.x * 2.0 - 1.0, 1.0 - pixel.y / size.y * 2.0);
  let point = sky_light.ssr_unclip * vec4f(ndc, depth, 1.0);
  return point.xyz / point.w;
}

fn ssr_depth_at(texel: vec2i, layer: i32, size: vec2i) -> f32 {
  return textureLoad(ssr_depth, clamp(texel, vec2i(0), size - vec2i(1)), layer, 0);
}

// Where a ray from `origin` along `direction` (unit) first meets the depth
// buffer, marched in the pixels it crosses: linear steps with a pixel
// stride and the ray's depth as the perspective-correct line it is, then
// halving between the last step in front and the first behind. The surface
// it meets must lie within a thickness of the ray (a ray passing behind a
// thin one goes on). xy: the pixel; w: 1 for a hit, 0 for none.
fn ssr_march(origin: vec3f, direction: vec3f, pixel_start: vec2f, size: vec2f, layer: i32) -> vec4f {
  let miss = vec4f(0.0);
  let steps = i32(sky_light.ssr.x);
  let clip0 = sky_light.ssr_clip * vec4f(origin, 1.0);
  let near = 0.05;
  if (clip0.w <= near) {
    return miss;
  }
  let reach = sky_light.ssr.w;
  var end = origin + direction * reach;
  var clip1 = sky_light.ssr_clip * vec4f(end, 1.0);
  if (clip1.w < near) {
    // Cut where the ray crosses the near plane.
    let cut = (clip0.w - near) / (clip0.w - clip1.w);
    end = origin + direction * (reach * cut);
    clip1 = sky_light.ssr_clip * vec4f(end, 1.0);
  }
  let k0 = 1.0 / clip0.w;
  let k1 = 1.0 / clip1.w;
  let z0 = clip0.z * k0;
  let z1 = clip1.z * k1;
  let p0 = vec2f(clip0.x * k0 * 0.5 + 0.5, 0.5 - clip0.y * k0 * 0.5) * size;
  let p1 = vec2f(clip1.x * k1 * 0.5 + 0.5, 0.5 - clip1.y * k1 * 0.5) * size;
  let delta = p1 - p0;
  let span = max(abs(delta.x), abs(delta.y));
  if (span < 1.0) {
    return miss;
  }
  // The share of the segment that stays on the screen.
  var inside = 1.0;
  if (delta.x > 0.0) { inside = min(inside, (size.x - 1.0 - p0.x) / delta.x); }
  if (delta.x < 0.0) { inside = min(inside, (0.0 - p0.x) / delta.x); }
  if (delta.y > 0.0) { inside = min(inside, (size.y - 1.0 - p0.y) / delta.y); }
  if (delta.y < 0.0) { inside = min(inside, (0.0 - p0.y) / delta.y); }
  let visible = span * max(inside, 0.0);
  if (visible < 1.0) {
    return miss;
  }
  let stride = max(visible / f32(steps), 1.0);
  let count = min(steps, i32(ceil(visible / stride)));
  let jitter = ssr_noise(pixel_start, 3.0);
  let size_i = vec2i(size);
  var previous = 0.0;
  // How far the ray was behind the surface at the last sample (negative: in front),
  // and how far that surface was.
  var last_behind = -1.0;
  var last_scene = -1.0;
  for (var i = 0; i < count; i++) {
    // How far along the full segment, as a share of its pixels.
    let s = (f32(i) + 1.0 + jitter) * stride / span;
    if (s > inside) {
      break;
    }
    let pixel = p0 + delta * s;
    let depth = ssr_depth_at(vec2i(floor(pixel)), layer, size_i);
    let ahead = ssr_distance(mix(z0, z1, s));
    let step_gap = abs(ahead - ssr_distance(mix(z0, z1, previous)));
    if (depth < 1.0) {
      let scene = ssr_distance(depth);
      let behind = ahead - scene;
      // A surface seen at a grazing angle changes its depth a good deal from one
      // step to the next, so the ray may be well behind it at the first sample
      // past it; allow that slope (within a quarter of the distance, so that the
      // jump to a different, nearer surface does not count) as well as the
      // thickness and the depth the ray itself crosses in a step.
      var slope = 0.0;
      if (last_scene > 0.0) {
        slope = min(abs(scene - last_scene), 0.25 * scene);
      }
      let thick = sky_light.ssr.z * scene + step_gap + slope;
      // The ray must have come from in front of it, not from behind it already.
      if (behind > 0.0 && behind < thick && last_behind <= 0.0) {
        // Halve between the last step and this one.
        var low = previous;
        var high = s;
        for (var r = 0; r < i32(sky_light.ssr_layers.w); r++) {
          let mid = 0.5 * (low + high);
          let found = ssr_depth_at(vec2i(floor(p0 + delta * mid)), layer, size_i);
          if (found < 1.0 && ssr_distance(mix(z0, z1, mid)) > ssr_distance(found)) {
            high = mid;
          } else {
            low = mid;
          }
        }
        let hit = p0 + delta * high;
        return vec4f(hit, 0.0, 1.0);
      }
      last_behind = behind;
      last_scene = scene;
    } else {
      last_behind = -1.0;
    }
    previous = s;
  }
  return miss;
}

// The surface's normal at a pixel from the depth around it, toward the eye:
// across whichever neighbor lies closer in depth on each axis, so an edge
// does not bend it.
fn ssr_normal(texel: vec2i, centre: vec3f, depth: f32, layer: i32, size: vec2i, sized: vec2f) -> vec3f {
  let right = ssr_depth_at(texel + vec2i(1, 0), layer, size);
  let left = ssr_depth_at(texel - vec2i(1, 0), layer, size);
  let down = ssr_depth_at(texel + vec2i(0, 1), layer, size);
  let up = ssr_depth_at(texel - vec2i(0, 1), layer, size);
  let here = ssr_distance(depth);
  let use_right = abs(ssr_distance(right) - here) < abs(ssr_distance(left) - here);
  let use_down = abs(ssr_distance(down) - here) < abs(ssr_distance(up) - here);
  let middle = vec2f(texel) + vec2f(0.5);
  let across = select(
    centre - ssr_position(middle - vec2f(1.0, 0.0), left, sized),
    ssr_position(middle + vec2f(1.0, 0.0), right, sized) - centre,
    use_right);
  let along = select(
    centre - ssr_position(middle - vec2f(0.0, 1.0), up, sized),
    ssr_position(middle + vec2f(0.0, 1.0), down, sized) - centre,
    use_down);
  // `across` runs right on the screen and `along` down, so this faces the eye.
  return normalize(cross(along, across));
}

// A glossy surface's screen-space reflection: rgb the light a ray from
// `world` meets in the last frame's picture where the camera's depth says it
// hits something, and a in 0..1 how far to trust it. It fades with roughness
// (the ray's direction is jittered through a GGX lobe, which temporal
// antialiasing averages), toward the screen's edges, along the ray's reach,
// and where the last frame did not see that surface (it moved or was hidden).
// A ray that hits a surface facing away from it, the back of what the
// camera sees the front of, hits nothing.
fn screen_reflection(world: vec3f, normal: vec3f, view: vec3f, roughness: f32) -> vec4f {
  let miss = vec4f(0.0);
  if (sky_light.ssr.x < 0.5 || sky_light.ssr_layers.z < 0.5 || roughness >= 0.8) {
    return miss;
  }
  let size_u = textureDimensions(ssr_depth);
  let size = vec2f(size_u);
  let size_i = vec2i(size_u);
  let clip = sky_light.ssr_clip * vec4f(world, 1.0);
  if (clip.w <= 0.05) {
    return miss;
  }
  let start = vec2f(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5) * size;
  if (any(start < vec2f(0.0)) || any(start >= size)) {
    return miss;
  }
  // A direction through the GGX lobe of the surface's roughness.
  var direction = reflect(-view, normal);
  if (roughness > 0.04) {
    let a = roughness * roughness;
    // The lobe's farthest tenth is left out: one sample in a thousand that far out would
    // meet a different wall altogether, and be a speck that temporal antialiasing cannot
    // average away.
    let u = vec2f(ssr_noise(start, 0.0), 0.9 * ssr_noise(start, 1.0));
    let cos_theta = sqrt((1.0 - u.y) / (1.0 + (a * a - 1.0) * u.y));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = 6.2831853 * u.x;
    let tangent = normalize(cross(select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.999), normal));
    let bitangent = cross(normal, tangent);
    let half_way = tangent * (sin_theta * cos(phi)) + bitangent * (sin_theta * sin(phi)) + normal * cos_theta;
    let jittered = reflect(-view, half_way);
    if (dot(jittered, normal) > 0.0) {
      direction = jittered;
    }
  }
  let layer = i32(sky_light.ssr_layers.x);
  let away = ssr_distance(clip.z / clip.w);
  let origin = world + normal * (0.003 + 0.004 * away);
  let found = ssr_march(origin, direction, start, size, layer);
  if (found.w < 0.5) {
    return miss;
  }
  let texel = vec2i(floor(found.xy));
  let depth = ssr_depth_at(texel, layer, size_i);
  let hit = ssr_position(vec2f(texel) + vec2f(0.5), depth, size);
  let hit_normal = ssr_normal(texel, hit, depth, layer, size_i, size);
  // The surface met must face the ray that meets it.
  if (dot(hit_normal, direction) >= 0.0) {
    return miss;
  }
  let travelled = length(hit - origin);
  // Where the last frame saw that point, and whether it was a surface there.
  let before = sky_light.ssr_previous * vec4f(hit, 1.0);
  if (before.w <= 0.05) {
    return miss;
  }
  let uv = vec2f(before.x / before.w * 0.5 + 0.5, 0.5 - before.y / before.w * 0.5);
  if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) {
    return miss;
  }
  let last = ssr_depth_at(vec2i(uv * size), i32(sky_light.ssr_layers.y), size_i);
  if (last >= 1.0) {
    return miss;
  }
  let error = abs(ssr_distance(last) - before.w) / before.w;
  let same = 1.0 - smoothstep(0.02, 0.06, error);
  let colour = min(textureSampleLevel(ssr_history, sky_sampler, uv, 0.0).rgb, vec3f(64.0));
  let edge = min(min(found.x, size.x - found.x) / size.x, min(found.y, size.y - found.y) / size.y);
  let on_screen = smoothstep(0.0, 0.08, edge);
  let near_end = 1.0 - smoothstep(0.6, 1.0, travelled / sky_light.ssr.w);
  let glossy = 1.0 - smoothstep(0.5, 0.8, roughness);
  return vec4f(colour, same * on_screen * near_end * glossy);
}

// The light a surface at `world` mirrors, looked up along the reflection of
// `view` about `normal` (both unit length, `view` toward the eye), of
// `roughness` (0 mirror to 1 matte), times the split-sum table's scale and
// bias for a face-on reflectance `f0` and a grazing one `f90`, read at
// `normal_dot_view`: what it reflects (a ray from it meets the screen's
// picture, the local reflection probes it is in the boxes of, or else the
// prefiltered sky). The picture takes the probes' and the sky's place by its
// confidence; the probes take the sky's by theirs. Zero where there is none
// of these. `normal` is where to look and `normal_dot_view` how much to
// reflect: they are the same cosine unless the lookup is bent (an anisotropic
// surface) or belongs to another layer's normal (a clear coat).
fn sky_reflection_lobe_split(world: vec3f, normal: vec3f, view: vec3f, roughness: f32, f0: vec3f, f90: f32, split: vec2f) -> vec3f {
  let probes = reflection_probes.counts.x > 0.5;
  let traced = sky_light.ssr.x > 0.5;
  if (sky_light.tint.w < 0.5 && !probes && !traced) {
    return vec3f(0.0);
  }
  let rough = clamp(roughness, 0.04, 1.0);
  let fresnel_scale = f0 * split.x + vec3f(f90 * split.y);
  var screen = vec4f(0.0);
  if (traced) {
    screen = screen_reflection(world, normal, view, rough);
    if (screen.a >= 0.999) {
      return screen.rgb * fresnel_scale;
    }
  }
  let toward = reflect(-view, normal);
  var radiance = vec3f(0.0);
  var local_cov = 0.0;
  if (probes) {
    let local = local_reflection(world, toward, rough);
    radiance = local.rgb;
    local_cov = local.a;
  }
  if (local_cov < 0.999 && sky_light.tint.w >= 0.5) {
    let sky_r = textureSampleLevel(sky_cube, sky_sampler, toward, rough * sky_light.reflection.x).rgb * sky_light.tint.rgb;
    radiance += sky_r * (1.0 - local_cov);
  }
  if (traced) {
    radiance = mix(radiance, screen.rgb, screen.a);
  }
  return radiance * fresnel_scale;
}

fn sky_reflection_lobe(world: vec3f, normal: vec3f, view: vec3f, normal_dot_view: f32, roughness: f32, f0: vec3f, f90: f32) -> vec3f {
  let nv = max(normal_dot_view, 0.0001);
  let rough = clamp(roughness, 0.04, 1.0);
  let split = sky_brdf(nv, rough);
  return sky_reflection_lobe_split(world, normal, view, rough, f0, f90, split);
}

// The light a surface at `world` facing `normal`, seen along `view` (both unit
// length, `view` toward the eye), of `roughness` (0 mirror to 1 matte) and
// with Fresnel reflectance `f0` face on, mirrors (`sky_reflection_lobe`).
fn sky_reflection(world: vec3f, normal: vec3f, view: vec3f, roughness: f32, f0: vec3f) -> vec3f {
  return sky_reflection_lobe(world, normal, view, dot(normal, view), roughness, f0, 1.0);
}

// How much of the sky's reflection a surface with ambient occlusion
// `occlusion` (1 open to 0 closed) shows: a glossy lobe is less hidden than a
// rough one, and a grazing view more.
fn specular_occlusion(normal_dot_view: f32, occlusion: f32, roughness: f32) -> f32 {
  return clamp(pow(normal_dot_view + occlusion, exp2(-16.0 * roughness - 1.0)) - 1.0 + occlusion, 0.0, 1.0);
}
// END environment
// BEGIN surface_lobes
// The lobes of a standard surface, as pure functions with no bindings: the
// microfacet base (GGX, height-correlated Smith, Schlick), a clear coat, sheen
// (Charlie with Neubelt's visibility), anisotropic GGX, a cheap subsurface
// (wrap and a back-lit term), the transmission and volume maths, and parallax
// occlusion mapping. `SURFACE_WGSL` (the sky, the frame's lights and the sun)
// and `SURFACE_TRANSMISSION_WGSL` build on them; a test can run any of them in
// a compute shader on its own.
//
// Conventions are Ink's `lighting.brdf`: a lobe's response is per unit
// radiance with the cosine in it. The frame's lights are scaled so that a
// diffuse surface is albedo times their color times the cosine, with no
// division by pi, so every lobe that comes out of `surface_light` is Ink's
// times pi (the specular ones here; the diffuse one is Lambert's 1 / pi * pi).
// The lobes below return Ink's own values; the multiplication by pi is
// `surface_light`'s.
//
// A shader builds a `StandardSurface` (`surface_default`, then sets what its
// material has), says which lobes exist in `lobes` (constants, so that the
// compiler folds the ones it leaves out, and a check of the parameter, so a
// weight of 0 costs nothing either) and calls `standard_surface`.
//
// Names: `surface_` for what a shader calls, `lobe_` for the maths under it,
// `parallax_` for POM; `SURFACE_` for flags.

const surface_coat: u32 = 1u;
const surface_sheen: u32 = 2u;
const surface_anisotropy: u32 = 4u;
const surface_transmission: u32 = 8u;
const surface_subsurface: u32 = 16u;
// The face-on reflectance comes from the ior, the specular factor and its color
// (`surface_apply_block`); without it f0 is the dielectric 0.04 of ior 1.5.
const surface_specular: u32 = 32u;

// What the frame's lights bring to a glossy surface, in two parts a material
// scales by its own: `diffuse` is for albedo (times 1 - metallic) to multiply,
// and `specular` is everything that does not take the albedo (the base lobe's
// highlight, already Fresnel-weighted, the coat, the sheen), to add as it is.
struct LightResponse {
  diffuse: vec3f,
  specular: vec3f,
};

// A surface seen from `view`: what `standard_surface` and `standard_sun` take.
// Build it with `surface_default` and set the fields of the lobes it has.
struct StandardSurface {
  // Where it is, which way it faces (unit) and the unit vector toward the eye.
  world: vec3f,
  normal: vec3f,
  view: vec3f,
  // The fragment's @builtin(position) and its view-space z (negative ahead of
  // the eye), which find its cluster of lights.
  position: vec4f,
  view_z: f32,
  // The base: albedo (or a metal's color), metallic and perceptual roughness (0
  // to 1), ambient occlusion (1 open to 0 closed), and the reflectance face on
  // (`f0`) and at grazing angles (`f90`).
  base_color: vec3f,
  metallic: f32,
  roughness: f32,
  occlusion: f32,
  f0: vec3f,
  f90: f32,
  // `SURFACE_*` flags: which lobes exist.
  lobes: u32,
  // Clear coat: its weight, roughness and normal (the mesh's own, with no
  // normal map, unless the material says).
  coat: f32,
  coat_roughness: f32,
  coat_normal: vec3f,
  // Sheen: color and roughness.
  sheen_color: vec3f,
  sheen_roughness: f32,
  // Anisotropy in -1 to 1 along `tangent`; with `bitangent`, an orthonormal
  // frame with `normal` (`surface_frame`).
  anisotropy: f32,
  tangent: vec3f,
  bitangent: vec3f,
  // Transmission: its weight, the index of refraction, the thickness in world
  // units (0 for a thin wall), the volume's color and the distance in which
  // it falls to that color (0 for never), and the light from behind
  // (`surface_transmit`, in `SURFACE_TRANSMISSION_WGSL`).
  transmission: f32,
  ior: f32,
  thickness: f32,
  attenuation_color: vec3f,
  attenuation_distance: f32,
  behind: vec3f,
  // Subsurface: how much light goes through (0 to 1), its color, and how
  // thin the surface is (0 a leaf, 1 opaque).
  subsurface: f32,
  subsurface_color: vec3f,
  thin: f32,
  // Thin-film iridescence (surface_iridescence): weight, film IOR, thickness (nm).
  iridescence: f32,
  iridescence_ior: f32,
  iridescence_thickness: f32,
  // Baked lightmap diffuse irradiance (surface_lightmap).
  lightmap_irradiance: vec3f,
  has_lightmap: f32,
};

// A surface of albedo (or metal color) `base_color`, `metallic` and
// `roughness`: the base lobe alone, seen at the fragment `position` (its
// @builtin(position)) whose view-space z is `view_z`.
fn surface_default(world: vec3f, normal: vec3f, view: vec3f, base_color: vec3f, metallic: f32, roughness: f32, position: vec4f, view_z: f32) -> StandardSurface {
  var s: StandardSurface;
  s.world = world;
  s.normal = normal;
  s.view = view;
  s.position = position;
  s.view_z = view_z;
  s.base_color = base_color;
  s.metallic = metallic;
  s.roughness = roughness;
  s.occlusion = 1.0;
  s.f0 = mix(vec3f(0.04), base_color, clamp(metallic, 0.0, 1.0));
  s.f90 = 1.0;
  s.coat_normal = normal;
  s.sheen_roughness = 0.5;
  s.ior = 1.5;
  s.attenuation_color = vec3f(1.0);
  s.thin = 1.0;
  s.iridescence = 0.0;
  s.iridescence_ior = 1.3;
  s.iridescence_thickness = 400.0;
  s.lightmap_irradiance = vec3f(0.0);
  s.has_lightmap = 0.0;
  return s;
}

// Sets the lobes in `lobes` (`SURFACE_*` flags, constants the shader chooses)
// from a material's `surface` block, the draw uniform's last five `vec4f`s:
//   [0] clearcoat, clearcoat roughness, sheen roughness, anisotropy
//   [1] sheen color, anisotropy rotation (radians; for `surface_frame`)
//   [2] transmission, ior, thickness (model units), attenuation distance
//   [3] attenuation color, specular
//   [4] specular color, emissive strength
// `model_scale` is what the thickness is scaled by to reach world units
// (`length(draw.model[0].xyz)`). With `surface_specular` or
// `surface_transmission` the reflectance (f0 and f90) follows the ior, the
// specular factor and its color; otherwise it stays the 0.04 of `surface_default`.
fn surface_apply_block(s: StandardSurface, block: array<vec4f, 5>, lobes: u32, model_scale: f32) -> StandardSurface {
  var r = s;
  r.lobes = lobes;
  if ((lobes & surface_coat) != 0u) {
    r.coat = block[0].x;
    r.coat_roughness = block[0].y;
  }
  if ((lobes & surface_sheen) != 0u) {
    r.sheen_roughness = block[0].z;
    r.sheen_color = block[1].xyz;
  }
  if ((lobes & surface_anisotropy) != 0u) {
    r.anisotropy = block[0].w;
  }
  if ((lobes & (surface_specular | surface_transmission)) != 0u) {
    r.ior = block[2].y;
    let specular = block[3].w;
    let dielectric = min(lobe_ior_f0(r.ior) * block[4].xyz, vec3f(1.0)) * specular;
    r.f0 = mix(dielectric, s.base_color, clamp(s.metallic, 0.0, 1.0));
    r.f90 = mix(specular, 1.0, clamp(s.metallic, 0.0, 1.0));
  }
  if ((lobes & surface_transmission) != 0u) {
    r.transmission = block[2].x;
    r.thickness = block[2].z * model_scale;
    r.attenuation_distance = block[2].w;
    r.attenuation_color = block[3].xyz;
  }
  return r;
}

// A mesh's tangent stream (xyz, w the handedness), or where it is zero (a mesh
// with none) `derived`, from `surface_derived_tangent`.
fn surface_tangent(stream: vec4f, derived: vec4f) -> vec4f {
  return select(derived, stream, dot(stream.xyz, stream.xyz) > 0.000001);
}

// ---- The base lobe.

// GGX's distribution for the squared alpha `a2` (alpha = roughness squared).
fn lobe_ggx_distribution(nh: f32, a2: f32) -> f32 {
  let denominator = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / (3.14159265359 * denominator * denominator);
}

// The height-correlated Smith visibility, which has the lobe's 1 / (4 nl nv).
fn lobe_smith_visibility(nl: f32, nv: f32, a2: f32) -> f32 {
  return 0.5 / max(0.000001, nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2));
}

// Schlick's Fresnel, from `f0` face on to `f90` at grazing angles.
fn lobe_schlick(f0: vec3f, f90: f32, vh: f32) -> vec3f {
  let fade = 1.0 - vh;
  return f0 + (vec3f(f90) - f0) * (fade * fade * fade * fade * fade);
}

// The dielectric's face-on reflectance from its index of refraction (at least
// 1), as two products and a division, so that an ior of 1.5 is the f32 nearest 0.04.
fn lobe_ior_f0(ior: f32) -> f32 {
  let i = max(ior, 1.0);
  return (i - 1.0) * (i - 1.0) / ((i + 1.0) * (i + 1.0));
}

// ---- Clear coat: a second, dielectric GGX layer over everything, with the
// base lobe's own Smith visibility at the coat's alpha and normal.

// Its Fresnel at `vh` (cosine between the view and the half vector), f0 0.04.
fn lobe_coat_fresnel(vh: f32) -> f32 {
  let fade = 1.0 - vh;
  return 0.04 + 0.96 * (fade * fade * fade * fade * fade);
}

// ---- Sheen: Charlie's distribution and Neubelt's visibility, for cloth.

// The sheen roughness a shader gives, held where the lobe stays well behaved.
fn lobe_sheen_roughness(roughness: f32) -> f32 {
  return clamp(roughness, 0.07, 1.0);
}

fn lobe_sheen_distribution(nh: f32, roughness: f32) -> f32 {
  let alpha = roughness * roughness;
  let sin2 = max(1.0 - nh * nh, 0.0);
  return (2.0 + 1.0 / alpha) * pow(sin2, 0.5 / alpha) / 6.28318530718;
}

fn lobe_sheen_visibility(nl: f32, nv: f32) -> f32 {
  return 1.0 / (4.0 * (nl + nv - nl * nv));
}

// The sheen lobe's directional albedo for a sheen color of 1, capped at 1:
// what the lobe takes from the light under it (`nv` the cosine of the view,
// `roughness` the sheen's, through `lobe_sheen_roughness`). It is the integral
// of the lobe's D * V * cos over the light's hemisphere, past 1 for a grazing
// view of a sharp sheen. A table of 32 columns by 16 rows, entry `32 row +
// column`, bilinear: the columns evenly spaced in the square root of nv and the
// rows in the square root of (roughness - 0.07) / 0.93. It is Ink's table
// (tools/native/src/surface.rs writes it; its sum is 220.61942091), and
// pen-render's tests integrate it again and compare it to this text.
var<private> lobe_sheen_table = array<f32, 512>(
  // SHEEN TABLE BEGIN
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  1.00000000, 0.96609209, 0.66635188, 0.43914545, 0.27494452, 0.16243548, 0.08983930, 0.04608312,
  0.02168285, 0.00923651, 0.00350696, 0.00116469, 0.00033065, 0.00007799, 0.00001475, 0.00000213,
  0.00000022, 0.00000002, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  1.00000000, 0.99849743, 0.70382454, 0.47581233, 0.30694153, 0.18781249, 0.10823914, 0.05826881,
  0.02901557, 0.01321075, 0.00542376, 0.00197461, 0.00062460, 0.00016736, 0.00003679, 0.00000637,
  0.00000082, 0.00000007, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  1.00000000, 1.00000000, 0.79120667, 0.56797785, 0.39292684, 0.26082237, 0.16527702, 0.09938102,
  0.05630328, 0.02980166, 0.01458949, 0.00652583, 0.00262752, 0.00093481, 0.00028706, 0.00007381,
  0.00001526, 0.00000240, 0.00000026, 0.00000002, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  1.00000000, 1.00000000, 0.87482038, 0.67105278, 0.50103574, 0.36310585, 0.25457281, 0.17199340,
  0.11146346, 0.06891247, 0.04037954, 0.02224737, 0.01141374, 0.00538712, 0.00230358, 0.00087483,
  0.00028734, 0.00007868, 0.00001703, 0.00000268, 0.00000027, 0.00000001, 0.00000000, 0.00000000,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  1.00000000, 1.00000000, 0.91595046, 0.74369491, 0.59229018, 0.46187804, 0.35194184, 0.26139767,
  0.18869614, 0.13193549, 0.08898340, 0.05760236, 0.03557141, 0.02079630, 0.01139987, 0.00578623,
  0.00267422, 0.00109958, 0.00038892, 0.00011233, 0.00002426, 0.00000330, 0.00000018, 0.00000000,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  1.00000000, 1.00000000, 0.90920665, 0.77150637, 0.64560985, 0.53219713, 0.43159173, 0.34378996,
  0.26849076, 0.20512874, 0.15291174, 0.11086346, 0.07787063, 0.05273348, 0.03421768, 0.02110542,
  0.01224284, 0.00658119, 0.00320904, 0.00137332, 0.00048788, 0.00012911, 0.00001932, 0.00000026,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  1.00000000, 0.98102820, 0.86882158, 0.76205773, 0.66160758, 0.56814115, 0.48214657, 0.40394357,
  0.33369393, 0.27141090, 0.21696873, 0.17011324, 0.13047412, 0.09757910, 0.07087035, 0.04972273,
  0.03346384, 0.02139528, 0.01281466, 0.00703751, 0.00341839, 0.00137031, 0.00038154, 0.00002884,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
  0.98277184, 0.89634067, 0.81178558, 0.72985974, 0.65118486, 0.57627302, 0.50554118, 0.43932106,
  0.37786569, 0.32135404, 0.26989436, 0.22352695, 0.18222671, 0.14590578, 0.11441668, 0.08755592,
  0.06506831, 0.04665211, 0.03196491, 0.02063031, 0.01224544, 0.00638909, 0.00263055, 0.00053878,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 0.94502083,
  0.87992735, 0.81491643, 0.75060375, 0.68749593, 0.62601584, 0.56651993, 0.50931011, 0.45464221,
  0.40273167, 0.35375757, 0.30786537, 0.26516875, 0.22575081, 0.18966490, 0.15693516, 0.12755696,
  0.10149724, 0.07869493, 0.05906142, 0.04248124, 0.02881280, 0.01788945, 0.00952070, 0.00349374,
  1.00000000, 1.00000000, 1.00000000, 1.00000000, 0.98340343, 0.93834126, 0.89090605, 0.84200093,
  0.79230903, 0.74236191, 0.69258323, 0.64331760, 0.59485042, 0.54742171, 0.50123597, 0.45646924,
  0.41327429, 0.37178437, 0.33211591, 0.29437057, 0.25863659, 0.22498985, 0.19349450, 0.16420342,
  0.13715847, 0.11239058, 0.08991975, 0.06975496, 0.05189404, 0.03632342, 0.02301797, 0.01194067,
  0.96083395, 0.95130653, 0.92968785, 0.90165811, 0.86940935, 0.83430004, 0.79726916, 0.75899487,
  0.71998297, 0.68062055, 0.64121042, 0.60199396, 0.56316703, 0.52489117, 0.48730163, 0.45051341,
  0.41462561, 0.37972478, 0.34588747, 0.31318216, 0.28167071, 0.25140954, 0.22245055, 0.19484174,
  0.16862782, 0.14385056, 0.12054914, 0.09876037, 0.07851887, 0.05985716, 0.04280577, 0.02739325,
  0.85583864, 0.84815146, 0.83071649, 0.80812013, 0.78212586, 0.75382052, 0.72394998, 0.69304750,
  0.66150557, 0.62961961, 0.59761598, 0.56567085, 0.53392314, 0.50248383, 0.47144269, 0.44087321,
  0.41083638, 0.38138354, 0.35255851, 0.32439934, 0.29693967, 0.27020975, 0.24423731, 0.21904830,
  0.19466745, 0.17111876, 0.14842596, 0.12661285, 0.10570369, 0.08572354, 0.06669855, 0.04865633,
  0.77679509, 0.77036767, 0.75580614, 0.73695585, 0.71529142, 0.69171744, 0.66685106, 0.64112984,
  0.61487218, 0.58831418, 0.56163327, 0.53496417, 0.50840985, 0.48204943, 0.45594386, 0.43014016,
  0.40467461, 0.37957512, 0.35486313, 0.33055503, 0.30666328, 0.28319735, 0.26016440, 0.23756984,
  0.21541786, 0.19371178, 0.17245440, 0.15164825, 0.13129592, 0.11140024, 0.09196451, 0.07299265,
  0.71697164, 0.71142504, 0.69887748, 0.68266018, 0.66404822, 0.64382094, 0.62250753, 0.60048076,
  0.57800943, 0.55529039, 0.53246921, 0.50965395, 0.48692483, 0.46434101, 0.44194562, 0.41976940,
  0.39783347, 0.37615137, 0.35473071, 0.33357435, 0.31268135, 0.29204771, 0.27166689, 0.25153028,
  0.23162749, 0.21194660, 0.19247427, 0.17319591, 0.15409562, 0.13515622, 0.11635915, 0.09768477,
  0.67144043, 0.66652196, 0.65541313, 0.64108069, 0.62465867, 0.60683840, 0.58808760, 0.56873412,
  0.54901274, 0.52909382, 0.50910171, 0.48912709, 0.46923564, 0.44947405, 0.42987455, 0.41045811,
  0.39123696, 0.37221637, 0.35339603, 0.33477118, 0.31633332, 0.29807086, 0.27996951, 0.26201260,
  0.24418128, 0.22645456, 0.20880934, 0.19122031, 0.17365971, 0.15609709, 0.13849886, 0.12083050,
  0.63657542, 0.63211312, 0.62205054, 0.60909083, 0.59426633, 0.57820518, 0.56133123, 0.54394050,
  0.52624394, 0.50839359, 0.49049941, 0.47264056, 0.45487327, 0.43723644, 0.41975565, 0.40244617,
  0.38531519, 0.36836348, 0.35158661, 0.33497592, 0.31851919, 0.30220115, 0.28600380, 0.26990667,
  0.25388683, 0.23791896, 0.22197516, 0.20602478, 0.19003399, 0.17396535, 0.15777709, 0.14142957,
  // SHEEN TABLE END
);

fn lobe_sheen_albedo(nv: f32, roughness: f32) -> f32 {
  let x = sqrt(clamp(nv, 0.0, 1.0)) * 31.0;
  let y = sqrt(clamp((lobe_sheen_roughness(roughness) - 0.07) / 0.93, 0.0, 1.0)) * 15.0;
  let x0 = min(u32(floor(x)), 30u);
  let y0 = min(u32(floor(y)), 14u);
  let fx = x - f32(x0);
  let fy = y - f32(y0);
  let a = mix(lobe_sheen_table[y0 * 32u + x0], lobe_sheen_table[y0 * 32u + x0 + 1u], fx);
  let b = mix(lobe_sheen_table[(y0 + 1u) * 32u + x0], lobe_sheen_table[(y0 + 1u) * 32u + x0 + 1u], fx);
  return mix(a, b, fy);
}

// ---- Anisotropic GGX: the lobe stretched along the tangent (a > 0 widens it
// across the grain, a < 0 along it, as glTF has it).

// The widths along the tangent and the bitangent for perceptual `roughness`
// and anisotropy `a`: x along the tangent, y along the bitangent. They are
// both alpha (roughness squared) when `a` is 0.
fn lobe_aniso_alpha(roughness: f32, anisotropy: f32) -> vec2f {
  let rough = clamp(roughness, 0.04, 1.0);
  let alpha = rough * rough;
  return vec2f(mix(alpha, 1.0, anisotropy * anisotropy), alpha);
}

// th and bh: the half vector's parts along the tangent and the bitangent.
fn lobe_aniso_distribution(nh: f32, th: f32, bh: f32, alpha: vec2f) -> f32 {
  let ratio = vec2f(th, bh) / alpha;
  let d = dot(ratio, ratio) + nh * nh;
  return 1.0 / (3.14159265359 * alpha.x * alpha.y * d * d);
}

// tl, bl, tv, bv: the light's and the view's parts along the tangent and bitangent.
fn lobe_aniso_visibility(nl: f32, nv: f32, tl: f32, bl: f32, tv: f32, bv: f32, alpha: vec2f) -> f32 {
  let shadowing = nv * length(vec3f(alpha.x * tl, alpha.y * bl, nl)) + nl * length(vec3f(alpha.x * tv, alpha.y * bv, nv));
  return 0.5 / max(0.000001, shadowing);
}

// The normal a reflection of an anisotropic surface looks up the sky by:
// bent toward the view across the grain, by as much as the anisotropy and the
// roughness (the reflection stretches along the grain, Filament's trick).
fn lobe_bent_normal(normal: vec3f, tangent: vec3f, bitangent: vec3f, view: vec3f, roughness: f32, anisotropy: f32) -> vec3f {
  let grain = select(tangent, bitangent, anisotropy >= 0.0);
  let across = cross(grain, view);
  let bent = cross(across, grain);
  let mixed = mix(normal, bent, abs(anisotropy) * clamp(5.0 * roughness, 0.0, 1.0));
  if (dot(mixed, mixed) < 0.000001) {
    return normal;
  }
  return normalize(mixed);
}

// Sets the anisotropy's frame: `tangent` (any vector not along the normal; a
// mesh's tangent stream, or `surface_derived_tangent`) made orthogonal to the
// normal and turned about it by `rotation` (radians), and the bitangent across
// it, `handedness` (+1 or -1) giving its side.
fn surface_frame(s: StandardSurface, tangent: vec3f, handedness: f32, rotation: f32) -> StandardSurface {
  var r = s;
  let t0 = normalize(tangent - s.normal * dot(s.normal, tangent));
  r.tangent = t0 * cos(rotation) + cross(s.normal, t0) * sin(rotation);
  r.bitangent = cross(s.normal, r.tangent) * handedness;
  return r;
}

// ---- Cheap subsurface: wrapped lighting and a back-lit term.

fn lobe_wrap(x: f32, w: f32) -> f32 {
  return clamp((x + w) / ((1.0 + w) * (1.0 + w)), 0.0, 1.0);
}

// The diffuse lobe's response for `amount` of subsurface of `color`, for the
// unclamped cosine `signed` to the light: 1 / pi of Ink's, so that
// Lambert is `max(signed, 0)`.
fn lobe_subsurface_diffuse(signed: f32, amount: f32, color: vec3f) -> vec3f {
  let nl = max(signed, 0.0);
  return vec3f((1.0 - amount) * nl) + amount * lobe_wrap(signed, 0.5) * clamp(color + vec3f(nl), vec3f(0.0), vec3f(1.0));
}

// What leaks through to a viewer on the other side: the light `way` (unit,
// toward the light) bent by the normal, seen along `view` (toward the eye).
// Ink's term is times base / pi * 2; this has the base and pi left out.
fn lobe_subsurface_back(way: vec3f, normal: vec3f, view: vec3f, amount: f32, color: vec3f, thin: f32) -> vec3f {
  let bent = -normalize(way + 0.2 * normal);
  let through = clamp(dot(view, bent), 0.0, 1.0);
  let squared = through * through;
  return color * (amount * (1.0 - thin) * squared * squared * 2.0);
}

// ---- Transmission and volume (KHR_materials_transmission, _volume, _ior).

// The direction a ray arriving along -`view` takes entering a medium, as
// `refract` has it: `eta` is the index of the medium it comes from over the one
// it enters (1 / ior from the air into glass). The zero vector at total
// internal reflection, which only happens leaving a denser medium (eta over 1).
fn lobe_refract(view: vec3f, normal: vec3f, eta: f32) -> vec3f {
  return refract(-view, normal, eta);
}

// How much of its color a volume lets through over `thickness` (world units)
// when it lets `color` through over `distance`: Beer-Lambert, per channel. A
// distance of 0 is a medium that absorbs nothing, and so is no thickness.
fn lobe_volume_attenuation(color: vec3f, thickness: f32, distance: f32) -> vec3f {
  if (thickness <= 0.0 || distance <= 0.0) {
    return vec3f(1.0);
  }
  return pow(color, vec3f(thickness / distance));
}

// The mip level of the picture behind to read through a surface of
// `roughness` and index `ior` (as the glTF sample viewer picks it): `size_log2` is the base-2
// logarithm of the picture's width, `last` its last level.
fn lobe_transmission_level(roughness: f32, ior: f32, size_log2: f32, last: f32) -> f32 {
  return min(size_log2 * clamp(roughness * clamp(2.0 * ior - 2.0, 0.0, 1.0), 0.0, 1.0), last);
}

// ---- Parallax occlusion mapping: a heightfield the surface's uv are marched
// through. The heights are the red of the material's layer texture (binding 4
// of group 1; pass it with its sampler): 1 is the surface, 0 `scale` (in uv
// units) below it.

struct Parallax {
  // Where every texture read of the material goes instead of the mesh's uv.
  uv: vec2f,
  // The height of the surface there (1 at the top).
  height: f32,
  // The light's visibility: 1 lit, 0 in a groove (1 when not asked for).
  shadow: f32,
};

// How many layers the march takes along a direction whose tangent-space z is
// `z` (|z| of 1 straight down, 0 across): floor(32 - 24 |z|), 8 to 32.
fn parallax_layers(z: f32) -> f32 {
  return floor(32.0 - 24.0 * min(abs(z), 1.0));
}

// The view `v` (world, toward the eye) in the tangent space of `tangent`,
// `bitangent` and `normal` (unit and orthogonal; the uv's axes): the transpose
// of the frame times `v`, normalized.
fn parallax_tangent_space(tangent: vec3f, bitangent: vec3f, normal: vec3f, v: vec3f) -> vec3f {
  return normalize(vec3f(dot(tangent, v), dot(bitangent, v), dot(normal, v)));
}

fn parallax_height(layers: texture_2d<f32>, layer_sampler: sampler, uv: vec2f, lod: f32) -> f32 {
  return clamp(textureSampleLevel(layers, layer_sampler, uv, lod).r, 0.0, 1.0);
}

// How much of the light reaches the surface at `uv` where its height is
// `height`, along `light` (unit, tangent space, toward the light): the ray
// rises from the surface to the top in as many layers as the march takes,
// sliding toward the light by `light.xy / z * scale` of the height it climbs,
// and the most any sample stands above it, times 8, darkens the light. 0 for a
// light below the surface.
fn parallax_shadow(layers: texture_2d<f32>, layer_sampler: sampler, uv: vec2f, height: f32, light: vec3f, scale: f32, lod: f32) -> f32 {
  if (light.z <= 0.0) {
    return 0.0;
  }
  let count = parallax_layers(light.z);
  let rise = 1.0 - height;
  let slide = light.xy / max(light.z, 0.0001) * (scale * rise);
  var above = 0.0;
  for (var k = 1.0; k <= count; k += 1.0) {
    let along = k / count;
    above = max(above, parallax_height(layers, layer_sampler, uv + slide * along, lod) - (height + rise * along));
  }
  return 1.0 - clamp(8.0 * above, 0.0, 1.0);
}

// Marches the view `view` (unit, tangent space, toward the eye, z up) down
// through the heightfield from `uv` in layers of equal depth, to the first
// layer where the surface is at or above the ray, and puts the hit where the
// line between that sample and the one before it crosses. `scale` is the height
// range in uv units; `lod` the layer texture's level (0 for the base). A view
// at or below the surface, or a hit at the first layer, leaves `uv` alone. With
// `with_shadow`, `light` (tangent space, toward the light) gives `shadow` too.
fn parallax_uv(layers: texture_2d<f32>, layer_sampler: sampler, uv: vec2f, view: vec3f, light: vec3f, scale: f32, lod: f32, with_shadow: bool) -> Parallax {
  var result = Parallax(uv, parallax_height(layers, layer_sampler, uv, lod), 1.0);
  if (view.z > 0.0) {
    let count = parallax_layers(view.z);
    let step = (-view.xy / max(view.z, 0.0001)) * scale / count;
    var before = uv;
    var before_gap = 0.0;
    var at = uv;
    for (var k = 0.0; k <= count; k += 1.0) {
      at = uv + step * k;
      let gap = parallax_height(layers, layer_sampler, at, lod) - (1.0 - k / count);
      if (gap >= 0.0 || k >= count) {
        if (k > 0.0) {
          result.uv = at + (before - at) * (gap / (gap - before_gap));
        }
        break;
      }
      before = at;
      before_gap = gap;
    }
    result.height = parallax_height(layers, layer_sampler, result.uv, lod);
  }
  if (with_shadow) {
    result.shadow = parallax_shadow(layers, layer_sampler, result.uv, result.height, light, scale, lod);
  }
  return result;
}

// ---- One light on a surface.

// The response of `s` to one light shining from the unit direction `way` (toward
// the light) with `radiance` (its color times its strength and its visibility,
// scaled as the frame's lights are). `diffuse` and `specular` are what
// `LightResponse` says. With only the base lobe, `diffuse` is `radiance` times
// the cosine and `specular` the GGX highlight, as the frame's lights have
// always been summed. With lobes, the diffuse and specular lobes (and the
// subsurface's back light, in `specular` for it takes the albedo but not
// 1 - metallic) are scaled by what the sheen and the coat leave of them, and
// the sheen and the coat add theirs: `(diffuse * (1 - transmission) + specular) *
// sheen_keep * coat_keep + sheen + coat`. The sheen is not dimmed by the coat.
fn surface_light(s: StandardSurface, way: vec3f, radiance: vec3f) -> LightResponse {
  var response = LightResponse(vec3f(0.0), vec3f(0.0));
  let signed = dot(s.normal, way);
  let nl = max(signed, 0.0);
  let wrapping = (s.lobes & surface_subsurface) != 0u && s.subsurface > 0.0;
  if (nl <= 0.0 && !wrapping) {
    return response;
  }
  var diffuse = vec3f(nl);
  var back = vec3f(0.0);
  if (wrapping) {
    let amount = clamp(s.subsurface, 0.0, 1.0);
    diffuse = lobe_subsurface_diffuse(signed, amount, s.subsurface_color);
    back = s.base_color * lobe_subsurface_back(way, s.normal, s.view, amount, s.subsurface_color, s.thin);
  }
  if ((s.lobes & surface_transmission) != 0u && s.transmission > 0.0) {
    let kept = 1.0 - clamp(s.transmission, 0.0, 1.0);
    diffuse *= kept;
    back *= kept;
  }
  let facing = dot(s.normal, s.view);
  if (nl <= 0.0 || facing <= 0.0) {
    response.diffuse = radiance * diffuse;
    response.specular = radiance * back;
    return response;
  }
  let nv = max(facing, 0.00001);
  let half_way = normalize(s.view + way);
  let nh = clamp(dot(s.normal, half_way), 0.0, 1.0);
  let vh = clamp(dot(s.view, half_way), 0.0, 1.0);
  var distribution = 0.0;
  var visibility = 0.0;
  let anisotropy = clamp(s.anisotropy, -1.0, 1.0);
  if ((s.lobes & surface_anisotropy) != 0u && anisotropy != 0.0) {
    let alpha = lobe_aniso_alpha(s.roughness, anisotropy);
    distribution = lobe_aniso_distribution(nh, dot(s.tangent, half_way), dot(s.bitangent, half_way), alpha);
    visibility = lobe_aniso_visibility(nl, nv, dot(s.tangent, way), dot(s.bitangent, way), dot(s.tangent, s.view), dot(s.bitangent, s.view), alpha);
  } else {
    let rough = clamp(s.roughness, 0.04, 1.0);
    let a2 = rough * rough * rough * rough;
    distribution = lobe_ggx_distribution(nh, a2);
    visibility = lobe_smith_visibility(nl, nv, a2);
  }
  var fresnel = lobe_schlick(s.f0, s.f90, vh);
  if ((s.lobes & surface_iridescence) != 0u && s.iridescence > 0.0) {
    let irid_f = lobe_iridescence_fresnel(1.0, max(s.iridescence_ior, 1.0001), vh, max(s.iridescence_thickness, 0.0), s.f0);
    fresnel = mix(fresnel, irid_f, clamp(s.iridescence, 0.0, 1.0));
  }
  var specular = radiance * (fresnel * (distribution * visibility * nl * 3.14159265359));
  // The layers over the base lobes take their share of the light, and add theirs.
  var layers = 1.0;
  var added = vec3f(0.0);
  if ((s.lobes & surface_sheen) != 0u && dot(s.sheen_color, s.sheen_color) > 0.0) {
    let tint = max(s.sheen_color, vec3f(0.0));
    let rough = lobe_sheen_roughness(s.sheen_roughness);
    let lobe = lobe_sheen_distribution(nh, rough) * lobe_sheen_visibility(nl, nv) * nl;
    added = radiance * tint * (lobe * 3.14159265359);
    layers = max(0.0, 1.0 - max(tint.x, max(tint.y, tint.z)) * lobe_sheen_albedo(nv, rough));
  }
  var coat = vec3f(0.0);
  if ((s.lobes & surface_coat) != 0u && s.coat > 0.0) {
    let weight = clamp(s.coat, 0.0, 1.0);
    let rc = clamp(s.coat_roughness, 0.04, 1.0);
    let a2 = rc * rc * rc * rc;
    let nhc = clamp(dot(s.coat_normal, half_way), 0.0, 1.0);
    let nlc = clamp(dot(s.coat_normal, way), 0.0, 1.0);
    let nvc = max(dot(s.coat_normal, s.view), 0.00001);
    // The coat's own lobe has its Fresnel at the half vector; what it leaves of
    // the lobes under it is the same for every light, its Fresnel at the view.
    let fc = weight * lobe_coat_fresnel(vh);
    layers *= 1.0 - weight * lobe_coat_fresnel(min(nvc, 1.0));
    coat = radiance * (fc * lobe_ggx_distribution(nhc, a2) * lobe_smith_visibility(nlc, nvc, a2) * nlc * 3.14159265359);
  }
  response.diffuse = radiance * diffuse * layers;
  response.specular = (specular + radiance * back) * layers + added + coat;
  return response;
}

// The light of the sun (`sun_color` times its visibility, scaled as the
// frame's lights are; `sun_dir` the way its light travels, as the scene's
// `light_direction`) on `s`: the same lobes the frame's lights get, diffuse
// times the albedo and (1 - metallic), and the specular ones as they are.
// Lambert alone is `base_color * sun_color * visibility * max(dot(normal, -sun_dir), 0)`.
fn standard_sun(s: StandardSurface, sun_dir: vec3f, sun_color: vec3f, visibility: f32) -> vec3f {
  let response = surface_light(s, -normalize(sun_dir), sun_color * visibility);
  return response.diffuse * s.base_color * (1.0 - clamp(s.metallic, 0.0, 1.0)) + response.specular;
}

// Screen-door 4x4 Bayer dither predicate for smooth LOD crossfading without
// transparency sorting (returns true when the fragment should be discarded).
// Pass `fade` in (0, 1] for the outgoing LOD and `-fade` in [-1, 0) for the
// incoming LOD so both halves cover complementary pixel masks with zero holes
// and zero double-drawn pixels. `fade >= 1.0` never discards; `fade == 0.0` or
// `fade <= -1.0` always discards.
fn surface_lod_dither(pixel: vec2f, fade: f32) -> bool {
  if (fade >= 1.0) {
    return false;
  }
  if (fade == 0.0 || fade <= -1.0) {
    return true;
  }
  let p = vec2u(floor(max(pixel, vec2f(0.0)))) & vec2u(3u, 3u);
  let idx = p.y * 4u + p.x;
  let bayer = array<f32, 16>(
     0.0,  8.0,  2.0, 10.0,
    12.0,  4.0, 14.0,  6.0,
     3.0, 11.0,  1.0,  9.0,
    15.0,  7.0, 13.0,  5.0,
  );
  let threshold = (bayer[idx] + 0.5) / 16.0;
  if (fade > 0.0) {
    return threshold >= fade;
  }
  return threshold < (-fade);
}

// Applies glTF `KHR_texture_transform` (scale -> rotation -> translation) to `uv`.
// `offset_scale` holds `(offset.x, offset.y, scale.x, scale.y)` and `rot_cs` holds `(cos(r), sin(r))`.
fn surface_transform_uv(uv: vec2f, offset_scale: vec4f, rot_cs: vec2f) -> vec2f {
  let s = uv * offset_scale.zw;
  let r = vec2f(s.x * rot_cs.x + s.y * rot_cs.y, -s.x * rot_cs.y + s.y * rot_cs.x);
  return r + offset_scale.xy;
}

// Applies a linear ORM texture sample (`R = occlusion`, `G = roughness`, `B = metallic`)
// onto `s` when `enabled > 0.5`.
fn surface_apply_orm(s: StandardSurface, orm: vec4f, enabled: f32) -> StandardSurface {
  var out = s;
  if (enabled > 0.5) {
    out.occlusion *= clamp(orm.r, 0.0, 1.0);
    out.roughness = clamp(out.roughness * orm.g, 0.04, 1.0);
    out.metallic = clamp(out.metallic * orm.b, 0.0, 1.0);
  }
  return out;
}

// ---- Thin-film iridescence (KHR_materials_iridescence), baked lightmaps & chromatic dispersion.
const surface_iridescence: u32 = 64u;
const surface_lightmap: u32 = 128u;

fn lobe_iridescence_xyz_to_srgb(xyz: vec3f) -> vec3f {
  let m = mat3x3f(
    vec3f( 3.2404542, -0.9692660,  0.0556434),
    vec3f(-1.5371385,  1.8760108, -0.2040259),
    vec3f(-0.4985314,  0.0415560,  1.0572252)
  );
  return max(m * xyz, vec3f(0.0));
}

// Belcour & Barla (2017) analytical Airy thin-film interference reflectance (`KHR_materials_iridescence`).
// Preserves signed interface reflection phases so zero thickness matches the exact bare outside/base Fresnel.
fn lobe_iridescence_fresnel(outside_ior: f32, film_ior: f32, cos_theta1: f32, thickness_nm: f32, base_f0: vec3f) -> vec3f {
  let c1 = clamp(cos_theta1, 0.0, 1.0);
  let eta = outside_ior / max(film_ior, 1.0001);
  let sin2_t2 = eta * eta * max(1.0 - c1 * c1, 0.0);
  if (sin2_t2 >= 1.0) {
    return vec3f(1.0);
  }
  let cos_t2 = sqrt(1.0 - sin2_t2);
  let r0_12 = (outside_ior - film_ior) / (outside_ior + film_ior);
  let r0_12_sq = r0_12 * r0_12;
  let r12 = r0_12_sq + (1.0 - r0_12_sq) * pow(1.0 - c1, 5.0);
  let phi12 = select(0.0, 3.14159265359, film_ior > outside_ior);

  let s = sqrt(clamp(base_f0, vec3f(0.0), vec3f(0.999)));
  let base_ior = (vec3f(1.0) + s) / max(vec3f(1.0) - s, vec3f(0.001));
  let r0_23 = (vec3f(film_ior) - base_ior) / (vec3f(film_ior) + base_ior);
  let r0_23_sq = r0_23 * r0_23;
  let r23 = r0_23_sq + (vec3f(1.0) - r0_23_sq) * pow(1.0 - cos_t2, 5.0);
  let phi23 = select(vec3f(0.0), vec3f(3.14159265359), base_ior > vec3f(film_ior));

  let opd = 2.0 * film_ior * max(thickness_nm, 0.0) * cos_t2;
  let phi = 2.0 * 3.14159265359 * opd / vec3f(650.0, 510.0, 475.0) + vec3f(phi12) + phi23;
  let r123 = clamp(sqrt(vec3f(r12) * r23), vec3f(0.0), vec3f(0.999));
  let term = 2.0 * r123 * cos(phi);
  let num = vec3f(r12) + r23 + term;
  let den = vec3f(1.0) + vec3f(r12) * r23 + term;
  return clamp(num / max(den, vec3f(0.0001)), vec3f(0.0), vec3f(1.0));
}

// Per-channel IOR spread for `KHR_materials_dispersion` (`dispersion = 20 / abbe_number`).
fn lobe_dispersion_iors(ior: f32, dispersion: f32) -> vec3f {
  let half_spread = (ior - 1.0) * 0.052356 * max(dispersion, 0.0);
  return vec3f(max(ior - half_spread, 1.0), ior, ior + half_spread);
}

// Samples a tile from the 4x3 SurfaceMapAtlas bound at @group(1) @binding(4) when `settings.w == 5.0`.
fn surface_atlas_tile_uv(tile: u32, uv: vec2f, atlas_dims: vec2f, clamp_uv: bool) -> vec2f {
  let col = f32(tile % 4u);
  let row = f32(tile / 4u);
  let cell = atlas_dims / vec2f(4.0, 3.0);
  let wrapped = select(fract(uv), clamp(uv, vec2f(0.0), vec2f(1.0)), clamp_uv);
  let inner = cell - vec2f(2.0);
  return (vec2f(col, row) * cell + vec2f(1.0) + wrapped * inner) / atlas_dims;
}

fn surface_read_atlas(atlas: texture_2d<f32>, samp: sampler, tile: u32, uv: vec2f, clamp_uv: bool) -> vec4f {
  let dims = vec2f(textureDimensions(atlas, 0));
  return textureSampleLevel(atlas, samp, surface_atlas_tile_uv(tile, uv, dims, clamp_uv), 0.0);
}

// Applies authored iridescence factors and enabled tiles from the GPU-packed 4x3 SurfaceMapAtlas
// (@group(1) @binding(4)) onto `s`. Preserves authored surface block factors and custom dielectric F0,
// and decodes ORM, Clearcoat, ClearcoatNormal, Sheen, Anisotropy, Transmission, Specular,
// Iridescence, Lightmap, and LightmapDir tiles.
fn surface_apply_atlas(
  s: StandardSurface,
  atlas: texture_2d<f32>,
  samp: sampler,
  uv: vec2f,
  uv1: vec2f,
  atlas_mode: f32,
  tile_mask: u32,
  block: array<vec4f, 5>,
  irid_params: vec4f,
  model_scale: f32,
) -> StandardSurface {
  var out = s;
  let authored_ior = max(block[2].y, 1.0);
  let authored_spec_w = max(block[3].w, 0.0);
  let authored_spec_c = max(block[4].xyz, vec3f(0.0));

  if (irid_params.x > 0.0) {
    out.lobes |= surface_iridescence;
    out.iridescence = clamp(irid_params.x, 0.0, 1.0);
    out.iridescence_ior = max(irid_params.y, 1.0001);
    out.iridescence_thickness = max(irid_params.w, 0.0);
  }

  if (atlas_mode <= 4.5 || tile_mask == 0u) {
    return out;
  }

  if ((tile_mask & 1u) != 0u) {
    let orm = surface_read_atlas(atlas, samp, 0u, uv, false);
    out = surface_apply_orm(out, orm, 1.0);
    let m = clamp(out.metallic, 0.0, 1.0);
    let dielectric = min(lobe_ior_f0(authored_ior) * authored_spec_c, vec3f(1.0)) * authored_spec_w;
    out.ior = authored_ior;
    out.f0 = mix(dielectric, out.base_color, m);
    out.f90 = mix(authored_spec_w, 1.0, m);
  }
  if ((tile_mask & 8u) != 0u) {
    let cc = surface_read_atlas(atlas, samp, 3u, uv, false);
    if ((out.lobes & surface_coat) == 0u) {
      out.coat = block[0].x;
      out.coat_roughness = block[0].y;
    }
    out.lobes |= surface_coat;
    out.coat = clamp(out.coat * cc.r, 0.0, 1.0);
    out.coat_roughness = clamp(out.coat_roughness * cc.g, 0.04, 1.0);
  }
  if ((tile_mask & 16u) != 0u) {
    let ccn = surface_read_atlas(atlas, samp, 4u, uv, false);
    let txy = ccn.xy * 2.0 - vec2f(1.0);
    let tz = sqrt(max(1.0 - dot(txy, txy), 0.0001));
    let n_base = normalize(out.coat_normal);
    var t_vec = out.tangent;
    var b_vec = out.bitangent;
    if (dot(t_vec, t_vec) < 0.25 || dot(b_vec, b_vec) < 0.25) {
      let up_ref = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n_base.y) > 0.9);
      t_vec = normalize(cross(up_ref, n_base));
      b_vec = cross(n_base, t_vec);
    }
    out.coat_normal = normalize(t_vec * txy.x + b_vec * txy.y + n_base * tz);
  }
  if ((tile_mask & 32u) != 0u) {
    let sh = surface_read_atlas(atlas, samp, 5u, uv, false);
    if ((out.lobes & surface_sheen) == 0u) {
      out.sheen_color = block[1].xyz;
      out.sheen_roughness = block[0].z;
    }
    out.lobes |= surface_sheen;
    out.sheen_color *= sh.rgb;
    out.sheen_roughness = clamp(out.sheen_roughness * sh.a, 0.07, 1.0);
  }
  if ((tile_mask & 64u) != 0u) {
    let an = surface_read_atlas(atlas, samp, 6u, uv, false);
    if ((out.lobes & surface_anisotropy) == 0u) {
      out.anisotropy = block[0].w;
    }
    out.lobes |= surface_anisotropy;
    out.anisotropy = clamp(out.anisotropy * an.b, -1.0, 1.0);
    let dir = an.rg * 2.0 - vec2f(1.0);
    if (dot(dir, dir) > 1e-4 && dot(out.tangent, out.tangent) > 0.25 && dot(out.bitangent, out.bitangent) > 0.25) {
      let d = normalize(dir);
      let new_t = normalize(out.tangent * d.x + out.bitangent * d.y);
      let hand = select(-1.0, 1.0, dot(cross(out.normal, out.tangent), out.bitangent) >= 0.0);
      out.tangent = new_t;
      out.bitangent = cross(out.normal, new_t) * hand;
    }
  }
  if ((tile_mask & 128u) != 0u) {
    let tr = surface_read_atlas(atlas, samp, 7u, uv, false);
    if ((out.lobes & surface_transmission) == 0u) {
      out.transmission = block[2].x;
      out.ior = authored_ior;
      out.thickness = block[2].z * model_scale;
      out.attenuation_distance = block[2].w;
      out.attenuation_color = block[3].xyz;
    }
    out.lobes |= surface_transmission;
    out.transmission = clamp(out.transmission * tr.r, 0.0, 1.0);
    out.thickness *= tr.g;
  }
  if ((tile_mask & 256u) != 0u) {
    let sp = surface_read_atlas(atlas, samp, 8u, uv, false);
    let spec_w = authored_spec_w * sp.a;
    let spec_c = authored_spec_c * sp.rgb;
    let m = clamp(out.metallic, 0.0, 1.0);
    let dielectric = min(lobe_ior_f0(authored_ior) * spec_c, vec3f(1.0)) * spec_w;
    out.lobes |= surface_specular;
    out.ior = authored_ior;
    out.f0 = mix(dielectric, out.base_color, m);
    out.f90 = mix(spec_w, 1.0, m);
  }
  if ((tile_mask & 512u) != 0u) {
    let ir = surface_read_atlas(atlas, samp, 9u, uv, false);
    out.lobes |= surface_iridescence;
    out.iridescence = clamp(irid_params.x * ir.r, 0.0, 1.0);
    out.iridescence_ior = max(irid_params.y, 1.0001);
    out.iridescence_thickness = mix(max(irid_params.z, 0.0), max(irid_params.w, 0.0), clamp(ir.g, 0.0, 1.0));
  }
  if ((tile_mask & 1024u) != 0u) {
    let lm = surface_read_atlas(atlas, samp, 10u, uv1, true);
    out.lobes |= surface_lightmap;
    out.lightmap_irradiance = lm.rgb;
    out.has_lightmap = 1.0;
  }
  if ((tile_mask & 2048u) != 0u && out.has_lightmap > 0.5) {
    let ld = surface_read_atlas(atlas, samp, 11u, uv1, true);
    let raw_dir = ld.xyz * 2.0 - vec3f(1.0);
    if (dot(raw_dir, raw_dir) > 1e-4) {
      let half_lambert = 0.5 + 0.5 * clamp(dot(normalize(out.normal), normalize(raw_dir)), -1.0, 1.0);
      let dir_weight = clamp(ld.a, 0.0, 1.0);
      out.lightmap_irradiance *= mix(1.0, half_lambert / max(dir_weight, 0.25), dir_weight);
    }
  }
  return out;
}
// END surface_lobes
// BEGIN surface
// A standard surface lit by the sky and the frame's lights: `standard_surface`
// takes a `StandardSurface` (surface_lobes.wgsl: add `SURFACE_LOBES_WGSL`
// first) and gives the light it sends the eye from the sky's diffuse light and
// reflections (`ENVIRONMENT_WGSL`) and from the point and spot lights of its
// cluster (`LIGHTS_WGSL`), through whichever lobes it has: the base GGX lobe,
// a clear coat, sheen, anisotropy, transmission and the cheap subsurface. The
// sun comes separately (`standard_sun`, surface_lobes.wgsl) with its shadow's
// visibility. `surface_ambient` and `clustered_surface` are the base lobe's
// halves alone, for shaders that light the albedo themselves.

// The unit vector along increasing u at a fragment, from the screen-space
// derivatives of its world position and uv, in xyz, and the handedness of the
// uv's frame (+1 or -1) in w. Call it in uniform control flow (derivatives),
// where a mesh's tangent stream may be zero (`surface_tangent`).
fn surface_derived_tangent(world: vec3f, normal: vec3f, uv: vec2f) -> vec4f {
  let dp1 = dpdx(world);
  let dp2 = dpdy(world);
  let duv1 = dpdx(uv);
  let duv2 = dpdy(uv);
  let dp2perp = cross(dp2, normal);
  let dp1perp = cross(normal, dp1);
  let t = dp2perp * duv1.x + dp1perp * duv2.x;
  let b = dp2perp * duv1.y + dp1perp * duv2.y;
  let longest = max(dot(t, t), dot(b, b));
  if (longest < 0.000000000001) {
    return vec4f(0.0);
  }
  let scale = inverseSqrt(longest);
  return vec4f(t * scale, select(-1.0, 1.0, dot(cross(normal, t), b) >= 0.0));
}

// The light a surface sends the eye from the sky: its diffuse light (the probe
// volume's or the sky's, with whatever it keeps from the reflection) and the
// reflection (the screen, the reflection probes or the sky), through the coat
// and the sheen over them where it has either.
fn surface_sky(s: StandardSurface) -> vec3f {
  let nv = max(dot(s.normal, s.view), 0.0001);
  let split = sky_brdf(nv, clamp(s.roughness, 0.04, 1.0));
  let reflecting = max(sky_light.tint.w, select(0.0, 1.0, reflection_probes.counts.x > 0.5 || sky_light.ssr.x > 0.5));
  let coating = (s.lobes & surface_coat) != 0u && s.coat > 0.0;
  let sheening = (s.lobes & surface_sheen) != 0u && dot(s.sheen_color, s.sheen_color) > 0.0;
  let brushing = (s.lobes & surface_anisotropy) != 0u && s.anisotropy != 0.0;
  let transmitting = (s.lobes & surface_transmission) != 0u && s.transmission > 0.0;
  let iridescent = (s.lobes & surface_iridescence) != 0u && s.iridescence > 0.0;
  let lightmapped = (s.lobes & surface_lightmap) != 0u && s.has_lightmap > 0.5;
  if (!coating && !sheening && !brushing && !transmitting && !iridescent && !lightmapped && s.occlusion >= 1.0) {
    let reflected = s.f0 * split.x + vec3f(s.f90 * split.y);
    let diffuse = s.base_color * (1.0 - clamp(s.metallic, 0.0, 1.0)) * (vec3f(1.0) - reflected * reflecting);
    // The base lobe alone, written as one expression: the very arithmetic
    // `surface_ambient` always had, to the bit.
    return diffuse * ambient_light_view(s.world, s.normal, s.view)
      + sky_reflection_lobe_split(s.world, s.normal, s.view, s.roughness, s.f0, s.f90, split);
  }
  var irradiance = s.lightmap_irradiance;
  if (!lightmapped) {
    irradiance = ambient_light_view(s.world, s.normal, s.view);
  }
  var lookup = s.normal;
  if (brushing) {
    lookup = lobe_bent_normal(s.normal, s.tangent, s.bitangent, s.view, s.roughness, clamp(s.anisotropy, -1.0, 1.0));
  }
  var eff_f0 = s.f0;
  if (iridescent) {
    let irid_f = lobe_iridescence_fresnel(1.0, max(s.iridescence_ior, 1.0001), nv, max(s.iridescence_thickness, 0.0), s.f0);
    eff_f0 = mix(eff_f0, irid_f, clamp(s.iridescence, 0.0, 1.0));
  }
  let reflected = eff_f0 * split.x + vec3f(s.f90 * split.y);
  let diffuse = s.base_color * (1.0 - clamp(s.metallic, 0.0, 1.0)) * (vec3f(1.0) - reflected * reflecting);
  var reflection = sky_reflection_lobe_split(s.world, lookup, s.view, s.roughness, eff_f0, s.f90, split);
  if (s.occlusion < 1.0) {
    irradiance *= s.occlusion;
    reflection *= specular_occlusion(nv, s.occlusion, s.roughness);
  }
  var base = diffuse * irradiance;
  if (transmitting) {
    // The diffuse lobe gives way to the picture behind: what the base lobe's
    // reflection leaves of it, by the albedo and not the metal.
    let weight = clamp(s.transmission, 0.0, 1.0);
    let through = s.behind * s.base_color * max(vec3f(1.0) - reflected, vec3f(0.0)) * (1.0 - clamp(s.metallic, 0.0, 1.0));
    base = diffuse * (irradiance * (1.0 - weight)) + through * weight;
  }
  var layers = 1.0;
  var added = vec3f(0.0);
  if (sheening) {
    let tint = max(s.sheen_color, vec3f(0.0));
    let albedo = lobe_sheen_albedo(nv, s.sheen_roughness);
    added = tint * (albedo * irradiance);
    layers = max(0.0, 1.0 - max(tint.x, max(tint.y, tint.z)) * albedo);
  }
  var coated = 1.0;
  var coat = vec3f(0.0);
  if (coating) {
    let weight = clamp(s.coat, 0.0, 1.0);
    let nvc = dot(s.coat_normal, s.view);
    coated = 1.0 - weight * lobe_coat_fresnel(clamp(nvc, 0.00001, 1.0));
    coat = weight * sky_reflection_lobe(s.world, s.coat_normal, s.view, nvc, s.coat_roughness, vec3f(0.04), 1.0);
    if (s.occlusion < 1.0) {
      coat *= specular_occlusion(max(nvc, 0.0001), s.occlusion, s.coat_roughness);
    }
  }
  // The sheen sits over the base lobes and the coat over those; as in the
  // lights, the sheen is not dimmed by the coat.
  return (base + reflection) * (layers * coated) + added + coat;
}

// What the frame's lights bring to a surface (`surface_light`, summed): its
// cluster's lights, each with its shadow.
fn clustered_lobes(s: StandardSurface) -> LightResponse {
  var sum = LightResponse(vec3f(0.0), vec3f(0.0));
  if (cluster_grid.counts.w == 0u) {
    return sum;
  }
  let behind = (s.lobes & surface_subsurface) != 0u && s.subsurface > 0.0;
  let range = cluster_ranges[cluster_of(s.position, -s.view_z)];
  for (var index = range.x; index < range.x + range.y; index++) {
    let taken = cluster_light_sample(cluster_lights[cluster_indices[index]], s.world, s.normal, behind);
    let one = surface_light(s, taken.way, taken.radiance);
    sum.diffuse += one.diffuse;
    sum.specular += one.specular;
  }
  return sum;
}

// The light a standard surface sends the eye from the sky, the frame's
// point/spot lights, and LTC area lights (not the sun: `standard_sun`).
fn standard_surface(s: StandardSurface) -> vec3f {
  var light = surface_sky(s);
  let local = clustered_lobes(s);
  light += local.diffuse * s.base_color * (1.0 - clamp(s.metallic, 0.0, 1.0));
  light += local.specular;
  if (cluster_grid.options.y >= 0.5) {
    let nv = max(dot(s.normal, s.view), 0.0001);
    var eff_f0 = s.f0;
    if ((s.lobes & surface_iridescence) != 0u && s.iridescence > 0.0) {
      let irid_f = lobe_iridescence_fresnel(1.0, max(s.iridescence_ior, 1.0001), nv, max(s.iridescence_thickness, 0.0), s.f0);
      eff_f0 = mix(eff_f0, irid_f, clamp(s.iridescence, 0.0, 1.0));
    }
    var layers = 1.0;
    var sheen_tint = vec3f(0.0);
    if ((s.lobes & surface_sheen) != 0u && dot(s.sheen_color, s.sheen_color) > 0.0) {
      let tint = max(s.sheen_color, vec3f(0.0));
      let albedo = lobe_sheen_albedo(nv, s.sheen_roughness);
      sheen_tint = tint * albedo;
      layers = max(0.0, 1.0 - max(tint.x, max(tint.y, tint.z)) * albedo);
    }
    var coat_weight = 0.0;
    var coat_rough = 0.04;
    if ((s.lobes & surface_coat) != 0u && s.coat > 0.0) {
      coat_weight = clamp(s.coat, 0.0, 1.0);
      coat_rough = clamp(s.coat_roughness, 0.04, 1.0);
      let nvc = max(dot(s.coat_normal, s.view), 0.00001);
      layers *= 1.0 - coat_weight * lobe_coat_fresnel(min(nvc, 1.0));
    }
    var diff_scale = layers;
    if ((s.lobes & surface_transmission) != 0u && s.transmission > 0.0) {
      diff_scale *= 1.0 - clamp(s.transmission, 0.0, 1.0);
    }
    light += clustered_area_lights_surface(
      s.world,
      s.normal,
      s.view,
      s.base_color,
      s.metallic,
      eff_f0,
      s.f90,
      s.roughness,
      diff_scale,
      layers,
      coat_weight,
      coat_rough,
      s.coat_normal,
      sheen_tint,
    );
  }
  return light;
}

// A surface's light from the sky, diffuse and glossy: `albedo`, `metallic`
// and `roughness` as `lighting.Material` has them, at `world` facing
// `normal`, seen along `view` (unit length, toward the eye). The diffuse part
// gives up what the reflection takes. The base lobe of `standard_surface`.
fn surface_ambient(world: vec3f, normal: vec3f, view: vec3f, albedo: vec3f, metallic: f32, roughness: f32) -> vec3f {
  return surface_sky(surface_default(world, normal, view, albedo, metallic, roughness, vec4f(0.0), 0.0));
}

// The frame's point and spot lights at `world`: `clustered_light`'s diffuse
// sum and their GGX specular, for a surface of `roughness` and `f0` seen along
// `view` (see `surface_light`); `position` and `view_z` as there. The base
// lobe of `standard_surface`.
fn clustered_surface(world: vec3f, normal: vec3f, view: vec3f, roughness: f32, f0: vec3f, position: vec4f, view_z: f32) -> LightResponse {
  var s = surface_default(world, normal, view, vec3f(1.0), 0.0, roughness, position, view_z);
  s.f0 = f0;
  return clustered_lobes(s);
}
// END surface

struct Vertex {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(12) uv1: vec2f,
};
struct Surface {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
  @location(1) world: vec3f,
  @location(2) uv: vec2f,
  @location(3) uv1: vec2f,
};

@vertex fn vertex(input: Vertex) -> Surface {
  var output: Surface;
  let world = draw.model * vec4f(input.position, 1.0);
  output.position = scene.projection * scene.view * world;
  output.normal = (draw.normal * vec4f(input.normal, 0.0)).xyz;
  output.world = world.xyz;
  output.uv = input.uv;
  output.uv1 = input.uv1;
  return output;
}

// The lobes this material has, as `SURFACE_*` flags: constants, so that the
// compiler leaves out the ones it does not have. The base lobe (a GGX highlight
// and Lambert's diffuse, the sky's reflection) is always there. Name the others
// to read their parameters from the material's `surface` block:
//   const shader_lobes: u32 = surface_coat | surface_sheen;
// surface_coat (clearcoat, clearcoatRoughness: a glossy layer over the
// material), surface_sheen (sheenColor, sheenRoughness: cloth), surface_anisotropy
// (anisotropy, anisotropyRotation: brushed metal; needs a frame, below),
// surface_specular (ior, specular, specularColor change the face-on reflectance)
// and surface_transmission (transmission, ior, thickness, attenuationColor,
// attenuationDistance: glass; it needs the frame's picture of what is behind,
// `TRANSLUCENCY_WGSL` and `SURFACE_TRANSMISSION_WGSL`, and `s = surface_transmit(s);`
// before `standard_surface`). The cheap surface_subsurface (foliage, wax) has
// no block parameters: set `s.subsurface`, `s.subsurface_color` and `s.thin`.
const shader_lobes: u32 = 0u;

@fragment fn fragment(surface: Surface) -> @location(0) vec4f {
  let normal = normalize(surface.normal);
  let view = normalize(scene.eye.xyz - surface.world);
  let seen = scene.view * vec4f(surface.world, 1.0);
  let uv = surface_transform_uv(surface.uv, draw.uv_transform, draw.maps.xy);
  let base_sample = textureSampleBias(base_texture, base_sampler, uv, scene.shadow_params.w);
  let is_atlas = draw.settings.w > 4.5;
  let direct_layer = textureSampleBias(layer_texture, base_sampler, uv, scene.shadow_params.w);
  let emissive_sample = select(direct_layer.rgb, surface_read_atlas(layer_texture, base_sampler, 1u, uv, false).rgb, is_atlas);
  // The material's parameters are metallic (x) and roughness (y, 0.5 if it gives none).
  let roughness = select(draw.parameters.y, 0.5, draw.parameters.y <= 0.0);
  let model_scale = length(draw.model[0].xyz);
  var s = surface_default(
    surface.world,
    normal,
    view,
    draw.color.rgb * base_sample.rgb,
    draw.parameters.x,
    roughness,
    surface.position,
    seen.z,
  );
  s = surface_apply_block(s, draw.surface, shader_lobes, model_scale);
  if (!is_atlas) {
    s = surface_apply_orm(s, direct_layer, draw.maps.z);
  }
  let tile_mask = u32(max(draw.atlas.x, 0.0) + 0.5);
  s = surface_apply_atlas(
    s,
    layer_texture,
    base_sampler,
    uv,
    surface.uv1,
    draw.settings.w,
    tile_mask,
    draw.surface,
    draw.iridescence,
    model_scale,
  );
  let visibility = sun_shadow(surface.world, normal, surface.position);
  var light = standard_surface(s) + standard_sun(s, scene.light_direction.xyz, scene.light_color.rgb, visibility);
  if (draw.maps.w > 0.5) {
    light += emissive_sample * draw.surface[4].w;
  }
  return vec4f(light, draw.color.a * base_sample.a);
}
