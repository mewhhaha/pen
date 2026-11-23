// A material shader: what pen-render binds for every draw (group 0, the
// frame's scene, its sky light and the sun's shadows; group 1, the draw and its
// material) and the mesh's vertex layout. This one lights the material's color
// from the frame's sun, shadowed (`sun_shadow`: the cascaded maps, filtered,
// and contact shadows, as the game asks), and the sky's light all round
// (`ambient_light`: the game's probe volume where it has one, else its sky).
// The leading fields of each struct are fixed; a shader declares as many of
// them as it reads, but the shadows need the whole Scene as it is here. The
// blocks between the BEGIN and END lines are pen-render's SHADOWS_WGSL and
// ENVIRONMENT_WGSL, which declare their bindings; a test keeps them the same.
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
};
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
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
const SUN_MAX_RADIUS: f32 = 12.0;
// How far a surface is moved off along its normal (in texels, at a right
// angle to the light) before it reads the map, and how much depth a texel of
// the receiver's slope adds on top of what the plane already gives.
const SUN_NORMAL_OFFSET: f32 = 1.5;
const SUN_BIAS_BASE: f32 = 1.0;
const SUN_BIAS_SLOPE: f32 = 0.75;
const SUN_MAX_SLOPE: f32 = 6.0;
// The widest filter, in texels, hard shadows and plain filtering use.
const SUN_FILTER_RADIUS: f32 = 1.5;
// The farthest a blocker is searched for, in world units from the receiver.
const SUN_BLOCKER_REACH: f32 = 8.0;

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
  // The depth a step of one world unit along the map's x and y adds on the
  // receiver's plane, and a world unit's depth.
  slope: vec2f,
  unit: f32,
  // A texel's width in world units, and the unit of x and y in texture coordinates.
  texel: f32,
};

fn sun_map_of(map: u32, world: vec3f, normal: vec3f, reach: f32) -> SunMap {
  let texel = scene.cascade_texels[map];
  let matrix = scene.cascade_view_projection[map];
  let toward = -normalize(scene.light_direction.xyz);
  let facing = dot(normal, toward);
  let sine = sqrt(clamp(1.0 - facing * facing, 0.0, 1.0));
  let shifted = world + normal * (texel.y * SUN_NORMAL_OFFSET * sine);
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
  let sx = clamp(-dot(normal, row_x / length(row_x)) / flat_, -SUN_MAX_SLOPE, SUN_MAX_SLOPE);
  let sy = clamp(-dot(normal, row_y / length(row_y)) / flat_, -SUN_MAX_SLOPE, SUN_MAX_SLOPE);
  result.unit = unit_z;
  // The plane's depth change per world unit across: the sign follows from
  // the plane through the receiver, whose normal faces the light.
  result.slope = vec2f(sx, sy) * unit_z;
  result.texel = texel.y;
  return result;
}

// The depth a tap offset by `offset` texels from the receiver compares
// against: the receiver's plane there, and a bias that grows with its slope.
fn sun_tap_depth(m: SunMap, offset: vec2f) -> f32 {
  let world = offset * m.texel;
  let plane = dot(m.slope, world);
  let bias = (SUN_BIAS_BASE + SUN_BIAS_SLOPE * (abs(m.slope.x) + abs(m.slope.y)) / m.unit) * m.texel * m.unit;
  return m.depth + plane - bias;
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
  var reach = SUN_FILTER_RADIUS;
  if (mode == 2u) {
    reach = SUN_MAX_RADIUS;
  }
  let m = sun_map_of(map, world, normal, reach);
  if (m.inside < 0.0 || m.depth > 1.0 || m.depth < 0.0) {
    return vec2f(1.0, -1.0);
  }
  if (mode == 0u) {
    return vec2f(textureSampleCompareLevel(shadow_cascades, shadow_sampler, m.uv, map, sun_tap_depth(m, vec2f(0.0))), m.inside);
  }
  if (mode == 1u) {
    return vec2f(sun_filtered(map, m, SUN_FILTER_RADIUS, spin, 16u), m.inside);
  }
  // Soft shadows: look for blockers where the penumbra could reach from,
  // average how far they are, and filter as wide as the sun's size makes
  // the penumbra there.
  let softness = scene.shadow_params.x;
  let size = vec2f(textureDimensions(shadow_cascades));
  let search = clamp(softness * SUN_BLOCKER_REACH * 0.5 / m.texel, 1.0, SUN_MAX_RADIUS - 1.0);
  var gap = 0.0;
  var blockers = 0.0;
  for (var i = 0u; i < 12u; i++) {
    let tap = shadow_taps[i];
    let offset = vec2f(tap.x * spin.x - tap.y * spin.y, tap.x * spin.y + tap.y * spin.x) * search;
    let at = vec2i(floor((m.uv + offset * scene.cascade_texels[map].x) * size));
    let seen = textureLoad(shadow_cascades, at, map, 0);
    let wanted = sun_tap_depth(m, vec2f(offset.x, -offset.y));
    if (seen < wanted) {
      gap += (wanted - seen) / m.unit;
      blockers += 1.0;
    }
  }
  if (blockers == 0.0) {
    return vec2f(1.0, m.inside);
  }
  let blocked = gap / blockers;
  let radius = clamp(softness * blocked * 0.5 / m.texel, SUN_FILTER_RADIUS, SUN_MAX_RADIUS);
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
  var shadowed = 0.0;
  for (var i = 0u; i < steps; i++) {
    let along = (f32(i) + 0.25 + 0.75 * jitter) / f32(steps);
    let point = origin + toward * (along * length);
    let clip = scene.projection * (scene.view * vec4f(point, 1.0));
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
  let noise = shadow_noise(position.xy);
  let angle = 6.2831853 * noise;
  let cascaded = sun_visibility(world, normal, vec2f(cos(angle), sin(angle)));
  if (cascaded <= 0.0) {
    return 0.0;
  }
  return min(cascaded, contact_shadow(world, normal, fract(noise * 7.0 + 0.5)));
}
// END shadows
// BEGIN environment
// The sky as light for a material shader: diffuse irradiance from nine
// spherical-harmonic coefficients the game works out, and glossy reflections
// from a cube map of the same sky prefiltered by roughness (environment.rs)
// with the split-sum BRDF table. Declare these bindings (group 0 of every
// mesh pipeline, after the scene's) and call `ambient_light` where the flat
// ambient color went, and `sky_reflection` or `surface_ambient` for gloss.
struct SkyLight {
  // The sky's diffuse light, real SH terms of bands 0 to 2 in the order
  // y, z, x (band 1), xy, yz, 3zz - 1, xz, xx - yy (band 2): what a white
  // surface facing a normal shows, cosine lobe applied and divided by pi.
  sh: array<vec4f, 9>,
  // rgb: scale from the cube's sky colors to the light the sky gives all
  // round; w: 1 when the cube holds the frame's sky, 0 for none (an inset's).
  tint: vec4f,
  // x: the cube's last level; yzw: unused.
  reflection: vec4f,
  // xyz: where probe 0 of the baked probe volume is; w: the spacing between probes.
  probe_origin: vec4f,
  // xyz: probes along x, y and z; w: 1 when the frame has a probe volume.
  probe_counts: vec4f,
};
@group(0) @binding(4) var sky_cube: texture_cube<f32>;
@group(0) @binding(5) var sky_table: texture_2d<f32>;
@group(0) @binding(6) var sky_sampler: sampler;
@group(0) @binding(7) var<uniform> sky_light: SkyLight;
// Seven vec4s a probe: nine RGB terms (27 floats, order as `sky_ambient`) then
// a validity weight of 1 (open) or 0 (inside geometry); probe (z * ny + y) * nx + x.
@group(0) @binding(8) var<storage, read> probe_data: array<vec4f>;

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

// The light all round a surface at `world` facing `normal` (unit length), to
// multiply by albedo: the baked probe volume's where the frame has one, else
// (and past its edge, blending smoothly) the whole sky's. This is the one
// place a material asks for diffuse sky light.
fn ambient_light(world: vec3f, normal: vec3f) -> vec3f {
  let sky = sky_ambient(normal);
  if (sky_light.probe_counts.w < 0.5) {
    return sky;
  }
  let probes = probe_ambient(world, normal);
  return mix(sky, probes.rgb, probes.a);
}

// The split-sum table at `normal_dot_view` and `roughness`: x scales f0 and y
// adds to it. The same GGX, height-correlated Smith and Schlick Fresnel as the
// point and spot lights'.
fn sky_brdf(normal_dot_view: f32, roughness: f32) -> vec2f {
  let uv = vec2f(clamp(normal_dot_view, 0.0, 1.0), clamp(roughness, 0.0, 1.0));
  return textureSampleLevel(sky_table, sky_sampler, uv, 0.0).rg;
}

// The sky mirrored in a surface facing `normal`, seen along `view` (both unit
// length, `view` toward the eye), of `roughness` (0 mirror to 1 matte) and
// with Fresnel reflectance `f0` face on: the prefiltered sky at the
// reflection, times the table's scale and bias. Zero when the frame has no
// sky to reflect.
fn sky_reflection(normal: vec3f, view: vec3f, roughness: f32, f0: vec3f) -> vec3f {
  if (sky_light.tint.w < 0.5) {
    return vec3f(0.0);
  }
  let nv = max(dot(normal, view), 0.0001);
  let toward = reflect(-view, normal);
  let rough = clamp(roughness, 0.04, 1.0);
  let radiance = textureSampleLevel(sky_cube, sky_sampler, toward, rough * sky_light.reflection.x).rgb * sky_light.tint.rgb;
  let split = sky_brdf(nv, rough);
  return radiance * (f0 * split.x + vec3f(split.y));
}

// How much of the sky's reflection a surface with ambient occlusion
// `occlusion` (1 open to 0 closed) shows: a glossy lobe is less hidden than a
// rough one, and a grazing view more.
fn specular_occlusion(normal_dot_view: f32, occlusion: f32, roughness: f32) -> f32 {
  return clamp(pow(normal_dot_view + occlusion, exp2(-16.0 * roughness - 1.0)) - 1.0 + occlusion, 0.0, 1.0);
}

// A surface's light from the sky, diffuse and glossy: `albedo`, `metallic`
// and `roughness` as `lighting.Material` has them, at `world` facing
// `normal`, seen along `view` (unit length, toward the eye). The diffuse part
// gives up what the reflection takes.
fn surface_ambient(world: vec3f, normal: vec3f, view: vec3f, albedo: vec3f, metallic: f32, roughness: f32) -> vec3f {
  let f0 = mix(vec3f(0.04), albedo, clamp(metallic, 0.0, 1.0));
  let nv = max(dot(normal, view), 0.0001);
  let split = sky_brdf(nv, clamp(roughness, 0.04, 1.0));
  let reflected = f0 * split.x + vec3f(split.y);
  let diffuse = albedo * (1.0 - clamp(metallic, 0.0, 1.0)) * (vec3f(1.0) - reflected * sky_light.tint.w);
  return diffuse * ambient_light(world, normal) + sky_reflection(normal, view, roughness, f0);
}
// END environment

struct Vertex {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
};
struct Surface {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
  @location(1) world: vec3f,
};

@vertex fn vertex(input: Vertex) -> Surface {
  var output: Surface;
  let world = draw.model * vec4f(input.position, 1.0);
  output.position = scene.projection * scene.view * world;
  output.normal = (draw.normal * vec4f(input.normal, 0.0)).xyz;
  output.world = world.xyz;
  return output;
}

@fragment fn fragment(surface: Surface) -> @location(0) vec4f {
  let normal = normalize(surface.normal);
  let sun = max(dot(normal, -normalize(scene.light_direction.xyz)), 0.0) * sun_shadow(surface.world, normal, surface.position);
  let light = ambient_light(surface.world, normal) + scene.light_color.rgb * sun;
  return vec4f(draw.color.rgb * light, draw.color.a);
}
