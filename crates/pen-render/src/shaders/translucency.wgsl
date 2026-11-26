// What a translucent surface, a particle or an overlay reads of the frame
// behind it: the opaque picture and its depth, and the mist between the eye and
// the surface (translucent surfaces draw after the atmosphere pass has misted the
// opaque ones, and so mist themselves at their own depth). Declare these
// bindings (group 0 of every mesh pipeline, after the scene's, in the fragment
// stage), and `sky_sampler` (binding 6: `ENVIRONMENT_WGSL` declares it), then
//
//   out = fog_apply(lit_colour, world)
//
// where the surface's usual colour and alpha go, and the translucent blend
// composites it over the fogged picture behind it. Where there is no such
// pass (the opaque draws, the shadow maps, an inset, a reflection probe's
// capture) the bindings hold stand-ins and every function here is the
// identity: `fog_apply` returns its colour, `opaque_available` is false and
// `opaque_behind` gives black, so one shader serves every pass.
//
// The block at binding 17 is the froxel volume's uniform and what the analytic
// mist beyond the grid's reach and without a grid needs (the sky's colours, the
// sun, the clouds, the camera); its first fields are `Volume`'s of
// volume_common.wgsl, field for field.
struct FogBlock {
  view: mat4x4f,
  previous: mat4x4f,
  projection: vec4f,
  grid: vec4f,
  slicing: vec4f,
  jitter: vec4f,
  medium: vec4f,
  scatter: vec4f,
  noise: vec4f,
  drift: vec4f,
  ambient: vec4f,
  clip: mat4x4f,
  depth: vec4f,
  flags: vec4f,
  camera: vec4f,
  zenith: vec4f,
  horizon: vec4f,
  ground: vec4f,
  sun: vec4f,
  sun_color: vec4f,
  clear: vec4f,
  fog: vec4f,
  clouds: vec4f,
};
// The opaque surfaces' picture after the sky, the mist and the ambient
// occlusion, level 0 at the scene's size and each next level the last halved
// by a 4 x 4 tent, down to one texel; the nearest depth buffer value of each of
// its pixels (1 where nothing opaque is); the integrated froxel volume (layer
// k: the in-scatter and the optical depth up to boundary k); the fog block.
@group(0) @binding(14) var opaque_color: texture_2d<f32>;
@group(0) @binding(15) var opaque_depth: texture_2d<f32>;
@group(0) @binding(16) var fog_froxels: texture_3d<f32>;
@group(0) @binding(17) var<uniform> fog_block: FogBlock;

// Whether the pass reads an opaque picture (a translucent pass does).
fn opaque_available() -> bool {
  return fog_block.flags.x > 0.5;
}

// A fragment's place in the opaque picture, from its `@builtin(position)`: 0 to 1, y down.
fn opaque_screen_uv(position: vec4f) -> vec2f {
  return position.xy / vec2f(textureDimensions(opaque_color, 0));
}

// A world point's place in the opaque picture, through the projection the scene
// draws with (jitter included): 0 to 1, y down. Beyond the picture it is outside 0 to 1.
fn opaque_world_uv(world: vec3f) -> vec2f {
  let clip = fog_block.clip * vec4f(world, 1.0);
  let ndc = clip.xy / max(clip.w, 1e-6);
  return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

// The last level of the opaque picture's chain (a texel across).
fn opaque_levels() -> f32 {
  return f32(textureNumLevels(opaque_color)) - 1.0;
}

// The opaque picture at `uv`, clamped to it, read trilinearly at `level` (clamped
// to the chain). Black where there is none: ask `opaque_available` first.
fn opaque_behind(uv: vec2f, level: f32) -> vec3f {
  if (!opaque_available()) {
    return vec3f(0.0);
  }
  return textureSampleLevel(opaque_color, sky_sampler, clamp(uv, vec2f(0.0), vec2f(1.0)), clamp(level, 0.0, opaque_levels())).rgb;
}

// How far ahead of the eye (along the view, not the ray) the opaque surface
// behind the point of the picture at `uv` is; very large where there is none.
fn opaque_depth_at(uv: vec2f) -> f32 {
  if (!opaque_available()) {
    return 1.0e30;
  }
  let size = vec2f(textureDimensions(opaque_depth));
  let pixel = vec2i(clamp(uv * size, vec2f(0.0), size - vec2f(1.0)));
  let value = textureLoad(opaque_depth, pixel, 0).r;
  if (value >= 1.0) {
    return 1.0e30;
  }
  // The projection's z and w rows give a value's distance, for perspective
  // and orthographic cameras alike.
  let q = fog_block.depth;
  return (value * q.w - q.y) / (value * q.z - q.x);
}

// A boundary of the grid, in slices from the eye, of a depth ahead of it.
fn fog_boundary(depth: f32) -> f32 {
  return fog_block.grid.z * log(1.0 + depth / fog_block.slicing.x) / fog_block.slicing.y;
}

// pi times the Henyey-Greenstein phase function.
fn fog_phase(g: f32, c: f32) -> f32 {
  let k = clamp(g, -0.95, 0.95);
  let d = max(1.0 + k * k - 2.0 * k * c, 0.0001);
  return (1.0 - k * k) / (4.0 * d * sqrt(d));
}

// The clouds' noise (atmosphere_common.wgsl's, name for name).
fn fog_lattice(cell: vec2f) -> f32 {
  let c = vec2u(vec2i(cell) + vec2i(65536));
  var h = (c.x * 0x8da6b343u) ^ (c.y * 0xd8163841u);
  h ^= h >> 16u;
  h *= 0x7feb352du;
  h ^= h >> 15u;
  h *= 0x846ca68bu;
  h ^= h >> 16u;
  return f32(h >> 8u) / 16777216.0;
}
fn fog_smooth_noise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = p - i;
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(fog_lattice(i), fog_lattice(i + vec2f(1.0, 0.0)), u.x),
    mix(fog_lattice(i + vec2f(0.0, 1.0)), fog_lattice(i + vec2f(1.0, 1.0)), u.x),
    u.y,
  );
}
fn fog_billows(p: vec2f) -> f32 {
  var sum = 0.0;
  var weight = 0.5;
  var q = p;
  for (var octave = 0; octave < 5; octave++) {
    sum += weight * fog_smooth_noise(q);
    q = q * 2.03 + vec2f(1.7, -3.1);
    weight *= 0.5;
  }
  return sum / 0.96875;
}

// How much of the sun's light the clouds leave a point (the atmosphere pass
// darkens the surfaces it fogs by it).
fn fog_cloud_shadow(point: vec3f) -> f32 {
  if (fog_block.clouds.x <= 0.0) {
    return 1.0;
  }
  let sun = fog_block.sun.xyz;
  let at = (point.xz - point.y * sun.xz / max(sun.y, 0.25) + fog_block.clouds.yz) / 18.0;
  let edge = 1.0 - fog_block.clouds.x;
  let density = smoothstep(edge - 0.04, edge + 0.2, fog_billows(at));
  let daylight = smoothstep(0.15, 0.65, max(fog_block.sun_color.r, fog_block.sun_color.g));
  return 1.0 - density * 0.42 * daylight;
}

// The sky's colour in `direction` as the mist takes it up, or the clear colour
// where the frame has no sky.
fn fog_sky(direction: vec3f) -> vec3f {
  if (fog_block.camera.w <= 0.5) {
    return fog_block.clear.rgb;
  }
  let up = direction.y;
  var color = mix(fog_block.horizon.rgb, fog_block.zenith.rgb, sqrt(clamp(up, 0.0, 1.0)));
  color = mix(color, fog_block.ground.rgb, smoothstep(0.0, 0.3, -up));
  let toward = max(dot(direction, fog_block.sun.xyz), 0.0);
  return color + fog_block.sun_color.rgb * (0.1 * pow(toward, 4.0) + 0.25 * pow(toward, 32.0));
}

// The mist along the segment from a to b, whose density falls off exponentially with height.
fn fog_mist_along(a: vec3f, b: vec3f) -> f32 {
  let height = fog_block.fog.z;
  let low = max(a.y, 0.0);
  let high = max(b.y, 0.0);
  let length = distance(a, b);
  if (abs(high - low) < 0.01) {
    return length * exp(-low / height);
  }
  return length * height * (exp(-low / height) - exp(-high / height)) / (high - low);
}

// The mist between two points outside the arena, the vertical cylinder about the
// origin, which stays clear.
fn fog_mist_beyond(start: vec3f, end: vec3f) -> f32 {
  let radius = fog_block.fog.x + 0.5 * fog_block.scatter.w;
  let along = end.xz - start.xz;
  let a = dot(along, along);
  let b = 2.0 * dot(start.xz, along);
  let c = dot(start.xz, start.xz) - radius * radius;
  let discriminant = b * b - 4.0 * a * c;
  if (a < 1e-6 || discriminant <= 0.0) {
    return select(0.0, fog_mist_along(start, end), c > 0.0);
  }
  let root = sqrt(discriminant);
  let enter = clamp((-b - root) / (2.0 * a), 0.0, 1.0);
  let leave = clamp((-b + root) / (2.0 * a), 0.0, 1.0);
  return fog_mist_along(start, mix(start, end, enter)) + fog_mist_along(mix(start, end, leave), end);
}

// The light the analytic mist scatters toward the eye per unit of optical depth,
// unshadowed: the sun in the mist's phase and the ambient light.
fn fog_light(direction: vec3f, cloud_sh: f32) -> vec3f {
  let around = mix(fog_block.ambient.rgb, fog_sky(direction), fog_block.scatter.z) * fog_block.scatter.y;
  let sun = fog_block.sun_color.rgb * (fog_block.scatter.x * fog_phase(fog_block.medium.w, dot(fog_block.sun.xyz, direction)) * cloud_sh);
  return fog_block.medium.rgb * (around + sun);
}

// What lies between the eye and `world`, as the atmosphere pass takes it up for a
// surface there: the light the mist scatters toward the eye (rgb) and how much of
// the colour at `world` gets through it (a: the mist's transmittance, times the
// clouds' shadow on the surface). The froxel volume gives the first stretch, its
// optical depth read in slice space so that a homogeneous medium is exact, and the
// same medium goes on analytically past its reach, in the colour the volume ends
// on; without a volume the whole way is analytic. (0, 0, 0, 1) where nothing is.
fn fog_terms(world: vec3f) -> vec4f {
  if (fog_block.flags.y < 0.5) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  let camera = fog_block.camera.xyz;
  let direction = normalize(world - camera);
  let cloud_sh = fog_cloud_shadow(world);
  var scattered = vec3f(0.0);
  var optical = 0.0;
  var start = camera;
  var tint = fog_light(direction, cloud_sh);
  if (fog_block.ambient.w > 0.5) {
    let depth = max(-(fog_block.view * vec4f(world, 1.0)).z, 0.0);
    let reach = min(depth, fog_block.grid.w);
    let at = vec3f(opaque_world_uv(world), (fog_boundary(reach) + 0.5) / (fog_block.grid.z + 1.0));
    let held = textureSampleLevel(fog_froxels, sky_sampler, at, 0.0);
    scattered = held.rgb;
    optical = held.a;
    if (depth > fog_block.grid.w) {
      start = camera + (world - camera) * (fog_block.grid.w / depth);
    } else {
      start = world;
    }
    let seen = 1.0 - exp(-optical);
    tint = mix(tint, scattered / max(seen, 0.0001), smoothstep(0.0, 0.05, seen));
  }
  let beyond = fog_mist_beyond(start, world) * fog_block.fog.y;
  let transmitted = exp(-optical);
  let trans_beyond = exp(-beyond);
  let light = scattered + transmitted * tint * (1.0 - trans_beyond);
  return vec4f(light, cloud_sh * (transmitted * trans_beyond));
}

// `color`, the colour of a surface at `world`, as the eye sees it through the
// mist between: `color * transmittance + in-scatter`, the numbers the atmosphere
// pass gives an opaque surface there. Over a picture already misted to its own
// depth, the usual translucent blend then composites it right (alpha * this +
// (1 - alpha) * what is behind). Additive light takes only the transmittance
// (`fog_terms(world).a`): it covers nothing, so adds no mist.
fn fog_apply(color: vec3f, world: vec3f) -> vec3f {
  let terms = fog_terms(world);
  return color * terms.a + terms.rgb;
}
