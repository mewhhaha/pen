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
  let scene = textureLoad(scene_color, pixel, 0);
  let remaining = scene.a;
  let color = scene.rgb - remaining * atmosphere.clear.rgb;
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
    return vec4f(color + remaining * background, 1.0);
  }
  // Depth counts opaque coverage; alpha also accounts for translucent
  // surfaces and particles covering the background without writing depth.
  let surface = color / (1.0 - empty) *
    mix(1.0, occlusion(position.xy), atmosphere.fog.w);
  let point = unproject(position.xy, nearest);
  let mist = select(atmosphere.clear.rgb, sky(direction, false), with_sky);
  let thickness = mist_beyond_arena(camera, point) * atmosphere.fog.y;
  let misted = mix(surface * cloud_shadow(point), mist, 1.0 - exp(-thickness));
  return vec4f(misted * (1.0 - empty) + remaining * background, 1.0);
}
