// Fills the froxel grid with the medium's light: for each froxel, how thick
// the medium is there (extinction) and how much light it sends toward the
// eye per unit of length (global height mist, noise and local volumes, lit by
// the sun through its cascaded maps and the clouds' shadows, by the clustered
// lights and their shadow maps, and by the sky's or the probes' ambient light,
// all through a Henyey-Greenstein phase), then blends it with the last
// frame's froxel at the same place in the world.
@group(1) @binding(0) var<uniform> volume: Volume;
@group(1) @binding(1) var<uniform> atmosphere: Atmosphere;
// Four vec4s a local volume: centre and shape, half size and density, albedo
// and falloff, emission.
@group(1) @binding(2) var<storage, read> fog_volumes: array<vec4f>;
@group(1) @binding(3) var history: texture_3d<f32>;
@group(1) @binding(4) var history_sampler: sampler;
@group(1) @binding(5) var injected: texture_storage_3d<rgba16float, write>;

// The sun's filter in the volume: the frame's, at most the volume tier's
// (soft shadows are never worth their search in mist).
fn volume_filter(mode: u32) -> u32 {
  return min(mode, u32(volume.drift.w));
}

// The world point of a place in the picture at a depth ahead of the eye.
fn world_at(uv: vec2f, depth: f32) -> vec3f {
  let rotation = mat3x3f(volume.view[0].xyz, volume.view[1].xyz, volume.view[2].xyz);
  return atmosphere.camera.xyz + transpose(rotation) * (volume_ray(uv) * depth);
}

// ---------------------------------------------------------------- the noise
// Ink's `noise.value3` and `noise.fbm_value3`: the same hash, the same
// lattice, the same quintic blend.
fn noise_hash(x: u32, y: u32, z: u32, seed: u32) -> u32 {
  var h = (x * 2747636419u) ^ (y * 2654435769u) ^ (z * 2246822519u) ^ (seed * 3266489917u);
  h = h ^ (h >> 15u);
  h = h * 2246822507u;
  h = h ^ (h >> 13u);
  h = h * 3266489909u;
  return h ^ (h >> 16u);
}

fn noise_corner(x: u32, y: u32, z: u32, seed: u32) -> f32 {
  return f32(noise_hash(x, y, z, seed) >> 8u) / 16777216.0 * 2.0 - 1.0;
}

fn noise_cell(v: f32) -> u32 {
  return bitcast<u32>(i32(floor(v)));
}

fn noise_fade(t: f32) -> f32 {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn value3(seed: u32, p: vec3f) -> f32 {
  let xi = noise_cell(p.x);
  let yi = noise_cell(p.y);
  let zi = noise_cell(p.z);
  let u = noise_fade(p.x - floor(p.x));
  let v = noise_fade(p.y - floor(p.y));
  let w = noise_fade(p.z - floor(p.z));
  let x00 = mix(noise_corner(xi, yi, zi, seed), noise_corner(xi + 1u, yi, zi, seed), u);
  let x10 = mix(noise_corner(xi, yi + 1u, zi, seed), noise_corner(xi + 1u, yi + 1u, zi, seed), u);
  let x01 = mix(noise_corner(xi, yi, zi + 1u, seed), noise_corner(xi + 1u, yi, zi + 1u, seed), u);
  let x11 = mix(noise_corner(xi, yi + 1u, zi + 1u, seed), noise_corner(xi + 1u, yi + 1u, zi + 1u, seed), u);
  return mix(mix(x00, x10, v), mix(x01, x11, v), w);
}

fn fbm_value3(seed: u32, octaves: u32, p: vec3f) -> f32 {
  var sum = 0.0;
  var total = 0.0;
  var amplitude = 1.0;
  var frequency = 1.0;
  for (var octave = 0u; octave < octaves; octave++) {
    sum += amplitude * value3(seed + octave * 101u, p * frequency);
    total += amplitude;
    amplitude *= 0.5;
    frequency *= 2.0;
  }
  return select(sum / total, 0.0, total == 0.0);
}

// ---------------------------------------------------------------- the medium
struct Medium {
  // Extinction, and the share of it that scatters, by channel.
  extinction: f32,
  scattering: vec3f,
  // Light the medium gives out, per unit of length.
  emitted: vec3f,
};

// The global mist at `p`: the exponential height falloff, cleared inside the
// radius (softly over `edge`), varied by the noise.
fn global_density(p: vec3f) -> f32 {
  let density = atmosphere.fog.y;
  if (density <= 0.0) {
    return 0.0;
  }
  let across = length(p.xz);
  let radius = atmosphere.fog.x;
  let edge = volume.scatter.w;
  var mask = select(0.0, 1.0, across > radius);
  if (edge > 0.0) {
    mask = smoothstep(radius, radius + edge, across);
  }
  var factor = 1.0;
  if (volume.noise.x > 0.0) {
    let q = (p + volume.drift.xyz) / volume.noise.y;
    factor = max(0.0, 1.0 + volume.noise.x * fbm_value3(0u, u32(volume.noise.z), q));
  }
  return density * exp(-max(p.y, 0.0) / atmosphere.fog.z) * mask * factor;
}

fn medium_at(p: vec3f) -> Medium {
  var medium: Medium;
  let global = global_density(p);
  medium.extinction = global;
  medium.scattering = volume.medium.rgb * global;
  medium.emitted = vec3f(0.0);
  for (var i = 0u; i < u32(volume.noise.w); i++) {
    let a = fog_volumes[i * 4u];
    let b = fog_volumes[i * 4u + 1u];
    let c = fog_volumes[i * 4u + 2u];
    let d = fog_volumes[i * 4u + 3u];
    let offset = (p - a.xyz) / b.xyz;
    let m = select(max(abs(offset.x), max(abs(offset.y), abs(offset.z))), length(offset), a.w > 0.5);
    if (m >= 1.0) {
      continue;
    }
    var t = 1.0;
    if (c.w > 0.0) {
      t = clamp((1.0 - m) / c.w, 0.0, 1.0);
    }
    let sigma = b.w * t * t * (3.0 - 2.0 * t);
    medium.extinction += sigma;
    medium.scattering += c.xyz * sigma;
    medium.emitted += d.xyz * sigma;
  }
  return medium;
}

// The mean of the light all round `world`: what `ambient_mean` is as a
// luminance, by channel (the probe volume's constant term where the frame
// has one, the sky's past its edge).
fn medium_ambient(world: vec3f) -> vec3f {
  let sky = sky_light.sh[0].rgb * 0.2820947918;
  if (sky_light.probe_counts.w >= 1.5) {
    let ddgi = ddgi_packed_ambient(world, vec3f(0.0, 1.0, 0.0), vec3f(0.0, 1.0, 0.0));
    return mix(sky, ddgi.rgb, ddgi.a);
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
      let term = vec3f(probe_float(index * 28u), probe_float(index * 28u + 1u), probe_float(index * 28u + 2u));
      sum += tri * valid * term * 0.2820947918;
      weight += tri * valid;
    }
  }
  if (weight <= 0.000001 || total <= 0.0) {
    return sky;
  }
  let trust = reach * reach * (3.0 - 2.0 * reach) * (share / total);
  return mix(sky, sum / weight, trust);
}

// The clustered lights' in-scattering at `world`, seen along `direction`
// (from the eye): each light's strength as surfaces take it, without the
// cosine, times its phase and its shadow.
fn medium_lights(world: vec3f, direction: vec3f, uv: vec2f, depth: f32) -> vec3f {
  if (cluster_grid.counts.w == 0u) {
    return vec3f(0.0);
  }
  let counts = cluster_grid.counts;
  let tile = min(vec2u(max(uv, vec2f(0.0)) * vec2f(counts.xy)), counts.xy - 1u);
  let along = select(log(max(depth, 0.000001)), depth, cluster_grid.options.x > 0.5);
  let slice = u32(clamp(along * cluster_grid.scale.z + cluster_grid.scale.w, 0.0, f32(counts.z - 1u)));
  let range = cluster_ranges[(slice * counts.y + tile.y) * counts.x + tile.x];
  var sum = vec3f(0.0);
  for (var index = range.x; index < range.x + range.y; index++) {
    let light = cluster_lights[cluster_indices[index]];
    let toward = light.position.xyz - world;
    let distance = length(toward);
    let reach = light.position.w;
    if (distance >= reach) {
      continue;
    }
    let way = toward / max(distance, 0.0001);
    let ratio = distance / reach;
    let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
    var strength = window * window / (distance * distance + 1.0);
    if (light.color.w > -1.0) {
      strength *= smoothstep(light.color.w, max(light.direction.w, light.color.w + 0.0001), dot(-way, light.direction.xyz));
    }
    if (strength <= 0.0) {
      continue;
    }
    let seen = local_shadow(light, world, way, way, distance);
    sum += light.color.rgb * (strength * seen * phase_scaled(volume.medium.w, dot(way, direction)));
  }
  return sum;
}

// An interleaved gradient noise from a froxel and the frame, to turn the
// sun's filter.
fn froxel_spin(cell: vec2u, frame: f32) -> vec2f {
  let pixel = vec2f(cell) + vec2f(5.588238, 5.588238) * frame;
  let angle = 6.2831853 * fract(52.9829189 * fract(dot(pixel, vec2f(0.06711056, 0.00583715))));
  return vec2f(cos(angle), sin(angle));
}

@compute @workgroup_size(4, 4, 4)
fn inject(@builtin(global_invocation_id) id: vec3u) {
  let size = vec3u(volume.grid.xyz);
  if (any(id >= size)) {
    return;
  }
  let cell = vec3f(id);
  let centre_uv = (cell.xy + 0.5) / volume.grid.xy;
  let sample_uv = (cell.xy + 0.5 + volume.jitter.xy) / volume.grid.xy;
  let centre = world_at(centre_uv, volume_depth(cell.z + 0.5));
  let depth = volume_depth(cell.z + 0.5 + volume.jitter.z);
  let point = world_at(sample_uv, depth);
  let medium = medium_at(point);
  var current = vec4f(0.0);
  if (medium.extinction > 0.0000001 || any(medium.emitted > vec3f(0.0))) {
    let direction = normalize(point - atmosphere.camera.xyz);
    let toward_sun = atmosphere.sun.xyz;
    var sun = vec3f(0.0);
    let daylight = volume.scatter.x;
    if (daylight > 0.0 && any(atmosphere.sun_color.rgb > vec3f(0.0))) {
      let visible = sun_visibility(point, toward_sun, froxel_spin(id.xy, volume.jitter.w + f32(id.z)));
      sun = atmosphere.sun_color.rgb * (daylight * phase_scaled(volume.medium.w, dot(toward_sun, direction)) * visible * cloud_shadow(point));
    }
    let around = mix(medium_ambient(point), mist_sky(direction), volume.scatter.z) * volume.scatter.y;
    let lights = medium_lights(point, direction, sample_uv, depth);
    let source = medium.scattering * (sun + around + lights) + medium.emitted;
    current = vec4f(source, medium.extinction);
  }
  // The last frame's froxel at the same place in the world.
  var result = current;
  if (volume.slicing.z > 0.5) {
    let before = volume.previous * vec4f(centre, 1.0);
    if (before.w > 0.0) {
      let ndc = before.xy / before.w;
      let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
      let boundary = volume_boundary(before.w);
      if (min(uv.x, uv.y) >= 0.0 && max(uv.x, uv.y) <= 1.0 && boundary >= 0.0 && boundary <= volume.grid.z) {
        let kept = textureSampleLevel(history, history_sampler, vec3f(uv, boundary / volume.grid.z), 0.0);
        // A medium that thickened or thinned since is not what the history
        // saw: it counts for less the more it differs.
        let change = abs(current.a - kept.a) / max(max(current.a, kept.a), 0.00001);
        let share = volume.slicing.w * (1.0 - smoothstep(0.25, 0.75, change));
        result = mix(current, kept, share);
      }
    }
  }
  textureStore(injected, id, result);
}
