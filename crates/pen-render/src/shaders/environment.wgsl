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

// The mean of the light all round `world`, as a luminance: the average of
// `ambient_light` over every direction, which is its constant SH term alone.
// A reflection probe keeps it from when it was drawn, and scales what it
// shows by how the light has changed since (`probe_cube_light`).
fn ambient_mean(world: vec3f) -> f32 {
  let weights = vec3f(0.2126, 0.7152, 0.0722);
  let sky = dot(sky_light.sh[0].rgb, weights) * 0.2820947918;
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
  for (var i = 0; i < count; i++) {
    // How far along the full segment, as a share of its pixels.
    let s = (f32(i) + 1.0 + jitter) * stride / span;
    if (s > inside) {
      break;
    }
    let pixel = p0 + delta * s;
    let depth = ssr_depth_at(vec2i(floor(pixel)), layer, size_i);
    let step_gap = abs(ssr_distance(mix(z0, z1, s)) - ssr_distance(mix(z0, z1, previous)));
    if (depth < 1.0) {
      let ahead = ssr_distance(mix(z0, z1, s));
      let behind = ahead - ssr_distance(depth);
      let thick = sky_light.ssr.z * ssr_distance(depth) + step_gap;
      if (behind > 0.0 && behind < thick) {
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
    let u = vec2f(ssr_noise(start, 0.0), ssr_noise(start, 1.0));
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

// The light a surface at `world` facing `normal`, seen along `view` (both unit
// length, `view` toward the eye), of `roughness` (0 mirror to 1 matte) and
// with Fresnel reflectance `f0` face on, mirrors: what it reflects (a ray
// from it meets the screen's picture, the local reflection probes it is in
// the boxes of, or else the prefiltered sky) times the split-sum table's scale
// and bias. The picture takes the probes' and the sky's place by its
// confidence; the probes take the sky's by theirs. Zero where there is
// none of these.
fn sky_reflection(world: vec3f, normal: vec3f, view: vec3f, roughness: f32, f0: vec3f) -> vec3f {
  let probes = reflection_probes.counts.x > 0.5;
  let traced = sky_light.ssr.x > 0.5;
  if (sky_light.tint.w < 0.5 && !probes && !traced) {
    return vec3f(0.0);
  }
  let nv = max(dot(normal, view), 0.0001);
  let toward = reflect(-view, normal);
  let rough = clamp(roughness, 0.04, 1.0);
  var radiance = vec3f(0.0);
  if (sky_light.tint.w >= 0.5) {
    radiance = textureSampleLevel(sky_cube, sky_sampler, toward, rough * sky_light.reflection.x).rgb * sky_light.tint.rgb;
  }
  if (probes) {
    let local = local_reflection(world, toward, rough);
    radiance = radiance * (1.0 - local.a) + local.rgb;
  }
  if (traced) {
    let screen = screen_reflection(world, normal, view, rough);
    radiance = mix(radiance, screen.rgb, screen.a);
  }
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
  let reflecting = max(sky_light.tint.w, select(0.0, 1.0, reflection_probes.counts.x > 0.5 || sky_light.ssr.x > 0.5));
  let diffuse = albedo * (1.0 - clamp(metallic, 0.0, 1.0)) * (vec3f(1.0) - reflected * reflecting);
  return diffuse * ambient_light(world, normal) + sky_reflection(world, normal, view, roughness, f0);
}
