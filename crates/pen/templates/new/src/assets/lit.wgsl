// A material shader: what pen-render binds for every draw (group 0, the
// frame's scene and its sky light; group 1, the draw and its material) and the
// mesh's vertex layout. This one lights the material's color from the frame's
// sun and the sky's light all round (`ambient_light`: the game's probe volume
// where it has one, else its sky). The leading fields of each struct are
// fixed; a shader declares as many of them as it reads. The block between
// the BEGIN and END lines is pen-render's ENVIRONMENT_WGSL, which declares the
// sky's bindings; a test keeps it the same.
struct Scene {
  view: mat4x4f,
  projection: mat4x4f,
  ambient: vec4f,
  light_direction: vec4f,
  light_color: vec4f,
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
  let sun = max(dot(normal, -normalize(scene.light_direction.xyz)), 0.0);
  let light = ambient_light(surface.world, normal) + scene.light_color.rgb * sun;
  return vec4f(draw.color.rgb * light, draw.color.a);
}
