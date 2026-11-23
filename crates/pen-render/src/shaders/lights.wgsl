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
};
@group(2) @binding(0) var<uniform> cluster_grid: ClusterGrid;
@group(2) @binding(1) var<storage, read> cluster_lights: array<ClusterLight>;
@group(2) @binding(2) var<storage, read> cluster_ranges: array<vec2u>;
@group(2) @binding(3) var<storage, read> cluster_indices: array<u32>;

// How much of `light` reaches `world` on a surface facing `normal`: its color
// times the cosine of its angle to the surface, falling smoothly to nothing
// at its reach (and, for a spot, outside its cone).
fn cluster_light(light: ClusterLight, world: vec3f, normal: vec3f) -> vec3f {
  let toward = light.position.xyz - world;
  let distance = length(toward);
  let reach = light.position.w;
  if (distance >= reach) {
    return vec3f(0.0);
  }
  let way = toward / max(distance, 0.0001);
  let ratio = distance / reach;
  let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
  var strength = window * window / (distance * distance + 1.0);
  if (light.color.w > -1.0) {
    strength *= smoothstep(light.color.w, max(light.direction.w, light.color.w + 0.0001), dot(-way, light.direction.xyz));
  }
  return light.color.rgb * strength * max(dot(normal, way), 0.0);
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

// What the frame's lights bring to a glossy surface, in two parts a material
// scales by its own: `diffuse` is what `clustered_light` sums, for albedo to
// multiply, and `specular` the same lights' GGX highlights, already
// Fresnel-weighted, to add as they are.
struct LightResponse {
  diffuse: vec3f,
  specular: vec3f,
};

// One light's response at `world` for a surface facing `normal`, seen along
// `view` (unit vectors, `view` toward the eye), of perceptual `roughness`
// (0 to 1) and face-on Fresnel reflectance `f0`. The lobe is Ink's
// `lighting.brdf`: GGX distribution with alpha = roughness squared,
// height-correlated Smith visibility and Schlick Fresnel. Lights here are
// scaled so that a diffuse surface is albedo times their color times the
// cosine, with no division by pi, so the specular lobe is multiplied by pi.
fn cluster_light_response(light: ClusterLight, world: vec3f, normal: vec3f, view: vec3f, roughness: f32, f0: vec3f) -> LightResponse {
  var response = LightResponse(vec3f(0.0), vec3f(0.0));
  let toward = light.position.xyz - world;
  let distance = length(toward);
  let reach = light.position.w;
  if (distance >= reach) {
    return response;
  }
  let way = toward / max(distance, 0.0001);
  let ratio = distance / reach;
  let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
  var strength = window * window / (distance * distance + 1.0);
  if (light.color.w > -1.0) {
    strength *= smoothstep(light.color.w, max(light.direction.w, light.color.w + 0.0001), dot(-way, light.direction.xyz));
  }
  let radiance = light.color.rgb * strength;
  let nl = max(dot(normal, way), 0.0);
  response.diffuse = radiance * nl;
  let facing = dot(normal, view);
  if (nl <= 0.0 || facing <= 0.0) {
    return response;
  }
  let nv = max(facing, 0.00001);
  let half_way = normalize(view + way);
  let nh = clamp(dot(normal, half_way), 0.0, 1.0);
  let vh = clamp(dot(view, half_way), 0.0, 1.0);
  let rough = clamp(roughness, 0.04, 1.0);
  let a2 = rough * rough * rough * rough;
  let denominator = nh * nh * (a2 - 1.0) + 1.0;
  let distribution = a2 / (3.14159265359 * denominator * denominator);
  let visibility = 0.5 / max(0.000001, nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2));
  let fade = 1.0 - vh;
  let fresnel = f0 + (vec3f(1.0) - f0) * (fade * fade * fade * fade * fade);
  response.specular = radiance * (fresnel * (distribution * visibility * nl * 3.14159265359));
  return response;
}

// The frame's point and spot lights at `world`: `clustered_light`'s diffuse
// sum and their GGX specular, for a surface of `roughness` and `f0` seen along
// `view` (see `cluster_light_response`); `position` and `view_z` as there.
fn clustered_surface(world: vec3f, normal: vec3f, view: vec3f, roughness: f32, f0: vec3f, position: vec4f, view_z: f32) -> LightResponse {
  var sum = LightResponse(vec3f(0.0), vec3f(0.0));
  if (cluster_grid.counts.w == 0u) {
    return sum;
  }
  let range = cluster_ranges[cluster_of(position, -view_z)];
  for (var index = range.x; index < range.x + range.y; index++) {
    let one = cluster_light_response(cluster_lights[cluster_indices[index]], world, normal, view, roughness, f0);
    sum.diffuse += one.diffuse;
    sum.specular += one.specular;
  }
  return sum;
}
