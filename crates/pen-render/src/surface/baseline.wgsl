// Frozen copies of the functions the standard surface library replaced, as
// they stood in environment.wgsl and lights.wgsl at pen commit 8d5884d (the
// parent of the library), under `old_` names: the witness that the base lobe
// of the library is the same to the bit. Used by the tests only, and not to
// be edited.
// The light a surface at `world` facing `normal`, seen along `view` (both unit
// length, `view` toward the eye), of `roughness` (0 mirror to 1 matte) and
// with Fresnel reflectance `f0` face on, mirrors: what it reflects (a ray
// from it meets the screen's picture, the local reflection probes it is in
// the boxes of, or else the prefiltered sky) times the split-sum table's scale
// and bias. The picture takes the probes' and the sky's place by its
// confidence; the probes take the sky's by theirs. Zero where there is
// none of these.
fn old_sky_reflection(world: vec3f, normal: vec3f, view: vec3f, roughness: f32, f0: vec3f) -> vec3f {
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

// A surface's light from the sky, diffuse and glossy: `albedo`, `metallic`
// and `roughness` as `lighting.Material` has them, at `world` facing
// `normal`, seen along `view` (unit length, toward the eye). The diffuse part
// gives up what the reflection takes.
fn old_surface_ambient(world: vec3f, normal: vec3f, view: vec3f, albedo: vec3f, metallic: f32, roughness: f32) -> vec3f {
  let f0 = mix(vec3f(0.04), albedo, clamp(metallic, 0.0, 1.0));
  let nv = max(dot(normal, view), 0.0001);
  let split = sky_brdf(nv, clamp(roughness, 0.04, 1.0));
  let reflected = f0 * split.x + vec3f(split.y);
  let reflecting = max(sky_light.tint.w, select(0.0, 1.0, reflection_probes.counts.x > 0.5 || sky_light.ssr.x > 0.5));
  let diffuse = albedo * (1.0 - clamp(metallic, 0.0, 1.0)) * (vec3f(1.0) - reflected * reflecting);
  return diffuse * ambient_light(world, normal) + old_sky_reflection(world, normal, view, roughness, f0);
}

// What the frame's lights bring to a glossy surface, in two parts a material
// scales by its own: `diffuse` is what `clustered_light` sums, for albedo to
// multiply, and `specular` the same lights' GGX highlights, already
// Fresnel-weighted, to add as they are.
struct OldLightResponse {
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
fn old_cluster_light_response(light: ClusterLight, world: vec3f, normal: vec3f, view: vec3f, roughness: f32, f0: vec3f) -> OldLightResponse {
  var response = OldLightResponse(vec3f(0.0), vec3f(0.0));
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
  let nl = max(dot(normal, way), 0.0);
  if (nl <= 0.0) {
    return response;
  }
  let radiance = light.color.rgb * strength * local_shadow(light, world, normal, way, distance);
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
// `view` (see `old_cluster_light_response`); `position` and `view_z` as there.
fn old_clustered_surface(world: vec3f, normal: vec3f, view: vec3f, roughness: f32, f0: vec3f, position: vec4f, view_z: f32) -> OldLightResponse {
  var sum = OldLightResponse(vec3f(0.0), vec3f(0.0));
  if (cluster_grid.counts.w == 0u) {
    return sum;
  }
  let range = cluster_ranges[cluster_of(position, -view_z)];
  for (var index = range.x; index < range.x + range.y; index++) {
    let one = old_cluster_light_response(cluster_lights[cluster_indices[index]], world, normal, view, roughness, f0);
    sum.diffuse += one.diffuse;
    sum.specular += one.specular;
  }
  return sum;
}
