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
    return diffuse * ambient_light(s.world, s.normal)
      + sky_reflection_lobe(s.world, s.normal, s.view, dot(s.normal, s.view), s.roughness, s.f0, s.f90);
  }
  var irradiance = select(ambient_light(s.world, s.normal), s.lightmap_irradiance, lightmapped);
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
  var reflection = sky_reflection_lobe(s.world, lookup, s.view, dot(s.normal, s.view), s.roughness, eff_f0, s.f90);
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
