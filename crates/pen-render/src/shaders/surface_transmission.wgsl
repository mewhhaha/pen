// The light a transmissive standard surface lets through from behind it
// (KHR_materials_transmission and _volume): the opaque picture the frame keeps
// of what it has drawn so far, read where the view's ray leaves the surface.
// Add it after `SURFACE_LOBES_WGSL` and `SURFACE_WGSL` and after
// `TRANSLUCENCY_WGSL`, which supplies the four functions it calls and declares
// the picture's bindings (the frame's opaque colour and its mip chain):
//   opaque_available() -> bool
//   opaque_world_uv(world: vec3f) -> vec2f
//   opaque_behind(uv: vec2f, level: f32) -> vec3f
//   opaque_levels() -> f32
// then call `surface_transmit(s)` at the top of the fragment (it reads a
// texture, nothing else in it needs uniform control flow) and light the result
// with `standard_surface`: the diffuse lobe gives way to `behind` by the
// surface's transmission, scaled by what the base lobe's reflection leaves.
// Where there is no picture (shadow passes, probe captures) it shows the sky
// along the same ray instead.

// The sky along `direction`, blurred by `roughness`: what a transmissive
// surface shows where the frame has no picture of what is behind it.
fn surface_sky_behind(direction: vec3f, roughness: f32) -> vec3f {
  if (sky_light.tint.w < 0.5) {
    return vec3f(0.0);
  }
  return textureSampleLevel(sky_cube, sky_sampler, direction, clamp(roughness, 0.0, 1.0) * sky_light.reflection.x).rgb * sky_light.tint.rgb;
}

// `s` with `behind` set: the picture behind the surface where the view's ray
// leaves it, through the volume's attenuation. A thin wall (thickness 0)
// reads the pixel behind the fragment; a volume refracts the ray in at the
// surface and out `thickness` along. Without the lobe, a weight of 0, or at
// total internal reflection `behind` is left black.
fn surface_transmit(s: StandardSurface) -> StandardSurface {
  var r = s;
  if ((s.lobes & surface_transmission) == 0u || s.transmission <= 0.0) {
    return r;
  }
  var direction = -s.view;
  var exit = s.world;
  var attenuation = vec3f(1.0);
  if (s.thickness > 0.0) {
    direction = lobe_refract(s.view, s.normal, 1.0 / max(1.0, s.ior));
    if (dot(direction, direction) == 0.0) {
      return r;
    }
    exit = s.world + direction * s.thickness;
    attenuation = lobe_volume_attenuation(s.attenuation_color, s.thickness, s.attenuation_distance);
  }
  let levels = opaque_levels();
  let level = lobe_transmission_level(s.roughness, s.ior, levels, levels);
  if (opaque_available()) {
    r.behind = opaque_behind(opaque_world_uv(exit), level) * attenuation;
  } else {
    r.behind = surface_sky_behind(direction, s.roughness) * attenuation;
  }
  return r;
}
