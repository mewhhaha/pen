// The froxel volume's shared declarations: its uniform and the grid's depth
// slicing, which the injection, the integration and the atmosphere pass
// that applies the volume all agree on. The includer declares `volume`.
struct Volume {
  // The world into the camera's view, and last frame's world into its clip
  // space (both unjittered).
  view: mat4x4f,
  previous: mat4x4f,
  // x, y: the projection's scale across and down; z, w: its off-centre shift.
  projection: vec4f,
  // x, y, z: froxels across, down and slices deep; w: how far ahead the grid reaches.
  grid: vec4f,
  // x: the slicing's knee, y: ln(1 + reach / knee); z: 1 when the history is
  // good to read, w: its share of the blend.
  slicing: vec4f,
  // This frame's sample offsets within a froxel, -0.5 to 0.5, and its number.
  jitter: vec4f,
  // rgb: the global mist's albedo; w: its anisotropy g.
  medium: vec4f,
  // x: the sun's scale, y: the ambient's, z: how far the ambient colour is the
  // sky's own in the direction looked, w: the width of the mist's soft edge.
  scatter: vec4f,
  // x: the noise's amount, y: its coarsest cell, z: its octaves, w: how many
  // local volumes follow.
  noise: vec4f,
  // xyz: how far the noise was carried; w: the sun's filter (0 one tap, 1 PCF).
  drift: vec4f,
  // rgb: the sky's mean light; w: 1 while the volume is drawn this frame.
  ambient: vec4f,
  // What `fog_apply` (translucency.wgsl) takes beyond the above to fog a surface at
  // its own depth the way the atmosphere pass fogs the opaque ones; the volume's own
  // passes and the atmosphere pass ignore it. The scene's view-projection, jittered
  // as the draws are.
  clip: mat4x4f,
  // Elements 10, 14, 11 and 15 of the scene's projection: a depth buffer value's
  // distance ahead of the eye.
  depth: vec4f,
  // x: 1 in a translucent pass, which reads the opaque picture; y: 1 when
  // `fog_apply` fogs (a mist, a volume or clouds in the frame).
  flags: vec4f,
  // xyz: the eye; w: 1 with a sky.
  camera: vec4f,
  // The sky's colours (zenith.w: its stars' brightness), the sun's direction, its
  // light and the clear colour, as the atmosphere pass has them.
  zenith: vec4f,
  horizon: vec4f,
  ground: vec4f,
  sun: vec4f,
  sun_color: vec4f,
  clear: vec4f,
  // x: the mist's radius, y: its density, z: its scale height.
  fog: vec4f,
  // x: how much of the sky the clouds cover; yz: how far the wind carried them.
  clouds: vec4f,
};

// A boundary of the grid, in slices from the eye (0 to the slice count), as a
// depth ahead of the eye: exponential past the knee, linear within it.
fn volume_depth(boundary: f32) -> f32 {
  let knee = volume.slicing.x;
  return knee * (exp(boundary / volume.grid.z * volume.slicing.y) - 1.0);
}

// The boundary coordinate of a depth ahead of the eye, `volume_depth`'s inverse.
fn volume_boundary(depth: f32) -> f32 {
  return volume.grid.z * log(1.0 + depth / volume.slicing.x) / volume.slicing.y;
}

// The view-space ray, with z of -1, through a point of the picture (0 to 1,
// y down): its point at depth d ahead of the eye is `ray * d`.
fn volume_ray(uv: vec2f) -> vec3f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  return vec3f((ndc.x + volume.projection.z) / volume.projection.x, (ndc.y + volume.projection.w) / volume.projection.y, -1.0);
}

// pi times the Henyey-Greenstein phase function: `c` is the cosine of the angle
// between the direction toward the light and the direction the eye looks along.
fn phase_scaled(g: f32, c: f32) -> f32 {
  let k = clamp(g, -0.95, 0.95);
  let d = max(1.0 + k * k - 2.0 * k * c, 0.0001);
  return (1.0 - k * k) / (4.0 * d * sqrt(d));
}
