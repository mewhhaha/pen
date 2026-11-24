// Integrates each column of the grid front to back: the light scattered
// toward the eye and the optical depth up to every slice's far boundary.
// Layer 0 holds nothing (no light, no depth); layer k + 1, the sum through
// slice k. Between two boundaries the slice's light is the exact integral of
// a constant source against its own transmittance, so it conserves energy
// however thick the slice.
@group(0) @binding(0) var<uniform> volume: Volume;
@group(0) @binding(1) var injected: texture_3d<f32>;
@group(0) @binding(2) var integrated: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
  let across = u32(volume.grid.x);
  let down = u32(volume.grid.y);
  if (id.x >= across || id.y >= down) {
    return;
  }
  let uv = (vec2f(id.xy) + 0.5) / volume.grid.xy;
  // A step along the ray is longer than the depth it gains, by the ray's slant.
  let slant = length(volume_ray(uv));
  var light = vec3f(0.0);
  var transmittance = 1.0;
  var optical = 0.0;
  textureStore(integrated, vec3u(id.xy, 0u), vec4f(0.0));
  let slices = u32(volume.grid.z);
  var near = 0.0;
  for (var k = 0u; k < slices; k++) {
    let far = volume_depth(f32(k + 1u));
    let segment = (far - near) * slant;
    near = far;
    let source = textureLoad(injected, vec3u(id.xy, k), 0);
    let kept = exp(-source.a * segment);
    var gained = source.rgb * segment;
    if (source.a > 0.000001) {
      gained = source.rgb * ((1.0 - kept) / source.a);
    }
    light += transmittance * gained;
    transmittance *= kept;
    optical += source.a * segment;
    textureStore(integrated, vec3u(id.xy, k + 1u), vec4f(light, optical));
  }
}
