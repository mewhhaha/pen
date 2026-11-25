// Conservative farthest-depth base level of the Hi-Z occlusion pyramid.
// Reads the opaque scene depth buffer (1x or MSAA) and writes the farthest
// depth per pixel into mip 0 of the R32Float pyramid (max for Forward-Z, min for Reversed-Z).
@group(0) @binding(0) var depth_tex: ${depth_texture};
@group(0) @binding(1) var out_mip: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(out_mip);
  if (id.x >= size.x || id.y >= size.y) {
    return;
  }
  let coord = vec2i(id.xy);
  var farthest = ${init_depth};
  for (var s = 0; s < ${samples}; s++) {
    farthest = ${far_op}(farthest, textureLoad(depth_tex, coord, s));
  }
  textureStore(out_mip, coord, vec4f(farthest, 0.0, 0.0, 1.0));
}
