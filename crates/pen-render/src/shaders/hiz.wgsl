// Conservative farthest-depth reduction for the Hi-Z occlusion pyramid.
// Downsamples mip k-1 into mip k taking the farthest depth over the covering
// footprint (max for Forward-Z, min for Reversed-Z), including extra border
// texels whenever the source dimension is odd so no occluder gap is missed.
@group(0) @binding(0) var src_mip: texture_2d<f32>;
@group(0) @binding(1) var out_mip: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dst_size = textureDimensions(out_mip);
  if (id.x >= dst_size.x || id.y >= dst_size.y) {
    return;
  }
  let src_size = vec2i(textureDimensions(src_mip, 0));
  let max_c = src_size - vec2i(1);
  let base = vec2i(id.xy) * 2;

  var d = ${far_op}(
    ${far_op}(
      textureLoad(src_mip, min(base, max_c), 0).r,
      textureLoad(src_mip, min(base + vec2i(1, 0), max_c), 0).r,
    ),
    ${far_op}(
      textureLoad(src_mip, min(base + vec2i(0, 1), max_c), 0).r,
      textureLoad(src_mip, min(base + vec2i(1, 1), max_c), 0).r,
    ),
  );

  let odd_x = src_size.x > 1 && (src_size.x & 1) != 0;
  let odd_y = src_size.y > 1 && (src_size.y & 1) != 0;
  if (odd_x) {
    d = ${far_op}(d, ${far_op}(
      textureLoad(src_mip, min(base + vec2i(2, 0), max_c), 0).r,
      textureLoad(src_mip, min(base + vec2i(2, 1), max_c), 0).r,
    ));
  }
  if (odd_y) {
    d = ${far_op}(d, ${far_op}(
      textureLoad(src_mip, min(base + vec2i(0, 2), max_c), 0).r,
      textureLoad(src_mip, min(base + vec2i(1, 2), max_c), 0).r,
    ));
  }
  if (odd_x && odd_y) {
    d = ${far_op}(d, textureLoad(src_mip, min(base + vec2i(2, 2), max_c), 0).r);
  }

  textureStore(out_mip, vec2i(id.xy), vec4f(d, 0.0, 0.0, 1.0));
}
