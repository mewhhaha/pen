// One level of the opaque picture's chain from the one above it: each texel is
// the (1, 3, 3, 1) / 8 tent over the 4 x 4 texels round it, in four bilinear
// taps three quarters of a texel out from its centre, so a texel of the level
// below it spills into its neighbours' and a surface seen through rough glass
// blurs without blocks. The edges repeat.
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};
@vertex fn vertex(@builtin(vertex_index) index: u32) -> Corner {
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return Corner(vec4f(corner * 2.0 - 1.0, 0.0, 1.0), vec2f(corner.x, 1.0 - corner.y));
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let reach = 0.75 / vec2f(textureDimensions(source));
  var sum = textureSampleLevel(source, source_sampler, corner.uv + vec2f(-reach.x, -reach.y), 0.0);
  sum += textureSampleLevel(source, source_sampler, corner.uv + vec2f(reach.x, -reach.y), 0.0);
  sum += textureSampleLevel(source, source_sampler, corner.uv + vec2f(-reach.x, reach.y), 0.0);
  sum += textureSampleLevel(source, source_sampler, corner.uv + vec2f(reach.x, reach.y), 0.0);
  return sum * 0.25;
}
