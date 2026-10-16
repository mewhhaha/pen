struct Corner {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};
@vertex fn vertex(@builtin(vertex_index) index: u32) -> Corner {
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  var output: Corner;
  output.position = vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
  output.uv = vec2f(corner.x, 1.0 - corner.y);
  return output;
}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
fn tap(uv: vec2f, x: f32, y: f32) -> vec3f {
  let texel = 1.0 / vec2f(textureDimensions(source));
  return textureSampleLevel(source, source_sampler, uv + vec2f(x, y) * texel, 0.0).rgb;
}
