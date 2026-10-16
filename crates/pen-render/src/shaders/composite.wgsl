struct Placement {
  rect: vec4f,
  // The screen's size, then how much of the image across and down the inset
  // drew.
  screen: vec4f,
};
@group(0) @binding(0) var<uniform> placement: Placement;
@group(0) @binding(1) var image: texture_2d<f32>;
@group(0) @binding(2) var image_sampler: sampler;
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};
@vertex fn vertex(@builtin(vertex_index) vertex: u32) -> Corner {
  let corner = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0))[vertex];
  let pixel = placement.rect.xy + corner * placement.rect.zw;
  var output: Corner;
  output.position = vec4f(pixel.x / placement.screen.x * 2.0 - 1.0, 1.0 - pixel.y / placement.screen.y * 2.0, 0.0, 1.0);
  output.uv = corner * placement.screen.zw;
  return output;
}
${tonedWgsl}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let light = textureSample(image, image_sampler, corner.uv);
  if (light.a <= 0.0) {
    return vec4f(0.0);
  }
  return vec4f(toned(light.rgb / light.a) * light.a, light.a);
}
