// Gather's one sprite shader: a rectangle of the sprite's color, its corners
// rounded by parameters.x (0 square, 1 as round as it can be), with a soft
// pixel of edge. The bindings are every sprite shader's (pen-render's
// sprites.rs); the texture goes unused.
struct Screen {
  size: vec4f,
};
struct Sprite {
  rect: vec4f,
  uv: vec4f,
  color: vec4f,
  parameters: vec4f,
};
@group(0) @binding(0) var<uniform> screen: Screen;
@group(0) @binding(1) var<storage, read> sprites: array<Sprite>;
@group(0) @binding(2) var image: texture_2d<f32>;
@group(0) @binding(3) var image_sampler: sampler;

struct Corner {
  @builtin(position) position: vec4f,
  // Pixels from the rectangle's middle.
  @location(0) local: vec2f,
  @location(1) color: vec4f,
  // Half its size, and its corners' radius, in pixels.
  @location(2) @interpolate(flat) shape: vec3f,
};

@vertex fn vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> Corner {
  let sprite = sprites[index];
  let corner = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0))[vertex];
  let pixel = sprite.rect.xy + corner * sprite.rect.zw;
  let half = sprite.rect.zw * 0.5;
  var output: Corner;
  output.position = vec4f(pixel.x / screen.size.x * 2.0 - 1.0, 1.0 - pixel.y / screen.size.y * 2.0, 0.0, 1.0);
  output.local = (corner - 0.5) * sprite.rect.zw;
  output.color = sprite.color;
  output.shape = vec3f(half, clamp(sprite.parameters.x, 0.0, 1.0) * min(half.x, half.y));
  return output;
}

@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let inside = abs(corner.local) - (corner.shape.xy - corner.shape.z);
  let distance = length(max(inside, vec2f(0.0))) + min(max(inside.x, inside.y), 0.0) - corner.shape.z;
  let coverage = clamp(0.5 - distance, 0.0, 1.0) * corner.color.a;
  // Premultiplied, as sprites blend.
  return vec4f(corner.color.rgb * coverage, coverage);
}
