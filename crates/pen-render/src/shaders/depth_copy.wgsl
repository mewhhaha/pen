// The scene's depth, copied into a float texture for the passes that read
// it (downlevel.rs): each texel's middle, through a nearest sampler, which
// GLSL ES 3.0 must have to read a depth texture at all.
@group(0) @binding(0) var depth: texture_2d<f32>;
@group(0) @binding(1) var nearest: sampler;
${fullscreen}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let uv = position.xy / vec2f(textureDimensions(depth));
  return vec4f(textureSampleLevel(depth, nearest, uv, 0.0).x, 0.0, 0.0, 1.0);
}
