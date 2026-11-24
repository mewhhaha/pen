// The opaque picture, misted, written to every sample of the translucent pass's
// target before the translucent surfaces draw over it.
@group(0) @binding(0) var opaque_picture: texture_2d<f32>;
${fullscreen}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  return textureLoad(opaque_picture, vec2i(position.xy), 0);
}
