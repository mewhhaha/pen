// The nearest depth buffer value of each pixel of the opaque scene over its
// samples, 1 where nothing opaque is: translucent surfaces read it (`opaque_depth_at`).
@group(0) @binding(0) var scene_depth: ${depth_texture};
${fullscreen}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(position.xy);
  var nearest = 1.0;
  for (var sample = 0; sample < ${samples}; sample++) {
    nearest = min(nearest, textureLoad(scene_depth, pixel, sample));
  }
  return vec4f(nearest, 0.0, 0.0, 1.0);
}
