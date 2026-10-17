// 1 wherever the stencil test passes: the scene's stencil marks, copied into
// a texture the present pass reads (downlevel.rs).
${fullscreen}

@fragment fn fragment() -> @location(0) u32 {
  return 1u;
}
