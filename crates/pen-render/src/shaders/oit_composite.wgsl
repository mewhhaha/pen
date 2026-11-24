// McGuire & Bavoil (2013) Weighted Blended Order-Independent Transparency composite pass.
// Reads `accum` (Rgba16Float: sum(premultiplied_rgb * w, alpha * w)) and
// `reveal` (R8Unorm: prod(1 - alpha)) and composites onto the opaque HDR scene target.

@group(0) @binding(0) var accum_tex: texture_2d<f32>;
@group(0) @binding(1) var reveal_tex: texture_2d<f32>;

@vertex fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
  let y = 1.0 - f32(idx & 2u) * 2.0;
  return vec4f(x, y, 0.0, 1.0);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = vec2i(pos.xy);
  let reveal = textureLoad(reveal_tex, p, 0).r;
  if (reveal >= 0.9999) {
    discard;
  }
  let accum = textureLoad(accum_tex, p, 0);
  let safe_rgb = clamp(accum.rgb, vec3f(0.0), vec3f(65000.0));
  let average = safe_rgb / max(accum.a, 1e-5);
  return vec4f(average, 1.0 - reveal);
}
