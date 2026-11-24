@group(0) @binding(0) var<uniform> screen: vec4f;
struct SpriteData {
  rect: vec4f,
  uv: vec4f,
  color: vec4f,
  parameters: vec4f,
};
@group(0) @binding(1) var<storage, read> sprites: array<SpriteData>;
@group(0) @binding(2) var atlas: texture_2d<f32>;
@group(0) @binding(3) var atlas_sampler: sampler;

struct Out {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) local_uv: vec2f,
  @location(2) color: vec4f,
  @location(3) params: vec4f,
  @location(4) size: vec2f,
};

@vertex fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
  let corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
  );
  let s = sprites[i];
  let c = corners[v];
  let px = s.rect.xy + c * s.rect.zw;
  let ndc = vec2f(px.x / max(screen.x, 1.0) * 2.0 - 1.0, 1.0 - px.y / max(screen.y, 1.0) * 2.0);
  let uv = mix(s.uv.xy, s.uv.zw, c);
  return Out(vec4f(ndc, 0.0, 1.0), uv, c, s.color, s.parameters, s.rect.zw);
}

@fragment fn fragment(in: Out) -> @location(0) vec4f {
  if (in.params.x > 0.5) {
    let sample = textureSample(atlas, atlas_sampler, in.uv);
    let alpha = step(0.35, sample.a) * in.color.a;
    if (alpha <= 0.01) {
      discard;
    }
    return vec4f(in.color.rgb * sample.rgb, alpha);
  }
  let half_size = in.size * 0.5;
  let p = (in.local_uv - vec2f(0.5)) * in.size;
  let radius = min(in.params.y, min(half_size.x, half_size.y));
  let q = abs(p) - (half_size - vec2f(radius));
  let dist = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - radius;
  let edge = clamp(0.5 - dist, 0.0, 1.0);
  if (edge <= 0.01) {
    discard;
  }
  let border = smoothstep(-2.0, -0.5, dist) * in.params.z;
  let rgb = mix(in.color.rgb, min(in.color.rgb * 1.45 + vec3f(0.18), vec3f(1.0)), border);
  return vec4f(rgb, in.color.a * edge);
}
