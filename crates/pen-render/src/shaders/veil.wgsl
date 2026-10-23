struct Veil {
  zenith: vec4f,
  horizon: vec4f,
  // x: seconds since it rose, y: how much of it is left, z: the screen's
  // aspect.
  state: vec4f,
};
@group(0) @binding(0) var<uniform> veil: Veil;
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};
@vertex fn vertex(@builtin(vertex_index) index: u32) -> Corner {
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return Corner(vec4f(corner * 2.0 - 1.0, 0.0, 1.0), vec2f(corner.x, 1.0 - corner.y));
}
${tonedWgsl}
// Slow contour currents and a faceted light. Everything is procedural: no
// loading-screen assets compete with the scene being loaded.
fn grain(p: vec2f) -> f32 {
  let q = fract(p * vec2f(123.34, 456.21));
  return fract((q.x + dot(q, q + 45.32)) * (q.y + dot(q, q + 45.32)));
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  // The application supplies its own startup sprites over a neutral backing.
  if veil.state.w > 0.0 { return vec4f(0.0, 0.0, 0.0, veil.state.y); }
  let time = veil.state.x;
  let p = (corner.uv - 0.5) * vec2f(veil.state.z, 1.0);
  let radius = length(p);
  let current = radius + 0.035 * sin(p.x * 9.0 + time * 0.28)
    * sin(p.y * 7.0 - time * 0.21) + 0.018 * sin(p.x * 19.0 + p.y * 13.0 + time * 0.18);
  let contour = abs(fract(current * 28.0 - time * 0.08) - 0.5);
  let line = 1.0 - smoothstep(0.018, 0.075, contour);
  let breath = 0.5 + 0.5 * sin(time * 0.7);
  let mist = exp(-radius * 3.5);
  var light = vec3f(0.003, 0.006, 0.012) + vec3f(0.015, 0.035, 0.052) * mist;
  light += vec3f(0.08, 0.17, 0.20) * line * mist * (0.3 + breath * 0.2);

  let orbit = vec2f(p.x, p.y * 1.65);
  let angle = atan2(orbit.y, orbit.x);
  let ring = 1.0 - smoothstep(0.001, 0.004, abs(length(orbit) - 0.22));
  let sweep = pow(0.5 + 0.5 * cos(angle - time * 0.8), 10.0);
  light += vec3f(1.1, 0.85, 0.48) * ring * (0.06 + sweep);

  let turn = 0.10 * sin(time * 0.35);
  let q = mat2x2f(cos(turn), sin(turn), -sin(turn), cos(turn)) * p;
  let diamond = abs(q.x) * 1.6 + abs(q.y) - 0.095;
  let core = 1.0 - smoothstep(-0.002, 0.002, diamond);
  let facet = select(0.38, 0.85, q.x + q.y * 0.55 > 0.0);
  let edge = exp(-abs(diamond) * 220.0);
  light += vec3f(0.66, 0.83, 0.87) * core * facet;
  light += vec3f(1.2, 1.05, 0.77) * edge * (0.55 + breath * 0.15);
  light += vec3f(0.04, 0.10, 0.14) * exp(-radius * 14.0);

  let drift = p * 60.0 + vec2f(time * 0.14, -time * 0.22);
  let cell = floor(drift);
  let star = 1.0 - smoothstep(0.025, 0.08, length(fract(drift) - 0.5));
  light += vec3f(0.45, 0.56, 0.60) * star * step(0.992, grain(cell)) * mist;
  let left = veil.state.y;
  return vec4f(toned(light) * left, left);
}
