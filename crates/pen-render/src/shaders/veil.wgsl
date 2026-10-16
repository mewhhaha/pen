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
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  var light = mix(veil.horizon.rgb, veil.zenith.rgb, smoothstep(0.0, 1.0, 1.0 - corner.uv.y));
  let p = (corner.uv - 0.5) * vec2f(veil.state.z, 1.0);
  let ring = 1.0 - smoothstep(0.0, 0.005, abs(length(p) - 0.035));
  let tail = fract(atan2(p.y, p.x) / 6.2831853 - veil.state.x * 0.8);
  light += vec3f(1.0, 0.94, 0.82) * ring * tail * tail * 1.4;
  let left = veil.state.y;
  return vec4f(toned(light) * left, left);
}
