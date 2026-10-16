${bloomVertex}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let uv = corner.uv;
  let color = tap(uv, 0.0, 0.0) * 4.0 +
    (tap(uv, 0.0, -1.0) + tap(uv, -1.0, 0.0) + tap(uv, 1.0, 0.0) + tap(uv, 0.0, 1.0)) * 2.0 +
    tap(uv, -1.0, -1.0) + tap(uv, 1.0, -1.0) + tap(uv, -1.0, 1.0) + tap(uv, 1.0, 1.0);
  return vec4f(color / 16.0, 1.0);
}
