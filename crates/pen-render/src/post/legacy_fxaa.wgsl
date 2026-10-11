// FXAA after Timothy Lottes: blur along the edge through each pixel, unless
// that leaves the range of its neighbourhood.
fn fxaa(uv: vec2f) -> vec3f {
  let texel = present.source.xy;
  let north_west = luma(fetch(uv + vec2f(-1.0, -1.0) * texel));
  let north_east = luma(fetch(uv + vec2f(1.0, -1.0) * texel));
  let south_west = luma(fetch(uv + vec2f(-1.0, 1.0) * texel));
  let south_east = luma(fetch(uv + vec2f(1.0, 1.0) * texel));
  let middle = luma(fetch(uv));
  let lowest = min(middle, min(min(north_west, north_east), min(south_west, south_east)));
  let highest = max(middle, max(max(north_west, north_east), max(south_west, south_east)));
  var along = vec2f(
    (south_west + south_east) - (north_west + north_east),
    (north_west + south_west) - (north_east + south_east),
  );
  let reduce = max((north_west + north_east + south_west + south_east) * (0.25 / 8.0), 1.0 / 128.0);
  let shortest = 1.0 / (min(abs(along.x), abs(along.y)) + reduce);
  along = clamp(along * shortest, vec2f(-8.0), vec2f(8.0)) * texel;
  let near = 0.5 * (fetch(uv + along * (1.0 / 3.0 - 0.5)) + fetch(uv + along * (2.0 / 3.0 - 0.5)));
  let far = near * 0.5 + 0.25 * (fetch(uv - along * 0.5) + fetch(uv + along * 0.5));
  let spread = luma(far);
  if (spread < lowest || spread > highest) {
    return near;
  }
  return far;
}
