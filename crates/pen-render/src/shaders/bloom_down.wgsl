${bloomVertex}
${exposure}
fn brightness(color: vec3f) -> f32 {
  return max(color.r, max(color.g, color.b));
}

// A group of four taps, averaged; first, each weighted the less the brighter
// it is (Karis), so one blazing pixel cannot outshine its neighbours.
fn weight(color: vec3f) -> f32 {
  return ${weight};
}
fn group(a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f {
  let wa = weight(a);
  let wb = weight(b);
  let wc = weight(c);
  let wd = weight(d);
  return (a * wa + b * wb + c * wc + d * wd) / (wa + wb + wc + wd);
}

@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let uv = corner.uv;
  let a = tap(uv, -2.0, -2.0); let b = tap(uv, 0.0, -2.0); let c = tap(uv, 2.0, -2.0);
  let d = tap(uv, -2.0, 0.0); let e = tap(uv, 0.0, 0.0); let f = tap(uv, 2.0, 0.0);
  let g = tap(uv, -2.0, 2.0); let h = tap(uv, 0.0, 2.0); let i = tap(uv, 2.0, 2.0);
  let j = tap(uv, -1.0, -1.0); let k = tap(uv, 1.0, -1.0);
  let l = tap(uv, -1.0, 1.0); let m = tap(uv, 1.0, 1.0);
  var color = group(j, k, l, m) * 0.5 +
    (group(a, b, d, e) + group(b, c, e, f) + group(d, e, g, h) + group(e, f, h, i)) * 0.125;
  return vec4f(color, 1.0);
}
