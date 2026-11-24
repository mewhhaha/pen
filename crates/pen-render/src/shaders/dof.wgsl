// Bokeh depth of field (dof.rs): 32-tap Vogel spiral disk gather in linear HDR
// light with signed thin-lens Circle of Confusion and depth-aware foreground/
// background overlap weighting so sharp foreground silhouettes stay crisp while
// out-of-focus foreground objects bleed naturally over the background.
struct Dof {
  inverse_projection: mat4x4f,
  // x: focus_distance, y: focus_range, z: aperture, w: max_blur
  params: vec4f,
  // x: width, y: height, z: aspect (width / height), w: 0
  size: vec4f,
};
@group(0) @binding(0) var<uniform> dof: Dof;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var depth: ${depth_texture};
@group(0) @binding(3) var filtering: sampler;
${fullscreen}

const GOLDEN_ANGLE: f32 = 2.39996323;
const TAPS: i32 = 32;

fn depth_at_uv(uv: vec2f) -> f32 {
  let dsize = vec2i(textureDimensions(depth));
  let p = clamp(vec2i(uv * vec2f(dsize)), vec2i(0), dsize - 1);
  return textureLoad(depth, p, 0);
}

fn linear_depth(uv: vec2f, d: f32) -> f32 {
  if (d >= 0.999999) {
    return 1e5;
  }
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let view = dof.inverse_projection * vec4f(ndc, d, 1.0);
  return abs(view.z / max(abs(view.w), 1e-6));
}

fn signed_coc(z: f32) -> f32 {
  let focus = max(dof.params.x, 1e-4);
  let half_range = dof.params.y * 0.5;
  let aperture = dof.params.z;
  let max_blur = dof.params.w;
  let delta = max(abs(z - focus) - half_range, 0.0);
  let mag = clamp(delta / max(z, 1e-3) * aperture, 0.0, max_blur);
  return select(-mag, mag, z < focus);
}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let size = dof.size.xy;
  let uv = position.xy / size;
  let pixel = vec2i(position.xy);
  let center_color = textureLoad(source, clamp(pixel, vec2i(0), vec2i(size) - 1), 0);
  let max_blur = dof.params.w;
  if (max_blur <= 1e-5 || dof.params.z <= 1e-5 || dof.params.x <= 1e-5) {
    return center_color;
  }
  let center_z = linear_depth(uv, depth_at_uv(uv));
  let center_coc = signed_coc(center_z);
  let c_mag = abs(center_coc);
  let aspect = max(dof.size.z, 1e-4);
  let feather = max(max_blur * 0.08, 0.5 / max(size.y, 1.0));
  // Small per-pixel spin breaks concentric ring structure without flickering.
  let spin = fract(52.9829189 * fract(dot(floor(position.xy), vec2f(0.06711056, 0.00583715)))) * 6.2831853;
  var accum = center_color.rgb;
  var total = 1.0;
  for (var i = 0; i < TAPS; i++) {
    let u = sqrt((f32(i) + 0.5) / f32(TAPS));
    let r = u * max_blur;
    let angle = f32(i) * GOLDEN_ANGLE + spin;
    let offset = vec2f(cos(angle) / aspect, sin(angle)) * r;
    let sample_uv = clamp(uv + offset, vec2f(0.0), vec2f(1.0));
    let sz = linear_depth(sample_uv, depth_at_uv(sample_uv));
    let sc = signed_coc(sz);
    let s_mag = abs(sc);
    // A tap contributes if its CoC disk reaches the center pixel AND either
    // it lies in front of / at the center's depth, or the center itself is
    // blurred out to at least this tap's radius.
    let fg = sz <= center_z + 0.02 * max(center_z, 1.0);
    let overlap = smoothstep(r - feather, r + feather, s_mag);
    let bg_ok = smoothstep(r - feather, r + feather, c_mag);
    let w = select(overlap * bg_ok, overlap, fg);
    if (w > 1e-4) {
      let col = textureSampleLevel(source, filtering, sample_uv, 0.0).rgb;
      accum += col * w;
      total += w;
    }
  }
  return vec4f(accum / total, center_color.a);
}
