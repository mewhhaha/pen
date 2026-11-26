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

const TAPS: i32 = 32;

// Precomputed 32-tap Vogel spiral offsets: xy = (cos(i * GOLDEN_ANGLE) * u, sin(i * GOLDEN_ANGLE) * u), z = u.
const VOGEL_TAPS = array<vec3f, 32>(
  vec3f(0.1250000, 0.0000000, 0.1250000),
  vec3f(-0.1596450, 0.1462479, 0.2165064),
  vec3f(0.0244362, -0.2784383, 0.2795085),
  vec3f(0.2012222, 0.2624588, 0.3307189),
  vec3f(-0.3692676, -0.0653182, 0.3750000),
  vec3f(0.3498025, -0.2225157, 0.4145781),
  vec3f(-0.1170021, 0.4352419, 0.4506939),
  vec3f(-0.2231357, -0.4296341, 0.4841229),
  vec3f(0.4841151, 0.1767981, 0.5153882),
  vec3f(-0.5036411, 0.2078957, 0.5448624),
  vec3f(0.2427883, -0.5188245, 0.5728220),
  vec3f(0.1794144, 0.5720013, 0.5994789),
  vec3f(-0.5407570, -0.3133797, 0.6250000),
  vec3f(0.6343695, -0.1394644, 0.6495191),
  vec3f(-0.3871458, 0.5506751, 0.6731456),
  vec3f(-0.0894397, -0.6901996, 0.6959705),
  vec3f(0.5490718, 0.4627583, 0.7180703),
  vec3f(-0.7388785, 0.0305549, 0.7395100),
  vec3f(0.5389551, -0.5363323, 0.7603453),
  vec3f(-0.0360582, 0.7797915, 0.7806247),
  vec3f(-0.5128175, -0.6145268, 0.8003905),
  vec3f(0.8123596, 0.1093018, 0.8196798),
  vec3f(-0.6883106, 0.4789086, 0.8385255),
  vec3f(0.1880861, -0.8360614, 0.8569568),
  vec3f(0.4350333, 0.7591911, 0.8750000),
  vec3f(-0.8504484, -0.2713162, 0.8926786),
  vec3f(0.8261024, -0.3816803, 0.9100137),
  vec3f(-0.3578882, 0.8551556, 0.9270248),
  vec3f(-0.3194073, -0.8880338, 0.9437293),
  vec3f(0.8499086, 0.4466882, 0.9601432),
  vec3f(-0.9440346, 0.2488445, 0.9762812),
  vec3f(0.5365958, -0.8345298, 0.9921567),
);

fn depth_at_uv(uv: vec2f, dsize_f: vec2f, dsize_max: vec2i) -> f32 {
  let p = clamp(vec2i(uv * dsize_f), vec2i(0), dsize_max);
  return textureLoad(depth, p, 0);
}

fn linear_depth(uv: vec2f, d: f32) -> f32 {
  if (d >= 0.999999) {
    return 1e5;
  }
  let ndc_x = uv.x * 2.0 - 1.0;
  let ndc_y = 1.0 - uv.y * 2.0;
  let vz = dof.inverse_projection[0].z * ndc_x + dof.inverse_projection[1].z * ndc_y
         + dof.inverse_projection[2].z * d + dof.inverse_projection[3].z;
  let vw = dof.inverse_projection[0].w * ndc_x + dof.inverse_projection[1].w * ndc_y
         + dof.inverse_projection[2].w * d + dof.inverse_projection[3].w;
  return abs(vz / max(abs(vw), 1e-6));
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
  let dsize = vec2i(textureDimensions(depth));
  let dsize_f = vec2f(dsize);
  let dsize_max = dsize - vec2i(1);
  let center_z = linear_depth(uv, depth_at_uv(uv, dsize_f, dsize_max));
  let center_coc = signed_coc(center_z);
  let c_mag = abs(center_coc);
  let inv_aspect = 1.0 / max(dof.size.z, 1e-4);
  let feather = max(max_blur * 0.08, 0.5 / max(size.y, 1.0));
  // Small per-pixel spin breaks concentric ring structure without flickering.
  let spin = fract(52.9829189 * fract(dot(floor(position.xy), vec2f(0.06711056, 0.00583715)))) * 6.2831853;
  let spin_cs = vec2f(cos(spin), sin(spin));
  var accum = center_color.rgb;
  var total = 1.0;
  for (var i = 0; i < TAPS; i++) {
    let tap = VOGEL_TAPS[i];
    let r = tap.z * max_blur;
    let rot_x = tap.x * spin_cs.x - tap.y * spin_cs.y;
    let rot_y = tap.x * spin_cs.y + tap.y * spin_cs.x;
    let offset = vec2f(rot_x * inv_aspect, rot_y) * max_blur;
    let sample_uv = clamp(uv + offset, vec2f(0.0), vec2f(1.0));
    let sz = linear_depth(sample_uv, depth_at_uv(sample_uv, dsize_f, dsize_max));
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
