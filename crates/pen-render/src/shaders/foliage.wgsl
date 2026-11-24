// GPU-Instanced Wind-Swaying Foliage Pipeline:
// - Procedural tapered 4-segment grass/foliage blades pinned at root (h=0)
// - Dual-frequency spatial wind gust wave + flutter with arc-length droop
// - Accurate finite-difference outward normal (`cross(right, dP/dh)`), two-sided flip, and backlit SSS.

struct FoliageUniform {
  view_projection: mat4x4f,
  camera_pos_and_time: vec4f,     // xyz = camera world pos, w = time (s)
  sun_dir_and_wind_str: vec4f,    // xyz = toward sun, w = wind strength
  sun_color_and_gust: vec4f,      // rgb = sun radiance, w = gust frequency
  wind_dir_and_speed: vec4f,      // xz = wind dir, y = flutter amount, w = wind speed
  root_color_and_width: vec4f,    // rgb = base albedo, w = base half-width
  tip_color_and_height: vec4f,    // rgb = tip albedo, w = base blade height
  sss_color_and_power: vec4f,     // rgb = backlit translucency color, w = translucency power
};

struct FoliageInstance {
  pos_and_scale: vec4f,   // xyz = root world pos, w = height scale
  rot_phase_tint: vec4f,  // x = yaw radians, y = wind phase offset, z = lean bias, w = tint factor
};

@group(0) @binding(0) var<uniform> u: FoliageUniform;
@group(0) @binding(1) var<storage, read> instances: array<FoliageInstance>;

const PI: f32 = 3.14159265359;
const SEGMENTS: u32 = 4u;

struct VsOut {
  @builtin(position) clip_pos: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) blade_h: f32,
  @location(3) tint: f32,
};

fn blade_spine(
  root: vec3f,
  facing: vec3f,
  wind_vec: vec3f,
  gust: f32,
  gust_phase: f32,
  lean: f32,
  blade_h: f32,
  h: f32,
) -> vec3f {
  let flutter = sin(gust_phase * 3.7 + h * 2.5) * u.wind_dir_and_speed.y;
  let bend_mag = u.sun_dir_and_wind_str.w * (gust * h * h + flutter * h * h * h);
  let curve = facing * (lean * h * h * blade_h) + wind_vec * (bend_mag * blade_h);
  let horiz_len = length(curve.xz);
  let droop = min(horiz_len * horiz_len / max(2.0 * blade_h, 0.1), blade_h * 0.45);
  return root + vec3f(curve.x, h * blade_h - droop, curve.z);
}

@vertex fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
  let inst = instances[iid];
  let root = inst.pos_and_scale.xyz;
  let scale = max(inst.pos_and_scale.w, 0.1);
  let yaw = inst.rot_phase_tint.x;
  let phase = inst.rot_phase_tint.y;
  let lean = inst.rot_phase_tint.z;

  var seg: u32 = 0u;
  var side: f32 = 0.0;
  if (vid < 18u) {
    let q = vid / 6u;
    let c = vid % 6u;
    if (c == 0u) { seg = q; side = -1.0; }
    else if (c == 1u) { seg = q; side = 1.0; }
    else if (c == 2u) { seg = q + 1u; side = -1.0; }
    else if (c == 3u) { seg = q; side = 1.0; }
    else if (c == 4u) { seg = q + 1u; side = 1.0; }
    else { seg = q + 1u; side = -1.0; }
  } else {
    let c = vid - 18u;
    if (c == 0u) { seg = 3u; side = -1.0; }
    else if (c == 1u) { seg = 3u; side = 1.0; }
    else { seg = 4u; side = 0.0; }
  }

  let h = f32(seg) / f32(SEGMENTS);
  let blade_h = u.tip_color_and_height.w * scale;
  let half_w = u.root_color_and_width.w * scale * (1.0 - h * 0.92);

  let right = vec3f(cos(yaw), 0.0, -sin(yaw));
  let facing = vec3f(sin(yaw), 0.0, cos(yaw));

  // Guard zero wind vector or zero wind strength to avoid normalize(0) NaN.
  let wind_raw = vec2f(u.wind_dir_and_speed.x, u.wind_dir_and_speed.z);
  let wind_len = length(wind_raw);
  let has_wind = wind_len > 1e-5 && abs(u.sun_dir_and_wind_str.w) > 1e-6;
  let wind_dir = select(vec2f(1.0, 0.0), wind_raw / wind_len, has_wind);
  let wind_vec = select(vec3f(0.0), vec3f(wind_dir.x, 0.0, wind_dir.y), has_wind);
  let t = u.camera_pos_and_time.w;
  let gust_phase = dot(root.xz, wind_dir) * u.sun_color_and_gust.w - t * u.wind_dir_and_speed.w + phase;
  let gust = 0.65 + 0.35 * sin(gust_phase);

  let spine = blade_spine(root, facing, wind_vec, gust, gust_phase, lean, blade_h, h);
  let world_pos = spine + right * (side * half_w);

  // Finite-difference tangent along exact displaced spine; cross(right, dP/dh) matches CCW front winding (+Z at yaw=0).
  let h0 = max(h - 0.01, 0.0);
  let h1 = min(h + 0.01, 1.0);
  let p0 = blade_spine(root, facing, wind_vec, gust, gust_phase, lean, blade_h, h0);
  let p1 = blade_spine(root, facing, wind_vec, gust, gust_phase, lean, blade_h, h1);
  let tangent_up = normalize(p1 - p0);
  let normal = normalize(cross(right, tangent_up));

  return VsOut(u.view_projection * vec4f(world_pos, 1.0), world_pos, normal, h, inst.rot_phase_tint.w);
}

@fragment fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  var n = normalize(in.normal);
  if (!front) {
    n = -n;
  }
  let v = normalize(u.camera_pos_and_time.xyz - in.world_pos);
  let sun_raw = u.sun_dir_and_wind_str.xyz;
  let l = select(vec3f(0.0, 1.0, 0.0), normalize(sun_raw), length(sun_raw) > 1e-6);

  let albedo = mix(u.root_color_and_width.rgb, u.tip_color_and_height.rgb, in.blade_h) * (0.85 + 0.3 * in.tint);
  let ao = mix(0.45, 1.0, in.blade_h);

  let ndotl = max(dot(n, l), 0.0);
  let wrap = clamp((dot(n, l) + 0.25) / 1.25, 0.0, 1.0);
  let diffuse = albedo * (wrap / PI) * u.sun_color_and_gust.rgb;

  // Backlit subsurface translucency gated by back-face solar incidence so light actually passes through the blade.
  let back_lit = smoothstep(-0.15, 0.35, dot(-n, l));
  let trans_dir = normalize(-l + n * 0.35);
  let back_dot = max(dot(v, trans_dir), 0.0);
  let sss = u.sss_color_and_power.rgb
    * pow(back_dot, max(u.sss_color_and_power.w, 1.0))
    * back_lit
    * (0.25 + 0.75 * in.blade_h)
    * u.sun_color_and_gust.rgb;

  let h_vec = normalize(l + v);
  let ndoth = max(dot(n, h_vec), 0.0);
  let spec = pow(ndoth, 24.0) * 0.08 * ndotl * u.sun_color_and_gust.rgb;

  let ambient = albedo * vec3f(0.16, 0.22, 0.14) * ao;
  return vec4f(diffuse * ao + sss + spec + ambient, 1.0);
}
