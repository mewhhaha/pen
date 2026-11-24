// Analytical 4-Wave Gerstner Water Pipeline:
// - Analytical vertex displacement, tangent/binormal/normal, and crest Jacobian foam
// - Dry-cell discard + oblique refracted ray intersection for Beer-Lambert depth absorption
// - Dual-layer animated underwater caustics + Schlick Fresnel + full GGX Cook-Torrance sun glitter.

struct WaterUniform {
  view_projection: mat4x4f,
  camera_pos_and_time: vec4f,    // xyz = camera pos, w = time (seconds)
  sun_dir_and_level: vec4f,      // xyz = toward sun, w = water surface base y
  sun_color_and_bed: vec4f,      // rgb = sun radiance, w = flat bed y at x=0
  absorption_and_grid: vec4f,    // rgb = Beer-Lambert extinction per meter, w = grid cells
  deep_and_foam: vec4f,          // rgb = deep scatter tint, w = foam threshold
  sky_horizon: vec4f,            // rgb = horizon sky reflection, w = surface extent (half-size)
  sky_zenith: vec4f,             // rgb = zenith sky reflection, w = caustic intensity
  bed_color_and_slope: vec4f,    // rgb = submerged seabed albedo, w = bed slope along +X
  waves: array<vec4f, 4>,        // xy = dir, z = steepness [0..0.24], w = wavelength (m)
};

@group(0) @binding(0) var<uniform> u: WaterUniform;

const PI: f32 = 3.14159265359;
const GRAVITY: f32 = 9.81;

struct VsOut {
  @builtin(position) clip_pos: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) foam_jacobian: f32,
};

@vertex fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
  let cells = max(u32(round(u.absorption_and_grid.w)), 2u);
  let cell_idx = vid / 6u;
  let corner = vid % 6u;
  let cx = cell_idx % cells;
  let cz = cell_idx / cells;
  var ox = 0u;
  var oz = 0u;
  if (corner == 1u || corner == 3u || corner == 4u) { ox = 1u; }
  if (corner == 2u || corner == 4u || corner == 5u) { oz = 1u; }

  let half_ext = max(u.sky_horizon.w, 1.0);
  let uv = vec2f(f32(cx + ox), f32(cz + oz)) / f32(cells);
  let base_xz = (uv * 2.0 - 1.0) * half_ext;
  let t = u.camera_pos_and_time.w;

  var disp = vec3f(0.0);
  var tangent = vec3f(1.0, 0.0, 0.0);
  var binormal = vec3f(0.0, 0.0, 1.0);
  var jac_xx = 1.0;
  var jac_zz = 1.0;
  var jac_xz = 0.0;

  for (var i = 0u; i < 4u; i = i + 1u) {
    let w = u.waves[i];
    let dir_len = length(w.xy);
    if (w.z <= 1e-5 || w.w <= 1e-3 || dir_len <= 1e-5) {
      continue;
    }
    let dir = w.xy / dir_len;
    let k = (2.0 * PI) / w.w;
    let c = sqrt(GRAVITY / k);
    let steep = clamp(w.z, 0.0, 0.24);
    let amp = steep / k;
    let phase = k * (dot(dir, base_xz) - c * t);
    let sp = sin(phase);
    let cp = cos(phase);

    disp.x += dir.x * (amp * cp);
    disp.y += amp * sp;
    disp.z += dir.y * (amp * cp);

    tangent += vec3f(
      -dir.x * dir.x * (steep * sp),
      dir.x * (steep * cp),
      -dir.x * dir.y * (steep * sp),
    );
    binormal += vec3f(
      -dir.x * dir.y * (steep * sp),
      dir.y * (steep * cp),
      -dir.y * dir.y * (steep * sp),
    );
    jac_xx -= dir.x * dir.x * steep * sp;
    jac_zz -= dir.y * dir.y * steep * sp;
    jac_xz -= dir.x * dir.y * steep * sp;
  }

  let world_pos = vec3f(base_xz.x, u.sun_dir_and_level.w, base_xz.y) + disp;
  let normal = normalize(cross(binormal, tangent));
  let det_j = jac_xx * jac_zz - jac_xz * jac_xz;
  let crest_foam = clamp(1.0 - det_j, 0.0, 1.0);
  return VsOut(u.view_projection * vec4f(world_pos, 1.0), world_pos, normal, crest_foam);
}

fn caustic_layer(p: vec2f, t: f32) -> f32 {
  let a = sin(p.x * 2.1 + t * 1.3 + cos(p.y * 1.9 - t * 0.9));
  let b = cos(p.y * 2.4 - t * 1.1 + sin(p.x * 1.7 + t * 1.4));
  let ridge = 1.0 - abs(a * b);
  return pow(clamp(ridge, 0.0, 1.0), 4.0);
}

@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
  let n = normalize(in.normal);
  let v = normalize(u.camera_pos_and_time.xyz - in.world_pos);
  let sun_raw = u.sun_dir_and_level.xyz;
  let l = select(vec3f(0.0, 1.0, 0.0), normalize(sun_raw), length(sun_raw) > 1e-6);
  let t = u.camera_pos_and_time.w;

  // Discard dry cells where seabed rises above the water surface.
  let slope_x = u.bed_color_and_slope.w;
  let bed_y = u.sun_color_and_bed.w + in.world_pos.x * slope_x;
  let vertical_depth = in.world_pos.y - bed_y;
  if (vertical_depth <= 0.0) {
    discard;
  }

  // Refracted ray distance through the water column to the sloped seabed plane.
  let refr_dir = refract(-v, n, 1.0 / 1.333);
  let denom = -(refr_dir.y - slope_x * refr_dir.x);
  let optical_depth = select(
    vertical_depth / max(-refr_dir.y, 0.15),
    clamp(vertical_depth / max(denom, 0.15), 0.01, 200.0),
    denom > 1e-4,
  );
  let bed_hit_xz = in.world_pos.xz + refr_dir.xz * min(optical_depth, 20.0);

  // Dual-layer animated caustics at the refracted seabed hit point.
  let c1 = caustic_layer(bed_hit_xz * 0.9, t);
  let c2 = caustic_layer(bed_hit_xz * 1.45 + vec2f(3.7, 1.9), -t * 0.8);
  let caustic = (c1 + c2) * 0.5 * u.sky_zenith.w * exp(-optical_depth * 0.25);

  let transmittance = exp(-optical_depth * max(u.absorption_and_grid.rgb, vec3f(0.0)));
  let lit_bed = u.bed_color_and_slope.rgb * (0.35 + 0.65 * max(l.y, 0.0) + caustic * u.sun_color_and_bed.rgb);
  let refracted = mix(u.deep_and_foam.rgb, lit_bed, transmittance);

  // View-normal Schlick Fresnel for sky reflection (water IOR 1.333 -> F0 = 0.02).
  let ndotv = max(dot(n, v), 0.001);
  let fresnel_env = 0.02 + 0.98 * pow(1.0 - ndotv, 5.0);
  let refl_dir = reflect(-v, n);
  let sky_refl = mix(u.sky_horizon.rgb, u.sky_zenith.rgb, clamp(refl_dir.y, 0.0, 1.0));

  // Full Cook-Torrance GGX direct sun specular (V.H Fresnel + Smith masking-shadowing).
  let h = normalize(l + v);
  let ndotl = max(dot(n, l), 0.0);
  let ndoth = max(dot(n, h), 0.0);
  let vdoth = max(dot(v, h), 0.0);
  let rough = 0.06;
  let a2 = pow(rough, 4.0);
  let denom_d = ndoth * ndoth * (a2 - 1.0) + 1.0;
  let d = a2 / max(PI * denom_d * denom_d, 1e-6);
  let k = (rough + 1.0) * (rough + 1.0) * 0.125;
  let g = (ndotv / (ndotv * (1.0 - k) + k)) * (ndotl / max(ndotl * (1.0 - k) + k, 1e-5));
  let fresnel_sun = 0.02 + 0.98 * pow(1.0 - vdoth, 5.0);
  let spec = ((d * g * fresnel_sun) / max(4.0 * ndotv * max(ndotl, 0.001), 1e-5)) * ndotl * u.sun_color_and_bed.rgb;

  let shore_foam = 1.0 - smoothstep(0.05, 0.45, vertical_depth);
  let wave_foam = smoothstep(u.deep_and_foam.w, u.deep_and_foam.w + 0.35, in.foam_jacobian);
  let foam = clamp(max(shore_foam * 0.8, wave_foam), 0.0, 1.0);
  let foam_color = vec3f(0.92, 0.95, 0.98) * (0.4 + 0.6 * ndotl);

  let water_rgb = mix(refracted, sky_refl, fresnel_env) + spec;
  let final_rgb = mix(water_rgb, foam_color, foam);
  return vec4f(final_rgb, 1.0);
}
