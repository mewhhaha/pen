// CDLOD Heightfield Terrain Pipeline:
// - Procedural per-patch vertex grid with crack-free world-distance CDLOD morphing
// - Manual bilinear R32Float heightmap sampling (portable across WebGPU without FLOAT32_FILTERABLE)
// - Exact border-aware finite-difference world normals + 4-layer height/slope splatting + triplanar cliffs.

struct TerrainUniform {
  view_projection: mat4x4f,
  camera_pos_and_fog: vec4f,       // xyz = camera world pos, w = fog density
  sun_dir_and_height_scale: vec4f, // xyz = toward sun, w = height_scale
  sun_color_and_ambient: vec4f,    // rgb = sun radiance, w = ambient intensity
  world_bounds: vec4f,             // x = world_size_x, y = world_size_z, z = morph_start, w = morph_end
  layer_heights: vec4f,            // x = shore_max, y = grass_max, z = snow_start, w = cliff_slope_min
  fog_color: vec4f,                // rgb = fog color, w = triplanar_scale
};

struct TerrainPatch {
  origin_scale_morph: vec4f, // x = origin_x, y = origin_z, z = patch_world_size, w = manual morph override (<0 uses distance CDLOD)
  uv_rect_and_grid: vec4f,   // x = uv_min_x, y = uv_min_z, z = uv_size, w = grid_cells (even >= 2)
};

@group(0) @binding(0) var<uniform> u: TerrainUniform;
@group(0) @binding(1) var<storage, read> patches: array<TerrainPatch>;
@group(0) @binding(2) var heightmap: texture_2d<f32>;

const PI: f32 = 3.14159265359;

fn sample_height_uv(uv: vec2f) -> f32 {
  let dims = vec2f(textureDimensions(heightmap, 0));
  let max_coord = vec2i(i32(dims.x) - 1, i32(dims.y) - 1);
  let p = clamp(uv, vec2f(0.0), vec2f(1.0)) * max(dims - vec2f(1.0), vec2f(1.0));
  let p0 = vec2i(floor(p));
  let p1 = min(p0 + vec2i(1, 1), max_coord);
  let f = fract(p);
  let h00 = textureLoad(heightmap, vec2i(p0.x, p0.y), 0).r;
  let h10 = textureLoad(heightmap, vec2i(p1.x, p0.y), 0).r;
  let h01 = textureLoad(heightmap, vec2i(p0.x, p1.y), 0).r;
  let h11 = textureLoad(heightmap, vec2i(p1.x, p1.y), 0).r;
  return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y) * u.sun_dir_and_height_scale.w;
}

struct VsOut {
  @builtin(position) clip_pos: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) uv: vec2f,
};

@vertex fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
  let p = patches[iid];
  let cells = max(u32(round(p.uv_rect_and_grid.w)) & ~1u, 2u);
  let cell_idx = vid / 6u;
  let corner = vid % 6u;
  let cx = cell_idx % cells;
  let cz = cell_idx / cells;
  var ox = 0u;
  var oz = 0u;
  if (corner == 1u || corner == 3u || corner == 4u) { ox = 1u; }
  if (corner == 2u || corner == 4u || corner == 5u) { oz = 1u; }

  let ix = cx + ox;
  let iz = cz + oz;
  let gx = f32(ix);
  let gz = f32(iz);

  // Evaluate unmorphed world position first so adjacent patches share identical distance-based morph
  // at every shared edge vertex (Strugar CDLOD rule -> crack-free transitions).
  let unmorphed_local = vec2f(gx, gz) / f32(cells);
  let unmorphed_uv = p.uv_rect_and_grid.xy + unmorphed_local * p.uv_rect_and_grid.z;
  let unmorphed_wx = p.origin_scale_morph.x + unmorphed_local.x * p.origin_scale_morph.z;
  let unmorphed_wz = p.origin_scale_morph.y + unmorphed_local.y * p.origin_scale_morph.z;
  let unmorphed_wy = sample_height_uv(unmorphed_uv);

  var morph = clamp(p.origin_scale_morph.w, 0.0, 1.0);
  if (p.origin_scale_morph.w < 0.0 && u.world_bounds.w > u.world_bounds.z) {
    let d_cam = length(u.camera_pos_and_fog.xyz - vec3f(unmorphed_wx, unmorphed_wy, unmorphed_wz));
    morph = clamp((d_cam - u.world_bounds.z) / max(u.world_bounds.w - u.world_bounds.z, 1e-4), 0.0, 1.0);
  }

  let frac_odd = vec2f(f32(ix & 1u), f32(iz & 1u));
  let morphed = vec2f(gx, gz) - frac_odd * morph;

  let local_uv = morphed / f32(cells);
  let uv = p.uv_rect_and_grid.xy + local_uv * p.uv_rect_and_grid.z;
  let wx = p.origin_scale_morph.x + local_uv.x * p.origin_scale_morph.z;
  let wz = p.origin_scale_morph.y + local_uv.y * p.origin_scale_morph.z;
  let wy = sample_height_uv(uv);
  let world_pos = vec3f(wx, wy, wz);
  return VsOut(u.view_projection * vec4f(world_pos, 1.0), world_pos, uv);
}

fn triplanar_rock_pattern(p: vec3f, n: vec3f) -> f32 {
  let w = pow(abs(n), vec3f(4.0));
  let wn = w / max(w.x + w.y + w.z, 1e-5);
  let s = max(u.fog_color.w, 0.05);
  let px = sin(p.y * s * 3.1 + sin(p.z * s * 1.7)) * 0.5 + 0.5;
  let py = sin(p.x * s * 2.3 + cos(p.z * s * 2.9)) * 0.5 + 0.5;
  let pz = sin(p.y * s * 3.1 + cos(p.x * s * 1.9)) * 0.5 + 0.5;
  return px * wn.x + py * wn.y + pz * wn.z;
}

@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
  let hm_dims = vec2f(textureDimensions(heightmap, 0));
  let texel = vec2f(1.0 / max(hm_dims.x - 1.0, 1.0), 1.0 / max(hm_dims.y - 1.0, 1.0));
  let u_l = max(in.uv.x - texel.x, 0.0);
  let u_r = min(in.uv.x + texel.x, 1.0);
  let v_d = max(in.uv.y - texel.y, 0.0);
  let v_u = min(in.uv.y + texel.y, 1.0);
  let hl = sample_height_uv(vec2f(u_l, in.uv.y));
  let hr = sample_height_uv(vec2f(u_r, in.uv.y));
  let hd = sample_height_uv(vec2f(in.uv.x, v_d));
  let hu = sample_height_uv(vec2f(in.uv.x, v_u));
  // Divide by actual clamped world separation so heightmap borders retain true slope.
  let span_x = max((u_r - u_l) * max(u.world_bounds.x, 1e-3), 1e-4);
  let span_z = max((v_u - v_d) * max(u.world_bounds.y, 1e-3), 1e-4);
  let n_world = normalize(vec3f((hl - hr) / span_x, 1.0, (hd - hu) / span_z));

  let h = in.world_pos.y;
  let slope = 1.0 - clamp(n_world.y, 0.0, 1.0);
  let sand_col = vec3f(0.74, 0.66, 0.48);
  let grass_col = vec3f(0.20, 0.44, 0.16);
  let rock_strata = triplanar_rock_pattern(in.world_pos, n_world);
  let rock_col = mix(vec3f(0.26, 0.25, 0.24), vec3f(0.42, 0.39, 0.36), rock_strata);
  let snow_col = vec3f(0.88, 0.91, 0.95);

  let grass_w = smoothstep(u.layer_heights.x - 0.5, u.layer_heights.x + 1.0, h);
  let snow_w = smoothstep(u.layer_heights.z - 1.0, u.layer_heights.z + 2.0, h) * (1.0 - smoothstep(0.25, 0.55, slope));
  let cliff_w = smoothstep(u.layer_heights.w, u.layer_heights.w + 0.18, slope);

  var albedo = mix(sand_col, grass_col, grass_w);
  albedo = mix(albedo, rock_col, cliff_w);
  albedo = mix(albedo, snow_col, snow_w);
  let roughness = mix(mix(0.88, 0.78, cliff_w), 0.35, snow_w);

  let sun_raw = u.sun_dir_and_height_scale.xyz;
  let l_dir = select(vec3f(0.0, 1.0, 0.0), normalize(sun_raw), length(sun_raw) > 1e-6);
  let v_dir = normalize(u.camera_pos_and_fog.xyz - in.world_pos);
  let h_dir = normalize(l_dir + v_dir);
  let ndotl = max(dot(n_world, l_dir), 0.0);
  let ndotv = max(dot(n_world, v_dir), 0.001);
  let ndoth = max(dot(n_world, h_dir), 0.0);
  let vdoth = max(dot(v_dir, h_dir), 0.0);

  let a2 = pow(roughness, 4.0);
  let denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
  let d = a2 / max(PI * denom * denom, 1e-6);
  let fresnel = 0.04 + 0.96 * pow(1.0 - vdoth, 5.0);
  let k = (roughness + 1.0) * (roughness + 1.0) * 0.125;
  let g = (ndotv / (ndotv * (1.0 - k) + k)) * (ndotl / max(ndotl * (1.0 - k) + k, 1e-5));
  let spec = (d * g * fresnel) / max(4.0 * ndotv * max(ndotl, 0.001), 1e-5);
  let direct = (albedo * (1.0 - fresnel) / PI + vec3f(spec)) * u.sun_color_and_ambient.rgb * ndotl;

  let amb = mix(vec3f(0.18, 0.17, 0.15), vec3f(0.45, 0.58, 0.78), n_world.y * 0.5 + 0.5) * u.sun_color_and_ambient.w;
  var color = direct + albedo * amb;

  let dist = length(u.camera_pos_and_fog.xyz - in.world_pos);
  let fog_t = 1.0 - exp(-dist * max(u.camera_pos_and_fog.w, 0.0));
  color = mix(color, u.fog_color.rgb, clamp(fog_t, 0.0, 1.0));
  return vec4f(color, 1.0);
}
