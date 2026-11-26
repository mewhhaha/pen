// Dynamic Diffuse Global Illumination (DDGI, Majercik et al., JCGT 2019)
// with dynamic local-light bounce injection, backface inside-wall probe classification,
// toroidal camera scrolling, and flat-buffer cascade packing for zero-sampler PBR lookup.
// Prefixed at compile time with tlas_traversal.wgsl or tlas_traversal_hw.wgsl.
//
// Atlas layout:
//  - Physical storage coord (sx, sy, sz) wraps logical (ix, iy, iz) by scroll_offset
//  - 2D tile grid: col = sz * nx + sx (0 .. nx*nz), row = sy (0 .. ny)
//  - Irradiance tile: 8x8 interior + 1px octahedral wrap border = 10x10 per probe (rgb = gamma irr, a = validity)
//  - Visibility tile: 16x16 interior + 1px octahedral wrap border = 18x18 per probe (rg = mean, mean^2)

struct DdgiUniform {
  origin_and_hysteresis: vec4f,   // xyz = grid origin, w = hysteresis (0.0 on reset)
  spacing_and_max_dist: vec4f,    // xyz = grid spacing, w = max ray distance
  sun_dir_and_gamma: vec4f,       // xyz = normalized sun dir, w = irradiance gamma (e.g. 5.0)
  sun_color_and_sharp: vec4f,     // rgb = sun radiance, a = depth_sharpness (e.g. 50.0)
  sky_zenith_and_nbias: vec4f,    // rgb = sky zenith, a = normal_bias
  sky_horizon_and_vbias: vec4f,   // rgb = sky horizon, a = view_bias
  rot_col0: vec4f,                // SO(3) random rotation matrix col 0, w = bounce_scale
  rot_col1: vec4f,                // SO(3) random rotation matrix col 1, w = frame_index_f32
  rot_col2: vec4f,                // SO(3) random rotation matrix col 2, w = 0
  grid_dims: vec4u,               // x = nx, y = ny, z = nz, w = rays_per_probe
  tlas_counts: vec4u,             // x = tlas_nodes, y = tlas_insts, z = blas_tris, w = visibility_mask
  scroll_and_lights: vec4i,       // xyz = toroidal scroll offset, w = local_light_count
  scroll_delta: vec4i,            // xyz = cells scrolled this frame (dx, dy, dz), w = 0
};

struct RtLocalLight {
  position_range: vec4f,      // xyz = world position, w = reach/range
  color_outer_cos: vec4f,     // rgb = linear color * intensity, w = outer_cos (-1 for point)
  direction_inner_cos: vec4f, // xyz = unit spot/area direction, w = inner_cos
};

const DDGI_PI: f32 = 3.14159265359;
const IRR_INNER: u32 = 8u;
const IRR_TILE: u32 = 10u;
const VIS_INNER: u32 = 16u;
const VIS_TILE: u32 = 18u;

fn ddgi_wrap_coord(v: i32, n: u32) -> u32 {
  let ni = max(i32(n), 1);
  return u32(((v % ni) + ni) % ni);
}

fn ddgi_storage_coord(logical: vec3u, dims: vec3u, scroll: vec3i) -> vec3u {
  return vec3u(
    ddgi_wrap_coord(i32(logical.x) + scroll.x, dims.x),
    ddgi_wrap_coord(i32(logical.y) + scroll.y, dims.y),
    ddgi_wrap_coord(i32(logical.z) + scroll.z, dims.z),
  );
}

fn ddgi_is_scrolled_in(logical: vec3u, dims: vec3u, delta: vec3i) -> bool {
  let ix = i32(logical.x);
  let iy = i32(logical.y);
  let iz = i32(logical.z);
  let nx = i32(dims.x);
  let ny = i32(dims.y);
  let nz = i32(dims.z);
  if (delta.x > 0 && ix >= nx - delta.x) { return true; }
  if (delta.x < 0 && ix < -delta.x) { return true; }
  if (delta.y > 0 && iy >= ny - delta.y) { return true; }
  if (delta.y < 0 && iy < -delta.y) { return true; }
  if (delta.z > 0 && iz >= nz - delta.z) { return true; }
  if (delta.z < 0 && iz < -delta.z) { return true; }
  return false;
}

fn ddgi_sign_not_zero(v: vec2f) -> vec2f {
  return vec2f(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
}

fn ddgi_oct_encode(v: vec3f) -> vec2f {
  let l1 = abs(v.x) + abs(v.y) + abs(v.z);
  var p = v.xy / max(l1, 1e-8);
  if (v.z < 0.0) {
    p = (1.0 - abs(p.yx)) * ddgi_sign_not_zero(p);
  }
  return p;
}

fn ddgi_oct_decode(e: vec2f) -> vec3f {
  var v = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  if (v.z < 0.0) {
    let t = (1.0 - abs(v.yx)) * ddgi_sign_not_zero(v.xy);
    v.x = t.x;
    v.y = t.y;
  }
  return normalize(v);
}

fn ddgi_spherical_fibonacci(idx: u32, count: u32) -> vec3f {
  let phi = 1.61803398875;
  let i = f32(idx);
  let n = max(f32(count), 1.0);
  let cos_t = 1.0 - (2.0 * i + 1.0) / n;
  let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
  let theta = 2.0 * DDGI_PI * fract(i * (phi - 1.0));
  return vec3f(cos(theta) * sin_t, sin(theta) * sin_t, cos_t);
}

@group(0) @binding(10) var<storage, read> probe_offsets_ro: array<vec4f>;

fn ddgi_storage_index(sc: vec3u, dims: vec3u) -> u32 {
  return (sc.y * dims.z + sc.z) * dims.x + sc.x;
}

fn ddgi_probe_world_pos(grid_coord: vec3u, u: DdgiUniform) -> vec3f {
  let dims = u.grid_dims.xyz;
  let sc = ddgi_storage_coord(grid_coord, dims, u.scroll_and_lights.xyz);
  let off = probe_offsets_ro[ddgi_storage_index(sc, dims)].xyz;
  return u.origin_and_hysteresis.xyz + vec3f(grid_coord) * u.spacing_and_max_dist.xyz + off;
}

fn ddgi_probe_uv(
  storage_coord: vec3u,
  dir: vec3f,
  dims: vec3u,
  inner: u32,
  tile: u32,
) -> vec2f {
  let col = storage_coord.z * dims.x + storage_coord.x;
  let row = storage_coord.y;
  let atlas_w = f32(dims.x * dims.z * tile);
  let atlas_h = f32(dims.y * tile);
  let oct = ddgi_oct_encode(normalize(dir)) * 0.5 + vec2f(0.5);
  let px = vec2f(f32(col * tile + 1u), f32(row * tile + 1u)) + oct * f32(inner);
  return px / vec2f(atlas_w, atlas_h);
}

fn ddgi_eval_sky(dir: vec3f, u: DdgiUniform) -> vec3f {
  let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
  let ground = u.sky_horizon_and_vbias.rgb * 0.3;
  let upper = mix(u.sky_horizon_and_vbias.rgb, u.sky_zenith_and_nbias.rgb, pow(clamp(dir.y, 0.0, 1.0), 0.6));
  return select(mix(ground, u.sky_horizon_and_vbias.rgb, t * 2.0), upper, dir.y >= 0.0);
}

// Reusable leak-free Chebyshev-weighted trilinear DDGI irradiance sampling.
fn ddgi_sample_irradiance(
  world_pos: vec3f,
  normal: vec3f,
  view_dir: vec3f,
  u: DdgiUniform,
  irr_atlas: texture_2d<f32>,
  vis_atlas: texture_2d<f32>,
  atlas_samp: sampler,
) -> vec3f {
  let dims = u.grid_dims.xyz;
  if (dims.x == 0u || dims.y == 0u || dims.z == 0u) {
    return vec3f(0.0);
  }
  let spacing = max(u.spacing_and_max_dist.xyz, vec3f(1e-4));
  let min_sp = min(spacing.x, min(spacing.y, spacing.z));
  let bias = (normal * u.sky_zenith_and_nbias.w + view_dir * u.sky_horizon_and_vbias.w) * min_sp;
  let biased_pos = world_pos + bias;

  let rel = (biased_pos - u.origin_and_hysteresis.xyz) / spacing;
  let max_c = vec3f(f32(dims.x - 1u), f32(dims.y - 1u), f32(dims.z - 1u));
  let clamped = clamp(rel, vec3f(0.0), max_c);
  let base_i = vec3u(floor(clamped));
  let alpha = clamped - vec3f(base_i);
  let gamma = max(u.sun_dir_and_gamma.w, 1.0);
  let scroll = u.scroll_and_lights.xyz;

  var sum_irr = vec3f(0.0);
  var sum_w = 0.0;

  for (var i = 0u; i < 8u; i++) {
    let off = vec3u(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
    let coord = min(base_i + off, dims - vec3u(1u));
    let sc = ddgi_storage_coord(coord, dims, scroll);
    let trilinear = mix(vec3f(1.0) - alpha, alpha, vec3f(off));
    var w = max(trilinear.x * trilinear.y * trilinear.z, 1e-4);

    let probe_pos = ddgi_probe_world_pos(coord, u);
    let to_probe_unbiased = probe_pos - world_pos;
    let dist_unbiased = length(to_probe_unbiased);
    let dir_to_probe = select(normal, to_probe_unbiased / dist_unbiased, dist_unbiased > 1e-5);

    // Smooth backface wrap weighting.
    let wrap = (dot(dir_to_probe, normal) + 1.0) * 0.5;
    w *= wrap * wrap + 0.05;

    // Chebyshev visibility test from probe toward biased surface point (eliminates light leaks!).
    let probe_to_pt = biased_pos - probe_pos;
    let dist_to_pt = length(probe_to_pt);
    let dir_from_probe = select(-normal, probe_to_pt / dist_to_pt, dist_to_pt > 1e-5);
    let vis_uv = ddgi_probe_uv(sc, dir_from_probe, dims, VIS_INNER, VIS_TILE);
    let moments = textureSampleLevel(vis_atlas, atlas_samp, vis_uv, 0.0).rg;
    let mean = moments.x;
    let mean2 = moments.y;
    if (dist_to_pt > mean) {
      let variance = max(abs(mean2 - mean * mean), 1e-4);
      let d = dist_to_pt - mean;
      var chebyshev = variance / (variance + d * d);
      // Crush low Chebyshev weights to fully silence thin-wall leaks.
      chebyshev = max(chebyshev * chebyshev * chebyshev, 0.0);
      w *= chebyshev;
    }
    if (w <= 1e-6) {
      continue;
    }

    let irr_uv = ddgi_probe_uv(sc, normal, dims, IRR_INNER, IRR_TILE);
    let irr_texel = textureSampleLevel(irr_atlas, atlas_samp, irr_uv, 0.0);
    // Probe classification validity (0.0 for probes trapped inside solid geometry).
    let validity = clamp(irr_texel.a, 0.0, 1.0);
    w *= validity;
    if (w <= 1e-6) {
      continue;
    }
    let encoded = max(irr_texel.rgb, vec3f(0.0));
    var irr: vec3f;
    if (abs(gamma - 5.0) < 1e-4) {
      irr = encoded * encoded * sqrt(encoded);
    } else {
      irr = pow(encoded, vec3f(gamma * 0.5));
    }
    sum_irr += irr * w;
    sum_w += w;
  }

  if (sum_w <= 1e-6) {
    return vec3f(0.0);
  }
  let avg = sum_irr / sum_w;
  return avg * avg;
}

// ---------------- Pass 1: Trace Probe Rays ----------------
@group(0) @binding(0) var<uniform> u: DdgiUniform;
// __TLAS_BINDING_1__
@group(0) @binding(2) var<storage, read> tlas_instances: array<TlasInstance>;
@group(0) @binding(3) var<storage, read> blas_nodes: array<BvhNode>;
@group(0) @binding(4) var<storage, read> blas_tris: array<BlasTri>;
@group(0) @binding(5) var prev_irr_tex: texture_2d<f32>;
@group(0) @binding(6) var prev_vis_tex: texture_2d<f32>;
@group(0) @binding(7) var atlas_sampler: sampler;
@group(0) @binding(8) var<storage, read_write> ray_radiance: array<vec4f>;
@group(0) @binding(9) var<storage, read> local_lights: array<RtLocalLight>;

@compute @workgroup_size(64, 1, 1)
fn ddgi_trace_probes(@builtin(global_invocation_id) id: vec3u) {
  let dims = u.grid_dims.xyz;
  let rays_per_probe = max(u.grid_dims.w, 1u);
  let total_probes = dims.x * dims.y * dims.z;
  let global_idx = id.x;
  let probe_idx = global_idx / rays_per_probe;
  let ray_idx = global_idx % rays_per_probe;
  if (probe_idx >= total_probes) {
    return;
  }

  let ix = probe_idx % dims.x;
  let iz = (probe_idx / dims.x) % dims.z;
  let iy = probe_idx / (dims.x * dims.z);
  let probe_pos = ddgi_probe_world_pos(vec3u(ix, iy, iz), u);

  let rot = mat3x3f(u.rot_col0.xyz, u.rot_col1.xyz, u.rot_col2.xyz);
  let ray_dir = normalize(rot * ddgi_spherical_fibonacci(ray_idx, rays_per_probe));
  let max_dist = max(u.spacing_and_max_dist.w, 1.0);
  let mask = u.tlas_counts.w;

  let h = trace_tlas(
    probe_pos,
    ray_dir,
    1e-3,
    max_dist,
    mask,
    false,
    u.tlas_counts.x,
    u.tlas_counts.y,
    u.tlas_counts.z,
  );

  if (!h.hit) {
    let sky = ddgi_eval_sky(ray_dir, u);
    ray_radiance[global_idx] = vec4f(sky, max_dist);
    return;
  }

  // Flag backface hits with negative distance so blend pass classifies probes trapped inside walls.
  if (!h.front_face) {
    ray_radiance[global_idx] = vec4f(0.0, 0.0, 0.0, -h.t);
    return;
  }

  let hit_pos = probe_pos + ray_dir * h.t;
  let sun_raw = u.sun_dir_and_gamma.xyz;
  let sun_dir = select(vec3f(0.0, 1.0, 0.0), normalize(sun_raw), length(sun_raw) > 1e-6);
  let diffuse_albedo = h.albedo * (1.0 - h.metallic);

  var rad = h.emissive;
  let ndl = max(0.0, dot(h.world_normal, sun_dir));
  if (ndl > 0.0) {
    let blocked = trace_tlas_any_hit(
      hit_pos + h.world_normal * 0.01,
      sun_dir,
      1e-3,
      max_dist,
      mask,
      u.tlas_counts.x,
      u.tlas_counts.y,
      u.tlas_counts.z,
    );
    if (!blocked) {
      rad += (diffuse_albedo / DDGI_PI) * u.sun_color_and_sharp.rgb * ndl;
    }
  }

  // Dynamic point, spot, and area lights injected directly into DDGI secondary bounces!
  let n_lights = u32(max(u.scroll_and_lights.w, 0));
  for (var li = 0u; li < n_lights; li++) {
    let lt = local_lights[li];
    let to_light = lt.position_range.xyz - hit_pos;
    let reach = lt.position_range.w;
    let dist2 = dot(to_light, to_light);
    if (dist2 <= 1e-8 || dist2 >= reach * reach) {
      continue;
    }
    let dist = sqrt(dist2);
    let ldir = to_light / dist;
    let l_ndl = max(0.0, dot(h.world_normal, ldir));
    if (l_ndl <= 0.0) {
      continue;
    }
    let ratio = dist / reach;
    let r2 = ratio * ratio;
    let win = clamp(1.0 - r2 * r2, 0.0, 1.0);
    var atten = (win * win) / (dist2 + 1.0);
    if (lt.color_outer_cos.w > -0.999) {
      let cone = dot(-ldir, lt.direction_inner_cos.xyz);
      atten *= smoothstep(lt.color_outer_cos.w, max(lt.direction_inner_cos.w, lt.color_outer_cos.w + 1e-4), cone);
    }
    if (atten <= 1e-5) {
      continue;
    }
    let blocked = trace_tlas_any_hit(
      hit_pos + h.world_normal * 0.015,
      ldir,
      1e-3,
      max(dist - 0.03, 2e-3),
      mask,
      u.tlas_counts.x,
      u.tlas_counts.y,
      u.tlas_counts.z,
    );
    if (!blocked) {
      rad += (diffuse_albedo / DDGI_PI) * lt.color_outer_cos.rgb * (atten * l_ndl);
    }
  }

  // Multi-bounce feedback from previous frame's DDGI irradiance field!
  if (u.rot_col1.w > 0.5 && u.rot_col0.w > 0.0) {
    let prev_gi = ddgi_sample_irradiance(
      hit_pos,
      h.world_normal,
      -ray_dir,
      u,
      prev_irr_tex,
      prev_vis_tex,
      atlas_sampler,
    );
    rad += diffuse_albedo * prev_gi * u.rot_col0.w;
  }

  ray_radiance[global_idx] = vec4f(rad, h.t);
}

// ---------------- Pass 2: Blend Irradiance & Visibility Atlases ----------------
@group(0) @binding(1) var<storage, read> blend_rays: array<vec4f>;
@group(0) @binding(2) var blend_prev_irr: texture_2d<f32>;
@group(0) @binding(3) var blend_prev_vis: texture_2d<f32>;
@group(0) @binding(4) var blend_out_irr: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var blend_out_vis: texture_storage_2d<rgba16float, write>;

// Cooperative ray staging: every 8x8 workgroup covers threads from exactly one probe
// (both 8x8 irradiance tiles and 16x16 visibility quadrants are aligned to 8),
// so computing ray directions and loading ray payloads once per 64-ray chunk cuts work 64x.
var<workgroup> wg_ray_val: array<vec4f, 64>;
var<workgroup> wg_ray_dir: array<vec3f, 64>;

@compute @workgroup_size(8, 8, 1)
fn ddgi_blend_irradiance(
  @builtin(global_invocation_id) id: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let dims = u.grid_dims.xyz;
  let cols = dims.x * dims.z;
  let rows = dims.y;
  let tx = id.x % IRR_INNER;
  let ty = id.y % IRR_INNER;
  let col = id.x / IRR_INNER;
  let row = id.y / IRR_INNER;
  let in_bounds = col < cols && row < rows;

  let ix = col % max(dims.x, 1u);
  let iz = col / max(dims.x, 1u);
  let iy = row;
  let logical = vec3u(ix, iy, iz);
  let probe_idx = (iy * dims.z + iz) * dims.x + ix;
  let rays_per_probe = max(u.grid_dims.w, 1u);
  let rot = mat3x3f(u.rot_col0.xyz, u.rot_col1.xyz, u.rot_col2.xyz);

  let oct = ((vec2f(f32(tx), f32(ty)) + vec2f(0.5)) / f32(IRR_INNER)) * 2.0 - vec2f(1.0);
  let texel_dir = ddgi_oct_decode(oct);

  var acc = vec3f(0.0);
  var weight_sum = 0.0;
  var backface_count = 0u;
  let base_ray = probe_idx * rays_per_probe;
  let lidx = lid.y * 8u + lid.x;

  for (var chunk = 0u; chunk < rays_per_probe; chunk += 64u) {
    let r = chunk + lidx;
    if (in_bounds && r < rays_per_probe) {
      wg_ray_val[lidx] = blend_rays[base_ray + r];
      wg_ray_dir[lidx] = normalize(rot * ddgi_spherical_fibonacci(r, rays_per_probe));
    }
    workgroupBarrier();
    if (in_bounds) {
      let count = min(64u, rays_per_probe - chunk);
      for (var j = 0u; j < count; j++) {
        let ray_val = wg_ray_val[j];
        if (ray_val.w < 0.0) {
          backface_count += 1u;
          continue;
        }
        let w = dot(texel_dir, wg_ray_dir[j]);
        if (w > 1e-4) {
          acc += ray_val.rgb * w;
          weight_sum += w;
        }
      }
    }
    workgroupBarrier();
  }

  if (!in_bounds) {
    return;
  }

  // Classify probe as inactive when >= 25% of rays hit backfaces (trapped inside solid geometry).
  let valid = select(1.0, 0.0, backface_count * 4u >= rays_per_probe);
  var new_irr = select(vec3f(0.0), acc / weight_sum, weight_sum > 1e-5);
  let gamma = max(u.sun_dir_and_gamma.w, 1.0);
  new_irr = pow(max(new_irr, vec3f(0.0)), vec3f(1.0 / gamma));

  let sc = ddgi_storage_coord(logical, dims, u.scroll_and_lights.xyz);
  let s_col = sc.z * dims.x + sc.x;
  let s_row = sc.y;
  let px = vec2i(i32(s_col * IRR_TILE + 1u + tx), i32(s_row * IRR_TILE + 1u + ty));
  let prev = textureLoad(blend_prev_irr, px, 0);
  let scrolled_in = ddgi_is_scrolled_in(logical, dims, u.scroll_delta.xyz);
  let base_h = select(clamp(u.origin_and_hysteresis.w, 0.0, 0.99), 0.0, scrolled_in);
  // Adaptive perceptual hysteresis (RTXGI DDGI): drop hysteresis sharply on large gamma-space
  // radiance shifts so dynamic sun/torch/door transitions converge in ~4 frames while static areas stay smooth.
  let delta_rgb = abs(new_irr - prev.rgb);
  let max_delta = max(delta_rgb.r, max(delta_rgb.g, delta_rgb.b));
  let h = select(base_h, base_h * 0.25, max_delta > 0.25);
  let blended_rgb = mix(new_irr, prev.rgb, h);
  let blended_v = mix(valid, prev.a, base_h);
  textureStore(blend_out_irr, px, vec4f(blended_rgb, blended_v));
}

// ---------------- Pass 2b: Probe Offset Relocation (Escape Solid Geometry & Surface Grazing) ----------------
@group(0) @binding(11) var<storage, read_write> probe_offsets_rw: array<vec4f>;

@compute @workgroup_size(64, 1, 1)
fn ddgi_relocate_probes(@builtin(global_invocation_id) id: vec3u) {
  let dims = u.grid_dims.xyz;
  let total_probes = dims.x * dims.y * dims.z;
  let probe_idx = id.x;
  if (probe_idx >= total_probes) {
    return;
  }

  let ix = probe_idx % dims.x;
  let iz = (probe_idx / dims.x) % dims.z;
  let iy = probe_idx / (dims.x * dims.z);
  let logical = vec3u(ix, iy, iz);
  let sc = ddgi_storage_coord(logical, dims, u.scroll_and_lights.xyz);
  let s_idx = ddgi_storage_index(sc, dims);
  let rays_per_probe = max(u.grid_dims.w, 1u);
  let rot = mat3x3f(u.rot_col0.xyz, u.rot_col1.xyz, u.rot_col2.xyz);

  let scrolled_in = ddgi_is_scrolled_in(logical, dims, u.scroll_delta.xyz);
  let reset = (u.rot_col1.w < 0.5) || scrolled_in;
  let offset = select(probe_offsets_rw[s_idx].xyz, vec3f(0.0), reset);

  var closest_back_dist = 1e9;
  var closest_back_dir = vec3f(0.0);
  var closest_front_dist = 1e9;
  var closest_front_dir = vec3f(0.0);
  var farthest_front_dist = 0.0;
  var farthest_front_dir = vec3f(0.0);
  var backface_count = 0u;

  let base_ray = probe_idx * rays_per_probe;
  for (var r = 0u; r < rays_per_probe; r++) {
    let raw_d = blend_rays[base_ray + r].w;
    let ray_dir = normalize(rot * ddgi_spherical_fibonacci(r, rays_per_probe));
    if (raw_d < 0.0) {
      backface_count += 1u;
      let bd = -raw_d;
      if (bd < closest_back_dist) {
        closest_back_dist = bd;
        closest_back_dir = ray_dir;
      }
    } else {
      if (raw_d < closest_front_dist) {
        closest_front_dist = raw_d;
        closest_front_dir = ray_dir;
      }
      if (raw_d > farthest_front_dist) {
        farthest_front_dist = raw_d;
        farthest_front_dir = ray_dir;
      }
    }
  }

  let spacing = max(u.spacing_and_max_dist.xyz, vec3f(1e-3));
  let min_sp = min(spacing.x, min(spacing.y, spacing.z));
  let min_front = 0.25 * min_sp;
  var full_offset = offset;
  var state = 1.0;

  if (backface_count * 4u >= rays_per_probe && closest_back_dist < 1e8) {
    // Trapped inside geometry: push through nearest backface toward open space.
    full_offset = offset + closest_back_dir * (closest_back_dist + 0.15 * min_sp);
    state = 0.0;
  } else if (closest_front_dist < min_front && farthest_front_dist > closest_front_dist + 0.05 * min_sp) {
    // Grazing a surface: nudge away from the closest frontface toward open space.
    let nudge_raw = farthest_front_dir - closest_front_dir * 0.5;
    let nudge_len = length(nudge_raw);
    if (nudge_len > 1e-4) {
      let push = min(min_front - closest_front_dist, 0.20 * min_sp);
      full_offset = offset + (nudge_raw / nudge_len) * push;
    }
  }

  // Keep relocated probes strictly inside their cell ellipsoid (<= 0.45 * spacing) so grid topology never flips.
  let max_off = spacing * 0.45;
  let norm_off = full_offset / max_off;
  let norm_len = length(norm_off);
  if (norm_len > 1.0) {
    full_offset = (norm_off / norm_len) * max_off;
  }
  probe_offsets_rw[s_idx] = vec4f(full_offset, state);
}

@compute @workgroup_size(8, 8, 1)
fn ddgi_blend_visibility(
  @builtin(global_invocation_id) id: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let dims = u.grid_dims.xyz;
  let cols = dims.x * dims.z;
  let rows = dims.y;
  let tx = id.x % VIS_INNER;
  let ty = id.y % VIS_INNER;
  let col = id.x / VIS_INNER;
  let row = id.y / VIS_INNER;
  let in_bounds = col < cols && row < rows;

  let ix = col % max(dims.x, 1u);
  let iz = col / max(dims.x, 1u);
  let iy = row;
  let logical = vec3u(ix, iy, iz);
  let probe_idx = (iy * dims.z + iz) * dims.x + ix;
  let rays_per_probe = max(u.grid_dims.w, 1u);
  let rot = mat3x3f(u.rot_col0.xyz, u.rot_col1.xyz, u.rot_col2.xyz);

  let oct = ((vec2f(f32(tx), f32(ty)) + vec2f(0.5)) / f32(VIS_INNER)) * 2.0 - vec2f(1.0);
  let texel_dir = ddgi_oct_decode(oct);

  let max_dist = max(u.spacing_and_max_dist.w, 1.0);
  let max_probe_dist = length(u.spacing_and_max_dist.xyz) * 1.5;
  let clamped_max = min(max_dist, max_probe_dist);
  let sharp = max(u.sun_color_and_sharp.w, 1.0);

  var acc_d = 0.0;
  var acc_d2 = 0.0;
  var weight_sum = 0.0;
  let base_ray = probe_idx * rays_per_probe;
  let lidx = lid.y * 8u + lid.x;

  for (var chunk = 0u; chunk < rays_per_probe; chunk += 64u) {
    let r = chunk + lidx;
    if (in_bounds && r < rays_per_probe) {
      let raw_d = blend_rays[base_ray + r].w;
      let dist = select(min(raw_d, clamped_max), min(-raw_d * 0.2, max_probe_dist), raw_d < 0.0);
      wg_ray_val[lidx] = vec4f(dist, dist * dist, 0.0, 0.0);
      wg_ray_dir[lidx] = normalize(rot * ddgi_spherical_fibonacci(r, rays_per_probe));
    }
    workgroupBarrier();
    if (in_bounds) {
      let count = min(64u, rays_per_probe - chunk);
      for (var j = 0u; j < count; j++) {
        let ndot = dot(texel_dir, wg_ray_dir[j]);
        if (ndot > 0.0) {
          let w = pow(ndot, sharp);
          if (w > 1e-6) {
            let dm = wg_ray_val[j];
            acc_d += dm.x * w;
            acc_d2 += dm.y * w;
            weight_sum += w;
          }
        }
      }
    }
    workgroupBarrier();
  }

  if (!in_bounds) {
    return;
  }
  let new_m = select(vec2f(max_probe_dist, max_probe_dist * max_probe_dist), vec2f(acc_d, acc_d2) / weight_sum, weight_sum > 1e-6);

  let sc = ddgi_storage_coord(logical, dims, u.scroll_and_lights.xyz);
  let s_col = sc.z * dims.x + sc.x;
  let s_row = sc.y;
  let px = vec2i(i32(s_col * VIS_TILE + 1u + tx), i32(s_row * VIS_TILE + 1u + ty));
  let prev = textureLoad(blend_prev_vis, px, 0).rg;
  let scrolled_in = ddgi_is_scrolled_in(logical, dims, u.scroll_delta.xyz);
  let h = select(clamp(u.origin_and_hysteresis.w, 0.0, 0.99), 0.0, scrolled_in);
  let blended = mix(new_m, prev, h);
  textureStore(blend_out_vis, px, vec4f(blended.x, blended.y, 0.0, 1.0));
}

// ---------------- Pass 3: Octahedral Border Copy ----------------
@group(0) @binding(1) var border_src: texture_2d<f32>;
@group(0) @binding(2) var border_dst: texture_storage_2d<rgba16float, write>;

fn copy_tile_pixel(px: vec2i, inner: i32, tile: i32) {
  let tx = px.x % tile;
  let ty = px.y % tile;
  let ox = px.x - tx;
  let oy = px.y - ty;
  var src = px;
  let last = tile - 1;
  if (tx == 0 && ty == 0) {
    src = vec2i(ox + inner, oy + inner);
  } else if (tx == last && ty == 0) {
    src = vec2i(ox + 1, oy + inner);
  } else if (tx == 0 && ty == last) {
    src = vec2i(ox + inner, oy + 1);
  } else if (tx == last && ty == last) {
    src = vec2i(ox + 1, oy + 1);
  } else if (ty == 0) {
    src = vec2i(ox + (last - tx), oy + 1);
  } else if (ty == last) {
    src = vec2i(ox + (last - tx), oy + inner);
  } else if (tx == 0) {
    src = vec2i(ox + 1, oy + (last - ty));
  } else if (tx == last) {
    src = vec2i(ox + inner, oy + (last - ty));
  }
  textureStore(border_dst, px, textureLoad(border_src, src, 0));
}

@compute @workgroup_size(8, 8, 1)
fn ddgi_border_irradiance(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(border_src, 0));
  let px = vec2i(id.xy);
  if (px.x >= dims.x || px.y >= dims.y) {
    return;
  }
  copy_tile_pixel(px, i32(IRR_INNER), i32(IRR_TILE));
}

@compute @workgroup_size(8, 8, 1)
fn ddgi_border_visibility(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(border_src, 0));
  let px = vec2i(id.xy);
  if (px.x >= dims.x || px.y >= dims.y) {
    return;
  }
  copy_tile_pixel(px, i32(VIS_INNER), i32(VIS_TILE));
}

// ---------------- Pass 4: Evaluate Arbitrary World Surface Query Points ----------------
struct DdgiQueryPoint {
  pos: vec4f,    // xyz = world position
  normal: vec4f, // xyz = unit surface normal
  view: vec4f,   // xyz = unit direction toward camera
};

@group(0) @binding(1) var eval_irr_tex: texture_2d<f32>;
@group(0) @binding(2) var eval_vis_tex: texture_2d<f32>;
@group(0) @binding(3) var eval_samp: sampler;
@group(0) @binding(4) var<storage, read> query_points: array<DdgiQueryPoint>;
@group(0) @binding(5) var<storage, read_write> query_out: array<vec4f>;

@compute @workgroup_size(64, 1, 1)
fn ddgi_eval_points(@builtin(global_invocation_id) id: vec3u) {
  let count = arrayLength(&query_points);
  if (id.x >= count) {
    return;
  }
  let q = query_points[id.x];
  let n = normalize(q.normal.xyz);
  let v = select(n, normalize(q.view.xyz), length(q.view.xyz) > 1e-5);
  let irr = ddgi_sample_irradiance(q.pos.xyz, n, v, u, eval_irr_tex, eval_vis_tex, eval_samp);
  query_out[id.x] = vec4f(irr, 1.0);
}

// ---------------- Pass 5: Pack Cascade into Flat Storage Buffer for In-Shader PBR Lookup ----------------
// Packed buffer layout (`array<vec4f>` bound at `@group(0) @binding(8)` in `environment.wgsl`):
//   [0]:          vec4f(cascade_count, 0, 0, 0)
//   [1 + c*6 +0]: vec4f(origin.xyz, gamma)
//   [1 + c*6 +1]: vec4f(spacing.xyz, normal_bias)
//   [1 + c*6 +2]: vec4f(nx, ny, nz, view_bias)
//   [1 + c*6 +3]: vec4f(scroll_x, scroll_y, scroll_z, irr_offset_vec4s)
//   [1 + c*6 +4]: vec4f(vis_offset_vec4s, strength, irr_w, irr_h)
//   [1 + c*6 +5]: vec4f(vis_w, vis_h, 0, 0)
//   [irr_offset .. +irr_w*irr_h]:              vec4f(irr.rgb, validity) per irradiance atlas texel
//   [vis_offset .. +(vis_w*vis_h + 1)/2]:      vec4f(m0.x, m0.y, m1.x, m1.y) (2 visibility texels per vec4f)
struct DdgiPackUniform {
  pack_info: vec4u, // x = cascade_idx, y = cascade_count, z = irr_offset_vec4s, w = vis_offset_vec4s
  strength: vec4f,  // x = indirect_strength, yzw = 0
};

@group(0) @binding(1) var<uniform> pack_u: DdgiPackUniform;
@group(0) @binding(2) var pack_irr_tex: texture_2d<f32>;
@group(0) @binding(3) var pack_vis_tex: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> packed_out: array<vec4f>;

@compute @workgroup_size(64, 1, 1)
fn ddgi_pack_cascade(@builtin(global_invocation_id) id: vec3u) {
  let irr_dims = vec2u(textureDimensions(pack_irr_tex, 0));
  let vis_dims = vec2u(textureDimensions(pack_vis_tex, 0));
  let irr_count = irr_dims.x * irr_dims.y;
  let vis_total = vis_dims.x * vis_dims.y;
  let vis_pairs = (vis_total + 1u) / 2u;
  let offsets_off = pack_u.pack_info.w + vis_pairs;
  let total_probes = u.grid_dims.x * u.grid_dims.y * u.grid_dims.z;

  if (id.x == 0u) {
    let base = 1u + pack_u.pack_info.x * 6u;
    if (pack_u.pack_info.x == 0u) {
      packed_out[0u] = vec4f(f32(pack_u.pack_info.y), 0.0, 0.0, 0.0);
    }
    packed_out[base + 0u] = vec4f(u.origin_and_hysteresis.xyz, u.sun_dir_and_gamma.w);
    packed_out[base + 1u] = vec4f(u.spacing_and_max_dist.xyz, u.sky_zenith_and_nbias.w);
    packed_out[base + 2u] = vec4f(f32(u.grid_dims.x), f32(u.grid_dims.y), f32(u.grid_dims.z), u.sky_horizon_and_vbias.w);
    packed_out[base + 3u] = vec4f(
      f32(u.scroll_and_lights.x),
      f32(u.scroll_and_lights.y),
      f32(u.scroll_and_lights.z),
      f32(pack_u.pack_info.z),
    );
    packed_out[base + 4u] = vec4f(
      f32(pack_u.pack_info.w),
      pack_u.strength.x,
      f32(irr_dims.x),
      f32(irr_dims.y),
    );
    packed_out[base + 5u] = vec4f(f32(vis_dims.x), f32(vis_dims.y), f32(offsets_off), 0.0);
  }

  let idx = id.x;
  if (idx < irr_count) {
    let px = vec2i(i32(idx % irr_dims.x), i32(idx / irr_dims.x));
    packed_out[pack_u.pack_info.z + idx] = textureLoad(pack_irr_tex, px, 0);
  }
  if (idx < vis_pairs) {
    let i0 = idx * 2u;
    let i1 = min(i0 + 1u, vis_total - 1u);
    let p0 = vec2i(i32(i0 % vis_dims.x), i32(i0 / vis_dims.x));
    let p1 = vec2i(i32(i1 % vis_dims.x), i32(i1 / vis_dims.x));
    let m0 = textureLoad(pack_vis_tex, p0, 0).rg;
    let m1 = textureLoad(pack_vis_tex, p1, 0).rg;
    packed_out[pack_u.pack_info.w + idx] = vec4f(m0.x, m0.y, m1.x, m1.y);
  }
  if (idx < total_probes) {
    packed_out[offsets_off + idx] = probe_offsets_ro[idx];
  }
}
