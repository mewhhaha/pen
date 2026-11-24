// Dynamic Diffuse Global Illumination (DDGI, Majercik et al., JCGT 2019).
// Prefixed at compile time with tlas_traversal.wgsl.
//
// Atlas layout:
//  - Probe (ix, iy, iz) -> linear index = (iy * nz + iz) * nx + ix
//  - 2D tile grid: col = iz * nx + ix (0 .. nx*nz), row = iy (0 .. ny)
//  - Irradiance tile: 8x8 interior + 1px octahedral wrap border = 10x10 per probe
//  - Visibility tile: 16x16 interior + 1px octahedral wrap border = 18x18 per probe

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
};

const DDGI_PI: f32 = 3.14159265359;
const IRR_INNER: u32 = 8u;
const IRR_TILE: u32 = 10u;
const VIS_INNER: u32 = 16u;
const VIS_TILE: u32 = 18u;

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

fn ddgi_probe_world_pos(grid_coord: vec3u, u: DdgiUniform) -> vec3f {
  return u.origin_and_hysteresis.xyz + vec3f(grid_coord) * u.spacing_and_max_dist.xyz;
}

fn ddgi_probe_uv(
  grid_coord: vec3u,
  dir: vec3f,
  dims: vec3u,
  inner: u32,
  tile: u32,
) -> vec2f {
  let col = grid_coord.z * dims.x + grid_coord.x;
  let row = grid_coord.y;
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

  var sum_irr = vec3f(0.0);
  var sum_w = 0.0;

  for (var i = 0u; i < 8u; i++) {
    let off = vec3u(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
    let coord = min(base_i + off, dims - vec3u(1u));
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
    let vis_uv = ddgi_probe_uv(coord, dir_from_probe, dims, VIS_INNER, VIS_TILE);
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

    w = max(w, 1e-6);
    let irr_uv = ddgi_probe_uv(coord, normal, dims, IRR_INNER, IRR_TILE);
    let encoded = max(textureSampleLevel(irr_atlas, atlas_samp, irr_uv, 0.0).rgb, vec3f(0.0));
    let irr = pow(encoded, vec3f(gamma * 0.5));
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
@group(0) @binding(1) var<storage, read> tlas_nodes: array<BvhNode>;
@group(0) @binding(2) var<storage, read> tlas_instances: array<TlasInstance>;
@group(0) @binding(3) var<storage, read> blas_nodes: array<BvhNode>;
@group(0) @binding(4) var<storage, read> blas_tris: array<BlasTri>;
@group(0) @binding(5) var prev_irr_tex: texture_2d<f32>;
@group(0) @binding(6) var prev_vis_tex: texture_2d<f32>;
@group(0) @binding(7) var atlas_sampler: sampler;
@group(0) @binding(8) var<storage, read_write> ray_radiance: array<vec4f>;

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

  let hit_pos = probe_pos + ray_dir * h.t;
  let sun_raw = u.sun_dir_and_gamma.xyz;
  let sun_dir = select(vec3f(0.0, 1.0, 0.0), normalize(sun_raw), length(sun_raw) > 1e-6);
  let diffuse_albedo = h.albedo * (1.0 - h.metallic);

  var rad = h.emissive;
  let ndl = max(0.0, dot(h.world_normal, sun_dir));
  if (ndl > 0.0) {
    let sh = trace_tlas(
      hit_pos + h.world_normal * 0.01,
      sun_dir,
      1e-3,
      max_dist,
      mask,
      true,
      u.tlas_counts.x,
      u.tlas_counts.y,
      u.tlas_counts.z,
    );
    if (!sh.hit) {
      rad += (diffuse_albedo / DDGI_PI) * u.sun_color_and_sharp.rgb * ndl;
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

@compute @workgroup_size(8, 8, 1)
fn ddgi_blend_irradiance(@builtin(global_invocation_id) id: vec3u) {
  let dims = u.grid_dims.xyz;
  let cols = dims.x * dims.z;
  let rows = dims.y;
  let tx = id.x % IRR_INNER;
  let ty = id.y % IRR_INNER;
  let col = id.x / IRR_INNER;
  let row = id.y / IRR_INNER;
  if (col >= cols || row >= rows) {
    return;
  }

  let ix = col % dims.x;
  let iz = col / dims.x;
  let iy = row;
  let probe_idx = (iy * dims.z + iz) * dims.x + ix;
  let rays_per_probe = max(u.grid_dims.w, 1u);
  let rot = mat3x3f(u.rot_col0.xyz, u.rot_col1.xyz, u.rot_col2.xyz);

  let oct = ((vec2f(f32(tx), f32(ty)) + vec2f(0.5)) / f32(IRR_INNER)) * 2.0 - vec2f(1.0);
  let texel_dir = ddgi_oct_decode(oct);

  var acc = vec3f(0.0);
  var weight_sum = 0.0;
  let base_ray = probe_idx * rays_per_probe;
  for (var r = 0u; r < rays_per_probe; r++) {
    let ray_dir = normalize(rot * ddgi_spherical_fibonacci(r, rays_per_probe));
    let w = max(0.0, dot(texel_dir, ray_dir));
    if (w > 1e-4) {
      let sample_rad = blend_rays[base_ray + r].rgb;
      acc += sample_rad * w;
      weight_sum += w;
    }
  }
  var new_irr = select(vec3f(0.0), acc / weight_sum, weight_sum > 1e-5);
  let gamma = max(u.sun_dir_and_gamma.w, 1.0);
  new_irr = pow(max(new_irr, vec3f(0.0)), vec3f(1.0 / gamma));

  let px = vec2i(i32(col * IRR_TILE + 1u + tx), i32(row * IRR_TILE + 1u + ty));
  let prev = textureLoad(blend_prev_irr, px, 0).rgb;
  let h = clamp(u.origin_and_hysteresis.w, 0.0, 0.99);
  let blended = mix(new_irr, prev, h);
  textureStore(blend_out_irr, px, vec4f(blended, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn ddgi_blend_visibility(@builtin(global_invocation_id) id: vec3u) {
  let dims = u.grid_dims.xyz;
  let cols = dims.x * dims.z;
  let rows = dims.y;
  let tx = id.x % VIS_INNER;
  let ty = id.y % VIS_INNER;
  let col = id.x / VIS_INNER;
  let row = id.y / VIS_INNER;
  if (col >= cols || row >= rows) {
    return;
  }

  let ix = col % dims.x;
  let iz = col / dims.x;
  let iy = row;
  let probe_idx = (iy * dims.z + iz) * dims.x + ix;
  let rays_per_probe = max(u.grid_dims.w, 1u);
  let rot = mat3x3f(u.rot_col0.xyz, u.rot_col1.xyz, u.rot_col2.xyz);

  let oct = ((vec2f(f32(tx), f32(ty)) + vec2f(0.5)) / f32(VIS_INNER)) * 2.0 - vec2f(1.0);
  let texel_dir = ddgi_oct_decode(oct);

  let max_dist = max(u.spacing_and_max_dist.w, 1.0);
  let max_probe_dist = length(u.spacing_and_max_dist.xyz) * 1.5;
  let sharp = max(u.sun_color_and_sharp.w, 1.0);

  var acc_d = 0.0;
  var acc_d2 = 0.0;
  var weight_sum = 0.0;
  let base_ray = probe_idx * rays_per_probe;
  for (var r = 0u; r < rays_per_probe; r++) {
    let ray_dir = normalize(rot * ddgi_spherical_fibonacci(r, rays_per_probe));
    let w = pow(max(0.0, dot(texel_dir, ray_dir)), sharp);
    if (w > 1e-6) {
      let dist = min(abs(blend_rays[base_ray + r].w), min(max_dist, max_probe_dist));
      acc_d += dist * w;
      acc_d2 += dist * dist * w;
      weight_sum += w;
    }
  }
  let new_m = select(vec2f(max_probe_dist, max_probe_dist * max_probe_dist), vec2f(acc_d, acc_d2) / weight_sum, weight_sum > 1e-6);

  let px = vec2i(i32(col * VIS_TILE + 1u + tx), i32(row * VIS_TILE + 1u + ty));
  let prev = textureLoad(blend_prev_vis, px, 0).rg;
  let h = clamp(u.origin_and_hysteresis.w, 0.0, 0.99);
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
