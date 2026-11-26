// Portable GPU BVH Ray-Traced Soft Shadows & Ray-Traced Ambient Occlusion (RTAO).
// - Explicit u32 BVH node fields (no float/u32 bitcast hazards)
// - Signed parallel-axis-safe slab AABB test
// - Watertight Woop-Benthin-Wald (JCGT 2013) shear-space ray-triangle intersection
// - Scale-relative normal/bias reconstruction and continuous RTAO radius falloff.

struct BvhNode {
  aabb_min: vec3f,
  left_or_first: u32,
  aabb_max: vec3f,
  tri_count: u32,
};

struct BvhTri {
  v0: vec4f,
  v1: vec4f,
  v2: vec4f,
};

struct RayTraceUniform {
  inverse_view_projection: mat4x4f,
  camera_pos_and_clear: vec4f, // xyz = camera pos, w = depth clear value (1.0 standard, 0.0 rev-Z)
  sun_dir_and_radius: vec4f,   // xyz = normalized toward sun, w = sun angular cone radius (0 = hard shadow)
  // x = ao_radius, y = ao_samples (0..16), z = shadow_samples (1..8), w = frame_seed
  ao_shadow: vec4f,
  // x = min_normal_bias, yzw = reserved
  bias_params: vec4f,
  // x = node_count, y = tri_count, zw = reserved
  counts: vec4u,
};

@group(0) @binding(0) var<uniform> u: RayTraceUniform;
@group(0) @binding(1) var<storage, read> nodes: array<BvhNode>;
@group(0) @binding(2) var<storage, read> tris: array<BvhTri>;
@group(0) @binding(3) var depth_tex: texture_2d<f32>;
@group(0) @binding(4) var out_tex: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265359;

fn hash23(p3: vec3f) -> vec2f {
  var p = fract(p3 * vec3f(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yzx + 33.33);
  return fract((p.xx + p.yz) * p.zy);
}

struct BvhRayInv {
  inv_dir: vec3f,
  neg_origin_inv: vec3f,
  parallel: vec3<bool>,
};

fn bvh_ray_inv(origin: vec3f, dir: vec3f) -> BvhRayInv {
  let parallel = abs(dir) < vec3f(1e-12);
  let safe_dir = select(dir, vec3f(1.0), parallel);
  let inv_dir = 1.0 / safe_dir;
  return BvhRayInv(inv_dir, -origin * inv_dir, parallel);
}

// Exact slab AABB intersection handling zero/parallel ray components without losing negative signs.
fn ray_aabb_hit_inv(origin: vec3f, ri: BvhRayInv, bmin: vec3f, bmax: vec3f, t_min: f32, t_max: f32) -> f32 {
  if (any(ri.parallel & ((origin < bmin) | (origin > bmax)))) {
    return -1.0;
  }
  let t0 = bmin * ri.inv_dir + ri.neg_origin_inv;
  let t1 = bmax * ri.inv_dir + ri.neg_origin_inv;
  let near_v = select(min(t0, t1), vec3f(t_min), ri.parallel);
  let far_v = select(max(t0, t1), vec3f(t_max), ri.parallel);
  let t_enter = max(t_min, max(near_v.x, max(near_v.y, near_v.z)));
  let t_exit = min(t_max, min(far_v.x, min(far_v.y, far_v.z)));
  return select(-1.0, t_enter, t_exit >= t_enter);
}

// Precomputed shear transform for watertight Woop-Benthin-Wald (JCGT 2013) ray-triangle intersection.
struct RayShear {
  kx: u32,
  ky: u32,
  kz: u32,
  sx: f32,
  sy: f32,
  sz: f32,
};

fn build_ray_shear(dir: vec3f) -> RayShear {
  let ad = abs(dir);
  var kz = 2u;
  if (ad.x >= ad.y && ad.x >= ad.z) {
    kz = 0u;
  } else if (ad.y >= ad.z) {
    kz = 1u;
  }
  let kx = (kz + 1u) % 3u;
  let ky = (kx + 1u) % 3u;
  // Preserve winding sign when major axis direction is negative.
  var fx = kx;
  var fy = ky;
  if (dir[kz] < 0.0) {
    fx = ky;
    fy = kx;
  }
  let sz = 1.0 / dir[kz];
  let sx = dir[fx] * sz;
  let sy = dir[fy] * sz;
  return RayShear(fx, fy, kz, sx, sy, sz);
}

fn ray_tri_watertight(origin: vec3f, rs: RayShear, tri: BvhTri, t_min: f32, t_max: f32) -> f32 {
  let a = tri.v0.xyz - origin;
  let b = tri.v1.xyz - origin;
  let c = tri.v2.xyz - origin;

  let ax = a[rs.kx] - rs.sx * a[rs.kz];
  let ay = a[rs.ky] - rs.sy * a[rs.kz];
  let bx = b[rs.kx] - rs.sx * b[rs.kz];
  let by = b[rs.ky] - rs.sy * b[rs.kz];
  let cx = c[rs.kx] - rs.sx * c[rs.kz];
  let cy = c[rs.ky] - rs.sy * c[rs.kz];

  let u_e = cx * by - cy * bx;
  let v_e = ax * cy - ay * cx;
  let w_e = bx * ay - by * ax;

  if ((u_e < 0.0 || v_e < 0.0 || w_e < 0.0) && (u_e > 0.0 || v_e > 0.0 || w_e > 0.0)) {
    return -1.0;
  }
  let det = u_e + v_e + w_e;
  if (det == 0.0) {
    return -1.0;
  }

  let az = rs.sz * a[rs.kz];
  let bz = rs.sz * b[rs.kz];
  let cz = rs.sz * c[rs.kz];
  let t_scaled = u_e * az + v_e * bz + w_e * cz;

  if (det > 0.0) {
    if (t_scaled < t_min * det || t_scaled > t_max * det) {
      return -1.0;
    }
  } else {
    if (t_scaled > t_min * det || t_scaled < t_max * det) {
      return -1.0;
    }
  }
  return t_scaled / det;
}

fn trace_bvh(origin: vec3f, dir: vec3f, t_min: f32, t_max: f32, any_hit: bool) -> f32 {
  let node_count = u.counts.x;
  let tri_count_total = u.counts.y;
  if (node_count == 0u || tri_count_total == 0u) {
    return -1.0;
  }
  let ri = bvh_ray_inv(origin, dir);
  let root = nodes[0u];
  if (ray_aabb_hit_inv(origin, ri, root.aabb_min, root.aabb_max, t_min, t_max) < 0.0) {
    return -1.0;
  }
  let rs = build_ray_shear(dir);
  var stack: array<u32, 32>;
  var stack_t: array<f32, 32>;
  var sp: i32 = -1;
  var cur = 0u;
  var closest = t_max;
  var found = false;

  loop {
    let node = nodes[cur];
    if (node.tri_count > 0u) {
      let first = node.left_or_first;
      let end = min(first + node.tri_count, tri_count_total);
      for (var i = first; i < end; i++) {
        let t = ray_tri_watertight(origin, rs, tris[i], t_min, closest);
        if (t > 0.0) {
          closest = t;
          found = true;
          if (any_hit) {
            return t;
          }
        }
      }
    } else {
      let left = node.left_or_first;
      let right = left + 1u;
      var dl = -1.0;
      if (left < node_count) {
        let nl = nodes[left];
        dl = ray_aabb_hit_inv(origin, ri, nl.aabb_min, nl.aabb_max, t_min, closest);
      }
      var dr = -1.0;
      if (right < node_count) {
        let nr = nodes[right];
        dr = ray_aabb_hit_inv(origin, ri, nr.aabb_min, nr.aabb_max, t_min, closest);
      }
      if (dl >= 0.0 && dr >= 0.0) {
        let left_near = dl < dr;
        if (sp + 1 < 32) {
          sp += 1;
          stack[sp] = select(left, right, left_near);
          stack_t[sp] = select(dl, dr, left_near);
        }
        cur = select(right, left, left_near);
        continue;
      } else if (dl >= 0.0) {
        cur = left;
        continue;
      } else if (dr >= 0.0) {
        cur = right;
        continue;
      }
    }
    var popped = false;
    while (sp >= 0) {
      let next_n = stack[sp];
      let next_t = stack_t[sp];
      sp -= 1;
      if (next_t <= closest) {
        cur = next_n;
        popped = true;
        break;
      }
    }
    if (!popped) {
      break;
    }
  }
  return select(-1.0, closest, found);
}

struct DepthPt {
  world: vec3f,
  valid: bool,
};

fn load_world(px: vec2i, dims: vec2i) -> DepthPt {
  if (px.x < 0 || px.y < 0 || px.x >= dims.x || px.y >= dims.y) {
    return DepthPt(vec3f(0.0), false);
  }
  let d = textureLoad(depth_tex, px, 0).r;
  if (abs(d - u.camera_pos_and_clear.w) <= 1e-6) {
    return DepthPt(vec3f(0.0), false);
  }
  let ndc = vec2f(
    (f32(px.x) + 0.5) / f32(dims.x) * 2.0 - 1.0,
    1.0 - (f32(px.y) + 0.5) / f32(dims.y) * 2.0,
  );
  let clip = u.inverse_view_projection * vec4f(ndc, d, 1.0);
  if (abs(clip.w) <= 1e-6) {
    return DepthPt(vec3f(0.0), false);
  }
  return DepthPt(clip.xyz / clip.w, true);
}

fn basis_from_normal(n: vec3f) -> mat3x3f {
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t = normalize(cross(up, n));
  let b = cross(n, t);
  return mat3x3f(t, b, n);
}

@compute @workgroup_size(8, 8, 1)
fn raytrace_shadows_ao(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(depth_tex, 0));
  if (i32(id.x) >= dims.x || i32(id.y) >= dims.y) {
    return;
  }
  let px = vec2i(id.xy);
  let center = load_world(px, dims);
  if (!center.valid) {
    textureStore(out_tex, px, vec4f(1.0, 1.0, 1.0, 1.0));
    return;
  }
  let pos = center.world;

  // Scale-relative geometric normal reconstruction from valid same-surface neighbors.
  let sr = load_world(px + vec2i(1, 0), dims);
  let sl = load_world(px - vec2i(1, 0), dims);
  let sd = load_world(px + vec2i(0, 1), dims);
  let su = load_world(px - vec2i(0, 1), dims);
  var dx = vec3f(1.0, 0.0, 0.0);
  var has_dx = true;
  if (sr.valid && sl.valid) {
    let dr = sr.world - pos;
    let dl = pos - sl.world;
    dx = select(dl, dr, dot(dr, dr) < dot(dl, dl));
  } else if (sr.valid) {
    dx = sr.world - pos;
  } else if (sl.valid) {
    dx = pos - sl.world;
  } else {
    has_dx = false;
  }
  var dy = vec3f(0.0, 0.0, 1.0);
  var has_dy = true;
  if (sd.valid && su.valid) {
    let dd = sd.world - pos;
    let du = pos - su.world;
    dy = select(du, dd, dot(dd, dd) < dot(du, du));
  } else if (sd.valid) {
    dy = sd.world - pos;
  } else if (su.valid) {
    dy = pos - su.world;
  } else {
    has_dy = false;
  }
  var n = vec3f(0.0, 1.0, 0.0);
  if (has_dx && has_dy) {
    let raw_n = cross(dx, dy);
    let area2 = dot(raw_n, raw_n);
    let ref2 = dot(dx, dx) * dot(dy, dy);
    if (area2 > 1e-12 * ref2 && ref2 > 0.0) {
      n = raw_n * inverseSqrt(area2);
    }
  }
  let to_cam = u.camera_pos_and_clear.xyz - pos;
  let cam_dist = length(to_cam);
  if (dot(n, to_cam) < 0.0) {
    n = -n;
  }

  // Scale-relative normal bias avoids self-intersection at distance without skipping close contacts.
  let min_bias = max(u.bias_params.x, 1e-5);
  let pixel_scale = max(length(dx), length(dy)) * 0.05;
  let bias = max(min_bias, min(pixel_scale, cam_dist * 2e-4));
  let ray_origin = pos + n * bias;

  // 1. Ray-traced hard/soft penumbra sun visibility.
  let sun_raw = u.sun_dir_and_radius.xyz;
  let sun_dir = select(vec3f(0.0, 1.0, 0.0), normalize(sun_raw), length(sun_raw) > 1e-6);
  let sun_radius = max(u.sun_dir_and_radius.w, 0.0);
  let sh_samples = clamp(u32(round(u.ao_shadow.z)), 1u, 8u);
  let sun_basis = basis_from_normal(sun_dir);
  var sun_vis = 0.0;
  if (dot(n, sun_dir) > 0.0) {
    for (var s = 0u; s < sh_samples; s++) {
      var d_ray = sun_dir;
      if (sun_radius > 1e-5) {
        let rnd = hash23(vec3f(vec2f(id.xy), u.ao_shadow.w * 13.7 + f32(s) * 5.3));
        let r_disk = sqrt(rnd.x) * sun_radius;
        let phi = rnd.y * 2.0 * PI;
        d_ray = normalize(sun_basis * vec3f(r_disk * cos(phi), r_disk * sin(phi), 1.0));
      }
      let hit = trace_bvh(ray_origin, d_ray, 1e-4, 1e4, true);
      if (hit < 0.0) {
        sun_vis += 1.0;
      }
    }
    sun_vis /= f32(sh_samples);
  }

  // 2. Cosine-weighted hemispherical Ray-Traced Ambient Occlusion (RTAO) with smooth boundary falloff.
  let ao_samples = clamp(u32(round(u.ao_shadow.y)), 0u, 16u);
  let ao_radius = max(u.ao_shadow.x, 0.05);
  let n_basis = basis_from_normal(n);
  var ao_vis = 1.0;
  var dist_sum = 0.0;
  if (ao_samples > 0u) {
    var open = 0.0;
    for (var s = 0u; s < ao_samples; s++) {
      let rnd = hash23(vec3f(vec2f(id.xy) + vec2f(19.1, 7.3), u.ao_shadow.w * 17.3 + f32(s) * 11.1));
      let phi = 2.0 * PI * ((f32(s) + rnd.x) / f32(ao_samples));
      let cos_t = sqrt(clamp(1.0 - rnd.y, 0.0, 1.0));
      let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
      let local_dir = vec3f(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);
      let d_ray = normalize(n_basis * local_dir);
      let hit = trace_bvh(ray_origin, d_ray, 1e-4, ao_radius, false);
      if (hit < 0.0) {
        open += 1.0;
        dist_sum += 1.0;
      } else {
        // Smooth C1 falloff reaching 1.0 continuously at `hit == ao_radius` (zero radius popping).
        let frac = clamp(hit / ao_radius, 0.0, 1.0);
        open += smoothstep(0.0, 1.0, frac);
        dist_sum += frac;
      }
    }
    ao_vis = clamp(open / f32(ao_samples), 0.0, 1.0);
    dist_sum /= f32(ao_samples);
  }

  textureStore(out_tex, px, vec4f(sun_vis, ao_vis, dist_sum, 1.0));
}
