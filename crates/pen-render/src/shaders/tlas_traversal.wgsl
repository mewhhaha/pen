// Reusable 2-level TLAS/BLAS watertight ray-tracing traversal core.
// Shared by real-time RT shadows/RTAO/reflections/GI, DDGI probe tracing, and GPU lightmap baking.
//
// Expects the host shader to bind:
//   tlas_nodes: array<BvhNode>
//   tlas_instances: array<TlasInstance>
//   blas_nodes: array<BvhNode>
//   blas_tris: array<BlasTri>

struct BvhNode {
  aabb_min: vec3f,
  left_or_first: u32,
  aabb_max: vec3f,
  tri_count: u32,
};

struct BlasTri {
  v0: vec4f,
  v1: vec4f,
  v2: vec4f,
  n0: vec4f,
  n1: vec4f,
  n2: vec4f,
};

struct TlasInstance {
  inv_transform: mat4x4f,
  normal_matrix_0: vec4f,
  normal_matrix_1: vec4f,
  normal_matrix_2: vec4f,
  albedo_roughness: vec4f,  // rgb = linear albedo, a = roughness
  emissive_metallic: vec4f, // rgb = linear emissive, a = metallic
  blas_node_offset: u32,
  blas_node_count: u32,
  blas_tri_offset: u32,
  mask: u32,
};

struct TlasHit {
  hit: bool,
  front_face: bool,
  t: f32,
  world_normal: vec3f,
  albedo: vec3f,
  roughness: f32,
  emissive: vec3f,
  metallic: f32,
  instance_index: u32,
};

fn tlas_ray_aabb_hit(origin: vec3f, dir: vec3f, bmin: vec3f, bmax: vec3f, t_min: f32, t_max: f32) -> f32 {
  var t_enter = t_min;
  var t_exit = t_max;

  if (abs(dir.x) < 1e-12) {
    if (origin.x < bmin.x || origin.x > bmax.x) { return -1.0; }
  } else {
    let inv = 1.0 / dir.x;
    let t0 = (bmin.x - origin.x) * inv;
    let t1 = (bmax.x - origin.x) * inv;
    t_enter = max(t_enter, min(t0, t1));
    t_exit = min(t_exit, max(t0, t1));
  }
  if (abs(dir.y) < 1e-12) {
    if (origin.y < bmin.y || origin.y > bmax.y) { return -1.0; }
  } else {
    let inv = 1.0 / dir.y;
    let t0 = (bmin.y - origin.y) * inv;
    let t1 = (bmax.y - origin.y) * inv;
    t_enter = max(t_enter, min(t0, t1));
    t_exit = min(t_exit, max(t0, t1));
  }
  if (abs(dir.z) < 1e-12) {
    if (origin.z < bmin.z || origin.z > bmax.z) { return -1.0; }
  } else {
    let inv = 1.0 / dir.z;
    let t0 = (bmin.z - origin.z) * inv;
    let t1 = (bmax.z - origin.z) * inv;
    t_enter = max(t_enter, min(t0, t1));
    t_exit = min(t_exit, max(t0, t1));
  }

  return select(-1.0, t_enter, t_exit >= t_enter);
}

struct TlasRayShear {
  kx: u32,
  ky: u32,
  kz: u32,
  sx: f32,
  sy: f32,
  sz: f32,
};

fn tlas_build_ray_shear(dir: vec3f) -> TlasRayShear {
  let ad = abs(dir);
  var kz = 2u;
  if (ad.x >= ad.y && ad.x >= ad.z) {
    kz = 0u;
  } else if (ad.y >= ad.z) {
    kz = 1u;
  }
  let kx = (kz + 1u) % 3u;
  let ky = (kx + 1u) % 3u;
  var fx = kx;
  var fy = ky;
  if (dir[kz] < 0.0) {
    fx = ky;
    fy = kx;
  }
  let sz = 1.0 / dir[kz];
  let sx = dir[fx] * sz;
  let sy = dir[fy] * sz;
  return TlasRayShear(fx, fy, kz, sx, sy, sz);
}

// Returns vec3f(t, bary_u, bary_v); t < 0.0 on miss.
// Barycentric coordinates weight (1 - u - v) * v0 + u * v1 + v * v2.
fn tlas_ray_tri_watertight(origin: vec3f, rs: TlasRayShear, tri: BlasTri, t_min: f32, t_max: f32) -> vec3f {
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
    return vec3f(-1.0, 0.0, 0.0);
  }
  let det = u_e + v_e + w_e;
  if (det == 0.0) {
    return vec3f(-1.0, 0.0, 0.0);
  }

  let az = rs.sz * a[rs.kz];
  let bz = rs.sz * b[rs.kz];
  let cz = rs.sz * c[rs.kz];
  let t_scaled = u_e * az + v_e * bz + w_e * cz;

  if (det > 0.0) {
    if (t_scaled < t_min * det || t_scaled > t_max * det) {
      return vec3f(-1.0, 0.0, 0.0);
    }
  } else {
    if (t_scaled > t_min * det || t_scaled < t_max * det) {
      return vec3f(-1.0, 0.0, 0.0);
    }
  }
  let inv_det = 1.0 / det;
  // u_e corresponds to vertex 0, v_e to vertex 1, w_e to vertex 2.
  return vec3f(t_scaled * inv_det, v_e * inv_det, w_e * inv_det);
}

fn trace_tlas(
  origin: vec3f,
  dir: vec3f,
  t_min: f32,
  t_max: f32,
  mask: u32,
  any_hit: bool,
  tlas_node_count: u32,
  tlas_inst_count: u32,
  blas_tri_total: u32,
) -> TlasHit {
  var best = TlasHit(false, true, t_max, vec3f(0.0, 1.0, 0.0), vec3f(0.0), 0.5, vec3f(0.0), 0.0, 0u);
  if (tlas_node_count == 0u || tlas_inst_count == 0u || blas_tri_total == 0u) {
    return best;
  }

  var tlas_stack: array<u32, 24>;
  var t_sp: i32 = 0;
  tlas_stack[0] = 0u;

  var blas_stack: array<u32, 28>;

  while (t_sp >= 0) {
    let tn_idx = tlas_stack[t_sp];
    t_sp -= 1;
    if (tn_idx >= tlas_node_count) {
      continue;
    }
    let tn = tlas_nodes[tn_idx];
    if (tlas_ray_aabb_hit(origin, dir, tn.aabb_min, tn.aabb_max, t_min, best.t) < 0.0) {
      continue;
    }

    if (tn.tri_count > 0u) {
      let i_first = tn.left_or_first;
      let i_end = min(i_first + tn.tri_count, tlas_inst_count);
      for (var ii = i_first; ii < i_end; ii++) {
        let inst = tlas_instances[ii];
        if ((inst.mask & mask) == 0u || inst.blas_node_count == 0u) {
          continue;
        }
        // Transform ray to instance local space without normalizing dir so t is invariant!
        let local_origin = (inst.inv_transform * vec4f(origin, 1.0)).xyz;
        let local_dir = (inst.inv_transform * vec4f(dir, 0.0)).xyz;
        if (dot(local_dir, local_dir) < 1e-16) {
          continue;
        }
        let rs = tlas_build_ray_shear(local_dir);

        var b_sp: i32 = 0;
        blas_stack[0] = 0u;
        let b_off = inst.blas_node_offset;
        let b_cnt = inst.blas_node_count;
        let tri_off = inst.blas_tri_offset;

        while (b_sp >= 0) {
          let bn_rel = blas_stack[b_sp];
          b_sp -= 1;
          if (bn_rel >= b_cnt) {
            continue;
          }
          let bn = blas_nodes[b_off + bn_rel];
          if (tlas_ray_aabb_hit(local_origin, local_dir, bn.aabb_min, bn.aabb_max, t_min, best.t) < 0.0) {
            continue;
          }
          if (bn.tri_count > 0u) {
            let tf = tri_off + bn.left_or_first;
            let te = min(tf + bn.tri_count, blas_tri_total);
            for (var ti = tf; ti < te; ti++) {
              let tri = blas_tris[ti];
              let res = tlas_ray_tri_watertight(local_origin, rs, tri, t_min, best.t);
              if (res.x > 0.0) {
                best.hit = true;
                best.t = res.x;
                let bu = res.y;
                let bv = res.z;
                let bw = max(0.0, 1.0 - bu - bv);
                let local_n = tri.n0.xyz * bw + tri.n1.xyz * bu + tri.n2.xyz * bv;
                let nm = mat3x3f(inst.normal_matrix_0.xyz, inst.normal_matrix_1.xyz, inst.normal_matrix_2.xyz);
                var wn = nm * local_n;
                let nlen2 = dot(wn, wn);
                wn = select(vec3f(0.0, 1.0, 0.0), wn * inverseSqrt(nlen2), nlen2 > 1e-12);
                let is_front = dot(wn, dir) <= 0.0;
                if (!is_front) {
                  wn = -wn;
                }
                best.front_face = is_front;
                best.world_normal = wn;
                best.albedo = inst.albedo_roughness.rgb;
                best.roughness = inst.albedo_roughness.a;
                best.emissive = inst.emissive_metallic.rgb;
                best.metallic = inst.emissive_metallic.a;
                best.instance_index = ii;
                if (any_hit) {
                  return best;
                }
              }
            }
          } else {
            let left = bn.left_or_first;
            let right = bn.left_or_first + 1u;
            var dl = -1.0;
            if (left < b_cnt) {
              let nl = blas_nodes[b_off + left];
              dl = tlas_ray_aabb_hit(local_origin, local_dir, nl.aabb_min, nl.aabb_max, t_min, best.t);
            }
            var dr = -1.0;
            if (right < b_cnt) {
              let nr = blas_nodes[b_off + right];
              dr = tlas_ray_aabb_hit(local_origin, local_dir, nr.aabb_min, nr.aabb_max, t_min, best.t);
            }
            if (dl >= 0.0 && dr >= 0.0) {
              let near_c = select(right, left, dl < dr);
              let far_c = select(left, right, dl < dr);
              if (b_sp + 2 < 28) {
                b_sp += 1;
                blas_stack[b_sp] = far_c;
                b_sp += 1;
                blas_stack[b_sp] = near_c;
              }
            } else if (dl >= 0.0 && b_sp + 1 < 28) {
              b_sp += 1;
              blas_stack[b_sp] = left;
            } else if (dr >= 0.0 && b_sp + 1 < 28) {
              b_sp += 1;
              blas_stack[b_sp] = right;
            }
          }
        }
      }
    } else {
      let left = tn.left_or_first;
      let right = tn.left_or_first + 1u;
      var dl = -1.0;
      if (left < tlas_node_count) {
        let nl = tlas_nodes[left];
        dl = tlas_ray_aabb_hit(origin, dir, nl.aabb_min, nl.aabb_max, t_min, best.t);
      }
      var dr = -1.0;
      if (right < tlas_node_count) {
        let nr = tlas_nodes[right];
        dr = tlas_ray_aabb_hit(origin, dir, nr.aabb_min, nr.aabb_max, t_min, best.t);
      }
      if (dl >= 0.0 && dr >= 0.0) {
        let near_c = select(right, left, dl < dr);
        let far_c = select(left, right, dl < dr);
        if (t_sp + 2 < 24) {
          t_sp += 1;
          tlas_stack[t_sp] = far_c;
          t_sp += 1;
          tlas_stack[t_sp] = near_c;
        }
      } else if (dl >= 0.0 && t_sp + 1 < 24) {
        t_sp += 1;
        tlas_stack[t_sp] = left;
      } else if (dr >= 0.0 && t_sp + 1 < 24) {
        t_sp += 1;
        tlas_stack[t_sp] = right;
      }
    }
  }
  return best;
}
