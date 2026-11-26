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

struct TlasRayInv {
  inv_dir: vec3f,
  neg_origin_inv: vec3f,
  parallel: vec3<bool>,
};

fn tlas_ray_inv(origin: vec3f, dir: vec3f) -> TlasRayInv {
  let parallel = abs(dir) < vec3f(1e-12);
  let safe_dir = select(dir, vec3f(1.0), parallel);
  let inv_dir = 1.0 / safe_dir;
  return TlasRayInv(inv_dir, -origin * inv_dir, parallel);
}

fn tlas_ray_aabb_hit_inv(origin: vec3f, ri: TlasRayInv, bmin: vec3f, bmax: vec3f, t_min: f32, t_max: f32) -> f32 {
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

fn tlas_ray_aabb_hit(origin: vec3f, dir: vec3f, bmin: vec3f, bmax: vec3f, t_min: f32, t_max: f32) -> f32 {
  return tlas_ray_aabb_hit_inv(origin, tlas_ray_inv(origin, dir), bmin, bmax, t_min, t_max);
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
fn tlas_ray_tri_watertight_verts(origin: vec3f, rs: TlasRayShear, v0: vec3f, v1: vec3f, v2: vec3f, t_min: f32, t_max: f32) -> vec3f {
  let a = v0 - origin;
  let b = v1 - origin;
  let c = v2 - origin;

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

fn trace_tlas_any_hit(
  origin: vec3f,
  dir: vec3f,
  t_min: f32,
  t_max: f32,
  mask: u32,
  tlas_node_count: u32,
  tlas_inst_count: u32,
  blas_tri_total: u32,
) -> bool {
  if (tlas_node_count == 0u || tlas_inst_count == 0u || blas_tri_total == 0u) {
    return false;
  }
  let world_inv = tlas_ray_inv(origin, dir);
  let root_tn = tlas_nodes[0u];
  if (tlas_ray_aabb_hit_inv(origin, world_inv, root_tn.aabb_min, root_tn.aabb_max, t_min, t_max) < 0.0) {
    return false;
  }

  var tlas_stack: array<u32, 24>;
  var t_sp: i32 = -1;
  var blas_stack: array<u32, 28>;
  var cur_tn = 0u;

  loop {
    let tn = tlas_nodes[cur_tn];
    if (tn.tri_count > 0u) {
      let i_first = tn.left_or_first;
      let i_end = min(i_first + tn.tri_count, tlas_inst_count);
      for (var ii = i_first; ii < i_end; ii++) {
        let inst_mask = tlas_instances[ii].mask;
        let b_cnt = tlas_instances[ii].blas_node_count;
        if ((inst_mask & mask) == 0u || b_cnt == 0u) {
          continue;
        }
        let inv_xform = tlas_instances[ii].inv_transform;
        let local_origin = (inv_xform * vec4f(origin, 1.0)).xyz;
        let local_dir = (inv_xform * vec4f(dir, 0.0)).xyz;
        if (dot(local_dir, local_dir) < 1e-16) {
          continue;
        }
        let local_inv = tlas_ray_inv(local_origin, local_dir);
        let b_off = tlas_instances[ii].blas_node_offset;
        let root_bn = blas_nodes[b_off];
        if (tlas_ray_aabb_hit_inv(local_origin, local_inv, root_bn.aabb_min, root_bn.aabb_max, t_min, t_max) < 0.0) {
          continue;
        }
        let rs = tlas_build_ray_shear(local_dir);
        let tri_off = tlas_instances[ii].blas_tri_offset;
        var b_sp: i32 = -1;
        var cur_bn = 0u;
        loop {
          let bn = blas_nodes[b_off + cur_bn];
          if (bn.tri_count > 0u) {
            let tf = tri_off + bn.left_or_first;
            let te = min(tf + bn.tri_count, blas_tri_total);
            for (var ti = tf; ti < te; ti++) {
              let res = tlas_ray_tri_watertight_verts(
                local_origin,
                rs,
                blas_tris[ti].v0.xyz,
                blas_tris[ti].v1.xyz,
                blas_tris[ti].v2.xyz,
                t_min,
                t_max,
              );
              if (res.x > 0.0) {
                return true;
              }
            }
          } else {
            let left = bn.left_or_first;
            let right = left + 1u;
            var dl = -1.0;
            if (left < b_cnt) {
              let nl = blas_nodes[b_off + left];
              dl = tlas_ray_aabb_hit_inv(local_origin, local_inv, nl.aabb_min, nl.aabb_max, t_min, t_max);
            }
            var dr = -1.0;
            if (right < b_cnt) {
              let nr = blas_nodes[b_off + right];
              dr = tlas_ray_aabb_hit_inv(local_origin, local_inv, nr.aabb_min, nr.aabb_max, t_min, t_max);
            }
            if (dl >= 0.0 && dr >= 0.0) {
              if (b_sp + 1 < 28) {
                b_sp += 1;
                blas_stack[b_sp] = select(left, right, dl < dr);
              }
              cur_bn = select(right, left, dl < dr);
              continue;
            } else if (dl >= 0.0) {
              cur_bn = left;
              continue;
            } else if (dr >= 0.0) {
              cur_bn = right;
              continue;
            }
          }
          if (b_sp < 0) {
            break;
          }
          cur_bn = blas_stack[b_sp];
          b_sp -= 1;
        }
      }
    } else {
      let left = tn.left_or_first;
      let right = left + 1u;
      var dl = -1.0;
      if (left < tlas_node_count) {
        let nl = tlas_nodes[left];
        dl = tlas_ray_aabb_hit_inv(origin, world_inv, nl.aabb_min, nl.aabb_max, t_min, t_max);
      }
      var dr = -1.0;
      if (right < tlas_node_count) {
        let nr = tlas_nodes[right];
        dr = tlas_ray_aabb_hit_inv(origin, world_inv, nr.aabb_min, nr.aabb_max, t_min, t_max);
      }
      if (dl >= 0.0 && dr >= 0.0) {
        if (t_sp + 1 < 24) {
          t_sp += 1;
          tlas_stack[t_sp] = select(left, right, dl < dr);
        }
        cur_tn = select(right, left, dl < dr);
        continue;
      } else if (dl >= 0.0) {
        cur_tn = left;
        continue;
      } else if (dr >= 0.0) {
        cur_tn = right;
        continue;
      }
    }
    if (t_sp < 0) {
      break;
    }
    cur_tn = tlas_stack[t_sp];
    t_sp -= 1;
  }
  return false;
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

  let world_inv = tlas_ray_inv(origin, dir);
  let root_tn = tlas_nodes[0u];
  if (tlas_ray_aabb_hit_inv(origin, world_inv, root_tn.aabb_min, root_tn.aabb_max, t_min, t_max) < 0.0) {
    return best;
  }

  var hit_found = false;
  var best_t = t_max;
  var best_u = 0.0;
  var best_v = 0.0;
  var best_tri = 0u;
  var best_inst = 0u;

  var tlas_stack: array<u32, 24>;
  var tlas_stack_t: array<f32, 24>;
  var t_sp: i32 = -1;
  var cur_tn = 0u;

  var blas_stack: array<u32, 28>;
  var blas_stack_t: array<f32, 28>;

  loop {
    let tn = tlas_nodes[cur_tn];
    if (tn.tri_count > 0u) {
      let i_first = tn.left_or_first;
      let i_end = min(i_first + tn.tri_count, tlas_inst_count);
      for (var ii = i_first; ii < i_end; ii++) {
        let inst_mask = tlas_instances[ii].mask;
        let b_cnt = tlas_instances[ii].blas_node_count;
        if ((inst_mask & mask) == 0u || b_cnt == 0u) {
          continue;
        }
        // Transform ray to instance local space without normalizing dir so t is invariant!
        let inv_xform = tlas_instances[ii].inv_transform;
        let local_origin = (inv_xform * vec4f(origin, 1.0)).xyz;
        let local_dir = (inv_xform * vec4f(dir, 0.0)).xyz;
        if (dot(local_dir, local_dir) < 1e-16) {
          continue;
        }
        let local_inv = tlas_ray_inv(local_origin, local_dir);
        let b_off = tlas_instances[ii].blas_node_offset;
        let root_bn = blas_nodes[b_off];
        if (tlas_ray_aabb_hit_inv(local_origin, local_inv, root_bn.aabb_min, root_bn.aabb_max, t_min, best_t) < 0.0) {
          continue;
        }
        let rs = tlas_build_ray_shear(local_dir);
        let tri_off = tlas_instances[ii].blas_tri_offset;

        var b_sp: i32 = -1;
        var cur_bn = 0u;

        loop {
          let bn = blas_nodes[b_off + cur_bn];
          if (bn.tri_count > 0u) {
            let tf = tri_off + bn.left_or_first;
            let te = min(tf + bn.tri_count, blas_tri_total);
            for (var ti = tf; ti < te; ti++) {
              let res = tlas_ray_tri_watertight_verts(
                local_origin,
                rs,
                blas_tris[ti].v0.xyz,
                blas_tris[ti].v1.xyz,
                blas_tris[ti].v2.xyz,
                t_min,
                best_t,
              );
              if (res.x > 0.0) {
                if (any_hit) {
                  best.hit = true;
                  best.t = res.x;
                  best.instance_index = ii;
                  return best;
                }
                hit_found = true;
                best_t = res.x;
                best_u = res.y;
                best_v = res.z;
                best_tri = ti;
                best_inst = ii;
              }
            }
          } else {
            let left = bn.left_or_first;
            let right = left + 1u;
            var dl = -1.0;
            if (left < b_cnt) {
              let nl = blas_nodes[b_off + left];
              dl = tlas_ray_aabb_hit_inv(local_origin, local_inv, nl.aabb_min, nl.aabb_max, t_min, best_t);
            }
            var dr = -1.0;
            if (right < b_cnt) {
              let nr = blas_nodes[b_off + right];
              dr = tlas_ray_aabb_hit_inv(local_origin, local_inv, nr.aabb_min, nr.aabb_max, t_min, best_t);
            }
            if (dl >= 0.0 && dr >= 0.0) {
              let left_near = dl < dr;
              if (b_sp + 1 < 28) {
                b_sp += 1;
                blas_stack[b_sp] = select(left, right, left_near);
                blas_stack_t[b_sp] = select(dl, dr, left_near);
              }
              cur_bn = select(right, left, left_near);
              continue;
            } else if (dl >= 0.0) {
              cur_bn = left;
              continue;
            } else if (dr >= 0.0) {
              cur_bn = right;
              continue;
            }
          }
          var popped_b = false;
          while (b_sp >= 0) {
            let next_b = blas_stack[b_sp];
            let next_bt = blas_stack_t[b_sp];
            b_sp -= 1;
            if (next_bt <= best_t) {
              cur_bn = next_b;
              popped_b = true;
              break;
            }
          }
          if (!popped_b) {
            break;
          }
        }
      }
    } else {
      let left = tn.left_or_first;
      let right = left + 1u;
      var dl = -1.0;
      if (left < tlas_node_count) {
        let nl = tlas_nodes[left];
        dl = tlas_ray_aabb_hit_inv(origin, world_inv, nl.aabb_min, nl.aabb_max, t_min, best_t);
      }
      var dr = -1.0;
      if (right < tlas_node_count) {
        let nr = tlas_nodes[right];
        dr = tlas_ray_aabb_hit_inv(origin, world_inv, nr.aabb_min, nr.aabb_max, t_min, best_t);
      }
      if (dl >= 0.0 && dr >= 0.0) {
        let left_near = dl < dr;
        if (t_sp + 1 < 24) {
          t_sp += 1;
          tlas_stack[t_sp] = select(left, right, left_near);
          tlas_stack_t[t_sp] = select(dl, dr, left_near);
        }
        cur_tn = select(right, left, left_near);
        continue;
      } else if (dl >= 0.0) {
        cur_tn = left;
        continue;
      } else if (dr >= 0.0) {
        cur_tn = right;
        continue;
      }
    }
    var popped_t = false;
    while (t_sp >= 0) {
      let next_t = tlas_stack[t_sp];
      let next_tt = tlas_stack_t[t_sp];
      t_sp -= 1;
      if (next_tt <= best_t) {
        cur_tn = next_t;
        popped_t = true;
        break;
      }
    }
    if (!popped_t) {
      break;
    }
  }

  if (hit_found) {
    let inst = tlas_instances[best_inst];
    let tri = blas_tris[best_tri];
    let bw = max(0.0, 1.0 - best_u - best_v);
    let local_n = tri.n0.xyz * bw + tri.n1.xyz * best_u + tri.n2.xyz * best_v;
    let nm = mat3x3f(inst.normal_matrix_0.xyz, inst.normal_matrix_1.xyz, inst.normal_matrix_2.xyz);
    var wn = nm * local_n;
    let nlen2 = dot(wn, wn);
    wn = select(vec3f(0.0, 1.0, 0.0), wn * inverseSqrt(nlen2), nlen2 > 1e-12);
    let is_front = dot(wn, dir) <= 0.0;
    if (!is_front) {
      wn = -wn;
    }
    best.hit = true;
    best.front_face = is_front;
    best.t = best_t;
    best.world_normal = wn;
    best.albedo = inst.albedo_roughness.rgb;
    best.roughness = inst.albedo_roughness.a;
    best.emissive = inst.emissive_metallic.rgb;
    best.metallic = inst.emissive_metallic.a;
    best.instance_index = best_inst;
  }
  return best;
}
