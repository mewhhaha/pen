enable wgpu_ray_query;

// Hardware-accelerated 2-level TLAS/BLAS traversal using `wgpu_ray_query`
// (Vulkan KHR_ray_query / DXR / Metal RT). Drop-in replacement for `tlas_traversal.wgsl`.
//
// Expects the host shader to bind:
//   hw_tlas: acceleration_structure
//   tlas_instances: array<TlasInstance>
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
  if (tlas_inst_count == 0u || blas_tri_total == 0u) {
    return false;
  }
  var rq: ray_query;
  rayQueryInitialize(
    &rq,
    hw_tlas,
    RayDesc(RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_TERMINATE_ON_FIRST_HIT, mask & 0xffu, t_min, t_max, origin, dir),
  );
  rayQueryProceed(&rq);
  let inter = rayQueryGetCommittedIntersection(&rq);
  return inter.kind == RAY_QUERY_INTERSECTION_TRIANGLE && inter.instance_custom_data < tlas_inst_count;
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
  if (tlas_inst_count == 0u || blas_tri_total == 0u) {
    return best;
  }

  var flags = RAY_FLAG_FORCE_OPAQUE;
  if (any_hit) {
    flags = flags | RAY_FLAG_TERMINATE_ON_FIRST_HIT;
  }

  var rq: ray_query;
  rayQueryInitialize(&rq, hw_tlas, RayDesc(flags, mask & 0xffu, t_min, t_max, origin, dir));
  rayQueryProceed(&rq);
  let inter = rayQueryGetCommittedIntersection(&rq);
  if (inter.kind != RAY_QUERY_INTERSECTION_TRIANGLE) {
    return best;
  }

  let inst_idx = inter.instance_custom_data;
  if (inst_idx >= tlas_inst_count) {
    return best;
  }
  if (any_hit) {
    best.hit = true;
    best.t = inter.t;
    best.instance_index = inst_idx;
    return best;
  }
  let inst = tlas_instances[inst_idx];
  let tri_idx = inst.blas_tri_offset + inter.primitive_index;
  if (tri_idx >= blas_tri_total) {
    return best;
  }
  let tri = blas_tris[tri_idx];

  let bu = inter.barycentrics.x;
  let bv = inter.barycentrics.y;
  let bw = 1.0 - bu - bv;
  let local_n = tri.n0.xyz * bw + tri.n1.xyz * bu + tri.n2.xyz * bv;
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
  best.t = inter.t;
  best.world_normal = wn;
  best.albedo = inst.albedo_roughness.rgb;
  best.roughness = inst.albedo_roughness.a;
  best.emissive = inst.emissive_metallic.rgb;
  best.metallic = inst.emissive_metallic.a;
  best.instance_index = inst_idx;
  return best;
}
