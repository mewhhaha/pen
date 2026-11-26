// Meshlet DAG cut and culling. Packed draws and diagnostics share an atomic slot.
struct DagClusterGpu {
  bounds_sphere: vec4f,
  cone_axis_cutoff: vec4f,
  lod_sphere: vec4f,
  parent_sphere: vec4f,
  errors: vec4f,
  counts: vec4u,
  metadata: vec4u,
};

struct CullUniform {
  model: mat4x4f,
  hiz_vp: mat4x4f,
  // Pre-transformed, unit-length local-space frustum planes: normalize(transpose(model) * world_plane).
  local_planes: array<vec4f, 6>,
  camera_pos: vec4f,
  // Pre-computed inverse-transformed camera position for backface cone test.
  local_cam: vec4f,
  // cluster count, Hi-Z enabled, Hi-Z mip count, backface/reversed-Z flags
  counts_flags: vec4u,
  // model scale bound, projection scale in pixels, near plane, pixel error threshold
  projection: vec4f,
};

struct DrawIndexedIndirect {
  index_count: u32,
  instance_count: u32,
  first_index: u32,
  base_vertex: i32,
  first_instance: u32,
};

@group(0) @binding(0) var<uniform> u: CullUniform;
@group(0) @binding(1) var<storage, read> clusters: array<DagClusterGpu>;
@group(0) @binding(2) var<storage, read_write> draws: array<DrawIndexedIndirect>;
@group(0) @binding(3) var<storage, read_write> draw_count: atomic<u32>;
@group(0) @binding(4) var hiz_tex: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> selected_clusters: array<u32>;
@group(0) @binding(6) var<storage, read_write> phase2_flags: array<u32>;

// Exact local-space frustum test using CPU-precomputed normalized local planes.
fn in_frustum_local(local_center: vec3f, local_radius: f32) -> bool {
  let limit = -(local_radius * 1.001 + 1e-5);
  for (var i = 0u; i < 6u; i++) {
    let lp = u.local_planes[i];
    if (dot(lp.xyz, local_center) + lp.w < limit) {
      return false;
    }
  }
  return true;
}

fn hiz_occluded(world_center: vec3f, world_radius: f32, rev_z: bool) -> bool {
  if (u.counts_flags.y == 0u || u.counts_flags.z == 0u) {
    return false;
  }
  let mips = min(u.counts_flags.z, textureNumLevels(hiz_tex));
  let r = world_radius * 1.02 + 0.005;
  let clip0 = u.hiz_vp * vec4f(world_center - vec3f(r), 1.0);
  let dx = u.hiz_vp[0] * (2.0 * r);
  let dy = u.hiz_vp[1] * (2.0 * r);
  let dz = u.hiz_vp[2] * (2.0 * r);
  var min_uv = vec2f(1e9);
  var max_uv = vec2f(-1e9);
  var near_z = select(1.0, 0.0, rev_z);
  for (var c = 0u; c < 8u; c++) {
    var clip = clip0;
    if ((c & 1u) != 0u) { clip += dx; }
    if ((c & 2u) != 0u) { clip += dy; }
    if ((c & 4u) != 0u) { clip += dz; }
    if (clip.w <= 1e-4) {
      return false;
    }
    let inv_w = 1.0 / clip.w;
    let ndc = clip.xyz * inv_w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    min_uv = min(min_uv, uv);
    max_uv = max(max_uv, uv);
    near_z = select(min(near_z, ndc.z), max(near_z, ndc.z), rev_z);
  }
  if (min_uv.x < 0.0 || max_uv.x > 1.0 || min_uv.y < 0.0 || max_uv.y > 1.0 || near_z <= 0.0 || near_z >= 1.0) {
    return false;
  }
  let base_dims = vec2f(textureDimensions(hiz_tex, 0));
  let rect = (max_uv - min_uv) * base_dims;
  let span = max(rect.x, rect.y);
  let mip = min(u32(ceil(log2(max(span, 1.0)))), mips - 1u);
  let dims = vec2i(textureDimensions(hiz_tex, i32(mip)));
  let p0 = clamp(vec2i(floor(min_uv * vec2f(dims))), vec2i(0), dims - vec2i(1));
  let p1 = clamp(vec2i(floor(max_uv * vec2f(dims))), vec2i(0), dims - vec2i(1));
  let span_x = p1.x - p0.x;
  let span_y = p1.y - p0.y;
  // Never skip interior texels when the pyramid is too shallow; scan full footprint up to 4x4 or stay conservative.
  if (span_x > 3 || span_y > 3) {
    return false;
  }
  var bound_d = select(0.0, 1.0, rev_z);
  for (var y = p0.y; y <= p1.y; y++) {
    for (var x = p0.x; x <= p1.x; x++) {
      let d = textureLoad(hiz_tex, vec2i(x, y), i32(mip)).r;
      bound_d = select(max(bound_d, d), min(bound_d, d), rev_z);
    }
  }
  if (rev_z) {
    return near_z < bound_d - 0.0005;
  }
  return near_z > bound_d + 0.0005;
}

fn screen_error(sphere: vec4f, error: f32) -> f32 {
  let world_center = (u.model * vec4f(sphere.xyz, 1.0)).xyz;
  let world_radius = sphere.w * u.projection.x;
  let distance = length(world_center - u.camera_pos.xyz);
  return error * u.projection.x * u.projection.y / max(distance - world_radius, u.projection.z);
}

fn backface_culled(m: DagClusterGpu) -> bool {
  if ((u.counts_flags.w & 1u) == 0u || m.cone_axis_cutoff.w >= 0.999) {
    return false;
  }
  // CPU already cleared the backface bit when det3 <= 1e-6 and pre-computed local_cam.
  let to_center = m.bounds_sphere.xyz - u.local_cam.xyz;
  let dist = length(to_center);
  return dist > m.bounds_sphere.w
    && dot(to_center, m.cone_axis_cutoff.xyz) >= m.cone_axis_cutoff.w * dist + m.bounds_sphere.w;
}

var<workgroup> wg_count: atomic<u32>;
var<workgroup> wg_base: u32;

@compute @workgroup_size(64)
fn cull_meshlet_dag(
  @builtin(workgroup_id) group: vec3u,
  @builtin(num_workgroups) groups: vec3u,
  @builtin(local_invocation_index) lane: u32,
) {
  if (lane == 0u) {
    atomicStore(&wg_count, 0u);
  }
  workgroupBarrier();

  let idx = (group.y * groups.x + group.x) * 64u + lane;
  var emit = false;
  var counts = vec4u(0u);
  if (idx < u.counts_flags.x) {
    let m = clusters[idx];
    let sphere = m.bounds_sphere;
    if (in_frustum_local(sphere.xyz, sphere.w) && !backface_culled(m)) {
      let lod_pass = m.errors.x <= 0.0 || screen_error(m.lod_sphere, m.errors.x) <= u.projection.w;
      if (lod_pass && (m.errors.y >= 1e29 || screen_error(m.parent_sphere, m.errors.y) > u.projection.w)) {
        let world_center = (u.model * vec4f(sphere.xyz, 1.0)).xyz;
        let reversed_z = (u.counts_flags.w & 2u) != 0u;
        if (!hiz_occluded(world_center, sphere.w * u.projection.x, reversed_z)) {
          emit = true;
          counts = m.counts;
        }
      }
    }
  }

  var local_slot = 0u;
  if (emit) {
    local_slot = atomicAdd(&wg_count, 1u);
  }
  workgroupBarrier();
  if (lane == 0u) {
    let total = atomicLoad(&wg_count);
    if (total > 0u) {
      wg_base = atomicAdd(&draw_count, total);
    }
  }
  workgroupBarrier();
  if (emit) {
    let slot = wg_base + local_slot;
    draws[slot] = DrawIndexedIndirect(counts.y, 1u, counts.x, bitcast<i32>(counts.z), 0u);
    selected_clusters[slot] = idx;
  }
}

@compute @workgroup_size(64)
fn cull_meshlet_dag_phase1(
  @builtin(workgroup_id) group: vec3u,
  @builtin(num_workgroups) groups: vec3u,
  @builtin(local_invocation_index) lane: u32,
) {
  if (lane == 0u) {
    atomicStore(&wg_count, 0u);
  }
  workgroupBarrier();

  let idx = (group.y * groups.x + group.x) * 64u + lane;
  var emit = false;
  var counts = vec4u(0u);
  if (idx < u.counts_flags.x) {
    let m = clusters[idx];
    let sphere = m.bounds_sphere;
    var defer = 0u;
    if (in_frustum_local(sphere.xyz, sphere.w) && !backface_culled(m)) {
      let lod_pass = m.errors.x <= 0.0 || screen_error(m.lod_sphere, m.errors.x) <= u.projection.w;
      if (lod_pass && (m.errors.y >= 1e29 || screen_error(m.parent_sphere, m.errors.y) > u.projection.w)) {
        let world_center = (u.model * vec4f(sphere.xyz, 1.0)).xyz;
        let reversed_z = (u.counts_flags.w & 2u) != 0u;
        if (hiz_occluded(world_center, sphere.w * u.projection.x, reversed_z)) {
          defer = 1u;
        } else {
          emit = true;
          counts = m.counts;
        }
      }
    }
    phase2_flags[idx] = defer;
  }

  var local_slot = 0u;
  if (emit) {
    local_slot = atomicAdd(&wg_count, 1u);
  }
  workgroupBarrier();
  if (lane == 0u) {
    let total = atomicLoad(&wg_count);
    if (total > 0u) {
      wg_base = atomicAdd(&draw_count, total);
    }
  }
  workgroupBarrier();
  if (emit) {
    let slot = wg_base + local_slot;
    draws[slot] = DrawIndexedIndirect(counts.y, 1u, counts.x, bitcast<i32>(counts.z), 0u);
    selected_clusters[slot] = idx;
  }
}

@compute @workgroup_size(64)
fn cull_meshlet_dag_phase2(
  @builtin(workgroup_id) group: vec3u,
  @builtin(num_workgroups) groups: vec3u,
  @builtin(local_invocation_index) lane: u32,
) {
  if (lane == 0u) {
    atomicStore(&wg_count, 0u);
  }
  workgroupBarrier();

  let idx = (group.y * groups.x + group.x) * 64u + lane;
  var emit = false;
  var counts = vec4u(0u);
  if (idx < u.counts_flags.x && phase2_flags[idx] != 0u) {
    let m = clusters[idx];
    let sphere = m.bounds_sphere;
    let world_center = (u.model * vec4f(sphere.xyz, 1.0)).xyz;
    let reversed_z = (u.counts_flags.w & 2u) != 0u;
    if (!hiz_occluded(world_center, sphere.w * u.projection.x, reversed_z)) {
      emit = true;
      counts = m.counts;
    }
  }

  var local_slot = 0u;
  if (emit) {
    local_slot = atomicAdd(&wg_count, 1u);
  }
  workgroupBarrier();
  if (lane == 0u) {
    let total = atomicLoad(&wg_count);
    if (total > 0u) {
      wg_base = atomicAdd(&draw_count, total);
    }
  }
  workgroupBarrier();
  if (emit) {
    let slot = wg_base + local_slot;
    draws[slot] = DrawIndexedIndirect(counts.y, 1u, counts.x, bitcast<i32>(counts.z), 0u);
    selected_clusters[slot] = idx;
  }
}
