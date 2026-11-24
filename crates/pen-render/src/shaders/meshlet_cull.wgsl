// GPU Meshlet Cluster Culling Compute Shader:
// - Exact local-space frustum sphere test (conservative under arbitrary shear and non-uniform scale)
// - Exact local-space perspective normal-cone backface cluster test (invariant under shear/scale)
// - Conservative prior-frame Hi-Z occlusion test (covers full texel footprint, never skips holes)
// - Writes baseline-WebGPU-compatible DrawIndexedIndirect args (`first_instance = 0u`).

struct MeshletGpu {
  // xyz = local bounding sphere center, w = radius (rounded outward in f32)
  sphere: vec4f,
  // xyz = unit cone axis (mean outward normal), w = cone_cutoff (sin(half_angle); >= 1.0 disables backface cull)
  cone_axis_cutoff: vec4f,
  // x = first_index (exact u32), y = index_count, z = base_vertex (bitcast i32), w = tri_count
  counts: vec4u,
  // x = unique_verts, yzw = reserved
  extra: vec4u,
};

struct CullUniform {
  model: mat4x4f,
  inv_model: mat4x4f,
  hiz_vp: mat4x4f,
  planes: array<vec4f, 6>,
  // xyz = camera world position, w = meshlet_count (bitcast u32 via flags.z, kept float-safe via u32 field below)
  camera_pos: vec4f,
  // x = meshlet_count, y = hiz_enabled, z = hiz_mips, w = backface_cull | (reversed_z << 1)
  counts_flags: vec4u,
};

struct DrawIndexedIndirect {
  index_count: u32,
  instance_count: u32,
  first_index: u32,
  base_vertex: i32,
  first_instance: u32,
};

struct CullStats {
  visible: atomic<u32>,
  frustum_culled: atomic<u32>,
  backface_culled: atomic<u32>,
  hiz_culled: atomic<u32>,
};

@group(0) @binding(0) var<uniform> u: CullUniform;
@group(0) @binding(1) var<storage, read> meshlets: array<MeshletGpu>;
@group(0) @binding(2) var<storage, read_write> draws: array<DrawIndexedIndirect>;
@group(0) @binding(3) var<storage, read_write> stats: CullStats;
@group(0) @binding(4) var hiz_tex: texture_2d<f32>;

// Exact local-space frustum test: transforms each world plane by transpose(model) so non-uniform
// scaling and shear never under-estimate the projected bounding sphere radius.
fn in_frustum_local(local_center: vec3f, local_radius: f32) -> bool {
  let mt = transpose(u.model);
  for (var i = 0u; i < 6u; i++) {
    let lp = mt * u.planes[i];
    let n_len = length(lp.xyz);
    if (n_len <= 1e-7) {
      continue;
    }
    let dist = (dot(lp.xyz, local_center) + lp.w) / n_len;
    if (dist < -(local_radius * 1.001 + 1e-5)) {
      return false;
    }
  }
  return true;
}

// Conservative Frobenius / spectral upper bound on how much `model` stretches a local sphere.
fn conservative_world_radius(local_radius: f32) -> f32 {
  let a = u.model[0].xyz;
  let b = u.model[1].xyz;
  let c = u.model[2].xyz;
  let sq = vec3f(dot(a, a), dot(b, b), dot(c, c));
  let ab = dot(a, b);
  let ac = dot(a, c);
  let bc = dot(b, c);
  let ortho = ab * ab <= 1e-7 * sq.x * sq.y && ac * ac <= 1e-7 * sq.x * sq.z && bc * bc <= 1e-7 * sq.y * sq.z;
  let stretch = sqrt(select(sq.x + sq.y + sq.z, max(sq.x, max(sq.y, sq.z)), ortho));
  return local_radius * stretch;
}

fn hiz_occluded(world_center: vec3f, world_radius: f32, rev_z: bool) -> bool {
  if (u.counts_flags.y == 0u || u.counts_flags.z == 0u) {
    return false;
  }
  let mips = u.counts_flags.z;
  let r = world_radius * 1.02 + 0.005;
  var min_uv = vec2f(1e9);
  var max_uv = vec2f(-1e9);
  var near_z = select(1.0, 0.0, rev_z);
  for (var c = 0u; c < 8u; c++) {
    let sx = select(-r, r, (c & 1u) != 0u);
    let sy = select(-r, r, (c & 2u) != 0u);
    let sz = select(-r, r, (c & 4u) != 0u);
    let clip = u.hiz_vp * vec4f(world_center + vec3f(sx, sy, sz), 1.0);
    if (clip.w <= 1e-4) {
      return false;
    }
    let ndc = clip.xyz / clip.w;
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

@compute @workgroup_size(64)
fn cull_meshlets(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  let count = u.counts_flags.x;
  if (idx >= count) {
    return;
  }
  let m = meshlets[idx];
  let backface_enabled = (u.counts_flags.w & 1u) != 0u;
  let rev_z = (u.counts_flags.w & 2u) != 0u;

  var visible = true;
  if (!in_frustum_local(m.sphere.xyz, m.sphere.w)) {
    visible = false;
    atomicAdd(&stats.frustum_culled, 1u);
  }

  // Exact local-space perspective normal-cone backface cluster test (exact under arbitrary shear/scale):
  // Since (M * v) . (adj(M)^T * n) = det(M) * (v . n), testing in local space using inverse-transformed
  // camera preserves backface orientation identically for any positive-determinant affine matrix.
  if (visible && backface_enabled && m.cone_axis_cutoff.w < 0.999) {
    let det = dot(u.model[0].xyz, cross(u.model[1].xyz, u.model[2].xyz));
    if (det > 1e-6) {
      let local_cam = (u.inv_model * vec4f(u.camera_pos.xyz, 1.0)).xyz;
      let to_center = m.sphere.xyz - local_cam;
      let dist = length(to_center);
      if (dist > m.sphere.w && dot(to_center, m.cone_axis_cutoff.xyz) >= m.cone_axis_cutoff.w * dist + m.sphere.w) {
        visible = false;
        atomicAdd(&stats.backface_culled, 1u);
      }
    }
  }

  if (visible) {
    let world_center = (u.model * vec4f(m.sphere.xyz, 1.0)).xyz;
    let world_radius = conservative_world_radius(m.sphere.w);
    if (hiz_occluded(world_center, world_radius, rev_z)) {
      visible = false;
      atomicAdd(&stats.hiz_culled, 1u);
    }
  }

  if (visible) {
    atomicAdd(&stats.visible, 1u);
  }
  // Emit `first_instance = 0u` so indirect draws execute cleanly on baseline WebGPU without INDIRECT_FIRST_INSTANCE.
  draws[idx] = DrawIndexedIndirect(
    m.counts.y,
    select(0u, 1u, visible),
    m.counts.x,
    bitcast<i32>(m.counts.z),
    0u,
  );
}
