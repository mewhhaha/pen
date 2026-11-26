// Instanced batches culled on the GPU: one invocation per instance of each
// batch (the dispatch's y) tests the batch's bounding sphere, placed by the
// instance's transform, against each view the frame draws, picks its level
// of detail from the camera's distance, and appends it to the instances its
// batch draws in each view that sees it at that level, counting it into
// that draw's indirect arguments. The renderer's culling.rs does the same
// on the CPU for smaller batches.
struct Instance {
  x: vec4f,
  y: vec4f,
  z: vec4f,
  w: vec4f,
  parameters: vec4f,
};

struct Batch {
  // In the mesh's space, grown by its shader's margin.
  sphere: vec4f,
  // Each level of detail's error, the whole mesh's (0) first.
  errors: vec4f,
  // Its instances in `instances`.
  first: u32,
  count: u32,
  // The views it may be drawn in, a bit each, and its levels of detail.
  views: u32,
  levels: u32,
  // Where its culled instances go, `count` for each view and level, and
  // its first draw's arguments, one for each view and level.
  out: u32,
  draws: u32,
  unused0: u32,
  unused1: u32,
};

struct Draw {
  index_count: u32,
  instance_count: atomic<u32>,
  first_index: u32,
  base_vertex: i32,
  first_instance: u32,
};

struct Cull {
  views: u32,
  // How many pixels a unit spans at w = 1; 0 draws every mesh whole.
  pixels: f32,
  // Prior-frame Hi-Z occlusion culling: enabled flag and pyramid mip count.
  hiz_enabled: u32,
  hiz_mips: u32,
  // The camera's view-projection row that gives a point's w.
  row: vec4f,
  // (width, height, 1/width, 1/height) of mip 0 of `hiz`.
  hiz_size: vec4f,
  // The view-projection matrix that produced `hiz`.
  hiz_vp: mat4x4f,
  // Six planes a view, pointing in, normals of unit length.
  planes: array<vec4f, 144>,
};

@group(0) @binding(0) var<uniform> cull: Cull;
@group(0) @binding(1) var<storage, read> batches: array<Batch>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read_write> culled: array<Instance>;
@group(0) @binding(4) var<storage, read_write> draws: array<Draw>;
@group(0) @binding(5) var hiz: texture_2d<f32>;

fn sees(view: u32, center: vec3f, radius: f32) -> bool {
  let slack = radius * 1.0001 + 0.00001;
  for (var plane = 0u; plane < 6u; plane++) {
    let p = cull.planes[view * 6u + plane];
    if (dot(p.xyz, center) + p.w < -slack) {
      return false;
    }
  }
  return true;
}

// Conservative screen-space Hi-Z occlusion test against the previous frame's
// max-depth pyramid: projects the sphere's padded enclosing world AABB through
// `hiz_vp` and only culls when all 8 corners lie inside the previous screen
// and the nearest corner depth is strictly behind the pyramid's max depth.
fn occluded(center: vec3f, radius: f32) -> bool {
  if (cull.hiz_enabled == 0u || cull.hiz_mips == 0u) {
    return false;
  }
  let r = radius * 1.02 + 0.01;
  let clip0 = cull.hiz_vp * vec4f(center - vec3f(r), 1.0);
  let dx = cull.hiz_vp[0] * (2.0 * r);
  let dy = cull.hiz_vp[1] * (2.0 * r);
  let dz = cull.hiz_vp[2] * (2.0 * r);
  var min_uv = vec2f(1e9, 1e9);
  var max_uv = vec2f(-1e9, -1e9);
  var min_z = 1.0;
  for (var corner = 0u; corner < 8u; corner++) {
    var clip = clip0;
    if ((corner & 1u) != 0u) { clip += dx; }
    if ((corner & 2u) != 0u) { clip += dy; }
    if ((corner & 4u) != 0u) { clip += dz; }
    if (clip.w <= 0.0001) {
      return false;
    }
    let inv_w = 1.0 / clip.w;
    let ndc = clip.xyz * inv_w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    min_uv = min(min_uv, uv);
    max_uv = max(max_uv, uv);
    min_z = min(min_z, ndc.z);
  }
  // Never cull if near-plane clipped or partially outside the prior screen.
  if (min_z <= 0.0 || min_uv.x < 0.0 || max_uv.x > 1.0 || min_uv.y < 0.0 || max_uv.y > 1.0) {
    return false;
  }
  let rect_min = floor(min_uv * cull.hiz_size.xy);
  let rect_max = floor(max_uv * cull.hiz_size.xy);
  let span = max(rect_max.x - rect_min.x + 1.0, rect_max.y - rect_min.y + 1.0);
  var mip = min(u32(ceil(log2(max(span, 1.0)))), cull.hiz_mips - 1u);
  var dims = vec2i(textureDimensions(hiz, i32(mip)));
  var p0 = clamp(vec2i(floor(min_uv * vec2f(dims))), vec2i(0), dims - vec2i(1));
  var p1 = clamp(vec2i(floor(max_uv * vec2f(dims))), vec2i(0), dims - vec2i(1));
  if ((p1.x - p0.x > 1 || p1.y - p0.y > 1) && mip + 1u < cull.hiz_mips) {
    mip = mip + 1u;
    dims = vec2i(textureDimensions(hiz, i32(mip)));
    p0 = clamp(vec2i(floor(min_uv * vec2f(dims))), vec2i(0), dims - vec2i(1));
    p1 = clamp(vec2i(floor(max_uv * vec2f(dims))), vec2i(0), dims - vec2i(1));
  }
  let d00 = textureLoad(hiz, vec2i(p0.x, p0.y), i32(mip)).r;
  let d10 = textureLoad(hiz, vec2i(p1.x, p0.y), i32(mip)).r;
  let d01 = textureLoad(hiz, vec2i(p0.x, p1.y), i32(mip)).r;
  let d11 = textureLoad(hiz, vec2i(p1.x, p1.y), i32(mip)).r;
  let max_depth = max(max(d00, d10), max(d01, d11));
  return min_z > max_depth + 0.0005;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let batch = batches[id.y];
  if (id.x >= batch.count) {
    return;
  }
  let instance = instances[batch.first + id.x];
  let model = mat4x4f(instance.x, instance.y, instance.z, instance.w);
  let center = (model * vec4f(batch.sphere.xyz, 1.0)).xyz;
  // How much the transform stretches the sphere (culling.rs's `stretch`).
  let a = model[0].xyz;
  let b = model[1].xyz;
  let c = model[2].xyz;
  let squared = vec3f(dot(a, a), dot(b, b), dot(c, c));
  let ab = dot(a, b);
  let ac = dot(a, c);
  let bc = dot(b, c);
  let square = ab * ab <= 1e-8 * squared.x * squared.y && ac * ac <= 1e-8 * squared.x * squared.z &&
    bc * bc <= 1e-8 * squared.y * squared.z;
  let stretch = sqrt(select(squared.x + squared.y + squared.z, max(squared.x, max(squared.y, squared.z)), square));
  let radius = batch.sphere.w * stretch;
  // The simplest level whose error spans at most a pixel where the sphere
  // is nearest the camera (culling.rs's `Detail::level`).
  var level = 0u;
  if (cull.pixels > 0.0 && batch.levels > 1u) {
    let nearest = max(dot(cull.row.xyz, center) + cull.row.w - radius * length(cull.row.xyz), 0.0001);
    let per_unit = cull.pixels * stretch / nearest;
    for (var simpler = batch.levels - 1u; simpler > 0u; simpler--) {
      if (batch.errors[simpler] * per_unit <= 1.0) {
        level = simpler;
        break;
      }
    }
  }
  for (var view = 0u; view < cull.views; view++) {
    if ((batch.views & (1u << view)) == 0u || !sees(view, center, radius)) {
      continue;
    }
    if (view == 0u && occluded(center, radius)) {
      continue;
    }
    let draw = view * batch.levels + level;
    let slot = atomicAdd(&draws[batch.draws + draw].instance_count, 1u);
    culled[batch.out + draw * batch.count + slot] = instance;
  }
}
