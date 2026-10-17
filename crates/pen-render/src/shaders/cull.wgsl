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
  unused0: u32,
  unused1: u32,
  // The camera's view-projection row that gives a point's w.
  row: vec4f,
  // Six planes a view, pointing in, normals of unit length.
  planes: array<vec4f, 30>,
};

@group(0) @binding(0) var<uniform> cull: Cull;
@group(0) @binding(1) var<storage, read> batches: array<Batch>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read_write> culled: array<Instance>;
@group(0) @binding(4) var<storage, read_write> draws: array<Draw>;

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
    let draw = view * batch.levels + level;
    let slot = atomicAdd(&draws[batch.draws + draw].instance_count, 1u);
    culled[batch.out + draw * batch.count + slot] = instance;
  }
}
