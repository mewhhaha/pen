// Per-object and skeletal screen-space velocity pass.
// Rasterizes at the scene's jittered clip position so depth matches the scene,
// while computing screen-space UV motion (curr_uv - prev_uv) from unjittered
// current and previous clip positions. Uncovered pixels retain the sentinel
// (1024.0, 1024.0) so downstream TAA and motion blur fall back to analytical
// camera reprojection from depth.
struct Camera {
  curr_jittered_vp: mat4x4f,
  curr_vp: mat4x4f,
  prev_vp: mat4x4f,
};

struct DrawMotion {
  curr_model: mat4x4f,
  prev_model: mat4x4f,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> draw: DrawMotion;
@group(2) @binding(0) var<storage, read> curr_joints: array<mat4x4f>;
@group(2) @binding(1) var<storage, read> prev_joints: array<mat4x4f>;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) curr_clip: vec4f,
  @location(1) prev_clip: vec4f,
};

fn project(curr_world: vec4f, prev_world: vec4f) -> VertexOut {
  var out: VertexOut;
  out.position = camera.curr_jittered_vp * curr_world;
  out.curr_clip = camera.curr_vp * curr_world;
  out.prev_clip = camera.prev_vp * prev_world;
  return out;
}

@vertex fn vertex_plain(@location(0) position: vec3f) -> VertexOut {
  let local = vec4f(position, 1.0);
  return project(draw.curr_model * local, draw.prev_model * local);
}

@vertex fn vertex_instanced(
  @location(0) position: vec3f,
  @location(3) m0: vec4f,
  @location(4) m1: vec4f,
  @location(5) m2: vec4f,
  @location(6) m3: vec4f,
) -> VertexOut {
  let model = mat4x4f(m0, m1, m2, m3);
  let world = model * vec4f(position, 1.0);
  return project(world, world);
}

@vertex fn vertex_skinned(
  @location(0) position: vec3f,
  @location(9) joints: vec4u,
  @location(10) weights: vec4f,
) -> VertexOut {
  let curr_skin = curr_joints[joints.x] * weights.x
    + curr_joints[joints.y] * weights.y
    + curr_joints[joints.z] * weights.z
    + curr_joints[joints.w] * weights.w;
  let prev_skin = prev_joints[joints.x] * weights.x
    + prev_joints[joints.y] * weights.y
    + prev_joints[joints.z] * weights.z
    + prev_joints[joints.w] * weights.w;
  let local = vec4f(position, 1.0);
  return project(draw.curr_model * (curr_skin * local), draw.prev_model * (prev_skin * local));
}

@fragment fn fragment(in: VertexOut) -> @location(0) vec4f {
  if (in.curr_clip.w <= 1e-5 || in.prev_clip.w <= 1e-5) {
    return vec4f(1024.0, 1024.0, 0.0, 0.0);
  }
  let curr_ndc = in.curr_clip.xy / in.curr_clip.w;
  let prev_ndc = in.prev_clip.xy / in.prev_clip.w;
  let curr_uv = vec2f(curr_ndc.x * 0.5 + 0.5, 0.5 - curr_ndc.y * 0.5);
  let prev_uv = vec2f(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
  let vel = clamp(curr_uv - prev_uv, vec2f(-10.0), vec2f(10.0));
  return vec4f(vel, 0.0, 0.0);
}
