// Material constants and geometric normals for the first opaque hit.
struct DrawGbuf {
  model: mat4x4f,
  normal_mat: mat4x4f,
  albedo_metallic: vec4f,
  // Roughness after clearcoat, then the clearcoat roughness limit for instances.
  roughness_pad: vec4f,
};

@group(0) @binding(0) var<uniform> view_projection: mat4x4f;
@group(1) @binding(0) var<uniform> draw: DrawGbuf;
@group(2) @binding(0) var<storage, read> joints: array<mat4x4f>;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
  @location(1) @interpolate(flat) metallic_roughness: vec2f,
};

// For normal directions the cofactor matrix needs only the sign of the
// determinant. This preserves orientation under reflections and avoids
// dividing by a small determinant before normalization.
fn transform_normal(model: mat4x4f, normal: vec3f) -> vec3f {
  let a = model[0].xyz;
  let b = model[1].xyz;
  let c = model[2].xyz;
  let cofactor = mat3x3f(cross(b, c), cross(c, a), cross(a, b));
  let determinant = dot(a, cofactor[0]);
  if (determinant == 0.0) {
    return normal;
  }
  return cofactor * normal * sign(determinant);
}

fn finite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn project(world: vec4f, normal: vec3f, metallic_roughness: vec2f) -> VertexOut {
  return VertexOut(view_projection * world, normal, metallic_roughness);
}

@vertex fn vertex_plain(@location(0) position: vec3f, @location(1) normal: vec3f) -> VertexOut {
  return project(
    draw.model * vec4f(position, 1.0),
    (draw.normal_mat * vec4f(normal, 0.0)).xyz,
    vec2f(draw.albedo_metallic.a, draw.roughness_pad.x),
  );
}

@vertex fn vertex_instanced(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(3) m0: vec4f,
  @location(4) m1: vec4f,
  @location(5) m2: vec4f,
  @location(6) m3: vec4f,
  @location(7) parameters: vec4f,
) -> VertexOut {
  let model = mat4x4f(m0, m1, m2, m3);
  var metallic = draw.albedo_metallic.a;
  var roughness = draw.roughness_pad.x;
  if (draw.roughness_pad.z > 0.5) {
    if (finite(parameters.x)) {
      metallic = clamp(parameters.x, 0.0, 1.0);
    }
    if (finite(parameters.y) && parameters.y > 0.0) {
      roughness = min(clamp(parameters.y, 0.04, 1.0), draw.roughness_pad.y);
    }
  }
  return project(
    model * vec4f(position, 1.0),
    transform_normal(model, normal),
    vec2f(metallic, roughness),
  );
}

@vertex fn vertex_skinned(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(9) indices: vec4u,
  @location(10) weights: vec4f,
) -> VertexOut {
  let skin = joints[indices.x] * weights.x
    + joints[indices.y] * weights.y
    + joints[indices.z] * weights.z
    + joints[indices.w] * weights.w;
  // Skin transforms vary per vertex. Apply the determinant's magnitude as
  // well, so interpolating the resulting world normals uses their true
  // relative lengths.
  let a = skin[0].xyz;
  let b = skin[1].xyz;
  let c = skin[2].xyz;
  let cofactors = mat3x3f(cross(b, c), cross(c, a), cross(a, b));
  let determinant = dot(a, cofactors[0]);
  var posed_normal = normal;
  if (determinant != 0.0) {
    posed_normal = (cofactors * normal) / determinant;
  }
  return project(
    draw.model * (skin * vec4f(position, 1.0)),
    (draw.normal_mat * vec4f(posed_normal, 0.0)).xyz,
    vec2f(draw.albedo_metallic.a, draw.roughness_pad.x),
  );
}

struct GbufferOut {
  @location(0) normal_roughness: vec4f,
  @location(1) albedo_metallic: vec4f,
};

@fragment fn fragment(in: VertexOut) -> GbufferOut {
  let magnitude = max(abs(in.normal.x), max(abs(in.normal.y), abs(in.normal.z)));
  var normal = vec3f(0.0, 0.0, 1.0);
  if (magnitude > 0.0 && finite(magnitude)) {
    normal = normalize(in.normal / magnitude);
  }
  return GbufferOut(
    vec4f(normal, in.metallic_roughness.y),
    vec4f(draw.albedo_metallic.rgb, in.metallic_roughness.x),
  );
}
