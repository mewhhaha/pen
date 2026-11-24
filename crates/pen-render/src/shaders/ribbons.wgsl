// World-space camera-facing ribbon and beam trails drawn in the translucent pass,
// supporting soft opaque-depth fade, material textures, additive or alpha blending,
// and volumetric mist.
struct View {
  view_projection: mat4x4f,
  inverse_projection: mat4x4f,
};

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var shape: texture_2d<f32>;
@group(0) @binding(2) var shape_sampler: sampler;
@group(0) @binding(6) var sky_sampler: sampler;
${translucency}

const additive = ${additive};

struct VertexIn {
  @location(0) position: vec3f,
  @location(1) uv: vec2f,
  @location(2) color: vec4f,
  @location(3) soft_depth: f32,
};

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec4f,
  @location(2) fog: vec4f,
  @location(3) soft_depth: f32,
};

fn linear_depth(ndc_z: f32) -> f32 {
  let clip = view.inverse_projection * vec4f(0.0, 0.0, ndc_z, 1.0);
  return -clip.z / max(abs(clip.w), 1e-5);
}

@vertex fn vertex(input: VertexIn) -> VertexOut {
  var out: VertexOut;
  out.position = view.view_projection * vec4f(input.position, 1.0);
  out.uv = input.uv;
  out.color = input.color;
  out.fog = fog_terms(input.position);
  out.soft_depth = input.soft_depth;
  return out;
}

@fragment fn fragment(input: VertexOut) -> @location(0) vec4f {
  var color = input.color * textureSample(shape, shape_sampler, input.uv);
  if (input.soft_depth > 0.0001 && opaque_available()) {
    let dims = vec2i(textureDimensions(opaque_depth, 0));
    let coord = clamp(vec2i(input.position.xy), vec2i(0), dims - vec2i(1));
    let scene_d = textureLoad(opaque_depth, coord, 0).r;
    if (scene_d < 1.0) {
      let gap = linear_depth(scene_d) - linear_depth(input.position.z);
      color.a *= clamp(gap / input.soft_depth, 0.0, 1.0);
    }
  }
  let light = color.rgb * color.a * input.fog.a;
  return vec4f(select(light + input.fog.rgb * color.a, light, additive), color.a);
}
