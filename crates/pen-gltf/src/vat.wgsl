// Vertex animation, the cheap way to draw a crowd: each vertex's place and
// normal come from a texture baked from a skinned mesh's clips (`pen gltf
// --vat`, pen-gltf's vat.rs says how it is laid out), read by the vertex's
// index at its clip and time, between the two baked frames either side.
// The material's layerTexture is the texture, and a draw's (or each
// instance's) parameters say x: which clip, y: how many seconds into it
// (the clip loops). Lit like a plain matte surface, with the sun's shadow.
// `pen gltf` writes this file when the asset directory has none; a game
// may light it as it likes.
struct Scene {
  view: mat4x4f,
  projection: mat4x4f,
  ambient: vec4f,
  light_direction: vec4f,
  light_color: vec4f,
  light_view_projection: mat4x4f,
  shadow: vec4f,
  zenith: vec4f,
  horizon: vec4f,
  ground: vec4f,
  eye: vec4f,
};
struct Draw {
  model: mat4x4f,
  normal: mat4x4f,
  color: vec4f,
  settings: vec4f,
  parameters: vec4f,
};
@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
@group(1) @binding(0) var<uniform> draw: Draw;
@group(1) @binding(1) var base_color: texture_2d<f32>;
@group(1) @binding(2) var base_sampler: sampler;
@group(1) @binding(4) var frames: texture_2d<f32>;

struct VertexInput {
  @builtin(vertex_index) vertex: u32,
  @location(2) uv: vec2f,
};
struct InstanceInput {
  @location(3) column0: vec4f,
  @location(4) column1: vec4f,
  @location(5) column2: vec4f,
  @location(6) column3: vec4f,
  @location(7) parameters: vec4f,
};
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
  @location(1) uv: vec2f,
  @location(2) world: vec3f,
};

// Texel `index` of the texture, row by row, as bytes.
fn texel(index: u32) -> vec4f {
  let width = textureDimensions(frames).x;
  return round(textureLoad(frames, vec2u(index % width, index / width), 0) * 255.0);
}

// The little-endian word texel `index` holds.
fn word(index: u32) -> u32 {
  let bytes = vec4u(texel(index));
  return bytes.x | (bytes.y << 8u) | (bytes.z << 16u) | (bytes.w << 24u);
}

fn number(index: u32) -> f32 {
  return bitcast<f32>(word(index));
}

struct Frame {
  place: vec3f,
  normal: vec3f,
};

// Vertex `vertex` in baked frame `frame`: 16 bits a coordinate across the box
// all frames lie in, and an octahedral normal.
fn baked(frame: u32, vertex: u32) -> Frame {
  let at = 9u + word(2u) * 3u + (frame * word(1u) + vertex) * 2u;
  let high = texel(at);
  let low = texel(at + 1u);
  let corner = vec3f(number(3u), number(4u), number(5u));
  let size = vec3f(number(6u), number(7u), number(8u));
  let place = corner + (high.xyz * 256.0 + low.xyz) / 65535.0 * size;
  let oct = vec2f(high.w, low.w) / 255.0 * 2.0 - 1.0;
  var normal = vec3f(oct, 1.0 - abs(oct.x) - abs(oct.y));
  if (normal.z < 0.0) {
    normal = vec3f((1.0 - abs(normal.yx)) * select(vec2f(-1.0), vec2f(1.0), normal.xy >= vec2f(0.0)), normal.z);
  }
  return Frame(place, normalize(normal));
}

// Vertex `vertex` at `parameters`' clip and time.
fn animated(vertex: u32, parameters: vec4f) -> Frame {
  let clip = min(u32(max(parameters.x, 0.0)), max(word(2u), 1u) - 1u);
  let first = word(9u + clip * 3u);
  let count = max(word(10u + clip * 3u), 1u);
  let duration = number(11u + clip * 3u);
  var phase = 0.0;
  if (duration > 0.0) {
    phase = fract(parameters.y / duration) * f32(count);
  }
  let before = min(u32(phase), count - 1u);
  let after = (before + 1u) % count;
  let earlier = baked(first + before, vertex);
  let later = baked(first + after, vertex);
  let along = fract(phase);
  return Frame(
    mix(earlier.place, later.place, along),
    normalize(mix(earlier.normal, later.normal, along) + vec3f(0.0, 1e-6, 0.0)),
  );
}

fn placed(input: VertexInput, model: mat4x4f, normal: mat3x3f, parameters: vec4f) -> VertexOutput {
  let frame = animated(input.vertex, parameters);
  let world = model * vec4f(frame.place, 1.0);
  var output: VertexOutput;
  output.position = scene.projection * scene.view * world;
  output.normal = normal * frame.normal;
  output.uv = input.uv;
  output.world = world.xyz;
  return output;
}

@vertex fn vertex(input: VertexInput) -> VertexOutput {
  return placed(input, draw.model, mat3x3f(draw.normal[0].xyz, draw.normal[1].xyz, draw.normal[2].xyz), draw.parameters);
}

@vertex fn vertex_instanced(input: VertexInput, instance: InstanceInput) -> VertexOutput {
  let model = mat4x4f(instance.column0, instance.column1, instance.column2, instance.column3);
  let a = model[0].xyz;
  let b = model[1].xyz;
  let c = model[2].xyz;
  let inverse_determinant = 1.0 / dot(a, cross(b, c));
  let normal = mat3x3f(cross(b, c) * inverse_determinant, cross(c, a) * inverse_determinant, cross(a, b) * inverse_determinant);
  return placed(input, model, normal, instance.parameters);
}

// How much sunlight reaches `world`, from the shadow map.
fn sunlit(world: vec3f, normal: vec3f) -> f32 {
  if (scene.shadow.x < 0.5) {
    return 1.0;
  }
  let clip = scene.light_view_projection * vec4f(world + normal * scene.shadow.z, 1.0);
  let ndc = clip.xyz / clip.w;
  let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || ndc.z > 1.0) {
    return 1.0;
  }
  return textureSampleCompareLevel(shadow_map, shadow_sampler, uv, ndc.z - 0.0004);
}

@fragment fn fragment(input: VertexOutput) -> @location(0) vec4f {
  let albedo = draw.color.rgb * textureSample(base_color, base_sampler, input.uv).rgb;
  if (draw.settings.x < 0.5) {
    return vec4f(albedo, draw.color.a);
  }
  let normal = normalize(input.normal);
  let to_light = normalize(-scene.light_direction.xyz);
  let sun = sunlit(input.world, normal);
  let light = scene.ambient.xyz * (0.75 + 0.25 * normal.y) + scene.light_color.xyz * max(dot(normal, to_light), 0.0) * sun;
  return vec4f(albedo * light, draw.color.a);
}
