struct View {
  view_projection: mat4x4f,
  inverse_view_projection: mat4x4f,
  size: vec4f,
};
struct Decal {
  model: mat4x4f,
  inverse: mat4x4f,
  color: vec4f,
};
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> decals: array<Decal>;
@group(0) @binding(2) var depth: ${depth_texture};
@group(0) @binding(3) var image: texture_2d<f32>;
@group(0) @binding(4) var image_sampler: sampler;

// The cube -1..1, wound counterclockwise seen from outside.
const corners = array<vec3f, 8>(
  vec3f(-1.0, -1.0, -1.0), vec3f(1.0, -1.0, -1.0), vec3f(-1.0, 1.0, -1.0), vec3f(1.0, 1.0, -1.0),
  vec3f(-1.0, -1.0, 1.0), vec3f(1.0, -1.0, 1.0), vec3f(-1.0, 1.0, 1.0), vec3f(1.0, 1.0, 1.0),
);
const faces = array<u32, 36>(
  0u, 4u, 6u, 0u, 6u, 2u, 1u, 3u, 7u, 1u, 7u, 5u,
  0u, 1u, 5u, 0u, 5u, 4u, 2u, 6u, 7u, 2u, 7u, 3u,
  0u, 2u, 3u, 0u, 3u, 1u, 4u, 5u, 7u, 4u, 7u, 6u,
);

struct Box {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) index: u32,
};

@vertex fn vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> Box {
  let corner = corners[faces[vertex]];
  return Box(view.view_projection * decals[index].model * vec4f(corner, 1.0), index);
}

// The decal's tinted texture where the surface behind the box lies in it, its
// alpha how much of the surface it covers: none outside the box.
fn laid(box: Box) -> vec4f {
  let decal = decals[box.index];
  let pixel = vec2i(box.position.xy);
  let ndc = vec2f(box.position.x / view.size.x * 2.0 - 1.0, 1.0 - box.position.y / view.size.y * 2.0);
  let world = view.inverse_view_projection * vec4f(ndc, textureLoad(depth, pixel, 0), 1.0);
  let local = (decal.inverse * vec4f(world.xyz / world.w, 1.0)).xyz;
  if (any(abs(local) > vec3f(1.0))) {
    return vec4f(0.0);
  }
  let texel = textureSampleLevel(image, image_sampler, local.xz * 0.5 + 0.5, 0.0);
  let cover = texel.a * decal.color.a * (1.0 - smoothstep(0.6, 1.0, abs(local.y)));
  return vec4f(texel.rgb * decal.color.rgb, cover);
}

// Multiplied over the surface, so its lighting and shadow show through.
@fragment fn fragment(box: Box) -> @location(0) vec4f {
  let found = laid(box);
  if (found.a <= 0.0) {
    discard;
  }
  return vec4f(mix(vec3f(1.0), found.rgb, found.a), 1.0);
}

// Painted over the surface in its own color, premultiplied, whatever light
// falls there.
@fragment fn paint(box: Box) -> @location(0) vec4f {
  let found = laid(box);
  if (found.a <= 0.0) {
    discard;
  }
  return vec4f(found.rgb * found.a, found.a);
}
