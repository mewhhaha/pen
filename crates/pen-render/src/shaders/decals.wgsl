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

const PAINT: bool = ${paint};

// The cube -1..1, wound counterclockwise seen from outside.
fn cube_corner(vertex: u32) -> vec3f {
  let face = vertex / 6u;
  let slot = vertex % 6u;
  var packed = 0u;
  switch face {
    case 0u: { packed = 0u | (4u << 3u) | (6u << 6u) | (0u << 9u) | (6u << 12u) | (2u << 15u); }
    case 1u: { packed = 1u | (3u << 3u) | (7u << 6u) | (1u << 9u) | (7u << 12u) | (5u << 15u); }
    case 2u: { packed = 0u | (1u << 3u) | (5u << 6u) | (0u << 9u) | (5u << 12u) | (4u << 15u); }
    case 3u: { packed = 2u | (6u << 3u) | (7u << 6u) | (2u << 9u) | (7u << 12u) | (3u << 15u); }
    case 4u: { packed = 0u | (2u << 3u) | (3u << 6u) | (0u << 9u) | (3u << 12u) | (1u << 15u); }
    default: { packed = 4u | (5u << 3u) | (7u << 6u) | (4u << 9u) | (7u << 12u) | (6u << 15u); }
  }
  let c = (packed >> (slot * 3u)) & 7u;
  return vec3f(
    select(-1.0, 1.0, (c & 1u) != 0u),
    select(-1.0, 1.0, (c & 2u) != 0u),
    select(-1.0, 1.0, (c & 4u) != 0u),
  );
}

struct Box {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) index: u32,
};

@vertex fn vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> Box {
  let corner = cube_corner(vertex);
  return Box(view.view_projection * (decals[index].model * vec4f(corner, 1.0)), index);
}

// The decal's tinted texture where the surface behind the box lies in it, its
// alpha how much of the surface it covers: none outside the box.
fn laid(position: vec4f, index: u32) -> vec4f {
  let decal = decals[index];
  let pixel = vec2i(position.xy);
  let ndc = vec2f(position.x / view.size.x * 2.0 - 1.0, 1.0 - position.y / view.size.y * 2.0);
  let world = view.inverse_view_projection * vec4f(ndc, textureLoad(depth, pixel, 0), 1.0);
  let local = (decal.inverse * vec4f(world.xyz / world.w, 1.0)).xyz;
  if (any(abs(local) > vec3f(1.0))) {
    return vec4f(0.0);
  }
  let texel = textureSampleLevel(image, image_sampler, local.xz * 0.5 + 0.5, 0.0);
  let cover = texel.a * decal.color.a * (1.0 - smoothstep(0.6, 1.0, abs(local.y)));
  return vec4f(texel.rgb * decal.color.rgb, cover);
}

@fragment fn fragment(box: Box) -> @location(0) vec4f {
  let found = laid(box.position, box.index);
  if (found.a <= 0.0) {
    discard;
  }
  if (PAINT) {
    return vec4f(found.rgb * found.a, found.a);
  }
  return vec4f(mix(vec3f(1.0), found.rgb, found.a), 1.0);
}
