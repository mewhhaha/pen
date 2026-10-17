// A material shader: what pen-render binds for every draw (group 0, the
// frame's scene; group 1, the draw and its material) and the mesh's vertex
// layout. This one lights the material's color from the frame's sun and
// ambient light. The leading fields of each struct are fixed; a shader
// declares as many of them as it reads.
struct Scene {
  view: mat4x4f,
  projection: mat4x4f,
  ambient: vec4f,
  light_direction: vec4f,
  light_color: vec4f,
};
struct Draw {
  model: mat4x4f,
  normal: mat4x4f,
  color: vec4f,
  settings: vec4f,
  parameters: vec4f,
};
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;

struct Vertex {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
};
struct Surface {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
};

@vertex fn vertex(input: Vertex) -> Surface {
  var output: Surface;
  output.position = scene.projection * scene.view * draw.model * vec4f(input.position, 1.0);
  output.normal = (draw.normal * vec4f(input.normal, 0.0)).xyz;
  return output;
}

@fragment fn fragment(surface: Surface) -> @location(0) vec4f {
  let sun = max(dot(normalize(surface.normal), -normalize(scene.light_direction.xyz)), 0.0);
  let light = scene.ambient.rgb + scene.light_color.rgb * sun;
  return vec4f(draw.color.rgb * light, draw.color.a);
}
