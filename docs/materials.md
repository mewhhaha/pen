# Material shaders

pen has no shaders of its own for a game's surfaces: every material names a
WGSL file among the game's assets, and pen-render compiles the pipelines it
needs from it. This is what pen-render gives such a shader. The `lit.wgsl` a
game `pen new` makes starts here, and gdev's `src/assets/*.wgsl` go further.

## A material

```json
{ "shader": "lit.wgsl", "color": [0.95, 0.45, 0.12, 1], "lighting": "lit" }
```

`shader`, `color` (four 0 to 1 channels; an alpha below 1 makes it
translucent, blended over what is behind it) and `lighting` (`lit` or
`unlit`) are needed. It may also name a `texture`, a `normalTexture` and a
`layerTexture`, give `instanceParameters` (four numbers its draws read
unless a draw gives its own), an `alphaCutoff`, and say whether it is an
`overlay` (drawn over the scene, dimmed where the scene hides it), takes
`decals`, casts a `shadow` (yes unless false) or splits its colours by the
frame's `aberration`.

## What a draw binds

Group 0, the frame's, the same for every draw:

| Binding | What |
|---|---|
| 0 | The scene, a uniform: `view` and `projection` (column-major `mat4x4f`), `ambient`, `light_direction` and `light_color` (`vec4f`, xyz used), `light_view_projection` (the shadow map's), `shadow` (x 1 with a shadow map, y its texel in texture coordinates, z how far off the surface to read it), the sky's `zenith`, `horizon` and `ground` colours, and `eye`, where the camera is. |
| 1 | The shadow map, `texture_depth_2d`. |
| 2 | Its `sampler_comparison`. |

Group 1, the draw's:

| Binding | What |
|---|---|
| 0 | The draw, a uniform: `model` and `normal` matrices, the material's `color`, `settings` (x 1 when lit, y the alpha cutoff, z 1 when it has a normal texture or a field, w 1 when it has a layer texture), the four `parameters`, then 32 `vec4f` of the draw's `extra` (zero past what it sent). |
| 1 | The material's `texture`, or white without one (a checker when it failed to load). |
| 2 | A filtering sampler, at the frame's quality. |
| 3 | The normal texture, or the material's field (the render packet's `fields`), or a flat normal. |
| 4 | The layer texture, or transparent. |

A shader declares only what it reads: a struct may stop after its leading
fields, and an unused binding may be left out, as `lit.wgsl` leaves out all
but the two uniforms.

## Vertices and entry points

A mesh's vertex is `@location(0) position: vec3f`, `@location(1) normal:
vec3f`, `@location(2) uv: vec2f` and, when the mesh has `attributes`,
`@location(8) attributes: vec2f`.

- `@vertex fn vertex` and `@fragment fn fragment` draw the material. The
  fragment's colour is linear and may pass 1: the frame is drawn into a
  floating-point target that the present pass tones down to the screen.
- `@vertex fn vertex_instanced`, if the shader has one, lets opaque draws of
  the same mesh and material share one instanced draw: its instance's model
  matrix comes as `@location(3)` to `@location(6)` (four columns) and its
  parameters as `@location(7)`.
- A shader whose `vertex` takes only builtins (`vertex_index`,
  `instance_index`) draws sprites instead: group 0 is then the screen's size
  (a `vec4f` uniform at binding 0), the frame's sprites (a storage array at
  1: each one's rectangle, texture coordinates, colour and parameters), the
  material's texture (2) and a sampler (3).

Projections map depth into WebGPU's 0 to 1. pen compiles only the pipelines
the loaded materials draw with, on background threads; a shader that does
not compile is reported (an `AssetFailed` event, and the log) and its draws
are left out, never the frame.
