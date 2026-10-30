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

- `@vertex func vertex` and `@fragment func fragment` draw the material. The
  fragment's colour is linear and may pass 1: the frame is drawn into a
  floating-point target that the present pass tones down to the screen.
- `@vertex func vertex_instanced`, if the shader has one, lets opaque draws of
  the same mesh and material share one instanced draw: its instance's model
  matrix comes as `@location(3)` to `@location(6)` (four columns) and its
  parameters as `@location(7)`.
- `@vertex func vertex_persistent` draws a mesh's immutable `instances` records.
  Each record supplies five `vec4f`s at locations 3–7. Their meaning belongs
  to the shader; `draw.model`, parameters and extra vectors still describe
  the whole draw. This entry is separate from `vertex_instanced`, so the same
  material can also draw ordinary meshes without treating records as matrices.
- A shader whose `vertex` takes only builtins (`vertex_index`,
  `instance_index`) draws sprites instead: group 0 is then the screen's size
  (a `vec4f` uniform at binding 0), the frame's sprites (a storage array at
  1: each one's rectangle, texture coordinates, colour and parameters), the
  material's texture (2), a sampler (3), and the raw resolved scene (4).
- A sprite material with `"scene_filter": true` runs before UI. Binding 4
  instead reads the preceding filter's image, initially the scene after
  atmosphere/occlusion, TAA, bloom, tone mapping and FXAA. Write those display
  values directly; do not tone-map again. Each sprite runs as a separate pass
  in submission order, preserving uncovered pixels and alpha blending. The
  host applies radial blur and stencil-masked chromatic aberration afterwards;
  UI/insets and accessibility color filters retain their usual order. Target
  textures are reused across frames; with no active filters, no extra pass or
  target allocation occurs. Standalone overlays/loading do not run scene filters.

Projections map depth into WebGPU's 0 to 1. pen compiles only the pipelines
the loaded materials draw with, on background threads; a shader that does
not compile is reported (an `AssetFailed` event, and the log) and its draws
are left out, never the frame.

## Persistent mesh instances

A mesh JSON can add `"instances": {"data": [...], "min": [x, y, z],
"max": [x, y, z]}`. `data` contains one or more records of exactly 20 finite
numbers. `min` and `max` must enclose **all reconstructed geometry** in mesh
space. Pen culls the aggregate bounds, grown by the shader's `bounds_margin`,
and applies the draw transform to the whole patch. It does not cull individual
records. Split large patches into separate mesh assets when finer culling matters.

An optional positive finite `"scale"` (default 1) multiplies JSON record values
once on load. For example, integer records with `"scale": 0.000244140625`
use a 1/4096-unit grid. Bounds remain in mesh units. The asset author chooses
precision; Pen's packed files and GPU buffers hold the expanded float32 values.
This can make both source assets and compressed packages smaller without
changing the shader layout or the number of instances.

Records are uploaded with the asset, retained on the GPU, and replaced on hot
reload. A frame sends one ordinary mesh draw; there are no per-record packet
entries or frame uploads. Frame instance-upload counters exclude asset loading.
The renderer's instance and triangle counts include the repeated geometry.

Persistent meshes require an opaque material and `vertex_persistent`; they
support scene, inset and shadow passes, but cannot have a skin or a posed draw.
Automatic export simplification leaves these meshes alone; authored LOD errors
must bound the reconstructed geometry. The template's vertex attributes remain
available at the ordinary locations, including `vertex_index` and `instance_index`
builtins when declared.

In packed PMSH version 1, flag 16 appends the instance count (`u32`), bounds
(six `f32`s, minimum then maximum), and 20 `f32`s per record after indices and
any LODs. All are little-endian. Flag 16 cannot coexist with skin flag 4. Older
hosts reject this flag; rebuild native and browser hosts when adopting it.

Packaging writes PMSH version 2 when it is smaller: it groups corresponding
bytes of vertex, index, skin and instance records, then compresses with zlib.
Its 16-byte header is `PMSH`, version 2, the mesh flags, two zero bytes, the
expanded version-1 byte count (`u32`), and four zero bytes. The zlib payload
contains the complete shuffled version-1 file; count tables, bounds and the
skin's rig remain in place. Loaders reverse the shuffle before parsing, with
every float and index unchanged. Compressed meshes may expand to at most
256 MiB. Larger or incompressible meshes keep version 1. Native and browser
hosts read both; rebuild the hosts to ship version 2. Runtime compatibility
checks include the mesh codec so packaging cannot pair it with an older player.
