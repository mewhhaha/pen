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
frame's `aberration`. A `surface` object adds the lobes of glTF's clear coat,
sheen, anisotropy, transmission and volume, ior, specular and emissive strength
extensions (below).

### The `surface` block

Every key is optional, numbers are plain floats and a key left out takes its
default; a material without the block has all of them. It is part of the
material's render state, not an instance parameter: draws of materials that
differ in it never share an instanced batch.

| Key | Default | Range |
|---|---|---|
| `clearcoat`, `clearcoatRoughness` | 0, 0 | 0 to 1 |
| `sheenColor` (three channels), `sheenRoughness` | [0, 0, 0], 0 | 0 to 1 |
| `anisotropy`, `anisotropyRotation` | 0, 0 | -1 to 1; radians |
| `transmission` | 0 | 0 to 1 |
| `ior` | 1.5 | at least 1 |
| `thickness`, `attenuationDistance` | 0, 0 | at least 0, in model units; thickness 0 is thin-walled and distance 0 is infinite |
| `attenuationColor` (three channels) | [1, 1, 1] | 0 to 1 |
| `specular`, `specularColor` (three channels) | 1, [1, 1, 1] | at least 0 |
| `emissiveStrength` | 1 | at least 0 |

A material with `transmission` above 0 draws among the translucent surfaces
even with an opaque `color` alpha (it reads the opaque picture behind it), and
casts no shadow. Textures for these lobes are not supported.

## What a draw binds

Group 0, the frame's, the same for every draw:

| Binding | What |
|---|---|
| 0 | The scene, a uniform: `view` and `projection` (column-major `mat4x4f`), `ambient`, `light_direction` and `light_color` (`vec4f`, xyz used), `light_view_projection` (the farthest sun map's), `shadow` (x 1 with sun shadows, y a map texel in texture coordinates, z how far off the surface the single map reads, in world units), the sky's `zenith`, `horizon` and `ground` colours, `eye`, where the camera is, then the sun's cascades: `cascades` (x how many maps, y the share of a slice over which a map blends into the next, z the filter, 0 hard, 1 PCF, 2 soft, w the contact shadow's steps), `cascade_view_projection` (four `mat4x4f`, world to each map's clip space, nearest first), `cascade_texels` (four `vec4f`: a texel in texture coordinates, a texel in world units, how far ahead of the camera the map's slice ends, its depth range in world units) and `shadow_params` (x the penumbra's width per unit of blocker distance, y how far contact shadows look). |
| 1 | The farthest sun map, `texture_depth_2d` (what shaders that predate cascades read; they get one hard map). |
| 2 | Its `sampler_comparison`. |
| 3 | Every sun map, a `texture_depth_2d_array` (`shadow_cascades`). |
| 4 | The sky's reflection, a `texture_cube<f32>` whose levels are the sky blurred by GGX lobes of growing roughness. |
| 5 | The split-sum BRDF table, a `texture_2d<f32>`. |
| 6 | A filtering sampler for both. |
| 7 | The sky's light, a uniform: nine diffuse spherical-harmonic coefficients, the reflection tint, the cube's last level and the probe volume's origin, spacing and counts. |
| 8 | The baked light probes, a read-only storage array of `vec4f`: seven a probe (nine RGB terms and a validity weight). |
| 9 | The camera's depth from the frame's casters, a `texture_depth_2d` of the scene's size, drawn only when the frame asks for contact shadows. |

## The sun's shadows

The game fits the sun's shadow maps (Ink's `graphics.shadow`): up to four
cascades along one light view, each an orthographic window round a slice of the
camera's view whose centre moves in whole texels, so that shadows hold still as
the camera moves. They come in the render packet's `cascades` section, nearest
first, with `shadow.texels` (each map's width), `shadow.filter`,
`shadow.softness`, `shadow.contact` and `shadow.contact_length`. Which side
fits them is decided by who has the camera, the light and the shadow distance
policy: the game does, pen draws and validates what it is given. A shader adds
`pen_render::SHADOWS_WGSL` after declaring `Scene` with every field through
`shadow_params`; it declares bindings 2, 3 and 9 and gives

- `sun_shadow(world, normal, position)`: the sun's visibility, 1 in the sun and
  0 in shadow, with `position` the fragment's `@builtin(position)`: the
  cascades' filtered visibility, blended into the next map over the last
  `cascades.y` of each slice (and into the sun past the last), and the contact
  shadow's, whichever the frame asks for. The filter's pattern is fixed per
  pixel, so temporal antialiasing averages it away rather than shimmering.
- `sun_visibility(world, normal, spin)`: the cascades alone, with the filter's
  disk turned by `spin` (cos, sin).
- `contact_shadow(world, normal, jitter)`: a short march in screen space toward
  the sun through the depth buffer (binding 9), shadowing what dips just behind
  what the camera sees; it needs a perspective camera.

The filters: 0 is one hardware-compared tap; 1 averages sixteen taps over a
rotated Poisson disk of 1.5 texels; 2 (percentage-closer soft shadows) searches
twelve taps for blockers, averages how far they are from the receiver and
filters over a disk as wide as `softness` times that distance, so shadows are
sharp at a caster's foot and soft away from it. Depth is linear (the maps are
orthographic), so each tap compares against the receiver's own plane, and the
surface moves along its normal by 1.5 texels at grazing angles to the light:
no acne on a lit plane at any angle, and no shadow lifted off its caster.

## Local light shadows

A frame's point and spot lights may ask to cast (`Light::shadow`, with a stable
`id`). The host picks at most `shadow.local` of them each frame: those whose
reach touches the camera's view, by size on screen (reach over distance), then
lower id, then earlier index, as far as 18 map faces fit (a spot is one
perspective face along its cone, a wider-than-160-degree one or a point light a
cube of six). They share one depth texture array of `shadow.local_texels`
faces; each face's window reaches a few texels beyond its edge, so the filter
never leaves it and cube faces show no seams. A light keeps its layers while
it stays picked, and a face whose casters (and light) did not change since it
was drawn is not drawn again (`RenderStats::local_shadow_kept`). `LIGHTS_WGSL`
declares them at group 2 (binding 4 the maps, 5 their comparison sampler, 6
each layer's matrix, and `ClusterLight::shadow`); `clustered_surface` and
`clustered_light` use them, with eight taps over a disk, and `local_shadow(light,
world, normal, way, distance)` gives a light's visibility alone.

Bindings 4 to 8, 10 and 11 are the sky as light and the local reflection probes'
cubes and table; 12 and 13 are the camera's depth (two layers, this frame's and
last frame's) and last frame's picture for screen-space reflections. A shader
that wants them adds
`pen_render::ENVIRONMENT_WGSL`, which declares them and defines
`ambient_light(world, normal)` (the light all round, to multiply by albedo; the
flat `ambient` color's replacement: the game's baked probe volume where it sends
one, blended into the sky's light past its edge), `sky_reflection(world, normal, view, roughness,
f0)` (the prefiltered sky, the reflection probes the surface is inside of, with
box-projected lookups, and the screen-space march, by confidence: see
`ARCHITECTURE.md`), `surface_ambient(world, normal, view, albedo, metallic, roughness)` for
both together, and `specular_occlusion`. A game that sends no coefficients gets
its flat `ambient` from `ambient_light` and no reflections. The point and spot
lights of `LIGHTS_WGSL` also have `clustered_surface(world, normal, view,
roughness, f0, position, view_z)`, their diffuse sum and GGX highlights. Both
use the lobe of ink's `lighting.brdf` (GGX with alpha the square of the
roughness, height-correlated Smith visibility, Schlick Fresnel).

Group 1, the draw's:

| Binding | What |
|---|---|
| 0 | The draw, a uniform: `model` and `normal` matrices, the material's `color`, `settings` (x 1 when lit, y the alpha cutoff, z 1 when it has a normal texture or a field, w 1 when it has a layer texture), the four `parameters`, then 32 `vec4f` of the draw's `extra` (zero past what it sent), then, from byte 688, `surface: array<vec4f, 5>` from the material's `surface` block: (clearcoat, clearcoat roughness, sheen roughness, anisotropy), (sheen color, anisotropy rotation), (transmission, ior, thickness, attenuation distance), (attenuation color, specular) and (specular color, emissive strength). |
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

## Probe volumes

A game that bakes light probes (ink's `probes`) sends the volume with its frame:
`probe_volume` (revision, counts, spacing, origin) and, when the bake changed, the
`probes` section (28 lanes each: nine RGB spherical-harmonic terms, then 1 for a
probe in open space or 0 for one inside geometry). The host keeps the probes
it was last sent and uses them while `probe_volume.revision` and the counts
match, so a frame without the section costs nothing; one that sends it
replaces them (at most 16384, filling the volume). `ambient_light` interpolates
trilinearly between the eight probes round the point lifted a quarter spacing
along the normal, each weighted by validity and by how much it faces the
surface, and blends toward the sky's light with the volume's confidence: zero
outside it, fading in over one cell beyond the outermost probes.

## Look-up tables for the colour grade

The frame's `grade.lut_a` and `lut_b` name a table by a material: 1 plus its
catalog index, whose `texture` is the table's strip, and 0 for none. A table of
N slices (N from 2 to 64) is a strip N * N texels wide and N tall: the slice is
blue, red runs along it and green down, so texel (r, g, b) is at x = b * N + r,
y = g and an identity table holds (r, g, b) / (N - 1). It is read as data, not as
a colour picture: an 8-bit PNG key is read raw (the sRGB decoding a colour
texture gets is not applied), and its colours are in the grade's own log encoding,
ACEScc (the Academy's S-2014-003 curve, on the frame's own primaries): 0.41 is
middle grey, 0 a thousandth of SDR white and 1 more than two hundred times it,
with values outside 0 to 1 passing through unchanged. The material needs a
shader like any other (it is never drawn), and the host uses the table once its
finest level has streamed in. Ink's `color.grade` and `color.lut_lookup` evaluate
the same table on the CPU. A `.cube` file from a grading tool is display-referred,
so it does not drop in: bake it through the grade's log encoding first.
