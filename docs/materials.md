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
`decals`, casts a `shadow` (yes unless false), splits its colours by the
frame's `aberration` or is `subsurface` (its opaque surfaces are diffused on screen,
as skin is, [below](#subsurface-the-cheap-kind-and-the-screen-space-kind)). A `surface` object adds the lobes of glTF's clear coat,
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
even with an opaque `color` alpha (it reads the opaque picture behind it), casts no
shadow and does not share an instanced batch (the tests in `tests/materials_integration.rs`
hold each of these to the renderer). Textures for these lobes are not supported.

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
| 14 | The opaque picture, a `texture_2d<f32>` (`Rgba16Float`) with its whole chain of levels: level 0 is the frame's opaque surfaces after the sky, the ambient occlusion and the mist, at the scene's size, and each next level the last halved by a 4 x 4 tent, down to a texel. Fragment stage only. |
| 15 | The nearest depth buffer value of each pixel of that picture (the projection's own, 0 to 1; the nearest of a pixel's samples; 1 where nothing opaque is), a `texture_2d<f32>` (`R32Float`, read with `textureLoad`). Fragment stage only. |
| 16 | The integrated froxel volume of the mist, a `texture_3d<f32>`: layer k holds the in-scatter (rgb) and the optical depth (a) up to boundary k of the grid. Fragment stage only. |
| 17 | The fog block, a uniform of 512 bytes: the volume's own uniform followed by the eye, the sky's colours, the sun, the clouds and the scene's jittered view-projection. Fragment stage only. |

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
`ARCHITECTURE.md`; `sky_reflection_lobe` takes the lookup's normal and the
cosine to reflect separately, and a grazing reflectance), and `specular_occlusion`.
A game that sends no coefficients gets its flat `ambient` from `ambient_light`
and no reflections. `pen_render::LIGHTS_WGSL` declares the frame's lights (group
2) and gives `clustered_light(world, normal, position, view_z)`, their diffuse
sum. The lighting of a glossy surface with both is `SURFACE_LOBES_WGSL` and
`SURFACE_WGSL`, below: `surface_ambient(world, normal, view, albedo, metallic,
roughness)` and `clustered_surface(world, normal, view, roughness, f0, position,
view_z)` are its base lobe's two halves, as ink's `lighting.brdf` has it (GGX
with alpha the square of the roughness, height-correlated Smith visibility,
Schlick Fresnel), and give exactly the light they always have.

## Translucent surfaces: the opaque picture and the mist

A translucent material, a particle or an overlay is drawn after the frame's opaque
surfaces have had their sky, ambient occlusion and mist, over that picture, so it is
misted by the air between it and the eye and not by what is behind it, and it can read
what is behind it. `pen_render::TRANSLUCENCY_WGSL` (after `ENVIRONMENT_WGSL`, which declares
the `sky_sampler` of binding 6 it samples with) declares bindings 14 to 17 of group 0, for
the fragment stage only, and gives:

- `fog_apply(color, world)`: `color` as the eye sees it through the mist from the eye to
  the point `world` (the surface's fragment): `color * transmittance + in-scatter`, the
  numbers the atmosphere pass gives an opaque surface there (the froxel volume at the
  point's depth, then the same medium analytically past its reach or without a grid, and
  the clouds' shadow on the surface). A translucent material outputs
  `vec4f(fog_apply(lit, world), alpha)`; the ordinary blend then composites it right.
  Light that only adds (a particle's) takes the transmittance alone:
  `fog_terms(world)` is the in-scatter (rgb) and the transmittance (a).
- `opaque_available()`: whether this pass reads an opaque picture.
- `opaque_screen_uv(position)`: a fragment's place in it, from `@builtin(position)`, 0 to
  1, y down; `opaque_world_uv(world)`: a world point's place, through the projection the
  scene draws with, TAA's jitter included.
- `opaque_behind(uv, level)`: the picture at `uv`, clamped to it, read trilinearly at
  `level` (clamped to the chain, 0 sharp, `opaque_levels()` a texel): a refracted or a
  frosted view of what is behind. Black where there is no picture.
- `opaque_levels()`: the chain's last level.
- `opaque_depth_at(uv)`: how far ahead of the eye, along the view, the opaque surface
  behind `uv` is; 1e30 where there is none.

In every pass that has no opaque picture to give (the opaque draws themselves, the shadow
passes, a reflection probe's faces, insets) the bindings hold stand-ins, `fog_apply`
returns its colour, `opaque_available()` is false and the rest give black and 1e30: one
shader serves opaque and translucent draws, and the atmosphere pass mists the opaque
ones. Frames without translucent surfaces, particles or overlays do not make the picture
at all. With this the fragment stage has 15 sampled textures at most (12 before).
Particles are misted by the renderer, at their middle (`particle_quads.wgsl`).

Group 1, the draw's:

| Binding | What |
|---|---|
| 0 | The draw, a uniform: `model` and `normal` matrices, the material's `color`, `settings` (x 1 when lit, y the alpha cutoff, z 1 when it has a normal texture or a field, w 1 when it has a layer texture), the four `parameters`, then 32 `vec4f` of the draw's `extra` (zero past what it sent), then, from byte 688, `surface: array<vec4f, 5>` from the material's `surface` block: (clearcoat, clearcoat roughness, sheen roughness, anisotropy), (sheen color, anisotropy rotation), (transmission, ior, thickness, attenuation distance), (attenuation color, specular) and (specular color, emissive strength).  See [The standard surface](#the-standard-surface). |
| 1 | The material's `texture`, or white without one (a checker when it failed to load). |
| 2 | A filtering sampler, at the frame's quality. |
| 3 | The normal texture, or the material's field (the render packet's `fields`), or a flat normal. |
| 4 | The layer texture, or transparent. `parallax_uv` reads its red channel as a height field. |

A shader declares only what it reads: a struct may stop after its leading
fields, and an unused binding may be left out, as `lit.wgsl` leaves out all
but the two uniforms.

## The standard surface

A material shader that wants more than albedo, Lambert and a mirror adds four
libraries, in this order: `pen_render::LIGHTS_WGSL`, `ENVIRONMENT_WGSL`,
`SURFACE_LOBES_WGSL` and `SURFACE_WGSL` (the `lit.wgsl` of `pen new` has all of
them, and `SHADOWS_WGSL` for the sun's shadow). `SURFACE_LOBES_WGSL` is pure
functions with no bindings: a `StandardSurface`, every lobe's distribution,
visibility, Fresnel and tables, `surface_light` (one light on a surface), the
sun (`standard_sun`) and parallax occlusion mapping. `SURFACE_WGSL` lights a
`StandardSurface` with the sky and the frame's lights: `standard_surface(s)`.

```wgsl
var s = surface_default(world, normal, view, base_color, metallic, roughness, position, view_z);
s = surface_apply_block(s, draw.surface, LOBES, length(draw.model[0].xyz));
let light = standard_surface(s)
          + standard_sun(s, scene.light_direction.xyz, scene.light_color.rgb, sun_shadow(world, normal, position));
```

`surface_default` is the base lobe alone: `world`, `normal` and `view` (unit, toward
the eye), `base_color` (albedo, or a metal's color), `metallic`, `roughness`, and
the fragment's `@builtin(position)` and its view-space z (negative ahead of the
eye), which find its cluster of lights. Everything else of a `StandardSurface` is
set to what the base lobe needs: `occlusion` (1 open, 0 closed: it darkens the
sky's light, and the sky's reflection by `specular_occlusion`), `f0` (0.04
mixed to `base_color` by `metallic`) and `f90` (1), `coat_normal` (the shading
normal). A shader that has a normal map sets `s.coat_normal` to the mesh's own
normal.

**Lobes.** `s.lobes` holds `SURFACE_*` flags, named from constants so that the
compiler leaves out a lobe the material does not have, and each lobe also checks its
weight at run time, so that a weight of 0 gives the base lobe's bits.

| Flag | Parameters | What it adds and costs |
|---|---|---|
| (always) | `base_color`, `metallic`, `roughness`, `f0`, `f90`, `occlusion` | GGX with height-correlated Smith visibility and Schlick Fresnel, Lambert, the sky's diffuse and reflection. One distribution, one visibility and a Fresnel per light. |
| `SURFACE_COAT` | `coat` (0 to 1), `coat_roughness`, `coat_normal` | A second GGX lobe over everything, with the base lobe's visibility at the coat's alpha and normal and its Fresnel (0.04 to 1) at the half vector; the lobes under it keep `1 - coat * Fresnel` at the view's angle to the coat normal, for every light. One more GGX per light and one more sky reflection (a second cube lookup, a second probe and SSR march where the frame has them), which is the cost to watch. |
| `SURFACE_SHEEN` | `sheen_color`, `sheen_roughness` | Charlie's distribution with Neubelt's visibility (cloth's rim light), `s * D * V * cos`; the lobes under it keep `1 - max(s) * Es` where `Es` is the lobe's directional albedo from a 32 by 16 table. A few `pow` and a table read per light; the sky adds `s * Es` of its diffuse light. The sheen is not dimmed by the coat. |
| `SURFACE_ANISOTROPY` | `anisotropy` (-1 to 1), a frame (`surface_frame`) | The specular lobe stretched along the tangent: widths `mix(alpha, 1, a^2)` and `alpha`, with the visibility that goes with it; the sky's reflection looks up a normal bent toward the view across the grain. The sign of `anisotropy` only turns that bend (the lights' highlight depends on its square): turn the brushing with the frame's rotation. As cheap as the isotropic lobe plus a few dot products. |
| `SURFACE_SUBSURFACE` | `subsurface` (0 to 1), `subsurface_color`, `thin` (0 a leaf, 1 opaque) | The diffuse lobe's cosine becomes a wrapped one (`wrap(x, 0.5)` times the color plus the cosine), so light reaches past the terminator, and a back-lit term lets the light through to a viewer on the other side. Lights behind the surface count for it (their shadow maps are read). Cheap. The screen-space diffusion for skin is a separate pass, not this (see [Subsurface](#subsurface-the-cheap-kind-and-the-screen-space-kind)). |
| `SURFACE_TRANSMISSION` | `transmission` (0 to 1), `ior`, `thickness` (world units), `attenuation_color`, `attenuation_distance`, `behind` | The diffuse lobe gives way to the frame's picture of what is behind the surface (`surface_transmit`, in `SURFACE_TRANSMISSION_WGSL`, which needs the translucency library's `opaque_*` functions): a thin wall (`thickness` 0) reads the pixel behind the fragment, a volume refracts the view's ray in (Snell, `1 / ior`) and reads where it leaves `thickness` along, through Beer-Lambert's `pow(attenuation_color, thickness / attenuation_distance)`, at a mip level that grows with roughness and `2 ior - 2`. Lights have no transmitted term. One texture read. |
| `SURFACE_SPECULAR` | `ior`, `specular`, `specular_color` (`surface_apply_block`) | `f0 = min(((ior - 1) / (ior + 1))^2 * specular_color, 1) * specular` and `f90 = specular` for a dielectric, mixed to the base color by `metallic` (`SURFACE_TRANSMISSION` does the same). No run-time cost. |

The lobes layer as `(diffuse * (1 - transmission) + specular) * sheen_keep *
coat_keep + sheen + coat`. Every lobe is in Ink's `lighting.brdf` convention (a
response per unit radiance, the cosine in it); the frame's lights are scaled so
that a diffuse surface is albedo times color times the cosine, so the specular
lobes here carry the factor pi that Ink's carry in the lights' own scale.
`tools/native/src/surface.rs` in Ink and pen-render's `surface` tests hold each lobe
to an independent f64 reference.

**The sun.** The sun is not a cluster light: its visibility comes from
`sun_shadow`. `standard_sun(s, sun_dir, sun_color, visibility)` lights `s` with it
through the same lobes (`sun_dir` the way its light travels, `scene.light_direction`).
Lambert alone, as shaders have always lit with it, is `base_color * sun_color *
(max(dot(normal, -sun_dir), 0) * visibility)`.

**The material's `surface` block.** The draw uniform ends with five `vec4f`:

| | x | y | z | w |
|---|---|---|---|---|
| `surface[0]` | clearcoat | clearcoat roughness | sheen roughness | anisotropy |
| `surface[1]` | sheen color r | g | b | anisotropy rotation (radians) |
| `surface[2]` | transmission | ior | thickness (model units) | attenuation distance (0: none) |
| `surface[3]` | attenuation color r | g | b | specular |
| `surface[4]` | specular color r | g | b | emissive strength |

A material's JSON may give these as a `surface` object (every key optional, camelCase:
`clearcoat`, `clearcoatRoughness`, `sheenColor`, `sheenRoughness`, `anisotropy`,
`anisotropyRotation`, `transmission`, `ior`, `thickness`, `attenuationColor`,
`attenuationDistance`, `specular`, `specularColor`, `emissiveStrength`). A material without
one has ior 1.5, specular 1, specular color and attenuation color white, emissive
strength 1 and the rest 0. `surface_apply_block(s,
block, lobes, model_scale)` reads it for the lobes `lobes` names (the thickness times
`model_scale`, `length(draw.model[0].xyz)`, is in world units); the anisotropy's
rotation is `block[1].w` for `surface_frame`. The subsurface has no block
parameters: set `s.subsurface`, `s.subsurface_color` and `s.thin` yourself.

**Frames for anisotropy and parallax.** A mesh may carry tangents (`@location(11)
tangent: vec4f`, xyz and the handedness in w); a mesh without them binds a zero
stream. `surface_derived_tangent(world, normal, uv)` makes one from the screen-space
derivatives of the position and the uv (call it in uniform control flow, where every
fragment does), and `surface_tangent(stream, derived)` takes the stream's where it
has one. `surface_frame(s, tangent.xyz, tangent.w, rotation)` makes it perpendicular to the
normal and turns it about the normal by `rotation`.

**Parallax occlusion mapping.** `parallax_uv(layers, layer_sampler, uv, view, light,
scale, lod, with_shadow)` returns the `uv` every texture read of the material should
use instead of the mesh's, its `height` there and, with `with_shadow`, a `shadow` from
0 (in a groove) to 1 to multiply the sun's visibility by. `layers` is the
material's layer texture (group 1, binding 4: `@group(1) @binding(4) var heights:
texture_2d<f32>;`, with the material sampler at 2), and **its red channel is the height
field**: 1 is the surface, 0 the deepest point, `scale` uv units below it (a shader
sets its own, often from a draw parameter); a layer texture used for something else
cannot also be a height map. `view` and `light` are unit vectors toward the eye and the
light in tangent space (`parallax_tangent_space(tangent, bitangent, normal, v)`; the uv's
own axes, so a mesh needs a uv-aligned frame). The march takes 8 to 32 layers
(`floor(32 - 24 |view.z|)`, the fewer the closer to straight down), stops at the first
whose height is at or above the ray, and puts the hit where the line to the layer before
crosses the height field, so that a plateau's shift is exact: `(h1 - h2) * scale *
tan(angle)` between plateaus of heights `h1` and `h2`. A view straight down, a hit at
the first layer, or a view below the surface leaves the uv alone. The self-shadow
marches toward the light the same way. `lod` is the layer texture's level: 0 for the base
(a shader that wants derivatives' worth of filtering passes its own). Bricks lying at a
grazing angle, with and without it, are in pen-render's `tests/standard_surface.rs`.

## Subsurface: the cheap kind and the screen-space kind

Light that enters a surface and leaves a little way off shows in two ways, and pen has one
for each.

**The cheap kind** is a lobe of the standard surface (`SURFACE_SUBSURFACE`, above): a wrapped
diffuse term, so light reaches past the terminator, and a back-lit term for a viewer on the
far side. It costs a few instructions per light and changes nothing but the shading of the
pixel it is on. Use it for foliage, thin cloth, wax, ears against the sun, paper and fruit,
where what shows is light getting through a thin thing, and for skin's soft terminator when
a blur is too much.

**The screen-space kind** is a material that says `"subsurface": true`. Its opaque surfaces
carry a stencil mark (bit 4, beside the aberration's 1 and the decals' 2), and once the
opaque surfaces and their decals are drawn, two passes blur the frame's picture where the
mark is (along x, then along y; after Jimenez et al., "Separable Subsurface Scattering",
2015) with a kernel that is the sum of Gaussians d'Eon and Luebke fitted to skin: red
spreads farthest, then green, then blue, and every edge of detail, every freckle and every
terminator softens and takes a warm glow. It runs before the sky, the ambient occlusion and the
mist, so skin is diffused and then misted, and anything translucent is drawn over the diffused
picture. Use it for skin: faces and hands. It costs two full-screen passes over the picture
(a pixel that is not marked is copied; a marked one reads 15 taps of four texels) in frames
that have such a draw, and nothing at all in frames that do not: no passes, no textures.

```json
{ "shader": "skin.wgsl", "color": [0.78, 0.52, 0.42, 1], "lighting": "lit", "subsurface": true }
```

What it does and does not do:

- A pixel is blurred only if every one of its samples is a `subsurface` surface; any other pixel
  (the sky, other materials, a silhouette's partly covered pixels) is copied bit for bit and
  neither gives nor takes light. A tap on one of those, or on a surface more than four widths
  nearer or farther along the view (a linear ramp from two), gives the pixel's own colour
  instead, so the blur stops at an edge: a nose does not bleed into the cheek behind it, and a
  white of the eye is not tinted by skin.
- The taps reach `1 cm` (a world unit is a metre) from a pixel and, in pixels, as far as that is at
  the pixel's depth: a head close to the eye is blurred over more pixels than one far off (at
  most 64). The reach is one constant for every frame; the render packet has no field for it,
  and neither has the draw uniform. A character at another scale needs a model whose units
  are metres.
- The picture holds the specular light as well, so it is blurred with the diffuse: a sharp
  highlight on skin softens too, which it should not. A separate specular buffer is a
  follow-up.
- It acts on opaque surfaces of the camera's picture only; insets and reflection probes do not
  diffuse, and a translucent material that says `subsurface` is drawn as it would be without.
- It is not tied to the `quality` settings (the render packet has no field for it): a game
  that wants it off draws its skin with a material that does not say `subsurface`.
- Shade the skin as you would without it: the albedo, the lights and the shadows are the
  material shader's; the pass only spreads what was drawn. The result conserves the picture's
  light (a flat picture stays flat), so a skin that looked right unblurred gets softer, not
  brighter or darker.

Both kinds can be used on one material (a lit skin with the wrapped terminator that also
says `"subsurface": true`), but the blur already softens the terminator, so most skin wants
the wrap turned down.

## Vertices and entry points

A mesh's vertex is `@location(0) position: vec3f`, `@location(1) normal:
vec3f`, `@location(2) uv: vec2f` and, when the mesh has `attributes`,
`@location(8) attributes: vec2f`. Every mesh pipeline also offers
`@location(11) tangent: vec4f`, from a vertex buffer of its own: the mesh's
`tangents` (four numbers a vertex, xyz and a handedness w of -1 or 1, the
bitangent being `cross(normal, tangent.xyz) * tangent.w`), or zeros where it has
none, and for brush solids; a shader that does not declare the location is
unaffected, and one that does treats a tangent of no length as "derive the frame
from the screen-space derivatives of position and uv". Locations 3 to 7 are
an instance's below, so the tangent does not sit among them.

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
  material's texture (2), a sampler (3), and the raw resolved scene (4): the picture
  the frame's draws made before the sky, the mist and the passes after the scene, or,
  in a frame with translucent surfaces, particles or overlays, the picture the translucent
  pass resolved to, with them in it and the sky and mist the opaque picture got.
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
