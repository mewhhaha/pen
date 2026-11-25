# How pen-render works

pen-render draws a game's render packets (`pen_protocol::RenderFrame`) with
wgpu. It runs natively (Vulkan, DX12, Metal) and in the browser (WebGPU) from
the same code. The host owns the window, the device and the frame loop. It
calls `Renderer::submit` once per frame, and calls `Renderer::overlay` to draw
another program's sprites, such as pen's dev tools, over a finished frame.
[Measurements](../../docs/render-performance.md) cover load times, frame
costs, culling, levels of detail, shadows and FXAA.

## Public API

- `Renderer::new(device, queue, RendererOptions)`: `format` is the
  swapchain's non-sRGB format (the present pass encodes colours itself), and
  `display` how the surface shows colours: `Display::SDR`, or the HDR color
  space `choose_surface(capabilities, hdr_info, hdr)` picked with `format`.
  `assets` is the asset root, holding `catalog.json` and every asset key.
  `veil` is how long the loading veil takes to clear, or None for no veil.
  `pipeline_cache` is a directory to keep compiled pipelines in between runs.
- `optional_features(&adapter)`: device features the renderer uses when the
  adapter has them: `PIPELINE_CACHE` (only Vulkan offers it),
  `TIMESTAMP_QUERY`, which times the GPU's passes, and the texture
  compression families (BC, ASTC, ETC2) KTX2 variants are chosen by. The
  host requests them with the device.
- `preload(catalog)` prefetches every mesh and material, with their textures
  and shaders, behind whatever frames ask for.
- `preload_scene(assets, samples)` loads what a scene draws first, `(kind,
  key)` pairs, all at once and before any frame asks, with pipelines for the
  scene's sample count; the catalog's preload waits until they are in.
  `Prefetch::start(root, assets)` reads and decodes such a list on threads of
  its own before there is a renderer (while the host opens the GPU), and
  `warm(prefetch)` has the renderer take those assets' first loads from it.
- `inspect_texture`, `inspect_mesh`, `inspect_material` and
  `shader_draws_sprites` read an asset as the renderer would, without a GPU,
  for tools (pen's asset lints).
- `submit(frame, catalog, target)` validates and draws a frame. Asset IDs in
  the frame are catalog indices.
- `share(assets)` makes a second renderer for another program's assets. It
  shares the device, the loader threads and the pipeline cache, and keeps its
  own loads and events. `prepare_sprites` and `overlay` draw that program's
  sprites over the game's frame.
- `take_events()` returns `AssetReady`/`AssetFailed` per asset revision.
  `reload(keys)` and `reload_all()` hot-reload assets. `quiet()` and
  `pending()` report whether loading has settled. `wait(timeout)` blocks
  until the loader brings something.
- `pack_mesh(bytes)` turns mesh JSON into the compact binary form the
  renderer also reads (below). With the `simplify` feature,
  `pack_mesh_with_lods(bytes)` adds simpler levels of detail (pen's packaging
  uses it).
- `passes()` names the passes after the scene the last frame drew.
- `LIGHTS_WGSL`, `TRANSLUCENCY_WGSL` and `MSDF_WGSL` are WGSL that material
  and sprite shaders add to light with the frame's point and spot lights, to
  mist a translucent surface at its own depth and read the opaque picture
  behind it, and to draw text from a multi-channel distance field (below). `SURFACE_LOBES_WGSL`, `SURFACE_WGSL` and
  `SURFACE_TRANSMISSION_WGSL` are the standard surface (below).
- `stats()` says what the last frame drew: draw calls, instances, triangles,
  what culling left out, the render scale, and the GPU's time over each part
  of a recent frame (`GpuTimes`) where the device has timestamps.
- `gpu_culling(from)` sets how many instances a batch needs to be culled on
  the GPU (256 by default; None culls all on the CPU), and
  `frame_budget(milliseconds)` turns on dynamic resolution.

## Assets

`asset_store.rs` keeps every asset by kind and key in a slot whose index
stays fixed. A frame asks for what it draws with `get`, which loads the asset
at once. `preload` queues prefetches a few at a time, behind those. Each load
has a revision. A reload starts a new revision, and the old one draws on until
the new one is ready, or for good if the new one fails.

`loader.rs` runs the background work. Natively that is two to six threads;
in the browser, tasks on the page's event loop. Jobs are file reads, mesh and
material parsing, packed texture inflation, PNG/JPEG decoding (the browser
decodes with `createImageBitmap`), shader modules, and pipeline compiles. A
frame's requests go ahead of prefetches. Results come back over a channel
and are taken in at the start of each `submit`, or by `wait`. Renderers made
with `share` put their jobs on the same threads, each with its own results
channel. The threads end with the last renderer.

The GPU work happens on the thread that calls `submit`. Meshes become buffers
(`create_buffer_init`). Images are uploaded and get their mip levels drawn
(`mipmaps.rs`). Packed textures (`.gtex`, `packed_texture.rs`) stream in,
smallest level first. One shows once its levels up to 128 texels across are
in. Each finer level then fades in over about 0.12 s rather than popping. A
resource made on this thread is published once its error scopes come back
empty. Natively that is at once; in the browser the check settles later
(`shader::scoped`). An asset that fails raises `AssetFailed`; it never
crashes the renderer.

Meshes are JSON (`positions`, `normals`, `uvs`, optional `attributes`,
`indices`, optional `lods`) or the binary form `pack_mesh` makes. The binary
form starts with `PMSH`, version 1, flags (shader attributes present, 16-bit
indices, a skin, levels of detail), the vertex and index counts, then
little-endian vertex floats and indices; flagged, then the simpler levels'
count, each one's index count and error, and their indices, and then a
skin. A packaged `X.mesh.json` may hold either; the renderer goes by the
magic bytes. Both are checked the same way: matching arrays, indices inside
the vertex array, nonzero normals, finite float32 coordinates, and at most
three simpler levels with growing errors. Each mesh gets a bounding sphere
as it is parsed, which culls its draws. A skinned mesh adds four `joints`
and `weights` a vertex and its rig (`skeleton`, `skin`, `clips`;
`animation.rs`, which also writes and reads the rig's binary form after the
joints and weights); its joints and weights become a vertex buffer of their
own. A mesh may also have `tangents`, four floats a vertex (flag 32 of the
binary form: its own block after the vertices), which become a vertex buffer
too.

A level of detail keeps the mesh's vertices and leaves out triangles: its
indices follow the whole mesh's in the same index buffer, with the error (in
the mesh's units) its surface may stray from the whole mesh's.
`pack_mesh_with_lods` (`simplify.rs`, behind the `simplify` feature: pen's
packaging has it, the browser host does not) makes them with meshoptimizer's
simplifier (the pure Rust port, meshopt-rs) for meshes of 256 triangles or
more: as simple as it gets within 0.25%, 1% and 4% of the mesh's size, each
kept if it leaves out a quarter of the last one's triangles.

## Pipelines

A material shader (a WGSL asset) has up to six scene pipelines for each sample
count. They are opaque, opaque and instanced, opaque and skinned, translucent,
and two for overlays: one dimmed where the scene hides it, one at full
strength. It also has up to three depth-only shadow pipelines (plain,
instanced and skinned). Instanced ones need a `vertex_instanced` entry point
and skinned ones `vertex_skinned`. Only the variants its loaded materials draw
with are compiled (`Material::variants`):

- When a shader loads, the loader compiles its module. The renderer then
  queues one job per pipeline its materials need, at the sample count frames
  draw with, so the pipelines compile side by side on different threads.
  A caster's shadow pipelines are part of that set. The new revision is
  published only once that set is in, so a hot reload keeps drawing the old
  shader until then.
- A material that arrives after its shader asks for its variants (and
  shadows, if it casts) right away, ahead of the frames that draw it.
- The opaque pipeline at 4 samples, which insets draw with, compiles in the
  background for every shader, so an inset is never held up waiting for it.
- A new MSAA count takes effect only once every shader the frame draws with,
  and the passes around the scene, have compiled for it. Until then frames
  keep the old count. Nothing compiles on the frame's thread in steady state.

The renderer's own pipelines (`built_in.rs`) are the passes around the scene
in `post.rs`, decals, particles, and the inset composite. (The particles'
compute pipeline is made on the frame's thread the first time a frame has
emitters.) They compile on the loader threads too: the sample-independent ones
when the renderer is made, the rest when a sample count is first seen. The
loading veil covers the frame while they compile. A renderer without a veil
waits natively for the compiles already under way; in the browser it compiles
what it lacks then and there.

With `pipeline_cache` set and the feature granted, every pipeline is made
with one `wgpu::PipelineCache`. The cache is written to
`<dir>/<wgpu::util::pipeline_cache_key>` in the background when a burst of
compiles ends, and again when the renderer is dropped. wgpu and the driver
ignore a cache from another driver or wgpu version. pen passes
`~/.cache/pen/pipelines`.

A material with `transmission` above 0 is translucent even with an opaque
colour (`Material::opaque`), as it reads the opaque picture behind it.

Every mesh pipeline, shadows' included, reads vertex buffers in slots 0 to 2: the
mesh's vertices (locations 0 to 2 and 8), its tangents (location 11, one `vec4f`
a vertex), and the instances' transforms and parameters (locations 3 to 7) or a
skinned mesh's joints and weights (9 and 10). A mesh without tangents, and each
brush solid, binds one shared buffer of zeros that is as large as the largest
mesh meshed without any (`State::zero_tangents`, grown by doubling when a mesh
arrives), so no pipeline needs a variant for tangents and shaders that leave
out location 11 are unaffected.

## A frame

`renderer.rs` holds the state and the types its parts share. The parts are
in `renderer/`:

- `assets.rs`: what arrives from the loader, shader staging, streaming
  levels, detail fading in, hot reload.
- `materials.rs`: the frame's sample count, and what a draw binds. That is
  its material, its shader's pipelines, and its textures, in a bind group
  cached by texture views and sampler. A material's textures must all be
  ready before it draws; a texture that failed shows as a checker.
- `batching.rs`: placing the draws, the solids, the uniforms and the passes.
  Opaque draws of a mesh and material whose shader has `vertex_instanced`
  share an instanced batch. Materials with `instanceParameters` share one
  across their render state, so animation poses draw together. Draws with
  their own `extra` data draw alone. The render state includes the material's
  `surface` block, written for every draw after its extra vectors (byte 688 of
  the draw uniform, `DRAW_BYTES` 768) by the one function that fills the
  draws' uniforms. Brush solids are meshed from their planes (`solids.rs`); only the solids that changed are meshed again. Each
  draw is then culled (below). Each item then gets its uniforms and
  pipeline; one no view sees still loads and compiles what it draws with,
  but writes no uniforms. The passes are sorted next: opaque surfaces by
  stencil mark and material, translucent ones back to front, and casters.
  All of the frame's working lists live in a `Scratch` that the next frame
  reuses.
- `gpu_culling.rs`: instanced batches of 256 instances or more, culled by a
  compute pass (`shaders/cull.wgsl`) into indirect draws.
- `shadows.rs` and `cascades.rs`: the scene uniform, the sun's shadow maps,
  and the shadow passes (below).
- `timing.rs` and `readback.rs`: GPU timestamps around the passes, read back
  a few frames later, as results of the GPU are.
- `resolution.rs`: dynamic resolution (below).
- `insets.rs`: insets. Each is a few draws seen by their own camera, drawn at
  4× MSAA (and 2× resolution unless pixelated) into their own targets, then
  laid over the output. The first targets fit the inset; targets for one
  that grows have room to spare.
- `fields.rs`: per-material half-float fields. The game patches them each
  frame, and draws read them at binding 3 in place of the normal map.
- `skinning.rs`: posed draws. The frame's pose layers sample their mesh's
  clips (`animation.rs`); a posed draw of a skinned mesh whose shader has
  `vertex_skinned` draws alone with its skinned pipeline, its joint matrices
  laid in one storage buffer for the frame at group 2, and casts a skinned
  shadow.
- `particles.rs`: the frame's emitters. Which particles are alive follows
  from each emitter's clock; a compute pass places them from its seed, and
  they draw as camera-facing squares after the translucent surfaces.
- `veil.rs`: the loading veil. It shows the frame's sky with a turning arc,
  and clears once nothing the frame draws is still on its way (or after 12 s).
- `validate.rs`: what a frame must be for the renderer to draw it.
- `thin_gbuffer.rs`: rasterized Thin G-Buffer prepass (`normal_roughness` and
  `albedo_metallic` in `Rgba16Float` plus sampleable `Depth32Float`, supporting
  Forward-Z and Reversed-Z, plain/morphed, instanced, skinned, and brush draws)
  feeding ray tracing and ReSTIR without primary TLAS lookup rays.
- `wired_passes/`: modular high-fidelity render passes wired into `frame.rs`:
  - `sky_env.rs`: Hillaire physical atmosphere LUTs, volumetric clouds, and
    outdoor CDLOD terrain / Gerstner water / instanced foliage passes.
  - `decals_oit.rs`: PBR deferred decals and weighted-blended OIT.
  - `ddgi.rs`: Dynamic Diffuse Global Illumination probe grid updates and
    packed cascade upload.
  - `rt.rs`: Hardware ray-query / software BVH TLAS tracing (`ray_tracing/`),
    Tier-3 `RestirRenderer` (ReSTIR DI + GI with Jacobian-aware Pairwise MIS),
    `RestirSpecularRenderer` (glossy VNDF reservoir reuse with virtual
    reflected-point parallax motion & thin-wall visibility validation),
    `SharcCache` (spatial hash radiance cache for multi-bounce GI), and
    `SvgfDenoiser` (with hit-distance-adaptive specular A-Trous denoising).
  - `meshlets.rs`: Two-phase GPU meshlet occlusion culling (`meshlets.rs`) with
    Hi-Z re-testing of disoccluded clusters in a single submission, alongside
    standalone continuous LODs in `meshlet_dag.rs` (boundary-locked QEM cluster
    DAG + GPU screen-space error cut).
  - `post_split.rs`: AMD FSR 1/2 temporal upscaling + RCAS sharpening, polish
    post-FX, and render-to-texture / split-screen secondary cameras.
- `frame.rs`: `submit` puts the frame together and encodes its passes. The
  culling compute pass comes first, then the shadow passes, then the
  particles' compute pass. The scene draws opaque surfaces, then decals over
  them if there are any, and resolves. If any opaque surface's material is
  `subsurface`, two passes diffuse the resolved picture where they are marked
  (below). The opaque picture's passes follow (the
  occlusion, the sky and the mist on that picture alone, its nearest depth and
  a chain of blurred levels), then the translucent pass: the opaque picture
  written into every sample, and over it the translucent surfaces, the
  particles and the overlays. A frame with none of those has no such pass and
  no such chain. Then come the passes in `post.rs` (TAA, the metering, the
  bloom, present), then sprites (`sprites.rs`), insets and the veil (see
  "Translucent surfaces and particles" below).

### Culling and levels of detail

A material shader that declares how far its vertex stage may move a vertex
from where its mesh puts it, in the mesh's units (`const bounds_margin =
0.0;`), has its draws culled; one that does not is never culled, since the
renderer cannot know where its vertices land (`culling.rs`). The mesh's
sphere, grown by the margin, is placed by each draw's transform (its radius
stretched by the transform's longest axis, or the root of its axes' squared
lengths when it shears) and tested against each view the frame draws: the
camera's clip volume, each sun map's and each local light face's (at most
23 views, a bit each in a `u32`). A draw left out of a view would
put no pixel in it, so what is drawn does not change. A caster only the
shadow maps see still casts; one no view sees writes no uniforms.

The same draws pick a level of detail from the camera: the simplest whose
error, at the sphere's nearest point, spans at most a pixel of the scene's
target (at its render scale). Shadow maps draw what the camera picked. An
instanced batch is drawn once per level its instances use, each from its own
run of instances. Insets and solids are never culled.

Batches of 256 instances or more (`gpu_culling(from)`) are culled on the GPU,
where the device has compute shaders: their instances go into the instance
buffer whole, once, and `cull.wgsl` tests each against every view, picks its
level, and appends it to the instances its batch draws in that view at that
level, counting them into indirect draw arguments (one `draw_indexed_indirect`
per view and level; no indirect first instance is needed, as each draw binds
its own run of the culled instances). The order of instances within such a
batch follows the GPU's atomics, so it may change from frame to frame;
depth testing makes that invisible but where two instances are exactly
coplanar. How many instances each view kept comes back a few frames later,
into the stats.

### Shadows

The game fits the sun's shadow maps and the renderer draws them
(`cascades.rs`, `shadows.rs`). Up to four cascades (the packet's `cascades`
section, nearest first) share the frame's light view and differ in their
orthographic windows: the game (Ink's `graphics.shadow`) fits each round a
slice of the camera's view, bounding sphere and all, and moves its centre in
whole texels, so shadows hold still as the camera moves. The game has the
camera, the light, the shadow distance and the player's quality tier, so it
fits them; the renderer validates what it is given (orthographic, texels and
slice ends growing outward) and draws each into a layer of one array texture.
Where a loaded shader reads `shadow_cascades` (binding 3) every cascade is
drawn; otherwise only the farthest, at binding 1 as before. The scene uniform
carries the count, the blend band, the filter, the contact steps, each
cascade's view-projection, texels (texture and world), slice end and depth
range, and the soft shadows' settings after its first 320 bytes, which older
shaders need not declare. Each map has its own shadow pass, with casters
culled to it.

`shaders/shadows.wgsl` (`SHADOWS_WGSL`) is what materials read them with:
`sun_shadow` takes the cascade of the fragment's slice of the view, blending
into the next over the end of the slice, with a hardware tap, a rotated
Poisson PCF, or soft shadows (PCSS: a blocker search sets the penumbra, so
contact hardens). Maps are orthographic, so the receiver's own plane gives each
tap's depth and the bias follows its slope; a normal offset of 1.5 texels
at grazing angles covers the rest. The filter's pattern is fixed per pixel
(interleaved gradient noise), so TAA averages it. Shader and reference tests
live in `renderer/shadow_tests.rs`.

Contact shadows (`shadow.contact` steps) need the camera's depth in a form a
forward shader can read: the frame's casters are drawn once more, depth only,
through the camera's jittered projection, into a texture of the scene's size
(binding 9), before the scene pass; the shader marches a short ray toward the
sun along it. A frame without them draws nothing extra.

Local lights shadow through `local_shadows.rs`. Lights that ask
(`Light::shadow`) and whose reach touches the camera's view are ranked by size
on screen, then id, then index, and taken while `shadow.local` casters and 18
faces allow: a spot is one perspective face, a point light (or a spot wider
than 160 degrees) a cube of six, every face's window a few texels wider than
its edge so a filter never leaves it. Faces are layers of one depth array
(`shadow.local_texels` wide) bound with the lights at group 2; a light keeps
its layers while it stays picked, and a face is drawn again only when its
signature (the light's, and every casting draw's mesh, transform, parameters,
instance data and revision that the face's view sees) changed. The faces are
views after the sun's in the same culling, so their casters are culled to
each. `RenderStats::local_shadow_casters`, `_faces` and `_kept` say what was
picked and what was kept.

### Bloom

Bloom is glare, a small share of the picture's light scattered round where it
is, as a convolution of the whole exposed picture rather than a threshold added
on top (a threshold in exposed space made every pixel past it add all of its
excess, so under auto exposure a bright window flooded the room). The first
pass halves the picture after exposure, in linear light, each of its 13 taps
held to 32 times SDR white (`BLOOM_CEILING`) and each group of four weighted
`1 / (1 + brightest channel)` (Karis), so a lone 1e4 pixel weighs as one ordinary
one. Each next pass halves the last; then each level's 9-tap tent is added onto the one
above. The levels sum to as many blurred pictures as there are levels (up to
six), so the present pass mixes `BLOOM_INTENSITY / levels` (0.05) of the sum with
`1 - BLOOM_INTENSITY` of the picture, in linear light: the total light and a
uniform picture's mean do not change (the Karis weights hold back a little of a
very bright pixel, and the ceiling clips a source past 32). A source's glow
reaches about as far as the top level's tent (a few hundred pixels at 720p) and
falls off with it. `quality.bloom` turns it on, as before; the share and the
ceiling are constants. Witnesses (`tests/exposure_looks.rs`): the whole chain
against an f64 convolution for lamps +6 to +10 EV over mid-grey, with the share of the
frame they brighten under 2%; a uniform picture's mean; and a 1e4 pixel.

### Dynamic resolution

With a frame budget (`frame_budget`), `resolution.rs` scales the frame's
render scale down when the GPU takes longer than the budget (aiming at 90%
of it, as GPU time goes with the square of the scale), to half at most, and
back up a sixteenth at a time once frames take under 70% of it, changing at
most every 30 frames. The GPU's time is its timestamps' total, or, without
them, from each frame's submit (or the last frame finishing) to the queue
finishing its work.

`post.rs` draws the scene into a floating-point target (`SCENE_FORMAT`),
multisampled and resolved for MSAA, at the render scale. Then come ambient
occlusion from its depth, sky, fog (through the froxel volume, below) and cloud shadows, TAA, the metering of
auto exposure, and bloom, as a render graph (below). Last, the present pass
exposes the light, grades it, tones highlights into white (or toward an HDR
display's peak), applies FXAA Quality (`shaders/fxaa.wgsl`), splits the colours
of surfaces that ask for the frame's aberration, and scales the image to the
output.

### The sky as light

`environment.rs` holds what materials light with from the frame's sky. The game
works out nine spherical-harmonic coefficients of its sky's diffuse light and a
tint (`sky.sh0` to `sky.sh8`, `sky.tint`); the renderer writes them into a
uniform and never computes them. For glossy surfaces it keeps a 64-texel cube
map of the sky's gradient (the horizon to the zenith or the ground by the square
root of the height, the game's own sky function), rgba16float with six levels,
level `n` being the sky convolved with the GGX lobe of roughness `n / 5` by
importance sampling the gradient itself. The cube is drawn again on the GPU, 36
small passes at the start of the frame, only when the frame's zenith, horizon or
ground colors differ from the last frame's bits. The split-sum BRDF table (32 x
32, scale and bias by cosine and roughness) is drawn once. Both bind at group 0,
bindings 4 to 7 (`ENVIRONMENT_WGSL`, materials.md); shadow passes bind a zeroed
light and insets a flat one from their own ambient color, with no reflection.
`ambient_light(world, normal)` is the only place a material asks for diffuse sky
light. Where the frame has a probe volume (the game's baked grid of SH light
probes, `probe_volume` and the `probes` section) it interpolates the grid and
blends toward the sky's light past its edge (materials.md); the resident
probes live in one storage buffer (`MAX_PROBES` probes, bound at 8), uploaded
only when a frame sends them, and are used while the frame's revision and
counts match what the buffer holds. Screen-space occlusion still darkens the
whole picture after the scene, so sky light and reflections are occluded with
the rest.

### Reflections beyond the sky

`sky_reflection(world, normal, view, roughness, f0)` is the one entry point; it
takes the prefiltered sky cube, lays the local reflection probes over it where
the surface is inside their boxes, and lays screen-space reflections over both
by their confidence. `surface_ambient` (in `surface.wgsl` now) reaches it through
`sky_reflection_lobe`, which takes the lookup's normal and the cosine to reflect
separately (an anisotropic surface bends the lookup, a clear coat looks up its own
normal), so materials that call `surface_ambient` need no change.

**Reflection probes** (`environment/reflection_probes.rs`, `probe_finalise.wgsl`,
`probe_filter.wgsl`; the packet's `reflection_probes`). A probe is a position, an
influence box (centre, half size, blend distance), a parallax box, a priority,
a stable id, a revision and a capture flag. The host keeps `SLOTS` (8) cubes of
128 texels a face and six GGX levels in one cube array (binding 10, the table of
probes at 11). Which probes hold slots: priority first, then nearness of the
box to the camera (a held probe counts as 20% nearer: hysteresis), then id; the
same inputs always pick the same probes, and a dropped probe's slot goes to
the best newcomer. A cube is drawn when its probe has none, its revision
differs from the one it holds, or `capture` rises (not while it stays up), at
`reflection.budget` faces a frame, one probe at a time (cubeless first, then in
rank order), and not while assets are loading or a draw it would show is not
ready. Each face is the frame's own opaque draws seen from the probe by a 90
degree camera, through the same pipelines at the frame's sample count, lit by
the sun with the frame's shadow maps, by the lights that reach the probe
(`Clusters::update_one`: one cluster) and by the sky and the diffuse probes (so
captures bounce light from other probes' cubes). Culling has one more view for
it, a box round the probe (`culling.rs`, `MAX_VIEWS` 24), which every draw may be
in. A pass finishes the face (`probe_finalise.wgsl`: the draw's alpha, the share
of the sky, is filled with the sky cube times its tint; mirrored, because a
right-handed camera sees a cube's left and right swapped) into a scratch cube;
with the sixth face the scratch cube is halved to a texel and filtered into
the slot's levels (filtered importance sampling from the matching level), in
the same frame, so the slot keeps the old cube until the new one is whole.
Limits: the captures use the frame's sun maps, which are fit to the camera, so
a probe far from it sees the sun unshadowed outside them; no fog or atmosphere,
particles or translucent surfaces; no contact shadows. Shading
(`local_reflection`): up to four probes with weight, by priority groups, highest
first; within a priority the probes blend by their weights, covering
`1 - product(1 - w)`, which takes that share of what the higher priorities left;
the sky shows through the rest. A probe's weight is the smoothstep of the
distance to the nearest face of its box over its blend (1 for a blend of 0).
Lookups are box-projected onto the parallax box (the ray leaves the box
at `p + r t`, the lookup is that point minus the capture point); where
the box does not project (a half size of 0 or less), the direction is used as
it is. Brightness: a cube keeps, in its alpha, the mean light of the diffuse
probe lattice (or the sky's, without one) at its position when it was drawn
(`ambient_mean`, the constant SH term); the shader scales what it shows by
the mean now over the mean then, clamped to a factor of 4 either way, so a cube
captured under other light does not glow after the game re-bakes the lattice.
That scales the ambient share of the cube alone; light from the sun and the
lights stays as captured, so recapture (`revision`) when they change a lot.

**Screen-space reflections** (`quality.ssr` 1 low, 2 high; `renderer/reflections.rs`,
`ssr_*` in environment.wgsl). They need TAA and turn it on (the noise and the
history are TAA's), and a perspective camera. The depth is the camera's
casters drawn once more depth only (the contact-shadow pass, now two layers of
one array, this frame's and last frame's: bindings 9 and 12); the colour a ray
meets is last frame's TAA output at the point's reprojection (binding 13), so
a hit costs nothing to shade. In the material's own fragment shader, a ray (jittered through the
GGX lobe by a per-pixel, per-frame noise) is marched in the pixels it crosses,
linearly (16 or 32 steps at a stride that spans the visible part of the ray, with a start offset turned by noise) with
the ray's depth interpolated perspective-correctly, then halved 4 or 6 times
between the last step in front and the first behind; a hit must lie within
a thickness of the ray: 4% of the surface's distance from the eye, plus the depth
the ray itself crosses in a step, plus how much the surface's depth changed since the
last step (up to a quarter of its distance). That last term matters: a surface seen at a
grazing angle (the ceiling from the near floor, a side wall near its corner) changes its
depth by more than any fixed thickness between two steps of a coarse stride, and without
it the march missed it in stripes of rows, and left saw-toothed edges where such a
reflection ends. A hit must also come from in front of the surface (the last sample was
not already behind it), so a ray that has gone behind a thin occluder does not hit what
lies deeper. The jitter's far tenth is left out of the GGX lobe, so that a rare sample
does not meet another wall altogether and leave a speck that TAA cannot average away. The surface's normal is rebuilt from the depth around the hit, and a hit
whose normal faces away from the ray is dropped (a back face). Confidence is
the product of: the screen edge (the hit's distance to the border over 8% of the
picture), the ray's length against its reach (20 or 40 world units, fading over the last
40%), roughness (1 to 0.5, gone at 0.8) and history (last frame's depth at the
reprojected point must be the point's depth within 2 to 6%; else the hit is
dropped: a disocclusion or a mover). A mover's reflection lags a frame (there are no
motion vectors of the hit). The result mixes over the probe and sky reflection
by that confidence; TAA averages the rest. Casters only: a material that does not
cast shadows is not in the depth, so it is neither hit nor reflecting by march. The
history is the whole picture TAA made, translucent surfaces and particles included
(they come before it, in the translucent pass), but they are not in the depth: a ray
meets them as colour on what is behind them.
Without `quality.ssr` and probes the output is what it was before.

### Participating media

The frame's `fog` is a medium: a mist whose extinction `density` falls off
exponentially with height (`height`, constant below the ground at y = 0), clear
inside a cylinder of `radius` about the origin (rising over `edge` for a soft
rim), varied by a 3D noise (`noise`, `noise_scale`, carried by the game's own
`drift`, so the same drift draws the same mist), plus up to 32 local `fog_volumes`
(axis-aligned boxes and ellipsoids with their own density, albedo, emission and a
falloff toward the edge). It scatters by `albedo` through a Henyey-Greenstein
phase of asymmetry `anisotropy`. `quality.volumetrics` (0 off, 1 low, 2 high) and
`fog.distance` (the reach; 0 for none) decide how it is drawn; the formulas are
Ink's `volumetrics`.

**The froxel volume** (`volume.rs`, `shaders/volume_*.wgsl`). A grid aligned with
the camera: froxels across and down the picture, slices deep along the view (not
radial), 160 x 90 x 64 at high and 80 x 45 x 32 at low. Boundary `b` of `N` is at
depth `knee * (exp(b / N * ln(1 + reach / knee)) - 1)` with `knee = reach / 64`:
the first slice is thin and each next one a fixed share thicker, out to the reach.
The grid needs a perspective camera and a medium; otherwise nothing runs. Two
compute passes run after the shadow maps and before the scene:

- *Injection* (`volume_inject.wgsl`), one invocation a froxel, at a point of the
  froxel that moves each frame along a Halton (2, 3, 5) sequence of sixteen: the
  medium there (extinction, scattering by channel, emission) and the light it
  sends toward the eye per unit of length. The sun is its cascades' visibility
  (`sun_visibility` from `shadows.wgsl`, the frame's filter at most the tier's: a
  hardware tap at low, the 16-tap PCF at high, never the soft-shadow search) times
  the clouds' shadow and the phase; the clustered lights come from the froxel's
  cluster (`lights.wgsl`'s window and falloff without the cosine, times the phase
  and `local_shadow` toward the light); the ambient is the mean of the probe
  volume's light, or the sky's constant coefficient, blended by `fog.sky` toward
  the sky's colour in the direction looked. Lights are in the units of surfaces
  (irradiance over pi), so they scatter as `pi * HG`: `(1 - g^2) / (4 (1 + g^2 -
  2 g c)^1.5)`. A froxel with no medium costs one density evaluation.
  Then it blends with the froxel at the same place in the world last frame (its
  centre through last frame's unjittered camera, trilinear), 90% history. The
  history is dropped for the whole grid on the first frame, `exposure.cut`, a
  change of tier or reach, a camera that moved more than a tenth of the reach, a
  frame without a medium and `Renderer::reset_volumetrics()`; and for a froxel
  that reprojects outside the old frustum, or whose extinction differs from the
  history's by 25% to 75% of the larger (fading out the blend), so a medium that
  appears or vanishes does not ghost.
- *Integration* (`volume_integrate.wgsl`), a column a thread, front to back:
  each slice's source `L` against its extinction `s` over the segment `d` along the
  ray gives `L (1 - exp(-s d)) / s` of in-scatter, which conserves energy however
  thick the slice. A texture of `N + 1` layers holds, at layer `k`, the in-scatter
  and the optical depth up to boundary `k` (layer 0 is nothing).

**Applying it** (`atmosphere.wgsl`). The atmosphere pass samples the integral at
each surface's depth (trilinear, optical depth interpolated in slice space, so a
homogeneous medium is exact), and beyond the reach continues the same medium
analytically (`mist_beyond_arena`, the height integral outside the cylinder), in
the colour the grid ends on (its in-scatter over its opacity, faded to the
analytic colour where the grid holds little), so that the surface seam shows no
step: the transmittance and the in-scatter are both continuous in depth at the
grid's end. Without a grid (the off tier, no reach, an orthographic camera) the
whole path is analytic, with the same colours as an unshadowed, noiseless
medium: the sun times its phase plus the ambient (`fog.sun`, `fog.ambient`,
`fog.sky`); with `sun` 0 and `sky` 1 it is the sky's colour in the direction
looked, the way the mist was before there was a grid. The sky itself stays
unmisted. The pass works on the opaque surfaces' picture alone: translucent
surfaces, particles and overlays are drawn after it and mist themselves where
they are (next section).

Cost (an RTX 4080 SUPER, the hazy room of `tests/volumetrics.rs`, the injection and the integration
together, `GpuTimes::volume`): 0.03 ms at low, 0.18 ms at high with the 16-tap sun filter; the off tier
runs no pass at all.

Limits: a perspective camera only; the sun and the lights are those the
scene's own maps hold (a light's shadow only while it holds a face); no
multiple scattering; local volumes do not take the noise.

Tests (`tests/volumetrics.rs`, f64 references, headless): a homogeneous mist's
transmittance and ambient in-scatter at every froxel depth of both tiers; the
exponential height mist's optical depth along slanted rays against the closed
form; sun shafts behind a roof slit (the sun's share of in-scatter, 1 where lit
and 0 where shadowed); the phase's forward over back ratio for three values of
`g`; a point light's in-scatter against its falloff, its integral against a fine
f64 integration, and its shadow; local volumes only inside their bounds, with
emission; a still camera's convergence and the history's reset on cuts, jumps and
disocclusion; no seam at the grid's end against the analytic mist; and the off tier.

### Subsurface scattering on the screen

A material with `"subsurface": true` (render state, like `aberration` and `decals`; `Material::subsurface`)
stamps its opaque surfaces into the stencil with bit 4 (`subsurface::MARK`; the aberration's mark is bit 1
and the decals' bit 2, `stencil()` in `renderer/frame.rs` ORs them). Once the opaque surfaces and
their decals are drawn and resolved, a frame with any such draw runs two passes of
`shaders/subsurface.wgsl` (`subsurface.rs`, after Jimenez et al., "Separable Subsurface Scattering",
2015) over the resolved scene texture, before the opaque picture's passes: the first blurs it along x
into a texture of its own, the second along y back into the scene texture, so nothing after them (the
sky, the occlusion and the mist, the translucent pass, TAA, bloom) has to know. The sky, occlusion
and mist are applied to the diffused picture, so skin is diffused before it is misted.

- *Kernel.* Seven pairs of taps and a centre, spaced `2 * sign * (t / 2)^2` over `t` from -2 to 2
  (denser at the centre) out to the reach, each weighted by d'Eon and Luebke's sum of six Gaussians for
  skin (variances 0.0064 to 7.41 mm squared, summing to one) times the width of its cell, with a falloff per channel
  (1, 0.37, 0.3: red reaches farthest), normalised per channel, the centre keeping `1 - strength +
  strength * weight` and the others `strength * weight` for strengths (0.48, 0.41, 0.28), so a flat
  picture stays as it is. `subsurface::kernel()` builds it in f64 and the shader's constant table is
  generated from it; the tests build it again from the paper's recipe.
- *Reach.* The outermost tap is `subsurface::WIDTH` = 1 cm (world units: metres) from the pixel,
  which is `width / pixel size` pixels at the pixel's depth, the pixel's size in view space worked out
  from the inverse of the projection the scene drew with (TAA's jitter included) and capped at 64
  pixels. The width is one constant for every frame; a per-frame or per-material width would need a
  field in the render packet or the draw uniform, neither of which this has.
- *Taps.* A tap reads the picture bilinearly (four `textureLoad`s, so the reference is exact), and is
  valid only if its nearest pixel has every sample marked, so empty and unmarked pixels, a silhouette's
  partly covered pixels and anything off the picture give the pixel's own colour in the tap's place.
  A tap's surface more than two widths nearer or farther along the view counts partly (a linear ramp
  to nothing at four widths) with the pixel's own colour in the rest, so light does not cross an edge
  between a nose and a cheek, but a slope does not shut it out.
- *Marks.* A pixel is diffused only where all its samples carry the mark (the nearest depth over them
  is its surface's): any other is copied as it is, bit for bit, which is also what the second
  pass does to the first's copy. A silhouette pixel of a multisampled surface keeps its antialiased edge.
- *Cost and keys.* A frame with no such opaque draw makes no passes (`Post::passes()` does not list them)
  and no textures: the passes' graph (the one made texture, `diffused across`, and two uniforms) is made
  with the first frame that needs it and kept with the targets, which are keyed on the size, samples and
  quality only, so turning a skin on and off allocates nothing. `GpuTimes::subsurface` times the two
  passes: 0.34 ms for the two (an RTX 4080 SUPER, 1024 by 1024 at 4 samples, a head filling most
  of the picture, the frame 0.42 ms against 0.07 ms for the same head without the flag;
  `cargo test --release -p pen-render --test subsurface cost -- --ignored --nocapture`). Insets and
  reflection probe captures do not diffuse.
- *Limits.* The picture the passes blur has the specular light in it: it is diffused with the diffuse
  (a separate specular buffer is a follow-up); the width is one constant; one kernel (skin's) for every
  such material; and translucent surfaces, particles and overlays are drawn after, so a `subsurface`
  flag on one has no effect.

Tests (`tests/subsurface.rs`, f64 references, headless, half-float HDR output so that a pixel is the
scene's own half-float): the kernel built again from the paper's recipe against the shader's table
through the pictures of a line, a row, an edge and a point of light on a flat marked surface (every
pixel of the profile within two half-float steps, 1 and 4 samples, and an edge with TAA), the light of
the picture conserved, unmarked and half-marked pixels the same bits as without the flag and neither
giving nor taking light, the depth ramp at four gaps, no passes without the flag and flipping it
changing nothing else, a translucent pane over a diffused surface, and a head with and without the flag.

### Translucent surfaces and particles

Translucent surfaces (materials with an alpha below 1 or `blend`), particles and
overlays are drawn after the opaque picture has had its sky, ambient occlusion and mist,
so each is misted by the air between it and the eye and not by what lies behind it, and
each can read the opaque picture. A frame with none of them has no such pass and draws
the one graph it always has. The order (`post.rs`, `graph.rs`, `renderer/frame.rs`):

1. The `scene` pass draws the opaque surfaces (and, in the layered path, the `decals`
   pass draws the decals over them) into the multisampled picture and resolves it into the
   single-sample scene texture. The multisampled depth and stencil stay. With subsurface
   materials, the two diffusion passes (above) then blur the scene texture where they are marked.
2. The opaque graph (`Post::opaque`): `occlusion` if the quality has it, `atmosphere` (the
   same shader and numbers as before, on this picture alone: the sky, the occlusion, the
   froxel volume and its analytic continuation, the clouds' shadow; it runs even where the
   quality asks for none of them, to write level 0), `opaque depth` (the nearest depth
   buffer value over a pixel's samples into an `R32Float` texture, 1 where nothing is) and
   `opaque mip`, once for each level below level 0, which makes it from the one above
   with a (1, 3, 3, 1) / 8 tent, four bilinear taps three quarters of a texel out, down to a
   texel (eleven levels at 1280 by 720). The picture is `Rgba16Float` with the whole chain.
3. The `translucent` pass: its colour target is the multisampled texture again (resolving
   into the scene texture; at one sample the scene texture itself), cleared; the `backdrop`
   writes level 0 into every sample (no depth test; the depth and stencil attachment is
   loaded and kept, so what follows tests against the opaque surfaces); then the
   translucent surfaces back to front, the particles and the overlays, with the sorting,
   stencil marks and blend constants they always had.
4. The passes after: TAA, the metering hook, the bloom, present, reading what the
   translucent pass resolved to. The atmosphere is no longer among them.

An opaque edge pixel keeps its blend of surface and sky, which the backdrop writes to all
its samples; a translucent surface's own edges are antialiased by its samples. A frame
without the pass draws the graph it always did (occlusion, atmosphere into a made texture,
TAA, bloom, present), bit for bit. The targets are keyed on the size, samples and quality
only: the opaque picture and its depth are made with the first frame that needs them, and
kept, with the graphs that read them, until the targets change, so turning a translucent
surface on and off allocates nothing and drops no history (TAA reads and draws the same
history textures in both). `passes()` lists, for a frame with the pass, the opaque graph's
passes, `backdrop` and the passes after (the two `subsurface` passes first, for a frame that
has them); the opaque graph is timed as `GpuTimes::opaque` and the translucent pass counts
with the scene.

**What a shader reads** (`TRANSLUCENCY_WGSL`, `shaders/translucency.wgsl`, after
`ENVIRONMENT_WGSL`, which declares the `sky_sampler` it samples with; materials.md lists
it). Bindings 14 to 17 of the scene group, fragment stage only: the opaque picture and its
chain (14), the nearest depth (15), the integrated froxel volume (16) and the fog block
(17: the volume's uniform, with the eye, the sky's colours, the sun, the clouds and the
scene's jittered view-projection after it). They are stand-ins in every pass that has no
such picture: the opaque draws themselves (so `fog_apply` is the identity there and one
shader serves both, the atmosphere pass misting what it draws), the shadow passes, a
reflection probe's faces, insets and frames without a translucent pass. The scene group
has 15 sampled textures in the fragment stage with the draw's three and the lights' one
(it had 12; WebGPU allows 16 by default): `renderer.rs`'s `scene_layout_entries`,
`draw_layout_entries` and `lights::layout_entries` are the layouts' entries, a unit test counts
them, and a device limited to 15 draws a frame of glass (`tests/materials_integration.rs`).

**Compositing.** Each draw's destination already holds the mist to its own depth: the
backdrop is the opaque surface misted to where it is, and each earlier translucent surface
was drawn misted to where it is. A translucent surface at depth z with colour c and alpha a
sees the air in front of it with transmittance T and in-scatter L, over what is behind:
the eye gets `T (a c + (1 - a) B) + L`, where B is the light that reached z from behind,
and the destination is `D = T B + L` since the ray's mist includes the stretch in front of
z, so `T (1 - a) B = (1 - a) (D - L)` and the eye gets `a (T c + L) + (1 - a) D`. That is
the ordinary blend of the colour `fog_apply(c, world) = c T + L`, which is what
translucent surfaces output. For the other blends: an occluded overlay blends by its
constant as alpha, the same; a visible overlay replaces what is behind and so is
`c T + L`; a covering particle (premultiplied, source factor 1, destination 1 - a) outputs
`p T + a L` for `p = a c`; added light (destination 1) outputs `p T` alone, since it hides
nothing and adds no mist of its own. A particle takes `fog_terms` at its middle (all its
square is at one depth), once a vertex.

`fog_terms(world)` (L, and T with the clouds' shadow on a surface that the atmosphere pass
also applies) takes the froxel integral at the point's depth with the optical depth
interpolated in slice space (a homogeneous medium is exact), and continues the same
medium analytically beyond the grid's reach, in the colour the grid ends on, or the
whole way without a grid, with the numbers of `atmosphere.wgsl`: a surface at the wall's
pixel gets the wall's own mist (a test compares the two). It needs a perspective camera for
a grid, like the atmosphere pass. Without a medium it is (0, 0, 0, 1).

Cost (an RTX 4080 SUPER, release, 1280 by 720, 4 samples, a room with sky, mist at the
high tier, occlusion, bloom and FXAA, passes timed): a frame with nothing translucent 0.45
ms (0.49 with TAA), as before; with one translucent pane 0.58 ms (0.62 with TAA) where the
old pipeline took 0.46 (0.51): the opaque picture's passes (atmosphere, occlusion, depth
and the ten levels) 0.23 ms and the translucent pass 0.09, against the 0.27 ms the passes
after the scene took; with a plume of 2000 particles on top 0.66 (0.54 before). About
0.12 to 0.13 ms more for a frame that has them, and nothing for one that has not.

Limits: the opaque picture holds the opaque surfaces only, so a translucent surface seen
through another does not refract it (`opaque_behind` reads what is behind both); the
scene texture that sprite shaders read (binding 4) is, in a frame with the pass, the
picture with its translucent surfaces and its mist and sky, and in one without, as it was
drawn (before the sky and the mist); TAA reprojects a translucent surface by the opaque
depth behind it, as it always did; the chain's levels are a tent, not a Gaussian, and
blocky past a few levels.

Tests (`tests/translucency.rs`, f64 references, headless): a translucent quad in front of a
wall in a homogeneous and an exponential-height mist, at both tiers and without a grid,
against the closed forms and the froxel volume's own numbers, far clearer than the old
number; the plain blend without a mist; a particle covering and adding; the opaque
picture's chain against an f64 filter to within a step of the 8 bit output (0.8 of 255 at worst) at every level, and its nearest
depth (the nearest sample's at a multisampled edge); the library's identity where there is
no picture; the world point's place in the picture equal to the pixel's under TAA's
jitter; frames with a translucent surface that shows nothing bit for bit the frames
without it (both tiers, 1 and 4 samples, TAA), and turning one on and off keeping
the history; the passes' order; edges at 4 samples; screen-space reflections; auto exposure
metering the picture with the pane in it; scene filters seeing it.

### Exposure and grading

The frame's `exposure` and `grade` (header groups, defaults that change
nothing) decide how the scene's light is shown. Both work on the light the
present pass reads, which `display.rs` describes as sRGB-encoded (the scene
draws as for an SDR screen): the shaders decode it to linear light
(`shaders/transfer.wgsl`), work there, and encode it again before `toned`, so a
stop is a doubling of light and the roll-off toward white or an HDR display's
peak is unchanged. With a fixed exposure of 0 stops and the neutral grade the
present pass shows exactly what it showed before there was either.

**Exposure** (`exposure.rs`). `exposure.auto` off, `compensation` is a fixed
exposure in stops (-16 to 16); on, the scene is metered on the GPU and the
exposure adapts. Metering reads the lit scene, after the sky, fog, translucent
surfaces and TAA and before the bloom: the bloom's first pass exposes the picture itself before it
blurs it (so what glows is the picture as the screen shows it, whatever the exposure),
which makes metering after it circular, and a glow is the scene's light
spilling, not more of it. Two compute passes run each metered frame, from a
hook in the render graph (`Graph::hook`: it gives the caller the view of the
texture it reads, at its place in the order):

- `shaders/meter.wgsl` counts each pixel into one of 256 bins evenly spaced in
  log2 of its linear luminance over `range_low` to `range_high` (default -10 to
  6 stops, a sixteenth of a stop a bin); darker and brighter pixels fall in the
  first and last bins and NaN counts for nothing. A pixel counts for its weight
  in 1/256ths (all 256, or with `centre` above 0 mixed toward a Gaussian of the
  distance from the middle), so a 4K picture's counts fit a `u32`. Each
  workgroup counts in shared memory first.
- `shaders/adapt.wgsl` (one invocation) takes the average of the bins' log2
  luminance over the weight between the `low` and `high` percentiles (a bin that
  straddles one counts for the share inside), so the darkest and brightest
  pixels are ignored, then the exposure that brings that average to `target`
  (middle grey, 0.18) plus `compensation`, clamped to `min` and `max`. The
  exposure moves toward it by `1 - exp(-rate * dt)` of the distance, `rate`
  being `brighten` going up and `darken` going down (per second), `dt` the game's
  seconds since the last frame, so any split of time into frames lands in the
  same place and `dt` 0 (a paused game) holds it. The first frame, `exposure.cut`
  and `Renderer::reset_exposure()` (for a host that restores or replays) jump to
  the goal instead.

The exposure lives in a four-float storage buffer (stops, the metered average,
whether it holds a value, the goal) that the bloom and the present pass read:
nothing is read back for rendering. A fixed frame writes the buffer from the
CPU (so switching to auto starts from there). `RenderStats::exposure` is the
exposure of a recent frame, copied out and mapped a few frames later like the
GPU timings, for dev overlays and tests. The compute pipelines compile on the
first metered frame.

**Grade** (`grade.rs`, `shaders/grade.wgsl`). In the present pass, after
exposure and the bloom, in linear light, in this order: a vignette (`vignette`,
`vignette_radius`: light times a smoothstep from 1 at the radius to 1 - strength
at the corner), `exposure` in stops and white balance (`temperature` moves red
and blue half a stop against each other, `tint` green against both), contrast
about `pivot` in a log encoding (ACEScc, the S-2014-003 curve on the frame's
own primaries), ASC CDL (`gain` slope, `lift` offset, power 1 / `gamma`) in
linear light, saturation and vibrance about the luminance, and the look-up
tables. Grading before the roll-off, in light and in log, is what makes it right
for both displays: the grade shapes the scene, then `toned` fits it to the SDR
screen's white or the HDR display's peak, rather than a table authored for SDR
white being stretched over highlights it never saw. The same formula is Ink's
`color.grade` and the f64 reference in this file's tests, term for term.
Nothing set is the identity (the shader is skipped); evaluated in full with
neutral settings it agrees with its input to a few parts in 100,000.

A table is a strip texture named by a material (`grade.lut_a`, `lut_b`: 1 plus
the catalog index; 0 for none): N slices of N by N texels (N 2 to 64), side by
side, so N * N wide and N tall, blue picking the slice, red running along it and
green down, texel (r, g, b) of an identity table holding (r, g, b) / (N - 1). It
holds colors in the grade's log encoding (0 is a thousandth of SDR white, 0.41
middle grey, 1 more than two hundred times white), read as data (not an sRGB
picture) at its finest level and sampled trilinearly with texel centres at
i / (N - 1), eight loads, by the shader (`lut_lookup`). It is 8-bit in an
ordinary PNG, so its steps are 1/255 of the encoding, about a twentieth of a
stop; the part of the encoding outside 0 to 1 passes through unchanged, so black
and the brightest highlights are not clipped. `lut_blend` blends `lut_a` toward
`lut_b` in log (a zone transition), `lut_weight` is how much of the result
applies, and a table that has not streamed in yet is left out (the second stands
in for a missing first). The two tables bind at group 1 of the present pass,
rebuilt only when they change; the exposure and the grade's uniform are at
bindings 5 and 6 of group 0. Scene filters (sprite materials with
`scene_filter`) see the picture after exposure, grade and roll-off, as before.

### The render graph

The passes after the scene draw as a small render graph (`graph.rs`). Each
pass is a full-screen triangle of one of the renderer's pipelines; it names
the textures it reads, in its bind group, and the one it draws into or the
output, clearing it or blending over it. Planning checks that every texture is
drawn before it is read, leaves out passes whose pictures nothing reads, gives
each texture the graph makes a lifetime, and lets textures of one size and
format whose lifetimes do not overlap share an allocation. `Post` builds the
graph again only when the targets' size or the passes change (the
`TargetsKey`): occlusion, atmosphere, TAA, each bloom level down and up,
present. A frame with subsurface surfaces has a small graph of its own before these, made
the first time one comes and kept with the targets: its two passes, along x into a made
texture and along y back into the imported scene texture. The scene's own targets (depth, picture, stencil marks) and TAA's
history are imported. A frame with translucent surfaces, particles or overlays
splits it at the translucent pass (above): the opaque graph (occlusion,
atmosphere into level 0 of the opaque picture, `opaque depth`, `opaque mip` for
each level below) draws into imported views of the opaque textures, and
the graph after it (TAA, bloom, present) reads the scene the translucent pass
resolved to; both are made the first time such a frame comes, and the graph
for frames without them is the one made with the targets. `Graph::execute` encodes the passes each frame, and
`Renderer::passes()` lists them. A hook is a pass that draws nothing: it reads one
texture and `execute` hands its view to the caller at that place (the metering
of auto exposure); `passes()` leaves hooks out. A pass may bind a read-only
storage buffer, and the present pass has a late bind group at group 1.

### Temporal antialiasing

`quality.taa` turns it on (`taa.rs`, `shaders/taa.wgsl`). The scene's
projection is jittered within its pixels along the Halton (2, 3) sequence,
eight frames long; the scene uniform, ambient occlusion, the atmosphere and the
translucent pass's reads of the opaque picture use the jittered matrix. A pass after
the atmosphere (and the translucent pass, where the frame has one) reprojects each pixel:
the nearest depth of its 3 x 3 neighbourhood puts its surface in the world,
and the last frame's unjittered camera says where it was (shifted back by the
jitter its sample sat at, so a still picture reads its history in place and is
never resampled). The history there,
read through a five-tap Catmull-Rom filter and clipped toward the
neighbourhood's colour box (variance clipping in YCoCg), takes 90% of the
blend; light is squeezed below 1 for the blend, so highlights do not
flicker. The history is two textures read and drawn in turn, with a graph for
each that shares every other texture. Bloom and the present pass read the
result; FXAA still applies on top if asked. Without motion vectors, things
moving on their own keep only what the clipping lets through; decals still
reconstruct with the unjittered camera (half a pixel at most).

### Point and spot lights

The frame's `lights` (position, reach, colour, a spot's direction and cone
cosines) are sorted on the CPU into 16 x 9 tiles of the picture and 24
slices of depth, spaced by the logarithm of depth (evenly for an orthographic
camera; near and far come from the projection). Each light's reach is a
sphere (a spot's cone in the smaller sphere round it), and it goes into every
cluster its view-space box touches: a box of clusters per light, counted then
filled, at most 256 lights a cluster (`lights.rs`). Group 2 of every mesh
pipeline binds the grid uniform, the lights, each cluster's range and the
light indices; `LIGHTS_WGSL` declares them and `clustered_light(world,
normal, position, view_z)` sums the lights of a fragment's cluster as a
diffuse surface's irradiance, each falling smoothly to nothing at its reach.
Lights that cast shadows read their maps at binding 4 of the same group, with
their comparison sampler and each layer's matrix (see Shadows). Material shaders that do not light with them leave group 2 alone. Insets and
the shadow passes bind a group of no lights and a stand-in atlas. A frame without lights writes one
48-byte uniform.

### The standard surface

`surface_lobes.wgsl`, `surface.wgsl` and `surface_transmission.wgsl` (`surface.rs`;
materials.md for what a shader calls) are one lighting model for material
shaders: a `StandardSurface` of position, normal, view, base color, metallic,
roughness, occlusion, the fragment's position and view depth (for the light
clusters), the face-on reflectance and optional lobes (clear coat, sheen,
anisotropic GGX, transmission with thin and volume refraction, a cheap subsurface),
lit by the sky (`surface_sky`: diffuse SH or probe light, the sky, probe and
screen reflections through `sky_reflection_lobe`), the frame's clustered lights
(`clustered_lobes`, one `surface_light` per light of the cluster) and the sun
(`standard_sun`, with the visibility of `sun_shadow`). Three files so that the
parts that can run alone do: the lobes have no bindings (`surface_light`, the
sun's helper and `parallax_uv` included: tests run them in compute shaders), the
glue needs `ENVIRONMENT_WGSL` and `LIGHTS_WGSL`, and the transmission glue needs the
translucency library's four `opaque_*` functions (it reads the frame's picture of
what is behind a surface; where there is none it shows the sky along the ray).
`surface_ambient` and `clustered_surface` are the base lobe through the same code:
`surface_sky` takes one expression for the base lobe alone (so a compiler fuses it
as it always did, which is what makes the results the old functions' to the bit),
and the lobes' branches are behind flags a shader names as constants. The layering
is `(diffuse * (1 - transmission) + specular) * sheen_keep * coat_keep + sheen +
coat`. Sheen's directional albedo is a 32 by 16 table (Ink's, integrated by
`tools/native/src/surface.rs`; pen-render's tests integrate it again and compare it
with the table in the shader's text). The lit.wgsl of `pen new` embeds the five
libraries between BEGIN and END lines that a test keeps equal to the sources.

### HDR output

`choose_surface` picks, when HDR is asked for and the surface offers it, in
this order: `Rgba16Float` in extended sRGB (`SurfaceColorSpace::ExtendedSrgb`:
Vulkan, Metal and the browser), `Rgba16Float` in scRGB (`ExtendedSrgbLinear`,
as Windows offers), or `Rgb10a2Unorm` in BT.2100 PQ; else the SDR format as
before. A display known to be SDR stays SDR. The display's headroom
(`DisplayHdrInfo::tone_map_headroom`, or 1000 nits over 203 when unknown) is
how far past SDR white highlights may go. Frames draw as for SDR, in
sRGB-encoded light, but the present pass's `toned` rolls highlights off
toward the headroom (encoded) instead of toward white; below its shoulder
nothing changes, so the game looks as it does in SDR. For extended sRGB that
is all. For scRGB and PQ the frame, sprites, insets and veil draw into a
picture of their own (`display.rs`'s `Screen`, shared with renderers drawing
over this one's frames), and a last pass decodes it: to linear light over
80 nits (scRGB), or into BT.2020 primaries and the PQ curve (HDR10). pen
asks for HDR with `PEN_HDR=1` when it plays (not while recording or taking
shots, which read 8 bits back); the browser host stays SDR.

### Compressed textures (KTX2)

See [texture compression](../../docs/texture-compression.md). A texture key
ending in `.ktx2` reads the first of `name.bc7.ktx2`, `name.astc.ktx2`,
`name.etc2.ktx2` the device samples, then `name.rgba.ktx2`, then `name.ktx2`
(`compressed.rs`); its levels stream in as a packed texture's do, uploaded
in whole blocks.

### Text from distance fields (MSDF)

`pen font` makes a multi-channel signed distance field atlas of a font
(`pen-texture`'s `msdf`, after msdfgen): RGB the field whose median is a
glyph's edge, sharp at corners, and A the plain distance. It is an RGBA8
UNORM KTX2 file with no family variants, so it is read as data. A sprite
shader adds `MSDF_WGSL` and calls `msdf_coverage(msdf_median(field.rgb), uv,
range, size)`, which smooths the edge over one screen pixel whatever the
magnification.

### Extended live rendering passes (`wired_passes/*`)

Beyond the core protocol packet, `Renderer::draw` runs five additional GPU
rendering systems driven either by direct `Renderer` setters or optional
`RenderFrame::extensions` tags (preserving ABI 8 with zero wire changes):

- **Continuous Meshlet DAG LODs (`meshlet_dag.rs`, `wired_passes/meshlets.rs`)**:
  static LOD0 meshes with `>= 768` non-degenerate triangles build an
  endpoint-preserving multi-level `MeshletDagGpu` alongside single-level
  meshlets. `MeshletDagCuller` selects a seamless screen-space error cut on the
  GPU (default `1.5 px`, configurable via `Renderer::set_dag_lod_error_threshold`;
  values `< 0` fall back to two-phase Hi-Z meshlet culling) and draws via
  `multi_draw_indexed_indirect_count` (or indirect draw loop fallback).
- **Stateful GPU Mesh Particles (`gpu_particles.rs`, `wired_passes/gpu_particles.rs`)**:
  3-pass race-free compute simulation (`simulate_step` -> `apply_spawns` ->
  `compact_alive`) with 3D curl turbulence, swept depth-buffer bounce collisions
  against resolved `R32Float` opaque depth, GPU dead-slot reclamation, and
  `draw_indexed_indirect` 3D mesh instancing inside the HDR scene pass. Driven
  by `Renderer::set_gpu_particle_scene` / `seed_gpu_particles` or
  `GPU_PARTICLES_EXT_TAG` (`65_005`; `65_004` is reserved by `ACOUSTICS_EXT_TAG`).
- **3D/2D Antialiased Lines & X-Ray Debug (`lines.rs`, `wired_passes/lines.rs`)**:
  quad-extruded capsule-SDF polylines with homogeneous near/far clipping,
  analytic box-filtered dashing, Forward-Z & Reversed-Z depth testing, dimmed
  occluded X-ray pass, always-on-top 3D layer, 1px hardware `LineList` pass,
  and a color-only 2D viewport overlay layer. Driven by
  `Renderer::set_line_scene` or `LINES_EXT_TAG` (`65_006`).
- **Per-Object JFA Silhouette Outlines (`jfa_outline.rs`, `wired_passes/jfa_outline.rs`)**:
  256-style `R32Uint` entity mask pass (supporting both single-sampled depth
  and MSAA-resolved `R32Float` depth occlusion tagging in Forward-Z &
  Reversed-Z), `O(log2 R)` Jump-Flood + JFA+1 Euclidean Voronoi compute passes
  in portable storage buffers, and crisp antialiased alpha compositing onto the
  HDR scene before tonemapping. Driven by `Renderer::set_jfa_outlines` or
  `JFA_OUTLINE_EXT_TAG` (`65_007`).
- **First-Class 2D Renderer Suite (`render2d/`, `wired_passes/render2d.rs`)**:
  world-space rotated/Y-sorted/9-sliced sprites (`Sprites2dPass`), single-draw
  chunked GPU tilemaps with flip/rot90/frame animation (`TilemapPass`),
  resolution-independent analytic vector SDF UI shapes with per-corner radii,
  linear/radial gradients, crisp borders, Abramowitz & Stegun `erfc` drop
  shadows, and inline rounded clip rects (`SdfShapePass`), and multi-cascade 2D
  Radiance Cascades global illumination with bilinear-fix ray marching and zero
  wall light leak (`RadianceCascades2d`). Driven by `Renderer::set_scene_2d`,
  `RENDER2D_EXT_TAG` (`65_008`, vector SDF UI), or `RENDER2D_WORLD_EXT_TAG`
  (`65_009`, 2D camera, atlas-keyed tilemaps, sprites, and 2D Radiance Cascades GI).

WGSL lives in `src/shaders/`. A `${name}` in a template is filled in with
`str::replace`, for example the sample count or the shared `toned` function.
Material shaders are assets, read from the asset root.

## Logging

Warnings (a texture stream that broke, a pipeline cache that could not be
written) and debug timings (each pipeline compile) go to the `log` crate. The
host installs the logger. The `render_packet` example prints them with
`RUST_LOG` set.

## Testing

- `cargo test --release -p pen-render -- --test-threads=1` runs unit tests
  and GPU tests on a headless adapter. They cover loading and failures, hot
  reload, shared renderers, shadow and variant compiles, culling on the CPU
  and the GPU, levels of detail, and FXAA witnesses. With `--features
  simplify`, also making levels of detail. `tests/look.rs` checks TAA against
  an analytic edge, clustered lights against every light summed, KTX2
  variants by device features, HDR output in each color space, and MSDF text
  magnified, and writes what it drew to `target/look-shots/`.
  `tests/exposure_looks.rs` draws a dark room with a bright window with a fixed
  and with auto exposure, the adaptation over time (rates, frame-rate
  independence, cuts and resets), and the grade and its look-up tables on SDR
  and HDR displays (PNGs in the directory `EXPOSURE_SHOTS` names if set); the
  histogram, trimmed average, adaptation, clamps and cuts are checked against f64
  references by `exposure.rs`'s tests, and the grade and table sampling by
  `grade.rs`'s.
  `tests/volumetrics.rs` holds the froxel volume of the mist to f64 references and
  draws a hazy room's sun shafts and a point light in mist (PNGs in the directory
  `VOLUME_SHOTS` names if set); see Participating media.
  `surface.rs`'s tests (in `src/surface/`) run the standard surface's lobes in compute
  shaders against independent f64 references (every lobe, white furnaces for the coat and
  the sheen, the sheen table's integral, anisotropy at 0 against the isotropic lobe,
  parallax against analytic shifts) and compare the base lobe with the functions it
  replaced, frozen in `src/surface/baseline.wgsl`, to the bit, on a grid of inputs and on the
  real sky and lights. `tests/standard_surface.rs` draws a row of materials, the base lobe as the old
  functions drew it, an anisotropic highlight's stretch against the reference and bricks with
  and without parallax (PNGs in /tmp/fidelity/materials).
  `tests/materials_integration.rs` puts the merged material stack through the renderer from
  a material's JSON (`surface.transmission`, `ior`, `thickness`) with f64 references: the stripes of a
  wall seen through tilted slabs of glass at three indices against Snell's law and the
  library's sampling rule (to 0.15 px; an index of 1 and a thin wall move nothing; a rougher glass
  reads a blurrier level), a glass slab in the analytic mist against the closed form's composition
  (refracted misted picture times what the glass lets through, misted again at the glass's own
  depth, to 0.2 of a level of 255), a glass ball before a wall and pillars in mist
  (/tmp/fidelity/materials/glass-in-fog-e.png), a transmission material drawn among the
  translucent surfaces, casting no shadow and not instanced, and the layouts on a device of 15
  sampled textures.
  `tests/subsurface.rs` holds the screen-space subsurface diffusion to an f64
  reference (see Subsurface scattering on the screen) and writes the head with and
  without it to /tmp/fidelity/materials/subsurface-e.png.
  `tests/shadow_looks.rs` draws the sun's soft and contact shadows and the
  spot and point lights' shadows (before and after PNGs, in the directory
  `SHADOW_SHOTS` names if set); the shaders' filters, cascade blending, bias,
  contact march and local shadow projection are checked against f64 references
  by `renderer/shadow_tests.rs`.
  Test in release: a Rust 1.99 release-mode miscompile once hid in the
  sampler setup.
- `cargo run --release -p pen-render --example render_packet` renders
  packets from `dist/packets` (made by `just packets`) with the assets in
  `dist/gdev/assets`, writes PNGs, and reports how long loading takes to
  settle. `--sequence` replays a run of packets in order and times each
  frame. See its header for the options.
