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
- `LIGHTS_WGSL` and `MSDF_WGSL` are WGSL that material and sprite shaders add
  to light with the frame's point and spot lights and to draw text from a
  multi-channel distance field (below). `SURFACE_LOBES_WGSL`, `SURFACE_WGSL` and
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
joints and weights); its joints and weights become a second vertex buffer.

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
  their own `extra` data draw alone. Brush solids are meshed from their
  planes (`solids.rs`); only the solids that changed are meshed again. Each
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
- `frame.rs`: `submit` puts the frame together and encodes its passes. The
  culling compute pass comes first, then the shadow passes, then the
  particles' compute pass. The scene draws opaque surfaces, then decals over
  them if there are any, then translucent surfaces, particles and overlays. Then come the passes in `post.rs`, then sprites
  (`sprites.rs`), insets and the veil.

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
cast shadows is not in the depth, so it is neither hit nor reflecting by march.
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
unmisted. Translucent surfaces and particles are still misted with everything
else by what lies behind them (the pass sees one picture, and a translucent
surface writes no depth): see the limits below.

Cost (an RTX 4080 SUPER, the hazy room of `tests/volumetrics.rs`, the injection and the integration
together, `GpuTimes::volume`): 0.03 ms at low, 0.18 ms at high with the 16-tap sun filter; the off tier
runs no pass at all.

Limits: a perspective camera only; the sun and the lights are those the
scene's own maps hold (a light's shadow only while it holds a face); no
multiple scattering; local volumes do not take the noise; translucents are
misted at the depth behind them, so a translucent surface much nearer than what
is behind it looks hazier than its own distance says (a pass that misted the opaque
colour before the translucent surfaces draw would need its own multisampled
resolve).

Tests (`tests/volumetrics.rs`, f64 references, headless): a homogeneous mist's
transmittance and ambient in-scatter at every froxel depth of both tiers; the
exponential height mist's optical depth along slanted rays against the closed
form; sun shafts behind a roof slit (the sun's share of in-scatter, 1 where lit
and 0 where shadowed); the phase's forward over back ratio for three values of
`g`; a point light's in-scatter against its falloff, its integral against a fine
f64 integration, and its shadow; local volumes only inside their bounds, with
emission; a still camera's convergence and the history's reset on cuts, jumps and
disocclusion; no seam at the grid's end against the analytic mist; and the off tier.

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
exposure adapts. Metering reads the lit scene, after the sky, fog and TAA and
before the bloom: the bloom's first pass exposes the picture itself before it
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
present. The scene's own targets (depth, picture, stencil marks) and TAA's
history are imported. `Graph::execute` encodes the passes each frame, and
`Renderer::passes()` lists them. A hook is a pass that draws nothing: it reads one
texture and `execute` hands its view to the caller at that place (the metering
of auto exposure); `passes()` leaves hooks out. A pass may bind a read-only
storage buffer, and the present pass has a late bind group at group 1.

### Temporal antialiasing

`quality.taa` turns it on (`taa.rs`, `shaders/taa.wgsl`). The scene's
projection is jittered within its pixels along the Halton (2, 3) sequence,
eight frames long; the scene uniform, ambient occlusion and the atmosphere
use the jittered matrix. A pass after the atmosphere reprojects each pixel:
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
