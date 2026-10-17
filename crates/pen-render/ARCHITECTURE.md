# How pen-render works

pen-render draws a game's render packets (`pen_protocol::RenderFrame`) with
wgpu. It runs natively (Vulkan, DX12, Metal) and in the browser (WebGPU, or
WebGL2 where there is none) from the same code. The host owns the window, the device and the frame loop. It
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
  multi-channel distance field (below).
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
camera's clip volume, and each shadow map's. A draw left out of a view would
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

The game fits its shadow map (`frame.shadow`, binding 1 and the scene
uniform's `light_view_projection`). Where a loaded shader reads
`shadow_cascades` (binding 3, a `texture_depth_2d_array`), the renderer also
fits a nearer map of the same size inside the camera's view (`cascades.rs`;
the code fits as many as `NEARER` says, and a map costs about as much as the
game's): a sphere round the stretch of view nearer than the split (a
practical split between the near plane and how far the game's map reaches), along the game's light and with its depth range, rounded up to a
sixteenth and moved in whole texels, so their shadows hold still. The maps
are layers of one array texture, nearest first and the game's last; the
scene uniform carries their count, view-projections and texels after the
first 320 bytes, which older shaders need not declare. Each map has its own
shadow pass, with casters culled to it.

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
occlusion from its depth, sky, fog and cloud shadows, TAA, and bloom, as a
render graph (below). Last, the present pass tones highlights into white (or
toward an HDR display's peak), applies FXAA Quality (`shaders/fxaa.wgsl`),
splits the colours of surfaces that ask for the frame's aberration, and
scales the image to the output.

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
`Renderer::passes()` lists them.

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
Material shaders that do not light with them leave group 2 alone. Insets and
the shadow pass bind a group of no lights. A frame without lights writes one
48-byte uniform.

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

## WebGL2

A device without storage buffers is taken to be WebGL2-class (wgpu's GL
backend in a browser without WebGPU), and `downlevel.rs` draws on it:

- Shaders are rewritten in naga's IR and handed to wgpu as such. Each
  read-only storage array (`var<storage, read> X: array<T>`: sprites, decals)
  becomes a uniform array of as many elements as one uniform binding holds,
  at most 64 KiB. Sprites and decals are drawn a window of that many at a
  time, bound at offsets the device's alignment allows, the instances
  counted from the window's start. A depth texture a shader only reads with
  `textureLoad`, which GLSL ES 3.0 cannot, becomes a float texture over the
  same depth view.
- Multisampled textures cannot be sampled: the scene draws with one sample
  a pixel. Insets keep 4, as nothing samples theirs before they resolve.
- A depth-stencil attachment cannot be read while attached, so decals read
  a copy of the depth (R32Float); and a stencil cannot be sampled, so a
  stencil-tested pass copies the aberration's marks into an R8Uint texture
  the present pass reads.
- A texture cannot be viewed in another format, so an sRGB texture's linear
  view reads it decoded (a PNG normal map would be off; packed normal maps
  are linear already).

Natively, a device asked for `wgpu::Limits::downlevel_webgl2_defaults()`
takes the same path, which `tests/downlevel.rs` compares with the storage
buffer path on the same packet, and on wgpu's GL backend where there is one.

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
  Test in release: a Rust 1.99 release-mode miscompile once hid in the
  sampler setup.
- `cargo run --release -p pen-render --example render_packet` renders
  packets from `dist/packets` (made by `just packets`) with the assets in
  `dist/gdev/assets`, writes PNGs, and reports how long loading takes to
  settle. `--sequence` replays a run of packets in order and times each
  frame. See its header for the options.
