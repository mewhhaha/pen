# How pen-render works

pen-render draws a game's render packets (`pen_protocol::RenderFrame`) with
wgpu. It runs natively (Vulkan, DX12, Metal) and in the browser (WebGPU) from
the same code. The host owns the window, the device and the frame loop. It
calls `Renderer::submit` once per frame, and calls `Renderer::overlay` to draw
another program's sprites, such as pen's dev tools, over a finished frame.
[Measurements](../../docs/render-performance.md) cover load times, frame
costs and FXAA.

## Public API

- `Renderer::new(device, queue, RendererOptions)`: `format` is the
  swapchain's non-sRGB format (the present pass encodes colours itself).
  `assets` is the asset root, holding `catalog.json` and every asset key.
  `veil` is how long the loading veil takes to clear, or None for no veil.
  `pipeline_cache` is a directory to keep compiled pipelines in between runs.
- `optional_features(&adapter)`: device features the renderer uses when the
  adapter has them (today `PIPELINE_CACHE`, which only Vulkan offers). The host
  requests them with the device.
- `preload(catalog)` prefetches every mesh and material, with their textures
  and shaders, behind whatever frames ask for.
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
  renderer also reads (below).

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
`indices`) or the binary form `pack_mesh` makes. The binary form starts with
`PMSH`, version 1, flags (shader attributes present, 16-bit indices), the
vertex and index counts, then little-endian vertex floats and indices. A
packaged `X.mesh.json` may hold either; the renderer goes by the magic bytes.
Both are checked the same way: matching arrays, indices inside the vertex
array, nonzero normals, and finite float32 coordinates.

## Pipelines

A material shader (a WGSL asset) has up to five scene pipelines for each
sample count. They are opaque, opaque and instanced, translucent, and two for
overlays: one dimmed where the scene hides it, one at full strength. It also
has up to two depth-only shadow pipelines (plain and instanced). Only the
variants its loaded materials draw with are compiled (`Material::variants`):

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
in `post.rs`, decals, and the inset composite. They compile on the loader
threads too: the sample-independent ones when the renderer is made, the rest
when a sample count is first seen. The loading veil covers the frame while
they compile. A renderer without a veil waits natively for the compiles
already under way; in the browser it compiles what it lacks then and there.

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
  item then gets its uniforms and pipeline. The passes are sorted next:
  opaque surfaces by stencil mark and material, translucent ones back to
  front. All of the frame's working lists live in a `Scratch` that the next
  frame reuses.
- `shadows.rs`: the scene uniform, the sun's shadow map, and the shadow pass.
- `insets.rs`: insets. Each is a few draws seen by their own camera, drawn at
  4× MSAA (and 2× resolution unless pixelated) into their own targets, then
  laid over the output. The first targets fit the inset; targets for one
  that grows have room to spare.
- `fields.rs`: per-material half-float fields. The game patches them each
  frame, and draws read them at binding 3 in place of the normal map.
- `veil.rs`: the loading veil. It shows the frame's sky with a turning arc,
  and clears once nothing the frame draws is still on its way (or after 12 s).
- `validate.rs`: what a frame must be for the renderer to draw it.
- `frame.rs`: `submit` puts the frame together and encodes its passes. The
  shadow pass comes first. The scene draws opaque surfaces, then decals over
  them if there are any, then translucent surfaces and overlays. Then come
  the passes in `post.rs`, then sprites (`sprites.rs`), insets and the veil.

`post.rs` draws the scene into a floating-point target (`SCENE_FORMAT`),
multisampled and resolved for MSAA, at the render scale. Then come ambient
occlusion from its depth, sky, fog and cloud shadows, and bloom. Last, the
present pass tones highlights into white, applies FXAA Quality
(`shaders/fxaa.wgsl`), splits the colours of surfaces that ask for the
frame's aberration, and scales the image to the output.

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
  reload, shared renderers, shadow and variant compiles, and FXAA witnesses.
  Test in release: a Rust 1.99 release-mode miscompile once hid in the
  sampler setup.
- `cargo run --release -p pen-render --example render_packet` renders
  packets from `dist/packets` (made by `just packets`) with the assets in
  `dist/gdev/assets`, writes PNGs, and reports how long loading takes to
  settle. `--sequence` replays a run of packets in order and times each
  frame. See its header for the options.
