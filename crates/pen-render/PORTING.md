> Historical port measurements and shader notes. The TypeScript host is retired.
> Current runtime and browser entry points are described in pen's README.

# Renderer port notes gdev's renderer to wgpu

This crate is a Rust/wgpu port of the WebGPU renderer in the gdev checkout at
`/home/jacobt/src/gdev` (read-only: never edit it or `../blot2`). gdev already
runs that renderer on Deno Desktop, whose WebGPU is wgpu, so its WGSL already
passes naga and the API maps almost one to one.

## Sources to port

| gdev file | What it is |
| --- | --- |
| `renderer.ts` (≈2100 lines) | `createRenderer`: scene, shadows, insets, overlays, materials/shaders, streaming textures, veil, `validateFrame` |
| `post.ts` | `createPost`: scene targets, atmosphere/sky, occlusion, bloom, FXAA/present, `sceneFormat`, `tonedWgsl` |
| `decals.ts`, `sprites.ts`, `solids.ts`, `mipmaps.ts`, `matrix.ts` | helpers the renderer uses |
| `asset_store.ts`, `asset_files.ts` | mesh/material parsing, the asset store, reading files under the asset root |
| `packed_texture.ts`, `texture.ts` | `.gtex` packed textures (deflated mip levels) and PNG/JPEG decoding |

Skip: hot reload / `assets.watch()`, `timestampWrites`, anything Deno- or
browser-specific. `asset_parsers.ts` is compile-time only (not needed).

Keep the WGSL text verbatim (inline WGSL strings in the TS move to Rust
string constants or `.wgsl` files with `include_str!`; material shaders are
read from the asset directory at runtime as today). Only change WGSL where
naga in wgpu 30 rejects it, and list each such change at the bottom of this
file. Keep the TS comments that explain *why*; match their density.

## Public API (lib.rs)

The app crate (`crates/pen`, owned by someone else) will call exactly this;
keep it stable:

```rust
pub struct RendererOptions {
    /// The swapchain format. Like the browser's preferred canvas format it is
    /// NOT sRGB (e.g. Bgra8Unorm); the present pass encodes as the TS does.
    pub format: wgpu::TextureFormat,
    /// Directory holding catalog.json and every asset key (e.g. dist/gdev/assets).
    pub assets: std::path::PathBuf,
    /// Seconds the loading veil takes to clear (gdev passes 0.6); None for no veil.
    pub veil: Option<f32>,
}

pub trait RenderTarget {
    fn size(&self) -> (u32, u32);
    /// Acquire the frame's output view as late as possible, like `target.view()`.
    fn view(&mut self) -> Option<wgpu::TextureView>;
    fn present(&mut self);
}

pub struct Renderer { /* ... */ }
impl Renderer {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, options: RendererOptions) -> anyhow::Result<Self>;
    /// `preload(catalog)`: prefetch every mesh and material (and their textures).
    pub fn preload(&mut self, catalog: &pen_protocol::Catalog);
    /// `validateFrame` + `submit(frame)`. Asset IDs in the frame are catalog
    /// indices; resolve them with `catalog.mesh(i)` / `catalog.material(i)`.
    pub fn submit(&mut self, frame: &pen_protocol::RenderFrame, catalog: &pen_protocol::Catalog, target: &mut dyn RenderTarget) -> anyhow::Result<()>;
    /// Asset events the TS passes to `options.event` (AssetReady/AssetFailed);
    /// the app forwards them to the game.
    pub fn take_events(&mut self) -> Vec<pen_protocol::GameEvent>;
    /// `quiet()`.
    pub fn quiet(&self) -> bool;
    /// `stats()`, as a plain struct with the same fields.
    pub fn stats(&self) -> RenderStats;
}
```

Types: `pen_protocol` (crates/pen-protocol) already mirrors `protocol.ts`
and decodes packets (`boundary::decode_frame`). Matrices are `[f32; 16]`
column-major. Draw/sprite/decal/solid materials and meshes are catalog
indices, not strings.

## Mapping notes

- `layout: "auto"` → `layout: None` and `pipeline.get_bind_group_layout(i)`.
- Async asset loading (the TS `AssetSource` + promises) → background loader
  thread(s) that read and decode (JSON parse, gtex inflate, PNG decode) and
  send results over a channel; GPU uploads and pipeline creation happen on
  the thread calling `submit`. Keep the streaming behaviour (smallest gtex
  levels first, detail fading in), the veil and `quiet()`.
- GPU error scopes: wgpu has `push_error_scope`/`pop_error_scope` (the pop is
  a future; `pollster::block_on` is fine, or poll at the next submit). An
  asset that fails should raise `AssetFailed` like the TS, not crash.
- No device features beyond defaults are requested; keep it that way.
- Use `bytemuck` for uniform/vertex uploads.

## Testing

- `cargo build -p pen-render` and `cargo clippy -p pen-render` stay clean.
- A real game bundle is at `/home/jacobt/src/pen/dist/gdev` (assets in
  `assets/`). Write `examples/render_packet.rs`: headless device, offscreen
  target, loads the catalog, renders, and writes a PNG.
- Real render packets from the running game will appear in
  `/home/jacobt/src/pen/dist/packets/` as `NNNN.f32` (raw little-endian
  f32 lanes) plus `schema.txt`. Decode with
  `pen_protocol::boundary::decode_frame(&lanes, schema, &catalog)`. Until they
  exist, build synthetic frames. Render several real packets and look at the
  PNGs (Read the image) to check they look like a 3D level with sky, lit
  meshes, sprites/UI and post effects rather than garbage.

## WGSL changes for naga

Other than the FXAA Quality replacement described below, WGSL strings from
the TS (decals.ts, mipmaps.ts and renderer.ts's composite and veil shaders)
are carried over verbatim, with the
TS template holes (`${...}`) filled by `str::replace`, and all 31 shader
assets in dist/gdev/assets compile under naga 30 unchanged.

## How the port departs from the TS

- Background work runs on loader threads (src/loader.rs): file reads, JSON,
  gtex inflating and PNG/JPEG decoding, and, where the TS uses
  `createRenderPipelineAsync`, shader modules and pipelines (a shader's
  first pipelines, its inset pipeline, `warm`ed sample counts, sprite
  pipelines). Shadow pipelines compile there only when a loaded opaque,
  lit material casts shadows. A material arriving after its shader requests
  the same background work. Newly visible shaders also request missing MSAA
  variants there, even when the frame's MSAA setting did not change.
  wgpu's `Device` is `Send + Sync` and error scopes are per thread, so each
  thread scopes its own work. Buffer and texture uploads, mipmap generation
  and built-in post/decal pipelines stay on the thread calling `submit`.
  The shared painted-decal variant prepares during initial scene settling,
  before gameplay introduces world-space labels.
  Shadow and scene variants never fall back to frame-thread compilation.
- FXAA uses the shared WGSL Quality algorithm in `src/fxaa.wgsl`, with a
  bounded preset-12 search and measured subpixel blending. It filters toned
  display RGB before HUD and UI sprites. [Measurements](../../docs/render-performance.md)
  cover independent supersampled references and actual browser agreement.
- Loads that arrive are taken in at the start of `submit`, as are streamed
  texture levels; the TS takes them whenever their promises settle.
- A frame whose `RenderTarget::view()` returns None is dropped (the TS would
  throw from `getCurrentTexture`).
- Draw bind groups are cached by texture-view IDs instead of WeakMaps, and
  groups unused for 120 frames are let go.
- Catalog indices out of range are rejected by `submit`, like the TS
  rejecting empty keys.
- Hot reload is `Renderer::reload(keys)`, the store's `reload` without the
  file watching: the app watches and debounces. A replaced asset is handed
  back by the store so the renderer drops the bind groups holding its
  views (the TS's WeakMaps let them go by themselves); pipelines, meshes
  and materials belong to their asset and are looked up afresh each frame.
- Field patches (`RenderFrame::fields`) are written into one r16float
  texture per material key, bound at binding 3 in place of the normal map,
  with renderer.ts's `halfOf` rounding (`half_of`). Insets keep "roomy"
  targets and draw into their corner with a viewport and scissor, as the TS
  does.
- `timestampWrites`, `close()` and `idle()` are not ported; dropping the
  `Renderer` stops its threads.

## Image verification

Native GPU regressions and independent supersampled FXAA witnesses cover the
shared renderer. Headless browser witnesses compare the same WGSL against native
output; the game protocol also has a native/browser engine conformance harness.
See [renderer measurements](../../docs/render-performance.md) and
[the browser host](../pen-web/README.md).

## In the browser

The crate also builds for `wasm32-unknown-unknown` on wgpu's WebGPU backend.
What differs lives in `src/platform.rs`: assets are fetched from under the
asset root's URL instead of read from files, PNG and JPEG are decoded by the
browser (`createImageBitmap`, as the TS client's `decodeTexture` does), the
clock is `performance.now()`, and warnings go to the console. The loader's
work is the same async code either way: natively each loader thread runs it
with `pollster::block_on`; in the browser each job is a `spawn_local` task.
Error scopes return a check future (`shader::scoped`): loader work awaits it,
and an asset made on the submitting thread is published once its check has
settled, which natively is at once, in the same call as before. A packed
texture is fetched whole before its levels stream in, one per task turn.

`../pen-web` is the browser front end in a separate Cargo workspace. It owns
input, scheduling, audio and services, and calls this renderer directly. Build
it with `crates/pen-web/build.sh`; `pen build --web` exports the host with the
guest programs and assets. Browser and native builds use separate target
directories to keep their incremental build locks independent.
