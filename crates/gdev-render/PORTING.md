# Porting gdev's renderer to wgpu

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

The app crate (`crates/gdev`, owned by someone else) will call exactly this;
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
    pub fn preload(&mut self, catalog: &gdev_protocol::Catalog);
    /// `validateFrame` + `submit(frame)`. Asset IDs in the frame are catalog
    /// indices; resolve them with `catalog.mesh(i)` / `catalog.material(i)`.
    pub fn submit(&mut self, frame: &gdev_protocol::RenderFrame, catalog: &gdev_protocol::Catalog, target: &mut dyn RenderTarget) -> anyhow::Result<()>;
    /// Asset events the TS passes to `options.event` (AssetReady/AssetFailed);
    /// the app forwards them to the game.
    pub fn take_events(&mut self) -> Vec<gdev_protocol::GameEvent>;
    /// `quiet()`.
    pub fn quiet(&self) -> bool;
    /// `stats()`, as a plain struct with the same fields.
    pub fn stats(&self) -> RenderStats;
}
```

Types: `gdev_protocol` (crates/gdev-protocol) already mirrors `protocol.ts`
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

- `cargo build -p gdev-render` and `cargo clippy -p gdev-render` stay clean.
- A real game bundle is at `/home/jacobt/src/bun-desktop/dist/gdev` (assets in
  `assets/`). Write `examples/render_packet.rs`: headless device, offscreen
  target, loads the catalog, renders, and writes a PNG.
- Real render packets from the running game will appear in
  `/home/jacobt/src/bun-desktop/dist/packets/` as `NNNN.f32` (raw little-endian
  f32 lanes) plus `schema.txt`. Decode with
  `gdev_protocol::boundary::decode_frame(&lanes, schema, &catalog)`. Until they
  exist, build synthetic frames. Render several real packets and look at the
  PNGs (Read the image) to check they look like a 3D level with sky, lit
  meshes, sprites/UI and post effects rather than garbage.

## WGSL changes for naga

(List each change and why.)
