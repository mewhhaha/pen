//! wgpu renderer for gdev render packets: a port of gdev's renderer.ts and
//! the modules it uses (post.ts, decals.ts, sprites.ts, solids.ts,
//! mipmaps.ts, matrix.ts, asset_store.ts, asset_files.ts, packed_texture.ts
//! and texture.ts). See PORTING.md.
mod asset_store;
mod decals;
mod loader;
mod matrix;
mod mipmaps;
mod packed_texture;
mod platform;
mod post;
mod renderer;
mod shader;
mod solids;
mod sprites;
mod texture;

use pen_protocol::{Catalog, GameEvent, RenderFrame};

pub use renderer::validate_frame;

pub struct RendererOptions {
    /// The swapchain format. Like the browser's preferred canvas format it is
    /// NOT sRGB (e.g. Bgra8Unorm); the present pass encodes as the TS does.
    pub format: wgpu::TextureFormat,
    /// Directory holding catalog.json and every asset key (e.g. dist/gdev/assets).
    pub assets: std::path::PathBuf,
    /// Seconds the loading veil takes to clear (gdev passes 0.6); None for no veil.
    pub veil: Option<f32>,
}

/// Where frames are drawn: a window's surface, or an offscreen texture.
pub trait RenderTarget {
    fn size(&self) -> (u32, u32);
    /// Acquire the frame's output view as late as possible, like `target.view()`.
    fn view(&mut self) -> Option<wgpu::TextureView>;
    fn present(&mut self);
}

/// What the last frame drew (the TS's `stats()`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderStats {
    pub draw_calls: u32,
    pub instanced_draw_calls: u32,
    pub instances: u32,
}

pub struct Renderer {
    state: renderer::State,
}

impl Renderer {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, options: RendererOptions) -> anyhow::Result<Self> {
        Ok(Self { state: renderer::State::new(device, queue, options)? })
    }

    /// `preload(catalog)`: prefetch every mesh and material (and their textures).
    pub fn preload(&mut self, catalog: &Catalog) {
        self.state.preload(catalog);
    }

    /// `validateFrame` + `submit(frame)`. Asset IDs in the frame are catalog
    /// indices; resolve them with `catalog.mesh(i)` / `catalog.material(i)`.
    pub fn submit(&mut self, frame: &RenderFrame, catalog: &Catalog, target: &mut dyn RenderTarget) -> anyhow::Result<()> {
        self.state.submit(frame, catalog, target)
    }

    /// Asset events the TS passes to `options.event` (AssetReady/AssetFailed);
    /// the app forwards them to the game.
    pub fn take_events(&mut self) -> Vec<GameEvent> {
        self.state.take_events()
    }

    /// `reload(keys)`: the files of these asset keys (relative to the asset
    /// root) changed. Each asset of theirs that was loaded or asked for loads
    /// again; the old one draws until the new revision is ready, and stays if
    /// it fails. AssetReady/AssetFailed report the new revision.
    pub fn reload(&mut self, keys: &[String]) {
        self.state.reload(keys);
    }

    /// `quiet()`: whether nothing is streaming in or settling, and the veil is gone.
    pub fn quiet(&self) -> bool {
        self.state.quiet()
    }

    /// Asset work under way (the TS store's `stats().pending`): loads,
    /// textures still streaming in, background pipeline compiles.
    pub fn pending(&self) -> usize {
        self.state.pending()
    }

    /// `stats()`, as a plain struct with the same fields.
    pub fn stats(&self) -> RenderStats {
        self.state.stats()
    }
}
