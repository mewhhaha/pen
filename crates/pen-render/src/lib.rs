//! wgpu renderer for a game's render packets: lit meshes, instanced batches
//! and brush solids with shadows, decals, sky and fog, bloom, FXAA, sprites
//! and insets, over assets that stream in on loader threads. Natively and in
//! the browser (WebGPU) alike. See ARCHITECTURE.md for how it works.
mod asset_store;
mod built_in;
mod decals;
mod loader;
mod matrix;
mod mipmaps;
mod packed_texture;
mod pipeline_cache;
mod platform;
mod post;
mod renderer;
mod shader;
mod solids;
mod sprites;
mod texture;

use pen_protocol::{Catalog, GameEvent, RenderFrame};

pub use asset_store::pack_mesh;
pub use renderer::validate_frame;

pub struct RendererOptions {
    /// The swapchain format. Like the browser's preferred canvas format it is
    /// NOT sRGB (e.g. Bgra8Unorm); the present pass encodes colours itself.
    pub format: wgpu::TextureFormat,
    /// Directory holding catalog.json and every asset key (e.g. dist/gdev/assets).
    pub assets: std::path::PathBuf,
    /// Seconds the loading veil takes to clear (gdev passes 0.6); None for no veil.
    pub veil: Option<f32>,
    /// A directory to keep compiled pipelines in between runs, which only
    /// a device with `optional_features` (Vulkan) can; None to compile
    /// afresh each run. The browser keeps its own.
    pub pipeline_cache: Option<std::path::PathBuf>,
}

/// The features the renderer uses when a device has them, of those the
/// adapter offers: request them with the device.
pub fn optional_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    adapter.features() & wgpu::Features::PIPELINE_CACHE
}

/// Where frames are drawn: a window's surface, or an offscreen texture.
pub trait RenderTarget {
    fn size(&self) -> (u32, u32);
    /// Acquire the frame's output view as late as possible, like `target.view()`.
    fn view(&mut self) -> Option<wgpu::TextureView>;
    fn present(&mut self);
}

/// What the last frame drew.
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

    /// A renderer of the assets under `assets` sharing this one's device,
    /// loader threads and pipeline cache, for another program's sprites over
    /// this one's frames (pen's dev tools): its loads and asset events are
    /// its own, and it has no veil.
    pub fn share(&self, assets: std::path::PathBuf) -> anyhow::Result<Renderer> {
        Ok(Self { state: self.state.sharing(assets)? })
    }

    /// Prefetches every mesh and material of the catalog (and their textures).
    pub fn preload(&mut self, catalog: &Catalog) {
        self.state.preload(catalog);
    }

    /// Validates `frame` and draws it. Asset IDs in the frame are catalog
    /// indices; resolve them with `catalog.mesh(i)` / `catalog.material(i)`.
    pub fn submit(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        target: &mut dyn RenderTarget,
    ) -> anyhow::Result<()> {
        self.state.submit(frame, catalog, target)
    }

    /// Starts loading what `sprites` draw with; whether it is all ready.
    pub fn prepare_sprites(&mut self, sprites: &[pen_protocol::Sprite], catalog: &Catalog) -> anyhow::Result<bool> {
        self.state.prepare_sprites(sprites, catalog)
    }

    /// Draws `sprites` over `output`, a finished image of `size`: another
    /// program's interface (pen's dev tools) over the game's frame. Sprites
    /// whose assets are still loading wait for them.
    pub fn overlay(
        &mut self,
        sprites: &[pen_protocol::Sprite],
        catalog: &Catalog,
        output: &wgpu::TextureView,
        size: (u32, u32),
    ) -> anyhow::Result<()> {
        self.state.overlay(sprites, catalog, output, size)
    }

    /// Asset events (AssetReady/AssetFailed) since the last call; the app
    /// forwards them to the game.
    pub fn take_events(&mut self) -> Vec<GameEvent> {
        self.state.take_events()
    }

    /// The files of these asset keys (relative to the asset
    /// root) changed. Each asset of theirs that was loaded or asked for loads
    /// again; the old one draws until the new revision is ready, and stays if
    /// it fails. AssetReady/AssetFailed report the new revision.
    pub fn reload(&mut self, keys: &[String]) {
        self.state.reload(keys);
    }

    /// Reload every requested mesh, material, shader and texture.
    pub fn reload_all(&mut self) {
        self.state.reload_all();
    }

    /// Blocks until the loader brings something (an asset, a compiled
    /// pipeline) or `timeout` passes, takes it in, and says whether anything
    /// came: for waiting on loading without spinning. Texture levels
    /// streaming in and detail fading in do not end the wait, and in the
    /// browser it never blocks.
    pub fn wait(&mut self, timeout: std::time::Duration) -> bool {
        self.state.wait(timeout)
    }

    /// Whether nothing is streaming in or settling, and the veil is gone.
    pub fn quiet(&self) -> bool {
        self.state.quiet()
    }

    /// Asset work under way: loads, textures still streaming in, background
    /// pipeline compiles.
    pub fn pending(&self) -> usize {
        self.state.pending()
    }

    /// What the last frame drew.
    pub fn stats(&self) -> RenderStats {
        self.state.stats()
    }
}
