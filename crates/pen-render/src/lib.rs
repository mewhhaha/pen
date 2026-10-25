//! wgpu renderer for a game's render packets: lit meshes, instanced batches
//! and brush solids with shadows and clustered point and spot lights, decals,
//! sky and fog, bloom, FXAA and TAA, SDR or HDR output, sprites (MSDF text
//! among them) and insets, over assets (KTX2 compressed textures among them)
//! that stream in on loader threads. Natively and in the browser (WebGPU)
//! alike. See ARCHITECTURE.md for how it works.
pub mod animation;
mod asset_store;
mod built_in;
mod compressed;
mod culling;
mod decals;
mod display;
mod filter;
mod graph;
mod inspect;
mod lights;
mod loader;
mod matrix;
mod mipmaps;
mod packed_texture;
mod pipeline_cache;
mod platform;
mod post;
mod prefetch;
mod renderer;
mod shader;
#[cfg(feature = "simplify")]
mod simplify;
mod solids;
mod sprites;
mod taa;
mod texture;
mod texture_recipe;

use pen_protocol::{Catalog, GameEvent, RenderFrame};

pub use asset_store::{mesh_bounds, pack_mesh, validate_asset_key};
pub use display::{DEFAULT_HEADROOM, Display, REFERENCE_WHITE_NITS, choose_surface};
pub use filter::ColorFilter;
pub use inspect::{TextureInfo, inspect_material, inspect_mesh, inspect_texture, shader_draws_sprites};
pub use lights::LIGHTS_WGSL;
pub use prefetch::Prefetch;

/// WGSL a sprite shader adds to draw text from a multi-channel signed
/// distance field atlas (pen-texture's `msdf`), crisp at any scale:
/// `msdf_coverage(msdf_median(field.rgb), uv, range, size)`.
pub const MSDF_WGSL: &str = include_str!("shaders/msdf.wgsl");
pub use renderer::validate_frame;
#[cfg(feature = "simplify")]
pub use simplify::pack_mesh_with_lods;

pub struct RendererOptions {
    /// The swapchain format. Like the browser's preferred canvas format it is
    /// NOT sRGB (e.g. Bgra8Unorm); the present pass encodes colours itself.
    pub format: wgpu::TextureFormat,
    /// How the surface shows colours: `Display::SDR`, or an HDR color space
    /// `choose_surface` picked with `format`.
    pub display: Display,
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
/// adapter offers: request them with the device. `PIPELINE_CACHE` keeps
/// compiled pipelines between runs; `TIMESTAMP_QUERY` times the GPU's passes
/// (`RenderStats::gpu`); the texture compression families pick which KTX2
/// variant of a texture loads.
pub fn optional_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    adapter.features()
        & (wgpu::Features::PIPELINE_CACHE
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::TEXTURE_COMPRESSION_ASTC
            | wgpu::Features::TEXTURE_COMPRESSION_ETC2)
}

/// Where frames are drawn: a window's surface, or an offscreen texture.
pub trait RenderTarget {
    fn size(&self) -> (u32, u32);
    /// Acquire the frame's output view as late as possible, like `target.view()`.
    fn view(&mut self) -> Option<wgpu::TextureView>;
    fn present(&mut self);
}

/// What the last frame drew.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RenderStats {
    pub draw_calls: u32,
    pub instanced_draw_calls: u32,
    pub instances: u32,
    /// Instance bytes and queue writes submitted this frame; unchanged runs stay on the GPU.
    pub instance_upload_bytes: u64,
    pub instance_upload_calls: u32,
    /// Source bytes passed to field texture writes, queue calls and incoming samples.
    pub field_upload_bytes: u64,
    pub field_upload_calls: u32,
    pub field_input_texels: u64,
    /// Triangles drawn in the camera's view, levels of detail counted as
    /// drawn.
    pub triangles: u64,
    /// Draws and instances left out because they lie outside the camera's
    /// view, and casters left out of the shadow maps they would not reach
    /// (once for each map). Batches culled on the GPU count their instances,
    /// triangles and those culled as a recent frame's, read back.
    pub culled: u32,
    pub shadow_culled: u32,
    /// The render scale the scene drew at: the frame's, times what dynamic
    /// resolution took off it.
    pub scale: f32,
    /// How long the GPU took over the passes of a recent frame (a few frames
    /// back), while `Renderer::time_passes` or a frame budget has them timed
    /// and the device has timestamps.
    pub gpu: Option<GpuTimes>,
}

/// How long the GPU took over a frame's passes, in milliseconds: the shadow
/// maps, culling instances, the scene, the passes after it, and sprites and
/// insets; and the whole frame, from its first pass's start to its last's end.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuTimes {
    pub shadow: f32,
    pub culling: f32,
    pub scene: f32,
    pub post: f32,
    pub overlay: f32,
    pub total: f32,
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

    /// Loads the assets a scene draws first at once, as frames ask for
    /// theirs, ahead of the catalog: `(kind, key)` pairs, the kinds mesh,
    /// material, shader, spriteShader and texture. `samples`, the sample
    /// count its frames draw with (0 when unknown), has their pipelines
    /// compiled for it from the start.
    pub fn preload_scene(&mut self, assets: &[(String, String)], samples: u32) {
        self.state.preload_scene(assets, samples);
    }

    /// Takes the first loads of assets from `prefetch`, which read and
    /// decoded them before this renderer was made. Give it before anything
    /// loads.
    pub fn warm(&mut self, prefetch: Prefetch) {
        self.state.warm(prefetch);
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

    /// Whether the last frame drew without something it asked for, which
    /// is still on its way.
    pub fn waiting(&self) -> bool {
        self.state.waiting()
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

    /// The passes after the scene the last frame drew, in order (its render
    /// graph's, which leaves out passes nothing reads).
    pub fn passes(&self) -> Vec<&'static str> {
        self.state.passes()
    }

    /// Where the catalog's mesh or material (`pen_protocol::devtools_asset`
    /// `kind`) `key` stands (`devtools_asset_state`), and its revision.
    pub fn standing(&self, kind: u32, key: &str) -> (u32, u64) {
        self.state.standing(kind, key)
    }

    /// Material `index`'s texture once it is loaded, starting to load it
    /// until then: for pen's dev tools to draw pictures into.
    pub fn material_texture(&mut self, catalog: &Catalog, index: u32) -> anyhow::Result<Option<wgpu::Texture>> {
        self.state.material_texture(catalog, index)
    }

    /// The colour filter frames draw through from now on, over everything
    /// they draw (`ColorFilter::default()` for none).
    pub fn set_color_filter(&mut self, filter: ColorFilter) {
        self.state.filter.filter = filter;
    }

    /// Instanced batches of at least `from` instances are culled on the GPU,
    /// where the device can (from 256 unless set); with None, every batch is
    /// culled on the CPU.
    pub fn gpu_culling(&mut self, from: Option<u32>) {
        self.state.gpu_culling(from);
    }

    /// Times the GPU's passes (`RenderStats::gpu`), where the device has
    /// timestamps (`optional_features`). Off by default: timestamps between
    /// passes keep the GPU from overlapping them.
    pub fn time_passes(&mut self, timing: bool) {
        self.state.time_passes(timing);
    }

    /// Dynamic resolution: the milliseconds of GPU time a frame may take.
    /// The scene's render scale drops (to half the frame's at most) when
    /// frames take longer, and rises again when they take well under it
    /// (`RenderStats::scale`). None, the default, draws at the frame's scale.
    pub fn frame_budget(&mut self, milliseconds: Option<f32>) {
        self.state.frame_budget(milliseconds);
    }
}
