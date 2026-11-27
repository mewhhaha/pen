//! wgpu renderer for a game's render packets: lit meshes, instanced batches
//! and brush solids with shadows and clustered point and spot lights, decals,
//! sky and fog, bloom, FXAA and TAA, SDR or HDR output, sprites (MSDF text
//! among them) and insets, over assets (KTX2 compressed textures among them)
//! that stream in on loader threads. Natively and in the browser (WebGPU)
//! alike. See ARCHITECTURE.md for how it works.
macro_rules! wgsl {
    ($name:literal) => {
        include_str!(concat!(env!("OUT_DIR"), "/shaders/", $name))
    };
}
pub(crate) use wgsl;

#[cfg(feature = "render-3d")]
macro_rules! deferred_wgsl {
    ($name:literal) => {{
        #[cfg(all(target_arch = "wasm32", feature = "deferred-shaders"))]
        {
            crate::deferred_shaders::expect($name)
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "deferred-shaders")))]
        {
            crate::wgsl!($name)
        }
    }};
}
#[cfg(feature = "render-3d")]
pub(crate) use deferred_wgsl;

#[cfg(feature = "render-3d")]
mod deferred_shaders;

#[cfg(feature = "render-3d")]
pub mod animation;
#[cfg(feature = "render-3d")]
pub mod area_lights;
#[cfg(feature = "render-3d")]
mod asset_store;
#[cfg(feature = "render-3d")]
pub mod atmosphere_lut;
#[cfg(feature = "render-3d")]
mod built_in;
#[cfg(feature = "render-3d")]
pub mod camera_precision;
#[cfg(feature = "render-3d")]
mod compressed;
#[cfg(feature = "render-3d")]
mod culling;
#[cfg(feature = "render-3d")]
pub mod ddgi;
#[cfg(feature = "render-3d")]
pub mod decals;
mod display;
#[cfg(feature = "render-3d")]
mod dof;
#[cfg(feature = "render-3d")]
mod environment;
#[cfg(feature = "render-3d")]
pub mod environment_passes;
#[cfg(feature = "render-3d")]
mod exposure;
mod filter;
#[cfg(feature = "render-3d")]
pub mod gpu_particles;
#[cfg(feature = "render-3d")]
mod grade;
#[cfg(feature = "render-3d")]
mod graph;
#[cfg(feature = "render-3d")]
pub mod hdri;
#[cfg(feature = "render-3d")]
mod inspect;
#[cfg(feature = "render-3d")]
pub mod jfa_outline;
#[cfg(feature = "render-3d")]
pub mod lightmap;
#[cfg(feature = "render-3d")]
mod lights;
#[cfg(feature = "render-3d")]
pub mod lines;
#[cfg(feature = "render-3d")]
mod loader;
#[cfg(feature = "render-3d")]
mod local_shadows;
mod matrix;
#[cfg(feature = "render-3d")]
mod mesh_cache;
#[cfg(feature = "render-3d")]
pub mod meshlet_dag;
#[cfg(feature = "render-3d")]
pub mod meshlets;
#[cfg(feature = "render-3d")]
mod mipmaps;
#[cfg(feature = "render-3d")]
mod motion_blur;
#[cfg(feature = "render-3d")]
pub mod oit;
mod packed_texture;
mod pipeline_cache;
mod platform;
#[cfg(feature = "render-3d")]
mod post;
#[cfg(feature = "render-3d")]
pub mod post_fx;
#[cfg(feature = "render-3d")]
mod prefetch;
#[cfg(feature = "render-3d")]
pub mod ray_tracing;
pub mod render2d;
#[cfg(feature = "render-3d")]
pub mod render_targets;
#[cfg(feature = "render-3d")]
mod renderer;
#[cfg(not(feature = "render-3d"))]
mod renderer2d_host;
#[cfg(feature = "render-3d")]
pub mod restir;
#[cfg(feature = "render-3d")]
pub mod restir_specular;
#[cfg(feature = "render-3d")]
mod shader;
#[cfg(all(feature = "render-3d", feature = "shader-graph"))]
pub mod shader_graph;
#[cfg(feature = "render-3d")]
pub mod sharc;
#[cfg(all(feature = "render-3d", feature = "simplify"))]
mod simplify;
#[cfg(feature = "render-3d")]
mod solids;
mod sort;
mod sprites;
#[cfg(feature = "render-3d")]
mod ssgi;
#[cfg(feature = "render-3d")]
mod subsurface;
#[cfg(feature = "render-3d")]
mod surface;
#[cfg(feature = "render-3d")]
pub mod surface_atlas;
#[cfg(feature = "render-3d")]
pub mod svgf;
#[cfg(feature = "render-3d")]
mod taa;
#[cfg(all(test, feature = "render-3d"))]
mod testing;
mod texture;
#[cfg(feature = "texture-recipes")]
mod texture_recipe;
#[cfg(feature = "render-3d")]
pub mod tonemap;
#[cfg(feature = "render-3d")]
pub mod video;
#[cfg(feature = "render-3d")]
mod volume;

use pen_protocol::{Catalog, GameEvent, RenderFrame};

#[cfg(feature = "render-3d")]
pub use area_lights::{
    AREA_LIGHTS_WGSL, AreaLight, AreaLightKind, AreaLightResources, LTC_LUT_SIZE, generate_ltc_luts,
};
#[cfg(feature = "render-3d")]
pub use asset_store::{mesh_bounds, pack_mesh, validate_asset_key};
#[cfg(not(feature = "render-3d"))]
pub fn validate_asset_key(key: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !(key.is_empty()
            || key.starts_with('/')
            || key.contains('\\')
            || key.split('/').any(|part| part.is_empty() || part == "..")),
        "asset key must stay inside assets: {key}"
    );
    Ok(())
}
#[cfg(feature = "render-3d")]
pub use atmosphere_lut::{ATMOSPHERE_LUT_WGSL, AtmosphereParams, CLOUDS_WGSL, PhysicalAtmosphere};
#[cfg(feature = "render-3d")]
pub use camera_precision::{
    CameraRelativeFrame, DepthMode, detect_depth_mode, perspective_forward_z, perspective_reversed_z,
};
#[cfg(feature = "render-3d")]
pub use ddgi::{
    DDGI_IRRADIANCE_INNER, DDGI_IRRADIANCE_TILE, DDGI_PACKED_HEADER_VEC4S, DDGI_VISIBILITY_INNER, DDGI_VISIBILITY_TILE,
    DDGI_WGSL, DdgiFrameParams, DdgiGridConfig, DdgiQueryPoint, DdgiVolume,
};
#[cfg(feature = "render-3d")]
pub use decals::{PBR_DECAL_WGSL, PbrDecal, PbrDecalPass, PbrDecalView};
pub use display::{DEFAULT_HEADROOM, Display, REFERENCE_WHITE_NITS, choose_surface};
#[cfg(feature = "render-3d")]
pub use environment::ENVIRONMENT_WGSL;
#[cfg(feature = "render-3d")]
pub use environment_passes::{
    FOLIAGE_WGSL, FoliageInstance, FoliagePipeline, FoliageUniform, PipelineTarget, TERRAIN_WGSL, TerrainHeightmap,
    TerrainPatch, TerrainPipeline, TerrainUniform, WATER_WGSL, WaterPipeline, WaterUniform,
};
pub use filter::ColorFilter;
#[cfg(feature = "render-3d")]
pub use gpu_particles::{
    GPU_PARTICLES_DRAW_WGSL, GPU_PARTICLES_SIM_WGSL, GpuParticle, GpuParticleMeshVertex, GpuParticleSimParams,
    GpuParticleSystem, ParticleSimCounters, unit_cube_particle_mesh,
};
#[cfg(feature = "render-3d")]
pub use hdri::{DEFAULT_HDRI_FACE_SIZE, DEFAULT_HDRI_SAMPLES, HdriCubemap, HdriImage, HdriPrefilterer, blend_sh_l2};
#[cfg(feature = "render-3d")]
pub use inspect::{
    TextureInfo, inspect_material, inspect_material_base_texture, inspect_mesh, inspect_texture, shader_draws_sprites,
};
#[cfg(feature = "render-3d")]
pub use jfa_outline::{
    JFA_OUTLINE_WGSL, JfaOutlinePass, MeshMaskParams, OUTLINE_OCCLUDED_BIT, OUTLINE_STYLE_CAPACITY, OutlineStyle,
    PreparedMeshMask,
};
#[cfg(feature = "render-3d")]
pub use lightmap::{
    BakedLightmap, GpuLightmapBaker, LIGHTMAP_BAKE_WGSL, LightmapBakeParams, LightmapTriangle, PackedLightmapMesh,
    pack_lightmap_uv2,
};
#[cfg(feature = "render-3d")]
pub use lights::LIGHTS_WGSL;
#[cfg(feature = "render-3d")]
pub use lines::{
    LINES_WGSL, LineBatch, LineFrame, LineLayer, LineRenderer, LineSegmentGpu, LineStyle, LineVertex, PreparedLines,
    XRayStyle,
};
#[cfg(feature = "render-3d")]
pub use meshlet_dag::{
    DagClusterGpu, MESHLET_DAG_CUT_WGSL, MeshletDagConfig, MeshletDagCullOutput, MeshletDagCullParams,
    MeshletDagCuller, MeshletDagGpu, MeshletDagMesh, MeshletDagTwoPhaseOutput, build_meshlet_dag,
};
#[cfg(feature = "render-3d")]
pub use meshlets::{
    DEFAULT_MAX_MESHLET_TRIANGLES, DEFAULT_MAX_MESHLET_VERTICES, MESHLET_CULL_WGSL, MeshletBatchEntry,
    MeshletCullParams, MeshletCullStats, MeshletCuller, MeshletGpu, MeshletIndirectDraw, MeshletMesh,
    MeshletTwoPhaseBatchEntry, build_meshlets,
};
#[cfg(feature = "render-3d")]
pub use oit::{OIT_SHADER_HELPERS_WGSL, OitPass, oit_color_targets};
#[cfg(feature = "render-3d")]
pub use post_fx::{
    FsrEasuUpscaler, FsrQualityPreset, POST_FX_WGSL, PostFxParams, PostFxPass, RcasSharpener, TaauUpscaler,
    UPSCALE_RCAS_WGSL, apply_projection_jitter, fsr_halton_jitter, halton_jitter,
};
#[cfg(feature = "render-3d")]
pub use prefetch::Prefetch;
#[cfg(feature = "render-3d")]
pub use ray_tracing::{
    BVH_RAYTRACE_WGSL, Blas, BlasTriGpu, BlasTriangle, BvhNodeGpu, BvhScene, BvhTriGpu, BvhTriangle, GpuBlasPool,
    GpuRayTracer, MAX_RT_LOCAL_LIGHTS, RayTraceParams, RtLocalLight, TLAS_RAYTRACE_WGSL, TLAS_TRAVERSAL_WGSL,
    TlasInstance, TlasInstanceGpu, TlasRayTraceParams, TlasRayTracer, TlasScene, TlasSceneGpu,
    device_supports_hardware_ray_query, supports_hardware_ray_query,
};
#[cfg(feature = "render-3d")]
pub use render_targets::{
    ColorTransfer, OffscreenRenderTarget, SPLIT_SCREEN_WGSL, SplitScreenCompositor, SplitScreenLayout, ViewportRect,
    oblique_clip_projection, planar_reflection_camera, planar_reflection_view,
};
pub use render2d::{
    Camera2d, Draw2d, RadianceCascades2d, RadianceCascades2dConfig, Scene2dEmitter, SdfShape, SdfShapePass, Sprite2d,
    Sprites2dPass, TileCell, TilemapPass, TilemapUniform, rasterize_scene_2d, sort_sprites,
};
#[cfg(feature = "render-3d")]
pub use restir::{RESTIR_MAX_LIGHTS, RESTIR_WGSL, RestirInputs, RestirParams, RestirRenderer};
#[cfg(feature = "render-3d")]
pub use restir_specular::{
    RESTIR_SPECULAR_WGSL, RestirSpecularInputs, RestirSpecularOutputs, RestirSpecularParams, RestirSpecularRenderer,
};
#[cfg(all(feature = "render-3d", feature = "shader-graph"))]
pub use shader_graph::{
    CompiledShaderGraph, NodeRef, PortType, RampStop, ShaderGraph, ShaderNode, ShaderNodeKind, is_shader_graph_key,
};
#[cfg(feature = "render-3d")]
pub use sharc::{
    SHARC_COMMON_WGSL, SHARC_DEFAULT_CAPACITY, SHARC_HASH_WGSL, SHARC_QUERY_RO_FN_WGSL, SHARC_WGSL, SharcCache,
    SharcConfig, SharcParams,
};
#[cfg(feature = "render-3d")]
pub use surface_atlas::{SURFACE_ATLAS_WGSL, SurfaceAtlasPacker, SurfaceTile, TileCopy};
#[cfg(feature = "render-3d")]
pub use svgf::{SVGF_WGSL, SvgfDenoiser, SvgfInputs, SvgfParams};
#[cfg(feature = "render-3d")]
pub use taa::{fsr_jitter_phase_count, fsr_mip_bias};
#[cfg(feature = "render-3d")]
pub use tonemap::{TONEMAP_SUITE_WGSL, TonemapOperator, tonemap_linear_f64};
#[cfg(feature = "render-3d")]
pub use video::{
    ChromaSiting, ColorMatrix, ColorRange, ColorSpace, IndexedY4mStream, IvfFrame, IvfStream, VideoPlayer,
    VideoTexture, Y4mStream, Yuv420Frame,
};
#[cfg(feature = "render-3d")]
pub use volume::{TRANSLUCENCY_WGSL, VolumeImage, VolumeLayer};

/// WGSL a sprite shader adds to draw text from a multi-channel signed
/// distance field atlas (pen-texture's `msdf`), crisp at any scale:
/// `msdf_coverage(msdf_median(field.rgb), uv, range, size)`.
pub const MSDF_WGSL: &str = crate::wgsl!("msdf.wgsl");
#[cfg(feature = "render-3d")]
pub use renderer::{
    ENV_FOLIAGE_EXT_TAG, ENV_TERRAIN_EXT_TAG, ENV_WATER_EXT_TAG, EnvironmentFoliage, EnvironmentScene,
    EnvironmentTerrain, GPU_PARTICLES_EXT_TAG, GpuParticleScene, JFA_OUTLINE_EXT_TAG, JfaOutlineScene,
    JfaOutlineTarget, LINES_EXT_TAG, LineScene, NANITE_EXT_TAG, RENDER2D_EXT_TAG, RENDER2D_WORLD_EXT_TAG,
    RadianceCascades2dLayer, SHADOWS_WGSL, Scene2d, SpriteLayer2d, TilemapLayer2d, validate_frame,
};
#[cfg(not(feature = "render-3d"))]
pub use renderer2d_host::validate_frame;
#[cfg(all(feature = "render-3d", feature = "simplify"))]
pub use simplify::pack_mesh_with_lods;
#[cfg(feature = "render-3d")]
pub use surface::{SURFACE_LOBES_WGSL, SURFACE_TRANSMISSION_WGSL, SURFACE_WGSL};

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
/// Device features the renderer uses when the adapter offers them. Pass to
/// `DeviceDescriptor::required_features` alongside [`optional_limits`].
pub fn optional_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    #[allow(unused_mut)]
    let mut wanted = wgpu::Features::PIPELINE_CACHE
        | wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
        | wgpu::Features::TEXTURE_COMPRESSION_BC
        | wgpu::Features::TEXTURE_COMPRESSION_ASTC
        | wgpu::Features::TEXTURE_COMPRESSION_ETC2;
    #[cfg(not(target_arch = "wasm32"))]
    {
        let limits = adapter.limits();
        if limits.max_blas_primitive_count > 0 && limits.max_tlas_instance_count > 0 {
            wanted |= wgpu::Features::EXPERIMENTAL_RAY_QUERY;
        }
    }
    adapter.features() & wanted
}

/// Device limits matching [`optional_features`]. Request the renderer's core
/// limits, within the adapter's available storage capacity, and enable native
/// acceleration structure limits when supported.
pub fn optional_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let available = adapter.limits();
    let defaults = wgpu::Limits::default();
    let base = wgpu::Limits {
        max_storage_buffers_per_shader_stage: defaults
            .max_storage_buffers_per_shader_stage
            .min(available.max_storage_buffers_per_shader_stage),
        max_storage_buffer_binding_size: defaults
            .max_storage_buffer_binding_size
            .min(available.max_storage_buffer_binding_size),
        ..defaults
    };
    #[cfg(not(target_arch = "wasm32"))]
    if adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
        && available.max_blas_primitive_count > 0
        && available.max_tlas_instance_count > 0
    {
        return base.using_acceleration_structure_values(available);
    }
    base
}

/// Experimental-feature opt-in token matching [`optional_features`] (enables
/// `EXPERIMENTAL_RAY_QUERY` on native adapters that support hardware RT;
/// remains disabled on WASM / adapters without hardware RT).
#[allow(unused_variables)]
pub fn optional_experimental_features(adapter: &wgpu::Adapter) -> wgpu::ExperimentalFeatures {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if optional_features(adapter).contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
            // SAFETY: requested only on native adapters advertising EXPERIMENTAL_RAY_QUERY
            // with non-zero acceleration structure limits, via validated wgpu ray-query APIs.
            return unsafe { wgpu::ExperimentalFeatures::enabled() };
        }
    }
    wgpu::ExperimentalFeatures::disabled()
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
    /// The point and spot lights that cast shadows this frame, the map faces
    /// they own, and how many of those the frame kept from the one before
    /// (nothing that casts into them moved) rather than drawing again.
    pub local_shadow_casters: u32,
    pub local_shadow_faces: u32,
    pub local_shadow_kept: u32,
    /// The render scale the scene drew at: the frame's, times what dynamic
    /// resolution took off it.
    pub scale: f32,
    /// How long the GPU took over the passes of a recent frame (a few frames
    /// back), while `Renderer::time_passes` or a frame budget has them timed
    /// and the device has timestamps.
    pub gpu: Option<GpuTimes>,
    /// The exposure, in stops, that a recent frame was shown with (a few frames
    /// back): a fixed frame's setting, or where auto exposure had adapted to.
    /// Read back without waiting for the GPU.
    pub exposure: Option<f32>,
    /// The froxels the volumetric fog filled this frame (0 when it is off or the
    /// mist is analytic alone), and whether it blended last frame's.
    pub volume_froxels: u32,
    pub volume_history: bool,
}

/// How long the GPU took over a frame's passes, in milliseconds: the shadow
/// maps, culling instances, the mist's froxel volume, the scene, the subsurface
/// diffusion of the surfaces that ask for it, the opaque
/// picture's passes (sky, mist, ambient occlusion, depth and the chain of
/// levels translucent surfaces read; in frames that have such surfaces,
/// particles or overlays, else they count as `post`), the passes after it, and
/// sprites and insets; and the whole frame, from its first pass's start to its
/// last's end.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuTimes {
    pub shadow: f32,
    pub culling: f32,
    pub volume: f32,
    pub scene: f32,
    pub subsurface: f32,
    pub opaque: f32,
    pub post: f32,
    pub overlay: f32,
    pub total: f32,
}

#[cfg(feature = "render-3d")]
pub struct Renderer {
    state: renderer::State,
}

#[cfg(not(feature = "render-3d"))]
pub struct Renderer {
    state: renderer2d_host::State2d,
}

#[cfg(not(feature = "render-3d"))]
impl Renderer {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, options: RendererOptions) -> anyhow::Result<Self> {
        Ok(Self { state: renderer2d_host::State2d::new(device, queue, options)? })
    }

    pub fn share(&self, assets: std::path::PathBuf) -> anyhow::Result<Renderer> {
        Ok(Self { state: self.state.sharing(assets)? })
    }

    pub fn preload(&mut self, catalog: &Catalog) {
        self.state.preload(catalog);
    }

    pub fn submit(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        target: &mut dyn RenderTarget,
    ) -> anyhow::Result<()> {
        self.state.submit(frame, catalog, target)
    }

    /// Submits a frame already validated with [`validate_frame`], skipping a duplicate pass.
    pub fn submit_validated(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        target: &mut dyn RenderTarget,
    ) -> anyhow::Result<()> {
        self.state.submit_validated(frame, catalog, target)
    }

    pub fn prepare_sprites(&mut self, sprites: &[pen_protocol::Sprite], catalog: &Catalog) -> anyhow::Result<bool> {
        self.state.prepare_sprites(sprites, catalog)
    }

    pub fn overlay(
        &mut self,
        sprites: &[pen_protocol::Sprite],
        catalog: &Catalog,
        output: &wgpu::TextureView,
        size: (u32, u32),
    ) -> anyhow::Result<()> {
        self.state.overlay(sprites, catalog, output, size)
    }

    pub fn take_events(&mut self) -> Vec<GameEvent> {
        self.state.take_events()
    }

    pub fn reload_all(&mut self) {
        self.state.reload_all();
    }

    pub fn quiet(&self) -> bool {
        self.state.quiet()
    }

    pub fn pending(&self) -> usize {
        self.state.pending()
    }

    pub fn stats(&self) -> RenderStats {
        self.state.stats()
    }

    pub fn set_color_filter(&mut self, filter: ColorFilter) {
        self.state.filter.filter = filter;
    }

    pub fn time_passes(&mut self, timing: bool) {
        self.state.time_passes(timing);
    }
}

#[cfg(feature = "render-3d")]
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

    /// Submits a frame already validated with [`validate_frame`], skipping a duplicate pass.
    pub fn submit_validated(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        target: &mut dyn RenderTarget,
    ) -> anyhow::Result<()> {
        self.state.submit_validated(frame, catalog, target)
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

    /// Makes auto exposure take the exposure the next frame's scene asks for at
    /// once, rather than adapt from the last one: for a host that restores an
    /// earlier moment of the game or replays one (a frame's `exposure.cut` does
    /// the same from the game).
    pub fn reset_exposure(&mut self) {
        self.state.reset_exposure();
    }

    /// Makes the froxel volume of the mist take no history from the frames
    /// before for the next frame it draws, as a host that restores or replays
    /// an earlier moment of the game does (a frame's `exposure.cut` does the
    /// same from the game).
    pub fn reset_volumetrics(&mut self) {
        self.state.reset_volumetrics();
    }

    /// Reads a texture of the last frame's froxel volume back, waiting for the
    /// GPU: for tools and tests. None before a frame has drawn one.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_volume(&self, layer: VolumeLayer) -> Option<VolumeImage> {
        self.state.read_volume(layer)
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

    /// Enables or disables prior-frame Hi-Z occlusion culling inside the GPU
    /// culling pass (enabled by default).
    pub fn occlusion_culling(&mut self, enabled: bool) {
        self.state.occlusion_culling(enabled);
    }

    /// Enables or disables indirect draw-arg CPU readback for GPU culling stats.
    pub fn culling_stats(&mut self, enabled: bool) {
        self.state.culling_stats(enabled);
    }

    /// Total GPU memory (bytes) currently held by resident meshes and textures.
    pub fn gpu_asset_bytes(&self) -> u64 {
        self.state.gpu_asset_bytes()
    }

    /// Sets an optional byte budget for resident meshes and textures. When set,
    /// frames that exceed the budget automatically LRU-evict idle meshes and
    /// textures unused for at least 2 frames; evicted assets reload on demand.
    pub fn set_asset_budget(&mut self, bytes: Option<u64>) {
        self.state.set_asset_budget(bytes);
    }

    /// Explicitly evicts idle resident meshes and textures unused for at least
    /// `max_age_frames`, oldest first, until resident GPU memory is at or below
    /// `target_bytes`. Returns the number of evicted assets.
    pub fn evict_unused(&mut self, max_age_frames: u64, target_bytes: u64) -> usize {
        self.state.evict_unused(max_age_frames, target_bytes)
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

    /// Maximum 2D texture side supported by the active WebGPU device.
    pub fn max_texture_dimension_2d(&self) -> u32 {
        self.state.max_texture_dimension_2d()
    }

    /// Reserves a video texture key while its stream is loading asynchronously so the static
    /// image loader does not attempt to decode the video file as PNG/JPEG.
    pub fn reserve_video_texture(&mut self, key: &str) {
        self.state.reserve_video_texture(key);
    }

    /// Evicts a live or reserved video texture key from the renderer, releasing its GPU texture.
    pub fn evict_video_texture(&mut self, key: &str) {
        self.state.evict_video_texture(key);
    }

    /// Uploads a planar YUV420p video frame and converts it on the GPU into a
    /// live dynamic RGBA texture bound by any material whose `texture` key
    /// equals `key`.
    pub fn upload_video_frame(&mut self, key: &str, frame: &Yuv420Frame) -> anyhow::Result<()> {
        self.state.upload_video_frame(key, frame)
    }

    /// Zero-copy planar YUV420p upload directly from an [`IndexedY4mStream`] backing buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_video_planes(
        &mut self,
        key: &str,
        width: u32,
        height: u32,
        color_space: ColorSpace,
        siting: ChromaSiting,
        y: &[u8],
        u: &[u8],
        v: &[u8],
    ) -> anyhow::Result<()> {
        self.state.upload_video_planes(key, width, height, color_space, siting, y, u, v)
    }

    /// Renders a secondary camera's `frame` into a double-buffered offscreen
    /// texture bound by any material whose `texture` key equals `key` (for
    /// planar reflections, portals, security monitors, or minimaps).
    pub fn render_camera_to_texture(
        &mut self,
        key: &str,
        frame: &RenderFrame,
        catalog: &Catalog,
        size: (u32, u32),
    ) -> anyhow::Result<()> {
        self.state.render_camera_to_texture(key, frame, catalog, size)
    }

    /// Loads an equirectangular HDRI image (`.hdr`/`.exr`) as the active IBL
    /// environment: prefilters its GGX specular cubemap on the GPU and returns
    /// the extracted L2 spherical harmonic diffuse irradiance coefficients.
    pub fn set_hdri_environment(
        &mut self,
        image: &HdriImage,
        intensity: f32,
        rotation_rad: f32,
    ) -> anyhow::Result<[pen_protocol::Vector; 9]> {
        self.state.set_hdri_environment(image, intensity, rotation_rad)
    }

    /// Auto-detects Radiance `.hdr` or OpenEXR `.exr` bytes and activates the prefiltered IBL cubemap.
    pub fn set_hdri_bytes(
        &mut self,
        bytes: &[u8],
        intensity: f32,
        rotation_rad: f32,
    ) -> anyhow::Result<[pen_protocol::Vector; 9]> {
        self.state.set_hdri_bytes(bytes, intensity, rotation_rad)
    }

    /// Loads a `.hdr` or `.exr` file from disk and activates it as the live IBL environment.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_hdri_asset(
        &mut self,
        path: &std::path::Path,
        intensity: f32,
        rotation_rad: f32,
    ) -> anyhow::Result<[pen_protocol::Vector; 9]> {
        self.state.load_hdri_asset(path, intensity, rotation_rad)
    }

    /// Whether a custom HDRI cubemap environment is currently active.
    pub fn has_hdri_environment(&self) -> bool {
        self.state.has_hdri_environment()
    }

    /// Reverts the environment cubemap back to the procedural sky gradient.
    pub fn clear_hdri_environment(&mut self) {
        self.state.clear_hdri_environment();
    }

    /// Active scene depth convention (`ForwardZ` or `ReversedZ`).
    pub fn depth_mode(&self) -> DepthMode {
        self.state.depth_mode()
    }

    /// Switches scene depth convention (`ForwardZ` or `ReversedZ`), recompiling depth-tested passes.
    pub fn set_depth_mode(&mut self, mode: DepthMode) {
        self.state.set_depth_mode(mode);
    }

    /// Uploads a validated `R32Float` terrain heightmap texture for [`EnvironmentScene`].
    pub fn upload_terrain_heightmap(
        &self,
        width: u32,
        height: u32,
        heights: &[f32],
    ) -> anyhow::Result<TerrainHeightmap> {
        self.state.upload_terrain_heightmap(width, height, heights)
    }

    /// Attaches an outdoor environment scene (CDLOD terrain, Gerstner water, GPU foliage) drawn
    /// inside every opaque scene pass.
    pub fn set_environment_scene(&mut self, scene: EnvironmentScene) {
        self.state.set_environment_scene(scene);
    }

    /// Detaches the active outdoor environment scene.
    pub fn clear_environment_scene(&mut self) {
        self.state.clear_environment_scene();
    }

    /// Attaches a GPU compute-simulated 3D mesh particle system stepped and drawn in `Renderer::draw`.
    pub fn set_gpu_particle_scene(&mut self, scene: GpuParticleScene) {
        self.state.set_gpu_particle_scene(scene);
    }

    /// Seeds or re-seeds live GPU particles on the next submitted frame.
    pub fn seed_gpu_particles(&mut self, particles: Vec<GpuParticle>) {
        self.state.seed_gpu_particles(particles);
    }

    /// Detaches and releases the active live GPU particle system.
    pub fn clear_gpu_particle_scene(&mut self) {
        self.state.clear_gpu_particle_scene();
    }

    /// Attaches a 3D/2D antialiased debug & vector line scene (depth-tested, X-ray, on-top, 2D overlay).
    pub fn set_line_scene(&mut self, scene: LineScene) {
        self.state.set_line_scene(scene);
    }

    /// Detaches the active antialiased line scene.
    pub fn clear_line_scene(&mut self) {
        self.state.clear_line_scene();
    }

    /// Attaches a multi-entity Jump-Flood silhouette outline scene composited onto the HDR scene.
    pub fn set_jfa_outlines(&mut self, scene: JfaOutlineScene) {
        self.state.set_jfa_outlines(scene);
    }

    /// Detaches the active JFA silhouette outline scene.
    pub fn clear_jfa_outlines(&mut self) {
        self.state.clear_jfa_outlines();
    }

    /// Uploads a GPU tilemap cell grid buffer for [`TilemapLayer2d`].
    pub fn upload_tilemap_cells(&self, cells: &[TileCell]) -> std::sync::Arc<wgpu::Buffer> {
        self.state.upload_tilemap_cells(cells)
    }

    /// Uploads a linear `Rgba8Unorm` 2D texture for [`SpriteLayer2d`] or [`TilemapLayer2d`].
    pub fn upload_2d_texture_rgba8(&self, width: u32, height: u32, rgba: &[u8]) -> wgpu::TextureView {
        self.state.upload_2d_texture_rgba8(width, height, rgba)
    }

    /// Attaches a first-class 2D world + vector SDF UI scene (tilemaps, 2D sprites, 2D Radiance Cascades GI, SDF UI).
    pub fn set_scene_2d(&mut self, scene: Scene2d) {
        self.state.set_scene_2d(scene);
    }

    /// Detaches the active 2D scene.
    pub fn clear_scene_2d(&mut self) {
        self.state.clear_scene_2d();
    }

    /// Sets the continuous meshlet DAG screen-space error threshold in pixels (`< 0.0` disables DAG LODs).
    pub fn set_dag_lod_error_threshold(&mut self, pixels: f32) {
        self.state.set_dag_lod_error_threshold(pixels);
    }

    /// Active continuous meshlet DAG screen-space error threshold in pixels.
    pub fn dag_lod_error_threshold(&self) -> f32 {
        self.state.dag_lod_error_threshold()
    }

    /// Number of LOD0 mesh draws dispatched through the continuous meshlet DAG LOD culler on the last frame.
    pub fn last_meshlet_dag_draws(&self) -> u32 {
        self.state.last_meshlet_dag_draws()
    }

    /// Number of LOD0 mesh draws dispatched through GPU meshlet cluster culling on the last frame.
    pub fn last_meshlet_cull_draws(&self) -> u32 {
        self.state.last_meshlet_cull_draws()
    }

    /// Whether Phase-2 disocclusion recovery ran on the last frame's meshlet-culled opaque pass.
    pub fn last_meshlet_phase2_active(&self) -> bool {
        self.state.last_meshlet_phase2_active()
    }

    /// Selects the display tonemapping operator (`Shoulder`, `AgX`, `AcesFilmic`, `PbrNeutral`).
    pub fn set_tonemap(&mut self, operator: TonemapOperator) {
        self.state.set_tonemap(operator);
    }
}
