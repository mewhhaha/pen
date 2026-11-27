// The renderer: a frame's meshes, instanced batches and brush solids drawn
// lit into the scene's floating-point target, with shadows from the sun,
// decals, translucent surfaces and overlays; the passes after the scene
// (post.rs); then sprites, insets and the loading veil over the output.
// Assets load in the background and show once they are ready.
//
// Here are the renderer's state and the types its parts share. The parts:
// assets arriving from the loader (assets.rs), what a frame's draws bind
// (materials.rs), how its draws batch (batching.rs), the sun's shadows
// (shadows.rs), insets (insets.rs), fields (fields.rs), skinned draws'
// poses (skinning.rs), particles (particles.rs), the loading veil (veil.rs),
// checking a frame (validate.rs) and drawing it (frame.rs), and timing its
// passes on the GPU (timing.rs).
mod assets;
mod batching;
mod cascades;
mod fields;
mod frame;
mod gpu_culling;
mod grading;
mod hiz;
mod insets;
mod inspect;
mod instance_upload;
mod materials;
mod morphing;
mod particles;
pub(crate) mod readback;
mod reflections;
mod resolution;
mod ribbons;
#[cfg(test)]
mod shadow_tests;
mod shadows;
mod skinning;
mod thin_gbuffer;
mod timing;
mod validate;
mod veil;
mod velocity;
mod wired_passes;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::mpsc::Receiver;

use anyhow::Result;
use pen_protocol::boundary::MAX_DRAW_EXTRA;
use pen_protocol::{GameEvent, Matrix, Sky, Solid, Vector};

use crate::asset_store::{AssetStore, Lod, MAPS_FLOATS, MAX_LODS, MESH_VERTEX_FLOATS, Material, SURFACE_FLOATS};
use crate::built_in::{BuiltIn, BuiltIns};
use crate::culling::{Detail, MAX_VIEWS, Sphere, ViewMask, Views};
use crate::decals::{DecalBatch, Decals};
use crate::environment::Environment;
use crate::loader::{LevelMessage, Loader};
use crate::matrix::invert;
use crate::mipmaps::Mipmapper;
use crate::packed_texture::PackedHeader;
use crate::pipeline_cache::PipelineCache;
use crate::post::{Post, TONED_WGSL};
use crate::ray_tracing::BlasTriangle;
use crate::render2d::SdfShape;
use crate::shader::{self, EXTRA_VERTEX_BYTES, PipelineContext, Pipelines, TANGENT_BYTES, Variants};
use crate::solids::SolidBatch;
use crate::sprites::{SpriteBatch, Sprites, sprite_group_layout};
use crate::{RenderStats, RendererOptions};
use cascades::Cascade;
use pen_protocol::boundary::MAX_CASCADES;

#[cfg(test)]
pub(crate) use fields::half_of;
pub(crate) use gpu_culling::pipeline as culling_pipeline;
use gpu_culling::{GpuBatch, GpuCulling};
pub(crate) use insets::composite_pipeline;
pub(crate) use particles::pipeline as particle_pipeline;
pub(crate) use ribbons::pipeline as ribbon_pipeline;
pub use shadows::SHADOWS_WGSL;
pub(crate) use timing::{GpuTimer, Span};
pub use validate::validate_frame;
pub use wired_passes::{
    ENV_FOLIAGE_EXT_TAG, ENV_TERRAIN_EXT_TAG, ENV_WATER_EXT_TAG, EnvironmentFoliage, EnvironmentScene,
    EnvironmentTerrain, GPU_PARTICLES_EXT_TAG, GpuParticleScene, JFA_OUTLINE_EXT_TAG, JfaOutlineScene,
    JfaOutlineTarget, LINES_EXT_TAG, LineScene, NANITE_EXT_TAG, RENDER2D_EXT_TAG, RENDER2D_WORLD_EXT_TAG,
    RadianceCascades2dLayer, Scene2d, SpriteLayer2d, TilemapLayer2d,
};

pub(crate) fn velocity_sentinel_bytes() -> [u8; 4] {
    velocity::SENTINEL_F16_BYTES
}

use assets::{texture_from, whole};

// Insets draw with this many samples a pixel.
const INSET_SAMPLES: u32 = 4;

// The scene uniform: view, projection, ambient, light direction and color,
// the light's view-projection, the shadow settings, the sky's colors
// overhead, at the horizon and below it, and where the eye is (see
// scene.wgsl); then the shadow maps' count, band, filter and contact steps,
// view-projections and texels, and the soft shadows' settings (shadows.rs).
// Shaders may declare only its first part.
const SCENE_FLOATS: usize = 80 + 4 + 16 * MAX_CASCADES + 4 * MAX_CASCADES + 4;
const SCENE_BYTES: u64 = SCENE_FLOATS as u64 * 4;

// A draw's uniform: model and normal matrices, color, settings, the four
// parameters, up to 32 extra vectors (see pen-protocol's Draw), the material's
// five surface vectors (`asset_store::Surface::vectors`, byte 688), and the
// two UV-transform / PBR map-flag vectors (byte 768, total 800 bytes).
const SURFACE_AT: usize = 44 + MAX_DRAW_EXTRA * 4;
const MAPS_AT: usize = SURFACE_AT + SURFACE_FLOATS;
const DRAW_BYTES: u64 = (MAPS_AT + MAPS_FLOATS) as u64 * 4;

/// CPU geometry retained for BLAS construction on the first RT or DDGI use.
#[cfg_attr(not(feature = "render-rt"), allow(dead_code))]
pub(crate) struct RtMeshSource {
    id: u64,
    vertices: Vec<f32>,
    /// LOD0 triangle indices for mesh assets. Brush solids are already in triangle order.
    indices: Option<Vec<u32>>,
}

#[cfg_attr(not(feature = "render-rt"), allow(dead_code))]
impl RtMeshSource {
    fn triangles(&self) -> Vec<BlasTriangle> {
        let vertices = self.vertices.as_chunks::<MESH_VERTEX_FLOATS>().0;
        let triangle = |[a, b, c]: [&[f32; MESH_VERTEX_FLOATS]; 3]| {
            let position = |v: &[f32; MESH_VERTEX_FLOATS]| [v[0], v[1], v[2]];
            let normal = |v: &[f32; MESH_VERTEX_FLOATS]| [v[3], v[4], v[5]];
            BlasTriangle::smooth(position(a), position(b), position(c), normal(a), normal(b), normal(c))
        };
        if let Some(indices) = &self.indices {
            indices.as_chunks::<3>().0.iter().map(|indices| triangle(indices.map(|i| &vertices[i as usize]))).collect()
        } else {
            vertices.as_chunks::<3>().0.iter().map(|[a, b, c]| triangle([a, b, c])).collect()
        }
    }
}

/// GPU meshlet cluster descriptors + cluster-ordered index buffer for LOD0 indirect culling.
pub(crate) struct MeshletGpuAsset {
    pub indices: wgpu::Buffer,
    pub descriptors: wgpu::Buffer,
    #[cfg_attr(not(feature = "render-meshlets"), allow(dead_code))]
    pub count: u32,
}

pub(crate) struct MeshAsset {
    vertex_count: u32,
    vertices: wgpu::Buffer,
    /// Each vertex's tangent, if the mesh has them: pipelines read them at
    /// location 11, from the renderer's zeros when it has none.
    tangents: Option<wgpu::Buffer>,
    /// Optional packed `[uv1.x, uv1.y, r, g, b, a]` per vertex (`TEXCOORD_1` loc 12, `COLOR_0` loc 13).
    extra_attrs: Option<wgpu::Buffer>,
    /// Its triangles, then each simpler level of detail's.
    indices: wgpu::Buffer,
    /// A sphere holding its vertices.
    bounds: Sphere,
    /// Its levels of detail, the whole mesh first.
    lods: Vec<Lod>,
    /// A skinned mesh's joints and weights, and its rig.
    skin: Option<MeshSkin>,
    /// Blendshape / morph target deltas on the GPU, evaluated by compute pass.
    morphs: Option<MeshMorphs>,
    instances: Option<MeshInstanceBuffer>,
    /// LOD0 triangles for lazy ray-tracing / DDGI BLAS construction.
    #[cfg_attr(not(feature = "render-rt"), allow(dead_code))]
    rt_source: Option<Arc<RtMeshSource>>,
    /// LOD0 meshlet clusters for GPU frustum/backface/Hi-Z cluster culling.
    meshlets: Option<MeshletGpuAsset>,
    /// Multi-level continuous meshlet DAG for GPU screen-space error LOD cut selection.
    meshlet_dag: Option<crate::meshlet_dag::MeshletDagGpu>,
}

pub(crate) struct MeshMorphs {
    targets: u32,
    deltas: wgpu::Buffer,
}

pub(crate) struct MeshInstanceBuffer {
    buffer: wgpu::Buffer,
    count: u32,
}

pub(crate) struct MeshSkin {
    vertices: wgpu::Buffer,
    rig: Arc<crate::animation::Rig>,
}

pub(crate) struct MaterialAsset {
    material: Material,
    /// Which render state it has, but for its instance parameters: materials
    /// whose draws can share one instanced batch have the same.
    state: u32,
}

pub(crate) struct ShaderAsset {
    /// The load that made it, which background compiles answer to.
    revision: u64,
    module: wgpu::ShaderModule,
    /// Original WGSL source retained for on-demand WBOIT pipeline derivation.
    #[cfg_attr(not(feature = "render-oit"), allow(dead_code))]
    code: Arc<String>,
    /// Whether it has a `vertex_instanced` entry point.
    instancing: bool,
    /// Whether it has a `vertex_skinned` entry point.
    skinning: bool,
    /// Mesh-owned records, consumed by `vertex_persistent`.
    persistent: bool,
    /// Whether it reads the cascaded shadow maps (`shadow_cascades`).
    cascaded: bool,
    /// Whether it reads the opaque scene color mip chain (`opaque_color` / `opaque_behind`).
    reads_opaque_color: bool,
    /// Whether it reads the opaque nearest-depth texture (`opaque_depth`).
    reads_opaque_depth: bool,
    /// How far its vertex stage moves vertices from where their mesh puts
    /// them, if it says: its draws are culled by their bounds grown by it.
    margin: Option<f32>,
    /// The variants its loaded materials draw with.
    needs: Variants,
    /// Its pipelines by samples per pixel, compiled in the background.
    made: HashMap<(u32, crate::camera_precision::DepthMode), Pipelines>,
    /// Variants compiling in the background, and those that failed, by
    /// samples per pixel.
    compiling: HashMap<(u32, crate::camera_precision::DepthMode), Variants>,
    failed: HashMap<(u32, crate::camera_precision::DepthMode), Variants>,
    /// Its depth-only pipelines into the shadow map, plain, instanced and
    /// skinned (by `Stage`), compiled in the background when a loaded
    /// material casts shadows.
    shadow: [Option<wgpu::RenderPipeline>; 4],
    shadow_compiling: [bool; 4],
}

/// A shader's new revision, published once what its materials draw with has
/// compiled, or failed to; until then its last revision draws on.
struct Staged {
    shader: ShaderAsset,
    early: bool,
    /// The sample count frames drew with when it loaded.
    samples: u32,
    depth_mode: crate::camera_precision::DepthMode,
    /// Why a compile failed, if one did.
    failure: Option<String>,
}

// A texture on the GPU, perhaps still streaming in, its smallest levels
// first: the levels from `resident` (0 is the full size) down are there, and
// the level of detail drawn, `shown`, eases down to it as each finer one
// arrives, so detail fades in instead of popping.
pub(crate) struct TextureAsset {
    texture: wgpu::Texture,
    /// Its resident levels, as sampled.
    view: wgpu::TextureView,
    /// The same levels read as linear data, as normal maps are.
    linear: wgpu::TextureView,
    /// Which views these are: new ones get a new ID.
    view_id: u64,
    resident: u32,
    shown: f64,
    /// Whether a frame has drawn with it; detail only fades in once seen.
    seen: bool,
    /// The levels still to come.
    stream: Option<Receiver<LevelMessage>>,
}

pub(crate) enum GpuAsset {
    Mesh(MeshAsset),
    Material(Arc<MaterialAsset>),
    Shader(ShaderAsset),
    SpriteShader(wgpu::RenderPipeline),
    Texture(TextureAsset),
}

impl MeshAsset {
    pub(crate) fn gpu_bytes(&self) -> u64 {
        self.vertices.size()
            + self.indices.size()
            + self.tangents.as_ref().map_or(0, wgpu::Buffer::size)
            + self.extra_attrs.as_ref().map_or(0, wgpu::Buffer::size)
            + self.skin.as_ref().map_or(0, |s| s.vertices.size())
            + self.morphs.as_ref().map_or(0, |m| m.deltas.size())
            + self.instances.as_ref().map_or(0, |i| i.buffer.size())
            + self.meshlets.as_ref().map_or(0, |ml| ml.indices.size() + ml.descriptors.size())
            + self.meshlet_dag.as_ref().map_or(0, |d| d.clusters_buf.size() + d.index_buf.size())
    }
}

impl TextureAsset {
    pub(crate) fn gpu_bytes(&self) -> u64 {
        let format = self.texture.format();
        let (bw, bh) = format.block_dimensions();
        let block_bytes = u64::from(format.block_copy_size(None).unwrap_or(4));
        let mut w = self.texture.width().max(1);
        let mut h = self.texture.height().max(1);
        let mut total = 0u64;
        for _ in 0..self.texture.mip_level_count().max(1) {
            total += u64::from(w.div_ceil(bw)) * u64::from(h.div_ceil(bh)) * block_bytes;
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
        total * u64::from(self.texture.depth_or_array_layers().max(1))
    }
}

impl GpuAsset {
    pub(crate) fn gpu_bytes(&self) -> u64 {
        match self {
            GpuAsset::Mesh(mesh) => mesh.gpu_bytes(),
            GpuAsset::Texture(texture) => texture.gpu_bytes(),
            GpuAsset::Material(_) | GpuAsset::Shader(_) | GpuAsset::SpriteShader(_) => 0,
        }
    }
}

/// An asset made on this thread, published once its error scopes come back
/// empty.
struct Checking {
    slot: usize,
    revision: u64,
    early: bool,
    asset: GpuAsset,
    check: shader::Check,
}

/// A packed texture whose first levels are still arriving: it shows once
/// its levels up to 128 texels across are there.
struct PendingTexture {
    /// What the texture's allocation scopes caught, until it is known.
    check: Option<shader::Check>,
    slot: usize,
    revision: u64,
    early: bool,
    texture: wgpu::Texture,
    header: PackedHeader,
    finest: u32,
    levels: Receiver<LevelMessage>,
}

struct Veil {
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
    // When the first frame came, and when the veil began to clear.
    rose: Option<f64>,
    clearing: Option<f64>,
}

// The shadow maps and what draws into them, made when shadows are first on:
// each pass's scene uniform and bind group, which binds a stand-in map, as
// it renders into the real ones.
struct Shadows {
    /// One scene uniform and bind group for each pass that draws into a
    /// map: the sun's maps, then the local lights' faces, then the camera's
    /// depth for contact shadows.
    uniforms: Vec<wgpu::Buffer>,
    groups: Vec<wgpu::BindGroup>,
    sampler: wgpu::Sampler,
    stand_in: shadows::ShadowMaps,
    maps: Option<shadows::ShadowMaps>,
    /// Which maps `maps` are, to know when the scene's bind group must change.
    map_id: u64,
    /// The camera's depth, drawn for contact shadows when a frame asks, and
    /// which texture it is.
    contact: Option<shadows::ContactDepth>,
    contact_id: u64,
    contact_serial: u64,
    /// Which of `contact`'s two layers this frame draws (last frame's holds
    /// the other), and how many frames it has been drawn.
    contact_layer: usize,
    contact_frames: u64,
    /// A picture of one texel to bind where there is no last frame to read.
    blank: wgpu::TextureView,
    /// What a scene group binds at 14 to 17 where no translucent pass reads
    /// them: the picture above, a depth and a volume of one texel, and a block
    /// of zeros, whose flags say there is no picture and no mist.
    blank_depth: wgpu::TextureView,
    blank_volume: wgpu::TextureView,
    no_fog: wgpu::Buffer,
    /// The local lights' map faces, and what each holds.
    local: Option<shadows::LocalMaps>,
}

// Each inset's scene uniform and bind group, and the targets it draws into
// with 4 samples per pixel. The first targets fit the inset; one that
// outgrows them gets targets with room to spare and draws into their corner,
// so an inset that grows every frame keeps its textures rather than making
// new ones each frame.
struct InsetScene {
    uniform: wgpu::Buffer,
    /// Its sky light: its ambient color from every direction.
    light: wgpu::Buffer,
    group: wgpu::BindGroup,
}

struct InsetTargets {
    // The textures' size; an inset uses as much of it as it needs.
    width: u32,
    height: u32,
    pixelated: bool,
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
    resolved: wgpu::TextureView,
    placement: wgpu::Buffer,
    group: wgpu::BindGroup,
}

struct SolidMesh {
    /// The material's catalog index.
    material: u32,
    buffer: wgpu::Buffer,
    count: u32,
    /// Triangle snapshot for lazy ray-tracing / DDGI BLAS construction.
    #[cfg_attr(not(feature = "render-rt"), allow(dead_code))]
    rt_source: Option<Arc<RtMeshSource>>,
}

/// Independent per-camera temporal & history state so secondary views (RTT cameras and
/// split-screen viewports) never corrupt the primary camera's or each other's TAA, exposure,
/// volumetric fog, Hi-Z occlusion, or SSR history.
#[derive(Default)]
pub(super) struct ViewTemporalState {
    pub(super) post: Option<Post>,
    pub(super) volumetrics: Option<crate::volume::Volumetrics>,
    pub(super) hiz: Option<hiz::HiZ>,
    pub(super) reflected: Option<(wgpu::TextureView, Matrix, u64)>,
    pub(super) last_time: Option<f64>,
    depth_mode: Option<crate::camera_precision::DepthMode>,
    rt_history: wired_passes::RtHistory,
}

impl ViewTemporalState {
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            post: None,
            volumetrics: Some(crate::volume::Volumetrics::new(device, queue)),
            hiz: Some(hiz::HiZ::new(device, queue)),
            reflected: None,
            last_time: None,
            depth_mode: None,
            rt_history: Default::default(),
        }
    }
}

struct CameraTargetEntry {
    assets: [TextureAsset; 2],
    targets: [Option<crate::render_targets::OffscreenRenderTarget>; 2],
    front: usize,
    temporal: ViewTemporalState,
}

/// Which texture a draw binds: an asset's, or a built-in stand-in.
#[derive(Clone, Copy, PartialEq)]
enum TextureRef {
    Slot(usize),
    /// A field, by its index in `fields`.
    Field(usize),
    /// A live GPU-converted video texture, by its index in `videos`.
    Video(usize),
    /// A live render-to-texture secondary camera output, by its index in `camera_targets`.
    CameraTarget(usize),
    /// A GPU-packed 4x3 SurfaceMapAtlas (ORM + Emissive + PBR extension maps + Lightmaps).
    SurfaceAtlas(usize),
    White,
    FlatNormal,
    Transparent,
    Checker,
}

#[derive(Clone, Copy)]
enum Shape {
    /// A catalog mesh, drawn alone.
    Mesh(u32),
    /// An instanced batch.
    Batch(usize),
    /// A solid batch, in world space.
    Solid(usize),
}

/// The draw an item is, if it is one: of the frame, or of one of its insets.
#[derive(Clone, Copy)]
enum Source {
    Draw(usize),
    Inset(usize, usize),
    /// None: a batch or solid, in world space.
    World,
}

/// Something drawn with one draw call (in each view that sees it).
#[derive(Clone, Copy)]
struct Item {
    material: u32,
    source: Source,
    shape: Shape,
    /// The views that see it, a bit each (`Views::mask`), the camera's
    /// first; none, and it is left out of every pass.
    views: ViewMask,
    /// The level of detail it draws its mesh at, the whole mesh 0.
    lod: u8,
}

/// Where a frame's draw goes: alone, left out this frame, or into a batch,
/// with the parameters it takes if it has none of its own; how its mesh and
/// material go (`Scratch::choices`); and, unless the GPU culls its batch,
/// the views that see it and the level of detail it draws at.
#[derive(Clone, Copy)]
struct Placement {
    batch: u32,
    choice: u32,
    parameters: [f32; 4],
    views: ViewMask,
    lod: u8,
}

/// How a mesh and material's draws go this frame: alone or into a batch; the
/// sphere they are culled by, in the mesh's space grown by its shader's
/// margin, if they are; whether they cast shadows; and the errors of the
/// mesh's levels of detail.
#[derive(Clone, Copy)]
struct Choice {
    batch: Batching,
    bounds: Option<Sphere>,
    casts: bool,
    errors: [f32; MAX_LODS],
    levels: u8,
}

#[derive(Clone, Copy)]
enum Batching {
    Alone,
    /// Into this batch, with the parameters a draw without its own takes.
    Into(usize, [f32; 4]),
    /// Into a batch whose instanced pipeline is still compiling: left out.
    Waiting,
}

const ALONE: u32 = u32::MAX;

const LEFT_OUT: u32 = u32::MAX - 1;

struct InstanceBatch {
    mesh: usize,
    material: u32,
    pipeline: wgpu::RenderPipeline,
    /// Its draws' sphere and whether they cast (`Choice`), and how many there are.
    bounds: Option<Sphere>,
    casts: bool,
    count: u32,
    /// Culled on the GPU: where its draws lie; its instances, whole, in the
    /// instance buffer from `first`, `written` of them so far.
    gpu: Option<GpuBatch>,
    first: u32,
    written: u32,
    /// Culled here: its instances each view sees at each level of detail,
    /// in the instance buffer, the first and how many.
    ranges: [[(u32, u32); MAX_LODS]; MAX_VIEWS],
}

struct ReadyEntry {
    item: usize,
    offset: u32,
    group: wgpu::BindGroup,
    material: Arc<MaterialAsset>,
    shader: usize,
    /// The mesh's slot, for a mesh drawn alone.
    mesh: Option<usize>,
    /// What it draws with: opaque (or into an inset), instanced or
    /// translucent, or an overlay's dimmed pass and its full-strength one.
    pipeline: wgpu::RenderPipeline,
    visible: Option<wgpu::RenderPipeline>,
    /// A posed draw's joint matrices: where they lie in the frame's.
    skin: Option<u32>,
    /// A morphed draw's deformed vertex slice `(start, end)` in the frame's morph output buffer.
    morph: Option<(u64, u64)>,
    /// Optional GPU meshlet-culled indirect buffer slot for LOD0 main-camera draws.
    meshlet_slot: Option<usize>,
    /// Optional GPU continuous meshlet DAG LOD cut output slot for LOD0 main-camera draws.
    dag_slot: Option<usize>,
    persistent: bool,
}

/// What a frame works out, kept between frames so that drawing the next
/// reuses its memory.
#[derive(Default)]
struct Scratch {
    /// The views the frame draws: the camera's, then the shadow maps'; and
    /// what picks levels of detail.
    views: Views,
    detail: Detail,
    /// The bit of the view a reflection probe's capture sees by (0 for none),
    /// and the ready entries (indices) it draws.
    capture_bit: ViewMask,
    capture_list: Vec<usize>,
    /// How each mesh and material's draws go, and which of those is each's.
    choices: Vec<Choice>,
    chosen: HashMap<(u32, u32), u32>,
    /// Batches of materials with instance parameters, by mesh and render state.
    parameter_batches: HashMap<(u32, u32), usize>,
    placements: Vec<Placement>,
    batches: Vec<InstanceBatch>,
    gpu_order: Vec<usize>,
    items: Vec<Item>,
    ready: Vec<ReadyEntry>,
    /// Ready entries (indices) by pass.
    opaque: Vec<usize>,
    translucent: Vec<usize>,
    overlays: Vec<usize>,
    casters: Vec<usize>,
    /// The casters the camera sees, for the depth contact shadows read.
    contact_casters: Vec<usize>,
    /// The materials a frame draws meshes and solids with.
    materials: Vec<u32>,
    /// Sprites to draw, in batches, and decals.
    sprites: Vec<SpriteBatch>,
    sprite_order: Vec<u32>,
    decals: Vec<DecalBatch>,
    decal_order: Vec<u32>,
    /// The insets drawn, and how many pixels across and down each draws.
    insets: Vec<(usize, (u32, u32))>,
    /// Flat draw-target layout: scene draws occupy `0..scene_draw_count`, then
    /// inset `k`'s draws start at `inset_draw_bases[k]`.
    scene_draw_count: usize,
    inset_draw_bases: Vec<usize>,
    total_draw_targets: usize,
    /// The frame's pose layers by draw target index.
    pose_first: Vec<u32>,
    pose_order: Vec<u32>,
    /// Explicit custom joint matrices by draw target index.
    bone_first: Vec<u32>,
    bone_order: Vec<u32>,
    /// Morph target weights by draw target index.
    morph_first: Vec<u32>,
    morph_order: Vec<u32>,
    /// Previous-frame model transform index in `frame.motions` per scene draw (`u32::MAX` for static).
    motion_for_draw: Vec<u32>,
    /// The frame's emitters and ribbons ready to draw, in order.
    emitters: Vec<particles::Ready>,
    ribbons: Vec<ribbons::Ready>,
}

/// Indirect draw and two-phase occlusion candidate buffers for one meshlet-culled draw slot.
pub(super) struct MeshletIndirectSlot {
    pub count: u32,
    #[cfg_attr(not(feature = "render-meshlets"), allow(dead_code))]
    pub generation: u64,
    pub draws_phase1: wgpu::Buffer,
    pub draws_phase2: wgpu::Buffer,
    #[cfg_attr(not(feature = "render-meshlets"), allow(dead_code))]
    pub phase2_flags: wgpu::Buffer,
}

/// What drawing an item reads.
struct Drawing<'a> {
    assets: &'a AssetStore<GpuAsset>,
    items: &'a [Item],
    batches: &'a [InstanceBatch],
    solids: &'a [SolidMesh],
    instances: &'a wgpu::Buffer,
    /// Zero tangents for every vertex of the largest mesh without any.
    zero_tangents: &'a wgpu::Buffer,
    /// Default `[uv1=(0,0), color=(1,1,1,1)]` stream for meshes without `uvs1`/`colors`.
    default_extra_attrs: &'a wgpu::Buffer,
    /// The frame's joint matrices, for posed draws.
    skin: Option<&'a wgpu::BindGroup>,
    /// The frame's GPU-morphed vertex buffer, for draws with active morph weights.
    morphed: Option<&'a wgpu::Buffer>,
    culling: &'a GpuCulling,
    meshlet_indirect: &'a [MeshletIndirectSlot],
    dag_outputs: &'a [crate::meshlet_dag::MeshletDagTwoPhaseOutput],
    multi_draw_indirect: bool,
}

/// Floats for a uniform buffer, gathered on the stack.
struct Floats<const N: usize> {
    values: [f32; N],
    len: usize,
}

impl<const N: usize> Floats<N> {
    fn new() -> Self {
        Self { values: [0.0; N], len: 0 }
    }

    fn extend(&mut self, values: impl IntoIterator<Item = f32>) -> &mut Self {
        for value in values {
            self.values[self.len] = value;
            self.len += 1;
        }
        self
    }

    fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.values[..self.len])
    }
}

/// The light a scene is lit by: ambient, and the light's direction and
/// color.
fn lighting<const N: usize>(floats: &mut Floats<N>, ambient: &Vector, direction: &Vector, color: &Vector) {
    floats.extend(vector(ambient)).extend(vector(direction)).extend(vector(color));
}

/// The sky the scene sits under, which lights it from all round and which
/// glossy surfaces mirror, and where the eye is, seen by `view`.
fn heavens<const N: usize>(floats: &mut Floats<N>, sky: &Sky, view: &Matrix) {
    let eye = invert(view);
    floats.extend(vector(&sky.zenith)).extend(vector(&sky.horizon)).extend(vector(&sky.ground));
    floats.extend([eye[12], eye[13], eye[14], 1.0]);
}

fn vector(v: &Vector) -> [f32; 4] {
    [v.x, v.y, v.z, 0.0]
}

/// A premultiplied-alpha pipeline over the output, with the layout its
/// shader implies.
fn overlay_pipeline(context: &PipelineContext, code: &str, label: &str) -> wgpu::RenderPipeline {
    let device = &context.device;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(code.replace("${tonedWgsl}", TONED_WGSL).into()),
    });
    let premultiplied = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: context.format,
                blend: Some(wgpu::BlendState { color: premultiplied, alpha: premultiplied }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: context.cache.get(),
    })
}

fn uniform_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// A vertex buffer of `size` bytes, all zero (as every new buffer is): the
/// tangents of meshes that have none.
fn zero_tangent_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("zero tangents"),
        size,
        usage: wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    })
}

/// How many bytes the zero tangents of `current` bytes should have to reach
/// each of a mesh's `vertices`: as they are, or the next power of two (so the
/// stream doubles as meshes grow), at most the device's `limit`.
fn zero_tangent_size(current: u64, vertices: usize, limit: u64) -> u64 {
    let needed = vertices as u64 * TANGENT_BYTES;
    if needed <= current { current } else { needed.next_power_of_two().min(limit).max(needed) }
}

fn default_extra_attrs_buffer(device: &wgpu::Device, queue: &wgpu::Queue, vertices: usize) -> wgpu::Buffer {
    let count = vertices.max(1);
    let mut data = Vec::with_capacity(count * 6);
    for _ in 0..count {
        data.extend_from_slice(&[0.0f32, 0.0, 1.0, 1.0, 1.0, 1.0]);
    }
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("default extra vertex attrs"),
        size: count as u64 * EXTRA_VERTEX_BYTES,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck::cast_slice(&data));
    buf
}

impl State {
    /// Makes sure the zero tangents reach as many vertices as a mesh of
    /// `vertices` has: one stream serves every mesh without tangents, and it
    /// grows with the largest.
    fn ensure_zero_tangents(&mut self, vertices: usize) {
        let size = zero_tangent_size(self.zero_tangents.size(), vertices, self.device.limits().max_buffer_size);
        if size != self.zero_tangents.size() {
            self.zero_tangents = zero_tangent_buffer(&self.device, size);
        }
    }

    /// Makes sure the default `[uv1=(0,0), color=white]` vertex attribute stream covers `vertices`.
    fn ensure_default_extra_attrs(&mut self, vertices: usize) {
        let needed = vertices.max(1) as u64 * EXTRA_VERTEX_BYTES;
        if needed > self.default_extra_attrs.size() {
            let cap = vertices.next_power_of_two().max(1);
            let max_verts = (self.device.limits().max_buffer_size / EXTRA_VERTEX_BYTES) as usize;
            let count = cap.min(max_verts).max(vertices);
            self.default_extra_attrs = default_extra_attrs_buffer(&self.device, &self.queue, count);
        }
    }
}

fn extent(width: u32, height: u32) -> wgpu::Extent3d {
    wgpu::Extent3d { width, height, depth_or_array_layers: 1 }
}

type SurfaceAtlasKey = Vec<(u32, u64, u64)>;

pub(crate) struct State {
    device: wgpu::Device,
    queue: wgpu::Queue,
    context: PipelineContext,
    veil_seconds: Option<f32>,
    clock: crate::platform::Clock,
    // Dropped before the assets, so its streams stop.
    loader: Loader,
    assets: AssetStore<GpuAsset>,
    // The last frame's texture filtering and samples per pixel: those its
    // settings asked for, and those it drew with (which wait for the shaders).
    filtering: u32,
    samples_now: u32,
    samples_shown: u32,
    built_ins: BuiltIns,
    // Created on first use, so a renderer that never draws needs none.
    post: Option<Post>,
    // The froxel volume of the mist (volume.rs).
    volumetrics: crate::volume::Volumetrics,
    mipmaps: Mipmapper,
    /// The sky the scene is lit by: diffuse coefficients, reflection cube and table.
    environment: Environment,
    decals: Option<Decals>,
    sprites: Option<Sprites>,
    // The colour filter the whole picture draws through (filter.rs).
    pub(crate) filter: crate::filter::Filter,
    inset_scenes: Vec<Option<InsetScene>>,
    inset_targets: Vec<Option<InsetTargets>>,
    inset_sampler: wgpu::Sampler,
    inset_nearest_sampler: wgpu::Sampler,
    veil: Option<Veil>,
    // The scene uniform, the shadow map and its comparison sampler.
    scene_uniform: wgpu::Buffer,
    shadows: Option<Shadows>,
    // The frame's point and spot lights, sorted into clusters of its view.
    clusters: crate::lights::Clusters,
    // The lights as a reflection probe's capture sees them, and its faces' scene uniforms.
    capture_clusters: crate::lights::Clusters,
    capture_uniforms: Vec<wgpu::Buffer>,
    // Where frames for an HDR surface that needs decoding draw, shared with
    // the renderers drawing over this one's frames.
    screen: crate::display::SharedScreen,
    // The scene's bind group, and the maps, the layer and the contact depth it binds;
    // and the translucent pass's, which binds the opaque picture and the mist as well
    // where the other holds stand-ins.
    scene_group: Option<(wgpu::BindGroup, shadows::SceneKey)>,
    translucent_group: Option<(wgpu::BindGroup, shadows::SceneKey)>,
    // Last frame's resolved picture, which screen-space reflections read, the
    // view-projection it was seen through and which frame it is for; and this
    // frame's capture of a reflection probe.
    reflected: Option<(wgpu::TextureView, Matrix, u64)>,
    capture: Option<crate::environment::Step>,
    // The frame's sun shadow maps, nearest first, and how many.
    cascades: ([Cascade; MAX_CASCADES], usize),
    // The point and spot lights that cast this frame, and their faces.
    local: crate::local_shadows::Selection,
    stride: u64,
    capacity: usize,
    draw_uniform: wgpu::Buffer,
    uniform_data: Vec<f32>,
    // Draw bind groups by their three textures' views and their sampler, and
    // the frame that last drew with each.
    groups: HashMap<([u64; 3], i32), (wgpu::BindGroup, u64)>,
    instance_capacity: usize,
    gpu_instance_base: u32,
    instance_buffer: wgpu::Buffer,
    // What a mesh without tangents binds as them: a zero `vec4f` for each
    // vertex of the largest mesh loaded or meshed without (`ensure_zero_tangents`).
    zero_tangents: wgpu::Buffer,
    // Default `[uv1=(0,0), color=white]` stream for meshes without `uvs1`/`colors`.
    default_extra_attrs: wgpu::Buffer,
    // Cached GPU-packed 4x3 SurfaceMapAtlas textures keyed by sorted component texture view IDs.
    surface_atlases: Vec<TextureAsset>,
    surface_atlas_ids: HashMap<SurfaceAtlasKey, (usize, u32)>,
    surface_atlas_packer: Option<crate::surface_atlas::SurfaceAtlasPacker>,
    // Active display tonemap operator (`Shoulder` default).
    tonemap: crate::tonemap::TonemapOperator,
    // Wired end-to-end passes driven directly by RenderFrame (physical sky, area lights, PBR decals, RCAS/post-FX, OIT, split-screen).
    wired: wired_passes::WiredPasses,
    // Indirect draw-command & phase-2 candidate buffers populated by two-phase GPU meshlet culling.
    meshlet_indirect: Vec<MeshletIndirectSlot>,
    // Indirect draw-command outputs populated by continuous meshlet DAG LOD cut culling (two-phase Hi-Z).
    meshlet_dag_outputs: Vec<crate::meshlet_dag::MeshletDagTwoPhaseOutput>,
    instance_data: Vec<f32>,
    instance_uploads: instance_upload::Uploads,
    last_stats: RenderStats,
    // Timestamps around the frame's passes, where the device has them, and
    // whether the host wants them.
    timer: timing::GpuTimer,
    timing: bool,
    // Instances of large batches culled on the GPU, plus prior-frame Hi-Z occlusion pyramid.
    culling: GpuCulling,
    hiz: hiz::HiZ,
    /// Optional byte budget for resident meshes and textures (LRU-evicted after submit).
    asset_budget: Option<u64>,
    // The share of the frame's render scale drawn at, to hold a frame budget.
    resolution: resolution::Resolution,
    // Samplers by how many levels off the finest they keep, in quarters.
    samplers: HashMap<i32, wgpu::Sampler>,
    white: TextureAsset,
    flat_normal: TextureAsset,
    transparent: TextureAsset,
    checker: TextureAsset,
    // The fields the game keeps values in, by material key, as half floats
    // its draws read at binding 3; a new one reads zero until patches arrive.
    fields: Vec<(String, TextureAsset, fields::FieldShadow)>,
    field_ids: HashMap<String, usize>,
    // Live GPU-converted YUV420p video textures keyed by material texture name.
    videos: Vec<(String, TextureAsset, crate::video::VideoTexture)>,
    video_ids: HashMap<String, usize>,
    // Video texture keys whose streams are currently fetching asynchronously, kept out of the static image loader.
    reserved_videos: HashSet<String>,
    // Double-buffered render-to-texture secondary camera targets keyed by material texture name.
    // `front` is bound by materials while `back` is rendered into to prevent read/write hazards.
    camera_targets: Vec<CameraTargetEntry>,
    camera_target_ids: HashMap<String, usize>,
    next_view_id: u64,
    // Packed textures still bringing their first levels; published textures
    // still streaming the rest in, and those whose detail is still settling.
    pending: Vec<PendingTexture>,
    // Assets made on this thread whose error scopes have not come back yet.
    checking: Vec<Checking>,
    // Shaders' new revisions whose pipelines are compiling, by slot.
    staging: HashMap<usize, Staged>,
    streaming: Vec<usize>,
    fading: Vec<usize>,
    last_time: Option<f64>,
    // Whether the catalog is being loaded ahead of need.
    preloading: bool,
    // The slots a scene asked for first (`preload_scene`), and the catalog
    // to load ahead of need once they are in.
    scene: Vec<usize>,
    deferred: Option<pen_protocol::Catalog>,
    // Shaders the preload has sorted and asked for: one whose vertex stage
    // takes only builtins draws sprites; one that takes vertex attributes
    // draws meshes.
    sorted: HashSet<String>,
    // Each solid the last frame drew and its triangles by material, and the
    // triangles of all of them by material, remade when the solids change.
    solids: Vec<(Solid, Vec<SolidBatch>)>,
    // Counts the changes to the solids, for what caches what they cast.
    solids_revision: u64,
    solid_meshes: Vec<SolidMesh>,
    // Whether something this frame draws is still on its way.
    waiting: bool,
    scratch: Scratch,
    // Posed draws' joint matrices, GPU morph targets, velocity & thin G-buffer prepasses, particles, and ribbon trails.
    skinning: skinning::Skinning,
    morphing: morphing::Morphing,
    velocity: velocity::Velocity,
    thin_gbuffer: thin_gbuffer::ThinGbuffer,
    particles: Option<particles::Particles>,
    ribbons: Option<ribbons::Ribbons>,
    // Interned material render states (`MaterialAsset::state`).
    states: HashMap<String, u32>,
    // The frame the catalog's mesh and material keys were last checked in.
    checks: u64,
    checked_meshes: Vec<u64>,
    checked_materials: Vec<u64>,
    // Whether background work was under way at the last frame: once it ends,
    // the pipeline cache is written.
    busy: bool,
    frames: u64,
    loading_seen: bool,
    ready_sent: bool,
}

/// The scene group's layout (group 0 of every mesh pipeline): the scene uniform, the sun's
/// maps, the sky as light, the reflection probes, the depth and picture of screen-space
/// reflections, the camera's depth for contact shadows and what translucent surfaces read.
pub(crate) fn scene_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(SCENE_BYTES),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
            count: None,
        },
        // Every shadow map, the game's last (shadows.rs).
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        },
        // The sky as light: its reflection cube, the BRDF table, their
        // sampler and the diffuse coefficients (environment.rs).
        crate::environment::layout_entries()[0],
        crate::environment::layout_entries()[1],
        crate::environment::layout_entries()[2],
        crate::environment::layout_entries()[3],
        crate::environment::layout_entries()[4],
        // The reflection probes' cubes and their table.
        crate::environment::layout_entries()[5],
        crate::environment::layout_entries()[6],
        // The depth of this frame and last, and last frame's picture,
        // which screen-space reflections read.
        wgpu::BindGroupLayoutEntry {
            binding: 12,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 13,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        // The camera's depth, drawn for contact shadows (shadows.rs).
        wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        // What translucent surfaces read (translucency.wgsl): the opaque
        // picture with its chain of levels, its nearest depth, the froxel
        // volume and the fog's block. Stand-ins where there is no such pass.
        wgpu::BindGroupLayoutEntry {
            binding: 14,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 15,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 16,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 17,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(crate::volume::UNIFORM_BYTES),
            },
            count: None,
        },
    ]
}

/// The draw group's layout (group 1): the draw's uniform, its texture and sampler, and the
/// normal map and layer texture.
pub(crate) fn draw_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let texture_entry = |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(DRAW_BYTES),
            },
            count: None,
        },
        texture_entry(1, wgpu::ShaderStages::FRAGMENT),
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        // A normal map, or a field the vertices read too.
        texture_entry(3, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
        // Layer / emissive / standalone ORM / GPU-packed 4x3 SurfaceMapAtlas.
        texture_entry(4, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
    ]
}

impl State {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, options: RendererOptions) -> Result<Self> {
        let scene_group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene"),
            entries: &scene_layout_entries(),
        });
        let draw_group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("draw"),
            entries: &draw_layout_entries(),
        });
        let lights_group = crate::lights::group_layout(&device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene"),
            bind_group_layouts: &[Some(&scene_group), Some(&draw_group), Some(&lights_group)],
            immediate_size: 0,
        });
        let skin_group = skinning::group_layout(&device);
        let skinned_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skinned"),
            // The lights stay group 2, as in every mesh pipeline; the joints
            // come after them, the fourth and last group WebGPU allows.
            bind_group_layouts: &[Some(&scene_group), Some(&draw_group), Some(&lights_group), Some(&skin_group)],
            immediate_size: 0,
        });
        let sprite_group = sprite_group_layout(&device);
        let sprite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sprites"),
            bind_group_layouts: &[Some(&sprite_group)],
            immediate_size: 0,
        });
        let context = PipelineContext {
            device: device.clone(),
            layout,
            skinned_layout,
            sprite_layout,
            scene_group,
            draw_group,
            skin_group,
            lights_group,
            sprite_group,
            format: options.display.drawing_format(options.format),
            display: options.display,
            cache: PipelineCache::open(&device, options.pipeline_cache.as_deref()),
            depth_mode: Arc::new(std::sync::atomic::AtomicU8::new(0)),
        };
        let screen = crate::display::Screen::new(&device, options.display, options.format);
        let mesh_cache = crate::mesh_cache::MeshCache::from_pipeline_cache(options.pipeline_cache.as_deref());
        let loader = Loader::new(options.assets, context.clone(), mesh_cache)?;
        let screen = crate::display::SharedScreen::new(screen);
        let mut state = Self::with(device, queue, context, loader, options.veil, screen)?;
        state.request_culling();
        // What frames draw with at any quality, ahead of the first.
        state.built_ins.prepare([
            BuiltIn::BloomDown { first: true },
            BuiltIn::BloomDown { first: false },
            BuiltIn::BloomUp,
            BuiltIn::Composite,
        ]);
        Ok(state)
    }

    /// A renderer of the assets under `assets` on this one's device, loader
    /// threads and pipeline cache, with its own loads and events.
    pub fn sharing(&self, assets: std::path::PathBuf) -> Result<Self> {
        let loader = self.loader.sharing(assets, self.context.clone());
        let (device, queue, context) = (self.device.clone(), self.queue.clone(), self.context.clone());
        Self::with(device, queue, context, loader, None, self.screen.clone())
    }

    fn with(
        device: wgpu::Device,
        queue: wgpu::Queue,
        context: PipelineContext,
        loader: Loader,
        veil: Option<f32>,
        screen: crate::display::SharedScreen,
    ) -> Result<Self> {
        let assets = AssetStore::new(loader.jobs.clone());
        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment);
        let stride = DRAW_BYTES.div_ceil(alignment) * alignment;
        let mut mipmaps = Mipmapper::new(&device, context.cache.clone());
        let mut built_in = |id: u64, size: (u32, u32), pixels: &[u8]| -> Result<TextureAsset> {
            let texture =
                texture_from(&device, &queue, &mut mipmaps, size, pixels, wgpu::TextureFormat::Rgba8UnormSrgb, false)?;
            Ok(whole(texture, id))
        };
        let white = built_in(1, (1, 1), &[255, 255, 255, 255])?;
        let flat_normal = built_in(2, (1, 1), &[128, 128, 255, 255])?;
        let transparent = built_in(3, (1, 1), &[0, 0, 0, 0])?;
        let checker = built_in(4, (2, 2), &[255, 0, 255, 255, 30, 30, 30, 255, 30, 30, 30, 255, 255, 0, 255, 255])?;
        let veil_seconds = veil;
        let veil = veil.map(|_| Veil::new(&context));
        let linear_sampler = |filter: wgpu::FilterMode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: filter,
                min_filter: filter,
                ..Default::default()
            })
        };
        Ok(Self {
            veil_seconds,
            clock: crate::platform::Clock::new(),
            built_ins: BuiltIns::new(context.clone(), loader.jobs.clone()),
            loader,
            assets,
            filtering: 0,
            samples_now: 1,
            samples_shown: 1,
            post: None,
            volumetrics: crate::volume::Volumetrics::new(&device, &queue),
            environment: Environment::new(&device, context.cache.clone()),
            mipmaps,
            decals: None,
            sprites: None,
            filter: crate::filter::Filter::default(),
            inset_scenes: Vec::new(),
            inset_targets: Vec::new(),
            inset_sampler: linear_sampler(wgpu::FilterMode::Linear),
            inset_nearest_sampler: linear_sampler(wgpu::FilterMode::Nearest),
            veil,
            scene_uniform: uniform_buffer(&device, "scene", SCENE_BYTES),
            shadows: None,
            clusters: crate::lights::Clusters::new(&device, &context.lights_group),
            capture_clusters: crate::lights::Clusters::new(&device, &context.lights_group),
            capture_uniforms: Vec::new(),
            screen,
            scene_group: None,
            translucent_group: None,
            reflected: None,
            capture: None,
            cascades: ([Cascade::NONE; MAX_CASCADES], 0),
            local: Default::default(),
            stride,
            capacity: 1,
            draw_uniform: uniform_buffer(&device, "draws", stride),
            uniform_data: vec![0.0; (stride / 4) as usize],
            groups: HashMap::new(),
            instance_capacity: 1,
            gpu_instance_base: 0,
            instance_buffer: batching::instance_buffer(&device, 1),
            zero_tangents: zero_tangent_buffer(&device, TANGENT_BYTES),
            default_extra_attrs: default_extra_attrs_buffer(&device, &queue, 1),
            surface_atlases: Vec::new(),
            surface_atlas_ids: HashMap::new(),
            surface_atlas_packer: None,
            tonemap: crate::tonemap::TonemapOperator::Shoulder,
            wired: wired_passes::WiredPasses::default(),
            meshlet_indirect: Vec::new(),
            meshlet_dag_outputs: Vec::new(),
            instance_data: vec![0.0; 20],
            instance_uploads: instance_upload::Uploads::default(),
            last_stats: RenderStats::default(),
            timer: timing::GpuTimer::new(&device, &queue),
            timing: false,
            culling: GpuCulling::new(&device, &queue),
            hiz: hiz::HiZ::new(&device, &queue),
            asset_budget: None,
            resolution: resolution::Resolution::new(),
            samplers: HashMap::new(),
            white,
            flat_normal,
            transparent,
            checker,
            fields: Vec::new(),
            field_ids: HashMap::new(),
            videos: Vec::new(),
            video_ids: HashMap::new(),
            reserved_videos: HashSet::new(),
            camera_targets: Vec::new(),
            camera_target_ids: HashMap::new(),
            next_view_id: 5,
            pending: Vec::new(),
            checking: Vec::new(),
            staging: HashMap::new(),
            streaming: Vec::new(),
            fading: Vec::new(),
            last_time: None,
            preloading: false,
            scene: Vec::new(),
            deferred: None,
            sorted: HashSet::new(),
            solids: Vec::new(),
            solids_revision: 0,
            solid_meshes: Vec::new(),
            waiting: false,
            scratch: Scratch::default(),
            skinning: skinning::Skinning::new(&device),
            morphing: morphing::Morphing::new(&device),
            velocity: velocity::Velocity::new(&device),
            thin_gbuffer: thin_gbuffer::ThinGbuffer::new(&device),
            particles: None,
            ribbons: None,
            states: HashMap::new(),
            checks: 0,
            checked_meshes: Vec::new(),
            checked_materials: Vec::new(),
            busy: false,
            frames: 0,
            loading_seen: false,
            ready_sent: false,
            context,
            device,
            queue,
        })
    }

    fn now(&self) -> f64 {
        self.clock.now()
    }

    fn view_id(&mut self) -> u64 {
        self.next_view_id += 1;
        self.next_view_id
    }

    pub fn stats(&self) -> RenderStats {
        self.last_stats
    }

    pub fn gpu_culling(&mut self, from: Option<u32>) {
        self.culling.from = from;
    }

    pub fn occlusion_culling(&mut self, enabled: bool) {
        self.hiz.set_enabled(enabled);
    }

    pub fn gpu_asset_bytes(&self) -> u64 {
        self.assets.total_gpu_bytes()
    }

    pub fn set_asset_budget(&mut self, bytes: Option<u64>) {
        self.asset_budget = bytes;
    }

    pub fn frame_budget(&mut self, milliseconds: Option<f32>) {
        self.resolution.set_budget(milliseconds);
    }

    pub fn max_texture_dimension_2d(&self) -> u32 {
        self.device.limits().max_texture_dimension_2d
    }

    pub fn reserve_video_texture(&mut self, key: &str) {
        self.reserved_videos.insert(key.to_owned());
    }

    pub fn evict_video_texture(&mut self, key: &str) {
        self.reserved_videos.remove(key);
        if let Some(idx) = self.video_ids.remove(key) {
            let removed = self.videos.swap_remove(idx);
            self.forget_views(removed.1.view_id);
            if idx < self.videos.len() {
                let swapped_key = self.videos[idx].0.clone();
                self.video_ids.insert(swapped_key, idx);
            }
        }
    }

    pub fn upload_video_frame(&mut self, key: &str, frame: &crate::video::Yuv420Frame) -> Result<()> {
        let max_dim = self.device.limits().max_texture_dimension_2d;
        frame.validate(max_dim)?;
        self.upload_video_planes(
            key,
            frame.width,
            frame.height,
            frame.color_space,
            frame.siting,
            &frame.y,
            &frame.u,
            &frame.v,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upload_video_planes(
        &mut self,
        key: &str,
        width: u32,
        height: u32,
        color_space: crate::video::ColorSpace,
        siting: crate::video::ChromaSiting,
        y: &[u8],
        u: &[u8],
        v: &[u8],
    ) -> Result<()> {
        let max_dim = self.device.limits().max_texture_dimension_2d;
        anyhow::ensure!(
            width >= 2
                && height >= 2
                && width <= max_dim.min(crate::video::MAX_VIDEO_DIM)
                && height <= max_dim.min(crate::video::MAX_VIDEO_DIM)
                && width.is_multiple_of(2)
                && height.is_multiple_of(2),
            "invalid video frame dimensions {width}x{height} (limit {max_dim})"
        );
        self.reserved_videos.remove(key);
        let found = self.video_ids.get(key).copied();
        let needs_new = !matches!(found, Some(idx) if self.videos[idx].2.size() == (width, height));
        if needs_new {
            // Construct and upload into the new VideoTexture BEFORE swapping so an error
            // never invalidates a working texture.
            let video = crate::video::VideoTexture::new(&self.device, width, height)?;
            let mut encoder =
                self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("video_upload") });
            video.upload_planes(&self.queue, &mut encoder, width, height, color_space, siting, y, u, v)?;
            self.queue.submit(Some(encoder.finish()));
            let id = self.view_id();
            let texture = video.texture().clone();
            // Materials sample `view` via `Rgba8UnormSrgb` so display-referred YUV video
            // linearizes properly under PBR lighting & sRGB present encoding.
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                ..Default::default()
            });
            let linear = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let asset =
                TextureAsset { texture, view, linear, view_id: id, resident: 0, shown: 0.0, seen: false, stream: None };
            match found {
                Some(index) => {
                    let old = std::mem::replace(&mut self.videos[index].1, asset);
                    self.videos[index].2 = video;
                    self.forget_views(old.view_id);
                }
                None => {
                    let idx = self.videos.len();
                    self.videos.push((key.to_owned(), asset, video));
                    self.video_ids.insert(key.to_owned(), idx);
                }
            }
            return Ok(());
        }
        let index = found.unwrap();
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("video_upload") });
        self.videos[index].2.upload_planes(&self.queue, &mut encoder, width, height, color_space, siting, y, u, v)?;
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Executes a secondary camera `frame` into `target` with full temporal state isolation
    /// (`Post`/TAA, exposure, `Volumetrics`, `HiZ`, SSR `reflected`, `last_time`) and display
    /// output bypass (skipping parent `Screen::decode` and global `ColorFilter`), preserving all
    /// primary-frame counters, veil, and readiness state. Returns `(child_waiting, child_stats)`.
    pub(super) fn submit_secondary_view(
        &mut self,
        frame: &pen_protocol::RenderFrame,
        catalog: &pen_protocol::Catalog,
        target: &mut dyn crate::RenderTarget,
        temporal: &mut ViewTemporalState,
    ) -> Result<(bool, RenderStats)> {
        let parent_depth = self.depth_mode();
        let child_depth = crate::camera_precision::detect_depth_mode(&frame.projection).unwrap_or(parent_depth);
        if temporal.depth_mode != Some(child_depth) {
            temporal.post = None;
            temporal.volumetrics = None;
            temporal.hiz = None;
            temporal.reflected = None;
            temporal.rt_history = Default::default();
            temporal.depth_mode = Some(child_depth);
        }
        self.select_depth_mode(child_depth);
        std::mem::swap(&mut self.wired.rt_history, &mut temporal.rt_history);
        let mut cam_vol =
            temporal.volumetrics.take().unwrap_or_else(|| crate::volume::Volumetrics::new(&self.device, &self.queue));
        let mut cam_hiz = temporal.hiz.take().unwrap_or_else(|| hiz::HiZ::new(&self.device, &self.queue));
        std::mem::swap(&mut self.post, &mut temporal.post);
        std::mem::swap(&mut self.volumetrics, &mut cam_vol);
        std::mem::swap(&mut self.hiz, &mut cam_hiz);
        std::mem::swap(&mut self.reflected, &mut temporal.reflected);
        std::mem::swap(&mut self.last_time, &mut temporal.last_time);

        let saved_stats = self.last_stats;
        let saved_frames = self.frames;
        let saved_waiting = self.waiting;
        let saved_veil = self.veil.take();
        let saved_timing = self.timing;
        let saved_loading = self.loading_seen;
        let saved_ready = self.ready_sent;
        let saved_secondary = self.wired.in_secondary_camera;
        let saved_bypass = self.wired.bypass_display_output;

        self.timing = false;
        self.wired.in_secondary_camera = true;
        self.wired.bypass_display_output = true;
        let res = self.submit(frame, catalog, target);
        let child_waiting = self.waiting;
        let child_stats = self.last_stats;

        self.wired.bypass_display_output = saved_bypass;
        self.wired.in_secondary_camera = saved_secondary;
        self.last_stats = saved_stats;
        self.frames = saved_frames;
        self.waiting = saved_waiting;
        self.veil = saved_veil;
        self.timing = saved_timing;
        self.loading_seen = saved_loading;
        self.ready_sent = saved_ready;

        std::mem::swap(&mut self.post, &mut temporal.post);
        std::mem::swap(&mut self.volumetrics, &mut cam_vol);
        std::mem::swap(&mut self.hiz, &mut cam_hiz);
        std::mem::swap(&mut self.reflected, &mut temporal.reflected);
        std::mem::swap(&mut self.last_time, &mut temporal.last_time);
        temporal.volumetrics = Some(cam_vol);
        temporal.hiz = Some(cam_hiz);
        std::mem::swap(&mut self.wired.rt_history, &mut temporal.rt_history);
        self.select_depth_mode(parent_depth);
        res?;
        Ok((child_waiting, child_stats))
    }

    /// Renders a secondary camera's `frame` into a double-buffered offscreen target bound by any
    /// material whose `texture` key equals `key` (mirrors, portals, security monitors, minimaps).
    ///
    /// Maintains independent temporal state (`Post`/TAA, exposure, `Volumetrics`, `HiZ`, SSR) per key
    /// and preserves the main camera's stats, frame counter, veil, and readiness bookkeeping.
    /// Requires a right-handed view matrix (`det > 0`; use `planar_reflection_camera` for mirrors)
    /// and an SDR or linear scRGB output format (not PQ `Rgb10a2Unorm`).
    pub fn render_camera_to_texture(
        &mut self,
        key: &str,
        frame: &pen_protocol::RenderFrame,
        catalog: &pen_protocol::Catalog,
        size: (u32, u32),
    ) -> Result<()> {
        let format = self.context.format;
        anyhow::ensure!(
            format != wgpu::TextureFormat::Rgb10a2Unorm,
            "render_camera_to_texture cannot sample PQ-encoded Rgb10a2Unorm directly as material color"
        );
        let v = &frame.view;
        let det = v[0] * (v[5] * v[10] - v[6] * v[9]) - v[4] * (v[1] * v[10] - v[2] * v[9])
            + v[8] * (v[1] * v[6] - v[2] * v[5]);
        anyhow::ensure!(
            det > 1e-6,
            "render_camera_to_texture requires a right-handed view (det > 0, got {det}); use planar_reflection_camera for mirrors"
        );

        let found = self.camera_target_ids.get(key).copied();
        let make_asset = |rt: &crate::render_targets::OffscreenRenderTarget, view_id: u64| TextureAsset {
            texture: rt.texture().clone(),
            view: rt.sample_view().clone(),
            linear: rt.render_view().clone(),
            view_id,
            resident: 0,
            shown: 0.0,
            seen: false,
            stream: None,
        };

        let needs_alloc = match found {
            None => true,
            Some(idx) => {
                let entry = &self.camera_targets[idx];
                entry.targets[entry.front].as_ref().is_none_or(|rt| rt.dimensions() != size)
            }
        };
        if needs_alloc {
            let mut rt0 = crate::render_targets::OffscreenRenderTarget::new(&self.device, size.0, size.1, format)?;
            let rt1 = crate::render_targets::OffscreenRenderTarget::new(&self.device, size.0, size.1, format)?;
            let mut temporal = ViewTemporalState::new(&self.device, &self.queue);
            self.submit_secondary_view(frame, catalog, &mut rt0, &mut temporal)?;
            let id0 = self.view_id();
            let id1 = self.view_id();
            let assets = [make_asset(&rt0, id0), make_asset(&rt1, id1)];
            let targets = [Some(rt0), Some(rt1)];
            match found {
                Some(idx) => {
                    let old0 = self.camera_targets[idx].assets[0].view_id;
                    let old1 = self.camera_targets[idx].assets[1].view_id;
                    self.camera_targets[idx] = CameraTargetEntry { assets, targets, front: 0, temporal };
                    self.forget_views(old0);
                    self.forget_views(old1);
                }
                None => {
                    let idx = self.camera_targets.len();
                    self.camera_targets.push(CameraTargetEntry { assets, targets, front: 0, temporal });
                    self.camera_target_ids.insert(key.to_owned(), idx);
                }
            }
            return Ok(());
        }

        let idx = found.unwrap();
        let back = 1 - self.camera_targets[idx].front;
        let mut target = self.camera_targets[idx].targets[back].take().expect("camera back buffer present");
        let mut temporal = std::mem::take(&mut self.camera_targets[idx].temporal);
        let res = self.submit_secondary_view(frame, catalog, &mut target, &mut temporal);
        let entry = &mut self.camera_targets[idx];
        entry.targets[back] = Some(target);
        entry.temporal = temporal;
        res?;
        entry.front = back;
        Ok(())
    }

    pub fn time_passes(&mut self, timing: bool) {
        self.timing = timing;
    }

    pub fn take_events(&mut self) -> Vec<GameEvent> {
        let mut events = std::mem::take(&mut self.assets.events);
        // Only applications requesting a startup overlay receive this event.
        if self.loading_seen && !self.ready_sent && self.quiet() && self.pending() == 0 {
            self.ready_sent = true;
            events.push(GameEvent::RendererReady);
        }
        events
    }

    /// Work under way for assets: loads, textures streaming in or waiting
    /// on their error scopes, and pipelines compiling in the background.
    pub fn pending(&self) -> usize {
        let compiling = self
            .assets
            .slots
            .iter()
            .map(|slot| match &slot.current {
                Some(GpuAsset::Shader(shader)) => shader.compiling(),
                _ => 0,
            })
            .sum::<usize>();
        // A catalog waiting on a scene's assets is work to come.
        self.assets.in_flight()
            + usize::from(self.deferred.is_some())
            + self.pending.len()
            + self.checking.len()
            + self.streaming.len()
            + compiling
            + self.built_ins.compiling()
            + usize::from(self.culling.compiling())
    }

    /// Whether nothing is streaming in or settling, and the veil is gone.
    pub fn quiet(&self) -> bool {
        self.streaming.is_empty() && self.fading.is_empty() && self.veil.is_none()
    }

    /// Whether something the last frame draws is still on its way.
    pub fn waiting(&self) -> bool {
        self.waiting
    }

    pub fn set_hdri_environment(
        &mut self,
        image: &crate::hdri::HdriImage,
        intensity: f32,
        rotation_rad: f32,
    ) -> Result<[pen_protocol::Vector; 9]> {
        let sh = self.environment.set_hdri(&self.device, &self.queue, image, intensity, rotation_rad)?;
        // Invalidate scene bind groups so binding 4 switches to the HDRI cubemap view.
        self.scene_group = None;
        self.translucent_group = None;
        Ok(sh)
    }

    pub fn set_hdri_bytes(
        &mut self,
        bytes: &[u8],
        intensity: f32,
        rotation_rad: f32,
    ) -> Result<[pen_protocol::Vector; 9]> {
        let image = crate::hdri::HdriImage::from_bytes(bytes)?;
        self.set_hdri_environment(&image, intensity, rotation_rad)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_hdri_asset(
        &mut self,
        path: &std::path::Path,
        intensity: f32,
        rotation_rad: f32,
    ) -> Result<[pen_protocol::Vector; 9]> {
        let bytes = std::fs::read(path)?;
        self.set_hdri_bytes(&bytes, intensity, rotation_rad)
    }

    pub fn has_hdri_environment(&self) -> bool {
        self.environment.has_hdri()
    }

    pub fn clear_hdri_environment(&mut self) {
        self.environment.clear_hdri();
        self.scene_group = None;
        self.translucent_group = None;
    }

    pub fn depth_mode(&self) -> crate::camera_precision::DepthMode {
        self.context.depth_mode()
    }

    pub fn set_depth_mode(&mut self, mode: crate::camera_precision::DepthMode) {
        if self.context.depth_mode() == mode {
            return;
        }
        self.select_depth_mode(mode);
        self.post = None;
        self.wired.rt_history = Default::default();
        self.hiz.invalidate();
    }

    fn select_depth_mode(&mut self, mode: crate::camera_precision::DepthMode) {
        if self.depth_mode() == mode {
            return;
        }
        self.context.set_depth_mode(mode);
        self.particles = None;
        self.ribbons = None;
        self.decals = None;
        #[cfg(feature = "render-oit")]
        self.wired.oit_pipelines.clear();
        #[cfg(feature = "render-pbr-decals")]
        self.wired.pbr_decals.clear();
        #[cfg(feature = "render-env")]
        self.wired.env_pipelines.clear();
    }

    pub fn upload_terrain_heightmap(
        &self,
        width: u32,
        height: u32,
        heights: &[f32],
    ) -> Result<crate::environment_passes::TerrainHeightmap> {
        crate::environment_passes::TerrainPipeline::upload_heightmap(&self.device, &self.queue, width, height, heights)
    }

    pub fn set_environment_scene(&mut self, scene: EnvironmentScene) {
        self.wired.env_scene = Some(scene);
    }

    pub fn clear_environment_scene(&mut self) {
        self.wired.env_scene = None;
    }

    pub fn set_gpu_particle_scene(&mut self, scene: GpuParticleScene) {
        self.wired.gpu_particles.scene = Some(scene);
    }

    pub fn seed_gpu_particles(&mut self, particles: Vec<crate::gpu_particles::GpuParticle>) {
        if let Some(scene) = self.wired.gpu_particles.scene.as_mut() {
            scene.pending_seed = Some(particles);
        } else {
            self.wired.gpu_particles.scene =
                Some(GpuParticleScene { pending_seed: Some(particles), ..GpuParticleScene::default() });
        }
    }

    pub fn clear_gpu_particle_scene(&mut self) {
        self.wired.gpu_particles.scene = None;
        #[cfg(feature = "render-particles")]
        {
            self.wired.gpu_particles.system = None;
        }
        self.wired.gpu_particles.config_key = None;
        self.wired.gpu_particles.last_seed_snapshot.clear();
    }

    pub fn set_line_scene(&mut self, scene: LineScene) {
        self.wired.lines.scene = Some(scene);
    }

    pub fn clear_line_scene(&mut self) {
        self.wired.lines.scene = None;
    }

    pub fn set_jfa_outlines(&mut self, scene: JfaOutlineScene) {
        self.wired.jfa_outline.scene = Some(scene);
    }

    pub fn clear_jfa_outlines(&mut self) {
        self.wired.jfa_outline.scene = None;
    }

    pub fn upload_tilemap_cells(&self, cells: &[crate::render2d::TileCell]) -> Arc<wgpu::Buffer> {
        Arc::new(crate::render2d::TilemapPass::upload_cells(&self.device, cells))
    }

    pub fn upload_2d_texture_rgba8(&self, width: u32, height: u32, rgba: &[u8]) -> wgpu::TextureView {
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render2d_rgba8_texture"),
            size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            tex.as_image_copy(),
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width.max(1) * 4),
                rows_per_image: Some(height.max(1)),
            },
            wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
        );
        tex.create_view(&Default::default())
    }

    pub fn set_scene_2d(&mut self, scene: Scene2d) {
        self.wired.render2d.scene = Some(scene);
    }

    pub fn clear_scene_2d(&mut self) {
        self.wired.render2d.scene = None;
    }

    pub fn last_meshlet_cull_draws(&self) -> u32 {
        self.wired.last_meshlet_cull_draws
    }

    pub fn last_meshlet_phase2_active(&self) -> bool {
        self.wired.last_meshlet_phase2_active
    }

    pub fn set_dag_lod_error_threshold(&mut self, pixels: f32) {
        self.wired.dag_lod.enabled = pixels >= 0.0;
        if pixels >= 0.0 {
            self.wired.dag_lod.pixel_error_threshold = pixels;
        }
    }

    pub fn dag_lod_error_threshold(&self) -> f32 {
        self.wired.dag_lod.pixel_error_threshold
    }

    pub fn last_meshlet_dag_draws(&self) -> u32 {
        self.wired.last_meshlet_dag_draws
    }

    pub fn set_tonemap(&mut self, operator: crate::tonemap::TonemapOperator) {
        self.tonemap = operator;
        if let Some(post) = &mut self.post {
            post.set_tonemap(operator, self.context.display.encoded_peak());
        }
    }
}

#[cfg(test)]
mod tangent_tests {
    use super::*;

    #[test]
    fn the_zero_tangents_grow_to_the_largest_mesh_and_stay() {
        let limit = 1 << 30;
        // A first mesh of three vertices wants 48 bytes, taken as 64.
        assert_eq!(zero_tangent_size(TANGENT_BYTES, 3, limit), 64);
        // Meshes that fit change nothing.
        assert_eq!(zero_tangent_size(64, 4, limit), 64);
        assert_eq!(zero_tangent_size(64, 1, limit), 64);
        // A larger one doubles to a power of two that holds every vertex.
        let size = zero_tangent_size(64, 1000, limit);
        assert!(size >= 1000 * TANGENT_BYTES && size.is_power_of_two(), "{size}");
        // Never past the device's limit, but always every vertex's tangent.
        assert_eq!(zero_tangent_size(64, 1000, 20_000), 16_384);
        assert_eq!(zero_tangent_size(64, 1000, 16_100), 16_100);
        assert_eq!(zero_tangent_size(64, 1000, 100), 1000 * TANGENT_BYTES);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sampled textures the fragment stage of a mesh pipeline binds, across its groups.
    fn fragment_textures(entries: &[wgpu::BindGroupLayoutEntry]) -> Vec<u32> {
        entries
            .iter()
            .filter(|entry| {
                matches!(entry.ty, wgpu::BindingType::Texture { .. })
                    && entry.visibility.contains(wgpu::ShaderStages::FRAGMENT)
            })
            .map(|entry| entry.binding)
            .collect()
    }

    #[test]
    fn a_mesh_pipelines_fragment_stage_samples_fifteen_textures_within_strict_webgpu_limits() {
        let scene = fragment_textures(&scene_layout_entries());
        let draw = fragment_textures(&draw_layout_entries());
        let lights = fragment_textures(&crate::lights::layout_entries(false));
        // The sun's map and cascades, the sky's cube and BRDF table, the reflection probes' cubes,
        // the depth and picture of screen-space reflections, the camera's depth for contact shadows,
        // and the opaque picture, its depth and the froxel volume (bindings 14 to 16, fragment only).
        assert_eq!(scene, [1, 3, 4, 5, 10, 12, 13, 9, 14, 15, 16]);
        // The draw's albedo, normal map, and layer/emissive/ORM/SurfaceMapAtlas slot;
        // plus the local lights' shadow atlas.
        assert_eq!(draw, [1, 3, 4]);
        assert_eq!(lights, [4]);
        let total = scene.len() + draw.len() + lights.len();
        assert_eq!(total, 15, "mesh fragment stage must stay within strict 15-texture limit");
        assert!(total <= 15);
    }
}

impl crate::Renderer {
    pub fn set_devtools_jfa_outlines(&mut self, scene: JfaOutlineScene) {
        self.state.wired.jfa_outline.devtools_scene = Some(scene);
    }

    pub fn clear_devtools_jfa_outlines(&mut self) {
        self.state.wired.jfa_outline.devtools_scene = None;
    }

    pub fn set_devtools_line_scene(&mut self, scene: LineScene) {
        self.state.wired.lines.devtools_scene = Some(scene);
    }

    pub fn clear_devtools_line_scene(&mut self) {
        self.state.wired.lines.devtools_scene = None;
    }

    pub fn set_devtools_sdf_overlay(&mut self, shapes: Vec<SdfShape>) {
        self.state.wired.render2d.devtools_sdf_overlay = shapes;
    }

    pub fn clear_devtools_sdf_overlay(&mut self) {
        self.state.wired.render2d.devtools_sdf_overlay.clear();
    }
}
