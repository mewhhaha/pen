// The renderer: a frame's meshes, instanced batches and brush solids drawn
// lit into the scene's floating-point target, with shadows from the sun,
// decals, translucent surfaces and overlays; the passes after the scene
// (post.rs); then sprites, insets and the loading veil over the output.
// Assets load in the background and show once they are ready.
//
// Here are the renderer's state and the types its parts share. The parts:
// assets arriving from the loader (assets.rs), what a frame's draws bind
// (materials.rs), how its draws batch (batching.rs), the sun's shadows
// (shadows.rs), insets (insets.rs), fields (fields.rs), the loading veil
// (veil.rs), checking a frame (validate.rs) and drawing it (frame.rs).
mod assets;
mod batching;
mod fields;
mod frame;
mod insets;
mod materials;
mod shadows;
mod validate;
mod veil;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::mpsc::Receiver;

use anyhow::Result;
use pen_protocol::boundary::MAX_DRAW_EXTRA;
use pen_protocol::{GameEvent, Matrix, Sky, Solid, Vector};

use crate::asset_store::{AssetStore, Material};
use crate::built_in::{BuiltIn, BuiltIns};
use crate::decals::{DecalBatch, Decals};
use crate::loader::{LevelMessage, Loader};
use crate::matrix::invert;
use crate::mipmaps::Mipmapper;
use crate::packed_texture::PackedHeader;
use crate::pipeline_cache::PipelineCache;
use crate::post::{Post, TONED_WGSL};
use crate::shader::{self, PipelineContext, Pipelines, Variants};
use crate::solids::SolidBatch;
use crate::sprites::{SpriteBatch, Sprites, sprite_group_layout};
use crate::{RenderStats, RendererOptions};

#[cfg(test)]
pub(crate) use fields::half_of;
pub(crate) use insets::composite_pipeline;
pub use validate::validate_frame;

use assets::{texture_from, whole};

// Insets draw with this many samples a pixel.
const INSET_SAMPLES: u32 = 4;

// The scene uniform: view, projection, ambient, light direction and color,
// the light's view-projection, the shadow settings, the sky's colors
// overhead, at the horizon and below it, and where the eye is (see
// scene.wgsl).
const SCENE_BYTES: u64 = 320;

// A draw's uniform: model and normal matrices, color, settings, the four
// parameters, then up to 32 extra vectors (see pen-protocol's Draw).
const DRAW_BYTES: u64 = 176 + MAX_DRAW_EXTRA as u64 * 16;

pub(crate) struct MeshAsset {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    count: u32,
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
    /// Whether it has a `vertex_instanced` entry point.
    instancing: bool,
    /// The variants its loaded materials draw with.
    needs: Variants,
    /// Its pipelines by samples per pixel, compiled in the background.
    made: HashMap<u32, Pipelines>,
    /// Variants compiling in the background, and those that failed, by
    /// samples per pixel.
    compiling: HashMap<u32, Variants>,
    failed: HashMap<u32, Variants>,
    /// Its depth-only pipelines into the shadow map, plain and instanced,
    /// compiled in the background when a loaded material casts shadows.
    shadow: [Option<wgpu::RenderPipeline>; 2],
    shadow_compiling: [bool; 2],
}

/// A shader's new revision, published once what its materials draw with has
/// compiled, or failed to; until then its last revision draws on.
struct Staged {
    shader: ShaderAsset,
    early: bool,
    /// The sample count frames drew with when it loaded.
    samples: u32,
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

// The shadow map and what draws into it, made when shadows are first on.
// The shadow pass binds a stand-in map, as it renders into the real one.
struct Shadows {
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    stand_in: wgpu::TextureView,
    group: wgpu::BindGroup,
    map: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// Which map `map` is, to know when the scene's bind group must change.
    map_id: u64,
}

// Each inset's scene uniform and bind group, and the targets it draws into
// with 4 samples per pixel. The first targets fit the inset; one that
// outgrows them gets targets with room to spare and draws into their corner,
// so an inset that grows every frame keeps its textures rather than making
// new ones each frame.
struct InsetScene {
    uniform: wgpu::Buffer,
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
}

/// Which texture a draw binds: an asset's, or a built-in stand-in.
#[derive(Clone, Copy, PartialEq)]
enum TextureRef {
    Slot(usize),
    /// A field, by its index in `fields`.
    Field(usize),
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

/// Something drawn with one draw call.
#[derive(Clone, Copy)]
struct Item {
    material: u32,
    source: Source,
    shape: Shape,
}

/// Where a frame's draw goes: alone, left out this frame, or into a batch,
/// with the parameters it takes if it has none of its own.
#[derive(Clone, Copy)]
struct Placement {
    batch: u32,
    parameters: [f32; 4],
}

/// The batch a mesh and material's draws join, and the parameters a draw
/// without its own takes.
type BatchChoice = (usize, [f32; 4]);

const ALONE: u32 = u32::MAX;

const LEFT_OUT: u32 = u32::MAX - 1;

struct InstanceBatch {
    mesh: usize,
    material: u32,
    pipeline: wgpu::RenderPipeline,
    /// Its instances in the instance buffer.
    first: u32,
    count: u32,
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
}

/// What a frame works out, kept between frames so that drawing the next
/// reuses its memory.
#[derive(Default)]
struct Scratch {
    /// Each mesh and material's batch and parameters, if they batch.
    choices: HashMap<(u32, u32), Option<BatchChoice>>,
    /// Batches of materials with instance parameters, by mesh and render state.
    parameter_batches: HashMap<(u32, u32), usize>,
    placements: Vec<Placement>,
    batches: Vec<InstanceBatch>,
    items: Vec<Item>,
    ready: Vec<ReadyEntry>,
    /// Ready entries (indices) by pass.
    opaque: Vec<usize>,
    translucent: Vec<usize>,
    overlays: Vec<usize>,
    casters: Vec<usize>,
    /// The materials a frame draws meshes and solids with.
    materials: Vec<u32>,
    /// Sprites to draw, in batches, and decals.
    sprites: Vec<SpriteBatch>,
    sprite_order: Vec<u32>,
    decals: Vec<DecalBatch>,
    decal_order: Vec<u32>,
    /// The insets drawn, and how many pixels across and down each draws.
    insets: Vec<(usize, (u32, u32))>,
}

/// What drawing an item reads.
struct Drawing<'a> {
    assets: &'a AssetStore<GpuAsset>,
    items: &'a [Item],
    batches: &'a [InstanceBatch],
    solids: &'a [SolidMesh],
    instances: &'a wgpu::Buffer,
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
fn lighting(floats: &mut Floats<80>, ambient: &Vector, direction: &Vector, color: &Vector) {
    floats.extend(vector(ambient)).extend(vector(direction)).extend(vector(color));
}

/// The sky the scene sits under, which lights it from all round and which
/// glossy surfaces mirror, and where the eye is, seen by `view`.
fn heavens(floats: &mut Floats<80>, sky: &Sky, view: &Matrix) {
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

fn extent(width: u32, height: u32) -> wgpu::Extent3d {
    wgpu::Extent3d { width, height, depth_or_array_layers: 1 }
}

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
    mipmaps: Mipmapper,
    decals: Option<Decals>,
    sprites: Option<Sprites>,
    inset_scenes: Vec<Option<InsetScene>>,
    inset_targets: Vec<Option<InsetTargets>>,
    inset_sampler: wgpu::Sampler,
    inset_nearest_sampler: wgpu::Sampler,
    veil: Option<Veil>,
    // The scene uniform, the shadow map and its comparison sampler.
    scene_uniform: wgpu::Buffer,
    shadows: Option<Shadows>,
    scene_group: Option<(wgpu::BindGroup, u64)>,
    stride: u64,
    capacity: usize,
    draw_uniform: wgpu::Buffer,
    uniform_data: Vec<f32>,
    // Draw bind groups by their three textures' views and their sampler, and
    // the frame that last drew with each.
    groups: HashMap<(u64, u64, u64, i32), (wgpu::BindGroup, u64)>,
    instance_capacity: usize,
    instance_buffer: wgpu::Buffer,
    instance_data: Vec<f32>,
    last_stats: RenderStats,
    // Samplers by how many levels off the finest they keep, in quarters.
    samplers: HashMap<i32, wgpu::Sampler>,
    white: TextureAsset,
    flat_normal: TextureAsset,
    transparent: TextureAsset,
    checker: TextureAsset,
    // The fields the game keeps values in, by material key, as half floats
    // its draws read at binding 3; a new one reads zero until patches arrive.
    fields: Vec<(String, TextureAsset)>,
    field_ids: HashMap<String, usize>,
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
    // Shaders the preload has sorted and asked for: one whose vertex stage
    // takes only builtins draws sprites; one that takes vertex attributes
    // draws meshes.
    sorted: HashSet<String>,
    // Each solid the last frame drew and its triangles by material, and the
    // triangles of all of them by material, remade when the solids change.
    solids: Vec<(Solid, Vec<SolidBatch>)>,
    solid_meshes: Vec<SolidMesh>,
    // Whether something this frame draws is still on its way.
    waiting: bool,
    scratch: Scratch,
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
}

impl State {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, options: RendererOptions) -> Result<Self> {
        let scene_group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(SCENE_BYTES),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
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
        let draw_group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("draw"),
            entries: &[
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
                texture_entry(4, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene"),
            bind_group_layouts: &[Some(&scene_group), Some(&draw_group)],
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
            sprite_layout,
            scene_group,
            draw_group,
            sprite_group,
            format: options.format,
            cache: PipelineCache::open(&device, options.pipeline_cache.as_deref()),
        };
        let loader = Loader::new(options.assets, context.clone())?;
        let mut state = Self::with(device, queue, context, loader, options.veil)?;
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
        Self::with(self.device.clone(), self.queue.clone(), self.context.clone(), loader, None)
    }

    fn with(
        device: wgpu::Device,
        queue: wgpu::Queue,
        context: PipelineContext,
        loader: Loader,
        veil: Option<f32>,
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
            mipmaps,
            decals: None,
            sprites: None,
            inset_scenes: Vec::new(),
            inset_targets: Vec::new(),
            inset_sampler: linear_sampler(wgpu::FilterMode::Linear),
            inset_nearest_sampler: linear_sampler(wgpu::FilterMode::Nearest),
            veil,
            scene_uniform: uniform_buffer(&device, "scene", SCENE_BYTES),
            shadows: None,
            scene_group: None,
            stride,
            capacity: 1,
            draw_uniform: uniform_buffer(&device, "draws", stride),
            uniform_data: vec![0.0; (stride / 4) as usize],
            groups: HashMap::new(),
            instance_capacity: 1,
            instance_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instances"),
                size: 80,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            instance_data: vec![0.0; 20],
            last_stats: RenderStats::default(),
            samplers: HashMap::new(),
            white,
            flat_normal,
            transparent,
            checker,
            fields: Vec::new(),
            field_ids: HashMap::new(),
            next_view_id: 5,
            pending: Vec::new(),
            checking: Vec::new(),
            staging: HashMap::new(),
            streaming: Vec::new(),
            fading: Vec::new(),
            last_time: None,
            preloading: false,
            sorted: HashSet::new(),
            solids: Vec::new(),
            solid_meshes: Vec::new(),
            waiting: false,
            scratch: Scratch::default(),
            states: HashMap::new(),
            checks: 0,
            checked_meshes: Vec::new(),
            checked_materials: Vec::new(),
            busy: false,
            frames: 0,
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

    pub fn take_events(&mut self) -> Vec<GameEvent> {
        std::mem::take(&mut self.assets.events)
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
        self.assets.in_flight()
            + self.pending.len()
            + self.checking.len()
            + self.streaming.len()
            + compiling
            + self.built_ins.compiling()
    }

    /// Whether nothing is streaming in or settling, and the veil is gone.
    pub fn quiet(&self) -> bool {
        self.streaming.is_empty() && self.fading.is_empty() && self.veil.is_none()
    }
}
