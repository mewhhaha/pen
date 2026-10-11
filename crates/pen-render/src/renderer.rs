// The renderer (gdev's renderer.ts): a frame's meshes, instanced batches and
// brush solids drawn lit into the scene's floating-point target, with shadows
// from the sun, decals, translucent surfaces and overlays; the passes after
// the scene (post.rs); then sprites, insets and the loading veil over the
// output. Assets load in the background and show once they are ready.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, TryRecvError};

use anyhow::{Result, anyhow, ensure};
use pen_protocol::boundary::{MAX_DRAW_EXTRA, MAX_FIELD};
use pen_protocol::{Catalog, Draw, FieldPatch, GameEvent, Matrix, RenderFrame, Solid, Vector};

use crate::asset_store::{AssetKind, AssetStore, Material};
use crate::decals::{DECAL_MARK, DecalBatch, Decals};
use crate::loader::{INSET_SAMPLES, Job, LevelMessage, Loaded, Loader, Message, Priority};
use crate::matrix::{IDENTITY, invert, multiply};
use crate::mipmaps::Mipmapper;
use crate::packed_texture::{PackedHeader, PackedLevel, mip_levels, texel_bytes};
use crate::post::{Post, SCENE_FORMAT, TONED_WGSL};
use crate::shader::{self, Layer, PipelineContext, Pipelines};
use crate::solids::mesh_solids;
use crate::sprites::{SpriteBatch, Sprites, sprite_group_layout};
use crate::{RenderStats, RenderTarget, RendererOptions};

pub(crate) struct MeshAsset {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    count: u32,
}

pub(crate) struct MaterialAsset {
    material: Material,
    /// The material's render state but its instance parameters, which
    /// instanced batches share.
    state: String,
}

pub(crate) struct ShaderAsset {
    /// The load that made it, which background compiles answer to.
    revision: u64,
    module: wgpu::ShaderModule,
    /// Whether it has a `vertex_instanced` entry point.
    instancing: bool,
    /// Its pipelines by samples per pixel, made on first use.
    made: HashMap<u32, Pipelines>,
    /// Sample counts whose pipelines compile in the background.
    compiling: HashSet<u32>,
    /// Its opaque pipeline for insets, once compiled.
    inset: Option<wgpu::RenderPipeline>,
    /// Its depth-only pipelines into the shadow map, plain and instanced.
    shadow: [Option<wgpu::RenderPipeline>; 2],
}

impl ShaderAsset {
    fn inset(&self) -> Option<&wgpu::RenderPipeline> {
        self.inset.as_ref().or_else(|| self.made.get(&INSET_SAMPLES).map(|pipelines| &pipelines.opaque))
    }
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

// How long (seconds) detail takes to settle once a finer level arrives.
const SHARPENING: f64 = 0.12;
// A streamed texture shows once its levels up to 128 texels across are in.
const SHOWN_LEVELS: u32 = 8;
// How much of an overlay shows through the scene in front of it.
const OCCLUDED_STRENGTH: f64 = 0.3;
// The scene uniform: view, projection, ambient, light direction and color,
// the light's view-projection, the shadow settings, the sky's colors
// overhead, at the horizon and below it, and where the eye is (see
// scene.wgsl).
const SCENE_BYTES: u64 = 320;
// A draw's uniform: model and normal matrices, color, settings, the four
// parameters, then up to 32 extra vectors (see protocol.ts's Draw).
const DRAW_BYTES: u64 = 176 + MAX_DRAW_EXTRA as u64 * 16;
// How long (seconds) the veil waits on loads that bring nothing before it
// clears anyway.
const VEIL_PATIENCE: f64 = 12.0;
// Bind groups no frame has drawn with for this many frames are let go.
const GROUP_LIFETIME: u64 = 120;

/// The bits of the half float nearest `value` (renderer.ts's `halfOf`).
pub(crate) fn half_of(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xff) as i32 - 112;
    let mantissa = bits & 0x7fffff;
    if exponent >= 31 {
        return (sign | 0x7c00) as u16;
    }
    if exponent > 0 {
        return (sign | (((exponent as u32) << 10) + ((mantissa + 0x1000) >> 13))) as u16;
    }
    if exponent < -10 {
        return sign as u16;
    }
    (sign | ((((mantissa | 0x800000) >> (1 - exponent)) + 0x1000) >> 13)) as u16
}

pub fn validate_frame(frame: &RenderFrame) -> Result<()> {
    validate_matrix(&frame.view)?;
    validate_matrix(&frame.projection)?;
    for patch in &frame.fields {
        let most = MAX_FIELD as u32;
        ensure!(
            patch.width <= most
                && patch.height <= most
                && patch.x <= patch.width
                && patch.z <= patch.height
                && patch.columns <= patch.width - patch.x
                && patch.rows <= patch.height - patch.z
                && patch.values.len() == patch.columns as usize * patch.rows as usize
                && patch.values.iter().all(|value| value.is_finite()),
            "field patch must lie inside its field"
        );
    }
    let finite = |vector: &Vector| [vector.x, vector.y, vector.z].iter().all(|value| value.is_finite());
    for vector in [&frame.ambient, &frame.light_color, &frame.light_direction] {
        ensure!(finite(vector), "render lighting must contain finite float32 values");
    }
    ensure!(hypot(&frame.light_direction) >= 1e-8, "light direction must be nonzero");
    let aberration = &frame.aberration;
    ensure!(
        [aberration.x, aberration.y].iter().all(|value| value.is_finite() && value.abs() <= 0.25),
        "aberration must be finite and within a screen"
    );
    ensure!(frame.blur >= 0.0 && frame.blur <= 0.5, "blur must be within 0..0.5");
    let clear = &frame.clear;
    ensure!(
        [clear.r, clear.g, clear.b, clear.a].iter().all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
        "clear color must have 0..1 channels"
    );
    for draw in &frame.draws {
        validate_matrix(&draw.transform)?;
        if let Some(parameters) = &draw.parameters {
            ensure!(
                parameters.iter().all(|value| value.is_finite()),
                "draw parameters must contain four finite float32 values"
            );
        }
        if let Some(extra) = &draw.extra {
            ensure!(
                extra.len() % 4 == 0 && extra.len() <= MAX_DRAW_EXTRA * 4 && extra.iter().all(|value| value.is_finite()),
                "draw extra data must be up to 32 vectors of finite float32 values"
            );
        }
        determinant_of(&draw.transform)?;
    }
    for decal in &frame.decals {
        validate_matrix(&decal.transform)?;
        determinant_of(&decal.transform)?;
        ensure!(decal.opacity >= 0.0 && decal.opacity <= 1.0, "decal opacity must be within 0..1");
    }
    for solid in &frame.solids {
        for face in &solid.faces {
            let normal = &face.normal;
            ensure!(
                [normal.x, normal.y, normal.z, face.distance].iter().all(|value| value.is_finite()) && hypot(normal) >= 1e-6,
                "solid faces need a plane and a material"
            );
        }
    }
    Ok(())
}

fn hypot(vector: &Vector) -> f64 {
    let [x, y, z] = [vector.x, vector.y, vector.z].map(f64::from);
    (x * x + y * y + z * z).sqrt()
}

fn validate_matrix(matrix: &Matrix) -> Result<()> {
    ensure!(matrix.iter().all(|value| value.is_finite()), "render matrices must contain 16 finite float32 values");
    Ok(())
}

fn determinant_of(matrix: &Matrix) -> Result<f64> {
    let m = matrix.map(f64::from);
    let (a, b, c) = (m[0], m[1], m[2]);
    let (d, e, f) = (m[4], m[5], m[6]);
    let (g, h, i) = (m[8], m[9], m[10]);
    let determinant = a * (e * i - f * h) - d * (b * i - c * h) + g * (b * f - c * e);
    ensure!(determinant.is_finite() && determinant.abs() >= 1e-12, "draw transform must be invertible");
    Ok(determinant)
}

fn normal_matrix(matrix: &Matrix) -> Result<[f32; 16]> {
    let m = matrix.map(f64::from);
    let (a, b, c) = (m[0], m[1], m[2]);
    let (d, e, f) = (m[4], m[5], m[6]);
    let (g, h, i) = (m[8], m[9], m[10]);
    let determinant = determinant_of(matrix)?;
    Ok([
        (e * i - f * h) / determinant,
        (f * g - d * i) / determinant,
        (d * h - e * g) / determinant,
        0.0,
        (c * h - b * i) / determinant,
        (a * i - c * g) / determinant,
        (b * g - a * h) / determinant,
        0.0,
        (b * f - c * e) / determinant,
        (c * d - a * f) / determinant,
        (a * e - b * d) / determinant,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    .map(|value| value as f32))
}

fn vector(v: &Vector) -> [f32; 4] {
    [v.x, v.y, v.z, 0.0]
}

// Texture filtering as the frame's quality gives it: 0 reads only the
// finest level there is, 1 is trilinear, more is that many anisotropic
// samples. `blur` keeps it that many levels off the finest, while a level
// that has just arrived fades in.
fn sampler_for(device: &wgpu::Device, filtering: u32, blur: f32) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: if filtering == 0 { wgpu::MipmapFilterMode::Nearest } else { wgpu::MipmapFilterMode::Linear },
        lod_min_clamp: if filtering == 0 { 0.0 } else { blur },
        lod_max_clamp: if filtering == 0 { 0.0 } else { 32.0 },
        // WebGPU clamps maxAnisotropy to what it supports; wgpu takes 1..=16.
        // Written as a clamp: Rust 1.99's optimizer turned the equivalent
        // `if filtering > 1 { filtering.min(16) } else { 1 }` into
        // `filtering.min(16)`, an invalid 0 for unfiltered frames.
        anisotropy_clamp: filtering.clamp(1, 16) as u16,
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        ..Default::default()
    })
}

// Lays an inset's resolved image over the screen in its rectangle, toned as
// the scene is; its color is premultiplied by its coverage, so edges blend
// smoothly.
const COMPOSITE_SHADER: &str = r#"
struct Placement {
  rect: vec4f,
  // The screen's size, then how much of the image across and down the inset
  // drew.
  screen: vec4f,
};
@group(0) @binding(0) var<uniform> placement: Placement;
@group(0) @binding(1) var image: texture_2d<f32>;
@group(0) @binding(2) var image_sampler: sampler;
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};
@vertex fn vertex(@builtin(vertex_index) vertex: u32) -> Corner {
  let corner = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0))[vertex];
  let pixel = placement.rect.xy + corner * placement.rect.zw;
  var output: Corner;
  output.position = vec4f(pixel.x / placement.screen.x * 2.0 - 1.0, 1.0 - pixel.y / placement.screen.y * 2.0, 0.0, 1.0);
  output.uv = corner * placement.screen.zw;
  return output;
}
${tonedWgsl}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let light = textureSample(image, image_sampler, corner.uv);
  if (light.a <= 0.0) {
    return vec4f(0.0);
  }
  return vec4f(toned(light.rgb / light.a) * light.a, light.a);
}
"#;

// The loading veil: the frame's sky, from its horizon up, with a short arc
// of light turning round the middle. It is small, so it compiles at once,
// and lies over the first frames until what they draw has loaded, then
// clears.
const VEIL_SHADER: &str = r#"
struct Veil {
  zenith: vec4f,
  horizon: vec4f,
  // x: seconds since it rose, y: how much of it is left, z: the screen's
  // aspect.
  state: vec4f,
};
@group(0) @binding(0) var<uniform> veil: Veil;
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};
@vertex fn vertex(@builtin(vertex_index) index: u32) -> Corner {
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return Corner(vec4f(corner * 2.0 - 1.0, 0.0, 1.0), vec2f(corner.x, 1.0 - corner.y));
}
${tonedWgsl}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  var light = mix(veil.horizon.rgb, veil.zenith.rgb, smoothstep(0.0, 1.0, 1.0 - corner.uv.y));
  let p = (corner.uv - 0.5) * vec2f(veil.state.z, 1.0);
  let ring = 1.0 - smoothstep(0.0, 0.005, abs(length(p) - 0.035));
  let tail = fract(atan2(p.y, p.x) / 6.2831853 - veil.state.x * 0.8);
  light += vec3f(1.0, 0.94, 0.82) * ring * tail * tail * 1.4;
  let left = veil.state.y;
  return vec4f(toned(light) * left, left);
}
"#;

/// A premultiplied-alpha pipeline over the output, with the layout its
/// shader implies.
fn overlay_pipeline(device: &wgpu::Device, code: &str, format: wgpu::TextureFormat, label: &str) -> wgpu::RenderPipeline {
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
        vertex: wgpu::VertexState { module: &module, entry_point: Some("vertex"), compilation_options: Default::default(), buffers: &[] },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState { color: premultiplied, alpha: premultiplied }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
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
    stand_in: wgpu::Texture,
    group: wgpu::BindGroup,
    map: Option<wgpu::Texture>,
    /// Which map `map` is, to know when the scene's bind group must change.
    map_id: u64,
}

// Each inset's scene uniform and bind group, and the targets it draws into
// with 4 samples per pixel. Smooth insets use twice the display resolution;
// pixelated insets use the requested block size and nearest sampling. The
// targets are made with room to spare and an inset draws into their corner,
// so one that changes size every frame, like the arm reaching to the
// pointer, keeps its textures rather than making new ones each frame.
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

/// A packed texture whose first levels are still arriving: it shows once
/// its levels up to 128 texels across are there.
/// An asset made on this thread, published once its error scopes come back
/// empty.
struct Checking {
    slot: usize,
    revision: u64,
    early: bool,
    asset: GpuAsset,
    check: shader::Check,
}

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

struct Item<'f> {
    material: u32,
    transform: &'f Matrix,
    parameters: Option<[f32; 4]>,
    extra: Option<&'f [f32]>,
    shape: Shape,
    /// The inset it belongs to, if any.
    inset: Option<usize>,
}

/// One mesh, drawn alone; `inset` is the inset it belongs to, if any.
fn mesh_item(draw: &Draw, inset: Option<usize>) -> Item<'_> {
    Item {
        material: draw.material,
        transform: &draw.transform,
        parameters: draw.parameters,
        extra: draw.extra.as_deref(),
        shape: Shape::Mesh(draw.mesh),
        inset,
    }
}

/// The batch a mesh and material's draws join, and the parameters a draw
/// without its own takes.
type BatchChoice = (usize, [f32; 4]);

struct InstanceBatch<'f> {
    mesh: usize,
    material: u32,
    pipeline: wgpu::RenderPipeline,
    transforms: Vec<&'f Matrix>,
    parameters: Vec<[f32; 4]>,
    first: u32,
}

struct ReadyEntry {
    item: usize,
    offset: u32,
    group: wgpu::BindGroup,
    material: Arc<MaterialAsset>,
    shader: usize,
    /// The mesh's slot, for a mesh drawn alone.
    mesh: Option<usize>,
}

/// What drawing an item reads.
struct Drawing<'a, 'f> {
    assets: &'a AssetStore<GpuAsset>,
    items: &'a [Item<'f>],
    batches: &'a [InstanceBatch<'f>],
    solids: &'a [SolidMesh],
    instances: &'a wgpu::Buffer,
}

impl Drawing<'_, '_> {
    /// Draws `entry`'s item into `pass`, counting it into `stats` if given.
    fn draw(&self, pass: &mut wgpu::RenderPass<'_>, entry: &ReadyEntry, stats: Option<&mut RenderStats>) -> bool {
        match self.items[entry.item].shape {
            Shape::Mesh(_) => {
                let Some(GpuAsset::Mesh(mesh)) = entry.mesh.and_then(|slot| self.assets.current(slot)) else {
                    return false;
                };
                pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.count, 0, 0..1);
                if let Some(stats) = stats {
                    stats.draw_calls += 1;
                }
            }
            Shape::Batch(index) => {
                let batch = &self.batches[index];
                let Some(GpuAsset::Mesh(mesh)) = self.assets.current(batch.mesh) else { return false };
                let count = batch.transforms.len() as u32;
                pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                pass.set_vertex_buffer(1, self.instances.slice(..));
                pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.count, 0, batch.first..batch.first + count);
                if let Some(stats) = stats {
                    stats.draw_calls += 1;
                    stats.instanced_draw_calls += 1;
                    stats.instances += count;
                }
            }
            Shape::Solid(index) => {
                let solid = &self.solids[index];
                pass.set_vertex_buffer(0, solid.buffer.slice(..));
                pass.draw(0..solid.count, 0..1);
                if let Some(stats) = stats {
                    stats.draw_calls += 1;
                }
            }
        }
        true
    }

    fn shader(&self, entry: &ReadyEntry) -> &ShaderAsset {
        match self.assets.current(entry.shader) {
            Some(GpuAsset::Shader(shader)) => shader,
            _ => unreachable!("a ready entry's shader is loaded"),
        }
    }
}

pub(crate) struct State {
    device: wgpu::Device,
    queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    veil_seconds: Option<f32>,
    clock: crate::platform::Clock,
    // Dropped before the assets, so its threads stop streaming.
    loader: Loader,
    assets: AssetStore<GpuAsset>,
    context: PipelineContext,
    // The last frame's texture filtering and samples per pixel: those its
    // settings asked for, and those it drew with (which wait for the shaders).
    filtering: u32,
    samples_now: u32,
    samples_shown: u32,
    // Created on first use, so a renderer that never draws needs no pipelines.
    post: Option<Post>,
    mipmaps: Option<Mipmapper>,
    decals: Option<Decals>,
    sprites: Option<Sprites>,
    sprite_group_layout: wgpu::BindGroupLayout,
    inset_scenes: Vec<Option<InsetScene>>,
    inset_targets: Vec<Option<InsetTargets>>,
    inset_composite: Option<wgpu::RenderPipeline>,
    inset_sampler: wgpu::Sampler,
    inset_nearest_sampler: wgpu::Sampler,
    veil: Option<Veil>,
    // The scene uniform, the shadow map and its comparison sampler.
    scene_layout: wgpu::BindGroupLayout,
    draw_layout: wgpu::BindGroupLayout,
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
    // Samplers by how many levels off the finest they keep, in eighths.
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
    streaming: Vec<usize>,
    fading: Vec<usize>,
    last_time: Option<f64>,
    // Whether the catalog is being loaded ahead of need.
    preloading: bool,
    // Shaders the preload has sorted and asked for: one whose vertex stage
    // takes only builtins draws sprites; one that takes vertex attributes
    // draws meshes.
    sorted: HashSet<String>,
    // Solid meshes, rebuilt when the frame's solids change.
    solids_key: Vec<Solid>,
    solid_meshes: Vec<SolidMesh>,
    // Whether something this frame draws is still on its way.
    waiting: bool,
    frames: u64,
}

fn texture_views(texture: &wgpu::Texture, finest: u32) -> (wgpu::TextureView, wgpu::TextureView) {
    let levels = |format: Option<wgpu::TextureFormat>| wgpu::TextureViewDescriptor {
        format,
        base_mip_level: finest,
        mip_level_count: Some(texture.mip_level_count() - finest),
        ..Default::default()
    };
    let linear = match texture.format() {
        wgpu::TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8Unorm,
        format => format,
    };
    (texture.create_view(&levels(None)), texture.create_view(&levels(Some(linear))))
}

/// A texture with all its levels there.
fn whole(texture: wgpu::Texture, view_id: u64) -> TextureAsset {
    let (view, linear) = texture_views(&texture, 0);
    TextureAsset { texture, view, linear, view_id, resident: 0, shown: 0.0, seen: false, stream: None }
}

fn upload_level(queue: &wgpu::Queue, texture: &wgpu::Texture, level: &PackedLevel) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture, mip_level: level.level, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &level.pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(level.width * texel_bytes(texture.format())),
            rows_per_image: Some(level.height),
        },
        extent(level.width, level.height),
    );
}

// Mipmapped textures get every smaller level drawn after their upload.
fn texture_from(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mipmaps: &mut Option<Mipmapper>,
    (width, height): (u32, u32),
    pixels: &[u8],
    format: wgpu::TextureFormat,
    mipmapped: bool,
) -> Result<wgpu::Texture> {
    let limit = device.limits().max_texture_dimension_2d;
    ensure!(width <= limit && height <= limit, "texture exceeds device dimensions");
    let levels = if mipmapped { mip_levels(width, height) } else { 1 };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: extent(width, height),
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | if levels > 1 { wgpu::TextureUsages::RENDER_ATTACHMENT } else { wgpu::TextureUsages::empty() },
        // Reinterpret the same allocation as linear data for object-space normals.
        view_formats: if format == wgpu::TextureFormat::Rgba8UnormSrgb { &[wgpu::TextureFormat::Rgba8Unorm] } else { &[] },
    });
    queue.write_texture(
        texture.as_image_copy(),
        pixels,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width * texel_bytes(format)), rows_per_image: Some(height) },
        extent(width, height),
    );
    if levels > 1 {
        mipmaps.get_or_insert_with(|| Mipmapper::new(device)).generate(device, queue, &texture);
    }
    Ok(texture)
}

impl State {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, options: RendererOptions) -> Result<Self> {
        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            bind_group_layouts: &[Some(&scene_layout), Some(&draw_layout)],
            immediate_size: 0,
        });
        let sprite_group_layout = sprite_group_layout(&device);
        let sprite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sprites"),
            bind_group_layouts: &[Some(&sprite_group_layout)],
            immediate_size: 0,
        });
        let context = PipelineContext { device: device.clone(), layout, sprite_layout, format: options.format };
        let loader = Loader::new(options.assets.clone(), context.clone())?;
        let assets = AssetStore::new(loader.jobs.clone());
        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment);
        let stride = DRAW_BYTES.div_ceil(alignment) * alignment;
        let mut mipmaps = None;
        let mut built_in = |id: u64, size: (u32, u32), pixels: &[u8]| -> Result<TextureAsset> {
            let texture = texture_from(&device, &queue, &mut mipmaps, size, pixels, wgpu::TextureFormat::Rgba8UnormSrgb, false)?;
            Ok(whole(texture, id))
        };
        let white = built_in(1, (1, 1), &[255, 255, 255, 255])?;
        let flat_normal = built_in(2, (1, 1), &[128, 128, 255, 255])?;
        let transparent = built_in(3, (1, 1), &[0, 0, 0, 0])?;
        let checker = built_in(4, (2, 2), &[255, 0, 255, 255, 30, 30, 30, 255, 30, 30, 30, 255, 255, 0, 255, 255])?;
        let veil = options.veil.map(|_| {
            let pipeline = overlay_pipeline(&device, VEIL_SHADER, options.format, "veil");
            let uniform = uniform_buffer(&device, "veil", 48);
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("veil"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }],
            });
            Veil { pipeline, uniform, group, rose: None, clearing: None }
        });
        let linear_sampler = |filter: wgpu::FilterMode| {
            device.create_sampler(&wgpu::SamplerDescriptor { mag_filter: filter, min_filter: filter, ..Default::default() })
        };
        Ok(Self {
            format: options.format,
            veil_seconds: options.veil,
            clock: crate::platform::Clock::new(),
            loader,
            assets,
            context,
            filtering: 0,
            samples_now: 1,
            samples_shown: 1,
            post: None,
            mipmaps,
            decals: None,
            sprites: None,
            sprite_group_layout,
            inset_scenes: Vec::new(),
            inset_targets: Vec::new(),
            inset_composite: None,
            inset_sampler: linear_sampler(wgpu::FilterMode::Linear),
            inset_nearest_sampler: linear_sampler(wgpu::FilterMode::Nearest),
            veil,
            scene_uniform: uniform_buffer(&device, "scene", SCENE_BYTES),
            scene_layout,
            draw_layout,
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
            streaming: Vec::new(),
            fading: Vec::new(),
            last_time: None,
            preloading: false,
            sorted: HashSet::new(),
            solids_key: Vec::new(),
            solid_meshes: Vec::new(),
            waiting: false,
            frames: 0,
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
                Some(GpuAsset::Shader(shader)) => shader.compiling.len(),
                _ => 0,
            })
            .sum::<usize>();
        self.assets.in_flight() + self.pending.len() + self.checking.len() + self.streaming.len() + compiling
    }

    /// Whether nothing is streaming in or settling, and the veil is gone.
    pub fn quiet(&self) -> bool {
        self.streaming.is_empty() && self.fading.is_empty() && self.veil.is_none()
    }

    /// Loads the catalog's meshes and materials, and the materials' textures,
    /// ahead of need, behind whatever a frame asks for, so nothing pops in
    /// when first drawn.
    pub fn preload(&mut self, catalog: &Catalog) {
        self.preloading = true;
        for key in &catalog.meshes {
            if let Err(error) = self.assets.prefetch(AssetKind::Mesh, key) {
                self.reject(AssetKind::Mesh, key, error);
            }
        }
        for key in &catalog.materials {
            if let Err(error) = self.assets.prefetch(AssetKind::Material, key) {
                self.reject(AssetKind::Material, key, error);
            }
        }
    }

    /// Reports an asset that cannot even be asked for.
    fn reject(&mut self, kind: AssetKind, key: &str, error: anyhow::Error) {
        self.assets.events.push(GameEvent::AssetFailed {
            key: key.to_owned(),
            kind: kind.name().to_owned(),
            revision: 0,
            message: format!("{error:#}"),
        });
    }

    fn prefetch_shader(&mut self, key: &str) {
        if !self.sorted.insert(key.to_owned()) {
            return;
        }
        self.loader.jobs.push(Job::Sort { key: key.to_owned() }, Priority::Background);
    }

    // ----- Assets arriving from the loader -----

    /// Takes in whatever the loader has brought since the last frame.
    fn receive(&mut self, now: f64) {
        while let Ok(message) = self.loader.results.try_recv() {
            match message {
                Message::Loaded { slot, revision, early, result } => {
                    let prepared = result.and_then(|loaded| self.prepare(slot, revision, early, loaded));
                    match prepared {
                        // A packed texture finishes loading once its first levels are in.
                        Ok(None) => {}
                        // Not published until its allocation and upload have
                        // passed the error scopes (at once natively).
                        Ok(Some((asset, Some(mut check)))) => match shader::settled(&mut check) {
                            Some(Ok(())) => {
                                let previous = self.assets.finish(slot, revision, early, Ok(asset), now);
                                self.retire(previous);
                            }
                            Some(Err(message)) => {
                                let previous = self.assets.finish(slot, revision, early, Err(message), now);
                                self.retire(previous);
                            }
                            None => self.checking.push(Checking { slot, revision, early, asset, check }),
                        },
                        Ok(Some((asset, None))) => {
                            let previous = self.assets.finish(slot, revision, early, Ok(asset), now);
                            self.retire(previous);
                        }
                        Err(message) => {
                            let previous = self.assets.finish(slot, revision, early, Err(message), now);
                            self.retire(previous);
                        }
                    }
                }
                Message::Sorted { key, sprite } => {
                    let kind = if sprite { AssetKind::SpriteShader } else { AssetKind::Shader };
                    let _ = self.assets.prefetch(kind, &key);
                }
                Message::Compiled { slot, revision, samples, result } => {
                    if let Some(GpuAsset::Shader(shader)) = self.assets.current_mut(slot)
                        && shader.revision == revision
                    {
                        shader.compiling.remove(&samples);
                        if let Ok(pipelines) = result {
                            shader.made.insert(samples, pipelines);
                        }
                    }
                }
                Message::Inset { slot, revision, result } => {
                    if let Some(GpuAsset::Shader(shader)) = self.assets.current_mut(slot)
                        && shader.revision == revision
                        && let Ok(pipeline) = result
                    {
                        shader.inset = Some(pipeline);
                    }
                }
            }
        }
        // Assets whose error scopes came back since the last frame.
        let mut index = 0;
        while index < self.checking.len() {
            let Some(result) = shader::settled(&mut self.checking[index].check) else {
                index += 1;
                continue;
            };
            let Checking { slot, revision, early, asset, .. } = self.checking.swap_remove(index);
            let previous = self.assets.finish(slot, revision, early, result.map(|()| asset), now);
            self.retire(previous);
        }
        self.receive_levels(now);
    }

    /// Makes what the loader brought into a GPU asset; None for a packed
    /// texture, which shows once its first levels are in.
    fn prepare(
        &mut self,
        slot: usize,
        revision: u64,
        early: bool,
        loaded: Loaded,
    ) -> Result<Option<(GpuAsset, Option<shader::Check>)>, String> {
        let device = self.device.clone();
        match loaded {
            Loaded::Mesh(mesh) => {
                let vertex_bytes: &[u8] = bytemuck::cast_slice(&mesh.vertices);
                let index_bytes: &[u8] = bytemuck::cast_slice(&mesh.indices);
                if vertex_bytes.len().max(index_bytes.len()) as u64 > device.limits().max_buffer_size {
                    return Err("mesh exceeds device buffer size".into());
                }
                let buffer = |bytes: &[u8], usage: wgpu::BufferUsages| {
                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: bytes.len() as u64,
                        usage: usage | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    self.queue.write_buffer(&buffer, 0, bytes);
                    buffer
                };
                let (asset, check) = shader::scoped(&device, || {
                    GpuAsset::Mesh(MeshAsset {
                        vertices: buffer(vertex_bytes, wgpu::BufferUsages::VERTEX),
                        indices: buffer(index_bytes, wgpu::BufferUsages::INDEX),
                        count: mesh.indices.len() as u32,
                    })
                });
                Ok(Some((asset, Some(check))))
            }
            Loaded::Material(material) => {
                if self.preloading {
                    for key in [&material.texture, &material.normal_texture, &material.layer_texture].into_iter().flatten() {
                        let _ = self.assets.prefetch(AssetKind::Texture, key);
                    }
                    self.prefetch_shader(&material.shader);
                }
                let state = material.state_key();
                Ok(Some((GpuAsset::Material(Arc::new(MaterialAsset { material, state })), None)))
            }
            Loaded::Shader { module, instancing, samples, pipelines } => Ok(Some((
                GpuAsset::Shader(ShaderAsset {
                    revision,
                    module,
                    instancing,
                    made: HashMap::from([(samples, pipelines)]),
                    compiling: HashSet::new(),
                    inset: None,
                    shadow: [None, None],
                }),
                None,
            ))),
            Loaded::SpriteShader(pipeline) => Ok(Some((GpuAsset::SpriteShader(pipeline), None))),
            Loaded::Image(image) => {
                let id = self.view_id();
                let (queue, mipmaps) = (&self.queue, &mut self.mipmaps);
                let (texture, check) = shader::scoped(&device, || {
                    texture_from(
                        &device,
                        queue,
                        mipmaps,
                        (image.width, image.height),
                        &image.pixels,
                        wgpu::TextureFormat::Rgba8UnormSrgb,
                        true,
                    )
                });
                let texture = texture.map_err(|error| format!("{error:#}"))?;
                Ok(Some((GpuAsset::Texture(whole(texture, id)), Some(check))))
            }
            Loaded::Packed { header, levels } => {
                let limit = device.limits().max_texture_dimension_2d;
                if header.width > limit || header.height > limit {
                    return Err("texture exceeds device dimensions".into());
                }
                let (texture, mut check) = shader::scoped(&device, || {
                    device.create_texture(&wgpu::TextureDescriptor {
                        label: None,
                        size: extent(header.width, header.height),
                        mip_level_count: header.levels,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: header.format,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        // Reinterpret the same allocation as linear data for normal maps.
                        view_formats: if header.format == wgpu::TextureFormat::Rgba8UnormSrgb {
                            &[wgpu::TextureFormat::Rgba8Unorm]
                        } else {
                            &[]
                        },
                    })
                });
                let check = match shader::settled(&mut check) {
                    Some(result) => result.map(|()| None)?,
                    None => Some(check),
                };
                self.pending.push(PendingTexture { check, slot, revision, early, texture, header, finest: header.levels, levels });
                Ok(None)
            }
        }
    }

    /// Uploads the levels that have arrived: a pending texture's first ones,
    /// until it shows, and a shown one's finer ones, which fade in.
    fn receive_levels(&mut self, now: f64) {
        let mut index = 0;
        while index < self.pending.len() {
            let pending = &mut self.pending[index];
            let shown_from = pending.header.levels.saturating_sub(SHOWN_LEVELS);
            let mut outcome: Option<Result<(), String>> = None;
            // Its levels wait until its allocation has passed the error
            // scopes, as the TS awaits them first (at once natively).
            if let Some(check) = &mut pending.check {
                match shader::settled(check) {
                    None => {
                        index += 1;
                        continue;
                    }
                    Some(Ok(())) => pending.check = None,
                    Some(Err(message)) => outcome = Some(Err(message)),
                }
            }
            while outcome.is_none() && pending.finest > shown_from {
                match pending.levels.try_recv() {
                    Ok(LevelMessage::Level(level)) => {
                        upload_level(&self.queue, &pending.texture, &level);
                        pending.finest = level.level;
                    }
                    Ok(LevelMessage::Done) | Err(TryRecvError::Disconnected) => {
                        outcome = Some(Err("packed texture ends early".into()));
                        break;
                    }
                    Ok(LevelMessage::Failed(message)) => {
                        outcome = Some(Err(message));
                        break;
                    }
                    Err(TryRecvError::Empty) => break,
                }
            }
            if outcome.is_none() && pending.finest <= shown_from {
                outcome = Some(Ok(()));
            }
            let Some(outcome) = outcome else {
                index += 1;
                continue;
            };
            let pending = self.pending.swap_remove(index);
            match outcome {
                Err(message) => {
                    let previous = self.assets.finish(pending.slot, pending.revision, pending.early, Err(message), now);
                    self.retire(previous);
                }
                Ok(()) => {
                    let (view, linear) = texture_views(&pending.texture, pending.finest);
                    let asset = TextureAsset {
                        texture: pending.texture,
                        view,
                        linear,
                        view_id: self.view_id(),
                        resident: pending.finest,
                        shown: f64::from(pending.finest),
                        seen: false,
                        stream: Some(pending.levels),
                    };
                    let current = self.assets.slots[pending.slot].revision == pending.revision;
                    let previous =
                        self.assets.finish(pending.slot, pending.revision, pending.early, Ok(GpuAsset::Texture(asset)), now);
                    self.retire(previous);
                    // The rest, in the background: each finer level joins the
                    // views and fades in. (A revision it replaced may still be
                    // listed; its stream went with it.)
                    if current && !self.streaming.contains(&pending.slot) {
                        self.streaming.push(pending.slot);
                    }
                }
            }
        }
        let mut index = 0;
        while index < self.streaming.len() {
            let slot = self.streaming[index];
            let key = self.assets.slots[slot].key.clone();
            let mut new_views = false;
            let mut ended = false;
            let mut old_views = None;
            if let Some(GpuAsset::Texture(texture)) = self.assets.slots[slot].current.as_mut() {
                old_views = Some(texture.view_id);
                loop {
                    let Some(levels) = &texture.stream else {
                        ended = true;
                        break;
                    };
                    match levels.try_recv() {
                        Ok(LevelMessage::Level(level)) => {
                            upload_level(&self.queue, &texture.texture, &level);
                            (texture.view, texture.linear) = texture_views(&texture.texture, level.level);
                            texture.resident = level.level;
                            new_views = true;
                            // A level nobody saw the coarser one before needs
                            // no fading in.
                            if !texture.seen {
                                texture.shown = f64::from(level.level);
                            }
                        }
                        Ok(LevelMessage::Done) | Err(TryRecvError::Disconnected) => {
                            texture.stream = None;
                        }
                        Ok(LevelMessage::Failed(message)) => {
                            crate::platform::warn(&format!("texture {key}: {message}"));
                            texture.stream = None;
                        }
                        Err(TryRecvError::Empty) => break,
                    }
                }
            } else {
                ended = true;
            }
            if new_views {
                // Groups of the coarser views are not drawn with again.
                if let Some(old) = old_views {
                    self.forget_views(old);
                }
                let id = self.view_id();
                if let Some(GpuAsset::Texture(texture)) = self.assets.slots[slot].current.as_mut() {
                    texture.view_id = id;
                    if texture.shown > f64::from(texture.resident) && !self.fading.contains(&slot) {
                        self.fading.push(slot);
                    }
                }
            }
            if ended {
                self.streaming.swap_remove(index);
            } else {
                index += 1;
            }
        }
    }

    /// Lets go of what depended on an asset a new revision replaced. Shader,
    /// sprite shader, mesh and material assets carry their own pipelines and
    /// buffers, and frames look them up afresh, so dropping them is enough;
    /// a texture's views are also held by cached bind groups.
    fn retire(&mut self, previous: Option<GpuAsset>) {
        if let Some(GpuAsset::Texture(texture)) = previous {
            self.forget_views(texture.view_id);
        }
    }

    /// Drops the cached draw bind groups that bind views `id`.
    fn forget_views(&mut self, id: u64) {
        self.groups.retain(|&(texture, normal, outline, _), _| texture != id && normal != id && outline != id);
    }

    /// Loads the changed `keys` again (asset_store.ts's `reload`): every
    /// asset of those keys that was loaded or asked for gets a new revision,
    /// drawn once it is ready; until then, or if it fails, the last one
    /// draws on.
    pub fn reload(&mut self, keys: &[String]) {
        self.assets.reload(keys);
    }

    // ----- Looking assets up for a frame -----

    /// The asset's slot, asking for it if it is not loaded yet; notes when a
    /// frame waits on it.
    fn resolve(&mut self, kind: AssetKind, key: &str) -> Result<usize> {
        let slot = self.assets.get(kind, key)?;
        if self.assets.current(slot).is_none() && self.assets.loading(slot) {
            self.waiting = true;
        }
        Ok(slot)
    }

    fn material(&mut self, catalog: &Catalog, index: u32) -> Result<Option<Arc<MaterialAsset>>> {
        let slot = self.resolve(AssetKind::Material, catalog.material(index))?;
        Ok(match self.assets.current(slot) {
            Some(GpuAsset::Material(material)) => Some(material.clone()),
            _ => None,
        })
    }

    /// The shader's slot, if it is loaded.
    fn shader(&mut self, key: &str) -> Result<Option<usize>> {
        let slot = self.resolve(AssetKind::Shader, key)?;
        Ok(matches!(self.assets.current(slot), Some(GpuAsset::Shader(_))).then_some(slot))
    }

    fn texture(&self, texture: TextureRef) -> &TextureAsset {
        match texture {
            TextureRef::Slot(slot) => match self.assets.current(slot) {
                Some(GpuAsset::Texture(texture)) => texture,
                _ => unreachable!("a texture reference names a loaded texture"),
            },
            TextureRef::White => &self.white,
            TextureRef::FlatNormal => &self.flat_normal,
            TextureRef::Transparent => &self.transparent,
            TextureRef::Checker => &self.checker,
            TextureRef::Field(index) => &self.fields[index].1,
        }
    }

    /// The loaded texture of `key`, if any, asking for it if need be.
    fn loaded_texture(&mut self, key: &Option<String>) -> Result<Option<usize>> {
        let Some(key) = key else { return Ok(None) };
        let slot = self.resolve(AssetKind::Texture, key)?;
        Ok(matches!(self.assets.current(slot), Some(GpuAsset::Texture(_))).then_some(slot))
    }

    /// Makes sure the shader has pipelines for `samples` per pixel, compiling
    /// them now if the background has not.
    fn ensure_pipelines(&mut self, slot: usize, samples: u32) {
        let context = &self.context;
        if let Some(GpuAsset::Shader(shader)) = self.assets.current_mut(slot)
            && !shader.made.contains_key(&samples)
        {
            let pipelines = shader::pipelines(context, &shader.module, shader.instancing, samples);
            shader.made.insert(samples, pipelines);
        }
    }

    fn pipelines(&self, slot: usize, samples: u32) -> &Pipelines {
        match self.assets.current(slot) {
            Some(GpuAsset::Shader(shader)) => &shader.made[&samples],
            _ => unreachable!("pipelines of a loaded shader"),
        }
    }

    fn sampler_at(&mut self, blur: f64) -> (i32, wgpu::Sampler) {
        let key = if self.filtering == 0 { 0 } else { (blur * 8.0).ceil() as i32 };
        let (device, filtering) = (&self.device, self.filtering);
        let sampler = self.samplers.entry(key).or_insert_with(|| sampler_for(device, filtering, key as f32 / 8.0));
        (key, sampler.clone())
    }

    fn shadow_resources(&mut self) -> &mut Shadows {
        let device = &self.device;
        let scene_layout = &self.scene_layout;
        self.shadows.get_or_insert_with(|| {
            let uniform = uniform_buffer(device, "shadow scene", SCENE_BYTES);
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                compare: Some(wgpu::CompareFunction::Less),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            let stand_in = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("shadow stand-in"),
                size: extent(1, 1),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow scene"),
                layout: scene_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&stand_in.create_view(&Default::default())),
                    },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                ],
            });
            Shadows { uniform, sampler, stand_in, group, map: None, map_id: 0 }
        })
    }

    /// The textures of a draw's material, or None while one is on its way:
    /// nothing is drawn half-dressed. Also whether the normal and layer
    /// textures loaded, as the draw's settings tell the shader.
    fn dress(&mut self, material: &Material, field: Option<usize>) -> Result<Option<([TextureRef; 3], f32, f32)>> {
        let loaded_texture = self.loaded_texture(&material.texture)?;
        let loaded_normal = self.loaded_texture(&material.normal_texture)?;
        let loaded_outline = self.loaded_texture(&material.layer_texture)?;
        // A texture on its way holds the draw back; one that failed shows as
        // the checker.
        if [&material.texture, &material.normal_texture, &material.layer_texture]
            .into_iter()
            .flatten()
            .any(|key| self.assets.loading_key(AssetKind::Texture, key))
        {
            return Ok(None);
        }
        let mut texture = match (&material.texture, loaded_texture) {
            (None, _) => TextureRef::White,
            (Some(_), Some(slot)) => TextureRef::Slot(slot),
            (Some(_), None) => TextureRef::Checker,
        };
        let mut normal = match field {
            Some(field) => TextureRef::Field(field),
            None => loaded_normal.map_or(TextureRef::FlatNormal, TextureRef::Slot),
        };
        let mut outline = loaded_outline.map_or(TextureRef::Transparent, TextureRef::Slot);
        // Index and palette must become usable together: an ordinary fallback
        // texture must never be interpreted as palette indices or oct normals.
        let indexed = material.indexed;
        let packed_ready = indexed
            && loaded_texture.is_some()
            && loaded_normal.is_some()
            && loaded_outline.is_some()
            && {
                let (index, normals, palette) =
                    (&self.texture(texture).texture, &self.texture(normal).texture, &self.texture(outline).texture);
                index.format() == wgpu::TextureFormat::R8Unorm
                    && normals.format() == wgpu::TextureFormat::Rg8Unorm
                    && palette.format() == wgpu::TextureFormat::Rgba8UnormSrgb
                    && palette.width() == 256
                    && palette.height() == 2
                    && index.width() == normals.width()
                    && index.height() == normals.height()
            };
        if indexed && !packed_ready {
            texture = TextureRef::Checker;
            normal = TextureRef::FlatNormal;
            outline = TextureRef::Transparent;
        }
        let normal_setting = if indexed { if packed_ready { 1.0 } else { 0.0 } } else if loaded_normal.is_some() { 1.0 } else { 0.0 };
        let outline_setting = if indexed { if packed_ready { 2.0 } else { 0.0 } } else if loaded_outline.is_some() { 1.0 } else { 0.0 };
        Ok(Some(([texture, normal, outline], normal_setting, outline_setting)))
    }

    /// The draw bind group for these textures, with the sampler their
    /// settling detail asks for.
    fn draw_group(&mut self, textures: [TextureRef; 3]) -> wgpu::BindGroup {
        for texture in textures {
            if let TextureRef::Slot(slot) = texture
                && let Some(GpuAsset::Texture(texture)) = self.assets.current_mut(slot)
            {
                texture.seen = true;
            }
        }
        let blur = textures
            .iter()
            .map(|&texture| {
                let texture = self.texture(texture);
                texture.shown - f64::from(texture.resident)
            })
            .fold(f64::NEG_INFINITY, f64::max);
        let (sampler_key, sampler) = self.sampler_at(blur);
        // Every texture's resident levels participate in cache identity.
        // Reloading any layer, or a finer level arriving, replaces its group.
        let [texture_id, normal_id, outline_id] = textures.map(|texture| self.texture(texture).view_id);
        let key = (texture_id, normal_id, outline_id, sampler_key);
        let frame = self.frames;
        if let Some((group, used)) = self.groups.get_mut(&key) {
            *used = frame;
            return group.clone();
        }
        let [texture, normal, outline] = textures.map(|texture| self.texture(texture));
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.draw_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.draw_uniform,
                        offset: 0,
                        size: wgpu::BufferSize::new(DRAW_BYTES),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&texture.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&normal.linear) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&outline.view) },
            ],
        });
        self.groups.insert(key, (group.clone(), frame));
        group
    }

    fn update_solids(&mut self, solids: &[Solid]) {
        if self.solids_key == solids {
            return;
        }
        self.solids_key = solids.to_vec();
        let device = &self.device;
        self.solid_meshes = mesh_solids(solids)
            .into_iter()
            .filter(|batch| !batch.vertices.is_empty())
            .map(|batch| {
                let bytes: &[u8] = bytemuck::cast_slice(&batch.vertices);
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("solids"),
                    size: bytes.len() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.queue.write_buffer(&buffer, 0, bytes);
                SolidMesh {
                    material: batch.material,
                    buffer,
                    count: (batch.vertices.len() / crate::asset_store::MESH_VERTEX_FLOATS) as u32,
                }
            })
            .collect();
    }

    /// The inset's targets, made anew when its size or look changes.
    fn targets_for(&mut self, index: usize, width: u32, height: u32, pixelated: bool) -> &InsetTargets {
        if self.inset_targets.len() <= index {
            self.inset_targets.resize_with(index + 1, || None);
        }
        let limit = self.device.limits().max_texture_dimension_2d;
        let roomy = |size: u32| limit.min((f64::from(size) * 1.25 / 256.0).ceil() as u32 * 256);
        // Kept while they hold the inset and are not four times what it would
        // be given afresh.
        let fits = self.inset_targets[index].as_ref().is_some_and(|found| {
            found.pixelated == pixelated
                && found.width >= width
                && found.height >= height
                && u64::from(found.width) * u64::from(found.height) <= 4 * u64::from(roomy(width)) * u64::from(roomy(height))
        });
        if !fits {
            let (width, height) = (roomy(width), roomy(height));
            let device = &self.device;
            let composite = self
                .inset_composite
                .get_or_insert_with(|| overlay_pipeline(device, COMPOSITE_SHADER, self.format, "inset composite"));
            let texture = |samples: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("inset"),
                        size: extent(width, height),
                        mip_level_count: 1,
                        sample_count: samples,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage,
                        view_formats: &[],
                    })
                    .create_view(&Default::default())
            };
            let resolved = texture(
                1,
                SCENE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            let placement = uniform_buffer(device, "inset placement", 32);
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inset"),
                layout: &composite.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: placement.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&resolved) },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(if pixelated {
                            &self.inset_nearest_sampler
                        } else {
                            &self.inset_sampler
                        }),
                    },
                ],
            });
            self.inset_targets[index] = Some(InsetTargets {
                width,
                height,
                pixelated,
                color: texture(INSET_SAMPLES, SCENE_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT),
                depth: texture(INSET_SAMPLES, wgpu::TextureFormat::Depth24PlusStencil8, wgpu::TextureUsages::RENDER_ATTACHMENT),
                resolved,
                placement,
                group,
            });
        }
        self.inset_targets[index].as_ref().unwrap()
    }

    /// Writes a patch into its material's field, made (zeroed) when first
    /// patched or when its size changes.
    fn write_field(&mut self, material: &str, patch: &FieldPatch) {
        let (width, height) = (patch.width.max(1), patch.height.max(1));
        let found = self.field_ids.get(material).copied();
        let index = match found {
            Some(index) if self.fields[index].1.texture.width() == width && self.fields[index].1.texture.height() == height => {
                index
            }
            _ => {
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("field"),
                    size: extent(width, height),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let field = whole(texture, self.view_id());
                match found {
                    Some(index) => {
                        let old = std::mem::replace(&mut self.fields[index].1, field);
                        self.forget_views(old.view_id);
                        index
                    }
                    None => {
                        self.fields.push((material.to_owned(), field));
                        self.field_ids.insert(material.to_owned(), self.fields.len() - 1);
                        self.fields.len() - 1
                    }
                }
            }
        };
        if patch.columns == 0 || patch.rows == 0 {
            return;
        }
        let halves: Vec<u16> = patch.values.iter().map(|&value| half_of(value)).collect();
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.fields[index].1.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: patch.x, y: patch.z, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&halves),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(patch.columns * 2), rows_per_image: Some(patch.rows) },
            extent(patch.columns, patch.rows),
        );
    }

    // ----- A frame -----

    pub fn submit(&mut self, frame: &RenderFrame, catalog: &Catalog, target: &mut dyn RenderTarget) -> Result<()> {
        validate_frame(frame)?;
        // Every asset the frame names is in the catalog and stays inside the
        // asset root.
        let mut validated_meshes = HashSet::new();
        let mut validated_materials = HashSet::new();
        let mut check_mesh = |index: u32| -> Result<()> {
            if validated_meshes.insert(index) {
                let key = catalog.meshes.get(index as usize).ok_or_else(|| anyhow!("draw must name a mesh and material"))?;
                crate::asset_store::validate_asset_key(key)?;
            }
            Ok(())
        };
        let mut check_material = |index: u32| -> Result<()> {
            if validated_materials.insert(index) {
                let key = catalog.materials.get(index as usize).ok_or_else(|| anyhow!("draw must name a mesh and material"))?;
                crate::asset_store::validate_asset_key(key)?;
            }
            Ok(())
        };
        for draw in frame.draws.iter().chain(frame.insets.iter().flat_map(|inset| &inset.draws)) {
            check_mesh(draw.mesh)?;
            check_material(draw.material)?;
        }
        for face in frame.solids.iter().flat_map(|solid| &solid.faces) {
            check_material(face.material)?;
        }
        for material in frame.decals.iter().map(|decal| decal.material).chain(frame.sprites.iter().map(|sprite| sprite.material)) {
            check_material(material)?;
        }
        for patch in &frame.fields {
            check_material(patch.material)?;
            self.write_field(catalog.material(patch.material), patch);
        }

        // What the loader brought, and resources the GPU is done with.
        let _ = self.device.poll(wgpu::PollType::Poll);
        let now = self.now();
        self.receive(now);
        self.frames += 1;
        self.waiting = false;
        if self.frames.is_multiple_of(GROUP_LIFETIME) {
            let oldest = self.frames - GROUP_LIFETIME;
            self.groups.retain(|_, (_, used)| *used >= oldest);
        }
        // Detail that arrived since the last frame settles in.
        let step = (now - self.last_time.unwrap_or(now)).min(0.1);
        self.last_time = Some(now);
        let assets = &mut self.assets;
        self.fading.retain(|&slot| {
            let Some(GpuAsset::Texture(texture)) = assets.current_mut(slot) else { return false };
            let resident = f64::from(texture.resident);
            texture.shown = resident + (texture.shown - resident) * (-step / SHARPENING).exp();
            if texture.shown - resident < 0.01 {
                texture.shown = resident;
                return false;
            }
            true
        });
        if self.post.is_none() {
            self.post = Some(Post::new(&self.device, &self.queue, self.format));
        }
        if frame.quality.filtering != self.filtering {
            self.filtering = frame.quality.filtering;
            self.samplers.clear();
            // Draw bind groups hold the sampler.
            self.groups.clear();
        }
        // A new sample count waits until each shader this frame draws with has
        // compiled its pipelines for it in the background; until then the
        // frame keeps the count it had.
        self.samples_now = frame.quality.msaa;
        self.loader.samples_now.store(self.samples_now, Ordering::Relaxed);
        if self.samples_now != self.samples_shown {
            let mut compiled = true;
            let mut materials: Vec<u32> = frame
                .draws
                .iter()
                .map(|draw| draw.material)
                .chain(frame.solids.iter().flat_map(|solid| solid.faces.iter().map(|face| face.material)))
                .collect();
            materials.sort_unstable();
            materials.dedup();
            for index in materials {
                let Some(material) = self.material(catalog, index)? else { continue };
                let Some(slot) = self.shader(&material.material.shader)? else { continue };
                let samples = self.samples_now;
                let Some(GpuAsset::Shader(shader)) = self.assets.current_mut(slot) else { continue };
                if shader.made.contains_key(&samples) {
                    continue;
                }
                // Compiles its pipelines for `samples` per pixel in the background.
                if shader.compiling.insert(samples) {
                    self.loader.jobs.push(
                        Job::Compile {
                            slot,
                            revision: shader.revision,
                            samples,
                            module: shader.module.clone(),
                            instancing: shader.instancing,
                        },
                        Priority::Urgent,
                    );
                }
                compiled = false;
            }
            if compiled {
                self.samples_shown = self.samples_now;
            } else {
                self.waiting = true;
            }
        }
        let quality = pen_protocol::Quality { msaa: self.samples_shown, ..frame.quality };
        let targets = self.post.as_mut().unwrap().begin(frame, quality, target.size());
        let samples = targets.samples;
        self.update_solids(&frame.solids);
        let mut stats = RenderStats::default();

        // Only depth-writing objects can be regrouped freely. Transparent
        // objects retain their global back-to-front order below.
        let mut batches: Vec<InstanceBatch<'_>> = Vec::new();
        let mut batch_choices: HashMap<(u32, u32), Option<BatchChoice>> = HashMap::new();
        let mut parameter_batches: HashMap<(u32, String), usize> = HashMap::new();
        let mut regular: Vec<&Draw> = Vec::new();
        for draw in &frame.draws {
            // Its own uniform data: drawn alone.
            if draw.extra.as_ref().is_some_and(|extra| !extra.is_empty()) {
                regular.push(draw);
                continue;
            }
            if let Some(choice) = batch_choices.get(&(draw.mesh, draw.material)) {
                match choice {
                    Some((batch, parameters)) => {
                        let batch = &mut batches[*batch];
                        batch.transforms.push(&draw.transform);
                        batch.parameters.push(draw.parameters.unwrap_or(*parameters));
                    }
                    None => regular.push(draw),
                }
                continue;
            }
            let material = self.material(catalog, draw.material)?;
            let shader = match &material {
                Some(material) => self.shader(&material.material.shader)?,
                None => None,
            };
            let mesh = self.resolve(AssetKind::Mesh, catalog.mesh(draw.mesh))?;
            let mesh_loaded = matches!(self.assets.current(mesh), Some(GpuAsset::Mesh(_)));
            let instancing = shader.is_some_and(|slot| matches!(self.assets.current(slot), Some(GpuAsset::Shader(shader)) if shader.instancing));
            match (material, shader) {
                (Some(material), Some(shader)) if material.material.opaque() && instancing && mesh_loaded => {
                    // All render state must match; only the explicit instance
                    // parameters may differ. This lets animation poses share
                    // one opaque draw.
                    let key = material.material.instance_parameters.map(|_| (draw.mesh, material.state.clone()));
                    let found = key.as_ref().and_then(|key| parameter_batches.get(key).copied());
                    let batch = match found {
                        Some(batch) => batch,
                        None => {
                            self.ensure_pipelines(shader, samples);
                            let pipeline = self.pipelines(shader, samples).instanced.clone().unwrap();
                            batches.push(InstanceBatch {
                                mesh,
                                material: draw.material,
                                pipeline,
                                transforms: Vec::new(),
                                parameters: Vec::new(),
                                first: 0,
                            });
                            if let Some(key) = key {
                                parameter_batches.insert(key, batches.len() - 1);
                            }
                            batches.len() - 1
                        }
                    };
                    let parameters = material.material.instance_parameters.map_or([0.0; 4], |values| values.map(|value| value as f32));
                    batches[batch].transforms.push(&draw.transform);
                    batches[batch].parameters.push(draw.parameters.unwrap_or(parameters));
                    batch_choices.insert((draw.mesh, draw.material), Some((batch, parameters)));
                }
                _ => {
                    batch_choices.insert((draw.mesh, draw.material), None);
                    regular.push(draw);
                }
            }
        }
        let instance_count: usize = batches.iter().map(|batch| batch.transforms.len()).sum();
        if instance_count > self.instance_capacity {
            self.instance_capacity = instance_count.next_power_of_two();
            ensure!(
                self.instance_capacity as u64 * 80 <= self.device.limits().max_buffer_size,
                "instance list exceeds device buffer size"
            );
            self.instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instances"),
                size: self.instance_capacity as u64 * 80,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_data = vec![0.0; self.instance_capacity * 20];
        }
        let mut next_instance = 0;
        for batch in &mut batches {
            batch.first = next_instance as u32;
            for (transform, parameters) in batch.transforms.iter().zip(&batch.parameters) {
                self.instance_data[next_instance * 20..next_instance * 20 + 16].copy_from_slice(*transform);
                self.instance_data[next_instance * 20 + 16..next_instance * 20 + 20].copy_from_slice(parameters);
                next_instance += 1;
            }
        }
        // Mesh draws, then one draw per solid batch in world space, then the
        // insets' draws.
        let mut items: Vec<Item<'_>> = regular.iter().map(|draw| mesh_item(draw, None)).collect();
        items.extend(batches.iter().enumerate().map(|(index, batch)| Item {
            material: batch.material,
            transform: &IDENTITY,
            parameters: None,
            extra: None,
            shape: Shape::Batch(index),
            inset: None,
        }));
        items.extend(self.solid_meshes.iter().enumerate().map(|(index, solid)| Item {
            material: solid.material,
            transform: &IDENTITY,
            parameters: None,
            extra: None,
            shape: Shape::Solid(index),
            inset: None,
        }));
        for (index, inset) in frame.insets.iter().enumerate() {
            items.extend(inset.draws.iter().map(|draw| mesh_item(draw, Some(index))));
        }
        if instance_count > 0 {
            self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&self.instance_data[..instance_count * 20]));
        }
        if items.len() > self.capacity {
            self.capacity = items.len().next_power_of_two();
            ensure!(
                self.capacity as u64 * self.stride <= self.device.limits().max_buffer_size,
                "draw list exceeds device buffer size"
            );
            self.draw_uniform = uniform_buffer(&self.device, "draws", self.capacity as u64 * self.stride);
            self.groups.clear();
            self.uniform_data = vec![0.0; self.capacity * (self.stride / 4) as usize];
        }
        // The game fits the shadow map's view; 0 texels for none.
        let shadow_texels = frame.shadow.texels;
        let sun = (shadow_texels > 0).then_some(&frame.shadow);
        let mut lighting = Vec::with_capacity(12);
        lighting.extend(vector(&frame.ambient));
        lighting.extend(vector(&frame.light_direction));
        lighting.extend(vector(&frame.light_color));
        let sun_view_projection = sun.map_or(IDENTITY, |sun| multiply(&sun.projection, &sun.view));
        // The sky the scene sits under, which lights it from all round and
        // which glossy surfaces mirror, and where the eye is, seen by `view`.
        let heavens = |view: &Matrix| {
            let eye = invert(view);
            let mut data = Vec::with_capacity(16);
            data.extend(vector(&frame.sky.zenith));
            data.extend(vector(&frame.sky.horizon));
            data.extend(vector(&frame.sky.ground));
            data.extend([eye[12], eye[13], eye[14], 1.0]);
            data
        };
        let mut scene = Vec::with_capacity(80);
        scene.extend(frame.view);
        scene.extend(frame.projection);
        scene.extend(&lighting);
        scene.extend(sun_view_projection);
        scene.extend([
            if sun.is_some() { 1.0 } else { 0.0 },
            1.0 / shadow_texels.max(1) as f32,
            sun.map_or(0.0, |sun| sun.texel * 1.5),
            0.0,
        ]);
        scene.extend(heavens(&frame.view));
        self.queue.write_buffer(&self.scene_uniform, 0, bytemuck::cast_slice(&scene));
        // The shadow pass sees the scene from the light.
        let mut shadow_map: Option<wgpu::TextureView> = None;
        if let Some(sun) = sun {
            let device = self.device.clone();
            let queue = self.queue.clone();
            let resources = self.shadow_resources();
            if resources.map.as_ref().is_none_or(|map| map.width() != shadow_texels) {
                resources.map = Some(device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("shadow map"),
                    size: extent(shadow_texels, shadow_texels),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                }));
                resources.map_id += 1;
            }
            shadow_map = resources.map.as_ref().map(|map| map.create_view(&Default::default()));
            let mut data = sun.view.to_vec();
            data.extend(sun.projection);
            data.extend(&lighting);
            queue.write_buffer(&resources.uniform, 0, bytemuck::cast_slice(&data));
        }
        let bound_map = if shadow_map.is_some() { self.shadow_resources().map_id } else { 0 };
        if self.scene_group.as_ref().is_none_or(|(_, map)| *map != bound_map) {
            let resources = self.shadow_resources();
            let view = match (bound_map, &resources.map) {
                (0, _) | (_, None) => resources.stand_in.create_view(&Default::default()),
                (_, Some(map)) => map.create_view(&Default::default()),
            };
            let sampler = resources.sampler.clone();
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene"),
                layout: &self.scene_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.scene_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                ],
            });
            self.scene_group = Some((group, bound_map));
        }
        // Resolve each item's assets and write its uniforms once.
        let floats = (self.stride / 4) as usize;
        let mut ready: Vec<ReadyEntry> = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let Some(material) = self.material(catalog, item.material)? else { continue };
            let Some(shader) = self.shader(&material.material.shader)? else { continue };
            let field = self.field_ids.get(catalog.material(item.material)).copied();
            let Some((textures, normal_setting, outline_setting)) = self.dress(&material.material, field)? else { continue };
            let group = self.draw_group(textures);
            let at = index * floats;
            let data = &mut self.uniform_data[at..at + floats];
            let m = &material.material;
            data[..16].copy_from_slice(item.transform);
            data[16..32].copy_from_slice(&normal_matrix(item.transform)?);
            data[32..36].copy_from_slice(&m.color.map(|channel| channel as f32));
            data[36..40].copy_from_slice(&[if m.lit { 1.0 } else { 0.0 }, m.alpha_cutoff as f32, normal_setting, outline_setting]);
            let parameters = item
                .parameters
                .or(m.instance_parameters.map(|values| values.map(|value| value as f32)))
                .unwrap_or([0.0; 4]);
            data[40..44].copy_from_slice(&parameters);
            // Unused extra vectors read as zero.
            let extra = &mut data[44..44 + MAX_DRAW_EXTRA * 4];
            extra.fill(0.0);
            if let Some(values) = item.extra {
                let count = values.len().min(MAX_DRAW_EXTRA * 4);
                extra[..count].copy_from_slice(&values[..count]);
            }
            self.ensure_pipelines(shader, samples);
            let mesh = match item.shape {
                Shape::Mesh(mesh) => Some(self.resolve(AssetKind::Mesh, catalog.mesh(mesh))?),
                _ => None,
            };
            ready.push(ReadyEntry { item: index, offset: (index as u64 * self.stride) as u32, group, material, shader, mesh });
        }
        if !items.is_empty() {
            self.queue.write_buffer(&self.draw_uniform, 0, bytemuck::cast_slice(&self.uniform_data[..items.len() * floats]));
        }
        // Translucent draws go back to front by their origin's view depth.
        let view = &frame.view;
        let depth_of = |transform: &Matrix| {
            f64::from(view[2]) * f64::from(transform[12])
                + f64::from(view[6]) * f64::from(transform[13])
                + f64::from(view[10]) * f64::from(transform[14])
                + f64::from(view[14])
        };
        let scene_entries: Vec<&ReadyEntry> = ready.iter().filter(|entry| items[entry.item].inset.is_none()).collect();
        let opaque: Vec<&ReadyEntry> = scene_entries.iter().copied().filter(|entry| entry.material.material.opaque()).collect();
        let mut translucent: Vec<&ReadyEntry> = scene_entries
            .iter()
            .copied()
            .filter(|entry| {
                let m = &entry.material.material;
                !m.overlay && (m.blend || m.color[3] < 1.0)
            })
            .collect();
        translucent.sort_by(|a, b| depth_of(items[a.item].transform).total_cmp(&depth_of(items[b.item].transform)));
        let overlays: Vec<&ReadyEntry> = scene_entries.iter().copied().filter(|entry| entry.material.material.overlay).collect();
        // Lit, opaque surfaces cast shadows, unless their material says not.
        let casters: Vec<&ReadyEntry> = if shadow_map.is_some() {
            opaque.iter().copied().filter(|entry| entry.material.material.lit && entry.material.material.shadow).collect()
        } else {
            Vec::new()
        };
        for entry in &casters {
            let instanced = matches!(items[entry.item].shape, Shape::Batch(_));
            let context = &self.context;
            if let Some(GpuAsset::Shader(shader)) = self.assets.current_mut(entry.shader) {
                let module = &shader.module;
                shader.shadow[instanced as usize].get_or_insert_with(|| shader::shadow_pipeline(context, module, instanced));
            }
        }
        // Decals whose material and texture are ready, by material.
        let mut decal_batches: Vec<DecalBatch<'_>> = Vec::new();
        for decal in &frame.decals {
            let material_slot = self.resolve(AssetKind::Material, catalog.material(decal.material))?;
            let Some(GpuAsset::Material(material)) = self.assets.current(material_slot) else { continue };
            let material = material.clone();
            let Some(key) = &material.material.texture else { continue };
            let image = self.resolve(AssetKind::Texture, key)?;
            let Some(GpuAsset::Texture(image)) = self.assets.current(image) else { continue };
            let paint = !material.material.lit;
            let found = decal_batches
                .iter_mut()
                .find(|other| other.texture_id == image.view_id && other.paint == paint && other.material == material_slot);
            match found {
                Some(batch) => batch.decals.push(decal),
                None => decal_batches.push(DecalBatch {
                    texture: image.view.clone(),
                    texture_id: image.view_id,
                    material: material_slot,
                    color: material.material.color,
                    paint,
                    decals: vec![decal],
                }),
            }
        }
        // Sprites over the finished image, in order.
        let mut sprite_batches: Vec<SpriteBatch<'_>> = Vec::new();
        for sprite in &frame.sprites {
            let Some(material) = self.material(catalog, sprite.material)? else { continue };
            let shader_slot = self.resolve(AssetKind::SpriteShader, &material.material.shader)?;
            let Some(GpuAsset::SpriteShader(pipeline)) = self.assets.current(shader_slot) else { continue };
            let pipeline = pipeline.clone();
            let image = match &material.material.texture {
                None => Some((self.white.view.clone(), self.white.view_id)),
                Some(key) => {
                    let slot = self.resolve(AssetKind::Texture, key)?;
                    match self.assets.current(slot) {
                        Some(GpuAsset::Texture(texture)) => Some((texture.view.clone(), texture.view_id)),
                        _ => None,
                    }
                }
            };
            let Some((texture, texture_id)) = image else { continue };
            match sprite_batches.last_mut() {
                Some(last) if last.key == (texture_id, shader_slot) => last.sprites.push(sprite),
                _ => sprite_batches.push(SpriteBatch { pipeline, texture, key: (texture_id, shader_slot), sprites: vec![sprite] }),
            }
        }
        // Each inset's targets and scene, ready to draw into.
        let (output_width, output_height) = target.size();
        let mut insets = Vec::new();
        for (index, inset) in frame.insets.iter().enumerate() {
            let (width, height) = (inset.width, inset.height);
            if width <= 0.0
                || height <= 0.0
                || inset.x >= output_width as f32
                || inset.y >= output_height as f32
                || inset.x + width <= 0.0
                || inset.y + height <= 0.0
            {
                continue;
            }
            let pixelated = inset.pixel_size > 0.0;
            let (width64, height64) = (f64::from(width), f64::from(height));
            let resolution = (if pixelated { 1.0 / f64::from(inset.pixel_size).max(1.0) } else { 2.0 })
                .min(4096.0 / width64)
                .min(4096.0 / height64)
                .min((3840.0 * 2160.0 / (width64 * height64)).sqrt());
            let size = |value: f64| ((value * resolution).ceil() as u32).max(1);
            let drawn = (size(width64), size(height64));
            self.targets_for(index, drawn.0, drawn.1, pixelated);
            if self.inset_scenes.len() <= index {
                self.inset_scenes.resize_with(index + 1, || None);
            }
            if self.inset_scenes[index].is_none() {
                let uniform = uniform_buffer(&self.device, "inset scene", SCENE_BYTES);
                let device = self.device.clone();
                let scene_layout = self.scene_layout.clone();
                let resources = self.shadow_resources();
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("inset scene"),
                    layout: &scene_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&resources.stand_in.create_view(&Default::default())),
                        },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&resources.sampler) },
                    ],
                });
                self.inset_scenes[index] = Some(InsetScene { uniform, group });
            }
            let mut data = inset.view.to_vec();
            data.extend(inset.projection);
            data.extend(vector(&inset.ambient));
            data.extend(vector(&inset.light_direction));
            data.extend(vector(&inset.light_color));
            data.extend(IDENTITY);
            data.extend([0.0; 4]);
            data.extend(heavens(&inset.view));
            let scene = self.inset_scenes[index].as_ref().unwrap();
            self.queue.write_buffer(&scene.uniform, 0, bytemuck::cast_slice(&data));
            let targets = self.inset_targets[index].as_ref().unwrap();
            self.queue.write_buffer(
                &targets.placement,
                0,
                bytemuck::cast_slice(&[
                    inset.x,
                    inset.y,
                    width,
                    height,
                    output_width as f32,
                    output_height as f32,
                    drawn.0 as f32 / targets.width as f32,
                    drawn.1 as f32 / targets.height as f32,
                ]),
            );
            insets.push((index, drawn));
        }

        let device = self.device.clone();
        let mut encoder = device.create_command_encoder(&Default::default());
        let drawing = Drawing {
            assets: &self.assets,
            items: &items,
            batches: &batches,
            solids: &self.solid_meshes,
            instances: &self.instance_buffer,
        };
        if let Some(map) = &shadow_map {
            let mut casting = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: map,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            casting.set_bind_group(0, &self.shadows.as_ref().unwrap().group, &[]);
            for entry in &casters {
                let instanced = matches!(items[entry.item].shape, Shape::Batch(_));
                casting.set_pipeline(drawing.shader(entry).shadow[instanced as usize].as_ref().unwrap());
                casting.set_bind_group(1, &entry.group, &[entry.offset]);
                drawing.draw(&mut casting, entry, None);
            }
        }
        // Acquired as late as possible.
        let Some(output) = target.view() else { return Ok(()) };
        // Opaque surfaces, then any decals over them, then translucent ones
        // and overlays; without decals, all in one pass.
        let stencil = |material: &Material| (if material.aberration { 1 } else { 0 }) | if material.decals { DECAL_MARK } else { 0 };
        let layered = !decal_batches.is_empty();
        let scene_group = &self.scene_group.as_ref().unwrap().0;
        let draw_layers = |pass: &mut wgpu::RenderPass<'_>, layers: &[(&[&ReadyEntry], Layer)], stats: &mut RenderStats| {
            pass.set_bind_group(0, scene_group, &[]);
            pass.set_blend_constant(wgpu::Color {
                r: OCCLUDED_STRENGTH,
                g: OCCLUDED_STRENGTH,
                b: OCCLUDED_STRENGTH,
                a: OCCLUDED_STRENGTH,
            });
            let mut mark = 0;
            for (entries, layer) in layers {
                for entry in entries.iter() {
                    let wanted = stencil(&entry.material.material);
                    if mark != wanted {
                        mark = wanted;
                        pass.set_stencil_reference(mark);
                    }
                    let pipeline = match items[entry.item].shape {
                        Shape::Batch(index) => &batches[index].pipeline,
                        _ => drawing.shader(entry).made[&samples].layer(*layer),
                    };
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(1, &entry.group, &[entry.offset]);
                    drawing.draw(pass, entry, Some(stats));
                }
            }
        };
        let later: [(&[&ReadyEntry], Layer); 3] =
            [(&translucent, Layer::Translucent), (&overlays, Layer::Occluded), (&overlays, Layer::Visible)];
        let clear = wgpu::Color {
            r: f64::from(frame.clear.r),
            g: f64::from(frame.clear.g),
            b: f64::from(frame.clear.b),
            a: f64::from(frame.clear.a),
        };
        {
            let mut first = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(targets.color(clear, true, !layered))],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: wgpu::StoreOp::Store }),
                }),
                ..Default::default()
            });
            let mut layers = vec![(&opaque[..], Layer::Opaque)];
            if !layered {
                layers.extend(later);
            }
            draw_layers(&mut first, &layers, &mut stats);
        }
        if layered {
            let decals = self.decals.get_or_insert_with(|| Decals::new(&device, SCENE_FORMAT));
            {
                let mut decal_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("decals"),
                    color_attachments: &[Some(targets.color(clear, false, false))],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &targets.depth,
                        depth_ops: None,
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                decals.draw(&device, &self.queue, &mut decal_pass, frame, &decal_batches, &targets.depth_only, samples, targets.size);
            }
            let mut last = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene over decals"),
                color_attachments: &[Some(targets.color(clear, false, true))],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                }),
                ..Default::default()
            });
            draw_layers(&mut last, &later, &mut stats);
        }
        self.post.as_mut().unwrap().finish(&mut encoder, frame, &output);
        if !sprite_batches.is_empty() {
            let layout = self.sprite_group_layout.clone();
            let sprites = self.sprites.get_or_insert_with(|| Sprites::new(&device, layout));
            sprites.draw(&device, &self.queue, &mut encoder, &output, (output_width, output_height), &sprite_batches);
        }
        // An inset: supersampled and multisampled, lit by its own light without
        // shadows, seen by its own camera in its rectangle of the screen.
        for &(index, drawn) in &insets {
            let targets = self.inset_targets[index].as_ref().unwrap();
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("inset"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &targets.color,
                        depth_slice: None,
                        resolve_target: Some(&targets.resolved),
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Discard },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &targets.depth,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                        stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: wgpu::StoreOp::Discard }),
                    }),
                    ..Default::default()
                });
                pass.set_viewport(0.0, 0.0, drawn.0 as f32, drawn.1 as f32, 0.0, 1.0);
                pass.set_scissor_rect(0, 0, drawn.0, drawn.1);
                pass.set_bind_group(0, &self.inset_scenes[index].as_ref().unwrap().group, &[]);
                for entry in &ready {
                    if items[entry.item].inset != Some(index) {
                        continue;
                    }
                    let Some(pipeline) = drawing.shader(entry).inset() else { continue };
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(1, &entry.group, &[entry.offset]);
                    drawing.draw(&mut pass, entry, Some(&mut stats));
                }
            }
            let mut laid = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("inset composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            laid.set_pipeline(self.inset_composite.as_ref().unwrap());
            laid.set_bind_group(0, &targets.group, &[]);
            laid.draw(0..6, 0..1);
        }
        let waiting = self.waiting;
        let arrived = self.assets.arrived;
        let mut gone = false;
        if let Some(veil) = &mut self.veil {
            let rose = *veil.rose.get_or_insert(now);
            if veil.clearing.is_none() && (!waiting || now - rose.max(arrived) > VEIL_PATIENCE) {
                veil.clearing = Some(now);
            }
            let cleared = veil
                .clearing
                .map_or(0.0, |clearing| ((now - clearing) / f64::from(self.veil_seconds.unwrap_or(0.0)).max(1e-3)).min(1.0));
            let left = 1.0 - cleared * cleared * (3.0 - 2.0 * cleared);
            if left > 0.0 {
                let mut data = Vec::with_capacity(12);
                data.extend(vector(&frame.sky.zenith));
                data.extend(vector(&frame.sky.horizon));
                data.extend([(now - rose) as f32, left as f32, output_width as f32 / output_height.max(1) as f32, 0.0]);
                self.queue.write_buffer(&veil.uniform, 0, bytemuck::cast_slice(&data));
                let mut covering = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("veil"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &output,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    ..Default::default()
                });
                covering.set_pipeline(&veil.pipeline);
                covering.set_bind_group(0, &veil.group, &[]);
                covering.draw(0..3, 0..1);
            } else {
                gone = true;
            }
        }
        if gone {
            self.veil = None;
        }
        self.last_stats = stats;
        self.queue.submit([encoder.finish()]);
        target.present();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_round_as_renderer_ts_does() {
        // From renderer.ts's halfOf.
        for (value, half) in [
            (0.0, 0x0000),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (0.5, 0x3800),
            (65504.0, 0x7bff),
            (1e6, 0x7c00),
            (6.1035156e-5, 0x0400),
            (3.0e-5, 0x01f7),
            (1e-9, 0x0000),
            (0.1, 0x2e66),
            (1.0009766, 0x3c01),
        ] {
            assert_eq!(half_of(value), half, "{value}");
        }
    }
}
