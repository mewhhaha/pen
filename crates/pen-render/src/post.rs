// The passes around the scene. The scene draws offscreen in floating point
// (`SCENE_FORMAT`), so light can go past white: multisampled and resolved for
// MSAA, at the render scale's size. Ambient occlusion is then worked out from
// its depth; a sky fills what the scene left empty, the occlusion darkens the
// rest and fog mists everything beyond the arena. With bloom, what is
// brighter than white spills light round it (`bloom`). A last pass adds the
// bloom, rolls the highlights off into white (`toned`), applies FXAA, splits
// the colors of the surfaces whose material asks for the frame's aberration,
// and scales the image to the screen. The passes after the scene draw as a
// render graph (graph.rs), made again when the targets' size or the passes
// change.
//
// `${...}` in the WGSL marks where a value is put in.
use std::collections::HashMap;

use pen_protocol::{Matrix, Quality, RenderFrame};

use crate::built_in::{BuiltIn, BuiltIns};
use crate::graph::{Binding, Graph, GraphBuilder, Handle, Spec, Target};
use crate::matrix::{invert, multiply};
use crate::packed_texture::level_size;
use crate::renderer::GpuTimer;
use crate::shader::PipelineContext;

/// The scene's format: floating point, so its light can go past white.
pub(crate) const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// WGSL: the scene's light as the screen shows it. Up to `shoulder` each
/// channel shows as it is; above, it rolls off smoothly toward 1 without
/// reaching it, so bright surfaces go white as they did when the screen
/// clipped them, keeping their shading, and nothing hard-clips.
pub(crate) const TONED_WGSL: &str = include_str!("shaders/toned.wgsl");

const FULLSCREEN: &str = include_str!("shaders/fullscreen.wgsl");

fn depth_texture(samples: u32) -> &'static str {
    if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" }
}

// The sky's colors come from the frame's lighting. Each pixel blends the sky
// and fog by how many of its samples the scene left empty, so MSAA keeps its
// smooth edges against the sky.
const ATMOSPHERE_SHADER: &str = include_str!("shaders/atmosphere.wgsl");

pub(crate) fn atmosphere_shader(samples: u32) -> String {
    ATMOSPHERE_SHADER
        .replace("${depth_texture}", depth_texture(samples))
        .replace("${fullscreen}", FULLSCREEN)
        .replace("${samples}", &samples.to_string())
        .replace("${share}", &format!("{:.4}", 1.0 / f64::from(samples)))
}

// Directions around a surface's normal (z) to test for occluders, spread on a
// golden-angle spiral and gathered toward the surface.
fn kernel() -> String {
    (0..12)
        .map(|index| {
            let index = f64::from(index);
            let z = 1.0 - (index + 0.5) / 12.0 * 0.9;
            let around = (1.0 - z * z).sqrt();
            let angle = index * std::f64::consts::PI * (3.0 - 5f64.sqrt());
            let reach = 0.25 + 0.75 * ((index + 1.0) / 12.0).powi(2);
            let values = [angle.cos() * around, angle.sin() * around, z].map(|value| format!("{:.5}", value * reach));
            format!("vec3f({})", values.join(", "))
        })
        .collect::<Vec<_>>()
        .join(", ")
}

// Screen-space ambient occlusion: how much of the hemisphere above each
// surface the depth buffer shows to be filled, within a world-space radius.
const OCCLUSION_SHADER: &str = include_str!("shaders/occlusion.wgsl");

pub(crate) fn occlusion_shader(samples: u32) -> String {
    OCCLUSION_SHADER
        .replace("${depth_texture}", depth_texture(samples))
        .replace("${fullscreen}", FULLSCREEN)
        .replace("${kernel}", &kernel())
}

const PRESENT_SHADER: &str = include_str!("shaders/present.wgsl");

pub(crate) fn present_shader(samples: u32, display: &crate::Display) -> String {
    // An HDR display's highlights roll off toward its peak, not white.
    let toned = if display.is_hdr() { crate::display::toned_wgsl(display.encoded_peak()) } else { TONED_WGSL.into() };
    PRESENT_SHADER
        .replace("${marks_texture}", if samples > 1 { "texture_multisampled_2d<u32>" } else { "texture_2d<u32>" })
        .replace("${fullscreen}", FULLSCREEN)
        .replace("${tonedWgsl}", &toned)
        .replace("${fxaa}", include_str!("shaders/fxaa.wgsl"))
}

// Bloom: the light past white, spilling round where it is. The first pass
// halves the scene, keeping only what is brighter than `threshold` (eased
// in over `knee` below it) and weighting each group of taps against the
// brightest pixels, so lone sparkles do not flicker; each next pass halves
// the last, a mip down. Then each level, blurred with a tent, is added onto
// the one above, back up to half the scene's size, which the last pass adds
// over the scene (after Jorge Jimenez's bloom for Call of Duty: Advanced
// Warfare: a 13-tap filter down, a 9-tap tent up).
const BLOOM_VERTEX: &str = include_str!("shaders/bloom_vertex.wgsl");

const BLOOM_DOWN_SHADER: &str = include_str!("shaders/bloom_down.wgsl");

pub(crate) fn bloom_down_shader(first: bool) -> String {
    BLOOM_DOWN_SHADER
        .replace("${bloomVertex}", BLOOM_VERTEX)
        .replace("${weight}", if first { "1.0 / (1.0 + brightness(color))" } else { "1.0" })
        .replace("${past_white}", if first { "color = past_white(color);" } else { "" })
}

const BLOOM_UP_SHADER: &str = include_str!("shaders/bloom_up.wgsl");

pub(crate) fn bloom_up_shader() -> String {
    BLOOM_UP_SHADER.replace("${bloomVertex}", BLOOM_VERTEX)
}

// How much of the bloom the last pass adds, and how many times at most the
// scene is halved for it.
const BLOOM_STRENGTH: f32 = 0.35;
const BLOOM_LEVELS: u32 = 6;

const MAX_RENDER_PIXELS: f64 = 3840.0 * 2160.0;

// How far ambient occlusion darkens, and its reach, curve and bias.
const OCCLUSION_STRENGTH: f32 = 1.0;
const OCCLUSION_RADIUS: f32 = 1.8;
const OCCLUSION_POWER: f32 = 2.2;
const OCCLUSION_BIAS: f32 = 0.04;

// The atmosphere pass's uniform: the camera, the sky the game worked out, the
// fog and how far ambient occlusion darkens.
fn atmosphere_of(frame: &RenderFrame, projection: &Matrix, quality: &Quality, width: u32, height: u32) -> [f32; 56] {
    let camera = invert(&frame.view);
    let sky = &frame.sky;
    let clear = &frame.clear;
    let light = &frame.light_color;
    let mut data = [0.0; 56];
    data[..16].copy_from_slice(&invert(&multiply(projection, &frame.view)));
    data[16..].copy_from_slice(&[
        camera[12],
        camera[13],
        camera[14],
        if quality.sky { 1.0 } else { 0.0 },
        sky.zenith.x,
        sky.zenith.y,
        sky.zenith.z,
        sky.night,
        sky.horizon.x,
        sky.horizon.y,
        sky.horizon.z,
        0.0,
        sky.ground.x,
        sky.ground.y,
        sky.ground.z,
        0.0,
        sky.sun.x,
        sky.sun.y,
        sky.sun.z,
        0.0,
        light.x,
        light.y,
        light.z,
        0.0,
        clear.r,
        clear.g,
        clear.b,
        1.0,
        frame.fog.radius,
        frame.fog.density,
        frame.fog.height,
        if quality.occlusion { OCCLUSION_STRENGTH } else { 0.0 },
        width as f32,
        height as f32,
        0.0,
        0.0,
        frame.clouds.cover,
        frame.clouds.drift.x,
        frame.clouds.drift.z,
        0.0,
    ]);
    data
}

#[derive(Clone, Copy, PartialEq)]
struct TargetsKey {
    width: u32,
    height: u32,
    samples: u32,
    occluding: bool,
    atmosphere: bool,
    levels: u32,
    taa: bool,
}

/// The scene's attachments, as the passes view them, and the graph of the
/// passes after the scene.
struct Targets {
    key: TargetsKey,
    /// Depth, and the stencil marks of surfaces the aberration splits; and
    /// the depth alone.
    depth: wgpu::TextureView,
    depth_only: wgpu::TextureView,
    marks: wgpu::TextureView,
    /// The scene's samples, resolved into `scene`.
    multisampled: Option<wgpu::TextureView>,
    /// The resolved scene.
    scene: wgpu::TextureView,
    /// The graph of the passes after the scene; with TAA, one for each
    /// history texture it reads, turn about.
    graphs: Vec<Graph>,
}

/// The attachments of one frame's scene passes, from `Post::begin`.
pub(crate) struct SceneTargets {
    pub size: (u32, u32),
    pub depth: wgpu::TextureView,
    /// The depth alone, for passes that read it while it stays attached.
    pub depth_only: wgpu::TextureView,
    pub scene: wgpu::TextureView,
    multisampled: Option<wgpu::TextureView>,
}

impl SceneTargets {
    /// The color attachment of a scene pass: the first clears, the last
    /// resolves samples, and one pass can be both.
    pub fn color(&self, clear: wgpu::Color, first: bool, last: bool) -> wgpu::RenderPassColorAttachment<'_> {
        let load = if first { wgpu::LoadOp::Clear(clear) } else { wgpu::LoadOp::Load };
        match &self.multisampled {
            None => wgpu::RenderPassColorAttachment {
                view: &self.scene,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
            },
            Some(multisampled) => wgpu::RenderPassColorAttachment {
                view: multisampled,
                depth_slice: None,
                resolve_target: last.then_some(&self.scene),
                ops: wgpu::Operations { load, store: if last { wgpu::StoreOp::Discard } else { wgpu::StoreOp::Store } },
            },
        }
    }
}

/// One of the passes' pipelines, its layout made from its shader.
pub(crate) fn pipeline(context: &PipelineContext, which: BuiltIn) -> wgpu::RenderPipeline {
    let add = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    let (code, target, blend) = match which {
        BuiltIn::Occlusion(samples) => (occlusion_shader(samples), wgpu::TextureFormat::R8Unorm, None),
        BuiltIn::Atmosphere(samples) => (atmosphere_shader(samples), SCENE_FORMAT, None),
        BuiltIn::Taa(samples) => (crate::taa::shader(samples), SCENE_FORMAT, None),
        BuiltIn::Present(samples) => (present_shader(samples, &context.display), context.format, None),
        BuiltIn::BloomDown { first } => (bloom_down_shader(first), SCENE_FORMAT, None),
        // Each level up is added onto the one above.
        BuiltIn::BloomUp => (bloom_up_shader(), SCENE_FORMAT, Some(wgpu::BlendState { color: add, alpha: add })),
        BuiltIn::Decal { .. } | BuiltIn::Composite | BuiltIn::Particles { .. } => {
            unreachable!("not a pass around the scene: {which:?}")
        }
    };
    let label = format!("{which:?}");
    let module = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&label),
        source: wgpu::ShaderSource::Wgsl(code.into()),
    });
    context.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&label),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format: target, blend, write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: context.cache.get(),
    })
}

struct FilteredPresent {
    source: wgpu::TextureView,
    marks: wgpu::TextureView,
    group: wgpu::BindGroup,
}

pub(crate) struct Post {
    device: wgpu::Device,
    queue: wgpu::Queue,
    uniforms: HashMap<&'static str, wgpu::Buffer>,
    sampler: wgpu::Sampler,
    // Bound in place of the occlusion when it is off, and of the bloom.
    unoccluded: wgpu::Texture,
    unbloomed: wgpu::Texture,
    targets: Option<Targets>,
    filtered_present: Option<FilteredPresent>,
    /// The begun frame's quality and output size.
    quality: Option<Quality>,
    size: (u32, u32),
    /// With TAA: frames jittered so far, the begun frame's jittered
    /// projection, and what the history remembers.
    jitters: u64,
    jittered: Option<(Matrix, (f64, f64))>,
    memory: crate::taa::Memory,
}

fn texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    usage: wgpu::TextureUsages,
    sample_count: u32,
    format: wgpu::TextureFormat,
    mip_level_count: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

impl Post {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // A 1 x 1 texture holding one value, bound in place of a pass that is off.
        let single = |format: wgpu::TextureFormat, value: u8| {
            let made = texture(
                device,
                1,
                1,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                1,
                format,
                1,
            );
            if format == wgpu::TextureFormat::R8Unorm {
                queue.write_texture(
                    made.as_image_copy(),
                    &[value],
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(1), rows_per_image: None },
                    wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                );
            }
            made
        };
        Self {
            device: device.clone(),
            queue: queue.clone(),
            uniforms: HashMap::new(),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            unoccluded: single(wgpu::TextureFormat::R8Unorm, 255),
            unbloomed: single(SCENE_FORMAT, 0),
            targets: None,
            filtered_present: None,
            quality: None,
            size: (1, 1),
            jitters: 0,
            jittered: None,
            memory: Default::default(),
        }
    }

    /// The passes after the scene that the last frame drew, in order.
    pub fn passes(&self) -> Vec<&'static str> {
        self.targets.as_ref().map(|targets| targets.graphs[0].names()).unwrap_or_default()
    }

    /// The begun frame's projection, jittered within its pixels, when TAA is
    /// on: the scene draws with it.
    pub fn jittered(&self) -> Option<Matrix> {
        self.jittered.map(|(projection, _)| projection)
    }

    fn uniform(&mut self, name: &'static str, size: u64) -> wgpu::Buffer {
        let device = &self.device;
        self.uniforms
            .entry(name)
            .or_insert_with(|| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(name),
                    size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .clone()
    }

    /// Prepares the attachments for `frame` drawn with `quality` on an
    /// output of `size`, and returns the scene pass's color and depth
    /// attachments.
    pub fn begin(
        &mut self,
        built_ins: &mut BuiltIns,
        frame: &RenderFrame,
        quality: Quality,
        size: (u32, u32),
    ) -> SceneTargets {
        self.quality = Some(quality);
        self.size = size;
        let occluding = quality.occlusion;
        let atmosphere = quality.sky || frame.fog.density > 0.0 || occluding || frame.clouds.cover > 0.0;
        // At most the pixels of a 4K screen, whatever the scale and display:
        // the multisampled targets of more would take gigabytes.
        let limit = self.device.limits().max_texture_dimension_2d;
        let budget =
            f64::from(quality.scale).min((MAX_RENDER_PIXELS / (f64::from(size.0) * f64::from(size.1)).max(1.0)).sqrt());
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "at least 1, and saturating")]
        let scaled = |value: u32| ((f64::from(value) * budget).round().max(1.0) as u32).min(limit);
        let (width, height) = (scaled(size.0), scaled(size.1));
        let samples = quality.msaa;
        // Halving down to no smaller than a few pixels.
        let half = (width.div_ceil(2), height.div_ceil(2));
        let levels = if quality.bloom {
            // floor(log2(side / 4)) + 1
            half.0.min(half.1).checked_ilog2().unwrap_or(0).saturating_sub(1).clamp(1, BLOOM_LEVELS)
        } else {
            0
        };
        let taa = quality.taa;
        let key = TargetsKey { width, height, samples, occluding, atmosphere, levels, taa };
        if self.targets.as_ref().is_none_or(|targets| targets.key != key) {
            self.targets = None;
            let attachment = wgpu::TextureUsages::RENDER_ATTACHMENT;
            let sampled = attachment | wgpu::TextureUsages::TEXTURE_BINDING;
            let device = &self.device;
            // Decals read the depth.
            let depth = texture(device, width, height, sampled, samples, wgpu::TextureFormat::Depth24PlusStencil8, 1);
            let depth_only = || {
                depth.create_view(&wgpu::TextureViewDescriptor {
                    aspect: wgpu::TextureAspect::DepthOnly,
                    ..Default::default()
                })
            };
            let multisampled =
                (samples > 1).then(|| texture(device, width, height, attachment, samples, SCENE_FORMAT, 1));
            let scene = texture(device, width, height, sampled, 1, SCENE_FORMAT, 1);
            let marks = depth.create_view(&wgpu::TextureViewDescriptor {
                aspect: wgpu::TextureAspect::StencilOnly,
                ..Default::default()
            });
            let view = |texture: &wgpu::Texture| texture.create_view(&Default::default());
            // With TAA, a graph for each history texture read, sharing the
            // rest; a new history starts afresh.
            self.memory = Default::default();
            let histories = taa.then(|| [0, 1].map(|_| crate::taa::history(device, width, height, SCENE_FORMAT)));
            let mut graphs: Vec<Graph> = Vec::new();
            for read in 0..if taa { 2 } else { 1 } {
                let history = histories.as_ref().map(|textures| [view(&textures[read]), view(&textures[1 - read])]);
                let graph = self
                    .graph(built_ins, key, depth_only(), view(&scene), marks.clone(), history, graphs.first())
                    .expect("the passes after the scene read only what earlier ones draw");
                log::debug!("passes after the scene: {:?}, {} textures", graph.names(), graph.allocations());
                graphs.push(graph);
            }
            self.targets = Some(Targets {
                key,
                depth: view(&depth),
                depth_only: depth_only(),
                multisampled: multisampled.as_ref().map(view),
                scene: view(&scene),
                marks,
                graphs,
            });
        }
        // The scene's samples move within their pixels from frame to frame.
        self.jittered = taa.then(|| {
            self.jitters += 1;
            let offset = crate::taa::jitter(self.jitters);
            (crate::taa::jittered(&frame.projection, offset, (width, height)), offset)
        });
        let current = self.targets.as_ref().unwrap();
        SceneTargets {
            size: (current.key.width, current.key.height),
            depth: current.depth.clone(),
            depth_only: current.depth_only.clone(),
            scene: current.scene.clone(),
            multisampled: current.multisampled.clone(),
        }
    }

    /// The passes after the scene for frames of `key`, reading the scene's
    /// depth, its picture and its stencil marks, and ending on the output.
    /// With TAA, `history` is the texture it reads and the one it draws.
    #[allow(clippy::too_many_arguments, reason = "the scene's views, each its own")]
    fn graph(
        &mut self,
        built_ins: &mut BuiltIns,
        key: TargetsKey,
        depth: wgpu::TextureView,
        scene: wgpu::TextureView,
        marks: wgpu::TextureView,
        history: Option<[wgpu::TextureView; 2]>,
        sharing: Option<&Graph>,
    ) -> anyhow::Result<Graph> {
        let TargetsKey { width, height, samples, occluding, atmosphere, levels, .. } = key;
        let full = |format: wgpu::TextureFormat| Spec { width, height, format };
        let sampler = Binding::Sampler(self.sampler.clone());
        let mut graph = GraphBuilder::default();
        let depth = graph.import("depth", depth);
        let scene = graph.import("scene", scene);
        let marks = graph.import("marks", marks);
        let occlusion = if occluding {
            let occlusion = graph.texture("occlusion", full(wgpu::TextureFormat::R8Unorm));
            let uniform = Binding::Uniform(self.uniform("occlusion", 160));
            let bindings = vec![uniform, Binding::Texture(depth)];
            graph.pass("occlusion", BuiltIn::Occlusion(samples), bindings, Target::Texture(occlusion), false);
            occlusion
        } else {
            graph.import("unoccluded", self.unoccluded.create_view(&Default::default()))
        };
        let mut lit = scene;
        if atmosphere {
            let composed = graph.texture("composed", full(SCENE_FORMAT));
            let uniform = Binding::Uniform(self.uniform("atmosphere", 224));
            let bindings = vec![
                uniform,
                Binding::Texture(scene),
                Binding::Texture(depth),
                Binding::Texture(occlusion),
                sampler.clone(),
            ];
            graph.pass("atmosphere", BuiltIn::Atmosphere(samples), bindings, Target::Texture(composed), false);
            lit = composed;
        }
        // TAA blends the lit picture into the history, which the passes
        // after it read.
        if let Some([read, drawn]) = history {
            let (read, drawn) = (graph.import("history", read), graph.import("history", drawn));
            let uniform = Binding::Uniform(self.uniform("taa", 160));
            let bindings =
                vec![uniform, Binding::Texture(lit), Binding::Texture(read), Binding::Texture(depth), sampler.clone()];
            graph.pass("taa", BuiltIn::Taa(samples), bindings, Target::Texture(drawn), false);
            lit = drawn;
        }
        // Bloom halves the picture level by level, then adds each level onto
        // the one above, back up to half the scene's size.
        let half = (width.div_ceil(2), height.div_ceil(2));
        let bloom: Vec<Handle> = (0..levels)
            .map(|level| {
                let size = (level_size(half.0, level), level_size(half.1, level));
                graph.texture("bloom", Spec { width: size.0, height: size.1, format: SCENE_FORMAT })
            })
            .collect();
        for (index, &level) in bloom.iter().enumerate() {
            let (first, from) = if index == 0 { (true, lit) } else { (false, bloom[index - 1]) };
            let bindings = vec![Binding::Texture(from), sampler.clone()];
            graph.pass("bloom down", BuiltIn::BloomDown { first }, bindings, Target::Texture(level), false);
        }
        for pair in bloom.windows(2).rev() {
            let bindings = vec![Binding::Texture(pair[1]), sampler.clone()];
            graph.pass("bloom up", BuiltIn::BloomUp, bindings, Target::Texture(pair[0]), true);
        }
        let glow = match bloom.first() {
            Some(&level) => level,
            None => graph.import("unbloomed", self.unbloomed.create_view(&Default::default())),
        };
        let uniform = Binding::Uniform(self.uniform("present", 64));
        let bindings = vec![uniform, Binding::Texture(lit), sampler, Binding::Texture(marks), Binding::Texture(glow)];
        graph.pass("present", BuiltIn::Present(samples), bindings, Target::Output, false);
        graph.compile(&self.device, built_ins, sharing)
    }

    /// Encodes the ordinary path without additional targets or passes.
    pub fn finish(
        &mut self,
        built_ins: &mut BuiltIns,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        output: &wgpu::TextureView,
        timer: &GpuTimer,
    ) {
        self.finish_base(built_ins, encoder, frame, output, timer, false);
    }

    /// With scene filters, defer camera effects until their output is ready.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_base(
        &mut self,
        built_ins: &mut BuiltIns,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        output: &wgpu::TextureView,
        timer: &GpuTimer,
        filtered: bool,
    ) {
        let quality = self.quality.expect("Post::begin comes first");
        let size = self.size;
        let TargetsKey { width, height, occluding, atmosphere, levels, taa, .. } = self.targets.as_ref().unwrap().key;
        // What the scene drew with.
        let (projection, offset) = self.jittered.unwrap_or((frame.projection, (0.0, 0.0)));
        if occluding {
            let uniform = self.uniform("occlusion", 160);
            let mut data = [0.0f32; 40];
            data[..16].copy_from_slice(&projection);
            data[16..32].copy_from_slice(&invert(&projection));
            data[32..].copy_from_slice(&[
                OCCLUSION_RADIUS,
                OCCLUSION_POWER,
                OCCLUSION_BIAS,
                0.0,
                width as f32,
                height as f32,
                0.0,
                0.0,
            ]);
            self.queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&data));
        }
        if atmosphere {
            let uniform = self.uniform("atmosphere", 224);
            let data = atmosphere_of(frame, &projection, &quality, width, height);
            self.queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&data));
        }
        // The graph reading the history this frame reads.
        let graph = if taa { self.memory.read() } else { 0 };
        if taa {
            let uniform = self.uniform("taa", 160);
            let data = self.memory.uniform(&frame.view, &frame.projection, (&projection, offset), (width, height));
            self.queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&data));
        }
        let uniform = self.uniform("present", 64);
        let (output_width, output_height) = (size.0 as f32, size.1 as f32);
        self.queue.write_buffer(
            &uniform,
            0,
            bytemuck::cast_slice(&[
                1.0 / output_width,
                1.0 / output_height,
                if quality.effects && !filtered { frame.blur } else { 0.0 },
                0.0,
                1.0 / width as f32,
                1.0 / height as f32,
                if quality.fxaa { 1.0 } else { 0.0 },
                0.0,
                // Screen heights, as texture coordinates.
                if quality.effects && !filtered { frame.aberration.x * output_height / output_width } else { 0.0 },
                if quality.effects && !filtered { frame.aberration.y } else { 0.0 },
                width as f32,
                height as f32,
                if levels > 0 { BLOOM_STRENGTH } else { 0.0 },
                0.0,
                0.0,
                0.0f32,
            ]),
        );
        self.targets.as_ref().unwrap().graphs[graph].execute(encoder, built_ins, output, timer);
    }
    /// Apply blur and stencil-masked aberration to the styled image, before UI.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_filtered(
        &mut self,
        built_ins: &mut BuiltIns,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        source: &wgpu::TextureView,
        output: &wgpu::TextureView,
        timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let uniform = self.uniform("filtered present", 64);
        let targets = self.targets.as_ref().unwrap();
        let key = targets.key;
        let marks = targets.marks.clone();
        let pipeline = built_ins.get(BuiltIn::Present(key.samples));
        let quality = self.quality.unwrap();
        let (width, height) = (self.size.0 as f32, self.size.1 as f32);
        self.queue.write_buffer(
            &uniform,
            0,
            bytemuck::cast_slice(&[
                1.0 / width,
                1.0 / height,
                if quality.effects { frame.blur } else { 0.0 },
                0.0,
                1.0 / width,
                1.0 / height,
                0.0,
                1.0,
                if quality.effects { frame.aberration.x * height / width } else { 0.0 },
                if quality.effects { frame.aberration.y } else { 0.0 },
                key.width as f32,
                key.height as f32,
                0.0,
                0.0,
                0.0,
                0.0f32,
            ]),
        );
        if self.filtered_present.as_ref().is_none_or(|kept| kept.source != *source || kept.marks != marks) {
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("filtered present"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(source) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&marks) },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&self.unbloomed.create_view(&Default::default())),
                    },
                ],
            });
            self.filtered_present = Some(FilteredPresent { source: source.clone(), marks, group });
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("filtered camera effects"),
            timestamp_writes: timestamps,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.filtered_present.as_ref().unwrap().group, &[]);
        pass.draw(0..3, 0..1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_fill_every_hole() {
        for code in [
            atmosphere_shader(1),
            atmosphere_shader(4),
            occlusion_shader(4),
            present_shader(1, &crate::Display::SDR),
            present_shader(
                4,
                &crate::Display { color_space: wgpu::SurfaceColorSpace::ExtendedSrgb, ..crate::Display::SDR },
            ),
            bloom_down_shader(true),
            bloom_down_shader(false),
            bloom_up_shader(),
            crate::taa::shader(1),
            crate::taa::shader(4),
        ] {
            assert!(!code.contains("${"), "{code}");
        }
        assert!(atmosphere_shader(4).contains("empty += 0.2500;"));
        // Five decimals, as WGSL literals.
        assert_eq!(
            kernel(),
            "vec3f(0.06923, 0.00000, 0.24564), vec3f(-0.09203, 0.08430, 0.24036), vec3f(0.01513, -0.17240, 0.24121), \
             vec3f(0.13697, 0.17865, 0.24583), vec3f(-0.28045, -0.04961, 0.25189), vec3f(0.29872, -0.19002, 0.25703), \
             vec3f(-0.11262, 0.41894, 0.25892), vec3f(-0.24177, -0.46551, 0.25521), vec3f(0.58818, 0.21480, 0.24355), \
             vec3f(-0.68243, 0.28170, 0.22161), vec3f(0.36455, -0.77903, 0.18704), vec3f(0.29644, 0.94510, 0.13750)"
        );
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "post/fxaa_tests.rs"]
mod fxaa_tests;
