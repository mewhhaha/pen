#![cfg_attr(
    not(all(feature = "render-ssgi", feature = "render-taa", feature = "render-dof", feature = "render-subsurface")),
    allow(dead_code)
)]
// The passes around the scene. The scene draws offscreen in floating point
// (`SCENE_FORMAT`), so light can go past white: multisampled and resolved for
// MSAA, at the render scale's size. Ambient occlusion is then worked out from
// its depth; a sky fills what the scene left empty, the occlusion darkens the
// rest and fog mists everything beyond the arena. With bloom, what is
// bright spills a little light round it (`bloom`). A last pass mixes in the
// bloom, rolls the highlights off into white (`toned`), applies FXAA, splits
// the colors of the surfaces whose material asks for the frame's aberration,
// and scales the image to the screen. The passes after the scene draw as a
// render graph (graph.rs), made again when the targets' size or the passes
// change.
//
// Surfaces whose material is `subsurface` are diffused first (subsurface.rs):
// once the opaque surfaces and their decals are resolved, two passes blur the
// scene's picture where the stencil marks them, along x and then along y, back
// into the scene's texture, so that everything below (the sky, the occlusion
// and the mist, then the rest) sees the diffused picture. A frame none of
// whose opaque draws are such makes no passes and no textures for it.
//
// A frame with translucent surfaces, particles or overlays splits that graph
// in two round a second pass over the scene's targets. What the scene drew
// until then, the opaque surfaces and their decals, is resolved, and the sky,
// the occlusion and the mist are applied to that alone (the opaque picture,
// level 0 of a chain of blurred levels, and the nearest depth of each of its
// pixels). The translucent pass then starts from that picture, written into
// every sample (the `backdrop`), under the same depth, and draws the
// translucent surfaces, the particles and the overlays, which mist
// themselves at their own depth (translucency.wgsl) and read the opaque
// picture. The passes that follow (TAA, the metering, the bloom, the present
// pass) read what it resolved to. A frame without any of them draws the one
// graph, as it always has.
//
// `${...}` in the WGSL marks where a value is put in.
use std::collections::HashMap;

use pen_protocol::{Matrix, Quality, RenderFrame};

use crate::built_in::{BuiltIn, BuiltIns};
use crate::exposure::Exposure;
use crate::grade::{self, Lut};
use crate::graph::{Binding, Graph, GraphBuilder, Handle, Spec, Target};
use crate::matrix::{invert, multiply};
use crate::packed_texture::level_size;
use crate::renderer::{GpuTimer, Span};
use crate::shader::PipelineContext;

/// The scene's format: floating point, so its light can go past white.
pub(crate) const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// WGSL: the scene's light as the screen shows it. Up to `shoulder` each
/// channel shows as it is; above, it rolls off smoothly toward 1 without
/// reaching it, so bright surfaces go white as they did when the screen
/// clipped them, keeping their shading, and nothing hard-clips.
pub(crate) const TONED_WGSL: &str = crate::wgsl!("toned.wgsl");

const FULLSCREEN: &str = crate::wgsl!("fullscreen.wgsl");

/// WGSL: the sRGB transfer function and exposure as light (`exposed`), which the
/// bloom's first pass, the metering and the present pass share.
pub(crate) const TRANSFER_WGSL: &str = crate::wgsl!("transfer.wgsl");

/// WGSL: the color grade (`grade_apply`), reference of Ink's `color.grade`.
pub(crate) const GRADE_WGSL: &str = crate::wgsl!("grade.wgsl");

fn depth_texture(samples: u32) -> &'static str {
    if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" }
}

// The sky's colors come from the frame's lighting. Each pixel blends the sky
// and fog by how many of its samples the scene left empty, so MSAA keeps its
// smooth edges against the sky.
const ATMOSPHERE_SHADER: &str = crate::wgsl!("atmosphere.wgsl");

const ATMOSPHERE_COMMON: &str = crate::wgsl!("atmosphere_common.wgsl");

pub(crate) fn atmosphere_shader(samples: u32, depth_mode: crate::camera_precision::DepthMode) -> String {
    let (depth_clear, empty_cond, nearer_op) = match depth_mode {
        crate::camera_precision::DepthMode::ForwardZ => ("1.0", "depth >= 1.0", "min"),
        crate::camera_precision::DepthMode::ReversedZ => ("0.0", "depth <= 0.0", "max"),
    };
    ATMOSPHERE_SHADER
        .replace("${common}", ATMOSPHERE_COMMON)
        .replace("${volume}", crate::volume::COMMON_WGSL)
        .replace("${depth_texture}", depth_texture(samples))
        .replace("${fullscreen}", FULLSCREEN)
        .replace("${samples}", &samples.to_string())
        .replace("${share}", &format!("{:.4}", 1.0 / f64::from(samples)))
        .replace("${depth_clear}", depth_clear)
        .replace("${empty_cond}", empty_cond)
        .replace("${nearer_op}", nearer_op)
}

// Ground-truth ambient occlusion (GTAO): horizon-based visibility from depth.
pub(crate) fn occlusion_shader(samples: u32) -> String {
    crate::deferred_wgsl!("occlusion.wgsl")
        .replace("${depth_texture}", depth_texture(samples))
        .replace("${fullscreen}", FULLSCREEN)
}

// The opaque scene's nearest depth to read from the translucent pass, its
// picture's chain of levels, and the picture written into the translucent
// pass's samples.
const OPAQUE_DEPTH_SHADER: &str = crate::wgsl!("opaque_depth.wgsl");

pub(crate) fn opaque_depth_shader(samples: u32, depth_mode: crate::camera_precision::DepthMode) -> String {
    let (depth_clear, nearer_op) = match depth_mode {
        crate::camera_precision::DepthMode::ForwardZ => ("1.0", "min"),
        crate::camera_precision::DepthMode::ReversedZ => ("0.0", "max"),
    };
    OPAQUE_DEPTH_SHADER
        .replace("${depth_texture}", depth_texture(samples))
        .replace("${fullscreen}", FULLSCREEN)
        .replace("${samples}", &samples.to_string())
        .replace("${depth_clear}", depth_clear)
        .replace("${nearer_op}", nearer_op)
}

const OPAQUE_MIP_SHADER: &str = crate::wgsl!("opaque_mip.wgsl");

const BACKDROP_SHADER: &str = crate::wgsl!("backdrop.wgsl");

pub(crate) fn backdrop_shader() -> String {
    BACKDROP_SHADER.replace("${fullscreen}", FULLSCREEN)
}

const PRESENT_SHADER: &str = crate::wgsl!("present.wgsl");

pub(crate) fn present_shader(samples: u32, display: &crate::Display) -> String {
    // An HDR display's highlights roll off toward its peak, not white.
    let toned = if display.is_hdr() { crate::display::toned_wgsl(display.encoded_peak()) } else { TONED_WGSL.into() };
    let toned_full = format!("{toned}\n{}", crate::tonemap::TONEMAP_SUITE_WGSL);
    PRESENT_SHADER
        .replace("${marks_texture}", if samples > 1 { "texture_multisampled_2d<u32>" } else { "texture_2d<u32>" })
        .replace("${fullscreen}", FULLSCREEN)
        .replace("${transfer}", TRANSFER_WGSL)
        .replace("${grade}", GRADE_WGSL)
        .replace("${tonedWgsl}", &toned_full)
        .replace("${fxaa}", crate::wgsl!("fxaa.wgsl"))
}

// Bloom: glare, a small share of the picture's light scattered round where it
// is, as a convolution of the whole exposed picture, not a threshold added on
// top. The first pass halves the picture in linear light after exposure, each
// tap held to `BLOOM_CEILING` and each group of taps weighted against its
// brightest pixel (Karis), so a lone sparkle weighs as one ordinary pixel; each
// next pass halves the last, a mip down. Then each level, blurred with a tent, is
// added onto the one above, back up to half the picture's size. The levels sum
// to as many pictures as there are levels, so the last pass takes
// `BLOOM_INTENSITY / levels` of the sum and keeps `1 - BLOOM_INTENSITY` of the
// picture: the light mixes in linear light and its total, and the mean of a
// uniform picture, do not change, and a bright window glows into a bounded
// neighbourhood at any exposure instead of flooding the frame (after Jorge
// Jimenez's bloom for Call of Duty: Advanced Warfare: a 13-tap filter down, a
// 9-tap tent up).
const BLOOM_VERTEX: &str = crate::wgsl!("bloom_vertex.wgsl");

const BLOOM_DOWN_SHADER: &str = crate::wgsl!("bloom_down.wgsl");

pub(crate) fn bloom_down_shader(first: bool) -> String {
    // The first level reads the picture as the camera exposes it, in linear light,
    // each tap held to `BLOOM_CEILING`, so that what spills is the picture's light
    // as the screen shows it, whatever the exposure.
    let convert = if first {
        format!(
            "{TRANSFER_WGSL}\n@group(0) @binding(2) var<storage, read> exposure: array<f32, 4>;\n\
             fn convert(color: vec3f) -> vec3f {{\n  \
               return min(srgb_decode(exposed(color, exposure[0])), vec3f({BLOOM_CEILING:.1}));\n}}"
        )
    } else {
        "fn convert(color: vec3f) -> vec3f { return color; }".to_owned()
    };
    BLOOM_DOWN_SHADER
        .replace("${bloomVertex}", BLOOM_VERTEX)
        .replace("${exposure}", &convert)
        .replace("${weight}", if first { "1.0 / (1.0 + brightness(color))" } else { "1.0" })
}

const BLOOM_UP_SHADER: &str = crate::wgsl!("bloom_up.wgsl");

pub(crate) fn bloom_up_shader() -> String {
    BLOOM_UP_SHADER.replace("${bloomVertex}", BLOOM_VERTEX)
}

// The share of the picture's light the bloom scatters, and how many times at
// most the picture is halved for it.
const BLOOM_INTENSITY: f32 = 0.05;

/// The most linear light (32 times SDR white, five stops over) one tap of the
/// bloom's source may bring, so a source of any brightness spills a bounded amount.
const BLOOM_CEILING: f32 = 32.0;
const BLOOM_LEVELS: u32 = 6;

const MAX_RENDER_PIXELS: f64 = 3840.0 * 2160.0;

// How far ambient occlusion darkens, and its reach, curve and bias.
const OCCLUSION_STRENGTH: f32 = 1.0;
const OCCLUSION_RADIUS: f32 = 1.8;
const OCCLUSION_POWER: f32 = 2.2;
const OCCLUSION_BIAS: f32 = 0.04;

// The atmosphere pass's uniform: the camera, the sky the game worked out, the
// fog and how far ambient occlusion darkens.
pub(crate) fn atmosphere_of(
    frame: &RenderFrame,
    projection: &Matrix,
    quality: &Quality,
    width: u32,
    height: u32,
) -> [f32; 56] {
    let camera = invert(&frame.view);
    let sky = &frame.sky;
    let clear = &frame.clear;
    let light = &frame.light_color;
    let sky_mode = if quality.physical_sky {
        if frame.clouds.cover > 0.001 { 3.0 } else { 2.0 }
    } else if quality.sky {
        1.0
    } else {
        0.0
    };
    let sun = if quality.physical_sky {
        crate::atmosphere_lut::normalize_sun_dir([
            -frame.light_direction.x,
            -frame.light_direction.y,
            -frame.light_direction.z,
        ])
    } else {
        [sky.sun.x, sky.sun.y, sky.sun.z]
    };
    let cam_alt_km = if quality.physical_sky {
        crate::atmosphere_lut::physical_camera_km([camera[12], camera[13], camera[14]])[1]
    } else {
        0.0
    };
    let mut data = [0.0; 56];
    data[..16].copy_from_slice(&invert(&multiply(projection, &frame.view)));
    data[16..].copy_from_slice(&[
        camera[12],
        camera[13],
        camera[14],
        sky_mode,
        sky.zenith.x,
        sky.zenith.y,
        sky.zenith.z,
        sky.night,
        sky.horizon.x,
        sky.horizon.y,
        sky.horizon.z,
        cam_alt_km,
        sky.ground.x,
        sky.ground.y,
        sky.ground.z,
        0.0,
        sun[0],
        sun[1],
        sun[2],
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

/// GPU views bound into `BuiltIn::Atmosphere` when `quality.physical_sky` is active.
pub(crate) struct PhysicalSkyBinding {
    /// Non-zero version key bumped whenever `sky_view`, `transmittance`, or `clouds` view identity changes.
    pub serial: u64,
    pub sky_view: wgpu::TextureView,
    pub transmittance: wgpu::TextureView,
    pub clouds: Option<wgpu::TextureView>,
}

/// Computes the scaled internal scene resolution `Post::begin` will allocate.
pub(crate) fn scene_size(device: &wgpu::Device, size: (u32, u32), scale: f32) -> (u32, u32) {
    let limit = device.limits().max_texture_dimension_2d;
    let budget = f64::from(scale).min((MAX_RENDER_PIXELS / (f64::from(size.0) * f64::from(size.1)).max(1.0)).sqrt());
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "at least 1, and saturating")]
    let scaled = |value: u32| ((f64::from(value) * budget).round().max(1.0) as u32).min(limit);
    (scaled(size.0), scaled(size.1))
}

#[derive(Clone, Copy, PartialEq)]
struct TargetsKey {
    width: u32,
    height: u32,
    /// Post-TAA target resolution (equals display resolution when temporal upscaling is active).
    taa_width: u32,
    taa_height: u32,
    samples: u32,
    occluding: bool,
    atmosphere: bool,
    ssgi: bool,
    levels: u32,
    taa: bool,
    dof: bool,
    motion_blur: bool,
    /// The froxel volume's grid the atmosphere pass reads (0 for none).
    volume: u64,
    /// Physical sky LUT + cloud texture generation (0 when off).
    physical_sky: u64,
}

/// What a frame draws beyond the scene's opaque surfaces.
#[derive(Clone, Copy, Default)]
pub(crate) struct Extras {
    /// Translucent surfaces, particles or overlays, drawn after the opaque picture is misted.
    pub translucent: bool,
    /// Whether any active translucent draw, particle or ribbon reads `opaque_depth`.
    pub opaque_depth: bool,
    /// Whether any active translucent draw reads `opaque_color` / `opaque_behind` mip levels.
    pub opaque_mips: bool,
    /// Opaque surfaces whose material is `subsurface`, diffused on screen.
    pub subsurface: bool,
}

/// What a frame with translucent surfaces draws and reads besides the scene's
/// own targets, made with the first such frame and kept until the targets change.
struct Split {
    opaque_depth: bool,
    opaque_mips: bool,
    opaque: OpaqueTargets,
    /// Level 0 of the opaque picture, as the backdrop pass reads it.
    backdrop: wgpu::BindGroup,
    /// The opaque picture's passes, run once the opaque surfaces are drawn.
    before: Graph,
    /// The passes that read the translucent pass's picture; with TAA, one for
    /// each history texture it reads, turn about.
    after: Vec<Graph>,
}

/// The opaque surfaces' picture and depth, as the translucent pass reads them.
#[derive(Clone)]
pub(crate) struct OpaqueTargets {
    /// Which these are (never 0), to know when a bind group of them is old.
    pub serial: u64,
    /// The picture after the sky, the occlusion and the mist, with its chain of levels.
    pub picture: wgpu::TextureView,
    /// The nearest depth of each of its pixels (1 for none).
    pub depth: wgpu::TextureView,
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
    /// Resolved color copied between ordered mesh filters.
    filter_picture: Option<wgpu::TextureView>,
    /// The graph of the passes after the scene; with TAA, one for each
    /// history texture it reads, turn about.
    graphs: Vec<Graph>,
    /// With TAA, the history textures: the picture each frame resolved to.
    histories: Option<[wgpu::TextureView; 2]>,
    /// With TAA, ping-pong lock status + previous NDC depth (`Rgba16Float`: `.r=lock`, `.g=prev_depth`, `.b=lock_luma`, `.a=valid`).
    locks: Option<[wgpu::TextureView; 2]>,
    /// With TAA or velocity motion blur, the 1x Rg16Float velocity target and its Depth32Float buffer.
    velocity: Option<(wgpu::TextureView, wgpu::TextureView)>,
    /// What frames with translucent surfaces add, once one has come.
    split: Option<Split>,
    /// The two passes that diffuse the marked surfaces of the resolved scene (into
    /// a texture of their own and back into the scene), once a frame needs them.
    subsurface: Option<Graph>,
}

/// The attachments of one frame's scene passes, from `Post::begin`.
pub(crate) struct SceneTargets {
    pub size: (u32, u32),
    pub depth: wgpu::TextureView,
    /// The depth alone, for passes that read it while it stays attached.
    pub depth_only: wgpu::TextureView,
    pub scene: wgpu::TextureView,
    multisampled: Option<wgpu::TextureView>,
    /// For a frame with translucent surfaces, the picture and depth of the
    /// opaque ones.
    pub opaque: Option<OpaqueTargets>,
    /// For a frame with TAA or velocity motion blur, `(velocity_view, velocity_depth)`.
    pub velocity: Option<(wgpu::TextureView, wgpu::TextureView)>,
    /// True when scene targets were freshly allocated this frame.
    pub targets_recreated: bool,
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

    /// Resolve for the next mesh filter while preserving the samples for subsequent draws.
    pub fn mesh_filter_color(&self, first: bool) -> wgpu::RenderPassColorAttachment<'_> {
        let mut attachment = self.color(wgpu::Color::BLACK, first, true);
        attachment.ops.store = wgpu::StoreOp::Store;
        attachment
    }

    /// The color attachment of the translucent pass: cleared, which the backdrop
    /// then overwrites everywhere, and resolved into the scene.
    pub fn translucent_color(&self) -> wgpu::RenderPassColorAttachment<'_> {
        let load = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
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
                resolve_target: Some(&self.scene),
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Discard },
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
    let depth_mode = context.depth_mode();
    let (code, target, blend) = match which {
        #[cfg(feature = "render-ssgi")]
        BuiltIn::Occlusion(samples) => (occlusion_shader(samples), wgpu::TextureFormat::R8Unorm, None),
        #[cfg(not(feature = "render-ssgi"))]
        BuiltIn::Occlusion(_) => unreachable!("render-ssgi disabled"),
        BuiltIn::Atmosphere(samples) => (atmosphere_shader(samples, depth_mode), SCENE_FORMAT, None),
        #[cfg(feature = "render-ssgi")]
        BuiltIn::Ssgi(samples) => (crate::ssgi::shader(samples), SCENE_FORMAT, None),
        #[cfg(not(feature = "render-ssgi"))]
        BuiltIn::Ssgi(_) => unreachable!("render-ssgi disabled"),
        BuiltIn::OpaqueDepth(samples) => {
            (opaque_depth_shader(samples, depth_mode), wgpu::TextureFormat::R32Float, None)
        }
        BuiltIn::OpaqueMip => (OPAQUE_MIP_SHADER.to_owned(), SCENE_FORMAT, None),
        #[cfg(feature = "render-subsurface")]
        BuiltIn::Subsurface(samples) => (crate::subsurface::shader(samples), SCENE_FORMAT, None),
        #[cfg(not(feature = "render-subsurface"))]
        BuiltIn::Subsurface(_) => unreachable!("render-subsurface disabled"),
        #[cfg(feature = "render-taa")]
        BuiltIn::Taa(samples) => (crate::taa::shader(samples), SCENE_FORMAT, None),
        #[cfg(not(feature = "render-taa"))]
        BuiltIn::Taa(_) => unreachable!("render-taa disabled"),
        #[cfg(feature = "render-dof")]
        BuiltIn::Dof(samples) => (crate::dof::shader(samples), SCENE_FORMAT, None),
        #[cfg(not(feature = "render-dof"))]
        BuiltIn::Dof(_) => unreachable!("render-dof disabled"),
        #[cfg(feature = "render-taa")]
        BuiltIn::MotionBlur(samples) => (crate::motion_blur::shader(samples), SCENE_FORMAT, None),
        #[cfg(not(feature = "render-taa"))]
        BuiltIn::MotionBlur(_) => unreachable!("render-taa disabled"),
        BuiltIn::Present(samples) => (present_shader(samples, &context.display), context.format, None),
        BuiltIn::BloomDown { first } => (bloom_down_shader(first), SCENE_FORMAT, None),
        // Each level up is added onto the one above.
        BuiltIn::BloomUp => (bloom_up_shader(), SCENE_FORMAT, Some(wgpu::BlendState { color: add, alpha: add })),
        BuiltIn::Decal { .. }
        | BuiltIn::Composite
        | BuiltIn::Particles { .. }
        | BuiltIn::Ribbons { .. }
        | BuiltIn::Backdrop(_) => {
            unreachable!("not a pass around the scene: {which:?}")
        }
    };
    let label = format!("{which:?}");
    let module = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&label),
        source: wgpu::ShaderSource::Wgsl(code.into()),
    });
    let mut color_targets =
        vec![Some(wgpu::ColorTargetState { format: target, blend, write_mask: wgpu::ColorWrites::ALL })];
    if matches!(which, BuiltIn::Taa(_)) {
        color_targets.push(Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }));
    }
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
            targets: &color_targets,
        }),
        multiview_mask: None,
        cache: context.cache.get(),
    })
}

/// The pipeline that lays the opaque picture into every sample of the
/// translucent pass, whose depth and stencil it leaves as they are.
pub(crate) fn backdrop_pipeline(context: &PipelineContext, samples: u32) -> wgpu::RenderPipeline {
    let module = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("backdrop"),
        source: wgpu::ShaderSource::Wgsl(backdrop_shader().into()),
    });
    context.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("backdrop"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState { read_mask: 0, write_mask: 0, ..Default::default() },
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SCENE_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: context.cache.get(),
    })
}

/// What a graph of the passes after the scene reads.
struct Inputs {
    depth: wgpu::TextureView,
    scene: wgpu::TextureView,
    marks: wgpu::TextureView,
    velocity: wgpu::TextureView,
    /// With TAA, the history texture read and the one drawn.
    history: Option<[wgpu::TextureView; 2]>,
    /// With TAA, the lock/prev-depth status texture read and the one drawn.
    lock: Option<[wgpu::TextureView; 2]>,
    /// Opaque-only lit backdrop (for Part::After) so TAA/FSR computes an automatic reactive mask
    /// over translucents, particles, ribbons, and overlays with zero extra passes.
    opaque_color: Option<wgpu::TextureView>,
}

/// Which of the passes after the scene a graph holds.
enum Part<'a> {
    /// All of them, from the scene as the passes drew it to the output: the
    /// frames without translucent surfaces.
    Whole,
    /// The opaque picture's, of a frame with translucent surfaces: the
    /// occlusion and the atmosphere into level 0 of the chain (views of each
    /// level), the nearest depth of each pixel when needed, and each level from
    /// the one above.
    Opaque(&'a [wgpu::TextureView], Option<&'a wgpu::TextureView>),
    /// The ones that read what the translucent pass resolved to.
    After,
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
    no_velocity: wgpu::Texture,
    targets: Option<Targets>,
    filtered_present: Option<FilteredPresent>,
    /// The begun frame's quality and output size, and whether it has translucent
    /// surfaces, particles or overlays.
    quality: Option<Quality>,
    size: (u32, u32),
    translucent: bool,
    /// Whether the begun frame diffuses surfaces on screen.
    subsurface: bool,
    /// How many opaque pictures have been made, to number them.
    opaques: u64,
    /// With TAA: frames jittered so far, the begun frame's jittered
    /// projection, and what the history remembers.
    jitters: u64,
    jittered: Option<(Matrix, (f64, f64))>,
    memory: crate::taa::Memory,
    /// Previous frame's unjittered view-projection matrix (tracked for both TAA and velocity motion blur).
    previous_vp: Option<Matrix>,
    /// The scene's metering and the exposure it leaves on the GPU.
    exposure: Exposure,
    /// The present pass's late bind group: the grade's two tables, and which they are.
    tables: Option<(([u64; 2], u32), wgpu::BindGroup)>,
    /// Active display tonemapper and encoded display peak (1.0 for SDR).
    tonemap: crate::tonemap::TonemapOperator,
    display_peak: f32,
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
            no_velocity: {
                let made = texture(
                    device,
                    1,
                    1,
                    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    1,
                    wgpu::TextureFormat::Rg16Float,
                    1,
                );
                queue.write_texture(
                    made.as_image_copy(),
                    &crate::renderer::velocity_sentinel_bytes(),
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: None },
                    wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                );
                made
            },
            targets: None,
            filtered_present: None,
            quality: None,
            size: (1, 1),
            translucent: false,
            subsurface: false,
            opaques: 0,
            jitters: 0,
            jittered: None,
            memory: Default::default(),
            previous_vp: None,
            exposure: Exposure::new(device, queue),
            tables: None,
            tonemap: crate::tonemap::TonemapOperator::Shoulder,
            display_peak: 1.0,
        }
    }

    pub fn set_tonemap(&mut self, tonemap: crate::tonemap::TonemapOperator, display_peak: f32) {
        self.tonemap = tonemap;
        self.display_peak = display_peak.max(1.0);
    }

    /// The next metered frame takes the exposure its scene asks for at once.
    pub fn reset_exposure(&mut self) {
        self.exposure.reset();
    }

    /// Takes in the exposures read back, and returns the latest one's stops.
    pub fn exposure_seen(&mut self) -> Option<f32> {
        self.exposure.take();
        self.exposure.latest()
    }

    /// Copies the frame's exposure out for reading back.
    pub fn exposure_finish(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.exposure.finish(encoder);
    }

    /// The frame is submitted.
    pub fn exposure_submitted(&mut self) {
        self.exposure.submitted();
    }

    /// The passes after the scene that the last frame drew, in order: the two that
    /// diffuse subsurface surfaces, if it had any; for a frame with translucent
    /// surfaces, those of the opaque picture, the backdrop the translucent pass
    /// starts with, and the passes that read its result.
    pub fn passes(&self) -> Vec<&'static str> {
        let Some(targets) = &self.targets else { return Vec::new() };
        let mut names = match (&targets.subsurface, self.subsurface) {
            (Some(diffusion), true) => diffusion.names(),
            _ => Vec::new(),
        };
        match (&targets.split, self.translucent) {
            (Some(split), true) => {
                names.extend(split.before.names());
                names.push("backdrop");
                names.extend(split.after[0].names());
            }
            _ => names.extend(targets.graphs[0].names()),
        }
        names
    }

    /// With TAA and a frame before this one: the picture the last frame
    /// resolved to (what TAA reads as its history, before exposure) and the
    /// view-projection that frame was seen through, unjittered.
    pub fn history(&self) -> Option<(wgpu::TextureView, Matrix)> {
        let histories = self.targets.as_ref()?.histories.as_ref()?;
        let previous = self.memory.previous()?;
        Some((histories[self.memory.read()].clone(), previous))
    }

    /// The begun frame's projection, jittered within its pixels, when TAA is
    /// on: the scene draws with it.
    pub fn jittered(&self) -> Option<Matrix> {
        self.jittered.map(|(projection, _)| projection)
    }

    /// Previous frame's unjittered view-projection matrix, if one has been drawn.
    pub fn previous_vp(&self) -> Option<Matrix> {
        self.previous_vp
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
    #[allow(clippy::too_many_arguments)]
    pub fn begin(
        &mut self,
        built_ins: &mut BuiltIns,
        frame: &RenderFrame,
        quality: Quality,
        size: (u32, u32),
        volume: &crate::volume::Applied,
        physical_sky: Option<&PhysicalSkyBinding>,
        extras: Extras,
    ) -> SceneTargets {
        let Extras { translucent, opaque_depth, opaque_mips, subsurface } = extras;
        #[cfg(feature = "render-subsurface")]
        let subsurface = subsurface && crate::deferred_shaders::ready_all(&["subsurface.wgsl"]);
        #[cfg(not(feature = "render-subsurface"))]
        let subsurface = {
            let _ = subsurface;
            false
        };
        self.quality = Some(quality);
        self.size = size;
        self.translucent = translucent;
        self.subsurface = subsurface;
        let occluding = cfg!(feature = "render-ssgi")
            && quality.occlusion
            && crate::deferred_shaders::ready_all(&["occlusion.wgsl"]);
        let atmosphere = quality.sky
            || quality.physical_sky
            || frame.fog.density > 0.0
            || occluding
            || frame.clouds.cover > 0.0
            || volume.serial != 0;
        // At most the pixels of a 4K screen, whatever the scale and display:
        // the multisampled targets of more would take gigabytes.
        let (width, height) = scene_size(&self.device, size, quality.scale);
        let samples = quality.msaa;
        let ssgi = cfg!(feature = "render-ssgi") && quality.ssgi && crate::deferred_shaders::ready_all(&["ssgi.wgsl"]);
        let taa = cfg!(feature = "render-taa")
            && quality.taa
            && crate::deferred_shaders::ready_all(&["taa.wgsl", "velocity.wgsl"]);
        let (taa_width, taa_height) = if taa && quality.upscale && (width < size.0 || height < size.1) {
            scene_size(&self.device, size, 1.0)
        } else {
            (width, height)
        };
        // Halving down to no smaller than a few pixels.
        let half = (taa_width.div_ceil(2), taa_height.div_ceil(2));
        let levels = if quality.bloom {
            // floor(log2(side / 4)) + 1
            half.0.min(half.1).checked_ilog2().unwrap_or(0).saturating_sub(1).clamp(1, BLOOM_LEVELS)
        } else {
            0
        };
        let dof = cfg!(feature = "render-dof")
            && quality.effects
            && frame.dof.focus_distance > 0.0
            && frame.dof.aperture > 0.0
            && frame.dof.max_blur > 0.0
            && crate::deferred_shaders::ready_all(&["dof.wgsl"]);
        let motion_blur = cfg!(feature = "render-taa")
            && quality.effects
            && frame.motion_blur.strength > 0.0
            && frame.motion_blur.max_radius > 0.0
            && crate::deferred_shaders::ready_all(&["motion_blur.wgsl", "velocity.wgsl"]);
        let key = TargetsKey {
            width,
            height,
            taa_width,
            taa_height,
            samples,
            occluding,
            atmosphere,
            ssgi,
            levels,
            taa,
            dof,
            motion_blur,
            volume: volume.serial,
            physical_sky: physical_sky.map_or(0, |b| b.serial),
        };
        let mut targets_recreated = false;
        if self.targets.as_ref().is_none_or(|targets| targets.key != key) {
            // Preserve full-resolution TAA/FSR history + lock textures and camera memory across dynamic
            // resolution scale changes (`--frame-budget` / FSR DRS) when output dimensions stay constant.
            let reusable_temporal = self.targets.take().and_then(|prev| {
                if taa
                    && prev.key.taa
                    && prev.key.taa_width == taa_width
                    && prev.key.taa_height == taa_height
                    && !frame.exposure.cut
                {
                    prev.histories.zip(prev.locks)
                } else {
                    None
                }
            });
            targets_recreated = true;
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
            let scene = texture(device, width, height, sampled | wgpu::TextureUsages::COPY_SRC, 1, SCENE_FORMAT, 1);
            let marks = depth.create_view(&wgpu::TextureViewDescriptor {
                aspect: wgpu::TextureAspect::StencilOnly,
                ..Default::default()
            });
            let view = |texture: &wgpu::Texture| texture.create_view(&Default::default());
            let velocity = (taa || motion_blur).then(|| {
                let vel = texture(device, width, height, sampled, 1, wgpu::TextureFormat::Rg16Float, 1);
                let vel_depth = texture(device, width, height, attachment, 1, wgpu::TextureFormat::Depth32Float, 1);
                (view(&vel), view(&vel_depth))
            });
            let vel_view = velocity
                .as_ref()
                .map(|(v, _)| v.clone())
                .unwrap_or_else(|| self.no_velocity.create_view(&Default::default()));
            let (kept, kept_locks) = if let Some((prev_views, prev_locks)) = reusable_temporal {
                (Some(prev_views), Some(prev_locks))
            } else {
                // With TAA, a graph for each history texture read, sharing the
                // rest; a new history starts afresh.
                self.memory = Default::default();
                self.previous_vp = None;
                #[cfg(feature = "render-taa")]
                let histories = taa.then(|| {
                    let textures = [0, 1].map(|_| crate::taa::history(device, taa_width, taa_height, SCENE_FORMAT));
                    [view(&textures[0]), view(&textures[1])]
                });
                #[cfg(not(feature = "render-taa"))]
                let histories = None;
                #[cfg(feature = "render-taa")]
                let locks = taa.then(|| {
                    let textures = [0, 1]
                        .map(|_| crate::taa::history(device, taa_width, taa_height, wgpu::TextureFormat::Rgba16Float));
                    [view(&textures[0]), view(&textures[1])]
                });
                #[cfg(not(feature = "render-taa"))]
                let locks = None;
                (histories, locks)
            };
            let mut graphs: Vec<Graph> = Vec::new();
            for read in 0..if taa { 2 } else { 1 } {
                let history = kept.as_ref().map(|views| [views[read].clone(), views[1 - read].clone()]);
                let lock = kept_locks.as_ref().map(|views| [views[read].clone(), views[1 - read].clone()]);
                let inputs = Inputs {
                    depth: depth_only(),
                    scene: view(&scene),
                    marks: marks.clone(),
                    velocity: vel_view.clone(),
                    history,
                    lock,
                    opaque_color: None,
                };
                let graph = self
                    .graph(built_ins, key, inputs, Part::Whole, volume, physical_sky, graphs.first())
                    .expect("the passes after the scene read only what earlier ones draw");
                log::debug!("passes after the scene: {:?}, {} textures", graph.names(), graph.allocations());
                graphs.push(graph);
            }
            self.targets = Some(Targets {
                key,
                histories: kept,
                locks: kept_locks,
                velocity,
                depth: view(&depth),
                depth_only: depth_only(),
                multisampled: multisampled.as_ref().map(view),
                scene: view(&scene),
                filter_picture: None,
                marks,
                graphs,
                split: None,
                subsurface: None,
            });
        } else if frame.exposure.cut {
            self.memory.cut();
            self.previous_vp = None;
        }
        if translucent
            && self
                .targets
                .as_ref()
                .unwrap()
                .split
                .as_ref()
                .is_none_or(|s| s.opaque_depth != opaque_depth || s.opaque_mips != opaque_mips)
        {
            let split = self.split(built_ins, volume, physical_sky, opaque_depth, opaque_mips);
            self.targets.as_mut().unwrap().split = Some(split);
        }
        if subsurface && self.targets.as_ref().unwrap().subsurface.is_none() {
            let diffusion = self.diffusion(built_ins);
            self.targets.as_mut().unwrap().subsurface = Some(diffusion);
        }
        // The scene's samples move within their pixels from frame to frame across an FSR-scaled Halton cycle.
        #[cfg(feature = "render-taa")]
        {
            self.jittered = taa.then(|| {
                self.jitters += 1;
                let phases = crate::taa::fsr_jitter_phase_count(width, taa_width);
                let offset = crate::taa::jitter_for(self.jitters, phases);
                (crate::taa::jittered(&frame.projection, offset, (width, height)), offset)
            });
        }
        #[cfg(not(feature = "render-taa"))]
        {
            self.jittered = None;
        }
        let current = self.targets.as_ref().unwrap();
        SceneTargets {
            size: (current.key.width, current.key.height),
            depth: current.depth.clone(),
            depth_only: current.depth_only.clone(),
            scene: current.scene.clone(),
            multisampled: current.multisampled.clone(),
            opaque: current.split.as_ref().filter(|_| translucent).map(|split| split.opaque.clone()),
            velocity: current.velocity.clone(),
            targets_recreated,
        }
    }

    /// What a frame with translucent surfaces adds to the targets: the opaque
    /// picture and its depth, the graph that makes them, the backdrop's bind
    /// group and the graphs of the passes that read what the translucent pass
    /// draws.
    fn split(
        &mut self,
        built_ins: &mut BuiltIns,
        volume: &crate::volume::Applied,
        physical_sky: Option<&PhysicalSkyBinding>,
        opaque_depth: bool,
        opaque_mips: bool,
    ) -> Split {
        let targets = self.targets.as_ref().unwrap();
        let key = targets.key;
        let (depth, scene, marks) = (targets.depth_only.clone(), targets.scene.clone(), targets.marks.clone());
        let velocity = targets
            .velocity
            .as_ref()
            .map(|(v, _)| v.clone())
            .unwrap_or_else(|| self.no_velocity.create_view(&Default::default()));
        let histories = targets.histories.clone();
        let locks = targets.locks.clone();
        let TargetsKey { width, height, samples, taa, .. } = key;
        let sampled = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        // Every level down to one texel when shaders sample `opaque_color`; level 0 alone for backdrop otherwise.
        let count = if opaque_mips { width.max(height).ilog2() + 1 } else { 1 };
        let picture = texture(&self.device, width, height, sampled, 1, SCENE_FORMAT, count);
        let levels: Vec<wgpu::TextureView> = (0..count)
            .map(|level| {
                picture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let (depth_w, depth_h) = if opaque_depth { (width, height) } else { (1, 1) };
        let nearest = texture(&self.device, depth_w, depth_h, sampled, 1, wgpu::TextureFormat::R32Float, 1)
            .create_view(&Default::default());
        let inputs = |read: usize, opaque_color: Option<wgpu::TextureView>| Inputs {
            depth: depth.clone(),
            scene: scene.clone(),
            marks: marks.clone(),
            velocity: velocity.clone(),
            history: histories.as_ref().map(|textures| [textures[read].clone(), textures[1 - read].clone()]),
            lock: locks.as_ref().map(|textures| [textures[read].clone(), textures[1 - read].clone()]),
            opaque_color,
        };
        let before = self
            .graph(
                built_ins,
                key,
                inputs(0, None),
                Part::Opaque(&levels, opaque_depth.then_some(&nearest)),
                volume,
                physical_sky,
                None,
            )
            .expect("the opaque picture's passes read only what earlier ones draw");
        log::debug!("opaque picture: {:?}, {} textures", before.names(), before.allocations());
        let mut after: Vec<Graph> = Vec::new();
        for read in 0..if taa { 2 } else { 1 } {
            let graph = self
                .graph(
                    built_ins,
                    key,
                    inputs(read, Some(levels[0].clone())),
                    Part::After,
                    volume,
                    physical_sky,
                    after.first(),
                )
                .expect("the passes after the translucent pass read only what earlier ones draw");
            log::debug!("passes after the translucent pass: {:?}, {} textures", graph.names(), graph.allocations());
            after.push(graph);
        }
        let backdrop = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("backdrop"),
            layout: &built_ins.get(BuiltIn::Backdrop(samples)).get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&levels[0]) }],
        });
        self.opaques += 1;
        let opaque =
            OpaqueTargets { serial: self.opaques, picture: picture.create_view(&Default::default()), depth: nearest };
        Split { opaque_depth, opaque_mips, opaque, backdrop, before, after }
    }

    /// The graph of the two passes that diffuse the marked surfaces of the resolved
    /// scene: along x into a texture of their own, then along y back into the scene.
    fn diffusion(&mut self, built_ins: &mut BuiltIns) -> Graph {
        let targets = self.targets.as_ref().unwrap();
        let TargetsKey { width, height, samples, .. } = targets.key;
        let mut graph = GraphBuilder::default();
        let depth = graph.import("depth", targets.depth_only.clone());
        let scene = graph.import("scene", targets.scene.clone());
        let marks = graph.import("marks", targets.marks.clone());
        let across = graph.texture("diffused across", Spec { width, height, format: SCENE_FORMAT });
        let (horizontal, vertical) = (self.uniform("subsurface x", 96), self.uniform("subsurface y", 96));
        let pass = |graph: &mut GraphBuilder, name, uniform, from, target| {
            let bindings = vec![
                Binding::Uniform(uniform),
                Binding::Texture(from),
                Binding::Texture(depth),
                Binding::Texture(marks),
            ];
            graph.pass(name, BuiltIn::Subsurface(samples), bindings, Target::Texture(target), false);
        };
        pass(&mut graph, "subsurface x", horizontal, scene, across);
        pass(&mut graph, "subsurface y", vertical, across, scene);
        graph
            .compile(&self.device, built_ins, None)
            .expect("the diffusion reads the scene and what its first pass draws")
    }

    /// For a frame with subsurface surfaces, once the opaque surfaces and their
    /// decals are drawn and resolved: blurs the scene's picture where they are,
    /// before anything else reads it.
    pub fn diffuse(
        &mut self,
        #[cfg_attr(not(feature = "render-subsurface"), allow(unused_variables))] built_ins: &mut BuiltIns,
        #[cfg_attr(not(feature = "render-subsurface"), allow(unused_variables))] encoder: &mut wgpu::CommandEncoder,
        #[cfg_attr(not(feature = "render-subsurface"), allow(unused_variables))] frame: &RenderFrame,
        #[cfg_attr(not(feature = "render-subsurface"), allow(unused_variables))] timer: &GpuTimer,
    ) {
        #[cfg(feature = "render-subsurface")]
        {
            let TargetsKey { width, height, .. } = self.targets.as_ref().unwrap().key;
            // What the scene drew with.
            let (projection, _) = self.jittered.unwrap_or((frame.projection, (0.0, 0.0)));
            let inverse = invert(&projection);
            for (name, vertical) in [("subsurface x", false), ("subsurface y", true)] {
                let uniform = self.uniform(name, 96);
                let data = crate::subsurface::uniform(&inverse, vertical, (width, height));
                self.queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&data));
            }
            let diffusion = self
                .targets
                .as_ref()
                .unwrap()
                .subsurface
                .as_ref()
                .expect("a frame with subsurface surfaces has its diffusion");
            diffusion.execute(encoder, built_ins, None, timer, Span::Subsurface, None, &mut |_, _| {});
        }
    }

    /// Reuse a resolved HDR image for the next mesh filter's binding 14.
    pub fn mesh_filter_picture(&mut self) -> wgpu::TextureView {
        let targets = self.targets.as_mut().expect("Post::begin comes first");
        targets
            .filter_picture
            .get_or_insert_with(|| {
                texture(
                    &self.device,
                    targets.key.width,
                    targets.key.height,
                    wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                    1,
                    SCENE_FORMAT,
                    1,
                )
                .create_view(&Default::default())
            })
            .clone()
    }

    /// Draws the opaque picture into every sample of the translucent pass, which
    /// then draws over it.
    pub fn draw_backdrop(&self, built_ins: &mut BuiltIns, pass: &mut wgpu::RenderPass<'_>) {
        let targets = self.targets.as_ref().expect("Post::begin comes first");
        let split = targets.split.as_ref().expect("a frame with translucent surfaces has its opaque picture");
        pass.set_pipeline(built_ins.get(BuiltIn::Backdrop(targets.key.samples)));
        pass.set_bind_group(0, &split.backdrop, &[]);
        pass.draw(0..3, 0..1);
    }

    /// The passes after the scene for frames of `key`, as `part` says, reading
    /// the scene's depth, its picture and its stencil marks and, for the last
    /// part, ending on the output. With TAA, `history` is the texture it reads
    /// and the one it draws.
    #[allow(clippy::too_many_arguments)]
    fn graph(
        &mut self,
        built_ins: &mut BuiltIns,
        key: TargetsKey,
        inputs: Inputs,
        part: Part<'_>,
        volume: &crate::volume::Applied,
        physical_sky: Option<&PhysicalSkyBinding>,
        sharing: Option<&Graph>,
    ) -> anyhow::Result<Graph> {
        let TargetsKey {
            width,
            height,
            taa_width,
            taa_height,
            samples,
            occluding,
            atmosphere,
            ssgi,
            levels,
            dof,
            motion_blur,
            ..
        } = key;
        let full = |format: wgpu::TextureFormat| Spec { width, height, format };
        let full_post = |format: wgpu::TextureFormat| Spec { width: taa_width, height: taa_height, format };
        let sampler = Binding::Sampler(self.sampler.clone());
        let mut graph = GraphBuilder::default();
        let depth = graph.import("depth", inputs.depth);
        let scene = graph.import("scene", inputs.scene);
        let marks = graph.import("marks", inputs.marks);
        let velocity = graph.import("velocity", inputs.velocity);
        let mut lit = scene;
        // The sky, the occlusion and the mist go onto the opaque picture alone:
        // a frame with translucent surfaces draws them into level 0 of its chain,
        // whatever the quality asks, as that is what the translucent pass starts from.
        if !matches!(part, Part::After) {
            let occlusion = if occluding {
                let occlusion = graph.texture(
                    "occlusion",
                    Spec { width: width.div_ceil(2), height: height.div_ceil(2), format: wgpu::TextureFormat::R8Unorm },
                );
                let uniform = Binding::Uniform(self.uniform("occlusion", 160));
                let bindings = vec![uniform, Binding::Texture(depth)];
                graph.pass("occlusion", BuiltIn::Occlusion(samples), bindings, Target::Texture(occlusion), false);
                occlusion
            } else {
                graph.import("unoccluded", self.unoccluded.create_view(&Default::default()))
            };
            if atmosphere || matches!(part, Part::Opaque(..)) {
                let composed = match part {
                    Part::Opaque(chain, _) if !ssgi => graph.import("opaque", chain[0].clone()),
                    _ => graph.texture("composed", full(SCENE_FORMAT)),
                };
                let uniform = Binding::Uniform(self.uniform("atmosphere", 224));
                let froxels = graph.import("froxels", volume.texture.clone());
                let phys_sky_view = physical_sky
                    .map(|b| b.sky_view.clone())
                    .unwrap_or_else(|| self.unbloomed.create_view(&Default::default()));
                let phys_clouds_view = physical_sky
                    .and_then(|b| b.clouds.clone())
                    .unwrap_or_else(|| self.unbloomed.create_view(&Default::default()));
                let phys_trans_view = physical_sky
                    .map(|b| b.transmittance.clone())
                    .unwrap_or_else(|| self.unoccluded.create_view(&Default::default()));
                let phys_sky = graph.import("physical_sky_lut", phys_sky_view);
                let phys_clouds = graph.import("physical_clouds", phys_clouds_view);
                let phys_trans = graph.import("physical_transmittance_lut", phys_trans_view);
                let bindings = vec![
                    uniform,
                    Binding::Texture(scene),
                    Binding::Texture(depth),
                    Binding::Texture(occlusion),
                    sampler.clone(),
                    Binding::Uniform(volume.uniform.clone()),
                    Binding::Texture(froxels),
                    Binding::Texture(phys_sky),
                    Binding::Texture(phys_clouds),
                    Binding::Texture(phys_trans),
                ];
                graph.pass("atmosphere", BuiltIn::Atmosphere(samples), bindings, Target::Texture(composed), false);
                lit = composed;
            }
            if ssgi {
                let bounced = match part {
                    Part::Opaque(chain, _) => graph.import("opaque", chain[0].clone()),
                    _ => graph.texture("ssgi", full(SCENE_FORMAT)),
                };
                let uniform = Binding::Uniform(self.uniform("ssgi", 160));
                let bindings = vec![uniform, Binding::Texture(lit), Binding::Texture(depth), sampler.clone()];
                graph.pass("ssgi", BuiltIn::Ssgi(samples), bindings, Target::Texture(bounced), false);
                lit = bounced;
            }
        }
        // The nearest depth of each pixel, and each level of the picture from the
        // one above it, only when active translucent passes read them.
        if let Part::Opaque(chain, nearest) = part {
            if let Some(nearest) = nearest {
                let nearest = graph.import("opaque depth", nearest.clone());
                graph.pass(
                    "opaque depth",
                    BuiltIn::OpaqueDepth(samples),
                    vec![Binding::Texture(depth)],
                    Target::Texture(nearest),
                    false,
                );
            }
            for level in &chain[1..] {
                let level = graph.import("opaque", level.clone());
                let bindings = vec![Binding::Texture(lit), sampler.clone()];
                graph.pass("opaque mip", BuiltIn::OpaqueMip, bindings, Target::Texture(level), false);
                lit = level;
            }
            return graph.compile(&self.device, built_ins, sharing);
        }
        // TAA / FSR 2.2/3.1-class temporal super resolution blends the lit picture into the history,
        // deriving a 3x3 dilated auto-reactive mask against `opaque_color`, applying pre-exposure,
        // checking geometric depth disocclusion, and maintaining dedicated thin-feature lock status.
        if let (Some([read, drawn]), Some([lock_read, lock_drawn])) = (inputs.history, inputs.lock) {
            let (read, drawn) = (graph.import("history", read), graph.import("history", drawn));
            let (lock_read, lock_drawn) = (graph.import("taa_lock", lock_read), graph.import("taa_lock", lock_drawn));
            let opaque_reactive = match inputs.opaque_color {
                Some(view) => graph.import("opaque_reactive", view),
                None => lit,
            };
            let uniform = Binding::Uniform(self.uniform("taa", 176));
            let exposure = Binding::Storage(self.exposure.state().clone());
            let bindings = vec![
                uniform,
                Binding::Texture(lit),
                Binding::Texture(read),
                Binding::Texture(depth),
                sampler.clone(),
                Binding::Texture(velocity),
                Binding::Texture(opaque_reactive),
                exposure,
                Binding::Texture(lock_read),
            ];
            graph.pass("taa", BuiltIn::Taa(samples), bindings, Target::Pair(drawn, lock_drawn), false);
            lit = drawn;
        }
        if dof {
            let focused = graph.texture("dof", full_post(SCENE_FORMAT));
            let uniform = Binding::Uniform(self.uniform("dof", 96));
            let bindings = vec![uniform, Binding::Texture(lit), Binding::Texture(depth), sampler.clone()];
            graph.pass("dof", BuiltIn::Dof(samples), bindings, Target::Texture(focused), false);
            lit = focused;
        }
        if motion_blur {
            let smeared = graph.texture("motion blur", full_post(SCENE_FORMAT));
            let uniform = Binding::Uniform(self.uniform("motion blur", 160));
            let bindings = vec![
                uniform,
                Binding::Texture(lit),
                Binding::Texture(velocity),
                Binding::Texture(depth),
                sampler.clone(),
            ];
            graph.pass("motion blur", BuiltIn::MotionBlur(samples), bindings, Target::Texture(smeared), false);
            lit = smeared;
        }
        // Auto exposure meters the lit picture, before the bloom (exposure.rs).
        graph.hook("meter", lit);
        let exposure = Binding::Storage(self.exposure.state().clone());
        // Bloom halves the picture level by level, then adds each level onto
        // the one above, back up to half the post-TAA picture's size.
        let half = (taa_width.div_ceil(2), taa_height.div_ceil(2));
        let bloom: Vec<Handle> = (0..levels)
            .map(|level| {
                let size = (level_size(half.0, level), level_size(half.1, level));
                graph.texture("bloom", Spec { width: size.0, height: size.1, format: SCENE_FORMAT })
            })
            .collect();
        for (index, &level) in bloom.iter().enumerate() {
            let (first, from) = if index == 0 { (true, lit) } else { (false, bloom[index - 1]) };
            let mut bindings = vec![Binding::Texture(from), sampler.clone()];
            if first {
                bindings.push(exposure.clone());
            }
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
        let grade = Binding::Uniform(self.uniform("grade", size_of::<grade::GradeUniform>() as u64));
        let bindings = vec![
            uniform,
            Binding::Texture(lit),
            sampler,
            Binding::Texture(marks),
            Binding::Texture(glow),
            grade,
            exposure,
        ];
        graph.pass("present", BuiltIn::Present(samples), bindings, Target::Output, false);
        graph.compile(&self.device, built_ins, sharing)
    }

    /// The present pass's late bind group (group 1) for these look-up tables, made
    /// again only when they change.
    fn tables_group(&mut self, built_ins: &mut BuiltIns, samples: u32, tables: &[Option<Lut>; 2]) -> wgpu::BindGroup {
        let key = [tables[0].as_ref().map_or(0, |table| table.id), tables[1].as_ref().map_or(0, |table| table.id)];
        let key = (key, samples);
        if self.tables.as_ref().is_none_or(|(kept, _)| *kept != key) {
            let layout = built_ins.get(BuiltIn::Present(samples)).get_bind_group_layout(1);
            let blank = self.unbloomed.create_view(&Default::default());
            let view = |table: &Option<Lut>| table.as_ref().map_or_else(|| blank.clone(), |table| table.view.clone());
            let (a, b) = (view(&tables[0]), view(&tables[1]));
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("grade tables"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&a) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&b) },
                ],
            });
            self.tables = Some((key, group));
        }
        self.tables.as_ref().unwrap().1.clone()
    }

    /// Encodes the ordinary path without additional targets or passes.
    pub fn finish(
        &mut self,
        built_ins: &mut BuiltIns,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        output: &wgpu::TextureView,
        timer: &GpuTimer,
        tables: &[Option<Lut>; 2],
    ) {
        self.finish_base(built_ins, encoder, frame, output, timer, false, tables);
    }

    /// Writes the uniforms of the occlusion and the atmosphere passes, which
    /// work out the opaque surfaces' picture, for the scene as it was drawn.
    fn write_opaque_uniforms(&mut self, frame: &RenderFrame) {
        let quality = self.quality.expect("Post::begin comes first");
        let TargetsKey { width, height, occluding, atmosphere, ssgi, .. } = self.targets.as_ref().unwrap().key;
        // What the scene drew with.
        let (projection, _) = self.jittered.unwrap_or((frame.projection, (0.0, 0.0)));
        if occluding {
            let uniform = self.uniform("occlusion", 160);
            let reversed_z = crate::camera_precision::detect_depth_mode(&frame.projection)
                == Some(crate::camera_precision::DepthMode::ReversedZ);
            let ao_w = width.div_ceil(2) as f32;
            let ao_h = height.div_ceil(2) as f32;
            let mut data = [0.0f32; 40];
            data[..16].copy_from_slice(&projection);
            data[16..32].copy_from_slice(&invert(&projection));
            data[32..].copy_from_slice(&[
                OCCLUSION_RADIUS,
                OCCLUSION_POWER,
                OCCLUSION_BIAS,
                if reversed_z { 1.0 } else { 0.0 },
                width as f32,
                height as f32,
                width as f32 / ao_w,
                height as f32 / ao_h,
            ]);
            self.queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&data));
        }
        if atmosphere || self.translucent {
            let uniform = self.uniform("atmosphere", 224);
            let data = atmosphere_of(frame, &projection, &quality, width, height);
            self.queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&data));
        }
        #[cfg(feature = "render-ssgi")]
        if ssgi {
            let uniform = self.uniform("ssgi", 160);
            let data = crate::ssgi::uniform(&projection, (width, height));
            self.queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&data));
        }
        #[cfg(not(feature = "render-ssgi"))]
        let _ = ssgi;
    }

    /// For a frame with translucent surfaces, once the opaque surfaces are
    /// drawn and resolved: the sky, the occlusion and the mist on them alone,
    /// their nearest depth and the chain of levels, ready for the translucent
    /// pass.
    pub fn opaque(
        &mut self,
        built_ins: &mut BuiltIns,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        timer: &GpuTimer,
    ) {
        self.write_opaque_uniforms(frame);
        let split = self
            .targets
            .as_ref()
            .unwrap()
            .split
            .as_ref()
            .expect("a frame with translucent surfaces has its opaque picture");
        split.before.execute(encoder, built_ins, None, timer, Span::Opaque, None, &mut |_, _| {});
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
        tables: &[Option<Lut>; 2],
    ) {
        let quality = self.quality.expect("Post::begin comes first");
        let size = self.size;
        let TargetsKey { width, height, taa_width, taa_height, levels, taa, dof, motion_blur, .. } =
            self.targets.as_ref().unwrap().key;
        // The opaque picture of a frame with translucent surfaces was worked out
        // before the translucent pass.
        if !self.translucent {
            self.write_opaque_uniforms(frame);
        }
        // What the scene drew with.
        let (projection, offset) = self.jittered.unwrap_or((frame.projection, (0.0, 0.0)));
        // The graph reading the history this frame reads.
        let graph = if taa { self.memory.read() } else { 0 };
        #[cfg(feature = "render-taa")]
        if taa {
            let uniform = self.uniform("taa", 176);
            let reversed_z = crate::camera_precision::detect_depth_mode(&frame.projection)
                == Some(crate::camera_precision::DepthMode::ReversedZ);
            let data = self.memory.uniform(
                &frame.view,
                &frame.projection,
                (&projection, offset),
                (width, height),
                (taa_width, taa_height),
                reversed_z,
                self.translucent,
            );
            self.queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&data));
        }
        #[cfg(not(feature = "render-taa"))]
        let _ = (&projection, offset);
        #[cfg(feature = "render-dof")]
        if dof {
            let uniform = self.uniform("dof", 96);
            let data = crate::dof::uniform(&projection, &frame.dof, (taa_width, taa_height));
            self.queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&data));
        }
        #[cfg(not(feature = "render-dof"))]
        let _ = dof;
        #[cfg(feature = "render-taa")]
        if motion_blur {
            let uniform = self.uniform("motion blur", 160);
            let data = crate::motion_blur::uniform(
                &frame.view,
                (&projection, offset),
                self.previous_vp,
                &frame.motion_blur,
                (taa_width, taa_height),
            );
            self.queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&data));
        }
        #[cfg(not(feature = "render-taa"))]
        let _ = motion_blur;
        let uniform = self.uniform("present", 64);
        let (output_width, output_height) = (size.0 as f32, size.1 as f32);
        // EASU spatial upscaling now coexists cleanly with FXAA (FXAA uses cheap bilinear edge probes
        // via `fetch_fast` and reserves 12-tap EASU reconstruction for the center & final shifted read).
        let use_easu = taa_width < size.0 || taa_height < size.1;
        self.queue.write_buffer(
            &uniform,
            0,
            bytemuck::cast_slice(&[
                1.0 / output_width,
                1.0 / output_height,
                if quality.effects && !filtered { frame.blur } else { 0.0 },
                if use_easu { 1.0 } else { 0.0 },
                1.0 / taa_width as f32,
                1.0 / taa_height as f32,
                if quality.fxaa { 1.0 } else { 0.0 },
                0.0,
                // Screen heights, as texture coordinates.
                if quality.effects && !filtered { frame.aberration.x * output_height / output_width } else { 0.0 },
                if quality.effects && !filtered { frame.aberration.y } else { 0.0 },
                width as f32,
                height as f32,
                if levels > 0 { BLOOM_INTENSITY / levels as f32 } else { 0.0 },
                1.0 - BLOOM_INTENSITY,
                0.0,
                0.0f32,
            ]),
        );
        // The exposure, fixed or metered, and the grade, for the present pass.
        self.exposure.begin(&frame.exposure);
        let uniform = self.uniform("grade", size_of::<grade::GradeUniform>() as u64);
        self.queue.write_buffer(
            &uniform,
            0,
            bytemuck::bytes_of(&grade::uniform_with_tonemap(&frame.grade, tables, self.tonemap, self.display_peak)),
        );
        let late = self.tables_group(built_ins, self.targets.as_ref().unwrap().key.samples, tables);
        let (targets, exposure) = (self.targets.as_ref().unwrap(), &mut self.exposure);
        let graphs = match &targets.split {
            Some(split) if self.translucent => &split.after,
            _ => &targets.graphs,
        };
        graphs[graph].execute(
            encoder,
            built_ins,
            Some(output),
            timer,
            Span::Post,
            Some(&late),
            &mut |picture, encoder| {
                exposure.meter(encoder, picture, (taa_width, taa_height));
            },
        );
        self.previous_vp = Some(multiply(&frame.projection, &frame.view));
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
        let ungraded = self.uniform("ungraded", size_of::<grade::GradeUniform>() as u64);
        let late = self.tables_group(built_ins, self.targets.as_ref().unwrap().key.samples, &[None, None]);
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
                    // Neither is read: the picture is shown already.
                    wgpu::BindGroupEntry { binding: 5, resource: ungraded.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.exposure.state().as_entire_binding() },
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
        pass.set_bind_group(1, &late, &[]);
        pass.draw(0..3, 0..1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_fill_every_hole() {
        let fz = crate::camera_precision::DepthMode::ForwardZ;
        let rz = crate::camera_precision::DepthMode::ReversedZ;
        for code in [
            atmosphere_shader(1, fz),
            atmosphere_shader(4, fz),
            atmosphere_shader(1, rz),
            occlusion_shader(1),
            occlusion_shader(4),
            crate::ssgi::shader(1),
            crate::ssgi::shader(4),
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
            crate::dof::shader(1),
            crate::dof::shader(4),
            crate::motion_blur::shader(1),
            crate::motion_blur::shader(4),
            opaque_depth_shader(1, fz),
            opaque_depth_shader(4, fz),
            opaque_depth_shader(1, rz),
            backdrop_shader(),
        ] {
            assert!(!code.contains("${"), "{code}");
        }
        assert!(atmosphere_shader(4, fz).contains("empty += 0.2500;"));
        assert!(atmosphere_shader(1, rz).contains("depth <= 0.0"));
        assert!(occlusion_shader(4).contains("gtao_arc"));
        assert!(
            opaque_depth_shader(4, fz).contains("texture_depth_multisampled_2d")
                && opaque_depth_shader(4, fz).contains("sample < 4")
        );
        assert!(opaque_depth_shader(1, rz).contains("max(nearest"));
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "post/fxaa_tests.rs"]
mod fxaa_tests;
