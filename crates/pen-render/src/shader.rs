// Material shaders (WGSL assets) and the pipelines they draw with, which the
// loader's threads compile in the background: the variants a shader's loaded
// materials draw with, for the sample counts frames draw at.
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use anyhow::{Result, bail};

use crate::asset_store::MESH_VERTEX_FLOATS;
use crate::pipeline_cache::PipelineCache;
use crate::post::SCENE_FORMAT;

/// What compiling a shader's pipelines needs; cheap to clone, shareable
/// between threads.
#[derive(Clone)]
pub(crate) struct PipelineContext {
    pub device: wgpu::Device,
    /// The scene uniform's group, then the draw's.
    pub layout: wgpu::PipelineLayout,
    /// The layout every sprite pipeline shares.
    pub sprite_layout: wgpu::PipelineLayout,
    /// The groups of those layouts.
    pub scene_group: wgpu::BindGroupLayout,
    pub draw_group: wgpu::BindGroupLayout,
    pub sprite_group: wgpu::BindGroupLayout,
    /// The output (swapchain) format sprites blend over.
    pub format: wgpu::TextureFormat,
    /// Where compiled pipelines are kept between runs, if anywhere.
    pub cache: PipelineCache,
}

/// What error scopes caught: Ok once they come back empty, else the
/// message. Natively it is ready at once; in the browser it settles later.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type Check = Pin<Box<dyn Future<Output = Result<(), String>> + Send>>;
#[cfg(target_arch = "wasm32")]
pub(crate) type Check = Pin<Box<dyn Future<Output = Result<(), String>>>>;

/// Runs `work` inside error scopes, popped before returning so that other
/// work cannot land in them, and returns what it made with the check of
/// what they caught: the allocation failure over follow-on "invalid
/// resource" errors.
pub(crate) fn scoped<T>(device: &wgpu::Device, work: impl FnOnce() -> T) -> (T, Check) {
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let value = work();
    // Popped in reverse.
    let validation = validation.pop();
    let memory = memory.pop();
    let internal = internal.pop();
    let check = async move {
        let (validation, memory, internal) = (validation.await, memory.await, internal.await);
        match memory.or(validation).or(internal) {
            Some(error) => Err(error.to_string()),
            None => Ok(()),
        }
    };
    (value, Box::pin(check))
}

/// `work`'s result once its error scopes come back empty.
pub(crate) async fn checked<T>(device: &wgpu::Device, work: impl FnOnce() -> T) -> Result<T, String> {
    let (value, check) = scoped(device, work);
    check.await.map(|()| value)
}

/// The check's outcome if it has settled, without waiting for it.
pub(crate) fn settled(check: &mut Check) -> Option<Result<(), String>> {
    match check.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(result) => Some(result),
        Poll::Pending => None,
    }
}

/// A WGSL module, or the compiler's errors.
pub(crate) async fn compile_shader(device: &wgpu::Device, bytes: &[u8]) -> Result<(wgpu::ShaderModule, String)> {
    let code = std::str::from_utf8(bytes)?.to_owned();
    let module = checked(device, || {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(code.as_str().into()),
        })
    })
    .await;
    match module {
        Ok(module) => Ok((module, code)),
        Err(message) => bail!(message),
    }
}

/// Each `@vertex fn NAME` in the code (`/@vertex\s+fn\s+NAME\b/`): its
/// name and the code after it.
fn vertex_entry_points(code: &str) -> impl Iterator<Item = (&str, &str)> {
    code.match_indices("@vertex").filter_map(|(at, marker)| {
        let after = &code[at + marker.len()..];
        let after_fn = after.trim_start();
        let named = after_fn.strip_prefix("fn").filter(|_| after_fn.len() < after.len())?;
        let name = named.trim_start();
        if name.len() == named.len() {
            return None;
        }
        let end = name.find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(name.len());
        Some((&name[..end], &name[end..]))
    })
}

/// Whether a shader draws sprites (its `vertex` entry point takes only
/// builtins), or None if it has no `vertex` entry point.
pub(crate) fn draws_sprites(code: &str) -> Option<bool> {
    let (_, rest) = vertex_entry_points(code).find(|&(name, _)| name == "vertex")?;
    let inputs = rest.trim_start().strip_prefix('(')?;
    let inputs = &inputs[..inputs.find(')')?];
    Some(inputs.split(',').all(|input| input.trim().is_empty() || input.trim().starts_with("@builtin")))
}

/// Whether it has a `vertex_instanced` entry point.
pub(crate) fn has_instancing(code: &str) -> bool {
    vertex_entry_points(code).any(|(name, _)| name == "vertex_instanced")
}

/// Which of a shader's pipelines: opaque surfaces (alone, or instanced),
/// translucent ones blended over them, and overlays drawn once dimmed where
/// the scene hides them and once at full strength where it does not.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Variants(u8);

const VARIANT_NAMES: [(Variants, &str); 5] = [
    (Variants::OPAQUE, "opaque"),
    (Variants::TRANSLUCENT, "translucent"),
    (Variants::OCCLUDED, "occluded"),
    (Variants::VISIBLE, "visible"),
    (Variants::INSTANCED, "instanced"),
];

impl std::fmt::Debug for Variants {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> =
            VARIANT_NAMES.iter().filter(|(variant, _)| self.contains(*variant)).map(|(_, name)| *name).collect();
        write!(formatter, "{}", if names.is_empty() { "none".to_owned() } else { names.join("+") })
    }
}

impl Variants {
    pub const NONE: Variants = Variants(0);
    pub const OPAQUE: Variants = Variants(1);
    pub const TRANSLUCENT: Variants = Variants(2);
    pub const OCCLUDED: Variants = Variants(4);
    pub const VISIBLE: Variants = Variants(8);
    pub const INSTANCED: Variants = Variants(16);

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    pub fn contains(self, other: Variants) -> bool {
        self.0 & other.0 == other.0
    }
    pub fn without(self, other: Variants) -> Variants {
        Variants(self.0 & !other.0)
    }
    /// Each variant it has, alone.
    pub fn each(self) -> impl Iterator<Item = Variants> {
        VARIANT_NAMES.into_iter().map(|(variant, _)| variant).filter(move |&variant| self.contains(variant))
    }
}

impl std::ops::BitOr for Variants {
    type Output = Variants;
    fn bitor(self, other: Variants) -> Variants {
        Variants(self.0 | other.0)
    }
}

impl std::ops::BitOrAssign for Variants {
    fn bitor_assign(&mut self, other: Variants) {
        self.0 |= other.0;
    }
}

/// Some of a shader's pipelines for one sample count.
#[derive(Clone, Default)]
pub(crate) struct Pipelines {
    pub opaque: Option<wgpu::RenderPipeline>,
    pub translucent: Option<wgpu::RenderPipeline>,
    pub occluded: Option<wgpu::RenderPipeline>,
    pub visible: Option<wgpu::RenderPipeline>,
    pub instanced: Option<wgpu::RenderPipeline>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Layer {
    Opaque,
    Translucent,
    Occluded,
    Visible,
}

impl Pipelines {
    /// The variants it has.
    pub fn variants(&self) -> Variants {
        [
            (&self.opaque, Variants::OPAQUE),
            (&self.translucent, Variants::TRANSLUCENT),
            (&self.occluded, Variants::OCCLUDED),
            (&self.visible, Variants::VISIBLE),
            (&self.instanced, Variants::INSTANCED),
        ]
        .into_iter()
        .filter(|(pipeline, _)| pipeline.is_some())
        .fold(Variants::NONE, |variants, (_, variant)| variants | variant)
    }

    /// Takes `other`'s pipelines in where it has none.
    pub fn merge(&mut self, other: Pipelines) {
        for (mine, theirs) in [
            (&mut self.opaque, other.opaque),
            (&mut self.translucent, other.translucent),
            (&mut self.occluded, other.occluded),
            (&mut self.visible, other.visible),
            (&mut self.instanced, other.instanced),
        ] {
            if mine.is_none() {
                *mine = theirs;
            }
        }
    }
}

const MESH_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 2 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 32, shader_location: 8 },
];
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 3 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 4 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 5 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 6 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 64, shader_location: 7 },
];

/// The buffers a mesh pipeline reads: the mesh's vertices and, instanced,
/// each instance's transform and parameters.
fn vertex_buffers(instanced: bool) -> Vec<Option<wgpu::VertexBufferLayout<'static>>> {
    let mut buffers = vec![Some(wgpu::VertexBufferLayout {
        array_stride: (MESH_VERTEX_FLOATS * 4) as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &MESH_ATTRIBUTES,
    })];
    if instanced {
        buffers.push(Some(wgpu::VertexBufferLayout {
            array_stride: 80,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_ATTRIBUTES,
        }));
    }
    buffers
}

// Surfaces that write depth stamp their material's aberration mark into the
// stencil, which the last pass reads.
fn stamping(depth_write: bool) -> wgpu::StencilFaceState {
    wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Always,
        pass_op: if depth_write { wgpu::StencilOperation::Replace } else { wgpu::StencilOperation::Keep },
        ..Default::default()
    }
}

/// One of a shader's scene pipelines.
fn variant_of(
    context: &PipelineContext,
    module: &wgpu::ShaderModule,
    samples: u32,
    depth_write: bool,
    depth_compare: wgpu::CompareFunction,
    blend: Option<wgpu::BlendComponent>,
    instanced: bool,
) -> wgpu::RenderPipeline {
    let buffers = vertex_buffers(instanced);
    context.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&context.layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(if instanced { "vertex_instanced" } else { "vertex" }),
            compilation_options: Default::default(),
            buffers: &buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            front_face: wgpu::FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: Some(depth_write),
            depth_compare: Some(depth_compare),
            stencil: wgpu::StencilState {
                front: stamping(depth_write),
                back: stamping(depth_write),
                read_mask: !0,
                write_mask: !0,
            },
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SCENE_FORMAT,
                blend: blend.map(|component| wgpu::BlendState { color: component, alpha: component }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: context.cache.get(),
    })
}

/// The `variants` of a shader's pipelines for `samples` per pixel (its
/// instanced one only if it has a `vertex_instanced` entry point).
pub(crate) fn pipelines(
    context: &PipelineContext,
    module: &wgpu::ShaderModule,
    instancing: bool,
    samples: u32,
    variants: Variants,
) -> Pipelines {
    use wgpu::{BlendFactor, BlendOperation, CompareFunction};
    let make = |variant: Variants, depth_write: bool, depth_compare, blend, instanced: bool| {
        variants
            .contains(variant)
            .then(|| variant_of(context, module, samples, depth_write, depth_compare, blend, instanced))
    };
    Pipelines {
        opaque: make(Variants::OPAQUE, true, CompareFunction::Less, None, false),
        translucent: make(
            Variants::TRANSLUCENT,
            false,
            CompareFunction::Less,
            Some(wgpu::BlendComponent {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            }),
            false,
        ),
        // The blend constant is OCCLUDED_STRENGTH.
        occluded: make(
            Variants::OCCLUDED,
            false,
            CompareFunction::Greater,
            Some(wgpu::BlendComponent {
                src_factor: BlendFactor::Constant,
                dst_factor: BlendFactor::OneMinusConstant,
                operation: BlendOperation::Add,
            }),
            false,
        ),
        visible: make(Variants::VISIBLE, true, CompareFunction::LessEqual, None, false),
        instanced: if instancing { make(Variants::INSTANCED, true, CompareFunction::Less, None, true) } else { None },
    }
}

/// A shader's depth-only pipeline into the shadow map.
pub(crate) fn shadow_pipeline(
    context: &PipelineContext,
    module: &wgpu::ShaderModule,
    instanced: bool,
) -> wgpu::RenderPipeline {
    let buffers = vertex_buffers(instanced);
    context.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow"),
        layout: Some(&context.layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(if instanced { "vertex_instanced" } else { "vertex" }),
            compilation_options: Default::default(),
            buffers: &buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            front_face: wgpu::FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
        }),
        multisample: Default::default(),
        fragment: None,
        multiview_mask: None,
        cache: context.cache.get(),
    })
}

/// A sprite shader's pipeline, blending over the output format.
pub(crate) fn sprite_pipeline(context: &PipelineContext, module: &wgpu::ShaderModule) -> wgpu::RenderPipeline {
    use wgpu::{BlendComponent, BlendFactor, BlendOperation};
    context.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sprite"),
        layout: Some(&context.sprite_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: context.format,
                blend: Some(wgpu::BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::SrcAlpha,
                        dst_factor: BlendFactor::OneMinusSrcAlpha,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::OneMinusSrcAlpha,
                        operation: BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: context.cache.get(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_sprite_shaders_from_mesh_shaders() {
        let sprite =
            "@vertex fn vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> Corner {";
        let mesh = "@vertex fn vertex(input: VertexInput) -> VertexOutput {";
        assert_eq!(draws_sprites(sprite), Some(true));
        assert_eq!(draws_sprites(mesh), Some(false));
        assert_eq!(draws_sprites("@vertex fn vertex_instanced(input: I) {}"), None);
        assert!(has_instancing("@vertex\nfn vertex_instanced(input: I) {}"));
        assert!(!has_instancing("@vertex fn vertex_instancedx(input: I) {}"));
        assert!(!has_instancing("@vertexfn vertex_instanced(input: I) {}"));
        assert_eq!(
            draws_sprites("@vertex fn vertex_instanced(i: I) {}\n@vertex fn vertex(@builtin(vertex_index) v: u32) {}"),
            Some(true)
        );
    }
}
