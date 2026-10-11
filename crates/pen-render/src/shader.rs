// Material shaders (WGSL assets) and the pipelines they draw with, which the
// loader's threads compile in the background as the TS compiles them with
// `createRenderPipelineAsync`, and the renderer's thread on first use as it
// does with `createRenderPipeline`.
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use anyhow::{Result, bail};

use crate::asset_store::MESH_VERTEX_FLOATS;
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
    /// The output (swapchain) format sprites blend over.
    pub format: wgpu::TextureFormat,
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

/// The parameters of the shader's `@vertex fn vertex(...)`, as the TS's
/// `/@vertex\s+fn\s+vertex\s*\(([^)]*)\)/` finds them.
fn vertex_inputs<'a>(code: &'a str, name: &str) -> Option<&'a str> {
    let mut rest = code;
    while let Some(at) = rest.find("@vertex") {
        let after = &rest[at + "@vertex".len()..];
        rest = after;
        let trimmed = after.trim_start();
        if trimmed.len() == after.len() {
            continue;
        }
        let Some(after_fn) = trimmed.strip_prefix("fn") else { continue };
        let named = after_fn.trim_start();
        if named.len() == after_fn.len() {
            continue;
        }
        let Some(after_name) = named.strip_prefix(name) else { continue };
        let open = after_name.trim_start();
        let Some(inputs) = open.strip_prefix('(') else { continue };
        let end = inputs.find(')')?;
        return Some(&inputs[..end]);
    }
    None
}

/// Whether a shader draws sprites (its vertex stage takes only builtins),
/// or None if it has no `vertex` entry point.
pub(crate) fn draws_sprites(code: &str) -> Option<bool> {
    let inputs = vertex_inputs(code, "vertex")?;
    Some(inputs.split(',').all(|input| input.trim().is_empty() || input.trim().starts_with("@builtin")))
}

/// Whether it has a `vertex_instanced` entry point
/// (`/@vertex\s+fn\s+vertex_instanced\b/`).
pub(crate) fn has_instancing(code: &str) -> bool {
    let mut rest = code;
    while let Some(at) = rest.find("@vertex") {
        rest = &rest[at + "@vertex".len()..];
        let trimmed = rest.trim_start();
        if trimmed.len() == rest.len() {
            continue;
        }
        let Some(after_fn) = trimmed.strip_prefix("fn") else { continue };
        let named = after_fn.trim_start();
        if named.len() == after_fn.len() {
            continue;
        }
        if let Some(after) = named.strip_prefix("vertex_instanced")
            && !after.starts_with(|c: char| c.is_alphanumeric() || c == '_')
        {
            return true;
        }
    }
    false
}

/// A shader's pipelines: opaque surfaces, translucent ones blended over them,
/// and overlays drawn once dimmed where the scene hides them and once at full
/// strength where it does not.
#[derive(Clone)]
pub(crate) struct Pipelines {
    pub opaque: wgpu::RenderPipeline,
    pub translucent: wgpu::RenderPipeline,
    pub occluded: wgpu::RenderPipeline,
    pub visible: wgpu::RenderPipeline,
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
    pub fn layer(&self, layer: Layer) -> &wgpu::RenderPipeline {
        match layer {
            Layer::Opaque => &self.opaque,
            Layer::Translucent => &self.translucent,
            Layer::Occluded => &self.occluded,
            Layer::Visible => &self.visible,
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
pub(crate) fn variant(
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
        cache: None,
    })
}

/// A shader's pipelines for `samples` per pixel.
pub(crate) fn pipelines(context: &PipelineContext, module: &wgpu::ShaderModule, instancing: bool, samples: u32) -> Pipelines {
    use wgpu::{BlendFactor, BlendOperation, CompareFunction};
    Pipelines {
        opaque: variant(context, module, samples, true, CompareFunction::Less, None, false),
        translucent: variant(
            context,
            module,
            samples,
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
        occluded: variant(
            context,
            module,
            samples,
            false,
            CompareFunction::Greater,
            Some(wgpu::BlendComponent {
                src_factor: BlendFactor::Constant,
                dst_factor: BlendFactor::OneMinusConstant,
                operation: BlendOperation::Add,
            }),
            false,
        ),
        visible: variant(context, module, samples, true, CompareFunction::LessEqual, None, false),
        instanced: instancing.then(|| variant(context, module, samples, true, CompareFunction::Less, None, true)),
    }
}

/// A shader's depth-only pipeline into the shadow map.
pub(crate) fn shadow_pipeline(context: &PipelineContext, module: &wgpu::ShaderModule, instanced: bool) -> wgpu::RenderPipeline {
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
        cache: None,
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
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_sprite_shaders_from_mesh_shaders() {
        let sprite = "@vertex fn vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> Corner {";
        let mesh = "@vertex fn vertex(input: VertexInput) -> VertexOutput {";
        assert_eq!(draws_sprites(sprite), Some(true));
        assert_eq!(draws_sprites(mesh), Some(false));
        assert_eq!(draws_sprites("@vertex fn vertex_instanced(input: I) {}"), None);
        assert!(has_instancing("@vertex\nfn vertex_instanced(input: I) {}"));
        assert!(!has_instancing("@vertex fn vertex_instancedx(input: I) {}"));
    }
}
