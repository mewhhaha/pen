// Decals: a material's texture laid onto the surfaces inside a box. Each box
// is drawn from the inside, between the opaque surfaces and the translucent
// ones; every pixel it covers finds the surface the depth buffer holds there,
// and where that surface lies within the box, projected along the box's y
// axis and faded toward its ends, a lit material's texture darkens it and an
// unlit one's paints over it. Only surfaces whose material takes decals are
// touched: the scene pass marks them in the stencil (DECAL_MARK), and the
// decal pass tests for the mark. A device without storage buffers reads the
// decals from a uniform array a window of them at a time, and the depth from
// a copy (downlevel.rs).
use pen_protocol::RenderFrame;

use crate::built_in::{BuiltIn, BuiltIns};
use crate::downlevel::{DepthCopy, Window, array_usage};
use crate::matrix::{invert, multiply, narrow};
use crate::post::SCENE_FORMAT;
use crate::shader::PipelineContext;

/// The stencil bit of surfaces that take decals; bit 1 is the aberration's.
pub(crate) const DECAL_MARK: u32 = 2;
// Model and inverse matrices, then color times opacity, per decal.
const DECAL_FLOATS: usize = 36;
#[allow(clippy::cast_possible_truncation, reason = "36 floats")]
const DECAL_BYTES: u32 = DECAL_FLOATS as u32 * 4;

const SHADER: &str = include_str!("shaders/decals.wgsl");

pub(crate) fn shader(samples: u32) -> String {
    SHADER.replace("${depth_texture}", if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" })
}

/// Decals of one material and texture: `count` of them from `first` in the
/// drawing order.
pub(crate) struct DecalBatch {
    pub texture: wgpu::TextureView,
    /// Which texture view it is, to batch by.
    pub texture_id: u64,
    /// The material's slot, whose color it shares.
    pub material: usize,
    pub color: [f64; 4],
    /// Painted over the surface (an unlit material), not multiplied.
    pub paint: bool,
    pub first: u32,
    pub count: u32,
}

/// The pipeline drawing decals into a scene of `samples`: painted (an unlit
/// material's) or multiplied.
pub(crate) fn pipeline(context: &PipelineContext, samples: u32, paint: bool) -> wgpu::RenderPipeline {
    use wgpu::{BlendComponent, BlendFactor, BlendOperation};
    let device = &context.device;
    let module = crate::downlevel::shader_module(device, Some("decals"), &shader(samples));
    let keep = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Equal,
        pass_op: wgpu::StencilOperation::Keep,
        ..Default::default()
    };
    let color = if paint {
        BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        }
    } else {
        BlendComponent { src_factor: BlendFactor::Dst, dst_factor: BlendFactor::Zero, operation: BlendOperation::Add }
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("decals"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            buffers: &[],
        },
        // The inside of the box, so it covers the surfaces even with the
        // camera within it.
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Front),
            front_face: wgpu::FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState { front: keep, back: keep, read_mask: DECAL_MARK, write_mask: 0 },
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some(if paint { "paint" } else { "fragment" }),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SCENE_FORMAT,
                blend: Some(wgpu::BlendState {
                    color,
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::One,
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

pub(crate) struct Decals {
    view_uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// How the decals are bound, and the depth they read, where there are
    /// no storage buffers.
    window: Option<Window>,
    depth_copy: Option<DepthCopy>,
    capacity: usize,
    storage: Option<wgpu::Buffer>,
    data: Vec<f32>,
}

impl Decals {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            view_uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("decal view"),
                size: 144,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            window: Window::of(device, DECAL_BYTES),
            depth_copy: crate::downlevel::limited(device).then(|| DepthCopy::new(device)),
            capacity: 0,
            storage: None,
            data: Vec::new(),
        }
    }

    /// The depth decals read, `depth` being a `size` scene's depth alone:
    /// that view, or on a device that cannot read it while it is attached, a
    /// copy of it made now.
    pub fn depth(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        size: (u32, u32),
    ) -> wgpu::TextureView {
        match &mut self.depth_copy {
            Some(copy) => {
                let target = copy.target(device, size);
                copy.copy(device, encoder, depth);
                target
            }
            None => depth.clone(),
        }
    }

    /// Draws `batches` of the frame's decals (by their indices in `order`)
    /// into `pass`, whose depth and stencil are read-only and whose depth
    /// `depth` holds (from `depth()`), for a frame of `frame`'s view.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        built_ins: &mut BuiltIns,
        pass: &mut wgpu::RenderPass<'_>,
        frame: &RenderFrame,
        batches: &[DecalBatch],
        order: &[u32],
        depth: &wgpu::TextureView,
        samples: u32,
        size: (u32, u32),
    ) {
        let count = order.len();
        if count == 0 {
            return;
        }
        if count > self.capacity {
            self.capacity = count.next_power_of_two();
            let room = self.window.map_or(self.capacity, |window| window.room(self.capacity));
            self.storage = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("decals"),
                size: (room * DECAL_FLOATS * 4) as u64,
                usage: array_usage(self.window),
                mapped_at_creation: false,
            }));
        }
        let data = &mut self.data;
        data.clear();
        for batch in batches {
            for &index in &order[batch.first as usize..(batch.first + batch.count) as usize] {
                let decal = &frame.decals[index as usize];
                data.extend_from_slice(&decal.transform);
                data.extend_from_slice(&invert(&decal.transform));
                let [red, green, blue, alpha] = batch.color;
                data.extend([red, green, blue, alpha * f64::from(decal.opacity)].map(narrow));
            }
        }
        let storage = self.storage.clone().unwrap();
        queue.write_buffer(&storage, 0, bytemuck::cast_slice(data));
        let view_projection = multiply(&frame.projection, &frame.view);
        let mut view = view_projection.to_vec();
        view.extend(invert(&view_projection));
        view.extend([size.0 as f32, size.1 as f32, 0.0, 0.0]);
        queue.write_buffer(&self.view_uniform, 0, bytemuck::cast_slice(&view));
        pass.set_stencil_reference(DECAL_MARK);
        for batch in batches {
            let selected = built_ins.get(BuiltIn::Decal { samples, paint: batch.paint });
            pass.set_pipeline(selected);
            let group = |offset: u32| {
                let decals = wgpu::BufferBinding {
                    buffer: &storage,
                    offset: u64::from(offset),
                    size: self.window.and_then(|window| wgpu::BufferSize::new(window.bytes())),
                };
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &selected.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.view_uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Buffer(decals) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth) },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&batch.texture),
                        },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    ],
                })
            };
            let instances = batch.first..batch.first + batch.count;
            match self.window {
                None => {
                    pass.set_bind_group(0, &group(0), &[]);
                    pass.draw(0..36, instances);
                }
                Some(window) => {
                    for (offset, instances) in window.draws(instances) {
                        pass.set_bind_group(0, &group(offset), &[]);
                        pass.draw(0..36, instances);
                    }
                }
            }
        }
    }
}
