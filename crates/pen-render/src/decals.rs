// Decals: a material's texture laid onto the surfaces inside a box. Each box
// is drawn from the inside, between the opaque surfaces and the translucent
// ones; every pixel it covers finds the surface the depth buffer holds there,
// and where that surface lies within the box, projected along the box's y
// axis and faded toward its ends, a lit material's texture darkens it and an
// unlit one's paints over it. Only surfaces whose material takes decals are
// touched: the scene pass marks them in the stencil (DECAL_MARK), and the
// decal pass tests for the mark.
use std::collections::HashMap;

use pen_protocol::{Decal, RenderFrame};

use crate::matrix::{invert, multiply};

/// The stencil bit of surfaces that take decals; bit 1 is the aberration's.
pub(crate) const DECAL_MARK: u32 = 2;
// Model and inverse matrices, then color times opacity, per decal.
const DECAL_FLOATS: usize = 36;

const SHADER: &str = r#"
struct View {
  view_projection: mat4x4f,
  inverse_view_projection: mat4x4f,
  size: vec4f,
};
struct Decal {
  model: mat4x4f,
  inverse: mat4x4f,
  color: vec4f,
};
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> decals: array<Decal>;
@group(0) @binding(2) var depth: ${depth_texture};
@group(0) @binding(3) var image: texture_2d<f32>;
@group(0) @binding(4) var image_sampler: sampler;

// The cube -1..1, wound counterclockwise seen from outside.
const corners = array<vec3f, 8>(
  vec3f(-1.0, -1.0, -1.0), vec3f(1.0, -1.0, -1.0), vec3f(-1.0, 1.0, -1.0), vec3f(1.0, 1.0, -1.0),
  vec3f(-1.0, -1.0, 1.0), vec3f(1.0, -1.0, 1.0), vec3f(-1.0, 1.0, 1.0), vec3f(1.0, 1.0, 1.0),
);
const faces = array<u32, 36>(
  0u, 4u, 6u, 0u, 6u, 2u, 1u, 3u, 7u, 1u, 7u, 5u,
  0u, 1u, 5u, 0u, 5u, 4u, 2u, 6u, 7u, 2u, 7u, 3u,
  0u, 2u, 3u, 0u, 3u, 1u, 4u, 5u, 7u, 4u, 7u, 6u,
);

struct Box {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) index: u32,
};

@vertex fn vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> Box {
  let corner = corners[faces[vertex]];
  return Box(view.view_projection * decals[index].model * vec4f(corner, 1.0), index);
}

// The decal's tinted texture where the surface behind the box lies in it, its
// alpha how much of the surface it covers: none outside the box.
fn laid(box: Box) -> vec4f {
  let decal = decals[box.index];
  let pixel = vec2i(box.position.xy);
  let ndc = vec2f(box.position.x / view.size.x * 2.0 - 1.0, 1.0 - box.position.y / view.size.y * 2.0);
  let world = view.inverse_view_projection * vec4f(ndc, textureLoad(depth, pixel, 0), 1.0);
  let local = (decal.inverse * vec4f(world.xyz / world.w, 1.0)).xyz;
  if (any(abs(local) > vec3f(1.0))) {
    return vec4f(0.0);
  }
  let texel = textureSampleLevel(image, image_sampler, local.xz * 0.5 + 0.5, 0.0);
  let cover = texel.a * decal.color.a * (1.0 - smoothstep(0.6, 1.0, abs(local.y)));
  return vec4f(texel.rgb * decal.color.rgb, cover);
}

// Multiplied over the surface, so its lighting and shadow show through.
@fragment fn fragment(box: Box) -> @location(0) vec4f {
  let found = laid(box);
  if (found.a <= 0.0) {
    discard;
  }
  return vec4f(mix(vec3f(1.0), found.rgb, found.a), 1.0);
}

// Painted over the surface in its own color, premultiplied, whatever light
// falls there.
@fragment fn paint(box: Box) -> @location(0) vec4f {
  let found = laid(box);
  if (found.a <= 0.0) {
    discard;
  }
  return vec4f(found.rgb * found.a, found.a);
}
"#;

fn shader(samples: u32) -> String {
    SHADER.replace("${depth_texture}", if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" })
}

pub(crate) struct DecalBatch<'a> {
    pub texture: wgpu::TextureView,
    /// Which texture view it is, to batch by.
    pub texture_id: u64,
    /// The material's slot, whose color it shares.
    pub material: usize,
    pub color: [f64; 4],
    /// Painted over the surface (an unlit material), not multiplied.
    pub paint: bool,
    pub decals: Vec<&'a Decal>,
}

pub(crate) struct Decals {
    format: wgpu::TextureFormat,
    pipelines: HashMap<(u32, bool), wgpu::RenderPipeline>,
    view_uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    capacity: usize,
    storage: Option<wgpu::Buffer>,
}

impl Decals {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        Self {
            format,
            pipelines: HashMap::new(),
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
            capacity: 0,
            storage: None,
        }
    }

    fn pipeline(&mut self, device: &wgpu::Device, samples: u32, paint: bool) -> wgpu::RenderPipeline {
        let format = self.format;
        self.pipelines
            .entry((samples, paint))
            .or_insert_with(|| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("decals"),
                    source: wgpu::ShaderSource::Wgsl(shader(samples).into()),
                });
                let keep = wgpu::StencilFaceState {
                    compare: wgpu::CompareFunction::Equal,
                    pass_op: wgpu::StencilOperation::Keep,
                    ..Default::default()
                };
                use wgpu::{BlendComponent, BlendFactor, BlendOperation};
                let color = if paint {
                    BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::OneMinusSrcAlpha, operation: BlendOperation::Add }
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
                    // The inside of the box, so it covers the surfaces even
                    // with the camera within it.
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
                            format,
                            blend: Some(wgpu::BlendState {
                                color,
                                alpha: BlendComponent { src_factor: BlendFactor::Zero, dst_factor: BlendFactor::One, operation: BlendOperation::Add },
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            })
            .clone()
    }

    /// Draws `batches` into `pass`, whose depth and stencil are read-only and
    /// whose depth `depth` also views, for a frame of `frame`'s view.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        frame: &RenderFrame,
        batches: &[DecalBatch<'_>],
        depth: &wgpu::TextureView,
        samples: u32,
        size: (u32, u32),
    ) {
        let count: usize = batches.iter().map(|batch| batch.decals.len()).sum();
        if count == 0 {
            return;
        }
        if count > self.capacity {
            self.capacity = count.next_power_of_two();
            self.storage = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("decals"),
                size: (self.capacity * DECAL_FLOATS * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let mut data = Vec::with_capacity(count * DECAL_FLOATS);
        for batch in batches {
            for decal in &batch.decals {
                data.extend_from_slice(&decal.transform);
                data.extend_from_slice(&invert(&decal.transform));
                data.extend([
                    batch.color[0] as f32,
                    batch.color[1] as f32,
                    batch.color[2] as f32,
                    (batch.color[3] * f64::from(decal.opacity)) as f32,
                ]);
            }
        }
        let storage = self.storage.clone().unwrap();
        queue.write_buffer(&storage, 0, bytemuck::cast_slice(&data));
        let view_projection = multiply(&frame.projection, &frame.view);
        let mut view = view_projection.to_vec();
        view.extend(invert(&view_projection));
        view.extend([size.0 as f32, size.1 as f32, 0.0, 0.0]);
        queue.write_buffer(&self.view_uniform, 0, bytemuck::cast_slice(&view));
        pass.set_stencil_reference(DECAL_MARK);
        let mut first = 0;
        for batch in batches {
            let selected = self.pipeline(device, samples, batch.paint);
            pass.set_pipeline(&selected);
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &selected.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.view_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: storage.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&batch.texture) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            pass.set_bind_group(0, &group, &[]);
            let count = batch.decals.len() as u32;
            pass.draw(0..36, first..first + count);
            first += count;
        }
    }
}
