// Single-sampled opaque geometry and material inputs for screen-space RT.
use pen_protocol::{Matrix, RenderFrame};

use super::skinning::{self, Skinning};
use super::timing::{GpuTimer, Span};
use super::{Drawing, GpuAsset, ReadyEntry, Scratch, Shape};
use crate::asset_store::{MESH_VERTEX_FLOATS, Material};
use crate::camera_precision::DepthMode;
use crate::matrix::{invert, multiply};
use crate::shader::SKIN_VERTEX_BYTES;

const SHADER: &str = include_str!("../shaders/thin_gbuffer.wgsl");
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const INSTANCE_BYTES: u64 = 80;

const MESH_ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
];
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 3 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 4 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 5 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 6 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 64, shader_location: 7 },
];
const SKIN_ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint16x4, offset: 0, shader_location: 9 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 8, shader_location: 10 },
];

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawGbuf {
    model: Matrix,
    normal_mat: Matrix,
    albedo_metallic: [f32; 4],
    roughness_pad: [f32; 4],
}

const CAMERA_BYTES: u64 = size_of::<Matrix>() as u64;
const DRAW_BYTES: u64 = size_of::<DrawGbuf>() as u64;

impl DrawGbuf {
    fn new(model: Matrix, albedo_metallic: [f32; 4], roughness: f32, roughness_limit: f32) -> Self {
        let inv = invert(&model);
        let normal_mat = [
            inv[0], inv[4], inv[8], 0.0, inv[1], inv[5], inv[9], 0.0, inv[2], inv[6], inv[10], 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        Self {
            model,
            normal_mat,
            albedo_metallic,
            roughness_pad: [roughness.min(roughness_limit), roughness_limit, 0.0, 0.0],
        }
    }

    #[allow(clippy::cast_possible_truncation, reason = "same narrowing as the TLAS material constants")]
    fn material(model: Matrix, m: &Material, draw_params: Option<[f32; 4]>) -> Self {
        let (mut metallic, mut roughness) = match m.instance_parameters {
            Some(p) => ((p[0] as f32).clamp(0.0, 1.0), if p[1] > 0.0 { (p[1] as f32).clamp(0.04, 1.0) } else { 0.5 }),
            None => (0.0, 0.5),
        };
        if let Some(p) = draw_params {
            if p[0].is_finite() {
                metallic = p[0].clamp(0.0, 1.0);
            }
            if p[1].is_finite() && p[1] > 0.0 {
                roughness = p[1].clamp(0.04, 1.0);
            }
        }
        let roughness_limit =
            if m.surface.clearcoat > 0.0 { m.surface.clearcoat_roughness.clamp(0.04, 0.35) } else { 1.0 };
        Self::new(
            model,
            [
                m.color[0].clamp(0.0, 10.0) as f32,
                m.color[1].clamp(0.0, 10.0) as f32,
                m.color[2].clamp(0.0, 10.0) as f32,
                metallic,
            ],
            roughness,
            roughness_limit,
        )
    }
}

struct Targets {
    size: (u32, u32),
    normal_roughness: wgpu::TextureView,
    albedo_metallic: wgpu::TextureView,
    depth: wgpu::TextureView,
}

impl Targets {
    fn new(device: &wgpu::Device, size: (u32, u32)) -> Self {
        let view = |label, format| {
            let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
            if cfg!(test) && format == COLOR_FORMAT {
                usage |= wgpu::TextureUsages::COPY_SRC;
            }
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        Self {
            size,
            normal_roughness: view("thin gbuffer normal roughness", COLOR_FORMAT),
            albedo_metallic: view("thin gbuffer albedo metallic", COLOR_FORMAT),
            depth: view("thin gbuffer depth", DEPTH_FORMAT),
        }
    }

    fn begin<'a>(
        &self,
        encoder: &'a mut wgpu::CommandEncoder,
        depth_mode: DepthMode,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) -> wgpu::RenderPass<'a> {
        let color = |view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("thin gbuffer"),
            timestamp_writes,
            color_attachments: &[color(&self.normal_roughness), color(&self.albedo_metallic)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(depth_mode.clear_value()),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        })
    }
}

struct Pipelines {
    plain: wgpu::RenderPipeline,
    instanced: wgpu::RenderPipeline,
    skinned: wgpu::RenderPipeline,
}

struct Resources {
    shader: wgpu::ShaderModule,
    camera_layout: wgpu::BindGroupLayout,
    camera_uniform: wgpu::Buffer,
    camera_group: wgpu::BindGroup,
    draw_layout: wgpu::BindGroupLayout,
    skin_layout: wgpu::BindGroupLayout,
    pipelines: [Option<Pipelines>; 2],
}

impl Resources {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("thin gbuffer"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let uniform_layout = |label, visibility, has_dynamic_offset, bytes| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset,
                        min_binding_size: wgpu::BufferSize::new(bytes),
                    },
                    count: None,
                }],
            })
        };
        let camera_layout = uniform_layout("thin gbuffer camera", wgpu::ShaderStages::VERTEX, false, CAMERA_BYTES);
        let camera_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("thin gbuffer camera"),
            size: CAMERA_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_group = buffer_group(device, "thin gbuffer camera", &camera_layout, &camera_uniform, CAMERA_BYTES);
        let draw_layout = uniform_layout("thin gbuffer draw", wgpu::ShaderStages::VERTEX_FRAGMENT, true, DRAW_BYTES);
        let skin_layout = skinning::group_layout(device);
        Self { shader, camera_layout, camera_uniform, camera_group, draw_layout, skin_layout, pipelines: [None, None] }
    }

    fn ensure_pipelines(&mut self, device: &wgpu::Device, depth_mode: DepthMode) {
        let reversed_z = depth_mode == DepthMode::ReversedZ;
        self.pipelines[usize::from(reversed_z)].get_or_insert_with(|| {
            let make = |label,
                        entry,
                        layouts: &[Option<&wgpu::BindGroupLayout>],
                        buffers: &[Option<wgpu::VertexBufferLayout<'_>>]| {
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: layouts,
                    immediate_size: 0,
                });
                let color = || {
                    Some(wgpu::ColorTargetState {
                        format: COLOR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                };
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &self.shader,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        buffers,
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        cull_mode: Some(wgpu::Face::Back),
                        front_face: wgpu::FrontFace::Ccw,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(depth_mode.compare_nearer()),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &self.shader,
                        entry_point: Some("fragment"),
                        compilation_options: Default::default(),
                        targets: &[color(), color()],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            };
            let mesh = Some(wgpu::VertexBufferLayout {
                array_stride: (MESH_VERTEX_FLOATS * 4) as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &MESH_ATTRIBUTES,
            });
            let instanced = Some(wgpu::VertexBufferLayout {
                array_stride: INSTANCE_BYTES,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &INSTANCE_ATTRIBUTES,
            });
            let skinned = Some(wgpu::VertexBufferLayout {
                array_stride: SKIN_VERTEX_BYTES,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &SKIN_ATTRIBUTES,
            });
            let draw_layouts = [Some(&self.camera_layout), Some(&self.draw_layout)];
            Pipelines {
                plain: make("thin gbuffer plain", "vertex_plain", &draw_layouts, std::slice::from_ref(&mesh)),
                instanced: make(
                    "thin gbuffer instanced",
                    "vertex_instanced",
                    &draw_layouts,
                    &[mesh.clone(), instanced],
                ),
                skinned: make(
                    "thin gbuffer skinned",
                    "vertex_skinned",
                    &[draw_layouts[0], draw_layouts[1], Some(&self.skin_layout)],
                    &[mesh, skinned],
                ),
            }
        });
    }

    fn pipelines(&self, depth_mode: DepthMode) -> &Pipelines {
        self.pipelines[usize::from(depth_mode == DepthMode::ReversedZ)].as_ref().unwrap()
    }
}

fn buffer_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
    bytes: u64,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: wgpu::BufferSize::new(bytes),
            }),
        }],
    })
}

struct DrawBuffer {
    buffer: wgpu::Buffer,
    group: wgpu::BindGroup,
    data: Vec<u8>,
}

impl DrawBuffer {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, bytes: u64) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("thin gbuffer draws"),
            size: bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = buffer_group(device, "thin gbuffer draws", layout, &buffer, DRAW_BYTES);
        Self { buffer, group, data: vec![0; usize::try_from(bytes).expect("draw buffer fits in memory")] }
    }

    fn write(&mut self, offset: u64, draw: DrawGbuf) {
        let offset = usize::try_from(offset).expect("draw offset fits in memory");
        self.data[offset..offset + size_of::<DrawGbuf>()].copy_from_slice(bytemuck::bytes_of(&draw));
    }
}

pub(super) struct ThinGbuffer {
    stride: u64,
    targets: Option<Targets>,
    resources: Option<Resources>,
    draws: Option<DrawBuffer>,
    skin: Option<(wgpu::Buffer, u64, wgpu::BindGroup)>,
}

impl ThinGbuffer {
    pub fn new(device: &wgpu::Device) -> Self {
        let align = u64::from(device.limits().min_uniform_buffer_offset_alignment).max(1);
        Self { stride: DRAW_BYTES.next_multiple_of(align), targets: None, resources: None, draws: None, skin: None }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        size: (u32, u32),
        depth_mode: DepthMode,
        frame: &RenderFrame,
        active_projection: &Matrix,
        scratch: &Scratch,
        drawing: &Drawing<'_>,
        skinning: &Skinning,
        ready: &[ReadyEntry],
        timer: &GpuTimer,
    ) {
        if size.0 == 0 || size.1 == 0 {
            self.targets = None;
            return;
        }
        if self.size() != Some(size) {
            self.targets = Some(Targets::new(device, size));
        }
        let targets = self.targets.as_ref().unwrap();
        if scratch.opaque.is_empty() {
            targets.begin(encoder, depth_mode, timer.pass(Span::Scene));
            return;
        }

        let resources = self.resources.get_or_insert_with(|| Resources::new(device));
        let view_projection = multiply(active_projection, &frame.view);
        queue.write_buffer(&resources.camera_uniform, 0, bytemuck::bytes_of(&view_projection));

        let used = self.stride * scratch.opaque.len() as u64;
        if self.draws.as_ref().is_none_or(|draws| draws.buffer.size() < used) {
            let capacity = scratch.opaque.len().next_power_of_two().max(16) as u64 * self.stride;
            self.draws = Some(DrawBuffer::new(device, &resources.draw_layout, capacity));
        }
        let draws = self.draws.as_mut().unwrap();
        for (slot, &idx) in scratch.opaque.iter().enumerate() {
            let entry = &ready[idx];
            let item = &drawing.items[entry.item];
            let draw_params = item.draw(frame).and_then(|draw| draw.parameters);
            draws.write(
                slot as u64 * self.stride,
                DrawGbuf::material(*item.transform(frame), &entry.material.material, draw_params),
            );
        }
        queue.write_buffer(&draws.buffer, 0, &draws.data[..usize::try_from(used).expect("draw data fits in memory")]);

        if let [Some((buffer, bytes)), _] = skinning.buffers() {
            if self.skin.as_ref().is_none_or(|(b, size, _)| b != buffer || *size != bytes) {
                let group = buffer_group(device, "thin gbuffer joints", &resources.skin_layout, buffer, bytes);
                self.skin = Some((buffer.clone(), bytes, group));
            }
        } else {
            self.skin = None;
        }

        resources.ensure_pipelines(device, depth_mode);
        let pipelines = resources.pipelines(depth_mode);
        let mut pass = targets.begin(encoder, depth_mode, timer.pass(Span::Scene));
        pass.set_bind_group(0, &resources.camera_group, &[]);
        for (slot, &idx) in scratch.opaque.iter().enumerate() {
            let entry = &ready[idx];
            // Persistent instance records have application-defined vertex semantics.
            if entry.persistent {
                continue;
            }
            let offset = u32::try_from(slot as u64 * self.stride).expect("thin gbuffer draw offset fits a u32");
            pass.set_bind_group(1, &draws.group, &[offset]);
            let item = &drawing.items[entry.item];
            match item.shape {
                Shape::Mesh(_) => {
                    let Some(GpuAsset::Mesh(mesh)) = entry.mesh.and_then(|slot| drawing.assets.current(slot)) else {
                        continue;
                    };
                    let lod = mesh.lods.get(usize::from(item.lod)).unwrap_or(&mesh.lods[0]);
                    if let (Some(at), Some(skin), Some((_, _, group))) = (entry.skin, &mesh.skin, &self.skin) {
                        pass.set_pipeline(&pipelines.skinned);
                        pass.set_bind_group(2, group, &[at]);
                        pass.set_vertex_buffer(1, skin.vertices.slice(..));
                    } else {
                        pass.set_pipeline(&pipelines.plain);
                    }
                    if let (Some((start, end)), Some(morphed)) = (entry.morph, drawing.morphed) {
                        pass.set_vertex_buffer(0, morphed.slice(start..end));
                    } else {
                        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                    }
                    pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(lod.first..lod.first + lod.count, 0, 0..1);
                }
                Shape::Batch(batch_idx) => {
                    let batch = &drawing.batches[batch_idx];
                    let Some(GpuAsset::Mesh(mesh)) = drawing.assets.current(batch.mesh) else { continue };
                    // The packed lane contains the draw override or its
                    // placement's material defaults. This also survives GPU culling.
                    pass.set_pipeline(&pipelines.instanced);
                    if let Some(culled) = &batch.gpu {
                        drawing.culling.draw_positions(&mut pass, culled, 0, mesh);
                    } else {
                        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                        pass.set_vertex_buffer(1, drawing.instances.slice(..));
                        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                        for (lod, &(first, count)) in mesh.lods.iter().zip(&batch.ranges[0]) {
                            if count != 0 {
                                pass.draw_indexed(lod.first..lod.first + lod.count, 0, first..first + count);
                            }
                        }
                    }
                }
                Shape::Solid(idx) => {
                    let solid = &drawing.solids[idx];
                    pass.set_pipeline(&pipelines.plain);
                    pass.set_vertex_buffer(0, solid.buffer.slice(..));
                    pass.draw(0..solid.count, 0..1);
                }
            }
        }
    }

    pub fn normal_roughness_view(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().map(|targets| &targets.normal_roughness)
    }

    pub fn albedo_metallic_view(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().map(|targets| &targets.albedo_metallic)
    }

    pub fn depth_view(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().map(|targets| &targets.depth)
    }

    pub fn size(&self) -> Option<(u32, u32)> {
        self.targets.as_ref().map(|targets| targets.size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::IDENTITY;
    use crate::testing::{device, unhalf};
    use wgpu::util::DeviceExt;

    const SIZE: (u32, u32) = (32, 16);
    const ROW_BYTES: u32 = 256;

    fn render_witness(device: &wgpu::Device, queue: &wgpu::Queue, depth_mode: DepthMode) {
        let targets = Targets::new(device, SIZE);
        let mut resources = Resources::new(device);
        resources.ensure_pipelines(device, depth_mode);
        let pipelines = resources.pipelines(depth_mode);
        queue.write_buffer(&resources.camera_uniform, 0, bytemuck::bytes_of(&IDENTITY));

        let make_buffer = |label, contents, usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
        };
        let vertices = [
            [-0.22, -0.35, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.22, -0.35, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.35, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let vertices = make_buffer(
            "thin gbuffer witness mesh",
            bytemuck::cast_slice::<[f32; MESH_VERTEX_FLOATS], u8>(&vertices),
            wgpu::BufferUsages::VERTEX,
        );

        let (near, far) = match depth_mode {
            DepthMode::ForwardZ => (0.25, 0.75),
            DepthMode::ReversedZ => (0.75, 0.25),
        };
        let mut plain = IDENTITY;
        plain[12] = -0.65;
        plain[14] = near;
        // Non-uniform scale and shear. The transformed XY plane has
        // z = 0.5x + 0.25y + near, so its normal is proportional to
        // (-0.5, -0.25, 1) and differs from model * (0, 0, 1).
        let shear = [1.25, 0.0, 0.625, 0.0, 0.25, 0.75, 0.3125, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, near, 1.0];
        let mut occluded = shear;
        occluded[14] = far;

        let stride = DRAW_BYTES.next_multiple_of(u64::from(device.limits().min_uniform_buffer_offset_alignment));
        let mut posed = IDENTITY;
        posed[13] = 0.65;
        posed[14] = near;
        let mut draws = DrawBuffer::new(device, &resources.draw_layout, 5 * stride);
        for (slot, draw) in [
            DrawGbuf::new(plain, [0.125, 0.5, 2.0, 0.25], 0.625, 1.0),
            DrawGbuf::new(shear, [1.0, 0.25, 0.125, 0.75], 0.875, 0.25),
            DrawGbuf::new(occluded, [8.0, 8.0, 8.0, 1.0], 1.0, 1.0),
            DrawGbuf::new(IDENTITY, [0.25, 0.75, 0.5, 0.125], 0.125, 0.3125),
            DrawGbuf::new(posed, [0.5, 0.125, 0.25, 0.5], 0.75, 1.0),
        ]
        .into_iter()
        .enumerate()
        {
            draws.write(slot as u64 * stride, draw);
        }
        queue.write_buffer(&draws.buffer, 0, &draws.data);

        let mut instance = [0.0; 20];
        instance[..16].copy_from_slice(&shear);
        instance[12] = 0.65;
        instance[16..].copy_from_slice(&[0.875, 0.75, 0.0, 0.0]);
        let instances =
            make_buffer("thin gbuffer witness instance", bytemuck::cast_slice(&instance), wgpu::BufferUsages::VERTEX);

        // Two equally weighted joints reproduce the same shear in skin space.
        let mut palette = [shear; 2];
        palette[0][2] = 0.5;
        palette[1][2] = 0.75;
        palette[0][14] = 0.0;
        palette[1][14] = 0.0;
        let joints =
            make_buffer("thin gbuffer witness joints", bytemuck::cast_slice(&palette), wgpu::BufferUsages::STORAGE);
        let skin_group =
            buffer_group(device, "thin gbuffer witness joints", &resources.skin_layout, &joints, joints.size());
        let indices = [0u16, 1, 0, 0];
        let weights = [0.5f32, 0.5, 0.0, 0.0];
        const { assert!(SKIN_VERTEX_BYTES == 24) };
        let mut skin_vertex = [0u8; 24];
        skin_vertex[..8].copy_from_slice(bytemuck::bytes_of(&indices));
        skin_vertex[8..].copy_from_slice(bytemuck::bytes_of(&weights));
        let skin =
            make_buffer("thin gbuffer witness skin", bytemuck::bytes_of(&[skin_vertex; 3]), wgpu::BufferUsages::VERTEX);

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = targets.begin(&mut encoder, depth_mode, None);
            pass.set_pipeline(&pipelines.plain);
            pass.set_bind_group(0, &resources.camera_group, &[]);
            pass.set_vertex_buffer(0, vertices.slice(..));
            // Submit the far triangle after the near triangle at the same
            // screen position to witness depth rejection in both modes.
            for slot in 0..3 {
                pass.set_bind_group(1, &draws.group, &[u32::try_from(slot * stride).unwrap()]);
                pass.draw(0..3, 0..1);
            }
            pass.set_pipeline(&pipelines.instanced);
            pass.set_bind_group(1, &draws.group, &[u32::try_from(3 * stride).unwrap()]);
            pass.set_vertex_buffer(1, instances.slice(..));
            pass.draw(0..3, 0..1);

            pass.set_pipeline(&pipelines.skinned);
            pass.set_bind_group(1, &draws.group, &[u32::try_from(4 * stride).unwrap()]);
            pass.set_bind_group(2, &skin_group, &[0]);
            pass.set_vertex_buffer(1, skin.slice(..));
            pass.draw(0..3, 0..1);
        }

        let target_bytes = u64::from(ROW_BYTES * SIZE.1);
        let copy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("thin gbuffer witness readback"),
            size: 2 * target_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        for (index, view) in [&targets.normal_roughness, &targets.albedo_metallic].into_iter().enumerate() {
            encoder.copy_texture_to_buffer(
                view.texture().as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &copy,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: index as u64 * target_bytes,
                        bytes_per_row: Some(ROW_BYTES),
                        rows_per_image: Some(SIZE.1),
                    },
                },
                wgpu::Extent3d { width: SIZE.0, height: SIZE.1, depth_or_array_layers: 1 },
            );
        }
        queue.submit([encoder.finish()]);
        let slice = copy.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| result.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mapped = slice.get_mapped_range().unwrap();
        let pixel = |target: usize, x: usize, y: usize| {
            let at = target * usize::try_from(target_bytes).unwrap() + y * ROW_BYTES as usize + x * 8;
            std::array::from_fn::<_, 4, _>(|i| {
                let lane = at + 2 * i;
                unhalf(u16::from_le_bytes(mapped[lane..lane + 2].try_into().unwrap()))
            })
        };

        let length = 1.3125_f32.sqrt();
        let sheared_normal = [-0.5 / length, -0.25 / length, 1.0 / length];
        for ((x, y), normal, roughness, albedo_metallic) in [
            ((5, 8), [0.0, 0.0, 1.0], 0.625, [0.125, 0.5, 2.0, 0.25]),
            ((16, 8), sheared_normal, 0.25, [1.0, 0.25, 0.125, 0.75]),
            ((26, 8), sheared_normal, 0.3125, [0.25, 0.75, 0.5, 0.875]),
            ((16, 3), sheared_normal, 0.75, [0.5, 0.125, 0.25, 0.5]),
        ] {
            let got = pixel(0, x, y);
            for (actual, expected) in got[..3].iter().zip(normal) {
                assert!((actual - expected).abs() < 0.0005, "{depth_mode:?}, pixel ({x}, {y}): normal {got:?}");
            }
            let normal_sq: f32 = got[..3].iter().map(|v| v * v).sum();
            assert!((normal_sq - 1.0).abs() < 0.001, "{depth_mode:?}, pixel ({x}, {y}): length squared {normal_sq}");
            assert_eq!(got[3], roughness);
            assert_eq!(pixel(1, x, y), albedo_metallic);
        }
        assert_eq!(pixel(0, 0, 0), [0.0; 4]);
        assert_eq!(pixel(1, 0, 0), [0.0; 4]);
    }

    #[test]
    fn gpu_thin_gbuffer_rasterizes_exact_normals_roughness_albedo_metallic_witness() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        render_witness(&device, &queue, DepthMode::ForwardZ);
        render_witness(&device, &queue, DepthMode::ReversedZ);
    }
}
