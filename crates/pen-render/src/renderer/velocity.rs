// Per-object and skeletal screen-space velocity pass (shaders/velocity.wgsl).
// Renders opaque scene geometry into a single-sampled Rg16Float velocity
// texture (UV delta curr_uv - prev_uv) whenever a frame contains moving draws
// (`frame.motions`) or skinned draws with a previous-frame joint palette.
// Frames with no object motion leave the texture at the sentinel (1024, 1024)
// so TAA and motion blur fall back to exact analytical depth reprojection.
use pen_protocol::{Matrix, RenderFrame};

use super::skinning::Skinning;
use super::timing::{GpuTimer, Span};
use super::{Drawing, GpuAsset, ReadyEntry, Scratch, Shape, Source};
use crate::asset_store::MESH_VERTEX_FLOATS;
use crate::matrix::{IDENTITY, multiply};
use crate::shader::SKIN_VERTEX_BYTES;

const SHADER: &str = include_str!("../shaders/velocity.wgsl");

/// Clear value marking pixels without explicit object velocity (sky or idle frame).
pub(crate) const SENTINEL: f64 = 1024.0;
/// IEEE-754 binary16 bits for 1024.0 (2^10 -> exp = 25 = 0b11001, frac = 0 -> 0x6400).
pub(crate) const SENTINEL_F16_BYTES: [u8; 4] = [0x00, 0x64, 0x00, 0x64];

const DRAW_MOTION_BYTES: u64 = 128;
const CAMERA_BYTES: u64 = 192;

const POS_ATTRIBUTE: [wgpu::VertexAttribute; 1] =
    [wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 }];

const INSTANCE_MODEL_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 3 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 4 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 5 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 6 },
];

const SKIN_ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint16x4, offset: 0, shader_location: 9 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 8, shader_location: 10 },
];

struct Pipelines {
    plain: wgpu::RenderPipeline,
    instanced: wgpu::RenderPipeline,
    skinned: wgpu::RenderPipeline,
    camera_uniform: wgpu::Buffer,
    camera_group: wgpu::BindGroup,
    draw_layout: wgpu::BindGroupLayout,
    skin_layout: wgpu::BindGroupLayout,
}

pub(super) struct Velocity {
    stride: u64,
    pipelines: Option<Pipelines>,
    draw_capacity: usize,
    draw_buffer: Option<wgpu::Buffer>,
    draw_group: Option<wgpu::BindGroup>,
    draw_data: Vec<f32>,
    /// Cached bind groups for skinned velocity: `(curr_buf, curr_size, prev_buf, prev_size)`.
    skin_both: Option<([wgpu::Buffer; 2], [u64; 2], wgpu::BindGroup)>,
    skin_same: Option<(wgpu::Buffer, u64, wgpu::BindGroup)>,
    /// Whether the current velocity target may hold non-sentinel pixels from an earlier frame.
    dirty: bool,
}

impl Velocity {
    pub fn new(device: &wgpu::Device) -> Self {
        let align = u64::from(device.limits().min_uniform_buffer_offset_alignment).max(1);
        let stride = DRAW_MOTION_BYTES.next_multiple_of(align);
        Self {
            stride,
            pipelines: None,
            draw_capacity: 0,
            draw_buffer: None,
            draw_group: None,
            draw_data: Vec::new(),
            skin_both: None,
            skin_same: None,
            dirty: true,
        }
    }

    /// Marks the velocity target as needing a sentinel clear before it is read.
    pub fn reset_target(&mut self) {
        self.dirty = true;
    }

    fn ensure_pipelines(&mut self, device: &wgpu::Device) {
        if self.pipelines.is_some() {
            return;
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("velocity"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("velocity camera"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(CAMERA_BYTES),
                },
                count: None,
            }],
        });
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("velocity draw"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(DRAW_MOTION_BYTES),
                },
                count: None,
            }],
        });
        let skin_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("velocity skin"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                },
            ],
        });
        let make = |label: &str,
                    layouts: &[Option<&wgpu::BindGroupLayout>],
                    entry: &str,
                    buffers: &[Option<wgpu::VertexBufferLayout<'_>>]| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
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
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fragment"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let pos_buf = Some(wgpu::VertexBufferLayout {
            array_stride: (MESH_VERTEX_FLOATS * 4) as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &POS_ATTRIBUTE,
        });
        let inst_buf = Some(wgpu::VertexBufferLayout {
            array_stride: 80,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_MODEL_ATTRIBUTES,
        });
        let skin_buf = Some(wgpu::VertexBufferLayout {
            array_stride: SKIN_VERTEX_BYTES,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &SKIN_ATTRIBUTES,
        });
        let plain = make(
            "velocity plain",
            &[Some(&camera_layout), Some(&draw_layout)],
            "vertex_plain",
            std::slice::from_ref(&pos_buf),
        );
        let instanced = make(
            "velocity instanced",
            &[Some(&camera_layout)],
            "vertex_instanced",
            &[pos_buf.clone(), inst_buf],
        );
        let skinned = make(
            "velocity skinned",
            &[Some(&camera_layout), Some(&draw_layout), Some(&skin_layout)],
            "vertex_skinned",
            &[pos_buf, skin_buf],
        );
        let camera_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("velocity camera"),
            size: CAMERA_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("velocity camera"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_uniform.as_entire_binding() }],
        });
        self.pipelines =
            Some(Pipelines { plain, instanced, skinned, camera_uniform, camera_group, draw_layout, skin_layout });
    }

    fn ensure_draw_buffer(&mut self, device: &wgpu::Device, count: usize) {
        if count <= self.draw_capacity && self.draw_buffer.is_some() {
            return;
        }
        self.draw_capacity = count.next_power_of_two().max(16);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("velocity draws"),
            size: self.draw_capacity as u64 * self.stride,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = &self.pipelines.as_ref().unwrap().draw_layout;
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("velocity draws"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(DRAW_MOTION_BYTES),
                }),
            }],
        });
        self.draw_buffer = Some(buffer);
        self.draw_group = Some(group);
        self.draw_data.resize(self.draw_capacity * (self.stride / 4) as usize, 0.0);
    }

    /// Encodes the velocity pass into `target` (`Rg16Float`) using `depth` (`Depth32Float`).
    /// When no opaque object has explicit motion or skeletal history, skips drawing and
    /// ensures `target` holds the sentinel value `(1024.0, 1024.0)`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        frame: &RenderFrame,
        jittered_projection: &Matrix,
        previous_vp: Option<Matrix>,
        scratch: &Scratch,
        drawing: &Drawing<'_>,
        skinning: &Skinning,
        ready: &[ReadyEntry],
        timer: &GpuTimer,
    ) {
        let has_object_motion = !frame.motions.is_empty()
            || scratch.opaque.iter().any(|&idx| {
                let entry = &ready[idx];
                entry.skin.is_some()
                    && entry.mesh.is_some_and(|mesh| {
                        scratch
                            .target_of_source(drawing.items[entry.item].source)
                            .is_some_and(|t| skinning.previous_offset(t, mesh).is_some())
                    })
            });

        let clear = wgpu::Color { r: SENTINEL, g: SENTINEL, b: 0.0, a: 0.0 };
        if !has_object_motion {
            if self.dirty {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("velocity clear"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
                    })],
                    ..Default::default()
                });
                self.dirty = false;
            }
            return;
        }

        self.ensure_pipelines(device);
        let curr_vp = multiply(&frame.projection, &frame.view);
        let curr_jittered_vp = multiply(jittered_projection, &frame.view);
        let prev_vp = previous_vp.unwrap_or(curr_vp);
        let mut cam_floats = [0.0f32; 48];
        cam_floats[..16].copy_from_slice(&curr_jittered_vp);
        cam_floats[16..32].copy_from_slice(&curr_vp);
        cam_floats[32..48].copy_from_slice(&prev_vp);
        queue.write_buffer(&self.pipelines.as_ref().unwrap().camera_uniform, 0, bytemuck::cast_slice(&cam_floats));

        // Slot 0 is always identity (used for world-space solids), followed by each opaque non-batch mesh.
        let mut motion_slots: Vec<(usize, u32)> = Vec::new();
        let mut slot_count: usize = 1;
        for &idx in &scratch.opaque {
            let entry = &ready[idx];
            if entry.persistent {
                continue;
            }
            if matches!(drawing.items[entry.item].shape, Shape::Mesh(_)) {
                motion_slots.push((idx, u32::try_from(slot_count).unwrap_or(u32::MAX)));
                slot_count += 1;
            }
        }
        self.ensure_draw_buffer(device, slot_count);
        let stride_floats = (self.stride / 4) as usize;
        // Slot 0: identity current and previous.
        self.draw_data[0..16].copy_from_slice(&IDENTITY);
        self.draw_data[16..32].copy_from_slice(&IDENTITY);
        for &(idx, slot) in &motion_slots {
            let entry = &ready[idx];
            let item = &drawing.items[entry.item];
            let curr = item.transform(frame);
            let prev = match item.source {
                Source::Draw(draw_idx) => scratch.motion_previous(frame, draw_idx).unwrap_or(curr),
                _ => curr,
            };
            let base = slot as usize * stride_floats;
            self.draw_data[base..base + 16].copy_from_slice(curr);
            self.draw_data[base + 16..base + 32].copy_from_slice(prev);
        }
        queue.write_buffer(
            self.draw_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&self.draw_data[..slot_count * stride_floats]),
        );

        // Prepare skinned velocity bind groups if any skinned buffers are live.
        let [curr_skin_buf, prev_skin_buf] = skinning.buffers();
        if let Some((curr_buf, curr_size)) = curr_skin_buf {
            let layout = &self.pipelines.as_ref().unwrap().skin_layout;
            if self.skin_same.as_ref().is_none_or(|(b, s, _)| b != curr_buf || *s != curr_size) {
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("velocity skin same"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: curr_buf,
                                offset: 0,
                                size: wgpu::BufferSize::new(curr_size),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: curr_buf,
                                offset: 0,
                                size: wgpu::BufferSize::new(curr_size),
                            }),
                        },
                    ],
                });
                self.skin_same = Some((curr_buf.clone(), curr_size, group));
            }
            if let Some((prev_buf, prev_size)) = prev_skin_buf
                && self
                    .skin_both
                    .as_ref()
                    .is_none_or(|(b, s, _)| &b[0] != curr_buf || &b[1] != prev_buf || *s != [curr_size, prev_size])
            {
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("velocity skin both"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: curr_buf,
                                offset: 0,
                                size: wgpu::BufferSize::new(curr_size),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: prev_buf,
                                offset: 0,
                                size: wgpu::BufferSize::new(prev_size),
                            }),
                        },
                    ],
                });
                self.skin_both = Some(([curr_buf.clone(), prev_buf.clone()], [curr_size, prev_size], group));
            }
        }

        let pipelines = self.pipelines.as_ref().unwrap();
        let draw_group = self.draw_group.as_ref().unwrap();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("velocity"),
            timestamp_writes: timer.pass(Span::Scene),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_bind_group(0, &pipelines.camera_group, &[]);

        let mut next_slot_cursor = 0usize;
        for &idx in &scratch.opaque {
            let entry = &ready[idx];
            if entry.persistent {
                continue;
            }
            let item = &drawing.items[entry.item];
            match item.shape {
                Shape::Mesh(_) => {
                    let slot = motion_slots[next_slot_cursor].1;
                    next_slot_cursor += 1;
                    let Some(mesh_slot) = entry.mesh else { continue };
                    let Some(GpuAsset::Mesh(mesh)) = drawing.assets.current(mesh_slot) else { continue };
                    if mesh.instances.is_some() {
                        continue;
                    }
                    let offset = u32::try_from(u64::from(slot) * self.stride).unwrap_or(0);
                    let lod = mesh.lods.get(usize::from(item.lod)).unwrap_or(&mesh.lods[0]);
                    if let (Some(curr_off), Some(skin)) = (entry.skin, &mesh.skin) {
                        let prev_off = scratch
                            .target_of_source(item.source)
                            .and_then(|t| skinning.previous_offset(t, mesh_slot));
                        let bound = match (prev_off, &self.skin_both, &self.skin_same) {
                            (Some(p_off), Some((_, _, group)), _) => Some((group, p_off)),
                            (_, _, Some((_, _, group))) => Some((group, curr_off)),
                            _ => None,
                        };
                        if let Some((skin_group, p_off)) = bound {
                            pass.set_pipeline(&pipelines.skinned);
                            pass.set_bind_group(1, draw_group, &[offset]);
                            pass.set_bind_group(2, skin_group, &[curr_off, p_off]);
                            if let (Some((start, end)), Some(morphed)) = (entry.morph, drawing.morphed) {
                                pass.set_vertex_buffer(0, morphed.slice(start..end));
                            } else {
                                pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                            }
                            pass.set_vertex_buffer(1, skin.vertices.slice(..));
                            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(lod.first..lod.first + lod.count, 0, 0..1);
                            continue;
                        }
                    }
                    pass.set_pipeline(&pipelines.plain);
                    pass.set_bind_group(1, draw_group, &[offset]);
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
                    pass.set_pipeline(&pipelines.instanced);
                    if let Some(culled) = &batch.gpu {
                        drawing.culling.draw_positions(&mut pass, culled, 0, mesh);
                    } else {
                        let ranges = &batch.ranges[0];
                        if ranges.iter().all(|&(_, count)| count == 0) {
                            continue;
                        }
                        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                        pass.set_vertex_buffer(1, drawing.instances.slice(..));
                        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                        for (lod, &(first, count)) in mesh.lods.iter().zip(ranges) {
                            if count > 0 {
                                pass.draw_indexed(lod.first..lod.first + lod.count, 0, first..first + count);
                            }
                        }
                    }
                }
                Shape::Solid(solid_idx) => {
                    let solid = &drawing.solids[solid_idx];
                    pass.set_pipeline(&pipelines.plain);
                    pass.set_bind_group(1, draw_group, &[0]);
                    pass.set_vertex_buffer(0, solid.buffer.slice(..));
                    pass.draw(0..solid.count, 0..1);
                }
            }
        }
        self.dirty = true;
    }
}
