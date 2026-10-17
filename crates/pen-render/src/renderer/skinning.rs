// Skinned draws' poses. A frame's pose layers name the draws they pose; a
// posed draw whose mesh has a skin and whose material's shader has a
// `vertex_skinned` entry point is drawn alone with its skinned pipeline, its
// joint matrices worked out here from its layers (the mesh's clips sampled
// and blended, animation.rs) and laid in one storage buffer for the frame,
// which the draw binds at group 3 with a dynamic offset. Other posed draws
// are drawn as they are, in their mesh's bind pose.
use std::sync::Arc;

use pen_protocol::RenderFrame;

use super::{GpuAsset, Item, MeshAsset, ReadyEntry, Scratch, Shape, State};
use crate::animation::{Layer, Matrix, Rig};
use crate::asset_store::Material;
use crate::shader::{Stage, Variants};

/// Group 3 of a skinned pipeline (after the lights): the draw's joint matrices, read by its
/// vertices, at a dynamic offset into the frame's. Empty on a WebGL2-class
/// device, which has no storage buffers and draws nothing skinned.
pub(super) fn group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    if crate::downlevel::limited(device) {
        return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("skin"), entries: &[] });
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("skin"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(64),
            },
            count: None,
        }],
    })
}

impl Scratch {
    /// Sorts the frame's pose layers by the draw they pose.
    pub(super) fn group_poses(&mut self, frame: &RenderFrame) {
        self.pose_first.clear();
        self.pose_order.clear();
        if frame.poses.is_empty() {
            return;
        }
        // Validation keeps every layer's draw among the frame's.
        let first = &mut self.pose_first;
        first.resize(frame.draws.len() + 1, 0);
        for pose in &frame.poses {
            first[pose.draw as usize + 1] += 1;
        }
        for index in 1..first.len() {
            first[index] += first[index - 1];
        }
        let mut next: Vec<u32> = first[..frame.draws.len()].to_vec();
        self.pose_order.resize(frame.poses.len(), 0);
        for (index, pose) in frame.poses.iter().enumerate() {
            let slot = &mut next[pose.draw as usize];
            self.pose_order[*slot as usize] = u32::try_from(index).unwrap_or(u32::MAX);
            *slot += 1;
        }
    }

    /// Whether the frame's draw `draw` has pose layers.
    pub(super) fn posed(&self, draw: usize) -> bool {
        self.pose_first.get(draw + 1).is_some_and(|&end| end > self.pose_first[draw])
    }

    /// Draw `draw`'s pose layers.
    pub(super) fn layers<'a>(&'a self, frame: &'a RenderFrame, draw: usize) -> impl Iterator<Item = Layer> + 'a {
        let range = match self.pose_first.get(draw + 1) {
            Some(&end) => self.pose_first[draw] as usize..end as usize,
            None => 0..0,
        };
        self.pose_order[range].iter().map(move |&index| {
            let pose = &frame.poses[index as usize];
            Layer { clip: pose.clip, time: pose.time, weight: pose.weight }
        })
    }
}

/// The frame's joint matrices: each posed draw's palette at an offset that
/// storage bindings allow, in one buffer bound once with each offset.
pub(super) struct Skinning {
    alignment: usize,
    /// This frame's palettes, as floats.
    data: Vec<f32>,
    /// The largest palette this frame, in bytes: what the binding spans.
    largest: u64,
    buffer: Option<wgpu::Buffer>,
    /// The bind group, for the buffer and the binding's size it was made
    /// with.
    group: Option<(wgpu::BindGroup, u64)>,
    layers: Vec<Layer>,
    palette: Vec<Matrix>,
}

/// What became of a posed draw.
pub(super) enum Posed {
    /// It is drawn as it is: its mesh has no skin, its shader no skinned
    /// entry point, or its material is not opaque.
    Unskinned,
    /// Its skinned pipeline is compiling.
    Waiting,
    /// Drawn with this pipeline and the joint matrices at this offset.
    Ready(wgpu::RenderPipeline, u32),
}

impl Skinning {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            alignment: device.limits().min_storage_buffer_offset_alignment as usize,
            data: Vec::new(),
            largest: 0,
            buffer: None,
            group: None,
            layers: Vec::new(),
            palette: Vec::new(),
        }
    }

    /// Starts a frame's palettes.
    pub fn begin(&mut self) {
        self.data.clear();
        self.largest = 0;
    }

    /// Lays `rig`'s joint matrices for `layers` after the frame's others,
    /// and returns their offset in bytes.
    fn pose(&mut self, rig: &Rig, layers: impl Iterator<Item = Layer>) -> u32 {
        self.layers.clear();
        self.layers.extend(layers);
        rig.palette(&self.layers, &mut self.palette);
        let aligned = (self.data.len() * 4).next_multiple_of(self.alignment);
        self.data.resize(aligned / 4, 0.0);
        for matrix in &self.palette {
            self.data.extend_from_slice(matrix);
        }
        self.largest = self.largest.max(self.palette.len() as u64 * 64);
        u32::try_from(aligned).unwrap_or(u32::MAX)
    }

    /// Writes the frame's palettes for its draws to bind; the bind group,
    /// if any draw is posed.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, layout: &wgpu::BindGroupLayout) {
        if self.data.is_empty() || crate::downlevel::limited(device) {
            return;
        }
        // Room for the last palette's whole binding past its offset.
        let used = (self.data.len() * 4) as u64;
        let size = used + self.largest;
        if self.buffer.as_ref().is_none_or(|buffer| buffer.size() < size) {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("joints"),
                size: size.next_power_of_two(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.buffer = Some(buffer);
            self.group = None;
        }
        let buffer = self.buffer.as_ref().unwrap();
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&self.data));
        if self.group.as_ref().is_none_or(|(_, binding)| *binding != self.largest) {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("joints"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(self.largest),
                    }),
                }],
            });
            self.group = Some((group, self.largest));
        }
    }

    /// The bind group of this frame's palettes, if any draw is posed.
    pub fn group(&self) -> Option<&wgpu::BindGroup> {
        if self.data.is_empty() { None } else { self.group.as_ref().map(|(group, _)| group) }
    }
}

impl ReadyEntry {
    /// How its vertices come: skinned, instanced or plain.
    pub(super) fn stage(&self, items: &[Item]) -> Stage {
        if self.skin.is_some() {
            Stage::Skinned
        } else if matches!(items[self.item].shape, Shape::Batch(_)) {
            Stage::Instanced
        } else {
            Stage::Plain
        }
    }
}

impl State {
    /// The frame's draw `draw` of `mesh` with `material` and `shader`,
    /// posed by its layers: drawn skinned once its pipeline is there, or as
    /// it is.
    pub(super) fn pose_draw(
        &mut self,
        frame: &RenderFrame,
        scratch: &Scratch,
        draw: usize,
        (mesh, material, shader): (usize, &Material, usize),
        samples: u32,
    ) -> Posed {
        let rig: Arc<Rig> = match self.assets.current(mesh) {
            Some(GpuAsset::Mesh(MeshAsset { skin: Some(skin), .. })) => skin.rig.clone(),
            _ => return Posed::Unskinned,
        };
        let skinning = matches!(self.assets.current(shader), Some(GpuAsset::Shader(shader)) if shader.skinning);
        if !skinning || !material.opaque() {
            return Posed::Unskinned;
        }
        if !self.ensure_variants(shader, samples, Variants::SKINNED) {
            return Posed::Waiting;
        }
        let pipeline = self.pipelines(shader, samples).skinned.clone().expect("an ensured variant is compiled");
        let offset = self.skinning.pose(&rig, scratch.layers(frame, draw));
        Posed::Ready(pipeline, offset)
    }
}
