// Skinned draws' poses. A frame's clip pose layers (`poses`) and/or explicit
// joint matrices (`bone_matrices`) name the draws they pose (either scene draws
// with `inset == 0` or inset draws with `inset == 1 + k`). A posed draw whose
// mesh has a skin and whose material's shader has a `vertex_skinned` entry
// point draws alone with its matching skinned pipeline (opaque, translucent,
// inset, or blended inset), its joint matrices laid in one storage buffer for
// the frame, bound at group 3 with a dynamic offset.
use std::collections::HashMap;
use std::sync::Arc;

use pen_protocol::RenderFrame;

use super::{GpuAsset, INSET_SAMPLES, Item, MeshAsset, ReadyEntry, Scratch, Shape, Source, State};
use crate::animation::{Layer, Matrix, Rig};
use crate::asset_store::Material;
use crate::matrix::IDENTITY;
use crate::shader::{Stage, Variants};

/// Group 3 of a skinned pipeline (after the lights): the draw's joint matrices, read by its
/// vertices, at a dynamic offset into the frame's.
pub(super) fn group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
    /// Builds the flat draw-target index mapping (`0..draws.len()` for scene
    /// draws, followed by each inset's draws) and sorts the frame's pose layers
    /// and explicit bone matrices by target index.
    pub(super) fn group_poses(&mut self, frame: &RenderFrame) {
        self.scene_draw_count = frame.draws.len();
        self.inset_draw_bases.clear();
        let mut total = frame.draws.len();
        for inset in &frame.insets {
            self.inset_draw_bases.push(total);
            total += inset.draws.len();
        }
        self.total_draw_targets = total;

        self.pose_first.clear();
        self.pose_order.clear();
        if !frame.poses.is_empty() {
            self.pose_first.resize(total + 1, 0);
            for pose in &frame.poses {
                if let Some(target) = self.draw_target_index(pose.inset, pose.draw) {
                    self.pose_first[target + 1] += 1;
                }
            }
            for index in 1..self.pose_first.len() {
                self.pose_first[index] += self.pose_first[index - 1];
            }
            let mut next: Vec<u32> = self.pose_first[..total].to_vec();
            self.pose_order.resize(frame.poses.len(), 0);
            for (index, pose) in frame.poses.iter().enumerate() {
                if let Some(target) = self.draw_target_index(pose.inset, pose.draw) {
                    let slot = &mut next[target];
                    self.pose_order[*slot as usize] = u32::try_from(index).unwrap_or(u32::MAX);
                    *slot += 1;
                }
            }
        }

        self.bone_first.clear();
        self.bone_order.clear();
        if !frame.bone_matrices.is_empty() {
            self.bone_first.resize(total + 1, 0);
            for bone in &frame.bone_matrices {
                if let Some(target) = self.draw_target_index(bone.inset, bone.draw) {
                    self.bone_first[target + 1] += 1;
                }
            }
            for index in 1..self.bone_first.len() {
                self.bone_first[index] += self.bone_first[index - 1];
            }
            let mut next: Vec<u32> = self.bone_first[..total].to_vec();
            self.bone_order.resize(frame.bone_matrices.len(), 0);
            for (index, bone) in frame.bone_matrices.iter().enumerate() {
                if let Some(target) = self.draw_target_index(bone.inset, bone.draw) {
                    let slot = &mut next[target];
                    self.bone_order[*slot as usize] = u32::try_from(index).unwrap_or(u32::MAX);
                    *slot += 1;
                }
            }
        }
    }

    pub(super) fn total_draw_targets(&self) -> usize {
        self.total_draw_targets
    }

    pub(super) fn draw_target_index(&self, inset: u32, draw: u32) -> Option<usize> {
        let draw = draw as usize;
        if inset == 0 {
            (draw < self.scene_draw_count).then_some(draw)
        } else {
            let k = (inset - 1) as usize;
            let base = *self.inset_draw_bases.get(k)?;
            let end = self.inset_draw_bases.get(k + 1).copied().unwrap_or(self.total_draw_targets);
            let idx = base + draw;
            (idx < end).then_some(idx)
        }
    }

    pub(super) fn target_of_source(&self, source: Source) -> Option<usize> {
        match source {
            Source::Draw(draw) => (draw < self.scene_draw_count).then_some(draw),
            Source::Inset(inset, draw) => {
                let base = *self.inset_draw_bases.get(inset)?;
                let end = self.inset_draw_bases.get(inset + 1).copied().unwrap_or(self.total_draw_targets);
                let idx = base + draw;
                (idx < end).then_some(idx)
            }
            Source::World => None,
        }
    }

    /// Whether draw target `target` has clip pose layers or explicit bone matrices.
    pub(super) fn posed(&self, target: usize) -> bool {
        let has_poses = self.pose_first.get(target + 1).is_some_and(|&end| end > self.pose_first[target]);
        let has_bones = self.bone_first.get(target + 1).is_some_and(|&end| end > self.bone_first[target]);
        has_poses || has_bones
    }

    /// Draw target `target`'s clip pose layers.
    pub(super) fn layers<'a>(&'a self, frame: &'a RenderFrame, target: usize) -> impl Iterator<Item = Layer> + 'a {
        let range = match self.pose_first.get(target + 1) {
            Some(&end) => self.pose_first[target] as usize..end as usize,
            None => 0..0,
        };
        self.pose_order[range].iter().map(move |&index| {
            let pose = &frame.poses[index as usize];
            Layer { clip: pose.clip, time: pose.time, weight: pose.weight }
        })
    }

    /// Draw target `target`'s explicit joint matrix overrides.
    pub(super) fn bone_matrices<'a>(
        &'a self,
        frame: &'a RenderFrame,
        target: usize,
    ) -> impl Iterator<Item = (u32, Matrix)> + 'a {
        let range = match self.bone_first.get(target + 1) {
            Some(&end) => self.bone_first[target] as usize..end as usize,
            None => 0..0,
        };
        self.bone_order[range].iter().map(move |&index| {
            let bone = &frame.bone_matrices[index as usize];
            (bone.joint, bone.matrix)
        })
    }
}

/// One frame's GPU joint buffer and per-draw offset table (double-buffered so
/// the velocity pass can read both current and previous skeletal palettes).
#[derive(Default)]
struct SkinSlot {
    buffer: Option<wgpu::Buffer>,
    /// The bind group, for the buffer and the binding's size it was made with.
    group: Option<(wgpu::BindGroup, u64)>,
    largest: u64,
    /// Maps `(draw_target, mesh_slot)` to its byte offset in `buffer`.
    offsets: HashMap<(usize, usize), u32>,
}

/// The frame's joint matrices: each posed draw's palette at an offset that
/// storage bindings allow, in one buffer bound once with each offset.
/// Double-buffered across frames so skeletal motion vectors have exact
/// previous-frame palettes without extra copies.
pub(super) struct Skinning {
    alignment: usize,
    /// This frame's palettes, as floats.
    data: Vec<f32>,
    /// The largest palette this frame, in bytes: what the binding spans.
    largest: u64,
    slots: [SkinSlot; 2],
    current: usize,
    layers: Vec<Layer>,
    palette: Vec<Matrix>,
}

/// What became of a posed draw.
pub(super) enum Posed {
    /// It is drawn as it is: its mesh has no skin, its shader no skinned
    /// entry point, or it is an overlay.
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
            slots: [SkinSlot::default(), SkinSlot::default()],
            current: 0,
            layers: Vec::new(),
            palette: Vec::new(),
        }
    }

    /// Starts a frame's palettes, swapping current/previous double-buffer slots.
    /// When `cut` is true, drops previous-frame history so cuts do not smear.
    pub fn begin(&mut self, cut: bool) {
        self.current = 1 - self.current;
        if cut {
            self.slots[1 - self.current].offsets.clear();
        }
        self.data.clear();
        self.largest = 0;
        self.slots[self.current].offsets.clear();
    }

    /// Lays `rig`'s joint matrices for `layers` and explicit `overrides` after
    /// the frame's others, and returns their offset in bytes.
    fn pose(
        &mut self,
        target: usize,
        mesh_slot: usize,
        rig: &Rig,
        layers: impl Iterator<Item = Layer>,
        overrides: impl Iterator<Item = (u32, Matrix)>,
    ) -> u32 {
        self.layers.clear();
        self.layers.extend(layers);
        if self.layers.is_empty() {
            self.palette.clear();
            self.palette.resize(rig.joints.len(), IDENTITY);
        } else {
            rig.palette(&self.layers, &mut self.palette);
        }
        for (joint, matrix) in overrides {
            if let Some(slot) = self.palette.get_mut(joint as usize) {
                *slot = matrix;
            }
        }
        let aligned = (self.data.len() * 4).next_multiple_of(self.alignment);
        self.data.resize(aligned / 4, 0.0);
        for matrix in &self.palette {
            self.data.extend_from_slice(matrix);
        }
        self.largest = self.largest.max(self.palette.len() as u64 * 64);
        let offset = u32::try_from(aligned).unwrap_or(u32::MAX);
        self.slots[self.current].offsets.insert((target, mesh_slot), offset);
        offset
    }

    /// Writes the frame's palettes for its draws to bind; the bind group,
    /// if any draw is posed.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, layout: &wgpu::BindGroupLayout) {
        let slot = &mut self.slots[self.current];
        if self.data.is_empty() {
            slot.largest = 0;
            return;
        }
        // Room for the last palette's whole binding past its offset.
        let used = (self.data.len() * 4) as u64;
        let size = used + self.largest;
        if slot.buffer.as_ref().is_none_or(|buffer| buffer.size() < size) {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("joints"),
                size: size.next_power_of_two(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            slot.buffer = Some(buffer);
            slot.group = None;
        }
        let buffer = slot.buffer.as_ref().unwrap();
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&self.data));
        slot.largest = self.largest;
        if slot.group.as_ref().is_none_or(|(_, binding)| *binding != self.largest) {
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
            slot.group = Some((group, self.largest));
        }
    }

    /// The bind group of this frame's palettes, if any draw is posed.
    pub fn group(&self) -> Option<&wgpu::BindGroup> {
        if self.data.is_empty() { None } else { self.slots[self.current].group.as_ref().map(|(group, _)| group) }
    }

    /// Byte offset of `(target, mesh_slot)` in the previous frame's joint buffer, if any.
    #[cfg_attr(not(feature = "render-taa"), allow(dead_code))]
    pub fn previous_offset(&self, target: usize, mesh_slot: usize) -> Option<u32> {
        let prev = &self.slots[1 - self.current];
        if prev.buffer.is_none() || prev.largest == 0 {
            return None;
        }
        prev.offsets.get(&(target, mesh_slot)).copied()
    }

    /// Current and previous joint storage buffers with their per-draw binding sizes.
    pub fn buffers(&self) -> [Option<(&wgpu::Buffer, u64)>; 2] {
        let curr = &self.slots[self.current];
        let prev = &self.slots[1 - self.current];
        let current = if self.data.is_empty() || curr.largest == 0 {
            None
        } else {
            curr.buffer.as_ref().map(|b| (b, curr.largest))
        };
        let previous = if prev.largest == 0 || prev.offsets.is_empty() {
            None
        } else {
            prev.buffer.as_ref().map(|b| (b, prev.largest))
        };
        [current, previous]
    }
}

impl ReadyEntry {
    /// How its vertices come: skinned, instanced or plain.
    pub(super) fn stage(&self, items: &[Item]) -> Stage {
        if self.persistent {
            Stage::Persistent
        } else if self.skin.is_some() {
            Stage::Skinned
        } else if matches!(items[self.item].shape, Shape::Batch(_)) {
            Stage::Instanced
        } else {
            Stage::Plain
        }
    }
}

impl State {
    /// The frame's draw target `target` of `mesh` with `material` and `shader`,
    /// posed by its clip layers and/or explicit bone matrices: drawn skinned
    /// once its pipeline is there (opaque, translucent, inset, or blended inset),
    /// or as it is.
    pub(super) fn pose_draw(
        &mut self,
        frame: &RenderFrame,
        scratch: &Scratch,
        target: usize,
        (mesh, material, shader): (usize, &Material, usize),
        samples: u32,
        inset: bool,
    ) -> Posed {
        let rig: Arc<Rig> = match self.assets.current(mesh) {
            Some(GpuAsset::Mesh(MeshAsset { skin: Some(skin), .. })) => skin.rig.clone(),
            _ => return Posed::Unskinned,
        };
        let skinning = matches!(self.assets.current(shader), Some(GpuAsset::Shader(shader)) if shader.skinning);
        if !skinning || material.overlay {
            return Posed::Unskinned;
        }
        let (target_samples, variant) = match (inset, material.opaque()) {
            (false, true) => (samples, Variants::SKINNED),
            (false, false) => (samples, Variants::SKINNED_TRANSLUCENT),
            (true, true) => (INSET_SAMPLES, Variants::INSET_SKINNED),
            (true, false) => (INSET_SAMPLES, Variants::INSET_SKINNED_BLENDED),
        };
        if !self.ensure_variants(shader, target_samples, variant) {
            return Posed::Waiting;
        }
        let made = self.pipelines(shader, target_samples);
        let pipeline = match (inset, material.opaque()) {
            (false, true) => made.skinned.clone(),
            (false, false) => made.skinned_translucent.clone(),
            (true, true) => made.inset_skinned.clone(),
            (true, false) => made.inset_skinned_blended.clone(),
        }
        .expect("an ensured variant is compiled");
        let offset =
            self.skinning.pose(target, mesh, &rig, scratch.layers(frame, target), scratch.bone_matrices(frame, target));
        Posed::Ready(pipeline, offset)
    }
}
