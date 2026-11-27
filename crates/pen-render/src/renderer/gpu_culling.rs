#![cfg_attr(not(feature = "render-gpu-culling"), allow(dead_code, unused_imports))]
// Instanced batches culled on the GPU (shaders/cull.wgsl): a batch of many
// instances is written to the instance buffer once, whole, and a compute pass
// before the frame's passes tests each instance against every view, picks
// its level of detail and appends it to the instances its batch draws in each
// view at that level, counting them into indirect draw arguments. The
// batch then draws with `draw_indexed_indirect`, one draw per view and level,
// and the CPU never touches its instances one by one. Smaller batches, and
// devices without compute shaders, cull on the CPU (batching.rs).
//
// How many instances each view kept comes back a few frames later
// (readback.rs), for the frame's counters.
use super::readback::Readbacks;
use super::timing::{GpuTimer, Span};
use super::{MeshAsset, State};
use crate::asset_store::{Lod, MAX_LODS};
use crate::culling::{Detail, MAX_VIEWS, Sphere, ViewMask, Views};
use crate::shader::PipelineContext;

/// Batches of at least this many instances cull on the GPU, where it can.
pub(crate) const GPU_CULLING_FROM: u32 = 32;

// An instance's floats: its transform's four columns and its parameters.
const INSTANCE_BYTES: u64 = 80;
// A batch's description (`Batch` in cull.wgsl), in words.
const BATCH_WORDS: usize = 16;
// An indirect draw's arguments, in words.
const DRAW_WORDS: usize = 5;
// The shader's `planes` array holds six planes for each of the views.
const _: () = assert!(MAX_VIEWS == 24, "cull.wgsl's planes array is 6 * MAX_VIEWS");

// The views uniform: header (4), dispatch (4), camera row (4), hiz size (4), hiz vp (16), six planes a view.
const UNIFORM_WORDS: usize = 32 + 4 * 6 * MAX_VIEWS;
const WORKGROUP: u32 = 64;

/// The compute pipeline that culls, made on the loader's threads.
pub(crate) fn pipeline(context: &PipelineContext) -> wgpu::ComputePipeline {
    let device = &context.device;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cull"),
        source: wgpu::ShaderSource::Wgsl(crate::deferred_wgsl!("cull.wgsl").into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cull"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: context.cache.get(),
    })
}

/// Whether the device can cull on the GPU: compute shaders, and storage
/// buffers enough.
fn supported(device: &wgpu::Device) -> bool {
    let limits = device.limits();
    limits.max_compute_workgroups_per_dimension > 0
        && limits.max_compute_invocations_per_workgroup >= WORKGROUP
        && limits.max_compute_workgroup_size_x >= WORKGROUP
        && limits.max_storage_buffers_per_shader_stage >= 4
}

/// A batch culled on the GPU this frame: where its culled instances and its
/// draws' arguments lie, plus its active view bitmask (masked to the frame's views).
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuBatch {
    out: u32,
    draws: u32,
    count: u32,
    levels: u32,
    views: ViewMask,
}

/// What a batch's readback counts mean: its instances, the views it may be
/// drawn in, its levels and each level's indices.
#[derive(Clone, Copy, Default)]
struct Counted {
    count: u32,
    views: ViewMask,
    levels: u32,
    indices: [u32; MAX_LODS],
}

/// What culling on the GPU kept, as last read back.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Kept {
    /// Instances the camera's view drew, and their triangles.
    pub instances: u32,
    pub triangles: u64,
    /// Instances left out of the camera's view, and out of shadow maps.
    pub culled: u32,
    pub shadow_culled: u32,
}

/// A buffer that grows (to the next power of two) as a frame needs more.
struct Grown {
    label: &'static str,
    usage: wgpu::BufferUsages,
    buffer: Option<wgpu::Buffer>,
}

impl Grown {
    fn new(label: &'static str, usage: wgpu::BufferUsages) -> Self {
        Self { label, usage, buffer: None }
    }

    fn at_least(&mut self, device: &wgpu::Device, bytes: u64) -> &wgpu::Buffer {
        if self.buffer.as_ref().is_none_or(|buffer| buffer.size() < bytes) {
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: bytes.next_power_of_two().max(256),
                usage: self.usage,
                mapped_at_creation: false,
            }));
        }
        self.buffer.as_ref().unwrap()
    }
}

pub(crate) struct GpuCulling {
    device: wgpu::Device,
    queue: wgpu::Queue,
    supported: bool,
    /// Asked of the loader; there once it has compiled, unless it failed.
    requested: bool,
    pipeline: Option<wgpu::ComputePipeline>,
    failed: bool,
    /// Batches of at least this many instances cull on the GPU; none, and
    /// all cull on the CPU.
    pub from: Option<u32>,
    uniform: wgpu::Buffer,
    batches: Grown,
    culled: Grown,
    draws: Grown,
    /// The bind group, the buffers it binds, and the Hi-Z pyramid view ID it bound.
    group: Option<(wgpu::BindGroup, [wgpu::Buffer; 4], u64)>,
    // This frame's batches, their draws' arguments and what they mean, the
    // instances they cull into, and total flattened instances across batches.
    batch_words: Vec<u32>,
    draw_words: Vec<u32>,
    counted: Vec<Counted>,
    out: u32,
    total_instances: u32,
    batch_count: u32,
    views: Views,
    detail: Detail,
    /// Each readback's view count and batches.
    readbacks: Readbacks<(usize, Vec<Counted>)>,
    readback: Option<usize>,
    latest: Kept,
    /// Whether the last frame culled any batch on the GPU.
    dispatched: bool,
    stats_readback: bool,
}

impl GpuCulling {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        Self {
            device: device.clone(),
            queue: queue.clone(),
            supported: supported(device),
            requested: false,
            pipeline: None,
            failed: false,
            from: Some(GPU_CULLING_FROM),
            uniform: super::uniform_buffer(device, "cull views", (UNIFORM_WORDS * 4) as u64),
            batches: Grown::new("cull batches", storage),
            culled: Grown::new("culled instances", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX),
            draws: Grown::new("culled draws", storage | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC),
            group: None,
            batch_words: Vec::new(),
            draw_words: Vec::new(),
            counted: Vec::new(),
            out: 0,
            total_instances: 0,
            batch_count: 0,
            views: Views::default(),
            detail: Detail::default(),
            readbacks: Readbacks::new("culled draws read back", 3, device),
            readback: None,
            latest: Kept::default(),
            dispatched: false,
            stats_readback: true,
        }
    }

    /// Whether the device can, and whether its pipeline should be compiled.
    pub fn wanted(&mut self) -> bool {
        #[cfg(not(feature = "render-gpu-culling"))]
        return false;
        #[cfg(feature = "render-gpu-culling")]
        {
            let wanted = self.supported && !self.requested;
            self.requested = true;
            wanted
        }
    }

    /// The pipeline the loader compiled.
    pub fn arrived(&mut self, result: Result<wgpu::ComputePipeline, String>) {
        match result {
            Ok(pipeline) => self.pipeline = Some(pipeline),
            Err(message) => {
                self.failed = true;
                log::warn!("culling pipeline: {message}");
            }
        }
    }

    /// Whether its pipeline is compiling.
    pub fn compiling(&self) -> bool {
        // A device that cannot cull on the GPU never asks.
        self.supported && self.requested && self.pipeline.is_none() && !self.failed
    }

    /// Whether GPU compute culling dispatched at least one batch this frame.
    pub fn active(&self) -> bool {
        self.dispatched
    }

    /// Enables or disables indirect draw-arg CPU readback for culling stats.
    pub fn set_stats_readback(&mut self, enabled: bool) {
        self.stats_readback = enabled;
        if !enabled {
            self.latest = Kept::default();
        }
    }

    /// Whether a batch of `count` instances culls on the GPU this frame.
    pub fn takes(&self, count: u32) -> bool {
        #[cfg(not(feature = "render-gpu-culling"))]
        {
            let _ = count;
            false
        }
        #[cfg(feature = "render-gpu-culling")]
        {
            self.pipeline.is_some() && self.from.is_some_and(|from| count >= from)
        }
    }

    /// What was kept, as last read back.
    pub fn latest(&self) -> Kept {
        self.latest
    }

    /// Starts a frame of `views`, its levels picked by `detail`; takes in
    /// what has come back.
    pub fn begin(&mut self, views: Views, detail: Detail) {
        self.batch_words.clear();
        self.draw_words.clear();
        self.counted.clear();
        self.out = 0;
        self.total_instances = 0;
        self.batch_count = 0;
        self.views = views;
        self.detail = detail;
        let latest = &mut self.latest;
        self.readbacks.take(|bytes, (views, counted)| *latest = kept(bytes, counted, *views));
        // Without batches of its own, the last frame kept nothing here; what
        // comes back still is from frames before it.
        if !std::mem::take(&mut self.dispatched) {
            self.latest = Kept::default();
        }
    }

    /// Adds a batch of `count` instances from `first` in the instance
    /// buffer, culled by `sphere` (in its mesh's space) in `views` (a bit
    /// each), drawing its mesh's `lods`; None if its instances, or those it
    /// culls, would pass what a storage buffer binds, and it culls on the
    /// CPU.
    pub fn add(&mut self, sphere: &Sphere, lods: &[Lod], first: u32, count: u32, views: ViewMask) -> Option<GpuBatch> {
        let view_count = u32::try_from(self.views.count()).expect("at most MAX_VIEWS");
        let view_mask_all = if view_count >= 32 { u32::MAX } else { (1u32 << view_count) - 1 };
        let active_mask = views & view_mask_all;
        let active_views = active_mask.count_ones();
        let levels = u32::try_from(lods.len().clamp(1, MAX_LODS)).expect("at most MAX_LODS");
        let draws = u32::try_from(self.draw_words.len() / DRAW_WORDS).ok()?;
        let mut indices = [0; MAX_LODS];
        for (dst, lod) in indices.iter_mut().zip(lods) {
            *dst = lod.count;
        }
        if count == 0 || active_views == 0 {
            if count > 0 {
                self.counted.push(Counted { count, views: active_mask, levels, indices });
            }
            return Some(GpuBatch { out: self.out, draws, count, levels, views: active_mask });
        }
        let out = self.out.checked_add(count.checked_mul(active_views * levels)?)?;
        let next_total = self.total_instances.checked_add(count)?;
        let limits = self.device.limits();
        let bound = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
        let inputs = u64::from(first) + u64::from(count);
        if (u64::from(out).max(inputs)) * INSTANCE_BYTES > bound {
            return None;
        }
        let batch = GpuBatch { out: self.out, draws, count, levels, views: active_mask };
        let mut errors = [0.0f32; 4];
        for (error, lod) in errors.iter_mut().zip(lods) {
            *error = lod.error;
        }
        let float = f32::to_bits;
        self.batch_words.extend(sphere.center.map(float));
        self.batch_words.push(float(sphere.radius));
        self.batch_words.extend(errors.map(float));
        self.batch_words.extend([first, count, active_mask, levels, batch.out, draws, self.total_instances, 0]);
        debug_assert_eq!(self.batch_words.len() % BATCH_WORDS, 0, "a batch is cull.wgsl's Batch");
        for _ in 0..active_views {
            for lod in &lods[..levels as usize] {
                self.draw_words.extend([lod.count, 0, lod.first, 0, 0]);
            }
        }
        self.counted.push(Counted { count, views: active_mask, levels, indices });
        self.out = out;
        self.total_instances = next_total;
        self.batch_count += 1;
        Some(batch)
    }

    /// Encodes the compute pass that culls this frame's batches, reading
    /// their instances from `instances` and testing camera visibility against `hiz`.
    pub fn dispatch(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        timer: &GpuTimer,
        instances: &wgpu::Buffer,
        hiz: &super::hiz::HiZ,
    ) {
        let Some(pipeline) = &self.pipeline else { return };
        if self.batch_count == 0 || self.total_instances == 0 {
            return;
        }
        let device = &self.device;
        let (hiz_view, hiz_view_id, hiz_vp, hiz_size, hiz_mips, hiz_active) = hiz.source();
        let hiz_active = hiz_active && hiz.depth_mode() == crate::camera_precision::DepthMode::ForwardZ;
        let max_wg = device.limits().max_compute_workgroups_per_dimension.max(1);
        let total_wg = self.total_instances.div_ceil(WORKGROUP);
        let (wg_x, wg_y) = if total_wg <= max_wg { (total_wg, 1) } else { (max_wg, total_wg.div_ceil(max_wg)) };
        let row_stride = wg_x.saturating_mul(WORKGROUP);
        let mut uniform = [0u32; UNIFORM_WORDS];
        let view_count = u32::try_from(self.views.count()).expect("at most MAX_VIEWS");
        uniform[0] = view_count;
        uniform[1] = self.detail.pixels.to_bits();
        uniform[2] = u32::from(hiz_active);
        uniform[3] = hiz_mips;
        uniform[4] = self.batch_count;
        uniform[5] = self.total_instances;
        uniform[6] = row_stride;
        uniform[7] = 0;
        for (dst, lane) in uniform[8..12].iter_mut().zip(self.detail.row) {
            *dst = lane.to_bits();
        }
        #[allow(clippy::cast_precision_loss, reason = "screen dimensions in pixels")]
        let (hw, hh) = (hiz_size.0.max(1) as f32, hiz_size.1.max(1) as f32);
        for (dst, lane) in uniform[12..16].iter_mut().zip([hw, hh, 1.0 / hw, 1.0 / hh]) {
            *dst = lane.to_bits();
        }
        for (dst, lane) in uniform[16..32].iter_mut().zip(hiz_vp) {
            *dst = lane.to_bits();
        }
        for (dst, lane) in uniform[32..].iter_mut().zip(self.views.planes().flatten()) {
            *dst = lane.to_bits();
        }
        self.queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&uniform));
        let batches = self.batches.at_least(device, (self.batch_words.len() * 4) as u64).clone();
        self.queue.write_buffer(&batches, 0, bytemuck::cast_slice(&self.batch_words));
        let draws = self.draws.at_least(device, (self.draw_words.len() * 4) as u64).clone();
        self.queue.write_buffer(&draws, 0, bytemuck::cast_slice(&self.draw_words));
        let culled = self.culled.at_least(device, u64::from(self.out) * INSTANCE_BYTES).clone();
        // The batches' instances lie first in the buffer, within what a
        // storage buffer binds (`add`).
        let limit = device.limits().max_storage_buffer_binding_size;
        let inputs = wgpu::BufferSize::new(instances.size().min(limit / INSTANCE_BYTES * INSTANCE_BYTES));
        let bound = [batches, instances.clone(), culled, draws];
        if self.group.as_ref().is_none_or(|(_, buffers, view_id)| *buffers != bound || *view_id != hiz_view_id) {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cull"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: bound[0].as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &bound[1],
                            offset: 0,
                            size: inputs,
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 3, resource: bound[2].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: bound[3].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(hiz_view) },
                ],
            });
            self.group = Some((group, bound, hiz_view_id));
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cull"),
            timestamp_writes: timer.compute(Span::Culling),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.group.as_ref().unwrap().0, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        self.dispatched = true;
    }

    /// Draws `batch`'s instances `view` kept, at each level, with `mesh`.
    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        batch: &GpuBatch,
        view: usize,
        mesh: &MeshAsset,
        tangents: &wgpu::Buffer,
        extra_attrs: &wgpu::Buffer,
    ) -> u32 {
        let view = u32::try_from(view).expect("at most MAX_VIEWS");
        if batch.count == 0 || (batch.views & (1u32 << view)) == 0 {
            return 0;
        }
        let (Some(culled), Some(draws)) = (&self.culled.buffer, &self.draws.buffer) else { return 0 };
        let compact_view = (batch.views & ((1u32 << view) - 1)).count_ones();
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, tangents.slice(..));
        pass.set_vertex_buffer(3, extra_attrs.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        for level in 0..batch.levels {
            let draw = compact_view * batch.levels + level;
            let first = u64::from(batch.out + draw * batch.count) * INSTANCE_BYTES;
            let last = first + u64::from(batch.count) * INSTANCE_BYTES;
            pass.set_vertex_buffer(2, culled.slice(first..last));
            pass.draw_indexed_indirect(draws, u64::from(batch.draws + draw) * (DRAW_WORDS * 4) as u64);
        }
        batch.levels
    }

    /// Draws `batch`'s culled instances with vertex buffer 0 = positions and buffer 1 = instances
    /// (used by the velocity pass, which skips tangents).
    pub fn draw_positions(&self, pass: &mut wgpu::RenderPass<'_>, batch: &GpuBatch, view: usize, mesh: &MeshAsset) {
        let view = u32::try_from(view).expect("at most MAX_VIEWS");
        if batch.count == 0 || (batch.views & (1u32 << view)) == 0 {
            return;
        }
        let (Some(culled), Some(draws)) = (&self.culled.buffer, &self.draws.buffer) else { return };
        let compact_view = (batch.views & ((1u32 << view) - 1)).count_ones();
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        for level in 0..batch.levels {
            let draw = compact_view * batch.levels + level;
            let first = u64::from(batch.out + draw * batch.count) * INSTANCE_BYTES;
            let last = first + u64::from(batch.count) * INSTANCE_BYTES;
            pass.set_vertex_buffer(1, culled.slice(first..last));
            pass.draw_indexed_indirect(draws, u64::from(batch.draws + draw) * (DRAW_WORDS * 4) as u64);
        }
    }

    /// Copies the frame's draws' arguments out to be read back; `submitted`
    /// follows once the encoder is submitted.
    pub fn finish(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.readback = None;
        if !self.stats_readback {
            return;
        }
        let Some(draws) = &self.draws.buffer else { return };
        if self.draw_words.is_empty() || self.counted.is_empty() || self.pipeline.is_none() {
            return;
        }
        let bytes = (self.draw_words.len() * 4) as u64;
        self.readback = self.readbacks.free(&self.device, bytes);
        if let Some(index) = self.readback {
            encoder.copy_buffer_to_buffer(draws, 0, self.readbacks.buffer(index), 0, bytes);
        }
    }

    pub fn submitted(&mut self) {
        if let Some(index) = self.readback.take() {
            let about = (self.views.count(), std::mem::take(&mut self.counted));
            self.readbacks.submitted(index, (self.draw_words.len() * 4) as u64, about);
        }
    }
}

/// What each view kept of the batches `counted` describes, from their
/// draws' arguments as read back.
fn kept(bytes: &[u8], counted: &[Counted], views: usize) -> Kept {
    let words = bytes.as_chunks::<4>().0;
    let mut kept = Kept::default();
    let mut draw = 0;
    let view_mask_all = if views >= 32 { u32::MAX } else { (1u32 << views) - 1 };
    for batch in counted {
        let active_mask = batch.views & view_mask_all;
        for view in 0..views {
            let active = (active_mask & (1u32 << view)) != 0;
            if !active {
                if view == 0 {
                    kept.culled += batch.count;
                }
                continue;
            }
            let mut drawn = 0;
            for level in 0..batch.levels as usize {
                let instances = words.get((draw + level) * DRAW_WORDS + 1).map_or(0, |&word| u32::from_le_bytes(word));
                drawn += instances;
                if view == 0 {
                    kept.triangles += u64::from(instances) * u64::from(batch.indices[level] / 3);
                }
            }
            draw += batch.levels as usize;
            if view == 0 {
                kept.instances += drawn;
                kept.culled += batch.count.saturating_sub(drawn);
            } else {
                kept.shadow_culled += batch.count.saturating_sub(drawn);
            }
        }
    }
    kept
}

impl State {
    /// Asks the loader for the culling pipeline, where the device can cull.
    pub(super) fn request_culling(&mut self) {
        if self.culling.wanted() {
            self.loader.jobs.push(crate::loader::Job::Culling, crate::loader::Priority::Background);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readbacks_count_what_each_view_kept() {
        // Three views; a batch of 10 instances at two levels (300 and 30 indices)
        // active in views 0 and 2 (skipping view 1), and one of 4 at one level (6)
        // active in camera view 0 only.
        let counted = [
            Counted { count: 10, views: 0b101, levels: 2, indices: [300, 30, 0, 0] },
            Counted { count: 4, views: 0b001, levels: 1, indices: [6, 0, 0, 0] },
        ];
        // Compact draw slots only for active views:
        // batch 0: view 0 (3, 4), view 2 (5, 0)
        // batch 1: view 0 (2)
        let instances = [3, 4, 5, 0, 2];
        let words: Vec<u32> = instances.iter().flat_map(|&count| [0, count, 0, 0, 0]).collect();
        let kept = kept(bytemuck::cast_slice(&words), &counted, 3);
        assert_eq!(kept, Kept { instances: 9, triangles: 3 * 100 + 4 * 10 + 2 * 2, culled: 3 + 2, shadow_culled: 5 });
    }
}
