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
use crate::culling::{Detail, MAX_VIEWS, Sphere, Views};
use crate::shader::PipelineContext;

pub(crate) const CULL_WGSL: &str = include_str!("../shaders/cull.wgsl");

/// Batches of at least this many instances cull on the GPU, where it can.
pub(crate) const GPU_CULLING_FROM: u32 = 256;

// An instance's floats: its transform's four columns and its parameters.
const INSTANCE_BYTES: u64 = 80;
// A batch's description (`Batch` in cull.wgsl), in words.
const BATCH_WORDS: usize = 16;
// An indirect draw's arguments, in words.
const DRAW_WORDS: usize = 5;
// The views uniform: four words, the camera's row, six planes a view.
const UNIFORM_WORDS: usize = 8 + 4 * 6 * MAX_VIEWS;
const WORKGROUP: u32 = 64;

/// The compute pipeline that culls, made on the loader's threads.
pub(crate) fn pipeline(context: &PipelineContext) -> wgpu::ComputePipeline {
    let device = &context.device;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cull"),
        source: wgpu::ShaderSource::Wgsl(CULL_WGSL.into()),
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
/// draws' arguments lie.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuBatch {
    out: u32,
    draws: u32,
    count: u32,
    levels: u32,
}

/// What a batch's readback counts mean: its instances, the views it may be
/// drawn in, its levels and each level's indices.
#[derive(Clone, Copy, Default)]
struct Counted {
    count: u32,
    views: u8,
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
    /// The bind group, and the buffers it binds.
    group: Option<(wgpu::BindGroup, [wgpu::Buffer; 4])>,
    // This frame's batches, their draws' arguments and what they mean, the
    // instances they cull into, and the widest batch.
    batch_words: Vec<u32>,
    draw_words: Vec<u32>,
    counted: Vec<Counted>,
    out: u32,
    widest: u32,
    views: Views,
    detail: Detail,
    /// Each readback's view count and batches.
    readbacks: Readbacks<(usize, Vec<Counted>)>,
    readback: Option<usize>,
    latest: Kept,
    /// Whether the last frame culled any batch on the GPU.
    dispatched: bool,
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
            widest: 0,
            views: Views::default(),
            detail: Detail::default(),
            readbacks: Readbacks::new("culled draws read back", 3, device),
            readback: None,
            latest: Kept::default(),
            dispatched: false,
        }
    }

    /// Whether the device can, and whether its pipeline should be compiled.
    pub fn wanted(&mut self) -> bool {
        let wanted = self.supported && !self.requested;
        self.requested = true;
        wanted
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

    /// Whether a batch of `count` instances culls on the GPU this frame.
    pub fn takes(&self, count: u32) -> bool {
        self.pipeline.is_some() && self.from.is_some_and(|from| count >= from)
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
        self.widest = 0;
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
    pub fn add(&mut self, sphere: &Sphere, lods: &[Lod], first: u32, count: u32, views: u8) -> Option<GpuBatch> {
        let view_count = u32::try_from(self.views.count()).expect("at most MAX_VIEWS");
        let levels = u32::try_from(lods.len().clamp(1, MAX_LODS)).expect("at most MAX_LODS");
        let out = self.out.checked_add(count.checked_mul(view_count * levels)?)?;
        let limits = self.device.limits();
        let bound = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
        let inputs = u64::from(first) + u64::from(count);
        if (u64::from(out).max(inputs)) * INSTANCE_BYTES > bound {
            return None;
        }
        let draws = u32::try_from(self.draw_words.len() / DRAW_WORDS).ok()?;
        let batch = GpuBatch { out: self.out, draws, count, levels };
        let mut errors = [0.0f32; 4];
        for (error, lod) in errors.iter_mut().zip(lods) {
            *error = lod.error;
        }
        let float = f32::to_bits;
        self.batch_words.extend(sphere.center.map(float));
        self.batch_words.push(float(sphere.radius));
        self.batch_words.extend(errors.map(float));
        self.batch_words.extend([first, count, u32::from(views), levels, batch.out, draws, 0, 0]);
        debug_assert_eq!(self.batch_words.len() % BATCH_WORDS, 0, "a batch is cull.wgsl's Batch");
        for _ in 0..view_count {
            for lod in &lods[..levels as usize] {
                self.draw_words.extend([lod.count, 0, lod.first, 0, 0]);
            }
        }
        let mut indices = [0; MAX_LODS];
        for (indices, lod) in indices.iter_mut().zip(lods) {
            *indices = lod.count;
        }
        self.counted.push(Counted { count, views, levels, indices });
        self.out = out;
        self.widest = self.widest.max(count);
        Some(batch)
    }

    /// Encodes the compute pass that culls this frame's batches, reading
    /// their instances from `instances`.
    pub fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder, timer: &GpuTimer, instances: &wgpu::Buffer) {
        let Some(pipeline) = &self.pipeline else { return };
        if self.counted.is_empty() {
            return;
        }
        let device = &self.device;
        let mut uniform = Vec::with_capacity(UNIFORM_WORDS);
        let view_count = u32::try_from(self.views.count()).expect("at most MAX_VIEWS");
        uniform.extend([view_count, self.detail.pixels.to_bits(), 0, 0]);
        uniform.extend(self.detail.row.map(f32::to_bits));
        uniform.extend(self.views.planes().flatten().map(|lane| lane.to_bits()));
        uniform.resize(UNIFORM_WORDS, 0);
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
        if self.group.as_ref().is_none_or(|(_, buffers)| *buffers != bound) {
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
                ],
            });
            self.group = Some((group, bound));
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cull"),
            timestamp_writes: timer.compute(Span::Culling),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.group.as_ref().unwrap().0, &[]);
        let batches = u32::try_from(self.counted.len()).expect("batches fit a u32");
        pass.dispatch_workgroups(self.widest.div_ceil(WORKGROUP), batches, 1);
        self.dispatched = true;
    }

    /// Draws `batch`'s instances `view` kept, at each level, with `mesh`.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, batch: &GpuBatch, view: usize, mesh: &MeshAsset) -> u32 {
        let (Some(culled), Some(draws)) = (&self.culled.buffer, &self.draws.buffer) else { return 0 };
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        let view = u32::try_from(view).expect("at most MAX_VIEWS");
        for level in 0..batch.levels {
            let draw = view * batch.levels + level;
            let first = u64::from(batch.out + draw * batch.count) * INSTANCE_BYTES;
            let last = first + u64::from(batch.count) * INSTANCE_BYTES;
            pass.set_vertex_buffer(1, culled.slice(first..last));
            pass.draw_indexed_indirect(draws, u64::from(batch.draws + draw) * (DRAW_WORDS * 4) as u64);
        }
        batch.levels
    }

    /// Copies the frame's draws' arguments out to be read back; `submitted`
    /// follows once the encoder is submitted.
    pub fn finish(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.readback = None;
        let Some(draws) = &self.draws.buffer else { return };
        if self.counted.is_empty() || self.pipeline.is_none() {
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
    let words: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|&word| u32::from_le_bytes(word)).collect();
    let mut kept = Kept::default();
    let mut draw = 0;
    for batch in counted {
        for view in 0..views {
            let mut drawn = 0;
            for level in 0..batch.levels as usize {
                let instances = words.get((draw + view * batch.levels as usize + level) * DRAW_WORDS + 1).copied();
                let instances = instances.unwrap_or(0);
                drawn += instances;
                if view == 0 {
                    kept.triangles += u64::from(instances) * u64::from(batch.indices[level] / 3);
                }
            }
            if view == 0 {
                kept.instances += drawn;
                kept.culled += batch.count.saturating_sub(drawn);
            } else if batch.views & (1 << view) != 0 {
                kept.shadow_culled += batch.count.saturating_sub(drawn);
            }
        }
        draw += views * batch.levels as usize;
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
        // Two views; a batch of 10 instances at two levels (300 and 30
        // indices) that casts, and one of 4 at one level (6) that does not.
        let counted = [
            Counted { count: 10, views: 0b11, levels: 2, indices: [300, 30, 0, 0] },
            Counted { count: 4, views: 0b01, levels: 1, indices: [6, 0, 0, 0] },
        ];
        // Instance counts by batch, view and level: camera 3 + 4, shadow
        // 5 + 0; camera 2, shadow 0.
        let instances = [3, 4, 5, 0, 2, 0];
        let words: Vec<u32> = instances.iter().flat_map(|&count| [0, count, 0, 0, 0]).collect();
        let kept = kept(bytemuck::cast_slice(&words), &counted, 2);
        assert_eq!(kept, Kept { instances: 9, triangles: 3 * 100 + 4 * 10 + 2 * 2, culled: 3 + 2, shadow_culled: 5 });
    }
}
