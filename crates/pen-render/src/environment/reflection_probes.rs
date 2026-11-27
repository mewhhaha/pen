// Local reflection probes. The game publishes boxes of the world (the
// packet's `reflection_probes`); the host draws each probe's cube from its
// point of view, with the frame's own opaque surfaces, lights, sun and shadows
// over the sky, keeps up to `SLOTS` of them prefiltered (GGX levels, as the
// sky cube's) in one cube array, and shaders blend those whose boxes hold a
// surface (`local_reflection` in environment.wgsl).
//
// A cube is drawn when the probe has none, its revision differs from the one
// it holds, or its `capture` flag rises, `reflection.budget` faces a frame:
// the frame's draws seen from the probe, one face at a time into a scratch
// target, finished into a scratch cube (`probe_finalise.wgsl`: the sky laid
// behind, mirrored as a cube's texels are, and the light of the diffuse probe
// lattice there in alpha). With the sixth face the scratch cube is halved
// into levels and filtered into the probe's slot (`probe_filter.wgsl`) in
// that frame, so the slot holds the old cube until the new one is whole.
// One probe is drawn at a time; the nearest, highest priority one that has
// no cube goes first.
//
// Which probes hold slots: the frame's probes by priority, then by how near
// their box is to the camera (those already held count as 20% nearer, so two
// equally near ones do not trade a slot back and forth), then by id.
use std::collections::HashMap;
use std::ops::Range;

use pen_protocol::{Matrix, ReflectionProbe, RenderFrame};

use super::{Environment, FORMAT, STRIDE, clear, layout_entries};
use crate::pipeline_cache::PipelineCache;
use crate::post::SCENE_FORMAT;

/// The cubes the host keeps (`MAX_REFLECTION_SLOTS`).
pub const SLOTS: usize = 8;
const SLOT_COUNT: u32 = 8;
/// A cube face's width in texels, and the levels down to four texels; level
/// `n` holds roughness `n / (LEVELS - 1)`.
pub(crate) const SIZE: u32 = 128;
pub(crate) const LEVELS: u32 = 6;
/// The scratch cube's levels, down to one texel.
const SCRATCH_LEVELS: u32 = 8;
/// GGX samples per texel of a rough level.
const SAMPLES: f32 = 96.0;
/// How near the camera a probe is to hold its slot over another equally near,
/// once it holds one.
const KEEP: f32 = 0.64;
/// The uniform of the probes' table: its count, then each probe's five vec4s.
pub(crate) const TABLE_BYTES: u64 = 16 + 80 * SLOTS as u64;
/// How near the capture's frustum reaches, in world units.
pub(crate) const NEAR: f32 = 0.05;

const FINALISE: &str = crate::wgsl!("probe_finalise.wgsl");
const FILTER: &str = crate::wgsl!("probe_filter.wgsl");

// Pass uniforms: faces finished in a frame, then the halvings, then the filters.
const FINALISE_PASSES: u64 = 6;
const DOWNSAMPLE_PASSES: u64 = 6 * (SCRATCH_LEVELS as u64 - 1);
const FILTER_PASSES: u64 = 6 * (LEVELS as u64 - 1);
const DOWNSAMPLE_FIRST: u64 = FINALISE_PASSES;
const FILTER_FIRST: u64 = DOWNSAMPLE_FIRST + DOWNSAMPLE_PASSES;
const PASS_COUNT: u64 = FILTER_FIRST + FILTER_PASSES;

/// What the host holds for a slot.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Held {
    id: u32,
    /// The revision its cube was drawn at; none while it holds no cube.
    captured: Option<u32>,
    /// Asked to be drawn again by a rising `capture`.
    dirty: bool,
}

/// The probe being drawn: its slot, the revision it draws for, and how many
/// faces are done.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Job {
    pub id: u32,
    pub slot: usize,
    pub revision: u32,
    pub done: u32,
}

/// What a frame is to draw of a probe: `faces` of the job's probe (the
/// whole cube is `0..6`), and whether the last of them finishes it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Step {
    pub job: Job,
    pub probe: ReflectionProbe,
    pub faces: Range<u32>,
    pub finishes: bool,
}

/// Which probes hold slots, which of them hold cubes, and what is being drawn.
#[derive(Default)]
pub(crate) struct Memory {
    held: [Option<Held>; SLOTS],
    job: Option<Job>,
    /// Each probe's `capture` flag in the last frame, to see it rise.
    flags: HashMap<u32, bool>,
}

fn distance_to_box(probe: &ReflectionProbe, eye: [f32; 3]) -> f32 {
    let centre = [probe.center.x, probe.center.y, probe.center.z];
    let extent = [probe.extent.x, probe.extent.y, probe.extent.z];
    (0..3).map(|axis| (eye[axis] - centre[axis]).abs().sub_or_zero(extent[axis]).powi(2)).sum::<f32>().sqrt()
}

trait SubOrZero {
    fn sub_or_zero(self, other: f32) -> f32;
}
impl SubOrZero for f32 {
    fn sub_or_zero(self, other: f32) -> f32 {
        (self - other).max(0.0)
    }
}

impl Memory {
    /// The indices of the frame's probes that hold slots, best first: by
    /// priority (highest), then nearness of their box to `eye` (nearest;
    /// held ones as if 20% nearer), then id. A probe whose id repeats is
    /// the first of that id.
    pub fn choose(&self, probes: &[ReflectionProbe], eye: [f32; 3]) -> Vec<usize> {
        let held = |id: u32| self.held.iter().flatten().any(|held| held.id == id);
        let mut seen = std::collections::HashSet::new();
        let mut ranked: Vec<(usize, f32)> = probes
            .iter()
            .enumerate()
            .filter(|(_, probe)| seen.insert(probe.id))
            .map(|(index, probe)| {
                let near = distance_to_box(probe, eye).powi(2);
                (index, if held(probe.id) { near * KEEP } else { near })
            })
            .collect();
        crate::sort::insertion_sort_by(&mut ranked, |&(a, near_a), &(b, near_b)| {
            probes[b]
                .priority
                .cmp(&probes[a].priority)
                .then(near_a.total_cmp(&near_b))
                .then(probes[a].id.cmp(&probes[b].id))
        });
        ranked.truncate(SLOTS);
        ranked.into_iter().map(|(index, _)| index).collect()
    }

    /// Which probe of `probes` holds `slot`, if any.
    fn slot_of(&self, id: u32) -> Option<usize> {
        self.held.iter().position(|held| held.is_some_and(|held| held.id == id))
    }

    /// Plans the frame: gives slots to the chosen probes (those held keep
    /// theirs, the rest take the lowest free ones, and a dropped probe's
    /// slot goes with it), notes `capture` flags rising, and picks what to
    /// draw of the probe being captured under `budget` faces, if `allowed`.
    /// Nothing is drawn until `commit`.
    pub fn plan(&mut self, probes: &[ReflectionProbe], eye: [f32; 3], budget: u32, allowed: bool) -> Option<Step> {
        let chosen = self.choose(probes, eye);
        let ids: Vec<u32> = chosen.iter().map(|&index| probes[index].id).collect();
        // Rising flags, over every probe this frame has.
        for probe in probes {
            let before = self.flags.insert(probe.id, probe.capture).unwrap_or(false);
            if let (true, Some(slot)) = (probe.capture && !before, self.slot_of(probe.id)) {
                self.held[slot].as_mut().unwrap().dirty = true;
                if self.job.is_some_and(|job| job.id == probe.id) {
                    self.job.as_mut().unwrap().done = 0;
                }
            }
        }
        self.flags.retain(|id, _| probes.iter().any(|probe| probe.id == *id));
        for slot in 0..SLOTS {
            if self.held[slot].is_some_and(|held| !ids.contains(&held.id)) {
                self.held[slot] = None;
            }
        }
        for &id in &ids {
            if self.slot_of(id).is_none() {
                let free = self.held.iter().position(Option::is_none).expect("at most SLOTS probes are chosen");
                self.held[free] = Some(Held { id, captured: None, dirty: false });
            }
        }
        // Stale ones, the cubeless first, then in the order the probes rank.
        let stale = |held: &Held, probe: &ReflectionProbe| held.captured != Some(probe.revision) || held.dirty;
        let wanted = chosen
            .iter()
            .map(|&index| &probes[index])
            .filter(|probe| {
                let held = self.held[self.slot_of(probe.id).unwrap()].as_ref().unwrap();
                stale(held, probe)
            })
            .min_by_key(|probe| {
                let held = self.held[self.slot_of(probe.id).unwrap()].as_ref().unwrap();
                (held.captured.is_some(), chosen.iter().position(|&i| probes[i].id == probe.id))
            })
            .copied();
        // The job goes on while its probe is still chosen and wanted at its revision.
        if let Some(job) = self.job {
            let current = chosen.iter().map(|&index| &probes[index]).find(|probe| probe.id == job.id);
            match current {
                Some(probe) if self.slot_of(job.id) == Some(job.slot) => {
                    let held = self.held[job.slot].as_ref().unwrap();
                    if probe.revision != job.revision && stale(held, probe) {
                        self.job = Some(Job { revision: probe.revision, done: 0, ..job });
                    } else if !stale(held, probe) {
                        self.job = None;
                    }
                }
                _ => self.job = None,
            }
        }
        if self.job.is_none() {
            self.job = wanted.map(|probe| Job {
                id: probe.id,
                slot: self.slot_of(probe.id).unwrap(),
                revision: probe.revision,
                done: 0,
            });
        }
        let job = self.job?;
        let probe = *probes.iter().find(|probe| probe.id == job.id)?;
        let faces = if allowed { budget.min(6 - job.done) } else { 0 };
        (faces > 0).then_some(Step { job, probe, faces: job.done..job.done + faces, finishes: job.done + faces == 6 })
    }

    /// The frame drew `step` (or, with `drawn` false, could not: the job
    /// waits). A finished cube is the probe's from now on.
    pub fn commit(&mut self, step: Option<&Step>, drawn: bool) {
        let Some(step) = step.filter(|_| drawn) else { return };
        let Some(job) = self.job.as_mut() else { return };
        job.done = step.faces.end;
        if step.finishes {
            if let Some(held) = self.held[step.job.slot].as_mut() {
                held.captured = Some(step.job.revision);
                held.dirty = false;
            }
            self.job = None;
        }
    }

    /// The table rows: the probes of `probes` that hold a cube, with their
    /// slots, by priority (highest first) then id.
    pub fn rows(&self, probes: &[ReflectionProbe]) -> Vec<(usize, usize)> {
        let mut rows: Vec<(usize, usize)> = probes
            .iter()
            .enumerate()
            .filter_map(|(index, probe)| {
                let slot = self.slot_of(probe.id)?;
                let held = self.held[slot]?;
                (held.captured.is_some() && probes.iter().position(|other| other.id == probe.id) == Some(index))
                    .then_some((index, slot))
            })
            .collect();
        crate::sort::insertion_sort_by(&mut rows, |&(a, _), &(b, _)| {
            probes[b].priority.cmp(&probes[a].priority).then_with(|| probes[a].id.cmp(&probes[b].id))
        });
        rows
    }
}

/// The table `ENVIRONMENT_WGSL` reads: the count, then each probe's position
/// and slot, influence box and blend, priority, and parallax box.
pub(crate) fn table_floats(probes: &[ReflectionProbe], rows: &[(usize, usize)]) -> Vec<f32> {
    let mut floats = vec![0.0; (TABLE_BYTES / 4) as usize];
    floats[0] = rows.len() as f32;
    for (row, &(index, slot)) in rows.iter().enumerate() {
        let probe = &probes[index];
        let at = 4 + row * 20;
        let parallax = [probe.parallax_extent.x, probe.parallax_extent.y, probe.parallax_extent.z];
        let boxed = parallax.iter().all(|&half| half > 0.0);
        let values = [
            [probe.position.x, probe.position.y, probe.position.z, slot as f32],
            [probe.center.x, probe.center.y, probe.center.z, probe.blend.max(0.0)],
            [probe.extent.x, probe.extent.y, probe.extent.z, probe.priority as f32],
            [probe.parallax_center.x, probe.parallax_center.y, probe.parallax_center.z, if boxed { 1.0 } else { 0.0 }],
            [parallax[0], parallax[1], parallax[2], 0.0],
        ];
        for (lane, value) in values.iter().enumerate() {
            floats[at + lane * 4..at + lane * 4 + 4].copy_from_slice(value);
        }
    }
    floats
}

/// The camera's view for cube face `face` of a probe at `position`, with
/// `face_projection`: it sees the cube's texels (see `face_direction` in the
/// shaders) mirrored left to right, which the finishing pass undoes.
pub(crate) fn face_view(face: u32, position: [f32; 3]) -> Matrix {
    // Where each face looks, and which way is down its texels.
    let (forward, down): ([f32; 3], [f32; 3]) = match face {
        0 => ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        1 => ([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        2 => ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        3 => ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
        4 => ([0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
        _ => ([0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
    };
    let up = down.map(|v| -v);
    // A right-handed camera's right, which is the cube's left.
    let right = [
        forward[1] * up[2] - forward[2] * up[1],
        forward[2] * up[0] - forward[0] * up[2],
        forward[0] * up[1] - forward[1] * up[0],
    ];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let back = forward.map(|v| -v);
    #[rustfmt::skip]
    let view = [
        right[0], up[0], back[0], 0.0,
        right[1], up[1], back[1], 0.0,
        right[2], up[2], back[2], 0.0,
        -dot(right, position), -dot(up, position), -dot(back, position), 1.0,
    ];
    view
}

/// A perspective projection of 90 degrees both ways from `near` to `far`.
pub(crate) fn face_projection(far: f32) -> Matrix {
    let mut m = [0.0; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = far / (NEAR - far);
    m[11] = -1.0;
    m[14] = NEAR * far / (NEAR - far);
    m
}

/// The view-projection of the box of half size `far` round `position`: the
/// clip volume that sees every draw a face of the probe may, for culling.
pub(crate) fn box_view(position: [f32; 3], far: f32) -> Matrix {
    let mut m = [0.0; 16];
    let scale = 1.0 / far;
    m[0] = scale;
    m[5] = scale;
    m[10] = 0.5 * scale;
    m[12] = -position[0] * scale;
    m[13] = -position[1] * scale;
    m[14] = 0.5 - 0.5 * position[2] * scale;
    m[15] = 1.0;
    m
}

/// What the capture draws into: the scene's samples (when multisampled), the
/// image they resolve to, and the depth.
#[derive(Clone)]
pub(crate) struct CaptureTargets {
    pub samples: u32,
    pub multisampled: Option<wgpu::TextureView>,
    pub resolved: wgpu::TextureView,
    pub depth: wgpu::TextureView,
}

pub(crate) struct ReflectionProbes {
    pub(super) cubes: wgpu::Texture,
    pub(super) cubes_view: wgpu::TextureView,
    /// The table the scene reads, and one with no probes.
    pub(super) table: wgpu::Buffer,
    pub(super) empty_table: wgpu::Buffer,
    scratch: wgpu::Texture,
    scratch_cube: wgpu::TextureView,
    passes: wgpu::Buffer,
    statics: bool,
    targets: Option<CaptureTargets>,
    pipelines: Option<Pipelines>,
    pub(crate) memory: Memory,
}

struct Pipelines {
    finalise: wgpu::RenderPipeline,
    environment: wgpu::BindGroupLayout,
    face: wgpu::BindGroupLayout,
    downsample: wgpu::RenderPipeline,
    downsample_layout: wgpu::BindGroupLayout,
    prefilter: wgpu::RenderPipeline,
    prefilter_layout: wgpu::BindGroupLayout,
    filter_sampler: wgpu::Sampler,
    /// Cached per-(level, face) 2D views of `scratch` (`SCRATCH_LEVELS * 6`).
    scratch_views: Vec<wgpu::TextureView>,
    /// Cached downsample bind groups for `(level - 1, face)` (`(SCRATCH_LEVELS - 1) * 6`).
    downsample_groups: Vec<wgpu::BindGroup>,
    /// Cached single prefilter bind group reading `scratch_cube` + dynamic pass uniform.
    prefilter_group: wgpu::BindGroup,
    /// Cached per-(slot, level, face) 2D target views of `cubes` (`SLOTS * (LEVELS - 1) * 6`).
    slot_filter_views: Vec<wgpu::TextureView>,
}

fn pass_uniform(entry: u32, visibility: wgpu::ShaderStages, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: entry,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

fn texture_entry(binding: u32, dimension: wgpu::TextureViewDimension) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: dimension,
            multisampled: false,
        },
        count: None,
    }
}

impl ReflectionProbes {
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = |label: &str, layers: u32, levels: u32, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: layers },
                mip_level_count: levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage,
                view_formats: &[],
            })
        };
        let cubes = texture(
            "reflection cubes",
            6 * SLOT_COUNT,
            LEVELS,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
        );
        let cubes_view = cubes.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        });
        let scratch = texture(
            "reflection scratch",
            6,
            SCRATCH_LEVELS,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let scratch_cube = scratch.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        Self {
            cubes,
            cubes_view,
            table: super::buffer(device, "reflection probes", TABLE_BYTES, wgpu::BufferUsages::UNIFORM),
            empty_table: super::buffer(device, "no reflection probes", TABLE_BYTES, wgpu::BufferUsages::UNIFORM),
            scratch,
            scratch_cube,
            passes: super::buffer(device, "reflection passes", STRIDE * PASS_COUNT, wgpu::BufferUsages::UNIFORM),
            statics: false,
            targets: None,
            pipelines: None,
            memory: Memory::default(),
        }
    }

    /// The uniform of pass `index`, in `STRIDE`-sized slots.
    fn offset(index: u64) -> u32 {
        u32::try_from(index * STRIDE).expect("a few kilobytes")
    }
}

/// The eye of a view matrix: where it sees from.
pub(crate) fn eye_of(frame: &RenderFrame) -> [f32; 3] {
    let inverse = crate::matrix::invert(&frame.view);
    [inverse[12], inverse[13], inverse[14]]
}

impl Environment {
    /// Plans what the frame captures of its reflection probes, when `allowed`.
    pub fn plan_reflections(&mut self, frame: &RenderFrame, allowed: bool) -> Option<Step> {
        self.reflections.memory.plan(&frame.reflection_probes, eye_of(frame), frame.reflection.budget, allowed)
    }

    /// The frame drew its capture (`drawn`) or did not: what holds a cube now goes
    /// into the table the scene reads.
    pub fn commit_reflections(&mut self, queue: &wgpu::Queue, frame: &RenderFrame, step: Option<&Step>, drawn: bool) {
        self.reflections.memory.commit(step, drawn);
        self.write_table(queue, &frame.reflection_probes);
    }

    /// The captured cubes' table: written with what holds a cube now.
    pub fn write_table(&mut self, queue: &wgpu::Queue, probes: &[ReflectionProbe]) {
        let rows = self.reflections.memory.rows(probes);
        queue.write_buffer(&self.reflections.table, 0, bytemuck::cast_slice(&table_floats(probes, &rows)));
    }

    /// The images a capture draws the scene into, for `samples` per pixel.
    pub fn capture_targets(&mut self, device: &wgpu::Device, samples: u32) -> CaptureTargets {
        let reflections = &mut self.reflections;
        if reflections.targets.as_ref().is_none_or(|targets| targets.samples != samples) {
            let texture = |samples: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("reflection capture"),
                        size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: samples,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage,
                        view_formats: &[],
                    })
                    .create_view(&Default::default())
            };
            let attachment = wgpu::TextureUsages::RENDER_ATTACHMENT;
            reflections.targets = Some(CaptureTargets {
                samples,
                multisampled: (samples > 1).then(|| texture(samples, SCENE_FORMAT, attachment)),
                resolved: texture(1, SCENE_FORMAT, attachment | wgpu::TextureUsages::TEXTURE_BINDING),
                depth: texture(samples, wgpu::TextureFormat::Depth24PlusStencil8, attachment),
            });
        }
        reflections.targets.clone().expect("made above")
    }

    fn reflection_pipelines(&mut self, device: &wgpu::Device) {
        if self.reflections.pipelines.is_some() {
            return;
        }
        let cache: &PipelineCache = &self.cache;
        let module = |label: &str, source: String| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        };
        let entries = layout_entries();
        let environment = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reflection finish environment"),
            entries: &entries[..5],
        });
        let face = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reflection finish face"),
            entries: &[
                pass_uniform(0, wgpu::ShaderStages::FRAGMENT, 32),
                texture_entry(1, wgpu::TextureViewDimension::D2),
            ],
        });
        let sampler = wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let downsample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reflection halving"),
            entries: &[
                pass_uniform(0, wgpu::ShaderStages::FRAGMENT, 32),
                texture_entry(2, wgpu::TextureViewDimension::D2),
                sampler,
            ],
        });
        let prefilter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reflection filter"),
            entries: &[
                pass_uniform(0, wgpu::ShaderStages::FRAGMENT, 32),
                texture_entry(1, wgpu::TextureViewDimension::Cube),
                sampler,
            ],
        });
        let make =
            |label: &str, module: &wgpu::ShaderModule, fragment: &str, layouts: &[Option<&wgpu::BindGroupLayout>]| {
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: layouts,
                    immediate_size: 0,
                });
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module,
                        entry_point: Some("vertex"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module,
                        entry_point: Some(fragment),
                        compilation_options: Default::default(),
                        targets: &[Some(FORMAT.into())],
                    }),
                    multiview_mask: None,
                    cache: cache.get(),
                })
            };
        let finishing = module("reflection finish", format!("{}\n{FINALISE}", super::ENVIRONMENT_WGSL));
        let filtering = module("reflection filter", FILTER.to_string());
        let filter_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("reflection filter"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let face_view = |texture: &wgpu::Texture, level: u32, layer: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: level,
                mip_level_count: Some(1),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        };
        let mut scratch_views = Vec::with_capacity((SCRATCH_LEVELS * 6) as usize);
        for level in 0..SCRATCH_LEVELS {
            for face in 0..6u32 {
                scratch_views.push(face_view(&self.reflections.scratch, level, face));
            }
        }
        let pass_uniform_entry = wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.reflections.passes,
                offset: 0,
                size: wgpu::BufferSize::new(32),
            }),
        };
        let mut downsample_groups = Vec::with_capacity(((SCRATCH_LEVELS - 1) * 6) as usize);
        for level in 1..SCRATCH_LEVELS {
            for face in 0..6u32 {
                let from = &scratch_views[((level - 1) * 6 + face) as usize];
                downsample_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("reflection halving"),
                    layout: &downsample_layout,
                    entries: &[
                        pass_uniform_entry.clone(),
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(from) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&filter_sampler) },
                    ],
                }));
            }
        }
        let prefilter_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reflection filter"),
            layout: &prefilter_layout,
            entries: &[
                pass_uniform_entry,
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.reflections.scratch_cube),
                },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&filter_sampler) },
            ],
        });
        let mut slot_filter_views = Vec::with_capacity(SLOTS * ((LEVELS - 1) * 6) as usize);
        for slot in 0..SLOT_COUNT {
            for level in 1..LEVELS {
                for face in 0..6u32 {
                    slot_filter_views.push(face_view(&self.reflections.cubes, level, slot * 6 + face));
                }
            }
        }
        self.reflections.pipelines = Some(Pipelines {
            finalise: make("reflection finish", &finishing, "fragment", &[Some(&environment), Some(&face)]),
            downsample: make("reflection halving", &filtering, "downsample", &[Some(&downsample_layout)]),
            prefilter: make("reflection filter", &filtering, "prefilter", &[Some(&prefilter_layout)]),
            environment,
            face,
            downsample_layout,
            prefilter_layout,
            filter_sampler,
            scratch_views,
            downsample_groups,
            prefilter_group,
            slot_filter_views,
        });
    }

    /// Writes the passes' uniforms that never change.
    fn write_statics(&mut self, queue: &wgpu::Queue) {
        if self.reflections.statics {
            return;
        }
        self.reflections.statics = true;
        let mut data = vec![0.0f32; (STRIDE * PASS_COUNT / 4) as usize];
        let mut put = |pass: u64, values: [f32; 8]| {
            let at = (pass * STRIDE / 4) as usize;
            data[at..at + 8].copy_from_slice(&values);
        };
        for level in 1..SCRATCH_LEVELS {
            let size = (SIZE >> level).max(1) as f32;
            for face in 0..6u32 {
                put(
                    DOWNSAMPLE_FIRST + u64::from(level - 1) * 6 + u64::from(face),
                    [face as f32, 0.0, 0.0, size, 0.0, 0.0, 0.0, 0.0],
                );
            }
        }
        for level in 1..LEVELS {
            let roughness = level as f32 / (LEVELS - 1) as f32;
            let size = (SIZE >> level).max(1) as f32;
            for face in 0..6u32 {
                put(
                    FILTER_FIRST + u64::from(level - 1) * 6 + u64::from(face),
                    [face as f32, roughness, SAMPLES, size, SIZE as f32, 0.0, 0.0, 0.0],
                );
            }
        }
        queue.write_buffer(&self.reflections.passes, 0, bytemuck::cast_slice(&data));
    }

    #[allow(clippy::too_many_arguments, reason = "the pass's pieces, each its own")]
    /// Finishes face `face` of the probe at `position` from the scene drawn
    /// into `targets.resolved`, into the scratch cube; `index` is how many
    /// faces this frame has finished before it.
    pub fn finish_face(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        targets: &CaptureTargets,
        face: u32,
        index: u64,
        position: [f32; 3],
    ) {
        self.reflection_pipelines(device);
        self.write_statics(queue);
        queue.write_buffer(
            &self.reflections.passes,
            index * STRIDE,
            bytemuck::cast_slice(&[face as f32, SIZE as f32, 0.0, 0.0, position[0], position[1], position[2], 0.0]),
        );
        let pipelines = self.reflections.pipelines.as_ref().unwrap();
        let all = self.entries(&self.light);
        let first = [all[0].clone(), all[1].clone(), all[2].clone(), all[3].clone(), all[4].clone()];
        let environment = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reflection finish environment"),
            layout: &pipelines.environment,
            entries: &first,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reflection finish face"),
            layout: &pipelines.face,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.reflections.passes,
                        offset: 0,
                        size: wgpu::BufferSize::new(32),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&targets.resolved) },
            ],
        });
        let view = &pipelines.scratch_views[face as usize];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("reflection finish"),
            color_attachments: &[Some(clear(view))],
            ..Default::default()
        });
        pass.set_pipeline(&pipelines.finalise);
        pass.set_bind_group(0, &environment, &[]);
        pass.set_bind_group(1, &group, &[ReflectionProbes::offset(index)]);
        pass.draw(0..3, 0..1);
    }

    /// With the scratch cube's six faces finished: halves them down to a
    /// texel, and fills `slot`'s levels, the first a copy and the rest GGX
    /// filtered.
    pub fn finish_probe(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        slot: usize,
    ) {
        self.reflection_pipelines(device);
        self.write_statics(queue);
        let reflections = &self.reflections;
        let pipelines = reflections.pipelines.as_ref().unwrap();
        let _ = &pipelines.filter_sampler;
        let _ = (&pipelines.downsample_layout, &pipelines.prefilter_layout);
        // The scratch cube, halved level by level.
        for level in 1..SCRATCH_LEVELS {
            for face in 0..6u32 {
                let to = &pipelines.scratch_views[(level * 6 + face) as usize];
                let group = &pipelines.downsample_groups[((level - 1) * 6 + face) as usize];
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("reflection halving"),
                    color_attachments: &[Some(clear(to))],
                    ..Default::default()
                });
                pass.set_pipeline(&pipelines.downsample);
                let index = DOWNSAMPLE_FIRST + u64::from(level - 1) * 6 + u64::from(face);
                pass.set_bind_group(0, group, &[ReflectionProbes::offset(index)]);
                pass.draw(0..3, 0..1);
            }
        }
        // Level 0 of the slot is the scratch's.
        for face in 0..6u32 {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &reflections.scratch,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: face },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &reflections.cubes,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: u32::try_from(slot).expect("at most SLOTS") * 6 + face },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            );
        }
        let slot_stride = ((LEVELS - 1) * 6) as usize;
        for level in 1..LEVELS {
            for face in 0..6u32 {
                let to = &pipelines.slot_filter_views[slot * slot_stride + ((level - 1) * 6 + face) as usize];
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("reflection filter"),
                    color_attachments: &[Some(clear(to))],
                    ..Default::default()
                });
                pass.set_pipeline(&pipelines.prefilter);
                let index = FILTER_FIRST + u64::from(level - 1) * 6 + u64::from(face);
                pass.set_bind_group(0, &pipelines.prefilter_group, &[ReflectionProbes::offset(index)]);
                pass.draw(0..3, 0..1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_protocol::Vector;

    fn probe(id: u32, priority: u32, at: [f32; 3]) -> ReflectionProbe {
        let v = |a: [f32; 3]| Vector { x: a[0], y: a[1], z: a[2] };
        ReflectionProbe {
            id,
            revision: 1,
            priority,
            position: v(at),
            center: v(at),
            extent: v([1.0; 3]),
            far: 20.0,
            ..Default::default()
        }
    }

    /// Frames until `probes` all hold cubes at `budget` faces a frame.
    fn frames_to_capture(probes: &[ReflectionProbe], budget: u32) -> (u32, Memory) {
        let mut memory = Memory::default();
        for frame in 1..=200 {
            let step = memory.plan(probes, [0.0; 3], budget, true);
            memory.commit(step.as_ref(), true);
            if memory.rows(probes).len() == probes.len().min(SLOTS) {
                return (frame, memory);
            }
        }
        panic!("never finished");
    }

    #[test]
    fn the_budget_spreads_a_capture_over_the_frames_it_takes() {
        let probes: Vec<_> = (0..3).map(|id| probe(id + 1, 0, [id as f32 * 5.0, 0.0, 0.0])).collect();
        // Three probes of six faces: 18 faces.
        for (budget, frames) in [(1, 18), (2, 9), (3, 6), (4, 6), (6, 3)] {
            assert_eq!(frames_to_capture(&probes, budget).0, frames, "budget {budget}");
        }
        // Each frame's faces are consecutive, and the sixth finishes it.
        let mut memory = Memory::default();
        let one = [probe(7, 0, [0.0; 3])];
        let first = memory.plan(&one, [0.0; 3], 4, true).unwrap();
        assert_eq!((first.faces.clone(), first.finishes), (0..4, false));
        memory.commit(Some(&first), true);
        let second = memory.plan(&one, [0.0; 3], 4, true).unwrap();
        assert_eq!((second.faces.clone(), second.finishes), (4..6, true));
        // Nothing drawn (an asset was missing) keeps its place.
        let mut memory = Memory::default();
        let step = memory.plan(&one, [0.0; 3], 4, true).unwrap();
        memory.commit(Some(&step), false);
        assert_eq!(memory.plan(&one, [0.0; 3], 4, true).unwrap().faces, 0..4);
        // None while not allowed.
        assert!(memory.plan(&one, [0.0; 3], 4, false).is_none());
    }

    #[test]
    fn a_new_revision_or_a_rising_flag_captures_again_and_the_old_cube_stays_until_then() {
        let mut probes = vec![probe(1, 0, [0.0; 3])];
        let (_, mut memory) = frames_to_capture(&probes, 6);
        assert!(memory.plan(&probes, [0.0; 3], 6, true).is_none(), "up to date: nothing to draw");
        probes[0].revision = 2;
        let step = memory.plan(&probes, [0.0; 3], 3, true).unwrap();
        memory.commit(Some(&step), true);
        // Half drawn: the probe still has its cube in the table.
        assert_eq!(memory.rows(&probes), vec![(0, 0)]);
        let step = memory.plan(&probes, [0.0; 3], 3, true).unwrap();
        assert!(step.finishes);
        memory.commit(Some(&step), true);
        assert!(memory.plan(&probes, [0.0; 3], 6, true).is_none());
        // The flag rises once, and a flag that stays up does not repeat it.
        probes[0].capture = true;
        let step = memory.plan(&probes, [0.0; 3], 6, true).expect("rose");
        memory.commit(Some(&step), true);
        assert!(memory.plan(&probes, [0.0; 3], 6, true).is_none(), "still up: no new capture");
        probes[0].capture = false;
        assert!(memory.plan(&probes, [0.0; 3], 6, true).is_none());
        probes[0].capture = true;
        assert!(memory.plan(&probes, [0.0; 3], 6, true).is_some(), "rose again");
    }

    #[test]
    fn slots_go_to_priority_then_nearness_deterministically_and_hold_steady() {
        // Ten probes along x, the farther ones of higher priority except one.
        let mut probes: Vec<_> = (0..10).map(|id| probe(id + 1, 0, [id as f32 * 10.0, 0.0, 0.0])).collect();
        probes[9].priority = 5;
        let memory = Memory::default();
        let chosen: Vec<u32> = memory.choose(&probes, [0.0; 3]).iter().map(|&i| probes[i].id).collect();
        assert_eq!(chosen, vec![10, 1, 2, 3, 4, 5, 6, 7]);
        // The same inputs in another order choose the same probes.
        let mut shuffled = probes.clone();
        shuffled.reverse();
        let again: Vec<u32> = memory.choose(&shuffled, [0.0; 3]).iter().map(|&i| shuffled[i].id).collect();
        assert_eq!(again, chosen);
        // Two equally near probes do not trade a slot as the camera sways: the holder keeps it.
        let two = [probe(1, 0, [-3.0, 0.0, 0.0]), probe(2, 0, [3.0, 0.0, 0.0])];
        let crowd: Vec<_> = (0..SLOT_COUNT - 1).map(|n| probe(100 + n, 1, [0.0, 50.0, 0.0])).collect();
        let all: Vec<_> = crowd.iter().chain(&two).copied().collect();
        let mut memory = Memory::default();
        let first = memory.plan(&all, [0.0, 0.0, 0.0], 0, true);
        assert!(first.is_none());
        let held = |memory: &Memory, id| memory.slot_of(id).is_some();
        let had_one = held(&memory, 1);
        for x in [-0.2, 0.2, -0.2, 0.2] {
            memory.plan(&all, [x, 0.0, 0.0], 0, true);
            assert_eq!(held(&memory, 1), had_one, "a swaying camera does not move the slots");
        }
    }

    #[test]
    fn faces_look_down_the_cube_directions_mirrored() {
        // `face_direction` of the shaders, in f64.
        let direction = |face: u32, u: f64, v: f64| -> [f64; 3] {
            let (sc, tc) = (u * 2.0 - 1.0, v * 2.0 - 1.0);
            match face {
                0 => [1.0, -tc, -sc],
                1 => [-1.0, -tc, sc],
                2 => [sc, 1.0, tc],
                3 => [sc, -1.0, -tc],
                4 => [sc, -tc, 1.0],
                _ => [-sc, -tc, -1.0],
            }
        };
        let projection = face_projection(50.0);
        for face in 0..6 {
            let view = face_view(face, [1.0, 2.0, 3.0]);
            for (u, v) in [(0.5, 0.5), (0.1, 0.3), (0.9, 0.8), (0.25, 0.75)] {
                let d = direction(face, u, v);
                let point = [1.0 + d[0] * 5.0, 2.0 + d[1] * 5.0, 3.0 + d[2] * 5.0, 1.0];
                let multiply = |m: &Matrix, p: [f64; 4]| -> [f64; 4] {
                    std::array::from_fn(|row| (0..4).map(|k| f64::from(m[k * 4 + row]) * p[k]).sum())
                };
                let clip = multiply(&projection, multiply(&view, point));
                let (x, y) = (clip[0] / clip[3] * 0.5 + 0.5, 0.5 - clip[1] / clip[3] * 0.5);
                // The image is mirrored left to right: finishing reads it at 1 - u.
                assert!((x - (1.0 - u)).abs() < 1e-5 && (y - v).abs() < 1e-5, "face {face} ({u},{v}): {x} {y}");
                assert!(clip[2] / clip[3] > 0.0 && clip[2] / clip[3] < 1.0);
            }
        }
    }

    #[test]
    fn the_table_packs_each_probe_in_the_order_given() {
        let mut a = probe(1, 3, [1.0, 2.0, 3.0]);
        a.blend = 0.5;
        a.parallax_extent = Vector { x: 4.0, y: 5.0, z: 6.0 };
        let floats = table_floats(&[a], &[(0, 5)]);
        assert_eq!(&floats[..4], &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(&floats[4..8], &[1.0, 2.0, 3.0, 5.0]);
        assert_eq!(&floats[8..12], &[1.0, 2.0, 3.0, 0.5]);
        assert_eq!(&floats[12..16], &[1.0, 1.0, 1.0, 3.0]);
        assert_eq!(floats[19], 1.0, "a box to project onto");
        assert_eq!(&floats[20..23], &[4.0, 5.0, 6.0]);
        // No projection when any half size is not positive.
        a.parallax_extent.y = 0.0;
        assert_eq!(table_floats(&[a], &[(0, 0)])[19], 0.0);
    }

    #[test]
    fn the_capture_box_sees_what_the_probe_does() {
        let view = box_view([10.0, 0.0, -4.0], 20.0);
        let frustum = crate::culling::Frustum::new(&view);
        let ball = |x: f32, y: f32, z: f32| crate::culling::Sphere { center: [x, y, z], radius: 1.0 };
        assert!(frustum.sees(&ball(10.0, 0.0, -4.0)));
        assert!(frustum.sees(&ball(29.0, 0.0, -4.0)));
        assert!(!frustum.sees(&ball(32.0, 0.0, -4.0)));
        assert!(!frustum.sees(&ball(10.0, 0.0, 17.5)));
        assert!(frustum.sees(&ball(10.0, 0.0, 15.5)));
        assert!(!frustum.sees(&ball(10.0, -22.5, -4.0)));
    }
}
