// Particles: the frame's emitters (pen-protocol's Emitter) simulated on the
// GPU and drawn over the scene's opaque and translucent surfaces. Each frame
// works out, on the CPU, which of each emitter's particles are alive (from
// its time, rate, lifetime and duration); a compute pass (particles.wgsl)
// then places each of them from its birth number and the emitter's seed
// alone, and they are drawn as camera-facing squares (particle_quads.wgsl).
// Nothing is kept between frames but buffers, so particles follow time
// travel and replays exactly. Alpha-blended particles are not sorted.
use anyhow::Result;
use pen_protocol::boundary::MAX_PARTICLES;
use pen_protocol::{Catalog, Emitter, RenderFrame};

use super::assets::texture_from;
use super::{GpuAsset, Scratch, State, uniform_buffer};
use crate::asset_store::AssetKind;
use crate::built_in::BuiltIn;
use crate::matrix::{multiply, narrow};
use crate::post::SCENE_FORMAT;
use crate::shader::PipelineContext;

const SIMULATE: &str = include_str!("../shaders/particles.wgsl");
const QUADS: &str = include_str!("../shaders/particle_quads.wgsl");

// The most particles a frame draws, all emitters together: 64 MB of them,
// within a storage binding and a dispatch.
const MOST: u32 = 1 << 21;
// What particles.wgsl's Emitter and Particle take.
const EMITTER_BYTES: u64 = 192;
const PARTICLE_BYTES: u64 = 32;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    transform: [f32; 16],
    velocity: [f32; 4],
    gravity: [f32; 4],
    color_start: [f32; 4],
    color_end: [f32; 4],
    shape: [f32; 4],
    timing: [f32; 4],
    counts: [u32; 4],
    seed: [u32; 4],
}

/// Which of an emitter's particles are alive at its time: `slots` of them
/// at most (its rate times its lifetime, and one), of which the first
/// `alive`, from birth number `first` on, oldest first; and the age of the
/// first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Living {
    pub slots: u32,
    pub alive: u32,
    pub first: f64,
    pub age: f64,
}

/// Its living particles, or None for an emitter with none to keep.
pub(crate) fn living(emitter: &Emitter) -> Option<Living> {
    let (time, rate, lifetime) = (f64::from(emitter.time), f64::from(emitter.rate), f64::from(emitter.lifetime));
    if !(rate > 0.0 && lifetime > 0.0 && time >= 0.0) {
        return None;
    }
    let most = MAX_PARTICLES as f64;
    let slots = ((lifetime * rate).ceil() + 1.0).min(most);
    // Born at k / rate: the last at or before now (and before the emitter
    // stops), the first that has not lived out its lifetime.
    let mut last = (time * rate).floor();
    if emitter.duration > 0.0 {
        last = last.min((f64::from(emitter.duration) * rate).ceil() - 1.0);
    }
    let first = ((time - lifetime) * rate).floor().max(-1.0) + 1.0;
    // The youngest, when there are more than the slots.
    let first = first.max(last - slots + 1.0);
    let alive = (last - first + 1.0).clamp(0.0, slots);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "whole numbers within MAX_PARTICLES")]
    Some(Living { slots: slots as u32, alive: alive as u32, first, age: time - first / rate })
}

/// The pipeline drawing particles into a scene of `samples`, adding their
/// light or covering what is behind them.
pub(crate) fn pipeline(context: &PipelineContext, samples: u32, additive: bool) -> wgpu::RenderPipeline {
    use wgpu::{BlendComponent, BlendFactor, BlendOperation};
    let device = &context.device;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("particles"),
        source: wgpu::ShaderSource::Wgsl(QUADS.into()),
    });
    let color = BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: if additive { BlendFactor::One } else { BlendFactor::OneMinusSrcAlpha },
        operation: BlendOperation::Add,
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("particles"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState { read_mask: 0, write_mask: 0, ..Default::default() },
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fragment"),
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

/// An emitter ready to draw: its pipeline, its bind group (the view, the
/// particles and its shape) and its slots among the frame's particles.
pub(crate) struct Ready {
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
    first: u32,
    count: u32,
}

pub(crate) struct Particles {
    /// The shape of particles whose material has no texture: a soft dot.
    dot: wgpu::TextureView,
    sampler: wgpu::Sampler,
    view: wgpu::Buffer,
    params: Vec<Params>,
    emitters: Option<wgpu::Buffer>,
    particles: Option<wgpu::Buffer>,
    simulate: Option<wgpu::ComputePipeline>,
    /// This frame's compute bind group, and its particles.
    group: Option<wgpu::BindGroup>,
    total: u32,
}

impl Particles {
    fn new(state: &mut State) -> Result<Self> {
        const SIZE: u32 = 64;
        let mut pixels = Vec::with_capacity((SIZE * SIZE * 4) as usize);
        for y in 0..SIZE {
            for x in 0..SIZE {
                let at = |value: u32| (f64::from(value) + 0.5) / f64::from(SIZE) * 2.0 - 1.0;
                let reach = (at(x).powi(2) + at(y).powi(2)).min(1.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "0..=255")]
                let alpha = ((1.0 - reach).powi(2) * 255.0).round() as u8;
                pixels.extend([255, 255, 255, alpha]);
            }
        }
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let dot = texture_from(&state.device, &state.queue, &mut state.mipmaps, (SIZE, SIZE), &pixels, format, true)?;
        let device = &state.device;
        Ok(Self {
            dot: dot.create_view(&Default::default()),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }),
            view: uniform_buffer(device, "particle view", 96),
            params: Vec::new(),
            emitters: None,
            particles: None,
            simulate: None,
            group: None,
            total: 0,
        })
    }

    /// The compute pass placing this frame's particles, if it has any.
    pub fn simulate(&self, encoder: &mut wgpu::CommandEncoder) {
        let (Some(group), Some(pipeline)) = (&self.group, &self.simulate) else { return };
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("particles"), timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(self.total.div_ceil(64), 1, 1);
    }
}

/// Draws the frame's ready emitters' particles into `pass`.
pub(super) fn draw(pass: &mut wgpu::RenderPass<'_>, emitters: &[Ready]) {
    for emitter in emitters {
        pass.set_pipeline(&emitter.pipeline);
        pass.set_bind_group(0, &emitter.group, &[]);
        pass.draw(0..6, emitter.first..emitter.first + emitter.count);
    }
}

/// A buffer of at least `size` bytes for `usage`, made anew if `buffer` is
/// smaller.
fn sized(device: &wgpu::Device, buffer: &mut Option<wgpu::Buffer>, size: u64, usage: wgpu::BufferUsages, label: &str) {
    if buffer.as_ref().is_none_or(|buffer| buffer.size() < size) {
        *buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.next_power_of_two(),
            usage,
            mapped_at_creation: false,
        }));
    }
}

impl State {
    /// Works out the frame's emitters: which particles are alive, the
    /// compute pass's input, and what each draws with. An emitter whose
    /// material, texture or pipeline is on its way waits.
    pub(super) fn prepare_particles(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        samples: u32,
        scratch: &mut Scratch,
    ) -> Result<()> {
        if let Some(particles) = &mut self.particles {
            particles.group = None;
            particles.total = 0;
        }
        if frame.emitters.is_empty() {
            return Ok(());
        }
        if self.particles.is_none() {
            self.particles = Some(Particles::new(self)?);
        }
        // Each emitter's slots, and the texture its particles take.
        let mut placed = Vec::new();
        let mut params = std::mem::take(&mut self.particles.as_mut().unwrap().params);
        params.clear();
        let mut total = 0u32;
        for emitter in &frame.emitters {
            let Some(living) = living(emitter) else { continue };
            if total + living.slots > MOST {
                break;
            }
            let Some(material) = self.material(catalog, emitter.material)? else { continue };
            let material = &material.material;
            let texture = match &material.texture {
                None => None,
                Some(key) => {
                    let slot = self.resolve(AssetKind::Texture, key)?;
                    match self.assets.current(slot) {
                        Some(GpuAsset::Texture(texture)) => Some(texture.view.clone()),
                        // A texture that failed shows its particles as dots.
                        _ if self.assets.loading(slot) => continue,
                        _ => None,
                    }
                }
            };
            let which = BuiltIn::Particles { samples, additive: emitter.additive };
            if !self.built_ins.ready([which]) {
                self.built_ins.prepare([which]);
                self.waiting = true;
                continue;
            }
            let tint = material.color.map(narrow);
            let color = |color: &pen_protocol::Color| {
                [color.r * tint[0], color.g * tint[1], color.b * tint[2], color.a * tint[3]]
            };
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a birth number, wrapping")]
            let first_born = living.first.rem_euclid(4_294_967_296.0) as u32;
            params.push(Params {
                transform: emitter.transform,
                velocity: [emitter.velocity.x, emitter.velocity.y, emitter.velocity.z, emitter.spread],
                gravity: [emitter.gravity.x, emitter.gravity.y, emitter.gravity.z, emitter.drag],
                color_start: color(&emitter.color_start),
                color_end: color(&emitter.color_end),
                shape: [emitter.radius, emitter.size_start, emitter.size_end, emitter.lifetime],
                timing: [narrow(living.age), narrow(1.0 / f64::from(emitter.rate)), 0.0, 0.0],
                counts: [total, living.slots, living.alive, first_born],
                seed: [emitter.seed, 0, 0, 0],
            });
            placed.push((which, texture, total, living.slots));
            total += living.slots;
        }
        let particles = self.particles.as_mut().unwrap();
        particles.params = params;
        if total == 0 {
            return Ok(());
        }
        let device = &self.device;
        let emitter_bytes = particles.params.len() as u64 * EMITTER_BYTES;
        let particle_bytes = u64::from(total) * PARTICLE_BYTES;
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        sized(device, &mut particles.emitters, emitter_bytes, storage, "emitters");
        sized(device, &mut particles.particles, particle_bytes, wgpu::BufferUsages::STORAGE, "particles");
        let (emitters, buffer) = (particles.emitters.as_ref().unwrap(), particles.particles.as_ref().unwrap());
        self.queue.write_buffer(emitters, 0, bytemuck::cast_slice(&particles.params));
        let view_projection = multiply(&frame.projection, &frame.view);
        let v = &frame.view;
        let mut view = view_projection.to_vec();
        view.extend([v[0], v[4], v[8], 0.0, v[1], v[5], v[9], 0.0]);
        self.queue.write_buffer(&particles.view, 0, bytemuck::cast_slice(&view));
        let simulate = particles.simulate.get_or_insert_with(|| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("particles"),
                source: wgpu::ShaderSource::Wgsl(SIMULATE.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("particles"),
                layout: None,
                module: &module,
                entry_point: Some("simulate"),
                compilation_options: Default::default(),
                cache: self.context.cache.get(),
            })
        });
        fn binding(buffer: &wgpu::Buffer, size: u64) -> wgpu::BindingResource<'_> {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset: 0, size: wgpu::BufferSize::new(size) })
        }
        particles.group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particles"),
            layout: &simulate.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: binding(emitters, emitter_bytes) },
                wgpu::BindGroupEntry { binding: 1, resource: binding(buffer, particle_bytes) },
            ],
        }));
        particles.total = total;
        for (which, texture, first, count) in placed {
            let pipeline = self.built_ins.get(which).clone();
            let particles = self.particles.as_ref().unwrap();
            let shape = texture.as_ref().unwrap_or(&particles.dot);
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particle quads"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: particles.view.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: binding(particles.particles.as_ref().unwrap(), particle_bytes),
                    },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(shape) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&particles.sampler) },
                ],
            });
            scratch.emitters.push(Ready { pipeline, group, first, count });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emitter(time: f32, rate: f32, lifetime: f32, duration: f32) -> Emitter {
        Emitter { time, rate, lifetime, duration, ..Default::default() }
    }

    #[test]
    fn the_living_follow_from_the_clock_alone() {
        // Ten a second living half a second: at 2.04 s those born at 1.6 ..
        // 2.0 s (birth numbers 16 to 20; 1.5 s's lived out its half second at 2.0).
        let living = super::living(&emitter(2.04, 10.0, 0.5, 0.0)).unwrap();
        assert_eq!((living.slots, living.alive, living.first), (6, 5, 16.0));
        assert!((living.age - 0.44).abs() < 1e-6, "{living:?}");
        // Just begun: only the first, at age 0.
        let living = super::living(&emitter(0.0, 10.0, 0.5, 0.0)).unwrap();
        assert_eq!((living.alive, living.first, living.age), (1, 0.0, 0.0));
        // Stopped at 1 s: births 0 to 9 only, and by 1.6 s none live on.
        let living = super::living(&emitter(1.15, 10.0, 0.5, 1.0)).unwrap();
        assert_eq!((living.alive, living.first), (3, 7.0));
        assert_eq!(super::living(&emitter(1.6, 10.0, 0.5, 1.0)).unwrap().alive, 0);
        // Nothing to keep: no rate, no lifetime, or before it began.
        assert!(super::living(&emitter(1.0, 0.0, 0.5, 0.0)).is_none());
        assert!(super::living(&emitter(1.0, 10.0, 0.0, 0.0)).is_none());
        assert!(super::living(&emitter(-1.0, 10.0, 0.5, 0.0)).is_none());
        // More than MAX_PARTICLES: the youngest of them.
        let living = super::living(&emitter(100.0, 10_000.0, 10.0, 0.0)).unwrap();
        let most = u32::try_from(MAX_PARTICLES).unwrap();
        assert_eq!((living.slots, living.alive), (most, most));
        assert_eq!(living.first, 1_000_000.0 - MAX_PARTICLES as f64 + 1.0);
    }
}
