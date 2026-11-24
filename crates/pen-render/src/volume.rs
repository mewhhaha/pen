// Participating media: the frame's mist drawn through a froxel volume. A grid
// aligned with the camera (froxels across and down the picture, slices deep
// along the view, spaced exponentially from the eye out to `fog.distance`) is
// filled each frame with the medium's extinction and the light it sends toward
// the eye (`shaders/volume_inject.wgsl`: the global height mist with its
// noise, the frame's local fog volumes, lit by the sun's cascades and the
// clouds' shadows, the clustered lights and their shadow maps, and the sky's
// or the probes' ambient light, through a Henyey-Greenstein phase), blended
// with the last frame's grid at the same place in the world, whose samples sit
// at jittered positions within their froxels, then integrated front to back
// down each column (`shaders/volume_integrate.wgsl`). The atmosphere pass
// (`shaders/atmosphere.wgsl`) reads that integral at each surface's depth, and
// continues the same medium analytically past the grid's reach.
//
// Ink's `volumetrics` is the CPU reference of the formulas; the f64 versions
// of the slicing, the medium and the integral live in this file's tests and
// in `tests/volumetrics.rs`.
use pen_protocol::{Matrix, Quality, RenderFrame};

use crate::matrix::multiply;
use crate::post::atmosphere_of;
use crate::shader::PipelineContext;

const COMMON: &str = include_str!("shaders/volume_common.wgsl");
const INJECT: &str = include_str!("shaders/volume_inject.wgsl");
const INTEGRATE: &str = include_str!("shaders/volume_integrate.wgsl");
const ATMOSPHERE_COMMON: &str = include_str!("shaders/atmosphere_common.wgsl");

/// WGSL: the volume's uniform and slicing, which the passes that fill the
/// volume and the atmosphere pass that applies it share.
pub(crate) const COMMON_WGSL: &str = COMMON;

/// The scene uniform as the volume's injection declares it.
const SCENE: &str = "struct Scene {
  view: mat4x4f, projection: mat4x4f, ambient: vec4f, light_direction: vec4f, light_color: vec4f,
  light_view_projection: mat4x4f, shadow: vec4f, zenith: vec4f, horizon: vec4f, ground: vec4f, eye: vec4f,
  cascades: vec4f,
  cascade_view_projection: array<mat4x4f, 4>,
  cascade_texels: array<vec4f, 4>,
  shadow_params: vec4f,
};
@group(0) @binding(0) var<uniform> scene: Scene;
";

/// The slices' depths follow `knee * (exp(b / slices * span) - 1)` with the
/// knee this share of the reach: the first slice is as thin as the camera
/// looks at, the last as thick as the distance asks.
pub(crate) const KNEE_DIVISOR: f64 = 64.0;

/// The most local fog volumes a frame publishes (the wire's MAX_FOG_VOLUMES).
const MAX_LOCAL: usize = pen_protocol::boundary::MAX_FOG_VOLUMES;

const UNIFORM_FLOATS: usize = 68;
const UNIFORM_BYTES: u64 = UNIFORM_FLOATS as u64 * 4;
const LOCAL_BYTES: u64 = (MAX_LOCAL * 64) as u64;

/// The weight of the history in the temporal blend of a froxel that holds.
const HISTORY: f32 = 0.9;

/// How far the camera may move in a frame, as a share of the reach, before
/// the history is dropped as a cut.
const CUT: f64 = 0.1;

/// What a quality tier draws: the grid, the sun's filter in it (0 a single
/// hardware tap, 1 sixteen) and the noise's octaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tier {
    pub level: u32,
    pub width: u32,
    pub height: u32,
    pub slices: u32,
    pub filter: u32,
    pub octaves: u32,
}

/// The tier of `quality.volumetrics`: none for 0.
pub(crate) fn tier(level: u32) -> Option<Tier> {
    match level {
        0 => None,
        1 => Some(Tier { level: 1, width: 80, height: 45, slices: 32, filter: 0, octaves: 2 }),
        _ => Some(Tier { level: 2, width: 160, height: 90, slices: 64, filter: 1, octaves: 3 }),
    }
}

/// The depth ahead of the eye of boundary `b` (0 to the slice count) of a grid
/// of `slices` that reaches `distance`.
#[cfg(test)]
pub(crate) fn boundary_depth(distance: f64, slices: u32, b: f64) -> f64 {
    let knee = distance / KNEE_DIVISOR;
    let span = (1.0 + distance / knee).ln();
    knee * ((b / f64::from(slices) * span).exp() - 1.0)
}

/// The boundary coordinate of `depth`: `boundary_depth`'s inverse.
#[cfg(test)]
pub(crate) fn depth_boundary(distance: f64, slices: u32, depth: f64) -> f64 {
    let knee = distance / KNEE_DIVISOR;
    let span = (1.0 + distance / knee).ln();
    f64::from(slices) * (1.0 + depth / knee).ln() / span
}

/// Whether `frame` draws a froxel volume at `level` (`quality.volumetrics`):
/// a perspective camera, a reach, and a medium to fill it with.
pub(crate) fn wanted(frame: &RenderFrame, level: u32) -> Option<Tier> {
    let medium = frame.fog.density > 0.0 || !frame.fog_volumes.is_empty();
    let perspective = frame.projection[11] < -0.5;
    (frame.fog.distance > 0.0 && medium && perspective).then(|| tier(level)).flatten()
}

/// What the atmosphere pass reads of the volume.
#[derive(Clone)]
pub(crate) struct Applied {
    /// Which grid this is (0 for none, a stand-in), so that the graph that
    /// binds its texture is made again when it changes.
    pub serial: u64,
    pub texture: wgpu::TextureView,
    pub uniform: wgpu::Buffer,
}

struct Pipelines {
    inject: wgpu::ComputePipeline,
    integrate: wgpu::ComputePipeline,
    inject_layout: wgpu::BindGroupLayout,
    integrate_layout: wgpu::BindGroupLayout,
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code, reason = "the textures are read back only natively"))]
struct Grid {
    tier: Tier,
    serial: u64,
    /// Two grids, taking turns being filled and being the history.
    injected: [wgpu::Texture; 2],
    integrated: wgpu::Texture,
    integrated_view: wgpu::TextureView,
    inject: [wgpu::BindGroup; 2],
    integrate: [wgpu::BindGroup; 2],
}

/// What the grid drew the frame before, to read as history.
#[derive(Clone, Copy)]
struct Remembered {
    view_projection: Matrix,
    position: [f32; 3],
    distance: f32,
    tier: u32,
}

/// The volume's state: its textures, its pipelines and what it remembers.
pub(crate) struct Volumetrics {
    device: wgpu::Device,
    queue: wgpu::Queue,
    uniform: wgpu::Buffer,
    atmosphere: wgpu::Buffer,
    local: wgpu::Buffer,
    stand_in: wgpu::TextureView,
    pipelines: Option<Pipelines>,
    grid: Option<Grid>,
    serial: u64,
    frames: u64,
    remembered: Option<Remembered>,
    /// The turn of the grid this frame fills, and whether it does.
    turn: usize,
    drawing: bool,
    /// Froxels filled by the last frame, and whether it read history.
    froxels: u32,
    historic: bool,
}

fn buffer(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn halton(mut index: u64, base: u64) -> f32 {
    let (mut result, mut fraction) = (0.0, 1.0);
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

/// Where frame `frame`'s samples sit within their froxels across, down and
/// along, from -0.5 to 0.5: the Halton (2, 3, 5) points, repeating every sixteen.
pub(crate) fn jitter(frame: u64) -> [f32; 3] {
    let index = frame % 16 + 1;
    [halton(index, 2) - 0.5, halton(index, 3) - 0.5, halton(index, 5) - 0.5]
}

fn storage_texture(device: &wgpu::Device, label: &str, size: (u32, u32, u32)) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: size.2 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

impl Volumetrics {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let stand_in = storage_texture(device, "no volume", (1, 1, 1)).create_view(&Default::default());
        Self {
            device: device.clone(),
            queue: queue.clone(),
            uniform: buffer(device, "volume", UNIFORM_BYTES, wgpu::BufferUsages::UNIFORM),
            atmosphere: buffer(device, "volume atmosphere", 224, wgpu::BufferUsages::UNIFORM),
            local: buffer(device, "fog volumes", LOCAL_BYTES, wgpu::BufferUsages::STORAGE),
            stand_in,
            pipelines: None,
            grid: None,
            serial: 0,
            frames: 0,
            remembered: None,
            turn: 0,
            drawing: false,
            froxels: 0,
            historic: false,
        }
    }

    /// The froxels the last frame filled (0 for none), and whether it blended
    /// the frame before's.
    pub fn stats(&self) -> (u32, bool) {
        (self.froxels, self.historic)
    }

    /// Drops what the volume remembers: the next frame it draws takes no history.
    pub fn reset(&mut self) {
        self.remembered = None;
    }

    fn pipelines(&mut self, context: &PipelineContext) -> &Pipelines {
        let device = &self.device;
        self.pipelines.get_or_insert_with(|| {
            let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            };
            let uniform = |size: u64| wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size),
            };
            let texture = |filterable: bool| wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            };
            let storage = wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D3,
            };
            let inject_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("volume injection"),
                entries: &[
                    entry(0, uniform(UNIFORM_BYTES)),
                    entry(1, uniform(224)),
                    entry(
                        2,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(16),
                        },
                    ),
                    entry(3, texture(true)),
                    entry(4, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                    entry(5, storage),
                ],
            });
            let integrate_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("volume integration"),
                entries: &[entry(0, uniform(UNIFORM_BYTES)), entry(1, texture(false)), entry(2, storage)],
            });
            let inject_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("volume injection"),
                bind_group_layouts: &[Some(&context.scene_group), Some(&inject_layout), Some(&context.lights_group)],
                immediate_size: 0,
            });
            let integrate_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("volume integration"),
                bind_group_layouts: &[Some(&integrate_layout)],
                immediate_size: 0,
            });
            let pipeline = |label: &str, source: String, layout: &wgpu::PipelineLayout, entry_point: &str| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: context.cache.get(),
                })
            };
            Pipelines {
                inject: pipeline("volume injection", inject_shader(), &inject_pipeline_layout, "inject"),
                integrate: pipeline("volume integration", integrate_shader(), &integrate_pipeline_layout, "integrate"),
                inject_layout,
                integrate_layout,
            }
        })
    }

    fn grid(&mut self, context: &PipelineContext, tier: Tier) -> &Grid {
        if self.grid.as_ref().is_none_or(|grid| grid.tier != tier) {
            self.serial += 1;
            self.remembered = None;
            let serial = self.serial;
            self.pipelines(context);
            let pipelines = self.pipelines.as_ref().unwrap();
            let device = &self.device;
            let size = (tier.width, tier.height, tier.slices);
            let injected = [storage_texture(device, "froxels", size), storage_texture(device, "froxels", size)];
            let integrated = storage_texture(device, "integrated froxels", (size.0, size.1, size.2 + 1));
            let views = injected.each_ref().map(|texture| texture.create_view(&Default::default()));
            let integrated_view = integrated.create_view(&Default::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("volume history"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            let inject = [0, 1].map(|turn| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("volume injection"),
                    layout: &pipelines.inject_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: self.atmosphere.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: self.local.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&views[1 - turn]),
                        },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&sampler) },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&views[turn]) },
                    ],
                })
            });
            let integrate = [0, 1].map(|turn| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("volume integration"),
                    layout: &pipelines.integrate_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&views[turn]) },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&integrated_view),
                        },
                    ],
                })
            });
            self.grid = Some(Grid { tier, serial, injected, integrated, integrated_view, inject, integrate });
        }
        self.grid.as_ref().unwrap()
    }

    /// Works out whether `frame` draws a volume at the player's `quality`,
    /// makes its grid, writes the volume's uniform, the atmosphere's the
    /// injection reads and the local volumes; and says what the atmosphere
    /// pass reads. The uniform is written for every frame the atmosphere pass
    /// draws, because the analytic mist takes its colours from it.
    pub fn prepare(
        &mut self,
        context: &PipelineContext,
        frame: &RenderFrame,
        quality: &Quality,
        size: (u32, u32),
    ) -> Applied {
        let tier = wanted(frame, quality.volumetrics);
        self.drawing = tier.is_some();
        let view_projection = multiply(&frame.projection, &frame.view);
        let distance = frame.fog.distance;
        let mut data = [0.0f32; UNIFORM_FLOATS];
        data[..16].copy_from_slice(&frame.view);
        let mut serial = 0;
        let mut texture = self.stand_in.clone();
        let mut filter = 0.0;
        if let Some(tier) = tier {
            let grid = self.grid(context, tier);
            serial = grid.serial;
            texture = grid.integrated_view.clone();
            filter = tier.filter as f32;
            let octaves = tier.octaves as f32;
            // The grid it drew with last frame, read as history while the
            // camera and the grid agree with this frame's, and nothing cut.
            let position = crate::matrix::invert(&frame.view);
            let position = [position[12], position[13], position[14]];
            let good = self.remembered.is_some_and(|before| {
                let moved =
                    (0..3).map(|axis| f64::from(position[axis] - before.position[axis]).powi(2)).sum::<f64>().sqrt();
                before.tier == tier.level
                    && before.distance == distance
                    && moved <= CUT * f64::from(distance)
                    && !frame.exposure.cut
            });
            let before = self.remembered.filter(|_| good);
            self.historic = good;
            data[16..32].copy_from_slice(&before.map_or(view_projection, |before| before.view_projection));
            let knee = f64::from(distance) / KNEE_DIVISOR;
            let span = (1.0 + f64::from(distance) / knee).ln();
            let jitter = jitter(self.frames);
            let projection = &frame.projection;
            let fog = &frame.fog;
            data[32..36].copy_from_slice(&[projection[0], projection[5], projection[8], projection[9]]);
            data[36..40].copy_from_slice(&[tier.width as f32, tier.height as f32, tier.slices as f32, distance]);
            data[40..44].copy_from_slice(&[
                crate::matrix::narrow(knee),
                crate::matrix::narrow(span),
                if good { 1.0 } else { 0.0 },
                HISTORY,
            ]);
            data[44..48].copy_from_slice(&[jitter[0], jitter[1], jitter[2], (self.frames % 4096) as f32]);
            let count = frame.fog_volumes.len().min(MAX_LOCAL);
            data[56..60].copy_from_slice(&[fog.noise, fog.noise_scale, octaves, count as f32]);
            self.remembered = Some(Remembered { view_projection, position, distance, tier: tier.level });
            self.upload_local(frame);
            self.queue.write_buffer(
                &self.atmosphere,
                0,
                bytemuck::cast_slice(&atmosphere_of(frame, &frame.projection, quality, size.0, size.1)),
            );
            self.turn = (self.frames % 2) as usize;
            self.frames += 1;
            self.froxels = tier.width * tier.height * tier.slices;
        } else {
            self.remembered = None;
            self.historic = false;
            self.froxels = 0;
        }
        let fog = &frame.fog;
        data[48..52].copy_from_slice(&[
            fog.albedo.x.max(0.0),
            fog.albedo.y.max(0.0),
            fog.albedo.z.max(0.0),
            fog.anisotropy.clamp(-0.95, 0.95),
        ]);
        data[52..56].copy_from_slice(&[
            fog.sun.max(0.0),
            fog.ambient.max(0.0),
            fog.sky.clamp(0.0, 1.0),
            fog.edge.max(0.0),
        ]);
        if tier.is_none() {
            data[56..60].copy_from_slice(&[fog.noise, fog.noise_scale, 0.0, 0.0]);
        }
        data[60..64].copy_from_slice(&[fog.drift.x, fog.drift.y, fog.drift.z, filter]);
        let sh = &frame.sky.sh0;
        data[64..68].copy_from_slice(&[
            sh.x * 0.282_094_8,
            sh.y * 0.282_094_8,
            sh.z * 0.282_094_8,
            if tier.is_some() { 1.0 } else { 0.0 },
        ]);
        self.queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&data));
        Applied { serial, texture, uniform: self.uniform.clone() }
    }

    fn upload_local(&self, frame: &RenderFrame) {
        let mut data = vec![0.0f32; MAX_LOCAL * 16];
        for (slot, volume) in frame.fog_volumes.iter().take(MAX_LOCAL).enumerate() {
            let at = slot * 16;
            let extent = [volume.extent.x.max(1e-4), volume.extent.y.max(1e-4), volume.extent.z.max(1e-4)];
            data[at..at + 16].copy_from_slice(&[
                volume.center.x,
                volume.center.y,
                volume.center.z,
                volume.shape as f32,
                extent[0],
                extent[1],
                extent[2],
                volume.density.max(0.0),
                volume.albedo.x.max(0.0),
                volume.albedo.y.max(0.0),
                volume.albedo.z.max(0.0),
                volume.falloff.clamp(0.0, 1.0),
                volume.emission.x.max(0.0),
                volume.emission.y.max(0.0),
                volume.emission.z.max(0.0),
                0.0,
            ]);
        }
        self.queue.write_buffer(&self.local, 0, bytemuck::cast_slice(&data));
    }

    /// Fills and integrates the grid, if this frame draws it: after the shadow
    /// maps and the lights it reads are drawn, before the scene.
    pub fn encode(&mut self, encoder: &mut wgpu::CommandEncoder, scene: &wgpu::BindGroup, lights: &wgpu::BindGroup) {
        if !self.drawing {
            return;
        }
        let (Some(grid), Some(pipelines)) = (&self.grid, &self.pipelines) else { return };
        let tier = grid.tier;
        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("volume"), ..Default::default() });
        pass.set_pipeline(&pipelines.inject);
        pass.set_bind_group(0, scene, &[]);
        pass.set_bind_group(1, &grid.inject[self.turn], &[]);
        pass.set_bind_group(2, lights, &[]);
        pass.dispatch_workgroups(tier.width.div_ceil(4), tier.height.div_ceil(4), tier.slices.div_ceil(4));
        // Dispatches of one pass see each other's writes only across passes.
        drop(pass);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("volume integration"),
            ..Default::default()
        });
        pass.set_pipeline(&pipelines.integrate);
        pass.set_bind_group(0, &grid.integrate[self.turn], &[]);
        pass.dispatch_workgroups(tier.width.div_ceil(8), tier.height.div_ceil(8), 1);
    }
}

/// Which of the volume's textures to read back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolumeLayer {
    /// The last frame's froxels: the light the medium sends toward the eye per
    /// unit of length (rgb) and its extinction (a), after the temporal blend.
    Injected,
    /// The integral down each column: the light scattered toward the eye (rgb)
    /// and the optical depth (a) up to each slice's far boundary, layer 0 first
    /// (nothing yet).
    Integrated,
}

/// A grid texture read back: froxels across, down and layers deep, each its
/// four channels, x varying fastest.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeImage {
    pub size: [u32; 3],
    pub texels: Vec<[f32; 4]>,
}

impl VolumeImage {
    /// The texel at froxel (x, y) of `layer`.
    pub fn at(&self, x: u32, y: u32, layer: u32) -> [f32; 4] {
        self.texels[((layer * self.size[1] + y) * self.size[0] + x) as usize]
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 if mantissa == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Volumetrics {
    /// Reads a texture of the last frame's grid back, waiting for the GPU:
    /// for tools and tests. None before a frame has drawn a volume.
    pub fn read(&self, layer: VolumeLayer) -> Option<VolumeImage> {
        let grid = self.grid.as_ref()?;
        let texture = match layer {
            VolumeLayer::Injected => &grid.injected[self.turn],
            VolumeLayer::Integrated => &grid.integrated,
        };
        let size = texture.size();
        let row = (size.width * 8).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let bytes = u64::from(row) * u64::from(size.height) * u64::from(size.depth_or_array_layers);
        let read = buffer(&self.device, "volume read back", bytes, wgpu::BufferUsages::MAP_READ);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &read,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(size.height),
                },
            },
            size,
        );
        self.queue.submit([encoder.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mapped = read.slice(..).get_mapped_range().unwrap();
        let mut texels = Vec::with_capacity((size.width * size.height * size.depth_or_array_layers) as usize);
        for layer in 0..size.depth_or_array_layers {
            for y in 0..size.height {
                let start = ((layer * size.height + y) * row) as usize;
                for x in 0..size.width as usize {
                    let at = start + x * 8;
                    let lane = |i: usize| half_to_f32(u16::from_le_bytes([mapped[at + i * 2], mapped[at + i * 2 + 1]]));
                    texels.push([lane(0), lane(1), lane(2), lane(3)]);
                }
            }
        }
        Some(VolumeImage { size: [size.width, size.height, size.depth_or_array_layers], texels })
    }
}

/// The injection's WGSL: the scene's uniform, the sun's shadows (filtered at
/// most as the volume's tier says), the sky's light, the clustered lights, the
/// sky and the clouds, then the volume's own.
pub(crate) fn inject_shader() -> String {
    let shadows = crate::renderer::SHADOWS_WGSL
        .replace("let mode = u32(scene.cascades.z);", "let mode = volume_filter(u32(scene.cascades.z));");
    assert!(shadows.contains("volume_filter"), "the sun's filter reads the frame's mode where the volume takes it");
    format!(
        "{SCENE}{shadows}\n{}\n{}\n{ATMOSPHERE_COMMON}\n{COMMON}\n{INJECT}",
        crate::environment::ENVIRONMENT_WGSL,
        crate::lights::LIGHTS_WGSL,
    )
}

/// The integration's WGSL.
pub(crate) fn integrate_shader() -> String {
    format!("{COMMON}\n{INTEGRATE}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_grow_exponentially_to_the_reach() {
        for slices in [32, 64] {
            let depths: Vec<f64> = (0..=slices).map(|b| boundary_depth(200.0, slices, f64::from(b))).collect();
            assert_eq!(depths[0], 0.0);
            assert!((depths[slices as usize] - 200.0).abs() < 1e-9);
            assert!(depths.windows(2).all(|pair| pair[1] > pair[0]));
            // Each slice a fixed share thicker than the last, past the knee.
            let widths: Vec<f64> = depths.windows(2).map(|pair| pair[1] - pair[0]).collect();
            for pair in widths.windows(2) {
                let ratio = pair[1] / pair[0];
                let expected = (1.0 + 64.0f64).powf(1.0 / f64::from(slices));
                assert!((ratio - expected).abs() < 1e-9, "{ratio} {expected}");
            }
            for b in [0.0, 0.5, 3.25, f64::from(slices) - 0.5] {
                let depth = boundary_depth(200.0, slices, b);
                assert!((depth_boundary(200.0, slices, depth) - b).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn tiers_shrink_with_the_setting() {
        assert!(tier(0).is_none());
        let (low, high) = (tier(1).unwrap(), tier(2).unwrap());
        assert!(low.width * low.height * low.slices < high.width * high.height * high.slices);
        assert_eq!((high.width, high.height, high.slices), (160, 90, 64));
        assert_eq!(tier(7), tier(2));
    }

    #[test]
    fn jitter_stays_within_the_froxel_and_repeats() {
        let points: Vec<[f32; 3]> = (0..16).map(jitter).collect();
        assert!(points.iter().all(|p| p.iter().all(|v| v.abs() < 0.5)));
        assert_eq!(jitter(16), points[0]);
        let mean = points.iter().fold([0.0f32; 3], |sum, p| [sum[0] + p[0], sum[1] + p[1], sum[2] + p[2]]);
        assert!(mean.iter().all(|v| (v / 16.0).abs() < 0.07), "{mean:?}");
    }

    #[test]
    fn the_shaders_assemble() {
        assert!(!inject_shader().contains("${"));
        assert!(!integrate_shader().contains("${"));
    }
}
