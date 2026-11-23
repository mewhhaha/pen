// The sky as light for the scene's materials. Diffuse light comes from nine
// spherical-harmonic coefficients the game works out from its sky (they ride
// in the render packet; the host only passes them on). Glossy reflections
// come from a cube map of the same sky, drawn here from the frame's three sky
// colors, whose levels are the sky blurred by GGX lobes of growing roughness,
// and the split-sum BRDF table, drawn once. The cube is redrawn on the GPU in
// the frame whose sky colors differ from the last one's, and not otherwise.
// `ENVIRONMENT_WGSL` is what a material shader adds to read all of it, at
// group 0 bindings 4 to 8 and 10 to 13 of every mesh pipeline: besides the
// sky's, the local reflection probes' cubes (reflection_probes.rs) and the
// screen-space reflections' depth and picture (renderer/screen_reflections.rs).
use pen_protocol::{Probe, ProbeVolume, RenderFrame, Sky, Vector};

use crate::pipeline_cache::PipelineCache;

mod reflection_probes;
pub(crate) use reflection_probes::{CaptureTargets, ReflectionProbes, Step, box_view, face_projection, face_view};

/// WGSL a material shader adds to light with the sky: `ambient_light(world,
/// normal)`, `sky_reflection(world, normal, view, roughness, f0)`,
/// `surface_ambient` and `specular_occlusion`.
pub const ENVIRONMENT_WGSL: &str = include_str!("shaders/environment.wgsl");

const PREFILTER: &str = include_str!("shaders/environment_prefilter.wgsl");
const TABLE_SHADER: &str = include_str!("shaders/environment_table.wgsl");

/// A face's width in texels, and the levels down to two texels; level `n`
/// holds roughness `n / (LEVELS - 1)`.
pub(crate) const SIZE: u32 = 64;
pub(crate) const LEVELS: u32 = 6;
/// The table's width and height in texels (the shader has the same).
const TABLE: u32 = 32;
/// GGX samples per texel of a rough level.
const SAMPLES: f32 = 128.0;
/// A pass's uniform: three colors and its settings, in a slot of this stride.
const PASS_BYTES: u64 = 64;
const STRIDE: u64 = 256;
const PASSES: u64 = 6 * LEVELS as u64;
/// The uniform `ENVIRONMENT_WGSL` reads: nine coefficients, the tint, the
/// reflection settings, the probe volume's header, and the screen-space
/// reflections' (floats 52 on).
pub(crate) const LIGHT_BYTES: u64 = 16 * 28;
/// The floats of `LIGHT_BYTES`, and where the screen-space reflections' begin.
pub(crate) const LIGHT_FLOATS: usize = 112;
pub(crate) const SCREEN_FLOATS: usize = 60;
const SCREEN_AT: usize = LIGHT_FLOATS - SCREEN_FLOATS;
/// Floats of one probe on the GPU: nine RGB terms and a validity weight.
pub(crate) const PROBE_FLOATS: usize = 28;
/// The most probes the volume buffer holds (`MAX_PROBES`, as one volume).
const MAX_PROBES: usize = 16384;
const PROBE_BYTES: u64 = (MAX_PROBES * PROBE_FLOATS * 4) as u64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub(crate) struct Environment {
    cache: PipelineCache,
    cube: wgpu::Texture,
    cube_view: wgpu::TextureView,
    _table: wgpu::Texture,
    table_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// The frame's light, the same without screen-space reflections (what
    /// the probes' captures are lit by), and the zeros the shadow passes bind.
    light: wgpu::Buffer,
    capture_light: wgpu::Buffer,
    empty: wgpu::Buffer,
    /// The local reflection probes' cubes and table.
    pub(crate) reflections: ReflectionProbes,
    /// The passes' uniforms, in slots with dynamic offsets.
    passes: wgpu::Buffer,
    /// The resident probe volume: every probe's 28 floats, and which bake it
    /// is (revision and counts).
    probe_data: wgpu::Buffer,
    resident: Option<(u32, [u32; 3])>,
    /// The sky colors the cube holds, or is to hold once encoded.
    drawn: Option<[u32; 9]>,
    stale: bool,
    table_drawn: bool,
    pipelines: Option<Pipelines>,
}

struct Pipelines {
    prefilter: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
    table: wgpu::RenderPipeline,
}

/// The scene group's bindings 4 to 8, 10 and 11: the sky's cube, the table, a
/// filtering sampler, the light, the diffuse probes, the reflection probes'
/// cubes and their table. (Bindings 12 and 13, the screen-space
/// reflections', come from the renderer.)
pub(crate) fn layout_entries() -> [wgpu::BindGroupLayoutEntry; 7] {
    let texture = |binding: u32, view_dimension: wgpu::TextureViewDimension| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension,
            multisampled: false,
        },
        count: None,
    };
    [
        texture(4, wgpu::TextureViewDimension::Cube),
        texture(5, wgpu::TextureViewDimension::D2),
        wgpu::BindGroupLayoutEntry {
            binding: 6,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 7,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(LIGHT_BYTES),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(16),
            },
            count: None,
        },
        texture(10, wgpu::TextureViewDimension::CubeArray),
        wgpu::BindGroupLayoutEntry {
            binding: 11,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(reflection_probes::TABLE_BYTES),
            },
            count: None,
        },
    ]
}

fn buffer(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The light uniform of a sky with these coefficients: nine colors, the tint
/// (w: 1 when the cube is the sky to reflect), the cube's last level, then the
/// probe volume's origin and spacing and its counts (w: 1 when it is in use).
pub(crate) fn light_floats(sh: &[Vector; 9], tint: &Vector, reflecting: bool) -> [f32; LIGHT_FLOATS] {
    let mut floats = [0.0; LIGHT_FLOATS];
    for (slot, v) in sh.iter().enumerate() {
        floats[slot * 4..slot * 4 + 3].copy_from_slice(&[v.x, v.y, v.z]);
    }
    floats[36..40].copy_from_slice(&[tint.x, tint.y, tint.z, if reflecting { 1.0 } else { 0.0 }]);
    floats[40] = (LEVELS - 1) as f32;
    floats[41] = (reflection_probes::LEVELS - 1) as f32;
    floats
}

/// A flat light, `ambient` from every direction, which reflects nothing.
pub(crate) fn flat_floats(ambient: &Vector) -> [f32; LIGHT_FLOATS] {
    // Coefficient 0 of a constant is the constant over the first basis function.
    let scale = 1.0 / 0.282_094_8;
    let zero = Vector { x: 0.0, y: 0.0, z: 0.0 };
    let mut sh = [zero; 9];
    sh[0] = Vector { x: ambient.x * scale, y: ambient.y * scale, z: ambient.z * scale };
    light_floats(&sh, &zero, false)
}

/// The sky's coefficients and tint from the frame, or its flat `ambient`
/// when the game sent none, so a game that knows nothing of them still has
/// light. `probes` says whether a probe volume is resident for the frame.
pub(crate) fn sky_floats(sky: &Sky, ambient: &Vector, volume: &ProbeVolume, probes: bool) -> [f32; LIGHT_FLOATS] {
    let sh = [sky.sh0, sky.sh1, sky.sh2, sky.sh3, sky.sh4, sky.sh5, sky.sh6, sky.sh7, sky.sh8];
    let sent = sh.iter().chain([&sky.tint]).any(|v| v.x != 0.0 || v.y != 0.0 || v.z != 0.0);
    let mut floats = if sent { light_floats(&sh, &sky.tint, true) } else { flat_floats(ambient) };
    if probes {
        floats[44..48].copy_from_slice(&[volume.origin.x, volume.origin.y, volume.origin.z, volume.spacing]);
        floats[48..52].copy_from_slice(&[volume.nx as f32, volume.ny as f32, volume.nz as f32, 1.0]);
    }
    floats
}

fn clear(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
    }
}

fn key(sky: &Sky) -> [u32; 9] {
    let colors = [sky.zenith, sky.horizon, sky.ground];
    std::array::from_fn(|index| {
        let color = colors[index / 3];
        [color.x, color.y, color.z][index % 3].to_bits()
    })
}

impl Environment {
    pub fn new(device: &wgpu::Device, cache: PipelineCache) -> Self {
        let cube = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sky cube"),
            size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 6 },
            mip_level_count: LEVELS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let cube_view = cube.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let table = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sky table"),
            size: wgpu::Extent3d { width: TABLE, height: TABLE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let table_view = table.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self {
            cache,
            cube,
            cube_view,
            _table: table,
            table_view,
            sampler,
            light: buffer(device, "sky light", LIGHT_BYTES, wgpu::BufferUsages::UNIFORM),
            capture_light: buffer(device, "capture sky light", LIGHT_BYTES, wgpu::BufferUsages::UNIFORM),
            reflections: ReflectionProbes::new(device),
            empty: buffer(device, "no sky light", LIGHT_BYTES, wgpu::BufferUsages::UNIFORM),
            passes: buffer(device, "sky passes", STRIDE * PASSES, wgpu::BufferUsages::UNIFORM),
            probe_data: buffer(device, "probes", PROBE_BYTES, wgpu::BufferUsages::STORAGE),
            resident: None,
            drawn: None,
            stale: false,
            table_drawn: false,
            pipelines: None,
        }
    }

    /// The bindings 4 to 8, 10 and 11 of a scene group, with `light` as the
    /// sky's light.
    pub fn entries<'a>(&'a self, light: &'a wgpu::Buffer) -> [wgpu::BindGroupEntry<'a>; 7] {
        self.entries_with(light, &self.reflections.table)
    }

    /// `entries`, with `table` as the reflection probes' (`empty_probes` for none).
    pub fn entries_with<'a>(
        &'a self,
        light: &'a wgpu::Buffer,
        table: &'a wgpu::Buffer,
    ) -> [wgpu::BindGroupEntry<'a>; 7] {
        [
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.cube_view) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.table_view) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            wgpu::BindGroupEntry { binding: 7, resource: light.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: self.probe_data.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::TextureView(&self.reflections.cubes_view),
            },
            wgpu::BindGroupEntry { binding: 11, resource: table.as_entire_binding() },
        ]
    }

    /// The table of the reflection probes that hold cubes.
    pub fn probe_table(&self) -> &wgpu::Buffer {
        &self.reflections.table
    }

    /// A table of no probes: what insets and the passes that draw into maps bind.
    pub fn empty_probes(&self) -> &wgpu::Buffer {
        &self.reflections.empty_table
    }

    /// The light the probes' captures are lit by.
    pub fn capture_light(&self) -> &wgpu::Buffer {
        &self.capture_light
    }

    /// Writes the screen-space reflections' part of the frame's light.
    pub fn set_screen(&self, queue: &wgpu::Queue, floats: &[f32; SCREEN_FLOATS]) {
        queue.write_buffer(&self.light, (SCREEN_AT * 4) as u64, bytemuck::cast_slice(floats));
    }

    /// The frame's light buffer, and the one that holds none.
    pub fn light(&self) -> &wgpu::Buffer {
        &self.light
    }

    pub fn empty(&self) -> &wgpu::Buffer {
        &self.empty
    }

    /// Takes the frame's sky: writes its light, and notes whether the cube
    /// must be drawn again.
    pub fn update(&mut self, queue: &wgpu::Queue, frame: &RenderFrame) {
        let volume = &frame.probe_volume;
        let using = self.set_probes(queue, volume, &frame.probes);
        self.set_sky(queue, &frame.sky, &frame.ambient, volume, using);
    }

    /// Takes the frame's probes: those sent replace the resident ones; none
    /// keeps them while the revision and counts are the same. Whether the
    /// volume is in use this frame.
    pub fn set_probes(&mut self, queue: &wgpu::Queue, volume: &ProbeVolume, probes: &[Probe]) -> bool {
        let counts = [volume.nx, volume.ny, volume.nz];
        let total = counts.iter().map(|&n| n as usize).product::<usize>();
        let fits = total > 0 && total <= MAX_PROBES;
        if fits && probes.len() == total {
            let mut floats = Vec::with_capacity(total * PROBE_FLOATS);
            for probe in probes {
                for v in
                    [probe.sh0, probe.sh1, probe.sh2, probe.sh3, probe.sh4, probe.sh5, probe.sh6, probe.sh7, probe.sh8]
                {
                    floats.extend([v.x, v.y, v.z]);
                }
                floats.push(probe.valid);
            }
            queue.write_buffer(&self.probe_data, 0, bytemuck::cast_slice(&floats));
            self.resident = Some((volume.revision, counts));
        }
        fits && self.resident == Some((volume.revision, counts))
    }

    /// Writes the sky's light, and notes whether the cube must be drawn again.
    pub fn set_sky(&mut self, queue: &wgpu::Queue, sky: &Sky, ambient: &Vector, volume: &ProbeVolume, probes: bool) {
        let floats = sky_floats(sky, ambient, volume, probes);
        queue.write_buffer(&self.light, 0, bytemuck::cast_slice(&floats));
        queue.write_buffer(&self.capture_light, 0, bytemuck::cast_slice(&floats));
        let wanted = key(sky);
        if self.drawn != Some(wanted) {
            self.drawn = Some(wanted);
            self.stale = true;
            // Written now, ahead of the submit that reads it.
            let mut data = vec![0.0f32; (STRIDE * PASSES / 4) as usize];
            for level in 0..LEVELS {
                for face in 0..6u32 {
                    let at = ((u64::from(level) * 6 + u64::from(face)) * STRIDE / 4) as usize;
                    let roughness = level as f32 / (LEVELS - 1) as f32;
                    let samples = if level == 0 { 1.0 } else { SAMPLES };
                    let size = (SIZE >> level).max(1) as f32;
                    for (index, color) in [sky.zenith, sky.horizon, sky.ground].iter().enumerate() {
                        data[at + index * 4..at + index * 4 + 3].copy_from_slice(&[color.x, color.y, color.z]);
                    }
                    data[at + 12..at + 16].copy_from_slice(&[face as f32, roughness, samples, size]);
                }
            }
            queue.write_buffer(&self.passes, 0, bytemuck::cast_slice(&data));
        }
    }

    /// Whether `encode` has passes to record.
    pub fn pending(&self) -> bool {
        self.stale || !self.table_drawn
    }

    fn pipelines(&mut self, device: &wgpu::Device) -> &Pipelines {
        let cache = &self.cache;
        let passes = &self.passes;
        self.pipelines.get_or_insert_with(|| {
            let make = |label: &str, source: &str| {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                })
            };
            let pipeline = |label: &str, module: &wgpu::ShaderModule, layout: Option<&wgpu::PipelineLayout>| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout,
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
                        entry_point: Some("fragment"),
                        compilation_options: Default::default(),
                        targets: &[Some(FORMAT.into())],
                    }),
                    multiview_mask: None,
                    cache: cache.get(),
                })
            };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sky pass"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(PASS_BYTES),
                    },
                    count: None,
                }],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sky pass"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sky pass"),
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: passes,
                        offset: 0,
                        size: wgpu::BufferSize::new(PASS_BYTES),
                    }),
                }],
            });
            Pipelines {
                prefilter: pipeline("sky prefilter", &make("sky prefilter", PREFILTER), Some(&pipeline_layout)),
                group,
                table: pipeline("sky table", &make("sky table", TABLE_SHADER), None),
            }
        })
    }

    /// Records the passes that draw the table (once) and the cube (when the
    /// sky's colors changed) into `encoder`.
    pub fn encode(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if !self.pending() {
            return;
        }
        self.pipelines(device);
        let pipelines = self.pipelines.as_ref().unwrap();
        if !self.table_drawn {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sky table"),
                color_attachments: &[Some(clear(&self.table_view))],
                ..Default::default()
            });
            pass.set_pipeline(&pipelines.table);
            pass.draw(0..3, 0..1);
            self.table_drawn = true;
        }
        if self.stale {
            for level in 0..LEVELS {
                for face in 0..6u32 {
                    let view = self.cube.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_mip_level: level,
                        mip_level_count: Some(1),
                        base_array_layer: face,
                        array_layer_count: Some(1),
                        ..Default::default()
                    });
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("sky cube"),
                        color_attachments: &[Some(clear(&view))],
                        ..Default::default()
                    });
                    pass.set_pipeline(&pipelines.prefilter);
                    let offset = (u64::from(level) * 6 + u64::from(face)) * STRIDE;
                    pass.set_bind_group(0, &pipelines.group, &[u32::try_from(offset).expect("a few kilobytes")]);
                    pass.draw(0..3, 0..1);
                }
            }
            self.stale = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_light_is_its_ambient_color_in_every_direction() {
        let floats = flat_floats(&Vector { x: 0.2, y: 0.3, z: 0.4 });
        // The constant term evaluates at 0.2820948 per unit of coefficient.
        assert!((floats[0] * 0.282_094_8 - 0.2).abs() < 1e-6);
        assert!((floats[2] * 0.282_094_8 - 0.4).abs() < 1e-6);
        assert!(floats[4..36].iter().all(|v| *v == 0.0));
        assert_eq!(floats[39], 0.0);
    }

    #[test]
    fn a_game_that_sends_no_coefficients_gets_the_flat_ambient() {
        let sky = Sky::default();
        let ambient = Vector { x: 0.5, y: 0.5, z: 0.5 };
        assert_eq!(sky_floats(&sky, &ambient, &ProbeVolume::default(), false), flat_floats(&ambient));
    }
}

#[cfg(test)]
mod gpu {
    //! The sky's light on a real (headless) GPU: the cube and table the
    //! frame's sky makes, read back through `ENVIRONMENT_WGSL` as materials do.
    use super::*;

    const SIDE: u32 = 16;

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default())).ok()
    }

    fn vector(x: f32, y: f32, z: f32) -> Vector {
        Vector { x, y, z }
    }

    /// A sky of three colors with these coefficients.
    fn sky(colors: [Vector; 3], sh: [Vector; 9], tint: Vector) -> Sky {
        Sky {
            zenith: colors[0],
            horizon: colors[1],
            ground: colors[2],
            tint,
            sh0: sh[0],
            sh1: sh[1],
            sh2: sh[2],
            sh3: sh[3],
            sh4: sh[4],
            sh5: sh[5],
            sh6: sh[6],
            sh7: sh[7],
            sh8: sh[8],
            ..Sky::default()
        }
    }

    /// Draws a `SIDE` x `SIDE` picture whose pixel (x, y) is `body`'s value for
    /// `pixel` (x, y) and a unit `direction` spread over the sphere: the
    /// scene group's bindings 4 to 7 with `sky`'s light, and
    /// `ENVIRONMENT_WGSL`. The pixels' RGBA, row by row.
    fn probe(device: &wgpu::Device, queue: &wgpu::Queue, sky: &Sky, ambient: &Vector, body: &str) -> Vec<[f32; 4]> {
        probe_with(device, queue, sky, ambient, (&ProbeVolume::default(), &[]), body)
    }

    /// `probe` with a probe volume and its probes uploaded.
    fn probe_with(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sky: &Sky,
        ambient: &Vector,
        probes: (&ProbeVolume, &[Probe]),
        body: &str,
    ) -> Vec<[f32; 4]> {
        probe_full(device, queue, sky, ambient, probes, None, body)
    }

    /// `probe_with`, the reflection probes' table (`table_floats`) written too.
    pub(super) fn probe_full(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sky: &Sky,
        ambient: &Vector,
        probes: (&ProbeVolume, &[Probe]),
        table: Option<&[f32]>,
        body: &str,
    ) -> Vec<[f32; 4]> {
        let mut environment = Environment::new(device, PipelineCache::open(device, None));
        let using = environment.set_probes(queue, probes.0, probes.1);
        environment.set_sky(queue, sky, ambient, probes.0, using);
        if let Some(table) = table {
            queue.write_buffer(&environment.reflections.table, 0, bytemuck::cast_slice(table));
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        environment.encode(device, &mut encoder);
        let source = format!(
            "{ENVIRONMENT_WGSL}
@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {{
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
}}
// The directions of a latitude and longitude grid, poles left out.
fn direction_at(pixel: vec2f) -> vec3f {{
  let longitude = pixel.x / {SIDE}.0 * 6.2831853;
  let y = 1.0 - 2.0 * pixel.y / {SIDE}.0;
  let ring = sqrt(1.0 - y * y);
  return vec3f(ring * cos(longitude), y, ring * sin(longitude));
}}
@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {{
  let direction = direction_at(position.xy);
  {body}
}}"
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let mut entries = layout_entries().to_vec();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 12,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 13,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        let layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
        let stand = |format: wgpu::TextureFormat, dimension: wgpu::TextureViewDimension| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 2 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor { dimension: Some(dimension), ..Default::default() })
        };
        let depth_stand_in = stand(wgpu::TextureFormat::Depth32Float, wgpu::TextureViewDimension::D2Array);
        let picture_stand_in = {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            texture.create_view(&Default::default())
        };
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("probe"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::TextureFormat::Rgba32Float.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &{
                let mut group = environment.entries(environment.light()).to_vec();
                group.push(wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(&depth_stand_in),
                });
                group.push(wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(&picture_stand_in),
                });
                group
            },
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: SIDE, height: SIDE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(clear(&view))],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(SIDE * SIDE * 16),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(SIDE * 16), rows_per_image: None },
            },
            wgpu::Extent3d { width: SIDE, height: SIDE, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mapped = buffer.slice(..).get_mapped_range().unwrap();
        let floats: &[f32] = bytemuck::cast_slice(&mapped);
        floats.chunks(4).map(|texel| [texel[0], texel[1], texel[2], texel[3]]).collect()
    }

    fn direction_at(x: u32, y: u32) -> [f64; 3] {
        let longitude = (f64::from(x) + 0.5) / f64::from(SIDE) * std::f64::consts::TAU;
        let height = 1.0 - 2.0 * (f64::from(y) + 0.5) / f64::from(SIDE);
        let ring = (1.0 - height * height).sqrt();
        [ring * longitude.cos(), height, ring * longitude.sin()]
    }

    fn gradient(colors: &[Vector; 3], d: &[f64; 3]) -> [f64; 3] {
        let [zenith, horizon, ground] = colors.map(|c| [f64::from(c.x), f64::from(c.y), f64::from(c.z)]);
        let (to, t) = if d[1] >= 0.0 { (zenith, d[1].sqrt()) } else { (ground, (-d[1]).sqrt()) };
        std::array::from_fn(|c| horizon[c] + (to[c] - horizon[c]) * t)
    }

    fn colors() -> [Vector; 3] {
        [vector(1.0, 0.2, 0.1), vector(0.3, 0.5, 0.2), vector(0.05, 0.1, 0.6)]
    }

    #[test]
    fn the_cube_is_the_sky_in_every_direction_it_is_sampled() {
        let Some((device, queue)) = device() else { return };
        let sky = sky(colors(), [vector(0.0, 0.0, 0.0); 9], vector(1.0, 1.0, 1.0));
        let picture = probe(
            &device,
            &queue,
            &sky,
            &Vector::default(),
            "return vec4f(textureSampleLevel(sky_cube, sky_sampler, direction, 0.0).rgb, 1.0);",
        );
        for y in 0..SIDE {
            for x in 0..SIDE {
                let want = gradient(&colors(), &direction_at(x, y));
                let got = picture[(y * SIDE + x) as usize];
                for c in 0..3 {
                    assert!(
                        (f64::from(got[c]) - want[c]).abs() < 0.03,
                        "pixel ({x}, {y}) channel {c}: {} vs {}",
                        got[c],
                        want[c]
                    );
                }
            }
        }
    }

    /// The prefiltered sky toward `n` at `roughness`, by summing GGX lobe
    /// samples of the analytic gradient in f64.
    fn prefiltered(colors: &[Vector; 3], n: [f64; 3], roughness: f64) -> [f64; 3] {
        let a2 = roughness.powi(4);
        let up = if n[1].abs() > 0.999 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        };
        let unit = |v: [f64; 3]| {
            let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / length, v[1] / length, v[2] / length]
        };
        let tangent = unit(cross(up, n));
        let bitangent = cross(n, tangent);
        let (mut total, mut weight) = ([0.0; 3], 0.0);
        let count: u32 = 20_000;
        for i in 0..count {
            let u = [(f64::from(i) + 0.5) / f64::from(count), f64::from(i.reverse_bits()) / 4_294_967_296.0];
            let phi = std::f64::consts::TAU * u[0];
            let cos_theta = ((1.0 - u[1]) / (1.0 + (a2 - 1.0) * u[1])).sqrt();
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let h: [f64; 3] = std::array::from_fn(|c| {
                tangent[c] * sin_theta * phi.cos() + bitangent[c] * sin_theta * phi.sin() + n[c] * cos_theta
            });
            let dot_nh = n[0] * h[0] + n[1] * h[1] + n[2] * h[2];
            let l: [f64; 3] = std::array::from_fn(|c| 2.0 * dot_nh * h[c] - n[c]);
            let facing = n[0] * l[0] + n[1] * l[1] + n[2] * l[2];
            if facing > 0.0 {
                let radiance = gradient(colors, &l);
                for c in 0..3 {
                    total[c] += radiance[c] * facing;
                }
                weight += facing;
            }
        }
        total.map(|v| v / weight)
    }

    #[test]
    fn rougher_levels_hold_the_ggx_blur_of_the_sky() {
        let Some((device, queue)) = device() else { return };
        let sky = sky(colors(), [vector(0.0, 0.0, 0.0); 9], vector(1.0, 1.0, 1.0));
        for level in [2.0f64, 3.0, 4.0, 5.0] {
            let roughness = level / f64::from(LEVELS - 1);
            let picture = probe(
                &device,
                &queue,
                &sky,
                &Vector::default(),
                &format!("return vec4f(textureSampleLevel(sky_cube, sky_sampler, direction, {level:?}).rgb, 1.0);"),
            );
            for (x, y) in [(0, 0), (3, 4), (8, 8), (12, 11), (5, 15)] {
                let want = prefiltered(&colors(), direction_at(x, y), roughness);
                let got = picture[(y * SIDE + x) as usize];
                for c in 0..3 {
                    // The cube's own 64-texel faces and their filtering allow a few
                    // percent; the last level, two texels a face, blends six faces'
                    // lobes and allows a few more.
                    let tolerance = if level > 4.0 { 0.08 } else { 0.04 };
                    assert!(
                        (f64::from(got[c]) - want[c]).abs() < tolerance,
                        "level {level} pixel ({x}, {y}) channel {c}: {} vs {}",
                        got[c],
                        want[c]
                    );
                }
            }
        }
    }

    /// The split-sum integral for Schlick Fresnel, summed in f64.
    fn table_reference(nv: f64, roughness: f64) -> [f64; 2] {
        let a2 = roughness.powi(4);
        let view = [(1.0 - nv * nv).sqrt(), 0.0, nv];
        let (mut scale, mut bias) = (0.0, 0.0);
        let count: u32 = 50_000;
        for i in 0..count {
            let u = [(f64::from(i) + 0.5) / f64::from(count), f64::from(i.reverse_bits()) / 4_294_967_296.0];
            let phi = std::f64::consts::TAU * u[0];
            let cos_theta = ((1.0 - u[1]) / (1.0 + (a2 - 1.0) * u[1])).sqrt();
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let h = [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta];
            let v_dot_h = view[0] * h[0] + view[1] * h[1] + view[2] * h[2];
            let l: [f64; 3] = std::array::from_fn(|c| 2.0 * v_dot_h * h[c] - view[c]);
            let nl = l[2].clamp(0.0, 1.0);
            if nl > 0.0 {
                let visibility =
                    0.5 / (nl * (nv * nv * (1.0 - a2) + a2).sqrt() + nv * (nl * nl * (1.0 - a2) + a2).sqrt()).max(1e-6);
                let weight = visibility * nl * 4.0 * v_dot_h.clamp(0.0, 1.0) / h[2].max(1e-4);
                let fresnel = (1.0 - v_dot_h.clamp(0.0, 1.0)).powi(5);
                scale += (1.0 - fresnel) * weight;
                bias += fresnel * weight;
            }
        }
        [scale / f64::from(count), bias / f64::from(count)]
    }

    #[test]
    fn the_table_is_the_split_sum_of_the_lights_ggx_lobe() {
        let Some((device, queue)) = device() else { return };
        let sky = sky(colors(), [vector(0.0, 0.0, 0.0); 9], vector(1.0, 1.0, 1.0));
        // The picture's pixel (x, y) looks up (x / 16, y / 16): the cosine across, roughness down.
        let picture = probe(
            &device,
            &queue,
            &sky,
            &Vector::default(),
            "let uv = (position.xy - vec2f(0.5)) / 15.0;
  let split = sky_brdf(max(uv.x, 0.05), max(uv.y, 0.05));
  return vec4f(split.x, split.y, 0.0, 1.0);",
        );
        for (x, y) in [(15, 15), (15, 8), (8, 8), (4, 4), (12, 1), (2, 14), (8, 15)] {
            let uv = [f64::from(x) / 15.0, f64::from(y) / 15.0];
            let want = table_reference(uv[0].max(0.05), uv[1].max(0.05).clamp(0.04, 1.0));
            let got = picture[(y * SIDE + x) as usize];
            assert!(
                (f64::from(got[0]) - want[0]).abs() < 0.03 && (f64::from(got[1]) - want[1]).abs() < 0.03,
                "table ({x}, {y}): {got:?} vs {want:?}"
            );
        }
        // White light off any surface is at most all of it.
        for texel in &picture {
            assert!(texel[0] + texel[1] <= 1.02 && texel[0] >= 0.0 && texel[1] >= 0.0);
        }
    }

    #[test]
    fn the_coefficients_evaluate_in_the_basis_the_game_projects_on() {
        let Some((device, queue)) = device() else { return };
        // A light with every coefficient in its own channel mix.
        let sh: [Vector; 9] =
            std::array::from_fn(|i| vector(0.1 * (i as f32 + 1.0), 0.05 * i as f32, 0.2 - 0.02 * i as f32));
        let sky = sky(colors(), sh, vector(1.0, 1.0, 1.0));
        let picture = probe(
            &device,
            &queue,
            &sky,
            &Vector::default(),
            "return vec4f(ambient_light(vec3f(0.0), direction), 1.0);",
        );
        for y in 0..SIDE {
            for x in 0..SIDE {
                let d = direction_at(x, y);
                let (px, py, pz) = (d[0], d[1], d[2]);
                let basis = [
                    0.282_094_791_8,
                    0.488_602_511_9 * py,
                    0.488_602_511_9 * pz,
                    0.488_602_511_9 * px,
                    1.092_548_430_6 * px * py,
                    1.092_548_430_6 * py * pz,
                    0.315_391_565_3 * (3.0 * pz * pz - 1.0),
                    1.092_548_430_6 * px * pz,
                    0.546_274_215_3 * (px * px - py * py),
                ];
                let got = picture[(y * SIDE + x) as usize];
                for (c, got) in got.iter().take(3).enumerate() {
                    let want: f64 =
                        (0..9).map(|i| f64::from([sh[i].x, sh[i].y, sh[i].z][c]) * basis[i]).sum::<f64>().max(0.0);
                    assert!((f64::from(*got) - want).abs() < 2e-4, "pixel ({x}, {y}) channel {c}: {got} vs {want}");
                }
            }
        }
    }

    #[test]
    fn a_sky_of_one_color_reflects_it_scaled_by_the_fresnel_table() {
        let Some((device, queue)) = device() else { return };
        let same = vector(0.5, 0.5, 0.5);
        let sky = sky([same; 3], [vector(0.1, 0.1, 0.1); 9], vector(2.0, 2.0, 2.0));
        // A mirror facing the viewer along +z: radiance 1 times (f0 * scale + bias).
        let picture = probe(
            &device,
            &queue,
            &sky,
            &Vector::default(),
            "let f0 = vec3f(0.04);
  let got = sky_reflection(vec3f(0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 1.0), 0.5, f0);
  let split = sky_brdf(1.0, 0.5);
  return vec4f(got, split.x * 0.04 + split.y);",
        );
        let texel = picture[0];
        assert!((texel[0] - texel[3]).abs() < 1e-3, "{texel:?}");
        assert!(texel[3] > 0.02, "a face-on surface reflects at least f0: {texel:?}");
    }

    #[test]
    fn no_sky_to_reflect_means_no_reflection() {
        let Some((device, queue)) = device() else { return };
        // A game without coefficients gets its flat ambient and no mirror.
        let sky = Sky { zenith: vector(1.0, 1.0, 1.0), ..Sky::default() };
        let picture = probe(
            &device,
            &queue,
            &sky,
            &vector(0.2, 0.3, 0.4),
            "return vec4f(ambient_light(vec3f(0.0), direction), length(sky_reflection(vec3f(0.0), direction, direction, 0.3, vec3f(1.0))));",
        );
        for texel in &picture {
            assert!((texel[0] - 0.2).abs() < 1e-4 && (texel[1] - 0.3).abs() < 1e-4 && (texel[2] - 0.4).abs() < 1e-4);
            assert_eq!(texel[3], 0.0);
        }
    }

    const VOLUME: ([f32; 3], f32, [u32; 3]) = ([-1.0, 0.5, 2.0], 1.5, [4, 3, 3]);

    fn volume() -> ProbeVolume {
        let (origin, spacing, counts) = VOLUME;
        ProbeVolume {
            revision: 3,
            nx: counts[0],
            ny: counts[1],
            nz: counts[2],
            spacing,
            origin: vector(origin[0], origin[1], origin[2]),
        }
    }

    /// 36 probes of deterministic made-up light; every seventh is invalid.
    fn made_probes() -> Vec<Probe> {
        let mut seed = 7u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        (0..36)
            .map(|index| {
                let mut term = |scale: f32| vector(next() * scale, next() * scale, next() * scale);
                Probe {
                    sh0: term(1.0),
                    sh1: term(0.3),
                    sh2: term(0.3),
                    sh3: term(0.3),
                    sh4: term(0.1),
                    sh5: term(0.1),
                    sh6: term(0.1),
                    sh7: term(0.1),
                    sh8: term(0.1),
                    valid: if index % 7 == 3 { 0.0 } else { 1.0 },
                }
            })
            .collect()
    }

    fn eval(probe: &Probe, n: &[f64; 3]) -> [f64; 3] {
        let (x, y, z) = (n[0], n[1], n[2]);
        let basis = [
            0.282_094_791_8,
            0.488_602_511_9 * y,
            0.488_602_511_9 * z,
            0.488_602_511_9 * x,
            1.092_548_430_6 * x * y,
            1.092_548_430_6 * y * z,
            0.315_391_565_3 * (3.0 * z * z - 1.0),
            1.092_548_430_6 * x * z,
            0.546_274_215_3 * (x * x - y * y),
        ];
        let terms = [probe.sh0, probe.sh1, probe.sh2, probe.sh3, probe.sh4, probe.sh5, probe.sh6, probe.sh7, probe.sh8];
        std::array::from_fn(|c| {
            terms.iter().zip(basis).map(|(t, b)| f64::from([t.x, t.y, t.z][c]) * b).sum::<f64>().max(0.0)
        })
    }

    /// The probe volume's light and confidence at `world` facing `n`, in f64:
    /// the CPU interpolation `ink/probes.sample` does (bias a quarter spacing).
    fn interpolate(probes: &[Probe], world: [f64; 3], n: [f64; 3]) -> [f64; 4] {
        let (origin, spacing, counts) = VOLUME;
        let (spacing, origin) = (f64::from(spacing), origin.map(f64::from));
        let g: [f64; 3] = std::array::from_fn(|a| (world[a] + n[a] * 0.25 * spacing - origin[a]) / spacing);
        let last: [f64; 3] = std::array::from_fn(|a| f64::from(counts[a]) - 1.0);
        let edge = (0..3).map(|a| g[a].min(last[a] - g[a])).fold(f64::INFINITY, f64::min);
        let reach = (edge + 1.0).clamp(0.0, 1.0);
        if reach <= 0.0 {
            return [0.0; 4];
        }
        let c: [f64; 3] = std::array::from_fn(|a| g[a].clamp(0.0, last[a]));
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped into the lattice")]
        let low: [usize; 3] = std::array::from_fn(|a| (c[a].floor() as usize).min((last[a] - 1.0).max(0.0) as usize));
        let (mut sum, mut weight, mut share, mut total) = ([0.0; 3], 0.0, 0.0, 0.0);
        for corner in 0..8usize {
            let d = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
            let tri: f64 = (0..3)
                .map(|a| {
                    let f = c[a] - low[a] as f64;
                    if d[a] == 1 { f } else { 1.0 - f }
                })
                .product();
            if tri <= 0.0 {
                continue;
            }
            let at: [usize; 3] = std::array::from_fn(|a| (low[a] + d[a]).min(counts[a] as usize - 1));
            let index = (at[2] * counts[1] as usize + at[1]) * counts[0] as usize + at[0];
            let probe = &probes[index];
            total += tri;
            share += tri * f64::from(probe.valid);
            if probe.valid > 0.0 {
                let toward: [f64; 3] = std::array::from_fn(|a| origin[a] + at[a] as f64 * spacing - world[a]);
                let gap = toward.iter().map(|v| v * v).sum::<f64>().sqrt();
                let facing =
                    if gap < 1e-4 { 1.0 } else { 0.5 + 0.5 * (0..3).map(|a| toward[a] * n[a]).sum::<f64>() / gap };
                let w = tri * f64::from(probe.valid) * (facing * facing).max(0.02);
                let light = eval(probe, &n);
                for ch in 0..3 {
                    sum[ch] += w * light[ch];
                }
                weight += w;
            }
        }
        if weight <= 1e-6 || total <= 0.0 {
            return [0.0; 4];
        }
        [sum[0] / weight, sum[1] / weight, sum[2] / weight, reach * reach * (3.0 - 2.0 * reach) * share / total]
    }

    #[test]
    fn the_shader_samples_the_probe_volume_as_the_cpu_interpolation_does() {
        let Some((device, queue)) = device() else { return };
        let probes = made_probes();
        let volume = volume();
        let sky = sky(colors(), [vector(0.0, 0.0, 0.0); 9], vector(1.0, 1.0, 1.0));
        // Pixels spread over, round and beyond the lattice (x -2 to 5, y -0.5 to 4.5, z 1 to 6).
        let picture = probe_with(
            &device,
            &queue,
            &sky,
            &Vector::default(),
            (&volume, &probes),
            "let world = vec3f(position.x / 16.0 * 7.0 - 2.0, position.y / 16.0 * 5.0 - 0.5,
    f32((u32(position.x) * 7u + u32(position.y) * 3u) % 16u) / 16.0 * 5.0 + 1.0);
  return probe_ambient(world, direction);",
        );
        let mut inside = 0;
        for y in 0..SIDE {
            for x in 0..SIDE {
                let world = [
                    (f64::from(x) + 0.5) / 16.0 * 7.0 - 2.0,
                    (f64::from(y) + 0.5) / 16.0 * 5.0 - 0.5,
                    f64::from((x * 7 + y * 3) % 16) / 16.0 * 5.0 + 1.0,
                ];
                let want = interpolate(&probes, world, direction_at(x, y));
                let got = picture[(y * SIDE + x) as usize];
                inside += usize::from(want[3] > 0.5);
                for c in 0..4 {
                    assert!(
                        (f64::from(got[c]) - want[c]).abs() < 2e-4,
                        "pixel ({x}, {y}) channel {c}: {} vs {}",
                        got[c],
                        want[c]
                    );
                }
            }
        }
        assert!(inside > 40, "the pixels should mostly land in the volume: {inside}");
    }

    #[test]
    fn a_probe_is_sampled_exactly_at_its_centre_and_an_invalid_one_gives_nothing() {
        let Some((device, queue)) = device() else { return };
        let probes = made_probes();
        let volume = volume();
        let sky = sky(colors(), [vector(0.0, 0.0, 0.0); 9], vector(1.0, 1.0, 1.0));
        // Pixel (x, y) is probe (x + 16 y) % 36 at its centre; the lookup is lifted
        // by the bias, so ask where the lifted point lands on the probe.
        let picture = probe_with(
            &device,
            &queue,
            &sky,
            &Vector::default(),
            (&volume, &probes),
            "let index = (u32(position.x) + 16u * u32(position.y)) % 36u;
  let at = vec3u(index % 4u, (index / 4u) % 3u, index / 12u);
  let world = sky_light.probe_origin.xyz + vec3f(at) * sky_light.probe_origin.w - direction * (0.25 * sky_light.probe_origin.w);
  return probe_ambient(world, direction);",
        );
        for y in 0..SIDE {
            for x in 0..SIDE {
                let index = ((x + 16 * y) % 36) as usize;
                let probe = &probes[index];
                let got = picture[(y * SIDE + x) as usize];
                let n = direction_at(x, y);
                if probe.valid == 0.0 {
                    // Its neighbours may still weigh in a little (the lift moved off its centre).
                    continue;
                }
                // The lifted point is off the centre by a quarter cell, so this is the
                // interpolation's own answer, close to but not exactly the probe's.
                let reference = interpolate(
                    &probes,
                    {
                        let (origin, spacing, _) = VOLUME;
                        let at = [index % 4, (index / 4) % 3, index / 12];
                        std::array::from_fn(|a| {
                            f64::from(origin[a]) + at[a] as f64 * f64::from(spacing) - n[a] * 0.25 * f64::from(spacing)
                        })
                    },
                    n,
                );
                for c in 0..3 {
                    assert!((f64::from(got[c]) - reference[c]).abs() < 2e-4);
                }
            }
        }
    }

    #[test]
    fn the_cpu_interpolation_is_exact_at_probe_centres() {
        let probes = made_probes();
        for (index, probe) in probes.iter().enumerate() {
            let (origin, spacing, _) = VOLUME;
            let at = [index % 4, (index / 4) % 3, index / 12];
            let world: [f64; 3] = std::array::from_fn(|a| f64::from(origin[a]) + at[a] as f64 * f64::from(spacing));
            // With no lift the sample sits on the probe; the lift is part of the lookup, so
            // undo it to land on the centre.
            let n = [0.0, 1.0, 0.0];
            let lifted_back: [f64; 3] = std::array::from_fn(|a| world[a] - n[a] * 0.25 * f64::from(spacing));
            let got = interpolate(&probes, lifted_back, n);
            if probe.valid == 0.0 {
                assert!(got[3] < 1.0);
            } else {
                // Only the centre probe has trilinear weight: its own light, in full.
                let want = eval(probe, &n);
                for c in 0..3 {
                    assert!((got[c] - want[c]).abs() < 1e-5, "probe {index}");
                }
            }
        }
    }

    // --------------------------------------------------- reflection probes

    use pen_protocol::ReflectionProbe;

    fn v3(a: [f32; 3]) -> Vector {
        Vector { x: a[0], y: a[1], z: a[2] }
    }

    fn rp(id: u32, priority: u32, centre: [f32; 3], extent: [f32; 3], blend: f32) -> ReflectionProbe {
        ReflectionProbe {
            id,
            revision: 1,
            priority,
            capture: false,
            position: v3(centre),
            center: v3(centre),
            extent: v3(extent),
            blend,
            parallax_center: v3(centre),
            parallax_extent: v3(extent),
            far: 50.0,
        }
    }

    /// The table of `probes` in slot order, ranked as the shader reads them.
    fn table_of(probes: &[ReflectionProbe]) -> Vec<f32> {
        let rows: Vec<(usize, usize)> = (0..probes.len()).map(|index| (index, index)).collect();
        reflection_probes::table_floats(probes, &rows)
    }

    /// The influence weight, in f64: 0 outside, else the smoothstep of the distance in from the
    /// nearest face over `blend` (1 for a blend of 0).
    fn weight_of(probe: &ReflectionProbe, world: [f64; 3]) -> f64 {
        let (c, e) =
            ([probe.center.x, probe.center.y, probe.center.z], [probe.extent.x, probe.extent.y, probe.extent.z]);
        let edge = (0..3).map(|a| f64::from(e[a]) - (world[a] - f64::from(c[a])).abs()).fold(f64::INFINITY, f64::min);
        if edge <= 0.0 {
            return 0.0;
        }
        let t = if probe.blend > 0.0 { (edge / f64::from(probe.blend)).clamp(0.0, 1.0) } else { 1.0 };
        t * t * (3.0 - 2.0 * t)
    }

    /// The coverage of `probes` (best priority first, equal priorities blending), as the shader
    /// takes it: each priority covers 1 - the product of its misses, of what the higher left.
    /// At most four probes with a weight count.
    fn coverage_of(probes: &[ReflectionProbe], world: [f64; 3]) -> f64 {
        let mut order: Vec<&ReflectionProbe> = probes.iter().collect();
        order.sort_by_key(|probe| (std::cmp::Reverse(probe.priority), probe.id));
        let (mut remaining, mut missed, mut weight, mut priority, mut used) = (1.0, 1.0, 0.0, None, 0);
        for probe in order {
            let w = weight_of(probe, world);
            if w <= 0.0 {
                continue;
            }
            if used >= 4 {
                break;
            }
            used += 1;
            if priority != Some(probe.priority) {
                if weight > 0.0 {
                    remaining *= missed;
                }
                (missed, weight, priority) = (1.0, 0.0, Some(probe.priority));
            }
            weight += w;
            missed *= 1.0 - w;
        }
        if weight > 0.0 {
            remaining *= missed;
        }
        1.0 - remaining
    }

    fn world_at(x: u32, y: u32) -> [f64; 3] {
        // At the pixel's middle, as `position` is.
        let longitude = (f64::from(x) + 0.5) / f64::from(SIDE) * std::f64::consts::TAU;
        let up = 1.0 - 2.0 * (f64::from(y) + 0.5) / f64::from(SIDE);
        let ring = (1.0 - up * up).sqrt();
        [ring * longitude.cos(), up, ring * longitude.sin()]
    }

    #[test]
    fn box_projection_lands_where_the_reflected_ray_leaves_the_box() {
        let Some((device, queue)) = device() else { return };
        let mut probe = rp(1, 0, [1.0, 0.5, 0.0], [4.0, 2.0, 3.0], 0.0);
        probe.position = v3([1.5, 0.25, -0.5]);
        probe.parallax_center = v3([0.0, 0.5, 0.25]);
        probe.parallax_extent = v3([5.0, 2.5, 4.0]);
        let table = table_of(&[probe]);
        // The surface point is inside the box, a different place each pixel's direction.
        let got = probe_full(
            &device,
            &queue,
            &Sky::default(),
            &vector(0.0, 0.0, 0.0),
            (&ProbeVolume::default(), &[]),
            Some(&table),
            "return vec4f(box_projected(reflection_probes.probes[0], direction * 0.5 + vec3f(0.2, 0.1, -0.3), direction), 1.0);",
        );
        for y in 0..SIDE {
            for x in 0..SIDE {
                let d = world_at(x, y);
                let world = [d[0] * 0.5 + 0.2, d[1] * 0.5 + 0.1, d[2] * 0.5 - 0.3];
                let (centre, half): ([f64; 3], [f64; 3]) = ([0.0, 0.5, 0.25], [5.0, 2.5, 4.0]);
                // The slab method in f64: the nearest face the ray leaves through.
                let along = (0..3)
                    .filter(|&a| d[a].abs() > 1e-9)
                    .map(|a| ((centre[a] + half[a].copysign(d[a])) - world[a]) / d[a])
                    .fold(f64::INFINITY, f64::min);
                let hit: [f64; 3] = std::array::from_fn(|a| world[a] + d[a] * along);
                // The hit is on a face of the box, and no farther along would stay inside.
                let on_face = (0..3).any(|a| ((hit[a] - centre[a]).abs() - half[a]).abs() < 1e-9);
                assert!(on_face && (0..3).all(|a| (hit[a] - centre[a]).abs() <= half[a] + 1e-9));
                let want: [f64; 3] = std::array::from_fn(|a| hit[a] - [1.5, 0.25, -0.5][a]);
                let at = got[(y * SIDE + x) as usize];
                for a in 0..3 {
                    assert!((f64::from(at[a]) - want[a]).abs() < 1e-3, "({x},{y}) axis {a}: {} vs {}", at[a], want[a]);
                }
            }
        }
    }

    #[test]
    fn influence_weights_fade_smoothly_and_combine_by_priority() {
        let Some((device, queue)) = device() else { return };
        // Two equal priorities overlapping, a higher one inside them, and one far away.
        // In the order the table keeps: priority, highest first, then id.
        let probes = [
            rp(3, 2, [0.0, 0.0, 0.0], [0.6, 0.6, 0.6], 0.4),
            rp(4, 1, [9.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0),
            rp(1, 0, [-1.0, 0.0, 0.0], [1.5, 1.5, 1.5], 1.0),
            rp(2, 0, [1.0, 0.0, 0.0], [1.5, 1.5, 1.5], 0.5),
        ];
        let table = table_of(&probes);
        let got = probe_full(
            &device,
            &queue,
            &Sky::default(),
            &vector(0.0, 0.0, 0.0),
            (&ProbeVolume::default(), &[]),
            Some(&table),
            "let c = local_reflection(direction * 1.6, direction, 0.5); return vec4f(c.a, reflection_weight(reflection_probes.probes[2], direction * 1.6), reflection_weight(reflection_probes.probes[0], direction * 1.6), 1.0);",
        );
        let mut inside = 0;
        for y in 0..SIDE {
            for x in 0..SIDE {
                let d = world_at(x, y);
                let world = d.map(|v| v * 1.6);
                let at = got[(y * SIDE + x) as usize];
                let want = coverage_of(&probes, world);
                assert!((f64::from(at[0]) - want).abs() < 2e-4, "coverage at ({x},{y}): {} vs {want}", at[0]);
                assert!((0.0..=1.0 + 1e-6).contains(&f64::from(at[0])));
                assert!((f64::from(at[1]) - weight_of(&probes[2], world)).abs() < 2e-4);
                assert!((f64::from(at[2]) - weight_of(&probes[0], world)).abs() < 2e-4);
                inside += usize::from(want > 0.0);
            }
        }
        assert!(inside > 40, "the boxes cover some of the sphere");
    }

    #[test]
    fn a_weight_is_smooth_across_a_box_and_its_blend() {
        // Moving a point straight in through a face: the weight rises monotonically from 0 to
        // 1 over the blend, with no step anywhere (hard edges aside) and a zero slope at both ends.
        let probe = rp(1, 0, [0.0; 3], [2.0, 2.0, 2.0], 0.8);
        let mut last = 0.0;
        let mut biggest_step = 0.0f64;
        for step in 0..=400 {
            let x = 2.0 - f64::from(step) * 0.005;
            let w = weight_of(&probe, [x, 0.0, 0.0]);
            assert!(w >= last - 1e-12, "monotonic");
            biggest_step = biggest_step.max(w - last);
            last = w;
        }
        assert_eq!(last, 1.0);
        assert!(biggest_step < 0.03, "no jump: {biggest_step}");
    }
}

#[cfg(test)]
mod template {
    #[test]
    fn the_new_game_template_carries_the_current_environment_block() {
        let template = include_str!("../../pen/templates/new/src/assets/lit.wgsl");
        let begin = template.find("// BEGIN environment\n").expect("a begin marker") + "// BEGIN environment\n".len();
        let end = template.find("// END environment").expect("an end marker");
        assert_eq!(&template[begin..end], super::ENVIRONMENT_WGSL, "copy shaders/environment.wgsl into lit.wgsl");
    }
}
