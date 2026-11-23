// The sky as light for the scene's materials. Diffuse light comes from nine
// spherical-harmonic coefficients the game works out from its sky (they ride
// in the render packet; the host only passes them on). Glossy reflections
// come from a cube map of the same sky, drawn here from the frame's three sky
// colors, whose levels are the sky blurred by GGX lobes of growing roughness,
// and the split-sum BRDF table, drawn once. The cube is redrawn on the GPU in
// the frame whose sky colors differ from the last one's, and not otherwise.
// `ENVIRONMENT_WGSL` is what a material shader adds to read all of it, at
// group 0 bindings 4 to 7 of every mesh pipeline.
use pen_protocol::{Sky, Vector};

use crate::pipeline_cache::PipelineCache;

/// WGSL a material shader adds to light with the sky: `ambient_light(world,
/// normal)`, `sky_reflection`, `surface_ambient` and `specular_occlusion`.
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
/// The uniform `ENVIRONMENT_WGSL` reads: nine coefficients, the tint and the
/// reflection settings.
pub(crate) const LIGHT_BYTES: u64 = 16 * 11;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub(crate) struct Environment {
    cache: PipelineCache,
    cube: wgpu::Texture,
    cube_view: wgpu::TextureView,
    _table: wgpu::Texture,
    table_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// The frame's light, and the zeros the shadow passes bind.
    light: wgpu::Buffer,
    empty: wgpu::Buffer,
    /// The passes' uniforms, in slots with dynamic offsets.
    passes: wgpu::Buffer,
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

/// The scene group's bindings 4 to 7: the cube, the table, a filtering
/// sampler and the light.
pub(crate) fn layout_entries() -> [wgpu::BindGroupLayoutEntry; 4] {
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
/// (w: 1 when the cube is the sky to reflect), and the cube's last level.
pub(crate) fn light_floats(sh: &[Vector; 9], tint: &Vector, reflecting: bool) -> [f32; 44] {
    let mut floats = [0.0; 44];
    for (slot, v) in sh.iter().enumerate() {
        floats[slot * 4..slot * 4 + 3].copy_from_slice(&[v.x, v.y, v.z]);
    }
    floats[36..40].copy_from_slice(&[tint.x, tint.y, tint.z, if reflecting { 1.0 } else { 0.0 }]);
    floats[40] = (LEVELS - 1) as f32;
    floats
}

/// A flat light, `ambient` from every direction, which reflects nothing.
pub(crate) fn flat_floats(ambient: &Vector) -> [f32; 44] {
    // Coefficient 0 of a constant is the constant over the first basis function.
    let scale = 1.0 / 0.282_094_8;
    let zero = Vector { x: 0.0, y: 0.0, z: 0.0 };
    let mut sh = [zero; 9];
    sh[0] = Vector { x: ambient.x * scale, y: ambient.y * scale, z: ambient.z * scale };
    light_floats(&sh, &zero, false)
}

/// The sky's coefficients and tint from the frame, or its flat `ambient`
/// when the game sent none, so a game that knows nothing of them still has
/// light.
pub(crate) fn sky_floats(sky: &Sky, ambient: &Vector) -> [f32; 44] {
    let sh = [sky.sh0, sky.sh1, sky.sh2, sky.sh3, sky.sh4, sky.sh5, sky.sh6, sky.sh7, sky.sh8];
    let sent = sh.iter().chain([&sky.tint]).any(|v| v.x != 0.0 || v.y != 0.0 || v.z != 0.0);
    if !sent {
        return flat_floats(ambient);
    }
    light_floats(&sh, &sky.tint, true)
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
            empty: buffer(device, "no sky light", LIGHT_BYTES, wgpu::BufferUsages::UNIFORM),
            passes: buffer(device, "sky passes", STRIDE * PASSES, wgpu::BufferUsages::UNIFORM),
            drawn: None,
            stale: false,
            table_drawn: false,
            pipelines: None,
        }
    }

    /// The bindings 4 to 7 of a scene group, with `light` as the sky's light.
    pub fn entries<'a>(&'a self, light: &'a wgpu::Buffer) -> [wgpu::BindGroupEntry<'a>; 4] {
        [
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.cube_view) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.table_view) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            wgpu::BindGroupEntry { binding: 7, resource: light.as_entire_binding() },
        ]
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
    pub fn update(&mut self, queue: &wgpu::Queue, sky: &Sky, ambient: &Vector) {
        queue.write_buffer(&self.light, 0, bytemuck::cast_slice(&sky_floats(sky, ambient)));
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
        assert_eq!(sky_floats(&sky, &ambient), flat_floats(&ambient));
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
        let mut environment = Environment::new(device, PipelineCache::open(device, None));
        environment.update(queue, sky, ambient);
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
        let layout = device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &layout_entries() });
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
            entries: &environment.entries(environment.light()),
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
  let got = sky_reflection(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 1.0), 0.5, f0);
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
            "return vec4f(ambient_light(vec3f(0.0), direction), length(sky_reflection(direction, direction, 0.3, vec3f(1.0))));",
        );
        for texel in &picture {
            assert!((texel[0] - 0.2).abs() < 1e-4 && (texel[1] - 0.3).abs() < 1e-4 && (texel[2] - 0.4).abs() < 1e-4);
            assert_eq!(texel[3], 0.0);
        }
    }
}
