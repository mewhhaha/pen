//! The shadows on a real (headless) GPU against f64 references: the sun's
//! filters, soft shadows, cascade blending and bias through `SHADOWS_WGSL`
//! as materials read it, and the local lights' maps through `LIGHTS_WGSL`.
//! Each test draws depth maps with its own tiny pipeline, so that what the
//! shader reads is known exactly, and reads the shaders' answers back.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, reason = "test geometry")]
use pen_protocol::{Light, RenderFrame, Vector};
use wgpu::util::DeviceExt;

use super::cascades::cascades;
use super::shadows::{SHADOWS_WGSL, scene_floats};
use crate::lights::LIGHTS_WGSL;
use crate::local_shadows;
use crate::matrix::{IDENTITY, multiply};

type Matrix = [f32; 16];

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    pollster::block_on(adapter.request_device(&Default::default())).ok()
}

// ---------------------------------------------------------------- geometry

/// A flat quad of two triangles, as vertices.
fn quad(corners: [[f32; 3]; 4]) -> Vec<[f32; 3]> {
    vec![corners[0], corners[1], corners[2], corners[0], corners[2], corners[3]]
}

/// The light's view looking along `ray`, as Ink's `graphics.shadow` anchors
/// it: 1000 back along the ray from the origin.
fn light_view(ray: [f64; 3]) -> Matrix {
    let length = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
    let ray = ray.map(|value| value / length);
    let across = if ray[1].abs() > 0.99 { [0.0, 0.0, 1.0] } else { [0.0, 1.0, 0.0] };
    let cross =
        |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let normalize = |a: [f64; 3]| {
        let length = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        a.map(|value| value / length)
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let right = normalize(cross(ray, across));
    let up = cross(right, ray);
    let origin = ray.map(|value| value * -1000.0);
    [
        right[0],
        up[0],
        -ray[0],
        0.0,
        right[1],
        up[1],
        -ray[1],
        0.0,
        right[2],
        up[2],
        -ray[2],
        0.0,
        -dot(right, origin),
        -dot(up, origin),
        dot(ray, origin),
        1.0,
    ]
    .map(|value| value as f32)
}

/// An orthographic window of half-width `radius` round (`x`, `y`) of the
/// light's view, for depths `near` to `far` (distances from its eye).
fn window(x: f64, y: f64, radius: f64, near: f64, far: f64) -> Matrix {
    let mut m = IDENTITY;
    m[0] = (1.0 / radius) as f32;
    m[5] = (1.0 / radius) as f32;
    m[10] = (-1.0 / (far - near)) as f32;
    m[12] = (-x / radius) as f32;
    m[13] = (-y / radius) as f32;
    m[14] = (-near / (far - near)) as f32;
    m
}

/// Where `matrix` puts `point`: texture coordinates and depth.
fn project(matrix: &Matrix, point: [f64; 3]) -> [f64; 3] {
    let m = matrix.map(f64::from);
    let clip = [0, 1, 2, 3].map(|row| m[row] * point[0] + m[4 + row] * point[1] + m[8 + row] * point[2] + m[12 + row]);
    [clip[0] / clip[3] * 0.5 + 0.5, 0.5 - clip[1] / clip[3] * 0.5, clip[2] / clip[3]]
}

// --------------------------------------------------------------------- GPU

struct Rig {
    device: wgpu::Device,
    queue: wgpu::Queue,
    depth: wgpu::RenderPipeline,
}

impl Rig {
    fn new() -> Option<Rig> {
        let (device, queue) = device()?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test depth"),
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<uniform> m: mat4x4f;
                 @vertex fn vertex(@location(0) p: vec3f) -> @builtin(position) vec4f { return m * vec4f(p, 1.0); }"
                    .into(),
            ),
        });
        let depth = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test depth"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            // What the renderer's shadow pipelines use.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });
        Some(Rig { device, queue, depth })
    }

    /// A depth texture array of `layers` maps of `size`, cleared to 1.
    fn maps(&self, size: u32, layers: u32) -> wgpu::Texture {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test maps"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        for layer in 0..layers {
            self.draw(&texture, layer, &IDENTITY, &[]);
        }
        texture
    }

    /// Draws `triangles` through `matrix` into `layer` of `maps`, clearing it first.
    fn draw(&self, maps: &wgpu::Texture, layer: u32, matrix: &Matrix, triangles: &[[f32; 3]]) {
        let view = maps.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        });
        self.draw_into(&view, matrix, triangles);
    }

    fn draw_into(&self, view: &wgpu::TextureView, matrix: &Matrix, triangles: &[[f32; 3]]) {
        let uniform = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(matrix),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.depth.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }],
        });
        let vertices = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(if triangles.is_empty() { &[[0.0f32; 3]; 3] } else { triangles }),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            if !triangles.is_empty() {
                pass.set_pipeline(&self.depth);
                pass.set_bind_group(0, &group, &[]);
                pass.set_vertex_buffer(0, vertices.slice(..));
                pass.draw(0..u32::try_from(triangles.len()).unwrap(), 0..1);
            }
        }
        self.queue.submit([encoder.finish()]);
    }

    /// Runs `kernel` (WGSL with an `main` entry) with `groups` bound at 0, 1, ... and reads `count` floats back from `out`.
    fn compute(
        &self,
        source: &str,
        groups: &[(u32, Vec<(u32, wgpu::BindingResource<'_>)>)],
        out: &wgpu::Buffer,
        count: usize,
    ) -> Vec<f32> {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test kernel"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("test kernel"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bound: Vec<wgpu::BindGroup> = groups
            .iter()
            .map(|(index, entries)| {
                let (index, entries) = (*index, entries);
                let entries: Vec<_> = entries
                    .iter()
                    .map(|(binding, resource)| wgpu::BindGroupEntry { binding: *binding, resource: resource.clone() })
                    .collect();
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(index),
                    entries: &entries,
                })
            })
            .collect();
        let bytes = (count * 4) as u64;
        let read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for ((index, _), group) in groups.iter().zip(&bound) {
                pass.set_bind_group(*index, group, &[]);
            }
            pass.dispatch_workgroups(u32::try_from(count.div_ceil(64)).unwrap(), 1, 1);
        }
        encoder.copy_buffer_to_buffer(out, 0, &read, 0, bytes);
        self.queue.submit([encoder.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mapped = read.slice(..).get_mapped_range().unwrap();
        bytemuck::cast_slice(&mapped).to_vec()
    }

    fn storage(&self, contents: &[f32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(contents),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    }
}

// ------------------------------------------------------------------ the sun

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

/// Reads the sun's visibility at each of `points` (world, normal, spin),
/// and the contact shadow's there, through the library as materials call it.
fn sun_kernel() -> String {
    format!(
        "{SCENE}{SHADOWS_WGSL}
@group(1) @binding(0) var<storage, read> points: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> answers: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {{
  if (id.x >= arrayLength(&answers) / 2u) {{ return; }}
  let world = points[id.x * 2u];
  let normal = points[id.x * 2u + 1u];
  answers[id.x * 2u] = sun_visibility(world.xyz, normal.xyz, vec2f(world.w, normal.w));
  answers[id.x * 2u + 1u] = contact_shadow(world.xyz, normal.xyz, 0.5);
}}"
    )
}

struct Sun {
    frame: RenderFrame,
    maps: wgpu::Texture,
    contact: wgpu::Texture,
}

impl Sun {
    /// A frame whose sun shines along `ray`, with `windows` (radius, centre
    /// x and y in the light's view, distance ahead of the camera where the
    /// slice ends) as cascades on maps of `size`.
    fn new(rig: &Rig, ray: [f64; 3], size: u32, windows: &[(f64, f64, f64, f64)]) -> Sun {
        let mut frame = RenderFrame {
            light_direction: Vector { x: ray[0] as f32, y: ray[1] as f32, z: ray[2] as f32 },
            shadow: pen_protocol::Shadow { view: light_view(ray), texels: size, filter: 1, ..Default::default() },
            ..Default::default()
        };
        for &(radius, x, y, far) in windows {
            frame.cascades.push(pen_protocol::Cascade {
                projection: window(x, y, radius, 980.0, 1020.0),
                texel: (2.0 * radius / f64::from(size)) as f32,
                far: far as f32,
            });
        }
        let maps = rig.maps(size, 4);
        let contact = rig.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test contact"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        Sun { frame, maps, contact }
    }

    /// Replaces the contact depth with the camera's view of `triangles`, `size` square.
    fn see(&mut self, rig: &Rig, size: u32, triangles: &[[f32; 3]]) {
        self.contact = rig.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test contact"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let matrix = multiply(&self.frame.projection, &self.frame.view);
        rig.draw_into(&self.contact.create_view(&Default::default()), &matrix, triangles);
    }

    /// Draws `triangles` into cascade `map`'s map.
    fn cast(&self, rig: &Rig, map: usize, triangles: &[[f32; 3]]) {
        let matrix = multiply(&self.frame.cascades[map].projection, &self.frame.shadow.view);
        rig.draw(&self.maps, u32::try_from(map).unwrap(), &matrix, triangles);
    }

    /// The sun's visibility and the contact shadow's at `points`.
    fn read(&self, rig: &Rig, points: &[([f32; 3], [f32; 3], [f32; 2])]) -> Vec<[f32; 2]> {
        let (maps, count) = cascades(&self.frame, true);
        let scene = scene_floats(&self.frame, &maps, count);
        let uniform = rig.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: scene.bytes(),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let sampler = rig.device.create_sampler(&wgpu::SamplerDescriptor {
            compare: Some(wgpu::CompareFunction::Less),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let all = self.maps.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let contact = self.contact.create_view(&Default::default());
        let mut lanes = Vec::new();
        for (world, normal, spin) in points {
            lanes.extend([world[0], world[1], world[2], spin[0], normal[0], normal[1], normal[2], spin[1]]);
        }
        let input = rig.storage(&lanes);
        let output = rig.storage(&vec![0.0; points.len() * 2]);
        let values = rig.compute(
            &sun_kernel(),
            &[
                (
                    0,
                    vec![
                        (0, uniform.as_entire_binding()),
                        (2, wgpu::BindingResource::Sampler(&sampler)),
                        (3, wgpu::BindingResource::TextureView(&all)),
                        (9, wgpu::BindingResource::TextureView(&contact)),
                    ],
                ),
                (1, vec![(0, input.as_entire_binding()), (1, output.as_entire_binding())]),
            ],
            &output,
            points.len() * 2,
        );
        values.chunks(2).map(|pair| [pair[0], pair[1]]).collect()
    }
}

/// The sixteen taps of `shadows.wgsl`.
const TAPS: [[f64; 2]; 16] = [
    [-0.94201624, -0.39906216],
    [0.94558609, -0.76890725],
    [-0.09418410, -0.92938870],
    [0.34495938, 0.29387760],
    [-0.91588581, 0.45771432],
    [-0.81544232, -0.87912464],
    [-0.38277543, 0.27676845],
    [0.97484398, 0.75648379],
    [0.44323325, -0.97511554],
    [0.53742981, -0.47373420],
    [-0.26496911, -0.41893023],
    [0.79197514, 0.19090188],
    [-0.24188840, 0.99706507],
    [-0.81409955, 0.91437590],
    [0.19984126, 0.78641367],
    [0.14383161, -0.14100790],
];

/// The f64 reference of the shaders' sun filters over a map whose content
/// `stored` gives (texel centre in texture coordinates to depth), for a
/// receiver square to the light: PCF's and PCSS's, as `shadows.wgsl`
/// writes them.
struct Reference {
    size: f64,
    /// A texel's width in world units, and the depth of one world unit.
    texel: f64,
    unit: f64,
    mode: u32,
    softness: f64,
}

impl Reference {
    fn depth_at(&self, stored: &dyn Fn(f64, f64) -> f64, tx: f64, ty: f64) -> f64 {
        stored((tx + 0.5) / self.size, (ty + 0.5) / self.size)
    }

    /// What a hardware 2 x 2 compare returns for `uv` and `reference`.
    fn compare(&self, stored: &dyn Fn(f64, f64) -> f64, uv: [f64; 2], reference: f64) -> f64 {
        let (x, y) = (uv[0] * self.size - 0.5, uv[1] * self.size - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let mut total = 0.0;
        for (dx, dy) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
            let weight = (if dx == 0.0 { 1.0 - fx } else { fx }) * (if dy == 0.0 { 1.0 - fy } else { fy });
            total += weight * f64::from(u8::from(reference < self.depth_at(stored, x0 + dx, y0 + dy)));
        }
        total
    }

    fn filtered(&self, stored: &dyn Fn(f64, f64) -> f64, uv: [f64; 2], depth: f64, radius: f64, spin: [f64; 2]) -> f64 {
        let bias = 1.0 * self.texel * self.unit;
        let mut light = 0.0;
        for tap in TAPS {
            let offset =
                [(tap[0] * spin[0] - tap[1] * spin[1]) * radius, (tap[0] * spin[1] + tap[1] * spin[0]) * radius];
            light += self.compare(stored, [uv[0] + offset[0] / self.size, uv[1] + offset[1] / self.size], depth - bias);
        }
        light / 16.0
    }

    /// The visibility of a receiver at `uv`, `depth`, as the shader finds it.
    fn visibility(&self, stored: &dyn Fn(f64, f64) -> f64, uv: [f64; 2], depth: f64, spin: [f64; 2]) -> f64 {
        let bias = 1.0 * self.texel * self.unit;
        match self.mode {
            0 => self.compare(stored, uv, depth - bias),
            1 => self.filtered(stored, uv, depth, 1.5, spin),
            _ => {
                let search = (self.softness * 8.0 * 0.5 / self.texel).clamp(1.0, 11.0);
                let (mut gap, mut blockers) = (0.0, 0.0);
                for tap in &TAPS[..12] {
                    let offset = [
                        (tap[0] * spin[0] - tap[1] * spin[1]) * search,
                        (tap[0] * spin[1] + tap[1] * spin[0]) * search,
                    ];
                    let (tx, ty) = ((uv[0] * self.size + offset[0]).floor(), (uv[1] * self.size + offset[1]).floor());
                    let seen = self.depth_at(stored, tx, ty);
                    // The same f32 compare as the shader's, to the bit where it matters.
                    if (seen as f32) < ((depth - bias) as f32) {
                        gap += (depth - bias - seen) / self.unit;
                        blockers += 1.0;
                    }
                }
                if blockers == 0.0 {
                    return 1.0;
                }
                let radius = (self.softness * (gap / blockers) * 0.5 / self.texel).clamp(1.5, 12.0);
                self.filtered(stored, uv, depth, radius, spin)
            }
        }
    }
}

/// A scene: the sun straight down, one cascade 16 wide round the origin on a
/// 1024 map, and a slab at height `height` covering everything left of x = 0.
/// Receivers lie on the floor, y = 0. `mode` is the frame's filter.
struct Slab {
    sun: Sun,
    reference: Reference,
    height: f64,
}

fn slab(rig: &Rig, mode: u32, height: f64, softness: f32) -> Slab {
    let mut sun = Sun::new(rig, [0.0, -1.0, 0.0], 1024, &[(16.0, 0.0, 0.0, 100.0)]);
    sun.frame.shadow.filter = mode;
    sun.frame.shadow.softness = softness;
    sun.frame.view = IDENTITY;
    sun.frame.shadow.contact = 0;
    // The camera looks down -z: depth ahead is -z, so points need z in (-100, 0).
    let h = height as f32;
    sun.cast(rig, 0, &quad([[-30.0, h, -30.0], [-30.0, h, 30.0], [0.0, h, 30.0], [0.0, h, -30.0]]));
    let texel = 2.0 * 16.0 / 1024.0;
    Slab {
        reference: Reference { size: 1024.0, texel, unit: 1.0 / 40.0, mode, softness: f64::from(softness) },
        sun,
        height,
    }
}

impl Slab {
    /// The slab's stored depth at texture coordinates (u, v), or the map's clear.
    fn stored(&self) -> impl Fn(f64, f64) -> f64 + '_ {
        let matrix = multiply(&self.sun.frame.cascades[0].projection, &self.sun.frame.shadow.view);
        let top = project(&matrix, [0.0, self.height, 0.0])[2];
        move |u, v| {
            // Invert uv to the world x on the floor: sample the matrix along x.
            let at_zero = project(&matrix, [0.0, 0.0, 0.0]);
            let at_one = project(&matrix, [1.0, 0.0, 0.0]);
            let at_z = project(&matrix, [0.0, 0.0, 1.0]);
            // uv is affine in (x, z); solve for x given (u, v).
            let (ax, ay) = (at_one[0] - at_zero[0], at_one[1] - at_zero[1]);
            let (bx, by) = (at_z[0] - at_zero[0], at_z[1] - at_zero[1]);
            let (du, dv) = (u - at_zero[0], v - at_zero[1]);
            let determinant = ax * by - ay * bx;
            let x = (du * by - dv * bx) / determinant;
            if x < 0.0 { top } else { 1.0 }
        }
    }

    /// The GPU's visibilities of receivers at `xs` (y = 0, z = -10, facing up) with `spin`.
    fn gpu(&self, rig: &Rig, xs: &[f64], spin: [f32; 2]) -> Vec<f32> {
        let points: Vec<_> = xs.iter().map(|&x| ([x as f32, 0.0, -10.0], [0.0, 1.0, 0.0], spin)).collect();
        self.sun.read(rig, &points).into_iter().map(|pair| pair[0]).collect()
    }

    /// The reference's.
    fn cpu(&self, xs: &[f64], spin: [f64; 2]) -> Vec<f64> {
        let matrix = multiply(&self.sun.frame.cascades[0].projection, &self.sun.frame.shadow.view);
        let stored = self.stored();
        xs.iter()
            .map(|&x| {
                let at = project(&matrix, [x, 0.0, -10.0]);
                self.reference.visibility(&stored, [at[0], at[1]], at[2], spin)
            })
            .collect()
    }
}

/// `count` positions across `from..to`.
fn across(from: f64, to: f64, count: usize) -> Vec<f64> {
    (0..count).map(|index| from + (to - from) * index as f64 / (count - 1) as f64).collect()
}

#[test]
fn filters_match_the_reference_and_lit_and_shadowed_stay_exact() {
    let Some(rig) = Rig::new() else { return };
    let xs = across(-1.0, 1.0, 161);
    for mode in [0, 1, 2] {
        let slab = slab(&rig, mode, 1.0, 0.2);
        let spin = [0.8f64.cos(), 0.8f64.sin()];
        let gpu = slab.gpu(&rig, &xs, [spin[0] as f32, spin[1] as f32]);
        let cpu = slab.cpu(&xs, spin);
        let worst = gpu.iter().zip(&cpu).map(|(g, c)| (f64::from(*g) - c).abs()).fold(0.0, f64::max);
        assert!(worst < 0.04, "mode {mode}: the GPU strays {worst} from the reference");
        // Far from the edge: exactly lit and exactly shadowed.
        let (lit, dark) = (slab.gpu(&rig, &[2.0, 5.0], [1.0, 0.0]), slab.gpu(&rig, &[-2.0, -5.0], [1.0, 0.0]));
        assert_eq!(lit, vec![1.0, 1.0], "mode {mode}: lit stays 1");
        assert_eq!(dark, vec![0.0, 0.0], "mode {mode}: shadowed stays 0");
        // The mode matters: hard shadows are a step, soft ones a ramp.
        let ramp = gpu.iter().filter(|value| **value > 0.05 && **value < 0.95).count();
        if mode == 0 {
            assert!(ramp <= 6, "hard shadows have a step, not a ramp: {ramp} in between");
        } else {
            assert!(ramp >= 6, "mode {mode} softens: {ramp} in between");
        }
    }
}

/// The distance over which `values` (at `xs`, rising) go from 10% to 90%.
fn ramp(xs: &[f64], values: &[f32]) -> f64 {
    let at = |level: f32| {
        let index = values.iter().position(|&value| value >= level).expect("reaches the level");
        let (x0, x1, v0, v1) = (xs[index - 1], xs[index], values[index - 1], values[index]);
        x0 + (x1 - x0) * f64::from((level - v0) / (v1 - v0))
    };
    at(0.9) - at(0.1)
}

#[test]
fn the_penumbra_widens_with_the_distance_to_the_blocker() {
    let Some(rig) = Rig::new() else { return };
    let xs = across(-1.0, 1.0, 401);
    let mut widths = Vec::new();
    for height in [0.5, 1.0, 2.0] {
        let slab = slab(&rig, 2, height, 0.2);
        let spin = [0.8f64.cos(), 0.8f64.sin()];
        let gpu = slab.gpu(&rig, &xs, [spin[0] as f32, spin[1] as f32]);
        let cpu = slab.cpu(&xs, spin);
        let worst = gpu.iter().zip(&cpu).map(|(g, c)| (f64::from(*g) - c).abs()).fold(0.0, f64::max);
        assert!(worst < 0.04, "height {height}: the GPU strays {worst} from the reference");
        widths.push(ramp(&xs, &gpu));
    }
    // The penumbra is `softness` wide per unit of distance to the blocker:
    // the sixteen taps turn a straight edge into a ramp from 10% to 90% over
    // about that much. The filter's own minimum keeps the nearest from
    // narrowing below its 1.5 texels' reach.
    for (height, width) in [0.5, 1.0, 2.0].iter().zip(&widths) {
        let wanted = 0.2 * height;
        assert!((width / wanted - 1.0).abs() < 0.25, "height {height}: width {width}, wanted {wanted}");
    }
    assert!(widths[1] > widths[0] * 1.2 && widths[2] > widths[1] * 1.6, "{widths:?}");
    let doubled = widths[2] / widths[1];
    assert!((doubled - 2.0).abs() < 0.4, "doubling the distance doubles the penumbra: {doubled}");
}

#[test]
fn cascades_blend_continuously_across_their_split() {
    let Some(rig) = Rig::new() else { return };
    // Two maps over the same ground; the near one holds a slab over all of
    // it, the far one is empty: the answer is the blend itself.
    let mut sun = Sun::new(&rig, [0.0, -1.0, 0.0], 512, &[(30.0, 0.0, 0.0, 20.0), (30.0, 0.0, 0.0, 60.0)]);
    sun.frame.view = IDENTITY;
    sun.frame.shadow.filter = 1;
    sun.cast(&rig, 0, &quad([[-40.0, 3.0, -40.0], [-40.0, 3.0, 40.0], [40.0, 3.0, 40.0], [40.0, 3.0, -40.0]]));
    let depths = across(1.0, 70.0, 691);
    let points: Vec<_> = depths.iter().map(|&d| ([0.0, 0.0, -d as f32], [0.0, 1.0, 0.0], [1.0, 0.0])).collect();
    let seen: Vec<f64> = sun.read(&rig, &points).iter().map(|pair| f64::from(pair[0])).collect();
    // The first map's slice is 0..20 and blends over its last share (0.15),
    // 17 to 20; the second's over 53 to 60 into the lit beyond it.
    let smooth = |t: f64| t.clamp(0.0, 1.0) * t.clamp(0.0, 1.0) * (3.0 - 2.0 * t.clamp(0.0, 1.0));
    let mut jump: f64 = 0.0;
    for (index, (&depth, &value)) in depths.iter().zip(&seen).enumerate() {
        // Shadowed by the near map until its last share, blending into the
        // far map's, which holds nothing, and lit past that.
        let expected = if depth < 20.0 { smooth((depth - 17.0) / 3.0) } else { 1.0 };
        assert!((value - expected).abs() < 0.02, "depth {depth}: {value}, expected {expected}");
        if index > 0 {
            jump = jump.max((value - seen[index - 1]).abs());
        }
    }
    // Neighbours 0.1 apart never differ by more than the blend's steepest slope allows.
    assert!(jump < 0.06, "the blend jumps by {jump}");
    assert_eq!(seen[0], 0.0);
}

#[test]
fn no_acne_on_a_lit_plane_at_grazing_angles_and_no_peter_panning() {
    let Some(rig) = Rig::new() else { return };
    let floor = quad([[-14.0, 0.0, -14.0], [-14.0, 0.0, 14.0], [14.0, 0.0, 14.0], [14.0, 0.0, -14.0]]);
    for mode in [0, 1, 2] {
        for elevation in [4.0f64, 8.0, 15.0, 30.0, 60.0, 89.0] {
            let angle = elevation.to_radians();
            let ray = [angle.cos(), -angle.sin(), 0.0];
            let mut sun = Sun::new(&rig, ray, 1024, &[(24.0, 0.0, 0.0, 100.0)]);
            sun.frame.view = IDENTITY;
            sun.frame.shadow.filter = mode;
            sun.frame.shadow.softness = 0.05;
            // The floor casts into its own map and receives the sun from it.
            sun.cast(&rig, 0, &floor);
            let mut points = Vec::new();
            for i in 0..40 {
                for j in 0..40 {
                    let (x, z) =
                        (-9.0 + 18.0 * f64::from(i) / 39.0 + 0.0137, -12.0 + 10.0 * f64::from(j) / 39.0 + 0.0291);
                    points.push(([x as f32, 0.0, z as f32 - 1.0], [0.0f32, 1.0, 0.0], [0.6f32, 0.8]));
                }
            }
            let seen = sun.read(&rig, &points);
            let darkest = seen.iter().map(|pair| pair[0]).fold(1.0, f32::min);
            assert!(darkest > 0.999, "mode {mode}, sun {elevation} degrees up: acne, darkest {darkest}");
        }
    }
    // A box on the floor casts a shadow that starts at its foot: the floor
    // just past its base, away from the sun, is dark, not lifted off it.
    let (x0, x1, height) = (-1.0f32, 1.0f32, 2.0f32);
    let mut solid = floor.clone();
    // Its top and its two long sides facing the sun and away (z extends 6 either way).
    solid.extend(quad([[x0, height, -6.0], [x0, height, 6.0], [x1, height, 6.0], [x1, height, -6.0]]));
    solid.extend(quad([[x1, 0.0, -6.0], [x1, height, -6.0], [x1, height, 6.0], [x1, 0.0, 6.0]]));
    solid.extend(quad([[x0, 0.0, 6.0], [x0, height, 6.0], [x0, height, -6.0], [x0, 0.0, -6.0]]));
    for mode in [0, 1, 2] {
        let ray = [0.6f64, -0.8, 0.0];
        let mut sun = Sun::new(&rig, ray, 2048, &[(24.0, 0.0, 0.0, 100.0)]);
        sun.frame.view = IDENTITY;
        sun.frame.shadow.filter = mode;
        sun.frame.shadow.softness = 0.02;
        sun.cast(&rig, 0, &solid);
        // The sun comes from the left going right: the shadow starts at the
        // box's right foot x = 1 and runs to x = 1 + 2 * 0.6 / 0.8 = 2.5.
        let probes = [
            // Just past the base: a few texels in, dark; and well inside, dark.
            ([1.0 + 0.06f32, 0.0, 0.0], 0.0),
            ([1.9f32, 0.0, 0.0], 0.0),
            // Past the shadow's far end, lit; and not shadowed in front of the box.
            ([2.5 + 0.12f32, 0.0, 0.0], 1.0),
            ([-1.2f32, 0.0, 0.0], 1.0),
        ];
        let points: Vec<_> = probes.iter().map(|(at, _)| (*at, [0.0f32, 1.0, 0.0], [1.0f32, 0.0])).collect();
        let seen = sun.read(&rig, &points);
        for ((at, want), got) in probes.iter().zip(&seen) {
            assert!((got[0] - want).abs() < 0.1, "mode {mode}: at {at:?} the sun is {} where {want} is wanted", got[0]);
        }
    }
}

/// A box from (-x, 0, -z) to (x, height, z), as triangles.
fn block(x: f32, height: f32, z: f32) -> Vec<[f32; 3]> {
    let corner = |i: u32| {
        [if i & 1 == 0 { -x } else { x }, if i & 2 == 0 { 0.0 } else { height }, if i & 4 == 0 { -z } else { z }]
    };
    let mut triangles = Vec::new();
    for quad_corners in [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]] {
        triangles.extend(quad(quad_corners.map(corner)));
    }
    triangles
}

#[test]
fn contact_shadows_darken_behind_a_small_caster_and_leave_the_rest() {
    let Some(rig) = Rig::new() else { return };
    // The sun runs along +x and down; a post 0.6 across and 1 tall stands at
    // the origin and its shadow on the floor reaches 1.6 from it, which the
    // coarse cascade (one texel is 0.5) cannot show.
    let mut sun = Sun::new(&rig, [0.8, -0.5, 0.0], 64, &[(16.0, 0.0, 0.0, 100.0)]);
    let eye = [0.0f32, 5.0, 6.0];
    // A camera looking from the eye at (0, 0.3, 0): view and a perspective projection.
    let forward = {
        let d = [0.0 - eye[0], 0.3 - eye[1], 0.0 - eye[2]];
        let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        d.map(|value| value / length)
    };
    let side = [1.0f32, 0.0, 0.0];
    let above = [
        side[1] * forward[2] - side[2] * forward[1],
        side[2] * forward[0] - side[0] * forward[2],
        side[0] * forward[1] - side[1] * forward[0],
    ];
    // The camera's up is `side x forward`'s negation: right x forward = down, so up = forward x side.
    let up = [-above[0], -above[1], -above[2]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    sun.frame.view = [
        side[0],
        up[0],
        -forward[0],
        0.0,
        side[1],
        up[1],
        -forward[1],
        0.0,
        side[2],
        up[2],
        -forward[2],
        0.0,
        -dot(side, eye),
        -dot(up, eye),
        dot(forward, eye),
        1.0,
    ];
    let (near, far, f) = (0.1f32, 100.0f32, 2.0f32);
    sun.frame.projection = [
        f,
        0.0,
        0.0,
        0.0,
        0.0,
        f,
        0.0,
        0.0,
        0.0,
        0.0,
        far / (near - far),
        -1.0,
        0.0,
        0.0,
        near * far / (near - far),
        0.0,
    ];
    sun.frame.shadow.contact = 24;
    sun.frame.shadow.contact_length = 2.5;
    let mut scene = quad([[-8.0, 0.0, -8.0], [-8.0, 0.0, 8.0], [8.0, 0.0, 8.0], [8.0, 0.0, -8.0]]);
    scene.extend(block(0.3, 1.0, 0.3));
    sun.see(&rig, 512, &scene);
    let probes = [
        ([0.7f32, 0.0, 0.0], true),
        ([1.1, 0.0, 0.0], true),
        ([1.5, 0.0, 0.0], true),
        // Toward the sun from the post, past the end of its shadow, and to the side: open.
        ([-1.5, 0.0, 0.0], false),
        ([3.0, 0.0, 0.0], false),
        ([1.1, 0.0, 2.0], false),
    ];
    let points: Vec<_> = probes.iter().map(|(at, _)| (*at, [0.0f32, 1.0, 0.0], [1.0f32, 0.0])).collect();
    let seen = sun.read(&rig, &points);
    for ((at, shadowed), got) in probes.iter().zip(&seen) {
        if *shadowed {
            assert!(got[1] < 0.75, "{at:?} is behind the post: contact shadow {}", got[1]);
        } else {
            assert_eq!(got[1], 1.0, "{at:?} is in the open: contact shadow {}", got[1]);
        }
    }
    // With no steps, nothing is ever shadowed.
    sun.frame.shadow.contact = 0;
    assert!(sun.read(&rig, &points).iter().all(|pair| pair[1] == 1.0));
    // An orthographic camera has no depth to march in: also nothing.
    sun.frame.shadow.contact = 24;
    sun.frame.projection[11] = 0.0;
    assert!(sun.read(&rig, &points).iter().all(|pair| pair[1] == 1.0));
}

// ------------------------------------------------------------ local lights

fn local_kernel() -> String {
    format!(
        "{LIGHTS_WGSL}
@group(3) @binding(0) var<storage, read> points: array<vec4f>;
@group(3) @binding(1) var<storage, read_write> answers: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {{
  if (id.x >= arrayLength(&answers)) {{ return; }}
  let light = cluster_lights[0];
  let world = points[id.x * 2u].xyz;
  let normal = points[id.x * 2u + 1u].xyz;
  let toward = light.position.xyz - world;
  let distance = length(toward);
  answers[id.x] = local_shadow(light, world, normal, toward / distance, distance) + 0.0 * f32(cluster_ranges[0].x + cluster_indices[0]) + 0.0 * cluster_grid.scale.x;
}}"
    )
}

/// A light at `at` shining where `direction` says (a point light when `cone` is -1), its shadow maps
/// drawn from `casters`, and the shaders' answers at `points` (world, normal).
fn local_answers(
    rig: &Rig,
    light: Light,
    texels: u32,
    casters: &[[f32; 3]],
    points: &[([f32; 3], [f32; 3])],
) -> (Vec<f32>, local_shadows::Selection) {
    let view = IDENTITY;
    let mut projection = [0.0f32; 16];
    projection[0] = 1.0;
    projection[5] = 1.0;
    projection[10] = -1.0001;
    projection[11] = -1.0;
    projection[14] = -0.10001;
    let lights = [light];
    let mut selection = local_shadows::select(&lights, &view, &projection, 4, texels);
    // The camera sees the light only if it is in front: tests put it there or move the camera by `view`.
    local_shadows::assign(&mut selection, &[None; local_shadows::MAX_FACES]);
    assert_eq!(selection.casters.len(), 1, "the light is picked");
    let maps = rig.maps(texels, local_shadows::MAX_FACES as u32);
    let caster = selection.casters[0];
    for index in 0..caster.faces() {
        rig.draw(
            &maps,
            u32::try_from(caster.layer + index).unwrap(),
            &selection.faces[caster.first + index].matrix,
            casters,
        );
    }
    let sampler = rig.device.create_sampler(&wgpu::SamplerDescriptor {
        compare: Some(wgpu::CompareFunction::Less),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let all = maps.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let mut faces = vec![0.0f32; local_shadows::MAX_FACES * 16];
    for index in 0..caster.faces() {
        let at = (caster.layer + index) * 16;
        faces[at..at + 16].copy_from_slice(&selection.faces[caster.first + index].matrix);
    }
    let faces = rig.storage(&faces);
    let (p, c, d) = (light.position, light.color, light.direction);
    let record = [
        p.x,
        p.y,
        p.z,
        light.radius,
        c.x,
        c.y,
        c.z,
        light.outer,
        d.x,
        d.y,
        d.z,
        light.inner,
        caster.layer as f32,
        if caster.cube { 1.0 } else { 0.0 },
        caster.texel,
        0.0,
    ];
    let lights_buffer = rig.storage(&record);
    let ranges = rig.storage(&[0.0, 0.0]);
    let indices = rig.storage(&[0.0]);
    let grid = rig.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[0u32; 12]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut lanes = Vec::new();
    for (world, normal) in points {
        lanes.extend([world[0], world[1], world[2], 0.0, normal[0], normal[1], normal[2], 0.0]);
    }
    let input = rig.storage(&lanes);
    let output = rig.storage(&vec![0.0; points.len()]);
    let values = rig.compute(
        &local_kernel(),
        &[
            (
                2,
                vec![
                    (0, grid.as_entire_binding()),
                    (1, lights_buffer.as_entire_binding()),
                    (2, ranges.as_entire_binding()),
                    (3, indices.as_entire_binding()),
                    (4, wgpu::BindingResource::TextureView(&all)),
                    (5, wgpu::BindingResource::Sampler(&sampler)),
                    (6, faces.as_entire_binding()),
                ],
            ),
            (3, vec![(0, input.as_entire_binding()), (1, output.as_entire_binding())]),
        ],
        &output,
        points.len(),
    );
    (values, selection)
}

fn light_at(at: [f32; 3], direction: [f32; 3], cone: f32) -> Light {
    Light {
        position: Vector { x: at[0], y: at[1], z: at[2] },
        radius: 40.0,
        color: Vector { x: 1.0, y: 1.0, z: 1.0 },
        direction: Vector { x: direction[0], y: direction[1], z: direction[2] },
        inner: if cone > 0.0 { cone + 0.05 } else { -1.0 },
        outer: cone,
        shadow: true,
        id: 1,
    }
}

#[test]
fn a_spot_s_shadow_lands_where_the_projection_says() {
    let Some(rig) = Rig::new() else { return };
    // A spot 10 above the floor, shining straight down, and a square plate
    // 2 across at height 4 right below it; the plate's shadow on the floor
    // is its projection from the light: the square scaled by 10 / 6 about
    // the point under the light.
    let plate = quad([[-1.0, 4.0, -21.0], [-1.0, 4.0, -19.0], [1.0, 4.0, -19.0], [1.0, 4.0, -21.0]]);
    let light = light_at([0.0, 10.0, -20.0], [0.0, -1.0, 0.0], 0.7);
    // The camera looks down -z from the origin and sees the light at z = -20.
    let scale = 10.0 / 6.0;
    let half = 1.0 * scale;
    let mut points = Vec::new();
    let mut want = Vec::new();
    for i in -12..=12 {
        for j in -12..=12 {
            let (x, z) = (f64::from(i) * 0.25, f64::from(j) * 0.25);
            // Skip the penumbra's neighbourhood: a few filter texels either side of the shadow's edge.
            let edge = (x.abs().max(z.abs()) - half).abs();
            if edge < 0.2 {
                continue;
            }
            points.push(([x as f32, 0.0, z as f32 - 20.0], [0.0f32, 1.0, 0.0]));
            want.push(if x.abs() < half && z.abs() < half { 0.0 } else { 1.0 });
        }
    }
    let (seen, selection) = local_answers(&rig, light, 1024, &plate, &points);
    assert!(!selection.casters[0].cube);
    for (index, (&got, &want)) in seen.iter().zip(&want).enumerate() {
        assert!((got - want).abs() < 0.05, "{:?}: the light reads {got}, the projection says {want}", points[index].0);
    }
}

#[test]
fn a_point_light_s_cube_has_no_seams() {
    let Some(rig) = Rig::new() else { return };
    // Occluders, small plates facing a point light at the origin, in the 26
    // directions of a cube's faces, edges and corners, 2 away. Receivers, 5
    // away along the same directions, are in their shadows; receivers turned
    // aside by more than the plate's shadow are not.
    let mut directions = Vec::new();
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                if (x, y, z) != (0, 0, 0) {
                    directions.push([f64::from(x), f64::from(y), f64::from(z)]);
                }
            }
        }
    }
    let unit = |d: [f64; 3]| {
        let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        d.map(|value| value / length)
    };
    let mut plates = Vec::new();
    for d in &directions {
        let d = unit(*d);
        // A plate square to the direction, 0.6 across, at distance 2.
        let helper = if d[1].abs() > 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        };
        let a = unit(cross(d, helper));
        let b = cross(d, a);
        let at = |s: f64, t: f64| -> [f32; 3] {
            std::array::from_fn(|axis| (d[axis] * 2.0 + a[axis] * s + b[axis] * t) as f32)
        };
        plates.extend(quad([at(-0.3, -0.3), at(0.3, -0.3), at(0.3, 0.3), at(-0.3, 0.3)]));
    }
    let light = light_at([0.0, 0.0, 0.0], [0.0, -1.0, 0.0], -1.0);
    let mut points = Vec::new();
    let mut want = Vec::new();
    for d in &directions {
        let d = unit(*d);
        let helper = if d[1].abs() > 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        };
        let a = unit(cross(d, helper));
        // Behind the plate: along the direction; beside it, 5 away and 0.5 off axis (the plate's shadow
        // there spans 0.3 * 5 / 2 = 0.75: so 0.5 off is inside it, 1.5 off outside).
        for (off, dark) in [(0.0, 0.0), (0.45, 0.0), (1.5, 1.0)] {
            let at: [f32; 3] = std::array::from_fn(|axis| (d[axis] * 5.0 + a[axis] * off) as f32);
            // A surface facing the light.
            let length = (f64::from(at[0]).powi(2) + f64::from(at[1]).powi(2) + f64::from(at[2]).powi(2)).sqrt();
            let normal: [f32; 3] = at.map(|value| -(f64::from(value) / length) as f32);
            points.push((at, normal));
            want.push(dark);
        }
    }
    // The camera sees the light: look from behind it, with the light ahead.
    let (seen, selection) = {
        let mut moved = light;
        moved.position = Vector { x: 0.0, y: 0.0, z: -3.0 };
        let moved_points: Vec<_> = points.iter().map(|(at, normal)| ([at[0], at[1], at[2] - 3.0], *normal)).collect();
        let plates: Vec<[f32; 3]> = plates.iter().map(|p| [p[0], p[1], p[2] - 3.0]).collect();
        local_answers(&rig, moved, 512, &plates, &moved_points)
    };
    assert!(selection.casters[0].cube && selection.faces.len() == 6);
    for (index, (&got, &want)) in seen.iter().zip(&want).enumerate() {
        let direction = directions[index / 3];
        assert!(
            (got - want).abs() < 0.1,
            "direction {direction:?}, probe {}: the light reads {got}, wanted {want}",
            index % 3
        );
    }
}
