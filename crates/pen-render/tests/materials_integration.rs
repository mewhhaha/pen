//! The material stack as a game meets it, on a real (headless) GPU: the standard surface's
//! glass from a material's JSON (`surface` block: transmission, ior, thickness), the opaque
//! picture it reads, and the mist, through the whole renderer. Held to f64 references:
//!
//! - the stripes of a wall seen through a slab of glass at a known angle move by what Snell's
//!   law says and the library's sampling rule (the ray refracted at the surface, `thickness`
//!   along, projected to the screen: DESIGN.md section 3) puts the picture's reading at; an
//!   index of 1 and a thin wall move nothing, and a rougher glass reads a blurrier level;
//! - glass in a mist shows the mist's opaque picture refracted, times what the glass lets
//!   through, then misted at the glass's own depth (`fog_apply`), as the closed form of the
//!   mist composes it;
//! - a material with `surface.transmission` is drawn among the translucent surfaces, casts no
//!   shadow and does not share an instanced batch, and the pipeline layouts fit a device that
//!   allows 15 sampled textures to a stage.
//!
//! The pictures are drawn in HDR (extended sRGB, half-float output), where the present pass
//! shows the scene's colours as they are below 0.9. Pictures go to /tmp/fidelity/materials.
//! Skips when no adapter is available.
#![cfg(not(target_arch = "wasm32"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "test geometry"
)]
#![allow(clippy::too_many_lines, clippy::many_single_char_names, reason = "test scenes")]
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

mod common;

use common::*;
use pen_protocol::*;
use pen_render::{Display, RenderTarget, Renderer, RendererOptions};

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, k: f64) -> V3 {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn unit(a: V3) -> V3 {
    scale(a, 1.0 / dot(a, a).sqrt())
}

/// GLSL's `refract`: the unit direction `incident` bent at a surface of unit normal `normal` (facing
/// against it) by the ratio of indices `eta`, or the zero vector at total internal reflection.
fn refract(incident: V3, normal: V3, eta: f64) -> V3 {
    let cosine = dot(normal, incident);
    let k = 1.0 - eta * eta * (1.0 - cosine * cosine);
    if k < 0.0 {
        return [0.0; 3];
    }
    sub(scale(incident, eta), scale(normal, eta * cosine + k.sqrt()))
}

/// The picture's size in pixels (a square) and the camera: looking down -z from `eye`, whose
/// field of view has `TAN` for the tangent of its half angle.
const SIZE: u32 = 256;
const TAN: f64 = 0.5;

fn projection() -> Matrix {
    let (near, far, f) = (0.1f64, 400.0f64, 1.0 / TAN);
    [
        f as f32,
        0.0,
        0.0,
        0.0,
        0.0,
        f as f32,
        0.0,
        0.0,
        0.0,
        0.0,
        (far / (near - far)) as f32,
        -1.0,
        0.0,
        0.0,
        (near * far / (near - far)) as f32,
        0.0,
    ]
}

/// The camera at `eye` looking along -z.
fn view(eye: V3) -> Matrix {
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -eye[0] as f32, -eye[1] as f32, -eye[2] as f32, 1.0]
}

/// The unit direction through the point (`x`, `y`) of the picture, in pixels (a pixel's
/// middle is half a pixel in).
fn ray(x: f64, y: f64) -> V3 {
    let s = f64::from(SIZE);
    unit([(x / s * 2.0 - 1.0) * TAN, (1.0 - y / s * 2.0) * TAN, -1.0])
}

/// Where a point relative to the eye lies in the picture, in pixels.
fn project(relative: V3) -> (f64, f64) {
    let s = f64::from(SIZE);
    let (x, y) = (relative[0] / -relative[2], relative[1] / -relative[2]);
    ((x / TAN + 1.0) / 2.0 * s, (1.0 - y / TAN) / 2.0 * s)
}

fn half(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f64::from(bits & 0x3ff);
    sign * if exponent == 0 {
        fraction / 1024.0 * 2f64.powi(-14)
    } else {
        (1.0 + fraction / 1024.0) * 2f64.powi(exponent - 15)
    }
}

struct Hdr(wgpu::Texture);

impl RenderTarget for Hdr {
    fn size(&self) -> (u32, u32) {
        (self.0.width(), self.0.height())
    }
    fn view(&mut self) -> Option<wgpu::TextureView> {
        Some(self.0.create_view(&Default::default()))
    }
    fn present(&mut self) {}
}

/// What a frame drew: each pixel's red, green and blue as half-floats.
#[derive(Clone, PartialEq, Debug)]
struct Picture {
    size: u32,
    bits: Vec<[u16; 4]>,
}

impl Picture {
    fn at(&self, x: u32, y: u32) -> V3 {
        let [r, g, b, _] = self.bits[(y * self.size + x) as usize];
        [half(r), half(g), half(b)]
    }

    fn row(&self, y: u32) -> Vec<V3> {
        (0..self.size).map(|x| self.at(x, y)).collect()
    }

    fn save(&self, name: &str) {
        let directory = PathBuf::from("/tmp/fidelity/materials");
        std::fs::create_dir_all(&directory).unwrap();
        let image = image::RgbImage::from_fn(self.size, self.size, |x, y| {
            image::Rgb(self.at(x, y).map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8))
        });
        image.save(directory.join(name)).unwrap();
    }
}

/// The meshes, shaders and materials (JSON) of a test, over a renderer drawing into an
/// HDR target.
struct Rig {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    catalog: Catalog,
    target: Hdr,
    size: u32,
}

impl Rig {
    fn new(name: &str, size: u32, files: &[(&str, String)], materials: &[String]) -> Option<Rig> {
        Self::with_limit(name, size, files, materials, None)
    }

    /// The same on a device that allows `sampled` sampled textures to a shader stage, if given.
    fn with_limit(
        name: &str,
        size: u32,
        files: &[(&str, String)],
        materials: &[String],
        sampled: Option<u32>,
    ) -> Option<Rig> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        let mut required_limits = wgpu::Limits::default();
        if let Some(sampled) = sampled {
            required_limits.max_sampled_textures_per_shader_stage = sampled;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: pen_render::optional_features(&adapter),
            required_limits,
            ..Default::default()
        }))
        .ok()?;
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("integration-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let mut meshes = Vec::new();
        for (key, text) in files {
            std::fs::write(root.join(key), text).unwrap();
            if key.ends_with(".mesh.json") {
                meshes.push((*key).to_owned());
            }
        }
        let mut keys = Vec::new();
        for (index, json) in materials.iter().enumerate() {
            let key = format!("m{index}.material.json");
            std::fs::write(root.join(&key), json).unwrap();
            keys.push(key);
        }
        let catalog = Catalog { sounds: vec![], meshes, materials: keys };
        let display = Display { color_space: wgpu::SurfaceColorSpace::ExtendedSrgb, headroom: 4.0, white_nits: 203.0 };
        let options = RendererOptions {
            format: wgpu::TextureFormat::Rgba16Float,
            display,
            assets: root,
            veil: None,
            pipeline_cache: None,
        };
        let renderer = Renderer::new(device.clone(), queue.clone(), options).unwrap();
        let target = Hdr(device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        Some(Rig { device, queue, renderer, catalog, target, size })
    }

    /// Submits `frame` until everything it names is loaded and compiled (with the pipelines of its
    /// own sample count), then once more, and reads it back.
    fn draw(&mut self, frame: &RenderFrame) -> Picture {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            self.renderer.submit(frame, &self.catalog, &mut self.target).unwrap();
            for event in self.renderer.take_events() {
                assert!(!matches!(event, GameEvent::AssetFailed { .. }), "an asset failed: {event:?}");
            }
            if self.renderer.pending() == 0
                && !self.renderer.waiting()
                && (frame.draws.is_empty() || self.renderer.stats().draw_calls > 0)
            {
                break;
            }
            assert!(Instant::now() < deadline, "assets did not load");
            std::thread::sleep(Duration::from_millis(5));
        }
        self.renderer.submit(frame, &self.catalog, &mut self.target).unwrap();
        self.read()
    }

    fn read(&self) -> Picture {
        let (texture, size) = (&self.target.0, self.size);
        let row = (size * 8).next_multiple_of(256);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row * size),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(size) },
            },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        );
        self.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mapped = buffer.slice(..).get_mapped_range().unwrap();
        let mut bits = Vec::with_capacity((size * size) as usize);
        for y in 0..size as usize {
            for x in 0..size as usize {
                let at = y * row as usize + x * 8;
                bits.push(std::array::from_fn(|i| u16::from_le_bytes([mapped[at + 2 * i], mapped[at + 2 * i + 1]])));
            }
        }
        Picture { size, bits }
    }
}

const HEAD: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
"#;

/// An unlit wall of its material's colour in vertical stripes in world x: `parameters` is (the stripes'
/// width in world units, 0 for none; the dark stripes' share of the colour; where the stripes start
/// in world units).
const WALL: &str = r#"
struct Corner { @builtin(position) position: vec4f, @location(0) world: vec3f };
@vertex fn vertex(@location(0) position: vec3f) -> Corner {
  let world = draw.model * vec4f(position, 1.0);
  return Corner(scene.projection * scene.view * world, world.xyz);
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  var dark = false;
  if (draw.parameters.x > 0.0) {
    dark = (i32(floor((corner.world.x + draw.parameters.z) / draw.parameters.x)) & 1) == 1;
  }
  return vec4f(draw.color.rgb * select(1.0, draw.parameters.y, dark), 1.0);
}
"#;

/// Glass as a game writes it: the standard surface's transmission lobe, its parameters from the
/// material's `surface` block as the host writes them into the draw uniform, lit by nothing but
/// what shows through (the frame has no light) and misted at its own depth. `parameters.x` is its
/// roughness.
const GLASS: &str = r#"
struct Scene {
  view: mat4x4f, projection: mat4x4f, ambient: vec4f, light_direction: vec4f, light_color: vec4f,
  light_view_projection: mat4x4f, shadow: vec4f, zenith: vec4f, horizon: vec4f, ground: vec4f, eye: vec4f,
};
struct Draw {
  model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f,
  extra: array<vec4f, 32>, surface: array<vec4f, 5>,
};
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
${lights}
${environment}
${surface}
${translucency}
${transmission}
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
};
@vertex fn vertex(@location(0) position: vec3f, @location(1) normal: vec3f) -> Corner {
  let world = draw.model * vec4f(position, 1.0);
  return Corner(scene.projection * scene.view * world, world.xyz, (draw.normal * vec4f(normal, 0.0)).xyz);
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let normal = normalize(corner.normal);
  let view = normalize(scene.eye.xyz - corner.world);
  let seen = scene.view * vec4f(corner.world, 1.0);
  var s = surface_default(corner.world, normal, view, draw.color.rgb, 0.0, draw.parameters.x, corner.position, seen.z);
  s = surface_apply_block(s, draw.surface, SURFACE_TRANSMISSION, length(draw.model[0].xyz));
  s = surface_transmit(s);
  return vec4f(fog_apply(standard_surface(s), corner.world), draw.color.a);
}
"#;

fn glass_shader() -> String {
    GLASS
        .replace("${lights}", pen_render::LIGHTS_WGSL)
        .replace("${environment}", pen_render::ENVIRONMENT_WGSL)
        .replace("${surface}", &format!("{}\n{}", pen_render::SURFACE_LOBES_WGSL, pen_render::SURFACE_WGSL))
        .replace("${translucency}", pen_render::TRANSLUCENCY_WGSL)
        .replace("${transmission}", pen_render::SURFACE_TRANSMISSION_WGSL)
}

const QUAD: &str = r#"{"positions":[-1,-1,0, 1,-1,0, -1,1,0, 1,1,0],"normals":[0,0,1,0,0,1,0,0,1,0,0,1],
                       "uvs":[0,0,1,0,0,1,1,1],"indices":[0,1,2,2,1,3]}"#;

/// A unit sphere's mesh, its normals its positions.
fn sphere(rings: usize, segments: usize) -> String {
    let (mut positions, mut uvs, mut indices) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..=rings {
        let polar = std::f64::consts::PI * r as f64 / rings as f64;
        for s in 0..=segments {
            let azimuth = std::f64::consts::TAU * s as f64 / segments as f64;
            positions.extend([polar.sin() * azimuth.cos(), polar.cos(), polar.sin() * azimuth.sin()]);
            uvs.extend([s as f64 / segments as f64, r as f64 / rings as f64]);
        }
    }
    for r in 0..rings {
        for s in 0..segments {
            let a = (r * (segments + 1) + s) as u32;
            let b = a + segments as u32 + 1;
            indices.extend([a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    let list = |v: &[f64]| v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(",");
    format!(
        r#"{{"positions":[{0}],"normals":[{0}],"uvs":[{1}],"indices":{2:?}}}"#,
        list(&positions),
        list(&uvs),
        indices
    )
}

/// A box from (-1, 0, -1) to (1, 1, 1), per-face normals.
fn pillar() -> String {
    let (x, z, y0, y1) = (1.0f32, 1.0f32, 0.0f32, 1.0f32);
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 1.0, 0.0], [[-x, y1, -z], [-x, y1, z], [x, y1, z], [x, y1, -z]]),
        ([0.0, -1.0, 0.0], [[-x, y0, z], [-x, y0, -z], [x, y0, -z], [x, y0, z]]),
        ([1.0, 0.0, 0.0], [[x, y0, -z], [x, y1, -z], [x, y1, z], [x, y0, z]]),
        ([-1.0, 0.0, 0.0], [[-x, y0, z], [-x, y1, z], [-x, y1, -z], [-x, y0, -z]]),
        ([0.0, 0.0, 1.0], [[x, y0, z], [x, y1, z], [-x, y1, z], [-x, y0, z]]),
        ([0.0, 0.0, -1.0], [[-x, y0, -z], [-x, y1, -z], [x, y1, -z], [x, y0, -z]]),
    ];
    let (mut positions, mut normals, mut uvs, mut indices) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (index, (normal, corners)) in faces.iter().enumerate() {
        for corner in corners {
            positions.extend(corner.iter().map(|value| value.to_string()));
            normals.extend(normal.iter().map(|value| value.to_string()));
            uvs.extend(["0".to_owned(), "0".to_owned()]);
        }
        let base = index * 4;
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3].map(|value| value.to_string()));
    }
    format!(
        r#"{{"positions":[{}],"normals":[{}],"uvs":[{}],"indices":[{}]}}"#,
        positions.join(","),
        normals.join(","),
        uvs.join(","),
        indices.join(",")
    )
}

/// A model matrix: scales, turns about y by `turn` radians, then places.
fn model(scales: V3, turn: f64, at: V3) -> Matrix {
    let (s, c) = turn.sin_cos();
    #[rustfmt::skip]
    let matrix = [
        (scales[0] * c) as f32, 0.0, (-scales[0] * s) as f32, 0.0,
        0.0, scales[1] as f32, 0.0, 0.0,
        (scales[2] * s) as f32, 0.0, (scales[2] * c) as f32, 0.0,
        at[0] as f32, at[1] as f32, at[2] as f32, 1.0,
    ];
    matrix
}

fn wall_material(color: V3) -> String {
    format!(
        r#"{{"shader":"wall.wgsl","color":[{},{},{},1],"lighting":"unlit","instanceParameters":[0,0,0,0]}}"#,
        color[0], color[1], color[2]
    )
}

/// Glass as a material's JSON says it: its `surface` block (the keys given, after `transmission` 1).
fn glass_material(surface: &str) -> String {
    format!(r#"{{"shader":"glass.wgsl","color":[1,1,1,1],"lighting":"lit","surface":{{"transmission":1{surface}}}}}"#)
}

fn draw(mesh: u32, material: u32, transform: Matrix, parameters: Option<[f32; 4]>) -> Draw {
    Draw { mesh, material, transform, parameters, extra: None }
}

fn frame_at(eye: V3, draws: Vec<Draw>) -> RenderFrame {
    let mut frame = frame(draws);
    frame.view = view(eye);
    frame.projection = projection();
    frame.light_color = Vector { x: 0.0, y: 0.0, z: 0.0 };
    frame
}

// ---- a) Refraction.

const WALL_DEPTH: f64 = 4.0;
const GLASS_DEPTH: f64 = 1.5;
const GLASS_SCALE: f64 = 1.5;
/// The stripes: 0.5 across, from -0.1, 0.8 and 0.1 in colour.
const STRIPE: f64 = 0.5;
const PHASE: f64 = 0.17;

/// A slab of glass `GLASS_DEPTH` ahead of the eye turned `turn` radians about y, of this
/// index and thickness (in the mesh's units: the draw scales it by `GLASS_SCALE`).
struct Slab {
    turn: f64,
    ior: f64,
    thickness: f64,
}

impl Slab {
    fn normal(&self) -> V3 {
        [self.turn.sin(), 0.0, self.turn.cos()]
    }

    /// How far (in pixels along x) the picture the glass shows at the point (`x`, `y`) of the
    /// picture lies from it: the view's ray refracted into the slab at the surface by `1 / ior`,
    /// `thickness` along (in the world's units) from where it meets it, the exit projected.
    fn shift(&self, x: f64, y: f64) -> f64 {
        let eye = [0.0; 3];
        let centre = [0.0, 0.0, -GLASS_DEPTH];
        let d = ray(x, y);
        let n = self.normal();
        let t = dot(n, sub(centre, eye)) / dot(n, d);
        let point = scale(d, t);
        let refracted = refract(d, n, 1.0 / self.ior);
        let exit = add(point, scale(refracted, self.thickness * GLASS_SCALE));
        project(exit).0 - x
    }
}

/// The pixel boundaries (in pixels, between two pixels) at which a row of the picture steps by more
/// than half of its range.
fn boundaries(row: &[V3]) -> Vec<f64> {
    let (low, high) = row.iter().fold((f64::MAX, f64::MIN), |(l, h), p| (l.min(p[1]), h.max(p[1])));
    (0..row.len() - 1)
        .filter(|&i| (row[i + 1][1] - row[i][1]).abs() > 0.5 * (high - low))
        .map(|i| (i + 1) as f64)
        .collect()
}

/// Where, in pixels, the green of a row crosses the middle of its levels there, by linear
/// interpolation between pixels' middles.
fn crossings(row: &[V3]) -> Vec<f64> {
    let mut found = Vec::new();
    for i in 3..row.len() - 4 {
        let around = &row[i - 2..i + 4];
        let (low, high) = around.iter().fold((f64::MAX, f64::MIN), |(l, h), p| (l.min(p[1]), h.max(p[1])));
        // An edge: the levels round it differ by more than they vary otherwise.
        if high - low < 0.3 * (row.iter().map(|p| p[1]).fold(0.0, f64::max)) {
            continue;
        }
        let middle = (low + high) / 2.0;
        let (a, b) = (row[i][1], row[i + 1][1]);
        if (a - middle) * (b - middle) < 0.0 {
            found.push(i as f64 + 0.5 + (middle - a) / (b - a));
        }
    }
    found
}

/// The shift at which `x + shift(x) = boundary`.
fn solve(slab: &Slab, y: f64, boundary: f64) -> f64 {
    let (mut low, mut high) = (-40.0, f64::from(SIZE) + 40.0);
    for _ in 0..80 {
        let middle = (low + high) / 2.0;
        if middle + slab.shift(middle, y) < boundary {
            low = middle;
        } else {
            high = middle;
        }
    }
    (low + high) / 2.0
}

fn refraction_files() -> Vec<(&'static str, String)> {
    vec![("quad.mesh.json", QUAD.to_owned()), ("wall.wgsl", format!("{HEAD}{WALL}")), ("glass.wgsl", glass_shader())]
}

fn refraction_frame(slab: Option<(&Slab, u32)>, roughness: f32) -> RenderFrame {
    let wall = draw(
        0,
        0,
        model([8.0, 8.0, 1.0], 0.0, [0.0, 0.0, -WALL_DEPTH]),
        Some([STRIPE as f32, 0.125, PHASE as f32, 0.0]),
    );
    let mut draws = vec![wall];
    if let Some((slab, material)) = slab {
        let transform = model([GLASS_SCALE, GLASS_SCALE, 1.0], slab.turn, [0.0, 0.0, -GLASS_DEPTH]);
        draws.push(draw(0, material, transform, Some([roughness, 0.0, 0.0, 0.0])));
    }
    frame_at([0.0; 3], draws)
}

#[test]
fn stripes_seen_through_a_slab_of_glass_move_as_snells_law_says() {
    let slabs = [
        Slab { turn: 30f64.to_radians(), ior: 1.5, thickness: 0.2 },
        Slab { turn: (-20f64).to_radians(), ior: 1.2, thickness: 0.4 },
        Slab { turn: 40f64.to_radians(), ior: 2.0, thickness: 0.15 },
    ];
    let mut materials = vec![wall_material([0.8, 0.8, 0.8])];
    for slab in &slabs {
        materials.push(glass_material(&format!(r#","ior":{},"thickness":{}"#, slab.ior, slab.thickness)));
    }
    // An index of 1 bends nothing, and a thin wall has no thickness to bend over.
    materials.push(glass_material(r#","ior":1.0,"thickness":0.3"#));
    materials.push(glass_material(r#","ior":1.5,"thickness":0"#));
    let Some(mut rig) = Rig::new("refraction", SIZE, &refraction_files(), &materials) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let row_y = SIZE / 2 + 20;
    let y = f64::from(row_y) + 0.5;
    let wall = rig.draw(&refraction_frame(None, 0.0)).row(row_y);
    let wall_edges = boundaries(&wall);
    assert!(wall_edges.len() >= 6, "the wall's stripes: {wall_edges:?}");
    // The wall alone, where its stripes step, to the pixel; the glass's stripes against Snell's law.
    for (index, slab) in slabs.iter().enumerate() {
        let picture = rig.draw(&refraction_frame(Some((slab, index as u32 + 1)), 0.0));
        if index == 0 {
            picture.save("glass-refraction-e.png");
        }
        let row = picture.row(row_y);
        let seen = crossings(&row);
        let mut compared = 0;
        let mut worst: f64 = 0.0;
        let mut moved: f64 = 0.0;
        for &edge in &wall_edges {
            let predicted = solve(slab, y, edge);
            if !(8.0..f64::from(SIZE) - 8.0).contains(&predicted) {
                continue;
            }
            let Some(&measured) = seen.iter().min_by(|a, b| (*a - predicted).abs().total_cmp(&(*b - predicted).abs()))
            else {
                panic!("slab {index}: no edge near {predicted} in {seen:?}");
            };
            let error = (measured - predicted).abs();
            eprintln!(
                "slab {index} (turn {:.0} deg, ior {}, thickness {}): wall edge at {edge}, predicted in the glass at {predicted:.3} (shifted {:.3} px), measured {measured:.3}",
                slab.turn.to_degrees(),
                slab.ior,
                slab.thickness,
                edge - predicted
            );
            worst = worst.max(error);
            moved = moved.max((edge - predicted).abs());
            compared += 1;
        }
        assert!(compared >= 4, "slab {index}: only {compared} edges compared");
        // The edges of a sharp picture, to 0.15 of a pixel (the crossing is read off the pixels' middles); and the
        // glass did move them, to see that.
        assert!(worst < 0.15, "slab {index}: an edge is {worst} px from where Snell's law puts it");
        assert!(moved > 2.0, "slab {index}: the stripes moved only {moved} px");
    }
    // An index of 1 and a thin wall: the stripes are where the wall alone has them.
    for (name, material) in [("ior 1", slabs.len() as u32 + 1), ("thin wall", slabs.len() as u32 + 2)] {
        let slab = Slab { turn: 30f64.to_radians(), ior: 1.0, thickness: 0.3 };
        let picture = rig.draw(&refraction_frame(Some((&slab, material)), 0.0));
        let seen = crossings(&picture.row(row_y));
        for &edge in &wall_edges {
            if !(8.0..f64::from(SIZE) - 8.0).contains(&edge) {
                continue;
            }
            let measured = seen.iter().copied().min_by(|a, b| (a - edge).abs().total_cmp(&(b - edge).abs())).unwrap();
            assert!((measured - edge).abs() < 0.05, "{name}: the edge at {edge} is seen at {measured}");
        }
    }
    // A rougher glass reads a blurrier level of the picture (level 8 r of the chain): the stripes' edges
    // soften at once, their contrast (the range of the row, which survives until the blur is as wide as
    // a stripe) only falls and keeps falling, to a tenth of what it was at the roughest.
    let slab = &slabs[0];
    let roughnesses = [0.0f32, 0.1, 0.2, 0.4, 0.7, 1.0];
    let (mut contrasts, mut steepness) = (Vec::new(), Vec::new());
    for &roughness in &roughnesses {
        let row = rig.draw(&refraction_frame(Some((slab, 1)), roughness)).row(row_y);
        let middle = &row[40..216];
        let (low, high) = middle.iter().fold((f64::MAX, f64::MIN), |(l, h), p| (l.min(p[1]), h.max(p[1])));
        contrasts.push((high - low) / (high + low));
        // The steepest step between neighbouring pixels, over the row's mean.
        let step = middle.windows(2).map(|pair| (pair[1][1] - pair[0][1]).abs()).fold(0.0, f64::max);
        steepness.push(step / (middle.iter().map(|p| p[1]).sum::<f64>() / middle.len() as f64));
    }
    eprintln!("by roughness {roughnesses:?}: stripe contrast {contrasts:.4?}, steepest edge {steepness:.4?}");
    assert!(steepness.windows(2).all(|pair| pair[1] < pair[0]), "{steepness:?}");
    assert!(contrasts.windows(2).all(|pair| pair[1] <= pair[0] + 1e-9), "{contrasts:?}");
    assert!(contrasts[4] < 0.2 * contrasts[0] && contrasts[5] < 0.1 * contrasts[0], "{contrasts:?}");
}

// ---- b) Glass in a mist.

/// The exponential height mist of a homogeneous albedo of 1 lit by an ambient light of `mean`
/// alone (no sun, no lights): what the eye sees of a colour through it, in closed form.
struct Mist {
    density: f64,
    height: f64,
    mean: V3,
}

/// The optical depth of the exponential height mist (`density` at the ground, falling by e over
/// `height` above it, constant below) from `eye` along the unit `direction` for `length`.
fn height_optical_depth(density: f64, height: f64, eye: V3, direction: V3, length: f64) -> f64 {
    let end = eye[1] + direction[1] * length;
    let (low, high) = (eye[1].min(end), eye[1].max(end));
    if high - low < 1e-9 {
        return density * length * (-eye[1].max(0.0) / height).exp();
    }
    let below = (0.0 - low).clamp(0.0, high - low) / (high - low);
    let above_low = low.max(0.0);
    let above = if high > 0.0 {
        height * ((-above_low / height).exp() - (-high / height).exp()) / (high - above_low).max(1e-12)
    } else {
        0.0
    };
    let along_above = (1.0 - below) * length;
    density * (below * length + along_above * if high > 0.0 { above } else { 0.0 })
}

impl Mist {
    /// The transmittance and the in-scatter between the eye and a point `length` along `direction`.
    fn between(&self, eye: V3, direction: V3, length: f64) -> (f64, V3) {
        let transmittance = (-height_optical_depth(self.density, self.height, eye, direction, length)).exp();
        (transmittance, self.mean.map(|mean| mean * (1.0 - transmittance)))
    }

    /// `colour` as the eye sees it through the mist from `length` away.
    fn fog(&self, colour: V3, eye: V3, direction: V3, length: f64) -> V3 {
        let (transmittance, scatter) = self.between(eye, direction, length);
        std::array::from_fn(|channel| colour[channel] * transmittance + scatter[channel])
    }
}

const SH: f64 = 0.282_094_8;

/// A sky whose mean light is `mean`: its constant coefficient.
fn sky_with_mean(mean: V3) -> Sky {
    let coefficient = |value: f64| (value / SH) as f32;
    Sky {
        sh0: Vector { x: coefficient(mean[0]), y: coefficient(mean[1]), z: coefficient(mean[2]) },
        sun: Vector { x: 0.0, y: 1.0, z: 0.0 },
        ..Sky::default()
    }
}

const FOG_EYE: V3 = [4.0, 1.5, 0.0];
const FOG_WALL: V3 = [0.55, 0.3, 0.2];
const MEAN: V3 = [0.8, 0.5, 0.3];

fn fog_files() -> Vec<(&'static str, String)> {
    vec![
        ("quad.mesh.json", QUAD.to_owned()),
        ("sphere.mesh.json", sphere(48, 96)),
        ("pillar.mesh.json", pillar()),
        ("wall.wgsl", format!("{HEAD}{WALL}")),
        ("glass.wgsl", glass_shader()),
    ]
}

/// A hazy frame: the analytic mist (no froxel grid), lit by the sky's mean alone.
fn hazy(mist: Option<&Mist>, draws: Vec<Draw>) -> RenderFrame {
    let mut frame = frame_at(FOG_EYE, draws);
    match mist {
        Some(mist) => {
            frame.fog =
                Fog { density: mist.density as f32, height: mist.height as f32, distance: 90.0, ..Fog::default() };
            frame.sky = sky_with_mean(mist.mean);
        }
        None => frame.sky = sky_with_mean(MEAN),
    }
    frame.quality.volumetrics = 0;
    frame
}

#[test]
fn glass_in_a_mist_shows_the_refracted_misted_picture_misted_again_at_its_own_depth() {
    let mist = Mist { density: 0.12, height: 2000.0, mean: MEAN };
    let slab = Slab { turn: 25f64.to_radians(), ior: 1.5, thickness: 0.15 };
    let (wall_z, glass_z, scale_glass) = (-6.0, -2.0, 3.0);
    let materials = vec![
        wall_material(FOG_WALL),
        wall_material([0.0; 3]),
        glass_material(&format!(
            r#","ior":{},"thickness":{},"attenuationColor":[0.8,0.95,0.9],"attenuationDistance":1.2"#,
            slab.ior, slab.thickness
        )),
    ];
    let Some(mut rig) = Rig::new("fog-glass", SIZE, &fog_files(), &materials) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let place = |z: f64| [FOG_EYE[0], FOG_EYE[1], z];
    let scene = |wall: u32| {
        vec![
            draw(0, wall, model([30.0, 30.0, 1.0], 0.0, place(wall_z)), Some([0.0; 4])),
            draw(0, 2, model([scale_glass, scale_glass, 1.0], slab.turn, place(glass_z)), Some([0.0; 4])),
        ]
    };
    // What the glass lets through and adds, by pixel, in a frame without mist: over the wall
    // (`k` times its colour, plus `s`) and over a black one (`s`).
    let (over_wall, over_black) = (rig.draw(&hazy(None, scene(0))), rig.draw(&hazy(None, scene(1))));
    let misted = rig.draw(&hazy(Some(&mist), scene(0)));
    let normal = slab.normal();
    let centre = [0.0, 0.0, glass_z - FOG_EYE[2]];
    let thickness = slab.thickness * scale_glass;
    let mut worst: f64 = 0.0;
    let (mut unlike_own, mut unlike_picture) = (f64::MAX, f64::MAX);
    let mut compared = 0;
    for y in (70..190).step_by(7) {
        for x in (70..190).step_by(7) {
            let d = ray(f64::from(x) + 0.5, f64::from(y) + 0.5);
            let t = dot(normal, centre) / dot(normal, d);
            // The glass's own mist, to where it is.
            let (transmittance, in_scatter) = mist.between(FOG_EYE, d, t);
            // The picture behind it, where the ray leaves the slab: the wall seen from the eye there, misted.
            let exit = add(scale(d, t), scale(refract(d, normal, 1.0 / slab.ior), thickness));
            let (px, py) = project(exit);
            let behind = ray(px, py);
            let wall_length = (wall_z - FOG_EYE[2]) / behind[2];
            let wall = mist.fog(FOG_WALL, FOG_EYE, behind, wall_length);
            let (plain, black) = (over_wall.at(x, y), over_black.at(x, y));
            let expected: V3 = std::array::from_fn(|c| {
                let through = (plain[c] - black[c]) / FOG_WALL[c];
                transmittance * (through * wall[c] + black[c]) + in_scatter[c]
            });
            let got = misted.at(x, y);
            let error = (0..3).map(|c| (got[c] - expected[c]).abs()).fold(0.0, f64::max);
            // What it would be if the glass took no mist of its own, or read the wall unmisted.
            let through: V3 = std::array::from_fn(|c| (plain[c] - black[c]) / FOG_WALL[c]);
            let without_own: V3 = std::array::from_fn(|c| through[c] * wall[c] + black[c]);
            let reading_unmisted: V3 =
                std::array::from_fn(|c| transmittance * (through[c] * FOG_WALL[c] + black[c]) + in_scatter[c]);
            let unlike = |other: V3| (0..3).map(|c| (expected[c] - other[c]).abs()).fold(0.0, f64::max);
            worst = worst.max(error);
            unlike_own = unlike_own.min(unlike(without_own));
            unlike_picture = unlike_picture.min(unlike(reading_unmisted));
            compared += 1;
            assert!(error < 0.006, "pixel {x},{y}: the picture has {got:.4?}, the composition {expected:.4?}");
        }
    }
    eprintln!(
        "glass in mist: {compared} pixels, worst difference {worst:.5} ({:.2} of 255); the composition is never nearer than {unlike_own:.3} to one with no mist of the glass's own and {unlike_picture:.3} to one reading the unmisted wall",
        worst * 255.0
    );
    assert!(unlike_own > 0.03 && unlike_picture > 0.03, "the mist does not tell the compositions apart");
}

#[test]
fn a_glass_sphere_in_a_mist_before_a_wall_and_pillars() {
    let mist = Mist { density: 0.075, height: 2000.0, mean: MEAN };
    let materials = vec![
        wall_material([0.62, 0.55, 0.5]),
        wall_material([0.72, 0.5, 0.34]),
        wall_material([0.3, 0.32, 0.36]),
        glass_material(r#","ior":1.5,"thickness":1.0,"attenuationColor":[0.8,1.0,0.9],"attenuationDistance":3.0"#),
    ];
    let Some(mut rig) = Rig::new("fog-sphere", 384, &fog_files(), &materials) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let at = |x: f64, y: f64, z: f64| [x, y, z];
    let mut draws = vec![
        // The wall in broad stripes, far off, and a floor a hand thick.
        draw(0, 0, model([40.0, 40.0, 1.0], 0.0, at(4.0, 1.5, -16.0)), Some([1.4, 0.55, 0.3, 0.0])),
        draw(2, 2, model([40.0, 0.1, 40.0], 0.0, at(4.0, -0.1, -20.0)), None),
    ];
    for (x, z) in [(0.6, -8.5), (7.2, -7.5), (2.4, -12.0), (5.6, -13.0), (9.5, -11.0), (-1.5, -10.0)] {
        draws.push(draw(2, 1, model([0.4, 4.2, 0.4], 0.0, at(x, 0.0, z)), None));
    }
    // The glass: a ball of it, a ball's radius of thickness.
    draws.push(draw(1, 3, model([0.75, 0.75, 0.75], 0.0, at(4.1, 1.35, -3.9)), Some([0.0; 4])));
    let picture = rig.draw(&hazy(Some(&mist), draws.clone()));
    picture.save("glass-in-fog-e.png");
    // Without the glass for the eye to compare, and without the mist.
    let mut bare = draws.clone();
    bare.pop();
    let without = rig.draw(&hazy(Some(&mist), bare));
    let clear = rig.draw(&hazy(None, draws));
    // The ball's disc (centre and radius in pixels, from where the ball is) shows the stripes and pillars
    // behind it refracted, so it is not what the picture shows there without it, and it is misted: it is
    // not what the glass shows in a frame without the mist, by much more.
    let (cx, cy, radius) = (192.0, 206.8, 55.0);
    let (mut refracted, mut misted): (f64, f64) = (0.0, 0.0);
    let mut inside = 0;
    for y in 0..384u32 {
        for x in 0..384u32 {
            if (f64::from(x) + 0.5 - cx).hypot(f64::from(y) + 0.5 - cy) > radius {
                continue;
            }
            let differ = |a: V3, b: V3| (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0, f64::max);
            refracted = refracted.max(differ(picture.at(x, y), without.at(x, y)));
            misted += differ(picture.at(x, y), clear.at(x, y));
            inside += 1;
        }
    }
    let misted = misted / f64::from(inside);
    eprintln!(
        "sphere's disc: most it differs from the picture without it {refracted:.3}, mean difference from the glass without mist {misted:.3}"
    );
    assert!(refracted > 0.05, "the glass shows nothing of its own: {refracted}");
    assert!(misted > 0.04, "the glass is not misted: {misted}");
}

// ---- c) A transmission material from the JSON.

/// An unlit, flat surface that can be instanced, in its material's colour.
const FLAT: &str = r#"
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
@vertex fn vertex_instanced(@location(0) position: vec3f, @location(3) x: vec4f, @location(4) y: vec4f,
    @location(5) z: vec4f, @location(6) w: vec4f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * mat4x4f(x, y, z, w) * vec4f(position, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return draw.color; }
"#;

fn flat_files() -> Vec<(&'static str, String)> {
    vec![("quad.mesh.json", QUAD.to_owned()), ("flat.wgsl", format!("{HEAD}{FLAT}"))]
}

#[test]
fn a_transmission_material_is_translucent_and_does_not_share_an_instanced_batch() {
    let plain = r#"{"shader":"flat.wgsl","color":[0.3,0.5,0.9,1],"lighting":"unlit"}"#.to_owned();
    let glass =
        r#"{"shader":"flat.wgsl","color":[0.3,0.5,0.9,1],"lighting":"unlit","surface":{"transmission":1}}"#.to_owned();
    // Alpha 1 and no `blend`: only the block says it is glass. A block of zeros does not.
    let coated = r#"{"shader":"flat.wgsl","color":[0.3,0.5,0.9,1],"lighting":"unlit","surface":{"clearcoat":1,"transmission":0}}"#.to_owned();
    let Some(mut rig) = Rig::new("instancing", 64, &flat_files(), &[plain, glass, coated]) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let quads = |material: u32| -> RenderFrame {
        let draws = (0..3)
            .map(|i| draw(0, material, model([0.1, 0.1, 1.0], 0.0, [f64::from(i) * 0.3 - 0.3, 0.0, -3.0]), None))
            .collect();
        frame_at([0.0; 3], draws)
    };
    // Three opaque draws of one mesh and material are one instanced draw call.
    rig.draw(&quads(0));
    let opaque = rig.renderer.stats();
    let opaque_passes = rig.renderer.passes();
    // A block that leaves transmission at 0 changes nothing about it.
    rig.draw(&quads(2));
    let coated = rig.renderer.stats();
    // Three draws of the transmissive one are three, drawn among the translucent surfaces.
    rig.draw(&quads(1));
    let transmissive = rig.renderer.stats();
    let transmissive_passes = rig.renderer.passes();
    eprintln!(
        "opaque: {} calls, {} instanced; clear coat only: {} calls, {} instanced; transmission: {} calls, {} instanced; passes {opaque_passes:?} / {transmissive_passes:?}",
        opaque.draw_calls,
        opaque.instanced_draw_calls,
        coated.draw_calls,
        coated.instanced_draw_calls,
        transmissive.draw_calls,
        transmissive.instanced_draw_calls
    );
    assert_eq!((opaque.draw_calls, opaque.instanced_draw_calls), (1, 1), "the opaque material batches");
    assert_eq!((coated.draw_calls, coated.instanced_draw_calls), (1, 1), "a block with no transmission is opaque");
    assert_eq!((transmissive.draw_calls, transmissive.instanced_draw_calls), (3, 0), "glass is not instanced");
    assert!(!opaque_passes.contains(&"backdrop"), "{opaque_passes:?}");
    assert!(transmissive_passes.contains(&"backdrop"), "glass draws in the translucent pass: {transmissive_passes:?}");
}

// A look-at camera, and the sun's cascades fitted to it, as shadow_looks.rs and a game have them.

fn look_at(eye: [f32; 3], target: [f32; 3]) -> (Matrix, Matrix, f64) {
    let length = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let normalize = |v: [f32; 3]| [v[0] / length(v), v[1] / length(v), v[2] / length(v)];
    let cross =
        |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let back = normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
    let right = normalize(cross([0.0, 1.0, 0.0], back));
    let up = cross(back, right);
    #[rustfmt::skip]
    let view = [
        right[0], up[0], back[0], 0.0,
        right[1], up[1], back[1], 0.0,
        right[2], up[2], back[2], 0.0,
        -dot(right, eye), -dot(up, eye), -dot(back, eye), 1.0,
    ];
    (view, projection(), TAN)
}

/// The light's view along `ray`, anchored 1000 back.
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
    let right = normalize(cross(ray, across));
    let up = cross(right, ray);
    let origin = ray.map(|value| value * -1000.0);
    #[rustfmt::skip]
    let matrix = [
        right[0], up[0], -ray[0], 0.0,
        right[1], up[1], -ray[1], 0.0,
        right[2], up[2], -ray[2], 0.0,
        -dot(right, origin), -dot(up, origin), dot(ray, origin), 1.0,
    ]
    .map(|value| value as f32);
    matrix
}

/// One cascade over the camera's view out to `far`: a sphere round the slice's corners, its centre
/// moved in whole texels of the light's view.
fn one_cascade(view: &Matrix, ray: [f64; 3], texels: u32, far: f64) -> (Matrix, Vec<Cascade>) {
    let light = light_view(ray);
    let wide = |m: &Matrix| m.map(f64::from);
    let (v, l) = (wide(view), wide(&light));
    let to_world = |p: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|i| (0..3).map(|row| v[i * 4 + row] * (p[row] - v[12 + row])).sum())
    };
    let to_light = |p: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|row| l[row] * p[0] + l[4 + row] * p[1] + l[8 + row] * p[2] + l[12 + row])
    };
    let corners: Vec<[f64; 3]> = [0.1, far]
        .into_iter()
        .flat_map(|d| {
            [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)].map(|(x, y)| to_world([x * d * TAN, y * d * TAN, -d]))
        })
        .collect();
    let centre: [f64; 3] = std::array::from_fn(|axis| corners.iter().map(|c| c[axis]).sum::<f64>() / 8.0);
    let radius = corners
        .iter()
        .map(|c| (0..3).map(|axis| (c[axis] - centre[axis]).powi(2)).sum::<f64>().sqrt())
        .fold(0.0, f64::max);
    let radius = (radius * 16.0).ceil() / 16.0 * f64::from(texels) / f64::from(texels - 1);
    let texel = 2.0 * radius / f64::from(texels);
    let [x, y, z] = to_light(centre);
    let (x, y) = ((x / texel).round() * texel, (y / texel).round() * texel);
    let distance = -z;
    let (near_plane, far_plane) = (distance - radius - 60.0, distance + radius + 60.0);
    let mut matrix = IDENTITY;
    matrix[0] = (1.0 / radius) as f32;
    matrix[5] = (1.0 / radius) as f32;
    matrix[10] = (-1.0 / (far_plane - near_plane)) as f32;
    matrix[12] = (-x / radius) as f32;
    matrix[13] = (-y / radius) as f32;
    matrix[14] = (-near_plane / (far_plane - near_plane)) as f32;
    (light, vec![Cascade { projection: matrix, texel: texel as f32, far: far as f32 }])
}

/// A lit floor that reads the sun's shadow map.
const FLOOR: &str = r#"
struct Scene {
  view: mat4x4f, projection: mat4x4f, ambient: vec4f, light_direction: vec4f, light_color: vec4f,
  light_view_projection: mat4x4f, shadow: vec4f, zenith: vec4f, horizon: vec4f, ground: vec4f, eye: vec4f,
  cascades: vec4f,
  cascade_view_projection: array<mat4x4f, 4>,
  cascade_texels: array<vec4f, 4>,
  shadow_params: vec4f,
};
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
${shadows}
struct Surface { @builtin(position) position: vec4f, @location(0) world: vec3f, @location(1) normal: vec3f };
@vertex fn vertex(@location(0) position: vec3f, @location(1) normal: vec3f) -> Surface {
  let world = draw.model * vec4f(position, 1.0);
  return Surface(scene.projection * scene.view * world, world.xyz, (draw.normal * vec4f(normal, 0.0)).xyz);
}
@fragment fn fragment(surface: Surface) -> @location(0) vec4f {
  let n = normalize(surface.normal);
  let to_sun = -normalize(scene.light_direction.xyz);
  let sun = max(dot(n, to_sun), 0.0) * sun_shadow(surface.world, n, surface.position);
  return vec4f(draw.color.rgb * (scene.ambient.rgb + scene.light_color.rgb * sun), 1.0);
}
"#;

/// A slab that is flat magenta, lit (so that an opaque one casts a shadow).
const SLAB: &str = r#"
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return draw.color; }
"#;

#[test]
fn a_transmission_material_casts_no_shadow() {
    let floor_mesh = r#"{"positions":[-14,0,-14, 14,0,-14, -14,0,14, 14,0,14],"normals":[0,1,0,0,1,0,0,1,0,0,1,0],
                         "uvs":[0,0,1,0,0,1,1,1],"indices":[0,2,1,1,2,3]}"#;
    let files = vec![
        ("floor.mesh.json", floor_mesh.to_owned()),
        ("floor.wgsl", FLOOR.replace("${shadows}", pen_render::SHADOWS_WGSL)),
        ("slab.wgsl", format!("{HEAD}{SLAB}")),
    ];
    let magenta = r#""color":[0.9,0.1,0.9,1],"lighting":"lit""#;
    let materials = vec![
        r#"{"shader":"floor.wgsl","color":[0.62,0.6,0.55,1],"lighting":"lit"}"#.to_owned(),
        format!(r#"{{"shader":"slab.wgsl",{magenta}}}"#),
        format!(r#"{{"shader":"slab.wgsl",{magenta},"surface":{{"transmission":1,"thickness":0.2}}}}"#),
    ];
    let Some(mut rig) = Rig::new("shadow", SIZE, &files, &materials) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let ray = [0.6, -0.5, 0.3];
    let (view, _, _) = look_at([7.0, 6.0, 9.0], [0.0, 1.2, 0.5]);
    let scene = |slab: Option<u32>| {
        let floor = draw(0, 0, IDENTITY, None);
        // The floor's mesh as a slab of 2.8 across, two metres up, between the sun and the floor.
        let mut draws = vec![floor];
        if let Some(material) = slab {
            let mut transform = IDENTITY;
            (transform[0], transform[10], transform[13]) = (0.1, 0.1, 2.0);
            draws.push(draw(0, material, transform, None));
        }
        let mut frame = frame(draws);
        frame.view = view;
        frame.projection = projection();
        frame.ambient = Vector { x: 0.16, y: 0.18, z: 0.22 };
        frame.light_color = Vector { x: 1.15, y: 1.05, z: 0.9 };
        frame.light_direction = Vector { x: ray[0] as f32, y: ray[1] as f32, z: ray[2] as f32 };
        frame.clear = Color { r: 0.5, g: 0.6, b: 0.75, a: 1.0 };
        frame.shadow.texels = 1024;
        let (light, cascades) = one_cascade(&frame.view, ray, 1024, 30.0);
        (frame.shadow.view, frame.cascades) = (light, cascades);
        frame.shadow.filter = 0;
        frame
    };
    let bare = rig.draw(&scene(None));
    let opaque = rig.draw(&scene(Some(1)));
    let opaque_passes = rig.renderer.passes();
    let glass = rig.draw(&scene(Some(2)));
    let glass_passes = rig.renderer.passes();
    let is_slab = |p: V3| (p[0] - 0.9).abs() < 0.02 && (p[1] - 0.1).abs() < 0.02 && (p[2] - 0.9).abs() < 0.02;
    let lum = |p: V3| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    let (mut shadow, mut slab_opaque, mut slab_glass, mut worst) = (0, 0, 0, 0.0f64);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let (b, o, g) = (bare.at(x, y), opaque.at(x, y), glass.at(x, y));
            slab_opaque += usize::from(is_slab(o));
            slab_glass += usize::from(is_slab(g));
            if is_slab(o) || is_slab(g) {
                continue;
            }
            if lum(b) - lum(o) > 0.08 {
                // In the opaque slab's shadow: the glass slab's frame has the floor as it is without either.
                shadow += 1;
                worst = worst.max((0..3).map(|c| (b[c] - g[c]).abs()).fold(0.0, f64::max));
            }
        }
    }
    eprintln!(
        "slab pixels: opaque {slab_opaque}, glass {slab_glass}; floor pixels in the opaque slab's shadow {shadow}, the glass frame's greatest difference from the bare floor there {worst:.4}"
    );
    assert!(
        slab_opaque > 500 && slab_glass.abs_diff(slab_opaque) < slab_opaque / 20,
        "the glass slab is drawn as large as the opaque one"
    );
    assert!(shadow > 300, "the opaque slab casts a shadow on the floor: {shadow} pixels");
    assert!(worst < 0.01, "the glass slab's shadow is {worst} dark");
    assert!(
        !opaque_passes.contains(&"backdrop") && glass_passes.contains(&"backdrop"),
        "{opaque_passes:?} {glass_passes:?}"
    );
}

#[test]
fn the_pipeline_layouts_fit_a_device_that_allows_fifteen_sampled_textures_to_a_stage() {
    let materials = vec![wall_material([0.8, 0.8, 0.8]), glass_material(r#","ior":1.5,"thickness":0.2"#)];
    let Some(mut rig) = Rig::with_limit("limit", 64, &refraction_files(), &materials, Some(15)) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    assert_eq!(rig.device.limits().max_sampled_textures_per_shader_stage, 15);
    // The renderer's layouts built on it and a frame with glass drew (a binding past 15 would be a
    // validation error on the way, which fails the test).
    let slab = Slab { turn: 0.3, ior: 1.5, thickness: 0.2 };
    let mut frame = refraction_frame(Some((&slab, 1)), 0.0);
    frame.view = view([0.0; 3]);
    let picture = rig.draw(&frame);
    assert!(picture.at(32, 32).iter().all(|&value| value > 0.05), "{:?}", picture.at(32, 32));
    assert!(rig.renderer.passes().contains(&"backdrop"));
}
