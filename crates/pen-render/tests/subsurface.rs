//! Screen-space subsurface scattering on a real (headless) GPU: a material that
//! says `"subsurface": true` stamps its surfaces into the stencil, and once the
//! opaque surfaces are drawn two passes blur the scene's picture there (along x,
//! then along y) before the sky and the mist are applied. Held to an f64
//! reference written here from the paper (Jimenez et al., "Separable Subsurface
//! Scattering", 2015) and d'Eon and Luebke's profile of skin:
//!
//! - the picture of a line of light, a row of light and a point on a flat marked
//!   surface equals the reference's diffusion pixel by pixel (the kernel's weights
//!   by channel, taps at their pixel offsets at the surface's depth, read
//!   bilinearly), with 1 and 4 samples and with TAA;
//! - the total light of the picture is conserved;
//! - pixels that are not marked, or only in part, are the same bits as without
//!   the flag, and take no light from marked ones or give none;
//! - light does not cross a depth edge between marked surfaces, by the documented
//!   ramp (kept below two widths of gap, dropped from four);
//! - frames without a subsurface material have no passes for it, and flipping
//!   one on and off changes no other picture.
//!
//! The pictures are drawn in HDR (extended sRGB, half-float output), where the
//! present pass shows the scene's colours as they are below 0.9, so a pixel of
//! the output is the scene's own half-float value. A sphere and a head drawn
//! with and without the flag go to /tmp/fidelity/materials/subsurface-e.png.
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

const HEAD: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
"#;

/// An unlit surface that draws `draw.color` where its mode says and `draw.parameters.w`
/// elsewhere, by pixel: `parameters` is (column, row, mode, base) and mode 1 a
/// column of light, 2 a row, 3 the one pixel where they cross, 4 everything left of
/// the column.
const LIGHT: &str = r#"
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let column = floor(position.x);
  let row = floor(position.y);
  let mode = u32(draw.parameters.z);
  let in_column = column == draw.parameters.x;
  let in_row = row == draw.parameters.y;
  var on = false;
  if (mode == 1u) { on = in_column; }
  if (mode == 2u) { on = in_row; }
  if (mode == 3u) { on = in_column && in_row; }
  if (mode == 4u) { on = column < draw.parameters.x; }
  return vec4f(select(vec3f(draw.parameters.w), draw.color.rgb, on), 1.0);
}
"#;

/// A pane: its colour and alpha, as the material has them.
const PANE_SHADER: &str = r#"
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return draw.color; }
"#;

/// The picture's size in pixels (a square) and the camera: a look down -z through
/// the origin, `TAN` the tangent of its half field of view, so that a surface
/// `depth` ahead of it has `SIZE / (2 depth TAN)` pixels to the metre.
const SIZE: u32 = 192;
const TAN: f64 = 0.5;
/// The depth of the surfaces most scenes draw, so that a centimetre (the reach
/// of the taps) is `SIZE / (2 DEPTH TAN) / 100` = 9.6 pixels.
const DEPTH: f64 = 0.2;
/// The world distance from a pixel to its outermost tap, as documented.
const WIDTH: f64 = 0.01;
/// The depth ramp, in widths, as documented: a tap counts in full up to the first
/// and not at all from the second.
const KEEP_UNTIL: f64 = 2.0;
const DROP_FROM: f64 = 4.0;
/// A colour of half-float values, so that a surface draws it exactly.
const LIGHT_COLOUR: V3 = [0.875, 0.75, 0.5];

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

/// Half-float bits as a number.
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

/// A half-float output.
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

/// What a frame drew: each pixel's red, green and blue as half-floats, and the bits
/// they are.
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
}

/// One of a test's materials: its colour and the material's own JSON keys.
#[derive(Clone, Copy)]
struct Material {
    shader: &'static str,
    color: V3,
    alpha: f64,
    extra: &'static str,
}

const PLAIN: Material = Material { shader: "light.wgsl", color: LIGHT_COLOUR, alpha: 1.0, extra: "" };
const SKIN: Material =
    Material { shader: "light.wgsl", color: LIGHT_COLOUR, alpha: 1.0, extra: r#""subsurface":true"# };
/// A half-transparent grey pane.
const PANE: Material = Material { shader: "pane.wgsl", color: [0.25; 3], alpha: 0.5, extra: "" };

const QUAD: &str = r#"{"positions":[-1,-1,0, 1,-1,0, -1,1,0, 1,1,0],"normals":[0,0,1,0,0,1,0,0,1,0,0,1],
                       "uvs":[0,0,1,0,0,1,1,1],"indices":[0,1,2,2,1,3]}"#;

struct Rig {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    catalog: Catalog,
    target: Hdr,
    size: u32,
}

impl Rig {
    fn new(name: &str, materials: &[Material], files: &[(&str, String)]) -> Option<Rig> {
        Self::sized(name, SIZE, materials, files)
    }

    fn sized(name: &str, size: u32, materials: &[Material], files: &[(&str, String)]) -> Option<Rig> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        let features = pen_render::optional_features(&adapter);
        let (device, queue) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor { required_features: features, ..Default::default() }),
        )
        .ok()?;
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("subsurface-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let write = |key: &str, bytes: &[u8]| std::fs::write(root.join(key), bytes).unwrap();
        write("quad.mesh.json", QUAD.as_bytes());
        write("light.wgsl", format!("{HEAD}{LIGHT}").as_bytes());
        write("pane.wgsl", format!("{HEAD}{PANE_SHADER}").as_bytes());
        for (key, text) in files {
            write(key, text.as_bytes());
        }
        let mut keys = Vec::new();
        for (index, material) in materials.iter().enumerate() {
            let key = format!("m{index}.material.json");
            let [r, g, b] = material.color;
            let extra = if material.extra.is_empty() { String::new() } else { format!(",{}", material.extra) };
            write(
                &key,
                format!(
                    r#"{{"shader":"{}","color":[{r},{g},{b},{}],"lighting":"unlit","instanceParameters":[0,0,0,0]{extra}}}"#,
                    material.shader,
                    material.alpha
                )
                .as_bytes(),
            );
            keys.push(key);
        }
        let mut meshes = vec!["quad.mesh.json".to_owned()];
        meshes.extend(files.iter().filter(|(key, _)| key.ends_with(".mesh.json")).map(|(key, _)| (*key).to_owned()));
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

    /// Submits `frame` until everything it names is loaded and compiled, then reads it.
    fn draw(&mut self, frame: &RenderFrame) -> Picture {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            self.renderer.submit(frame, &self.catalog, &mut self.target).unwrap();
            for event in self.renderer.take_events() {
                assert!(!matches!(event, GameEvent::AssetFailed { .. }), "an asset failed: {event:?}");
            }
            if self.renderer.pending() == 0
                && !self.renderer.waiting()
                && self.renderer.stats().draw_calls >= frame.draws.len() as u32
            {
                break;
            }
            assert!(Instant::now() < deadline, "assets did not load");
            std::thread::sleep(Duration::from_millis(5));
        }
        self.renderer.submit(frame, &self.catalog, &mut self.target).unwrap();
        self.read()
    }

    /// Draws `frame` `count` times (a still camera) and returns the last picture.
    fn settle(&mut self, frame: &RenderFrame, count: usize) -> Picture {
        let mut last = self.draw(frame);
        for _ in 1..count {
            self.renderer.submit(frame, &self.catalog, &mut self.target).unwrap();
            last = self.read();
        }
        last
    }

    fn read(&self) -> Picture {
        let texture = &self.target.0;
        let size = self.size;
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

    /// Loads everything the catalog names, with pipelines for `samples` per pixel.
    fn load(&mut self, samples: u32, shaders: &[&str]) {
        let mut assets: Vec<(String, String)> =
            self.catalog.meshes.iter().map(|key| ("mesh".to_owned(), key.clone())).collect();
        assets.extend(self.catalog.materials.iter().map(|key| ("material".to_owned(), key.clone())));
        assets.extend(shaders.iter().map(|key| ("shader".to_owned(), (*key).to_owned())));
        self.renderer.preload_scene(&assets, samples);
        let deadline = Instant::now() + Duration::from_secs(60);
        while self.renderer.pending() > 0 {
            assert!(Instant::now() < deadline, "assets did not load");
            self.renderer.wait(Duration::from_millis(20));
        }
    }
}

fn frame_of(draws: Vec<Draw>, samples: u32, taa: bool) -> RenderFrame {
    let mut frame = frame(draws);
    frame.projection = projection();
    frame.quality.msaa = samples;
    frame.quality.taa = taa;
    frame
}

/// A surface `depth` ahead of the eye covering the pictures' columns `from` to `to` (they may
/// lie outside the picture) and every row, of material `material`, whose pixels
/// `parameters` (column, row, mode, base) say.
fn plane(material: u32, depth: f64, from: f64, to: f64, parameters: [f32; 4]) -> Draw {
    let x = |column: f64| (column / f64::from(SIZE) * 2.0 - 1.0) * depth * TAN;
    let (left, right) = (x(from), x(to));
    let tall = depth * TAN * 4.0;
    let (centre, half_width) = ((left + right) / 2.0, (right - left) / 2.0);
    #[rustfmt::skip]
    let transform = [
        half_width as f32, 0.0, 0.0, 0.0,
        0.0, tall as f32, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        centre as f32, 0.0, -depth as f32, 1.0,
    ];
    Draw { mesh: 0, material, transform, parameters: Some(parameters), extra: None }
}

// d'Eon and Luebke's profile of skin and Jimenez's recipe for a kernel from it, written
// again here from the paper, as a list of all fifteen taps instead of seven pairs.

/// The sum of Gaussians: (variance in mm squared, weight).
const PROFILE: [(f64, f64); 6] =
    [(0.0064, 0.233), (0.0484, 0.100), (0.187, 0.118), (0.567, 0.113), (1.99, 0.358), (7.41, 0.078)];
const FALLOFF: V3 = [1.0, 0.37, 0.3];
const STRENGTH: V3 = [0.48, 0.41, 0.28];

#[derive(Clone, Copy, Debug)]
struct Tap {
    /// Out of the reach, -1 to 1.
    offset: f64,
    weights: V3,
}

fn reference_kernel() -> Vec<Tap> {
    let count = 15usize;
    let range = 2.0f64;
    let spacing = 2.0 * range / (count as f64 - 1.0);
    // Offsets, denser near the middle: sign(o) * o^2 / range.
    let offsets: Vec<f64> = (0..count)
        .map(|i| {
            let o = -range + i as f64 * spacing;
            o.signum() * o * o / range
        })
        .map(|offset| if offset == 0.0 { 0.0 } else { offset })
        .collect();
    let profile = |r: f64, channel: usize| -> f64 {
        PROFILE
            .iter()
            .map(|&(variance, weight)| {
                let scaled = r / (0.001 + FALLOFF[channel]);
                weight * (-scaled * scaled / (2.0 * variance)).exp() / (2.0 * std::f64::consts::PI * variance)
            })
            .sum()
    };
    // Each tap stands for the stretch halfway to each neighbour.
    let area = |i: usize| {
        let before = if i > 0 { (offsets[i] - offsets[i - 1]).abs() } else { 0.0 };
        let after = if i + 1 < count { (offsets[i] - offsets[i + 1]).abs() } else { 0.0 };
        (before + after) / 2.0
    };
    let raw: Vec<V3> = (0..count).map(|i| std::array::from_fn(|c| area(i) * profile(offsets[i], c))).collect();
    let sums: V3 = std::array::from_fn(|c| raw.iter().map(|weights| weights[c]).sum());
    (0..count)
        .map(|i| {
            let mut weights: V3 = std::array::from_fn(|c| raw[i][c] / sums[c]);
            weights = if i == count / 2 {
                std::array::from_fn(|c| (1.0 - STRENGTH[c]) + STRENGTH[c] * weights[c])
            } else {
                std::array::from_fn(|c| STRENGTH[c] * weights[c])
            };
            Tap { offset: offsets[i] / range, weights }
        })
        .collect()
}

/// What one axis of the diffusion does to a line of pixels: `values` (what the
/// scene drew), whether each is a marked surface, its depth ahead of the eye, and how
/// many pixels a world distance is at a depth.
fn diffuse_line(values: &[V3], marked: &[bool], depth: &[f64]) -> Vec<V3> {
    let kernel = reference_kernel();
    let centre = kernel.len() / 2;
    let count = i64::try_from(values.len()).unwrap();
    (0..values.len())
        .map(|p| {
            if !marked[p] {
                return values[p];
            }
            // Pixels to the outermost tap.
            let reach = (WIDTH / (2.0 * depth[p] * TAN / f64::from(SIZE))).min(64.0);
            let mut sum: V3 = std::array::from_fn(|c| values[p][c] * kernel[centre].weights[c]);
            for (i, tap) in kernel.iter().enumerate() {
                if i == centre {
                    continue;
                }
                let at = p as f64 + 0.5 + tap.offset * reach;
                let q = at.floor() as i64;
                let seen: V3 = if q < 0 || q >= count || !marked[q as usize] {
                    values[p]
                } else {
                    let keep = 1.0
                        - (((depth[q as usize] - depth[p]).abs() / WIDTH - KEEP_UNTIL) / (DROP_FROM - KEEP_UNTIL))
                            .clamp(0.0, 1.0);
                    let corner = at - 0.5;
                    let base = corner.floor();
                    let share = corner - base;
                    let read = |i: i64| values[i.clamp(0, count - 1) as usize];
                    let (a, b) = (read(base as i64), read(base as i64 + 1));
                    std::array::from_fn(|c| {
                        let bilinear = a[c] * (1.0 - share) + b[c] * share;
                        values[p][c] * (1.0 - keep) + bilinear * keep
                    })
                };
                for c in 0..3 {
                    sum[c] += tap.weights[c] * seen[c];
                }
            }
            sum
        })
        .collect()
}

/// The largest difference between two pixels' channels, relative to the reference
/// with a floor, and where.
fn worst(got: &[V3], want: &[V3]) -> (f64, usize) {
    let mut worst = (0.0f64, 0);
    for (i, (got, want)) in got.iter().zip(want).enumerate() {
        for c in 0..3 {
            let error = (got[c] - want[c]).abs() / want[c].abs().max(1e-3);
            if error > worst.0 {
                worst = (error, i);
            }
        }
    }
    worst
}

/// The relative error of a half-float result: a half-float step (up to 2^-10 of a
/// value, and the GPU truncates rather than rounds) for the intermediate picture and
/// another for the scene it is drawn back into.
const TOLERANCE: f64 = 2.0e-3;

fn light(parameters_mode: f32, column: u32, row: u32) -> [f32; 4] {
    [column as f32, row as f32, parameters_mode, 0.0]
}

fn row_of(picture: &Picture, y: u32) -> Vec<V3> {
    (0..SIZE).map(|x| picture.at(x, y)).collect()
}

fn column_of(picture: &Picture, x: u32) -> Vec<V3> {
    (0..SIZE).map(|y| picture.at(x, y)).collect()
}

fn lit(value: V3, on: bool) -> V3 {
    if on { value } else { [0.0; 3] }
}

#[test]
fn the_kernel_is_normalised_and_wider_for_red_than_for_blue() {
    let kernel = reference_kernel();
    for c in 0..3 {
        let sum: f64 = kernel.iter().map(|tap| tap.weights[c]).sum();
        assert!((sum - 1.0).abs() < 1e-12, "channel {c}: {sum}");
    }
    // Symmetric, centre in the middle.
    for i in 0..7 {
        assert!((kernel[i].offset + kernel[14 - i].offset).abs() < 1e-12);
        for c in 0..3 {
            assert!((kernel[i].weights[c] - kernel[14 - i].weights[c]).abs() < 1e-12);
        }
    }
    assert_eq!(kernel[7].offset, 0.0);
    // The share of the blur beyond half the reach: red reaches farther than green, green than blue.
    let beyond: Vec<f64> = (0..3)
        .map(|c| kernel.iter().filter(|tap| tap.offset.abs() > 0.5).map(|tap| tap.weights[c] / STRENGTH[c]).sum())
        .collect();
    assert!(beyond[0] > beyond[1] && beyond[1] > beyond[2], "{beyond:?}");
}

#[test]
fn a_line_a_row_and_a_point_of_light_diffuse_as_the_kernel_says() {
    let materials = [PLAIN, SKIN];
    for (samples, taa) in [(1, false), (4, false), (1, true)] {
        let Some(mut rig) = Rig::new("profile", &materials, &[]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        rig.load(samples, &["light.wgsl"]);
        let (x0, y0) = (SIZE / 2 + 3, SIZE / 2 - 5);
        let draw = |mode: f32| vec![plane(1, DEPTH, -50.0, f64::from(SIZE) + 50.0, light(mode, x0, y0))];
        let frames = if taa { 12 } else { 1 };
        // A tolerance for the picture TAA has made of the same picture twelve times.
        let tolerance = if taa { 4.0e-3 } else { TOLERANCE };
        let marked = vec![true; SIZE as usize];
        let depth = vec![DEPTH; SIZE as usize];
        let colour = LIGHT_COLOUR;
        // A line, a row, and an edge (all that TAA keeps of a line is another matter):
        // the reference's profile along the other axis.
        let modes: &[(f32, &str)] = if taa { &[(4.0, "edge")] } else { &[(1.0, "line"), (2.0, "row"), (4.0, "edge")] };
        for &(mode, name) in modes {
            let picture = rig.settle(&frame_of(draw(mode), samples, taa), frames);
            let vertical = mode == 2.0;
            let place = if vertical { y0 } else { x0 };
            let on: Vec<V3> =
                (0..SIZE).map(|i| lit(colour, if mode == 4.0 { i < place } else { i == place })).collect();
            let want = diffuse_line(&on, &marked, &depth);
            let got = if vertical { column_of(&picture, SIZE / 2 - 30) } else { row_of(&picture, SIZE / 2 + 20) };
            let (error, at) = worst(&got, &want);
            eprintln!("{name}, {samples} samples, taa {taa}: worst relative error {error:.2e} at pixel {at}");
            assert!(
                error < tolerance,
                "{name}, {samples} samples, taa {taa}: pixel {at} is {:?}, the reference {:?}",
                got[at],
                want[at]
            );
            // Nothing past the reach of a line, and the line's own pixel the brightest.
            if mode != 4.0 {
                let line = place as usize;
                for far in [0usize, line.saturating_sub(12), line + 12, SIZE as usize - 1] {
                    assert!(got[far].iter().all(|&value| value == 0.0), "{name}: {:?} at {far}", got[far]);
                }
                assert!(got[line][0] > got[line + 1][0] && got[line][0] > got[line - 1][0]);
            }
        }
        if taa {
            continue;
        }
        // A point: the product of the two axes' profiles.
        let picture = rig.settle(&frame_of(draw(3.0), samples, taa), frames);
        let along = |position: u32| -> Vec<V3> {
            diffuse_line(&(0..SIZE).map(|i| lit([1.0; 3], i == position)).collect::<Vec<_>>(), &marked, &depth)
        };
        let (across, down) = (along(x0), along(y0));
        let mut got = Vec::new();
        let mut want = Vec::new();
        for y in y0 - 14..=y0 + 14 {
            for x in x0 - 14..=x0 + 14 {
                got.push(picture.at(x, y));
                want.push(std::array::from_fn(|c| colour[c] * across[x as usize][c] * down[y as usize][c]));
            }
        }
        let (error, at) = worst(&got, &want);
        eprintln!("point, {samples} samples, taa {taa}: worst relative error {error:.2e} at patch pixel {at}");
        assert!(error < tolerance, "point, {samples} samples, taa {taa}: {:?} vs {:?}", got[at], want[at]);
        // It is the same on the diagonal as on the axes: a product, not a ring of one width.
        let corner = picture.at(x0 + 4, y0 + 4);
        assert!(corner[0] > 0.0);
    }
}

/// The total of every channel of every pixel: the light the picture holds.
fn light_of(picture: &Picture) -> V3 {
    let mut total = [0.0; 3];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let pixel = picture.at(x, y);
            for c in 0..3 {
                total[c] += pixel[c];
            }
        }
    }
    total
}

#[test]
fn the_light_of_the_picture_is_kept() {
    let materials = [PLAIN, SKIN];
    for samples in [1, 4] {
        let Some(mut rig) = Rig::new("energy", &materials, &[]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        rig.load(samples, &["light.wgsl"]);
        let (x0, y0) = (SIZE / 2 + 3, SIZE / 2 - 5);
        for (mode, name) in [(1.0, "line"), (2.0, "row"), (3.0, "point"), (4.0, "edge")] {
            let scene = |material: u32| {
                frame_of(
                    vec![plane(material, DEPTH, -50.0, f64::from(SIZE) + 50.0, light(mode, x0, y0))],
                    samples,
                    false,
                )
            };
            let (before, after) = (rig.draw(&scene(0)), rig.draw(&scene(1)));
            let (kept, spread) = (light_of(&before), light_of(&after));
            let change: Vec<f64> = (0..3).map(|c| (spread[c] - kept[c]).abs() / kept[c]).collect();
            // How much of the picture moved: the diffusion did something.
            let moved = (0..SIZE)
                .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
                .filter(|&(x, y)| (0..3).any(|c| (before.at(x, y)[c] - after.at(x, y)[c]).abs() > 0.01))
                .count();
            eprintln!(
                "{name}, {samples} samples: light {kept:.3?} -> {spread:.3?} (relative change {:.1e} {:.1e} {:.1e}), {moved} pixels moved by more than 0.01",
                change[0], change[1], change[2]
            );
            // Each stage's half-float steps, at most, in every pixel that has any light.
            assert!(change.iter().all(|&change| change < 3.0e-3), "{name}: the light changed by {change:?}");
            assert!(moved >= 5, "{name}: the picture did not change");
        }
    }
}

#[test]
fn pixels_that_are_not_marked_or_not_wholly_are_the_same_bits_and_give_and_take_no_light() {
    let materials = [PLAIN, SKIN];
    for samples in [1, 4] {
        let Some(mut rig) = Rig::new("seam", &materials, &[]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        rig.load(samples, &["light.wgsl"]);
        // The marked surface's edge: on a pixel's edge with one sample, through the
        // middle of the pixel with four, where the pixel is half marked.
        let edge = if samples == 1 { 96.0 } else { 96.5 };
        let (marked_line, plain_line) = (92u32, 100u32);
        let scene = |left: u32| {
            frame_of(
                vec![
                    plane(left, DEPTH, -50.0, edge, light(1.0, marked_line, 0)),
                    plane(0, DEPTH, edge, f64::from(SIZE) + 50.0, light(1.0, plain_line, 0)),
                ],
                samples,
                false,
            )
        };
        let (plain, skin) = (rig.draw(&scene(0)), rig.draw(&scene(1)));
        let first_unmarked = 96;
        let mut same = 0;
        for y in 0..SIZE {
            for x in first_unmarked..SIZE {
                assert_eq!(
                    skin.bits[(y * SIZE + x) as usize],
                    plain.bits[(y * SIZE + x) as usize],
                    "{samples} samples: pixel {x},{y} is not marked in full and changed"
                );
                same += 1;
            }
        }
        assert!(same > 0);
        if samples == 4 {
            // The pixel through which the edge runs is neither surface's colour alone: not a trivial case.
            let both = skin.at(96, 40);
            assert!(both[0] == 0.0 || both == LIGHT_COLOUR || both[0] > 0.0);
        }
        // The marked side: the reference's diffusion of its line, where the unmarked
        // pixels (the partly marked one among them) are not part of the surface and
        // their light does not reach it.
        let values: Vec<V3> = (0..SIZE).map(|i| lit(LIGHT_COLOUR, i == marked_line || i == plain_line)).collect();
        let marked: Vec<bool> = (0..SIZE).map(|i| i < first_unmarked).collect();
        let want = diffuse_line(&values, &marked, &vec![DEPTH; SIZE as usize]);
        let got = row_of(&skin, 60);
        let (error, at) = worst(&got[..first_unmarked as usize], &want[..first_unmarked as usize]);
        eprintln!("seam, {samples} samples: worst relative error on the marked side {error:.2e} at {at}");
        assert!(error < TOLERANCE, "{samples} samples: pixel {at} is {:?}, the reference {:?}", got[at], want[at]);
        // The marked line's light did spread toward the edge, and took none from the plain line 8 pixels on.
        assert!(got[(first_unmarked - 1) as usize][0] > 0.0);
        let without = diffuse_line(
            &(0..SIZE).map(|i| lit(LIGHT_COLOUR, i == marked_line)).collect::<Vec<_>>(),
            &marked,
            &vec![DEPTH; SIZE as usize],
        );
        for i in 0..first_unmarked as usize {
            assert_eq!(want[i], without[i], "the reference lets the plain line reach pixel {i}");
        }
    }
}

#[test]
fn light_does_not_cross_an_edge_between_surfaces_at_different_depths() {
    let materials = [PLAIN, SKIN];
    let Some(mut rig) = Rig::new("depth", &materials, &[]) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    rig.load(1, &["light.wgsl"]);
    let edge = 96u32;
    let line = 97u32;
    for gap in [0.01, 0.03, 0.06, 0.10] {
        let back = DEPTH + gap;
        // The near surface covers the left of the picture, the far one all of it (and is
        // seen where the near one is not); the far one carries a line of light a pixel past the edge.
        let frame = frame_of(
            vec![
                plane(1, DEPTH, -50.0, f64::from(edge), light(0.0, 0, 0)),
                plane(1, back, -50.0, f64::from(SIZE) + 50.0, light(1.0, line, 0)),
            ],
            1,
            false,
        );
        let picture = rig.draw(&frame);
        let values: Vec<V3> = (0..SIZE).map(|i| lit(LIGHT_COLOUR, i == line)).collect();
        let marked = vec![true; SIZE as usize];
        let depth: Vec<f64> = (0..SIZE).map(|i| if i < edge { DEPTH } else { back }).collect();
        let want = diffuse_line(&values, &marked, &depth);
        let got = row_of(&picture, 30);
        let (error, at) = worst(&got, &want);
        // What it would be if the surfaces were one.
        let one_surface = diffuse_line(&values, &marked, &vec![back; SIZE as usize]);
        let leak = got[(edge - 1) as usize][0];
        eprintln!(
            "gap {gap}: worst error {error:.2e} at {at}; near pixel beside the edge holds {leak:.5} (one surface: {:.5})",
            one_surface[(edge - 1) as usize][0]
        );
        assert!(error < TOLERANCE, "gap {gap}: pixel {at} is {:?}, the reference {:?}", got[at], want[at]);
        // Past the ramp (four widths) nothing crosses: the near surface holds no light at all.
        if gap >= DROP_FROM * WIDTH - 1e-9 {
            assert!(got[..edge as usize].iter().all(|pixel| *pixel == [0.0; 3]), "gap {gap}: light crossed the edge");
            assert!(one_surface[(edge - 1) as usize][0] > 5e-3, "the test would not have seen a leak");
        }
        // Inside two widths the surfaces count as one.
        if gap <= KEEP_UNTIL * WIDTH + 1e-9 {
            assert!(leak > 5e-3, "gap {gap}: the light did not cross a small step");
        }
    }
}

#[test]
fn frames_without_a_subsurface_material_have_no_passes_for_it_and_flipping_one_in_changes_nothing_else() {
    let materials = [PLAIN, SKIN, PANE];
    let Some(mut rig) = Rig::new("flip", &materials, &[]) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    rig.load(4, &["light.wgsl", "pane.wgsl"]);
    let (x0, y0) = (SIZE / 2 + 3, SIZE / 2 - 5);
    let scene = |material: u32, glass: bool| {
        let mut draws = vec![plane(material, DEPTH, -50.0, f64::from(SIZE) + 50.0, light(1.0, x0, y0))];
        if glass {
            draws.push(plane(2, 0.15, f64::from(SIZE) / 4.0, f64::from(SIZE) * 0.75, [0.0; 4]));
        }
        frame_of(draws, 4, false)
    };
    let names = |rig: &Rig| -> Vec<&'static str> { rig.renderer.passes() };
    for glass in [false, true] {
        let plain = rig.draw(&scene(0, glass));
        assert!(names(&rig).iter().all(|name| !name.starts_with("subsurface")), "{:?}", names(&rig));
        let skin = rig.draw(&scene(1, glass));
        let passes = names(&rig);
        assert_eq!(passes[..2], ["subsurface x", "subsurface y"], "{passes:?}");
        if glass {
            // Before the opaque picture's own passes: diffused before it is misted.
            let atmosphere = passes
                .iter()
                .position(|name| *name == "atmosphere")
                .expect("a frame with glass draws the opaque picture");
            assert!(atmosphere > 1 && passes.contains(&"backdrop"), "{passes:?}");
        }
        assert!(plain != skin, "glass {glass}: the flag did nothing");
        let plain_again = rig.draw(&scene(0, glass));
        assert!(names(&rig).iter().all(|name| !name.starts_with("subsurface")));
        assert!(plain == plain_again, "glass {glass}: a frame without the flag changed after one with it");
        let skin_again = rig.draw(&scene(1, glass));
        assert!(skin == skin_again, "glass {glass}: the diffusion is not repeatable");
    }
}

#[test]
fn glass_over_a_diffused_surface_is_drawn_over_the_diffused_picture() {
    let materials = [PLAIN, SKIN, PANE];
    let Some(mut rig) = Rig::new("glass", &materials, &[]) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    rig.load(1, &["light.wgsl", "pane.wgsl"]);
    let (x0, y0) = (SIZE / 2 + 3, SIZE / 2 - 5);
    let draws = vec![
        plane(1, DEPTH, -50.0, f64::from(SIZE) + 50.0, light(1.0, x0, y0)),
        plane(2, 0.15, f64::from(SIZE) / 4.0, f64::from(SIZE) * 0.75, [0.0; 4]),
    ];
    let picture = rig.draw(&frame_of(draws, 1, false));
    let on: Vec<V3> = (0..SIZE).map(|i| lit(LIGHT_COLOUR, i == x0)).collect();
    let diffused = diffuse_line(&on, &vec![true; SIZE as usize], &vec![DEPTH; SIZE as usize]);
    // Behind the pane (its middle half), half its grey over half the diffused picture.
    let want: Vec<V3> = diffused.iter().map(|pixel| pixel.map(|value| 0.5 * 0.25 + 0.5 * value)).collect();
    let got = row_of(&picture, 20);
    let span = (SIZE / 4 + 2) as usize..(SIZE * 3 / 4 - 2) as usize;
    let (error, at) = worst(&got[span.clone()], &want[span.clone()]);
    eprintln!("glass over skin: worst relative error {error:.2e} at {}", span.start + at);
    assert!(
        error < 4.0e-3,
        "pixel {} is {:?}, the reference {:?}",
        span.start + at,
        got[span.start + at],
        want[span.start + at]
    );
    assert!(got[x0 as usize][0] > 0.4 && got[x0 as usize + 3][0] > 0.125 + 0.5 * 0.0058);
}

/// A unit sphere's mesh, `rings` by `segments`, its normals its positions.
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

/// Skin as a shader can fake it without a map: freckles and a web of fine veins
/// in the surface's own coordinates, and light that stops at the terminator
/// (an ambient light of a fifth of the sun's and a sixth under it), so that
/// every edge of detail is a hard one for the diffusion to soften.
const SKIN_SHADER: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f, ambient: vec4f, light_direction: vec4f, light_color: vec4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
struct Corner { @builtin(position) position: vec4f, @location(0) normal: vec3f, @location(1) local: vec3f };
@vertex fn vertex(@location(0) position: vec3f, @location(1) normal: vec3f) -> Corner {
  let world = draw.model * vec4f(position, 1.0);
  return Corner(scene.projection * scene.view * world, (draw.normal * vec4f(normal, 0.0)).xyz, position);
}
fn hash(p: vec3f) -> f32 { return fract(sin(dot(p, vec3f(127.1, 311.7, 74.7))) * 43758.5453); }
fn value_noise(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let x00 = mix(hash(i), hash(i + vec3f(1.0, 0.0, 0.0)), u.x);
  let x10 = mix(hash(i + vec3f(0.0, 1.0, 0.0)), hash(i + vec3f(1.0, 1.0, 0.0)), u.x);
  let x01 = mix(hash(i + vec3f(0.0, 0.0, 1.0)), hash(i + vec3f(1.0, 0.0, 1.0)), u.x);
  let x11 = mix(hash(i + vec3f(0.0, 1.0, 1.0)), hash(i + vec3f(1.0, 1.0, 1.0)), u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let n = normalize(corner.normal);
  let ndl = dot(n, normalize(-scene.light_direction.xyz));
  let freckles = smoothstep(0.74, 0.8, value_noise(corner.local * 34.0 + 7.0));
  let veins = 1.0 - smoothstep(0.0, 0.04, abs(value_noise(corner.local * 11.0) - 0.5));
  var albedo = draw.color.rgb * (1.0 - 0.6 * freckles);
  albedo = mix(albedo, albedo * vec3f(0.8, 0.5, 0.55), veins * 0.9);
  let direct = select(0.0, 0.5 + 0.5 * ndl, ndl > 0.0);
  return vec4f(albedo * (scene.ambient.rgb + scene.light_color.rgb * direct), 1.0);
}
"#;

const FLESH: V3 = [0.78, 0.52, 0.42];

/// A model matrix: scales then places.
fn model(scale: [f64; 3], at: [f64; 3]) -> Matrix {
    #[rustfmt::skip]
    let matrix = [
        scale[0] as f32, 0.0, 0.0, 0.0,
        0.0, scale[1] as f32, 0.0, 0.0,
        0.0, 0.0, scale[2] as f32, 0.0,
        at[0] as f32, at[1] as f32, at[2] as f32, 1.0,
    ];
    matrix
}

/// The head, from the front: an ellipsoid with a nose, two ears and two eyes (a
/// white that is not skin, a material of its own), a world metre to the unit.
fn head_frame(skin: u32) -> RenderFrame {
    let draw = |material: u32, scale: [f64; 3], at: [f64; 3], parameters: Option<[f32; 4]>| Draw {
        mesh: 1,
        material,
        transform: model(scale, at),
        parameters,
        extra: None,
    };
    let eye = Some([0.0, 0.0, 0.0, 0.85]);
    let draws = vec![
        draw(skin, [0.09, 0.115, 0.1], [0.0, 0.0, 0.0], None),
        draw(skin, [0.022, 0.03, 0.03], [0.0, -0.015, 0.1], None),
        draw(skin, [0.012, 0.03, 0.02], [0.088, 0.0, -0.005], None),
        draw(skin, [0.012, 0.03, 0.02], [-0.088, 0.0, -0.005], None),
        draw(2, [0.013, 0.013, 0.008], [0.035, 0.03, 0.093], eye),
        draw(2, [0.013, 0.013, 0.008], [-0.035, 0.03, 0.093], eye),
    ];
    let mut frame = frame_of(draws, 4, false);
    frame.view = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.01, -0.33, 1.0];
    frame.ambient = Vector { x: 0.2, y: 0.2, z: 0.24 };
    frame.light_color = Vector { x: 0.85, y: 0.8, z: 0.72 };
    frame.light_direction = Vector { x: -0.6, y: -0.3, z: -0.75 };
    frame.clear = Color { r: 0.1, g: 0.1, b: 0.12, a: 1.0 };
    frame
}

#[test]
fn a_head_with_and_without_the_flag() {
    const HEAD_SIZE: u32 = 384;
    let flesh = |extra: &'static str| Material { shader: "skin.wgsl", color: FLESH, alpha: 1.0, extra };
    let materials = [
        flesh(""),
        flesh(r#""subsurface":true"#),
        Material { shader: "light.wgsl", color: [0.9; 3], alpha: 1.0, extra: "" },
    ];
    let files = [("skin.wgsl", SKIN_SHADER.to_owned()), ("sphere.mesh.json", sphere(48, 96))];
    let Some(mut rig) = Rig::sized("head", HEAD_SIZE, &materials, &files) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    rig.load(4, &["light.wgsl", "skin.wgsl"]);
    let (plain, skin) = (rig.draw(&head_frame(0)), rig.draw(&head_frame(1)));
    // What the flag changes: the skin, not what is round it, and not the eyes.
    let size = HEAD_SIZE as usize;
    let clear = plain.bits[2 * size + 2];
    let mut changed = 0;
    for y in 0..size {
        for x in 0..size {
            let (a, b) = (plain.bits[y * size + x], skin.bits[y * size + x]);
            if a == clear {
                assert_eq!(a, b, "pixel {x},{y} is empty and changed");
            }
            changed += usize::from(a != b);
        }
    }
    // The picture as one: the plain head, the diffused one and what the diffusion changed (eight
    // times as strong), and below each a patch of the cheek and the terminator, three times as large.
    let (crop, zoom) = ((80u32, 200u32, 128u32), 3u32);
    let mut image = image::RgbImage::new(HEAD_SIZE * 3, HEAD_SIZE + crop.2 * zoom);
    let rgb8 = |pixel: V3| image::Rgb(pixel.map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8));
    let difference = |x: u32, y: u32| -> V3 {
        let (a, b) = (plain.at(x, y), skin.at(x, y));
        std::array::from_fn(|c| 0.5 + (b[c] - a[c]) * 8.0)
    };
    for column in 0..3u32 {
        let shot = |x: u32, y: u32| match column {
            0 => plain.at(x, y),
            1 => skin.at(x, y),
            _ => difference(x, y),
        };
        let shift = column * HEAD_SIZE;
        for y in 0..HEAD_SIZE {
            for x in 0..HEAD_SIZE {
                image.put_pixel(x + shift, y, rgb8(shot(x, y)));
            }
        }
        for y in 0..crop.2 * zoom {
            for x in 0..crop.2 * zoom {
                image.put_pixel(x + shift, HEAD_SIZE + y, rgb8(shot(crop.0 + x / zoom, crop.1 + y / zoom)));
            }
        }
    }
    let directory = PathBuf::from("/tmp/fidelity/materials");
    std::fs::create_dir_all(&directory).unwrap();
    image.save(directory.join("subsurface-e.png")).unwrap();
    assert!(changed > 5000, "the head barely changed: {changed} pixels");
    // The eye is not skin: its pixels, and the skin round it that the eye gave no light to, hold.
    for y in 157..162u32 {
        for x in 246..251u32 {
            assert_eq!(
                plain.bits[(y * HEAD_SIZE + x) as usize],
                skin.bits[(y * HEAD_SIZE + x) as usize],
                "the eye at {x},{y}"
            );
        }
    }
    // The light is kept (to the diffusion at the silhouette, where unmarked pixels give the pixel its own colour
    // back, and the half-float steps).
    let (light_plain, light_skin) = (light_of(&plain), light_of(&skin));
    let change: Vec<f64> = (0..3).map(|c| (light_skin[c] - light_plain[c]) / light_plain[c]).collect();
    eprintln!("head: light {light_plain:.1?} -> {light_skin:.1?}, relative change {change:?}");
    assert!(change.iter().all(|change| change.abs() < 0.01), "{change:?}");
    // The detail softens: the sum of the steps between neighbours, over the cheek.
    let steps = |picture: &Picture| -> f64 {
        let mut total = 0.0;
        for y in crop.1..crop.1 + crop.2 {
            for x in crop.0..crop.0 + crop.2 - 1 {
                total += (0..3).map(|c| (picture.at(x + 1, y)[c] - picture.at(x, y)[c]).abs()).sum::<f64>();
            }
        }
        total
    };
    let (sharp, soft) = (steps(&plain), steps(&skin));
    eprintln!("head: the steps between neighbouring pixels over the cheek, {sharp:.1} -> {soft:.1}");
    assert!(soft < 0.85 * sharp, "{soft} vs {sharp}");
    // Past the terminator the light of the lit side bleeds in, red farther than blue: the shadowed
    // pixels with a lit one within eight pixels gain more red than green than blue.
    let luma = |pixel: V3| pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722;
    let (mut gain, mut count) = ([0.0; 3], 0);
    for y in 8..HEAD_SIZE - 8 {
        for x in 8..HEAD_SIZE - 8 {
            let here = plain.at(x, y);
            if luma(here) > 0.3 || luma(here) < 0.05 {
                continue;
            }
            let near_light = (1..=8).any(|d| luma(plain.at(x + d, y)) > 0.55);
            if near_light {
                let after = skin.at(x, y);
                for c in 0..3 {
                    gain[c] += after[c] - here[c];
                }
                count += 1;
            }
        }
    }
    let gain = gain.map(|gain| gain / f64::from(count));
    eprintln!("head: {count} shadowed pixels beside light gain {gain:.4?}");
    assert!(count > 50 && gain[0] > gain[1] && gain[1] > gain[2] && gain[2] > 0.0, "{gain:?}");
}
