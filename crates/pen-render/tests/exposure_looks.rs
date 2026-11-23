//! Exposure and grading as a game sees them, on a real (headless) GPU: a dark
//! room with a bright window drawn with a fixed exposure and with auto exposure,
//! the adaptation over time and after a cut, and the grade (white balance,
//! contrast, saturation, a vignette and look-up tables) on SDR and HDR displays.
//! The shaders are checked against f64 references in the crate's unit tests;
//! these check the frame's settings reach them, and what they do to a picture.
//! Each test writes what it drew to target/look-shots (or the directory in
//! `EXPOSURE_SHOTS`). Skips when no adapter is available.
#![cfg(not(target_arch = "wasm32"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test pictures"
)]
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

mod common;

use common::*;
use pen_protocol::*;
use pen_render::{Display, RenderTarget, Renderer, RendererOptions};

// Draws its mesh in its material's color times the draw's first parameter.
const GAIN: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return vec4f(draw.color.rgb * draw.parameters.x, 1.0); }
"#;

const WIDTH: u32 = 384;
const HEIGHT: u32 = 256;

fn shots() -> PathBuf {
    let directory = std::env::var_os("EXPOSURE_SHOTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/look-shots"));
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

struct Screen(wgpu::Texture);

impl Screen {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        Screen(device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }))
    }
}

impl RenderTarget for Screen {
    fn size(&self) -> (u32, u32) {
        (self.0.width(), self.0.height())
    }
    fn view(&mut self) -> Option<wgpu::TextureView> {
        Some(self.0.create_view(&Default::default()))
    }
    fn present(&mut self) {}
}

/// Every texel's bytes, rows packed.
fn read(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let texel = texture.format().block_copy_size(None).unwrap();
    let (width, height) = (texture.width(), texture.height());
    let row = (width * texel).next_multiple_of(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(height) },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mapped = buffer.slice(..).get_mapped_range().unwrap();
    (0..height as usize)
        .flat_map(|y| mapped[y * row as usize..y * row as usize + (width * texel) as usize].to_vec())
        .collect()
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

fn srgb_decode(value: f64) -> f64 {
    let c = value.max(0.0);
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn srgb_encode(value: f64) -> f64 {
    let l = value.max(0.0);
    if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 }
}

/// The present pass's highlights: each channel as it is up to 0.9, then rolling
/// off toward `peak`.
fn toned(value: f64, peak: f64) -> f64 {
    let d = peak - 0.9;
    if value > 0.9 { peak - d * d / (value + d - 0.9) } else { value }
}

/// The two displays: SDR, and extended sRGB reaching four times white.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Sdr,
    Hdr,
}

impl Kind {
    fn format(self) -> wgpu::TextureFormat {
        match self {
            Kind::Sdr => FORMAT,
            Kind::Hdr => wgpu::TextureFormat::Rgba16Float,
        }
    }

    fn display(self) -> Display {
        match self {
            Kind::Sdr => Display::SDR,
            Kind::Hdr => {
                Display { color_space: wgpu::SurfaceColorSpace::ExtendedSrgb, headroom: 4.0, white_nits: 203.0 }
            }
        }
    }

    fn peak(self) -> f64 {
        match self {
            Kind::Sdr => 1.0,
            Kind::Hdr => srgb_encode(4.0),
        }
    }

    /// What the display shows (in its encoding, 0 to its peak) of encoded light
    /// that is `light` before the present pass tones it.
    fn shown(self, light: [f64; 3]) -> [f64; 3] {
        light.map(|value| toned(srgb_encode(value), self.peak()))
    }
}

/// A rectangle of the picture, in clip space, drawn flat in `color` times `gain`.
#[derive(Clone, Copy)]
struct Rect {
    x: (f32, f32),
    y: (f32, f32),
    depth: f32,
    color: [f32; 3],
    gain: f32,
}

struct Rig {
    kind: Kind,
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    catalog: Catalog,
    target: Screen,
    rects: usize,
}

fn strip(size: u32, map: impl Fn([f64; 3]) -> [f64; 3]) -> Vec<u8> {
    let mut image = image::RgbaImage::new(size * size, size);
    for g in 0..size {
        for b in 0..size {
            for r in 0..size {
                let top = f64::from(size - 1);
                let out = map([f64::from(r) / top, f64::from(g) / top, f64::from(b) / top]);
                let [r8, g8, b8] = out.map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8);
                image.put_pixel(b * size + r, g, image::Rgba([r8, g8, b8, 255]));
            }
        }
    }
    let mut bytes = Vec::new();
    image.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png).unwrap();
    bytes
}

/// The catalog's materials: a gain-shaded material of each color, then each table's.
const COLORS: [[f32; 3]; 8] = [
    [0.8, 0.78, 0.72],
    [0.6, 0.55, 0.5],
    [0.85, 0.92, 1.0],
    [0.7, 0.2, 0.15],
    [1.0, 0.9, 0.6],
    [0.5, 0.5, 0.5],
    [0.8, 0.3, 0.1],
    [0.2, 0.6, 0.9],
];

impl Rig {
    fn new(name: &str, kind: Kind) -> Option<Rig> {
        let (device, queue) = featured_device()?;
        let quad = br#"{"positions":[-1,-1,0.5, 1,-1,0.5, -1,1,0.5, 1,1,0.5],"normals":[0,0,1,0,0,1,0,0,1,0,0,1],
                        "uvs":[0,0,1,0,0,1,1,1],"indices":[0,1,2,2,1,3]}"#;
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("exposure-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let mut materials: Vec<String> = Vec::new();
        let mut files: Vec<(String, Vec<u8>)> =
            vec![("quad.mesh.json".into(), quad.to_vec()), ("gain.wgsl".into(), GAIN.as_bytes().to_vec())];
        for (index, color) in COLORS.iter().enumerate() {
            let key = format!("color{index}.material.json");
            let json = format!(
                r#"{{"shader":"gain.wgsl","color":[{},{},{},1],"lighting":"unlit"}}"#,
                color[0], color[1], color[2]
            );
            files.push((key.clone(), json.into_bytes()));
            materials.push(key);
        }
        // Look-up tables of 16 slices: none, red and blue swapped, and a warm look.
        let tables: [(&str, Vec<u8>); 3] = [
            ("identity", strip(16, |c| c)),
            ("swap", strip(16, |c| [c[2], c[1], c[0]])),
            ("warm", strip(16, |c| [(c[0] * 1.15).min(1.0), c[1], c[2] * 0.8])),
        ];
        for (name, bytes) in tables {
            files.push((format!("{name}.lut.png"), bytes));
            let json =
                format!(r#"{{"shader":"gain.wgsl","color":[1,1,1,1],"lighting":"unlit","texture":"{name}.lut.png"}}"#);
            files.push((format!("{name}.material.json"), json.into_bytes()));
            materials.push(format!("{name}.material.json"));
        }
        let catalog_json = format!(
            r#"{{"meshes":["quad.mesh.json"],"materials":[{}]}}"#,
            materials.iter().map(|key| format!("\"{key}\"")).collect::<Vec<_>>().join(",")
        );
        files.push(("catalog.json".into(), catalog_json.into_bytes()));
        for (key, bytes) in files {
            std::fs::write(root.join(key), bytes).unwrap();
        }
        let catalog = boundary::read_catalog(&root).unwrap();
        let options = RendererOptions {
            format: kind.format(),
            display: kind.display(),
            assets: root,
            veil: None,
            pipeline_cache: None,
        };
        let renderer = Renderer::new(device.clone(), queue.clone(), options).unwrap();
        let target = Screen::new(&device, kind.format());
        Some(Rig { kind, device, queue, renderer, catalog, target, rects: 0 })
    }

    /// A frame of these rectangles, nearest depth in front.
    fn scene(&mut self, rects: &[Rect]) -> RenderFrame {
        let mut draws = Vec::new();
        for rect in rects {
            let material = COLORS.iter().position(|color| *color == rect.color).expect("a catalog color") as u32;
            let (width, height) = ((rect.x.1 - rect.x.0) / 2.0, (rect.y.1 - rect.y.0) / 2.0);
            let (x, y) = ((rect.x.0 + rect.x.1) / 2.0, (rect.y.0 + rect.y.1) / 2.0);
            draws.push(Draw {
                mesh: 0,
                material,
                transform: [
                    width,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    height,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    x,
                    y,
                    rect.depth - 0.5,
                    1.0,
                ],
                parameters: Some([rect.gain, 0.0, 0.0, 0.0]),
                extra: None,
            });
        }
        self.rects = draws.len();
        frame(draws)
    }

    /// Submits `frame` once everything it draws has loaded, and again.
    fn settle(&mut self, frame: &RenderFrame) {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            self.renderer.submit(frame, &self.catalog, &mut self.target).unwrap();
            if self.renderer.pending() == 0 && self.renderer.stats().draw_calls >= self.rects as u32 {
                break;
            }
            assert!(Instant::now() < deadline, "assets did not load: {:?}", self.renderer.take_events());
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// What the display shows after `frame`, each pixel's channels in the display's encoding.
    fn show(&mut self, frame: &RenderFrame) -> Vec<[f64; 3]> {
        self.settle(frame);
        self.renderer.submit(frame, &self.catalog, &mut self.target).unwrap();
        self.shown()
    }

    fn shown(&mut self) -> Vec<[f64; 3]> {
        let bytes = read(&self.device, &self.queue, &self.target.0);
        match self.kind {
            Kind::Sdr => bytes.chunks(4).map(|texel| [2, 1, 0].map(|i| f64::from(texel[i]) / 255.0)).collect(),
            Kind::Hdr => bytes
                .chunks(8)
                .map(|texel| [0, 1, 2].map(|i| half(u16::from_le_bytes([texel[i * 2], texel[i * 2 + 1]]))))
                .collect(),
        }
    }

    /// The exposure the frame settles at, read back: `frame`'s again without time
    /// passing, until a frame's exposure has come back.
    fn exposure(&mut self, frame: &RenderFrame) -> f64 {
        let mut hold = frame.clone();
        hold.exposure.dt = 0.0;
        hold.exposure.cut = false;
        for _ in 0..8 {
            self.settle(&hold);
            self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        }
        f64::from(self.renderer.stats().exposure.expect("an exposure came back"))
    }

    fn save(&mut self, name: &str, pixels: &[[f64; 3]]) {
        let peak = self.kind.peak();
        let image = image::RgbaImage::from_fn(WIDTH, HEIGHT, |x, y| {
            let pixel = pixels[(y * WIDTH + x) as usize];
            let [r, g, b] = pixel.map(|value| (value / peak * 255.0).round().clamp(0.0, 255.0) as u8);
            image::Rgba([r, g, b, 255])
        });
        image.save(shots().join(name)).unwrap();
    }
}

fn at(pixels: &[[f64; 3]], x: u32, y: u32) -> [f64; 3] {
    pixels[(y * WIDTH + x) as usize]
}

/// A dark room, lamp and couch, with a bright window onto the day.
fn room(rig: &mut Rig, daylight: f32) -> (RenderFrame, Vec<Rect>) {
    let rects = vec![
        Rect { x: (-1.0, 1.0), y: (-1.0, 1.0), depth: 0.9, color: COLORS[0], gain: 0.05 },
        Rect { x: (-1.0, 1.0), y: (-1.0, -0.45), depth: 0.85, color: COLORS[1], gain: 0.025 },
        Rect { x: (0.2, 0.8), y: (-0.1, 0.75), depth: 0.75, color: COLORS[2], gain: daylight },
        Rect { x: (-0.8, -0.2), y: (-0.7, -0.2), depth: 0.7, color: COLORS[3], gain: 0.06 },
        Rect { x: (-0.55, -0.45), y: (0.2, 0.5), depth: 0.7, color: COLORS[4], gain: 1.5 },
    ];
    (rig.scene(&rects), rects)
}

/// The luminance of each pixel of `rects` as the shader leaves it, before exposure.
fn lit(rects: &[Rect]) -> Vec<f64> {
    (0..WIDTH * HEIGHT)
        .map(|index| {
            let (x, y) = ((index % WIDTH) as f32 + 0.5, (index / WIDTH) as f32 + 0.5);
            let (x, y) = (x / WIDTH as f32 * 2.0 - 1.0, 1.0 - y / HEIGHT as f32 * 2.0);
            let nearest = rects
                .iter()
                .filter(|rect| rect.x.0 <= x && x < rect.x.1 && rect.y.0 <= y && y < rect.y.1)
                .min_by(|a, b| a.depth.total_cmp(&b.depth))
                .expect("the wall covers the picture");
            let light = nearest.color.map(|channel| srgb_decode(f64::from(channel * nearest.gain)));
            0.2126 * light[0] + 0.7152 * light[1] + 0.0722 * light[2]
        })
        .collect()
}

/// The exposure, in stops, that brings the trimmed mean of the log luminance to
/// the target, computed pixel by pixel without bins.
fn goal(lit: &[f64], low: f64, high: f64, target: f64) -> f64 {
    let mut logs: Vec<f64> = lit.iter().map(|value| value.max(2f64.powi(-10)).log2()).collect();
    logs.sort_by(f64::total_cmp);
    let count = logs.len() as f64;
    let (mut weight, mut sum) = (0.0, 0.0);
    for (index, log) in logs.iter().enumerate() {
        let inside = ((index as f64 + 1.0).min(high * count) - (index as f64).max(low * count)).max(0.0);
        weight += inside;
        sum += inside * log;
    }
    target.log2() - sum / weight
}

fn auto(dt: f32) -> Exposure {
    Exposure { auto: true, dt, ..Default::default() }
}

fn mean_luminance(pixels: &[[f64; 3]], x: std::ops::Range<u32>, y: std::ops::Range<u32>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0.0;
    for row in y {
        for column in x.clone() {
            let p = at(pixels, column, row);
            sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
            count += 1.0;
        }
    }
    sum / count
}

#[test]
fn a_fixed_exposure_is_todays_look_and_scales_the_light_on_sdr_and_hdr_displays() {
    for kind in [Kind::Sdr, Kind::Hdr] {
        let Some(mut rig) = Rig::new(&format!("fixed-{kind:?}"), kind) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        // A flat picture of encoded light 0.2 and one of 0.6 (which an HDR display rolls off less).
        for value in [0.2f32, 0.6, 2.0] {
            let rect = Rect { x: (-1.0, 1.0), y: (-1.0, 1.0), depth: 0.5, color: COLORS[5], gain: value / 0.5 };
            let mut frame = rig.scene(&[rect]);
            let plain = at(&rig.show(&frame), 100, 100);
            // As it was: nothing between the scene and the roll-off.
            let was = [f64::from(value); 3].map(|v| toned(v, kind.peak()));
            for (shown, wanted) in plain.iter().zip(was) {
                assert!((shown - wanted).abs() < 0.004, "{kind:?} {value}: {plain:?} vs {was:?}");
            }
            // A stop more or less is twice or half the light.
            for stops in [1.0f32, -2.0, 0.5] {
                frame.exposure.compensation = stops;
                let seen = at(&rig.show(&frame), 100, 100);
                let light = srgb_decode(f64::from(value)) * 2f64.powf(f64::from(stops));
                let wanted = kind.shown([light; 3]);
                for (shown, wanted) in seen.iter().zip(wanted) {
                    assert!(
                        (shown - wanted).abs() < 0.006,
                        "{kind:?} {value} at {stops} stops: {seen:?} vs {wanted:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn auto_exposure_brings_a_dark_room_up_and_adapts_to_the_day_outside() {
    let Some(mut rig) = Rig::new("room", Kind::Sdr) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let (mut frame, rects) = room(&mut rig, 7.0);
    let fixed = rig.show(&frame);
    rig.save("exposure-room-manual.png", &fixed);
    // With a fixed exposure the room is almost black and the window blown out.
    let (room_fixed, window_fixed) =
        (mean_luminance(&fixed, 10..90, 10..60), mean_luminance(&fixed, 240..280, 100..140));
    assert!(room_fixed < 0.25 && window_fixed > 0.95, "{room_fixed} {window_fixed}");
    // The first frame, or a cut, takes the exposure the room asks for: the trimmed
    // mean of its luminance at middle grey.
    frame.exposure = Exposure { cut: true, ..auto(1.0 / 60.0) };
    let adapted = rig.show(&frame);
    rig.save("exposure-room-auto.png", &adapted);
    let stops = rig.exposure(&frame);
    let wanted = goal(&lit(&rects), 0.1, 0.9, 0.18);
    assert!((stops - wanted).abs() < 0.08, "{stops} stops, the room asks for {wanted}");
    assert!(stops > 1.0, "{stops}");
    let (room_auto, window_auto) =
        (mean_luminance(&adapted, 10..90, 10..60), mean_luminance(&adapted, 240..280, 100..140));
    assert!(room_auto > room_fixed * 1.8, "the room is lifted: {room_fixed} to {room_auto}");
    assert!(window_auto > 0.95, "and the window stays bright: {window_auto}");
    // Compensation moves it, a stop at a time, and the limits hold it.
    let mut biased = frame.clone();
    biased.exposure.compensation = -1.0;
    rig.settle(&biased);
    assert!((rig.exposure(&biased) - (wanted - 1.0)).abs() < 0.08);
    let mut capped = frame.clone();
    capped.exposure.max = 0.5;
    rig.settle(&capped);
    assert_eq!(rig.exposure(&capped), 0.5);
    let mut floored = frame.clone();
    floored.exposure.min = (wanted + 1.0) as f32;
    floored.exposure.max = (wanted + 2.0) as f32;
    rig.settle(&floored);
    assert_eq!(rig.exposure(&floored), f64::from(floored.exposure.min));
    // The picture is dark again where exposure is fixed again, as it was.
    frame.exposure = Exposure::default();
    assert_eq!(rig.show(&frame), fixed);
}

#[test]
fn exposure_adapts_over_time_at_its_own_speeds_whatever_the_frame_rate() {
    let Some(mut rig) = Rig::new("adapt", Kind::Sdr) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    // Walking from the room into the day: the same camera, a few hundred times the light.
    let (mut inside, rects_in) = room(&mut rig, 7.0);
    let (mut outside, rects_out) = {
        let rects = vec![
            Rect { x: (-1.0, 1.0), y: (-1.0, 1.0), depth: 0.9, color: COLORS[2], gain: 1.0 },
            Rect { x: (-1.0, 1.0), y: (-1.0, -0.5), depth: 0.85, color: COLORS[1], gain: 0.4 },
        ];
        (rig.scene(&rects), rects)
    };
    let (dim, bright) = (goal(&lit(&rects_in), 0.1, 0.9, 0.18), goal(&lit(&rects_out), 0.1, 0.9, 0.18));
    assert!(dim > bright + 4.0, "{dim} {bright}");
    let settings = Exposure { brighten: 3.0, darken: 1.0, ..auto(0.0) };
    inside.exposure = Exposure { cut: true, dt: 1.0 / 60.0, ..settings };
    rig.settle(&inside);
    let start = rig.exposure(&inside);
    assert!((start - dim).abs() < 0.08, "{start} vs {dim}");
    // Out into the day: darkening, at the darken rate, over a second of frames of 1/20.
    outside.exposure = Exposure { dt: 0.05, ..settings };
    for step in 1..=20 {
        rig.settle(&outside);
        let seen = rig.exposure(&outside);
        let expected = bright + (start - bright) * (-0.05 * f64::from(step)).exp();
        assert!((seen - expected).abs() < 0.1, "darkening, step {step}: {seen} vs {expected}");
    }
    // And back in: brightening, three times as quickly, in frames of another length.
    let from = rig.exposure(&outside);
    inside.exposure = Exposure { dt: 0.1, ..settings };
    for step in 1..=8 {
        rig.settle(&inside);
        let seen = rig.exposure(&inside);
        let expected = dim + (from - dim) * (-0.3 * f64::from(step)).exp();
        assert!((seen - expected).abs() < 0.1, "brightening, step {step}: {seen} vs {expected}");
    }
    // The same second in 1 or 60 frames ends in the same place.
    let mut at_end = Vec::new();
    for frames in [1u32, 60] {
        inside.exposure = Exposure { cut: true, dt: 1.0 / 60.0, ..settings };
        rig.settle(&inside);
        outside.exposure = Exposure { dt: 1.0 / frames as f32, ..settings };
        for _ in 0..frames {
            rig.settle(&outside);
        }
        at_end.push(rig.exposure(&outside));
    }
    assert!((at_end[0] - at_end[1]).abs() < 0.03, "{at_end:?}");
    // A cut jumps to the day's exposure in one frame, and so does the host's reset.
    inside.exposure = Exposure { cut: true, dt: 1.0 / 60.0, ..settings };
    rig.settle(&inside);
    outside.exposure = Exposure { cut: true, dt: 1.0 / 60.0, ..settings };
    rig.settle(&outside);
    assert!((rig.exposure(&outside) - bright).abs() < 0.08);
    rig.settle(&inside);
    outside.exposure = Exposure { dt: 1.0 / 60.0, ..settings };
    rig.renderer.reset_exposure();
    rig.settle(&outside);
    assert!((rig.exposure(&outside) - bright).abs() < 0.08, "the host's reset takes the day at once");
    // Bloom takes the exposure into account: the lamp spills light in the room
    // once it is exposed up, and the picture is saved for a look.
    inside.exposure = Exposure { cut: true, dt: 1.0 / 60.0, ..settings };
    inside.quality.bloom = true;
    let glowing = rig.show(&inside);
    rig.save("exposure-room-auto-bloom.png", &glowing);
    inside.exposure = Exposure::default();
    let dark_glow = rig.show(&inside);
    let beside = |pixels: &[[f64; 3]]| mean_luminance(pixels, 108..118, 80..100);
    assert!(beside(&glowing) > beside(&dark_glow) + 0.05, "{} vs {}", beside(&glowing), beside(&dark_glow));
}

#[test]
fn the_grade_moves_the_picture_as_it_says_on_sdr_and_hdr_displays() {
    for kind in [Kind::Sdr, Kind::Hdr] {
        let Some(mut rig) = Rig::new(&format!("grade-{kind:?}"), kind) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let swatch = Rect { x: (-1.0, 1.0), y: (-1.0, 1.0), depth: 0.5, color: COLORS[6], gain: 1.0 };
        let mut frame = rig.scene(&[swatch]);
        let plain = rig.show(&frame);
        let base = at(&plain, 192, 128);
        // An explicit identity grade is the same picture to the pixel.
        frame.grade = Grade::default();
        assert_eq!(rig.show(&frame), plain);
        let light = swatch.color.map(|c| srgb_decode(f64::from(c)));
        let close = |seen: [f64; 3], wanted: [f64; 3], tolerance: f64, what: &str| {
            for (a, b) in seen.iter().zip(wanted) {
                assert!((a - b).abs() < tolerance, "{kind:?} {what}: {seen:?} vs {wanted:?}");
            }
        };
        // Gain 2 is a stop of light, and so is the grade's own exposure offset, and the
        // frame's exposure: three ways to the same pixel.
        let stop = kind.shown(light.map(|c| c * 2.0));
        frame.grade.gain = Vector { x: 2.0, y: 2.0, z: 2.0 };
        close(at(&rig.show(&frame), 192, 128), stop, 0.006, "gain 2");
        frame.grade = Grade { exposure: 1.0, ..Default::default() };
        close(at(&rig.show(&frame), 192, 128), stop, 0.006, "a stop of grade exposure");
        frame.grade = Grade::default();
        frame.exposure.compensation = 1.0;
        close(at(&rig.show(&frame), 192, 128), stop, 0.006, "a stop of exposure");
        frame.exposure = Exposure::default();
        // Contrast about the pivot leaves middle grey, and steepens what is off it.
        let middle = rig.scene(&[Rect { color: COLORS[5], gain: (srgb_encode(0.18) / 0.5) as f32, ..swatch }]);
        let mut contrasty = middle.clone();
        contrasty.grade.contrast = 2.0;
        let (before, after) = (at(&rig.show(&middle), 192, 128), at(&rig.show(&contrasty), 192, 128));
        close(after, before, 0.006, "contrast leaves the pivot");
        // Off the pivot it steepens: a brighter grey goes brighter, a darker one darker.
        let brighter = rig.scene(&[Rect { color: COLORS[5], gain: (srgb_encode(0.4) / 0.5) as f32, ..swatch }]);
        let mut steeper = brighter.clone();
        steeper.grade.contrast = 2.0;
        assert!(at(&rig.show(&steeper), 192, 128)[1] > at(&rig.show(&brighter), 192, 128)[1] + 0.05);
        // No contrast puts every color at the pivot's grey.
        frame.grade.contrast = 0.0;
        let flat = at(&rig.show(&frame), 192, 128);
        close(flat, [kind.shown([0.18; 3])[0]; 3], 0.006, "no contrast is the pivot's grey");
        frame.grade = Grade::default();
        // Desaturation is grey of the same luminance; warmth is redder and less blue.
        frame.grade.saturation = 0.0;
        let grey_out = at(&rig.show(&frame), 192, 128);
        close(grey_out, [grey_out[0]; 3], 0.004, "no saturation");
        frame.grade = Grade { temperature: 1.0, ..Default::default() };
        let warm = at(&rig.show(&frame), 192, 128);
        assert!(warm[0] > base[0] && warm[2] < base[2], "{kind:?} warm {warm:?} vs {base:?}");
        // Lift raises black; gamma raises midtones (above 1) and gain scales.
        let black = rig.scene(&[Rect { gain: 0.0, ..swatch }]);
        let mut lifted = black.clone();
        lifted.grade.lift = Vector { x: 0.1, y: 0.1, z: 0.1 };
        let dark = at(&rig.show(&black), 192, 128);
        close(dark, [0.0; 3], 0.002, "black");
        close(at(&rig.show(&lifted), 192, 128), kind.shown([0.1; 3]), 0.006, "lift");
        // A vignette darkens the corners by its strength in light and leaves the middle.
        frame.grade = Grade { vignette: 0.5, vignette_radius: 0.4, ..Default::default() };
        let vignetted = rig.show(&frame);
        close(at(&vignetted, 192, 128), base, 0.004, "the middle");
        close(at(&vignetted, 0, 0), kind.shown(light.map(|c| c * 0.5)), 0.03, "the corner");
        rig.save(&format!("grade-{kind:?}-ungraded.png"), &plain);
        // A graded look, for a look: the room, warmer, punchier, vignetted.
        let (mut room, _) = room(&mut rig, 7.0);
        room.exposure = Exposure { cut: true, ..auto(1.0 / 60.0) };
        let ungraded = rig.show(&room);
        rig.save(&format!("grade-{kind:?}-room-ungraded.png"), &ungraded);
        room.grade = Grade {
            temperature: 0.35,
            tint: -0.1,
            contrast: 1.35,
            saturation: 1.25,
            vibrance: 0.3,
            gain: Vector { x: 1.05, y: 1.0, z: 0.92 },
            lift: Vector { x: 0.004, y: 0.003, z: 0.006 },
            vignette: 0.55,
            vignette_radius: 0.35,
            ..Default::default()
        };
        let graded = rig.show(&room);
        rig.save(&format!("grade-{kind:?}-room-graded.png"), &graded);
        assert_ne!(graded, ungraded);
        let corner = |pixels: &[[f64; 3]]| mean_luminance(pixels, 0..12, 0..12);
        assert!(corner(&graded) < corner(&ungraded) * 0.8, "{} vs {}", corner(&graded), corner(&ungraded));
    }
}

#[test]
fn look_up_tables_grade_blend_and_weigh_on_sdr_and_hdr_displays() {
    for kind in [Kind::Sdr, Kind::Hdr] {
        let Some(mut rig) = Rig::new(&format!("tables-{kind:?}"), kind) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let swatch = Rect { x: (-1.0, 1.0), y: (-1.0, 1.0), depth: 0.5, color: COLORS[6], gain: 1.0 };
        let mut frame = rig.scene(&[swatch]);
        let plain = at(&rig.show(&frame), 192, 128);
        let light = swatch.color.map(|c| srgb_decode(f64::from(c)));
        // The tables are the catalog's last three materials, named 1 more than their index.
        let [identity, swap, warm] = [8u32, 9, 10].map(|index| index + 1);
        let close = |seen: [f64; 3], wanted: [f64; 3], tolerance: f64, what: &str| {
            for (a, b) in seen.iter().zip(wanted) {
                assert!((a - b).abs() < tolerance, "{kind:?} {what}: {seen:?} vs {wanted:?}");
            }
        };
        // The identity table changes nothing but its own 8-bit steps.
        frame.grade.lut_a = identity;
        close(at(&rig.show(&frame), 192, 128), plain, 0.03, "the identity table");
        // Red and blue swapped.
        frame.grade.lut_a = swap;
        let swapped = at(&rig.show(&frame), 192, 128);
        close(swapped, kind.shown([light[2], light[1], light[0]]), 0.04, "the swapping table");
        // Blending the identity table toward it goes half way in log light: the
        // geometric mean of each channel and its swap.
        frame.grade.lut_a = identity;
        frame.grade.lut_b = swap;
        frame.grade.lut_blend = 0.5;
        let mid = [(light[0] * light[2]).sqrt(), light[1], (light[2] * light[0]).sqrt()];
        close(at(&rig.show(&frame), 192, 128), kind.shown(mid), 0.04, "half way between two tables");
        frame.grade.lut_blend = 1.0;
        close(at(&rig.show(&frame), 192, 128), swapped, 0.01, "blended all the way");
        frame.grade.lut_blend = 0.0;
        close(
            at(&rig.show(&frame), 192, 128),
            at(
                &rig.show(&{
                    let mut f = frame.clone();
                    f.grade.lut_b = 0;
                    f
                }),
                192,
                128,
            ),
            0.002,
            "blend 0",
        );
        // The table's weight is how much of it applies.
        frame.grade.lut_a = swap;
        frame.grade.lut_b = 0;
        frame.grade.lut_weight = 0.0;
        close(at(&rig.show(&frame), 192, 128), plain, 0.002, "weight 0");
        frame.grade.lut_weight = 0.5;
        close(at(&rig.show(&frame), 192, 128), kind.shown(mid), 0.04, "weight a half");
        // A table alone, named second, stands in for the first.
        frame.grade.lut_a = 0;
        frame.grade.lut_b = swap;
        frame.grade.lut_weight = 1.0;
        close(at(&rig.show(&frame), 192, 128), swapped, 0.01, "a lone second table");
        // A look, for a look.
        let (mut room, _) = room(&mut rig, 7.0);
        room.exposure = Exposure { cut: true, ..auto(1.0 / 60.0) };
        let before = rig.show(&room);
        room.grade.lut_a = warm;
        let after = rig.show(&room);
        rig.save(&format!("lut-{kind:?}-room-plain.png"), &before);
        rig.save(&format!("lut-{kind:?}-room-warm.png"), &after);
        assert_ne!(before, after);
    }
}
