//! Renders gdev render packets headlessly and writes each as a PNG; the
//! measurements in docs/render-performance.md come from it.
//!
//! ```text
//! cargo run --release -p pen-render --example render_packet -- [options] [packet numbers...]
//!   --assets DIR    asset root holding catalog.json (default dist/gdev/assets)
//!   --packets DIR   NNNN.f32 packets and schema.txt (default dist/packets)
//!   --out DIR       where the PNGs go (default target/render_packet)
//!   --size WxH      output size (default 1280x720)
//!   --veil          draw the loading veil (0.6 s), as gdev does
//!   --synthetic     render a made-up frame instead of packets
//!   --sequence      settle on the first packet, as pen does on its first
//!                   frame, then submit each later one once, in order, timing
//!                   each (all packets if none are named)
//!   --pipelines DIR keep compiled pipelines there between runs (Vulkan)
//!   --msaa N, --fxaa, --filtering N, --scale S, --blur B, --aberration X,Y
//!                   override the frames' quality and effects
//!   --robots N      add an N x N grid of instanced walking robots
//!   --early         also write the first frame submitted (NNNN-early.png)
//! ```
//!
//! Otherwise each frame is submitted until its assets have loaded and the
//! renderer is quiet, as pen settles its first frame, then timed over 30
//! more submits and read back. `RUST_LOG=debug` (any value) also prints the
//! renderer's pipeline compile times.
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use pen_protocol::*;
use pen_render::{RenderTarget, Renderer, RendererOptions};

/// A change to every frame rendered, from the command line.
type Override = Box<dyn Fn(&mut RenderFrame)>;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

struct Offscreen {
    texture: wgpu::Texture,
}

impl RenderTarget for Offscreen {
    fn size(&self) -> (u32, u32) {
        (self.texture.width(), self.texture.height())
    }
    fn view(&mut self) -> Option<wgpu::TextureView> {
        Some(self.texture.create_view(&Default::default()))
    }
    fn present(&mut self) {}
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Result<image::RgbaImage> {
    let (width, height) = (texture.width(), texture.height());
    let row = (width * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read back"),
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
    buffer.slice(..).map_async(wgpu::MapMode::Read, |result| result.expect("map read-back buffer"));
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let mapped = buffer.slice(..).get_mapped_range()?;
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let start = (y * row) as usize;
        for texel in mapped[start..start + (width * 4) as usize].as_chunks::<4>().0 {
            // BGRA, as the swapchain lays it out.
            pixels.extend([texel[2], texel[1], texel[0], 255]);
        }
    }
    Ok(image::RgbaImage::from_raw(width, height, pixels).unwrap())
}

fn read_packet(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(bytes.as_chunks::<4>().0.iter().map(|&lanes| f32::from_le_bytes(lanes)).collect())
}

fn vector(x: f32, y: f32, z: f32) -> Vector {
    Vector { x, y, z }
}

/// A camera at `eye` looking at the origin, and a perspective projection
/// into WebGPU's 0..1 depth.
fn camera(eye: [f32; 3], aspect: f32) -> (Matrix, Matrix) {
    let length = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let normalize = |v: [f32; 3]| {
        let l = length(v);
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let cross =
        |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let back = normalize(eye);
    let right = normalize(cross([0.0, 1.0, 0.0], back));
    let up = cross(back, right);
    let view = [
        right[0],
        up[0],
        back[0],
        0.0,
        right[1],
        up[1],
        back[1],
        0.0,
        right[2],
        up[2],
        back[2],
        0.0,
        -dot(right, eye),
        -dot(up, eye),
        -dot(back, eye),
        1.0,
    ];
    let (near, far, f) = (0.1f32, 200.0f32, 1.0 / (0.5f32).tan());
    let projection = [
        f / aspect,
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
    (view, projection)
}

/// A made-up frame: a few meshes on a brush floor under a sky, with a sprite.
fn synthetic(catalog: &Catalog, (width, height): (u32, u32)) -> RenderFrame {
    let find = |list: &[String], key: &str| {
        list.iter().position(|entry| entry == key).map_or(0, |index| u32::try_from(index).unwrap())
    };
    let (view, projection) = camera([6.0, 4.0, 8.0], width as f32 / height as f32);
    let translate = |x: f32, y: f32, z: f32| [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0];
    let cube = find(&catalog.meshes, "cube.mesh.json");
    let blue = find(&catalog.materials, "blue.material.json");
    let draws = (0..5)
        .map(|index| Draw {
            material: blue,
            mesh: cube,
            transform: translate(index as f32 * 1.5 - 3.0, 0.5, 0.0),
            parameters: None,
            extra: None,
        })
        .collect();
    let face = |x: f32, y: f32, z: f32, distance: f32| SolidFace { normal: vector(x, y, z), distance, material: blue };
    let floor = Solid {
        faces: vec![
            face(0.0, 1.0, 0.0, 0.0),
            face(0.0, -1.0, 0.0, 0.5),
            face(1.0, 0.0, 0.0, 8.0),
            face(-1.0, 0.0, 0.0, 8.0),
            face(0.0, 0.0, 1.0, 8.0),
            face(0.0, 0.0, -1.0, 8.0),
        ],
    };
    RenderFrame {
        hide_cursor: false,
        ambient: vector(0.25, 0.27, 0.3),
        aberration: Aberration { x: 0.0, y: 0.0 },
        blur: 0.0,
        clear: Color { r: 0.4, g: 0.55, b: 0.75, a: 1.0 },
        quality: Quality {
            msaa: 4,
            fxaa: true,
            scale: 1.0,
            filtering: 4,
            occlusion: true,
            sky: true,
            effects: true,
            bloom: true,
        },
        sky: Sky {
            zenith: vector(0.2, 0.4, 0.8),
            horizon: vector(0.7, 0.8, 0.9),
            ground: vector(0.3, 0.3, 0.3),
            sun: vector(0.4, 0.8, 0.45),
            night: 0.0,
        },
        fog: Fog { radius: 30.0, density: 0.02, height: 10.0 },
        clouds: Clouds { cover: 0.0, drift: Drift { x: 0.0, z: 0.0 } },
        shadow: Shadow { view: IDENTITY, projection: IDENTITY, texel: 0.0, texels: 0 },
        draws,
        decals: Vec::new(),
        solids: vec![floor],
        sprites: vec![Sprite {
            x: 20.0,
            y: 20.0,
            width: 200.0,
            height: 40.0,
            u0: 0.0,
            v0: 0.0,
            u1: 1.0,
            v1: 1.0,
            color: [1.0, 1.0, 1.0, 0.6],
            material: find(&catalog.materials, "white.material.json"),
            parameters: [0.0; 4],
        }],
        insets: Vec::new(),
        fields: Vec::new(),
        ui: Vec::new(),
        services: Vec::new(),
        light_color: vector(1.0, 0.95, 0.9),
        light_direction: vector(0.4, 0.8, 0.45),
        view,
        projection,
        sounds: Vec::new(),
        buses: Vec::new(),
        voices: Vec::new(),
        preferences: None,
        load_requested: false,
        save_requested: false,
        scene_load_requested: false,
        scene_save_requested: false,
        checkpoint_requested: false,
        undo_requested: false,
        redo_requested: false,
        exit_requested: false,
    }
}

const IDENTITY: Matrix = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

/// Prints the renderer's log to stderr: warnings, and with RUST_LOG set,
/// its pipeline compile times.
struct Stderr;

impl log::Log for Stderr {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.target().starts_with("pen_render")
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}: {}", record.level(), record.args());
        }
    }
    fn flush(&self) {}
}

fn main() -> Result<()> {
    log::set_logger(&Stderr).map_err(|_| anyhow::anyhow!("a logger is already set"))?;
    log::set_max_level(if std::env::var_os("RUST_LOG").is_some() {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Warn
    });
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut assets = root.join("dist/gdev/assets");
    let mut packets = root.join("dist/packets");
    let mut out = root.join("target/render_packet");
    let mut size = (1280, 720);
    let mut veil = None;
    let mut use_synthetic = false;
    let mut numbers = Vec::new();
    let mut overrides: Vec<Override> = Vec::new();
    let mut robots = 0;
    let mut early = false;
    let mut sequence = false;
    let mut pipelines = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--assets" => assets = PathBuf::from(args.next().context("--assets DIR")?),
            "--packets" => packets = PathBuf::from(args.next().context("--packets DIR")?),
            "--out" => out = PathBuf::from(args.next().context("--out DIR")?),
            "--size" => {
                let value = args.next().context("--size WxH")?;
                let (w, h) = value.split_once('x').context("--size WxH")?;
                size = (w.parse()?, h.parse()?);
            }
            "--veil" => veil = Some(0.6),
            "--synthetic" => use_synthetic = true,
            "--msaa" => {
                let msaa: u32 = args.next().context("--msaa N")?.parse()?;
                overrides.push(Box::new(move |frame| frame.quality.msaa = msaa));
            }
            "--fxaa" => overrides.push(Box::new(|frame| frame.quality.fxaa = true)),
            "--filtering" => {
                let filtering: u32 = args.next().context("--filtering N")?.parse()?;
                overrides.push(Box::new(move |frame| frame.quality.filtering = filtering));
            }
            "--scale" => {
                let scale: f32 = args.next().context("--scale S")?.parse()?;
                overrides.push(Box::new(move |frame| frame.quality.scale = scale));
            }
            "--blur" => {
                let blur: f32 = args.next().context("--blur B")?.parse()?;
                overrides.push(Box::new(move |frame| frame.blur = blur));
            }
            "--aberration" => {
                let value = args.next().context("--aberration X,Y")?;
                let (x, y) = value.split_once(',').context("--aberration X,Y")?;
                let (x, y): (f32, f32) = (x.parse()?, y.parse()?);
                overrides.push(Box::new(move |frame| frame.aberration = Aberration { x, y }));
            }
            "--robots" => robots = args.next().context("--robots N")?.parse()?,
            "--early" => early = true,
            "--sequence" => sequence = true,
            "--pipelines" => pipelines = Some(PathBuf::from(args.next().context("--pipelines DIR")?)),
            number => numbers.push(number.parse::<u32>().with_context(|| format!("unknown argument {number}"))?),
        }
    }
    let catalog = pen_protocol::boundary::read_catalog(&assets)?;
    let schema_path = packets.join("schema.txt");
    if !use_synthetic && !schema_path.exists() {
        eprintln!("no packets at {}; rendering a synthetic frame", packets.display());
        use_synthetic = true;
    }
    let mut frames: Vec<(String, RenderFrame)> = Vec::new();
    // A grid of robots in their walk poses, as the game draws them: one
    // instanced batch for all eight pose materials.
    let robot_mesh =
        catalog.meshes.iter().position(|key| key == "robot3d.mesh.json").map(|index| u32::try_from(index).unwrap());
    let add_robots = |frame: &mut RenderFrame| {
        let Some(mesh) = robot_mesh else { return };
        for index in 0..robots * robots {
            let (column, row) = ((index % robots) as f32, (index / robots) as f32);
            let spread = robots as f32 - 1.0;
            let key = format!("robot.walk_{:02}.material.json", index % 8);
            let Some(material) = catalog.materials.iter().position(|entry| *entry == key) else { continue };
            let material = u32::try_from(material).unwrap();
            // Six times life size, so they show from the god view.
            let mut transform = IDENTITY.map(|value| value * 6.0);
            transform[15] = 1.0;
            transform[12] = (column - spread / 2.0) * 7.0;
            transform[14] = (row - spread / 2.0) * 7.0 + 10.0;
            frame.draws.push(Draw { material, mesh, transform, parameters: None, extra: None });
        }
    };
    if use_synthetic {
        frames.push(("synthetic".into(), synthetic(&catalog, size)));
    } else {
        let schema = pen_protocol::boundary::validate_schema(std::fs::read_to_string(&schema_path)?.trim().parse()?)?;
        if numbers.is_empty() && sequence {
            numbers = (0..).take_while(|number| packets.join(format!("{number:04}.f32")).exists()).collect();
        } else if numbers.is_empty() {
            numbers = vec![0, 30, 59, 60, 90, 119];
        }
        for number in numbers {
            let lanes = read_packet(&packets.join(format!("{number:04}.f32")))?;
            let decoded = pen_protocol::boundary::decode_frame(&lanes, schema, &catalog)?;
            frames.push((format!("{number:04}"), decoded.render));
        }
    }

    for (_, frame) in &mut frames {
        for set in &overrides {
            set(frame);
        }
        add_robots(frame);
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))?;
    eprintln!("adapter: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);
    let features = pen_render::optional_features(&adapter);
    let descriptor = wgpu::DeviceDescriptor { required_features: features, ..Default::default() };
    let (device, queue) = pollster::block_on(adapter.request_device(&descriptor))?;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output"),
        size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let mut target = Offscreen { texture };
    let started = Instant::now();
    let options = RendererOptions { format: FORMAT, assets, veil, pipeline_cache: pipelines };
    let mut renderer = Renderer::new(device.clone(), queue.clone(), options)?;
    renderer.preload(&catalog);
    std::fs::create_dir_all(&out)?;
    let report = |renderer: &mut Renderer| {
        for event in renderer.take_events() {
            if let GameEvent::AssetFailed { key, kind, message, .. } = event {
                eprintln!("{kind} {key}: {message}");
            }
        }
    };
    let settle = |renderer: &mut Renderer, target: &mut Offscreen, name: &str, frame: &RenderFrame| -> Result<()> {
        let begun = Instant::now();
        let mut submitted = 0;
        loop {
            renderer.submit(frame, &catalog, target)?;
            device.poll(wgpu::PollType::wait_indefinitely())?;
            submitted += 1;
            if early && submitted == 1 {
                read_back(&device, &queue, &target.texture)?.save(out.join(format!("{name}-early.png")))?;
            }
            report(renderer);
            if (renderer.quiet() && renderer.pending() == 0) || begun.elapsed() > Duration::from_secs(60) {
                break;
            }
            std::thread::sleep(Duration::from_millis(4));
        }
        eprintln!("{name}: settled in {:.0} ms ({submitted} submits)", begun.elapsed().as_secs_f64() * 1000.0);
        Ok(())
    };
    if sequence {
        let Some(((first_name, first), rest)) = frames.split_first() else { bail!("nothing to render") };
        settle(&mut renderer, &mut target, first_name, first)?;
        let mut times = Vec::new();
        for (name, frame) in rest {
            let begun = Instant::now();
            renderer.submit(frame, &catalog, &mut target)?;
            let submit = begun.elapsed().as_secs_f64() * 1000.0;
            device.poll(wgpu::PollType::wait_indefinitely())?;
            let total = begun.elapsed().as_secs_f64() * 1000.0;
            report(&mut renderer);
            eprintln!("{name}: submit {submit:.2} ms, with the GPU {total:.2} ms, {:?}", renderer.stats());
            times.push((total, name));
            // Frames come a display refresh apart, as in the game.
            std::thread::sleep(Duration::from_millis(16));
        }
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        let slowest: Vec<String> = times.iter().take(5).map(|(total, name)| format!("{name} {total:.2} ms")).collect();
        eprintln!("slowest: {}", slowest.join(", "));
        read_back(&device, &queue, &target.texture)?.save(out.join("sequence.png"))?;
        return Ok(());
    }
    for (name, frame) in &frames {
        settle(&mut renderer, &mut target, name, frame)?;
        // How long a settled frame takes to encode and submit.
        let timed = Instant::now();
        for _ in 0..30 {
            renderer.submit(frame, &catalog, &mut target)?;
            device.poll(wgpu::PollType::wait_indefinitely())?;
        }
        let per_frame = timed.elapsed().as_secs_f64() / 30.0 * 1000.0;
        let picture = read_back(&device, &queue, &target.texture)?;
        let path = out.join(format!("{name}.png"));
        picture.save(&path)?;
        eprintln!(
            "{}: {:.2} ms a frame (GPU included), {:?}, {:.1}s in all",
            path.display(),
            per_frame,
            renderer.stats(),
            started.elapsed().as_secs_f64()
        );
    }
    if frames.is_empty() {
        bail!("nothing to render");
    }
    Ok(())
}
