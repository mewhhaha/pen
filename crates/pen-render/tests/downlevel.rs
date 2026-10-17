//! The renderer on a WebGL2-class device (src/downlevel.rs): a real packet of
//! the game, with sprites and decals, drawn with WebGL2's limits (no storage
//! buffers) matches it drawn with WebGPU's, without a GPU error. With those
//! limits on the default adapter, and on wgpu's GL backend where there is
//! one. Skips without the packets and the game's bundle (`just packets`) or
//! an adapter. `PEN_PACKETS` names another packet directory than
//! dist/packets.
#![cfg(not(target_arch = "wasm32"))]
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pen_protocol::*;
use pen_render::{RenderTarget, Renderer, RendererOptions};

// What a WebGL2 canvas gives.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SIZE: (u32, u32) = (640, 360);

fn dist() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dist")
}

/// The first packet drawing both sprites and decals, by its number, with
/// the catalog; None without the fixtures.
fn packet() -> Option<(String, RenderFrame, Catalog)> {
    let packets = std::env::var_os("PEN_PACKETS").map_or_else(|| dist().join("packets"), PathBuf::from);
    let catalog = pen_protocol::boundary::read_catalog(&dist().join("gdev/assets")).ok()?;
    let schema = std::fs::read_to_string(packets.join("schema.txt")).ok()?;
    let schema = pen_protocol::boundary::validate_schema(schema.trim().parse().ok()?).ok()?;
    (0..).map_while(|number| Some((number, std::fs::read(packets.join(format!("{number:04}.f32"))).ok()?))).find_map(
        |(number, bytes)| {
            let lanes: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|&lanes| f32::from_le_bytes(lanes)).collect();
            let frame = pen_protocol::boundary::decode_frame(&lanes, schema, &catalog).ok()?.render;
            (!frame.sprites.is_empty() && !frame.decals.is_empty())
                .then(|| (format!("{number:04}"), frame, catalog.clone()))
        },
    )
}

struct Offscreen(wgpu::Texture);

impl RenderTarget for Offscreen {
    fn size(&self) -> (u32, u32) {
        (self.0.width(), self.0.height())
    }
    fn view(&mut self) -> Option<wgpu::TextureView> {
        Some(self.0.create_view(&Default::default()))
    }
    fn present(&mut self) {}
}

/// The output's pixels, RGBA, row after row.
fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let row = (width * 4).div_ceil(256) * 256;
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
    let mut pixels = Vec::new();
    for y in 0..height as usize {
        let start = y * row as usize;
        pixels.extend_from_slice(&mapped[start..start + width as usize * 4]);
    }
    pixels
}

/// Prints the renderer's warnings and anyone's errors (wgpu's GL backend
/// logs what its driver says) to stderr.
struct Stderr;

impl log::Log for Stderr {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Error
            || (metadata.level() <= log::Level::Warn && metadata.target().starts_with("pen_render"))
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{} {}: {}", record.level(), record.target(), record.args());
        }
    }
    fn flush(&self) {}
}

/// What drawing a frame on a device came to.
struct Drawn {
    pixels: Vec<u8>,
    /// Uncaptured GPU errors.
    errors: Vec<String>,
    /// The assets that failed to load.
    failed: BTreeSet<String>,
}

/// `frame` drawn on `adapter` with `limits`, with the catalog preloaded as
/// the hosts do, once its assets have settled.
fn draw(adapter: &wgpu::Adapter, limits: wgpu::Limits, frame: &RenderFrame, catalog: &Catalog) -> Drawn {
    let descriptor = wgpu::DeviceDescriptor { required_limits: limits, ..Default::default() };
    let (device, queue) = pollster::block_on(adapter.request_device(&descriptor)).unwrap();
    let errors = Arc::new(Mutex::new(Vec::new()));
    let uncaptured = errors.clone();
    device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| uncaptured.lock().unwrap().push(error.to_string())));
    let mut target = Offscreen(device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: SIZE.0, height: SIZE.1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }));
    let options = RendererOptions {
        format: FORMAT,
        display: pen_render::Display::SDR,
        assets: dist().join("gdev/assets"),
        veil: None,
        pipeline_cache: None,
    };
    let mut renderer = Renderer::new(device.clone(), queue.clone(), options).unwrap();
    renderer.preload(catalog);
    let mut failed = BTreeSet::new();
    let started = Instant::now();
    loop {
        renderer.submit(frame, catalog, &mut target).unwrap();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        for event in renderer.take_events() {
            if let GameEvent::AssetFailed { key, message, .. } = event {
                eprintln!("failed: {key}: {}", message.lines().last().unwrap_or_default().trim());
                failed.insert(key);
            }
        }
        if renderer.quiet() && renderer.pending() == 0 {
            break;
        }
        let quiet = renderer.quiet();
        let pending = renderer.pending();
        assert!(started.elapsed() < Duration::from_secs(120), "not settled: quiet {quiet}, {pending} pending");
        std::thread::sleep(Duration::from_millis(4));
    }
    let pixels = read_back(&device, &queue, &target.0);
    let errors = errors.lock().unwrap().clone();
    Drawn { pixels, errors, failed }
}

/// How two images differ: the largest and the mean difference of a colour
/// channel, and the share of pixels with a channel more than 2 apart.
fn difference(a: &[u8], b: &[u8]) -> (u8, f64, f64) {
    let (mut largest, mut sum, mut apart) = (0, 0u64, 0usize);
    for (a, b) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
        let channels = [0, 1, 2].map(|channel| a[channel].abs_diff(b[channel]));
        largest = largest.max(channels.into_iter().max().unwrap());
        sum += channels.iter().map(|&channel| u64::from(channel)).sum::<u64>();
        apart += usize::from(channels.iter().any(|&channel| channel > 2));
    }
    let pixels = a.len() / 4;
    (largest, sum as f64 / (pixels * 3) as f64, apart as f64 / pixels as f64)
}

fn save(name: &str, pixels: &[u8]) {
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join("downlevel");
    std::fs::create_dir_all(&directory).unwrap();
    let image = image::RgbaImage::from_raw(SIZE.0, SIZE.1, pixels.to_vec()).unwrap();
    image.save(directory.join(format!("{name}.png"))).unwrap();
}

#[test]
fn draws_a_packet_with_webgl2_limits_as_with_storage_buffers() {
    let _ = log::set_logger(&Stderr).map(|()| log::set_max_level(log::LevelFilter::Warn));
    let Some((number, mut frame, catalog)) = packet() else {
        eprintln!("skipped: no packet with sprites and decals in dist/packets, or no dist/gdev (just packets)");
        return;
    };
    // One sample a pixel, as a WebGL2 device draws, and the aberration on,
    // which reads the stencil's marks. Enough sprites and decals that their
    // uniform arrays take several windows: copies of the frame's own, drawn
    // over them.
    frame.quality.msaa = 1;
    frame.quality.effects = true;
    frame.aberration = Aberration { x: 0.004, y: 0.0 };
    let (sprites, decals) = (frame.sprites.clone(), frame.decals.clone());
    while frame.sprites.len() < 600 {
        frame.sprites.extend_from_slice(&sprites);
    }
    while frame.decals.len() < 250 {
        frame.decals.extend_from_slice(&decals);
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("skipped: no GPU adapter");
        return;
    };
    let info = adapter.get_info();
    eprintln!(
        "packet {number}: {} sprites, {} decals on {} ({:?})",
        frame.sprites.len(),
        frame.decals.len(),
        info.name,
        info.backend
    );
    // Some of the game's assets may fail anywhere (a sprite shader asked for
    // as a mesh's): those must be the same with WebGL2's limits.
    let reference = draw(&adapter, wgpu::Limits::default(), &frame, &catalog);
    assert!(reference.errors.is_empty(), "with storage buffers: {:#?}", reference.errors);
    save("storage", &reference.pixels);
    let webgl2 = wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
    let limited = draw(&adapter, webgl2, &frame, &catalog);
    assert!(limited.errors.is_empty(), "with WebGL2's limits: {:#?}", limited.errors);
    assert_eq!(limited.failed, reference.failed, "with WebGL2's limits");
    save("limited", &limited.pixels);
    // The same arithmetic on the same adapter: uniform arrays read what the
    // storage buffers held, the depth copy is exact (24 bits in a 32-bit
    // float) and the marks are the stencil's bit. Allowing for a driver
    // free to fuse or reorder float operations in the rewritten shaders,
    // a few pixels may move a level or two; it measures 0 here.
    let (largest, mean, apart) = difference(&reference.pixels, &limited.pixels);
    eprintln!("WebGL2's limits: largest difference {largest}, mean {mean:.4}, {:.3}% of pixels > 2", apart * 100.0);
    assert!(mean < 0.05 && apart < 0.001, "largest {largest}, mean {mean}, apart {apart}");

    // wgpu's GL backend (GLES through EGL here), as near WebGL2 as a native
    // test comes: GLSL from naga, a GL driver, and its downlevel flags, which
    // WebGL2's limits on another adapter do not bring (no read-only
    // depth-stencil attachments, no view formats). Without wgpu's checks of
    // indirect draws, which pen makes none of: they are a compute shader,
    // which GLES drivers may lack the GLSL version for (and WebGL2 has no
    // compute at all).
    let gl = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::GL,
        flags: wgpu::InstanceFlags::default() - wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let Ok(adapter) = pollster::block_on(gl.request_adapter(&Default::default())) else {
        eprintln!("skipped the GL backend: no GL adapter");
        return;
    };
    eprintln!("GL: {}", adapter.get_info().name);
    let limits = wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
    let gl = draw(&adapter, limits, &frame, &catalog);
    assert!(gl.errors.is_empty(), "on GL: {:#?}", gl.errors);
    assert_eq!(gl.failed, reference.failed, "on GL");
    // Its picture is not compared: NVIDIA's GLES through EGL draws this
    // frame mostly dark, without an error, while Chromium's WebGL2 draws the
    // game as WebGPU does (crates/pen-web/tests/browser.mjs, WebGPU hidden).
    save("gl", &gl.pixels);
}
