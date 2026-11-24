//! What the tests that read a picture back share: a renderer over a directory
//! of assets written for the test, frames drawn until everything they name is
//! ready, the picture's pixels, and what the present pass shows of a light.
//! Frames have no effects on, so a pixel is the scene's colour as it is
//! (below 0.9) and clips toward 1 above.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "picture coordinates and 8-bit colours of a 128-pixel test image"
)]
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use pen_protocol::*;
use pen_render::{Display, Renderer, RendererOptions};
use serde_json::json;

use super::{FORMAT, IDENTITY, Offscreen, device, pixels};

pub const SIZE: u32 = 128;

/// What a frame with its effects off shows, as 8 bits, of light `value`: the
/// value as it is up to 0.9, rolling off toward 1 above (the present pass's
/// highlights).
pub fn shown(value: f32) -> u8 {
    let value = f64::from(value);
    let toned = if value > 0.9 { 1.0 - 0.01 / (value + 0.1 - 0.9) } else { value };
    (toned.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub fn moved(x: f32, y: f32) -> Matrix {
    let mut matrix = IDENTITY;
    matrix[12] = x;
    matrix[13] = y;
    matrix
}

pub struct Rig {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub root: PathBuf,
    pub renderer: Renderer,
    pub catalog: Catalog,
    pub target: Offscreen,
}

pub fn write(root: &Path, key: &str, bytes: impl AsRef<[u8]>) {
    std::fs::write(root.join(key), bytes).unwrap();
}

impl Rig {
    /// A renderer over a directory of these `files` and a catalog of the meshes and materials.
    pub fn new(name: &str, meshes: &[&str], materials: &[&str], files: &[(&str, Vec<u8>)]) -> Option<Rig> {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter; skipping");
            return None;
        };
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("picture-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        write(&root, "catalog.json", json!({"meshes": meshes, "materials": materials}).to_string());
        for (key, bytes) in files {
            write(&root, key, bytes);
        }
        let catalog = boundary::read_catalog(&root).unwrap();
        let renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererOptions {
                format: FORMAT,
                display: Display::SDR,
                assets: root.clone(),
                veil: None,
                pipeline_cache: None,
            },
        )
        .unwrap();
        let target = Offscreen::new(&device, SIZE);
        Some(Rig { device, queue, root, renderer, catalog, target })
    }

    /// Draws `frame` once everything it names is loaded and compiled, and
    /// reads the picture back.
    pub fn show(&mut self, frame: &RenderFrame) -> Vec<[u8; 4]> {
        let started = Instant::now();
        loop {
            self.renderer.submit(frame, &self.catalog, &mut self.target).unwrap();
            for event in self.renderer.take_events() {
                if let GameEvent::AssetFailed { key, message, .. } = event {
                    panic!("{key} failed: {message}");
                }
            }
            if !self.renderer.waiting() && self.renderer.pending() == 0 {
                break;
            }
            assert!(started.elapsed() < Duration::from_secs(30), "assets still loading");
            std::thread::sleep(Duration::from_millis(5));
        }
        self.renderer.submit(frame, &self.catalog, &mut self.target).unwrap();
        pixels(&self.device, &self.queue, &self.target.0)
    }
}

/// The pixel showing the clip-space point (x, y).
pub fn at(pixels: &[[u8; 4]], x: f32, y: f32) -> [u8; 3] {
    let column = (((x + 1.0) / 2.0 * SIZE as f32) as usize).min(SIZE as usize - 1);
    let row = (((1.0 - y) / 2.0 * SIZE as f32) as usize).min(SIZE as usize - 1);
    let [r, g, b, _] = pixels[row * SIZE as usize + column];
    [r, g, b]
}

pub fn near(got: [u8; 3], want: [u8; 3]) -> bool {
    got.iter().zip(want).all(|(&got, want)| got.abs_diff(want) <= 2)
}
