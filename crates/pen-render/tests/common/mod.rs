//! What the GPU tests share: a headless device, a plain frame, an offscreen
//! target and reading its middle pixel back.
#![allow(dead_code)]
use std::time::{Duration, Instant};

use pen_protocol::*;
use pen_render::{RenderTarget, Renderer};

pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
pub const IDENTITY: Matrix = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

pub fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    pollster::block_on(adapter.request_device(&Default::default())).ok()
}

/// A frame of these draws seen through identity matrices, every effect off.
pub fn frame(draws: Vec<Draw>) -> RenderFrame {
    let zero = Vector::default();
    RenderFrame {
        hide_cursor: false,
        ambient: zero,
        aberration: Aberration { x: 0.0, y: 0.0 },
        blur: 0.0,
        clear: Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        quality: Quality {
            msaa: 1,
            fxaa: false,
            scale: 1.0,
            filtering: 1,
            occlusion: false,
            sky: false,
            effects: false,
            bloom: false,
        },
        sky: Sky { zenith: zero, horizon: zero, ground: zero, sun: Vector { x: 0.0, y: 1.0, z: 0.0 }, night: 0.0 },
        fog: Fog { radius: 0.0, density: 0.0, height: 1.0 },
        clouds: Clouds { cover: 0.0, drift: Drift { x: 0.0, z: 0.0 } },
        shadow: Shadow { view: IDENTITY, projection: IDENTITY, texel: 0.0, texels: 0 },
        draws,
        decals: Vec::new(),
        solids: Vec::new(),
        sprites: Vec::new(),
        insets: Vec::new(),
        fields: Vec::new(),
        ui: Vec::new(),
        services: Vec::new(),
        light_color: Vector { x: 1.0, y: 1.0, z: 1.0 },
        light_direction: Vector { x: 0.0, y: 1.0, z: 0.0 },
        view: IDENTITY,
        projection: IDENTITY,
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

pub struct Offscreen(pub wgpu::Texture);

impl Offscreen {
    pub fn new(device: &wgpu::Device, size: u32) -> Self {
        Self(device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }))
    }
}

impl RenderTarget for Offscreen {
    fn size(&self) -> (u32, u32) {
        (self.0.width(), self.0.height())
    }
    fn view(&mut self) -> Option<wgpu::TextureView> {
        Some(self.0.create_view(&Default::default()))
    }
    fn present(&mut self) {}
}

/// The output's middle pixel, as RGBA.
pub fn middle(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> [u8; 4] {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: texture.width() / 2, y: texture.height() / 2, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: None, rows_per_image: None },
        },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let texel = buffer.slice(..4).get_mapped_range().unwrap();
    [texel[2], texel[1], texel[0], texel[3]]
}

/// Submits `frame` until the events so far satisfy `done` (20 s at most),
/// then once more, and returns the events.
pub fn pump(
    renderer: &mut Renderer,
    frame: &RenderFrame,
    catalog: &Catalog,
    target: &mut Offscreen,
    done: impl Fn(&[GameEvent]) -> bool,
) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let started = Instant::now();
    loop {
        renderer.submit(frame, catalog, target).unwrap();
        events.extend(renderer.take_events());
        if done(&events) {
            break;
        }
        assert!(started.elapsed() < Duration::from_secs(20), "timed out with events {events:?}");
        std::thread::sleep(Duration::from_millis(5));
    }
    renderer.submit(frame, catalog, target).unwrap();
    events.extend(renderer.take_events());
    events
}

/// Whether `events` have `key` ready at `revision`.
pub fn ready(events: &[GameEvent], key: &str, revision: u64) -> bool {
    events.iter().any(|event| matches!(event, GameEvent::AssetReady { key: k, revision: r, .. } if k == key && *r == revision))
}

/// Whether `events` have `key` failing at `revision`.
pub fn failed(events: &[GameEvent], key: &str, revision: u64) -> bool {
    events.iter().any(|event| matches!(event, GameEvent::AssetFailed { key: k, revision: r, .. } if k == key && *r == revision))
}
