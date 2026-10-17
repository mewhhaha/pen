//! What the importer's tests share: a small glTF model built here, as an
//! exporter would write it, and (for the GPU tests) a headless device, an
//! offscreen target and its pixels read back.
#![allow(dead_code)]
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

/// A glTF file's JSON and its one buffer, built a view and accessor at a
/// time.
struct Builder {
    bytes: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

impl Builder {
    fn view(&mut self, data: &[u8]) -> usize {
        while !self.bytes.len().is_multiple_of(4) {
            self.bytes.push(0);
        }
        self.views.push(json!({"buffer": 0, "byteOffset": self.bytes.len(), "byteLength": data.len()}));
        self.bytes.extend_from_slice(data);
        self.views.len() - 1
    }

    /// An accessor of floats, `kind` (VEC3, ...) wide.
    fn floats(&mut self, values: &[f32], kind: &str, bounds: bool) -> usize {
        let width = match kind {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            _ => 16,
        };
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        let view = self.view(&bytes);
        let mut accessor =
            json!({"bufferView": view, "componentType": 5126, "count": values.len() / width, "type": kind});
        if bounds {
            let column = |axis: usize| values.iter().skip(axis).step_by(width).copied();
            accessor["min"] = (0..width).map(|axis| column(axis).fold(f32::INFINITY, f32::min)).collect();
            accessor["max"] = (0..width).map(|axis| column(axis).fold(f32::NEG_INFINITY, f32::max)).collect();
        }
        self.accessors.push(accessor);
        self.accessors.len() - 1
    }

    fn shorts(&mut self, values: &[u16], kind: &str) -> usize {
        let width = if kind == "VEC4" { 4 } else { 1 };
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        let view = self.view(&bytes);
        self.accessors
            .push(json!({"bufferView": view, "componentType": 5123, "count": values.len() / width, "type": kind}));
        self.accessors.len() - 1
    }
}

/// A 2 x 2 PNG of one color.
pub fn png(color: [u8; 4]) -> Vec<u8> {
    let image = image::RgbaImage::from_pixel(2, 2, image::Rgba(color));
    let mut bytes = std::io::Cursor::new(Vec::new());
    image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
    bytes.into_inner()
}

/// An arm: a column 0.2 wide and 2 tall bound to two joints, `base` at its
/// foot and `elbow` halfway up, under an `armature` node; and beside it a
/// plinth, a static triangle placed 2 along x. Two clips: `bend` turns the
/// elbow 90 degrees about z (toward -x) over a second, `lift` raises the
/// base a step after a second. The column's material is red and textured.
/// Returns the glTF JSON (without its buffer's `uri`) and the buffer.
pub fn arm() -> (Value, Vec<u8>) {
    let mut builder = Builder { bytes: Vec::new(), views: Vec::new(), accessors: Vec::new() };
    // Pairs of vertices at heights 0, 0.5, 1, 1.5 and 2.
    let heights = [0.0f32, 0.5, 1.0, 1.5, 2.0];
    let positions: Vec<f32> = heights.iter().flat_map(|&y| [-0.1, y, 0.0, 0.1, y, 0.0]).collect();
    let normals: Vec<f32> = heights.iter().flat_map(|_| [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).collect();
    let uvs: Vec<f32> = heights.iter().flat_map(|&y| [0.0, 1.0 - y / 2.0, 1.0, 1.0 - y / 2.0]).collect();
    let indices: Vec<u16> =
        (0..4u16).flat_map(|row| [row * 2, row * 2 + 1, row * 2 + 3, row * 2, row * 2 + 3, row * 2 + 2]).collect();
    // Below the elbow the base moves them, above it the elbow; at it, both.
    let influence = |y: f32| match y {
        _ if y < 1.0 => ([0u16, 0, 0, 0], [1.0f32, 0.0, 0.0, 0.0]),
        _ if y == 1.0 => ([0, 1, 0, 0], [0.5, 0.5, 0.0, 0.0]),
        _ => ([1, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]),
    };
    let joints: Vec<u16> = heights.iter().flat_map(|&y| influence(y).0.repeat(2)).collect();
    let weights: Vec<f32> = heights.iter().flat_map(|&y| influence(y).1.repeat(2)).collect();
    let position = builder.floats(&positions, "VEC3", true);
    let normal = builder.floats(&normals, "VEC3", false);
    let uv = builder.floats(&uvs, "VEC2", false);
    let index = builder.shorts(&indices, "SCALAR");
    let joint = builder.shorts(&joints, "VEC4");
    let weight = builder.floats(&weights, "VEC4", false);
    let triangle = builder.floats(&[0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0], "VEC3", true);
    let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let mut binds = identity.to_vec();
    binds.extend([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0]);
    let inverse_binds = builder.floats(&binds, "MAT4", false);
    let half = std::f32::consts::FRAC_1_SQRT_2;
    let bend_times = builder.floats(&[0.0, 1.0], "SCALAR", true);
    let bend_turns = builder.floats(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, half, half], "VEC4", false);
    let lift_times = builder.floats(&[0.0, 1.0, 2.0], "SCALAR", true);
    let lift_steps = builder.floats(&[0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0], "VEC3", false);
    let image = builder.view(&png([255, 255, 255, 255]));
    let document = json!({
        "asset": {"version": "2.0", "generator": "pen-gltf tests"},
        "scene": 0,
        "scenes": [{"nodes": [0, 4]}],
        "nodes": [
            {"name": "armature", "children": [1, 3]},
            {"name": "base", "children": [2]},
            {"name": "elbow", "translation": [0.0, 1.0, 0.0]},
            {"name": "column", "mesh": 0, "skin": 0},
            {"name": "plinth", "mesh": 1, "translation": [2.0, 0.0, 0.0]},
        ],
        "skins": [{"joints": [1, 2], "inverseBindMatrices": inverse_binds}],
        "meshes": [
            {"name": "column", "primitives": [{
                "attributes": {"POSITION": position, "NORMAL": normal, "TEXCOORD_0": uv, "JOINTS_0": joint, "WEIGHTS_0": weight},
                "indices": index,
                "material": 0,
            }]},
            {"name": "plinth", "primitives": [{"attributes": {"POSITION": triangle}}]},
        ],
        "materials": [{
            "name": "red",
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.0, 0.0, 1.0], "baseColorTexture": {"index": 0}, "metallicFactor": 0.25, "roughnessFactor": 0.5},
        }],
        "textures": [{"source": 0}],
        "images": [{"name": "white", "bufferView": image, "mimeType": "image/png"}],
        "animations": [
            {"name": "bend", "samplers": [{"input": bend_times, "output": bend_turns}],
             "channels": [{"sampler": 0, "target": {"node": 2, "path": "rotation"}}]},
            {"name": "lift", "samplers": [{"input": lift_times, "output": lift_steps, "interpolation": "STEP"}],
             "channels": [{"sampler": 0, "target": {"node": 1, "path": "translation"}}]},
        ],
        "bufferViews": builder.views,
        "accessors": builder.accessors,
    });
    let mut document = document;
    let length = builder.bytes.len();
    document["buffers"] = json!([{"byteLength": length}]);
    (document, builder.bytes)
}

fn base64(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut text = String::new();
    for chunk in bytes.chunks(3) {
        let word = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        for index in 0..=chunk.len() {
            text.push(char::from(DIGITS[(word >> (18 - 6 * index) & 63) as usize]));
        }
    }
    while !text.len().is_multiple_of(4) {
        text.push('=');
    }
    text
}

/// A directory of its own for a test.
pub fn directory(name: &str) -> PathBuf {
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pen-gltf-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// The arm as a `.gltf` with its buffer in a data URI.
pub fn arm_gltf(directory: &Path) -> PathBuf {
    let (document, bytes) = arm();
    gltf_file(&directory.join("arm.gltf"), document, &bytes)
}

/// `document` with `bytes` as its buffer's data URI, written to `path`.
pub fn gltf_file(path: &Path, mut document: Value, bytes: &[u8]) -> PathBuf {
    document["buffers"][0]["uri"] = format!("data:application/octet-stream;base64,{}", base64(bytes)).into();
    std::fs::write(path, serde_json::to_string_pretty(&document).unwrap()).unwrap();
    path.to_owned()
}

/// The arm as a `.glb`: its JSON and its buffer in one binary file.
pub fn arm_glb(directory: &Path) -> PathBuf {
    let (document, mut bytes) = arm();
    let mut json = serde_json::to_vec(&document).unwrap();
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    let length = u32::try_from(12 + 8 + json.len() + 8 + bytes.len()).unwrap();
    let mut glb = b"glTF".to_vec();
    glb.extend(2u32.to_le_bytes());
    glb.extend(length.to_le_bytes());
    glb.extend(u32::try_from(json.len()).unwrap().to_le_bytes());
    glb.extend(b"JSON");
    glb.extend(json);
    glb.extend(u32::try_from(bytes.len()).unwrap().to_le_bytes());
    glb.extend(b"BIN\0");
    glb.extend(bytes);
    let path = directory.join("arm.glb");
    std::fs::write(&path, glb).unwrap();
    path
}

pub fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

pub mod gpu {
    //! A headless device, frames seen through a fixed camera, and their
    //! pixels read back.
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use pen_protocol::*;
    use pen_render::{RenderTarget, Renderer, RendererOptions};

    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
    pub const SIZE: u32 = 128;
    pub const IDENTITY: Matrix = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    /// x from -1.5 to 1.5 and y from -0.5 to 2.5 fill the image; z from 1
    /// (near) to -1.
    pub const PROJECTION: Matrix =
        [2.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 2.0 / 3.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, -2.0 / 3.0, 0.5, 1.0];

    pub fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default())).ok()
    }

    pub fn renderer(device: &wgpu::Device, queue: &wgpu::Queue, assets: &Path) -> Renderer {
        let options = RendererOptions {
            format: FORMAT,
            display: pen_render::Display::SDR,
            assets: assets.to_owned(),
            veil: None,
            pipeline_cache: None,
        };
        Renderer::new(device.clone(), queue.clone(), options).unwrap()
    }

    /// A placement `x` along.
    pub fn at(x: f32) -> Matrix {
        let mut matrix = IDENTITY;
        matrix[12] = x;
        matrix
    }

    /// A frame of these draws and poses, lit by white ambient light alone,
    /// every effect off.
    pub fn frame(draws: Vec<Draw>, poses: Vec<Pose>) -> RenderFrame {
        let zero = Vector::default();
        RenderFrame {
            ambient: Vector { x: 1.0, y: 1.0, z: 1.0 },
            light_direction: Vector { x: 0.0, y: -1.0, z: 0.0 },
            light_color: zero,
            clear: Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            quality: Quality { msaa: 1, scale: 1.0, filtering: 1, ..Default::default() },
            sky: Sky { sun: Vector { x: 0.0, y: 1.0, z: 0.0 }, ..Default::default() },
            fog: Fog { radius: 0.0, density: 0.0, height: 1.0 },
            shadow: Shadow { view: IDENTITY, projection: IDENTITY, texel: 0.0, texels: 0 },
            view: IDENTITY,
            projection: PROJECTION,
            draws,
            poses,
            ..Default::default()
        }
    }

    pub struct Offscreen(pub wgpu::Texture);

    impl Offscreen {
        pub fn new(device: &wgpu::Device) -> Self {
            Self(device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
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
            (SIZE, SIZE)
        }
        fn view(&mut self) -> Option<wgpu::TextureView> {
            Some(self.0.create_view(&Default::default()))
        }
        fn present(&mut self) {}
    }

    /// Draws `frame` until every asset of `keys` is ready and nothing is
    /// compiling or loading, then once more.
    pub fn settle(
        renderer: &mut Renderer,
        frame: &RenderFrame,
        catalog: &Catalog,
        target: &mut Offscreen,
        keys: &[&str],
    ) {
        let mut ready = std::collections::HashSet::new();
        let started = Instant::now();
        loop {
            renderer.submit(frame, catalog, target).unwrap();
            for event in renderer.take_events() {
                match event {
                    GameEvent::AssetReady { key, .. } => {
                        ready.insert(key);
                    }
                    GameEvent::AssetFailed { key, message, .. } => panic!("{key} failed: {message}"),
                    _ => {}
                }
            }
            if keys.iter().all(|key| ready.contains(*key)) && renderer.pending() == 0 {
                break;
            }
            assert!(started.elapsed() < Duration::from_secs(30), "assets still loading: {ready:?}");
            std::thread::sleep(Duration::from_millis(5));
        }
        renderer.submit(frame, catalog, target).unwrap();
    }

    /// The image's pixels, RGBA, row by row.
    pub fn pixels(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<[u8; 4]> {
        let row = SIZE * 4;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row * SIZE),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
            },
            wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let bytes = buffer.slice(..).get_mapped_range().unwrap();
        bytes.as_chunks::<4>().0.iter().map(|&[b, g, r, a]| [r, g, b, a]).collect()
    }

    /// The pixel showing the point (x, y).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "points inside the picture")]
    pub fn seen(pixels: &[[u8; 4]], x: f32, y: f32) -> [u8; 4] {
        let column = ((x + 1.5) / 3.0 * SIZE as f32) as usize;
        let row = ((2.5 - y) / 3.0 * SIZE as f32) as usize;
        pixels[row.min(SIZE as usize - 1) * SIZE as usize + column.min(SIZE as usize - 1)]
    }

    pub fn lit(pixel: [u8; 4]) -> bool {
        pixel[0] > 100 && pixel[1] < 40
    }

    /// Where shots go: $PEN_SHOTS, or the tests' temporary directory.
    pub fn shots() -> PathBuf {
        let directory = std::env::var_os("PEN_SHOTS")
            .map_or_else(|| Path::new(env!("CARGO_TARGET_TMPDIR")).join("shots"), PathBuf::from);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    pub fn save(pixels: &[[u8; 4]], name: &str) -> PathBuf {
        let path = shots().join(name);
        image::RgbaImage::from_raw(SIZE, SIZE, pixels.as_flattened().to_vec()).unwrap().save(&path).unwrap();
        path
    }
}
