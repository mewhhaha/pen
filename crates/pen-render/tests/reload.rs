//! Hot reload on a real (headless) GPU: assets whose files change load again
//! with a new revision and replace the live ones once ready; a broken edit is
//! reported and the last good revision keeps drawing; preloading a changed
//! catalog loads what it adds. Skips when no adapter is available.
// Native only: in the browser the renderer runs on the page, not under cargo test.
#![cfg(not(target_arch = "wasm32"))]
mod common;

use std::path::{Path, PathBuf};

use common::*;
use pen_protocol::*;
use pen_render::{Renderer, RendererOptions};

// Its triangle in its material's color times the texture's middle texel.
const TINTED: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@group(1) @binding(1) var base: texture_2d<f32>;
@group(1) @binding(2) var base_sampler: sampler;
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f {
  return draw.color * textureSampleLevel(base, base_sampler, vec2f(0.5), 0.0);
}
"#;

// The same, but plain blue whatever the material says.
const BLUE: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return vec4f(0.0, 0.0, 1.0, 1.0); }
"#;

fn mesh(left: f32) -> String {
    let right = left + 1.0;
    let middle = left + 0.5;
    format!(
        r#"{{"positions":[{left},-0.5,0.5, {right},-0.5,0.5, {middle},0.5,0.5],"normals":[0,0,1,0,0,1,0,0,1],
            "uvs":[0,0,1,0,0.5,1],"indices":[0,1,2]}}"#
    )
}

fn png(color: [u8; 4]) -> Vec<u8> {
    let image = image::RgbaImage::from_pixel(4, 4, image::Rgba(color));
    let mut bytes = std::io::Cursor::new(Vec::new());
    image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
    bytes.into_inner()
}

/// A 512 x 512 colour texture of one color, packed with all ten levels.
fn gtex(color: [u8; 4]) -> Vec<u8> {
    use std::io::Write;
    let mut bytes = b"GTEX".to_vec();
    bytes.extend([2, 4, 1, 10]);
    bytes.extend(512u32.to_le_bytes());
    bytes.extend(512u32.to_le_bytes());
    for level in (0..10).rev() {
        let size = 512usize >> level;
        let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(&color.repeat(size * size)).unwrap();
        let packed = encoder.finish().unwrap();
        bytes.extend((packed.len() as u32).to_le_bytes());
        bytes.extend(packed);
    }
    bytes
}

fn write(root: &Path, key: &str, bytes: impl AsRef<[u8]>) {
    std::fs::write(root.join(key), bytes).unwrap();
}

fn assets() -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pen-render-reload-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    write(&root, "catalog.json", br#"{"meshes":["tri.mesh.json"],"materials":["paint.material.json"]}"#);
    write(&root, "tri.mesh.json", mesh(-0.5));
    write(&root, "tinted.wgsl", TINTED);
    write(&root, "paint.png", png([255, 0, 0, 255]));
    write(&root, "paint.material.json", br#"{"shader":"tinted.wgsl","texture":"paint.png","color":[1,1,1,1],"lighting":"unlit"}"#);
    root
}

fn keys(keys: &[&str]) -> Vec<String> {
    keys.iter().map(|key| key.to_string()).collect()
}

/// Whether `pixel` is within 2 of `want` in each channel.
fn near(pixel: [u8; 4], want: [u8; 3]) -> bool {
    pixel.iter().zip(want).all(|(&got, want)| got.abs_diff(want) <= 2)
}

#[test]
fn reloaded_assets_replace_the_live_ones() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let root = assets();
    let catalog = boundary::read_catalog(&root).unwrap();
    let mut renderer =
        Renderer::new(device.clone(), queue.clone(), RendererOptions { format: FORMAT, assets: root.clone(), veil: None })
            .unwrap();
    renderer.preload(&catalog);
    let mut target = Offscreen::new(&device, 64);
    let draw = Draw { material: 0, mesh: 0, transform: IDENTITY, parameters: None, extra: None };
    let frame = frame(vec![draw.clone()]);
    // Full channels roll off past the shoulder to 1 - 0.01 / 0.2 of white.
    let full = 242;

    let events = pump(&mut renderer, &frame, &catalog, &mut target, |events| {
        ["tri.mesh.json", "tinted.wgsl", "paint.png", "paint.material.json"].iter().all(|key| ready(events, key, 1))
    });
    assert!(!events.iter().any(|event| matches!(event, GameEvent::AssetFailed { .. })), "{events:?}");
    assert!(near(middle(&device, &queue, &target.0), [full, 0, 0]), "red at first");

    // A new texture: its bind groups give way to the new one's.
    write(&root, "paint.png", png([0, 255, 0, 255]));
    renderer.reload(&keys(&["paint.png"]));
    pump(&mut renderer, &frame, &catalog, &mut target, |events| ready(events, "paint.png", 2));
    assert!(near(middle(&device, &queue, &target.0), [0, full, 0]), "green once the texture reloads");

    // A new material: its color halves the texture's.
    write(&root, "paint.material.json", br#"{"shader":"tinted.wgsl","texture":"paint.png","color":[0.5,0.5,0.5,1],"lighting":"unlit"}"#);
    renderer.reload(&keys(&["paint.material.json"]));
    pump(&mut renderer, &frame, &catalog, &mut target, |events| ready(events, "paint.material.json", 2));
    let pixel = middle(&device, &queue, &target.0);
    assert!(near(pixel, [0, 128, 0]), "half green once the material reloads: {pixel:?}");

    // A new shader: its pipelines replace the old ones.
    write(&root, "tinted.wgsl", BLUE);
    renderer.reload(&keys(&["tinted.wgsl"]));
    pump(&mut renderer, &frame, &catalog, &mut target, |events| ready(events, "tinted.wgsl", 2));
    assert!(near(middle(&device, &queue, &target.0), [0, 0, full]), "blue once the shader reloads");

    // A broken edit is reported, and the last good shader draws on.
    write(&root, "tinted.wgsl", "@fragment fn fragment( -> {");
    renderer.reload(&keys(&["tinted.wgsl"]));
    let events = pump(&mut renderer, &frame, &catalog, &mut target, |events| failed(events, "tinted.wgsl", 3));
    assert!(!ready(&events, "tinted.wgsl", 3));
    for _ in 0..5 {
        renderer.submit(&frame, &catalog, &mut target).unwrap();
    }
    assert!(near(middle(&device, &queue, &target.0), [0, 0, full]), "still blue after a broken edit");

    // Fixed again.
    write(&root, "tinted.wgsl", TINTED);
    renderer.reload(&keys(&["tinted.wgsl", "tinted.wgsl"]));
    let events = pump(&mut renderer, &frame, &catalog, &mut target, |events| ready(events, "tinted.wgsl", 4));
    assert!(!ready(&events, "tinted.wgsl", 5), "a key named twice loads once");
    let pixel = middle(&device, &queue, &target.0);
    assert!(near(pixel, [0, 128, 0]), "half green with the fixed shader: {pixel:?}");

    // Keys nothing asked for load nothing.
    renderer.reload(&keys(&["unused.png", "catalog.json"]));
    let events = pump(&mut renderer, &frame, &catalog, &mut target, |_| true);
    assert!(events.is_empty(), "{events:?}");

    // A catalog with another material, ahead of the first, preloaded again.
    write(&root, "white.material.json", br#"{"shader":"tinted.wgsl","color":[1,1,1,1],"lighting":"unlit"}"#);
    write(&root, "catalog.json", br#"{"meshes":["tri.mesh.json"],"materials":["white.material.json","paint.material.json"]}"#);
    let catalog = boundary::read_catalog(&root).unwrap();
    renderer.preload(&catalog);
    pump(&mut renderer, &frame, &catalog, &mut target, |events| ready(events, "white.material.json", 1));
    assert!(near(middle(&device, &queue, &target.0), [full, full, full]), "white from the new catalog's first material");
    let paint = common::frame(vec![Draw { material: 1, ..draw.clone() }]);
    renderer.submit(&paint, &catalog, &mut target).unwrap();
    assert!(near(middle(&device, &queue, &target.0), [0, 128, 0]), "the old material at its new index");

    // The material now names a packed texture, which streams in; then that
    // texture changes too, while the old one may still be streaming.
    write(&root, "paint.gtex", gtex([255, 255, 0, 255]));
    write(&root, "paint.material.json", br#"{"shader":"tinted.wgsl","texture":"paint.gtex","color":[1,1,1,1],"lighting":"unlit"}"#);
    renderer.reload(&keys(&["paint.material.json"]));
    pump(&mut renderer, &paint, &catalog, &mut target, |events| ready(events, "paint.gtex", 1));
    assert!(near(middle(&device, &queue, &target.0), [full, full, 0]), "yellow from the packed texture");
    write(&root, "paint.gtex", gtex([0, 255, 255, 255]));
    renderer.reload(&keys(&["paint.gtex"]));
    pump(&mut renderer, &paint, &catalog, &mut target, |events| ready(events, "paint.gtex", 2));
    assert!(near(middle(&device, &queue, &target.0), [0, full, full]), "cyan once the packed texture reloads");
    let started = std::time::Instant::now();
    while !renderer.quiet() {
        assert!(started.elapsed().as_secs() < 20, "the reloaded texture finishes streaming");
        renderer.submit(&paint, &catalog, &mut target).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    // A new mesh: the triangle moves off the middle.
    write(&root, "tri.mesh.json", mesh(-0.95));
    renderer.reload(&keys(&["tri.mesh.json"]));
    pump(&mut renderer, &paint, &catalog, &mut target, |events| ready(events, "tri.mesh.json", 2));
    assert!(near(middle(&device, &queue, &target.0), [0, 0, 0]), "the moved triangle leaves the middle empty");
}
