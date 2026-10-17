//! Particles on a real (headless) GPU: emitters declared in the frame, their
//! particles placed by a compute pass from each emitter's clock and seed
//! alone. A column of sparks rises where its velocity takes it; the same
//! emitter at the same time draws the same picture whatever came before; a
//! seed changes the scatter; textures shape them; MSAA draws them too.
//! Writes PNG shots ($PEN_SHOTS, or the tests' temporary directory). Skips
//! when no adapter is available.
// Native only: in the browser the renderer runs on the page, not under cargo test.
#![cfg(not(target_arch = "wasm32"))]
mod common;

use std::path::{Path, PathBuf};

use common::*;
use pen_protocol::*;
use pen_render::{Display, Renderer, RendererOptions};

const SIZE: u32 = 256;
// x from -1.5 to 1.5 and y from -0.5 to 2.5 fill the image.
const PROJECTION: Matrix =
    [2.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 2.0 / 3.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, -2.0 / 3.0, 0.5, 1.0];

fn assets() -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pen-render-particles-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let write = |key: &str, bytes: &[u8]| std::fs::write(root.join(key), bytes).unwrap();
    write("catalog.json", br#"{"meshes":["tri.mesh.json"],"materials":["spark.material.json","smoke.material.json"]}"#);
    write(
        "tri.mesh.json",
        br#"{"positions":[0,0,0,1,0,0,0,1,0],"normals":[0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,0,1],"indices":[0,1,2]}"#,
    );
    write("spark.material.json", br#"{"shader":"none.wgsl","color":[1,1,1,1],"lighting":"unlit"}"#);
    // A square, half bright: textures shape particles and tint them.
    let image = image::RgbaImage::from_pixel(4, 4, image::Rgba([128, 128, 128, 255]));
    let mut png = std::io::Cursor::new(Vec::new());
    image.write_to(&mut png, image::ImageFormat::Png).unwrap();
    write("square.png", &png.into_inner());
    write(
        "smoke.material.json",
        br#"{"shader":"none.wgsl","texture":"square.png","color":[1,1,1,1],"lighting":"unlit"}"#,
    );
    root
}

/// Sparks rising two units a second from the origin, twenty a second, each
/// living a second.
fn column(time: f32) -> Emitter {
    let white = Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    Emitter {
        material: 0,
        transform: IDENTITY,
        seed: 1,
        time,
        rate: 20.0,
        lifetime: 1.0,
        velocity: Vector { x: 0.0, y: 2.0, z: 0.0 },
        size_start: 0.15,
        size_end: 0.15,
        color_start: white,
        color_end: white,
        additive: true,
        ..Default::default()
    }
}

/// A fountain: up and scattered, falling back under gravity, warm and small
/// at birth, fading out larger.
fn fountain(time: f32, seed: u32) -> Emitter {
    Emitter {
        material: 0,
        transform: IDENTITY,
        seed,
        time,
        rate: 200.0,
        lifetime: 1.5,
        radius: 0.05,
        velocity: Vector { x: 0.0, y: 3.0, z: 0.0 },
        spread: 1.0,
        gravity: Vector { x: 0.0, y: -4.0, z: 0.0 },
        drag: 0.3,
        size_start: 0.04,
        size_end: 0.12,
        color_start: Color { r: 1.0, g: 0.7, b: 0.2, a: 1.0 },
        color_end: Color { r: 0.4, g: 0.1, b: 0.6, a: 0.0 },
        additive: false,
        ..Default::default()
    }
}

fn scene(emitters: Vec<Emitter>, msaa: u32) -> RenderFrame {
    let mut frame = frame(Vec::new());
    frame.projection = PROJECTION;
    frame.quality.msaa = msaa;
    frame.emitters = emitters;
    frame
}

/// The pixel showing the point (x, y).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "points inside the picture")]
fn seen(pixels: &[[u8; 4]], x: f32, y: f32) -> [u8; 4] {
    let column = ((x + 1.5) / 3.0 * SIZE as f32) as usize;
    let row = ((2.5 - y) / 3.0 * SIZE as f32) as usize;
    pixels[row * SIZE as usize + column]
}

fn bright(pixel: [u8; 4]) -> bool {
    pixel[0] > 150
}

fn save(pixels: &[[u8; 4]], name: &str) {
    let directory = std::env::var_os("PEN_SHOTS")
        .map_or_else(|| Path::new(env!("CARGO_TARGET_TMPDIR")).join("shots"), PathBuf::from);
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join(name);
    image::RgbaImage::from_raw(SIZE, SIZE, pixels.as_flattened().to_vec()).unwrap().save(&path).unwrap();
    eprintln!("{}", path.display());
}

#[test]
fn emitters_place_their_particles_from_their_clock_and_seed() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let root = assets();
    let catalog = boundary::read_catalog(&root).unwrap();
    let options =
        RendererOptions { format: FORMAT, display: Display::SDR, assets: root, veil: None, pipeline_cache: None };
    let mut renderer = Renderer::new(device.clone(), queue.clone(), options).unwrap();
    renderer.preload(&catalog);
    let mut target = Offscreen::new(&device, SIZE);
    // Draws a frame until its particles have stopped changing (their
    // material and pipeline have arrived), and returns its pixels.
    let mut draw = |frame: &RenderFrame| {
        let mut last = Vec::new();
        for _ in 0..2000 {
            renderer.submit(frame, &catalog, &mut target).unwrap();
            assert!(!renderer.take_events().iter().any(|event| matches!(event, GameEvent::AssetFailed { .. })));
            let now = pixels(&device, &queue, &target.0);
            if now == last && renderer.pending() == 0 {
                return now;
            }
            last = now;
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("the particles never settled");
    };

    // Two seconds in, the sparks born in the last second rise along y up to
    // two units; nothing beside them, nothing above.
    let sparks = draw(&scene(vec![column(2.0)], 1));
    save(&sparks, "particles-column-2.00s.png");
    for y in [0.2, 0.7, 1.2, 1.7] {
        assert!(bright(seen(&sparks, 0.0, y)), "a spark at height {y}: {:?}", seen(&sparks, 0.0, y));
    }
    assert!(!bright(seen(&sparks, 0.6, 1.0)) && !bright(seen(&sparks, 0.0, 2.3)));
    // Half a second in, they reach only one unit up.
    let early = draw(&scene(vec![column(0.5)], 1));
    assert!(bright(seen(&early, 0.0, 0.5)) && !bright(seen(&early, 0.0, 1.5)));
    // Drawn again after other times, the same emitter at the same time
    // draws the same picture: nothing is kept from frame to frame.
    assert_eq!(draw(&scene(vec![column(2.0)], 1)), sparks);
    // Stopped after a second: by three seconds every spark has died.
    let stopped = draw(&scene(vec![Emitter { duration: 1.0, ..column(3.0) }], 1));
    assert!(stopped.iter().all(|pixel| pixel[0] < 10), "nothing lives on");

    // A fountain's scatter follows its seed: the same seed, the same spray.
    let fountain_one = draw(&scene(vec![fountain(2.0, 7)], 1));
    save(&fountain_one, "particles-fountain-2.00s.png");
    assert_eq!(draw(&scene(vec![fountain(2.0, 7)], 1)), fountain_one);
    let fountain_two = draw(&scene(vec![fountain(2.0, 8)], 1));
    save(&fountain_two, "particles-fountain-seed-8.png");
    assert_ne!(fountain_two, fountain_one);
    let fountain_later = draw(&scene(vec![fountain(2.5, 7)], 1));
    save(&fountain_later, "particles-fountain-2.50s.png");
    assert_ne!(fountain_later, fountain_one);

    // With four samples a pixel, and textured: the sparks are squares of
    // the texture's grey (sRGB 128, a fifth of white's light).
    let textured = Emitter { material: 1, ..column(2.0) };
    let smoke = draw(&scene(vec![textured], 4));
    save(&smoke, "particles-textured-msaa.png");
    let pixel = seen(&smoke, 0.0, 1.2);
    assert!((45..=65).contains(&pixel[0]) && pixel[0] == pixel[2], "grey: {pixel:?}");
    assert!(!bright(seen(&smoke, 0.0, 1.2)) && seen(&smoke, 0.5, 1.2)[0] < 10);
}
