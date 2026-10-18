//! The renderer on a real (headless) GPU, over a small asset directory: what
//! loads draws, what fails is reported and left out, packed textures stream
//! in until the renderer is quiet. Skips when no adapter is available.
// Native only: in the browser the renderer runs on the page, not under cargo test.
#![cfg(not(target_arch = "wasm32"))]
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

mod common;

use common::*;
use pen_protocol::*;
use pen_render::{Renderer, RendererOptions};

// Draws its triangle in its material's color, wherever the draw puts it.
const SHADER: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return draw.color; }
"#;

fn deflate(bytes: &[u8]) -> Vec<u8> {
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    encoder.write_all(bytes).unwrap();
    encoder.finish().unwrap()
}

/// A 512 x 512 colour texture, packed with all ten levels.
fn packed_texture() -> Vec<u8> {
    let mut bytes = b"GTEX".to_vec();
    bytes.extend([2, 4, 1, 10]);
    bytes.extend(512u32.to_le_bytes());
    bytes.extend(512u32.to_le_bytes());
    for level in (0..10).rev() {
        let size = 512usize >> level;
        let packed = deflate(&vec![200; size * size * 4]);
        bytes.extend(u32::try_from(packed.len()).unwrap().to_le_bytes());
        bytes.extend(packed);
    }
    bytes
}

fn assets() -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pen-render-assets-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let write = |key: &str, bytes: &[u8]| std::fs::write(root.join(key), bytes).unwrap();
    write(
        "catalog.json",
        br#"{"meshes":["tri.mesh.json","bad.mesh.json"],
             "materials":["green.material.json","broken.material.json","missing.material.json"]}"#,
    );
    write(
        "tri.mesh.json",
        br#"{"positions":[-0.5,-0.5,0.5, 0.5,-0.5,0.5, 0,0.5,0.5],"normals":[0,0,1,0,0,1,0,0,1],
             "uvs":[0,0,1,0,0.5,1],"indices":[0,1,2]}"#,
    );
    write("bad.mesh.json", br#"{"positions":[0,0,0],"normals":[0,0,0],"uvs":[0,0],"indices":[0,0,0]}"#);
    write("flat.wgsl", SHADER.as_bytes());
    write("broken.wgsl", b"@vertex fn vertex( -> {");
    write("stream.gtex", &packed_texture());
    write(
        "green.material.json",
        br#"{"shader":"flat.wgsl","texture":"stream.gtex","color":[0,1,0,1],"lighting":"unlit"}"#,
    );
    write("broken.material.json", br#"{"shader":"broken.wgsl","color":[1,0,0,1],"lighting":"unlit"}"#);
    write(
        "missing.material.json",
        br#"{"shader":"flat.wgsl","texture":"nowhere.png","color":[0,0,1,1],"lighting":"unlit"}"#,
    );
    root
}

#[test]
fn draws_what_loads_and_reports_what_fails() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let root = assets();
    let catalog = boundary::read_catalog(&root).unwrap();
    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions { format: FORMAT, assets: root, veil: Some(0.1), pipeline_cache: None },
    )
    .unwrap();
    renderer.preload(&catalog);
    let mut target = Offscreen::new(&device, 64);
    let draw = |mesh: u32, material: u32, x: f32| {
        let mut transform = IDENTITY;
        transform[12] = x;
        Draw { material, mesh, transform, parameters: None, extra: None }
    };
    // The green triangle in the middle; the others off to the side.
    let frame = frame(vec![draw(0, 0, 0.0), draw(0, 1, 2.0), draw(1, 0, 2.0), draw(0, 2, 2.0)]);
    let mut ready = Vec::new();
    let mut failed = Vec::new();
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(20) {
        renderer.submit(&frame, &catalog, &mut target).unwrap();
        for event in renderer.take_events() {
            match event {
                GameEvent::AssetReady { key, .. } => ready.push(key),
                GameEvent::AssetFailed { key, kind, message, .. } => failed.push((key, kind, message)),
                _ => {}
            }
        }
        if renderer.quiet() && failed.len() >= 3 && ready.iter().any(|key| key == "stream.gtex") {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let failed_keys: Vec<&str> = failed.iter().map(|(key, _, _)| key.as_str()).collect();
    for key in ["broken.wgsl", "bad.mesh.json", "nowhere.png"] {
        assert!(failed_keys.contains(&key), "{key} should fail: {failed:?}");
    }
    for key in ["tri.mesh.json", "green.material.json", "flat.wgsl", "stream.gtex"] {
        assert!(ready.iter().any(|ready| ready == key), "{key} should load: {ready:?}");
    }
    assert!(renderer.quiet(), "streaming, fading and the veil are done");
    renderer.submit(&frame, &catalog, &mut target).unwrap();
    // The green triangle, and the one whose texture failed (as the checker);
    // the broken shader and the bad mesh draw nothing.
    assert_eq!(renderer.stats().draw_calls, 2);
    // Green past the shoulder rolls off: 1 - 0.01 / 0.2.
    let [r, g, b, _] = middle(&device, &queue, &target.0);
    assert!(r < 2 && b < 2 && (240..=244).contains(&g), "middle pixel {r} {g} {b}");

    // Four samples a pixel: the frames keep one until the shader's
    // pipelines for four have compiled in the background, then switch.
    let mut multisampled = frame.clone();
    multisampled.quality.msaa = 4;
    multisampled.quality.fxaa = true;
    for _ in 0..50 {
        renderer.submit(&multisampled, &catalog, &mut target).unwrap();
        std::thread::sleep(Duration::from_millis(10));
    }
    let [r, g, b, _] = middle(&device, &queue, &target.0);
    assert!(r < 2 && b < 2 && (240..=244).contains(&g), "middle pixel {r} {g} {b} with MSAA");
}

#[test]
fn renderer_moves_between_threads() {
    fn send<T: Send>() {}
    send::<Renderer>();
}

#[test]
fn translucent_insets_preserve_coverage_and_draw_order() {
    let (device, queue) = device().expect("the inset regression needs a headless GPU");
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pen-render-inset-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("flat.wgsl"), SHADER).unwrap();
    std::fs::write(
        root.join("tri.mesh.json"),
        br#"{"positions":[-1,-1,0.5, 1,-1,0.5, 0,1,0.5],"normals":[0,0,1,0,0,1,0,0,1],
             "uvs":[0,0,1,0,0.5,1],"indices":[0,1,2]}"#,
    )
    .unwrap();
    std::fs::write(
        root.join("green.material.json"),
        br#"{"shader":"flat.wgsl","color":[0,0.4,0,0.5],"lighting":"unlit","blend":true}"#,
    )
    .unwrap();
    std::fs::write(
        root.join("blue.material.json"),
        br#"{"shader":"flat.wgsl","color":[0,0,0.4,0.5],"lighting":"unlit","blend":true}"#,
    )
    .unwrap();
    let catalog = Catalog {
        meshes: vec!["tri.mesh.json".into()],
        materials: vec!["green.material.json".into(), "blue.material.json".into()],
        ..Default::default()
    };
    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions { format: FORMAT, assets: root, veil: None, pipeline_cache: None },
    )
    .unwrap();
    renderer.preload(&catalog);
    let mut target = Offscreen::new(&device, 64);
    let mut shown = frame(Vec::new());
    shown.clear.r = 0.8;
    shown.insets.push(Inset {
        x: 0.0,
        y: 0.0,
        width: 64.0,
        height: 64.0,
        pixel_size: 0.0,
        view: IDENTITY,
        projection: IDENTITY,
        ambient: Vector::default(),
        light_direction: shown.light_direction,
        light_color: shown.light_color,
        draws: (0..2)
            .map(|material| Draw { mesh: 0, material, transform: IDENTITY, parameters: None, extra: None })
            .collect(),
    });
    pump(&mut renderer, &shown, &catalog, &mut target, |events| {
        ready(events, "flat.wgsl", 1)
            && ready(events, "green.material.json", 1)
            && ready(events, "blue.material.json", 1)
    });
    // Half blue over half green over red: 0.2 red, 0.1 green, 0.2 blue.
    let [r, g, b, _] = middle(&device, &queue, &target.0);
    assert!((48..=54).contains(&r) && (23..=29).contains(&g) && (48..=54).contains(&b), "inset pixel {r} {g} {b}");
}

#[test]
fn rejects_invalid_frames() {
    let mut bad = frame(Vec::new());
    bad.light_direction = Vector::default();
    assert!(pen_render::validate_frame(&bad).is_err());
    let mut bad = frame(vec![Draw { material: 0, mesh: 0, transform: [0.0; 16], parameters: None, extra: None }]);
    assert!(pen_render::validate_frame(&bad).is_err());
    bad.draws.clear();
    bad.blur = 0.75;
    assert!(pen_render::validate_frame(&bad).is_err());
    assert!(pen_render::validate_frame(&frame(Vec::new())).is_ok());
}

// A sprite in its color, wherever it is on the screen.
const SPRITE: &str = r#"
@group(0) @binding(0) var<uniform> screen: vec4f;
struct Sprite { rect: vec4f, uv: vec4f, color: vec4f, parameters: vec4f };
@group(0) @binding(1) var<storage, read> sprites: array<Sprite>;
struct Corner { @builtin(position) position: vec4f, @location(0) color: vec4f };
@vertex fn vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> Corner {
  let corner = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0))[vertex];
  let sprite = sprites[index];
  let pixel = sprite.rect.xy + corner * sprite.rect.zw;
  return Corner(vec4f(pixel.x / screen.x * 2.0 - 1.0, 1.0 - pixel.y / screen.y * 2.0, 0.0, 1.0), sprite.color);
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f { return corner.color; }
"#;

#[test]
fn shared_renderers_load_their_own_assets() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let game = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions { format: FORMAT, assets: assets(), veil: None, pipeline_cache: None },
    )
    .unwrap();
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pen-render-shared-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("sprite.wgsl"), SPRITE).unwrap();
    std::fs::write(root.join("dot.material.json"), br#"{"shader":"sprite.wgsl","color":[1,1,1,1],"lighting":"unlit"}"#)
        .unwrap();
    let mut tools = game.share(root).unwrap();
    let catalog = Catalog { materials: vec!["dot.material.json".into()], ..Default::default() };
    let sprite = Sprite {
        x: 16.0,
        y: 16.0,
        width: 32.0,
        height: 32.0,
        u0: 0.0,
        v0: 0.0,
        u1: 1.0,
        v1: 1.0,
        color: [1.0, 0.0, 0.0, 1.0],
        material: 0,
        parameters: [0.0; 4],
    };
    let target = Offscreen::new(&device, 64);
    let view = target.0.create_view(&Default::default());
    let started = Instant::now();
    while !tools.prepare_sprites(&[sprite], &catalog).unwrap() {
        assert!(started.elapsed() < Duration::from_secs(20), "the sprite's assets load");
        tools.wait(Duration::from_millis(100));
    }
    let ready = |events: Vec<GameEvent>| -> Vec<(String, u64)> {
        events
            .into_iter()
            .filter_map(|event| match event {
                GameEvent::AssetReady { key, revision, .. } => Some((key, revision)),
                _ => None,
            })
            .collect()
    };
    let loaded = ready(tools.take_events());
    for key in ["dot.material.json", "sprite.wgsl"] {
        assert!(loaded.contains(&(key.to_owned(), 1)), "{key}: {loaded:?}");
    }
    let mut game = game;
    assert!(game.take_events().is_empty(), "the game's renderer hears none of its events");
    // It outlives the renderer whose loader threads it shares.
    drop(game);
    tools.reload(&["dot.material.json".to_owned()]);
    let started = Instant::now();
    while !ready(tools.take_events()).contains(&("dot.material.json".to_owned(), 2)) {
        assert!(started.elapsed() < Duration::from_secs(20), "the material loads again");
        tools.wait(Duration::from_millis(100));
        tools.prepare_sprites(&[sprite], &catalog).unwrap();
    }
    tools.overlay(&[sprite], &catalog, &view, (64, 64)).unwrap();
    let [r, g, b, _] = middle(&device, &queue, &target.0);
    assert!(r > 250 && g < 5 && b < 5, "middle pixel {r} {g} {b}");
}
