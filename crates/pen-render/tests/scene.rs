//! A scene's assets loaded first (`Renderer::preload_scene`): all of them at
//! once, before any frame asks, and the rest of the catalog only after them.
//! Skips when no adapter is available.
#![cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::time::{Duration, Instant};

mod common;

use common::*;
use pen_protocol::*;
use pen_render::{Renderer, RendererOptions};

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

const MESH: &[u8] = br#"{"positions":[-0.5,-0.5,0.5, 0.5,-0.5,0.5, 0,0.5,0.5],"normals":[0,0,1,0,0,1,0,0,1],
    "uvs":[0,0,1,0,0.5,1],"indices":[0,1,2]}"#;

fn ready_keys(events: &[GameEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            GameEvent::AssetReady { key, .. } => Some(key.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn a_scenes_assets_load_before_the_rest_of_the_catalog() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pen-render-scene-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let write = |key: &str, bytes: &[u8]| std::fs::write(root.join(key), bytes).unwrap();
    write("flat.wgsl", SHADER.as_bytes());
    for name in ["near", "far", "later"] {
        write(&format!("{name}.mesh.json"), MESH);
        write(&format!("{name}.material.json"), br#"{"shader":"flat.wgsl","color":[0,1,0,1],"lighting":"unlit"}"#);
    }
    let catalog = Catalog {
        meshes: vec!["near.mesh.json".into(), "far.mesh.json".into(), "later.mesh.json".into()],
        materials: vec!["near.material.json".into(), "far.material.json".into(), "later.material.json".into()],
        sounds: Vec::new(),
    };
    let mut renderer = Renderer::new(
        device.clone(),
        queue,
        RendererOptions {
            format: FORMAT,
            display: pen_render::Display::SDR,
            assets: root.clone(),
            veil: None,
            pipeline_cache: None,
        },
    )
    .unwrap();
    let scene: Vec<(String, String)> = [
        ("material", "near.material.json"),
        ("shader", "flat.wgsl"),
        ("mesh", "near.mesh.json"),
        ("mesh", "far.mesh.json"),
        ("teapot", "ignored"),
    ]
    .iter()
    .map(|(kind, key)| (kind.to_string(), key.to_string()))
    .collect();
    renderer.preload_scene(&scene, 0);
    renderer.preload(&catalog);
    // The catalog waits, and counts as work to come.
    assert!(renderer.pending() > 0);
    let started = Instant::now();
    let mut events = Vec::new();
    while renderer.pending() > 0 {
        assert!(started.elapsed() < Duration::from_secs(30), "loading ends; so far {events:?}");
        renderer.wait(Duration::from_millis(50));
        events.extend(renderer.take_events());
    }
    events.extend(renderer.take_events());
    let keys = ready_keys(&events);
    let place = |key: &str| keys.iter().position(|ready| ready == key).unwrap_or_else(|| panic!("{key} in {keys:?}"));
    for first in ["near.material.json", "flat.wgsl", "near.mesh.json", "far.mesh.json"] {
        for after in ["far.material.json", "later.material.json", "later.mesh.json"] {
            assert!(place(first) < place(after), "{first} before {after}: {keys:?}");
        }
    }
    // Each once.
    assert_eq!(keys.len(), 7, "{keys:?}");
    let target = Offscreen::new(&device, 8);
    let frame = frame(vec![Draw { mesh: 0, material: 0, transform: IDENTITY, parameters: None, extra: None }]);
    let mut target = target;
    renderer.submit(&frame, &catalog, &mut target).unwrap();
    assert!(!renderer.waiting(), "the scene's draw waits on nothing");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn assets_read_ahead_are_taken_for_their_first_load() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pen-render-prefetch-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let write = |key: &str, bytes: &[u8]| std::fs::write(root.join(key), bytes).unwrap();
    write("flat.wgsl", SHADER.as_bytes());
    write("tri.mesh.json", MESH);
    write("paint.material.json", br#"{"shader":"flat.wgsl","color":[0,1,0,1],"lighting":"unlit"}"#);
    let scene: Vec<(String, String)> =
        [("material", "paint.material.json"), ("shader", "flat.wgsl"), ("mesh", "tri.mesh.json")]
            .iter()
            .map(|(kind, key)| (kind.to_string(), key.to_string()))
            .collect();
    // Read ahead before there is a renderer, as a host does while its GPU opens.
    let prefetch = pen_render::Prefetch::start(root.clone(), &scene);
    std::thread::sleep(Duration::from_millis(200));
    // The material read ahead is green; the file is now red.
    write("paint.material.json", br#"{"shader":"flat.wgsl","color":[1,0,0,1],"lighting":"unlit"}"#);
    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions {
            format: FORMAT,
            display: pen_render::Display::SDR,
            assets: root.clone(),
            veil: None,
            pipeline_cache: None,
        },
    )
    .unwrap();
    renderer.warm(prefetch);
    renderer.preload_scene(&scene, 1);
    let catalog = Catalog {
        meshes: vec!["tri.mesh.json".into()],
        materials: vec!["paint.material.json".into()],
        sounds: Vec::new(),
    };
    let mut target = Offscreen::new(&device, 8);
    let frame = frame(vec![Draw { mesh: 0, material: 0, transform: IDENTITY, parameters: None, extra: None }]);
    pump(&mut renderer, &frame, &catalog, &mut target, |events| {
        ready(events, "paint.material.json", 1) && ready(events, "flat.wgsl", 1)
    });
    let started = Instant::now();
    while renderer.waiting() {
        assert!(started.elapsed() < Duration::from_secs(20), "the draw's assets load");
        renderer.wait(Duration::from_millis(20));
        renderer.submit(&frame, &catalog, &mut target).unwrap();
    }
    let [r, g, _, _] = middle(&device, &queue, &target.0);
    assert!(g > 200 && r < 50, "drawn with the material read ahead: {r} {g}");
    // A reload reads the file.
    renderer.reload(&["paint.material.json".to_owned()]);
    pump(&mut renderer, &frame, &catalog, &mut target, |events| ready(events, "paint.material.json", 2));
    renderer.submit(&frame, &catalog, &mut target).unwrap();
    let [r, g, _, _] = middle(&device, &queue, &target.0);
    assert!(r > 200 && g < 50, "drawn with the file's material after a reload: {r} {g}");
    std::fs::remove_dir_all(root).unwrap();
}
