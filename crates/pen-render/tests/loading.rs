//! Opt-in startup sprites remain visible above the veil; readiness is not sent to other frames.
mod common;
use common::*;
use pen_protocol::{Catalog, GameEvent, Sprite};
use pen_render::{Display, Renderer, RendererOptions};
use std::time::Duration;
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
fn startup_sprites_and_readiness_are_opt_in() {
    let (device, queue) = device().expect("the startup regression requires a headless GPU");
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("loading-overlay");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("sprite.wgsl"), SPRITE).unwrap();
    std::fs::write(
        root.join("sprite.material.json"),
        r#"{"shader":"sprite.wgsl","color":[1,1,1,1],"lighting":"unlit"}"#,
    )
    .unwrap();
    let catalog = Catalog { materials: vec!["sprite.material.json".into()], ..Default::default() };
    let mut scene = frame(Vec::new());
    scene.sprites.push(Sprite {
        x: 16.0,
        y: 16.0,
        width: 32.0,
        height: 32.0,
        u0: 0.0,
        v0: 0.0,
        u1: 1.0,
        v1: 1.0,
        color: [0.0, 1.0, 0.0, 1.0],
        material: 0,
        parameters: [0.0; 4],
    });
    let mut target = Offscreen::new(&device, 64);
    let make = |seconds| {
        Renderer::new(
            device.clone(),
            queue.clone(),
            RendererOptions {
                format: FORMAT,
                display: Display::SDR,
                assets: root.clone(),
                veil: Some(seconds),
                pipeline_cache: None,
            },
        )
        .unwrap()
    };
    let mut shown = Vec::new();
    for loading in [false, true] {
        let mut renderer = make(600.0);
        scene.loading = loading;
        for _ in 0..1000 {
            renderer.submit(&scene, &catalog, &mut target).unwrap();
            if renderer.pending() == 0 {
                break;
            }
            renderer.wait(Duration::from_millis(5));
        }
        renderer.submit(&scene, &catalog, &mut target).unwrap();
        shown.push(middle(&device, &queue, &target.0));
        assert!(!renderer.quiet(), "the long veil is still up");
        let events = renderer.take_events();
        assert!(!events.iter().any(|e| matches!(e, GameEvent::AssetFailed { .. })), "{events:?}");
        assert!(!events.iter().any(|e| matches!(e, GameEvent::RendererReady)));
        if loading {
            assert_eq!(pixels(&device, &queue, &target.0)[0], [0, 0, 0, 255], "neutral backing around startup UI");
        }
    }
    assert_ne!(shown[0], shown[1], "ordinary sprites remain underneath the veil");
    assert_eq!(shown[1], [0, 255, 0, 255], "startup sprites draw fully over it");
    let mut renderer = make(0.001);
    let events = pump(&mut renderer, &scene, &catalog, &mut target, |events| {
        events.iter().any(|e| matches!(e, GameEvent::RendererReady))
    });
    assert_eq!(events.iter().filter(|e| matches!(e, GameEvent::RendererReady)).count(), 1);
    assert!(renderer.quiet());
    assert!(!renderer.take_events().iter().any(|e| matches!(e, GameEvent::RendererReady)));
    scene.loading = false;
    renderer.submit(&scene, &catalog, &mut target).unwrap();
    assert!(!renderer.take_events().iter().any(|e| matches!(e, GameEvent::RendererReady)));
    scene.loading = true;
    renderer.submit(&scene, &catalog, &mut target).unwrap();
    assert!(
        renderer.take_events().iter().any(|e| matches!(e, GameEvent::RendererReady)),
        "a restored startup can finish too"
    );
}

#[test]
fn modal_insets_split_a_shared_sprite_batch_at_the_layer_boundary() {
    use pen_protocol::{Draw, Inset};
    let (device, queue) = device().expect("UI layering requires a headless GPU");
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("modal-layers");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("sprite.wgsl"), SPRITE).unwrap();
    std::fs::write(
        root.join("sprite.material.json"),
        r#"{"shader":"sprite.wgsl","color":[1,1,1,1],"lighting":"unlit","blend":true}"#,
    )
    .unwrap();
    std::fs::write(root.join("quad.mesh.json"), r#"{"positions":[-1,-1,0.5,1,-1,0.5,-1,1,0.5,1,1,0.5],"normals":[0,0,1,0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,0,1,1,1],"indices":[0,1,2,2,1,3]}"#).unwrap();
    std::fs::write(
        root.join("green.wgsl"),
        r#"
struct V { @builtin(position) position: vec4f };
@vertex fn vertex(@location(0) position: vec3f) -> V { return V(vec4f(position,1)); }
@fragment fn fragment() -> @location(0) vec4f { return vec4f(0,1,0,1); }
"#,
    )
    .unwrap();
    std::fs::write(root.join("green.material.json"), r#"{"shader":"green.wgsl","color":[1,1,1,1],"lighting":"unlit"}"#)
        .unwrap();
    let catalog = Catalog {
        meshes: vec!["quad.mesh.json".into()],
        materials: vec!["sprite.material.json".into(), "green.material.json".into()],
        ..Default::default()
    };
    let mut scene = frame(vec![]);
    let sprite = |color| Sprite {
        x: 0.0,
        y: 0.0,
        width: 64.0,
        height: 64.0,
        u0: 0.0,
        v0: 0.0,
        u1: 1.0,
        v1: 1.0,
        color,
        material: 0,
        parameters: [0.0; 4],
    };
    scene.sprites = vec![sprite([1.0, 0.0, 0.0, 1.0]), sprite([0.0, 0.0, 1.0, 0.5])];
    scene.insets = vec![Inset {
        x: 0.0,
        y: 0.0,
        width: 64.0,
        height: 64.0,
        after_sprites: 1,
        view: IDENTITY,
        projection: IDENTITY,
        draws: vec![Draw { mesh: 0, material: 1, transform: IDENTITY, parameters: None, extra: None }],
        ..Default::default()
    }];
    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions { format: FORMAT, display: Display::SDR, assets: root, veil: None, pipeline_cache: None },
    )
    .unwrap();
    renderer.preload(&catalog);
    let mut target = Offscreen::new(&device, 64);
    pump(&mut renderer, &scene, &catalog, &mut target, |events| {
        ready(events, "green.wgsl", 1) && ready(events, "sprite.wgsl", 1)
    });
    // Insets pass through scene tone mapping; sprites are already display
    // colors. Measure the opaque inset before checking the compositing order.
    scene.insets[0].after_sprites = 2;
    renderer.submit(&scene, &catalog, &mut target).unwrap();
    let green = middle(&device, &queue, &target.0);
    assert!(green[0] <= 3 && green[1] > 200 && green[2] <= 3, "{green:?}");
    for (boundary, expected) in [(1, [0, i32::from(green[1]) / 2, 128]), (0, [128, 0, 128])] {
        scene.insets[0].after_sprites = boundary;
        renderer.submit(&scene, &catalog, &mut target).unwrap();
        let pixel = middle(&device, &queue, &target.0);
        for channel in 0..3 {
            assert!((i32::from(pixel[channel]) - expected[channel]).abs() <= 3, "boundary {boundary}: {pixel:?}");
        }
    }
}
