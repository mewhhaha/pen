//! Culling on a real (headless) GPU: draws whose shader declares its bounds
//! margin are left out of the views their mesh's sphere lies outside of,
//! alone and in instanced batches (culled on the CPU, and on the GPU), for
//! the camera and the shadow map alike, and what any view sees still draws,
//! at the level of detail its size on screen allows. Skips when no adapter
//! is available.
#![cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

mod common;

use common::*;
use pen_protocol::*;
use pen_render::{RenderStats, Renderer, RendererOptions};

// A triangle in its material's color, alone or instanced; its vertices stay
// where the mesh puts them.
const SHADER: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
const bounds_margin = 0.0;
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
@vertex fn vertex_instanced(@location(0) position: vec3f, @location(3) x: vec4f, @location(4) y: vec4f,
    @location(5) z: vec4f, @location(6) w: vec4f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * mat4x4f(x, y, z, w) * vec4f(position, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return draw.color; }
"#;

fn assets(name: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pen-render-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let write = |key: &str, bytes: &[u8]| std::fs::write(root.join(key), bytes).unwrap();
    write("catalog.json", br#"{"meshes":["tri.mesh.json","square.mesh.json"],"materials":["green.material.json"]}"#);
    write(
        "tri.mesh.json",
        br#"{"positions":[-0.5,-0.5,0.5, 0.5,-0.5,0.5, 0,0.5,0.5],"normals":[0,0,1,0,0,1,0,0,1],
             "uvs":[0,0,1,0,0.5,1],"indices":[0,1,2]}"#,
    );
    // A square, 0.2 across; simpler, the lower left half of it, within an
    // error of 0.01: a third of a pixel when seen whole on 64 pixels.
    write(
        "square.mesh.json",
        br#"{"positions":[-0.1,-0.1,0.5, 0.1,-0.1,0.5, 0.1,0.1,0.5, -0.1,0.1,0.5],
             "normals":[0,0,1,0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,1,1,0,1],"indices":[0,1,2,0,2,3],
             "lods":[{"error":0.01,"indices":[0,1,3]}]}"#,
    );
    write("bounded.wgsl", SHADER.as_bytes());
    // Lit and opaque: it casts shadows.
    write("green.material.json", br#"{"shader":"bounded.wgsl","color":[0,1,0,1],"lighting":"lit"}"#);
    root
}

fn at(x: f32, extra: bool) -> Draw {
    let mut transform = IDENTITY;
    transform[12] = x;
    Draw { material: 0, mesh: 0, transform, parameters: None, extra: extra.then(|| vec![1.0; 4]) }
}

/// Submits `frame` until it draws and nothing is pending, then a few more
/// times, each finished on the GPU, so that what the GPU culled has come
/// back; and returns the last frame's stats.
fn settle(
    renderer: &mut Renderer,
    device: &wgpu::Device,
    frame: &RenderFrame,
    catalog: &Catalog,
    target: &mut Offscreen,
) -> RenderStats {
    let started = Instant::now();
    loop {
        renderer.submit(frame, catalog, target).unwrap();
        let stats = renderer.stats();
        if renderer.pending() == 0 && stats.draw_calls > 0 {
            break;
        }
        assert!(started.elapsed() < Duration::from_secs(20), "the frame settles: {stats:?}");
        renderer.wait(Duration::from_millis(20));
    }
    for _ in 0..4 {
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        renderer.submit(frame, catalog, target).unwrap();
    }
    renderer.stats()
}

fn renderer(device: &wgpu::Device, queue: &wgpu::Queue, name: &str, gpu: bool) -> (Renderer, Catalog) {
    let root = assets(name);
    let catalog = boundary::read_catalog(&root).unwrap();
    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions {
            format: FORMAT,
            display: pen_render::Display::SDR,
            assets: root,
            veil: None,
            pipeline_cache: None,
        },
    )
    .unwrap();
    // Every batch on the GPU, or none; timed, to tell which.
    renderer.gpu_culling(gpu.then_some(1));
    renderer.time_passes(true);
    (renderer, catalog)
}

#[test]
fn culls_what_no_view_sees_and_draws_what_any_does() {
    let Some((device, queue)) = featured_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    for gpu in [false, true] {
        culls_on(&device, &queue, gpu);
    }
}

fn culls_on(device: &wgpu::Device, queue: &wgpu::Queue, gpu: bool) {
    let (mut renderer, catalog) = renderer(device, queue, if gpu { "gpu-culling" } else { "culling" }, gpu);
    let mut target = Offscreen::new(device, 64);
    let settle = |renderer: &mut Renderer, frame: &RenderFrame, target: &mut Offscreen| {
        settle(renderer, device, frame, &catalog, target)
    };

    // The camera sees x within -1..1. Instanced: one in the middle, one
    // reaching in from the right (0.4..1.4), and two well outside; alone
    // (with its own data), one outside.
    let frame = frame(vec![at(0.0, false), at(0.9, false), at(3.0, false), at(-3.0, false), at(5.0, true)]);
    let stats = settle(&mut renderer, &frame, &mut target);
    eprintln!("culled on the {}: {stats:?}", if gpu { "GPU" } else { "CPU" });
    // Where the GPU has timestamps, they show whether it culled.
    if let Some(times) = stats.gpu {
        assert_eq!(times.culling > 0.0, gpu, "{times:?}");
    }
    assert_eq!((stats.draw_calls, stats.instanced_draw_calls, stats.instances), (1, 1, 2), "{gpu}: {stats:?}");
    assert_eq!((stats.culled, stats.shadow_culled, stats.triangles), (3, 0, 2), "{gpu}: {stats:?}");
    let [r, g, b, _] = middle(device, queue, &target.0);
    assert!(r < 2 && b < 2 && g > 200, "middle pixel {r} {g} {b}");

    // With shadows, the camera looking 10 to the right of the shadow map: the
    // one at 0 only casts, the one at 10 only shows, and the one at 30 does
    // neither; alone or instanced alike.
    let mut shadowed = frame.clone();
    shadowed.view[12] = -10.0;
    shadowed.shadow = Shadow { view: IDENTITY, projection: IDENTITY, texel: 1.0 / 64.0, texels: 64 };
    shadowed.draws = vec![at(0.0, false), at(10.0, false), at(30.0, false), at(0.0, true), at(10.0, true)];
    let stats = settle(&mut renderer, &shadowed, &mut target);
    assert_eq!((stats.instanced_draw_calls, stats.instances), (1, 1), "{gpu}: {stats:?}");
    // Camera: the batch and the one alone at 10.
    assert_eq!(stats.draw_calls, 2, "{gpu}: {stats:?}");
    // Out of the camera's view: 0, 30 and 0 alone; out of the shadow map's:
    // 10, 30 and 10 alone.
    assert_eq!((stats.culled, stats.shadow_culled), (3, 3), "{gpu}: {stats:?}");
    let [r, g, b, _] = middle(device, queue, &target.0);
    assert!(r < 2 && b < 2 && g > 200, "middle pixel {r} {g} {b} with shadows");
}

#[test]
fn draws_the_level_of_detail_its_size_on_screen_allows() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    for gpu in [false, true] {
        let (mut renderer, catalog) = renderer(&device, &queue, if gpu { "gpu-lod" } else { "lod" }, gpu);
        let mut target = Offscreen::new(&device, 64);
        // Two squares (instanced) a little up and right of the middle, so
        // that the middle pixel lies in the half the simpler level leaves out.
        let square = |scale: f32| {
            let mut transform = IDENTITY.map(|value| value * scale);
            transform[12] = -0.05 * scale;
            transform[13] = -0.05 * scale;
            transform[14] = 0.5 - 0.5 * scale;
            transform[15] = 1.0;
            Draw { material: 0, mesh: 1, transform, parameters: None, extra: None }
        };
        // As they are, the simpler level's error is a third of a pixel: it
        // draws, one triangle each, and leaves the middle dark.
        let small = frame(vec![square(1.0), square(1.0)]);
        let stats = settle(&mut renderer, &device, &small, &catalog, &mut target);
        assert_eq!((stats.instances, stats.triangles), (2, 2), "{gpu}: {stats:?}");
        let [_, g, _, _] = middle(&device, &queue, &target.0);
        assert!(g < 2, "middle pixel green {g} with the simpler level, gpu {gpu}");
        // Four times the size, 1.3 pixels: whole, and the middle green.
        let large = frame(vec![square(4.0), square(4.0)]);
        let stats = settle(&mut renderer, &device, &large, &catalog, &mut target);
        assert_eq!((stats.instances, stats.triangles), (2, 4), "{gpu}: {stats:?}");
        let [_, g, _, _] = middle(&device, &queue, &target.0);
        assert!(g > 200, "middle pixel green {g} whole, gpu {gpu}");
    }
}
