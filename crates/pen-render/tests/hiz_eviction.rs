//! Headless GPU witnesses for Checkpoint 7:
//! 1. Prior-frame Hi-Z occlusion culling in `cull.wgsl` culls instanced batches
//!    hidden behind a foreground opaque wall without false-culling visible ones.
//! 2. Runtime GPU asset residency tracking and LRU eviction reclaim mesh/texture
//!    memory under a budget and transparently reload evicted assets on demand.
#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::cast_precision_loss, reason = "test coordinates")]
mod common;

use std::path::{Path, PathBuf};

use common::{FORMAT, IDENTITY, Offscreen, device, frame, pixels, pump, ready};
use pen_protocol::*;
use pen_render::{Renderer, RendererOptions};
use serde_json::json;

struct TempDir(PathBuf);
impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("pen-render-hiz-evict-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const INSTANCED_SHADER: &str = r#"
const bounds_margin = 0.0;
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@group(1) @binding(1) var base: texture_2d<f32>;
@group(1) @binding(2) var base_sampler: sampler;

struct Out {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex fn vertex(
  @location(0) position: vec3f,
  @location(1) _normal: vec3f,
  @location(2) uv: vec2f,
) -> Out {
  var out: Out;
  out.position = scene.projection * scene.view * draw.model * vec4f(position, 1.0);
  out.uv = uv;
  return out;
}

@vertex fn vertex_instanced(
  @location(0) position: vec3f,
  @location(1) _normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) m0: vec4f,
  @location(4) m1: vec4f,
  @location(5) m2: vec4f,
  @location(6) m3: vec4f,
  @location(7) _params: vec4f,
) -> Out {
  let model = mat4x4f(m0, m1, m2, m3);
  var out: Out;
  out.position = scene.projection * scene.view * model * vec4f(position, 1.0);
  out.uv = uv;
  return out;
}

@fragment fn fragment(input: Out) -> @location(0) vec4f {
  let tex = textureSample(base, base_sampler, input.uv).rgb;
  return vec4f(draw.color.rgb * tex, draw.color.a);
}
"#;

fn perspective_90(aspect: f32) -> Matrix {
    let (near, far) = (0.1f32, 100.0f32);
    let mut m = [0.0; 16];
    m[0] = 1.0 / aspect;
    m[5] = 1.0;
    m[10] = far / (near - far);
    m[11] = -1.0;
    m[14] = near * far / (near - far);
    m
}

fn translate_scale(x: f32, y: f32, z: f32, sx: f32, sy: f32, sz: f32) -> Matrix {
    [sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, sz, 0.0, x, y, z, 1.0]
}

fn write_quad_mesh(dir: &Path, name: &str, half_w: f32, half_h: f32) {
    let mesh = json!({
        "positions": [
            -half_w, -half_h, 0.0,
             half_w, -half_h, 0.0,
             half_w,  half_h, 0.0,
            -half_w,  half_h, 0.0
        ],
        "normals": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        "uvs": [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "indices": [0, 1, 2, 0, 2, 3]
    });
    std::fs::write(dir.join(name), serde_json::to_vec(&mesh).unwrap()).unwrap();
}

fn write_solid_png(path: &Path, size: u32, rgba: [u8; 4]) {
    image::RgbaImage::from_pixel(size, size, image::Rgba(rgba)).save(path).unwrap();
}

#[test]
fn hiz_occlusion_culls_instances_hidden_behind_an_opaque_wall() {
    let Some((device, queue)) = device() else { return };
    let dir = TempDir::new("hiz");
    let root = dir.path();
    std::fs::write(root.join("inst.wgsl"), INSTANCED_SHADER).unwrap();
    write_quad_mesh(root, "wall.json", 1.0, 1.0);
    write_quad_mesh(root, "card.json", 0.25, 0.25);
    std::fs::write(
        root.join("wall.mat.json"),
        serde_json::to_vec(&json!({ "shader": "inst.wgsl", "color": [0.1, 0.2, 0.9, 1.0], "lighting": "unlit" }))
            .unwrap(),
    )
    .unwrap();
    std::fs::write(
        root.join("card.mat.json"),
        serde_json::to_vec(&json!({ "shader": "inst.wgsl", "color": [0.95, 0.15, 0.1, 1.0], "lighting": "unlit" }))
            .unwrap(),
    )
    .unwrap();
    let catalog = Catalog {
        meshes: vec!["wall.json".into(), "card.json".into()],
        materials: vec!["wall.mat.json".into(), "card.mat.json".into()],
        sounds: vec![],
    };
    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions {
            format: FORMAT,
            display: pen_render::Display::SDR,
            assets: root.to_path_buf(),
            veil: None,
            pipeline_cache: None,
        },
    )
    .unwrap();
    // Lower the GPU culling threshold so our 64-instance batch runs on the GPU compute pass.
    renderer.gpu_culling(Some(8));

    let mut target = Offscreen::new(&device, 256);
    let mut draws = Vec::with_capacity(65);
    // Large blue wall covering the entire left half of the viewport at z = -2.5.
    draws.push(Draw {
        mesh: 0,
        material: 0,
        transform: translate_scale(-1.4, 0.0, -2.5, 1.4, 2.8, 1.0),
        parameters: None,
        extra: None,
    });
    // 64 instanced red cards at z = -8.0: 32 hidden behind the left wall, 32 visible on the right.
    for i in 0..32u32 {
        let col = (i % 4) as f32;
        let row = (i / 4) as f32;
        let y = -3.5 + row * 1.0;
        draws.push(Draw {
            mesh: 1,
            material: 1,
            transform: translate_scale(-5.5 + col * 1.2, y, -8.0, 1.0, 1.0, 1.0),
            parameters: None,
            extra: None,
        });
        draws.push(Draw {
            mesh: 1,
            material: 1,
            transform: translate_scale(1.6 + col * 1.2, y, -8.0, 1.0, 1.0, 1.0),
            parameters: None,
            extra: None,
        });
    }
    let mut f = frame(draws);
    f.view = IDENTITY;
    f.projection = perspective_90(1.0);
    f.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    f.quality.msaa = 1;
    f.quality.fxaa = false;
    f.quality.bloom = false;
    f.quality.sky = false;
    f.quality.occlusion = false;

    pump(&mut renderer, &f, &catalog, &mut target, |events| {
        ready(events, "wall.json", 1)
            && ready(events, "card.json", 1)
            && ready(events, "inst.wgsl", 1)
            && ready(events, "wall.mat.json", 1)
            && ready(events, "card.mat.json", 1)
    });
    // Wait until background GPU culling pipeline compile finishes too.
    while renderer.pending() > 0 {
        renderer.wait(std::time::Duration::from_millis(50));
        renderer.submit(&f, &catalog, &mut target).unwrap();
    }

    // Run several frames so Hi-Z builds and GPU culling readbacks settle.
    for _ in 0..8 {
        renderer.submit(&f, &catalog, &mut target).unwrap();
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }
    let stats_on = renderer.stats();
    assert_eq!(stats_on.instances, 33, "wall (1) + 32 visible cards should draw: {stats_on:?}");
    assert_eq!(stats_on.culled, 32, "all 32 cards behind the opaque wall should be Hi-Z culled: {stats_on:?}");

    // Verify rendered pixels: left half is blue wall, right half still shows visible red cards.
    let img = pixels(&device, &queue, &target.0);
    let wall_px = img[128 * 256 + 64];
    assert!(wall_px[2] > 180 && wall_px[0] < 80, "left half should show blue wall: {wall_px:?}");
    let right_red_pixels = (0..256)
        .flat_map(|y| (140..240).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            let p = img[y * 256 + x];
            p[0] > 180 && p[2] < 80
        })
        .count();
    assert!(right_red_pixels > 200, "visible red cards must not be false-culled (count={right_red_pixels})");

    // Disable Hi-Z occlusion culling: all 64 cards (+ 1 wall) must now survive frustum culling.
    renderer.occlusion_culling(false);
    for _ in 0..8 {
        renderer.submit(&f, &catalog, &mut target).unwrap();
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }
    let stats_off = renderer.stats();
    assert_eq!(stats_off.instances, 65, "with Hi-Z off all 64 cards + wall draw: {stats_off:?}");
    assert_eq!(stats_off.culled, 0, "none should be frustum culled: {stats_off:?}");
}

#[test]
fn gpu_asset_lru_eviction_reclaims_memory_and_reloads_on_demand() {
    let Some((device, queue)) = device() else { return };
    let dir = TempDir::new("evict");
    let root = dir.path();
    std::fs::write(root.join("inst.wgsl"), INSTANCED_SHADER).unwrap();
    write_quad_mesh(root, "a.json", 0.8, 0.8);
    write_quad_mesh(root, "b.json", 0.8, 0.8);
    write_solid_png(&root.join("a.png"), 64, [240, 30, 30, 255]);
    write_solid_png(&root.join("b.png"), 64, [30, 240, 40, 255]);
    std::fs::write(
        root.join("a.mat.json"),
        serde_json::to_vec(&json!({
            "shader": "inst.wgsl", "texture": "a.png", "color": [1.0, 1.0, 1.0, 1.0], "lighting": "unlit"
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        root.join("b.mat.json"),
        serde_json::to_vec(&json!({
            "shader": "inst.wgsl", "texture": "b.png", "color": [1.0, 1.0, 1.0, 1.0], "lighting": "unlit"
        }))
        .unwrap(),
    )
    .unwrap();
    let catalog = Catalog {
        meshes: vec!["a.json".into(), "b.json".into()],
        materials: vec!["a.mat.json".into(), "b.mat.json".into()],
        sounds: vec![],
    };
    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions {
            format: FORMAT,
            display: pen_render::Display::SDR,
            assets: root.to_path_buf(),
            veil: None,
            pipeline_cache: None,
        },
    )
    .unwrap();
    let mut target = Offscreen::new(&device, 128);

    let make_frame = |draw_a: bool, draw_b: bool| {
        let mut draws = Vec::new();
        if draw_a {
            draws.push(Draw {
                mesh: 0,
                material: 0,
                transform: translate_scale(0.0, 0.0, -2.0, 1.0, 1.0, 1.0),
                parameters: None,
                extra: None,
            });
        }
        if draw_b {
            draws.push(Draw {
                mesh: 1,
                material: 1,
                transform: translate_scale(0.0, 0.0, -2.2, 1.0, 1.0, 1.0),
                parameters: None,
                extra: None,
            });
        }
        let mut f = frame(draws);
        f.view = IDENTITY;
        f.projection = perspective_90(1.0);
        f.quality.msaa = 1;
        f.quality.fxaa = false;
        f.quality.bloom = false;
        f.quality.sky = false;
        f.quality.occlusion = false;
        f
    };

    // Load both A and B.
    let both = make_frame(true, true);
    pump(&mut renderer, &both, &catalog, &mut target, |events| {
        ready(events, "a.json", 1)
            && ready(events, "b.json", 1)
            && ready(events, "a.png", 1)
            && ready(events, "b.png", 1)
            && ready(events, "inst.wgsl", 1)
    });
    let both_bytes = renderer.gpu_asset_bytes();
    assert!(both_bytes > 32_000, "two 64x64 mipmapped textures + meshes must occupy >32 KB: {both_bytes}");

    // Draw only A for 5 frames so B ages out.
    let only_a = make_frame(true, false);
    for _ in 0..5 {
        renderer.submit(&only_a, &catalog, &mut target).unwrap();
    }

    // Evict stale assets (age >= 3 frames) down to half the resident footprint.
    let evicted = renderer.evict_unused(3, both_bytes / 2);
    assert!(evicted >= 1, "B's unused texture/mesh should be evicted");
    let after_evict = renderer.gpu_asset_bytes();
    assert!(after_evict <= both_bytes / 2, "footprint {after_evict} should drop below {}", both_bytes / 2);
    assert_eq!(
        renderer.standing(devtools_asset::MESH, "a.json").0,
        devtools_asset_state::READY,
        "recently drawn mesh A must remain resident"
    );

    // Budget-driven automatic eviction: set a tight budget and draw only A; all of B evicts cleanly.
    renderer.set_asset_budget(Some(1));
    for _ in 0..3 {
        renderer.submit(&only_a, &catalog, &mut target).unwrap();
    }
    assert_eq!(
        renderer.standing(devtools_asset::MESH, "b.json").0,
        devtools_asset_state::UNUSED,
        "stale mesh B should be evicted by budget"
    );
    assert_eq!(
        renderer.standing(devtools_asset::MESH, "a.json").0,
        devtools_asset_state::READY,
        "active mesh A must never be evicted by budget"
    );

    // Clear budget and draw only B: evicted mesh + texture must transparently reload and render green!
    renderer.set_asset_budget(None);
    let only_b = make_frame(false, true);
    pump(&mut renderer, &only_b, &catalog, &mut target, |events| {
        ready(events, "b.json", 1) && ready(events, "b.png", 1)
    });
    renderer.submit(&only_b, &catalog, &mut target).unwrap();
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    assert_eq!(renderer.standing(devtools_asset::MESH, "b.json").0, devtools_asset_state::READY);
    let img = pixels(&device, &queue, &target.0);
    let center = img[64 * 128 + 64];
    assert!(center[1] > 180 && center[0] < 80, "reloaded asset B must render bright green: {center:?}");
}
