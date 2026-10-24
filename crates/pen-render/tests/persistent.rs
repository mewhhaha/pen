//! Asset-owned instance records draw, cull and reload without frame uploads.
#![cfg(not(target_arch = "wasm32"))]
mod common;
use common::*;
use pen_protocol::*;
use pen_render::{Renderer, RendererOptions};
use serde_json::json;

const SHADER: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
const bounds_margin = 0.0;
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
// Deliberately elsewhere: persistent meshes must not enter dynamic batches.
@vertex fn vertex_instanced(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return vec4f(position + vec3f(100.0, 0.0, 0.0), 1.0);
}
@vertex fn vertex_persistent(@location(0) position: vec3f, @location(3) offset: vec4f)
    -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position + offset.xyz, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return draw.color; }
"#;

fn mesh(first: f32) -> Vec<u8> {
    let mut records = vec![0.; 40];
    records[0] = first;
    records[20] = -9.3;
    pen_render::pack_mesh(
        &serde_json::to_vec(&json!({
            "positions":[9.8,-0.2,0.5,10.2,-0.2,0.5,10.,0.2,0.5],
            "normals":[0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,0.5,1],"indices":[0,1,2],
            "instances":{"data":records,"min":[-0.2,-0.2,0.5],"max":[0.9,0.2,0.5]}
        }))
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn persistent_geometry_survives_culling_movement_and_reload() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("persistent-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("catalog.json"), br#"{"meshes":["grass.mesh.json"],"materials":["green.material.json"]}"#)
        .unwrap();
    std::fs::write(root.join("grass.mesh.json"), mesh(-10.)).unwrap();
    std::fs::write(root.join("grass.wgsl"), SHADER).unwrap();
    std::fs::write(root.join("green.material.json"), br#"{"shader":"grass.wgsl","color":[0,1,0,1],"lighting":"lit"}"#)
        .unwrap();
    let catalog = boundary::read_catalog(&root).unwrap();
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
    let mut target = Offscreen::new(&device, 64);
    let mut frame = frame(vec![Draw { mesh: 0, material: 0, transform: IDENTITY, parameters: None, extra: None }]);
    frame.shadow = Shadow { view: IDENTITY, projection: IDENTITY, texel: 1. / 64., texels: 64 };
    pump(&mut renderer, &frame, &catalog, &mut target, |events| {
        ready(events, "grass.wgsl", 1) && ready(events, "grass.mesh.json", 1)
    });
    let stats = renderer.stats();
    assert_eq!((stats.draw_calls, stats.instanced_draw_calls, stats.instances, stats.triangles), (1, 1, 2, 2));
    assert_eq!(stats.instance_upload_bytes, 0);
    assert!(
        middle(&device, &queue, &target.0)[1] > 200,
        "aggregate bounds keep visible instances despite offscreen template"
    );
    frame.draws[0].transform[12] = 5.;
    renderer.submit(&frame, &catalog, &mut target).unwrap();
    assert_eq!(renderer.stats().draw_calls, 0);
    assert_eq!(renderer.stats().culled, 1);
    frame.draws[0].transform = IDENTITY;
    std::fs::write(root.join("grass.mesh.json"), mesh(-9.3)).unwrap();
    renderer.reload(&["grass.mesh.json".into()]);
    pump(&mut renderer, &frame, &catalog, &mut target, |events| ready(events, "grass.mesh.json", 2));
    assert!(middle(&device, &queue, &target.0)[1] < 2, "reloading replaces instance data");
    frame.draws[0].transform[12] = -0.7;
    renderer.submit(&frame, &catalog, &mut target).unwrap();
    assert!(middle(&device, &queue, &target.0)[1] > 200, "a draw transform moves the whole patch");
    assert_eq!(renderer.stats().instance_upload_bytes, 0, "moving the patch never rewrites its records");
    // Insets use a separate coverage-preserving pipeline and the same records.
    frame.insets.push(Inset {
        x: 0.,
        y: 0.,
        width: 64.,
        height: 64.,
        view: IDENTITY,
        projection: IDENTITY,
        draws: frame.draws.clone(),
        light_direction: frame.light_direction,
        light_color: frame.light_color,
        ..Default::default()
    });
    frame.draws.clear();
    renderer.submit(&frame, &catalog, &mut target).unwrap();
    assert!(middle(&device, &queue, &target.0)[1] > 200, "persistent instances also draw inside insets");
}
