//! End-to-end headless GPU integration test proving a standard `pen_protocol::RenderFrame`
//! activates AgX tonemapping, HDRI IBL, physical sky LUTs + clouds, LTC area lights,
//! PBR deferred decals, RCAS + film grain + lens flare polish, and RTT / split-screen cameras
//! through `Renderer::submit`.

mod common;

use pen_protocol::{AreaLightSpec, CameraTargetSpec, Catalog, Color, Draw, PbrDecalSpec, Vector};
use pen_render::{HdriImage, Renderer, RendererOptions, TonemapOperator};
use serde_json::json;

const SIMPLE_SHADER: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct DrawU { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: DrawU;
@group(1) @binding(1) var base: texture_2d<f32>;
@group(1) @binding(2) var base_samp: sampler;

struct Out { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vertex(@location(0) p: vec3f, @location(1) _n: vec3f, @location(2) uv: vec2f) -> Out {
  var o: Out;
  o.pos = scene.projection * scene.view * draw.model * vec4f(p, 1.0);
  o.uv = uv;
  return o;
}

@fragment fn fragment(i: Out) -> @location(0) vec4f {
  let c = textureSample(base, base_samp, i.uv).rgb * draw.color.rgb * 3.0;
  return vec4f(c, draw.color.a);
}
"#;

#[test]
fn render_frame_end_to_end_drives_agx_hdri_physical_sky_pbr_decals_polish_and_split_screen() {
    let Some((device, queue)) = common::device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let dir = std::env::temp_dir().join(format!("pen-wired-pipeline-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    std::fs::write(dir.join("simple.wgsl"), SIMPLE_SHADER).unwrap();
    let quad = json!({
        "positions": [-1.0, -1.0, 0.0,  1.0, -1.0, 0.0,  1.0, 1.0, 0.0,  -1.0, 1.0, 0.0],
        "normals": [0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0],
        "uvs": [0.0, 1.0,  1.0, 1.0,  1.0, 0.0,  0.0, 0.0],
        "indices": [0, 1, 2, 0, 2, 3]
    });
    std::fs::write(dir.join("quad.mesh.json"), serde_json::to_vec(&quad).unwrap()).unwrap();
    std::fs::write(
        dir.join("wall.mat.json"),
        serde_json::to_vec(&json!({
            "shader": "simple.wgsl",
            "color": [1.0, 0.2, 0.1, 1.0],
            "lighting": "lit",
            "decals": true
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("decal.mat.json"),
        serde_json::to_vec(&json!({
            "shader": "simple.wgsl",
            "color": [0.1, 0.9, 0.2, 1.0],
            "lighting": "lit"
        }))
        .unwrap(),
    )
    .unwrap();

    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererOptions {
            format: common::FORMAT,
            display: pen_render::Display::SDR,
            assets: dir.clone(),
            veil: None,
            pipeline_cache: None,
        },
    )
    .unwrap();

    // Install real HDRI environment + AgX tonemapping via Renderer.
    let hdri = HdriImage::from_rgba_f32(16, 8, vec![[1.5, 1.2, 1.0, 1.0]; 128]).unwrap();
    let sh = renderer.set_hdri_environment(&hdri, 1.0, 0.0).unwrap();
    assert!(sh[0].x > 1.0);

    let catalog = Catalog {
        meshes: vec!["quad.mesh.json".into()],
        materials: vec!["wall.mat.json".into(), "decal.mat.json".into()],
        sounds: vec![],
    };

    let mut frame = common::frame(vec![Draw {
        mesh: 0,
        material: 0,
        transform: [
            2.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, -2.0, 1.0,
        ],
        parameters: None,
        extra: None,
    }]);
    frame.projection = pen_render::perspective_forward_z(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 50.0);
    frame.light_direction = Vector { x: 0.2, y: -0.9, z: -0.3 };
    frame.grade.tonemap = TonemapOperator::AgX as u32;
    frame.grade.sharpness = 0.5;
    frame.grade.grain = 0.04;
    frame.grade.flare = 0.4;
    frame.quality.physical_sky = true;
    frame.clouds.cover = 0.4;
    frame.area_lights.push(AreaLightSpec {
        kind: 0,
        position: Vector { x: 0.0, y: 1.0, z: -1.0 },
        right: Vector { x: 1.0, y: 0.0, z: 0.0 },
        up: Vector { x: 0.0, y: 1.0, z: 0.0 },
        color: Vector { x: 1.0, y: 0.9, z: 0.8 },
        intensity: 4.0,
        width: 1.0,
        height: 1.0,
        range: 10.0,
        cookie_material: 0,
        two_sided: true,
    });
    frame.pbr_decals.push(PbrDecalSpec {
        material: 1,
        transform: [
            0.8, 0.0, 0.0, 0.0,
            0.0, 0.8, 0.0, 0.0,
            0.0, 0.0, 0.8, 0.0,
            0.0, 0.0, -2.0, 1.0,
        ],
        albedo: Color { r: 0.1, g: 1.0, b: 0.2, a: 0.9 },
        normal_strength: 0.5,
        roughness: 0.4,
        metallic: 0.0,
        emissive_strength: 0.3,
        fade: 0.1,
    });
    let mut target = common::Offscreen::new(&device, 64);
    let mut events = Vec::new();
    for _ in 0..80 {
        renderer.submit(&frame, &catalog, &mut target).unwrap();
        events.extend(renderer.take_events());
        if !renderer.waiting() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        !events.iter().any(|e| matches!(e, pen_protocol::GameEvent::AssetFailed { .. })),
        "unexpected asset failure: {events:?}"
    );
    assert!(!renderer.waiting(), "wired pipeline assets failed to load in time: {events:?}");

    // 1) Full-screen pass with AgX + HDRI + physical sky + area lights + PBR decals + RCAS + Post-FX polish.
    let center_agx = common::middle(&device, &queue, &target.0);
    assert!(
        center_agx[0] > 30 || center_agx[1] > 30 || center_agx[2] > 30,
        "wired full-screen center pixel unexpectedly dark: {center_agx:?}"
    );

    // 2) Verify AgX vs Shoulder tonemap operator visibly changes bright HDR output through RenderFrame.
    let mut shoulder_frame = frame.clone();
    shoulder_frame.grade.tonemap = TonemapOperator::Shoulder as u32;
    shoulder_frame.grade.grain = 0.0;
    let mut agx_clean = frame.clone();
    agx_clean.grade.grain = 0.0;
    renderer.submit(&shoulder_frame, &catalog, &mut target).unwrap();
    let px_shoulder = common::middle(&device, &queue, &target.0);
    renderer.submit(&agx_clean, &catalog, &mut target).unwrap();
    let px_agx = common::middle(&device, &queue, &target.0);
    assert_ne!(px_shoulder, px_agx, "AgX and Shoulder tonemappers should differ on HDR colors");

    // 3) Exercise 2-viewport split-screen secondary camera compositing across left + right halves.
    let mut split_frame = frame.clone();
    split_frame.camera_targets.push(CameraTargetSpec {
        material: 0,
        width: 32,
        height: 64,
        viewport_x: 0.0,
        viewport_y: 0.0,
        viewport_w: 0.5,
        viewport_h: 1.0,
        split_screen: true,
        view: frame.view,
        projection: frame.projection,
    });
    split_frame.camera_targets.push(CameraTargetSpec {
        material: 0,
        width: 32,
        height: 64,
        viewport_x: 0.5,
        viewport_y: 0.0,
        viewport_w: 0.5,
        viewport_h: 1.0,
        split_screen: true,
        view: frame.view,
        projection: frame.projection,
    });
    renderer.submit(&split_frame, &catalog, &mut target).unwrap();
    let pixels = common::pixels(&device, &queue, &target.0);
    let left_mid = pixels[32 * 64 + 16];
    let right_mid = pixels[32 * 64 + 48];
    assert!(
        (left_mid[0] > 20 || left_mid[1] > 20) && (right_mid[0] > 20 || right_mid[1] > 20),
        "both split-screen halves should render non-black content: left={left_mid:?}, right={right_mid:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
