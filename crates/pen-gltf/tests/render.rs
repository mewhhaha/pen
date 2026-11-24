//! Imported models drawn on a real (headless) GPU: a skinned arm posed by
//! its clips at two times, blended, and in its bind pose without layers;
//! and a crowd of the same arm baked into a vertex-animation texture, one
//! instanced draw whose members each play their own time. Each writes PNG
//! shots ($PEN_SHOTS, or the tests' temporary directory). Skips when no
//! adapter is available.
mod common;

use common::gpu::*;
use common::*;
use pen_gltf::{Options, VatOptions, import};
use pen_protocol::*;

// The arm's red, moved by its joints when posed; in its bind pose otherwise.
const SKINNED: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@group(3) @binding(0) var<storage, read> joints: array<mat4x4f>;
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
struct Skinned { @location(0) position: vec3f, @location(9) joints: vec4u, @location(10) weights: vec4f };
@vertex fn vertex_skinned(input: Skinned) -> @builtin(position) vec4f {
  let skin = joints[input.joints.x] * input.weights.x + joints[input.joints.y] * input.weights.y
    + joints[input.joints.z] * input.weights.z + joints[input.joints.w] * input.weights.w;
  return scene.projection * scene.view * draw.model * skin * vec4f(input.position, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return draw.color; }
"#;

fn layer(clip: u32, time: f32, weight: f32) -> Pose {
    Pose { draw: 0, clip, time, weight, inset: 0 }
}

#[test]
fn skinned_draws_take_their_pose() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let directory = directory("skinned");
    let assets = directory.join("assets");
    let options = Options { shader: "skinned.wgsl".to_owned(), ..Options::default() };
    import(&arm_gltf(&directory), &assets, &options).unwrap();
    std::fs::write(assets.join("skinned.wgsl"), SKINNED).unwrap();
    std::fs::write(
        assets.join("catalog.json"),
        r#"{"meshes":["arm_column.mesh.json"],"materials":["arm_red.material.json"]}"#,
    )
    .unwrap();
    let catalog = boundary::read_catalog(&assets).unwrap();
    let mut renderer = renderer(&device, &queue, &assets);
    renderer.preload(&catalog);
    let mut target = Offscreen::new(&device);
    let draw = Draw { mesh: 0, material: 0, transform: IDENTITY, parameters: None, extra: None };
    let keys = ["arm_column.mesh.json", "arm_red.material.json", "skinned.wgsl", "arm_white.png"];
    // The assets' readiness is reported once, to the first frames.
    let mut waiting: &[&str] = &keys;
    let mut shot = |poses: Vec<Pose>, name: &str| {
        let frame = frame(vec![draw.clone()], poses);
        settle(&mut renderer, &frame, &catalog, &mut target, std::mem::take(&mut waiting));
        let pixels = pixels(&device, &queue, &target.0);
        eprintln!("{}", save(&pixels, name).display());
        pixels
    };

    // The bend's start: the arm straight up, as it was bound.
    let start = shot(vec![layer(0, 0.0, 1.0)], "skinned-bend-0.00s.png");
    assert!(lit(seen(&start, 0.0, 1.75)) && lit(seen(&start, 0.0, 0.25)), "the arm stands straight");
    assert!(!lit(seen(&start, -0.6, 1.0)));
    // Near its end: the elbow has turned the forearm nearly 90 degrees,
    // toward -x; below the elbow nothing moved.
    let bent = shot(vec![layer(0, 0.99, 1.0)], "skinned-bend-0.99s.png");
    assert!(!lit(seen(&bent, 0.0, 1.75)), "the forearm has left the vertical");
    assert!(lit(seen(&bent, -0.6, 1.0)) && lit(seen(&bent, 0.0, 0.25)), "it lies along -x from the elbow");
    assert_ne!(start, bent);
    // Both evenly: halfway, 45 degrees.
    let blended = shot(vec![layer(0, 0.0, 1.0), layer(0, 0.99, 1.0)], "skinned-blend.png");
    assert!(lit(seen(&blended, -0.45, 1.45)) && !lit(seen(&blended, 0.0, 1.75)) && !lit(seen(&blended, -0.6, 1.0)));
    // The lift's step a second in raises the whole arm half a unit.
    let lifted = shot(vec![layer(1, 1.5, 1.0)], "skinned-lift-1.50s.png");
    assert!(!lit(seen(&lifted, 0.0, 0.25)) && lit(seen(&lifted, 0.0, 2.25)));
    // Without layers it draws as it is: its bind pose, straight up.
    let unposed = shot(Vec::new(), "skinned-unposed.png");
    assert_eq!(unposed, start);

    // Skinned shadows: the same pose cast into the sun's shadow map.
    let mut sunny = frame(vec![draw.clone()], vec![layer(0, 0.99, 1.0)]);
    sunny.shadow = Shadow { view: IDENTITY, texels: 256, ..Shadow::default() };
    sunny.cascades = vec![Cascade { projection: PROJECTION, texel: 0.01, far: 100.0 }];
    settle(&mut renderer, &sunny, &catalog, &mut target, &[]);
    assert_eq!(pixels(&device, &queue, &target.0), bent);
}

#[test]
fn vertex_animation_crowds_draw_as_one_instanced_draw() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let directory = directory("crowd");
    let assets = directory.join("assets");
    let options = Options { vat: Some(VatOptions { fps: 30.0, shader: "vat.wgsl".to_owned() }), ..Options::default() };
    import(&arm_gltf(&directory), &assets, &options).unwrap();
    std::fs::write(
        assets.join("catalog.json"),
        r#"{"meshes":["arm_column.mesh.json"],"materials":["arm_column_vat.material.json"]}"#,
    )
    .unwrap();
    let catalog = boundary::read_catalog(&assets).unwrap();
    let mut renderer = renderer(&device, &queue, &assets);
    renderer.preload(&catalog);
    let mut target = Offscreen::new(&device);
    let keys =
        ["arm_column.mesh.json", "arm_column_vat.material.json", "vat.wgsl", "arm_column_vat.gtex", "arm_white.png"];
    // Three arms, each its own time into the bend: none, halfway, nearly all.
    let crowd = |offset: f32| {
        [(-1.0, 0.0), (0.0, 0.5), (1.0, 0.9)]
            .map(|(x, time)| Draw {
                mesh: 0,
                material: 0,
                transform: at(x),
                parameters: Some([0.0, time + offset, 0.0, 0.0]),
                extra: None,
            })
            .to_vec()
    };
    let frame_now = frame(crowd(0.0), Vec::new());
    settle(&mut renderer, &frame_now, &catalog, &mut target, &keys);
    assert_eq!(
        (renderer.stats().draw_calls, renderer.stats().instanced_draw_calls, renderer.stats().instances),
        (1, 1, 3),
        "the crowd is one instanced draw"
    );
    let now = pixels(&device, &queue, &target.0);
    eprintln!("{}", save(&now, "vat-crowd-0.00s.png").display());
    assert!(lit(seen(&now, -1.0, 1.75)), "the first stands straight");
    assert!(lit(seen(&now, -0.45, 1.45)) && !lit(seen(&now, 0.0, 1.75)), "the second leans halfway");
    assert!(lit(seen(&now, 0.45, 1.08)) && !lit(seen(&now, 1.0, 1.75)), "the third lies nearly flat");
    // A tenth of a second on, each has moved on along its clip, and the
    // last has come round to its start.
    let frame_later = frame(crowd(0.1), Vec::new());
    settle(&mut renderer, &frame_later, &catalog, &mut target, &[]);
    let later = pixels(&device, &queue, &target.0);
    eprintln!("{}", save(&later, "vat-crowd-0.10s.png").display());
    assert!(!lit(seen(&later, -1.0, 1.95)) && lit(seen(&later, -1.15, 1.95)), "the first has begun to bend");
    assert!(lit(seen(&later, 1.0, 1.75)), "the third stands again");
    assert_ne!(now, later);
}

// A quad's first tangent as its colour: xyz scaled to 0.2 to 0.8.
const TANGENT: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
struct Corner { @builtin(position) position: vec4f, @location(0) tangent: vec4f };
@vertex fn vertex(@location(0) position: vec3f, @location(11) tangent: vec4f) -> Corner {
  return Corner(scene.projection * scene.view * draw.model * vec4f(position, 1.0), tangent);
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  return vec4f(corner.tangent.xyz * 0.3 + 0.5, 1.0);
}
"#;

#[test]
fn imported_tangents_reach_the_shader() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let directory = directory("tangent-shader");
    let assets = directory.join("assets");
    // A normal-mapped quad with its texture mirrored: made tangents face -x, left-handed.
    let mut quad = square();
    for uv in &mut quad.uvs {
        uv[0] = 1.0 - uv[0];
    }
    let material = serde_json::json!({"name": "mapped", "normalTexture": {"index": 0}});
    let file = model_file(&directory, "quad", model(&quad, material, &[], serde_json::json!({})));
    let options = Options { shader: "tangent.wgsl".to_owned(), ..Options::default() };
    let imported = import(&file, &assets, &options).unwrap();
    std::fs::write(assets.join("tangent.wgsl"), TANGENT).unwrap();
    std::fs::write(
        assets.join("catalog.json"),
        serde_json::json!({"meshes": [imported.meshes[0].mesh], "materials": [imported.materials[0]]}).to_string(),
    )
    .unwrap();
    let catalog = boundary::read_catalog(&assets).unwrap();
    let mut renderer = renderer(&device, &queue, &assets);
    renderer.preload(&catalog);
    let mut target = Offscreen::new(&device);
    let draw = Draw { mesh: 0, material: 0, transform: IDENTITY, parameters: None, extra: None };
    let frame = frame(vec![draw], Vec::new());
    let keys = [imported.meshes[0].mesh.as_str(), imported.materials[0].as_str(), "tangent.wgsl"];
    settle(&mut renderer, &frame, &catalog, &mut target, &keys);
    let shot = pixels(&device, &queue, &target.0);
    eprintln!("{}", save(&shot, "imported-tangents.png").display());
    let got = seen(&shot, 0.5, 0.5);
    // (-1, 0, 0) as 0.2, 0.5, 0.5 light: 51, 128, 128.
    assert!(got[0].abs_diff(51) <= 2 && got[1].abs_diff(128) <= 2 && got[2].abs_diff(128) <= 2, "{got:?}");
}

// Half-transparent green, opaque red.
const GLASS: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 0.5); }
"#;

const RED: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }
"#;

#[test]
fn an_imported_transmissive_material_draws_among_the_translucent_ones() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let directory = directory("transmission");
    let assets = directory.join("assets");
    // Red behind, glass in front: the glass half the size, with a factor of 0 or 0.5.
    let behind = model_file(
        &directory,
        "behind",
        model(&square(), serde_json::json!({"name": "red"}), &[], serde_json::json!({"translation": [0.0, 0.0, -0.3]})),
    );
    let behind = import(&behind, &assets, &Options { shader: "red.wgsl".to_owned(), ..Options::default() }).unwrap();
    std::fs::write(assets.join("red.wgsl"), RED).unwrap();
    std::fs::write(assets.join("glass.wgsl"), GLASS).unwrap();
    let mut pictures = Vec::new();
    for (name, transmission) in [("clear", 0.0), ("glass", 0.5)] {
        let glass = serde_json::json!({
            "name": name,
            "extensions": {"KHR_materials_transmission": {"transmissionFactor": transmission}},
        });
        let placement = serde_json::json!({"translation": [0.25, 0.25, 0.3], "scale": [0.5, 0.5, 1.0]});
        let file = model_file(&directory, name, model(&square(), glass, &["KHR_materials_transmission"], placement));
        let options = Options { name: Some(name.to_owned()), shader: "glass.wgsl".to_owned(), ..Options::default() };
        let front = import(&file, &assets, &options).unwrap();
        std::fs::write(
            assets.join("catalog.json"),
            serde_json::json!({
                "meshes": [behind.meshes[0].mesh, front.meshes[0].mesh],
                "materials": [behind.materials[0], front.materials[0]],
            })
            .to_string(),
        )
        .unwrap();
        let catalog = boundary::read_catalog(&assets).unwrap();
        let mut renderer = renderer(&device, &queue, &assets);
        renderer.preload(&catalog);
        let mut target = Offscreen::new(&device);
        let draws = vec![
            Draw { mesh: 0, material: 0, transform: IDENTITY, parameters: None, extra: None },
            Draw { mesh: 1, material: 1, transform: IDENTITY, parameters: None, extra: None },
        ];
        let frame = frame(draws, Vec::new());
        settle(&mut renderer, &frame, &catalog, &mut target, &[]);
        let shot = pixels(&device, &queue, &target.0);
        eprintln!("{}", save(&shot, &format!("imported-{name}.png")).display());
        pictures.push((seen(&shot, 0.5, 0.5), seen(&shot, 0.1, 0.1)));
    }
    let [(clear, clear_outside), (glass, glass_outside)] = pictures[..] else { unreachable!() };
    // Without transmission the quad is opaque: its green replaces the red.
    assert!(clear[1] > 200 && clear[0] < 30, "{clear:?}");
    // With it the same alpha blends over the red behind: half and half.
    assert!(glass[0].abs_diff(128) <= 3 && glass[1].abs_diff(128) <= 3 && glass[2] < 5, "{glass:?}");
    // Beyond the glass the red shows alone.
    assert!(clear_outside[0] > 200 && glass_outside[0] > 200 && glass_outside[1] < 30, "{glass_outside:?}");
}
