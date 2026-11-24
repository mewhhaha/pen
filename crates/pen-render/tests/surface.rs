//! A material's `surface` block reaches the shader: the draw uniform's last
//! five `vec4f`s (`draw.surface`, at byte 688) hold it for every way a draw is
//! made (alone, in an instanced batch, posed and skinned), the defaults stand
//! without a block, two surfaces never share a batch, a hot reload of the
//! block takes, and a transmissive material draws among the translucent ones.
//! Each test shader writes a swatch of one component of `draw.surface` (half of
//! it, to fit the picture) as its colour, and the pixels are compared with what
//! the present pass shows of that light. Skips when no adapter is available.
#![cfg(not(target_arch = "wasm32"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "picture coordinates and 8-bit colours of a 128-pixel test image"
)]
mod common;

use common::picture::*;
use common::*;
use pen_protocol::*;
use serde_json::json;

// The draw uniform to its end, and a swatch for each of the ten things the
// surface block holds in `draw.surface`: swatches 0 to 4 show the xyz of each
// of its five vectors and 5 to 9 their w, as grey. Which swatch is the
// draw's first parameter, which a batch brings with each instance.
const BASE: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw {
  model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f,
  extra: array<vec4f, 32>, surface: array<vec4f, 5>,
};
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) which: f32,
};
@vertex fn vertex(@location(0) position: vec3f) -> Corner {
  return Corner(scene.projection * scene.view * draw.model * vec4f(position, 1.0), draw.parameters.x);
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let which = u32(round(corner.which));
  let vector = draw.surface[which % 5u];
  return vec4f(select(vector.xyz, vec3f(vector.w), which >= 5u) * 0.5, 1.0);
}
"#;

const INSTANCED: &str = r#"
@vertex fn vertex_instanced(
  @location(0) position: vec3f,
  @location(3) m0: vec4f, @location(4) m1: vec4f, @location(5) m2: vec4f, @location(6) m3: vec4f,
  @location(7) parameters: vec4f,
) -> Corner {
  return Corner(scene.projection * scene.view * mat4x4f(m0, m1, m2, m3) * vec4f(position, 1.0), parameters.x);
}
"#;

const SKINNED: &str = r#"
@group(3) @binding(0) var<storage, read> joints: array<mat4x4f>;
struct Skinned { @location(0) position: vec3f, @location(9) joints: vec4u, @location(10) weights: vec4f };
@vertex fn vertex_skinned(input: Skinned) -> Corner {
  let skin = joints[input.joints.x] * input.weights.x + joints[input.joints.y] * input.weights.y
    + joints[input.joints.z] * input.weights.z + joints[input.joints.w] * input.weights.w;
  let world = scene.projection * scene.view * draw.model * skin * vec4f(input.position, 1.0);
  return Corner(world, draw.parameters.x);
}
"#;

const SQUARE: [[f32; 3]; 4] = [[-0.12, -0.12, 0.5], [0.12, -0.12, 0.5], [0.12, 0.12, 0.5], [-0.12, 0.12, 0.5]];

fn square(skinned: bool) -> Vec<u8> {
    let mut mesh = json!({
        "positions": SQUARE.concat(),
        "normals": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        "uvs": [0, 0, 1, 0, 1, 1, 0, 1],
        "indices": [0, 1, 2, 0, 2, 3],
    });
    if skinned {
        mesh["joints"] = json!([0, 0, 0, 0].repeat(4));
        mesh["weights"] = json!([1, 0, 0, 0].repeat(4));
        mesh["skeleton"] = json!([{
            "name": "root", "translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1],
        }]);
        mesh["skin"] = json!([{
            "node": 0, "inverseBind": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        }]);
        mesh["clips"] = json!([{"name": "still", "duration": 1, "channels": []}]);
    }
    serde_json::to_vec(&mesh).unwrap()
}

/// What the five vectors hold for the surface a material gives in `block`:
/// the documented defaults, with these keys set.
const DEFAULTS: [[f32; 4]; 5] =
    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.5, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]];

/// A block that sets every key to something its own, so that no two of the
/// twenty numbers are mistaken for each other; and its vectors.
fn full_block() -> (serde_json::Value, [[f32; 4]; 5]) {
    let block = json!({
        "clearcoat": 0.9, "clearcoatRoughness": 0.5, "sheenColor": [0.2, 0.6, 0.4], "sheenRoughness": 0.3,
        "anisotropy": 0.8, "anisotropyRotation": 1.2, "transmission": 0.0, "ior": 1.7, "thickness": 1.5,
        "attenuationColor": [0.3, 0.7, 0.9], "attenuationDistance": 0.6, "specular": 0.45,
        "specularColor": [1.6, 0.8, 0.2], "emissiveStrength": 1.8,
    });
    let vectors =
        [[0.9, 0.5, 0.3, 0.8], [0.2, 0.6, 0.4, 1.2], [0.0, 1.7, 1.5, 0.6], [0.3, 0.7, 0.9, 0.45], [1.6, 0.8, 0.2, 1.8]];
    (block, vectors)
}

/// The swatch `which`'s colour for `vectors`: what the shader writes of it.
fn swatch_color(vectors: &[[f32; 4]; 5], which: usize) -> [u8; 3] {
    let vector = vectors[which % 5];
    let rgb = if which >= 5 { [vector[3]; 3] } else { [vector[0], vector[1], vector[2]] };
    rgb.map(|value| shown(value * 0.5))
}

/// Where swatch `which` sits in clip space.
fn center(which: usize) -> (f32, f32) {
    (-0.8 + 0.4 * (which % 5) as f32, if which < 5 { 0.5 } else { -0.5 })
}

fn draw(mesh: u32, material: u32, which: usize) -> Draw {
    let (x, y) = center(which);
    Draw { mesh, material, transform: moved(x, y), parameters: Some([which as f32, 0.0, 0.0, 0.0]), extra: None }
}

/// Every swatch of `pixels` against what `vectors` should show, as one
/// message of those that differ.
fn wrong_swatches(pixels: &[[u8; 4]], swatches: &[(usize, &[[f32; 4]; 5])]) -> Vec<String> {
    const NAMES: [&str; 10] = [
        "clearcoat, clearcoat roughness, sheen roughness",
        "sheen colour",
        "transmission, ior, thickness",
        "attenuation colour",
        "specular colour",
        "anisotropy",
        "anisotropy rotation",
        "attenuation distance",
        "specular",
        "emissive strength",
    ];
    swatches
        .iter()
        .filter_map(|&(which, vectors)| {
            let (x, y) = center(which);
            let (got, want) = (at(pixels, x, y), swatch_color(vectors, which));
            (!near(got, want)).then(|| format!("swatch {which} ({}): drew {got:?}, wanted {want:?}", NAMES[which]))
        })
        .collect()
}

const ALL: [usize; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

fn surface_material(shader: &str, block: Option<&serde_json::Value>, parameters: bool) -> Vec<u8> {
    let mut material = json!({"shader": shader, "color": [1, 1, 1, 1], "lighting": "lit"});
    if let Some(block) = block {
        material["surface"] = block.clone();
    }
    if parameters {
        material["instanceParameters"] = json!([0, 0, 0, 0]);
    }
    material.to_string().into_bytes()
}

#[test]
fn every_draw_of_a_plain_path_reads_its_materials_surface() {
    let (block, vectors) = full_block();
    let files = [
        ("quad.mesh.json", square(false)),
        ("plain.wgsl", BASE.as_bytes().to_vec()),
        ("coated.material.json", surface_material("plain.wgsl", Some(&block), false)),
        ("bare.material.json", surface_material("plain.wgsl", None, false)),
    ];
    let Some(mut rig) =
        Rig::new("surface-plain", &["quad.mesh.json"], &["coated.material.json", "bare.material.json"], &files)
    else {
        return;
    };
    // The top row with the block, the bottom row without: in one frame, so
    // each draw's uniform has its own material's.
    let draws: Vec<Draw> = ALL.iter().map(|&which| draw(0, if which < 5 { 0 } else { 1 }, which)).collect();
    let picture = rig.show(&frame(draws));
    let top: Vec<(usize, &[[f32; 4]; 5])> = (0..5).map(|which| (which, &vectors)).collect();
    let bottom: Vec<(usize, &[[f32; 4]; 5])> = (5..10).map(|which| (which, &DEFAULTS)).collect();
    let wrong = wrong_swatches(&picture, &[top, bottom].concat());
    assert!(wrong.is_empty(), "{wrong:#?}");
    assert_eq!(rig.renderer.stats().instanced_draw_calls, 0, "a plain shader draws each alone");
    // Again with the materials the other way round: nothing sticks to a slot.
    let draws: Vec<Draw> = ALL.iter().map(|&which| draw(0, if which < 5 { 1 } else { 0 }, which)).collect();
    let picture = rig.show(&frame(draws));
    let top: Vec<(usize, &[[f32; 4]; 5])> = (0..5).map(|which| (which, &DEFAULTS)).collect();
    let bottom: Vec<(usize, &[[f32; 4]; 5])> = (5..10).map(|which| (which, &vectors)).collect();
    let wrong = wrong_swatches(&picture, &[top, bottom].concat());
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn an_instanced_batch_reads_the_surface_of_its_material() {
    let (block, vectors) = full_block();
    let shader = format!("{BASE}{INSTANCED}");
    let files = [
        ("quad.mesh.json", square(false)),
        ("batch.wgsl", shader.into_bytes()),
        ("coated.material.json", surface_material("batch.wgsl", Some(&block), false)),
        ("bare.material.json", surface_material("batch.wgsl", None, false)),
    ];
    let Some(mut rig) =
        Rig::new("surface-batch", &["quad.mesh.json"], &["coated.material.json", "bare.material.json"], &files)
    else {
        return;
    };
    // Each material's five swatches are one batch of its own.
    let draws: Vec<Draw> = ALL.iter().map(|&which| draw(0, if which < 5 { 0 } else { 1 }, which)).collect();
    let picture = rig.show(&frame(draws));
    let stats = rig.renderer.stats();
    assert_eq!((stats.instanced_draw_calls, stats.instances), (2, 10), "two batches of five: {stats:?}");
    let top: Vec<(usize, &[[f32; 4]; 5])> = (0..5).map(|which| (which, &vectors)).collect();
    let bottom: Vec<(usize, &[[f32; 4]; 5])> = (5..10).map(|which| (which, &DEFAULTS)).collect();
    let wrong = wrong_swatches(&picture, &[top, bottom].concat());
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn a_posed_skinned_draw_reads_the_surface_of_its_material() {
    let (block, vectors) = full_block();
    let shader = format!("{BASE}{INSTANCED}{SKINNED}");
    let files = [
        ("quad.mesh.json", square(true)),
        ("skin.wgsl", shader.into_bytes()),
        ("coated.material.json", surface_material("skin.wgsl", Some(&block), false)),
        ("bare.material.json", surface_material("skin.wgsl", None, false)),
    ];
    let Some(mut rig) =
        Rig::new("surface-skin", &["quad.mesh.json"], &["coated.material.json", "bare.material.json"], &files)
    else {
        return;
    };
    let draws: Vec<Draw> = ALL.iter().map(|&which| draw(0, if which < 5 { 0 } else { 1 }, which)).collect();
    let mut posed = frame(draws);
    posed.poses =
        ALL.iter().map(|&which| Pose { draw: which as u32, clip: 0, time: 0.0, weight: 1.0, inset: 0 }).collect();
    let picture = rig.show(&posed);
    // Posed draws of a skinned mesh draw alone, each with the skinned pipeline.
    assert_eq!(rig.renderer.stats().draw_calls, 10);
    let top: Vec<(usize, &[[f32; 4]; 5])> = (0..5).map(|which| (which, &vectors)).collect();
    let bottom: Vec<(usize, &[[f32; 4]; 5])> = (5..10).map(|which| (which, &DEFAULTS)).collect();
    let wrong = wrong_swatches(&picture, &[top, bottom].concat());
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn materials_of_different_surfaces_never_share_a_batch() {
    let (block, vectors) = full_block();
    let other = json!({"clearcoat": 0.25, "ior": 1.25, "specularColor": [0.5, 0.25, 0.75]});
    let other_vectors = [
        [0.25, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.25, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.25, 0.75, 1.0],
    ];
    let shader = format!("{BASE}{INSTANCED}");
    // With instance parameters, materials of one render state share a
    // batch: the first two differ in nothing else, the third in its surface.
    let files = [
        ("quad.mesh.json", square(false)),
        ("batch.wgsl", shader.into_bytes()),
        ("a.material.json", surface_material("batch.wgsl", Some(&block), true)),
        ("same.material.json", surface_material("batch.wgsl", Some(&block), true)),
        ("other.material.json", surface_material("batch.wgsl", Some(&other), true)),
    ];
    let Some(mut rig) = Rig::new(
        "surface-state",
        &["quad.mesh.json"],
        &["a.material.json", "same.material.json", "other.material.json"],
        &files,
    ) else {
        return;
    };
    // Swatches 0 and 1 from `a` and `same`, 5 and 6 from `other`, 2 and 3 from `a` again.
    let draws = vec![draw(0, 0, 0), draw(0, 1, 1), draw(0, 2, 5), draw(0, 2, 6), draw(0, 0, 2), draw(0, 1, 3)];
    let picture = rig.show(&frame(draws));
    let stats = rig.renderer.stats();
    assert_eq!(
        (stats.instanced_draw_calls, stats.instances),
        (2, 6),
        "the same surface shares a batch, a different one does not: {stats:?}"
    );
    let wrong = wrong_swatches(
        &picture,
        &[(0, &vectors), (1, &vectors), (2, &vectors), (3, &vectors), (5, &other_vectors), (6, &other_vectors)],
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn a_reloaded_surface_replaces_the_one_a_material_drew_with() {
    let (block, vectors) = full_block();
    let shader = format!("{BASE}{INSTANCED}");
    let files = [
        ("quad.mesh.json", square(false)),
        ("batch.wgsl", shader.into_bytes()),
        ("paint.material.json", surface_material("batch.wgsl", None, false)),
    ];
    let Some(mut rig) = Rig::new("surface-reload", &["quad.mesh.json"], &["paint.material.json"], &files) else {
        return;
    };
    let draws = || -> Vec<Draw> { ALL.iter().map(|&which| draw(0, 0, which)).collect() };
    let picture = rig.show(&frame(draws()));
    let wrong = wrong_swatches(&picture, &ALL.map(|which| (which, &DEFAULTS)));
    assert!(wrong.is_empty(), "before: {wrong:#?}");
    // The block arrives.
    write(&rig.root, "paint.material.json", surface_material("batch.wgsl", Some(&block), false));
    rig.renderer.reload(&["paint.material.json".to_owned()]);
    let frame_now = frame(draws());
    pump(&mut rig.renderer, &frame_now, &rig.catalog, &mut rig.target, |events| {
        ready(events, "paint.material.json", 2)
    });
    let picture = pixels(&rig.device, &rig.queue, &rig.target.0);
    let wrong = wrong_swatches(&picture, &ALL.map(|which| (which, &vectors)));
    assert!(wrong.is_empty(), "after the block came: {wrong:#?}");
    // And goes again.
    write(&rig.root, "paint.material.json", surface_material("batch.wgsl", None, false));
    rig.renderer.reload(&["paint.material.json".to_owned()]);
    pump(&mut rig.renderer, &frame_now, &rig.catalog, &mut rig.target, |events| {
        ready(events, "paint.material.json", 3)
    });
    let picture = pixels(&rig.device, &rig.queue, &rig.target.0);
    let wrong = wrong_swatches(&picture, &ALL.map(|which| (which, &DEFAULTS)));
    assert!(wrong.is_empty(), "after the block went: {wrong:#?}");
}

// A half-transparent green quad over an opaque red one: the opaque pipeline
// writes the green as it is, the translucent one blends it.
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

const BEHIND: &str = r#"
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
fn a_transmissive_material_blends_over_what_is_behind_it_even_with_an_opaque_alpha() {
    let behind = serde_json::to_vec(&json!({
        "positions": [-0.5, -0.5, 0.7, 0.5, -0.5, 0.7, 0.5, 0.5, 0.7, -0.5, 0.5, 0.7],
        "normals": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        "uvs": [0, 0, 1, 0, 1, 1, 0, 1],
        "indices": [0, 1, 2, 0, 2, 3],
    }))
    .unwrap();
    let glass = |block: serde_json::Value| {
        json!({"shader": "glass.wgsl", "color": [1, 1, 1, 1], "lighting": "unlit", "surface": block})
            .to_string()
            .into_bytes()
    };
    let files = [
        ("behind.mesh.json", behind),
        ("quad.mesh.json", square(false)),
        ("glass.wgsl", GLASS.as_bytes().to_vec()),
        ("behind.wgsl", BEHIND.as_bytes().to_vec()),
        ("red.material.json", br#"{"shader":"behind.wgsl","color":[1,0,0,1],"lighting":"unlit"}"#.to_vec()),
        ("clear.material.json", glass(json!({"transmission": 0.0, "thickness": 1.0}))),
        ("glass.material.json", glass(json!({"transmission": 0.5}))),
    ];
    let Some(mut rig) = Rig::new(
        "surface-glass",
        &["behind.mesh.json", "quad.mesh.json"],
        &["red.material.json", "clear.material.json", "glass.material.json"],
        &files,
    ) else {
        return;
    };
    let in_front = |material: u32| {
        vec![
            Draw { mesh: 0, material: 0, transform: IDENTITY, parameters: None, extra: None },
            Draw { mesh: 1, material, transform: IDENTITY, parameters: None, extra: None },
        ]
    };
    // Without transmission the quad is opaque, so its green replaces the red.
    let picture = rig.show(&frame(in_front(1)));
    let green = at(&picture, 0.0, 0.0);
    assert!(near(green, [0, shown(1.0), 0]), "an opaque quad writes its colour as it is: {green:?}");
    // With it, the same alpha 0.5 blends: half green and half red, in light.
    let picture = rig.show(&frame(in_front(2)));
    let blended = at(&picture, 0.0, 0.0);
    let half = shown(0.5);
    assert!(near(blended, [half, half, 0]), "a transmissive quad is blended: {blended:?}, wanted {half}");
    // Outside the quad the red alone shows.
    assert!(near(at(&picture, 0.4, 0.4), [shown(1.0), 0, 0]));
}

#[test]
fn an_inset_reads_the_surface_of_its_material_too() {
    let (block, vectors) = full_block();
    let files = [
        ("quad.mesh.json", square(false)),
        ("plain.wgsl", BASE.as_bytes().to_vec()),
        ("coated.material.json", surface_material("plain.wgsl", Some(&block), false)),
    ];
    let Some(mut rig) = Rig::new("surface-inset", &["quad.mesh.json"], &["coated.material.json"], &files) else {
        return;
    };
    let mut inset_frame = frame(Vec::new());
    inset_frame.insets.push(Inset {
        x: 0.0,
        y: 0.0,
        width: SIZE as f32,
        height: SIZE as f32,
        view: IDENTITY,
        projection: IDENTITY,
        draws: ALL.iter().map(|&which| draw(0, 0, which)).collect(),
        light_direction: inset_frame.light_direction,
        light_color: inset_frame.light_color,
        ..Default::default()
    });
    let picture = rig.show(&inset_frame);
    let wrong = wrong_swatches(&picture, &ALL.map(|which| (which, &vectors)));
    assert!(wrong.is_empty(), "{wrong:#?}");
}
