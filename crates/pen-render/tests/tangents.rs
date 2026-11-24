//! A mesh's tangents reach the shader at `@location(11)`, in every kind of
//! mesh pipeline (alone, instanced, skinned, persistent, and the sun's shadow
//! maps, which compile the same entry points); a mesh without tangents, and a
//! brush solid, read zeros whatever their size; and a hot reload swaps the
//! stream. Each test shader writes a vertex's tangent (xyz in one swatch, the
//! handedness w in the other) as the colour of a quad, which the pixels are
//! compared with. Skips when no adapter is available.
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

// Five quads in one mesh, each with a tangent of its own: swatch k.
const TANGENTS: [[f32; 4]; 5] =
    [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, -1.0], [0.6, 0.8, 0.0, 1.0], [0.0, 0.0, 1.0, -1.0], [-0.28, 0.0, 0.96, 1.0]];

// What a quad shows of its tangent: xyz as colours in 0.1 to 0.9 (a draw whose
// first parameter is 0), or w as grey (1).
const BASE: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) tangent: vec4f,
  @location(1) @interpolate(flat) mode: f32,
};
@vertex fn vertex(@location(0) position: vec3f, @location(11) tangent: vec4f) -> Corner {
  return Corner(scene.projection * scene.view * draw.model * vec4f(position, 1.0), tangent, draw.parameters.x);
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let shown = select(corner.tangent.xyz, vec3f(corner.tangent.w), corner.mode > 0.5);
  return vec4f(shown * 0.4 + 0.5, 1.0);
}
"#;

const INSTANCED: &str = r#"
@vertex fn vertex_instanced(
  @location(0) position: vec3f,
  @location(3) m0: vec4f, @location(4) m1: vec4f, @location(5) m2: vec4f, @location(6) m3: vec4f,
  @location(7) parameters: vec4f,
  @location(11) tangent: vec4f,
) -> Corner {
  let world = scene.projection * scene.view * mat4x4f(m0, m1, m2, m3) * vec4f(position, 1.0);
  return Corner(world, tangent, parameters.x);
}
"#;

const SKINNED: &str = r#"
@group(3) @binding(0) var<storage, read> joints: array<mat4x4f>;
struct Skinned {
  @location(0) position: vec3f,
  @location(9) joints: vec4u,
  @location(10) weights: vec4f,
  @location(11) tangent: vec4f,
};
@vertex fn vertex_skinned(input: Skinned) -> Corner {
  let skin = joints[input.joints.x] * input.weights.x + joints[input.joints.y] * input.weights.y
    + joints[input.joints.z] * input.weights.z + joints[input.joints.w] * input.weights.w;
  let world = scene.projection * scene.view * draw.model * skin * vec4f(input.position, 1.0);
  return Corner(world, input.tangent, draw.parameters.x);
}
"#;

const PERSISTENT: &str = r#"
const bounds_margin = 0.0;
@vertex fn vertex_persistent(
  @location(0) position: vec3f, @location(3) offset: vec4f, @location(11) tangent: vec4f,
) -> Corner {
  let world = scene.projection * scene.view * draw.model * vec4f(position + offset.xyz, 1.0);
  return Corner(world, tangent, draw.parameters.x);
}
"#;

/// Where quad `k` sits across the picture.
fn x_of(k: usize) -> f32 {
    -0.8 + 0.4 * k as f32
}

/// Five quads of a mesh as JSON: `tangents` give each its four vertices' tangent; and,
/// as asked, a skin, or persistent instances (one record, no offset).
fn quads(tangents: Option<&[[f32; 4]; 5]>, skinned: bool, persistent: bool) -> Vec<u8> {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    let mut tangent_stream = Vec::new();
    for k in 0..5 {
        let x = x_of(k);
        for (dx, dy) in [(-0.12, -0.12), (0.12, -0.12), (0.12, 0.12), (-0.12, 0.12)] {
            positions.extend([x + dx, dy, 0.5]);
            if let Some(tangents) = tangents {
                tangent_stream.extend(tangents[k]);
            }
        }
        let first = (k * 4) as u32;
        indices.extend([first, first + 1, first + 2, first, first + 2, first + 3]);
    }
    let (normals, uvs) = ([0, 0, 1].repeat(20), [0, 0].repeat(20));
    let mut mesh = json!({"positions": positions, "normals": normals, "uvs": uvs, "indices": indices});
    if tangents.is_some() {
        mesh["tangents"] = json!(tangent_stream);
    }
    if skinned {
        let (joints, weights) = ([0, 0, 0, 0].repeat(20), [1, 0, 0, 0].repeat(20));
        mesh["joints"] = json!(joints);
        mesh["weights"] = json!(weights);
        mesh["skeleton"] = json!([{
            "name": "root", "translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1],
        }]);
        mesh["skin"] = json!([{
            "node": 0, "inverseBind": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        }]);
        mesh["clips"] = json!([{"name": "still", "duration": 1, "channels": []}]);
    }
    if persistent {
        mesh["instances"] = json!({"data": vec![0.0; 20], "min": [-1, -0.2, 0.5], "max": [1, 0.2, 0.5]});
    }
    serde_json::to_vec(&mesh).unwrap()
}

fn material(shader: &str) -> Vec<u8> {
    json!({"shader": shader, "color": [1, 1, 1, 1], "lighting": "lit"}).to_string().into_bytes()
}

/// The draw of mesh `mesh` showing xyz (mode 0) or w (mode 1) at height `y`.
fn row(mesh: u32, mode: f32, y: f32) -> Draw {
    Draw { mesh, material: 0, transform: moved(0.0, y), parameters: Some([mode, 0.0, 0.0, 0.0]), extra: None }
}

/// The rows of a frame: the mesh with its tangents showing xyz then w, and
/// the mesh without any showing xyz.
fn rows(with: u32, without: u32) -> Vec<Draw> {
    vec![row(with, 0.0, 0.66), row(with, 1.0, 0.0), row(without, 0.0, -0.66)]
}

/// What the three rows should show of these tangents: xyz and w, then zeros
/// (grey 0.5, the colour of nothing).
fn wrong_rows(picture: &[[u8; 4]], tangents: &[[f32; 4]; 5]) -> Vec<String> {
    let mut wrong = Vec::new();
    for (k, tangent) in tangents.iter().enumerate() {
        let x = x_of(k);
        let xyz = [tangent[0], tangent[1], tangent[2]].map(|value| shown(value * 0.4 + 0.5));
        let w = [shown(tangent[3] * 0.4 + 0.5); 3];
        for (name, y, want) in [("xyz", 0.66, xyz), ("w", 0.0, w), ("a mesh without tangents", -0.66, [shown(0.5); 3])]
        {
            let got = at(picture, x, y);
            if !near(got, want) {
                wrong.push(format!("quad {k}, {name}: drew {got:?}, wanted {want:?}"));
            }
        }
    }
    wrong
}

/// A frame of `draws`, with the sun's shadow map on, which every opaque lit
/// material's shadow pipeline draws into.
fn shadowed(draws: Vec<Draw>) -> RenderFrame {
    let mut frame = frame(draws);
    frame.shadow = Shadow { view: IDENTITY, texels: 64, ..Shadow::default() };
    frame.cascades = vec![Cascade { projection: IDENTITY, texel: 1.0 / 32.0, far: 100.0 }];
    frame
}

fn files(shader: String) -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("with.mesh.json", quads(Some(&TANGENTS), false, false)),
        ("without.mesh.json", quads(None, false, false)),
        ("tangent.wgsl", shader.into_bytes()),
        ("paint.material.json", material("tangent.wgsl")),
    ]
}

const MESHES: [&str; 2] = ["with.mesh.json", "without.mesh.json"];

#[test]
fn a_meshs_tangents_reach_the_plain_pipeline_and_a_mesh_without_reads_zeros() {
    let Some(mut rig) = Rig::new("tangent-plain", &MESHES, &["paint.material.json"], &files(BASE.to_owned())) else {
        return;
    };
    let picture = rig.show(&shadowed(rows(0, 1)));
    let wrong = wrong_rows(&picture, &TANGENTS);
    // The zero stream is every vertex of the largest mesh; both meshes have twenty.
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn instanced_draws_read_the_tangents_beside_their_instance_data() {
    let Some(mut rig) =
        Rig::new("tangent-batch", &MESHES, &["paint.material.json"], &files(format!("{BASE}{INSTANCED}")))
    else {
        return;
    };
    let picture = rig.show(&shadowed(rows(0, 1)));
    // The mesh with tangents has two rows in one batch, the one without a batch of its own.
    let stats = rig.renderer.stats();
    assert_eq!((stats.instanced_draw_calls, stats.instances), (2, 3), "{stats:?}");
    let wrong = wrong_rows(&picture, &TANGENTS);
    assert!(wrong.is_empty(), "{wrong:#?}");
    // Several instances of one mesh share a batch, each showing the mesh's tangent.
    let more = vec![row(0, 0.0, 0.66), row(0, 0.0, 0.0), row(1, 0.0, -0.66), row(1, 0.0, -0.33)];
    let picture = rig.show(&shadowed(more));
    assert_eq!(rig.renderer.stats().instanced_draw_calls, 2);
    for (k, tangent) in TANGENTS.iter().enumerate() {
        let want = [tangent[0], tangent[1], tangent[2]].map(|value| shown(value * 0.4 + 0.5));
        assert!(near(at(&picture, x_of(k), 0.0), want), "quad {k}");
        assert!(near(at(&picture, x_of(k), -0.33), [shown(0.5); 3]), "quad {k} without");
    }
}

#[test]
fn a_posed_skinned_mesh_reads_its_tangents_beside_its_joints() {
    let shader = format!("{BASE}{INSTANCED}{SKINNED}");
    let files = [
        ("with.mesh.json", quads(Some(&TANGENTS), true, false)),
        ("without.mesh.json", quads(None, true, false)),
        ("tangent.wgsl", shader.into_bytes()),
        ("paint.material.json", material("tangent.wgsl")),
    ];
    let Some(mut rig) = Rig::new("tangent-skin", &MESHES, &["paint.material.json"], &files) else { return };
    let mut posed = shadowed(rows(0, 1));
    posed.poses = (0..3).map(|draw| Pose { draw, clip: 0, time: 0.0, weight: 1.0 }).collect();
    let picture = rig.show(&posed);
    assert_eq!(rig.renderer.stats().instanced_draw_calls, 0, "posed draws are drawn alone");
    let wrong = wrong_rows(&picture, &TANGENTS);
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn persistent_instances_read_the_tangents_of_their_template() {
    let shader = format!("{BASE}{INSTANCED}{PERSISTENT}");
    let files = [
        ("with.mesh.json", quads(Some(&TANGENTS), false, true)),
        ("without.mesh.json", quads(None, false, true)),
        ("tangent.wgsl", shader.into_bytes()),
        ("paint.material.json", material("tangent.wgsl")),
    ];
    let Some(mut rig) = Rig::new("tangent-persistent", &MESHES, &["paint.material.json"], &files) else { return };
    let picture = rig.show(&shadowed(rows(0, 1)));
    let wrong = wrong_rows(&picture, &TANGENTS);
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn a_brush_solid_reads_zero_tangents() {
    let files = [
        ("with.mesh.json", quads(Some(&TANGENTS), false, false)),
        ("tangent.wgsl", BASE.as_bytes().to_vec()),
        ("paint.material.json", material("tangent.wgsl")),
    ];
    let Some(mut rig) = Rig::new("tangent-solid", &["with.mesh.json"], &["paint.material.json"], &files) else {
        return;
    };
    // A box in front of the camera: its faces' vertices have no tangents.
    let faces = [
        ([1.0, 0.0, 0.0], 0.3),
        ([-1.0, 0.0, 0.0], 0.3),
        ([0.0, 1.0, 0.0], 0.3),
        ([0.0, -1.0, 0.0], 0.3),
        ([0.0, 0.0, 1.0], 0.8),
        ([0.0, 0.0, -1.0], -0.2),
    ]
    .map(|([x, y, z], distance)| SolidFace { normal: Vector { x, y, z }, distance, material: 0 });
    let mut solid = frame(Vec::new());
    solid.solids = vec![Solid { faces: faces.to_vec() }];
    let picture = rig.show(&solid);
    let got = at(&picture, 0.0, 0.0);
    // Mode 0 of a draw's first parameter, which a solid has none of: zeros
    // read as the colour of nothing.
    assert!(near(got, [shown(0.5); 3]), "the solid drew {got:?}");
    assert!(near(at(&picture, 0.0, 0.9), [0, 0, 0]), "and nothing else");
}

#[test]
fn a_mesh_larger_than_any_before_gets_zero_tangents_all_the_same() {
    // The first mesh has twenty vertices; the second has far more, its one visible
    // quad the last: the zeros reach it, however much they had to grow.
    let quads_in_it = 6000usize;
    let (mut positions, mut indices) = (Vec::<f32>::new(), Vec::<u32>::new());
    for k in 0..quads_in_it {
        let (x, y) = if k + 1 == quads_in_it { (0.0, 0.0) } else { (5.0, 5.0) };
        for (dx, dy) in [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)] {
            positions.extend([x + dx, y + dy, 0.5]);
        }
        let first = (k * 4) as u32;
        indices.extend([first, first + 1, first + 2, first, first + 2, first + 3]);
    }
    let (normals, uvs) = ([0, 0, 1].repeat(quads_in_it * 4), vec![0.0; quads_in_it * 8]);
    let big = json!({"positions": positions, "normals": normals, "uvs": uvs, "indices": indices});
    let files = [
        ("with.mesh.json", quads(Some(&TANGENTS), false, false)),
        ("big.mesh.json", serde_json::to_vec(&big).unwrap()),
        ("tangent.wgsl", BASE.as_bytes().to_vec()),
        ("paint.material.json", material("tangent.wgsl")),
    ];
    let Some(mut rig) =
        Rig::new("tangent-grow", &["with.mesh.json", "big.mesh.json"], &["paint.material.json"], &files)
    else {
        return;
    };
    // The small one alone first, then the big one's arrival (a frame asking for it).
    let picture = rig.show(&frame(vec![row(0, 0.0, 0.66)]));
    assert!(near(at(&picture, x_of(0), 0.66), [shown(0.9), shown(0.5), shown(0.5)]));
    let picture = rig.show(&frame(vec![row(0, 0.0, 0.66), row(1, 0.0, -0.4)]));
    assert!(near(at(&picture, 0.0, -0.4), [shown(0.5); 3]), "the last quad of a large mesh without tangents");
    // Back to a small one: the stream stays as large as the largest.
    let picture = rig.show(&frame(vec![row(0, 0.0, 0.0)]));
    assert!(near(at(&picture, x_of(1), 0.0), [shown(0.5), shown(0.9), shown(0.5)]));
}

#[test]
fn a_reloaded_mesh_swaps_its_tangents_for_the_new_ones_or_for_none() {
    let Some(mut rig) =
        Rig::new("tangent-reload", &["with.mesh.json"], &["paint.material.json"], &files(BASE.to_owned()))
    else {
        return;
    };
    let draws = || vec![row(0, 0.0, 0.66), row(0, 1.0, 0.0)];
    let picture = rig.show(&frame(draws()));
    let before = at(&picture, x_of(0), 0.66);
    assert!(near(before, [shown(0.9), shown(0.5), shown(0.5)]), "{before:?}");
    // Other tangents.
    let mut swapped = TANGENTS;
    swapped[0] = [0.0, 0.0, -1.0, 1.0];
    write(&rig.root, "with.mesh.json", quads(Some(&swapped), false, false));
    rig.renderer.reload(&["with.mesh.json".to_owned()]);
    let frame_now = frame(draws());
    pump(&mut rig.renderer, &frame_now, &rig.catalog, &mut rig.target, |events| ready(events, "with.mesh.json", 2));
    let picture = pixels(&rig.device, &rig.queue, &rig.target.0);
    let after = at(&picture, x_of(0), 0.66);
    assert!(near(after, [shown(0.5), shown(0.5), shown(0.1)]), "{after:?}");
    // And none.
    write(&rig.root, "with.mesh.json", quads(None, false, false));
    rig.renderer.reload(&["with.mesh.json".to_owned()]);
    pump(&mut rig.renderer, &frame_now, &rig.catalog, &mut rig.target, |events| ready(events, "with.mesh.json", 3));
    let picture = pixels(&rig.device, &rig.queue, &rig.target.0);
    for k in 0..5 {
        assert!(near(at(&picture, x_of(k), 0.66), [shown(0.5); 3]), "quad {k} has no tangents now");
    }
}
