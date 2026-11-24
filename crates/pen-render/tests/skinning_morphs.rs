//! GPU witnesses for explicit Ink-supplied bone matrices (`bone_matrices`),
//! GPU compute morph targets (`morph_weights`), combined morph + skinning,
//! translucent skinned draws (`SKINNED_TRANSLUCENT`), and skinned/morphed
//! inset draws (`INSET_SKINNED` / `INSET_SKINNED_BLENDED`). Skips without GPU.
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

const SHADER: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@group(3) @binding(0) var<storage, read> joints: array<mat4x4f>;

struct Out {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
};

struct Plain {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
};

struct Skinned {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(9) joints: vec4u,
  @location(10) weights: vec4f,
};

@vertex fn vertex(input: Plain) -> Out {
  return Out(scene.projection * scene.view * draw.model * vec4f(input.position, 1.0), input.normal);
}

@vertex fn vertex_skinned(input: Skinned) -> Out {
  let skin = joints[input.joints.x] * input.weights.x
    + joints[input.joints.y] * input.weights.y
    + joints[input.joints.z] * input.weights.z
    + joints[input.joints.w] * input.weights.w;
  return Out(scene.projection * scene.view * draw.model * skin * vec4f(input.position, 1.0), input.normal);
}

@fragment fn fragment(input: Out) -> @location(0) vec4f {
  let col = select(draw.color.rgb, input.normal * 0.5 + 0.5, draw.parameters.x > 0.5);
  return vec4f(col, draw.color.a);
}
"#;

/// A small centered quad (`[-0.15, 0.15]^2` at `z = 0.5`) with 2 skin joints
/// and 2 morph targets:
/// - Target 0: shifts +X by 0.5, normal unchanged.
/// - Target 1: shifts +Y by 0.5, adds (+1, 0, 0) to normal (so at weight 1.0
///   the normalized normal becomes `(0.7071, 0, 0.7071)`).
fn skinned_morphed_quad() -> Vec<u8> {
    let positions = vec![-0.15, -0.15, 0.5, 0.15, -0.15, 0.5, 0.15, 0.15, 0.5, -0.15, 0.15, 0.5];
    let normals = [0.0f32, 0.0, 1.0].repeat(4);
    let uvs = [0.0f32, 0.0].repeat(4);
    let indices = vec![0u32, 1, 2, 0, 2, 3];
    let joints = [0u16, 0, 0, 0].repeat(4);
    let weights = [1.0f32, 0.0, 0.0, 0.0].repeat(4);
    let t0_pos = [0.5f32, 0.0, 0.0].repeat(4);
    let t1_pos = [0.0f32, 0.5, 0.0].repeat(4);
    let t1_norm = [1.0f32, 0.0, 0.0].repeat(4);
    let mesh = json!({
        "positions": positions,
        "normals": normals,
        "uvs": uvs,
        "indices": indices,
        "joints": joints,
        "weights": weights,
        "skeleton": [{
            "name": "root",
            "translation": [0, 0, 0],
            "rotation": [0, 0, 0, 1],
            "scale": [1, 1, 1],
        }],
        "skin": [{
            "node": 0,
            "inverseBind": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        }],
        "clips": [{
            "name": "left",
            "duration": 1.0,
            "channels": [{
                "node": 0,
                "path": "translation",
                "interpolation": "step",
                "times": [0.0],
                "values": [-0.6, 0.0, 0.0],
            }],
        }],
        "morphs": [
            { "positions": t0_pos },
            { "positions": t1_pos, "normals": t1_norm },
        ],
    });
    serde_json::to_vec(&mesh).unwrap()
}

fn bg_quad() -> Vec<u8> {
    let positions = vec![-1.0, -1.0, 0.8, 1.0, -1.0, 0.8, 1.0, 1.0, 0.8, -1.0, 1.0, 0.8];
    let normals = [0.0f32, 0.0, 1.0].repeat(4);
    let uvs = [0.0f32, 0.0].repeat(4);
    let indices = vec![0u32, 1, 2, 0, 2, 3];
    serde_json::to_vec(&json!({
        "positions": positions, "normals": normals, "uvs": uvs, "indices": indices,
    }))
    .unwrap()
}

fn mat(color: [f32; 4], blend: bool) -> Vec<u8> {
    json!({
        "shader": "anim.wgsl",
        "color": color,
        "lighting": "unlit",
        "blend": blend,
    })
    .to_string()
    .into_bytes()
}

fn test_files() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("quad.mesh.json", skinned_morphed_quad()),
        ("bg.mesh.json", bg_quad()),
        ("anim.wgsl", SHADER.as_bytes().to_vec()),
        ("white.material.json", mat([1.0, 1.0, 1.0, 1.0], false)),
        ("red.material.json", mat([1.0, 0.0, 0.0, 1.0], false)),
        ("blue_half.material.json", mat([0.0, 0.0, 1.0, 0.5], true)),
    ]
}

const MESHES: [&str; 2] = ["quad.mesh.json", "bg.mesh.json"];
const MATERIALS: [&str; 3] = ["white.material.json", "red.material.json", "blue_half.material.json"];

#[test]
fn explicit_bone_matrices_override_clips_and_drive_skinning() {
    let Some(mut rig) = Rig::new("bone-matrices", &MESHES, &MATERIALS, &test_files()) else {
        return;
    };
    let mut f = frame(vec![Draw { mesh: 0, material: 0, transform: IDENTITY, parameters: None, extra: None }]);
    // Clip 0 moves joint 0 to x = -0.6; explicit BoneMatrix overrides it to x = +0.5.
    f.poses = vec![Pose { draw: 0, clip: 0, time: 0.0, weight: 1.0, inset: 0 }];
    f.bone_matrices = vec![BoneMatrix { inset: 0, draw: 0, joint: 0, matrix: moved(0.5, 0.0) }];
    let pic = rig.show(&f);
    let white = [shown(1.0); 3];
    assert!(near(at(&pic, 0.5, 0.0), white), "expected white {white:?} at +0.5, got {:?}", at(&pic, 0.5, 0.0));
    assert!(near(at(&pic, -0.6, 0.0), [0, 0, 0]), "clip position should be overridden, got {:?}", at(&pic, -0.6, 0.0));
    assert!(near(at(&pic, 0.0, 0.0), [0, 0, 0]), "origin should be empty, got {:?}", at(&pic, 0.0, 0.0));
}

#[test]
fn gpu_morph_targets_deform_positions_normals_and_compose_with_skinning() {
    let Some(mut rig) = Rig::new("morph-and-skin", &MESHES, &MATERIALS, &test_files()) else {
        return;
    };
    // Target 0 shifts +0.5 X, target 1 shifts +0.5 Y and adds (+1,0,0) to normal.
    // BoneMatrix adds another -0.25 X AFTER morphing -> final quad center is (+0.25, +0.5).
    let mut f = frame(vec![Draw {
        mesh: 0,
        material: 0,
        transform: IDENTITY,
        parameters: Some([1.0, 0.0, 0.0, 0.0]), // show normal as color
        extra: None,
    }]);
    f.morph_weights = vec![
        MorphWeight { inset: 0, draw: 0, target: 0, weight: 1.0 },
        MorphWeight { inset: 0, draw: 0, target: 1, weight: 1.0 },
    ];
    f.bone_matrices = vec![BoneMatrix { inset: 0, draw: 0, joint: 0, matrix: moved(-0.25, 0.0) }];
    let pic = rig.show(&f);
    let n = std::f32::consts::FRAC_1_SQRT_2;
    let want = [shown(n * 0.5 + 0.5), shown(0.5), shown(n * 0.5 + 0.5)];
    let got = at(&pic, 0.25, 0.5);
    assert!(near(got, want), "expected morphed+skinned normal color {want:?} at (0.25, 0.5), got {got:?}");
    assert!(near(at(&pic, 0.0, 0.0), [0, 0, 0]), "origin should be empty, got {:?}", at(&pic, 0.0, 0.0));
}

#[test]
fn skinned_translucent_and_inset_draws_render_posed() {
    let Some(mut rig) = Rig::new("skinned-translucent-inset", &MESHES, &MATERIALS, &test_files()) else {
        return;
    };
    // Fullscreen red background (draw 0) + translucent blue skinned quad (draw 1)
    // moved by BoneMatrix to x = +0.5 -> blended purple pixel at (+0.5, 0.0).
    // Plus an opaque skinned quad inside inset 0 covering the top-left quadrant,
    // posed via inset: 1, draw: 0 to shift +0.5 X inside the inset.
    let mut f = frame(vec![
        Draw { mesh: 1, material: 1, transform: IDENTITY, parameters: None, extra: None },
        Draw { mesh: 0, material: 2, transform: IDENTITY, parameters: None, extra: None },
    ]);
    f.insets = vec![Inset {
        x: 0.0,
        y: 0.0,
        width: 64.0,
        height: 64.0,
        pixel_size: 0.0,
        after_sprites: 0,
        view: IDENTITY,
        projection: IDENTITY,
        ambient: Vector { x: 1.0, y: 1.0, z: 1.0 },
        light_direction: Vector { x: 0.0, y: 1.0, z: 0.0 },
        light_color: Vector { x: 1.0, y: 1.0, z: 1.0 },
        draws: vec![Draw { mesh: 0, material: 0, transform: IDENTITY, parameters: None, extra: None }],
    }];
    f.bone_matrices = vec![
        BoneMatrix { inset: 0, draw: 1, joint: 0, matrix: moved(0.5, 0.0) },
        BoneMatrix { inset: 1, draw: 0, joint: 0, matrix: moved(0.5, 0.0) },
    ];
    let pic = rig.show(&f);
    // At (+0.5, 0.0): 50% blue over 100% red -> linear (0.5, 0, 0.5).
    let blended = at(&pic, 0.5, 0.0);
    let want_blend = [shown(0.5), 0, shown(0.5)];
    assert!(near(blended, want_blend), "skinned translucent expected {want_blend:?}, got {blended:?}");
    // Pure red at (0.0, -0.5) where translucent quad moved away from.
    let red = [shown(1.0), 0, 0];
    let white = [shown(1.0); 3];
    assert!(near(at(&pic, 0.0, -0.5), red), "background red expected {red:?}, got {:?}", at(&pic, 0.0, -0.5));
    // Inside inset (top-left 64x64 of 128x128): inset NDC x=+0.5, y=0.0 maps to
    // pixel x = 64 * 0.75 = 48 (screen NDC -0.25), pixel y = 32 (screen NDC +0.5).
    let inset_hit = at(&pic, -0.25, 0.5);
    assert!(near(inset_hit, white), "skinned inset expected white {white:?} at (-0.25, 0.5), got {inset_hit:?}");
    // Inset center (-0.5, 0.5) should still show red background since quad moved right.
    assert!(
        near(at(&pic, -0.5, 0.5), red),
        "inset origin should be unoccluded red {red:?}, got {:?}",
        at(&pic, -0.5, 0.5)
    );
}
