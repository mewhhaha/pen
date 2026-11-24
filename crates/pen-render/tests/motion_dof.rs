//! Headless GPU pixel witnesses for Checkpoint 6:
//! 1. Moving-object TAA reprojection via per-object screen-space velocity vectors.
//! 2. Directional velocity motion blur (per-object and automatic skeletal joint motion).
//! 3. Bokeh depth of field (in-focus sharpness vs out-of-focus blur + rack focus).
#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::cast_precision_loss, reason = "test coordinates")]
mod common;

use std::path::{Path, PathBuf};

use common::picture::shown;
use common::{FORMAT, IDENTITY, Offscreen, device, frame, pixels, pump, ready};
use pen_protocol::*;
use pen_render::{Renderer, RendererOptions};
use serde_json::json;

struct TempDir(PathBuf);
impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("pen-render-motion-dof-{name}-{}", std::process::id()));
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

const PLAIN_AND_SKINNED_SHADER: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@group(1) @binding(1) var base: texture_2d<f32>;
@group(1) @binding(2) var base_sampler: sampler;
@group(3) @binding(0) var<storage, read> joints: array<mat4x4f>;

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

@vertex fn vertex_skinned(
  @location(0) position: vec3f,
  @location(1) _normal: vec3f,
  @location(2) uv: vec2f,
  @location(9) j: vec4u,
  @location(10) w: vec4f,
) -> Out {
  let skin = joints[j.x] * w.x + joints[j.y] * w.y + joints[j.z] * w.z + joints[j.w] * w.w;
  var out: Out;
  out.position = scene.projection * scene.view * draw.model * (skin * vec4f(position, 1.0));
  out.uv = uv;
  return out;
}

@fragment fn fragment(input: Out) -> @location(0) vec4f {
  let tex = textureSample(base, base_sampler, input.uv).rgb;
  return vec4f(draw.color.rgb * tex, draw.color.a);
}
"#;

fn translate(x: f32, y: f32, z: f32) -> Matrix {
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0]
}

fn write_quad_mesh(dir: &Path, name: &str, half: f32, skinned: bool) {
    let mut mesh = json!({
        "positions": [
            -half, -half, 0.0,
             half, -half, 0.0,
             half,  half, 0.0,
            -half,  half, 0.0
        ],
        "normals": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        "uvs": [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "indices": [0, 1, 2, 0, 2, 3]
    });
    if skinned {
        mesh["joints"] = json!([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        mesh["weights"] = json!([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        mesh["skeleton"] = json!([{
            "name": "root",
            "translation": [0, 0, 0],
            "rotation": [0, 0, 0, 1],
            "scale": [1, 1, 1],
        }]);
        mesh["skin"] = json!([{
            "node": 0,
            "inverseBind": IDENTITY,
        }]);
    }
    std::fs::write(dir.join(name), mesh.to_string()).unwrap();
}

fn write_stripes_png(path: &Path, size: u32, period: u32) {
    let mut img = image::RgbaImage::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let white = ((x / period) + (y / period)).is_multiple_of(2);
            let v = if white { 255 } else { 0 };
            img.put_pixel(x, y, image::Rgba([v, v, v, 255]));
        }
    }
    img.save(path).unwrap();
}

#[test]
fn moving_object_taa_tracks_velocity_without_trailing_ghosts() {
    let Some((device, queue)) = device() else { return };
    let temp = TempDir::new("taa");
    let dir = temp.path();
    write_quad_mesh(dir, "quad.mesh.json", 0.2, false);
    std::fs::write(dir.join("flat.wgsl"), PLAIN_AND_SKINNED_SHADER).unwrap();
    std::fs::write(
        dir.join("red.material.json"),
        json!({"shader": "flat.wgsl", "color": [1.0, 0.0, 0.0, 1.0], "lighting": "unlit"}).to_string(),
    )
    .unwrap();
    let catalog =
        Catalog { meshes: vec!["quad.mesh.json".into()], materials: vec!["red.material.json".into()], sounds: vec![] };
    let options = RendererOptions {
        format: FORMAT,
        display: pen_render::Display::SDR,
        assets: dir.to_path_buf(),
        veil: None,
        pipeline_cache: None,
    };
    let mut renderer = Renderer::new(device.clone(), queue.clone(), options).unwrap();
    let mut target = Offscreen::new(&device, 64);

    let mut base = frame(vec![Draw {
        mesh: 0,
        material: 0,
        transform: translate(-0.4, 0.0, 0.5),
        parameters: None,
        extra: None,
    }]);
    base.quality.taa = true;
    pump(&mut renderer, &base, &catalog, &mut target, |e| {
        ready(e, "quad.mesh.json", 1) && ready(e, "red.material.json", 1) && ready(e, "flat.wgsl", 1)
    });
    assert_eq!(renderer.passes(), ["taa", "present"]);

    // Advance the red square rightward by +0.15 NDC per frame with explicit previous transforms.
    let mut prev_x = -0.4f32;
    for step in 1..=5 {
        let next_x = -0.4 + step as f32 * 0.15;
        let mut f = base.clone();
        f.draws[0].transform = translate(next_x, 0.0, 0.5);
        f.motions.push(Motion { draw: 0, previous: translate(prev_x, 0.0, 0.5) });
        renderer.submit(&f, &catalog, &mut target).unwrap();
        prev_x = next_x;
    }

    // Final square covers NDC x in [0.15, 0.55] -> pixel x in [36.8, 49.6].
    let image = pixels(&device, &queue, &target.0);
    let at = |x: usize, y: usize| image[y * 64 + x];
    let center = at(43, 32);
    let trail = at(28, 32); // Where the square sat on the previous frame (NDC x = -0.125).
    assert!(
        center[0].abs_diff(shown(1.0)) <= 4 && center[1] <= 2 && center[2] <= 2,
        "moving quad interior should stay bright red under TAA, got {center:?}"
    );
    assert!(trail[0] <= 2 && trail[1] <= 2 && trail[2] <= 2, "trailing background must be black without ghosting, got {trail:?}");
}

#[test]
fn velocity_motion_blur_smears_along_motion_axis_including_skeletal_motion() {
    let Some((device, queue)) = device() else { return };
    let temp = TempDir::new("motion_blur");
    let dir = temp.path();
    write_quad_mesh(dir, "quad.mesh.json", 0.2, false);
    write_quad_mesh(dir, "skinned.mesh.json", 0.2, true);
    std::fs::write(dir.join("flat.wgsl"), PLAIN_AND_SKINNED_SHADER).unwrap();
    std::fs::write(
        dir.join("white.material.json"),
        json!({"shader": "flat.wgsl", "color": [1.0, 1.0, 1.0, 1.0], "lighting": "unlit"}).to_string(),
    )
    .unwrap();
    let catalog = Catalog {
        meshes: vec!["quad.mesh.json".into(), "skinned.mesh.json".into()],
        materials: vec!["white.material.json".into()],
        sounds: vec![],
    };
    let options = RendererOptions {
        format: FORMAT,
        display: pen_render::Display::SDR,
        assets: dir.to_path_buf(),
        veil: None,
        pipeline_cache: None,
    };
    let mut renderer = Renderer::new(device.clone(), queue.clone(), options).unwrap();
    let mut target = Offscreen::new(&device, 64);

    // 1. Rigid per-object horizontal motion blur (moving right along +X, zero Y motion).
    let mut rigid = frame(vec![Draw {
        mesh: 0,
        material: 0,
        transform: translate(0.0, 0.0, 0.5),
        parameters: None,
        extra: None,
    }]);
    rigid.quality.effects = true;
    rigid.motion_blur = MotionBlur { strength: 1.0, max_radius: 0.16, samples: 16 };
    rigid.motions.push(Motion { draw: 0, previous: translate(-0.32, 0.0, 0.5) });
    pump(&mut renderer, &rigid, &catalog, &mut target, |e| {
        ready(e, "quad.mesh.json", 1) && ready(e, "white.material.json", 1) && ready(e, "flat.wgsl", 1)
    });
    assert_eq!(renderer.passes(), ["motion blur", "present"]);

    // Quad spans NDC [-0.2, 0.2] => pixel [25.6, 38.4] on a 64x64 image.
    let img = pixels(&device, &queue, &target.0);
    let at = |x: usize, y: usize| img[y * 64 + x];
    let left_smear = at(24, 32);
    let right_smear = at(40, 32);
    let top_outside = at(32, 23);
    let bottom_outside = at(32, 41);
    assert!(
        left_smear[0] >= 15 && right_smear[0] >= 15,
        "horizontal motion must smear past left/right edges: left={left_smear:?}, right={right_smear:?}"
    );
    assert!(
        top_outside[0] == 0 && bottom_outside[0] == 0,
        "purely horizontal motion must not smear vertically: top={top_outside:?}, bottom={bottom_outside:?}"
    );

    // 2. Automatic skeletal joint motion blur (draw transform stays fixed at IDENTITY while joint 0 translates).
    let mut skinned_a = frame(vec![Draw {
        mesh: 1,
        material: 0,
        transform: translate(0.0, 0.0, 0.5),
        parameters: None,
        extra: None,
    }]);
    skinned_a.quality.effects = true;
    skinned_a.motion_blur = MotionBlur { strength: 1.0, max_radius: 0.16, samples: 16 };
    skinned_a.bone_matrices.push(BoneMatrix { inset: 0, draw: 0, joint: 0, matrix: translate(-0.32, 0.0, 0.0) });
    pump(&mut renderer, &skinned_a, &catalog, &mut target, |e| ready(e, "skinned.mesh.json", 1));
    // Now move joint 0 to x = 0.0 in the very next frame (no explicit Motion record needed!).
    let mut skinned_b = skinned_a.clone();
    skinned_b.bone_matrices[0].matrix = IDENTITY;
    renderer.submit(&skinned_b, &catalog, &mut target).unwrap();
    let sk_img = pixels(&device, &queue, &target.0);
    let sk_at = |x: usize, y: usize| sk_img[y * 64 + x];
    assert!(
        sk_at(24, 32)[0] >= 15 && sk_at(40, 32)[0] >= 15,
        "skeletal joint motion should generate automatic velocity motion blur: left={:?}, right={:?}",
        sk_at(24, 32),
        sk_at(40, 32)
    );
    assert_eq!(sk_at(32, 23)[0], 0, "skeletal horizontal motion should not smear vertically");
}

#[test]
fn bokeh_dof_keeps_focal_plane_sharp_and_blurs_defocused_plane() {
    let Some((device, queue)) = device() else { return };
    let temp = TempDir::new("dof");
    let dir = temp.path();
    write_quad_mesh(dir, "quad.mesh.json", 0.45, false);
    write_stripes_png(&dir.join("checker.png"), 64, 4);
    std::fs::write(dir.join("flat.wgsl"), PLAIN_AND_SKINNED_SHADER).unwrap();
    std::fs::write(
        dir.join("checker.material.json"),
        json!({"shader": "flat.wgsl", "texture": "checker.png", "color": [1.0, 1.0, 1.0, 1.0], "lighting": "unlit"})
            .to_string(),
    )
    .unwrap();
    let catalog = Catalog {
        meshes: vec!["quad.mesh.json".into()],
        materials: vec!["checker.material.json".into()],
        sounds: vec![],
    };
    let options = RendererOptions {
        format: FORMAT,
        display: pen_render::Display::SDR,
        assets: dir.to_path_buf(),
        veil: None,
        pipeline_cache: None,
    };
    let mut renderer = Renderer::new(device.clone(), queue.clone(), options).unwrap();
    let mut target = Offscreen::new(&device, 128);

    // Left quad at depth z = 0.20, right quad at depth z = 0.80.
    let mut f = frame(vec![
        Draw { mesh: 0, material: 0, transform: translate(-0.5, 0.0, 0.2), parameters: None, extra: None },
        Draw { mesh: 0, material: 0, transform: translate(0.5, 0.0, 0.8), parameters: None, extra: None },
    ]);
    f.quality.effects = true;
    f.dof = Dof { focus_distance: 0.2, focus_range: 0.02, aperture: 0.08, max_blur: 0.06 };
    pump(&mut renderer, &f, &catalog, &mut target, |e| {
        ready(e, "quad.mesh.json", 1)
            && ready(e, "checker.material.json", 1)
            && ready(e, "checker.png", 1)
            && ready(e, "flat.wgsl", 1)
    });
    assert_eq!(renderer.passes(), ["dof", "present"]);

    let contrast = |img: &[[u8; 4]], x0: usize, x1: usize| -> u8 {
        let mut lo = 255u8;
        let mut hi = 0u8;
        for y in 44..84 {
            for x in x0..x1 {
                let v = img[y * 128 + x][0];
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        hi.saturating_sub(lo)
    };

    let near_focus = pixels(&device, &queue, &target.0);
    let left_c = contrast(&near_focus, 16, 48);
    let right_c = contrast(&near_focus, 80, 112);
    assert!(
        left_c >= 180 && right_c * 2 < left_c,
        "focus at 0.2 should keep left quad crisp ({left_c}) and blur right quad ({right_c})"
    );

    // Rack focus to the far quad (z = 0.80) AND verify combined TAA + DoF + MotionBlur graph!
    f.dof.focus_distance = 0.8;
    renderer.submit(&f, &catalog, &mut target).unwrap();
    let far_focus = pixels(&device, &queue, &target.0);
    let left_c2 = contrast(&far_focus, 16, 48);
    let right_c2 = contrast(&far_focus, 80, 112);
    assert!(
        right_c2 >= 180 && left_c2 * 2 < right_c2,
        "focus at 0.8 should keep right quad crisp ({right_c2}) and blur left quad ({left_c2})"
    );

    f.quality.taa = true;
    f.motion_blur = MotionBlur { strength: 0.5, max_radius: 0.05, samples: 8 };
    renderer.submit(&f, &catalog, &mut target).unwrap();
    assert_eq!(renderer.passes(), ["taa", "dof", "motion blur", "present"]);
}
