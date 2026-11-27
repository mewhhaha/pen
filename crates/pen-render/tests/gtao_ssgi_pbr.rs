//! Headless GPU pixel witnesses for Checkpoint 8b:
//! 1. Horizon-based GTAO darkens concave geometry creases and SSGI bounces red
//!    indirect light from a bright floor onto an adjacent shadowed neutral wall.
//! 2. Modern GPU particles (flipbook sub-UV grid, soft depth fade against opaque
//!    geometry, directional sun lighting) and camera-facing GPU ribbon trails.
//! 3. Per-pixel PBR ORM (`ormTexture`), `emissiveTexture`, and `uvTransform`
//!    (`KHR_texture_transform`) bindings and shader helpers.
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
        let dir = std::env::temp_dir().join(format!("pen-render-ch8b-{name}-{}", std::process::id()));
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

const UNLIT_SHADER: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw {
  model: mat4x4f,
  normal: mat4x4f,
  color: vec4f,
  settings: vec4f,
  parameters: vec4f,
  extra: array<vec4f, 32>,
  surface: array<vec4f, 5>,
  uv_transform: vec4f,
  maps: vec4f,
};
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@group(1) @binding(1) var base: texture_2d<f32>;
@group(1) @binding(2) var base_sampler: sampler;
@group(1) @binding(4) var layer_tex: texture_2d<f32>;

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

fn sample_atlas_tile(tex: texture_2d<f32>, samp: sampler, tile: u32, uv: vec2f) -> vec4f {
  let dims = vec2f(textureDimensions(tex, 0));
  let cell = dims / vec2f(4.0, 3.0);
  let col = f32(tile % 4u);
  let row = f32(tile / 4u);
  let auv = (vec2f(col, row) * cell + vec2f(1.0) + fract(uv) * (cell - vec2f(2.0))) / dims;
  return textureSampleLevel(tex, samp, auv, 0.0);
}

@fragment fn fragment(input: Out) -> @location(0) vec4f {
  let s = input.uv * draw.uv_transform.zw;
  let uv = vec2f(s.x * draw.maps.x + s.y * draw.maps.y, -s.x * draw.maps.y + s.y * draw.maps.x) + draw.uv_transform.xy;
  let base_rgb = textureSample(base, base_sampler, uv).rgb;
  let is_atlas = draw.settings.w > 4.5;
  let orm = select(textureSample(layer_tex, base_sampler, uv), sample_atlas_tile(layer_tex, base_sampler, 0u, uv), is_atlas);
  let ao = select(1.0, orm.r, draw.maps.z > 0.5);
  var rgb = draw.color.rgb * base_rgb * ao;
  if (draw.maps.w > 0.5) {
    let em = select(textureSample(layer_tex, base_sampler, uv).rgb, sample_atlas_tile(layer_tex, base_sampler, 1u, uv).rgb, is_atlas);
    rgb += em * draw.surface[4].w;
  }
  return vec4f(rgb, draw.color.a);
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

fn write_quad_xy(dir: &Path, name: &str, half_w: f32, half_h: f32) {
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

fn write_floor_xz(dir: &Path, name: &str, half_w: f32, half_d: f32) {
    let mesh = json!({
        "positions": [
            -half_w, 0.0,  half_d,
             half_w, 0.0,  half_d,
             half_w, 0.0, -half_d,
            -half_w, 0.0, -half_d
        ],
        "normals": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "uvs": [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "indices": [0, 1, 2, 0, 2, 3]
    });
    std::fs::write(dir.join(name), serde_json::to_vec(&mesh).unwrap()).unwrap();
}

fn avg_region(img: &[[u8; 4]], size: usize, x0: usize, x1: usize, y0: usize, y1: usize) -> [f64; 3] {
    let mut sum = [0.0; 3];
    let mut n = 0.0;
    for y in y0..y1 {
        for x in x0..x1 {
            let p = img[y * size + x];
            sum[0] += f64::from(p[0]);
            sum[1] += f64::from(p[1]);
            sum[2] += f64::from(p[2]);
            n += 1.0;
        }
    }
    sum.map(|c| c / n)
}

#[test]
fn gtao_darkens_concave_corners_and_ssgi_bleeds_red_floor_onto_white_wall() {
    let Some((device, queue)) = device() else { return };
    let dir = TempDir::new("gtao-ssgi");
    let root = dir.path();
    std::fs::write(root.join("unlit.wgsl"), UNLIT_SHADER).unwrap();
    write_quad_xy(root, "wall.json", 4.0, 4.0);
    write_floor_xz(root, "floor.json", 4.0, 2.5);
    std::fs::write(
        root.join("wall.mat.json"),
        serde_json::to_vec(&json!({ "shader": "unlit.wgsl", "color": [0.35, 0.35, 0.35, 1.0], "lighting": "unlit" }))
            .unwrap(),
    )
    .unwrap();
    std::fs::write(
        root.join("floor.mat.json"),
        serde_json::to_vec(&json!({ "shader": "unlit.wgsl", "color": [1.0, 0.02, 0.02, 1.0], "lighting": "unlit" }))
            .unwrap(),
    )
    .unwrap();
    let catalog = Catalog {
        meshes: vec!["wall.json".into(), "floor.json".into()],
        materials: vec!["wall.mat.json".into(), "floor.mat.json".into()],
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
    let mut target = Offscreen::new(&device, 192);

    // Camera at (0, 0.6, 0) looking down -Z:
    // Back gray wall at z = -3.0 (y in [-1, 7], seam at y = -1.0 -> screen y ≈ 0.5 + (1.6 / 3.0) * 0.5 ≈ 0.77).
    // Bright red floor at y = -1.0, z in [-3.0, +2.0] meeting the back wall in a 90-degree concave crease!
    let mut view = IDENTITY;
    view[13] = -0.6;
    let base_draws = vec![
        Draw {
            mesh: 0,
            material: 0,
            transform: translate_scale(0.0, 2.0, -3.0, 1.0, 1.0, 1.0),
            parameters: None,
            extra: None,
        },
        Draw {
            mesh: 1,
            material: 1,
            transform: translate_scale(0.0, -1.0, -0.5, 1.0, 1.0, 1.0),
            parameters: None,
            extra: None,
        },
    ];
    let mut f = frame(base_draws);
    f.view = view;
    f.projection = perspective_90(1.0);
    f.quality.msaa = 1;
    f.quality.fxaa = false;
    f.quality.bloom = false;
    f.quality.sky = false;

    pump(&mut renderer, &f, &catalog, &mut target, |events| {
        ready(events, "wall.json", 1) && ready(events, "floor.json", 1) && ready(events, "unlit.wgsl", 1)
    });

    // 1. GTAO on: crease just above y_seam (~y = 140..146 in 192px) must be darker than open upper wall (y = 32..64).
    f.quality.occlusion = true;
    f.quality.ssgi = false;
    renderer.submit(&f, &catalog, &mut target).unwrap();
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let gtao_img = pixels(&device, &queue, &target.0);
    let upper_wall = avg_region(&gtao_img, 192, 64, 128, 32, 64);
    let crease_wall = avg_region(&gtao_img, 192, 64, 128, 132, 142);
    assert!(
        crease_wall[0] + 5.0 < upper_wall[0],
        "GTAO must darken the concave floor-wall crease: upper={upper_wall:?}, crease={crease_wall:?}"
    );

    // 2. SSGI off vs on: bright red floor must bleed warm red light onto the neutral wall above the crease.
    f.quality.occlusion = false;
    f.quality.ssgi = false;
    renderer.submit(&f, &catalog, &mut target).unwrap();
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let no_gi = pixels(&device, &queue, &target.0);
    let wall_no_gi = avg_region(&no_gi, 192, 64, 128, 118, 140);

    f.quality.ssgi = true;
    renderer.submit(&f, &catalog, &mut target).unwrap();
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    assert!(renderer.passes().contains(&"ssgi"), "SSGI pass must run in post graph: {:?}", renderer.passes());
    let with_gi = pixels(&device, &queue, &target.0);
    let wall_with_gi = avg_region(&with_gi, 192, 64, 128, 118, 140);
    let bleed_before = wall_no_gi[0] - wall_no_gi[2];
    let bleed_after = wall_with_gi[0] - wall_with_gi[2];
    assert!(
        bleed_after > bleed_before + 8.0,
        "SSGI must bleed red bounce onto neutral wall: before={wall_no_gi:?} (R-B={bleed_before:.1}), after={wall_with_gi:?} (R-B={bleed_after:.1})"
    );
}

#[test]
fn particles_flipbook_soft_depth_lighting_and_ribbon_trails() {
    let Some((device, queue)) = device() else { return };
    let dir = TempDir::new("particles-ribbons");
    let root = dir.path();
    std::fs::write(root.join("unlit.wgsl"), UNLIT_SHADER).unwrap();
    write_quad_xy(root, "wall.json", 4.0, 4.0);

    // 2x2 flipbook texture: TL=red, TR=green, BL=blue, BR=yellow.
    let mut sheet = image::RgbaImage::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            let color = match (x >= 32, y >= 32) {
                (false, false) => [255, 20, 20, 255],
                (true, false) => [20, 255, 20, 255],
                (false, true) => [20, 40, 255, 255],
                (true, true) => [255, 240, 20, 255],
            };
            sheet.put_pixel(x, y, image::Rgba(color));
        }
    }
    sheet.save(root.join("sheet.png")).unwrap();
    image::RgbaImage::from_pixel(16, 16, image::Rgba([255, 255, 255, 255])).save(root.join("white.png")).unwrap();

    std::fs::write(
        root.join("wall.mat.json"),
        serde_json::to_vec(&json!({ "shader": "unlit.wgsl", "color": [0.05, 0.05, 0.05, 1.0], "lighting": "unlit" }))
            .unwrap(),
    )
    .unwrap();
    std::fs::write(
        root.join("sheet.mat.json"),
        serde_json::to_vec(&json!({
            "shader": "unlit.wgsl", "texture": "sheet.png", "color": [1.0, 1.0, 1.0, 1.0], "lighting": "unlit"
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        root.join("white.mat.json"),
        serde_json::to_vec(&json!({
            "shader": "unlit.wgsl", "texture": "white.png", "color": [1.0, 1.0, 1.0, 1.0], "lighting": "unlit"
        }))
        .unwrap(),
    )
    .unwrap();

    let catalog = Catalog {
        meshes: vec!["wall.json".into()],
        materials: vec!["wall.mat.json".into(), "sheet.mat.json".into(), "white.mat.json".into()],
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

    let make_emitter =
        |material: u32, z: f32, time: f32, soft_depth: f32, lit: f32, cols: u32, rows: u32, fps: f32| Emitter {
            material,
            transform: translate_scale(0.0, 0.0, z, 1.0, 1.0, 1.0),
            seed: 1,
            time,
            duration: 0.1,
            rate: 1.0,
            lifetime: 10.0,
            size_start: 1.8,
            size_end: 1.8,
            color_start: Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
            color_end: Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
            additive: false,
            soft_depth,
            lit,
            flipbook_cols: cols,
            flipbook_rows: rows,
            flipbook_fps: fps,
            ..Default::default()
        };

    let mut f = frame(vec![Draw {
        mesh: 0,
        material: 0,
        transform: translate_scale(0.0, 0.0, -3.0, 1.0, 1.0, 1.0),
        parameters: None,
        extra: None,
    }]);
    f.view = IDENTITY;
    f.projection = perspective_90(1.0);
    f.emitters = vec![make_emitter(1, -2.0, 0.25, 0.0, 0.0, 2, 2, 1.0)];
    f.ribbons = vec![
        RibbonPoint {
            ribbon: 1,
            material: 2,
            additive: false,
            position: Vector { x: -1.5, y: 0.0, z: -2.0 },
            width: 0.5,
            color: Color { r: 0.1, g: 0.9, b: 0.95, a: 1.0 },
            soft_depth: 0.0,
        },
        RibbonPoint {
            ribbon: 1,
            material: 2,
            additive: false,
            position: Vector { x: 1.5, y: 0.0, z: -2.0 },
            width: 0.5,
            color: Color { r: 0.1, g: 0.9, b: 0.95, a: 1.0 },
            soft_depth: 0.0,
        },
    ];
    pump(&mut renderer, &f, &catalog, &mut target, |events| {
        ready(events, "wall.json", 1)
            && ready(events, "sheet.png", 1)
            && ready(events, "white.png", 1)
            && ready(events, "unlit.wgsl", 1)
    });
    while renderer.pending() > 0 {
        renderer.wait(std::time::Duration::from_millis(50));
        renderer.submit(&f, &catalog, &mut target).unwrap();
    }

    // 1. Flipbook frame 0 (t=0.25s @ 1fps -> red) vs frame 1 (t=1.25s -> green).
    f.ribbons.clear();
    f.emitters = vec![make_emitter(1, -2.0, 0.25, 0.0, 0.0, 2, 2, 1.0)];
    renderer.submit(&f, &catalog, &mut target).unwrap();
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let px_f0 = pixels(&device, &queue, &target.0)[64 * 128 + 64];
    assert!(px_f0[0] > 180 && px_f0[1] < 80, "flipbook frame 0 must sample red quadrant: {px_f0:?}");

    f.emitters = vec![make_emitter(1, -2.0, 1.25, 0.0, 0.0, 2, 2, 1.0)];
    renderer.submit(&f, &catalog, &mut target).unwrap();
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let px_f1 = pixels(&device, &queue, &target.0)[64 * 128 + 64];
    assert!(px_f1[1] > 180 && px_f1[0] < 80, "flipbook frame 1 must sample green quadrant: {px_f1:?}");

    // 2. Soft depth fade: particle 0.1 units in front of opaque wall at z=-3.0 fades dramatically with soft_depth=2.0.
    f.emitters = vec![make_emitter(2, -2.9, 0.25, 0.0, 0.0, 1, 1, 0.0)];
    renderer.submit(&f, &catalog, &mut target).unwrap();
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let hard_px = pixels(&device, &queue, &target.0)[64 * 128 + 64];

    f.emitters = vec![make_emitter(2, -2.9, 0.25, 2.0, 0.0, 1, 1, 0.0)];
    renderer.submit(&f, &catalog, &mut target).unwrap();
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let soft_px = pixels(&device, &queue, &target.0)[64 * 128 + 64];
    assert!(
        u32::from(hard_px[0]) > u32::from(soft_px[0]) + 80,
        "soft depth fade must attenuate particle near wall: hard={hard_px:?}, soft={soft_px:?}"
    );

    // 3. Directional sun + ambient lighting on white particle: bright blue sun lights white particle blue.
    f.light_direction = Vector { x: 0.0, y: 0.0, z: -1.0 };
    f.light_color = Vector { x: 0.0, y: 0.1, z: 1.0 };
    f.ambient = Vector { x: 0.05, y: 0.05, z: 0.05 };
    f.emitters = vec![make_emitter(2, -2.0, 0.25, 0.0, 1.0, 1, 1, 0.0)];
    renderer.submit(&f, &catalog, &mut target).unwrap();
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let lit_px = pixels(&device, &queue, &target.0)[64 * 128 + 64];
    assert!(lit_px[2] > lit_px[0] + 60, "lit particle must respond to blue directional sun: {lit_px:?}");

    // 4. GPU Ribbon trail renders a continuous cyan strip across the viewport.
    f.emitters.clear();
    f.ribbons = vec![
        RibbonPoint {
            ribbon: 7,
            material: 2,
            additive: false,
            position: Vector { x: -1.5, y: 0.0, z: -2.0 },
            width: 0.6,
            color: Color { r: 0.05, g: 0.9, b: 0.95, a: 1.0 },
            soft_depth: 0.0,
        },
        RibbonPoint {
            ribbon: 7,
            material: 2,
            additive: false,
            position: Vector { x: 0.0, y: 0.0, z: -2.0 },
            width: 0.6,
            color: Color { r: 0.05, g: 0.9, b: 0.95, a: 1.0 },
            soft_depth: 0.0,
        },
        RibbonPoint {
            ribbon: 7,
            material: 2,
            additive: false,
            position: Vector { x: 1.5, y: 0.0, z: -2.0 },
            width: 0.6,
            color: Color { r: 0.05, g: 0.9, b: 0.95, a: 1.0 },
            soft_depth: 0.0,
        },
    ];
    renderer.submit(&f, &catalog, &mut target).unwrap();
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let rib_img = pixels(&device, &queue, &target.0);
    let mid = rib_img[64 * 128 + 64];
    let top = rib_img[16 * 128 + 64];
    assert!(mid[1] > 170 && mid[2] > 170 && mid[0] < 80, "ribbon center must render bright cyan: {mid:?}");
    assert!(top[1] < 90 && top[2] < 90, "above ribbon should show dark wall: {top:?}");
}

#[test]
fn pbr_orm_emissive_and_uv_transform_maps_shade_per_pixel() {
    let Some((device, queue)) = device() else { return };
    let dir = TempDir::new("pbr-maps");
    let root = dir.path();
    std::fs::write(root.join("unlit.wgsl"), UNLIT_SHADER).unwrap();
    write_quad_xy(root, "quad.json", 1.5, 1.5);

    // Base texture: left half red, right half blue.
    let mut base = image::RgbaImage::new(32, 32);
    // ORM texture: left half AO=0.15, right half AO=1.0.
    let mut orm = image::RgbaImage::new(32, 32);
    // Emissive texture: right half bright green glow, left half black.
    let mut emissive = image::RgbaImage::new(32, 32);
    for y in 0..32 {
        for x in 0..32 {
            let right = x >= 16;
            base.put_pixel(x, y, image::Rgba(if right { [20, 40, 240, 255] } else { [240, 40, 20, 255] }));
            orm.put_pixel(x, y, image::Rgba(if right { [255, 128, 0, 255] } else { [38, 128, 0, 255] }));
            emissive.put_pixel(x, y, image::Rgba(if right { [0, 220, 0, 255] } else { [0, 0, 0, 255] }));
        }
    }
    base.save(root.join("base.png")).unwrap();
    orm.save(root.join("orm.png")).unwrap();
    emissive.save(root.join("emissive.png")).unwrap();

    // Material with ORM + emissive + uvTransform shifting U by +0.5 (so left side of quad samples right side of textures!).
    std::fs::write(
        root.join("mapped.mat.json"),
        serde_json::to_vec(&json!({
            "shader": "unlit.wgsl",
            "texture": "base.png",
            "ormTexture": "orm.png",
            "emissiveTexture": "emissive.png",
            "uvTransform": { "offset": [0.5, 0.0], "scale": [1.0, 1.0], "rotation": 0.0 },
            "color": [1.0, 1.0, 1.0, 1.0],
            "lighting": "unlit"
        }))
        .unwrap(),
    )
    .unwrap();

    let catalog =
        Catalog { meshes: vec!["quad.json".into()], materials: vec!["mapped.mat.json".into()], sounds: vec![] };
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
    let mut f = frame(vec![Draw {
        mesh: 0,
        material: 0,
        transform: translate_scale(0.0, 0.0, -2.0, 1.0, 1.0, 1.0),
        parameters: None,
        extra: None,
    }]);
    f.view = IDENTITY;
    f.projection = perspective_90(1.0);
    pump(&mut renderer, &f, &catalog, &mut target, |events| {
        ready(events, "quad.json", 1)
            && ready(events, "base.png", 1)
            && ready(events, "orm.png", 1)
            && ready(events, "emissive.png", 1)
            && ready(events, "unlit.wgsl", 1)
    });
    renderer.submit(&f, &catalog, &mut target).unwrap();
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let img = pixels(&device, &queue, &target.0);
    // Because uvTransform offset.x = 0.5, left of quad (u=0.25 -> 0.75) reads right half of textures
    // (blue base * AO=1.0 + bright green emissive -> bright cyan-green), while right of quad (u=0.75 -> 1.25 wraps to 0.25)
    // reads left half of textures (red base * AO=0.15 + zero emissive -> dark dim red)!
    let left_px = img[64 * 128 + 36];
    let right_px = img[64 * 128 + 92];
    assert!(
        left_px[1] > 160 && left_px[2] > 160,
        "uvTransform + emissive + AO=1.0 should produce bright cyan-green on left: {left_px:?}"
    );
    assert!(
        right_px[0] > right_px[2] + 30 && right_px[0] < 160 && right_px[1] < 80,
        "uvTransform + AO=0.15 + no emissive should produce dim red on right: {right_px:?}"
    );
}
