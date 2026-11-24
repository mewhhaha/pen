//! Shadows as the game sees them, on a real (headless) GPU: a sun casting
//! soft shadows from a tall column onto a floor, contact shadows under a
//! small thing the maps' texels are too coarse for, and a spot and a point
//! light shadowing objects, drawn with the render packet's fields and
//! `SHADOWS_WGSL` / `LIGHTS_WGSL` as a material reads them. Each test writes
//! what it drew to target/look-shots (or the directory in `SHADOW_SHOTS`).
//! Skips when no adapter is available.
#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::cast_possible_truncation, reason = "test geometry")]
use std::path::{Path, PathBuf};

mod common;

use common::*;
use pen_protocol::*;
use pen_render::{Display, Renderer, RendererOptions};

const LIT: &str = r#"
struct Scene {
  view: mat4x4f, projection: mat4x4f, ambient: vec4f, light_direction: vec4f, light_color: vec4f,
  light_view_projection: mat4x4f, shadow: vec4f, zenith: vec4f, horizon: vec4f, ground: vec4f, eye: vec4f,
  cascades: vec4f,
  cascade_view_projection: array<mat4x4f, 4>,
  cascade_texels: array<vec4f, 4>,
  shadow_params: vec4f,
};
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
${shadows}
${lights}
${environment}
${surface}
struct Surface {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
};
const bounds_margin = 0.0;
@vertex fn vertex(@location(0) position: vec3f, @location(1) normal: vec3f) -> Surface {
  let world = draw.model * vec4f(position, 1.0);
  return Surface(scene.projection * scene.view * world, world.xyz, (draw.normal * vec4f(normal, 0.0)).xyz);
}
@fragment fn fragment(surface: Surface) -> @location(0) vec4f {
  let n = normalize(surface.normal);
  let view_dir = normalize(scene.eye.xyz - surface.world);
  let to_sun = -normalize(scene.light_direction.xyz);
  let sun = max(dot(n, to_sun), 0.0) * sun_shadow(surface.world, n, surface.position);
  let seen = scene.view * vec4f(surface.world, 1.0);
  let local = clustered_surface(surface.world, n, view_dir, 0.9, vec3f(0.04), surface.position, seen.z);
  let light = scene.ambient.rgb + scene.light_color.rgb * sun + local.diffuse;
  return vec4f(draw.color.rgb * light, 1.0);
}
"#;

fn shots() -> PathBuf {
    let directory = std::env::var_os("SHADOW_SHOTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/look-shots"));
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

fn boxed(x: f32, z: f32) -> String {
    // A box from (-x, 0, -z) to (x, 1, z), per-face normals.
    let (y0, y1) = (0.0, 1.0);
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 1.0, 0.0], [[-x, y1, -z], [-x, y1, z], [x, y1, z], [x, y1, -z]]),
        ([0.0, -1.0, 0.0], [[-x, y0, z], [-x, y0, -z], [x, y0, -z], [x, y0, z]]),
        ([1.0, 0.0, 0.0], [[x, y0, -z], [x, y1, -z], [x, y1, z], [x, y0, z]]),
        ([-1.0, 0.0, 0.0], [[-x, y0, z], [-x, y1, z], [-x, y1, -z], [-x, y0, -z]]),
        ([0.0, 0.0, 1.0], [[x, y0, z], [x, y1, z], [-x, y1, z], [-x, y0, z]]),
        ([0.0, 0.0, -1.0], [[-x, y0, -z], [-x, y1, -z], [x, y1, -z], [x, y0, -z]]),
    ];
    let (mut positions, mut normals, mut uvs, mut indices) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (index, (normal, corners)) in faces.iter().enumerate() {
        for corner in corners {
            positions.extend(corner.iter().map(|value| value.to_string()));
            normals.extend(normal.iter().map(|value| value.to_string()));
            uvs.extend(["0".to_owned(), "0".to_owned()]);
        }
        let base = index * 4;
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3].map(|value| value.to_string()));
    }
    format!(
        r#"{{"positions":[{}],"normals":[{}],"uvs":[{}],"indices":[{}]}}"#,
        positions.join(","),
        normals.join(","),
        uvs.join(","),
        indices.join(",")
    )
}

fn renderer(device: &wgpu::Device, queue: &wgpu::Queue, root: PathBuf) -> Renderer {
    let options =
        RendererOptions { format: FORMAT, display: Display::SDR, assets: root, veil: None, pipeline_cache: None };
    Renderer::new(device.clone(), queue.clone(), options).unwrap()
}

/// A look-at camera from `eye` to `target` and a perspective projection.
fn camera(eye: [f32; 3], target: [f32; 3], aspect: f32) -> (Matrix, Matrix, f64) {
    let length = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let normalize = |v: [f32; 3]| [v[0] / length(v), v[1] / length(v), v[2] / length(v)];
    let cross =
        |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let back = normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
    let right = normalize(cross([0.0, 1.0, 0.0], back));
    let up = cross(back, right);
    let view = [
        right[0],
        up[0],
        back[0],
        0.0,
        right[1],
        up[1],
        back[1],
        0.0,
        right[2],
        up[2],
        back[2],
        0.0,
        -dot(right, eye),
        -dot(up, eye),
        -dot(back, eye),
        1.0,
    ];
    let tangent = 0.5f32;
    let (near, far, f) = (0.1f32, 100.0f32, 1.0 / tangent);
    let projection = [
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        f,
        0.0,
        0.0,
        0.0,
        0.0,
        far / (near - far),
        -1.0,
        0.0,
        0.0,
        near * far / (near - far),
        0.0,
    ];
    (view, projection, f64::from(tangent))
}

/// The light's view along `ray`, anchored 1000 back, as Ink's `graphics.shadow` has it.
fn light_view(ray: [f64; 3]) -> Matrix {
    let length = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
    let ray = ray.map(|value| value / length);
    let across = if ray[1].abs() > 0.99 { [0.0, 0.0, 1.0] } else { [0.0, 1.0, 0.0] };
    let cross =
        |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let normalize = |a: [f64; 3]| {
        let length = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        a.map(|value| value / length)
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let right = normalize(cross(ray, across));
    let up = cross(right, ray);
    let origin = ray.map(|value| value * -1000.0);
    [
        right[0],
        up[0],
        -ray[0],
        0.0,
        right[1],
        up[1],
        -ray[1],
        0.0,
        right[2],
        up[2],
        -ray[2],
        0.0,
        -dot(right, origin),
        -dot(up, origin),
        dot(ray, origin),
        1.0,
    ]
    .map(|value| value as f32)
}

/// Fits cascades to slices of the camera's view cut at `splits`, as a game
/// does (ink's `shadow.cascades`): a sphere round each slice's corners, its
/// radius rounded up and its centre moved in whole texels of the light's view.
fn fit(view: &Matrix, tangent: f64, aspect: f64, ray: [f64; 3], texels: u32, splits: &[f64]) -> (Matrix, Vec<Cascade>) {
    let light = light_view(ray);
    let wide = |m: &Matrix| m.map(f64::from);
    let (v, l) = (wide(view), wide(&light));
    let to_world = |p: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|i| (0..3).map(|row| v[i * 4 + row] * (p[row] - v[12 + row])).sum())
    };
    let to_light = |p: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|row| l[row] * p[0] + l[4 + row] * p[1] + l[8 + row] * p[2] + l[12 + row])
    };
    let mut cascades = Vec::new();
    let mut near = 0.1;
    for &far in splits {
        let corners: Vec<[f64; 3]> = [near, far]
            .into_iter()
            .flat_map(|d| {
                [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
                    .map(|(x, y)| to_world([x * d * tangent * aspect, y * d * tangent, -d]))
            })
            .collect();
        let centre: [f64; 3] = std::array::from_fn(|axis| corners.iter().map(|c| c[axis]).sum::<f64>() / 8.0);
        let radius = corners
            .iter()
            .map(|c| (0..3).map(|axis| (c[axis] - centre[axis]).powi(2)).sum::<f64>().sqrt())
            .fold(0.0, f64::max);
        let radius = (radius * 16.0).ceil() / 16.0 * f64::from(texels) / f64::from(texels - 1);
        let texel = 2.0 * radius / f64::from(texels);
        let [x, y, z] = to_light(centre);
        let (x, y) = ((x / texel).round() * texel, (y / texel).round() * texel);
        let distance = -z;
        let (near_plane, far_plane) = (distance - radius - 60.0, distance + radius + 60.0);
        let mut projection = IDENTITY;
        projection[0] = (1.0 / radius) as f32;
        projection[5] = (1.0 / radius) as f32;
        projection[10] = (-1.0 / (far_plane - near_plane)) as f32;
        projection[12] = (-x / radius) as f32;
        projection[13] = (-y / radius) as f32;
        projection[14] = (-near_plane / (far_plane - near_plane)) as f32;
        cascades.push(Cascade { projection, texel: texel as f32, far: far as f32 });
        near = far;
    }
    (light, cascades)
}

fn save(name: &str, pixels: &[[u8; 4]], size: u32) {
    let image = image::RgbaImage::from_fn(size, size, |x, y| {
        let [r, g, b, _] = pixels[(y * size + x) as usize];
        image::Rgba([r, g, b, 255])
    });
    image.save(shots().join(name)).unwrap();
}

fn luminance(pixel: [u8; 4]) -> f64 {
    0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
}

fn scale_at(sx: f32, sy: f32, sz: f32, x: f32, y: f32, z: f32) -> Matrix {
    [sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, sz, 0.0, x, y, z, 1.0]
}

struct Scene {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    catalog: Catalog,
    target: Offscreen,
    size: u32,
}

impl Scene {
    fn new(name: &str) -> Option<Scene> {
        let source = LIT
            .replace("${shadows}", pen_render::SHADOWS_WGSL)
            .replace("${lights}", pen_render::LIGHTS_WGSL)
            .replace("${environment}", pen_render::ENVIRONMENT_WGSL)
            .replace("${surface}", &format!("{}\n{}", pen_render::SURFACE_LOBES_WGSL, pen_render::SURFACE_WGSL));
        Scene::with_shader(name, source)
    }

    fn with_shader(name: &str, source: String) -> Option<Scene> {
        let (device, queue) = featured_device()?;
        let floor = br#"{"positions":[-14,0,-14, 14,0,-14, -14,0,14, 14,0,14],"normals":[0,1,0,0,1,0,0,1,0,0,1,0],
                         "uvs":[0,0,1,0,0,1,1,1],"indices":[0,2,1,1,2,3]}"#;
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("shadows-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let files: Vec<(&str, Vec<u8>)> = vec![
            ("catalog.json", br#"{"meshes":["floor.mesh.json","box.mesh.json"],"materials":["floor.material.json","stone.material.json","red.material.json"]}"#.to_vec()),
            ("floor.mesh.json", floor.to_vec()),
            ("box.mesh.json", boxed(0.5, 0.5).into_bytes()),
            ("lit.wgsl", source.into_bytes()),
            ("floor.material.json", br#"{"shader":"lit.wgsl","color":[0.62,0.6,0.55,1],"lighting":"lit"}"#.to_vec()),
            ("stone.material.json", br#"{"shader":"lit.wgsl","color":[0.75,0.72,0.68,1],"lighting":"lit"}"#.to_vec()),
            ("red.material.json", br#"{"shader":"lit.wgsl","color":[0.8,0.25,0.2,1],"lighting":"lit"}"#.to_vec()),
        ];
        for (key, bytes) in files {
            std::fs::write(root.join(key), bytes).unwrap();
        }
        let catalog = boundary::read_catalog(&root).unwrap();
        let renderer = renderer(&device, &queue, root);
        let size = 512;
        let target = Offscreen::new(&device, size);
        Some(Scene { device, queue, renderer, catalog, target, size })
    }

    fn draw(&mut self, frame: &RenderFrame) -> Vec<[u8; 4]> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            self.renderer.submit(frame, &self.catalog, &mut self.target).unwrap();
            if self.renderer.pending() == 0 && self.renderer.stats().draw_calls >= frame.draws.len() as u32 {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "assets did not load: {:?}", self.renderer.take_events());
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        pixels(&self.device, &self.queue, &self.target.0)
    }
}

fn draw(mesh: u32, material: u32, transform: Matrix) -> Draw {
    Draw { mesh, material, transform, parameters: None, extra: None }
}

fn vector(x: f32, y: f32, z: f32) -> Vector {
    Vector { x, y, z }
}

/// The sun frame: a column, a short block beside it, a floor.
fn sun_frame(scene: &Scene, quality: u32) -> RenderFrame {
    let aspect = 1.0f32;
    let (view, projection, tangent) = camera([7.0, 6.0, 9.0], [0.0, 1.2, 0.5], aspect);
    let mut frame = frame(vec![
        draw(0, 0, IDENTITY),
        draw(1, 1, scale_at(1.0, 5.0, 1.0, 0.0, 0.0, 0.0)),
        draw(1, 2, scale_at(0.5, 0.35, 0.5, 2.3, 0.0, -1.6)),
        draw(1, 1, scale_at(0.12, 0.9, 0.12, -2.4, 0.0, 1.6)),
    ]);
    frame.view = view;
    frame.projection = projection;
    frame.ambient = vector(0.16, 0.18, 0.22);
    frame.light_color = vector(1.15, 1.05, 0.9);
    frame.light_direction = vector(0.6, -0.5, 0.3);
    frame.clear = Color { r: 0.5, g: 0.6, b: 0.75, a: 1.0 };
    let ray = [0.6, -0.5, 0.3];
    frame.shadow.texels = 1024;
    if quality == 0 {
        // As it was: one hard map over the view, no contact shadows.
        let (light, cascades) = fit(&frame.view, tangent, f64::from(aspect), ray, 1024, &[30.0]);
        (frame.shadow.view, frame.cascades) = (light, cascades);
        frame.shadow.filter = 0;
    } else {
        let (light, cascades) = fit(&frame.view, tangent, f64::from(aspect), ray, 1024, &[9.0, 16.0, 30.0]);
        (frame.shadow.view, frame.cascades) = (light, cascades);
        frame.shadow.filter = 2;
        frame.shadow.softness = 0.07;
        frame.shadow.contact = 24;
        frame.shadow.contact_length = 0.5;
    }
    let _ = scene;
    frame
}

#[test]
fn a_sun_casts_soft_shadows_and_contact_shadows_from_a_tall_column() {
    let Some(mut scene) = Scene::new("sun") else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let before = scene.draw(&sun_frame(&scene, 0));
    let after_frame = sun_frame(&scene, 1);
    let after = scene.draw(&after_frame);
    save("shadows-sun-before.png", &before, scene.size);
    save("shadows-sun-after.png", &after, scene.size);
    // Both show the column's shadow: dark floor pixels where there was light.
    let darkest = |image: &[[u8; 4]]| image.iter().map(|p| luminance(*p)).fold(f64::MAX, f64::min);
    assert!(darkest(&before) < 60.0 && darkest(&after) < 60.0);
    // Soft shadows have far more pixels between lit and shadowed than hard ones.
    let between = |image: &[[u8; 4]]| {
        let values: Vec<f64> = image.iter().map(|p| luminance(*p)).collect();
        let (low, high) = (values.iter().cloned().fold(f64::MAX, f64::min), values.iter().cloned().fold(0.0, f64::max));
        let mut count = 0;
        // Along every row, the steps between neighbours that are a fraction of the whole range.
        for row in values.chunks(scene.size as usize) {
            for pair in row.windows(2) {
                let step = (pair[1] - pair[0]).abs();
                if step > (high - low) * 0.01 && step < (high - low) * 0.12 {
                    count += 1;
                }
            }
        }
        count
    };
    assert!(between(&after) > between(&before), "soft shadows ramp: {} vs {}", between(&after), between(&before));
    // Contact shadows darken the floor by the small block's and the thin post's feet.
    // The maps are made coarse (256 texels), so that a post and a block are small things to them.
    let mut coarse = sun_frame(&scene, 1);
    coarse.shadow.texels = 256;
    let (light, cascades) = fit(&coarse.view, 0.5, 1.0, [0.6, -0.5, 0.3], 256, &[9.0, 16.0, 30.0]);
    (coarse.shadow.view, coarse.cascades) = (light, cascades);
    let with = scene.draw(&coarse);
    coarse.shadow.contact = 0;
    let off = scene.draw(&coarse);
    save("shadows-sun-coarse-contact.png", &with, scene.size);
    save("shadows-sun-coarse-no-contact.png", &off, scene.size);
    let darker: usize = with.iter().zip(&off).filter(|(a, b)| luminance(**a) + 6.0 < luminance(**b)).count();
    let brighter: usize = with.iter().zip(&off).filter(|(a, b)| luminance(**a) > luminance(**b) + 6.0).count();
    assert!(darker > 20, "contact shadows darken the floor by what the maps miss: {darker}");
    assert!(brighter < darker / 4 + 50, "and do not lighten it: {brighter}");
}

#[test]
fn spot_and_point_lights_cast_shadows_within_their_budget() {
    let Some(mut scene) = Scene::new("local") else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let aspect = 1.0;
    let (view, projection, _) = camera([6.0, 7.0, 9.0], [0.0, 0.6, 0.0], aspect);
    let mut base = frame(vec![
        draw(0, 0, IDENTITY),
        draw(1, 1, scale_at(0.8, 1.6, 0.8, -1.0, 0.0, 0.0)),
        draw(1, 2, scale_at(0.9, 0.9, 0.9, 2.2, 0.0, 1.0)),
    ]);
    base.view = view;
    base.projection = projection;
    base.ambient = vector(0.1, 0.11, 0.14);
    base.light_color = vector(0.0, 0.0, 0.0);
    base.light_direction = vector(0.0, -1.0, 0.0);
    base.shadow.local = 4;
    base.shadow.local_texels = 1024;
    let spot = |shadow: bool, id: u32| Light {
        position: vector(-5.5, 4.0, 1.0),
        radius: 20.0,
        color: vector(60.0, 52.0, 38.0),
        direction: vector(0.85, -0.45, -0.1),
        inner: 0.95,
        outer: 0.8,
        shadow,
        id,
    };
    let point = |shadow: bool, id: u32| Light {
        position: vector(2.6, 3.2, 5.2),
        radius: 18.0,
        color: vector(14.0, 30.0, 48.0),
        direction: vector(0.0, -1.0, 0.0),
        inner: -1.0,
        outer: -1.0,
        shadow,
        id,
    };
    base.lights = vec![spot(false, 1), point(false, 2)];
    let before = scene.draw(&base);
    save("shadows-local-before.png", &before, scene.size);
    let mut lit = base.clone();
    lit.lights = vec![spot(true, 1), point(true, 2)];
    let after = scene.draw(&lit);
    let stats = scene.renderer.stats();
    save("shadows-local-after.png", &after, scene.size);
    // A spot is a face and a point light six.
    assert_eq!((stats.local_shadow_casters, stats.local_shadow_faces), (2, 7), "{stats:?}");
    let darker = after.iter().zip(&before).filter(|(a, b)| luminance(**a) + 8.0 < luminance(**b)).count();
    assert!(darker > 1500, "the lights' shadows darken the floor behind the objects: {darker} pixels");
    // A second identical frame keeps every face; moving a caster draws again.
    scene.draw(&lit);
    let again = scene.renderer.stats();
    assert_eq!((again.local_shadow_faces, again.local_shadow_kept), (7, 7), "nothing moved: {again:?}");
    let mut moved = lit.clone();
    moved.draws[1] = draw(1, 1, scale_at(0.8, 1.6, 0.8, -1.0, 0.0, 0.1));
    let moved_image = scene.draw(&moved);
    let after_move = scene.renderer.stats();
    assert!(after_move.local_shadow_kept < 7, "a moved caster redraws the faces it reaches: {after_move:?}");
    assert_ne!(moved_image, after);
    // A budget of one: the nearer, larger light only.
    let mut one = lit.clone();
    one.shadow.local = 1;
    scene.draw(&one);
    assert_eq!(scene.renderer.stats().local_shadow_casters, 1);
    let mut none = lit.clone();
    none.shadow.local = 0;
    let unshadowed = scene.draw(&none);
    assert_eq!(unshadowed, before, "no budget, no shadows");
}

/// The shader `pen new` gives a game, as it is, with the sun's shadows from the frame.
#[test]
fn the_new_game_template_shades_with_the_suns_shadows() {
    let template = include_str!("../../pen/templates/new/src/assets/lit.wgsl").to_owned();
    let Some(mut scene) = Scene::with_shader("template", template) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let lit = sun_frame(&scene, 1);
    let mut none = lit.clone();
    none.shadow.texels = 0;
    none.cascades.clear();
    let shadowed = scene.draw(&lit);
    let open = scene.draw(&none);
    save("shadows-template.png", &shadowed, scene.size);
    let darker = shadowed.iter().zip(&open).filter(|(a, b)| luminance(**a) + 20.0 < luminance(**b)).count();
    assert!(darker > 3000, "the template's floor is shadowed behind the column: {darker} pixels");
}
