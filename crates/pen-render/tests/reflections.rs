//! Reflections beyond the sky on a real (headless) GPU: local reflection
//! probes captured from a room (a coloured wall shows where the cube's faces
//! say, parallax corrected, within the budget's frame count) and screen-space
//! reflections of a mirror floor (the analytic mirror position, fading with
//! roughness and at the screen's edge, never meeting a back face), each
//! against the same scene without it. Pictures go to /tmp/fidelity/reflections.
#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::cast_sign_loss, reason = "pixels in range")]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "test geometry"
)]
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

mod common;

use common::*;
use pen_protocol::*;
use pen_render::{Display, RenderTarget, Renderer, RendererOptions};

fn shots() -> PathBuf {
    let directory = PathBuf::from("/tmp/fidelity/reflections");
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

fn assets(name: &str, files: &[(&str, &[u8])]) -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("look-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    for (key, bytes) in files {
        std::fs::write(root.join(key), bytes).unwrap();
    }
    root
}

fn renderer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    root: PathBuf,
    format: wgpu::TextureFormat,
    display: Display,
) -> Renderer {
    let options = RendererOptions { format, display, assets: root, veil: None, pipeline_cache: None };
    Renderer::new(device.clone(), queue.clone(), options).unwrap()
}

struct Target(wgpu::Texture);

impl Target {
    fn new(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        Target(device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }))
    }

    /// Every texel's bytes, rows packed.
    fn read(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u8> {
        let texture = &self.0;
        let texel = texture.format().block_copy_size(None).unwrap();
        let (width, height) = (texture.width(), texture.height());
        let row = (width * texel).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row * height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mapped = buffer.slice(..).get_mapped_range().unwrap();
        (0..height as usize)
            .flat_map(|y| mapped[y * row as usize..y * row as usize + (width * texel) as usize].to_vec())
            .collect()
    }
}

impl RenderTarget for Target {
    fn size(&self) -> (u32, u32) {
        (self.0.width(), self.0.height())
    }
    fn view(&mut self) -> Option<wgpu::TextureView> {
        Some(self.0.create_view(&Default::default()))
    }
    fn present(&mut self) {}
}

/// Submits `frame` until everything it draws has loaded.
fn settle(renderer: &mut Renderer, frame: &RenderFrame, catalog: &Catalog, target: &mut Target, draws: u32) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        renderer.submit(frame, catalog, target).unwrap();
        if renderer.pending() == 0 && renderer.stats().draw_calls >= draws {
            break;
        }
        assert!(Instant::now() < deadline, "assets did not load: {:?}", renderer.take_events());
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// A material that lights its color with the sun and the sky (`surface_ambient`), a draw's
/// parameters being metallic and roughness.
const ROOM: &str = r#"
struct Scene {
  view: mat4x4f, projection: mat4x4f, ambient: vec4f, light_direction: vec4f, light_color: vec4f,
  light_view_projection: mat4x4f, shadow: vec4f, zenith: vec4f, horizon: vec4f, ground: vec4f, eye: vec4f,
};
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
${environment}
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
};
@vertex fn vertex(@location(0) position: vec3f, @location(1) normal: vec3f) -> Corner {
  let world = draw.model * vec4f(position, 1.0);
  return Corner(scene.projection * scene.view * world, world.xyz, normal);
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let n = normalize(corner.normal);
  let view = normalize(scene.eye.xyz - corner.world);
  let albedo = draw.color.rgb;
  let metallic = draw.parameters.x;
  let roughness = draw.parameters.y;
  let sun = max(dot(n, -normalize(scene.light_direction.xyz)), 0.0);
  var light = surface_ambient(corner.world, n, view, albedo, metallic, roughness);
  light += albedo * (1.0 - metallic) * scene.light_color.rgb * sun;
  return vec4f(light, 1.0);
}
"#;

type V3 = [f32; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn unit(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}

/// Mesh JSON of quads, each four corners counterclockwise seen from the side its normal points to.
fn quads(list: &[[V3; 4]]) -> String {
    let (mut positions, mut normals, mut uvs, mut indices) = (vec![], vec![], vec![], vec![]);
    for (n, quad) in list.iter().enumerate() {
        let normal = unit(cross(sub(quad[1], quad[0]), sub(quad[2], quad[0])));
        for (corner, uv) in quad.iter().zip([[0, 0], [1, 0], [1, 1], [0, 1]]) {
            positions.extend(corner.iter().map(|v| v.to_string()));
            normals.extend(normal.iter().map(|v| v.to_string()));
            uvs.extend(uv.iter().map(|v: &i32| v.to_string()));
        }
        let first = n * 4;
        indices.extend([first, first + 1, first + 2, first, first + 2, first + 3].map(|i| i.to_string()));
    }
    format!(
        r#"{{"positions":[{}],"normals":[{}],"uvs":[{}],"indices":[{}]}}"#,
        positions.join(","),
        normals.join(","),
        uvs.join(","),
        indices.join(",")
    )
}

/// The outside of a box, its six faces facing out.
fn cuboid(low: V3, high: V3) -> Vec<[V3; 4]> {
    let [x0, y0, z0] = low;
    let [x1, y1, z1] = high;
    vec![
        [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]],
        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
        [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
        [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
    ]
}

fn look_at(eye: V3, target: V3, aspect: f32) -> (Matrix, Matrix) {
    let back = unit(sub(eye, target));
    let right = unit(cross([0.0, 1.0, 0.0], back));
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
    let (near, far, f) = (0.1f32, 100.0f32, 1.4f32);
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
    (view, projection)
}

fn project(view: &Matrix, projection: &Matrix, point: V3, size: (u32, u32)) -> (f32, f32) {
    let apply = |m: &Matrix, v: [f32; 4]| -> [f32; 4] {
        std::array::from_fn(|row| (0..4).map(|k| m[k * 4 + row] * v[k]).sum())
    };
    let clip = apply(projection, apply(view, [point[0], point[1], point[2], 1.0]));
    ((clip[0] / clip[3] * 0.5 + 0.5) * size.0 as f32, (0.5 - clip[1] / clip[3] * 0.5) * size.1 as f32)
}

/// The world point a pixel's ray meets the plane y = 0 at, and the ray's direction.
fn floor_point(eye: V3, view: &Matrix, projection: &Matrix, pixel: (f32, f32), size: (u32, u32)) -> Option<(V3, V3)> {
    let ndc = [pixel.0 / size.0 as f32 * 2.0 - 1.0, 1.0 - pixel.1 / size.1 as f32 * 2.0];
    // The view-space direction, then the world's (the view's rotation transposed).
    let d = [ndc[0] / projection[0], ndc[1] / projection[5], -1.0];
    let world = [
        view[0] * d[0] + view[1] * d[1] + view[2] * d[2],
        view[4] * d[0] + view[5] * d[1] + view[6] * d[2],
        view[8] * d[0] + view[9] * d[1] + view[10] * d[2],
    ];
    let world = unit(world);
    if world[1] >= -1e-4 {
        return None;
    }
    let t = -eye[1] / world[1];
    Some(([eye[0] + world[0] * t, 0.0, eye[2] + world[2] * t], world))
}

struct Scene {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    catalog: Catalog,
    target: Target,
    size: (u32, u32),
}

impl Scene {
    /// A scene of `meshes` (JSON, in the catalog's order) and `materials` (name, color), all
    /// drawn with the ROOM shader.
    fn new(name: &str, meshes: &[String], materials: &[(&str, [f32; 3])], size: (u32, u32)) -> Option<Scene> {
        let (device, queue) = device()?;
        let shader = ROOM.replace("${environment}", pen_render::ENVIRONMENT_WGSL);
        let mut files: Vec<(String, Vec<u8>)> = vec![("room.wgsl".into(), shader.into_bytes())];
        let names: Vec<String> = (0..meshes.len()).map(|i| format!("\"m{i}.mesh.json\"")).collect();
        let mats: Vec<String> = materials.iter().map(|(n, _)| format!("\"{n}.material.json\"")).collect();
        files.push((
            "catalog.json".into(),
            format!(r#"{{"meshes":[{}],"materials":[{}]}}"#, names.join(","), mats.join(",")).into_bytes(),
        ));
        for (i, mesh) in meshes.iter().enumerate() {
            files.push((format!("m{i}.mesh.json"), mesh.clone().into_bytes()));
        }
        for (n, c) in materials {
            files.push((
                format!("{n}.material.json"),
                // A material named `..._nocast` casts no shadow, so it is not in the depth
                // the march reads.
                format!(
                    r#"{{"shader":"room.wgsl","color":[{},{},{},1],"lighting":"lit","shadow":{}}}"#,
                    c[0],
                    c[1],
                    c[2],
                    !n.ends_with("_nocast")
                )
                .into_bytes(),
            ));
        }
        let borrowed: Vec<(&str, &[u8])> = files.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        let root = assets(name, &borrowed);
        let catalog = boundary::read_catalog(&root).unwrap();
        let renderer = renderer(&device, &queue, root, FORMAT, Display::SDR);
        let target = Target::new(&device, size.0, size.1, FORMAT);
        Some(Scene { device, queue, renderer, catalog, target, size })
    }

    fn submit(&mut self, frame: &RenderFrame) {
        self.renderer.submit(frame, &self.catalog, &mut self.target).unwrap();
    }

    fn settle(&mut self, frame: &RenderFrame, draws: u32) {
        settle(&mut self.renderer, frame, &self.catalog, &mut self.target, draws);
    }

    /// Submits `frames` more, then the picture, as RGB floats in 0..1, rows down.
    fn shoot(&mut self, frame: &RenderFrame, frames: u32) -> Vec<[f32; 3]> {
        for _ in 0..frames {
            self.submit(frame);
        }
        self.read()
    }

    fn read(&self) -> Vec<[f32; 3]> {
        self.target
            .read(&self.device, &self.queue)
            .chunks(4)
            .map(|p| [f32::from(p[2]) / 255.0, f32::from(p[1]) / 255.0, f32::from(p[0]) / 255.0])
            .collect()
    }

    fn save(&self, pixels: &[[f32; 3]], name: &str) {
        let (w, h) = self.size;
        let image = image::RgbImage::from_fn(w, h, |x, y| {
            let p = pixels[(y * w + x) as usize];
            image::Rgb(p.map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8))
        });
        image.save(shots().join(name)).unwrap();
    }
}

fn at(pixels: &[[f32; 3]], size: (u32, u32), x: f32, y: f32) -> [f32; 3] {
    pixels[(y as u32).min(size.1 - 1) as usize * size.0 as usize + (x as u32).min(size.0 - 1) as usize]
}

fn sky(frame: &mut RenderFrame) {
    let v = |x, y, z| Vector { x, y, z };
    frame.sky = Sky {
        zenith: v(0.25, 0.5, 1.0),
        horizon: v(0.8, 0.9, 1.0),
        ground: v(0.3, 0.3, 0.3),
        sun: v(0.0, 1.0, 0.0),
        tint: v(1.0, 1.0, 1.0),
        sh0: v(2.2, 2.4, 3.0),
        ..Sky::default()
    };
    frame.light_direction = v(0.3, -1.0, -0.2);
    frame.light_color = v(0.9, 0.85, 0.7);
}

// ------------------------------------------------------------------ the room

const WALLS: [(&str, [f32; 3]); 7] = [
    ("floor", [0.9, 0.9, 0.9]),
    ("red", [0.9, 0.08, 0.08]),
    ("blue", [0.08, 0.2, 0.9]),
    ("green", [0.08, 0.8, 0.12]),
    ("yellow", [0.9, 0.8, 0.08]),
    ("ceiling", [0.8, 0.8, 0.8]),
    ("box", [0.7, 0.7, 0.7]),
];

fn room() -> Vec<String> {
    let q = |c: [V3; 4]| quads(&[c]);
    vec![
        q([[-4.0, 0.0, 4.0], [4.0, 0.0, 4.0], [4.0, 0.0, -4.0], [-4.0, 0.0, -4.0]]),
        q([[-4.0, 0.0, 4.0], [-4.0, 0.0, -4.0], [-4.0, 3.0, -4.0], [-4.0, 3.0, 4.0]]),
        q([[4.0, 0.0, -4.0], [4.0, 0.0, 4.0], [4.0, 3.0, 4.0], [4.0, 3.0, -4.0]]),
        q([[-4.0, 0.0, -4.0], [4.0, 0.0, -4.0], [4.0, 3.0, -4.0], [-4.0, 3.0, -4.0]]),
        q([[4.0, 0.0, 4.0], [-4.0, 0.0, 4.0], [-4.0, 3.0, 4.0], [4.0, 3.0, 4.0]]),
        q([[-4.0, 3.0, -4.0], [4.0, 3.0, -4.0], [4.0, 3.0, 4.0], [-4.0, 3.0, 4.0]]),
    ]
}

fn draw(mesh: u32, material: u32, metallic: f32, roughness: f32) -> Draw {
    Draw { mesh, material, transform: IDENTITY, parameters: Some([metallic, roughness, 0.0, 0.0]), extra: None }
}

fn room_frame(size: (u32, u32), eye: V3, target: V3) -> RenderFrame {
    let mut frame = frame(vec![
        draw(0, 0, 1.0, 0.1),
        draw(1, 1, 0.0, 0.9),
        draw(2, 2, 0.0, 0.9),
        draw(3, 3, 0.0, 0.9),
        draw(4, 4, 0.0, 0.9),
        draw(5, 5, 0.0, 0.9),
    ]);
    (frame.view, frame.projection) = look_at(eye, target, size.0 as f32 / size.1 as f32);
    sky(&mut frame);
    frame
}

fn probe(id: u32, revision: u32) -> ReflectionProbe {
    let v = |x, y, z| Vector { x, y, z };
    ReflectionProbe {
        id,
        revision,
        priority: 0,
        capture: false,
        position: v(0.0, 1.5, 0.0),
        // The box reaches below the floor, so that the floor is inside it.
        center: v(0.0, 1.0, 0.0),
        extent: v(4.5, 2.5, 4.5),
        blend: 0.0,
        parallax_center: v(0.0, 1.5, 0.0),
        parallax_extent: v(4.0, 1.5, 4.0),
        far: 30.0,
    }
}

/// Which of the room's surfaces a ray from `from` along `d` leaves through: a color name.
fn wall_hit(from: V3, d: V3) -> (&'static str, V3) {
    let faces: [(&str, usize, f32); 6] = [
        ("red", 0, -4.0),
        ("blue", 0, 4.0),
        ("green", 2, -4.0),
        ("yellow", 2, 4.0),
        ("ceiling", 1, 3.0),
        ("floor", 1, 0.0),
    ];
    let mut best = ("none", f32::INFINITY);
    for (name, axis, plane) in faces {
        if d[axis].abs() < 1e-6 {
            continue;
        }
        let t = (plane - from[axis]) / d[axis];
        if t > 1e-4 && t < best.1 {
            best = (name, t);
        }
    }
    (best.0, [from[0] + d[0] * best.1, from[1] + d[1] * best.1, from[2] + d[2] * best.1])
}

fn dominant(p: [f32; 3], name: &str) -> bool {
    match name {
        "red" => p[0] > 1.6 * p[1].max(p[2]),
        "blue" => p[2] > 1.4 * p[0].max(p[1]),
        "green" => p[1] > 1.5 * p[0].max(p[2]),
        "yellow" => p[0].min(p[1]) > 1.8 * p[2],
        _ => false,
    }
}

#[test]
fn a_probe_cube_reflects_the_room_where_its_walls_are_and_the_budget_sets_the_frames_it_takes() {
    let size = (480, 300);
    let Some(mut scene) = Scene::new("probe-room", &room(), &WALLS, size) else { return };
    let eye = [0.5, 1.5, 3.2];
    let mut frame = room_frame(size, eye, [0.0, 0.2, -2.0]);
    scene.settle(&frame, 6);
    let before = scene.shoot(&frame, 2);
    scene.save(&before, "room-sky-only.png");
    // The floor the camera sees, on a grid, with what its reflection leaves the room through.
    let samples: Vec<(f32, f32, &str, V3)> = (0..30)
        .flat_map(|j| (0..48).map(move |i| (i, j)))
        .filter_map(|(i, j)| {
            let pixel = (i as f32 * 10.0 + 5.0, 140.0 + j as f32 * 5.0 + 2.5);
            let (point, d) = floor_point(eye, &frame.view, &frame.projection, pixel, size)?;
            if point[0].abs() > 3.9 || point[2].abs() > 3.9 {
                return None;
            }
            let (name, hit) = wall_hit(point, [d[0], -d[1], d[2]]);
            // Well inside a wall: its filtering and the cube's texel are not at an edge.
            let edge = match name {
                "red" | "blue" => hit[1].min(3.0 - hit[1]).min(4.0 - hit[2].abs()),
                "green" | "yellow" => hit[1].min(3.0 - hit[1]).min(4.0 - hit[0].abs()),
                _ => 0.0,
            };
            (edge > 0.5 && !name.eq("ceiling") && !name.eq("floor")).then_some((pixel.0, pixel.1, name, hit))
        })
        .collect();
    assert!(samples.len() > 100, "the grid should reach the walls: {}", samples.len());
    let wrong_before = samples.iter().filter(|s| dominant(at(&before, size, s.0, s.1), s.2)).count();
    // How many frames until the probe shows, at each budget: its 6 faces over `budget` a frame.
    for (budget, frames) in [(6, 1), (3, 2), (2, 3), (1, 6)] {
        let mut fresh = Scene::new("probe-room", &room(), &WALLS, size).unwrap();
        let mut f = frame.clone();
        fresh.settle(&f, 6);
        fresh.shoot(&f, 2);
        let reference = fresh.read();
        f.reflection_probes = vec![probe(1, 1)];
        f.reflection.budget = budget;
        let mut changed = None;
        for n in 1..=8 {
            let now = fresh.shoot(&f, 1);
            let differs = samples.iter().filter(|s| now_differs(&now, &reference, size, s)).count();
            if differs > samples.len() / 2 {
                changed = Some(n);
                break;
            }
        }
        assert_eq!(changed, Some(frames), "budget {budget}: faces drawn {budget} a frame show after {frames}");
        // And it stays: the frame after, with the probe held, is the same picture.
        let again = fresh.shoot(&f, 1);
        let settled = fresh.shoot(&f, 1);
        assert_eq!(again, settled, "nothing is captured again while the revision holds");
    }
    frame.reflection_probes = vec![probe(1, 1)];
    frame.reflection.budget = 6;
    let after = scene.shoot(&frame, 3);
    scene.save(&after, "room-probe.png");
    let right = samples.iter().filter(|s| dominant(at(&after, size, s.0, s.1), s.2)).count();
    eprintln!(
        "wall colours found in the floor's reflection: sky only {wrong_before}, probe {right} of {}",
        samples.len()
    );
    assert!(right as f32 > 0.9 * samples.len() as f32, "{right} of {}", samples.len());
    assert!(wrong_before < samples.len() / 2, "without the probe the floor shows the sky");
    // A new revision draws it again, and a rising flag does too: the picture is the same room.
    frame.reflection_probes[0].revision = 2;
    let revised = scene.shoot(&frame, 2);
    assert_eq!(revised, scene.shoot(&frame, 1), "an unchanged room captured again is the same cube");
    // Taking the probes away gives the sky-only picture back, exactly.
    frame.reflection_probes = Vec::new();
    assert!(scene.shoot(&frame, 1) == before, "no probes, the previous output");
}

fn now_differs(now: &[[f32; 3]], then: &[[f32; 3]], size: (u32, u32), s: &(f32, f32, &str, V3)) -> bool {
    let (a, b) = (at(now, size, s.0, s.1), at(then, size, s.0, s.1));
    (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs() > 0.08
}

#[test]
fn a_probe_is_normalised_to_the_light_it_is_shown_under() {
    let size = (240, 150);
    let Some(mut scene) = Scene::new("probe-light", &room(), &WALLS, size) else { return };
    let eye = [0.5, 1.5, 3.2];
    let mut frame = room_frame(size, eye, [0.0, 0.2, -2.0]);
    frame.reflection_probes = vec![probe(1, 1)];
    frame.reflection.budget = 6;
    scene.settle(&frame, 6);
    let bright = scene.shoot(&frame, 3);
    // The sky dims to a quarter: the captured cube is scaled by the change in the light's mean,
    // so the reflection follows the light (within the clamp) without capturing again.
    let mut dim = frame.clone();
    dim.sky.sh0 = Vector { x: 0.55, y: 0.6, z: 0.75 };
    dim.light_color = Vector { x: 0.225, y: 0.2125, z: 0.175 };
    let dimmer = scene.shoot(&dim, 2);
    // The floor's reflection of the walls, against the walls themselves (the picture above
    // the floor): the light fell on both, so their ratio holds.
    let luma = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let region = |p: &[[f32; 3]], rows: std::ops::Range<u32>| {
        let n = rows.len() as f32 * size.0 as f32;
        rows.flat_map(|y| (0..size.0).map(move |x| (x, y)))
            .map(|(x, y)| luma(p[(y * size.0 + x) as usize]))
            .sum::<f32>()
            / n
    };
    let (floor_rows, wall_rows) = (100..150, 5..50);
    let ratio = |p: &[[f32; 3]]| region(p, floor_rows.clone()) / region(p, wall_rows.clone());
    scene.save(&dimmer, "room-probe-dimmed.png");
    eprintln!("floor / wall light {} -> {}", ratio(&bright), ratio(&dimmer));
    assert!(region(&dimmer, wall_rows.clone()) < 0.75 * region(&bright, wall_rows.clone()), "the light dimmed");
    assert!((ratio(&dimmer) / ratio(&bright) - 1.0).abs() < 0.3, "the reflection followed the light");
}

// -------------------------------------------------- screen-space reflections

fn mirror_scene(name: &str, size: (u32, u32), box_at: V3) -> Option<Scene> {
    let floor = quads(&[[[-30.0, 0.0, 30.0], [30.0, 0.0, 30.0], [30.0, 0.0, -30.0], [-30.0, 0.0, -30.0]]]);
    let block = quads(&cuboid([box_at[0] - 0.6, 0.0, box_at[2] - 0.6], [box_at[0] + 0.6, 1.2, box_at[2] + 0.6]));
    // A one-sided red table top at height 1.2, seen from above: a ray from below meets its back.
    let table = quads(&[[[-3.0, 1.2, -1.0], [-3.0, 1.2, 1.0], [-1.0, 1.2, 1.0], [-1.0, 1.2, -1.0]]]);
    Scene::new(name, &[floor, block, table], &[("floor", [0.9, 0.9, 0.9]), ("red", [0.95, 0.05, 0.05])], size)
}

fn mirror_frame(size: (u32, u32), roughness: f32, ssr: u32, with_table: bool) -> (RenderFrame, V3) {
    let eye = [0.0, 2.2, 4.0];
    let mut draws = vec![draw(0, 0, 1.0, roughness), draw(1, 1, 0.0, 0.9)];
    if with_table {
        draws.push(draw(2, 1, 0.0, 0.9));
    }
    let mut frame = frame(draws);
    (frame.view, frame.projection) = look_at(eye, [0.0, 0.5, -3.0], size.0 as f32 / size.1 as f32);
    sky(&mut frame);
    frame.quality.ssr = ssr;
    (frame, eye)
}

/// The red block's reflection in the floor: its mirror image's middle on the screen.
fn mirror_pixel(frame: &RenderFrame, size: (u32, u32), box_at: V3) -> (f32, f32) {
    project(&frame.view, &frame.projection, [box_at[0], -0.6, box_at[2]], size)
}

fn redness(p: [f32; 3]) -> f32 {
    p[0] - p[1].max(p[2])
}

#[test]
fn a_mirror_floor_reflects_the_block_where_the_mirror_puts_it() {
    let size = (480, 300);
    let box_at = [0.0, 0.0, -3.0];
    let Some(mut scene) = mirror_scene("mirror", size, box_at) else { return };
    let (off, _) = mirror_frame(size, 0.04, 0, false);
    scene.settle(&off, 2);
    let without = scene.shoot(&off, 2);
    scene.save(&without, "mirror-off.png");
    let (on, _) = mirror_frame(size, 0.04, 2, false);
    // TAA needs frames to gather its history and for the picture to converge.
    let with = scene.shoot(&on, 24);
    scene.save(&with, "mirror-ssr.png");
    let (px, py) = mirror_pixel(&on, size, box_at);
    eprintln!("the block's mirror image is at pixel ({px:.1}, {py:.1})");
    let (a, b) = (at(&without, size, px, py), at(&with, size, px, py));
    eprintln!("without {a:?} with {b:?}");
    assert!(redness(b) > 0.25 && redness(a) < 0.05, "the block shows red in the mirror only with the march");
    // Turned off again, the previous picture returns exactly.
    let again = scene.shoot(&off, 2);
    assert!(again == without, "no march, the previous output");
    let with = scene.shoot(&on, 24);
    // The reflection ends where the mirror image does: well above and below it, and left and
    // right of it, the floor is what it was.
    for (dx, dy) in [(0.0, -70.0), (-90.0, 0.0), (90.0, 0.0), (0.0, 90.0)] {
        let p = at(&with, size, px + dx, py + dy);
        assert!(redness(p) < 0.12, "({dx}, {dy}): {p:?}");
    }
    // The block's own pixel row: its image is as wide as the block's mirror image, within a few pixels.
    let (left, _) = project(&on.view, &on.projection, [-0.6, -0.6, -3.6], size);
    let (right, _) = project(&on.view, &on.projection, [0.6, -0.6, -3.6], size);
    let row: Vec<f32> = (0..size.0).map(|x| redness(at(&with, size, x as f32, py))).collect();
    let first = row.iter().position(|&r| r > 0.2).unwrap() as f32;
    let last = row.iter().rposition(|&r| r > 0.2).unwrap() as f32;
    eprintln!("the red spans {first}..{last}, the analytic edges {left:.1}..{right:.1} (z = -3.6, the far face)");
    assert!(first >= left.min(right) - 8.0 && last <= left.max(right) + 8.0);
    // A static camera converges: the last frames differ by the noise's last grain only.
    let more = scene.shoot(&on, 1);
    let still = scene.shoot(&on, 1);
    let diff: f32 = more
        .iter()
        .zip(&still)
        .map(|(a, b)| (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs())
        .sum::<f32>()
        / more.len() as f32;
    eprintln!("mean difference between consecutive converged frames: {diff:.5}");
    assert!(diff < 0.01, "a still camera does not shimmer: {diff}");
}

#[test]
fn reflections_fade_with_roughness_and_at_the_screens_edge() {
    let size = (480, 300);
    let box_at = [0.0, 0.0, -3.0];
    let Some(mut scene) = mirror_scene("fade", size, box_at) else { return };
    let (probe_frame, _) = mirror_frame(size, 0.04, 2, false);
    let (px, py) = mirror_pixel(&probe_frame, size, box_at);
    scene.settle(&probe_frame, 2);
    let mut strengths = Vec::new();
    for roughness in [0.04, 0.3, 0.5, 0.8] {
        let (on, _) = mirror_frame(size, roughness, 2, false);
        let (off, _) = mirror_frame(size, roughness, 0, false);
        scene.shoot(&off, 1);
        let base = scene.shoot(&off, 1);
        let lit = scene.shoot(&on, 24);
        scene.save(&lit, &format!("mirror-roughness-{roughness}.png"));
        let (a, b) = (at(&base, size, px, py), at(&lit, size, px, py));
        strengths.push(redness(b) - redness(a));
        if roughness >= 0.8 {
            // TAA is on with the march, so edges move; the open floor holds.
            let floor = (150 * size.0 as usize..size.0 as usize * size.1 as usize).step_by(3);
            let (sum, n) = floor.fold((0.0f32, 0u32), |(sum, n), i| {
                let (c, d) = (base[i], lit[i]);
                (sum + (c[0] - d[0]).abs() + (c[1] - d[1]).abs() + (c[2] - d[2]).abs(), n + 1)
            });
            assert!(sum / (n as f32) < 0.03, "a rough floor has no screen-space reflection: {}", sum / n as f32);
        }
    }
    eprintln!("red added by the march at roughness 0.04, 0.3, 0.5, 0.8: {strengths:?}");
    assert!(strengths[0] > strengths[1] && strengths[1] > strengths[2] && strengths[2] > strengths[3] - 0.02);
    assert!(strengths[0] > 0.25 && strengths[3].abs() < 0.05);
    // Toward the screen's edge: the same block, its image moved to the right border.
    // Found by halving: the block's front face, where the ray meets it, a few pixels from the border.
    let (frame_for_edge, _) = mirror_frame(size, 0.04, 2, false);
    let (mut low, mut high) = (0.0f32, 12.0f32);
    for _ in 0..30 {
        let middle = 0.5 * (low + high);
        let (x, _) = project(&frame_for_edge.view, &frame_for_edge.projection, [middle, 0.6, -2.4], size);
        if x < size.0 as f32 - 1.5 { low = middle } else { high = middle }
    }
    let near_edge = [low, 0.0, -3.0];
    let Some(mut edge) = mirror_scene("fade-edge", size, near_edge) else { return };
    let (on, _) = mirror_frame(size, 0.04, 2, false);
    let (off, _) = mirror_frame(size, 0.04, 0, false);
    edge.settle(&off, 2);
    let (ex, ey) = mirror_pixel(&on, size, near_edge);
    eprintln!("edge image at ({ex:.1}, {ey:.1}) of {}x{}", size.0, size.1);
    let base = edge.shoot(&off, 2);
    let lit = edge.shoot(&on, 24);
    edge.save(&lit, "mirror-edge.png");
    // Along the block's image toward the border, where its front face (the hit) nears the edge.
    let added = |x: f32| redness(at(&lit, size, x, ey)) - redness(at(&base, size, x, ey));
    let (inner, outer) = (added(ex), added(size.0 as f32 - 3.0));
    eprintln!("red added at x = {ex:.0}: {inner:.3}, at the border: {outer:.3}; in the middle {:.3}", strengths[0]);
    assert!(inner < 0.8 * strengths[0] && outer < 0.25 * inner.max(0.2), "the march fades toward the screen's edge");
}

#[test]
fn rays_that_meet_the_back_of_a_surface_hit_nothing() {
    let size = (480, 300);
    let box_at = [4.0, 0.0, -8.0];
    let Some(mut scene) = mirror_scene("backs", size, box_at) else { return };
    let (on, eye) = mirror_frame(size, 0.04, 2, true);
    let (off, _) = mirror_frame(size, 0.04, 0, true);
    scene.settle(&off, 3);
    let base = scene.shoot(&off, 2);
    let lit = scene.shoot(&on, 24);
    scene.save(&lit, "mirror-table-ssr.png");
    scene.save(&base, "mirror-table-off.png");
    // Floor points whose mirror ray crosses the table top (red, facing up, at height 1.2) from
    // below: that surface's back as the ray meets it.
    let mut crossings = 0;
    let mut red = 0;
    for j in 0..300 {
        for i in 0..480 {
            let pixel = (i as f32 + 0.5, j as f32 + 0.5);
            let Some((point, d)) = floor_point(eye, &on.view, &on.projection, pixel, size) else { continue };
            let ray = [d[0], -d[1], d[2]];
            let t = 1.2 / ray[1];
            let hit = [point[0] + ray[0] * t, 1.2, point[2] + ray[2] * t];
            let inside = hit[0] > -2.9 && hit[0] < -1.1 && hit[2] > -0.9 && hit[2] < 0.9;
            // The floor must be in the open (not under the table, where the table hides it) and
            // the ray must not first meet the table's front: it comes from below.
            let hidden = point[0] > -3.0 && point[0] < -1.0 && point[2] > -1.0 && point[2] < 1.0;
            if inside && !hidden && (t < 12.0) {
                crossings += 1;
                if redness(at(&lit, size, pixel.0, pixel.1)) > 0.2 {
                    red += 1;
                }
            }
        }
    }
    eprintln!("{crossings} floor pixels' rays cross the table top from below; {red} show it");
    assert!(crossings > 200, "the setup has such rays: {crossings}");
    assert!(red * 50 < crossings, "back faces never reflect: {red} of {crossings}");
}

/// The glossy room with an orange block on its floor, seen from near a corner.
fn glossy_room(name: &str, size: (u32, u32), ceiling_casts: bool) -> Option<(Scene, RenderFrame)> {
    let mut meshes = room();
    meshes.push(quads(&cuboid([0.6, 0.0, -1.6], [1.8, 1.3, -0.4])));
    let mut materials = WALLS.to_vec();
    if !ceiling_casts {
        materials[5].0 = "ceiling_nocast";
    }
    materials.push(("orange", [0.95, 0.4, 0.05]));
    let scene = Scene::new(name, &meshes, &materials, size)?;
    let mut frame = room_frame(size, [-0.8, 1.7, 3.3], [0.3, 0.5, -2.0]);
    frame.draws.push(draw(6, 7, 0.0, 0.9));
    Some((scene, frame))
}

#[test]
fn the_glossy_room_with_the_sky_alone_with_probes_and_with_the_march_too() {
    let size = (800, 500);
    let Some((mut scene, mut frame)) = glossy_room("room-shots", size, true) else { return };
    scene.settle(&frame, 7);
    let sky_only = scene.shoot(&frame, 3);
    scene.save(&sky_only, "room-1-sky-only.png");
    frame.reflection_probes = vec![probe(1, 1)];
    frame.reflection.budget = 6;
    let probes = scene.shoot(&frame, 4);
    scene.save(&probes, "room-2-probes.png");
    frame.quality.ssr = 2;
    let both = scene.shoot(&frame, 28);
    scene.save(&both, "room-3-probes-and-ssr.png");
    // The orange block's reflection: the march puts it where the mirror does.
    let (px, py) = project(&frame.view, &frame.projection, [1.2, -0.65, -1.0], size);
    let orange = |p: [f32; 3]| p[0] - p[2] > 0.5 && p[1] > 0.2 && p[0] > p[1] * 1.4;
    eprintln!(
        "block's mirror image at ({px:.0}, {py:.0}): sky {:?} probes {:?} both {:?}",
        at(&sky_only, size, px, py),
        at(&probes, size, px, py),
        at(&both, size, px, py)
    );
    assert!(orange(at(&both, size, px, py)), "orange in the mirror");
    assert!(!orange(at(&probes, size, px, py)) && !orange(at(&sky_only, size, px, py)));
}

/// The mean absolute second difference down each column of `rows`: a smooth gradient has
/// none, stripes have a great deal.
fn vertical_curvature(
    pixels: &[[f32; 3]],
    width: u32,
    columns: std::ops::Range<u32>,
    rows: std::ops::Range<u32>,
) -> f32 {
    let (mut sum, mut n) = (0.0, 0u32);
    for x in columns {
        for y in rows.start + 1..rows.end - 1 {
            let at = |y: u32| pixels[(y * width + x) as usize];
            let (a, b, c) = (at(y - 1), at(y), at(y + 1));
            sum += (0..3).map(|k| (a[k] - 2.0 * b[k] + c[k]).abs()).sum::<f32>();
            n += 1;
        }
    }
    sum / n as f32
}

fn mean_difference(
    a: &[[f32; 3]],
    b: &[[f32; 3]],
    width: u32,
    columns: std::ops::Range<u32>,
    rows: std::ops::Range<u32>,
) -> f32 {
    let (mut sum, mut n) = (0.0, 0u32);
    for y in rows {
        for x in columns.clone() {
            let (p, q) = (a[(y * width + x) as usize], b[(y * width + x) as usize]);
            sum += (p[0] - q[0]).abs() + (p[1] - q[1]).abs() + (p[2] - q[2]).abs();
            n += 1;
        }
    }
    sum / n as f32
}

// A march with a coarse step (a surface seen at a grazing angle, like the ceiling from the
// near floor, changes its depth by more than the thickness from one step to the next) once
// missed hits in stripes across the near floor and left saw-toothed edges where the wall
// reflections end. The near floor reflecting the ceiling and the far wall must be smooth,
// and where the march has nothing to hit the probe's reflection must remain.
#[test]
fn the_march_leaves_the_near_floor_smooth_and_falls_back_to_the_probe() {
    let size = (800, 500);
    // The near floor in the middle of the frame, below the block's reflection, and the
    // whole width of it.
    let (middle, wide) = ((250..470, 410..496), (150..650, 400..496));
    for ceiling_casts in [true, false] {
        let Some((mut scene, mut frame)) = glossy_room("room-smooth", size, ceiling_casts) else { return };
        frame.reflection_probes = vec![probe(1, 1)];
        frame.reflection.budget = 6;
        scene.settle(&frame, 7);
        let probes_only = scene.shoot(&frame, 4);
        frame.quality.ssr = 2;
        let marched = scene.shoot(&frame, 40);
        scene.save(&marched, if ceiling_casts { "room-smooth-ssr.png" } else { "room-smooth-no-ceiling-hits.png" });
        let curvature = vertical_curvature(&marched, size.0, middle.0.clone(), middle.1.clone());
        let reference = vertical_curvature(&probes_only, size.0, middle.0.clone(), middle.1.clone());
        let difference = mean_difference(&marched, &probes_only, size.0, middle.0.clone(), middle.1.clone());
        let across = mean_difference(&marched, &probes_only, size.0, wide.0.clone(), wide.1.clone());
        eprintln!(
            "ceiling casts {ceiling_casts}: curvature {curvature:.5} (probes {reference:.5}), difference {difference:.5}, across {across:.5}"
        );
        assert!(curvature < reference + 0.0015, "stripes: {curvature} against {reference}");
        if ceiling_casts {
            // The march hits the ceiling and the far wall: their own light, to the cube's blur.
            assert!(difference < 0.06 && across < 0.08, "{difference} {across}");
        } else {
            // Nothing to hit but sky past the screen: the probe's reflection, whole.
            assert!(difference < 0.012 && across < 0.03, "{difference} {across}");
        }
        // Edges hold after convergence: a frame later and a jitter's cycle later the floor's
        // reflections have not moved.
        let next = scene.shoot(&frame, 1);
        let later = scene.shoot(&frame, 8);
        let rows = 300..496;
        let step = mean_difference(&marched, &next, size.0, 0..size.0, rows.clone());
        let cycle = mean_difference(&marched, &later, size.0, 0..size.0, rows);
        eprintln!("floor between frames: {step:.5}, a cycle later {cycle:.5}");
        assert!(step < 0.004 && cycle < 0.004, "reflection edges shimmer: {step} {cycle}");
    }
}
