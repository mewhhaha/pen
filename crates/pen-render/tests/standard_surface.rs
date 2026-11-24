//! The standard surface through the whole renderer on a real (headless) GPU:
//! a row of materials under the sky, the sun and a light (clear coat, sheen,
//! anisotropy, subsurface, and glass over a stand-in picture), the base lobe
//! rendered bit for bit as the functions it replaced drew it, the stretch of
//! an anisotropic highlight against the f64 reference, and a brick wall seen
//! across its face with and without parallax. Pictures go to
//! /tmp/fidelity/materials.
//!
//! The material's `surface` block is not written by the host yet (pen-gltf
//! and the draw uniform's tail are another workstream's); these shaders read
//! the same five vectors from the draw's `extra` vectors instead, then a sixth
//! with the lobe flags: `extra[5] = (flags, subsurface, thin, unused)` and the
//! subsurface color in `extra[6]`.
#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::cast_sign_loss, reason = "pixels in range")]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    reason = "test geometry"
)]
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

mod common;

use common::*;
use pen_protocol::*;
use pen_render::{Display, RenderTarget, Renderer, RendererOptions};

fn shots() -> PathBuf {
    let directory = PathBuf::from("/tmp/fidelity/materials");
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

fn assets(name: &str, files: &[(&str, &[u8])]) -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("surface-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    for (key, bytes) in files {
        std::fs::write(root.join(key), bytes).unwrap();
    }
    root
}

struct Target(wgpu::Texture);

impl Target {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        Target(device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }))
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

/// A rendered picture: red, green, blue of every pixel, row by row.
#[derive(Clone, PartialEq)]
struct Picture {
    width: u32,
    height: u32,
    rgb: Vec<[u8; 3]>,
}

impl Picture {
    fn at(&self, x: u32, y: u32) -> [u8; 3] {
        self.rgb[(y * self.width + x) as usize]
    }

    fn save(&self, name: &str) {
        let image = image::RgbImage::from_fn(self.width, self.height, |x, y| image::Rgb(self.at(x, y)));
        image.save(shots().join(name)).unwrap();
    }
}

struct Scene {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    catalog: Catalog,
    target: Target,
}

impl Scene {
    fn new(name: &str, size: (u32, u32), files: &[(&str, &[u8])]) -> Option<Scene> {
        let (device, queue) = device()?;
        let root = assets(name, files);
        let catalog = boundary::read_catalog(&root).unwrap();
        let options =
            RendererOptions { format: FORMAT, display: Display::SDR, assets: root, veil: None, pipeline_cache: None };
        let renderer = Renderer::new(device.clone(), queue.clone(), options).unwrap();
        let target = Target::new(&device, size.0, size.1);
        Some(Scene { device, queue, renderer, catalog, target })
    }

    fn draw(&mut self, frame: &RenderFrame) -> Picture {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            self.renderer.submit(frame, &self.catalog, &mut self.target).unwrap();
            let events = self.renderer.take_events();
            for event in &events {
                assert!(!matches!(event, GameEvent::AssetFailed { .. }), "an asset failed: {event:?}");
            }
            if self.renderer.pending() == 0 && self.renderer.stats().draw_calls >= frame.draws.len() as u32 {
                break;
            }
            assert!(Instant::now() < deadline, "assets did not load");
            std::thread::sleep(Duration::from_millis(5));
        }
        let (width, height) = (self.target.0.width(), self.target.0.height());
        let rgba = pixels(&self.device, &self.queue, &self.target.0);
        Picture { width, height, rgb: rgba.iter().map(|p| [p[0], p[1], p[2]]).collect() }
    }
}

/// A unit sphere's mesh: `rings` by `segments`, uv around and down.
fn sphere(rings: usize, segments: usize) -> String {
    let (mut positions, mut uvs, mut indices) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..=rings {
        let polar = std::f64::consts::PI * r as f64 / rings as f64;
        for s in 0..=segments {
            let azimuth = std::f64::consts::TAU * s as f64 / segments as f64;
            positions.extend([polar.sin() * azimuth.cos(), polar.cos(), polar.sin() * azimuth.sin()]);
            uvs.extend([s as f64 / segments as f64, r as f64 / rings as f64]);
        }
    }
    for r in 0..rings {
        for s in 0..segments {
            let a = (r * (segments + 1) + s) as u32;
            let b = a + segments as u32 + 1;
            indices.extend([a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    let list = |v: &[f64]| v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(",");
    // A unit sphere's normals are its positions.
    format!(
        r#"{{"positions":[{0}],"normals":[{0}],"uvs":[{1}],"indices":{2:?}}}"#,
        list(&positions),
        list(&uvs),
        indices
    )
}

/// A quad of `size` half-width at height `y`, facing up, whose uv run `repeat` times.
fn floor(size: f32, y: f32, repeat: f32) -> String {
    format!(
        r#"{{"positions":[-{s},{y},-{s}, {s},{y},-{s}, -{s},{y},{s}, {s},{y},{s}],"normals":[0,1,0,0,1,0,0,1,0,0,1,0],
            "uvs":[0,0,{r},0,0,{r},{r},{r}],"indices":[0,2,1,1,2,3]}}"#,
        s = size,
        y = y,
        r = repeat
    )
}

/// A quad facing +z at depth `z`, half-width `size`.
fn wall(size: f32, z: f32) -> String {
    format!(
        r#"{{"positions":[-{s},-{s},{z}, {s},-{s},{z}, -{s},{s},{z}, {s},{s},{z}],"normals":[0,0,1,0,0,1,0,0,1,0,0,1],
            "uvs":[0,0,1,0,0,1,1,1],"indices":[0,1,2,2,1,3]}}"#,
        s = size,
        z = z
    )
}

fn translate(x: f32, y: f32, z: f32, scale: f32) -> Matrix {
    [scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, x, y, z, 1.0]
}

/// A look-at camera from `eye` to `target` and a perspective projection into
/// WebGPU's 0..1 depth, with a vertical field of `fov` radians' tangent-half-angle `0.6`.
fn camera(eye: [f64; 3], target: [f64; 3], aspect: f64) -> (Matrix, Matrix, [[f64; 3]; 3]) {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross =
        |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let unit = |a: [f64; 3]| {
        let l = dot(a, a).sqrt();
        [a[0] / l, a[1] / l, a[2] / l]
    };
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
    ]
    .map(|x| x as f32);
    let (near, far, f) = (0.1f64, 100.0f64, 1.0 / 0.6f64.tan());
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
    ]
    .map(|x| x as f32);
    (view, projection, [right, up, back])
}

fn vector(x: f32, y: f32, z: f32) -> Vector {
    Vector { x, y, z }
}

/// A sky of a gradient and the diffuse light of one that is brightest overhead.
fn sky() -> Sky {
    let scale = 1.0 / 0.282_094_8;
    Sky {
        zenith: vector(0.25, 0.45, 1.0),
        horizon: vector(0.85, 0.8, 0.75),
        ground: vector(0.12, 0.1, 0.09),
        sun: vector(0.4, 0.8, 0.45),
        tint: vector(1.0, 1.0, 1.0),
        sh0: vector(0.42 * scale, 0.46 * scale, 0.55 * scale),
        sh1: vector(0.2, 0.23, 0.3),
        ..Sky::default()
    }
}

fn point_light(x: f32, y: f32, z: f32, radius: f32, color: [f32; 3]) -> Light {
    Light {
        position: vector(x, y, z),
        radius,
        color: vector(color[0], color[1], color[2]),
        direction: vector(0.0, -1.0, 0.0),
        inner: -1.0,
        outer: -1.0,
        shadow: false,
        id: 0,
    }
}

/// The lobes' flags, as `surface_lobes.wgsl` has them.
const COAT: f32 = 1.0;
const SHEEN: f32 = 2.0;
const ANISOTROPY: f32 = 4.0;
const TRANSMISSION: f32 = 8.0;
const SUBSURFACE: f32 = 16.0;

/// A draw's `extra` as the shaders below read it: the surface block's five
/// vectors, `(flags, subsurface, thin, 0)` and the subsurface's color.
#[derive(Clone, Copy, Default)]
struct Look {
    flags: f32,
    coat: f32,
    coat_roughness: f32,
    sheen_roughness: f32,
    anisotropy: f32,
    sheen_color: [f32; 3],
    rotation: f32,
    transmission: f32,
    ior: f32,
    thickness: f32,
    attenuation_distance: f32,
    attenuation_color: [f32; 3],
    subsurface: f32,
    subsurface_color: [f32; 3],
    thin: f32,
}

impl Look {
    fn extra(&self) -> Vec<f32> {
        let c = |v: [f32; 3], w: f32| [v[0], v[1], v[2], w];
        [
            [self.coat, self.coat_roughness, self.sheen_roughness, self.anisotropy],
            c(self.sheen_color, self.rotation),
            [self.transmission, self.ior.max(1.0), self.thickness, self.attenuation_distance],
            c(self.attenuation_color, 1.0),
            [0.0, 0.0, 0.0, 1.0],
            [self.flags, self.subsurface, self.thin, 0.0],
            c(self.subsurface_color, 0.0),
        ]
        .concat()
    }
}

// Spheres lit as standard surfaces: the draw's color is the base color and its
// parameters metallic and roughness; the lobes and what they need come from
// `extra` (see the header). The brushed direction runs round the y axis.
const SPHERES: &str = r#"
struct Scene {
  view: mat4x4f, projection: mat4x4f, ambient: vec4f, light_direction: vec4f, light_color: vec4f,
  light_view_projection: mat4x4f, shadow: vec4f, zenith: vec4f, horizon: vec4f, ground: vec4f, eye: vec4f,
};
struct Draw {
  model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f,
  extra: array<vec4f, 32>, surface: array<vec4f, 5>,
};
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
${lights}
${environment}
${surface}
${stand_in}
${transmission}
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
};
@vertex fn vertex(@location(0) position: vec3f, @location(1) normal: vec3f) -> Corner {
  let world = draw.model * vec4f(position, 1.0);
  return Corner(scene.projection * scene.view * world, world.xyz, (draw.normal * vec4f(normal, 0.0)).xyz);
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let normal = normalize(corner.normal);
  let view = normalize(scene.eye.xyz - corner.world);
  let seen = scene.view * vec4f(corner.world, 1.0);
  let block = array<vec4f, 5>(draw.extra[0], draw.extra[1], draw.extra[2], draw.extra[3], draw.extra[4]);
  let lobes = u32(draw.extra[5].x);
  var s = surface_default(corner.world, normal, view, draw.color.rgb, draw.parameters.x, draw.parameters.y, corner.position, seen.z);
  s = surface_apply_block(s, block, lobes, 1.0);
  s.subsurface = draw.extra[5].y;
  s.thin = draw.extra[5].z;
  s.subsurface_color = draw.extra[6].xyz;
  if ((lobes & SURFACE_ANISOTROPY) != 0u) {
    s = surface_frame(s, cross(vec3f(0.0, 1.0, 0.0), normal), 1.0, draw.extra[1].w);
  }
  ${transmit}
  let light = standard_surface(s) + standard_sun(s, scene.light_direction.xyz, scene.light_color.rgb, 1.0);
  return vec4f(light, draw.color.a);
}
"#;

// A checkered wall in screen space: what the glass's stand-in picture shows
// where nothing else has been drawn (`opaque_behind` evaluates the same
// pattern at the uv it is asked for).
const CHECKER: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
fn checker(uv: vec2f) -> vec3f {
  let cell = floor(uv * vec2f(18.0, 12.0));
  let odd = (i32(cell.x) + i32(cell.y)) & 1;
  return select(vec3f(0.55, 0.32, 0.22), vec3f(0.2, 0.3, 0.5), odd == 1);
}
struct Corner { @builtin(position) position: vec4f };
@vertex fn vertex(@location(0) position: vec3f) -> Corner {
  return Corner(scene.projection * scene.view * draw.model * vec4f(position, 1.0));
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  return vec4f(checker(corner.position.xy / draw.parameters.xy), 1.0);
}
"#;

/// The translucency library's functions as this test stands them in: the
/// picture behind is the checkered wall's pattern, which the wall's shader
/// draws at the same screen uv.
const STAND_IN: &str = r#"
fn checker(uv: vec2f) -> vec3f {
  let cell = floor(uv * vec2f(18.0, 12.0));
  let odd = (i32(cell.x) + i32(cell.y)) & 1;
  return select(vec3f(0.55, 0.32, 0.22), vec3f(0.2, 0.3, 0.5), odd == 1);
}
fn opaque_available() -> bool { return true; }
fn opaque_levels() -> f32 { return 8.0; }
fn opaque_world_uv(world: vec3f) -> vec2f {
  let clip = scene.projection * scene.view * vec4f(world, 1.0);
  return vec2f(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
}
fn opaque_behind(uv: vec2f, level: f32) -> vec3f {
  return mix(checker(clamp(uv, vec2f(0.0), vec2f(0.9999))), vec3f(0.42, 0.3, 0.3), clamp(level / 8.0 * 1.4, 0.0, 1.0));
}
"#;

fn spheres_shader(with_transmission: bool) -> String {
    SPHERES
        .replace("${lights}", pen_render::LIGHTS_WGSL)
        .replace("${environment}", pen_render::ENVIRONMENT_WGSL)
        .replace("${surface}", &format!("{}\n{}", pen_render::SURFACE_LOBES_WGSL, pen_render::SURFACE_WGSL))
        .replace("${stand_in}", if with_transmission { STAND_IN } else { "" })
        .replace("${transmission}", if with_transmission { pen_render::SURFACE_TRANSMISSION_WGSL } else { "" })
        .replace("${transmit}", if with_transmission { "s = surface_transmit(s);" } else { "" })
}

/// The six materials of the sheet, left to right: name, base color, metallic and roughness, look.
fn sheet() -> [(&'static str, [f32; 4], [f32; 2], Look); 6] {
    [
        ("base", [0.75, 0.2, 0.15, 1.0], [0.0, 0.45], Look::default()),
        (
            "clear coat",
            [0.5, 0.04, 0.06, 1.0],
            [0.0, 0.5],
            Look { flags: COAT, coat: 1.0, coat_roughness: 0.04, ..Look::default() },
        ),
        (
            "sheen",
            [0.12, 0.05, 0.2, 1.0],
            [0.0, 0.9],
            Look { flags: SHEEN, sheen_color: [0.9, 0.7, 1.0], sheen_roughness: 0.45, ..Look::default() },
        ),
        (
            "anisotropy",
            [0.9, 0.7, 0.4, 1.0],
            [1.0, 0.35],
            Look { flags: ANISOTROPY, anisotropy: 0.85, rotation: 0.0, ..Look::default() },
        ),
        (
            "subsurface",
            [0.75, 0.62, 0.2, 1.0],
            [0.0, 0.7],
            Look {
                flags: SUBSURFACE,
                subsurface: 0.9,
                subsurface_color: [1.0, 0.45, 0.1],
                thin: 0.15,
                ..Look::default()
            },
        ),
        (
            "glass (over a stand-in picture)",
            [0.9, 0.97, 1.0, 1.0],
            [0.0, 0.05],
            Look {
                flags: TRANSMISSION,
                transmission: 1.0,
                ior: 1.5,
                thickness: 0.8,
                attenuation_distance: 2.5,
                attenuation_color: [0.7, 0.95, 0.85],
                ..Look::default()
            },
        ),
    ]
}

const SHEET_SIZE: (f64, f64) = (900.0, 600.0);
const EYE: [f64; 3] = [0.0, 0.0, 7.5];

/// Where the sphere `index` of the sheet is: three to a row, two rows.
fn sphere_world(index: usize) -> [f64; 3] {
    [(index % 3) as f64 * 2.7 - 2.7, if index < 3 { 1.45 } else { -1.45 }, 0.0]
}

/// The sheet's frame, with the lobes of the spheres in `plain` switched off and the glass left out when `glass` is false.
fn sheet_frame(plain: &[usize], glass: bool) -> RenderFrame {
    let (view, projection, _) = camera(EYE, [0.0, 0.0, 0.0], SHEET_SIZE.0 / SHEET_SIZE.1);
    let mut draws = vec![Draw {
        mesh: 1,
        material: 6,
        transform: IDENTITY,
        parameters: Some([SHEET_SIZE.0 as f32, SHEET_SIZE.1 as f32, 0.0, 0.0]),
        extra: None,
    }];
    for (index, (_, _, parameters, look)) in sheet().iter().enumerate() {
        if index == 5 && !glass {
            continue;
        }
        let look = if plain.contains(&index) { Look { flags: 0.0, ..*look } } else { *look };
        let at = sphere_world(index);
        draws.push(Draw {
            mesh: 0,
            material: index as u32,
            transform: translate(at[0] as f32, at[1] as f32, at[2] as f32, 1.0),
            parameters: Some([parameters[0], parameters[1], 0.0, 0.0]),
            extra: Some(look.extra()),
        });
    }
    let mut frame = frame(draws);
    (frame.view, frame.projection) = (view, projection);
    frame.quality.sky = true;
    frame.sky = sky();
    frame.light_direction = vector(-0.45, -0.6, -0.65);
    frame.light_color = vector(1.5, 1.4, 1.2);
    frame.lights =
        vec![point_light(-2.0, 0.2, 3.5, 9.0, [1.8, 1.3, 0.9]), point_light(3.5, -0.4, 3.0, 8.0, [0.3, 0.5, 1.1])];
    frame
}

/// Where the sphere `index` is on its picture, in pixels, and its radius there.
fn sphere_disc(index: usize) -> (f64, f64, f64) {
    let (view, projection, _) = camera(EYE, [0.0, 0.0, 0.0], SHEET_SIZE.0 / SHEET_SIZE.1);
    let clip = |m: &Matrix, v: [f64; 4]| -> [f64; 4] {
        std::array::from_fn(|row| (0..4).map(|k| f64::from(m[k * 4 + row]) * v[k]).sum())
    };
    let at = sphere_world(index);
    let c = clip(&projection, clip(&view, [at[0], at[1], at[2], 1.0]));
    let distance = (at[0] - EYE[0]).hypot(at[1] - EYE[1]).hypot(at[2] - EYE[2]);
    // A unit sphere's radius over its distance, through the projection's scale.
    let radius = 1.0 / (distance * distance - 1.0).sqrt() * (1.0 / 0.6f64.tan()) * SHEET_SIZE.1 / 2.0;
    ((c[0] / c[3] * 0.5 + 0.5) * SHEET_SIZE.0, (0.5 - c[1] / c[3] * 0.5) * SHEET_SIZE.1, radius)
}

/// The brightest green on the sphere's disc, and the mean of red, green and blue over the part of its disc between `inner` and
/// `outer` of its radius that `keep` says to count (it is given the offset from the middle in screen units, y up).
fn disc(picture: &Picture, index: usize, inner: f64, outer: f64, keep: impl Fn(f64, f64) -> bool) -> (u8, f64) {
    let (cx, cy, r) = sphere_disc(index);
    let (mut sum, mut count, mut best) = (0.0, 0.0, 0);
    for y in 0..picture.height {
        for x in 0..picture.width {
            let (dx, dy) = (f64::from(x) - cx, cy - f64::from(y));
            let d = dx.hypot(dy) / r;
            if (inner..=outer).contains(&d) && keep(dx / r, dy / r) {
                let [red, green, blue] = picture.at(x, y);
                sum += (f64::from(red) + f64::from(green) + f64::from(blue)) / 3.0;
                count += 1.0;
                best = best.max(green);
            }
        }
    }
    (best, sum / count)
}

#[test]
fn a_row_of_materials_under_the_sky_the_sun_and_a_light() {
    let spheres = sphere(48, 96);
    let mut files: Vec<(String, Vec<u8>)> = vec![
        ("sphere.mesh.json".into(), spheres.into_bytes()),
        ("wall.mesh.json".into(), wall(40.0, -4.0).into_bytes()),
        ("standard.wgsl".into(), spheres_shader(false).into_bytes()),
        ("glass.wgsl".into(), spheres_shader(true).into_bytes()),
        ("checker.wgsl".into(), CHECKER.as_bytes().to_vec()),
        ("checker.material.json".into(), br#"{"shader":"checker.wgsl","color":[1,1,1,1],"lighting":"unlit"}"#.to_vec()),
    ];
    let mut materials = Vec::new();
    for (index, (_, color, _, _)) in sheet().iter().enumerate() {
        let shader = if index == 5 { "glass.wgsl" } else { "standard.wgsl" };
        files.push((
            format!("look{index}.material.json"),
            format!(r#"{{"shader":"{shader}","color":{color:?},"lighting":"unlit"}}"#).into_bytes(),
        ));
        materials.push(format!("\"look{index}.material.json\""));
    }
    materials.push("\"checker.material.json\"".into());
    files.push((
        "catalog.json".into(),
        format!(r#"{{"meshes":["sphere.mesh.json","wall.mesh.json"],"materials":[{}]}}"#, materials.join(","))
            .into_bytes(),
    ));
    let borrowed: Vec<(&str, &[u8])> = files.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
    let Some(mut scene) = Scene::new("sheet", (900, 600), &borrowed) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let picture = scene.draw(&sheet_frame(&[], true));
    picture.save("material-sheet-b.png");
    // Every sphere is there, lit.
    for (index, (name, ..)) in sheet().iter().enumerate() {
        let (x, y, _) = sphere_disc(index);
        let pixel = picture.at(x as u32, y as u32);
        assert!(pixel.iter().any(|&c| c > 12), "the {name} sphere is black at its middle: {pixel:?}");
    }
    // Each lobe is in the picture: the sphere differs from the same one without it, where the lobe lives.
    let without = scene.draw(&sheet_frame(&[1, 2, 3, 4], true));
    without.save("material-sheet-b-lobes-off.png");
    let everywhere = |_: f64, _: f64| true;
    // Clear coat: a sharp, bright reflection of the light that the matte base has no peak like.
    let (coated, bare) = (disc(&picture, 1, 0.0, 0.95, everywhere).0, disc(&without, 1, 0.0, 0.95, everywhere).0);
    assert!(coated > bare + 30, "the clear coat's highlight peaks at {coated}, the bare paint's at {bare}");
    // Sheen: the cloth brightens toward its rim, where it is seen at a grazing angle.
    let (sheen_rim, plain_rim) =
        (disc(&picture, 2, 0.75, 0.95, everywhere).1, disc(&without, 2, 0.75, 0.95, everywhere).1);
    assert!(sheen_rim > plain_rim + 6.0, "the sheen's rim is {sheen_rim}, without it {plain_rim}");
    // Anisotropy: the highlight is not the isotropic one (the stretch is measured in its own test).
    let (cx, cy, r) = sphere_disc(3);
    let differing = (0..picture.height)
        .flat_map(|y| (0..picture.width).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            (f64::from(x) - cx).hypot(f64::from(y) - cy) < r * 0.95
                && picture.at(x, y)[1].abs_diff(without.at(x, y)[1]) > 12
        })
        .count();
    assert!(differing > 150, "the brushed metal differs from the isotropic one in {differing} pixels");
    // Subsurface: light wraps past the sun's terminator, the band where the sun's cosine is just below 0.
    let shadowed = |dx: f64, dy: f64| {
        let dz = (1.0 - dx * dx - dy * dy).max(0.0).sqrt();
        let cosine = (dx * 0.45 + dy * 0.6 + dz * 0.65) / (0.45f64 * 0.45 + 0.6 * 0.6 + 0.65 * 0.65).sqrt();
        (-0.4..-0.02).contains(&cosine)
    };
    let (wrapped, flat) = (disc(&picture, 4, 0.0, 0.95, shadowed).1, disc(&without, 4, 0.0, 0.95, shadowed).1);
    assert!(wrapped > flat + 4.0, "the wax past the terminator is {wrapped} on average, without the subsurface {flat}");
    // Glass shows the checkered wall through it, refracted: its pixels differ from the wall straight behind.
    let through = scene.draw(&sheet_frame(&[], false));
    let (gx, gy, gr) = sphere_disc(5);
    let mut changed = 0;
    for y in (gy - gr * 0.6) as u32..(gy + gr * 0.6) as u32 {
        for x in (gx - gr * 0.6) as u32..(gx + gr * 0.6) as u32 {
            if (f64::from(x) - gx).hypot(f64::from(y) - gy) < gr * 0.6 && picture.at(x, y) != through.at(x, y) {
                changed += 1;
            }
        }
    }
    assert!(changed > 2000, "the glass changes {changed} pixels of the picture behind it");
}

// A floor and spheres lit the way a game's shader lights with the library: the
// sky (`surface_ambient`), then the frame's lights' two parts, then the sun's
// Lambert. `${library}` is what the build has, `${ambient}` and `${local}` the
// functions to call (the library's, or the frozen old ones of the parent
// commit), `${standard}` set where `standard_surface` lights instead.
const GAME: &str = r#"
struct Scene {
  view: mat4x4f, projection: mat4x4f, ambient: vec4f, light_direction: vec4f, light_color: vec4f,
  light_view_projection: mat4x4f, shadow: vec4f, zenith: vec4f, horizon: vec4f, ground: vec4f, eye: vec4f,
};
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
${library}
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
};
@vertex fn vertex(@location(0) position: vec3f, @location(1) normal: vec3f) -> Corner {
  let world = draw.model * vec4f(position, 1.0);
  return Corner(scene.projection * scene.view * world, world.xyz, (draw.normal * vec4f(normal, 0.0)).xyz);
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let normal = normalize(corner.normal);
  let view = normalize(scene.eye.xyz - corner.world);
  let seen = scene.view * vec4f(corner.world, 1.0);
  let albedo = draw.color.rgb;
  let metallic = draw.parameters.x;
  let roughness = draw.parameters.y;
  let f0 = mix(vec3f(0.04), albedo, metallic);
  let sun = max(dot(normal, -normalize(scene.light_direction.xyz)), 0.0);
  ${shade}
  light += albedo * (1.0 - metallic) * scene.light_color.rgb * sun;
  return vec4f(light, 1.0);
}
"#;

const SHADE_GAME: &str = "
  var light = ${ambient}(corner.world, normal, view, albedo, metallic, roughness);
  let local = ${local}(corner.world, normal, view, roughness, f0, corner.position, seen.z);
  light += local.diffuse * albedo * (1.0 - metallic);
  light += local.specular;";

const SHADE_STANDARD: &str = "
  let s = surface_default(corner.world, normal, view, albedo, metallic, roughness, corner.position, seen.z);
  var light = standard_surface(s);";

/// The library as this build has it, for shaders that call `surface_ambient` and `clustered_surface`.
fn game_library() -> String {
    format!(
        "{}\n{}\n{}\n{}",
        pen_render::LIGHTS_WGSL,
        pen_render::ENVIRONMENT_WGSL,
        pen_render::SURFACE_LOBES_WGSL,
        pen_render::SURFACE_WGSL
    )
}

fn game_shader(library: &str, shade: &str, ambient: &str, local: &str) -> String {
    GAME.replace("${library}", library)
        .replace("${shade}", shade)
        .replace("${ambient}", ambient)
        .replace("${local}", local)
}

/// Draws the game scene with `shader`: a floor, three spheres (matte, rough
/// metal, glossy plastic), the sky, the sun and three lights, one a spot.
fn game_picture(name: &str, shader: &str) -> Option<Picture> {
    let size = (480u32, 270u32);
    let mut scene = Scene::new(
        name,
        size,
        &[
            (
                "catalog.json",
                br#"{"meshes":["sphere.mesh.json","floor.mesh.json"],"materials":["game.material.json"]}"#,
            ),
            ("sphere.mesh.json", sphere(32, 64).as_bytes()),
            ("floor.mesh.json", floor(9.0, -1.0, 1.0).as_bytes()),
            ("game.wgsl", shader.as_bytes()),
            ("game.material.json", br#"{"shader":"game.wgsl","color":[0.8,0.8,0.8,1],"lighting":"unlit"}"#),
        ],
    )?;
    let (view, projection, _) = camera([0.0, 1.6, 7.0], [0.0, -0.3, 0.0], f64::from(size.0) / f64::from(size.1));
    let spheres: [([f32; 3], [f32; 2]); 3] =
        [([-2.4, 0.0, 0.0], [0.0, 0.9]), ([0.0, 0.0, 0.0], [1.0, 0.45]), ([2.4, 0.0, 0.0], [0.0, 0.12])];
    let mut draws =
        vec![Draw { mesh: 1, material: 0, transform: IDENTITY, parameters: Some([0.0, 0.7, 0.0, 0.0]), extra: None }];
    for (at, parameters) in spheres {
        draws.push(Draw {
            mesh: 0,
            material: 0,
            transform: translate(at[0], at[1], at[2], 1.0),
            parameters: Some([parameters[0], parameters[1], 0.0, 0.0]),
            extra: None,
        });
    }
    let mut frame = frame(draws);
    (frame.view, frame.projection) = (view, projection);
    frame.quality.sky = true;
    frame.sky = sky();
    frame.light_direction = vector(-0.4, -0.7, -0.5);
    frame.light_color = vector(0.5, 0.45, 0.38);
    let mut spot = point_light(1.0, 2.5, 2.0, 12.0, [1.4, 1.0, 0.7]);
    (spot.inner, spot.outer, spot.direction) = (0.95, 0.75, vector(-0.2, -0.8, -0.4));
    frame.lights = vec![
        point_light(-3.0, 0.6, 2.5, 8.0, [0.9, 0.6, 0.4]),
        point_light(3.5, 0.2, 3.0, 7.0, [0.15, 0.3, 0.7]),
        spot,
    ];
    Some(scene.draw(&frame))
}

fn differing(a: &Picture, b: &Picture) -> (usize, u8) {
    let pairs = a.rgb.iter().zip(&b.rgb);
    let count = pairs.clone().filter(|(x, y)| x != y).count();
    let largest = pairs.flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| p.abs_diff(*q))).max().unwrap_or(0);
    (count, largest)
}

#[test]
fn the_base_lobe_renders_as_the_functions_it_replaced_did() {
    // The parent commit's own `surface_ambient` and `clustered_surface`, frozen under `old_` names,
    // against the library's: the same picture to the bit.
    let old_library = format!(
        "{}\n{}\n{}",
        pen_render::LIGHTS_WGSL,
        pen_render::ENVIRONMENT_WGSL,
        include_str!("../src/surface/baseline.wgsl")
    );
    let old = game_shader(&old_library, SHADE_GAME, "old_surface_ambient", "old_clustered_surface");
    let new = game_shader(&game_library(), SHADE_GAME, "surface_ambient", "clustered_surface");
    let standard = game_shader(&game_library(), SHADE_STANDARD, "", "");
    let (Some(before), Some(after), Some(through)) =
        (game_picture("old", &old), game_picture("new", &new), game_picture("standard", &standard))
    else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    after.save("base-lobe-library.png");
    // The picture is lit: it is not the sky and black.
    assert!(after.rgb.iter().filter(|p| p[0] > 30 || p[1] > 30).count() > 20_000, "the scene is dark");
    assert_eq!(differing(&before, &after), (0, 0), "the library's base lobe is not the old functions' picture");
    let (count, largest) = differing(&after, &through);
    eprintln!("standard_surface against the two old halves summed: {count} pixels differ, by at most {largest} of 255");
    assert!(
        largest <= 1,
        "standard_surface with the base lobe lights as the two halves summed: {count} pixels, up to {largest}"
    );
    // What a build left of it is for the cross-build check (README of the report): the same picture, from origin/main.
    if let Some(directory) = std::env::var_os("SURFACE_DUMP") {
        let bytes: Vec<u8> = after.rgb.iter().flatten().copied().collect();
        std::fs::write(Path::new(&directory).join("game-scene.raw"), bytes).unwrap();
    }
}

// A sphere of white metal lit by the sun alone, brushed along x (the draw's
// parameters are metallic, roughness, anisotropy and the rotation of the brushing about the normal): the anisotropic lobe's
// own picture. The tangent is the world's x, made perpendicular to the normal.
const BRUSHED: &str = r#"
struct Scene {
  view: mat4x4f, projection: mat4x4f, ambient: vec4f, light_direction: vec4f, light_color: vec4f,
  light_view_projection: mat4x4f, shadow: vec4f, zenith: vec4f, horizon: vec4f, ground: vec4f, eye: vec4f,
};
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
${surface}
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
};
@vertex fn vertex(@location(0) position: vec3f, @location(1) normal: vec3f) -> Corner {
  let world = draw.model * vec4f(position, 1.0);
  return Corner(scene.projection * scene.view * world, world.xyz, (draw.normal * vec4f(normal, 0.0)).xyz);
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let normal = normalize(corner.normal);
  let view = normalize(scene.eye.xyz - corner.world);
  var s = surface_default(corner.world, normal, view, vec3f(1.0), draw.parameters.x, draw.parameters.y, corner.position, 0.0);
  s.lobes = SURFACE_ANISOTROPY;
  s.anisotropy = draw.parameters.z;
  s = surface_frame(s, vec3f(1.0, 0.0, 0.0), 1.0, draw.parameters.w);
  return vec4f(standard_sun(s, scene.light_direction.xyz, scene.light_color.rgb, 1.0), 1.0);
}
"#;

type V3 = [f64; 3];

fn dot3(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn scale3(a: V3, k: f64) -> V3 {
    [a[0] * k, a[1] * k, a[2] * k]
}

fn add3(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn unit3(a: V3) -> V3 {
    scale3(a, 1.0 / dot3(a, a).sqrt())
}

/// The anisotropic GGX lobe of a white metal, per unit radiance with the cosine in it and times pi (the
/// frame's light scale): D, V and F = 1 written from the formulas, in f64.
fn brushed_response(n: V3, v: V3, l: V3, tangent: V3, roughness: f64, anisotropy: f64, rotation: f64) -> f64 {
    let nl = dot3(n, l).max(0.0);
    let nv = dot3(n, v).max(1e-5);
    if nl <= 0.0 || dot3(n, v) <= 0.0 {
        return 0.0;
    }
    let alpha = roughness.clamp(0.04, 1.0).powi(2);
    let at = alpha * (1.0 - anisotropy * anisotropy) + anisotropy * anisotropy;
    let ab = alpha;
    // The tangent made perpendicular to the normal, the bitangent across it.
    let t0 = unit3(add3(tangent, scale3(n, -dot3(n, tangent))));
    let t = add3(scale3(t0, rotation.cos()), scale3(cross3(n, t0), rotation.sin()));
    let b = cross3(n, t);
    let h = unit3(add3(v, l));
    let nh = dot3(n, h).clamp(0.0, 1.0);
    let (th, bh) = (dot3(t, h), dot3(b, h));
    let q = (th / at).powi(2) + (bh / ab).powi(2) + nh * nh;
    let d = 1.0 / (std::f64::consts::PI * at * ab * q * q);
    let len = |x: f64, y: f64, z: f64| (x * x + y * y + z * z).sqrt();
    let visibility = 0.5
        / (nv * len(at * dot3(t, l), ab * dot3(b, l), nl) + nl * len(at * dot3(t, v), ab * dot3(b, v), nv)).max(1e-6);
    d * visibility * nl * std::f64::consts::PI
}

/// The picture of the brushed sphere (green, 0..1) the f64 reference predicts, with the light's color 1.
fn brushed_reference(
    size: u32,
    eye: V3,
    axes: [V3; 3],
    to_sun: V3,
    roughness: f64,
    anisotropy: f64,
    rotation: f64,
) -> Vec<f64> {
    let [right, up, back] = axes;
    let (f, aspect) = (1.0 / 0.6f64.tan(), 1.0);
    let mut out = vec![0.0; (size * size) as usize];
    for py in 0..size {
        for px in 0..size {
            let ndc = (
                (f64::from(px) + 0.5) / f64::from(size) * 2.0 - 1.0,
                1.0 - (f64::from(py) + 0.5) / f64::from(size) * 2.0,
            );
            let (dx, dy) = (ndc.0 * aspect / f, ndc.1 / f);
            let direction = unit3(add3(add3(scale3(right, dx), scale3(up, dy)), scale3(back, -1.0)));
            // The unit sphere at the origin.
            let b = dot3(eye, direction);
            let c = dot3(eye, eye) - 1.0;
            let discriminant = b * b - c;
            if discriminant < 0.0 {
                continue;
            }
            let t = -b - discriminant.sqrt();
            let n = unit3(add3(eye, scale3(direction, t)));
            let v = scale3(direction, -1.0);
            out[(py * size + px) as usize] =
                brushed_response(n, v, to_sun, [1.0, 0.0, 0.0], roughness, anisotropy, rotation);
        }
    }
    out
}

/// The full width at half the peak of the brightest row and column through the peak of `image` (any
/// monotone display of green), in pixels, with linear interpolation of the crossings.
fn half_widths(image: &[f64], size: u32) -> (f64, f64) {
    let at = |x: i64, y: i64| image[(y as usize) * size as usize + x as usize];
    let (mut peak, mut px, mut py) = (0.0, 0, 0);
    for y in 0..i64::from(size) {
        for x in 0..i64::from(size) {
            if at(x, y) > peak {
                (peak, px, py) = (at(x, y), x, y);
            }
        }
    }
    let half = peak / 2.0;
    let width = |along: &dyn Fn(i64) -> f64, start: i64| {
        let crossing = |step: i64| {
            let mut i = start;
            while (0..i64::from(size)).contains(&(i + step)) && along(i + step) >= half {
                i += step;
            }
            if (0..i64::from(size)).contains(&(i + step)) {
                let (a, b) = (along(i), along(i + step));
                (i as f64) + (step as f64) * (a - half) / (a - b)
            } else {
                i as f64
            }
        };
        crossing(1) - crossing(-1)
    };
    (width(&|x| at(x, py), px), width(&|y| at(px, y), py))
}

#[test]
fn an_anisotropic_highlight_stretches_along_its_tangent_as_the_reference_predicts() {
    let size = 192u32;
    let eye = [0.0, 0.0, 4.0];
    let (view, projection, axes) = camera(eye, [0.0, 0.0, 0.0], 1.0);
    let to_sun = unit3([0.35, 0.45, 1.0]);
    let shader = BRUSHED.replace(
        "${surface}",
        &format!(
            "{}\n{}\n{}\n{}",
            pen_render::LIGHTS_WGSL,
            pen_render::ENVIRONMENT_WGSL,
            pen_render::SURFACE_LOBES_WGSL,
            pen_render::SURFACE_WGSL
        ),
    );
    let Some(mut scene) = Scene::new(
        "brushed",
        (size, size),
        &[
            ("catalog.json", br#"{"meshes":["sphere.mesh.json"],"materials":["brushed.material.json"]}"#),
            ("sphere.mesh.json", sphere(96, 192).as_bytes()),
            ("brushed.wgsl", shader.as_bytes()),
            ("brushed.material.json", br#"{"shader":"brushed.wgsl","color":[1,1,1,1],"lighting":"unlit"}"#),
        ],
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let roughness = 0.25;
    let mut render = |anisotropy: f64, rotation: f64, peak: f64| {
        let mut frame = frame(vec![Draw {
            mesh: 0,
            material: 0,
            transform: IDENTITY,
            parameters: Some([1.0, roughness as f32, anisotropy as f32, rotation as f32]),
            extra: None,
        }]);
        (frame.view, frame.projection) = (view, projection);
        frame.light_direction = vector(-to_sun[0] as f32, -to_sun[1] as f32, -to_sun[2] as f32);
        // The display shows light as it is below about 0.5: scale the brightest to 0.4.
        let k = 0.4 / peak;
        frame.light_color = vector(k as f32, k as f32, k as f32);
        scene.draw(&frame)
    };
    let mut stretches = Vec::new();
    for (anisotropy, rotation) in [(0.5, 0.0), (0.0, 0.0), (-0.5, 0.0), (0.5, std::f64::consts::FRAC_PI_2)] {
        let want = brushed_reference(size, eye, axes, to_sun, roughness, anisotropy, rotation);
        let peak = want.iter().copied().fold(0.0, f64::max);
        let picture = render(anisotropy, rotation, peak);
        picture.save(&format!("anisotropy-b-{anisotropy}-turned-{rotation:.2}.png"));
        let got: Vec<f64> = picture.rgb.iter().map(|p| f64::from(p[1]) / 255.0).collect();
        let expected: Vec<f64> = want.iter().map(|v| v / peak * 0.4).collect();
        // The picture is the reference's, to the display's 8 bits and the sphere's facets.
        let worst = got.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let mean = got.iter().zip(&expected).map(|(a, b)| (a - b).abs()).sum::<f64>() / got.len() as f64;
        let (gx, gy) = half_widths(&got, size);
        let (wx, wy) = half_widths(&expected, size);
        eprintln!(
            "anisotropy {anisotropy} turned {rotation:.2}: highlight {gx:.1} x {gy:.1} px (ratio {:.3}), f64 reference {wx:.1} x {wy:.1} (ratio {:.3}); picture differs by {mean:.4} on average, {worst:.3} at most",
            gx / gy,
            wx / wy
        );
        assert!(
            mean < 0.004 && worst < 0.08,
            "anisotropy {anisotropy} turned {rotation:.2}: the picture is off the reference's by {mean} on average, {worst} at most"
        );
        assert!(
            ((gx / gy) / (wx / wy) - 1.0).abs() < 0.04,
            "anisotropy {anisotropy} turned {rotation:.2}: stretch {} against the reference's {}",
            gx / gy,
            wx / wy
        );
        stretches.push(gx / gy);
    }
    // Along the tangent (x), round for none, the same for the other sign (the lobe's widths use the
    // square of the anisotropy; the sign only bends the sky's reflection), and across it once the
    // brushing is turned a quarter.
    assert!(stretches[0] > 1.4, "a positive anisotropy stretches the highlight along the tangent: {}", stretches[0]);
    assert!((stretches[1] - 1.0).abs() < 0.15, "no anisotropy gives a round highlight: {}", stretches[1]);
    assert!(
        (stretches[2] - stretches[0]).abs() < 0.05,
        "the sign does not change the lights' highlight: {stretches:?}"
    );
    assert!(stretches[3] < 0.75, "turned a quarter, the highlight stretches across x: {}", stretches[3]);
}

// A brick wall lying flat under a low sun, seen at a grazing angle: its uv
// are marched through the height field of the material's layer texture
// (binding 4) before every read, and its grooves shade themselves. The draw's
// parameters are the height range in uv units (0 for none: the plain wall),
// whether the grooves shade themselves, and the sun's visibility.
const BRICKS: &str = r#"
struct Scene {
  view: mat4x4f, projection: mat4x4f, ambient: vec4f, light_direction: vec4f, light_color: vec4f,
  light_view_projection: mat4x4f, shadow: vec4f, zenith: vec4f, horizon: vec4f, ground: vec4f, eye: vec4f,
};
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@group(1) @binding(1) var bricks: texture_2d<f32>;
@group(1) @binding(2) var bricks_sampler: sampler;
@group(1) @binding(4) var heights: texture_2d<f32>;
${library}
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
};
@vertex fn vertex(@location(0) position: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f) -> Corner {
  let world = draw.model * vec4f(position, 1.0);
  return Corner(scene.projection * scene.view * world, world.xyz, (draw.normal * vec4f(normal, 0.0)).xyz, uv);
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let normal = normalize(corner.normal);
  let view = normalize(scene.eye.xyz - corner.world);
  let seen = scene.view * vec4f(corner.world, 1.0);
  let to_sun = -normalize(scene.light_direction.xyz);
  // The uv's frame from the screen-space derivatives (the mesh has no tangents).
  let tangent = surface_derived_tangent(corner.world, normal, corner.uv);
  let bitangent = cross(normal, tangent.xyz) * tangent.w;
  let toward_eye = parallax_tangent_space(tangent.xyz, bitangent, normal, view);
  let toward_sun = parallax_tangent_space(tangent.xyz, bitangent, normal, to_sun);
  let found = parallax_uv(heights, bricks_sampler, corner.uv, toward_eye, toward_sun, draw.parameters.x, 0.0, draw.parameters.y > 0.5);
  let albedo = textureSample(bricks, bricks_sampler, found.uv).rgb;
  let s = surface_default(corner.world, normal, view, albedo, 0.0, 0.85, corner.position, seen.z);
  let light = standard_surface(s) + standard_sun(s, scene.light_direction.xyz, scene.light_color.rgb, found.shadow);
  return vec4f(light, 1.0);
}
"#;

/// A quad lying at height 0 over x in `x0..x1` and z in -3..3, its uv running `repeat` tiles across each way.
fn flat_quad(x0: f32, x1: f32, repeat: f32) -> String {
    format!(
        r#"{{"positions":[{x0},0,-3, {x1},0,-3, {x0},0,3, {x1},0,3],"normals":[0,1,0,0,1,0,0,1,0,0,1,0],
            "uvs":[0,0,{r},0,0,{r2},{r},{r2}],"indices":[0,2,1,1,2,3]}}"#,
        r = repeat / 2.0,
        r2 = repeat,
    )
}

/// A 256 by 256 brick tile, 2 bricks across and 4 courses down with staggered joints, as an albedo PNG and a height PNG
/// (bricks at 255, mortar at 0, a few texels of slope between).
fn brick_textures() -> (Vec<u8>, Vec<u8>) {
    let (brick_w, brick_h, mortar) = (128.0f64, 64.0f64, 10.0f64);
    let size = 256u32;
    let mut albedo = image::RgbImage::new(size, size);
    let mut height = image::RgbImage::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let course = (f64::from(y) / brick_h).floor();
            let offset = if (course as i64) % 2 == 0 { 0.0 } else { brick_w / 2.0 };
            let u = (f64::from(x) + offset).rem_euclid(brick_w);
            let v = f64::from(y).rem_euclid(brick_h);
            // How far inside the brick's edge the texel is: a mortar joint is 0, a brick's face 1 over a few texels.
            let edge = u.min(brick_w - u).min(v).min(brick_h - v);
            let h = ((edge - mortar / 2.0) / 3.0).clamp(0.0, 1.0);
            let id = ((f64::from(x) + offset) / brick_w).floor() as i64 + course as i64 * 7;
            let tint = 0.8 + 0.2 * (((id * 2654435761) & 0xff) as f64 / 255.0);
            let brick = [0.62 * tint, 0.27 * tint, 0.17 * tint];
            let joint = [0.55, 0.52, 0.47];
            let colour = |c: usize| (255.0 * (joint[c] + (brick[c] - joint[c]) * h)).round() as u8;
            albedo.put_pixel(x, y, image::Rgb([colour(0), colour(1), colour(2)]));
            let grey = (255.0 * h).round() as u8;
            height.put_pixel(x, y, image::Rgb([grey, grey, grey]));
        }
    }
    let png = |image: &image::RgbImage| {
        let mut bytes = Vec::new();
        image.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png).unwrap();
        bytes
    };
    (png(&albedo), png(&height))
}

#[test]
fn a_brick_wall_seen_at_a_grazing_angle_with_and_without_parallax() {
    let (albedo, height) = brick_textures();
    let shader = BRICKS.replace("${library}", &game_library());
    let size = (960u32, 540u32);
    let Some(mut scene) = Scene::new(
        "bricks",
        size,
        &[
            ("catalog.json", br#"{"meshes":["left.mesh.json","right.mesh.json"],"materials":["bricks.material.json"]}"#),
            ("left.mesh.json", flat_quad(-3.0, 0.0, 3.0).as_bytes()),
            ("right.mesh.json", flat_quad(0.0, 3.0, 3.0).as_bytes()),
            ("bricks.wgsl", shader.as_bytes()),
            ("bricks.png", &albedo),
            ("bricks-height.png", &height),
            (
                "bricks.material.json",
                br#"{"shader":"bricks.wgsl","texture":"bricks.png","layerTexture":"bricks-height.png","color":[1,1,1,1],"lighting":"unlit"}"#,
            ),
        ],
    ) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let (view, projection, _) = camera([0.0, 0.85, 3.4], [0.0, 0.0, -0.4], f64::from(size.0) / f64::from(size.1));
    let mut render = |left: [f32; 2], right: [f32; 2]| {
        let draw = |mesh: u32, p: [f32; 2]| Draw {
            mesh,
            material: 0,
            transform: IDENTITY,
            parameters: Some([p[0], p[1], 0.0, 0.0]),
            extra: None,
        };
        let mut frame = frame(vec![draw(0, left), draw(1, right)]);
        (frame.view, frame.projection) = (view, projection);
        frame.quality.sky = true;
        frame.sky = sky();
        // A low sun from the side, across the joints.
        frame.light_direction = vector(-0.8, -0.35, -0.3);
        frame.light_color = vector(1.5, 1.35, 1.1);
        scene.draw(&frame)
    };
    let scale = 0.02;
    let plain = render([0.0, 0.0], [0.0, 0.0]);
    let parallax = render([0.0, 0.0], [scale, 0.0]);
    let shaded = render([0.0, 0.0], [scale, 1.0]);
    // Left half, plain wall on the left and parallax on the right.
    shaded.save("pom-b.png");
    plain.save("pom-b-plain.png");
    parallax.save("pom-b-no-self-shadow.png");
    let half = size.0 / 2;
    let on_right = |a: &Picture, b: &Picture| {
        let (mut count, mut total) = (0usize, 0.0f64);
        for y in 0..size.1 {
            for x in half + 2..size.0 {
                let (p, q) = (a.at(x, y), b.at(x, y));
                let d: i32 = (0..3).map(|c| (i32::from(p[c]) - i32::from(q[c])).abs()).sum();
                if d > 24 {
                    count += 1;
                }
                total += f64::from(d);
            }
        }
        (count, total / f64::from((size.0 - half - 2) * size.1))
    };
    // The plain wall is what scale 0 draws, on whichever side: the left half is the same in every picture.
    for y in 0..size.1 {
        for x in 0..half - 2 {
            assert_eq!(plain.at(x, y), parallax.at(x, y), "the left half does not move ({x}, {y})");
            assert_eq!(plain.at(x, y), shaded.at(x, y), "the left half does not move ({x}, {y})");
        }
    }
    let (moved, mean) = on_right(&plain, &parallax);
    eprintln!("parallax moves {moved} pixels of the right half, by {mean:.1} on average (sum of channels)");
    assert!(moved > 20_000, "parallax moves only {moved} pixels");
    // The grooves' self-shadow darkens the right half further and changes nothing on the left.
    let lum = |p: &Picture| {
        let mut sum = 0.0;
        for y in 0..size.1 {
            for x in half + 2..size.0 {
                let [r, g, b] = p.at(x, y);
                sum += f64::from(r) + f64::from(g) + f64::from(b);
            }
        }
        sum
    };
    let (open, grooved) = (lum(&parallax), lum(&shaded));
    assert!(grooved < open * 0.995, "self-shadow darkens the wall: {grooved} vs {open}");
}
