//! Translucent surfaces, particles and overlays on a real (headless) GPU: drawn
//! after the opaque picture is misted, so that they are misted at their own depth
//! and can read what lies behind them. Held to f64 references: the mist between
//! the eye and a translucent quad or a particle (`fog_apply`) at both volumetric
//! tiers and without a grid, composited over the misted wall behind it; the opaque
//! picture's blur chain and its nearest depth (`opaque_behind`, `opaque_depth_at`);
//! and the order of the passes: edges kept, TAA, screen-space reflections, auto
//! exposure, scene filters and a frame with nothing translucent in it, which must
//! draw what it drew before. The pictures are written to the directory
//! `TRANSLUCENCY_SHOTS` names (default target/look-shots). Skips when no adapter is
//! available.
#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "test geometry")]
use std::path::{Path, PathBuf};

mod common;

use common::*;
use pen_protocol::*;
use pen_render::{Display, Renderer, RendererOptions, VolumeLayer};

/// The library as a shader adds it.
const TRANSLUCENCY: &str = pen_render::TRANSLUCENCY_WGSL;

const HEAD: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
"#;

/// Draws its mesh in its material's colour.
const FLAT: &str = r#"
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return vec4f(draw.color.rgb, 1.0); }
"#;

/// Draws its mesh in its material's colour and alpha, misted at its own depth.
const GLASS: &str = r#"
@group(0) @binding(6) var sky_sampler: sampler;
${translucency}
struct Surface { @builtin(position) position: vec4f, @location(0) world: vec3f };
@vertex fn vertex(@location(0) position: vec3f) -> Surface {
  let world = draw.model * vec4f(position, 1.0);
  return Surface(scene.projection * scene.view * world, world.xyz);
}
@fragment fn fragment(surface: Surface) -> @location(0) vec4f {
  return vec4f(fog_apply(draw.color.rgb, surface.world), draw.color.a);
}
"#;

/// Shows what the library says of the pixel it draws: by `draw.parameters.y`, 0 the
/// opaque picture at level `x`, 1 the opaque surface's distance over 100, 2 where
/// the world point's place in the picture differs from the pixel's (in tenths of a
/// pixel, from a half), 3 the last level over ten, 4 whether it reads one.
const PROBE: &str = r#"
@group(0) @binding(6) var sky_sampler: sampler;
${translucency}
struct Surface { @builtin(position) position: vec4f, @location(0) world: vec3f };
@vertex fn vertex(@location(0) position: vec3f) -> Surface {
  let world = draw.model * vec4f(position, 1.0);
  return Surface(scene.projection * scene.view * world, world.xyz);
}
@fragment fn fragment(surface: Surface) -> @location(0) vec4f {
  let uv = opaque_screen_uv(surface.position);
  let mode = u32(draw.parameters.y);
  if (mode == 0u) { return vec4f(opaque_behind(uv, draw.parameters.x), 1.0); }
  if (mode == 1u) { return vec4f(min(opaque_depth_at(uv) / 100.0, 0.9), 0.0, 0.0, 1.0); }
  if (mode == 2u) {
    let size = vec2f(textureDimensions(opaque_color, 0));
    let off = (opaque_world_uv(surface.world) - uv) * size;
    return vec4f(0.5 + off.x * 0.1, 0.5 + off.y * 0.1, 0.0, 1.0);
  }
  if (mode == 3u) { return vec4f(opaque_levels() / 10.0, 0.0, 0.0, 1.0); }
  return vec4f(select(0.5, 0.0, opaque_available()), select(0.0, 0.5, opaque_available()), 0.0, 1.0);
}
"#;

/// A scene filter that shows the picture it is given inverted.
const INVERT: &str = r#"
@group(0) @binding(0) var<uniform> screen: vec4f;
struct Sprite { rect: vec4f, uv: vec4f, color: vec4f, parameters: vec4f };
@group(0) @binding(1) var<storage,read> sprites: array<Sprite>;
@group(0) @binding(4) var scene: texture_2d<f32>;
struct Output { @builtin(position) position: vec4f, @location(0) color: vec4f };
@vertex fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Output {
  let corners = array<vec2f, 6>(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1), vec2f(0, 1), vec2f(1, 0), vec2f(1, 1));
  let sprite = sprites[i];
  let at = sprite.rect.xy + corners[v] * sprite.rect.zw;
  return Output(vec4f(at.x / screen.x * 2 - 1, 1 - at.y / screen.y * 2, 0, 1), sprite.color);
}
@fragment fn fragment(v: Output) -> @location(0) vec4f {
  let size = vec2i(textureDimensions(scene));
  let pixel = clamp(vec2i(v.position.xy / screen.xy * vec2f(size)), vec2i(0), size - vec2i(1));
  return vec4f(vec3f(1.0) - textureLoad(scene, pixel, 0).rgb, 1.0);
}
"#;

fn shader(body: &str) -> String {
    format!("{HEAD}{}", body.replace("${translucency}", TRANSLUCENCY))
}

fn shots() -> PathBuf {
    let directory = std::env::var_os("TRANSLUCENCY_SHOTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/look-shots"));
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

fn save(name: &str, pixels: &[[u8; 4]], size: u32) {
    let image = image::RgbaImage::from_fn(size, size, |x, y| {
        let [r, g, b, _] = pixels[(y * size + x) as usize];
        image::Rgba([r, g, b, 255])
    });
    image.save(shots().join(name)).unwrap();
}

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn unit(a: V3) -> V3 {
    let n = dot(a, a).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn at(origin: V3, direction: V3, length: f64) -> V3 {
    [origin[0] + direction[0] * length, origin[1] + direction[1] * length, origin[2] + direction[2] * length]
}

/// A look-at camera: the view and projection, and its axes.
struct Camera {
    view: Matrix,
    projection: Matrix,
    eye: V3,
    right: V3,
    up: V3,
    back: V3,
    scale: (f64, f64),
}

fn camera(eye: V3, target: V3, aspect: f64, tangent: f64) -> Camera {
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
    .map(|value| value as f32);
    let (near, far, f) = (0.1f64, 400.0f64, 1.0 / tangent);
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
    .map(|value| value as f32);
    Camera { view, projection, eye, right, up, back, scale: (f / aspect, f) }
}

impl Camera {
    /// The unit direction through the middle of pixel (x, y) of a `size` square
    /// picture and the length of the ray per unit of depth ahead of the eye.
    fn ray(&self, size: u32, x: u32, y: u32) -> (V3, f64) {
        let (u, v) = ((f64::from(x) + 0.5) / f64::from(size), (f64::from(y) + 0.5) / f64::from(size));
        let (nx, ny) = (u * 2.0 - 1.0, 1.0 - v * 2.0);
        let view = [nx / self.scale.0, ny / self.scale.1, -1.0];
        let world: V3 =
            std::array::from_fn(|i| view[0] * self.right[i] + view[1] * self.up[i] + view[2] * self.back[i]);
        (unit(world), dot(view, view).sqrt())
    }

    /// The unit-square mesh facing the camera, `depth` ahead of it along its axis,
    /// `half` across to each side (the mesh's corners are at plus and minus one).
    fn facing(&self, depth: f64, half: f64) -> Matrix {
        self.placed(depth, half, (0.0, 0.0))
    }

    /// The same, moved `shift` across and up its plane (in the plane's units).
    fn placed(&self, depth: f64, half: f64, shift: (f64, f64)) -> Matrix {
        let centre = at(self.eye, [-self.back[0], -self.back[1], -self.back[2]], depth);
        let centre = at(at(centre, self.right, shift.0), self.up, shift.1);
        let columns = [
            [self.right[0] * half, self.right[1] * half, self.right[2] * half, 0.0],
            [self.up[0] * half, self.up[1] * half, self.up[2] * half, 0.0],
            [self.back[0], self.back[1], self.back[2], 0.0],
            [centre[0], centre[1], centre[2], 1.0],
        ];
        let mut matrix = [0.0f32; 16];
        for (column, values) in columns.iter().enumerate() {
            for (row, value) in values.iter().enumerate() {
                matrix[column * 4 + row] = *value as f32;
            }
        }
        matrix
    }
}

fn vector(x: f32, y: f32, z: f32) -> Vector {
    Vector { x, y, z }
}

fn scale_at(sx: f32, sy: f32, sz: f32, x: f32, y: f32, z: f32) -> Matrix {
    [sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, sz, 0.0, x, y, z, 1.0]
}

fn draw(mesh: u32, material: u32, transform: Matrix) -> Draw {
    Draw { mesh, material, transform, parameters: None, extra: None }
}

fn with_parameters(mut draw: Draw, parameters: [f32; 4]) -> Draw {
    draw.parameters = Some(parameters);
    draw
}

const SH: f32 = 0.282_094_8;

/// A sky whose mean light is `mean`: its constant coefficient.
fn sky_with_mean(mean: V3) -> Sky {
    Sky {
        sh0: vector(
            (mean[0] / f64::from(SH)) as f32,
            (mean[1] / f64::from(SH)) as f32,
            (mean[2] / f64::from(SH)) as f32,
        ),
        sun: vector(0.0, 1.0, 0.0),
        ..Sky::default()
    }
}

const QUAD: &str = r#"{"positions":[-1,-1,0, 1,-1,0, -1,1,0, 1,1,0],"normals":[0,0,1,0,0,1,0,0,1,0,0,1],
                       "uvs":[0,0,1,0,0,1,1,1],"indices":[0,1,2,2,1,3]}"#;
// Its lower edge slants across the picture.
const WEDGE: &str = r#"{"positions":[0,0,0, 1,0,0, 0,1,0],"normals":[0,0,1,0,0,1,0,0,1],
                        "uvs":[0,0,1,0,0,1],"indices":[0,1,2]}"#;

/// One of a test's materials: the shader file, and the material's own JSON keys.
struct Material {
    shader: &'static str,
    color: [f32; 4],
    extra: &'static str,
}

fn flat(color: [f32; 3]) -> Material {
    Material { shader: "flat.wgsl", color: [color[0], color[1], color[2], 1.0], extra: "" }
}

fn glass(color: [f32; 4]) -> Material {
    Material { shader: "glass.wgsl", color, extra: "" }
}

/// A test renderer over the materials it is given, drawing into a square picture.
struct Rig {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    catalog: Catalog,
    target: Offscreen,
    size: u32,
    /// How many frames have been submitted.
    submits: usize,
}

impl Rig {
    fn new(name: &str, size: u32, materials: &[Material]) -> Option<Rig> {
        let (device, queue) = featured_device()?;
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("translucency-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let write = |key: &str, bytes: &[u8]| std::fs::write(root.join(key), bytes).unwrap();
        write("quad.mesh.json", QUAD.as_bytes());
        write("wedge.mesh.json", WEDGE.as_bytes());
        write("flat.wgsl", shader(FLAT).as_bytes());
        write("glass.wgsl", shader(GLASS).as_bytes());
        write("probe.wgsl", shader(PROBE).as_bytes());
        write("invert.wgsl", INVERT.as_bytes());
        let mut png = std::io::Cursor::new(Vec::new());
        image::RgbaImage::from_pixel(4, 4, image::Rgba([255; 4])).write_to(&mut png, image::ImageFormat::Png).unwrap();
        write("decal.png", &png.into_inner());
        let mut keys = Vec::new();
        for (index, material) in materials.iter().enumerate() {
            let key = format!("m{index}.material.json");
            let [r, g, b, a] = material.color;
            let extra = if material.extra.is_empty() { String::new() } else { format!(",{}", material.extra) };
            write(
                &key,
                format!(
                    r#"{{"shader":"{}","color":[{r},{g},{b},{a}],"lighting":"unlit","instanceParameters":[0,0,0,0]{extra}}}"#,
                    material.shader
                )
                .as_bytes(),
            );
            keys.push(key);
        }
        let catalog = Catalog {
            sounds: vec![],
            meshes: vec!["quad.mesh.json".into(), "wedge.mesh.json".into()],
            materials: keys,
        };
        let options =
            RendererOptions { format: FORMAT, display: Display::SDR, assets: root, veil: None, pipeline_cache: None };
        let renderer = Renderer::new(device.clone(), queue.clone(), options).unwrap();
        let target = Offscreen::new(&device, size);
        Some(Rig { device, queue, renderer, catalog, target, size, submits: 0 })
    }

    /// Loads everything the catalog names, with pipelines for `samples` per pixel,
    /// before the first frame, so that no frame is drawn without it: the frames that
    /// follow are as many as are submitted, which pictures that depend on the frames
    /// before them (the history of TAA and of the mist) need to be compared.
    fn load(&mut self, samples: u32) {
        let mut assets: Vec<(String, String)> = Vec::new();
        for mesh in &self.catalog.meshes {
            assets.push(("mesh".into(), mesh.clone()));
        }
        for material in &self.catalog.materials {
            assets.push(("material".into(), material.clone()));
        }
        for shader in ["flat.wgsl", "glass.wgsl", "probe.wgsl"] {
            assets.push(("shader".into(), shader.into()));
        }
        assets.push(("texture".into(), "decal.png".into()));
        self.renderer.preload_scene(&assets, samples);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while self.renderer.pending() > 0 {
            assert!(std::time::Instant::now() < deadline, "assets did not load: {:?}", self.renderer.take_events());
            self.renderer.wait(std::time::Duration::from_millis(20));
        }
    }

    fn draw(&mut self, frame: &RenderFrame) -> Vec<[u8; 4]> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        loop {
            self.submits += 1;
            self.renderer.submit(frame, &self.catalog, &mut self.target).unwrap();
            if self.renderer.pending() == 0
                && !self.renderer.waiting()
                && self.renderer.stats().draw_calls >= frame.draws.len() as u32
            {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "assets did not load: {:?}", self.renderer.take_events());
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        pixels(&self.device, &self.queue, &self.target.0)
    }

    /// Draws `frame` `count` times (a still camera) and returns the last picture.
    fn settle(&mut self, frame: &RenderFrame, count: usize) -> Vec<[u8; 4]> {
        let mut last = Vec::new();
        for _ in 0..count {
            last = self.draw(frame);
        }
        last
    }

    fn volume(&self, layer: VolumeLayer) -> pen_render::VolumeImage {
        self.renderer.read_volume(layer).expect("the frame drew a volume")
    }

    /// The pixel at (x, y), as colours from 0 to 1.
    fn pixel(&self, pixels: &[[u8; 4]], x: u32, y: u32) -> V3 {
        let [r, g, b, _] = pixels[(y * self.size + x) as usize];
        [f64::from(r) / 255.0, f64::from(g) / 255.0, f64::from(b) / 255.0]
    }
}

/// An empty frame seen by `camera`.
fn view_frame(camera: &Camera) -> RenderFrame {
    let mut frame = frame(Vec::new());
    frame.view = camera.view;
    frame.projection = camera.projection;
    frame
}

fn largest_difference(a: V3, b: V3) -> f64 {
    (0..3).map(|channel| (a[channel] - b[channel]).abs()).fold(0.0, f64::max)
}

/// The exponential height mist of a homogeneous albedo of 1 lit by an ambient light
/// of `mean` alone (no sun, no lights): what the eye sees of a colour through it.
struct Mist {
    density: f64,
    height: f64,
    mean: V3,
}

/// The optical depth of the exponential height mist (`density` at the ground,
/// falling by e over `height` above it, constant below) from `eye` along the
/// unit `direction` for `length`: a closed form, by cases of where the ray
/// crosses the ground.
fn height_optical_depth(density: f64, height: f64, eye: V3, direction: V3, length: f64) -> f64 {
    let end = eye[1] + direction[1] * length;
    let (low, high) = (eye[1].min(end), eye[1].max(end));
    if high - low < 1e-9 {
        return density * length * (-eye[1].max(0.0) / height).exp();
    }
    // The share of the path (by height) above the ground and below it.
    let below = (0.0 - low).clamp(0.0, high - low) / (high - low);
    let above_low = low.max(0.0);
    let above = if high > 0.0 {
        height * ((-above_low / height).exp() - (-high / height).exp()) / (high - above_low).max(1e-12)
    } else {
        0.0
    };
    let along_above = (1.0 - below) * length;
    density * (below * length + along_above * if high > 0.0 { above } else { 0.0 })
}

impl Mist {
    /// The transmittance and the in-scatter between the eye and a point `length` along `direction`.
    fn between(&self, eye: V3, direction: V3, length: f64) -> (f64, V3) {
        let transmittance = (-height_optical_depth(self.density, self.height, eye, direction, length)).exp();
        (transmittance, self.mean.map(|mean| mean * (1.0 - transmittance)))
    }

    /// `colour` as the eye sees it through the mist from `length` away.
    fn fog(&self, colour: V3, eye: V3, direction: V3, length: f64) -> V3 {
        let (transmittance, scatter) = self.between(eye, direction, length);
        std::array::from_fn(|channel| colour[channel] * transmittance + scatter[channel])
    }
}

/// `front` over `behind` at `alpha`.
fn over(front: V3, alpha: f64, behind: V3) -> V3 {
    std::array::from_fn(|channel| alpha * front[channel] + (1.0 - alpha) * behind[channel])
}

/// What the integrated froxel volume holds at pixel (x, y) of a `size` square
/// picture and `depth` ahead of the eye, read as a sampler reads it: trilinear,
/// the layer coordinate being the slice boundary (slices of `reach` out, as
/// the grid cuts them): the in-scatter and the optical depth.
fn froxel_integral(
    volume: &pen_render::VolumeImage,
    size: u32,
    pixel: (u32, u32),
    depth: f64,
    reach: f64,
) -> (V3, f64) {
    let [width, height, layers] = volume.size;
    let slices = f64::from(layers - 1);
    let knee = reach / 64.0;
    let boundary = slices * (1.0 + depth / knee).ln() / (1.0 + reach / knee).ln();
    let (u, v) = ((f64::from(pixel.0) + 0.5) / f64::from(size), (f64::from(pixel.1) + 0.5) / f64::from(size));
    let (x, y) = (u * f64::from(width) - 0.5, v * f64::from(height) - 0.5);
    let (x0, y0, z0) = (x.floor(), y.floor(), boundary.floor());
    let (fx, fy, fz) = (x - x0, y - y0, boundary - z0);
    let mut total = [0.0f64; 4];
    for (dz, wz) in [(0.0, 1.0 - fz), (1.0, fz)] {
        for (dy, wy) in [(0.0, 1.0 - fy), (1.0, fy)] {
            for (dx, wx) in [(0.0, 1.0 - fx), (1.0, fx)] {
                let at = |value: f64, count: u32| value.clamp(0.0, f64::from(count - 1)) as u32;
                let texel = volume.at(at(x0 + dx, width), at(y0 + dy, height), at(z0 + dz, layers));
                for channel in 0..4 {
                    total[channel] += f64::from(texel[channel]) * wx * wy * wz;
                }
            }
        }
    }
    ([total[0], total[1], total[2]], total[3])
}

/// How far a pixel may be from the f64 reference: the 8 bit output's rounding
/// and the transfer functions round trip.
const LEVELS: f64 = 2.5 / 255.0;

const WALL: [f32; 3] = [0.55, 0.3, 0.2];
const PANE: [f32; 4] = [0.15, 0.55, 0.25, 0.5];
const MEAN: V3 = [0.8, 0.5, 0.3];

fn rgb(color: [f32; 3]) -> V3 {
    color.map(f64::from)
}

/// A pane `near` from the eye over a wall `far` from it, in a mist of these tiers
/// (the wall covers the picture, the pane its middle half), and what the
/// picture holds at the middle and at a corner (wall alone).
struct Pane {
    near: f64,
    far: f64,
}

fn pane_frame(view: &Camera, mist: &Mist, tier: u32, pane: &Pane) -> RenderFrame {
    let mut frame = view_frame(view);
    frame.fog = Fog { density: mist.density as f32, height: mist.height as f32, distance: 90.0, ..Fog::default() };
    frame.quality.volumetrics = tier;
    frame.sky = sky_with_mean(mist.mean);
    frame.light_color = vector(0.0, 0.0, 0.0);
    frame.draws = vec![draw(0, 0, view.facing(pane.far, 200.0)), draw(0, 1, view.facing(pane.near, pane.near * 0.25))];
    frame
}

/// What the pane's pixel, the wall's pixel and the picture the old pipeline
/// drew (the pane misted at the wall's depth) hold, from the references.
fn pane_references(rig: &Rig, view: &Camera, mist: &Mist, pane: &Pane, at: (u32, u32)) -> (V3, V3, V3) {
    let (direction, slant) = view.ray(rig.size, at.0, at.1);
    let (near, far) = (pane.near * slant, pane.far * slant);
    let wall = mist.fog(rgb(WALL), view.eye, direction, far);
    let alpha = f64::from(PANE[3]);
    let through = over(mist.fog(rgb([PANE[0], PANE[1], PANE[2]]), view.eye, direction, near), alpha, wall);
    let before = mist.fog(over(rgb([PANE[0], PANE[1], PANE[2]]), alpha, rgb(WALL)), view.eye, direction, far);
    (through, wall, before)
}

#[test]
fn a_translucent_quad_is_misted_at_its_own_depth_over_the_misted_wall_behind_it() {
    let Some(mut rig) = Rig::new("pane", 64, &[flat(WALL), glass(PANE)]) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let pane = Pane { near: 4.0, far: 40.0 };
    // Nearly homogeneous (a scale height of two kilometres: f32 cannot tell the
    // mist's heights apart far past that) and thinning with height, looking along
    // the ground and up through the heights the mist thins over. The eye is off the
    // mist's clear axis, so the analytic path has a mist to take.
    let looks = [
        ("homogeneous", Mist { density: 0.03, height: 2000.0, mean: MEAN }, [4.0, 1.5, 0.0], [4.0, 1.5, -10.0]),
        ("height", Mist { density: 0.06, height: 12.0, mean: MEAN }, [4.0, 1.5, 0.0], [4.0, 14.0, -30.0]),
    ];
    for (name, mist, eye, target) in looks {
        let view = camera(eye, target, 1.0, 0.5);
        for (tier_name, tier) in [("high", 2), ("low", 1), ("analytic", 0)] {
            let frame = pane_frame(&view, &mist, tier, &pane);
            let picture = rig.settle(&frame, 3);
            assert_eq!(rig.renderer.stats().volume_froxels > 0, tier > 0, "{name} {tier_name}");
            save(&format!("translucency-pane-{name}-{tier_name}.png"), &picture, rig.size);
            let (through, wall, before) = pane_references(&rig, &view, &mist, &pane, (32, 32));
            let got = rig.pixel(&picture, 32, 32);
            let corner = rig.pixel(&picture, 3, 3);
            let (_, wall_corner, _) = pane_references(&rig, &view, &mist, &pane, (3, 3));
            eprintln!(
                "{name} {tier_name}: pane {got:.3?} expected {through:.3?} (misted at the wall's depth it would be {before:.3?}); wall {corner:.3?} expected {wall_corner:.3?}"
            );
            assert!(
                largest_difference(got, through) < LEVELS,
                "{name} {tier_name}: the pane's pixel {got:?} vs {through:?}"
            );
            assert!(
                largest_difference(corner, wall_corner) < LEVELS,
                "{name} {tier_name}: the wall's pixel {corner:?} vs {wall_corner:?}"
            );
            // The pane is far clearer than the wall's mist would make it: the old
            // number is not what is drawn.
            assert!(
                largest_difference(got, before) > 0.04,
                "{name} {tier_name}: near the old number {before:?}: {got:?}"
            );
            assert!(largest_difference(through, wall) > 0.04, "the pane shows over the wall");
            // Through a grid the numbers are the grid's: the froxel volume's own integral,
            // read in slice space, at the pane's depth and at the wall's, as the atmosphere
            // pass reads it for the wall.
            if tier > 0 {
                let volume = rig.volume(VolumeLayer::Integrated);
                let at = |depth: f64| froxel_integral(&volume, rig.size, (32, 32), depth, 90.0);
                let (scatter_pane, optical_pane) = at(pane.near);
                let (scatter_wall, optical_wall) = at(pane.far);
                let (t_pane, t_wall) = ((-optical_pane).exp(), (-optical_wall).exp());
                let wall_seen: V3 = std::array::from_fn(|c| rgb(WALL)[c] * t_wall + scatter_wall[c]);
                let pane_seen: V3 = std::array::from_fn(|c| f64::from(PANE[c]) * t_pane + scatter_pane[c]);
                let from_grid = over(pane_seen, f64::from(PANE[3]), wall_seen);
                assert!(
                    largest_difference(got, from_grid) < 1.3 / 255.0,
                    "{name} {tier_name}: the pane {got:?} against the froxel volume's own numbers {from_grid:?}"
                );
            }
        }
    }
}

#[test]
fn without_a_mist_a_translucent_quad_is_the_plain_blend() {
    let Some(mut rig) = Rig::new("pane-clear", 64, &[flat(WALL), glass(PANE)]) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([4.0, 1.5, 0.0], [4.0, 1.5, -10.0], 1.0, 0.5);
    let pane = Pane { near: 4.0, far: 40.0 };
    // No mist at all, and clouds that cover nothing: the identity.
    let mist = Mist { density: 0.0, height: 1.0, mean: MEAN };
    for tier in [0, 2] {
        let mut frame = pane_frame(&view, &mist, tier, &pane);
        frame.fog.distance = 0.0;
        let picture = rig.settle(&frame, 2);
        let expected = over([f64::from(PANE[0]), f64::from(PANE[1]), f64::from(PANE[2])], 0.5, rgb(WALL));
        let got = rig.pixel(&picture, 32, 32);
        assert!(largest_difference(got, expected) < LEVELS, "tier {tier}: {got:?} vs {expected:?}");
        assert!(largest_difference(rig.pixel(&picture, 3, 3), rgb(WALL)) < LEVELS);
        assert_eq!(rig.renderer.stats().volume_froxels, 0);
    }
}

fn translation(x: f64, y: f64, z: f64) -> Matrix {
    scale_at(1.0, 1.0, 1.0, x as f32, y as f32, z as f32)
}

/// An emitter of one particle (born at 0 s, at the emitter's age of half a second)
/// `size` across, of `color` and its alpha, at `place`.
fn particle(place: V3, size: f32, color: [f32; 4], additive: bool) -> Emitter {
    let color = Color { r: color[0], g: color[1], b: color[2], a: color[3] };
    Emitter {
        material: 2,
        transform: translation(place[0], place[1], place[2]),
        seed: 1,
        time: 0.5,
        rate: 1.0,
        lifetime: 10.0,
        size_start: size,
        size_end: size,
        color_start: color,
        color_end: color,
        additive,
        ..Default::default()
    }
}

#[test]
fn a_particle_is_misted_at_its_own_depth_covering_or_adding() {
    let materials = [flat(WALL), glass(PANE), Material { shader: "none.wgsl", color: [1.0, 1.0, 1.0, 1.0], extra: "" }];
    let Some(mut rig) = Rig::new("particle", 64, &materials) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([4.0, 1.5, 0.0], [4.0, 1.5, -10.0], 1.0, 0.5);
    let mist = Mist { density: 0.03, height: 2000.0, mean: MEAN };
    let (near, far) = (4.0, 40.0);
    let colour = [0.15, 0.55, 0.25, 0.5f32];
    let tint = [f64::from(colour[0]), f64::from(colour[1]), f64::from(colour[2])];
    let alpha = f64::from(colour[3]);
    for (tier_name, tier) in [("high", 2), ("analytic", 0)] {
        for additive in [false, true] {
            let mut frame = view_frame(&view);
            frame.fog = Fog { density: 0.03, height: 2000.0, distance: 90.0, ..Fog::default() };
            frame.quality.volumetrics = tier;
            frame.sky = sky_with_mean(MEAN);
            frame.light_color = vector(0.0, 0.0, 0.0);
            frame.draws = vec![draw(0, 0, view.facing(far, 200.0))];
            // Wider than the picture at its depth, so the pixel is its middle.
            frame.emitters = vec![particle([4.0, 1.5, -near], 8.0, colour, additive)];
            let picture = rig.settle(&frame, 3);
            save(
                &format!("translucency-particle-{tier_name}-{}.png", if additive { "added" } else { "covering" }),
                &picture,
                rig.size,
            );
            let (direction, slant) = view.ray(rig.size, 32, 32);
            let (t1, l1) = mist.between(view.eye, direction, near * slant);
            let wall = mist.fog(rgb(WALL), view.eye, direction, far * slant);
            let (expected, old) = if additive {
                // Light that only adds: attenuated by the mist before it, none of its own.
                (
                    std::array::from_fn(|c| wall[c] + alpha * tint[c] * t1),
                    // What the old pipeline made of it: the sum misted at the wall's depth.
                    mist.fog(std::array::from_fn(|c| rgb(WALL)[c] + alpha * tint[c]), view.eye, direction, far * slant),
                )
            } else {
                (
                    over(std::array::from_fn(|c| tint[c] * t1 + l1[c]), alpha, wall),
                    mist.fog(over(tint, alpha, rgb(WALL)), view.eye, direction, far * slant),
                )
            };
            let got = rig.pixel(&picture, 32, 32);
            eprintln!(
                "particle {tier_name} {}: {got:.3?} expected {expected:.3?}, old pipeline {old:.3?}",
                if additive { "added" } else { "covering" }
            );
            // The soft dot's alpha at its middle is 1 to a part in a thousand.
            assert!(
                largest_difference(got, expected) < LEVELS + 0.002,
                "{tier_name} additive {additive}: {got:?} vs {expected:?}"
            );
            assert!(
                largest_difference(got, old) > 0.03,
                "{tier_name} additive {additive}: the old number {old:?} is not what is drawn: {got:?}"
            );
        }
    }
}

/// A projection of the box x and y from 0 to `size` (y up), by the depth 0 to
/// `far` ahead of an eye looking down -z, for a view of the identity: a pixel is
/// a unit, so that quads laid on whole units cover whole pixels.
fn ortho(size: f32, far: f32) -> Matrix {
    [2.0 / size, 0.0, 0.0, 0.0, 0.0, 2.0 / size, 0.0, 0.0, 0.0, 0.0, -1.0 / far, 0.0, -1.0, -1.0, 0.0, 1.0]
}

const RED: [f32; 3] = [0.8, 0.2, 0.2];
const BLUE: [f32; 3] = [0.2, 0.3, 0.8];
const GREEN: [f32; 3] = [0.3, 0.7, 0.3];
const CLEAR: [f32; 3] = [0.1, 0.12, 0.14];
/// How far from the eye each surface of `corners` is.
const RED_AT: f32 = 20.0;
const BLUE_AT: f32 = 40.0;
const GREEN_AT: f32 = 60.0;

/// Which surface of `corners` covers the middle of pixel (x, y) of a picture of 64
/// (a wedge, red, nearest, whose slanted side crosses the picture; blue and green
/// squares side by side to its right; and nothing on its far side).
fn corners_at(x: u32, y: u32) -> Option<(usize, f32)> {
    let (px, py) = (f64::from(x) + 0.5, 64.0 - (f64::from(y) + 0.5));
    if px / 40.0 + py / 64.0 <= 1.0 {
        Some((0, RED_AT))
    } else if px >= 32.0 && py < 32.0 {
        Some((1, BLUE_AT))
    } else if px >= 32.0 {
        Some((2, GREEN_AT))
    } else {
        None
    }
}

/// Wedge, blue square and green square, nearest first, over a clear colour.
fn corners(msaa: u32) -> RenderFrame {
    let mut frame = frame(vec![
        draw(1, 0, scale_at(40.0, 64.0, 1.0, 0.0, 0.0, -RED_AT)),
        draw(0, 1, scale_at(16.0, 16.0, 1.0, 48.0, 16.0, -BLUE_AT)),
        draw(0, 2, scale_at(16.0, 16.0, 1.0, 48.0, 48.0, -GREEN_AT)),
    ]);
    frame.projection = ortho(64.0, 100.0);
    frame.clear = Color { r: CLEAR[0], g: CLEAR[1], b: CLEAR[2], a: 1.0 };
    frame.quality.msaa = msaa;
    frame
}

const CORNER_COLOURS: [[f32; 3]; 3] = [RED, BLUE, GREEN];

/// A material that draws what the library says (`PROBE`'s modes) over everything, in
/// the translucent pass where it can read the opaque picture.
fn probe() -> Material {
    Material { shader: "probe.wgsl", color: [1.0, 1.0, 1.0, 1.0], extra: r#""overlay":true"# }
}

/// The frame with a full-picture probe in front of everything, in `mode`.
fn probed(mut frame: RenderFrame, mode: f32, level: f32) -> RenderFrame {
    // At 0.5 ahead of an eye: nearer than any surface.
    frame.draws.push(with_parameters(draw(0, 3, scale_at(32.0, 32.0, 1.0, 32.0, 32.0, -0.5)), [level, mode, 0.0, 0.0]));
    frame
}

fn corner_materials() -> Vec<Material> {
    vec![flat(RED), flat(BLUE), flat(GREEN), probe()]
}

/// The corners picture's level 0 in f64, row by row from the top.
fn corners_level0() -> Vec<V3> {
    (0..64u32 * 64)
        .map(|index| match corners_at(index % 64, index / 64) {
            Some((surface, _)) => rgb(CORNER_COLOURS[surface]),
            None => rgb(CLEAR),
        })
        .collect()
}

/// A bilinear read of an image of `size` square texels at (u, v), the way a
/// sampler with clamped edges reads it.
fn bilinear(image: &[V3], size: usize, u: f64, v: f64) -> V3 {
    let (x, y) = (u * size as f64 - 0.5, v * size as f64 - 0.5);
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let texel = |tx: f64, ty: f64| {
        image[(ty.clamp(0.0, size as f64 - 1.0) as usize) * size + tx.clamp(0.0, size as f64 - 1.0) as usize]
    };
    let (a, b, c, d) = (texel(x0, y0), texel(x0 + 1.0, y0), texel(x0, y0 + 1.0), texel(x0 + 1.0, y0 + 1.0));
    std::array::from_fn(|channel| {
        (a[channel] * (1.0 - fx) + b[channel] * fx) * (1.0 - fy) + (c[channel] * (1.0 - fx) + d[channel] * fx) * fy
    })
}

/// The chain's next level from `above`: the (1, 3, 3, 1) / 8 tent over 4 x 4 texels,
/// as four bilinear reads three quarters of a texel out, edges repeating.
fn halve(above: &[V3], size: usize) -> Vec<V3> {
    let half = size / 2;
    let reach = 0.75 / size as f64;
    (0..half * half)
        .map(|index| {
            let (u, v) = (((index % half) as f64 + 0.5) / half as f64, ((index / half) as f64 + 0.5) / half as f64);
            let mut sum = [0.0; 3];
            for (dx, dy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                let tap = bilinear(above, size, u + dx * reach, v + dy * reach);
                for channel in 0..3 {
                    sum[channel] += tap[channel] * 0.25;
                }
            }
            sum
        })
        .collect()
}

#[test]
fn the_opaque_picture_reads_back_through_its_chain_of_blurred_levels() {
    let Some(mut rig) = Rig::new("chain", 64, &corner_materials()) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut levels = vec![corners_level0()];
    for level in 1..=6 {
        let size = 64usize >> (level - 1);
        levels.push(halve(&levels[level - 1], size));
    }
    for (level, expected) in levels.iter().enumerate() {
        let size = 64usize >> level;
        let picture = rig.settle(&probed(corners(1), 0.0, level as f32), 2);
        if level == 0 || level == 3 {
            save(&format!("translucency-opaque-level-{level}.png"), &picture, rig.size);
        }
        let mut worst = 0.0f64;
        for y in 0..64u32 {
            for x in 0..64u32 {
                let reference = bilinear(expected, size, (f64::from(x) + 0.5) / 64.0, (f64::from(y) + 0.5) / 64.0);
                worst = worst.max(largest_difference(rig.pixel(&picture, x, y), reference));
            }
        }
        eprintln!("level {level}: worst difference from the f64 chain {:.4} ({:.1} of 255)", worst, worst * 255.0);
        // Level 0 is the picture itself, a pixel at a time; each level above it rounds
        // to half floats and one more tent, by a hair.
        let limit = if level == 0 { 2.0 / 255.0 } else { 3.0 / 255.0 };
        assert!(worst < limit, "level {level}: {worst}");
    }
    // The last level is one texel, and the chain's length is known to the shader.
    let picture = rig.settle(&probed(corners(1), 3.0, 0.0), 2);
    let last = rig.pixel(&picture, 10, 10)[0] * 10.0;
    assert!((last - 6.0).abs() < 0.1, "a picture of 64 has levels 0 to 6: {last}");
}

#[test]
fn the_nearest_depth_of_each_pixel_is_the_distance_of_the_opaque_surface_behind_it() {
    let Some(mut rig) = Rig::new("nearest", 64, &corner_materials()) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let pure: Vec<[f64; 3]> = [RED, BLUE, GREEN, CLEAR].map(rgb).to_vec();
    for msaa in [1, 4] {
        let picture = rig.settle(&probed(corners(msaa), 1.0, 0.0), 2);
        let colours = rig.settle(&corners(msaa), 2);
        save(&format!("translucency-opaque-depth-{msaa}x.png"), &picture, rig.size);
        let byte = |distance: f32| f64::from(distance) / 100.0 * 255.0;
        let candidates = [byte(RED_AT), byte(BLUE_AT), byte(GREEN_AT), 0.9 * 255.0];
        let mut mixed = 0;
        for y in 0..64u32 {
            for x in 0..64u32 {
                // The distance over 100, as a byte, in the red channel.
                let got = f64::from(picture[(y * 64 + x) as usize][0]);
                // Never between two surfaces' depths: the nearest sample's, not the average.
                assert!(
                    candidates.iter().any(|candidate| (got - candidate).abs() <= 1.0),
                    "msaa {msaa}: pixel ({x}, {y}) holds {got}, the depth of no surface"
                );
                let colour = rig.pixel(&colours, x, y);
                let is_mixed = pure.iter().all(|pure| largest_difference(colour, *pure) > 3.0 / 255.0);
                if msaa == 1 {
                    // A pixel whose middle a surface covers has its depth.
                    let want = corners_at(x, y).map_or(0.9 * 255.0, |(_, distance)| byte(distance));
                    assert!((got - want).abs() <= 1.0, "({x}, {y}): {got} for {want}");
                } else if is_mixed {
                    // A pixel the wedge's edge splits shows a mix of its colour and what
                    // is behind, and the wedge is the nearest surface there.
                    mixed += 1;
                    assert!((got - byte(RED_AT)).abs() <= 1.0, "({x}, {y}) is a mix and has depth {got}");
                }
            }
        }
        if msaa > 1 {
            eprintln!("{mixed} pixels along the wedge's edge, each at its depth");
            assert!(mixed > 20, "{mixed}");
        }
    }
}

#[test]
fn a_world_point_finds_its_pixel_through_the_projection_the_scene_drew_with_jitter_included() {
    let Some(mut rig) = Rig::new("jitter", 64, &corner_materials()) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 1.0, 0.5);
    for taa in [false, true] {
        let mut frame = view_frame(&view);
        frame.quality.taa = taa;
        frame.quality.msaa = 4;
        frame.draws = vec![
            draw(0, 0, view.facing(30.0, 80.0)),
            with_parameters(draw(0, 3, view.facing(5.0, 2.4)), [0.0, 2.0, 0.0, 0.0]),
        ];
        // Eight frames walk the jitter through its sequence: the offset (in tenths
        // of a pixel from a half) stays at zero in every one.
        for frame_number in 0..9 {
            let picture = rig.draw(&frame);
            for (x, y) in [(20, 20), (32, 32), (40, 25)] {
                let [r, g, _, _] = picture[(y * 64 + x) as usize];
                // Within a twentieth of a pixel (a byte step is 0.04).
                assert!(
                    (i32::from(r) - 128).abs() <= 2 && (i32::from(g) - 128).abs() <= 2,
                    "taa {taa} frame {frame_number}: ({x}, {y}) is {r}, {g}"
                );
            }
        }
    }
}

#[test]
fn the_library_is_the_identity_where_no_opaque_picture_is_read() {
    // The same wall drawn by a shader that misted it itself and by one that did not,
    // in an opaque pass, which has stand-ins to read: bit for bit the same picture,
    // since the atmosphere pass is what mists an opaque surface.
    let opaque_probe = Material { shader: "probe.wgsl", color: [1.0, 1.0, 1.0, 1.0], extra: "" };
    let materials = [flat(WALL), glass([WALL[0], WALL[1], WALL[2], 1.0]), opaque_probe, probe()];
    let Some(mut rig) = Rig::new("identity", 64, &materials) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([4.0, 1.5, 0.0], [4.0, 1.5, -10.0], 1.0, 0.5);
    let mist = Mist { density: 0.03, height: 2000.0, mean: MEAN };
    for tier in [0, 2] {
        // Two renderers, each as many frames in: the volume blends with its history.
        let mut pictures = Vec::new();
        for material in [0, 1] {
            let mut fresh = Rig::new("identity-fresh", 64, &materials).unwrap();
            let mut frame = pane_frame(&view, &mist, tier, &Pane { near: 4.0, far: 40.0 });
            frame.draws = vec![draw(0, material, view.facing(40.0, 200.0))];
            pictures.push(fresh.settle(&frame, 3));
            assert!(!fresh.renderer.passes().contains(&"backdrop"));
        }
        assert_eq!(pictures[0], pictures[1], "tier {tier}: an opaque pass has the identity for fog_apply");
    }
    // An opaque surface asking whether it reads an opaque picture is told no (red),
    // an overlay, in the translucent pass, yes (green).
    for (material, answer) in [(2, [128, 0]), (3, [0, 128])] {
        let mut frame = view_frame(&view);
        frame.draws = vec![with_parameters(draw(0, material, view.facing(10.0, 100.0)), [0.0, 4.0, 0.0, 0.0])];
        let picture = rig.settle(&frame, 2);
        let [r, g, _, _] = picture[(32 * 64 + 32) as usize];
        assert!(r.abs_diff(answer[0]) <= 1 && g.abs_diff(answer[1]) <= 1, "material {material}: {r} {g}");
    }
}

/// A hazy room seen from the near end: sky, a mist through a grid or not, clouds,
/// ambient occlusion and the rest of what the passes after the scene do, over a
/// wall, a pillar, and a wedge whose slanted edge crosses them: what frames drew
/// before translucent surfaces were drawn in a pass of their own.
fn hazy(view: &Camera, tier: u32, msaa: u32) -> RenderFrame {
    let mut frame = view_frame(view);
    frame.fog = Fog { density: 0.02, height: 30.0, distance: 90.0, sky: 0.6, ..Fog::default() };
    frame.quality.volumetrics = tier;
    frame.quality.msaa = msaa;
    frame.quality.sky = true;
    frame.quality.occlusion = true;
    frame.quality.bloom = true;
    frame.quality.fxaa = true;
    frame.sky = Sky {
        zenith: vector(0.25, 0.45, 0.85),
        horizon: vector(0.7, 0.78, 0.9),
        ground: vector(0.3, 0.3, 0.32),
        sun: vector(0.3, 0.8, 0.2),
        sh0: vector(2.0, 2.2, 2.6),
        ..Sky::default()
    };
    frame.clouds = Clouds { cover: 0.5, drift: Drift { x: 3.0, z: 1.0 } };
    frame.light_color = vector(0.6, 0.55, 0.5);
    frame.clear = Color { r: 0.7, g: 0.78, b: 0.9, a: 1.0 };
    frame.draws = vec![
        draw(0, 0, view.placed(60.0, 40.0, (0.0, -12.0))),
        draw(0, 3, view.placed(25.0, 3.0, (-5.0, 1.0))),
        draw(1, 2, view.placed(12.0, 6.0, (1.5, -1.0))),
    ];
    frame
}

fn hazy_materials() -> Vec<Material> {
    vec![flat(WALL), glass([0.2, 0.7, 0.3, 0.5]), flat(RED), flat(BLUE)]
}

/// Draws `frame` on a renderer made for it, `count` times, and returns the last picture.
fn drawn_fresh(
    name: &str,
    materials: &[Material],
    frame: &RenderFrame,
    count: usize,
) -> (Vec<[u8; 4]>, Vec<&'static str>) {
    let mut rig = Rig::new(name, 64, materials).unwrap();
    rig.load(frame.quality.msaa);
    let picture = rig.settle(frame, count);
    assert_eq!(rig.submits, count, "every frame was drawn whole: the pictures are of the same frames");
    (picture, rig.renderer.passes())
}

#[test]
fn a_frame_whose_translucent_surface_shows_nothing_draws_as_if_it_were_not_there() {
    if Rig::new("hazy-probe", 64, &hazy_materials()).is_none() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let view = camera([3.0, 2.0, 18.0], [0.0, 1.0, -10.0], 1.0, 0.5);
    let configurations = [(2, 4, false), (0, 4, false), (2, 1, false), (2, 4, true), (0, 1, true)];
    let plain_frame = |tier: u32, msaa: u32, taa: bool| {
        let mut plain = hazy(&view, tier, msaa);
        plain.quality.taa = taa;
        plain
    };
    let frames_of = |taa: bool| if taa { 12 } else { 3 };
    // The frames without the pane, which the pipeline that draws one picture drew the
    // same before there was a pane to draw: the pictures are saved to hold them to
    // that build's (the same test of it).
    let mut plains = Vec::new();
    for (tier, msaa, taa) in configurations {
        let (before, passes) =
            drawn_fresh("hazy-plain", &hazy_materials(), &plain_frame(tier, msaa, taa), frames_of(taa));
        save(&format!("translucency-hazy-plain-tier{tier}-msaa{msaa}-taa{taa}.png"), &before, 64);
        assert!(!passes.contains(&"backdrop"), "{passes:?}");
        plains.push(before);
    }
    for ((tier, msaa, taa), before) in configurations.into_iter().zip(plains) {
        // The same room with a pane too small to cover the middle of a pixel, or a
        // sample, in front: translucent, so drawn in the translucent pass over the
        // opaque picture, and invisible, so changing nothing.
        let mut with_pane = plain_frame(tier, msaa, taa);
        with_pane.draws.push(draw(0, 1, view.facing(5.0, 1.0e-4)));
        let (after, split) = drawn_fresh("hazy-pane", &hazy_materials(), &with_pane, frames_of(taa));
        assert!(split.contains(&"backdrop") && split.contains(&"opaque depth"), "{split:?}");
        save(&format!("translucency-hazy-tier{tier}-msaa{msaa}-taa{taa}.png"), &after, 64);
        let differing = before.iter().zip(&after).filter(|(a, b)| a != b).count();
        eprintln!("tier {tier} msaa {msaa} taa {taa}: {differing} of {} pixels differ; passes {split:?}", before.len());
        assert_eq!(before, after, "tier {tier} msaa {msaa} taa {taa}: the split frame is the one-picture frame");
    }
}

#[test]
fn turning_a_translucent_surface_on_and_off_keeps_the_targets_and_the_temporal_history() {
    let Some(mut rig) = Rig::new("flip", 64, &hazy_materials()) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    rig.load(4);
    let view = camera([3.0, 2.0, 18.0], [0.0, 1.0, -10.0], 1.0, 0.5);
    let mut plain = hazy(&view, 2, 4);
    plain.quality.taa = true;
    let mut with_pane = plain.clone();
    with_pane.draws.push(draw(0, 1, view.facing(5.0, 1.0e-4)));
    // Frames 4 to 7 have the invisible pane and the others do not; the same frames
    // without it ever are the same picture, so the history (TAA's, the mist's) has
    // not been dropped when the passes changed.
    let (reference, _) = drawn_fresh("flip-reference", &hazy_materials(), &plain, 12);
    let mut last = Vec::new();
    for frame in 0..12 {
        last = rig.draw(if (4..8).contains(&frame) { &with_pane } else { &plain });
        let names = rig.renderer.passes();
        assert_eq!(names.contains(&"backdrop"), (4..8).contains(&frame), "frame {frame}: {names:?}");
    }
    assert_eq!(rig.submits, 12);
    assert_eq!(last, reference);
}

#[test]
fn passes_list_the_opaque_picture_the_translucent_pass_and_what_reads_it() {
    let Some(mut rig) = Rig::new("passes", 64, &hazy_materials()) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([3.0, 2.0, 18.0], [0.0, 1.0, -10.0], 1.0, 0.5);
    let mut frame = view_frame(&view);
    frame.draws = vec![draw(0, 0, view.facing(40.0, 200.0))];
    rig.settle(&frame, 2);
    assert_eq!(rig.renderer.passes(), ["present"], "nothing to mist, nothing translucent: the picture as it was drawn");
    // A picture of 64 has 7 levels, so 6 passes make the chain.
    let chain: Vec<&str> = ["atmosphere", "opaque depth"].into_iter().chain(["opaque mip"; 6]).collect();
    frame.draws.push(draw(0, 1, view.facing(5.0, 2.0)));
    rig.settle(&frame, 2);
    assert_eq!(rig.renderer.passes(), [chain.clone(), vec!["backdrop", "present"]].concat());
    // Occlusion, then TAA and the bloom's levels after the translucent pass.
    frame.quality.occlusion = true;
    frame.quality.taa = true;
    frame.quality.bloom = true;
    rig.settle(&frame, 2);
    let expected = [
        vec!["occlusion"],
        chain,
        vec!["backdrop", "taa"],
        vec!["bloom down"; 4],
        vec!["bloom up"; 3],
        vec!["present"],
    ]
    .concat();
    assert_eq!(rig.renderer.passes(), expected);
    // Without the pane the frame is the one picture again.
    frame.draws.pop();
    rig.settle(&frame, 2);
    let names = rig.renderer.passes();
    assert!(
        !names.contains(&"backdrop") && !names.contains(&"opaque depth") && names.contains(&"occlusion"),
        "{names:?}"
    );
}

/// How much of the way from `behind` to `front` a pixel's colour is, by its largest
/// channel's difference.
fn share(pixel: V3, behind: V3, front: V3) -> f64 {
    let channel =
        (0..3).max_by(|&a, &b| (front[a] - behind[a]).abs().total_cmp(&(front[b] - behind[b]).abs())).unwrap();
    (pixel[channel] - behind[channel]) / (front[channel] - behind[channel])
}

#[test]
fn edges_keep_their_samples_whether_opaque_or_translucent() {
    let materials = [flat(RED), flat(BLUE), flat(GREEN), glass([0.8, 0.8, 0.2, 0.5])];
    let Some(mut rig) = Rig::new("edges", 64, &materials) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    // The red wedge (opaque) and over the clear colour beside it a translucent
    // wedge whose slanted side crosses it: both at 4 samples a pixel.
    let mut frame = corners(4);
    frame.draws.push(draw(1, 3, scale_at(16.0, 10.0, 1.0, 16.0, 44.0, -10.0)));
    let picture = rig.settle(&frame, 3);
    save("translucency-edges.png", &picture, rig.size);
    let (clear, red) = (rgb(CLEAR), rgb(RED));
    let covered = over([0.8, 0.8, 0.2], 0.5, clear);
    let (mut opaque_edge, mut translucent_edge) = (Vec::new(), Vec::new());
    for y in 0..64u32 {
        for x in 0..32u32 {
            let pixel = rig.pixel(&picture, x, y);
            let (px, py) = (f64::from(x) + 0.5, 64.0 - (f64::from(y) + 0.5));
            // The red wedge's side, away from the translucent wedge.
            if py < 40.0 && (px / 40.0 + py / 64.0 - 1.0).abs() < 0.03 {
                opaque_edge.push(share(pixel, clear, red));
            }
            // The translucent wedge's side: its pixels hold the clear colour, the
            // wedge's blend, or a share of the one over the other.
            if (44.0..54.0).contains(&py)
                && (px >= 16.0)
                && ((px - 16.0) / 16.0 + (py - 44.0) / 10.0 - 1.0).abs() < 0.06
            {
                translucent_edge.push(share(pixel, clear, covered));
            }
        }
    }
    for (name, edge) in [("opaque", &opaque_edge), ("translucent", &translucent_edge)] {
        let mixed: Vec<f64> = edge.iter().copied().filter(|share| (0.1..0.9).contains(share)).collect();
        eprintln!("{name} edge: {} pixels along it, {} of them mixed: {mixed:.2?}", edge.len(), mixed.len());
        assert!(mixed.len() >= 5, "{name}: an edge of whole pixels only is aliased: {edge:.2?}");
        // Four samples a pixel: quarters of the way, to a part in forty.
        for share in &mixed {
            let nearest = (share * 4.0).round() / 4.0;
            assert!((share - nearest).abs() < 0.04, "{name}: {share} is not a number of samples of four");
        }
        assert!(mixed.iter().any(|share| (share - 0.5).abs() > 0.1), "{name}: more than one coverage");
    }
}

#[test]
fn screen_space_reflections_and_a_translucent_surface_draw_together() {
    let Some(mut rig) = Rig::new("ssr", 64, &[flat(WALL), glass(PANE)]) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([4.0, 1.5, 0.0], [4.0, 1.5, -10.0], 1.0, 0.5);
    let mist = Mist { density: 0.03, height: 2000.0, mean: MEAN };
    let pane = Pane { near: 4.0, far: 40.0 };
    let mut frame = pane_frame(&view, &mist, 2, &pane);
    frame.quality.ssr = 1;
    let picture = rig.settle(&frame, 12);
    // SSR brings TAA with it, and the passes of the translucent frame.
    let names = rig.renderer.passes();
    assert!(names.contains(&"taa") && names.contains(&"backdrop"), "{names:?}");
    let (through, _, _) = pane_references(&rig, &view, &mist, &pane, (32, 32));
    let (got, around) = (rig.pixel(&picture, 32, 32), rig.pixel(&picture, 3, 3));
    assert!(largest_difference(got, through) < 1.5 * LEVELS, "{got:?} vs {through:?}");
    let (_, corner, _) = pane_references(&rig, &view, &mist, &pane, (3, 3));
    assert!(largest_difference(around, corner) < 1.5 * LEVELS, "{around:?} vs {corner:?}");
}

#[test]
fn auto_exposure_meters_the_picture_with_its_translucent_surfaces_in_it() {
    let grey = [0.5f32; 3];
    let materials = [flat(grey), glass([0.9, 0.9, 0.9, 0.8])];
    let Some(mut rig) = Rig::new("exposure", 64, &materials) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    rig.load(1);
    let view = camera([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 1.0, 0.5);
    let mut frame = view_frame(&view);
    frame.exposure = Exposure { auto: true, cut: true, dt: 1.0 / 60.0, ..Default::default() };
    let linear =
        |display: f64| if display <= 0.04045 { display / 12.92 } else { ((display + 0.055) / 1.055).powf(2.4) };
    let stops = |display: f64| (0.18 / linear(display)).log2();
    let mut seen = Vec::new();
    for with_pane in [false, true] {
        frame.draws = vec![draw(0, 0, view.facing(40.0, 200.0))];
        if with_pane {
            frame.draws.push(draw(0, 1, view.facing(5.0, 200.0)));
        }
        // The metering comes back a few frames later: draw until it is the frame's.
        let mut latest = None;
        for _ in 0..30 {
            rig.draw(&frame);
            latest = rig.renderer.stats().exposure;
        }
        seen.push(latest.expect("a metered frame's exposure comes back"));
    }
    let expected = [stops(0.5), stops(0.8 * 0.9 + 0.2 * 0.5)];
    eprintln!(
        "auto exposure in stops: {seen:.3?} for a grey wall alone, and behind a bright pane; expected {expected:.3?}"
    );
    assert!((f64::from(seen[0]) - expected[0]).abs() < 0.07, "{} vs {}", seen[0], expected[0]);
    assert!((f64::from(seen[1]) - expected[1]).abs() < 0.07, "metered after the pane: {} vs {}", seen[1], expected[1]);
}

#[test]
fn scene_filters_see_the_picture_with_its_translucent_surfaces_in_it() {
    let filter = Material { shader: "invert.wgsl", color: [1.0; 4], extra: r#""scene_filter":true"# };
    let materials = [flat(WALL), glass(PANE), filter];
    let Some(mut rig) = Rig::new("filter", 64, &materials) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([4.0, 1.5, 0.0], [4.0, 1.5, -10.0], 1.0, 0.5);
    let mut frame = view_frame(&view);
    frame.draws = vec![draw(0, 0, view.facing(40.0, 200.0)), draw(0, 1, view.facing(4.0, 1.0))];
    frame.sprites = vec![Sprite {
        x: 0.0,
        y: 0.0,
        width: 64.0,
        height: 64.0,
        u0: 0.0,
        v0: 0.0,
        u1: 1.0,
        v1: 1.0,
        color: [1.0; 4],
        material: 2,
        parameters: [0.0; 4],
    }];
    let picture = rig.settle(&frame, 3);
    save("translucency-filtered.png", &picture, rig.size);
    let expected =
        over([f64::from(PANE[0]), f64::from(PANE[1]), f64::from(PANE[2])], 0.5, rgb(WALL)).map(|value| 1.0 - value);
    let got = rig.pixel(&picture, 32, 32);
    assert!(largest_difference(got, expected) < LEVELS, "{got:?} vs {expected:?}");
    let corner = rig.pixel(&picture, 3, 3);
    assert!(largest_difference(corner, rgb(WALL).map(|value| 1.0 - value)) < LEVELS, "{corner:?}");
}

/// How long the GPU takes over a frame of 1280 by 720 at 4 samples a pixel, in
/// milliseconds (the median of the frames timed), for a room with a sky, mist,
/// occlusion, bloom and TAA, drawn with and without a translucent pane and a
/// plume of particles. Run with `-- --ignored --nocapture`.
#[test]
#[ignore = "a measurement, not a witness"]
fn the_translucent_pass_costs_this_much_at_1280_by_720() {
    let mut materials = hazy_materials();
    materials.push(Material { shader: "none.wgsl", color: [1.0; 4], extra: "" });
    let Some(mut rig) = Rig::new("cost", 64, &materials) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    rig.target = Offscreen(rig.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: 1280, height: 720, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }));
    rig.renderer.time_passes(true);
    let view = camera([3.0, 2.0, 18.0], [0.0, 1.0, -10.0], 16.0 / 9.0, 0.5);
    for (label, taa, translucent) in [
        ("plain, no TAA", false, false),
        ("plain, TAA", true, false),
        ("with a pane, no TAA", false, true),
        ("with a pane, TAA", true, true),
        ("with a pane and particles, TAA", true, true),
    ] {
        let mut frame = hazy(&view, 2, 4);
        frame.quality.taa = taa;
        frame.draws = vec![
            draw(0, 0, view.placed(60.0, 60.0, (0.0, -12.0))),
            draw(0, 3, view.placed(25.0, 3.0, (-5.0, 1.0))),
            draw(1, 2, view.placed(12.0, 6.0, (1.5, -1.0))),
        ];
        if translucent {
            frame.draws.push(draw(0, 1, view.facing(6.0, 3.0)));
        }
        if label.contains("particles") {
            frame.emitters = vec![particle([0.0, 1.0, 4.0], 0.5, [0.9, 0.6, 0.2, 0.4], false)];
            frame.emitters[0].rate = 2000.0;
            frame.emitters[0].lifetime = 1.0;
            frame.emitters[0].radius = 3.0;
            frame.emitters[0].velocity = vector(0.0, 1.0, 0.0);
            frame.emitters[0].time = 2.0;
            frame.emitters[0].material = 4;
        }
        let mut samples: Vec<String> = Vec::new();
        let mut totals: Vec<f32> = Vec::new();
        for index in 0..90 {
            rig.draw(&frame);
            if let Some(gpu) = rig.renderer.stats().gpu.filter(|_| index >= 40) {
                totals.push(gpu.total);
                samples.push(format!("{gpu:?}"));
            }
            std::thread::sleep(std::time::Duration::from_millis(3));
        }
        totals.sort_by(f32::total_cmp);
        let median = totals.get(totals.len() / 2).copied().unwrap_or(f32::NAN);
        println!(
            "{label}: total {median:.3} ms (median of {}); a frame: {}",
            totals.len(),
            samples.last().cloned().unwrap_or_default()
        );
    }
}

#[test]
fn the_scene_a_sprite_reads_is_the_picture_with_its_translucent_surfaces_and_their_mist() {
    // The inverting shader as an ordinary sprite (no scene filter): binding 4 is the
    // resolved scene, which in a frame with a translucent surface has it, misted.
    let sprite = Material { shader: "invert.wgsl", color: [1.0; 4], extra: "" };
    let materials = [flat(WALL), glass(PANE), sprite];
    let Some(mut rig) = Rig::new("sprite-scene", 64, &materials) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([4.0, 1.5, 0.0], [4.0, 1.5, -10.0], 1.0, 0.5);
    let mist = Mist { density: 0.03, height: 2000.0, mean: MEAN };
    let pane = Pane { near: 4.0, far: 40.0 };
    let mut frame = pane_frame(&view, &mist, 2, &pane);
    frame.sprites = vec![Sprite {
        x: 0.0,
        y: 0.0,
        width: 64.0,
        height: 64.0,
        u0: 0.0,
        v0: 0.0,
        u1: 1.0,
        v1: 1.0,
        color: [1.0; 4],
        material: 2,
        parameters: [0.0; 4],
    }];
    let picture = rig.settle(&frame, 3);
    let (through, _, _) = pane_references(&rig, &view, &mist, &pane, (32, 32));
    let got = rig.pixel(&picture, 32, 32);
    let expected = through.map(|value| 1.0 - value);
    assert!(largest_difference(got, expected) < LEVELS, "{got:?} vs {expected:?}");
    let (_, corner, _) = pane_references(&rig, &view, &mist, &pane, (3, 3));
    assert!(largest_difference(rig.pixel(&picture, 3, 3), corner.map(|value| 1.0 - value)) < LEVELS);
}

#[test]
fn decals_lie_on_the_opaque_surfaces_beneath_translucent_ones() {
    // A wall that takes decals, a decal painted on it, and a pane over part of the
    // decal. The decals' passes run in the scene pass, ahead of the translucent one.
    let wall = Material { shader: "flat.wgsl", color: [WALL[0], WALL[1], WALL[2], 1.0], extra: r#""decals":true"# };
    let paint = Material { shader: "none.wgsl", color: [0.6, 0.2, 0.2, 1.0], extra: r#""texture":"decal.png""# };
    let materials = [wall, glass(PANE), paint];
    let Some(mut rig) = Rig::new("decals", 64, &materials) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([0.0, 0.0, 0.0], [0.0, 0.0, -10.0], 1.0, 0.5);
    let mut frame = view_frame(&view);
    frame.quality.msaa = 4;
    frame.draws = vec![draw(0, 0, view.facing(20.0, 100.0))];
    // A box of 3 across, its y axis (the way it projects) toward the eye, centred on the wall.
    let at_wall = [view.eye[0] + 2.0 * view.right[0], 0.0, -20.0];
    let along = |axis: V3, length: f64| axis.map(|component| component * length);
    let columns = [along(view.right, 3.0), along(view.back, 3.0), along(view.up, 3.0), at_wall];
    let mut model = [0.0f32; 16];
    for (column, values) in columns.iter().enumerate() {
        for (row, value) in values.iter().enumerate() {
            model[column * 4 + row] = *value as f32;
        }
        model[column * 4 + 3] = if column == 3 { 1.0 } else { 0.0 };
    }
    frame.decals = vec![Decal { material: 2, transform: model, opacity: 1.0 }];
    rig.load(4);
    let bare = rig.settle(&frame_without_decals(&frame), 2);
    let painted = rig.settle(&frame, 2);
    // Where the decal lies the wall is painted: x of 2 units right of the middle, at depth 20.
    let (x, y) = (32 + (2.0 / (20.0 * 0.5) * 32.0) as u32, 32);
    assert!(largest_difference(rig.pixel(&bare, x, y), rgb(WALL)) < LEVELS);
    assert!(largest_difference(rig.pixel(&painted, x, y), [0.6, 0.2, 0.2]) < LEVELS, "{:?}", rig.pixel(&painted, x, y));
    // With a pane covering it, the decal still shows beneath, and the frame is the one
    // picture without the pane wherever the pane is not.
    let mut with_pane = frame.clone();
    with_pane.draws.push(draw(0, 1, view.placed(10.0, 4.0, (2.0, 0.0))));
    let covered = rig.settle(&with_pane, 2);
    let expected = over([f64::from(PANE[0]), f64::from(PANE[1]), f64::from(PANE[2])], 0.5, [0.6, 0.2, 0.2]);
    assert!(
        largest_difference(rig.pixel(&covered, x, y), expected) < LEVELS,
        "{:?} vs {expected:?}",
        rig.pixel(&covered, x, y)
    );
    assert_eq!(rig.pixel(&covered, 4, 4), rig.pixel(&painted, 4, 4));
    // A pane that covers nothing: bit for bit the picture without it.
    let mut nothing = frame.clone();
    nothing.draws.push(draw(0, 1, view.facing(5.0, 1.0e-4)));
    assert_eq!(rig.settle(&nothing, 2), painted);
}

#[test]
fn the_translucent_pass_follows_the_render_scale_and_the_mist_s_grid() {
    let Some(mut rig) = Rig::new("scale", 64, &[flat(WALL), glass(PANE)]) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([4.0, 1.5, 0.0], [4.0, 1.5, -10.0], 1.0, 0.5);
    let mist = Mist { density: 0.03, height: 2000.0, mean: MEAN };
    let pane = Pane { near: 4.0, far: 40.0 };
    let (through, _, _) = pane_references(&rig, &view, &mist, &pane, (32, 32));
    // The picture's size and the grid (a tier, then none) change from frame to frame: the
    // opaque picture, its chain and the bind groups that read them are made again each time.
    for (scale, tier) in [(1.0, 2), (0.5, 2), (0.5, 1), (1.0, 0), (0.75, 2), (1.0, 2)] {
        let mut frame = pane_frame(&view, &mist, tier, &pane);
        frame.quality.scale = scale;
        let picture = rig.settle(&frame, 3);
        let got = rig.pixel(&picture, 32, 32);
        assert!(largest_difference(got, through) < LEVELS, "scale {scale} tier {tier}: {got:?} vs {through:?}");
        assert!(rig.renderer.passes().contains(&"backdrop"));
    }
}

#[test]
fn overlays_draw_dimmed_where_the_scene_hides_them_and_whole_where_it_does_not() {
    // An overlay behind the wall shows through it at the dimming's strength; one before
    // the wall replaces what is behind it: both in the translucent pass, after the picture.
    let overlay = Material { shader: "glass.wgsl", color: [0.15, 0.55, 0.25, 1.0], extra: r#""overlay":true"# };
    let Some(mut rig) = Rig::new("overlays", 64, &[flat(WALL), overlay]) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    for msaa in [1, 4] {
        let mut frame = frame(vec![
            draw(0, 0, scale_at(32.0, 32.0, 1.0, 32.0, 32.0, -40.0)),
            draw(0, 1, scale_at(16.0, 32.0, 1.0, 16.0, 32.0, -60.0)),
            draw(0, 1, scale_at(16.0, 32.0, 1.0, 48.0, 32.0, -10.0)),
        ]);
        frame.projection = ortho(64.0, 100.0);
        frame.quality.msaa = msaa;
        let picture = rig.settle(&frame, 2);
        let colour = [0.15, 0.55, 0.25];
        let behind = over(colour, 0.3, rgb(WALL));
        assert!(
            largest_difference(rig.pixel(&picture, 16, 32), behind) < LEVELS,
            "msaa {msaa}: {:?}",
            rig.pixel(&picture, 16, 32)
        );
        assert!(
            largest_difference(rig.pixel(&picture, 48, 32), colour) < LEVELS,
            "msaa {msaa}: {:?}",
            rig.pixel(&picture, 48, 32)
        );
        assert!(rig.renderer.passes().contains(&"backdrop"));
    }
}

fn frame_without_decals(frame: &RenderFrame) -> RenderFrame {
    let mut bare = frame.clone();
    bare.decals.clear();
    bare
}

/// A quad of `half` across and up, the first axis then the second, which faces the
/// way they cross, at `position`.
fn plane(position: V3, across: V3, up: V3) -> Matrix {
    let normal = unit(cross(across, up));
    let columns = [across, up, normal, position];
    let mut matrix = [0.0f32; 16];
    for (column, values) in columns.iter().enumerate() {
        for (row, value) in values.iter().enumerate() {
            matrix[column * 4 + row] = *value as f32;
        }
        matrix[column * 4 + 3] = if column == 3 { 1.0 } else { 0.0 };
    }
    matrix
}

/// A hazy room seen through a pane of tinted glass: walls, floor and pillars in a
/// mist, a sky above them, and two panes of glass before them, one near the camera
/// and one in the middle of the room. What the picture shows of a pane is how hazy
/// its own distance makes it.
#[test]
fn glass_in_a_hazy_room() {
    let materials = [
        flat([0.62, 0.55, 0.45]),
        flat([0.45, 0.5, 0.58]),
        flat([0.75, 0.3, 0.25]),
        flat([0.35, 0.4, 0.3]),
        glass([0.05, 0.5, 0.45, 0.55]),
        glass([0.9, 0.45, 0.1, 0.65]),
    ];
    let Some(mut rig) = Rig::new("glass-room", 512, &materials) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([0.0, 1.7, 12.0], [0.0, 2.2, -30.0], 1.0, 0.55);
    let mut frame = view_frame(&view);
    frame.fog = Fog {
        density: 0.045,
        height: 25.0,
        distance: 120.0,
        albedo: vector(0.9, 0.95, 1.0),
        anisotropy: 0.4,
        sun: 1.0,
        ambient: 1.0,
        sky: 0.5,
        ..Fog::default()
    };
    frame.quality.volumetrics = 2;
    frame.quality.sky = true;
    frame.quality.msaa = 4;
    frame.sky = Sky {
        zenith: vector(0.3, 0.5, 0.8),
        horizon: vector(0.75, 0.82, 0.9),
        ground: vector(0.4, 0.4, 0.42),
        sun: vector(0.4, 0.7, 0.3),
        sh0: vector(1.6, 1.8, 2.2),
        ..Sky::default()
    };
    frame.light_color = vector(1.1, 0.95, 0.75);
    frame.light_direction = vector(-0.4, -0.7, -0.3);
    frame.clear = Color { r: 0.75, g: 0.82, b: 0.9, a: 1.0 };
    frame.draws = vec![
        // The floor and the back and side walls of a room 28 across.
        draw(0, 0, plane([0.0, 0.0, -19.0], [14.0, 0.0, 0.0], [0.0, 0.0, -31.0])),
        draw(0, 1, plane([0.0, 7.0, -50.0], [14.0, 0.0, 0.0], [0.0, 7.0, 0.0])),
        draw(0, 3, plane([-14.0, 7.0, -19.0], [0.0, 0.0, -31.0], [0.0, 7.0, 0.0])),
        draw(0, 3, plane([14.0, 7.0, -19.0], [0.0, 0.0, 31.0], [0.0, 7.0, 0.0])),
    ];
    for (index, z) in [-10.0, -22.0, -34.0, -46.0].into_iter().enumerate() {
        for x in [-6.0, 6.0] {
            let colour = if (index + usize::from(x > 0.0)) % 2 == 0 { 2 } else { 1 };
            frame.draws.push(draw(0, colour, plane([x, 4.0, z], [0.8, 0.0, 0.0], [0.0, 4.0, 0.0])));
        }
    }
    // The panes: one a short way from the camera, one in the room.
    frame.draws.push(draw(0, 4, view.placed(7.0, 2.4, (0.6, -0.2))));
    frame.draws.push(draw(0, 5, view.placed(24.0, 3.2, (-5.5, 0.6))));
    let picture = rig.settle(&frame, 40);
    save("translucency-glass-in-fog.png", &picture, rig.size);
}
