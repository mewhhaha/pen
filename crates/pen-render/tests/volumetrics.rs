//! The froxel volume of the frame's mist, on a real (headless) GPU, held to f64
//! references: transmittance and in-scatter of a homogeneous mist, the
//! exponential height mist's optical depth along a slanted ray, sun shafts
//! behind an occluder, the Henyey-Greenstein phase, a point light's
//! in-scattering, local fog volumes, the temporal blend, the seam where the
//! grid ends and the analytic mist goes on, and the off tier. The pictures are
//! written to the directory `VOLUME_SHOTS` names (default target/look-shots).
//! Skips when no adapter is available.
#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::cast_possible_truncation, reason = "test geometry")]
use std::path::{Path, PathBuf};

mod common;

use common::*;
use pen_protocol::*;
use pen_render::{Display, Renderer, RendererOptions, VolumeImage, VolumeLayer};

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
    let directory = std::env::var_os("VOLUME_SHOTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/look-shots"));
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

fn boxed(x: f32, y0: f32, y1: f32, z: f32) -> String {
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

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: V3) -> V3 {
    let n = norm(a);
    [a[0] / n, a[1] / n, a[2] / n]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// A look-at camera: the view, the projection, the camera's right, up and
/// back axes, and the projection's scale across and down.
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
    /// The ray through the centre of froxel column (x, y) of a `w` by `h` grid:
    /// its unit direction in the world, and its length per unit of depth.
    fn ray(&self, grid: (u32, u32), x: u32, y: u32) -> (V3, f64) {
        let (u, v) = ((f64::from(x) + 0.5) / f64::from(grid.0), (f64::from(y) + 0.5) / f64::from(grid.1));
        let (nx, ny) = (u * 2.0 - 1.0, 1.0 - v * 2.0);
        let view = [nx / self.scale.0, ny / self.scale.1, -1.0];
        let world: V3 =
            std::array::from_fn(|i| view[0] * self.right[i] + view[1] * self.up[i] + view[2] * self.back[i]);
        (unit(world), norm(view))
    }
}

/// The depth ahead of the eye of boundary `b` of a grid of `slices` reaching `distance`.
fn depth(distance: f64, slices: u32, b: f64) -> f64 {
    let knee = distance / 64.0;
    let span = (1.0 + distance / knee).ln();
    knee * ((b / f64::from(slices) * span).exp() - 1.0)
}

/// pi times the Henyey-Greenstein phase.
fn phase(g: f64, c: f64) -> f64 {
    (1.0 - g * g) / (4.0 * (1.0 + g * g - 2.0 * g * c).powf(1.5))
}

fn vector(x: f32, y: f32, z: f32) -> Vector {
    Vector { x, y, z }
}

struct Rig {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    catalog: Catalog,
    target: Offscreen,
    size: u32,
}

impl Rig {
    fn new(name: &str, size: u32) -> Option<Rig> {
        let (device, queue) = featured_device()?;
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("volumes-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let source = LIT.replace("${shadows}", pen_render::SHADOWS_WGSL).replace("${lights}", pen_render::LIGHTS_WGSL);
        let floor = br#"{"positions":[-60,0,-60, 60,0,-60, -60,0,60, 60,0,60],"normals":[0,1,0,0,1,0,0,1,0,0,1,0],
                         "uvs":[0,0,1,0,0,1,1,1],"indices":[0,2,1,1,2,3]}"#;
        let files: Vec<(&str, Vec<u8>)> = vec![
            (
                "catalog.json",
                br#"{"meshes":["floor.mesh.json","box.mesh.json"],"materials":["floor.material.json","stone.material.json","red.material.json"]}"#.to_vec(),
            ),
            ("floor.mesh.json", floor.to_vec()),
            ("box.mesh.json", boxed(0.5, 0.0, 1.0, 0.5).into_bytes()),
            ("lit.wgsl", source.into_bytes()),
            ("floor.material.json", br#"{"shader":"lit.wgsl","color":[0.62,0.6,0.55,1],"lighting":"lit"}"#.to_vec()),
            ("stone.material.json", br#"{"shader":"lit.wgsl","color":[0.75,0.72,0.68,1],"lighting":"lit"}"#.to_vec()),
            ("red.material.json", br#"{"shader":"lit.wgsl","color":[0.8,0.25,0.2,1],"lighting":"lit"}"#.to_vec()),
        ];
        for (key, bytes) in files {
            std::fs::write(root.join(key), bytes).unwrap();
        }
        let catalog = boundary::read_catalog(&root).unwrap();
        let options =
            RendererOptions { format: FORMAT, display: Display::SDR, assets: root, veil: None, pipeline_cache: None };
        let renderer = Renderer::new(device.clone(), queue.clone(), options).unwrap();
        let target = Offscreen::new(&device, size);
        Some(Rig { device, queue, renderer, catalog, target, size })
    }

    fn draw(&mut self, frame: &RenderFrame) -> Vec<[u8; 4]> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
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

    /// Draws `frame` `count` times (a still camera, so that the jitter's
    /// frames blend into the grid) and returns the last picture.
    fn settle(&mut self, frame: &RenderFrame, count: usize) -> Vec<[u8; 4]> {
        let mut last = Vec::new();
        for _ in 0..count {
            last = self.draw(frame);
        }
        last
    }

    fn volume(&self, layer: VolumeLayer) -> VolumeImage {
        self.renderer.read_volume(layer).expect("the frame drew a volume")
    }
}

fn draw(mesh: u32, material: u32, transform: Matrix) -> Draw {
    Draw { mesh, material, transform, parameters: None, extra: None }
}

fn scale_at(sx: f32, sy: f32, sz: f32, x: f32, y: f32, z: f32) -> Matrix {
    [sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, sz, 0.0, x, y, z, 1.0]
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

/// An empty frame seen by `camera`, its mist `fog`, drawn at `tier`.
fn mist_frame(camera: &Camera, fog: Fog, tier: u32) -> RenderFrame {
    let mut frame = frame(Vec::new());
    frame.view = camera.view;
    frame.projection = camera.projection;
    frame.fog = fog;
    frame.quality.volumetrics = tier;
    frame
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

fn relative(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-6)
}

#[test]
fn a_homogeneous_mist_has_the_analytic_transmittance_and_ambient_in_scatter() {
    let Some(mut rig) = Rig::new("homogeneous", 64) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([0.0, 1.5, 0.0], [0.0, 1.5, -10.0], 1.6, 0.5);
    let (sigma, reach) = (0.03f64, 90.0f64);
    let mean = [0.8f64, 0.5, 0.3];
    for (tier, grid, slices) in [(2u32, (160u32, 90u32), 64u32), (1, (80, 45), 32)] {
        let fog = Fog { density: sigma as f32, height: 1.0e9, distance: reach as f32, ..Fog::default() };
        let mut frame = mist_frame(&view, fog, tier);
        frame.sky = sky_with_mean(mean);
        frame.light_color = vector(0.0, 0.0, 0.0);
        rig.settle(&frame, 3);
        let stats = rig.renderer.stats();
        assert_eq!(stats.volume_froxels, grid.0 * grid.1 * slices, "tier {tier}");
        let integral = rig.volume(VolumeLayer::Integrated);
        assert_eq!(integral.size, [grid.0, grid.1, slices + 1]);
        let mut worst = (0.0f64, 0.0f64);
        for (x, y) in [(0, 0), (grid.0 / 2, grid.1 / 2), (grid.0 - 1, grid.1 - 1), (grid.0 / 3, grid.1 * 2 / 3)] {
            let (_, slant) = view.ray(grid, x, y);
            assert_eq!(integral.at(x, y, 0), [0.0; 4]);
            for k in 1..=slices {
                let distance = depth(reach, slices, f64::from(k)) * slant;
                let optical = sigma * distance;
                let got = integral.at(x, y, k);
                worst.0 = worst.0.max(relative(f64::from(got[3]), optical));
                // The in-scatter of an isotropic ambient light: mean * (1 - T).
                for channel in 0..3 {
                    let expected = mean[channel] * (1.0 - (-optical).exp());
                    worst.1 = worst.1.max((f64::from(got[channel]) - expected).abs() / (expected + 1e-3));
                }
                let transmittance = (-f64::from(got[3])).exp();
                assert!((transmittance - (-optical).exp()).abs() < 2e-3, "T at slice {k}: {transmittance}");
            }
        }
        eprintln!("tier {tier}: worst optical depth error {:.2e}, in-scatter {:.2e}", worst.0, worst.1);
        assert!(worst.0 < 2.5e-3, "optical depth: {}", worst.0);
        assert!(worst.1 < 4e-3, "in-scatter: {}", worst.1);
    }
}

/// The light's view along `ray`, anchored 1000 back, as Ink's `graphics.shadow` has it.
fn light_view(ray: V3) -> Matrix {
    let ray = unit(ray);
    let across = if ray[1].abs() > 0.99 { [0.0, 0.0, 1.0] } else { [0.0, 1.0, 0.0] };
    let right = unit(cross(ray, across));
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
fn fit(camera: &Camera, aspect: f64, tangent: f64, ray: V3, texels: u32, splits: &[f64]) -> (Matrix, Vec<Cascade>) {
    let light = light_view(ray);
    let l = light.map(f64::from);
    let to_world = |p: V3| -> V3 {
        std::array::from_fn(|i| camera.eye[i] + p[0] * camera.right[i] + p[1] * camera.up[i] + p[2] * camera.back[i])
    };
    let to_light = |p: V3| -> V3 {
        std::array::from_fn(|row| l[row] * p[0] + l[4 + row] * p[1] + l[8 + row] * p[2] + l[12 + row])
    };
    let mut cascades = Vec::new();
    let mut near = 0.1;
    for &far in splits {
        let corners: Vec<V3> = [near, far]
            .into_iter()
            .flat_map(|d| {
                [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
                    .map(|(x, y)| to_world([x * d * tangent * aspect, y * d * tangent, -d]))
            })
            .collect();
        let centre: V3 = std::array::from_fn(|axis| corners.iter().map(|c| c[axis]).sum::<f64>() / 8.0);
        let radius = corners.iter().map(|c| norm(sub(*c, centre))).fold(0.0, f64::max);
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

/// The sun's cascades for `frame`, which `camera` sees along `ray`.
fn with_sun(frame: &mut RenderFrame, camera: &Camera, ray: V3, color: V3, texels: u32) {
    let (aspect, tangent) = (camera.scale.1 / camera.scale.0, 1.0 / camera.scale.1);
    frame.light_direction = vector(ray[0] as f32, ray[1] as f32, ray[2] as f32);
    frame.light_color = vector(color[0] as f32, color[1] as f32, color[2] as f32);
    let toward = unit(ray).map(|v| -v as f32);
    frame.sky.sun = vector(toward[0], toward[1], toward[2]);
    let (light, cascades) = fit(camera, aspect, tangent, ray, texels, &[12.0, 40.0, 120.0]);
    frame.shadow.view = light;
    frame.shadow.texels = texels;
    frame.shadow.filter = 1;
    frame.cascades = cascades;
}

/// The froxel (x, y, k)'s centre in the world, for a camera and a grid.
fn froxel_centre(view: &Camera, grid: (u32, u32), slices: u32, reach: f64, x: u32, y: u32, k: u32) -> V3 {
    let (direction, slant) = view.ray(grid, x, y);
    let along = depth(reach, slices, f64::from(k) + 0.5) * slant;
    std::array::from_fn(|i| view.eye[i] + direction[i] * along)
}

/// Whether the sun, shining along `ray`, reaches `p` past two roof slabs at
/// heights `roof` (y from, to) with a slit between x = -slit and x = slit.
fn sun_reaches(p: V3, ray: V3, roof: (f64, f64), slit: f64) -> bool {
    let toward = unit(ray).map(|v| -v);
    if p[1] >= roof.0 {
        return true;
    }
    // Where the ray toward the sun crosses the roof's underside.
    let t = (roof.0 - p[1]) / toward[1];
    let x = p[0] + toward[0] * t;
    let z = p[2] + toward[2] * t;
    (x.abs() < slit) || z.abs() > 30.0
}

#[test]
fn sun_shafts_through_a_roof_slit_have_the_shadowed_and_unshadowed_in_scatter_ratio() {
    let Some(mut rig) = Rig::new("shafts", 64) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([0.0, 3.0, 18.0], [0.0, 3.0, -5.0], 1.0, 0.6);
    let (sigma, reach, g) = (0.04f64, 60.0f64, 0.5f64);
    let fog =
        Fog { density: sigma as f32, height: 1.0e9, distance: reach as f32, anisotropy: g as f32, ..Fog::default() };
    let mut frame = mist_frame(&view, fog, 2);
    frame.draws = vec![
        draw(0, 0, IDENTITY),
        draw(1, 1, scale_at(27.5, 1.0, 60.0, -16.25, 8.0, 0.0)),
        draw(1, 1, scale_at(27.5, 1.0, 60.0, 16.25, 8.0, 0.0)),
    ];
    let mean = [0.04f64, 0.05, 0.07];
    frame.sky = sky_with_mean(mean);
    let sun = [3.0, 2.8, 2.4];
    let ray = [0.25, -1.0, 0.1];
    with_sun(&mut frame, &view, ray, sun, 2048);
    frame.clear = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    rig.settle(&frame, 40);
    let stats = rig.renderer.stats();
    assert_eq!(stats.volume_froxels, 160 * 90 * 64);
    let injected = rig.volume(VolumeLayer::Injected);
    let toward = unit(ray).map(|v| -v);
    let grid = (160u32, 90u32);
    let (mut lit, mut shadowed) = (Vec::new(), Vec::new());
    let mut worst: f64 = 0.0;
    for y in (0..grid.1).step_by(7) {
        for x in (0..grid.0).step_by(5) {
            for k in (8..64).step_by(3) {
                let p = froxel_centre(&view, grid, 64, reach, x, y, k);
                // Inside the room, away from the slit's edges and the geometry.
                if p[1] < 0.8 || p[1] > 7.0 || p[2].abs() > 28.0 || p[0].abs() > 28.0 {
                    continue;
                }
                let (direction, _) = view.ray(grid, x, y);
                let t = (8.0 - p[1]) / toward[1];
                let edge = (p[0] + toward[0] * t).abs() - 2.5;
                // Well inside the lit column or well inside the shadow: a metre and
                // a half past the edge covers the filter and the froxel's jitter.
                if edge.abs() < 1.5 {
                    continue;
                }
                let expected_sun = sun.map(|s| s * sigma * phase(g, dot(toward, direction)));
                let got = injected.at(x, y, k);
                let reaches = sun_reaches(p, ray, (8.0, 9.0), 2.5);
                // How much of the sun's in-scatter the froxel holds, past the ambient's.
                let mut seen = 0.0;
                for channel in 0..3 {
                    seen += (f64::from(got[channel]) - sigma * mean[channel]) / expected_sun[channel] / 3.0;
                }
                let deviation = (seen - if reaches { 1.0 } else { 0.0 }).abs();
                if deviation > worst {
                    worst = deviation;
                    eprintln!("worst so far at {x},{y},{k} p {p:?} lit {reaches} edge {edge:.2}: seen {seen}");
                }
                if reaches { lit.push(seen) } else { shadowed.push(seen) }
            }
        }
    }
    assert!(lit.len() > 40 && shadowed.len() > 40, "{} lit and {} shadowed froxels", lit.len(), shadowed.len());
    eprintln!(
        "shafts: {} lit, {} shadowed froxels; worst deviation of the sun's share {worst:.3}",
        lit.len(),
        shadowed.len()
    );
    // A lit froxel holds the sun's in-scatter in the phase's measure, a shadowed one none.
    assert!(worst < 0.06, "the sun's share of the in-scatter: {worst}");
    let lit_mean = lit.iter().sum::<f64>() / lit.len() as f64;
    let shadow_mean = shadowed.iter().sum::<f64>() / shadowed.len() as f64;
    assert!((lit_mean - 1.0).abs() < 0.01 && shadow_mean.abs() < 0.02, "means {lit_mean} and {shadow_mean}");
    let image = rig.draw(&frame);
    save("volumetrics-shafts.png", &image, rig.size);
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

#[test]
fn height_mist_has_the_analytic_exponential_optical_depth_along_a_slanted_ray() {
    let Some(mut rig) = Rig::new("height", 64) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    // Looking up at a slant from beside the ground, so that rays cross the
    // heights the mist thins through, and one looking down through the ground's level.
    let (density, height, reach) = (0.06f64, 12.0f64, 60.0f64);
    for (eye, target) in [([0.0, 2.0, 0.0], [0.0, 14.0, -30.0]), ([0.0, 6.0, 0.0], [0.0, -6.0, -30.0])] {
        let view = camera(eye, target, 1.6, 0.5);
        let fog = Fog { density: density as f32, height: height as f32, distance: reach as f32, ..Fog::default() };
        let mut frame = mist_frame(&view, fog, 2);
        frame.sky = sky_with_mean([0.5, 0.5, 0.5]);
        rig.settle(&frame, 40);
        let integral = rig.volume(VolumeLayer::Integrated);
        let grid = (160, 90);
        let mut worst: f64 = 0.0;
        for (x, y) in [(0, 0), (80, 45), (159, 89), (40, 10), (120, 80), (80, 0), (80, 89)] {
            let (direction, slant) = view.ray(grid, x, y);
            for k in 1..=64 {
                let length = depth(reach, 64, f64::from(k)) * slant;
                let expected = height_optical_depth(density, height, view.eye, direction, length);
                let got = f64::from(integral.at(x, y, k)[3]);
                // The grid takes the mist at one point of each slice (the jitter walks it over
                // the slice, and the average is the integral to second order): a few parts in a
                // thousand of the whole, and never more than a slice's share of the quadrature.
                let error = (got - expected).abs() / expected.max(0.05);
                worst = worst.max(error);
            }
        }
        eprintln!("height mist from {eye:?}: worst relative optical depth error {worst:.4}");
        assert!(worst < 0.012, "optical depth along slanted rays: {worst}");
    }
}

#[test]
fn the_henyey_greenstein_phase_brightens_toward_the_light_by_the_formula() {
    let Some(mut rig) = Rig::new("phase", 64) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let (sigma, reach) = (0.02f64, 40.0f64);
    let ray = [0.4, -0.3, -1.0];
    let toward = unit(ray).map(|v| -v);
    for g in [0.0f64, 0.7, -0.5] {
        let mut means = Vec::new();
        // Looking toward the sun and away from it.
        for sign in [1.0, -1.0] {
            let look = [toward[0] * sign, toward[1] * sign * 0.2, toward[2] * sign];
            let view = camera([0.0, 5.0, 0.0], [look[0] * 10.0, 5.0 + look[1] * 10.0, look[2] * 10.0], 1.0, 0.4);
            let fog = Fog {
                density: sigma as f32,
                height: 1.0e9,
                distance: reach as f32,
                anisotropy: g as f32,
                ..Fog::default()
            };
            let mut frame = mist_frame(&view, fog, 2);
            frame.sky = sky_with_mean([0.0, 0.0, 0.0]);
            frame.light_color = vector(2.0, 2.0, 2.0);
            frame.light_direction = vector(ray[0] as f32, ray[1] as f32, ray[2] as f32);
            frame.sky.sun = vector(toward[0] as f32, toward[1] as f32, toward[2] as f32);
            rig.settle(&frame, 3);
            let injected = rig.volume(VolumeLayer::Injected);
            let mut total = 0.0;
            let mut expected_total = 0.0;
            let mut count = 0.0;
            for (x, y, k) in [(80, 45, 30), (60, 30, 40), (100, 60, 20), (30, 70, 50), (130, 20, 35)] {
                let (direction, _) = view.ray((160, 90), x, y);
                let want = 2.0 * sigma * phase(g, dot(toward, direction));
                let got = f64::from(injected.at(x, y, k)[0]);
                assert!(relative(got, want) < 1.5e-2, "g {g} sign {sign}: {got} vs {want}");
                total += got;
                expected_total += want;
                count += 1.0;
            }
            means.push((total / count, expected_total / count));
        }
        let ratio = means[0].0 / means[1].0;
        let formula = means[0].1 / means[1].1;
        eprintln!("g {g}: forward over back {ratio:.4} (formula {formula:.4})");
        assert!(relative(ratio, formula) < 2e-2, "forward over back for g {g}: {ratio} vs {formula}");
        if g > 0.0 {
            assert!(ratio > 5.0, "forward scattering brightens toward the light");
        } else if g < 0.0 {
            assert!(ratio < 1.0, "back scattering brightens away from it");
        }
    }
}

/// The light's strength at a distance, as `clustered_light` takes it.
fn light_strength(distance: f64, reach: f64) -> f64 {
    if distance >= reach {
        return 0.0;
    }
    let ratio = distance / reach;
    let window = (1.0 - ratio.powi(4)).clamp(0.0, 1.0);
    window * window / (distance * distance + 1.0)
}

#[test]
fn a_point_light_in_the_mist_scatters_by_its_falloff_and_is_shadowed_by_casters() {
    let Some(mut rig) = Rig::new("point", 64) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([0.0, 2.0, 14.0], [0.0, 2.0, -8.0], 1.6, 0.5);
    let (sigma, reach, g) = (0.05f64, 50.0f64, 0.3f64);
    let fog =
        Fog { density: sigma as f32, height: 1.0e9, distance: reach as f32, anisotropy: g as f32, ..Fog::default() };
    let mut frame = mist_frame(&view, fog, 2);
    frame.sky = sky_with_mean([0.0, 0.0, 0.0]);
    frame.light_color = vector(0.0, 0.0, 0.0);
    let (position, radius, color) = ([2.0f64, 3.0, -2.0], 30.0f64, [12.0f64, 9.0, 6.0]);
    frame.lights = vec![Light {
        position: vector(position[0] as f32, position[1] as f32, position[2] as f32),
        radius: radius as f32,
        color: vector(color[0] as f32, color[1] as f32, color[2] as f32),
        direction: vector(0.0, -1.0, 0.0),
        inner: -1.0,
        outer: -1.0,
        shadow: false,
        id: 1,
    }];
    rig.settle(&frame, 40);
    let injected = rig.volume(VolumeLayer::Injected);
    let grid = (160u32, 90u32);
    let mut worst: f64 = 0.0;
    let mut checked = 0;
    for y in (0..grid.1).step_by(6) {
        for x in (0..grid.0).step_by(8) {
            for k in (4..64).step_by(2) {
                let p = froxel_centre(&view, grid, 64, reach, x, y, k);
                let to = sub(position, p);
                let distance = norm(to);
                // Where the light has a good share of its reach left, and the froxel is not
                // thinner than the light's own scale.
                let (direction, slant) = view.ray(grid, x, y);
                // The froxel's longest side: its slice's thickness along the ray, or its width.
                let thickness = (depth(reach, 64, f64::from(k) + 1.0) - depth(reach, 64, f64::from(k))) * slant;
                let width = depth(reach, 64, f64::from(k) + 0.5) * slant * 2.0 * 0.5 / 90.0 * 1.6;
                if distance < 8.0 * thickness.max(width) || distance > radius * 0.6 {
                    continue;
                }
                let want: Vec<f64> = color
                    .iter()
                    .map(|c| sigma * c * light_strength(distance, radius) * phase(g, dot(unit(to), direction)))
                    .collect();
                let got = injected.at(x, y, k);
                for channel in 0..3 {
                    worst = worst.max(relative(f64::from(got[channel]), want[channel]));
                }
                checked += 1;
            }
        }
    }
    eprintln!("point light: {checked} froxels, worst relative error {worst:.4}");
    assert!(checked > 100);
    // The grid takes the light at the sample's jitter, whose average is the froxel's
    // centre to second order: past six froxels from the light that is under a percent
    // of its curvature, and half-float storage adds a part in a thousand.
    assert!(worst < 0.03, "in-scatter of a point light against its falloff: {worst}");
    // Nothing beyond the light's reach.
    let integral = rig.volume(VolumeLayer::Integrated);
    let (x, y) = (4, 45);
    let (direction, slant) = view.ray(grid, x, y);
    // The integral along a ray against a fine f64 integration of the same source and transmittance.
    let mut worst_integral: f64 = 0.0;
    for k in [16u32, 32, 48, 64] {
        let end = depth(reach, 64, f64::from(k)) * slant;
        let steps = 20_000;
        let (mut light, mut optical) = ([0.0f64; 3], 0.0f64);
        for i in 0..steps {
            let s = end * (f64::from(i) + 0.5) / f64::from(steps);
            let p: V3 = std::array::from_fn(|axis| view.eye[axis] + direction[axis] * s);
            let to = sub(position, p);
            let d = norm(to);
            let ds = end / f64::from(steps);
            for channel in 0..3 {
                let source = sigma * color[channel] * light_strength(d, radius) * phase(g, dot(unit(to), direction));
                light[channel] += (-optical).exp() * source * ds;
            }
            optical += sigma * ds;
        }
        let got = integral.at(x, y, k);
        for channel in 0..3 {
            worst_integral =
                worst_integral.max((f64::from(got[channel]) - light[channel]).abs() / light[channel].max(0.02));
        }
    }
    eprintln!("point light integral along a ray: worst relative error {worst_integral:.4}");
    assert!(worst_integral < 0.06, "the integrated in-scatter: {worst_integral}");
    // A caster between the light and part of the mist shadows it: the froxels behind it
    // hold a small share of what they would.
    let before = injected.clone();
    let mut shadowed = frame.clone();
    shadowed.lights[0].shadow = true;
    shadowed.shadow.local = 2;
    shadowed.shadow.local_texels = 1024;
    // A wall between the light and the camera's right side of the view.
    shadowed.draws = vec![draw(1, 1, scale_at(0.4, 8.0, 8.0, 5.0, -1.0, -2.0))];
    rig.settle(&shadowed, 40);
    let stats = rig.renderer.stats();
    assert!(stats.local_shadow_casters >= 1, "{stats:?}");
    let after = rig.volume(VolumeLayer::Injected);
    let (mut lit_share, mut shadow_share) = (Vec::new(), Vec::new());
    for y in (0..grid.1).step_by(6) {
        for x in (0..grid.0).step_by(8) {
            for k in (4..64).step_by(2) {
                let p = froxel_centre(&view, grid, 64, reach, x, y, k);
                let (a, b) = (before.at(x, y, k)[0], after.at(x, y, k)[0]);
                if a < 1e-3 {
                    continue;
                }
                // Behind the wall seen from the light: past x = 5 and along the light's ray.
                let to = sub(p, position);
                let t = (5.0 - position[0]) / to[0];
                let behind = to[0] > 0.0 && t > 0.0 && t < 1.0;
                let hit = [position[1] + to[1] * t, position[2] + to[2] * t];
                let inside = (-5.0..7.0).contains(&hit[0]) && (-10.0..6.0).contains(&hit[1]);
                let share = f64::from(b) / f64::from(a);
                if behind
                    && inside
                    && (p[0] - 5.0) > 0.8
                    && hit[1] > -8.0
                    && hit[1] < 4.0
                    && hit[0] > -3.0
                    && hit[0] < 6.0
                {
                    shadow_share.push(share);
                } else if !behind && p[0] < 4.0 {
                    lit_share.push(share);
                }
            }
        }
    }
    assert!(
        lit_share.len() > 100 && shadow_share.len() > 20,
        "{} lit, {} shadowed",
        lit_share.len(),
        shadow_share.len()
    );
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    eprintln!("point light shadow: lit share {:.3}, shadowed share {:.3}", mean(&lit_share), mean(&shadow_share));
    assert!((mean(&lit_share) - 1.0).abs() < 0.03, "unshadowed froxels keep their light");
    assert!(mean(&shadow_share) < 0.2, "froxels behind the caster lose it");
}

#[test]
fn local_fog_volumes_hold_their_medium_only_inside_their_bounds() {
    let Some(mut rig) = Rig::new("local-volumes", 64) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([0.0, 2.0, 20.0], [0.0, 2.0, 0.0], 1.6, 0.5);
    let reach = 60.0f64;
    let mut frame = mist_frame(&view, Fog { height: 1.0, distance: reach as f32, ..Fog::default() }, 2);
    frame.sky = sky_with_mean([0.2, 0.2, 0.2]);
    frame.light_color = vector(0.0, 0.0, 0.0);
    let boxed = FogVolume {
        shape: 0,
        center: vector(-5.0, 2.0, 0.0),
        extent: vector(3.0, 2.0, 3.0),
        density: 0.4,
        albedo: vector(1.0, 1.0, 1.0),
        emission: vector(0.0, 0.0, 0.0),
        falloff: 0.0,
    };
    let ellipsoid = FogVolume {
        shape: 1,
        center: vector(5.0, 2.0, -2.0),
        extent: vector(2.5, 1.5, 4.0),
        density: 0.3,
        albedo: vector(0.0, 0.0, 0.0),
        emission: vector(0.5, 0.25, 0.0),
        falloff: 0.5,
    };
    frame.fog_volumes = vec![boxed, ellipsoid];
    rig.settle(&frame, 40);
    let injected = rig.volume(VolumeLayer::Injected);
    let grid = (160u32, 90u32);
    let (mut inside_box, mut inside_ellipsoid, mut outside) = (0, 0, 0);
    let mut worst_box: f64 = 0.0;
    let mut worst_ellipsoid: f64 = 0.0;
    for y in 0..grid.1 {
        for x in (0..grid.0).step_by(2) {
            for k in 0..64 {
                let p = froxel_centre(&view, grid, 64, reach, x, y, k);
                let at = injected.at(x, y, k);
                let (_, slant) = view.ray(grid, x, y);
                // The froxel's longest side: jitter moves the sample this far.
                let thickness = (depth(reach, 64, f64::from(k) + 1.0) - depth(reach, 64, f64::from(k))) * slant;
                let width = depth(reach, 64, f64::from(k) + 0.5) * slant * 1.6 / 90.0;
                let margin = thickness.max(width);
                let offset_ellipsoid = [(p[0] - 5.0) / 2.5, (p[1] - 2.0) / 1.5, (p[2] + 2.0) / 4.0];
                let m_ellipsoid = norm(offset_ellipsoid);
                let near_box = [(3.0 - (p[0] + 5.0).abs()), (2.0 - (p[1] - 2.0).abs()), (3.0 - p[2].abs())];
                let box_depth = near_box.iter().cloned().fold(f64::MAX, f64::min);
                if box_depth > margin {
                    inside_box += 1;
                    worst_box = worst_box.max(relative(f64::from(at[3]), 0.4));
                }
                // Well inside the ellipsoid's flat core (its falloff covers the outer half of its
                // size), where jitter does not change the weight.
                if m_ellipsoid < 0.35 {
                    inside_ellipsoid += 1;
                    // Emission only: its albedo is nothing, and the mist scatters no ambient.
                    let want = [0.3 * 0.5, 0.3 * 0.25, 0.0];
                    for channel in 0..3 {
                        worst_ellipsoid = worst_ellipsoid.max((f64::from(at[channel]) - want[channel]).abs() / 0.15);
                    }
                }
                let far_from_box = box_depth < -margin;
                if far_from_box && m_ellipsoid > 1.0 + 3.0 * margin / 1.5 {
                    outside += 1;
                    assert_eq!(at, [0.0; 4], "outside every volume at {p:?}");
                }
            }
        }
    }
    eprintln!(
        "{inside_box} froxels in the box, {inside_ellipsoid} in the ellipsoid's core, {outside} outside; worst {worst_box:.4}, {worst_ellipsoid:.4}"
    );
    assert!(inside_box > 100 && inside_ellipsoid > 5 && outside > 10_000);
    assert!(worst_box < 2e-3, "a box holds its density: {worst_box}");
    assert!(worst_ellipsoid < 3e-3, "an ellipsoid with no albedo gives out only its emission: {worst_ellipsoid}");
    // The falloff reaches nothing at the edge and rises smoothly inward: along a ray through the
    // ellipsoid's middle the extinction rises and falls once.
    let (x, y) = (grid.0 / 2 + 25, grid.1 / 2);
    let mut values = Vec::new();
    for k in 0..64 {
        let p = froxel_centre(&view, grid, 64, reach, x, y, k);
        if (p[0] - 5.0).abs() < 4.0 {
            values.push(f64::from(injected.at(x, y, k)[3]));
        }
    }
    assert!(values.iter().cloned().fold(0.0, f64::max) > 0.0);
    // Outside the volumes the mist is analytic only where there is none: stays at nothing in the frame's far part.
    let integral = rig.volume(VolumeLayer::Integrated);
    assert_eq!(integral.at(0, 0, 1)[3], 0.0);
}

#[test]
fn a_still_camera_converges_and_a_cut_resets_the_history() {
    let Some(mut rig) = Rig::new("temporal", 64) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([0.0, 3.0, 18.0], [0.0, 3.0, -5.0], 1.0, 0.6);
    let reach = 60.0f64;
    let fog = Fog { density: 0.04, height: 1.0e9, distance: reach as f32, anisotropy: 0.5, ..Fog::default() };
    let mut frame = mist_frame(&view, fog, 2);
    frame.draws = vec![
        draw(0, 0, IDENTITY),
        draw(1, 1, scale_at(27.5, 1.0, 60.0, -16.25, 8.0, 0.0)),
        draw(1, 1, scale_at(27.5, 1.0, 60.0, 16.25, 8.0, 0.0)),
    ];
    frame.sky = sky_with_mean([0.04, 0.05, 0.07]);
    with_sun(&mut frame, &view, [0.25, -1.0, 0.1], [3.0, 2.8, 2.4], 2048);
    let sample = |image: &VolumeImage| -> Vec<f32> {
        let mut values = Vec::new();
        for y in (0..90).step_by(5) {
            for x in (0..160).step_by(5) {
                for k in (6..64).step_by(3) {
                    values.push(image.at(x, y, k)[1]);
                }
            }
        }
        values
    };
    let change = |a: &[f32], b: &[f32]| -> f64 {
        let (mut sum, mut total) = (0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            sum += f64::from((x - y).abs());
            total += f64::from(y.abs());
        }
        sum / total
    };
    // The first draws load the assets and may submit the frame more than once: start afresh.
    rig.draw(&frame);
    rig.renderer.reset_volumetrics();
    let mut kept: Vec<Vec<f32>> = Vec::new();
    let mut changes: Vec<f64> = Vec::new();
    for index in 0..48 {
        rig.draw(&frame);
        let image = sample(&rig.volume(VolumeLayer::Injected));
        if let Some(last) = kept.last() {
            changes.push(change(&image, last));
        }
        kept.push(image);
        let stats = rig.renderer.stats();
        assert_eq!(stats.volume_history, index > 0, "frame {index} reads history after the first");
    }
    // The converged picture is the mean over a whole period of the jitter's sequence.
    let converged: Vec<f32> =
        (0..kept[0].len()).map(|i| kept[32..48].iter().map(|image| image[i]).sum::<f32>() / 16.0).collect();
    let errors: Vec<f64> = kept.iter().map(|image| change(image, &converged)).collect();
    eprintln!(
        "distance from the converged picture: frame 0 {:.4}, frame 1 {:.4}, frame 8 {:.4}, frame 40 {:.4}; \
         frame to frame at the end {:.4}",
        errors[0], errors[1], errors[8], errors[40], changes[46]
    );
    // The first frame is one jittered sample; the blend averages the sequence in, so later frames
    // sit nearer the converged picture, and once there a still camera is stable.
    assert!(errors[0] > 2.0 * errors[40], "the picture converges: {} then {}", errors[0], errors[40]);
    assert!(errors[40] < 0.006, "and sits within a part in 170 of the converged one: {}", errors[40]);
    let settled: f64 = changes[32..].iter().sum::<f64>() / changes[32..].len() as f64;
    assert!(settled < 0.004, "a still camera is stable frame to frame: {settled}");
    // A cut resets: `exposure.cut` reads nothing from history, and so does a jump of the camera.
    let before = rig.volume(VolumeLayer::Injected);
    let mut cut = frame.clone();
    cut.exposure.cut = true;
    rig.draw(&cut);
    assert!(!rig.renderer.stats().volume_history, "a cut takes no history");
    rig.draw(&frame);
    assert!(rig.renderer.stats().volume_history, "the frame after it does again");
    let mut jumped = frame.clone();
    let far_view = camera([0.0, 3.0, 18.0 + 0.2 * reach], [0.0, 3.0, -5.0 + 0.2 * reach], 1.0, 0.6);
    jumped.view = far_view.view;
    rig.draw(&jumped);
    assert!(!rig.renderer.stats().volume_history, "a camera that jumps takes no history");
    let _ = before;
    // A host's reset has the same effect.
    rig.draw(&frame);
    rig.renderer.reset_volumetrics();
    rig.draw(&frame);
    assert!(!rig.renderer.stats().volume_history, "a reset takes no history");
    // Disocclusion: a volume that vanishes is gone from the next frame, not faded out of history.
    let mut volume = frame.clone();
    volume.fog_volumes = vec![FogVolume {
        shape: 0,
        center: vector(0.0, 3.0, 4.0),
        extent: vector(3.0, 2.0, 3.0),
        density: 0.5,
        albedo: vector(1.0, 1.0, 1.0),
        emission: vector(0.0, 0.0, 0.0),
        falloff: 0.0,
    }];
    for _ in 0..20 {
        rig.draw(&volume);
    }
    let with = rig.volume(VolumeLayer::Injected);
    let mut gone = volume.clone();
    gone.fog_volumes.clear();
    rig.draw(&gone);
    let without = rig.volume(VolumeLayer::Injected);
    // The froxel on the middle column nearest the volume's middle.
    let (x, y) = (80, 45);
    let k = (0..64)
        .min_by(|&a, &b| {
            let at = |k| norm(sub(froxel_centre(&view, (160, 90), 64, reach, x, y, k), [0.0, 3.0, 4.0]));
            at(a).total_cmp(&at(b))
        })
        .unwrap();
    eprintln!(
        "froxel {k}: extinction {} with the volume, {} the frame after it is gone",
        with.at(x, y, k)[3],
        without.at(x, y, k)[3]
    );
    assert!(relative(f64::from(with.at(x, y, k)[3]), 0.54) < 5e-3);
    assert!(relative(f64::from(without.at(x, y, k)[3]), 0.04) < 5e-3, "a medium that is gone leaves no ghost");
}

/// A frame of the floor under a sky, in a homogeneous mist the colour of the sky, seen from
/// the near end of the floor so that the grid's reach falls inside the picture.
fn seam_frame(view: &Camera, reach: f32, tier: u32, radius: f32) -> RenderFrame {
    let fog = Fog { radius, density: 0.03, height: 2000.0, distance: reach, sun: 0.0, sky: 1.0, ..Fog::default() };
    let mut frame = mist_frame(view, fog, tier);
    frame.draws = vec![draw(0, 0, IDENTITY), draw(1, 2, scale_at(2.0, 6.0, 2.0, 0.0, 0.0, -30.0))];
    frame.quality.sky = true;
    frame.sky = Sky {
        zenith: vector(0.25, 0.45, 0.85),
        horizon: vector(0.7, 0.78, 0.9),
        ground: vector(0.3, 0.3, 0.32),
        sun: vector(0.0, 1.0, 0.0),
        sh0: vector(2.0, 2.2, 2.6),
        ..Sky::default()
    };
    frame.ambient = vector(0.4, 0.4, 0.4);
    frame.light_color = vector(0.6, 0.55, 0.5);
    frame.light_direction = vector(0.3, -1.0, 0.2);
    frame.clear = Color { r: 0.7, g: 0.78, b: 0.9, a: 1.0 };
    frame
}

#[test]
fn the_grids_end_leaves_no_seam_against_the_analytic_mist_beyond_it() {
    let Some(mut rig) = Rig::new("seam", 192) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([0.0, 1.5, 55.0], [0.0, 0.5, 10.0], 1.0, 0.5);
    // The same homogeneous mist drawn analytically (no grid) and through grids reaching 12, 30
    // and 100: in the picture the floor crosses each grid's end somewhere.
    for radius in [0.0f32, 20.0] {
        let analytic = rig.settle(&seam_frame(&view, 0.0, 2, radius), 2);
        assert_eq!(rig.renderer.stats().volume_froxels, 0, "no reach, no grid");
        save(&format!("volumetrics-seam-analytic-r{radius}.png"), &analytic, rig.size);
        for reach in [12.0f32, 30.0, 100.0] {
            let image = rig.settle(&seam_frame(&view, reach, 2, radius), 40);
            assert!(rig.renderer.stats().volume_froxels > 0);
            save(&format!("volumetrics-seam-r{radius}-reach-{reach}.png"), &image, rig.size);
            let (mut total, mut worst) = (0.0f64, 0.0f64);
            for (a, b) in image.iter().zip(&analytic) {
                let difference = (0..3).map(|c| (f64::from(a[c]) - f64::from(b[c])).abs()).fold(0.0, f64::max);
                total += difference;
                worst = worst.max(difference);
            }
            let mean = total / image.len() as f64;
            eprintln!("radius {radius} reach {reach}: mean difference {mean:.3} of 255, worst {worst}");
            assert!(mean < 0.8, "the same mist looks the same through the grid and without it: {mean}");
            assert!(worst < 12.0, "and nowhere across the grid's end does it show a seam: {worst}");
            // Down a column of the picture the grid's end makes no step: no pixel differs from its
            // neighbour more than the analytic picture's own steps do.
            let column = |image: &[[u8; 4]]| -> Vec<f64> {
                (0..rig.size).map(|y| luminance(image[(y * rig.size + rig.size / 4) as usize])).collect()
            };
            let (steps, base) = (column(&image), column(&analytic));
            let jump = |v: &[f64]| v.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max);
            assert!(
                jump(&steps) <= jump(&base) + 2.5,
                "no step at the grid's end: {} vs {}",
                jump(&steps),
                jump(&base)
            );
        }
    }
}

#[test]
fn the_off_tier_draws_no_volume_and_falls_back_to_the_analytic_mist() {
    let Some(mut rig) = Rig::new("off", 128) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([0.0, 1.5, 55.0], [0.0, 0.5, 10.0], 1.0, 0.5);
    let on = seam_frame(&view, 40.0, 2, 0.0);
    rig.settle(&on, 3);
    assert!(rig.renderer.stats().volume_froxels > 0);
    // Off: no froxel is filled and the picture is the analytic one, bit for bit.
    let off = seam_frame(&view, 40.0, 0, 0.0);
    let fallback = rig.settle(&off, 3);
    assert_eq!(rig.renderer.stats().volume_froxels, 0, "the off tier has no volumetric cost");
    assert!(!rig.renderer.stats().volume_history);
    let analytic = rig.settle(&seam_frame(&view, 0.0, 2, 0.0), 3);
    assert_eq!(fallback, analytic, "the off tier is the analytic mist");
    // No mist, no medium: nothing is filled either.
    let mut clear = seam_frame(&view, 40.0, 2, 0.0);
    clear.fog.density = 0.0;
    rig.settle(&clear, 2);
    assert_eq!(rig.renderer.stats().volume_froxels, 0);
    // The low tier fills fewer froxels than the high one.
    rig.settle(&seam_frame(&view, 40.0, 1, 0.0), 2);
    let low = rig.renderer.stats().volume_froxels;
    rig.settle(&on, 2);
    assert!(low > 0 && low < rig.renderer.stats().volume_froxels);
}

/// A room of pillars under a roof, its sun low on the pillared side, in a haze.
fn room_frame(view: &Camera, tier: u32, density: f32) -> RenderFrame {
    let mut draws = vec![draw(0, 0, IDENTITY), draw(1, 1, scale_at(40.0, 1.0, 40.0, 0.0, 10.0, 0.0))];
    // A colonnade along the sun's side, pillars a metre and a half wide, four apart.
    for i in -4..=4 {
        draws.push(draw(1, 1, scale_at(1.5, 10.0, 1.5, -12.0, 0.0, i as f32 * 4.0)));
    }
    draws.push(draw(1, 2, scale_at(2.0, 1.2, 2.0, 2.0, 0.0, -2.0)));
    draws.push(draw(1, 2, scale_at(1.0, 2.2, 1.0, -3.0, 0.0, 5.0)));
    let fog = Fog {
        density,
        height: 14.0,
        distance: 70.0,
        anisotropy: 0.55,
        sun: 1.0,
        ambient: 1.0,
        noise: 0.35,
        noise_scale: 9.0,
        ..Fog::default()
    };
    let mut frame = mist_frame(view, fog, tier);
    frame.draws = draws;
    frame.sky = sky_with_mean([0.05, 0.06, 0.08]);
    frame.ambient = vector(0.07, 0.08, 0.1);
    with_sun(&mut frame, view, [0.85, -0.38, 0.15], [3.2, 2.9, 2.4], 2048);
    frame.clear = Color { r: 0.02, g: 0.025, b: 0.03, a: 1.0 };
    frame
}

#[test]
fn a_hazy_room_shows_the_sun_in_shafts_past_its_pillars() {
    let Some(mut rig) = Rig::new("hazy-room", 384) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([12.0, 3.0, 14.0], [-4.0, 3.5, -2.0], 1.0, 0.55);
    let before = rig.settle(&room_frame(&view, 2, 0.0), 2);
    let after = rig.settle(&room_frame(&view, 2, 0.045), 48);
    assert_eq!(rig.renderer.stats().volume_froxels, 160 * 90 * 64);
    save("volumetrics-room-before.png", &before, rig.size);
    save("volumetrics-room-after.png", &after, rig.size);
    // The haze brightens the air between the pillars where the sun reaches it, and leaves
    // the shadowed air dark: the picture gains a wide range of new light in bands.
    let gain: Vec<f64> = after.iter().zip(&before).map(|(a, b)| luminance(*a) - luminance(*b)).collect();
    let bright = gain.iter().filter(|g| **g > 25.0).count();
    let dark = gain.iter().filter(|g| g.abs() < 4.0).count();
    eprintln!("room: {bright} pixels gain more than 25, {dark} stay within 4");
    assert!(bright > 3000, "shafts light the air: {bright}");
    assert!(dark > 3000, "the shadowed air and the lit surfaces' far side are left: {dark}");
    // Across a row of the picture the new light rises and falls between pillars: several bands.
    let row = rig.size as usize * 190;
    let band: Vec<f64> = (0..rig.size as usize).map(|x| gain[row + x]).collect();
    let crossings = band.windows(2).filter(|w| (w[0] - 20.0) * (w[1] - 20.0) < 0.0).count();
    eprintln!("room: row 190 crosses the shafts' edge {crossings} times");
    assert!(crossings >= 4, "the pillars cut the light into shafts: {crossings}");
}

#[test]
fn a_point_light_glows_in_the_mist_and_is_cut_by_what_stands_before_it() {
    let Some(mut rig) = Rig::new("point-glow", 384) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let view = camera([0.0, 2.2, 14.0], [0.0, 2.0, -6.0], 1.0, 0.55);
    let build = |tier: u32, density: f32| {
        let mut frame = mist_frame(
            &view,
            Fog {
                density,
                height: 9.0,
                distance: 60.0,
                anisotropy: 0.35,
                noise: 0.25,
                noise_scale: 6.0,
                ..Fog::default()
            },
            tier,
        );
        frame.draws = vec![
            draw(0, 0, IDENTITY),
            draw(1, 1, scale_at(1.2, 5.0, 1.2, -2.2, 0.0, -2.0)),
            draw(1, 2, scale_at(2.4, 2.0, 1.0, 1.8, 0.0, -1.0)),
        ];
        frame.sky = sky_with_mean([0.025, 0.03, 0.045]);
        frame.ambient = vector(0.03, 0.035, 0.05);
        frame.light_color = vector(0.0, 0.0, 0.0);
        frame.clear = Color { r: 0.01, g: 0.012, b: 0.02, a: 1.0 };
        frame.lights = vec![Light {
            position: vector(0.2, 3.2, -6.0),
            radius: 40.0,
            color: vector(110.0, 80.0, 48.0),
            direction: vector(0.0, -1.0, 0.0),
            inner: -1.0,
            outer: -1.0,
            shadow: true,
            id: 1,
        }];
        frame.shadow.local = 2;
        frame.shadow.local_texels = 1024;
        frame
    };
    let before = rig.settle(&build(2, 0.0), 2);
    let after = rig.settle(&build(2, 0.05), 48);
    save("volumetrics-point-before.png", &before, rig.size);
    save("volumetrics-point-after.png", &after, rig.size);
    let gain: Vec<f64> = after.iter().zip(&before).map(|(a, b)| luminance(*a) - luminance(*b)).collect();
    let glow = gain.iter().filter(|g| **g > 20.0).count();
    eprintln!("point light: {glow} pixels gain more than 20");
    assert!(glow > 4000, "the mist glows round the light: {glow}");
    // The glow falls off from the light: the brightest gain is near its screen position.
    let size = rig.size as usize;
    let window = |x0: usize, y0: usize, side: usize| -> f64 {
        let mut total = 0.0;
        for y in y0..y0 + side {
            for x in x0..x0 + side {
                total += gain[y * size + x];
            }
        }
        total / (side * side) as f64
    };
    let near_light = window(size / 2 - 40, size / 2 - 5, 80);
    let corner = window(0, size - 80, 80);
    assert!(near_light > 4.0 * corner.max(1.0), "the glow falls off with distance: {near_light} against {corner}");
}

/// What the volume costs the GPU, at each tier (`cargo test --release -- --ignored --nocapture
/// the_volumes_gpu_time`): the injection and the integration of the hazy room, in milliseconds.
#[test]
#[ignore = "a measurement, not a witness"]
fn the_volumes_gpu_time_by_tier() {
    let Some(mut rig) = Rig::new("cost", 512) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    rig.renderer.time_passes(true);
    let view = camera([12.0, 3.0, 14.0], [-4.0, 3.5, -2.0], 1.0, 0.55);
    for (tier, name) in [(0u32, "off"), (1, "low"), (2, "high")] {
        let frame = room_frame(&view, tier, 0.045);
        let mut best = 0.0f32;
        let mut total = 0.0;
        for index in 0..60 {
            rig.draw(&frame);
            if index >= 30 {
                if let Some(gpu) = rig.renderer.stats().gpu {
                    best = best.max(gpu.volume);
                    total = gpu.total;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        eprintln!("{name}: volume {best:.3} ms (largest of 30 frames) in a {total:.3} ms frame");
    }
}
