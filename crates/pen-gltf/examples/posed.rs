//! Imports a glTF file and draws each skinned mesh of it posed by each of its
//! clips at five times through the clip, headlessly, into one PNG a clip
//! (the times side by side): a look at how an import moves without a game.
//! Then a crowd of 25 of it baked into a vertex-animation texture, each
//! playing its own clip and time, as one instanced draw, at two moments.
//!
//! cargo run --release -p pen-gltf --example posed -- MODEL.glb OUT_DIR [SIZE]
//!
//! The assets go to OUT_DIR/assets, drawn with a plain lit skinned shader
//! (and the import's vat.wgsl) from a three-quarter view that fits them.
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use pen_protocol::*;
use pen_render::{RenderTarget, Renderer, RendererOptions};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
const TIMES: usize = 5;

// Lit by the frame's light and ambient, its material's texture and color;
// moved by its joints when posed.
const SHADER: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f, ambient: vec4f, light_direction: vec4f, light_color: vec4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@group(1) @binding(1) var base: texture_2d<f32>;
@group(1) @binding(2) var base_sampler: sampler;
@group(3) @binding(0) var<storage, read> joints: array<mat4x4f>;
struct Out { @builtin(position) position: vec4f, @location(0) normal: vec3f, @location(1) uv: vec2f };
struct Plain { @location(0) position: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f };
struct Skinned {
  @location(0) position: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f,
  @location(9) joints: vec4u, @location(10) weights: vec4f,
};
fn placed(skin: mat4x4f, position: vec3f, normal: vec3f, uv: vec2f) -> Out {
  var out: Out;
  out.position = scene.projection * scene.view * draw.model * skin * vec4f(position, 1.0);
  out.normal = (draw.normal * skin * vec4f(normal, 0.0)).xyz;
  out.uv = uv;
  return out;
}
@vertex fn vertex(input: Plain) -> Out {
  return placed(mat4x4f(vec4f(1.0, 0.0, 0.0, 0.0), vec4f(0.0, 1.0, 0.0, 0.0), vec4f(0.0, 0.0, 1.0, 0.0),
    vec4f(0.0, 0.0, 0.0, 1.0)), input.position, input.normal, input.uv);
}
@vertex fn vertex_skinned(input: Skinned) -> Out {
  let skin = joints[input.joints.x] * input.weights.x + joints[input.joints.y] * input.weights.y
    + joints[input.joints.z] * input.weights.z + joints[input.joints.w] * input.weights.w;
  return placed(skin, input.position, input.normal, input.uv);
}
@fragment fn fragment(input: Out) -> @location(0) vec4f {
  let albedo = draw.color.rgb * textureSample(base, base_sampler, input.uv).rgb;
  let normal = normalize(input.normal);
  let sun = max(dot(normal, -scene.light_direction.xyz), 0.0);
  return vec4f(albedo * (scene.ambient.xyz + scene.light_color.xyz * sun), 1.0);
}
"#;

struct Offscreen(wgpu::Texture);

impl RenderTarget for Offscreen {
    fn size(&self) -> (u32, u32) {
        (self.0.width(), self.0.height())
    }
    fn view(&mut self) -> Option<wgpu::TextureView> {
        Some(self.0.create_view(&Default::default()))
    }
    fn present(&mut self) {}
}

fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let (Some(model), Some(out)) = (arguments.first(), arguments.get(1)) else {
        bail!("posed MODEL.glb OUT_DIR [SIZE]");
    };
    let size: u32 = arguments.get(2).map_or(Ok(320), |size| size.parse())?;
    let (model, out) = (PathBuf::from(model), PathBuf::from(out));
    let assets = out.join("assets");
    let options = pen_gltf::Options {
        shader: "posed.wgsl".to_owned(),
        vat: Some(pen_gltf::VatOptions { fps: 30.0, shader: "vat.wgsl".to_owned() }),
        ..Default::default()
    };
    let imported = pen_gltf::import(&model, &assets, &options)?;
    std::fs::write(assets.join("posed.wgsl"), SHADER)?;
    let skinned: Vec<&pen_gltf::ImportedMesh> = imported.meshes.iter().filter(|mesh| mesh.joints > 0).collect();
    if skinned.is_empty() {
        bail!("{} has no skinned meshes", model.display());
    }
    // Each mesh's material, then its vertex-animation material.
    let materials: Vec<String> = skinned
        .iter()
        .map(|mesh| mesh.material.clone())
        .chain(skinned.iter().map(|mesh| mesh.vat.clone().unwrap_or_else(|| mesh.material.clone())))
        .collect();
    let catalog = serde_json::json!({
        "meshes": skinned.iter().map(|mesh| mesh.mesh.clone()).collect::<Vec<_>>(),
        "materials": materials,
    });
    std::fs::write(assets.join("catalog.json"), catalog.to_string())?;
    let catalog = boundary::read_catalog(&assets)?;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).context("no GPU adapter")?;
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default()))?;
    let options = RendererOptions {
        format: FORMAT,
        display: pen_render::Display::SDR,
        assets: assets.clone(),
        veil: None,
        pipeline_cache: None,
    };
    let mut renderer = Renderer::new(device.clone(), queue.clone(), options)?;
    renderer.preload(&catalog);
    let mut target = Offscreen(device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }));
    for (index, mesh) in skinned.iter().enumerate() {
        #[derive(serde::Deserialize)]
        struct Positions {
            positions: Vec<f32>,
        }
        let Positions { positions } = serde_json::from_slice(&std::fs::read(assets.join(&mesh.mesh))?)?;
        let (view, projection) = camera(&positions);
        let draw = Draw {
            mesh: u32::try_from(index)?,
            material: u32::try_from(index)?,
            transform: IDENTITY,
            parameters: None,
            extra: None,
        };
        for (clip_index, clip) in imported.clips.iter().enumerate() {
            let mut strip = Vec::new();
            for step in 0..TIMES {
                let time = clip.duration * step as f32 / (TIMES - 1) as f32;
                let pose = Pose {
                    draw: 0,
                    clip: u32::try_from(clip_index)?,
                    time: time.min(clip.duration * 0.999),
                    weight: 1.0,
                };
                let frame = frame(vec![draw.clone()], vec![pose], view, projection);
                settle(&mut renderer, &frame, &catalog, &mut target)?;
                strip.push(pixels(&device, &queue, &target.0));
            }
            let name = format!("{}-{}.png", mesh.mesh.trim_end_matches(".mesh.json"), clip.name);
            let path = out.join(name);
            write_strip(&strip, size, &path)?;
            println!(
                "{} at {:?} s: {}",
                clip.name,
                (0..TIMES).map(|step| clip.duration * step as f32 / (TIMES - 1) as f32).collect::<Vec<_>>(),
                path.display()
            );
        }
        if mesh.vat.is_none() {
            continue;
        }
        // A crowd: 5 x 5 of it, a step apart, each its own clip and time,
        // drawn with the vertex-animation material.
        let step = 1.3 * spread(&positions);
        let material = u32::try_from(skinned.len() + index)?;
        let clips = u32::try_from(imported.clips.len().max(1))?;
        let crowd = |moment: f32| -> Vec<Draw> {
            (0u32..25)
                .map(|member| {
                    let mut transform = IDENTITY;
                    transform[12] = (member % 5) as f32 * step;
                    transform[14] = (member / 5) as f32 * step;
                    let clip = member % clips;
                    let time = moment + member as f32 * 0.137;
                    Draw {
                        mesh: draw.mesh,
                        material,
                        transform,
                        parameters: Some([clip as f32, time, 0.0, 0.0]),
                        extra: None,
                    }
                })
                .collect()
        };
        let mut everywhere = Vec::new();
        for member in crowd(0.0) {
            everywhere.extend(
                positions
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .flat_map(|point| [point[0] + member.transform[12], point[1], point[2] + member.transform[14]]),
            );
        }
        let (view, projection) = camera(&everywhere);
        let mut strip = Vec::new();
        for moment in [0.0, 0.7] {
            let frame = frame(crowd(moment), Vec::new(), view, projection);
            settle(&mut renderer, &frame, &catalog, &mut target)?;
            let stats = renderer.stats();
            println!(
                "crowd at {moment} s: {} draw calls, {} instanced, {} instances",
                stats.draw_calls, stats.instanced_draw_calls, stats.instances
            );
            strip.push(pixels(&device, &queue, &target.0));
        }
        let path = out.join(format!("{}-crowd.png", mesh.mesh.trim_end_matches(".mesh.json")));
        write_strip(&strip, size, &path)?;
        println!("crowd: {}", path.display());
    }
    Ok(())
}

/// How far the mesh reaches across x and z.
fn spread(positions: &[f32]) -> f32 {
    let mut reach = [0.0f32; 2];
    for point in positions.as_chunks::<3>().0 {
        reach[0] = reach[0].max(point[0].abs());
        reach[1] = reach[1].max(point[2].abs());
    }
    2.0 * reach[0].max(reach[1]).max(0.05)
}

const IDENTITY: Matrix = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

/// A three-quarter view from the front right and above, and an orthographic
/// projection fitting the bind pose with room to move.
fn camera(positions: &[f32]) -> (Matrix, Matrix) {
    let mut low = [f32::INFINITY; 3];
    let mut high = [f32::NEG_INFINITY; 3];
    for point in positions.as_chunks::<3>().0 {
        for axis in 0..3 {
            low[axis] = low[axis].min(point[axis]);
            high[axis] = high[axis].max(point[axis]);
        }
    }
    let center = [0, 1, 2].map(|axis| (low[axis] + high[axis]) * 0.5);
    let reach =
        [0, 1, 2].map(|axis| high[axis] - low[axis]).iter().map(|extent| extent * extent).sum::<f32>().sqrt() * 0.75;
    // Looking from (1, 0.6, 1) toward the center: right, up and back axes.
    let back = normalize([1.0, 0.6, 1.0]);
    let right = normalize(cross([0.0, 1.0, 0.0], back));
    let up = cross(back, right);
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
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
        -dot(right, center),
        -dot(up, center),
        -dot(back, center),
        1.0,
    ];
    let scale = 1.0 / reach.max(1e-3);
    // z from reach (near) to -reach (far) into 0..1.
    let projection = [scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, -0.5 * scale, 0.0, 0.0, 0.0, 0.5, 1.0];
    (view, projection)
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|value| value / length)
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn frame(draws: Vec<Draw>, poses: Vec<Pose>, view: Matrix, projection: Matrix) -> RenderFrame {
    RenderFrame {
        ambient: Vector { x: 0.35, y: 0.37, z: 0.42 },
        light_direction: Vector { x: -0.4, y: -0.8, z: -0.45 },
        light_color: Vector { x: 0.95, y: 0.9, z: 0.8 },
        clear: Color { r: 0.12, g: 0.13, b: 0.16, a: 1.0 },
        quality: Quality { msaa: 4, scale: 1.0, filtering: 1, ..Default::default() },
        sky: Sky { sun: Vector { x: 0.0, y: 1.0, z: 0.0 }, ..Default::default() },
        fog: Fog { radius: 0.0, density: 0.0, height: 1.0 },
        shadow: Shadow { view: IDENTITY, projection: IDENTITY, texel: 0.0, texels: 0 },
        view,
        projection,
        draws,
        poses,
        ..Default::default()
    }
}

/// Draws `frame` until nothing is loading or compiling, then once more.
fn settle(renderer: &mut Renderer, frame: &RenderFrame, catalog: &Catalog, target: &mut Offscreen) -> Result<()> {
    let started = Instant::now();
    loop {
        renderer.submit(frame, catalog, target)?;
        for event in renderer.take_events() {
            if let GameEvent::AssetFailed { key, message, .. } = event {
                bail!("{key}: {message}");
            }
        }
        if renderer.pending() == 0 && started.elapsed() > Duration::from_millis(50) {
            break;
        }
        if started.elapsed() > Duration::from_secs(30) {
            bail!("assets still loading after 30 s");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    renderer.submit(frame, catalog, target)
}

fn pixels(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let row = (width * 4).next_multiple_of(256);
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
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let bytes = buffer.slice(..).get_mapped_range().unwrap();
    bytes
        .chunks(row as usize)
        .flat_map(|line| {
            line[..(width * 4) as usize].as_chunks::<4>().0.iter().flat_map(|&[b, g, r, _]| [r, g, b, 255])
        })
        .collect()
}

/// The images side by side.
fn write_strip(images: &[Vec<u8>], size: u32, path: &std::path::Path) -> Result<()> {
    let count = u32::try_from(images.len())?;
    let mut strip = image::RgbaImage::new(size * count, size);
    for (index, pixels) in (0u32..).zip(images) {
        let image = image::RgbaImage::from_raw(size, size, pixels.clone()).context("an image of the wrong size")?;
        image::imageops::replace(&mut strip, &image, i64::from(size * index), 0);
    }
    strip.save(path)?;
    Ok(())
}
