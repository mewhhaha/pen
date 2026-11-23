//! Image quality on a real (headless) GPU: temporal antialiasing against an
//! analytic edge, clustered lights against every light summed, KTX2 variants
//! by device features, HDR output read back as the display would get it, and
//! MSDF text magnified.
//! Each test writes what it drew to target/look-shots. Skips when no adapter
//! is available.
#![cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

mod common;

use common::*;
use pen_protocol::*;
use pen_render::{Display, RenderTarget, Renderer, RendererOptions};

// Draws its mesh in its material's color, wherever the draw puts it.
const FLAT: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return draw.color; }
"#;

// Draws its mesh with its texture's middle texel.
const TEXTURED: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
@group(1) @binding(1) var image: texture_2d<f32>;
@group(1) @binding(2) var image_sampler: sampler;
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return scene.projection * scene.view * draw.model * vec4f(position, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return textureSampleLevel(image, image_sampler, vec2f(0.5), 0.0); }
"#;

// The frame's lights on a floor facing up: through their clusters, or
// (`${sum}` set) every light summed, for the clusters to match.
const LIT: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
${lights}
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) view_z: f32,
};
@vertex fn vertex(@location(0) position: vec3f) -> Corner {
  let world = draw.model * vec4f(position, 1.0);
  let seen = scene.view * world;
  return Corner(scene.projection * seen, world.xyz, seen.z);
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let up = vec3f(0.0, 1.0, 0.0);
  var light = vec3f(0.0);
  if (${sum}) {
    for (var index = 0u; index < cluster_grid.counts.w; index++) {
      light += cluster_light(cluster_lights[index], corner.world, up);
    }
  } else {
    light = clustered_light(corner.world, up, corner.position, corner.view_z);
  }
  return vec4f(light, 1.0);
}
"#;

fn shots() -> PathBuf {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/look-shots");
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

/// A device with the features the renderer uses where the adapter has them.
fn featured_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    let descriptor =
        wgpu::DeviceDescriptor { required_features: pen_render::optional_features(&adapter), ..Default::default() };
    pollster::block_on(adapter.request_device(&descriptor)).ok()
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

    /// Each pixel's green, of a BGRA8 target.
    fn greens(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f64> {
        self.read(device, queue).chunks(4).map(|texel| f64::from(texel[1]) / 255.0).collect()
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

fn save_grey(name: &str, values: &[f64], width: u32, height: u32, zoom: u32) {
    let image = image::RgbaImage::from_fn(width * zoom, height * zoom, |x, y| {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a value in 0..=255")]
        let value = (values[((y / zoom) * width + x / zoom) as usize] * 255.0).round().clamp(0.0, 255.0) as u8;
        image::Rgba([value, value, value, 255])
    });
    image.save(shots().join(name)).unwrap();
}

// Transparent draws do not write depth. Their coverage must still replace
// the clear color with sky, including with MSAA and overlapping surfaces.
#[test]
fn transparent_surfaces_keep_the_sky_behind_them() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mesh = br#"{"positions":[-1,-1,0.5, 1,-1,0.5, -1,1,0.5, 1,1,0.5],"normals":[0,0,1,0,0,1,0,0,1,0,0,1],
                    "uvs":[0,0,1,0,0,1,1,1],"indices":[0,1,2,2,1,3]}"#;
    let root = assets(
        "transparent-sky",
        &[
            (
                "catalog.json",
                br#"{"meshes":["quad.mesh.json"],"materials":["glass.material.json","solid.material.json","inset.material.json"]}"#,
            ),
            ("quad.mesh.json", mesh),
            ("inset.material.json", br#"{"shader":"flat.wgsl","color":[0,0.8,0,1],"lighting":"unlit"}"#),
            ("flat.wgsl", FLAT.as_bytes()),
            ("glass.wgsl", FLAT.replace("break draw.color;", "break vec4f(draw.color.rgb, 0.5);").as_bytes()),
            ("low-alpha.wgsl", FLAT.replace("break draw.color;", "break vec4f(draw.color.rgb, 0.1);").as_bytes()),
            ("glass.material.json", br#"{"shader":"glass.wgsl","color":[0,0.8,0,1],"lighting":"unlit","blend":true}"#),
            ("solid.material.json", br#"{"shader":"low-alpha.wgsl","color":[0,0.8,0,1],"lighting":"unlit"}"#),
        ],
    );
    let catalog = boundary::read_catalog(&root).unwrap();
    let mut renderer = renderer(&device, &queue, root, FORMAT, Display::SDR);
    let mut target = Target::new(&device, 16, 16, FORMAT);
    let glass = Draw { mesh: 0, material: 0, transform: IDENTITY, parameters: None, extra: None };
    let mut frame = frame(vec![glass.clone()]);
    // A nonzero clear color checks that it is removed before sky is added.
    frame.clear = Color { r: 0.35, g: 0.1, b: 0.2, a: 0.3 };
    frame.quality.sky = true;
    let sky = Vector { x: 0.0, y: 0.2, z: 0.0 };
    frame.sky.zenith = sky;
    frame.sky.horizon = sky;
    frame.sky.ground = sky;
    frame.light_color = Vector::default();
    let check = |target: &Target, label: &str, expected: f32| {
        let actual = middle(&device, &queue, &target.0);
        assert!(actual[0] <= 1 && actual[2] <= 1, "{label}: leaked clear color {actual:?}");
        assert!(
            (f32::from(actual[1]) / 255.0 - expected).abs() < 0.01,
            "{label}: {actual:?}, expected green {expected}"
        );
    };
    for samples in [1, 4] {
        frame.quality.msaa = samples;
        frame.draws = vec![glass.clone()];
        settle(&mut renderer, &frame, &catalog, &mut target, 1);
        check(&target, "one translucent surface", 0.5);
        frame.draws.push(glass.clone());
        settle(&mut renderer, &frame, &catalog, &mut target, 2);
        check(&target, "overlapping translucent surfaces", 0.65);
        frame.draws = vec![Draw { material: 1, ..glass.clone() }];
        settle(&mut renderer, &frame, &catalog, &mut target, 1);
        check(&target, "opaque ignores shader alpha", 0.8);
    }
    // Insets retain their separate authored coverage: an opaque mesh with
    // alpha 1 must cover the sky, and empty pixels must leave it untouched.
    frame.draws.clear();
    let mut inset_draw = Draw { material: 2, ..glass };
    inset_draw.transform[0] = 0.5;
    inset_draw.transform[5] = 0.5;
    frame.insets = vec![Inset {
        width: 16.0,
        height: 16.0,
        view: IDENTITY,
        projection: IDENTITY,
        draws: vec![inset_draw],
        ..Inset::default()
    }];
    settle(&mut renderer, &frame, &catalog, &mut target, 1);
    check(&target, "opaque inset", 0.8);
    // The existing inset policy uses authored fragment alpha as coverage,
    // even for its opaque pipeline. Preserve that behavior exactly.
    frame.insets[0].draws[0].material = 1;
    settle(&mut renderer, &frame, &catalog, &mut target, 1);
    check(&target, "authored inset coverage", 0.27986);
    let corner = pixels(&device, &queue, &target.0)[0];
    assert!((f32::from(corner[1]) / 255.0 - 0.2).abs() < 0.01, "empty inset corner: {corner:?}");
}

// The edge the TAA test draws: green where y > 0.37 x + 0.11 in clip space.
fn covered(x: f64, y: f64) -> bool {
    y > 0.37 * x + 0.11
}

const SIZE: u32 = 48;

/// The share of pixel (x, y) the edge's shape covers, from 16 x 16 samples.
fn coverage(x: u32, y: u32) -> f64 {
    let mut inside = 0;
    for sy in 0..16 {
        for sx in 0..16 {
            let px = f64::from(x) + (f64::from(sx) + 0.5) / 16.0;
            let py = f64::from(y) + (f64::from(sy) + 0.5) / 16.0;
            let ndc = (px / f64::from(SIZE) * 2.0 - 1.0, 1.0 - py / f64::from(SIZE) * 2.0);
            inside += i32::from(covered(ndc.0, ndc.1));
        }
    }
    f64::from(inside) / 256.0
}

#[test]
fn taa_smooths_a_still_edge_and_follows_the_camera() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    // A shape whose lower edge crosses the picture: y = 0.37 x + 0.11.
    let mesh = br#"{"positions":[-3,-1.0,0.5, 3,1.22,0.5, -3,3,0.5, 3,3,0.5],"normals":[0,0,1,0,0,1,0,0,1,0,0,1],
                    "uvs":[0,0,1,0,0,1,1,1],"indices":[0,1,2,2,1,3]}"#;
    let root = assets(
        "taa",
        &[
            ("catalog.json", br#"{"meshes":["shape.mesh.json"],"materials":["green.material.json"]}"#),
            ("shape.mesh.json", mesh),
            ("flat.wgsl", FLAT.as_bytes()),
            ("green.material.json", br#"{"shader":"flat.wgsl","color":[0,0.8,0,1],"lighting":"unlit"}"#),
        ],
    );
    let catalog = boundary::read_catalog(&root).unwrap();
    let mut renderer = renderer(&device, &queue, root, FORMAT, Display::SDR);
    let mut target = Target::new(&device, SIZE, SIZE, FORMAT);
    let mut frame = frame(vec![Draw { mesh: 0, material: 0, transform: IDENTITY, parameters: None, extra: None }]);
    settle(&mut renderer, &frame, &catalog, &mut target, 1);
    assert_eq!(renderer.passes(), ["present"]);
    let aliased = target.greens(&device, &queue);
    frame.quality.taa = true;
    for _ in 0..40 {
        renderer.submit(&frame, &catalog, &mut target).unwrap();
    }
    assert_eq!(renderer.passes(), ["taa", "present"]);
    let smoothed = target.greens(&device, &queue);
    let truth: Vec<f64> = (0..SIZE * SIZE).map(|index| coverage(index % SIZE, index / SIZE) * 0.8).collect();
    // Pixels the edge crosses.
    let edge: Vec<usize> = (0..truth.len()).filter(|&index| truth[index] > 0.02 && truth[index] < 0.78).collect();
    let error =
        |image: &[f64]| edge.iter().map(|&index| (image[index] - truth[index]).abs()).sum::<f64>() / edge.len() as f64;
    let (before, after) = (error(&aliased), error(&smoothed));
    eprintln!("edge pixels {}: mean error {before:.4} aliased, {after:.4} with TAA", edge.len());
    assert!(edge.len() > 40);
    assert!(after < before * 0.5, "TAA should bring the edge toward its coverage: {before:.4} -> {after:.4}");
    // Away from the edge nothing changes.
    for index in 0..truth.len() {
        if truth[index] <= 0.0 || truth[index] >= 0.8 {
            let x = index % SIZE as usize;
            let y = index / SIZE as usize;
            let near_edge = edge.iter().any(|&other| (other % 48).abs_diff(x) <= 1 && (other / 48).abs_diff(y) <= 1);
            if !near_edge {
                assert!((smoothed[index] - aliased[index]).abs() < 2.0 / 255.0, "pixel {x},{y} moved");
            }
        }
    }
    save_grey("taa-edge-aliased.png", &aliased, SIZE, SIZE, 8);
    save_grey("taa-edge-smoothed.png", &smoothed, SIZE, SIZE, 8);
    save_grey("taa-edge-truth.png", &truth, SIZE, SIZE, 8);

    // The camera moves 0.7 pixels a frame: the history follows it, so the
    // edge stays where the camera puts it, with no trail behind.
    for step in 1..=24 {
        frame.view[12] = step as f32 * 0.7 * 2.0 / SIZE as f32;
        renderer.submit(&frame, &catalog, &mut target).unwrap();
    }
    let moving = target.greens(&device, &queue);
    frame.quality.taa = false;
    renderer.submit(&frame, &catalog, &mut target).unwrap();
    let still = target.greens(&device, &queue);
    let mut trail = 0;
    for index in 0..still.len() {
        let (x, y) = (index % SIZE as usize, index / SIZE as usize);
        // More than a pixel from the moved edge, the TAA picture is the plain one.
        let near = (0..still.len()).any(|other| {
            still[other] != still[index] && (other % 48).abs_diff(x) <= 1 && (other / 48).abs_diff(y) <= 1
        });
        if !near && (moving[index] - still[index]).abs() > 0.1 {
            trail += 1;
        }
    }
    save_grey("taa-edge-moving.png", &moving, SIZE, SIZE, 8);
    assert_eq!(trail, 0, "TAA left a trail behind the moving edge");
}

// A checkerboard of squares half a world unit across, from where each
// fragment is in the world.
const CHECKER: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
struct Corner { @builtin(position) position: vec4f, @location(0) world: vec3f };
@vertex fn vertex(@location(0) position: vec3f) -> Corner {
  let world = draw.model * vec4f(position, 1.0);
  return Corner(scene.projection * scene.view * world, world.xyz);
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let square = floor(corner.world.xy * 2.0);
  let dark = (i32(square.x) + i32(square.y)) % 2 == 0;
  return vec4f(0.0, select(0.75, 0.2, dark), 0.0, 1.0);
}
"#;

#[test]
fn taa_keeps_a_still_perspective_scene_sharp() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    // A checkered card 5 ahead of the eye, before a grey wall 15 ahead.
    let quad = |half: f32, z: f32| {
        format!(
            r#"{{"positions":[{l},{l},{z}, {h},{l},{z}, {l},{h},{z}, {h},{h},{z}],"normals":[0,0,1,0,0,1,0,0,1,0,0,1],
                "uvs":[0,0,1,0,0,1,1,1],"indices":[0,1,2,2,1,3]}}"#,
            l = -half,
            h = half
        )
    };
    let (card, wall) = (quad(1.5, -5.0), quad(30.0, -15.0));
    let root = assets(
        "taa-still",
        &[
            ("catalog.json", br#"{"meshes":["card.mesh.json","wall.mesh.json"],"materials":["card.material.json","wall.material.json"]}"#),
            ("card.mesh.json", card.as_bytes()),
            ("wall.mesh.json", wall.as_bytes()),
            ("checker.wgsl", CHECKER.as_bytes()),
            ("flat.wgsl", FLAT.as_bytes()),
            ("card.material.json", br#"{"shader":"checker.wgsl","color":[1,1,1,1],"lighting":"unlit"}"#),
            ("wall.material.json", br#"{"shader":"flat.wgsl","color":[0,0.4,0,1],"lighting":"unlit"}"#),
        ],
    );
    let catalog = boundary::read_catalog(&root).unwrap();
    let mut renderer = renderer(&device, &queue, root, FORMAT, Display::SDR);
    let side = 128;
    let mut target = Target::new(&device, side, side, FORMAT);
    let draw = |mesh| Draw { mesh, material: mesh, transform: IDENTITY, parameters: None, extra: None };
    let mut frame = frame(vec![draw(0), draw(1)]);
    let (near, far, f) = (0.1f32, 100.0f32, 1.0 / 0.5f32.tan());
    frame.projection = [
        f,
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
    settle(&mut renderer, &frame, &catalog, &mut target, 2);
    let plain = target.greens(&device, &queue);
    frame.quality.taa = true;
    for _ in 0..48 {
        renderer.submit(&frame, &catalog, &mut target).unwrap();
    }
    let smoothed = target.greens(&device, &queue);
    save_grey("taa-still-plain.png", &plain, side, side, 4);
    save_grey("taa-still-smoothed.png", &smoothed, side, side, 4);
    // The card covers about the middle half; its inside, and the wall well
    // away from it.
    let card_inside: Vec<usize> = (0..plain.len())
        .filter(|&index| {
            let (x, y) = (index % 128, index / 128);
            (40..88).contains(&x) && (40..88).contains(&y)
        })
        .collect();
    let wall_away: Vec<usize> = (0..plain.len())
        .filter(|&index| {
            let (x, y) = (index % 128, index / 128);
            !(16..112).contains(&x) || !(16..112).contains(&y)
        })
        .collect();
    let spread = |image: &[f64]| {
        let mean = card_inside.iter().map(|&index| image[index]).sum::<f64>() / card_inside.len() as f64;
        (card_inside.iter().map(|&index| (image[index] - mean).powi(2)).sum::<f64>() / card_inside.len() as f64).sqrt()
    };
    let kept = spread(&smoothed) / spread(&plain);
    let wall_moved = wall_away.iter().map(|&index| (smoothed[index] - plain[index]).abs()).fold(0.0, f64::max);
    eprintln!("TAA keeps {:.1}% of the checker's contrast; the wall moves {wall_moved:.4}", kept * 100.0);
    assert!(kept > 0.85, "TAA blurred a still picture: {kept:.3} of its contrast");
    assert!(wall_moved < 2.0 / 255.0, "the wall changed: {wall_moved}");
    // Against the scene itself, 16 x 16 samples a pixel: TAA comes closer
    // than one sample does, and nowhere far off (nothing moved).
    let truth: Vec<f64> = (0..plain.len())
        .map(|index| {
            let (x, y) = ((index % 128) as f64, (index / 128) as f64);
            let mut sum = 0.0;
            for sample in 0..256 {
                let (sx, sy) = (x + (f64::from(sample % 16) + 0.5) / 16.0, y + (f64::from(sample / 16) + 0.5) / 16.0);
                // Where the pixel's ray meets the card, 5 ahead.
                let ndc = (sx / 64.0 - 1.0, 1.0 - sy / 64.0);
                let world = (ndc.0 * 5.0 / f64::from(f), ndc.1 * 5.0 / f64::from(f));
                sum += if world.0.abs() > 1.5 || world.1.abs() > 1.5 {
                    0.4
                } else if ((world.0 * 2.0).floor() + (world.1 * 2.0).floor()).rem_euclid(2.0) == 0.0 {
                    0.2
                } else {
                    0.75
                };
            }
            sum / 256.0
        })
        .collect();
    let error = |image: &[f64]| {
        card_inside.iter().map(|&index| (image[index] - truth[index]).abs()).sum::<f64>() / card_inside.len() as f64
    };
    let worst = card_inside.iter().map(|&index| (smoothed[index] - truth[index]).abs()).fold(0.0, f64::max);
    eprintln!(
        "mean error against the scene {:.4} plain, {:.4} with TAA; worst {worst:.3}",
        error(&plain),
        error(&smoothed)
    );
    assert!(error(&smoothed) < error(&plain), "TAA should come closer to the scene");
    assert!(worst < 0.2, "TAA moved the picture: {worst}");
}

/// A look-at camera from `eye` to the origin and a perspective projection
/// into WebGPU's 0..1 depth.
fn camera(eye: [f32; 3], aspect: f32) -> (Matrix, Matrix) {
    let length = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let normalize = |v: [f32; 3]| [v[0] / length(v), v[1] / length(v), v[2] / length(v)];
    let cross =
        |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let back = normalize(eye);
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
    let (near, far, f) = (0.1f32, 100.0f32, 1.0 / 0.6f32.tan());
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

#[test]
fn clustered_lights_match_every_light_summed() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let lit = LIT.replace("${lights}", pen_render::LIGHTS_WGSL);
    let floor = br#"{"positions":[-12,0,-12, 12,0,-12, -12,0,12, 12,0,12],"normals":[0,1,0,0,1,0,0,1,0,0,1,0],
                     "uvs":[0,0,1,0,0,1,1,1],"indices":[0,2,1,1,2,3]}"#;
    let root = assets(
        "lights",
        &[
            (
                "catalog.json",
                br#"{"meshes":["floor.mesh.json"],"materials":["clustered.material.json","summed.material.json"]}"#,
            ),
            ("floor.mesh.json", floor),
            ("clustered.wgsl", lit.replace("${sum}", "false").as_bytes()),
            ("summed.wgsl", lit.replace("${sum}", "true").as_bytes()),
            ("clustered.material.json", br#"{"shader":"clustered.wgsl","color":[1,1,1,1],"lighting":"unlit"}"#),
            ("summed.material.json", br#"{"shader":"summed.wgsl","color":[1,1,1,1],"lighting":"unlit"}"#),
        ],
    );
    let catalog = boundary::read_catalog(&root).unwrap();
    let mut renderer = renderer(&device, &queue, root, FORMAT, Display::SDR);
    let (width, height) = (320, 180);
    let mut target = Target::new(&device, width, height, FORMAT);
    let (view, projection) = camera([0.0, 9.0, 14.0], width as f32 / height as f32);
    let mut seed = 11u32;
    let mut random = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    let lights: Vec<Light> = (0..400)
        .map(|index| {
            let spot = index % 3 == 0;
            Light {
                position: Vector { x: random() * 24.0 - 12.0, y: 0.3 + random() * 1.5, z: random() * 24.0 - 12.0 },
                radius: 0.6 + random() * 2.4,
                color: Vector { x: random() * 0.5, y: random() * 0.5, z: random() * 0.5 },
                direction: Vector { x: 0.0, y: -1.0, z: 0.0 },
                inner: if spot { 0.95 } else { -1.0 },
                outer: if spot { 0.8 } else { -1.0 },
            }
        })
        .collect();
    let draw = |material| Draw { mesh: 0, material, transform: IDENTITY, parameters: None, extra: None };
    let mut frame = frame(vec![draw(0)]);
    (frame.view, frame.projection, frame.lights) = (view, projection, lights);
    settle(&mut renderer, &frame, &catalog, &mut target, 1);
    let clustered = target.read(&device, &queue);
    frame.draws = vec![draw(1)];
    settle(&mut renderer, &frame, &catalog, &mut target, 1);
    let summed = target.read(&device, &queue);
    let largest = clustered.iter().zip(&summed).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
    let lit_pixels = clustered.chunks(4).filter(|texel| texel[..3].iter().any(|&value| value > 20)).count();
    eprintln!("{lit_pixels} lit pixels; largest difference from every light summed: {largest}");
    assert!(lit_pixels > 1500, "the lights should light the floor");
    assert!(largest <= 1, "clusters left out a light: difference {largest}");
    let image = image::RgbaImage::from_fn(width, height, |x, y| {
        let at = ((y * width + x) * 4) as usize;
        image::Rgba([clustered[at + 2], clustered[at + 1], clustered[at], 255])
    });
    image.save(shots().join("clustered-lights.png")).unwrap();
}

/// A BC7 block of one colour (mode 6, both endpoints the colour), its
/// channels odd so their low bit is the shared p-bit.
fn bc7_solid([r, g, b, a]: [u8; 4]) -> [u8; 16] {
    let mut bits = 0u128;
    let mut at = 0;
    let mut put = |value: u128, count: u32| {
        bits |= value << at;
        at += count;
    };
    put(1 << 6, 7);
    for channel in [r, g, b, a] {
        put(u128::from(channel >> 1), 7);
        put(u128::from(channel >> 1), 7);
    }
    put(1, 1);
    put(1, 1);
    bits.to_le_bytes()
}

fn linear(srgb: u8) -> f64 {
    let value = f64::from(srgb) / 255.0;
    if value <= 0.04045 { value / 12.92 } else { ((value + 0.055) / 1.055).powf(2.4) }
}

#[test]
fn ktx2_textures_load_the_variant_the_device_samples() {
    let Some((device, queue)) = featured_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    // Orange in BC7 blocks, blue as plain RGBA: which one draws says which
    // variant loaded.
    let orange = [201, 101, 51, 255];
    let bc7: Vec<Vec<u8>> = [16u32, 8, 4, 2, 1]
        .iter()
        .map(|&side| (0..side.div_ceil(4).pow(2)).flat_map(|_| bc7_solid(orange)).collect())
        .collect();
    let rgba: Vec<Vec<u8>> =
        (0..5).map(|level| [41u8, 81, 201, 255].repeat(((16u32 >> level) * (16 >> level)) as usize)).collect();
    let root = assets(
        "ktx2",
        &[
            ("catalog.json", br#"{"meshes":["quad.mesh.json"],"materials":["textured.material.json"]}"#),
            (
                "quad.mesh.json",
                br#"{"positions":[-1,-1,0.5, 1,-1,0.5, -1,1,0.5, 1,1,0.5],"normals":[0,0,1,0,0,1,0,0,1,0,0,1],
                     "uvs":[0,1,1,1,0,0,1,0],"indices":[0,1,2,2,1,3]}"#,
            ),
            ("textured.wgsl", TEXTURED.as_bytes()),
            (
                "textured.material.json",
                br#"{"shader":"textured.wgsl","texture":"paint.ktx2","color":[1,1,1,1],"lighting":"unlit"}"#,
            ),
            ("paint.bc7.ktx2", &pen_texture::ktx2::write(pen_texture::ktx2::Format::Bc7Srgb, 16, 16, &bc7).unwrap()),
            (
                "paint.rgba.ktx2",
                &pen_texture::ktx2::write(pen_texture::ktx2::Format::Rgba8Srgb, 16, 16, &rgba).unwrap(),
            ),
        ],
    );
    let catalog = boundary::read_catalog(&root).unwrap();
    let draw = Draw { mesh: 0, material: 0, transform: IDENTITY, parameters: None, extra: None };
    let frame = frame(vec![draw]);
    let colour = |device: &wgpu::Device, queue: &wgpu::Queue| {
        let mut renderer = renderer(device, queue, root.clone(), FORMAT, Display::SDR);
        let mut target = Target::new(device, 16, 16, FORMAT);
        settle(&mut renderer, &frame, &catalog, &mut target, 1);
        let texel = target.read(device, queue);
        [texel[2], texel[1], texel[0]]
    };
    // The present pass shows linear light as it is.
    let expect = |srgb: [u8; 4]| srgb[..3].iter().map(|&channel| (linear(channel) * 255.0).round()).collect::<Vec<_>>();
    let near =
        |got: [u8; 3], want: Vec<f64>| got.iter().zip(&want).all(|(got, want)| (f64::from(*got) - want).abs() <= 2.0);
    if device.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        let got = colour(&device, &queue);
        assert!(near(got, expect(orange)), "BC7 variant: {got:?} for {:?}", expect(orange));
    } else {
        eprintln!("the adapter has no BC compression; only the RGBA variant is checked");
    }
    // A device without block compression reads the plain variant.
    let (plain_device, plain_queue) = common::device().unwrap();
    let got = colour(&plain_device, &plain_queue);
    assert!(near(got, expect([41, 81, 201, 255])), "RGBA variant: {got:?}");
}

/// A half float's value.
fn half(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f64::from(bits & 0x3ff);
    sign * if exponent == 0 {
        fraction / 1024.0 * 2f64.powi(-14)
    } else {
        (1.0 + fraction / 1024.0) * 2f64.powi(exponent - 15)
    }
}

fn pq(nits: f64) -> f64 {
    let (m1, m2, c1, c2, c3) = (0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875);
    let y = (nits / 10000.0).clamp(0.0, 1.0).powf(m1);
    ((c1 + c2 * y) / (1.0 + c3 * y)).powf(m2)
}

#[test]
fn hdr_output_reaches_past_white_in_the_surface_s_encoding() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let quad = br#"{"positions":[-1,-1,0.5, 1,-1,0.5, -1,1,0.5, 1,1,0.5],"normals":[0,0,1,0,0,1,0,0,1,0,0,1],
                    "uvs":[0,0,1,0,0,1,1,1],"indices":[0,1,2,2,1,3]}"#;
    let root = assets(
        "hdr",
        &[
            (
                "catalog.json",
                br#"{"meshes":["quad.mesh.json"],"materials":["grey.material.json","bright.material.json"]}"#,
            ),
            ("quad.mesh.json", quad),
            ("flat.wgsl", FLAT.as_bytes()),
            ("bright.wgsl", FLAT.replace("break draw.color;", "break vec4f(draw.color.rgb * 6.0, 1.0);").as_bytes()),
            ("grey.material.json", br#"{"shader":"flat.wgsl","color":[0.5,0.5,0.5,1],"lighting":"unlit"}"#),
            // Six times as bright: light past white.
            ("bright.material.json", br#"{"shader":"bright.wgsl","color":[0.5,0.5,0.5,1],"lighting":"unlit"}"#),
        ],
    );
    let catalog = boundary::read_catalog(&root).unwrap();
    let hdr = |color_space| Display { color_space, headroom: 4.0, white_nits: 203.0 };
    // Each display's grey and bright pixels as the surface reads them.
    let shown = |format: wgpu::TextureFormat, display: Display| -> [f64; 2] {
        let mut renderer = renderer(&device, &queue, root.clone(), format, display);
        let mut target = Target::new(&device, 8, 8, format);
        [0, 1].map(|material| {
            let frame = frame(vec![Draw { mesh: 0, material, transform: IDENTITY, parameters: None, extra: None }]);
            settle(&mut renderer, &frame, &catalog, &mut target, 1);
            let texel = target.read(&device, &queue);
            match format {
                wgpu::TextureFormat::Rgba16Float => half(u16::from_le_bytes([texel[0], texel[1]])),
                wgpu::TextureFormat::Rgb10a2Unorm => {
                    f64::from(u32::from_le_bytes(texel[..4].try_into().unwrap()) & 0x3ff) / 1023.0
                }
                _ => f64::from(texel[1]) / 255.0,
            }
        })
    };
    let sdr = shown(FORMAT, Display::SDR);
    let extended = shown(wgpu::TextureFormat::Rgba16Float, hdr(wgpu::SurfaceColorSpace::ExtendedSrgb));
    let linear_light = shown(wgpu::TextureFormat::Rgba16Float, hdr(wgpu::SurfaceColorSpace::ExtendedSrgbLinear));
    let ten_bit = shown(wgpu::TextureFormat::Rgb10a2Unorm, hdr(wgpu::SurfaceColorSpace::Bt2100Pq));
    eprintln!("grey and bright: SDR {sdr:?}, extended sRGB {extended:?}, scRGB {linear_light:?}, PQ {ten_bit:?}");
    // SDR rolls highlights off below white; HDR keeps grey where SDR has it
    // and lets highlights past white, up to the display's peak (4x white is
    // 1.82 encoded).
    assert!((sdr[0] - 0.5).abs() < 0.003 && sdr[1] < 1.0);
    assert!((extended[0] - 0.5).abs() < 0.003, "{extended:?}");
    assert!(extended[1] > 1.3 && extended[1] < 1.83, "{extended:?}");
    // scRGB is linear light over 80 nits.
    let decode =
        |encoded: f64| if encoded <= 0.04045 { encoded / 12.92 } else { ((encoded + 0.055) / 1.055).powf(2.4) };
    assert!((linear_light[0] - decode(0.5) * 203.0 / 80.0).abs() < 0.01, "{linear_light:?}");
    assert!((linear_light[1] - decode(extended[1]) * 203.0 / 80.0).abs() < 0.05, "{linear_light:?}");
    // PQ is absolute: grey's 43 nits (a neutral grey keeps its value in BT.2020).
    assert!((ten_bit[0] - pq(decode(0.5) * 203.0)).abs() < 0.003, "{ten_bit:?} vs {}", pq(decode(0.5) * 203.0));
    assert!(ten_bit[1] > pq(203.0), "{ten_bit:?}");
}

// A sprite of the atlas's field: the median of its channels, or (parameter x
// set) its alpha, the single-channel field, for comparison.
const FIELD_SPRITE: &str = r#"
struct Screen { size: vec4f };
struct Sprite { rect: vec4f, uv: vec4f, color: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> screen: Screen;
@group(0) @binding(1) var<storage, read> sprites: array<Sprite>;
@group(0) @binding(2) var image: texture_2d<f32>;
@group(0) @binding(3) var image_sampler: sampler;
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) single: f32,
};
@vertex fn vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> Corner {
  let sprite = sprites[index];
  let corner = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0))[vertex];
  let pixel = sprite.rect.xy + corner * sprite.rect.zw;
  return Corner(vec4f(pixel.x / screen.size.x * 2.0 - 1.0, 1.0 - pixel.y / screen.size.y * 2.0, 0.0, 1.0),
    mix(sprite.uv.xy, sprite.uv.zw, corner), sprite.parameters.x);
}
${msdf}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let field = textureSampleLevel(image, image_sampler, corner.uv, 0.0);
  let distance = select(msdf_median(field.rgb), field.a, corner.single > 0.5);
  let ink = msdf_coverage(distance, corner.uv, 4.0, vec2f(textureDimensions(image)));
  return vec4f(1.0, 1.0, 1.0, ink);
}
"#;

/// A five-pointed star in a 32-texel cell: sharp outward and inward corners.
fn star() -> Vec<(f64, f64)> {
    (0..10)
        .map(|index| {
            let angle = f64::from(index) * std::f64::consts::PI / 5.0 - std::f64::consts::FRAC_PI_2;
            let radius = if index % 2 == 0 { 13.0 } else { 5.5 };
            (16.0 + radius * angle.cos(), 16.5 + radius * angle.sin())
        })
        .collect()
}

fn inside(polygon: &[(f64, f64)], (x, y): (f64, f64)) -> bool {
    let mut winding = false;
    for (index, &(x0, y0)) in polygon.iter().enumerate() {
        let (x1, y1) = polygon[(index + 1) % polygon.len()];
        if (y0 > y) != (y1 > y) && x < x0 + (y - y0) / (y1 - y0) * (x1 - x0) {
            winding = !winding;
        }
    }
    winding
}

#[test]
fn msdf_text_stays_sharp_magnified() {
    use pen_texture::msdf::{Point, Segment, Shape};
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let polygon = star();
    let points: Vec<Point> = polygon.iter().map(|&(x, y)| Point::new(x, y)).collect();
    let contour =
        (0..points.len()).map(|index| Segment::Line(points[index], points[(index + 1) % points.len()])).collect();
    let field = pen_texture::msdf::generate(&Shape { contours: vec![contour] }, 32, 32, 4.0);
    let atlas = pen_texture::ktx2::write(pen_texture::ktx2::Format::Rgba8Unorm, 32, 32, &[field]).unwrap();
    let root = assets(
        "msdf",
        &[
            ("catalog.json", br#"{"meshes":["none.mesh.json"],"materials":["star.material.json"]}"#),
            ("field.wgsl", FIELD_SPRITE.replace("${msdf}", pen_render::MSDF_WGSL).as_bytes()),
            (
                "star.material.json",
                br#"{"shader":"field.wgsl","texture":"star.ktx2","color":[1,1,1,1],"lighting":"unlit"}"#,
            ),
            ("star.ktx2", &atlas),
        ],
    );
    let catalog = boundary::read_catalog(&root).unwrap();
    let mut renderer = renderer(&device, &queue, root, FORMAT, Display::SDR);
    // Eight times the atlas's size.
    let side = 256;
    let mut target = Target::new(&device, side, side, FORMAT);
    let sprite = |single: f32| Sprite {
        x: 0.0,
        y: 0.0,
        width: side as f32,
        height: side as f32,
        u0: 0.0,
        v0: 0.0,
        u1: 1.0,
        v1: 1.0,
        color: [1.0; 4],
        material: 0,
        parameters: [single, 0.0, 0.0, 0.0],
    };
    let mut drawn = Vec::new();
    for single in [0.0, 1.0] {
        let mut frame = frame(Vec::new());
        frame.sprites = vec![sprite(single)];
        let deadline = Instant::now() + Duration::from_secs(20);
        while !renderer.prepare_sprites(&frame.sprites, &catalog).unwrap() || renderer.pending() > 0 {
            renderer.submit(&frame, &catalog, &mut target).unwrap();
            assert!(Instant::now() < deadline, "the atlas did not load: {:?}", renderer.take_events());
            std::thread::sleep(Duration::from_millis(5));
        }
        renderer.submit(&frame, &catalog, &mut target).unwrap();
        drawn.push(target.greens(&device, &queue));
    }
    // The star itself, at each pixel's middle, in the cell's texels.
    let scale = f64::from(side) / 32.0;
    let truth: Vec<bool> = (0..side * side)
        .map(|index| {
            inside(&polygon, ((f64::from(index % side) + 0.5) / scale, (f64::from(index / side) + 0.5) / scale))
        })
        .collect();
    // Pixels more than a pixel from the edge that come out on the wrong side.
    let wrong = |image: &[f64]| {
        (0..truth.len())
            .filter(|&index| {
                let at = u32::try_from(index).unwrap();
                let (x, y) = (at % side, at / side);
                let settled = (x.saturating_sub(1)..=(x + 1).min(side - 1)).all(|nx| {
                    (y.saturating_sub(1)..=(y + 1).min(side - 1))
                        .all(|ny| truth[(ny * side + nx) as usize] == truth[index])
                });
                settled && (image[index] > 0.5) != truth[index]
            })
            .count()
    };
    let (multi, single) = (wrong(&drawn[0]), wrong(&drawn[1]));
    // The edge stays a pixel wide, magnified: few partly covered pixels.
    let soft = drawn[0].iter().filter(|&&value| value > 0.1 && value < 0.9).count();
    eprintln!("wrong pixels magnified 8x: {multi} MSDF, {single} single-channel; {soft} pixels on the edge");
    assert!(multi * 5 < single, "MSDF keeps corners the single field rounds: {multi} vs {single}");
    assert!(multi < 40, "{multi}");
    assert!(soft < 1400, "the edge should stay about a pixel wide: {soft}");
    save_grey("msdf-star.png", &drawn[0], side, side, 1);
    save_grey("sdf-star.png", &drawn[1], side, side, 1);
}

/// gdev's UI font in MSDF and in its single-channel field at three sizes, for
/// a look by eye (target/look-shots/msdf-text.png): `--ignored`, with gdev
/// checked out beside pen.
#[test]
#[ignore]
fn msdf_text_shot() {
    let font = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../gdev/models/font/Fredoka.ttf");
    let (Ok(data), Some((device, queue))) = (std::fs::read(&font), device()) else {
        eprintln!("no {} or GPU adapter; skipping", font.display());
        return;
    };
    let text = "Pen MSDF: Sharp at 12x!";
    let options = pen_texture::msdf::Options {
        size: 48.0,
        range: 4.0,
        width: 512,
        variations: vec![(*b"wght", 600.0)],
        chars: text.chars().collect(),
    };
    let atlas = pen_texture::msdf::atlas(&data, &options).unwrap();
    let bytes =
        pen_texture::ktx2::write(pen_texture::ktx2::Format::Rgba8Unorm, atlas.width, atlas.height, &[atlas.pixels])
            .unwrap();
    let root = assets(
        "msdf-text",
        &[
            ("catalog.json", br#"{"meshes":["none.mesh.json"],"materials":["text.material.json"]}"#),
            ("field.wgsl", FIELD_SPRITE.replace("${msdf}", pen_render::MSDF_WGSL).as_bytes()),
            (
                "text.material.json",
                br#"{"shader":"field.wgsl","texture":"text.ktx2","color":[1,1,1,1],"lighting":"unlit"}"#,
            ),
            ("text.ktx2", &bytes),
        ],
    );
    let catalog = boundary::read_catalog(&root).unwrap();
    let mut renderer = renderer(&device, &queue, root, FORMAT, Display::SDR);
    let (width, height) = (1600, 900);
    let mut target = Target::new(&device, width, height, FORMAT);
    let mut sprites = Vec::new();
    let mut baseline = 0.0;
    for (em, single) in [(20.0, 0.0), (64.0, 0.0), (64.0, 1.0), (300.0, 0.0), (300.0, 1.0)] {
        baseline += em * 1.1;
        let mut pen = 16.0;
        for code in text.chars() {
            let glyph = atlas.glyphs.iter().find(|glyph| glyph.code == code).unwrap();
            if glyph.width > 0.0 {
                sprites.push(Sprite {
                    x: pen + glyph.left * em,
                    y: baseline + glyph.top * em,
                    width: glyph.width * em,
                    height: glyph.height * em,
                    u0: glyph.u0,
                    v0: glyph.v0,
                    u1: glyph.u1,
                    v1: glyph.v1,
                    color: [1.0; 4],
                    material: 0,
                    parameters: [single, 0.0, 0.0, 0.0],
                });
            }
            pen += glyph.advance * em;
        }
    }
    let mut frame = frame(Vec::new());
    frame.sprites = sprites;
    let deadline = Instant::now() + Duration::from_secs(20);
    while !renderer.prepare_sprites(&frame.sprites, &catalog).unwrap() || renderer.pending() > 0 {
        renderer.submit(&frame, &catalog, &mut target).unwrap();
        assert!(Instant::now() < deadline, "the atlas did not load: {:?}", renderer.take_events());
        std::thread::sleep(Duration::from_millis(5));
    }
    renderer.submit(&frame, &catalog, &mut target).unwrap();
    save_grey("msdf-text.png", &target.greens(&device, &queue), width, height, 1);
}

// A floor facing up lit by the sky (`ENVIRONMENT_WGSL`) and the frame's lights
// (`LIGHTS_WGSL`), with a draw's parameters as metallic and roughness; its
// light's highlight is left out where `${highlights}` is false.
const GLOSSY: &str = r#"
struct Scene {
  view: mat4x4f, projection: mat4x4f, ambient: vec4f, light_direction: vec4f, light_color: vec4f,
  light_view_projection: mat4x4f, shadow: vec4f, zenith: vec4f, horizon: vec4f, ground: vec4f, eye: vec4f,
};
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: Draw;
${lights}
${environment}
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) view_z: f32,
};
@vertex fn vertex(@location(0) position: vec3f) -> Corner {
  let world = draw.model * vec4f(position, 1.0);
  let seen = scene.view * world;
  return Corner(scene.projection * seen, world.xyz, seen.z);
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  let up = vec3f(0.0, 1.0, 0.0);
  let view = normalize(scene.eye.xyz - corner.world);
  let albedo = draw.color.rgb;
  let metallic = draw.parameters.x;
  let roughness = draw.parameters.y;
  let f0 = mix(vec3f(0.04), albedo, metallic);
  var light = surface_ambient(corner.world, up, view, albedo, metallic, roughness);
  let local = clustered_surface(corner.world, up, view, roughness, f0, corner.position, corner.view_z);
  light += local.diffuse * albedo * (1.0 - metallic);
  if (${highlights}) {
    light += local.specular;
  }
  return vec4f(light, 1.0);
}
"#;

fn project_to_pixel(view: &Matrix, projection: &Matrix, point: [f32; 3], size: (u32, u32)) -> (f32, f32) {
    let apply = |m: &Matrix, v: [f32; 4]| -> [f32; 4] {
        std::array::from_fn(|row| (0..4).map(|k| m[k * 4 + row] * v[k]).sum())
    };
    let clip = apply(projection, apply(view, [point[0], point[1], point[2], 1.0]));
    ((clip[0] / clip[3] * 0.5 + 0.5) * size.0 as f32, (0.5 - clip[1] / clip[3] * 0.5) * size.1 as f32)
}

#[test]
fn the_sky_and_lights_light_a_glossy_floor() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let source =
        GLOSSY.replace("${lights}", pen_render::LIGHTS_WGSL).replace("${environment}", pen_render::ENVIRONMENT_WGSL);
    let floor = br#"{"positions":[-12,0,-12, 12,0,-12, -12,0,12, 12,0,12],"normals":[0,1,0,0,1,0,0,1,0,0,1,0],
                     "uvs":[0,0,1,0,0,1,1,1],"indices":[0,2,1,1,2,3]}"#;
    let root = assets(
        "glossy",
        &[
            (
                "catalog.json",
                br#"{"meshes":["floor.mesh.json"],"materials":["with.material.json","without.material.json"]}"#,
            ),
            ("floor.mesh.json", floor),
            ("with.wgsl", source.replace("${highlights}", "true").as_bytes()),
            ("without.wgsl", source.replace("${highlights}", "false").as_bytes()),
            ("with.material.json", br#"{"shader":"with.wgsl","color":[0.9,0.9,0.9,1],"lighting":"unlit"}"#),
            ("without.material.json", br#"{"shader":"without.wgsl","color":[0.9,0.9,0.9,1],"lighting":"unlit"}"#),
        ],
    );
    let catalog = boundary::read_catalog(&root).unwrap();
    let mut renderer = renderer(&device, &queue, root, FORMAT, Display::SDR);
    let (width, height) = (320, 180);
    let mut target = Target::new(&device, width, height, FORMAT);
    let eye = [0.0, 3.0, 6.0];
    let (view, projection) = camera(eye, width as f32 / height as f32);
    let draw = |material, metallic: f32, roughness: f32| Draw {
        mesh: 0,
        material,
        transform: IDENTITY,
        parameters: Some([metallic, roughness, 0.0, 0.0]),
        extra: None,
    };
    let vector = |x, y, z| Vector { x, y, z };
    let mut frame = frame(vec![]);
    (frame.view, frame.projection) = (view, projection);
    frame.lights = vec![Light {
        position: vector(0.0, 1.5, 0.0),
        radius: 12.0,
        color: vector(1.0, 1.0, 1.0),
        direction: vector(0.0, -1.0, 0.0),
        inner: -1.0,
        outer: -1.0,
    }];
    let mut render = |frame: &RenderFrame| {
        settle(&mut renderer, frame, &catalog, &mut target, 1);
        target.read(&device, &queue)
    };
    // The highlight sits where the eye's mirror image sees the light: on the
    // floor between them, a quarter of the way from the light to the eye's
    // foot (heights 1.5 and 3).
    let spot = project_to_pixel(&view, &projection, [0.0, 0.0, 6.0 * 1.5 / 4.5], (width, height));
    let brightness =
        |image: &[u8]| image.chunks(4).map(|t| f32::from(t[0]) + f32::from(t[1]) + f32::from(t[2])).collect::<Vec<_>>();
    // Without any sky: the lights' highlight against the lights' diffuse.
    frame.draws = vec![draw(0, 0.0, 0.2)];
    let lit = brightness(&render(&frame));
    frame.draws = vec![draw(1, 0.0, 0.2)];
    let plain = brightness(&render(&frame));
    let peak = |values: &[f32]| {
        let (at, value) =
            values.iter().enumerate().fold((0, 0.0), |best, (i, v)| if *v > best.1 { (i, *v) } else { best });
        let at = u32::try_from(at).unwrap();
        ((at % width) as f32, (at / width) as f32, value)
    };
    let (px, py, highlight) = peak(&lit);
    eprintln!(
        "highlight peak {highlight} at ({px}, {py}); expected near ({}, {}); without: {}",
        spot.0,
        spot.1,
        peak(&plain).2
    );
    assert!(lit.iter().zip(&plain).all(|(a, b)| a >= b), "a highlight only adds light");
    assert!(highlight > peak(&plain).2 * 1.3 + 30.0, "a glossy floor shows the light's highlight");
    assert!(
        (px - spot.0).abs() < 14.0 && (py - spot.1).abs() < 14.0,
        "the highlight lies where the eye mirrors the light"
    );
    // A rough floor spreads the same light into a dimmer, wider highlight.
    frame.draws = vec![draw(0, 0.0, 0.9)];
    let rough = brightness(&render(&frame));
    frame.draws = vec![draw(1, 0.0, 0.9)];
    let rough_plain = brightness(&render(&frame));
    let rough_extra: f32 = rough.iter().zip(&rough_plain).map(|(a, b)| a - b).fold(0.0, f32::max);
    let glossy_extra = highlight - peak(&plain).2;
    assert!(rough_extra < glossy_extra, "roughness dims the peak: {rough_extra} vs {glossy_extra}");

    // The sky: a flat diffuse light lights a matte floor by its color.
    let base_sky = frame.sky;
    let flat = |value: f32| {
        let mut sky = base_sky;
        sky.sh0 = vector(value / 0.282_094_8, value / 0.282_094_8, value / 0.282_094_8);
        sky.tint = vector(1.0, 1.0, 1.0);
        sky
    };
    frame.lights = vec![];
    let mut mean_with = |sky: Sky, metallic, roughness| {
        frame.sky = sky;
        frame.draws = vec![draw(1, metallic, roughness)];
        let image = brightness(&render(&frame));
        image.iter().sum::<f32>() / image.len() as f32
    };
    let dim = mean_with(flat(0.1), 0.0, 1.0);
    let bright = mean_with(flat(0.4), 0.0, 1.0);
    assert!(bright > dim * 1.5, "the sky's coefficients light a matte floor: {dim} vs {bright}");
    // And a metal floor mirrors the sky's colors: a blue zenith and horizon tints it blue.
    let mut blue = flat(0.0);
    (blue.zenith, blue.horizon, blue.ground) = (vector(0.1, 0.2, 2.0), vector(0.1, 0.2, 2.0), vector(0.1, 0.2, 2.0));
    let mut red = flat(0.0);
    (red.zenith, red.horizon, red.ground) = (vector(2.0, 0.2, 0.1), vector(2.0, 0.2, 0.1), vector(2.0, 0.2, 0.1));
    let mut mirror = |sky: Sky, channel: usize| {
        frame.sky = sky;
        frame.draws = vec![draw(1, 1.0, 0.05)];
        let image = render(&frame);
        image.chunks(4).map(|t| f32::from(t[channel])).sum::<f32>() / (width * height) as f32
    };
    // BGRA: channel 0 is blue, 2 red.
    assert!(mirror(blue, 0) > mirror(red, 0) * 1.5, "a metal floor reflects a blue sky");
    assert!(mirror(red, 2) > mirror(blue, 2) * 1.5, "a metal floor reflects a red sky");
    // Reflections follow the sky when its colors change from frame to frame.
    let again = mirror(blue, 0);
    assert!(again > 0.0);
}

// Surfaces lit only by the sky's light all round (`ambient_light`), as their
// material's color: a view of global illumination alone.
const GALLERY: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct Draw { model: mat4x4f, normal: mat4x4f, color: vec4f };
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
  let light = ambient_light(corner.world, normalize(corner.normal));
  return vec4f(draw.color.rgb * light * 1.4, 1.0);
}
"#;

/// A quad of corners `a`, `b`, `c`, `d` in order, wound to face `normal`.
fn quad(corners: [[f32; 3]; 4], normal: [f32; 3]) -> String {
    let edge = |p: [f32; 3], q: [f32; 3]| [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
    let (e1, e2) = (edge(corners[0], corners[1]), edge(corners[0], corners[2]));
    let cross = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    let facing = cross[0] * normal[0] + cross[1] * normal[1] + cross[2] * normal[2] > 0.0;
    let indices = if facing { "[0,1,2,1,3,2]" } else { "[0,2,1,1,2,3]" };
    let positions: Vec<f32> = corners.iter().flatten().copied().collect();
    let normals: Vec<f32> = (0..4).flat_map(|_| normal).collect();
    format!(r#"{{"positions":{positions:?},"normals":{normals:?},"uvs":[0,0,1,0,0,1,1,1],"indices":{indices}}}"#)
}

// With PEN_PROBE_ROOM=gallery.json (ink's `INK_PROBE_DUMP`), draws a room open
// to the front, its right wall red, lit by the sky alone: first by the flat
// sky light, then through the baked probe volume, to target/look-shots.
#[test]
#[allow(clippy::cast_possible_truncation, reason = "JSON numbers to f32 and u32")]
fn probe_room_shots() {
    let Some(path) = std::env::var_os("PEN_PROBE_ROOM") else { return };
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let dump: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let floats =
        |key: &str| -> Vec<f32> { dump[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect() };
    let (values, sky_terms) = (floats("values"), floats("sky"));
    let origin = floats("origin");
    let counts: Vec<u32> = dump["counts"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let source = GALLERY.replace("${environment}", pen_render::ENVIRONMENT_WGSL);
    let meshes = [
        ("floor", quad([[-3.0, 0.0, -3.0], [3.0, 0.0, -3.0], [-3.0, 0.0, 4.0], [3.0, 0.0, 4.0]], [0.0, 1.0, 0.0])),
        ("ceiling", quad([[-3.0, 3.5, -3.0], [3.0, 3.5, -3.0], [-3.0, 3.5, 4.0], [3.0, 3.5, 4.0]], [0.0, -1.0, 0.0])),
        ("back", quad([[-3.0, 0.0, -3.0], [3.0, 0.0, -3.0], [-3.0, 3.5, -3.0], [3.0, 3.5, -3.0]], [0.0, 0.0, 1.0])),
        ("left", quad([[-3.0, 0.0, -3.0], [-3.0, 0.0, 4.0], [-3.0, 3.5, -3.0], [-3.0, 3.5, 4.0]], [1.0, 0.0, 0.0])),
        ("right", quad([[3.0, 0.0, -3.0], [3.0, 0.0, 4.0], [3.0, 3.5, -3.0], [3.0, 3.5, 4.0]], [-1.0, 0.0, 0.0])),
    ];
    let mut files: Vec<(String, Vec<u8>)> = vec![
        (
            "catalog.json".into(),
            br#"{"meshes":["floor.mesh.json","ceiling.mesh.json","back.mesh.json","left.mesh.json","right.mesh.json"],"materials":["white.material.json","red.material.json"]}"#.to_vec(),
        ),
        ("gallery.wgsl".into(), source.into_bytes()),
        ("white.material.json".into(), br#"{"shader":"gallery.wgsl","color":[0.75,0.75,0.72,1],"lighting":"unlit"}"#.to_vec()),
        ("red.material.json".into(), br#"{"shader":"gallery.wgsl","color":[0.85,0.08,0.06,1],"lighting":"unlit"}"#.to_vec()),
    ];
    for (name, json) in &meshes {
        files.push((format!("{name}.mesh.json"), json.clone().into_bytes()));
    }
    let borrowed: Vec<(&str, &[u8])> = files.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
    let root = assets("room", &borrowed);
    let catalog = boundary::read_catalog(&root).unwrap();
    let mut renderer = renderer(&device, &queue, root, FORMAT, Display::SDR);
    let (width, height) = (640, 400);
    let mut target = Target::new(&device, width, height, FORMAT);
    let (view, projection) = camera([0.0, 2.4, 8.0], width as f32 / height as f32);
    let draw = |mesh, material| Draw { mesh, material, transform: IDENTITY, parameters: None, extra: None };
    let mut frame = frame(vec![draw(0, 0), draw(1, 0), draw(2, 0), draw(3, 0), draw(4, 1)]);
    (frame.view, frame.projection) = (view, projection);
    frame.light_color = Vector::default();
    let vector = |a: &[f32]| Vector { x: a[0], y: a[1], z: a[2] };
    frame.sky.sh0 = vector(&sky_terms[0..3]);
    frame.sky.sh1 = vector(&sky_terms[3..6]);
    frame.sky.sh2 = vector(&sky_terms[6..9]);
    frame.sky.sh3 = vector(&sky_terms[9..12]);
    frame.sky.sh4 = vector(&sky_terms[12..15]);
    frame.sky.sh5 = vector(&sky_terms[15..18]);
    frame.sky.sh6 = vector(&sky_terms[18..21]);
    frame.sky.sh7 = vector(&sky_terms[21..24]);
    frame.sky.sh8 = vector(&sky_terms[24..27]);
    frame.sky.tint = Vector { x: 1.0, y: 1.0, z: 1.0 };
    let mut shoot = |frame: &RenderFrame, name: &str| {
        settle(&mut renderer, frame, &catalog, &mut target, 5);
        let pixels = target.read(&device, &queue);
        let image = image::RgbaImage::from_fn(width, height, |x, y| {
            let at = ((y * width + x) * 4) as usize;
            image::Rgba([pixels[at + 2], pixels[at + 1], pixels[at], 255])
        });
        image.save(shots().join(name)).unwrap();
        pixels
    };
    let flat = shoot(&frame, "probe-room-sky-only.png");
    let probes: Vec<Probe> = values
        .chunks(28)
        .map(|p| Probe {
            sh0: vector(&p[0..3]),
            sh1: vector(&p[3..6]),
            sh2: vector(&p[6..9]),
            sh3: vector(&p[9..12]),
            sh4: vector(&p[12..15]),
            sh5: vector(&p[15..18]),
            sh6: vector(&p[18..21]),
            sh7: vector(&p[21..24]),
            sh8: vector(&p[24..27]),
            valid: p[27],
        })
        .collect();
    frame.probe_volume = ProbeVolume {
        revision: 1,
        nx: counts[0],
        ny: counts[1],
        nz: counts[2],
        spacing: dump["spacing"].as_f64().unwrap() as f32,
        origin: vector(&origin),
    };
    frame.probes = probes;
    let lit = shoot(&frame, "probe-room-probes.png");
    // The next frame sends no probes and keeps them: the picture is the same.
    frame.probes = Vec::new();
    let kept = shoot(&frame, "probe-room-probes-kept.png");
    assert_eq!(lit, kept, "a frame that sends no probes uses the ones the host holds");
    // The floor by the red wall (bottom right of the room) turns red; without
    // probes it is as grey as the rest.
    let at = |pixels: &[u8], x: u32, y: u32| {
        let i = ((y * width + x) * 4) as usize;
        [f32::from(pixels[i + 2]), f32::from(pixels[i + 1]), f32::from(pixels[i])]
    };
    let mut changed = 0;
    for y in 0..height {
        for x in 0..width {
            let (a, b) = (at(&flat, x, y), at(&lit, x, y));
            if (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs() > 12.0 {
                changed += 1;
            }
        }
    }
    eprintln!("{changed} pixels differ with probes");
    assert!(changed > 5000, "the probes should change the lighting of the room");
}
