//! Required GPU witnesses: an independent analytic scene rasterizer, box
//! supersampling, temporal residuals and timestamps of the production pass.
use super::*;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

const WIDTH: u32 = 128;
const HEIGHT: u32 = 96;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const MODES: [&str; 3] = ["off", "legacy", "quality"];

fn scene(kind: &str, x: f64, y: f64, motion: f64) -> f64 {
    let x = x + motion;
    let y = y + motion * 0.37;
    match kind {
        "diagonal" => {
            let edge = x * 0.43 + 15.13;
            if y > edge {
                return 0.85;
            }
            0.10
        }
        "fence" => {
            let at = (x + y * 0.37).rem_euclid(11.0);
            if at < 1.2 {
                return 0.85;
            }
            0.10
        }
        "texture" => 0.45 + 0.35 * (x * 1.11).sin() * (y * 0.93).cos(),
        "detail" => 0.45 + 0.02 * (x * 1.11).sin() * (y * 0.93).cos(),
        _ => panic!("unknown analytic witness"),
    }
}

fn raster(kind: &str, width: u32, height: u32, motion: f64, samples: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for sy in 0..samples {
                for sx in 0..samples {
                    sum += scene(
                        kind,
                        f64::from(x) + (f64::from(sx) + 0.5) / f64::from(samples),
                        f64::from(y) + (f64::from(sy) + 0.5) / f64::from(samples),
                        motion,
                    );
                }
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a coverage in 0..=255")]
            let value = (sum / f64::from(samples * samples) * 255.0).round() as u8;
            pixels.extend([value, value, value, 255]);
        }
    }
    pixels
}

fn uniform(width: u32, height: u32, output: (u32, u32), enabled: bool) -> [f32; 16] {
    let mut values = [0.0; 16];
    values[0] = 1.0 / output.0 as f32;
    values[1] = 1.0 / output.1 as f32;
    values[4] = 1.0 / width as f32;
    values[5] = 1.0 / height as f32;
    if enabled {
        values[6] = 1.0;
    }
    values[10] = width as f32;
    values[11] = height as f32;
    values
}

fn code(mode: &str) -> String {
    let quality = include_str!("../shaders/fxaa.wgsl");
    let mut code = present_shader(1, &crate::Display::SDR);
    if mode == "legacy" {
        code = code.replace(quality, include_str!("legacy_fxaa.wgsl"));
    } else {
        // Shader iteration needs no relink of the release renderer tests.
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/fxaa.wgsl");
        code = code.replace(quality, &std::fs::read_to_string(path).unwrap());
        if let Ok(subpixel) = std::env::var("PEN_FXAA_TEST_SUBPIXEL") {
            let value: f32 = subpixel.parse().unwrap();
            assert!((0.0..=1.0).contains(&value));
            code = code.replace(
                "subpixel_curve * subpixel_curve * 0.25",
                &format!("subpixel_curve * subpixel_curve * {value:.2}"),
            );
        }
    }
    code
}

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::RenderPipeline>,
    timestamps: wgpu::QuerySet,
}

impl Harness {
    fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("FXAA image witnesses require a real or software WebGPU adapter");
        assert!(
            adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            "FXAA GPU measurements require TIMESTAMP_QUERY support"
        );
        let info = adapter.get_info();
        eprintln!("FXAA adapter: {} {:?} {:?}", info.name, info.backend, info.device_type);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            ..Default::default()
        }))
        .expect("FXAA GPU device creation failed");
        let pipelines = MODES
            .iter()
            .map(|mode| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(mode),
                    source: wgpu::ShaderSource::Wgsl(code(mode).into()),
                });
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(mode),
                    layout: None,
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vertex"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fragment"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            })
            .collect();
        let timestamps = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("FXAA measurements"),
            ty: wgpu::QueryType::Timestamp,
            count: 128,
        });
        Self { device, queue, pipelines, timestamps }
    }

    fn read(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let (tx, rx) = mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = buffer.slice(..).get_mapped_range().unwrap().to_vec();
        buffer.unmap();
        bytes
    }

    fn draw(
        &self,
        input: &[u8],
        size: (u32, u32),
        output: (u32, u32),
        mode: usize,
        measure: bool,
    ) -> (Vec<u8>, Vec<f64>) {
        self.draw_format(input, size, output, mode, measure, FORMAT)
    }

    fn draw_format(
        &self,
        input: &[u8],
        size: (u32, u32),
        output: (u32, u32),
        mode: usize,
        measure: bool,
        source_format: wgpu::TextureFormat,
    ) -> (Vec<u8>, Vec<f64>) {
        let device = &self.device;
        let texture = |size: (u32, u32), format, usage| super::texture(device, size.0, size.1, usage, 1, format, 1);
        let source = texture(size, source_format, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        self.queue.write_texture(
            source.as_image_copy(),
            input,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(u32::try_from(input.len()).unwrap() / size.1),
                rows_per_image: None,
            },
            source.size(),
        );
        let marks = texture(size, wgpu::TextureFormat::R8Uint, wgpu::TextureUsages::TEXTURE_BINDING);
        let glow = texture((1, 1), FORMAT, wgpu::TextureUsages::TEXTURE_BINDING);
        let target = texture(output, FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let values = uniform(size.0, size.1, output, mode != 0);
        let settings = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&settings, 0, bytemuck::cast_slice(&values));
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines[mode].get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: settings.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&source.create_view(&Default::default())),
                },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&marks.create_view(&Default::default())),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&glow.create_view(&Default::default())),
                },
            ],
        });
        let view = target.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut passes = 1;
        if measure {
            passes = 80;
        }
        for index in 0..passes {
            let mut timestamp_writes = None;
            if measure && index >= 16 {
                let query = (index - 16) * 2;
                timestamp_writes = Some(wgpu::RenderPassTimestampWrites {
                    query_set: &self.timestamps,
                    beginning_of_pass_write_index: Some(query),
                    end_of_pass_write_index: Some(query + 1),
                });
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(MODES[mode]),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                timestamp_writes,
                ..Default::default()
            });
            pass.set_pipeline(&self.pipelines[mode]);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let query_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if measure {
            encoder.resolve_query_set(&self.timestamps, 0..128, &resolve, 0);
            encoder.copy_buffer_to_buffer(&resolve, 0, &query_read, 0, 1024);
        }
        let image = self.read_texture(&target, encoder);
        let mut times = Vec::new();
        if measure {
            let bytes = self.read(&query_read);
            for pair in bytes.as_chunks::<16>().0 {
                let start = u64::from_ne_bytes(pair[..8].try_into().unwrap());
                let end = u64::from_ne_bytes(pair[8..].try_into().unwrap());
                times.push((end - start) as f64 * f64::from(self.queue.get_timestamp_period()) / 1e6);
            }
        }
        (image, times)
    }

    fn read_texture(&self, target: &wgpu::Texture, mut encoder: wgpu::CommandEncoder) -> Vec<u8> {
        let stride = (target.width() * 4).div_ceil(256) * 256;
        let pixels = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(stride * target.height()),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &pixels,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(stride), rows_per_image: None },
            },
            target.size(),
        );
        self.queue.submit([encoder.finish()]);
        let padded = self.read(&pixels);
        padded.chunks(stride as usize).flat_map(|row| &row[..(target.width() * 4) as usize]).copied().collect()
    }
}

struct Target(wgpu::Texture);

impl crate::RenderTarget for Target {
    fn size(&self) -> (u32, u32) {
        (self.0.width(), self.0.height())
    }
    fn view(&mut self) -> Option<wgpu::TextureView> {
        Some(self.0.create_view(&Default::default()))
    }
    fn present(&mut self) {}
}

fn hud_witness(gpu: &Harness, root: &Path) {
    use pen_protocol::{Catalog, Color, Draw, Sprite};
    let assets = root.join("hud-assets");
    std::fs::create_dir_all(&assets).unwrap();
    for (key, bytes) in [
        (
            "triangle.json",
            r#"{"positions":[-0.8,-0.8,0.5,0.8,-0.8,0.5,0.3,0.8,0.5],"normals":[0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,0,1],"indices":[0,1,2]}"#,
        ),
        ("background.json", r#"{"shader":"background.wgsl","color":[1,1,1,1],"lighting":"unlit"}"#),
        ("hud.json", r#"{"shader":"hud.wgsl","color":[1,1,1,1],"lighting":"unlit"}"#),
        (
            "background.wgsl",
            r#"
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f { return vec4f(position, 1.0); }
@fragment fn fragment() -> @location(0) vec4f { return vec4f(0.85, 0.85, 0.85, 1.0); }
"#,
        ),
        (
            "hud.wgsl",
            r#"
struct Sprite { rect: vec4f, uv: vec4f, color: vec4f, parameters: vec4f };
@group(0) @binding(0) var<uniform> screen: vec4f;
@group(0) @binding(1) var<storage, read> sprites: array<Sprite>;
@vertex fn vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> @builtin(position) vec4f {
    let corners = array<vec2f, 6>(vec2f(0,0), vec2f(1,0), vec2f(0,1), vec2f(0,1), vec2f(1,0), vec2f(1,1));
    let rect = sprites[instance].rect;
    let pixel = rect.xy + corners[vertex] * rect.zw;
    return vec4f(pixel.x / screen.x * 2.0 - 1.0, 1.0 - pixel.y / screen.y * 2.0, 0.0, 1.0);
}
@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
    if ((u32(position.x) + u32(position.y)) % 2u == 0u) { return vec4f(1.0); }
    return vec4f(0.0, 0.0, 0.0, 1.0);
}
"#,
        ),
    ] {
        std::fs::write(assets.join(key), bytes).unwrap();
    }
    let catalog = Catalog {
        meshes: vec!["triangle.json".into()],
        materials: vec!["background.json".into(), "hud.json".into()],
        ..Default::default()
    };
    let mut renderer = crate::Renderer::new(
        gpu.device.clone(),
        gpu.queue.clone(),
        crate::RendererOptions {
            format: FORMAT,
            display: crate::Display::SDR,
            assets,
            veil: None,
            pipeline_cache: None,
        },
    )
    .unwrap();
    renderer.preload(&catalog);
    let mut frame = RenderFrame {
        view: crate::matrix::IDENTITY,
        projection: crate::matrix::IDENTITY,
        quality: Quality { msaa: 1, scale: 1.0, filtering: 1, ..Default::default() },
        light_direction: pen_protocol::Vector { x: 0.0, y: 1.0, z: 0.0 },
        clear: Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
        draws: vec![Draw { mesh: 0, material: 0, transform: crate::matrix::IDENTITY, ..Default::default() }],
        sprites: vec![Sprite {
            x: 12.0,
            y: 16.0,
            width: 48.0,
            height: 20.0,
            color: [1.0; 4],
            material: 1,
            ..Default::default()
        }],
        ..Default::default()
    };
    let overlay = [Sprite { x: 70.0, y: 50.0, width: 16.0, height: 16.0, ..frame.sprites[0] }];
    for (size, scale) in [((WIDTH, HEIGHT), 1.0), ((91, 71), 0.5), ((192, 144), 1.0)] {
        let mut target = Target(super::texture(
            &gpu.device,
            size.0,
            size.1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            1,
            FORMAT,
            1,
        ));
        frame.quality.scale = scale;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        loop {
            renderer.submit(&frame, &catalog, &mut target).unwrap();
            if renderer.pending() == 0
                && renderer.stats().draw_calls > 0
                && renderer.prepare_sprites(&overlay, &catalog).unwrap()
            {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "HUD image assets did not load");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let mut images = Vec::new();
        for enabled in [false, true] {
            frame.quality.fxaa = enabled;
            renderer.submit(&frame, &catalog, &mut target).unwrap();
            renderer.overlay(&overlay, &catalog, &target.0.create_view(&Default::default()), size).unwrap();
            images.push(gpu.read_texture(&target.0, gpu.device.create_command_encoder(&Default::default())));
        }
        assert_ne!(images[0], images[1], "scene AA must actually run in HUD witness");
        for rect in [(12, 16, 48, 20), (70, 50, 16, 16)] {
            for y in rect.1..rect.1 + rect.3 {
                for x in rect.0..rect.0 + rect.2 {
                    let at = ((y * size.0 + x) * 4) as usize;
                    assert_eq!(&images[0][at..at + 4], &images[1][at..at + 4], "HUD/UI changed at {x},{y}");
                }
            }
        }
        save(root, &format!("hud-{}x{}-off.png", size.0, size.1), &images[0], size);
        save(root, &format!("hud-{}x{}-quality.png", size.0, size.1), &images[1], size);
    }
}

fn save(root: &Path, name: &str, pixels: &[u8], size: (u32, u32)) {
    std::fs::write(root.join(name).with_extension("rgba"), pixels).unwrap();
    image::save_buffer(root.join(name), pixels, size.0, size.1, image::ColorType::Rgba8).unwrap();
}

fn error(image: &[u8], reference: &[u8]) -> f64 {
    image
        .as_chunks::<4>()
        .0
        .iter()
        .zip(reference.as_chunks::<4>().0.iter())
        .map(|(a, b)| (f64::from(a[0]) - f64::from(b[0])).abs() / 255.0)
        .sum::<f64>()
        / (image.len() / 4) as f64
}

#[test]
fn fxaa_quality_images_motion_resize_and_gpu_cost() {
    let root = std::env::var_os("PEN_FXAA_WITNESSES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/fxaa"));
    std::fs::create_dir_all(&root).unwrap();
    for mode in ["quality", "legacy"] {
        std::fs::write(root.join(format!("present-{mode}.wgsl")), code(mode)).unwrap();
    }
    std::fs::write(root.join("manifest.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "width": WIDTH, "height": HEIGHT, "format": "rgba8unorm", "uniform": uniform(WIDTH, HEIGHT, (WIDTH, HEIGHT), true),
    })).unwrap()).unwrap();
    let gpu = Harness::new();
    hud_witness(&gpu, &root);
    let mut results = serde_json::Map::new();
    for kind in ["diagonal", "fence", "texture", "detail"] {
        let input = raster(kind, WIDTH, HEIGHT, 0.0, 1);
        let reference = raster(kind, WIDTH, HEIGHT, 0.0, 16);
        std::fs::write(root.join(format!("{kind}-input.rgba")), &input).unwrap();
        save(&root, &format!("{kind}-input.png"), &input, (WIDTH, HEIGHT));
        save(&root, &format!("{kind}-reference.png"), &reference, (WIDTH, HEIGHT));
        let mut errors = Vec::new();
        let mut changes = Vec::new();
        for (mode, name) in MODES.iter().enumerate() {
            let (image, _) = gpu.draw(&input, (WIDTH, HEIGHT), (WIDTH, HEIGHT), mode, false);
            errors.push(error(&image, &reference));
            changes.push(error(&image, &input));
            save(&root, &format!("{kind}-{name}.png"), &image, (WIDTH, HEIGHT));
        }
        eprintln!("FXAA {kind} MAE off/legacy/quality: {errors:?}");
        if kind == "diagonal" || kind == "fence" {
            assert!(errors[2] < errors[0], "quality must reduce supersampled {kind} error");
            assert!(errors[2] < errors[1], "quality must improve on legacy {kind} error");
        }
        if kind == "detail" {
            assert_eq!(changes[2], 0.0, "quality must preserve low-contrast texture exactly");
            results.insert("detail_change_mae".into(), serde_json::json!(changes));
        }
        results.insert(format!("{kind}_mae"), serde_json::json!(errors));
    }

    // HDR contrast compressed by tone mapping must not trigger filtering.
    let mut hdr = Vec::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let mut channel = 0x4000u16; // 2.0 in IEEE float16
            if f64::from(y) > f64::from(x) * 0.43 + 15.13 {
                channel = 0x4400;
            } // 4.0
            for value in [channel, channel, channel, 0x3c00] {
                hdr.extend(value.to_le_bytes());
            }
        }
    }
    let hdr_off = gpu.draw_format(&hdr, (WIDTH, HEIGHT), (WIDTH, HEIGHT), 0, false, SCENE_FORMAT).0;
    let hdr_aa = gpu.draw_format(&hdr, (WIDTH, HEIGHT), (WIDTH, HEIGHT), 2, false, SCENE_FORMAT).0;
    assert_eq!(hdr_off, hdr_aa, "FXAA contrast must use the toned display values");
    assert!(hdr_aa.as_chunks::<4>().0.iter().all(|pixel| (253..=254).contains(&pixel[0])));

    // Error in temporal change, relative to independently supersampled motion.
    let mut shimmer = [0.0; 3];
    let mut previous: Option<(Vec<Vec<u8>>, Vec<u8>)> = None;
    for frame in 0..12 {
        let motion = f64::from(frame) * 0.17;
        let input = raster("fence", WIDTH, HEIGHT, motion, 1);
        let reference = raster("fence", WIDTH, HEIGHT, motion, 16);
        let images: Vec<_> =
            (0..3).map(|mode| gpu.draw(&input, (WIDTH, HEIGHT), (WIDTH, HEIGHT), mode, false).0).collect();
        for mode in 0..3 {
            save(&root, &format!("motion-{frame:02}-{}.png", MODES[mode]), &images[mode], (WIDTH, HEIGHT));
        }
        save(&root, &format!("motion-{frame:02}-reference.png"), &reference, (WIDTH, HEIGHT));
        if let Some((last_images, last_reference)) = previous {
            for mode in 0..3 {
                for pixel in (0..input.len()).step_by(4) {
                    let change = f64::from(images[mode][pixel]) - f64::from(last_images[mode][pixel]);
                    let expected = f64::from(reference[pixel]) - f64::from(last_reference[pixel]);
                    shimmer[mode] += (change - expected).abs() / 255.0 / f64::from(WIDTH * HEIGHT) / 11.0;
                }
            }
        }
        previous = Some((images, reference));
    }
    eprintln!("FXAA temporal residual off/legacy/quality: {shimmer:?}");
    assert!(shimmer[2] < shimmer[0], "quality must reduce spatial aliasing shimmer in this motion witness");
    results.insert("temporal_mae".into(), serde_json::json!(shimmer));

    // Source texels stay separate from output pixels through resizes/scales.
    for output in [(91, 71), (192, 144), (WIDTH, HEIGHT)] {
        let input = raster("diagonal", WIDTH, HEIGHT, 0.0, 1);
        let (image, _) = gpu.draw(&input, (WIDTH, HEIGHT), output, 2, false);
        assert_eq!(image.len(), (output.0 * output.1 * 4) as usize);
        assert!(image.as_chunks::<4>().0.iter().all(|pixel| pixel[3] == 255));
        assert!(image.as_chunks::<4>().0.iter().all(|pixel| (25..=217).contains(&pixel[0])));
        save(&root, &format!("resize-{}x{}.png", output.0, output.1), &image, output);
    }

    for size in [(1920, 1080), (2560, 1440)] {
        let input = raster("fence", size.0, size.1, 0.0, 1);
        let half: Vec<u8> =
            input.iter().flat_map(|value| crate::renderer::half_of(f32::from(*value) / 255.0).to_le_bytes()).collect();
        for (mode, name) in MODES.iter().enumerate() {
            let (_, mut times) = gpu.draw_format(&half, size, size, mode, true, SCENE_FORMAT);
            times.sort_by(f64::total_cmp);
            let spread = [times[3], times[32], times[60]];
            eprintln!("FXAA GPU {}x{} {name} p05/median/p95 ms: {spread:?}", size.0, size.1);
            results.insert(format!("{}x{}_{name}_gpu_ms", size.0, size.1), serde_json::json!(spread));
        }
    }
    std::fs::write(root.join("results.json"), serde_json::to_vec_pretty(&results).unwrap()).unwrap();
    eprintln!("FXAA witnesses: {}", root.display());
}
