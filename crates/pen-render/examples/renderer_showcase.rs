//! Headless Vulkan visual showcase rendering four live `Renderer::submit` panels
//! (JFA Outlines + Meshlet DAGs, 3D/X-Ray/2D Lines, GPU Depth-Bounce Mesh Particles,
//! and the 2D Suite with Tilemap + 9-Slice Sprites + 2D Radiance Cascades GI + SDF UI)
//! into a 1280x720 PNG composite and optional raw RGB frame sequence for MP4 encoding.
//!
//! Usage:
//!   cargo run --release -p pen-render --example renderer_showcase -- <out_png> [frames_dir] [frame_count]

use anyhow::Result;
use image::{Rgba, RgbaImage};
use pen_protocol::{Catalog, Color, Draw, Light, RenderFrame, Vector};
use pen_render::{
    Camera2d, Display, GpuParticle, GpuParticleScene, JfaOutlineScene, JfaOutlineTarget, LineBatch, LineLayer,
    LineScene, LineStyle, OutlineStyle, RadianceCascades2dConfig, RadianceCascades2dLayer, RenderTarget, Renderer,
    RendererOptions, Scene2d, Scene2dEmitter, SdfShape, Sprite2d, SpriteLayer2d, TileCell, TilemapLayer2d,
    TilemapUniform, TonemapOperator, XRayStyle, perspective_reversed_z,
};
use serde_json::json;
use std::path::{Path, PathBuf};

const PANEL_W: u32 = 640;
const PANEL_H: u32 = 360;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

const LIT_SHADER: &str = r#"
struct Scene { view: mat4x4f, projection: mat4x4f };
struct DrawU { model: mat4x4f, normal: mat4x4f, color: vec4f, settings: vec4f };
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> draw: DrawU;
@group(1) @binding(1) var base: texture_2d<f32>;
@group(1) @binding(2) var base_samp: sampler;

struct Out {
  @builtin(position) pos: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) world_n: vec3f,
  @location(2) uv: vec2f,
};

@vertex fn vertex(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f) -> Out {
  var o: Out;
  let wp = draw.model * vec4f(p, 1.0);
  o.pos = scene.projection * scene.view * wp;
  o.world_pos = wp.xyz;
  o.world_n = normalize((draw.normal * vec4f(n, 0.0)).xyz);
  o.uv = uv;
  return o;
}

@fragment fn fragment(i: Out) -> @location(0) vec4f {
  let albedo = textureSample(base, base_samp, i.uv).rgb * draw.color.rgb;
  let n = normalize(i.world_n);
  let l1 = normalize(vec3f(0.55, 0.85, 0.45));
  let l2 = normalize(vec3f(-0.7, 0.35, -0.5));
  let ndl1 = max(dot(n, l1), 0.0);
  let ndl2 = max(dot(n, l2), 0.0);
  let v = normalize(vec3f(0.0, 2.0, 6.0) - i.world_pos);
  let h = normalize(l1 + v);
  let spec = pow(max(dot(n, h), 0.0), 36.0) * 0.45;
  let hemi = mix(vec3f(0.14, 0.17, 0.25), vec3f(0.36, 0.44, 0.60), n.y * 0.5 + 0.5);
  let lit = albedo * (vec3f(1.85, 1.72, 1.55) * ndl1 + vec3f(0.45, 0.65, 1.0) * ndl2 + hemi) + vec3f(spec);
  return vec4f(lit, draw.color.a);
}
"#;

struct Offscreen {
    texture: wgpu::Texture,
    width: u32,
    height: u32,
}

impl Offscreen {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("showcase_panel"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        Self { texture, width, height }
    }

    fn read_rgba(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u8> {
        let row = self.width * 4;
        let padded = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let size = u64::from(padded) * u64::from(self.height);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("showcase_readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range().unwrap();
        let bytes: &[u8] = &mapped;
        let mut out = Vec::with_capacity((self.width * self.height * 4) as usize);
        for y in 0..self.height as usize {
            let start = y * padded as usize;
            out.extend_from_slice(&bytes[start..start + row as usize]);
        }
        drop(mapped);
        buf.unmap();
        out
    }
}

impl RenderTarget for Offscreen {
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn view(&mut self) -> Option<wgpu::TextureView> {
        Some(self.texture.create_view(&Default::default()))
    }

    fn present(&mut self) {}
}

fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross =
        |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm = |v: [f32; 3]| {
        let l = dot(v, v).sqrt().max(1e-6);
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let f = norm(sub(target, eye));
    let s = norm(cross(f, up));
    let u = cross(s, f);
    [
        s[0],
        u[0],
        -f[0],
        0.0,
        s[1],
        u[1],
        -f[1],
        0.0,
        s[2],
        u[2],
        -f[2],
        0.0,
        -dot(s, eye),
        -dot(u, eye),
        dot(f, eye),
        1.0,
    ]
}

fn trs(pos: [f32; 3], yaw: f32, scale: [f32; 3]) -> [f32; 16] {
    let (s, c) = yaw.sin_cos();
    [
        c * scale[0],
        0.0,
        -s * scale[0],
        0.0,
        0.0,
        scale[1],
        0.0,
        0.0,
        s * scale[2],
        0.0,
        c * scale[2],
        0.0,
        pos[0],
        pos[1],
        pos[2],
        1.0,
    ]
}

fn build_assets(dir: &Path) -> Catalog {
    std::fs::create_dir_all(dir).unwrap();
    // 1024-triangle UV sphere (hits MeshletDagGpu >= 768 tris!)
    let (stacks, slices) = (16u32, 32u32);
    let (mut pos, mut norm, mut uv, mut idx) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for r in 0..=stacks {
        let v = r as f32 / stacks as f32;
        let phi = v * std::f32::consts::PI;
        for c in 0..=slices {
            let u = c as f32 / slices as f32;
            let theta = u * std::f32::consts::TAU;
            let (x, y, z) = (phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
            pos.extend_from_slice(&[x, y, z]);
            norm.extend_from_slice(&[x, y, z]);
            uv.extend_from_slice(&[u, v]);
        }
    }
    let stride = slices + 1;
    for r in 0..stacks {
        for c in 0..slices {
            let (i0, i1, i2, i3) = (r * stride + c, r * stride + c + 1, (r + 1) * stride + c, (r + 1) * stride + c + 1);
            idx.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
        }
    }
    std::fs::write(
        dir.join("sphere.mesh.json"),
        serde_json::to_vec(&json!({ "positions": pos, "normals": norm, "uvs": uv, "indices": idx })).unwrap(),
    )
    .unwrap();

    // Unit box with flat normals
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 0.0, 1.0], [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]),
        ([0.0, 0.0, -1.0], [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]]),
        ([1.0, 0.0, 0.0], [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]]),
        ([-1.0, 0.0, 0.0], [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]]),
        ([0.0, 1.0, 0.0], [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]]),
        ([0.0, -1.0, 0.0], [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]]),
    ];
    let (mut bp, mut bn, mut bu, mut bi) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (n, q) in faces {
        let base = u32::try_from(bp.len() / 3).unwrap();
        for (k, p) in q.iter().enumerate() {
            bp.extend_from_slice(p);
            bn.extend_from_slice(&n);
            bu.extend_from_slice(match k {
                0 => &[0.0f32, 1.0],
                1 => &[1.0, 1.0],
                2 => &[1.0, 0.0],
                _ => &[0.0, 0.0],
            });
        }
        bi.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    std::fs::write(
        dir.join("box.mesh.json"),
        serde_json::to_vec(&json!({ "positions": bp, "normals": bn, "uvs": bu, "indices": bi })).unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("lit.wgsl"), LIT_SHADER).unwrap();

    // Subtle checker albedo PNG
    let mut checker = RgbaImage::new(32, 32);
    for y in 0..32 {
        for x in 0..32 {
            let light = ((x / 8) + (y / 8)) % 2 == 0;
            let c = if light { 235 } else { 195 };
            checker.put_pixel(x, y, Rgba([c, c, c, 255]));
        }
    }
    checker.save(dir.join("checker.png")).unwrap();
    let white = RgbaImage::from_pixel(4, 4, Rgba([255, 255, 255, 255]));
    white.save(dir.join("white.png")).unwrap();

    let mats = [
        (
            "floor.mat.json",
            json!({ "shader": "lit.wgsl", "texture": "checker.png", "color": [0.28, 0.32, 0.40, 1.0], "lighting": "unlit" }),
        ),
        (
            "cobalt.mat.json",
            json!({ "shader": "lit.wgsl", "texture": "white.png", "color": [0.22, 0.48, 0.92, 1.0], "lighting": "unlit" }),
        ),
        (
            "coral.mat.json",
            json!({ "shader": "lit.wgsl", "texture": "white.png", "color": [0.90, 0.30, 0.26, 1.0], "lighting": "unlit" }),
        ),
        (
            "slate.mat.json",
            json!({ "shader": "lit.wgsl", "texture": "white.png", "color": [0.36, 0.40, 0.48, 1.0], "lighting": "unlit" }),
        ),
        (
            "emerald.mat.json",
            json!({ "shader": "lit.wgsl", "texture": "white.png", "color": [0.18, 0.72, 0.50, 1.0], "lighting": "unlit" }),
        ),
    ];
    for (name, v) in &mats {
        std::fs::write(dir.join(name), v.to_string()).unwrap();
    }
    Catalog {
        meshes: vec!["sphere.mesh.json".into(), "box.mesh.json".into()],
        materials: mats.iter().map(|(n, _)| (*n).into()).collect(),
        sounds: vec![],
    }
}

fn wait_ready(renderer: &mut Renderer, frame: &RenderFrame, catalog: &Catalog, target: &mut Offscreen) {
    for _ in 0..120 {
        renderer.submit(frame, catalog, target).unwrap();
        let _ = renderer.take_events();
        if !renderer.waiting() && renderer.pending() == 0 {
            break;
        }
        renderer.wait(std::time::Duration::from_millis(10));
    }
}

fn base_3d_frame(eye: [f32; 3], target_pt: [f32; 3], draws: Vec<Draw>) -> RenderFrame {
    let mut f = RenderFrame {
        view: look_at(eye, target_pt, [0.0, 1.0, 0.0]),
        projection: perspective_reversed_z(50.0f32.to_radians(), PANEL_W as f32 / PANEL_H as f32, 0.1, None),
        clear: Color { r: 0.03, g: 0.04, b: 0.07, a: 1.0 },
        ambient: Vector { x: 0.9, y: 0.95, z: 1.05 },
        light_direction: Vector { x: -0.5, y: -0.8, z: -0.35 },
        light_color: Vector { x: 2.2, y: 2.1, z: 1.95 },
        draws,
        ..RenderFrame::default()
    };
    f.quality.sky = false;
    f.quality.msaa = 4;
    f.quality.scale = 1.0;
    f.quality.bloom = true;
    f.exposure.auto = false;
    f
}

fn build_tile_atlas(renderer: &Renderer) -> wgpu::TextureView {
    // 4x1 atlas (64x16): 1=dark stone tile, 2=accent circuit tile, 3=moss tile, 4=alt glow tile
    let (w, h) = (64u32, 16u32);
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for ty in 0..16u32 {
        for tx in 0..64u32 {
            let tile = tx / 16;
            let lx = tx % 16;
            let ly = ty;
            let border = lx == 0 || ly == 0 || lx == 15 || ly == 15;
            let px: [u8; 4] = match tile {
                0 => {
                    if border {
                        [56, 62, 76, 255]
                    } else if (lx + ly) % 5 == 0 {
                        [92, 98, 114, 255]
                    } else {
                        [76, 82, 96, 255]
                    }
                }
                1 => {
                    if border {
                        [48, 72, 94, 255]
                    } else if lx == 7 || lx == 8 || ly == 7 || ly == 8 {
                        [115, 190, 220, 255]
                    } else {
                        [68, 92, 116, 255]
                    }
                }
                2 => {
                    if border {
                        [54, 84, 62, 255]
                    } else if (lx * 3 + ly * 7) % 11 == 0 {
                        [130, 200, 118, 255]
                    } else {
                        [82, 126, 86, 255]
                    }
                }
                _ => {
                    if border {
                        [112, 86, 48, 255]
                    } else {
                        [215, 168, 86, 255]
                    }
                }
            };
            let o = ((ty * w + tx) * 4) as usize;
            rgba[o..o + 4].copy_from_slice(&px);
        }
    }
    renderer.upload_2d_texture_rgba8(w, h, &rgba)
}

fn build_nineslice_atlas(renderer: &Renderer) -> wgpu::TextureView {
    // 16x16 9-slice badge with crisp 3px gold corners and teal interior
    let mut rgba = vec![0u8; 16 * 16 * 4];
    for y in 0..16u32 {
        for x in 0..16u32 {
            let corner_x = !(3..13).contains(&x);
            let corner_y = !(3..13).contains(&y);
            let px = if corner_x && corner_y {
                [255, 205, 70, 255]
            } else if corner_x || corner_y {
                [210, 150, 50, 255]
            } else {
                [46, 155, 175, 255]
            };
            let o = ((y * 16 + x) * 4) as usize;
            rgba[o..o + 4].copy_from_slice(&px);
        }
    }
    renderer.upload_2d_texture_rgba8(16, 16, &rgba)
}

fn blit_quadrant(dst: &mut RgbaImage, src: &[u8], ox: u32, oy: u32) {
    for y in 0..PANEL_H {
        for x in 0..PANEL_W {
            let o = ((y * PANEL_W + x) * 4) as usize;
            dst.put_pixel(ox + x, oy + y, Rgba([src[o], src[o + 1], src[o + 2], 255]));
        }
    }
}

fn draw_divider(img: &mut RgbaImage) {
    let w = img.width();
    let h = img.height();
    for x in 0..w {
        for dy in 0..2 {
            img.put_pixel(x, PANEL_H - 1 + dy, Rgba([220, 230, 245, 255]));
        }
    }
    for y in 0..h {
        for dx in 0..2 {
            img.put_pixel(PANEL_W - 1 + dx, y, Rgba([220, 230, 245, 255]));
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let out_png = PathBuf::from(args.get(1).map_or("/tmp/renderer_expansion_showcase.png", String::as_str));
    let frames_dir = args.get(2).map(PathBuf::from);
    let frame_count: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("headless GPU adapter required");
    let mut features = wgpu::Features::empty();
    if adapter.features().contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
        features |= wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("showcase"),
        required_features: features,
        required_limits: adapter.limits(),
        ..Default::default()
    }))?;

    let dir = std::env::temp_dir().join(format!("pen-renderer-showcase-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let catalog = build_assets(&dir);

    let make_renderer = || {
        let mut r = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererOptions {
                format: FORMAT,
                display: Display::SDR,
                assets: dir.clone(),
                veil: None,
                pipeline_cache: None,
            },
        )
        .unwrap();
        r.set_tonemap(TonemapOperator::AgX);
        r
    };
    let mut r_jfa = make_renderer();
    let mut r_lines = make_renderer();
    let mut r_particles = make_renderer();
    let mut r_2d = make_renderer();
    let mut panel = Offscreen::new(&device, PANEL_W, PANEL_H);

    // Pre-build 2D textures & tilemap grid (20 x 12 tiles of 32px = 640x384).
    let tile_atlas = build_tile_atlas(&r_2d);
    let sprite_atlas = build_nineslice_atlas(&r_2d);
    let mut cells = vec![TileCell::new(1, false, false, false, 1); 20 * 12];
    for y in 0..12usize {
        for x in 0..20usize {
            let idx = y * 20 + x;
            if (x + y) % 4 == 0 {
                cells[idx] = TileCell::new(2, (x % 2) == 1, false, (y % 2) == 1, 1);
            } else if (x == 9 || x == 10) && (3..=8).contains(&y) {
                cells[idx] = TileCell::new(3, false, false, false, 2);
            }
        }
    }
    let cell_buf = r_2d.upload_tilemap_cells(&cells);

    // Seed GPU particles: fountain of 320 glowing cubes dropping onto the lit floor!
    let mut seed = Vec::with_capacity(320);
    for i in 0..320u32 {
        let fi = i as f32;
        let angle = fi * 2.39996;
        let rad = 0.25 + (fi % 19.0) * 0.08;
        let h = 1.6 + (fi / 320.0) * 2.8;
        let vx = angle.cos() * (0.8 + (fi % 7.0) * 0.25);
        let vz = angle.sin() * (0.8 + (fi % 5.0) * 0.25);
        let warm = (fi % 3.0) / 2.0;
        let color = [2.6, 0.75 + warm * 1.1, 0.12 + warm * 0.2, 1.0];
        seed.push(GpuParticle::primary(
            [angle.cos() * rad, h, angle.sin() * rad],
            [vx, 1.4 - (fi % 4.0) * 0.5, vz],
            8.0,
            0.13 + (fi % 4.0) * 0.025,
            color,
        ));
    }
    let mut p_scene = GpuParticleScene {
        max_particles: 2048,
        gravity: [0.0, -7.5, 0.0],
        dt: 1.0 / 30.0,
        drag: 0.12,
        curl_strength: 2.2,
        curl_frequency: 0.6,
        restitution: 0.62,
        friction: 0.18,
        depth_thickness: 0.45,
        depth_collide: true,
        sub_emit_on_bounce: true,
        sub_emit_count: 3,
        sub_emit_speed: 3.4,
        sub_emit_life: 1.2,
        velocity_align: true,
        ambient: 0.45,
        pending_seed: Some(seed.clone()),
        ..GpuParticleScene::default()
    };
    r_particles.set_gpu_particle_scene(p_scene.clone());
    p_scene.pending_seed = None;

    if let Some(fdir) = &frames_dir {
        std::fs::create_dir_all(fdir)?;
    }

    let mut last_composite = RgbaImage::new(PANEL_W * 2, PANEL_H * 2);
    for frame_idx in 0..frame_count {
        let t = frame_idx as f32 / 30.0;
        let orbit = 0.35 + t * 0.55;
        let eye = [orbit.sin() * 5.6, 2.3, orbit.cos() * 5.6];

        // --- Panel 1 (Top-Left): Continuous Meshlet DAGs + Multi-Object JFA Silhouette Outlines ---
        let jfa_draws = vec![
            Draw {
                mesh: 1,
                material: 0,
                transform: trs([0.0, -1.1, 0.0], 0.0, [9.0, 0.2, 9.0]),
                parameters: None,
                extra: None,
            },
            Draw {
                mesh: 0,
                material: 1,
                transform: trs([-1.35, 0.0, 0.0], t, [0.92; 3]),
                parameters: None,
                extra: None,
            },
            Draw {
                mesh: 0,
                material: 2,
                transform: trs([1.35, 0.0, -0.35], -t, [0.92; 3]),
                parameters: None,
                extra: None,
            },
            Draw {
                mesh: 1,
                material: 3,
                transform: trs([0.2, -0.15, 1.35], 0.2, [1.5, 1.7, 0.32]),
                parameters: None,
                extra: None,
            },
            Draw {
                mesh: 0,
                material: 4,
                transform: trs([0.0, 0.55, -1.7], t * 0.5, [0.75; 3]),
                parameters: None,
                extra: None,
            },
        ];
        r_jfa.set_jfa_outlines(JfaOutlineScene {
            styles: vec![
                OutlineStyle {
                    color: [2.4, 1.8, 0.15, 1.0],
                    occluded_color: [0.18, 0.72, 0.9, 0.85],
                    width_px: 5.5,
                    feather_px: 1.2,
                    fill_alpha: 0.0,
                },
                OutlineStyle {
                    color: [0.2, 2.2, 2.5, 1.0],
                    occluded_color: [0.65, 0.2, 0.85, 0.8],
                    width_px: 5.0,
                    feather_px: 1.2,
                    fill_alpha: 0.0,
                },
                OutlineStyle {
                    color: [2.4, 0.3, 1.7, 1.0],
                    occluded_color: [0.55, 0.15, 0.5, 0.7],
                    width_px: 4.5,
                    feather_px: 1.2,
                    fill_alpha: 0.0,
                },
            ],
            targets: vec![
                JfaOutlineTarget { draw_index: 1, style_index: 0 },
                JfaOutlineTarget { draw_index: 2, style_index: 1 },
                JfaOutlineTarget { draw_index: 4, style_index: 2 },
            ],
            max_radius_px: 8,
            depth_bias: 2e-4,
        });
        let f_jfa = base_3d_frame(eye, [0.0, 0.0, 0.0], jfa_draws);
        if frame_idx == 0 {
            wait_ready(&mut r_jfa, &f_jfa, &catalog, &mut panel);
        } else {
            r_jfa.submit(&f_jfa, &catalog, &mut panel)?;
        }
        let p1 = panel.read_rgba(&device, &queue);

        // --- Panel 2 (Top-Right): Antialiased 3D + Dimmed X-Ray Lines, Dashed Trajectories & 2D Overlay ---
        let line_draws = vec![
            Draw {
                mesh: 1,
                material: 0,
                transform: trs([0.0, -1.1, 0.0], 0.0, [9.0, 0.2, 9.0]),
                parameters: None,
                extra: None,
            },
            Draw {
                mesh: 1,
                material: 3,
                transform: trs([0.0, 0.1, 0.6], 0.15, [2.2, 2.2, 0.45]),
                parameters: None,
                extra: None,
            },
            Draw {
                mesh: 1,
                material: 1,
                transform: trs([-1.7, -0.2, -0.8], 0.4, [1.1, 1.4, 1.1]),
                parameters: None,
                extra: None,
            },
        ];
        let mut lb = LineBatch::new();
        lb.axes([-2.2, -0.95, 1.4], 1.1, LineStyle { width_px: 4.0, layer: LineLayer::XRay, ..Default::default() });
        lb.aabb(
            [-2.3, -0.95, -1.4],
            [-1.1, 0.55, -0.2],
            [0.2, 2.1, 2.5, 1.0],
            LineStyle { width_px: 3.0, layer: LineLayer::XRay, ..Default::default() },
        );
        lb.circle(
            [0.0, -0.95, 0.0],
            [0.0, 1.0, 0.0],
            2.5,
            48,
            [1.8, 1.4, 0.25, 1.0],
            LineStyle {
                width_px: 3.0,
                dash_len_px: 10.0,
                gap_len_px: 6.0,
                layer: LineLayer::XRay,
                ..Default::default()
            },
        );
        let mut traj = Vec::new();
        for s in 0..40 {
            let u = s as f32 / 39.0;
            let x = -2.6 + u * 5.2;
            let y = -0.3 + (u * std::f32::consts::PI).sin() * 1.45;
            let z = -1.2 + (u * 5.0 + t * 2.0).sin() * 0.4;
            traj.push([x, y, z]);
        }
        lb.polyline(&traj, 4.5, [2.6, 0.35, 1.4, 1.0], false, 12.0, 6.0, true);
        // 2D screen-space oscilloscope / marquee overlay at bottom-left of panel 2
        let mut graph = Vec::new();
        for i in 0..36 {
            let u = i as f32 / 35.0;
            graph.push([18.0 + u * 165.0, 326.0 - (u * 12.0 + t * 4.0).sin() * 14.0]);
        }
        lb.polyline_2d(&graph, [0.2, 1.0, 0.65, 0.95], false, LineStyle { width_px: 2.5, ..Default::default() });
        r_lines.set_line_scene(LineScene {
            batch: lb,
            xray: XRayStyle { alpha: 0.32, desaturate: 0.25, solid_line_dash_px: Some([8.0, 5.0]) },
        });
        let f_lines = base_3d_frame(eye, [0.0, 0.0, 0.0], line_draws);
        if frame_idx == 0 {
            wait_ready(&mut r_lines, &f_lines, &catalog, &mut panel);
        } else {
            r_lines.submit(&f_lines, &catalog, &mut panel)?;
        }
        let p2 = panel.read_rgba(&device, &queue);

        // --- Panel 3 (Bottom-Left): Live Stateful GPU Mesh Particles + Depth-Buffer Bounce ---
        let p_draws = vec![
            Draw {
                mesh: 1,
                material: 0,
                transform: trs([0.0, -1.1, 0.0], 0.0, [10.0, 0.25, 10.0]),
                parameters: None,
                extra: None,
            },
            Draw {
                mesh: 1,
                material: 3,
                transform: trs([0.0, -0.55, 0.0], 0.4, [2.0, 0.8, 2.0]),
                parameters: None,
                extra: None,
            },
        ];
        if frame_idx > 0 && frame_idx % 75 == 0 {
            r_particles.seed_gpu_particles(seed.clone());
        }
        let mut f_gp = base_3d_frame([4.6, 2.5, 4.6], [0.0, -0.2, 0.0], p_draws);
        f_gp.lights.push(Light {
            position: Vector { x: 0.0, y: 1.5, z: 0.0 },
            color: Vector { x: 2.0, y: 1.0, z: 0.4 },
            radius: 10.0,
            ..Default::default()
        });
        if frame_idx == 0 {
            wait_ready(&mut r_particles, &f_gp, &catalog, &mut panel);
            // Warm up 18 physics frames so particles are mid-cascade and bouncing off the pedestal/floor!
            for _ in 0..18 {
                r_particles.submit(&f_gp, &catalog, &mut panel)?;
            }
        } else {
            r_particles.submit(&f_gp, &catalog, &mut panel)?;
        }
        let p3 = panel.read_rgba(&device, &queue);

        // --- Panel 4 (Bottom-Right): First-Class 2D Suite (Tilemap + 9-Slice Sprites + RC2D GI + SDF UI) ---
        let cam2d =
            Camera2d { center: [0.0, 0.0], zoom: 1.0, rotation: 0.0, viewport: [PANEL_W as f32, PANEL_H as f32] };
        let lantern_x = 125.0 + (t * 1.4).sin() * 48.0;
        let lantern_y = 180.0 + (t * 1.9).cos() * 54.0;
        let beacon_x = 515.0 + (t * 1.2).cos() * 40.0;
        let beacon_y = 185.0 + (t * 1.5).sin() * 55.0;
        let progress = 0.25 + 0.65 * (0.5 + 0.5 * (t * 1.3).sin());
        r_2d.set_scene_2d(Scene2d {
            camera: cam2d,
            tilemaps: vec![TilemapLayer2d {
                uniform: TilemapUniform {
                    camera_vp: cam2d.world_to_ndc(),
                    origin_and_tile_size: [-320.0, 192.0, 32.0, 32.0],
                    grid_size_and_atlas: [20, 12, 4, 1],
                    anim_time_and_tint: [t, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                },
                cells: cell_buf.clone(),
                atlas: tile_atlas.clone(),
                sampler: None,
            }],
            sprite_layers: vec![SpriteLayer2d {
                sprites: vec![
                    Sprite2d {
                        pos: [-120.0, -55.0],
                        size: [112.0, 36.0],
                        nine_slice: [4.0; 4],
                        color: [1.0, 1.0, 1.0, 1.0],
                        ..Default::default()
                    },
                    Sprite2d {
                        pos: [120.0, -50.0],
                        size: [48.0, 48.0],
                        pivot_rot_layer: [0.5, 0.5, t * 0.9, 1.0],
                        color: [1.0, 1.0, 1.0, 1.0],
                        ..Default::default()
                    },
                ],
                atlas: sprite_atlas.clone(),
                sampler: None,
                y_sort: true,
            }],
            radiance_cascades: Some(RadianceCascades2dLayer {
                emitters: vec![
                    Scene2dEmitter::Circle {
                        center: [lantern_x, lantern_y],
                        radius: 15.0,
                        material: [6.0, 3.2, 0.85, 1.0],
                    },
                    Scene2dEmitter::Circle {
                        center: [beacon_x, beacon_y],
                        radius: 14.0,
                        material: [0.5, 3.6, 5.5, 1.0],
                    },
                    Scene2dEmitter::Rect { rect: [286.0, 78.0, 26.0, 204.0], material: [0.0, 0.0, 0.0, 1.0] },
                    Scene2dEmitter::Rect { rect: [410.0, 110.0, 24.0, 140.0], material: [0.0, 0.0, 0.0, 1.0] },
                ],
                scene_view: None,
                config: RadianceCascades2dConfig::default(),
                composite_weights: [0.08, 1.15, 0.22],
                opaque_backdrop: true,
            }),
            sdf_shapes: vec![
                // Glass UI panel top-left with gradient, crisp border, and Gaussian drop shadow
                SdfShape {
                    rect: [18.0, 18.0, 210.0, 78.0],
                    corner_radii: [14.0, 14.0, 14.0, 14.0],
                    fill_color: [0.07, 0.10, 0.18, 0.90],
                    fill_color_end: [0.13, 0.18, 0.30, 0.92],
                    border_color: [0.45, 0.78, 1.0, 0.85],
                    shadow_color: [0.0, 0.0, 0.0, 0.72],
                    kind: 0,
                    border_width_px: 2.0,
                    gradient_kind: 1,
                    gradient_angle: 0.45,
                    shadow_offset: [0.0, 7.0],
                    shadow_blur_px: 8.0,
                    ..Default::default()
                },
                // Track bar
                SdfShape {
                    rect: [34.0, 56.0, 178.0, 18.0],
                    corner_radii: [9.0; 4],
                    fill_color: [0.04, 0.06, 0.11, 0.95],
                    border_color: [0.25, 0.34, 0.48, 0.75],
                    border_width_px: 1.5,
                    ..Default::default()
                },
                // Gradient progress fill clipped cleanly inside rounded track!
                SdfShape {
                    rect: [34.0, 56.0, 178.0 * progress, 18.0],
                    corner_radii: [9.0; 4],
                    fill_color: [1.0, 0.62, 0.12, 1.0],
                    fill_color_end: [0.15, 0.92, 0.78, 1.0],
                    gradient_kind: 1,
                    clip_rect: [34.0, 56.0, 178.0, 18.0],
                    clip_radius_px: 9.0,
                    ..Default::default()
                },
                // Glowing circular status badge
                SdfShape {
                    rect: [34.0, 30.0, 16.0, 16.0],
                    fill_color: [0.25, 1.0, 0.65, 1.0],
                    fill_color_end: [0.05, 0.55, 0.35, 1.0],
                    kind: 1,
                    gradient_kind: 2,
                    ..Default::default()
                },
            ],
            sdf_overlay_shapes: vec![],
        });
        let mut f_2d = RenderFrame {
            projection: pen_render::perspective_forward_z(
                std::f32::consts::FRAC_PI_2,
                PANEL_W as f32 / PANEL_H as f32,
                0.1,
                100.0,
            ),
            view: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            clear: Color { r: 0.02, g: 0.02, b: 0.04, a: 1.0 },
            ..RenderFrame::default()
        };
        f_2d.quality.msaa = 1;
        f_2d.quality.scale = 1.0;
        f_2d.quality.sky = false;
        r_2d.submit(&f_2d, &catalog, &mut panel)?;
        let p4 = panel.read_rgba(&device, &queue);

        let mut composite = RgbaImage::new(PANEL_W * 2, PANEL_H * 2);
        blit_quadrant(&mut composite, &p1, 0, 0);
        blit_quadrant(&mut composite, &p2, PANEL_W, 0);
        blit_quadrant(&mut composite, &p3, 0, PANEL_H);
        blit_quadrant(&mut composite, &p4, PANEL_W, PANEL_H);
        draw_divider(&mut composite);

        if let Some(fdir) = &frames_dir {
            composite.save(fdir.join(format!("frame_{frame_idx:04}.png")))?;
        }
        last_composite = composite;
    }

    last_composite.save(&out_png)?;
    println!(
        "wrote {} ({}x{}, {} frame(s))",
        out_png.display(),
        last_composite.width(),
        last_composite.height(),
        frame_count
    );
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
