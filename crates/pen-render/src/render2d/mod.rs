//! Standalone 2D draw and lighting passes. Callers own the render targets,
//! resource updates and the order in which passes are encoded.

pub mod radiance_cascades;
pub mod sdf_ui;
pub mod sprites2d;
pub mod tilemap;

pub use radiance_cascades::{RadianceCascades2d, RadianceCascades2dConfig, Scene2dEmitter, rasterize_scene_2d};
pub use sdf_ui::{SdfShape, SdfShapePass};
pub use sprites2d::{Camera2d, Sprite2d, Sprites2dPass, sort_sprites};
pub use tilemap::{TileCell, TilemapPass, TilemapUniform};

use wgpu::util::DeviceExt;

/// A prepared draw owns its bindings. Each batch can be encoded independently,
/// including several batches prepared before a single queue submission.
pub struct Draw2d {
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
    instances: u32,
}

impl Draw2d {
    /// Draw into a single-sampled color attachment matching the pass format.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.draw(0..6, 0..self.instances);
    }
}

fn buffer(device: &wgpu::Device, label: &str, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytes, usage })
}

fn pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    format: wgpu::TextureFormat,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend: Some(blend), write_mask: wgpu::ColorWrites::ALL })],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: crate::pipeline_cache::active().as_ref(),
    })
}

fn prepared(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    resources: &[wgpu::BindingResource<'_>],
    instances: u32,
) -> Draw2d {
    let entries: Vec<_> = resources
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, resource)| wgpu::BindGroupEntry { binding: u32::try_from(index).unwrap(), resource })
        .collect();
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    Draw2d { pipeline: pipeline.clone(), group, instances }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::testing;

    fn texture(
        device: &wgpu::Device,
        size: [u32; 2],
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("2D witness target"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: usage | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    fn read_texture(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, bpp: u32) -> Vec<u8> {
        let size = texture.size();
        let row = size.width * bpp;
        let stride = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let bytes = u64::from(stride) * u64::from(size.height);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("2D witness readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(size.height),
                },
            },
            size,
        );
        queue.submit([encoder.finish()]);
        testing::read(device, queue, &buffer, bytes)
            .chunks_exact(stride as usize)
            .flat_map(|padded| padded[..row as usize].iter().copied())
            .collect()
    }

    fn draw_rgba8(device: &wgpu::Device, queue: &wgpu::Queue, size: [u32; 2], draw: &Draw2d) -> Vec<[u8; 4]> {
        let target = texture(device, size, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let view = target.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            draw.draw(&mut pass);
        }
        queue.submit([encoder.finish()]);
        read_texture(device, queue, &target, 4).as_chunks::<4>().0.to_vec()
    }

    fn read_hdr(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<[f32; 4]> {
        read_texture(device, queue, texture, 8)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|rgba| std::array::from_fn(|i| testing::unhalf(u16::from_le_bytes([rgba[2 * i], rgba[2 * i + 1]]))))
            .collect()
    }

    const CLEAR: [u8; 4] = [0, 0, 0, 0];
    const RED: [u8; 4] = [255, 0, 0, 255];
    const GREEN: [u8; 4] = [0, 255, 0, 255];
    const BLUE: [u8; 4] = [0, 0, 255, 255];
    const YELLOW: [u8; 4] = [255, 255, 0, 255];
    const MAGENTA: [u8; 4] = [255, 0, 255, 255];
    const WHITE: [u8; 4] = [255; 4];

    fn linear(color: [u8; 4]) -> [f32; 4] {
        color.map(|channel| f32::from(channel) / 255.0)
    }

    #[test]
    fn sprites_rotate_sort_and_keep_nine_slice_corners() {
        let (device, queue) = testing::device().expect("headless GPU required");
        let size = [64, 40];
        let camera = Camera2d { center: [0.0; 2], zoom: 1.0, rotation: 0.0, viewport: [64.0, 40.0] };
        let world = |x: f32, y: f32| [x - 32.0, 20.0 - y];
        let palette: Vec<_> = (0..8)
            .flat_map(|y| {
                (0..16).map(move |x| {
                    linear(match (x, y) {
                        (0..=1, 0..=1) => RED,
                        (6..=7, 0..=1) => GREEN,
                        (0..=1, 6..=7) => YELLOW,
                        (6..=7, 6..=7) => MAGENTA,
                        (8.., _) => WHITE,
                        _ => BLUE,
                    })
                })
            })
            .collect();
        let atlas = testing::picture(&device, &queue, 16, 8, &palette).create_view(&Default::default());
        let sampler = device.create_sampler(&Default::default());
        let panel = Sprite2d {
            pos: world(20.0, 8.0),
            size: [32.0, 8.0],
            uv_rect: [0.0, 0.0, 0.5, 1.0],
            nine_slice: [2.0; 4],
            ..Default::default()
        };
        let rotated = Sprite2d {
            pos: world(48.0, 8.0),
            size: [8.0, 8.0],
            pivot_rot_layer: [0.5, 0.5, std::f32::consts::FRAC_PI_2, 0.0],
            nine_slice: [0.0; 4],
            ..panel
        };
        let upper = Sprite2d {
            pos: world(20.0, 24.0),
            size: [16.0, 10.0],
            uv_rect: [0.5, 0.0, 1.0, 1.0],
            color: linear(RED),
            ..Default::default()
        };
        let lower = Sprite2d { pos: world(20.0, 27.0), color: linear(GREEN), ..upper };
        let highest =
            Sprite2d { size: [4.0, 6.0], pivot_rot_layer: [0.5, 0.5, 0.0, 1.0], color: linear(YELLOW), ..upper };
        let mut sprites = [lower, highest, upper, panel, rotated];
        sort_sprites(&mut sprites, true);
        let pass = Sprites2dPass::new(&device, wgpu::TextureFormat::Rgba8Unorm);
        let draw = pass.prepare(&device, camera, &sprites, &atlas, &sampler).unwrap();
        let pixels = draw_rgba8(&device, &queue, size, &draw);
        let at = |x: usize, y: usize| pixels[y * 64 + x];
        let top: Vec<_> = (4..36).map(|x| at(x, 4)).collect();
        assert_eq!(&top[..2], &[RED; 2]);
        assert_eq!(&top[2..30], &[BLUE; 28]);
        assert_eq!(&top[30..], &[GREEN; 2]);
        assert_eq!(at(4, 5), RED);
        assert_eq!(at(4, 6), BLUE);
        assert_eq!(at(4, 11), YELLOW);
        assert_eq!(at(35, 11), MAGENTA);
        assert_eq!(at(51, 11), GREEN);
        assert_eq!(at(51, 4), RED);
        assert_eq!(at(15, 25), GREEN);
        assert_eq!(at(20, 25), YELLOW);

        let tied = Sprite2d { color: linear(BLUE), ..lower };
        let mut same_key = [lower, tied, upper];
        sort_sprites(&mut same_key, true);
        assert_eq!(same_key.map(|sprite| sprite.color), [upper.color, lower.color, tied.color]);
        sort_sprites(&mut same_key, false);
        assert_eq!(same_key.map(|sprite| sprite.color), [upper.color, lower.color, tied.color]);

        println!(
            "sprites: stretched row has 2 red + 28 blue + 2 green pixels; rotated (51,11)={:?}, lower-Y={:?}, higher layer={:?}",
            at(51, 11),
            at(15, 25),
            at(20, 25),
        );
    }

    #[test]
    fn camera_transform_handles_center_zoom_and_rotation() {
        let camera = Camera2d {
            center: [10.0, -8.0],
            zoom: 2.0,
            rotation: std::f32::consts::FRAC_PI_2,
            viewport: [100.0, 40.0],
        };
        let m = camera.world_to_ndc();
        let transform = |x: f32, y: f32| std::array::from_fn::<_, 2, _>(|i| m[0][i] * x + m[1][i] * y + m[3][i]);
        for (world, expected) in [([10.0, -8.0], [0.0, 0.0]), ([20.0, -8.0], [0.0, 1.0]), ([10.0, 17.0], [-1.0, 0.0])] {
            for (actual, expected) in transform(world[0], world[1]).into_iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn tilemap_one_draw_hides_empty_cells_flips_and_animates() {
        let (device, queue) = testing::device().expect("headless GPU required");
        let size = [32, 32];
        let camera = Camera2d { center: [0.0; 2], zoom: 1.0, rotation: 0.0, viewport: [32.0; 2] };
        let pixels: Vec<_> = [RED, RED, GREEN, BLUE, YELLOW, YELLOW, MAGENTA, MAGENTA]
            .into_iter()
            .cycle()
            .take(16)
            .map(linear)
            .collect();
        let atlas = testing::picture(&device, &queue, 8, 2, &pixels).create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mut cells = [TileCell::EMPTY; 16];
        cells[1] = TileCell::new(1, false, false, false, 0);
        cells[2] = TileCell::new(2, true, false, false, 1);
        cells[3] = TileCell::new(3, false, false, false, 2);
        cells[4] = TileCell::new(2, false, false, true, 0);
        cells[15] = cells[1];
        let cells = TilemapPass::upload_cells(&device, &cells);
        let pass = TilemapPass::new(&device, wgpu::TextureFormat::Rgba8Unorm);
        let frame = |t| {
            let uniform = TilemapUniform {
                camera_vp: camera.world_to_ndc(),
                origin_and_tile_size: [-16.0, 16.0, 8.0, 8.0],
                grid_size_and_atlas: [4, 4, 4, 1],
                anim_time_and_tint: [t, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            };
            let draw = pass.prepare(&device, &uniform, &cells, &atlas, &sampler);
            assert_eq!(draw.instances, 1);
            draw_rgba8(&device, &queue, size, &draw)
        };
        let at = |pixels: &[[u8; 4]], x: usize, y: usize| pixels[y * 32 + x];
        let start = frame(0.0);
        let next = frame(1.0);
        assert_eq!(at(&start, 4, 4), CLEAR);
        assert_eq!(at(&start, 12, 4), RED);
        assert_eq!(at(&start, 28, 28), RED);
        assert_eq!(at(&start, 16, 4), BLUE);
        assert_eq!(at(&start, 23, 4), GREEN);
        assert_eq!(at(&start, 4, 8), GREEN);
        assert_eq!(at(&start, 4, 15), BLUE);
        for (pixels, color) in [(&start, YELLOW), (&next, MAGENTA)] {
            for x in [24, 28, 31] {
                assert_eq!(at(pixels, x, 4), color);
            }
        }
        println!(
            "tilemap: empty={:?}, red={:?}, flipped={:?}/{:?}, animation={:?}/{:?}",
            at(&start, 4, 4),
            at(&start, 12, 4),
            at(&start, 16, 4),
            at(&start, 23, 4),
            at(&start, 28, 4),
            at(&next, 28, 4),
        );
    }

    #[test]
    fn sdf_rounded_gradient_border_shadow_and_clip() {
        let (device, queue) = testing::device().expect("headless GPU required");
        let panel = SdfShape {
            rect: [8.0, 8.0, 44.0, 26.0],
            corner_radii: [12.0, 5.0, 8.0, 2.0],
            fill_color: linear(RED),
            fill_color_end: linear(BLUE),
            border_color: linear(WHITE),
            shadow_color: [0.15, 0.15, 0.15, 0.8],
            border_width_px: 3.0,
            gradient_kind: 1,
            shadow_offset: [0.0, 6.0],
            shadow_blur_px: 3.0,
            ..Default::default()
        };
        let child = SdfShape {
            rect: [60.0, 10.0, 34.0, 30.0],
            fill_color: linear(GREEN),
            clip_rect: [66.0, 15.0, 18.0, 17.0],
            clip_radius_px: 4.0,
            ..Default::default()
        };
        let pass = SdfShapePass::new(&device, wgpu::TextureFormat::Rgba8Unorm);
        let draw = pass.prepare(&device, [96.0, 56.0], &[panel, child]).unwrap();
        let pixels = draw_rgba8(&device, &queue, [96, 56], &draw);
        let at = |x: usize, y: usize| pixels[y * 96 + x];
        let corner = at(8, 8);
        assert!(corner[0] <= 2 && corner[3] < 16, "rounded corner: {corner:?}");
        for y in 8..11 {
            assert_eq!(at(30, y), WHITE);
        }
        let left = at(16, 20);
        let right = at(44, 20);
        assert!(left[0] > 190 && left[2] < 55 && right[2] > 190 && right[0] < 55);
        let shadow = [35, 39, 43].map(|y| at(30, y));
        assert!(shadow[0][3] > shadow[1][3] && shadow[1][3] > shadow[2][3] && shadow[2][3] > 0);
        assert!(shadow[0][0] < 40);
        assert_eq!(at(30, 54), CLEAR);
        assert_eq!(at(74, 23), GREEN);
        for (x, y) in [(65, 23), (84, 23), (74, 14), (74, 32), (66, 15)] {
            assert_eq!(at(x, y), CLEAR, "outside clip at {x},{y}");
        }
        println!(
            "SDF: corner={corner:?}, border={:?}, gradient={left:?}/{right:?}, shadow={shadow:?}, clipped={:?}",
            at(30, 9),
            at(65, 23),
        );
    }

    #[test]
    fn sdf_ellipse_radial_and_rotated_capsule() {
        let (device, queue) = testing::device().expect("headless GPU required");
        let ellipse = SdfShape {
            rect: [4.5, 4.5, 28.0, 16.0],
            fill_color: linear(RED),
            fill_color_end: linear(BLUE),
            gradient_radius_px: 12.0,
            kind: 1,
            border_width_px: 2.0,
            gradient_kind: 2,
            border_color: linear(WHITE),
            ..Default::default()
        };
        let capsule = SdfShape {
            rect: [36.5, 9.5, 24.0, 6.0],
            fill_color: linear(GREEN),
            kind: 2,
            rotation: std::f32::consts::FRAC_PI_4,
            ..Default::default()
        };
        let pass = SdfShapePass::new(&device, wgpu::TextureFormat::Rgba8Unorm);
        let draw = pass.prepare(&device, [64.0, 28.0], &[ellipse, capsule]).unwrap();
        let pixels = draw_rgba8(&device, &queue, [64, 28], &draw);
        let at = |x: usize, y: usize| pixels[y * 64 + x];
        assert_eq!(at(18, 12), RED);
        assert!(at(29, 12)[2] > 220);
        assert_eq!(at(18, 19), WHITE);
        assert_eq!(at(5, 5), CLEAR);
        assert_eq!(at(55, 19), GREEN);
        assert_eq!(at(55, 6), CLEAR);
    }

    #[test]
    fn radiance_cascades_keep_umbra_and_resolve_penumbra() {
        let (device, queue) = testing::device().expect("headless GPU required");
        let size = [256, 256];
        let warm = Scene2dEmitter::Circle { center: [48.0, 128.0], radius: 54.0, material: [5.0, 2.6, 0.8, 1.0] };
        let wall = Scene2dEmitter::Rect { rect: [120.0, 64.0, 16.0, 128.0], material: [0.0, 0.0, 0.0, 1.0] };
        let green = Scene2dEmitter::Circle { center: [208.0, 128.0], radius: 18.0, material: [0.0, 6.0, 0.0, 1.0] };
        let rc = RadianceCascades2d::new(&device, size, RadianceCascades2dConfig::default()).unwrap();
        assert_eq!(rc.cascade_count(), 5);
        let albedo = testing::picture(&device, &queue, 256, 256, &vec![[0.25, 0.5, 0.75, 0.5]; 256 * 256])
            .create_view(&Default::default());
        let composite = texture(&device, size, wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING);
        let render = |emitters: &[Scene2dEmitter]| {
            let scene = testing::picture(&device, &queue, 256, 256, &rasterize_scene_2d(size, emitters));
            let mut encoder = device.create_command_encoder(&Default::default());
            rc.build(&device, &mut encoder, &scene.create_view(&Default::default()));
            rc.composite(
                &device,
                &mut encoder,
                &albedo,
                &composite.create_view(&Default::default()),
                [0.125, 1.0, 0.25],
            );
            queue.submit([encoder.finish()]);
            read_hdr(&device, &queue, rc.irradiance())
        };
        let luminance = |[r, g, b, _]: [f32; 4]| 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let sample = |pixels: &[[f32; 4]], x: usize, y: usize| pixels[y * 256 + x];
        let one_light = render(&[warm, wall]);
        let lit = luminance(sample(&one_light, 108, 128));
        let umbra = luminance(sample(&one_light, 160, 128));
        let transition = [48, 52, 56, 60, 64].map(|y| luminance(sample(&one_light, 160, y)));
        let combined = read_hdr(&device, &queue, &composite);
        let lit_rgb = sample(&one_light, 108, 128);
        let mixed = sample(&combined, 108, 128);
        for i in 0..3 {
            let expected = [0.25, 0.5, 0.75][i] * (0.125 + lit_rgb[i]) + lit_rgb[i] * 0.25;
            assert!((mixed[i] - expected).abs() < 0.002, "composite channel {i}: {mixed:?}");
        }
        assert_eq!(mixed[3].to_bits(), 0.5_f32.to_bits());
        let two_lights = render(&[warm, wall, green]);
        let green_side = sample(&two_lights, 160, 128);
        println!(
            "RC: {} cascades, lit luminance={lit:.8}, umbra={umbra:.8}, penumbra y=48..64 {transition:?}, two-light RGB={:?}, composite={mixed:?}",
            rc.cascade_count(),
            &green_side[..3],
        );
        assert!(lit > 0.3);
        assert!(umbra < lit * 0.1);
        assert!(transition[2] > umbra && transition[2] < lit);
        assert!(transition.windows(2).all(|v| v[0] >= v[1] && v[0] - v[1] < lit * 0.3));
        assert!(green_side[1] > 0.3 && green_side[1] > 4.0 * (green_side[0] + green_side[2]));
    }
}
