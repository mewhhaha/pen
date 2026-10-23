// Sprites: the rectangles a frame lays over the finished image, in order. Each
// draws with its material's sprite shader (a WGSL asset) over the material's
// texture. Every sprite shader sees the same bindings:
//
//   @group(0) @binding(0) var<uniform> screen: vec4f;  // width, height in px
//   @group(0) @binding(1) var<storage, read> sprites: array<Sprite>;
//   @group(0) @binding(2) var image: texture_2d<f32>;
//   @group(0) @binding(3) var image_sampler: sampler;
//   @group(0) @binding(4) var scene: texture_2d<f32>; // resolved linear scene, before post/UI
//
// Scene coordinates are screen UVs, independent of render scale. Standalone
// overlays and loading sprites receive white instead of a previous frame.
//
// where Sprite is { rect: vec4f, uv: vec4f, color: vec4f, parameters: vec4f }
// (x, y, width, height in pixels from the top left; u0, v0, u1, v1; RGBA;
// four values for the shader), and draws six vertices per instance, the
// instance being the sprite's index. See src/assets/sprite.wgsl.
use std::collections::HashMap;

use pen_protocol::Sprite;

const SPRITE_FLOATS: usize = 16;

/// The bind group layout every sprite pipeline shares.
pub(crate) fn sprite_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let both = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sprites"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: both,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: both,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

/// Consecutive sprites with one shader and texture: `count` of them from
/// `first` in the drawing order (sprites still loading are left out).
pub(crate) struct SpriteBatch {
    pub pipeline: wgpu::RenderPipeline,
    pub texture: wgpu::TextureView,
    /// Which texture view and pipeline they are, to batch by.
    pub key: (u64, usize),
    pub first: u32,
    pub count: u32,
}

pub(crate) struct Sprites {
    layout: wgpu::BindGroupLayout,
    screen: wgpu::Buffer,
    sampler: wgpu::Sampler,
    scene: Option<wgpu::TextureView>,
    white: wgpu::TextureView,
    capacity: usize,
    storage: Option<wgpu::Buffer>,
    data: Vec<f32>,
    /// Bind groups by texture view, with the draw that last used each.
    groups: HashMap<u64, (wgpu::BindGroup, u64)>,
    draws: u64,
}

// Bind groups no draw has used for this many draws are let go.
const GROUP_LIFETIME: u64 = 120;

impl Sprites {
    pub fn new(device: &wgpu::Device, layout: wgpu::BindGroupLayout) -> Self {
        Self {
            layout,
            screen: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sprite screen"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                ..Default::default()
            }),
            scene: None,
            white: device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("sprite scene fallback"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                })
                .create_view(&Default::default()),
            capacity: 0,
            storage: None,
            data: Vec::new(),
            groups: HashMap::new(),
            draws: 0,
        }
    }

    /// The scene is immutable during UI drawing, so overlapping sprites all
    /// sample the world, not each other's ink. Resizes invalidate cached groups.
    pub fn set_scene(&mut self, scene: Option<wgpu::TextureView>) {
        if self.scene != scene {
            self.scene = scene;
            self.groups.clear();
        }
    }

    /// Draws `batches` of `sprites` (by their indices in `order`), in order,
    /// over `output`, a `size` screen.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        size: (u32, u32),
        batches: &[SpriteBatch],
        sprites: &[Sprite],
        order: &[u32],
        timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        self.set_scene(None);
        self.prepare(device, queue, size, sprites, order);
        self.draw_prepared(device, encoder, output, batches, 0..order.len() as u32, timestamps);
    }

    /// Upload once, then draw ranges between the frame's 3D UI layers.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: (u32, u32),
        sprites: &[Sprite],
        order: &[u32],
    ) {
        let count = order.len();
        if count == 0 {
            return;
        }
        if self.scene.is_none() {
            queue.write_texture(
                self.white.texture().as_image_copy(),
                &[255, 255, 255, 255],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
        }
        self.draws += 1;
        if self.draws.is_multiple_of(GROUP_LIFETIME) {
            let oldest = self.draws - GROUP_LIFETIME;
            self.groups.retain(|_, (_, used)| *used >= oldest);
        }
        if count > self.capacity {
            // The groups bind the storage buffer this replaces.
            self.groups.clear();
            self.capacity = count.next_power_of_two();
            self.storage = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sprites"),
                size: (self.capacity * SPRITE_FLOATS * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let data = &mut self.data;
        data.clear();
        for sprite in order.iter().map(|&index| &sprites[index as usize]) {
            data.extend([sprite.x, sprite.y, sprite.width, sprite.height, sprite.u0, sprite.v0, sprite.u1, sprite.v1]);
            data.extend(sprite.color);
            data.extend(sprite.parameters);
        }
        let storage = self.storage.clone().unwrap();
        queue.write_buffer(&storage, 0, bytemuck::cast_slice(data));
        queue.write_buffer(&self.screen, 0, bytemuck::cast_slice(&[size.0 as f32, size.1 as f32, 0.0, 0.0]));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_prepared(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        batches: &[SpriteBatch],
        range: std::ops::Range<u32>,
        timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        if range.is_empty() {
            return;
        }
        let storage = self.storage.clone().unwrap();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sprites"),
            timestamp_writes: timestamps,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        let draws = self.draws;
        for batch in batches {
            let first = batch.first.max(range.start);
            let end = (batch.first + batch.count).min(range.end);
            if first >= end {
                continue;
            }
            pass.set_pipeline(&batch.pipeline);
            let (group, used) = self.groups.entry(batch.key.0).or_insert_with(|| {
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.screen.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: storage.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&batch.texture),
                        },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(self.scene.as_ref().unwrap_or(&self.white)),
                        },
                    ],
                });
                (group, draws)
            });
            *used = draws;
            pass.set_bind_group(0, &*group, &[]);
            pass.draw(0..6, first..end);
        }
    }
}
