// Sprites: the rectangles a frame lays over the finished image, in order. Each
// draws with its material's sprite shader (a WGSL asset) over the material's
// texture. Every sprite shader sees the same bindings:
//
//   @group(0) @binding(0) var<uniform> screen: vec4f;  // width, height in px
//   @group(0) @binding(1) var<storage, read> sprites: array<Sprite>;
//   @group(0) @binding(2) var image: texture_2d<f32>;
//   @group(0) @binding(3) var image_sampler: sampler;
//
// where Sprite is { rect: vec4f, uv: vec4f, color: vec4f, parameters: vec4f }
// (x, y, width, height in pixels from the top left; u0, v0, u1, v1; RGBA;
// four values for the shader), and draws six vertices per instance, the
// instance being the sprite's index. See src/assets/sprite.wgsl.
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
        ],
    })
}

/// Consecutive sprites with one shader and texture.
pub(crate) struct SpriteBatch<'a> {
    pub pipeline: wgpu::RenderPipeline,
    pub texture: wgpu::TextureView,
    /// Which texture view and pipeline they are, to batch by.
    pub key: (u64, usize),
    pub sprites: Vec<&'a Sprite>,
}

pub(crate) struct Sprites {
    layout: wgpu::BindGroupLayout,
    screen: wgpu::Buffer,
    sampler: wgpu::Sampler,
    capacity: usize,
    storage: Option<wgpu::Buffer>,
}

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
            capacity: 0,
            storage: None,
        }
    }

    /// Draws `batches`, in order, over `output`, a `size` screen.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        size: (u32, u32),
        batches: &[SpriteBatch<'_>],
    ) {
        let count: usize = batches.iter().map(|batch| batch.sprites.len()).sum();
        if count == 0 {
            return;
        }
        if count > self.capacity {
            self.capacity = count.next_power_of_two();
            self.storage = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sprites"),
                size: (self.capacity * SPRITE_FLOATS * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let mut data = Vec::with_capacity(count * SPRITE_FLOATS);
        for batch in batches {
            for sprite in &batch.sprites {
                data.extend([sprite.x, sprite.y, sprite.width, sprite.height, sprite.u0, sprite.v0, sprite.u1, sprite.v1]);
                data.extend(sprite.color);
                data.extend(sprite.parameters);
            }
        }
        let storage = self.storage.clone().unwrap();
        queue.write_buffer(&storage, 0, bytemuck::cast_slice(&data));
        queue.write_buffer(&self.screen, 0, bytemuck::cast_slice(&[size.0 as f32, size.1 as f32, 0.0, 0.0]));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sprites"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        let mut first = 0;
        for batch in batches {
            pass.set_pipeline(&batch.pipeline);
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.screen.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: storage.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&batch.texture) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            pass.set_bind_group(0, &group, &[]);
            let count = batch.sprites.len() as u32;
            pass.draw(0..6, first..first + count);
            first += count;
        }
    }
}
