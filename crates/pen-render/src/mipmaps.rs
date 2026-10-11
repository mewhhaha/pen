// WebGPU has no mipmap generation: each smaller level of a decoded image
// (packed textures bring their own) is drawn by filtering the one above it.
use std::collections::HashMap;

const SHADER: &str = r#"
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
struct Corner {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};
@vertex fn vertex(@builtin(vertex_index) index: u32) -> Corner {
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return Corner(vec4f(corner * 2.0 - 1.0, 0.0, 1.0), vec2f(corner.x, 1.0 - corner.y));
}
@fragment fn fragment(corner: Corner) -> @location(0) vec4f {
  return textureSampleLevel(source, source_sampler, corner.uv, 0.0);
}
"#;

pub(crate) struct Mipmapper {
    module: wgpu::ShaderModule,
    sampler: wgpu::Sampler,
    pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

impl Mipmapper {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            module: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mipmaps"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            }),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            pipelines: HashMap::new(),
        }
    }

    /// Fills every level of `texture` below the first from the first.
    pub fn generate(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) {
        if texture.mip_level_count() < 2 {
            return;
        }
        let module = &self.module;
        let pipeline = self.pipelines.entry(texture.format()).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mipmaps"),
                layout: None,
                vertex: wgpu::VertexState {
                    module,
                    entry_point: None,
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: None,
                    compilation_options: Default::default(),
                    targets: &[Some(texture.format().into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
        let level_view = |level: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        for level in 1..texture.mip_level_count() {
            let target = level_view(level);
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&level_view(level - 1)),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
    }
}
