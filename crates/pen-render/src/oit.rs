//! Weighted Blended Order-Independent Transparency (McGuire & Bavoil 2013).
//!
//! Renders overlapping transparent meshes, foliage, hair cards, and layered VFX
//! into dual offscreen targets (`accum: Rgba16Float`, `reveal: R8Unorm`) so
//! final pixels are strictly independent of draw submission order.

const OIT_COMPOSITE_WGSL: &str = include_str!("shaders/oit_composite.wgsl");

/// WGSL helper snippet for translucent fragment shaders targeting WBOIT.
pub const OIT_SHADER_HELPERS_WGSL: &str = r#"
struct OitOutput {
  @location(0) accum: vec4f,
  @location(1) reveal: f32,
};

// McGuire & Bavoil (2013) Equation 9 depth-aware weight function (compatible with forward-Z and reversed-Z).
fn oit_weight(linear_depth: f32, alpha: f32) -> f32 {
  let z = max(abs(linear_depth), 0.01);
  let w = alpha / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0));
  return clamp(w, 1e-2, 3000.0);
}

fn oit_pack(rgb: vec3f, alpha: f32, linear_depth: f32) -> OitOutput {
  let a = clamp(alpha, 0.0, 1.0);
  let w = oit_weight(linear_depth, a);
  return OitOutput(vec4f(rgb * a, a) * w, a);
}
"#;

/// Returns the two `ColorTargetState` entries required for WBOIT translucent render pipelines:
/// - `@location(0) accum`: `Rgba16Float`, additive (`One + One`)
/// - `@location(1) reveal`: `R8Unorm`, multiplicative (`Zero + OneMinusSrc`)
pub fn oit_color_targets() -> [Option<wgpu::ColorTargetState>; 2] {
    [
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::R8Unorm,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            write_mask: wgpu::ColorWrites::RED,
        }),
    ]
}

/// Offscreen accumulation + revealage targets and fullscreen composite pass for WBOIT.
pub struct OitPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    size: (u32, u32),
    accum: Option<(wgpu::Texture, wgpu::TextureView)>,
    reveal: Option<(wgpu::Texture, wgpu::TextureView)>,
    bind_group: Option<wgpu::BindGroup>,
}

impl OitPass {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat, cache: Option<&wgpu::PipelineCache>) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("oit_composite"),
            source: wgpu::ShaderSource::Wgsl(OIT_COMPOSITE_WGSL.into()),
        });
        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("oit_composite_layout"),
            entries: &[tex_entry(0), tex_entry(1)],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("oit_composite_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("oit_composite"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache,
        });
        Self { pipeline, layout, size: (0, 0), accum: None, reveal: None, bind_group: None }
    }

    /// Ensures offscreen `accum` and `reveal` textures match `(width, height)`.
    pub fn resize(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        let size = (size.0.max(1), size.1.max(1));
        if self.size == size && self.bind_group.is_some() {
            return;
        }
        let make = |label: &str, format: wgpu::TextureFormat| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            (tex, view)
        };
        let accum = make("oit_accum", wgpu::TextureFormat::Rgba16Float);
        let reveal = make("oit_reveal", wgpu::TextureFormat::R8Unorm);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oit_composite_bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&accum.1) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&reveal.1) },
            ],
        });
        self.accum = Some(accum);
        self.reveal = Some(reveal);
        self.bind_group = Some(bg);
        self.size = size;
    }

    /// Returns the two color attachments clearing `accum` to `(0,0,0,0)` and `reveal` to `1.0`.
    pub fn attachments(&self) -> [Option<wgpu::RenderPassColorAttachment<'_>>; 2] {
        [
            Some(wgpu::RenderPassColorAttachment {
                view: &self.accum.as_ref().expect("call resize before attachments").1,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &self.reveal.as_ref().expect("call resize before attachments").1,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
            }),
        ]
    }

    /// Composites accumulated OIT transparency over `scene_view` (loading existing opaque pixels).
    pub fn composite(&self, encoder: &mut wgpu::CommandEncoder, scene_view: &wgpu::TextureView) {
        let Some(bg) = &self.bind_group else { return };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("oit_composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: scene_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_wboit_produces_identical_pixels_regardless_of_draw_order() {
        let Some((device, queue)) = crate::testing::device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let quad_wgsl = format!(
            "{OIT_SHADER_HELPERS_WGSL}
            struct Quad {{ color: vec4f, depth: f32, _p0: f32, _p1: f32, _p2: f32 }};
            @group(0) @binding(0) var<uniform> q: Quad;
            @vertex fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {{
              let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
              let y = 1.0 - f32(idx & 2u) * 2.0;
              return vec4f(x, y, 0.5, 1.0);
            }}
            @fragment fn fs() -> OitOutput {{
              return oit_pack(q.color.rgb, q.color.a, q.depth);
            }}"
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("oit_test_quad"),
            source: wgpu::ShaderSource::Wgsl(quad_wgsl.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(32),
                },
                count: None,
            }],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("oit_quad_pipe"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &oit_color_targets(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let make_quad = |rgba: [f32; 4], depth: f32| {
            let data = [rgba[0], rgba[1], rgba[2], rgba[3], depth, 0.0, 0.0, 0.0];
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&data));
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
            })
        };
        let red_near = make_quad([1.0, 0.1, 0.0, 0.5], 5.0);
        let blue_far = make_quad([0.0, 0.2, 1.0, 0.6], 20.0);

        let mut oit = OitPass::new(&device, wgpu::TextureFormat::Rgba8Unorm, None);
        oit.resize(&device, (8, 8));

        let render_order = |first: &wgpu::BindGroup, second: &wgpu::BindGroup| -> [u8; 4] {
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = target.create_view(&Default::default());
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let _clear = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
            }
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("oit_accum_pass"),
                    color_attachments: &oit.attachments(),
                    ..Default::default()
                });
                pass.set_pipeline(&pipe);
                pass.set_bind_group(0, first, &[]);
                pass.draw(0..3, 0..1);
                pass.set_bind_group(0, second, &[]);
                pass.draw(0..3, 0..1);
            }
            oit.composite(&mut enc, &view);

            let rb = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256 * 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            enc.copy_texture_to_buffer(
                target.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &rb,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(8) },
                },
                wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 },
            );
            queue.submit([enc.finish()]);
            let slice = rb.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range().unwrap();
            let px = [data[0], data[1], data[2], data[3]];
            drop(data);
            rb.unmap();
            px
        };

        let ab = render_order(&red_near, &blue_far);
        let ba = render_order(&blue_far, &red_near);
        assert_eq!(ab, ba, "WBOIT must be independent of draw submission order: {ab:?} vs {ba:?}");
        // Nearer red quad should dominate far blue quad due to depth weighting.
        assert!(ab[0] > ab[2], "nearer red quad should dominate far blue quad: {ab:?}");
    }
}
