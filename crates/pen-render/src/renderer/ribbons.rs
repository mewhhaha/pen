#![cfg_attr(not(feature = "render-particles-cpu"), allow(dead_code, unused_imports))]
// World-space camera-facing ribbon and beam trails built from `frame.ribbons`
// and drawn in the translucent pass alongside particles.
use anyhow::Result;
use pen_protocol::{Catalog, RenderFrame, RibbonPoint};

use super::{GpuAsset, Scratch, State, uniform_buffer};
use crate::asset_store::AssetKind;
use crate::built_in::BuiltIn;
use crate::matrix::{invert, multiply, narrow};
use crate::post::SCENE_FORMAT;
use crate::shader::PipelineContext;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RibbonVertex {
    position: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
    soft_depth: f32,
}

const VERTEX_BYTES: u64 = size_of::<RibbonVertex>() as u64;

pub(crate) fn pipeline(context: &PipelineContext, samples: u32, additive: bool) -> wgpu::RenderPipeline {
    use wgpu::{BlendComponent, BlendFactor, BlendOperation};
    let device = &context.device;
    let code = crate::deferred_wgsl!("ribbons.wgsl")
        .replace("${translucency}", crate::volume::TRANSLUCENCY_WGSL)
        .replace("${additive}", if additive { "true" } else { "false" });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ribbons"),
        source: wgpu::ShaderSource::Wgsl(code.into()),
    });
    let color = BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: if additive { BlendFactor::One } else { BlendFactor::OneMinusSrcAlpha },
        operation: BlendOperation::Add,
    };
    let attrs = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 12, shader_location: 1 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 20, shader_location: 2 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 36, shader_location: 3 },
    ];
    let group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ribbons"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 15,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 16,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 17,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ribbons"),
        bind_group_layouts: &[Some(&group_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ribbons"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: VERTEX_BYTES,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &attrs,
            })],
        },
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: Some(false),
            depth_compare: Some(context.depth_mode().compare_nearer()),
            stencil: wgpu::StencilState { read_mask: 0, write_mask: 0, ..Default::default() },
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SCENE_FORMAT,
                blend: Some(wgpu::BlendState {
                    color,
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: context.cache.get(),
    })
}

pub(crate) struct Ready {
    pipeline: wgpu::RenderPipeline,
    shape: Option<wgpu::TextureView>,
    group: Option<wgpu::BindGroup>,
    first_vertex: u32,
    vertex_count: u32,
}

pub(crate) struct Ribbons {
    sampler: wgpu::Sampler,
    view: wgpu::Buffer,
    vertices: Vec<RibbonVertex>,
    vertex_buffer: Option<wgpu::Buffer>,
}

impl Ribbons {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }),
            view: uniform_buffer(device, "ribbon view", 128),
            vertices: Vec::new(),
            vertex_buffer: None,
        }
    }
}

pub(super) fn draw(pass: &mut wgpu::RenderPass<'_>, ribbons: Option<&Ribbons>, ready: &[Ready]) {
    let Some(ribbons) = ribbons else { return };
    let Some(vbuf) = &ribbons.vertex_buffer else { return };
    if ready.is_empty() {
        return;
    }
    pass.set_vertex_buffer(0, vbuf.slice(..));
    for batch in ready {
        pass.set_pipeline(&batch.pipeline);
        pass.set_bind_group(0, batch.group.as_ref().expect("ribbon bound before draw"), &[]);
        pass.draw(batch.first_vertex..batch.first_vertex + batch.vertex_count, 0..1);
    }
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-6 && len.is_finite() { [v[0] / len, v[1] / len, v[2] / len] } else { fallback }
}

fn emit_strip(points: &[RibbonPoint], camera_pos: [f32; 3], tint: [f32; 4], out: &mut Vec<RibbonVertex>) {
    let n = points.len();
    if n < 2 {
        return;
    }
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    let mut us = Vec::with_capacity(n);
    let mut total_len = 0.0f32;
    us.push(0.0);
    for i in 1..n {
        let a = points[i - 1].position;
        let b = points[i].position;
        let d = sub3([b.x, b.y, b.z], [a.x, a.y, a.z]);
        total_len += (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        us.push(total_len);
    }
    let inv_len = if total_len > 1e-6 { 1.0 / total_len } else { 0.0 };
    for (i, pt) in points.iter().enumerate() {
        us[i] *= inv_len;
        let p = [pt.position.x, pt.position.y, pt.position.z];
        let tangent = if i == 0 {
            let q = points[1].position;
            sub3([q.x, q.y, q.z], p)
        } else if i + 1 == n {
            let q = points[n - 2].position;
            sub3(p, [q.x, q.y, q.z])
        } else {
            let a = points[i - 1].position;
            let b = points[i + 1].position;
            sub3([b.x, b.y, b.z], [a.x, a.y, a.z])
        };
        let t_unit = unit_or(tangent, [0.0, 0.0, -1.0]);
        let to_cam = unit_or(sub3(camera_pos, p), [0.0, 1.0, 0.0]);
        let side = unit_or(cross3(t_unit, to_cam), [1.0, 0.0, 0.0]);
        let half_w = pt.width.max(0.0) * 0.5;
        left.push([p[0] - side[0] * half_w, p[1] - side[1] * half_w, p[2] - side[2] * half_w]);
        right.push([p[0] + side[0] * half_w, p[1] + side[1] * half_w, p[2] + side[2] * half_w]);
    }
    let mul_color = |c: &pen_protocol::Color| [c.r * tint[0], c.g * tint[1], c.b * tint[2], c.a * tint[3]];
    for i in 0..n - 1 {
        let c0 = mul_color(&points[i].color);
        let c1 = mul_color(&points[i + 1].color);
        let s0 = points[i].soft_depth.max(0.0);
        let s1 = points[i + 1].soft_depth.max(0.0);
        let v00 = RibbonVertex { position: left[i], uv: [us[i], 0.0], color: c0, soft_depth: s0 };
        let v01 = RibbonVertex { position: right[i], uv: [us[i], 1.0], color: c0, soft_depth: s0 };
        let v10 = RibbonVertex { position: left[i + 1], uv: [us[i + 1], 0.0], color: c1, soft_depth: s1 };
        let v11 = RibbonVertex { position: right[i + 1], uv: [us[i + 1], 1.0], color: c1, soft_depth: s1 };
        out.extend_from_slice(&[v00, v01, v11, v00, v11, v10]);
    }
}

impl State {
    pub(super) fn prepare_ribbons(
        &mut self,
        #[cfg_attr(not(feature = "render-particles-cpu"), allow(unused_variables))] frame: &RenderFrame,
        #[cfg_attr(not(feature = "render-particles-cpu"), allow(unused_variables))] catalog: &Catalog,
        #[cfg_attr(not(feature = "render-particles-cpu"), allow(unused_variables))] samples: u32,
        #[cfg_attr(not(feature = "render-particles-cpu"), allow(unused_variables))] scratch: &mut Scratch,
    ) -> Result<()> {
        #[cfg(not(feature = "render-particles-cpu"))]
        return Ok(());
        #[cfg(feature = "render-particles-cpu")]
        {
            if frame.ribbons.is_empty() || !crate::deferred_shaders::ready_all(&["ribbons.wgsl"]) {
                return Ok(());
            }
            let ribbons = self.ribbons.get_or_insert_with(|| Ribbons::new(&self.device));
            ribbons.vertices.clear();
            let inv_view = invert(&frame.view);
            let camera_pos = [inv_view[12], inv_view[13], inv_view[14]];

            let mut vertices = std::mem::take(&mut ribbons.vertices);
            let mut cursor = 0;
            while cursor < frame.ribbons.len() {
                let head = &frame.ribbons[cursor];
                let mut end = cursor + 1;
                while end < frame.ribbons.len()
                    && frame.ribbons[end].ribbon == head.ribbon
                    && frame.ribbons[end].material == head.material
                    && frame.ribbons[end].additive == head.additive
                {
                    end += 1;
                }
                let slice = &frame.ribbons[cursor..end];
                cursor = end;
                if slice.len() < 2 {
                    continue;
                }
                let Some(material) = self.material(catalog, head.material)? else { continue };
                let material = &material.material;
                let texture = match &material.texture {
                    None => None,
                    Some(key) => {
                        let slot = self.resolve(AssetKind::Texture, key)?;
                        match self.assets.current(slot) {
                            Some(GpuAsset::Texture(texture)) => Some(texture.view.clone()),
                            _ if self.assets.loading(slot) => continue,
                            _ => None,
                        }
                    }
                };
                let which = BuiltIn::Ribbons { samples, additive: head.additive };
                if !self.built_ins.ready([which]) {
                    self.built_ins.prepare([which]);
                    self.waiting = true;
                    continue;
                }
                let first_vertex = u32::try_from(vertices.len()).unwrap_or(0);
                emit_strip(slice, camera_pos, material.color.map(narrow), &mut vertices);
                let vertex_count = u32::try_from(vertices.len()).unwrap_or(first_vertex) - first_vertex;
                if vertex_count > 0 {
                    let pipeline = self.built_ins.get(which).clone();
                    scratch.ribbons.push(Ready { pipeline, shape: texture, group: None, first_vertex, vertex_count });
                }
            }
            let ribbons = self.ribbons.as_mut().unwrap();
            ribbons.vertices = vertices;
            if ribbons.vertices.is_empty() {
                return Ok(());
            }
            let bytes = ribbons.vertices.len() as u64 * VERTEX_BYTES;
            if ribbons.vertex_buffer.as_ref().is_none_or(|b| b.size() < bytes) {
                ribbons.vertex_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("ribbon vertices"),
                    size: bytes.next_power_of_two().max(256),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            self.queue.write_buffer(
                ribbons.vertex_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&ribbons.vertices),
            );
            Ok(())
        }
    }

    pub(super) fn bind_ribbons(
        &self,
        frame: &RenderFrame,
        projection: &pen_protocol::Matrix,
        opaque: Option<&crate::post::OpaqueTargets>,
        volume: &crate::volume::Applied,
        ready: &mut [Ready],
    ) {
        let Some(ribbons) = &self.ribbons else { return };
        if ready.is_empty() {
            return;
        }
        let mut view_data = [0.0f32; 32];
        view_data[..16].copy_from_slice(&multiply(projection, &frame.view));
        view_data[16..32].copy_from_slice(&invert(projection));
        self.queue.write_buffer(&ribbons.view, 0, bytemuck::cast_slice(&view_data));
        let depth_view = opaque.map_or(&self.white.linear, |o| &o.depth);
        for batch in ready {
            let shape = batch.shape.as_ref().unwrap_or(&self.white.view);
            batch.group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ribbons"),
                layout: &batch.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ribbons.view.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(shape) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&ribbons.sampler) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&ribbons.sampler) },
                    wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(depth_view) },
                    wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(&volume.texture) },
                    wgpu::BindGroupEntry { binding: 17, resource: volume.uniform.as_entire_binding() },
                ],
            }));
        }
    }
}
