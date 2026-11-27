#![cfg_attr(not(feature = "render-decals"), allow(dead_code))]
// Decals: a material's texture laid onto the surfaces inside a box. Each box
// is drawn from the inside, between the opaque surfaces and the translucent
// ones; every pixel it covers finds the surface the depth buffer holds there,
// and where that surface lies within the box, projected along the box's y
// axis and faded toward its ends, a lit material's texture darkens it and an
// unlit one's paints over it. Only surfaces whose material takes decals are
// touched: the scene pass marks them in the stencil (DECAL_MARK), and the
// decal pass tests for the mark.
use pen_protocol::RenderFrame;

use crate::built_in::{BuiltIn, BuiltIns};
use crate::matrix::{invert, multiply, narrow};
use crate::post::SCENE_FORMAT;
use crate::shader::PipelineContext;

/// The stencil bit of surfaces that take decals; bit 1 is the aberration's and
/// bit 4 the subsurface's (`subsurface::MARK`).
pub(crate) const DECAL_MARK: u32 = 2;
// Model and inverse matrices, then color times opacity, per decal.
const DECAL_FLOATS: usize = 36;

/// Standalone WGSL source for physically based deferred decals (albedo + tangent-space normal + ORM + emissive).
pub const PBR_DECAL_WGSL: &str = crate::wgsl!("decal_pbr.wgsl");

fn shader(samples: u32, paint: bool) -> String {
    crate::deferred_wgsl!("decals.wgsl")
        .replace("${depth_texture}", if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" })
        .replace("${paint}", if paint { "true" } else { "false" })
}

fn pbr_shader(samples: u32) -> String {
    let (depth_ty, sample_param, sample_arg) = if samples > 1 {
        ("texture_depth_multisampled_2d", ", @builtin(sample_index) sample_idx: u32", "i32(sample_idx)")
    } else {
        ("texture_depth_2d", "", "0")
    };
    crate::deferred_wgsl!("decal_pbr.wgsl")
        .replace("${depth_texture}", depth_ty)
        .replace("${sample_param}", sample_param)
        .replace("${sample_arg}", sample_arg)
}

/// Decals of one material and texture: `count` of them from `first` in the
/// drawing order.
pub(crate) struct DecalBatch {
    pub texture: wgpu::TextureView,
    /// Which texture view it is, to batch by.
    pub texture_id: u64,
    /// The material's slot, whose color it shares.
    pub material: usize,
    pub color: [f64; 4],
    /// Painted over the surface (an unlit material), not multiplied.
    pub paint: bool,
    pub first: u32,
    pub count: u32,
}

/// The pipeline drawing decals into a scene of `samples`: painted (an unlit
/// material's) or multiplied.
pub(crate) fn pipeline(context: &PipelineContext, samples: u32, paint: bool) -> wgpu::RenderPipeline {
    use wgpu::{BlendComponent, BlendFactor, BlendOperation};
    let device = &context.device;
    let label = if paint { "decals (paint)" } else { "decals (multiply)" };
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader(samples, paint).into()),
    });
    let group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("decals"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(144),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: samples > 1,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("decals"),
        bind_group_layouts: &[Some(&group_layout)],
        immediate_size: 0,
    });
    let keep = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Equal,
        pass_op: wgpu::StencilOperation::Keep,
        ..Default::default()
    };
    let color = if paint {
        BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        }
    } else {
        BlendComponent { src_factor: BlendFactor::Dst, dst_factor: BlendFactor::Zero, operation: BlendOperation::Add }
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        // The inside of the box, so it covers the surfaces even with the
        // camera within it.
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Front),
            front_face: wgpu::FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState { front: keep, back: keep, read_mask: DECAL_MARK, write_mask: 0 },
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

pub(crate) struct Decals {
    view_uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    capacity: usize,
    storage: Option<wgpu::Buffer>,
    data: Vec<f32>,
    bg_cache: Vec<(u32, bool, u64, wgpu::TextureView, wgpu::BindGroup)>,
}

impl Decals {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            view_uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("decal view"),
                size: 144,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            capacity: 0,
            storage: None,
            data: Vec::new(),
            bg_cache: Vec::new(),
        }
    }

    /// Draws `batches` of the frame's decals (by their indices in `order`)
    /// into `pass`, whose depth and stencil are read-only and whose depth
    /// `depth` also views, for a frame of `frame`'s view.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        built_ins: &mut BuiltIns,
        pass: &mut wgpu::RenderPass<'_>,
        frame: &RenderFrame,
        batches: &[DecalBatch],
        order: &[u32],
        depth: &wgpu::TextureView,
        samples: u32,
        size: (u32, u32),
    ) {
        let count = order.len();
        if count == 0 {
            return;
        }
        if count > self.capacity {
            self.capacity = count.next_power_of_two();
            self.storage = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("decals"),
                size: (self.capacity * DECAL_FLOATS * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.bg_cache.clear();
        }
        let data = &mut self.data;
        data.clear();
        for batch in batches {
            for &index in &order[batch.first as usize..(batch.first + batch.count) as usize] {
                let decal = &frame.decals[index as usize];
                data.extend_from_slice(&decal.transform);
                data.extend_from_slice(&invert(&decal.transform));
                let [red, green, blue, alpha] = batch.color;
                data.extend([red, green, blue, alpha * f64::from(decal.opacity)].map(narrow));
            }
        }
        let storage = self.storage.as_ref().unwrap();
        queue.write_buffer(storage, 0, bytemuck::cast_slice(data));
        let view_projection = multiply(&frame.projection, &frame.view);
        let inv_vp = invert(&view_projection);
        let mut view = [0.0f32; 36];
        view[..16].copy_from_slice(&view_projection);
        view[16..32].copy_from_slice(&inv_vp);
        view[32] = size.0 as f32;
        view[33] = size.1 as f32;
        queue.write_buffer(&self.view_uniform, 0, bytemuck::cast_slice(&view));
        if self.bg_cache.len() > 64 {
            self.bg_cache.clear();
        }
        pass.set_stencil_reference(DECAL_MARK);
        for batch in batches {
            let selected = built_ins.get(BuiltIn::Decal { samples, paint: batch.paint });
            pass.set_pipeline(selected);
            let cached_idx = self.bg_cache.iter().position(|(s, p, tid, cached_depth, _)| {
                *s == samples && *p == batch.paint && *tid == batch.texture_id && cached_depth == depth
            });
            let idx = if let Some(idx) = cached_idx {
                idx
            } else {
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &selected.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.view_uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: storage.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth) },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&batch.texture),
                        },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    ],
                });
                self.bg_cache.push((samples, batch.paint, batch.texture_id, depth.clone(), group));
                self.bg_cache.len() - 1
            };
            pass.set_bind_group(0, &self.bg_cache[idx].4, &[]);
            pass.draw(0..36, batch.first..batch.first + batch.count);
        }
    }
}

/// GPU-ready physically based deferred decal record matching `shaders/decal_pbr.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PbrDecal {
    pub model: [f32; 16],
    pub inverse: [f32; 16],
    /// `[r, g, b, opacity]` in linear space.
    pub albedo_opacity: [f32; 4],
    /// `[emissive_r, emissive_g, emissive_b, normal_strength]`.
    pub emissive_normal: [f32; 4],
    /// `[roughness_scale, metallic_scale, angle_fade_min_dot, y_fade_start]`.
    pub pbr_params: [f32; 4],
}

fn mat3_det(m: &[f32; 16]) -> f32 {
    let c0 = [m[0], m[1], m[2]];
    let c1 = [m[4], m[5], m[6]];
    let c2 = [m[8], m[9], m[10]];
    c0[0] * (c1[1] * c2[2] - c1[2] * c2[1]) - c1[0] * (c0[1] * c2[2] - c0[2] * c2[1])
        + c2[0] * (c0[1] * c1[2] - c0[2] * c1[1])
}

impl PbrDecal {
    /// Builds a `PbrDecal` from a box transform. If `model` has a negative 3x3 determinant
    /// (mirrored transform), its local Y axis is flipped so front-face culling still renders
    /// the interior of the unit cube while preserving `local.xz` UVs and `abs(local.y)` fade.
    pub fn from_transform(
        mut model: [f32; 16],
        albedo_opacity: [f32; 4],
        emissive: [f32; 3],
        normal_strength: f32,
        roughness_scale: f32,
        metallic_scale: f32,
    ) -> Self {
        if mat3_det(&model) < 0.0 {
            model[4] = -model[4];
            model[5] = -model[5];
            model[6] = -model[6];
        }
        Self {
            model,
            inverse: invert(&model),
            albedo_opacity,
            emissive_normal: [emissive[0], emissive[1], emissive[2], normal_strength.clamp(0.0, 4.0)],
            pbr_params: [roughness_scale.clamp(0.04, 1.0), metallic_scale.clamp(0.0, 1.0), 0.25, 0.6],
        }
    }
}

/// Scene view and lighting parameters for `PbrDecalPass`.
#[derive(Clone, Copy, Debug)]
pub struct PbrDecalView {
    pub view_projection: [f32; 16],
    pub camera_pos: [f32; 3],
    pub sun_direction: [f32; 3],
    pub sun_color: [f32; 3],
    pub sky_ambient: [f32; 3],
    pub ground_ambient: [f32; 3],
    pub ambient_intensity: f32,
    /// Clear depth value representing sky/background (1.0 for standard-Z, 0.0 for reversed-Z).
    pub depth_clear: f32,
    /// Optional sub-viewport `[x, y, width, height]`; `None` uses the full bound depth texture size.
    pub viewport: Option<[f32; 4]>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PbrViewUniform {
    view_projection: [f32; 16],
    inverse_view_projection: [f32; 16],
    camera_pos_and_clear: [f32; 4],
    sun_dir_and_unused: [f32; 4],
    sun_color_intensity: [f32; 4],
    sky_ambient: [f32; 4],
    ground_ambient: [f32; 4],
    viewport_size: [f32; 4],
}

fn pbr_bind_group_layout(device: &wgpu::Device, samples: u32, dynamic_uniform: bool) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pbr_decal_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: dynamic_uniform,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<PbrViewUniform>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: samples > 1,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
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
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn create_pbr_pipeline_raw(
    device: &wgpu::Device,
    samples: u32,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    use_stencil: bool,
    layout: Option<&wgpu::PipelineLayout>,
    cache: Option<&wgpu::PipelineCache>,
) -> wgpu::RenderPipeline {
    use wgpu::{BlendComponent, BlendFactor, BlendOperation};
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("decals_pbr"),
        source: wgpu::ShaderSource::Wgsl(pbr_shader(samples).into()),
    });
    let keep = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Equal,
        pass_op: wgpu::StencilOperation::Keep,
        ..Default::default()
    };
    let stencil = if use_stencil {
        wgpu::StencilState { front: keep, back: keep, read_mask: DECAL_MARK, write_mask: 0 }
    } else {
        wgpu::StencilState::default()
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("decals_pbr"),
        layout,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Front),
            front_face: wgpu::FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil,
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("pbr_fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::OneMinusSrcAlpha,
                        operation: BlendOperation::Add,
                    },
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
        cache,
    })
}

/// Standalone GPU pass for physically based deferred decals (albedo + tangent-space normal + ORM + emissive).
///
/// The render pass calling [`Self::draw`] must attach depth as read-only (`depth_ops: None`) and bind
/// a depth-only `TextureView` (`aspect: TextureAspect::DepthOnly` for combined depth-stencil formats).
/// Uses a dynamic uniform offset ring and appending storage buffer so multiple [`Self::draw`] calls
/// recorded in one submission never overwrite each other's inputs; call [`Self::begin_frame`] once
/// per submitted frame to reset ring cursors.
pub struct PbrDecalPass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    view_uniform: wgpu::Buffer,
    uniform_stride: u32,
    uniform_slot: usize,
    sampler: wgpu::Sampler,
    storage: wgpu::Buffer,
    capacity: usize,
    storage_cursor: usize,
    use_stencil: bool,
}

const PBR_UNIFORM_RING_SLOTS: usize = 64;

impl PbrDecalPass {
    pub fn new(
        device: &wgpu::Device,
        samples: u32,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        use_stencil: bool,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(matches!(samples, 1 | 4), "PbrDecalPass supports sample counts 1 or 4, got {samples}");
        anyhow::ensure!(
            !use_stencil || depth_format.has_stencil_aspect(),
            "PbrDecalPass use_stencil requires a stencil-capable depth format, got {depth_format:?}"
        );
        let bgl = pbr_bind_group_layout(device, samples, true);
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbr_decal_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline =
            create_pbr_pipeline_raw(device, samples, color_format, depth_format, use_stencil, Some(&pl), None);
        let align = device.limits().min_uniform_buffer_offset_alignment.max(256);
        let uniform_stride = u32::try_from(std::mem::size_of::<PbrViewUniform>())?.next_multiple_of(align);
        let view_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbr_decal_view_ring"),
            size: u64::from(uniform_stride) * PBR_UNIFORM_RING_SLOTS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("pbr_decal_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let capacity = 64;
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbr_decals"),
            size: (capacity * std::mem::size_of::<PbrDecal>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            pipeline,
            bgl,
            view_uniform,
            uniform_stride,
            uniform_slot: 0,
            sampler,
            storage,
            capacity,
            storage_cursor: 0,
            use_stencil,
        })
    }

    /// Resets per-frame ring cursors after the previous frame's commands have been submitted.
    pub fn begin_frame(&mut self) {
        self.uniform_slot = 0;
        self.storage_cursor = 0;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        view: &PbrDecalView,
        decals: &[PbrDecal],
        depth_view: &wgpu::TextureView,
        albedo_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        orm_view: &wgpu::TextureView,
    ) -> anyhow::Result<()> {
        if decals.is_empty() {
            return Ok(());
        }
        anyhow::ensure!(view.view_projection.iter().all(|v| v.is_finite()), "non-finite decal view_projection");
        let rec_bytes = std::mem::size_of::<PbrDecal>();
        let max_storage = device.limits().max_storage_buffer_binding_size;
        let needed_total =
            self.storage_cursor.checked_add(decals.len()).ok_or_else(|| anyhow::anyhow!("decal count overflow"))?;
        if needed_total > self.capacity {
            let new_cap = needed_total.next_power_of_two().max(64);
            let new_bytes = (new_cap as u64)
                .checked_mul(rec_bytes as u64)
                .ok_or_else(|| anyhow::anyhow!("decal storage size overflow"))?;
            anyhow::ensure!(
                new_bytes <= max_storage,
                "decal storage buffer {new_bytes} exceeds device limit {max_storage}"
            );
            // Allocate a fresh buffer so any prior draw in the same submission keeps its old buffer alive via its bind group.
            self.capacity = new_cap;
            self.storage = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbr_decals"),
                size: new_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.storage_cursor = 0;
        }
        let base_instance = self.storage_cursor;
        let byte_offset = (base_instance * rec_bytes) as u64;
        queue.write_buffer(&self.storage, byte_offset, bytemuck::cast_slice(decals));
        self.storage_cursor += decals.len();

        let slot = self.uniform_slot % PBR_UNIFORM_RING_SLOTS;
        self.uniform_slot = self.uniform_slot.wrapping_add(1);
        let dyn_offset = u32::try_from(slot)? * self.uniform_stride;
        let [vp_x, vp_y, vp_w, vp_h] = view.viewport.unwrap_or([0.0, 0.0, 0.0, 0.0]);
        let uniform = PbrViewUniform {
            view_projection: view.view_projection,
            inverse_view_projection: invert(&view.view_projection),
            camera_pos_and_clear: [view.camera_pos[0], view.camera_pos[1], view.camera_pos[2], view.depth_clear],
            sun_dir_and_unused: [view.sun_direction[0], view.sun_direction[1], view.sun_direction[2], 0.0],
            sun_color_intensity: [view.sun_color[0], view.sun_color[1], view.sun_color[2], view.ambient_intensity],
            sky_ambient: [view.sky_ambient[0], view.sky_ambient[1], view.sky_ambient[2], vp_x],
            ground_ambient: [view.ground_ambient[0], view.ground_ambient[1], view.ground_ambient[2], vp_y],
            viewport_size: [vp_w, vp_h, 0.0, 0.0],
        };
        queue.write_buffer(&self.view_uniform, u64::from(dyn_offset), bytemuck::bytes_of(&uniform));

        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pbr_decal_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.view_uniform,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<PbrViewUniform>() as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: self.storage.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(albedo_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(normal_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(orm_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        if self.use_stencil {
            pass.set_stencil_reference(DECAL_MARK);
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[dyn_offset]);
        let start = u32::try_from(base_instance)?;
        let end = u32::try_from(base_instance + decals.len())?;
        pass.draw(0..36, start..end);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upload_1x1_rgba8(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rgba: [u8; 4],
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("decal_1x1"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    #[test]
    fn gpu_pbr_deferred_decal_normal_uv_emissive_and_multi_draw_witness() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter required for PBR decal witness");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

        // Reject invalid stencil + depth-only combination; verify 4x MSAA stencil pipeline compiles cleanly.
        assert!(
            PbrDecalPass::new(&device, 1, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::Depth32Float, true)
                .is_err()
        );
        assert!(PbrDecalPass::new(&device, 4, SCENE_FORMAT, wgpu::TextureFormat::Depth24PlusStencil8, true).is_ok());

        let (w, h) = (32u32, 32u32);
        // Orthographic top-down camera looking straight down -Y at a flat floor at y = 0.
        let vp: [f32; 16] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -0.25, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0];
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pbr_decal_depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pbr_decal_color"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut pass_obj =
            PbrDecalPass::new(&device, 1, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::Depth32Float, false)
                .unwrap();
        let (_a_t, albedo_v) = upload_1x1_rgba8(&device, &queue, [220, 220, 220, 255]);
        let (_n_pos_u, n_pos_u_v) = upload_1x1_rgba8(&device, &queue, [240, 128, 180, 255]); // tilted +U (+X)
        let (_n_neg_u, n_neg_u_v) = upload_1x1_rgba8(&device, &queue, [15, 128, 180, 255]); // tilted -U (-X)
        let (_n_pos_v, n_pos_v_v) = upload_1x1_rgba8(&device, &queue, [128, 240, 180, 255]); // tilted +V (+Z)
        let (_n_neg_v, n_neg_v_v) = upload_1x1_rgba8(&device, &queue, [128, 15, 180, 255]); // tilted -V (-Z)
        let (_orm_t, orm_v) = upload_1x1_rgba8(&device, &queue, [255, 255, 255, 255]);

        let view_sun_x = PbrDecalView {
            view_projection: vp,
            camera_pos: [0.0, 4.0, 0.0],
            sun_direction: [0.85, 0.5268, 0.0], // Sun from +X (+U)
            sun_color: [2.2, 2.2, 2.2],
            sky_ambient: [0.08, 0.08, 0.08],
            ground_ambient: [0.02, 0.02, 0.02],
            ambient_intensity: 1.0,
            depth_clear: 1.0,
            viewport: None,
        };
        let view_sun_z = PbrDecalView { sun_direction: [0.0, 0.5268, 0.85], ..view_sun_x }; // Sun from +Z (+V)
        let box_model: [f32; 16] = [1.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 1.0];

        let run_decal =
            |p: &mut PbrDecalPass, view: &PbrDecalView, decal: PbrDecal, n_view: &wgpu::TextureView| -> [u8; 4] {
                p.begin_frame();
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                {
                    let _clear = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear_depth"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(0.5),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                }
                {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("pbr_decal_draw"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &color_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &depth_view,
                            depth_ops: None,
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    p.draw(&device, &queue, &mut rp, view, &[decal], &depth_view, &albedo_v, n_view, &orm_v).unwrap();
                }
                let bpr = (w * 4).next_multiple_of(256);
                let rb = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pbr_decal_rb"),
                    size: u64::from(bpr * h),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                enc.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &color_tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &rb,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(bpr),
                            rows_per_image: Some(h),
                        },
                    },
                    wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );
                queue.submit(Some(enc.finish()));
                let slice = rb.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                rx.recv().unwrap().unwrap();
                let bytes = slice.get_mapped_range().unwrap();
                let o = ((h / 2) * bpr + (w / 2) * 4) as usize;
                let px = [bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]];
                drop(bytes);
                rb.unmap();
                px
            };

        let lit_decal = PbrDecal::from_transform(box_model, [1.0, 1.0, 1.0, 1.0], [0.0; 3], 1.0, 0.7, 0.0);
        let px_u_pos = run_decal(&mut pass_obj, &view_sun_x, lit_decal, &n_pos_u_v);
        let px_u_neg = run_decal(&mut pass_obj, &view_sun_x, lit_decal, &n_neg_u_v);
        assert!(
            px_u_pos[0] > px_u_neg[0] + 40,
            "U-axis (+X) normal tilt toward +X sun must shade brighter: pos={px_u_pos:?} neg={px_u_neg:?}"
        );

        let px_v_pos = run_decal(&mut pass_obj, &view_sun_z, lit_decal, &n_pos_v_v);
        let px_v_neg = run_decal(&mut pass_obj, &view_sun_z, lit_decal, &n_neg_v_v);
        assert!(
            px_v_pos[0] > px_v_neg[0] + 40,
            "V-axis (+Z) normal tilt toward +Z sun must shade brighter: pos={px_v_pos:?} neg={px_v_neg:?}"
        );

        // Two distinct draws in ONE submission (left red emissive, right blue emissive) must not overwrite each other.
        pass_obj.begin_frame();
        let left_box: [f32; 16] = [0.8, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, -1.0, 0.0, 0.0, 1.0];
        let right_box: [f32; 16] = [0.8, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 1.0, 0.0, 0.0, 1.0];
        let d_red = PbrDecal::from_transform(left_box, [0.0, 0.0, 0.0, 1.0], [0.9, 0.0, 0.0], 1.0, 0.8, 0.0);
        let d_blue = PbrDecal::from_transform(right_box, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.9], 1.0, 0.8, 0.0);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let _clear = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear_depth_2"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.5), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pbr_decal_two_draws"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: None,
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass_obj
                .draw(&device, &queue, &mut rp, &view_sun_x, &[d_red], &depth_view, &albedo_v, &n_pos_u_v, &orm_v)
                .unwrap();
            pass_obj
                .draw(&device, &queue, &mut rp, &view_sun_z, &[d_blue], &depth_view, &albedo_v, &n_pos_v_v, &orm_v)
                .unwrap();
        }
        let bpr = (w * 4).next_multiple_of(256);
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbr_decal_rb_two"),
            size: u64::from(bpr * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &rb,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let bytes = slice.get_mapped_range().unwrap();
        let px_left = {
            let o = ((h / 2) * bpr + (w / 4) * 4) as usize;
            [bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]
        };
        let px_right = {
            let o = ((h / 2) * bpr + (3 * w / 4) * 4) as usize;
            [bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]
        };
        assert!(px_left[0] > 180 && px_left[2] < 30, "first draw must stay red: {px_left:?}");
        assert!(px_right[2] > 180 && px_right[0] < 30, "second draw must be blue: {px_right:?}");
        drop(bytes);
        rb.unmap();
    }
}
