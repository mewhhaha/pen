//! Per-style screen-space silhouette outlines for native wgpu and WebGPU.
//!
//! Draw selected geometry into the mask, run the jump flood, then composite
//! over the scene. The final unit step reduces nearest-seed errors inherent
//! to jump flooding. Distances to the chosen seeds are Euclidean.

use pen_protocol::Matrix;
use wgpu::util::DeviceExt;

use crate::camera_precision::DepthMode;

pub const JFA_OUTLINE_WGSL: &str = include_str!("shaders/jfa_outline.wgsl");
pub const OUTLINE_STYLE_CAPACITY: usize = 256;
/// Marks a masked pixel as occluded. The low 16 bits hold the palette index plus
/// one. A mask value of zero denotes background.
pub const OUTLINE_OCCLUDED_BIT: u32 = 1 << 16;

const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MAX_RADIUS: u32 = 64;
const WORKGROUP_SIZE: u32 = 8;
const FLOOD_STEPS: [u32; 7] = [1, 2, 4, 8, 16, 32, MAX_RADIUS];

#[derive(Clone, Copy, Debug, Default)]
pub struct OutlineStyle {
    /// Straight-alpha color in the target's linear color space. RGB may exceed 1 for HDR.
    pub color: [f32; 4],
    pub occluded_color: [f32; 4],
    /// Exterior radius in target pixels. Palette upload clamps it to 0–64.
    pub width_px: f32,
    /// Half the width of the smooth transition centered at `width_px`. Zero gives a hard edge.
    pub feather_px: f32,
    /// Multiplies the style color's alpha on selected pixels.
    pub fill_alpha: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OutlineStyleGpu {
    color: [f32; 4],
    occluded_color: [f32; 4],
    width_px: f32,
    feather_px: f32,
    fill_alpha: f32,
    padding: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshDraw {
    model: Matrix,
    view_proj: Matrix,
    packed_style: u32,
    reversed_z: u32,
    depth_bias: f32,
    padding: u32,
}

pub struct MeshMaskParams<'a> {
    /// Byte stride of each vertex. Position is a `Float32x3` at offset zero.
    pub vertex_stride: u64,
    /// Column-major model transform.
    pub model: Matrix,
    /// Column-major view-projection transform in WebGPU clip space.
    pub view_proj: Matrix,
    pub style_index: usize,
    /// Must match the depth convention of the mask pass and the view-projection matrix.
    pub depth_mode: DepthMode,
    /// Optional single-sampled depth-only view with the same extent and viewport as the mask.
    pub scene_depth: Option<&'a wgpu::TextureView>,
    /// Nonnegative tolerance in device depth units for matching the scene surface.
    pub depth_bias: f32,
}

/// Immutable draw state. Prepare it before opening the mask pass.
pub struct PreparedMeshMask {
    pipeline: wgpu::RenderPipeline,
    bindings: wgpu::BindGroup,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SceneDepthKind {
    None,
    Depth2d,
    R32Float,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct MeshPipelineKey {
    stride: u64,
    depth_mode: DepthMode,
    scene_depth: SceneDepthKind,
    skinned: bool,
}

struct MaskTargets {
    mask: wgpu::Texture,
    mask_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    seeds: [wgpu::Buffer; 2],
    jumps: [wgpu::BindGroup; 2],
    step_stride: u32,
    init: wgpu::BindGroup,
}

pub struct JfaOutlinePass {
    shader: wgpu::ShaderModule,
    styles: wgpu::Buffer,
    radii: wgpu::Buffer,
    radius_stride: u32,
    init: wgpu::ComputePipeline,
    jump: wgpu::ComputePipeline,
    mesh_pipelines: Vec<(MeshPipelineKey, wgpu::RenderPipeline)>,
    composite_pipelines: Vec<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
    targets: Option<MaskTargets>,
    resolved: Option<(usize, wgpu::TextureView, u32)>,
}

impl JfaOutlinePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("JFA outline"),
            source: wgpu::ShaderSource::Wgsl(JFA_OUTLINE_WGSL.into()),
        });
        let styles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("JFA outline palette"),
            size: size_of::<[OutlineStyleGpu; OUTLINE_STYLE_CAPACITY]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let radius_stride = device.limits().min_uniform_buffer_offset_alignment.max(16);
        let mut radius_bytes = vec![0u8; radius_stride as usize * MAX_RADIUS as usize];
        for (r, slot) in (1u32..=MAX_RADIUS).zip(radius_bytes.chunks_exact_mut(radius_stride as usize)) {
            slot[..4].copy_from_slice(bytemuck::bytes_of(&r));
        }
        let radii = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JFA composite radii"),
            contents: &radius_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let init = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("JFA init seeds"),
            layout: None,
            module: &shader,
            entry_point: Some("init_seeds"),
            compilation_options: Default::default(),
            cache: None,
        });
        let jump_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("JFA jump bindings"),
            entries: &[
                compute_buffer_layout(3, wgpu::BufferBindingType::Storage { read_only: true }, false, 8),
                compute_buffer_layout(4, wgpu::BufferBindingType::Storage { read_only: false }, false, 8),
                compute_buffer_layout(5, wgpu::BufferBindingType::Uniform, true, 16),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("JFA jump layout"),
            bind_group_layouts: &[Some(&jump_layout)],
            immediate_size: 0,
        });
        let jump = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("JFA jump"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("jfa_step"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            shader,
            styles,
            radii,
            radius_stride,
            init,
            jump,
            mesh_pipelines: Vec::new(),
            composite_pipelines: Vec::new(),
            targets: None,
            resolved: None,
        }
    }

    /// Allocate for a nonzero render extent. Each seed buffer uses 8 bytes per
    /// pixel (packed 16-bit coordinates + style), so large targets use half the memory.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        assert!(width > 0 && height > 0, "the outline extent must be nonzero");
        if self.targets.as_ref().is_some_and(|targets| targets.mask.width() == width && targets.mask.height() == height)
        {
            return;
        }
        let limits = device.limits();
        assert!(
            width <= limits.max_texture_dimension_2d
                && height <= limits.max_texture_dimension_2d
                && width <= 65535
                && height <= 65535,
            "the outline extent exceeds the device's 2D texture or packed seed limit"
        );
        let seed_bytes = (u64::from(width) * u64::from(height) * 8).max(16);
        assert!(
            seed_bytes <= limits.max_storage_buffer_binding_size.min(limits.max_buffer_size),
            "each outline seed buffer requires {seed_bytes} bytes, exceeding the device's buffer limits"
        );
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let texture = |label, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let mask = texture(
            "JFA mask",
            MASK_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
        );
        let mask_view = mask.create_view(&Default::default());
        let depth_view = texture("JFA mask depth", DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT)
            .create_view(&Default::default());
        let seeds = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("JFA seeds"),
                size: seed_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        let init = self.init_bindings(device, &mask_view, &seeds[0]);

        // Immutable per-step params keep every dispatch independent of queue writes.
        let step_stride = limits.min_uniform_buffer_offset_alignment.max(16);
        let mut steps = vec![0; step_stride as usize * FLOOD_STEPS.len()];
        for (&step, slot) in FLOOD_STEPS.iter().zip(steps.chunks_exact_mut(step_stride as usize)) {
            slot[..16].copy_from_slice(bytemuck::bytes_of(&[width, height, step, 0]));
        }
        let step_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JFA jump parameters"),
            contents: &steps,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let jump_layout = self.jump.get_bind_group_layout(0);
        let jumps = std::array::from_fn(|input| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("JFA ping-pong bindings"),
                layout: &jump_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 3, resource: seeds[input].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: seeds[1 - input].as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &step_buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new(16),
                        }),
                    },
                ],
            })
        });
        self.targets = Some(MaskTargets { mask, mask_view, depth_view, seeds, jumps, step_stride, init });
        self.resolved = None;
    }

    /// Replace the palette. Omitted palette entries become transparent.
    pub fn set_styles(&mut self, queue: &wgpu::Queue, styles: &[OutlineStyle]) {
        queue.write_buffer(&self.styles, 0, bytemuck::cast_slice(&Self::pack_styles(styles)));
    }

    pub(crate) fn encode_styles(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        styles: &[OutlineStyle],
    ) {
        let upload = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JFA palette upload"),
            contents: bytemuck::cast_slice(&Self::pack_styles(styles)),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        encoder.copy_buffer_to_buffer(&upload, 0, &self.styles, 0, self.styles.size());
    }

    fn pack_styles(styles: &[OutlineStyle]) -> [OutlineStyleGpu; OUTLINE_STYLE_CAPACITY] {
        use bytemuck::Zeroable as _;

        assert!(styles.len() <= OUTLINE_STYLE_CAPACITY, "the outline palette supports up to 256 styles");
        let mut gpu_styles = [OutlineStyleGpu::zeroed(); OUTLINE_STYLE_CAPACITY];
        for (gpu, style) in gpu_styles.iter_mut().zip(styles) {
            *gpu = OutlineStyleGpu {
                color: style.color,
                occluded_color: style.occluded_color,
                width_px: style.width_px.clamp(0.0, 64.0),
                feather_px: style.feather_px.clamp(0.0, 64.0),
                fill_alpha: style.fill_alpha.clamp(0.0, 1.0),
                padding: 0.0,
            };
        }
        gpu_styles
    }

    /// The pass-owned `R32Uint` mask. Callers may render or copy packed mask values into it directly.
    pub fn mask_texture(&self) -> &wgpu::Texture {
        &self.targets().mask
    }

    /// Clear the mask and begin a Forward-Z mesh pass.
    pub fn begin_mask_pass<'a>(&'a self, encoder: &'a mut wgpu::CommandEncoder) -> wgpu::RenderPass<'a> {
        self.begin_mask_pass_with_depth_mode(encoder, DepthMode::ForwardZ)
    }

    pub fn begin_mask_pass_with_depth_mode<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        depth_mode: DepthMode,
    ) -> wgpu::RenderPass<'a> {
        let targets = self.targets();
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("JFA mesh mask"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.mask_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &targets.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(depth_mode.clear_value()),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        })
    }

    /// Prepare a draw with immutable transforms. The scene depth only tags
    /// occlusion. The mask pass's own depth attachment resolves selected-mesh overlap.
    pub fn prepare_mesh_mask(&mut self, device: &wgpu::Device, params: &MeshMaskParams<'_>) -> PreparedMeshMask {
        let kind = if params.scene_depth.is_some() { SceneDepthKind::Depth2d } else { SceneDepthKind::None };
        self.prepare_mesh_mask_inner(device, params, kind, None)
    }

    /// Variant of [`Self::prepare_mesh_mask`] accepting a single-sampled `R32Float` resolved depth view
    /// (e.g. from MSAA depth resolve) for occlusion tagging.
    pub fn prepare_mesh_mask_r32(&mut self, device: &wgpu::Device, params: &MeshMaskParams<'_>) -> PreparedMeshMask {
        let kind = if params.scene_depth.is_some() { SceneDepthKind::R32Float } else { SceneDepthKind::None };
        self.prepare_mesh_mask_inner(device, params, kind, None)
    }

    pub(crate) fn prepare_scene_mask(
        &mut self,
        device: &wgpu::Device,
        params: &MeshMaskParams<'_>,
        resolved: bool,
        skin: Option<(u32, &wgpu::Buffer, u64)>,
    ) -> PreparedMeshMask {
        let kind = if resolved { SceneDepthKind::R32Float } else { SceneDepthKind::Depth2d };
        self.prepare_mesh_mask_inner(device, params, kind, skin)
    }

    fn prepare_mesh_mask_inner(
        &mut self,
        device: &wgpu::Device,
        params: &MeshMaskParams<'_>,
        scene_depth_kind: SceneDepthKind,
        skin: Option<(u32, &wgpu::Buffer, u64)>,
    ) -> PreparedMeshMask {
        assert!(params.style_index < OUTLINE_STYLE_CAPACITY, "style index is outside the outline palette");
        assert!(
            params.vertex_stride >= 12
                && params.vertex_stride.is_multiple_of(4)
                && params.vertex_stride <= u64::from(device.limits().max_vertex_buffer_array_stride),
            "invalid mesh vertex stride"
        );
        let key = MeshPipelineKey {
            stride: params.vertex_stride,
            depth_mode: params.depth_mode,
            scene_depth: scene_depth_kind,
            skinned: skin.is_some(),
        };
        let pipeline = if let Some((_, pipeline)) = self.mesh_pipelines.iter().find(|(candidate, _)| *candidate == key)
        {
            pipeline.clone()
        } else {
            let pipeline = self.create_mesh_pipeline(device, key);
            self.mesh_pipelines.push((key, pipeline.clone()));
            pipeline
        };
        let draw = MeshDraw {
            model: params.model,
            view_proj: params.view_proj,
            packed_style: u32::try_from(params.style_index + 1).unwrap(),
            reversed_z: u32::from(params.depth_mode == DepthMode::ReversedZ),
            depth_bias: params.depth_bias.max(0.0),
            padding: 0,
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JFA mesh draw"),
            contents: bytemuck::bytes_of(&draw),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }];
        if let Some(depth) = params.scene_depth {
            let binding = match scene_depth_kind {
                SceneDepthKind::R32Float => 7,
                _ => 1,
            };
            entries.push(wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(depth) });
        }
        if let Some((offset, buffer, size)) = skin {
            entries.push(wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: u64::from(offset),
                    size: wgpu::BufferSize::new(size),
                }),
            });
        }
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JFA mesh draw bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        PreparedMeshMask { pipeline, bindings }
    }

    /// Encode a triangle-list draw. With an index buffer, `range` counts indices.
    /// Use the same depth mode when preparing every draw for a given mask pass.
    pub fn draw_mesh_mask(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        prepared: &PreparedMeshMask,
        vertices: wgpu::BufferSlice<'_>,
        indices: Option<(wgpu::BufferSlice<'_>, wgpu::IndexFormat)>,
        range: std::ops::Range<u32>,
    ) {
        pass.set_pipeline(&prepared.pipeline);
        pass.set_bind_group(0, &prepared.bindings, &[]);
        pass.set_vertex_buffer(0, vertices);
        if let Some((indices, format)) = indices {
            pass.set_index_buffer(indices, format);
            pass.draw_indexed(range, 0, 0..1);
        } else {
            pass.draw(range, 0..1);
        }
    }

    /// Initialize seeds from the pass-owned mask and encode the jump flood.
    /// Supply the ceiling of the widest style's width plus feather, clamped to
    /// 1..=64. The radius determines the jump schedule.
    pub fn run_jfa(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        max_radius_px: u32,
    ) {
        let targets = self.targets();
        let final_seed = self.encode_flood(encoder, &targets.init, max_radius_px);
        self.resolved = Some((final_seed, targets.mask_view.clone(), max_radius_px.clamp(1, MAX_RADIUS)));
    }

    /// Initialize from an external, single-sampled `R32Uint` 2D mask with
    /// `TEXTURE_BINDING` usage and the same extent as the pass.
    pub fn run_jfa_from_mask(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mask: &wgpu::Texture,
        max_radius_px: u32,
    ) {
        let targets = self.targets();
        assert_eq!(mask.size(), targets.mask.size(), "external mask extent must match the outline pass");
        assert_eq!(mask.format(), MASK_FORMAT, "external mask format must be R32Uint");
        assert_eq!(mask.dimension(), wgpu::TextureDimension::D2, "external mask must be 2D");
        assert_eq!(mask.sample_count(), 1, "external mask must be single-sampled");
        let view = mask.create_view(&wgpu::TextureViewDescriptor { mip_level_count: Some(1), ..Default::default() });
        let init = self.init_bindings(device, &view, &targets.seeds[0]);
        let final_seed = self.encode_flood(encoder, &init, max_radius_px);
        self.resolved = Some((final_seed, view, max_radius_px.clamp(1, MAX_RADIUS)));
    }

    /// Blend over a single-sampled, blendable color target with the mask's extent.
    pub fn composite(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
    ) {
        let (seed, mask, radius) = self.resolved.as_ref().expect("run the outline flood before compositing");
        let targets = self.targets.as_ref().expect("resize the outline pass before use");
        let pipeline =
            if let Some((_, pipeline)) = self.composite_pipelines.iter().find(|(format, _)| *format == target_format) {
                pipeline
            } else {
                let pipeline = self.create_composite_pipeline(device, target_format);
                self.composite_pipelines.push((target_format, pipeline));
                &self.composite_pipelines.last().unwrap().1
            };
        let radius_offset = u64::from((*radius).clamp(1, MAX_RADIUS) - 1) * u64::from(self.radius_stride);
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JFA composite bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(mask) },
                wgpu::BindGroupEntry { binding: 3, resource: targets.seeds[*seed].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.styles.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.radii,
                        offset: radius_offset,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("JFA composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.draw(0..3, 0..1);
    }

    fn targets(&self) -> &MaskTargets {
        self.targets.as_ref().expect("resize the outline pass before use")
    }

    fn init_bindings(&self, device: &wgpu::Device, mask: &wgpu::TextureView, seeds: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JFA init bindings"),
            layout: &self.init.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(mask) },
                wgpu::BindGroupEntry { binding: 4, resource: seeds.as_entire_binding() },
            ],
        })
    }

    fn encode_flood(&self, encoder: &mut wgpu::CommandEncoder, init: &wgpu::BindGroup, radius: u32) -> usize {
        let targets = self.targets();
        let x = targets.mask.width().div_ceil(WORKGROUP_SIZE);
        let y = targets.mask.height().div_ceil(WORKGROUP_SIZE);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("JFA outline flood"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.init);
        pass.set_bind_group(0, init, &[]);
        pass.dispatch_workgroups(x, y, 1);

        pass.set_pipeline(&self.jump);
        let largest_step_index = radius.clamp(1, MAX_RADIUS).next_power_of_two().ilog2();
        let mut input = 0;
        for step_index in (0..=largest_step_index).rev().chain(std::iter::once(0)) {
            pass.set_bind_group(0, &targets.jumps[input], &[step_index * targets.step_stride]);
            pass.dispatch_workgroups(x, y, 1);
            input = 1 - input;
        }
        input
    }

    fn create_mesh_pipeline(&self, device: &wgpu::Device, key: MeshPipelineKey) -> wgpu::RenderPipeline {
        let buffers = [
            Some(wgpu::VertexBufferLayout {
                array_stride: key.stride,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3],
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: crate::shader::SKIN_VERTEX_BYTES,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![9 => Uint16x4, 10 => Float32x4],
            }),
        ];
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("JFA mesh mask pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some(if key.skinned { "mask_skinned_vs" } else { "mask_vs" }),
                compilation_options: Default::default(),
                buffers: &buffers[..if key.skinned { 2 } else { 1 }],
            },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(key.depth_mode.compare_nearer_or_equal()),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some(match key.scene_depth {
                    SceneDepthKind::None => "mask_fs",
                    SceneDepthKind::Depth2d => "depth_mask_fs",
                    SceneDepthKind::R32Float => "depth_r32_mask_fs",
                }),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: MASK_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }

    fn create_composite_pipeline(&self, device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("JFA composite pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some("fullscreen_vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some("composite"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }
}

fn compute_buffer_layout(
    binding: u32,
    ty: wgpu::BufferBindingType,
    has_dynamic_offset: bool,
    min_binding_size: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset,
            min_binding_size: wgpu::BufferSize::new(min_binding_size),
        },
        count: None,
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    const WIDTH: u32 = 128;
    const HEIGHT: u32 = 64;
    const BACKGROUND: [u8; 4] = [8, 12, 16, 255];
    const IDENTITY: Matrix = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    const BOX_VERTICES: [[f32; 3]; 8] = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    const BOX_INDICES: [u16; 36] =
        [0, 2, 1, 0, 3, 2, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7, 4, 5, 6, 4, 6, 7];

    struct Frame {
        device: wgpu::Device,
        queue: wgpu::Queue,
        outline: JfaOutlinePass,
        encoder: wgpu::CommandEncoder,
        output: wgpu::Texture,
    }

    impl Frame {
        fn new(width: u32, height: u32, styles: &[OutlineStyle]) -> Self {
            let (device, queue) = crate::testing::device().expect("headless GPU adapter required for JFA witnesses");
            let mut outline = JfaOutlinePass::new(&device);
            outline.resize(&device, width, height);
            outline.set_styles(&queue, styles);
            let output = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("JFA witness color"),
                size: outline.mask_texture().size(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            let clear = BACKGROUND.map(|value| f64::from(value) / 255.0);
            drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("JFA witness clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output.create_view(&Default::default()),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: clear[0], g: clear[1], b: clear[2], a: clear[3] }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            }));
            Self { device, queue, outline, encoder, output }
        }

        fn draw_boxes(&mut self, boxes: &[(f32, usize)], mode: DepthMode, scene_depth: Option<&wgpu::TextureView>) {
            let vertex = |bytes| {
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("JFA witness vertices"),
                    contents: bytes,
                    usage: wgpu::BufferUsages::VERTEX,
                })
            };
            let (vertices, stride) = if boxes.len() > 1 {
                let packed = BOX_VERTICES.map(|[x, y, z]| [x, y, z, 0.0, 0.0, 1.0, 0.0, 0.0]);
                (vertex(bytemuck::cast_slice(&packed)), 32)
            } else {
                (vertex(bytemuck::cast_slice(&BOX_VERTICES)), 12)
            };
            let indices = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("JFA witness indices"),
                contents: bytemuck::cast_slice(&BOX_INDICES),
                usage: wgpu::BufferUsages::INDEX,
            });
            let mut view_proj = IDENTITY;
            if mode == DepthMode::ReversedZ {
                view_proj[10] = -1.0;
                view_proj[14] = 1.0;
            }
            let draws: Vec<_> = boxes
                .iter()
                .map(|&(x, style_index)| {
                    let mut model = IDENTITY;
                    model[0] = 0.25;
                    model[5] = 0.5;
                    model[10] = 0.1;
                    model[12] = x;
                    model[14] = 0.5;
                    self.outline.prepare_mesh_mask(
                        &self.device,
                        &MeshMaskParams {
                            vertex_stride: stride,
                            model,
                            view_proj,
                            style_index,
                            depth_mode: mode,
                            scene_depth,
                            depth_bias: 0.0001,
                        },
                    )
                })
                .collect();
            let mut pass = match mode {
                DepthMode::ForwardZ => self.outline.begin_mask_pass(&mut self.encoder),
                DepthMode::ReversedZ => self.outline.begin_mask_pass_with_depth_mode(&mut self.encoder, mode),
            };
            for draw in &draws {
                self.outline.draw_mesh_mask(
                    &mut pass,
                    draw,
                    vertices.slice(..),
                    Some((indices.slice(..), wgpu::IndexFormat::Uint16)),
                    0..36,
                );
            }
        }

        fn scene_depth(&mut self, mode: DepthMode) -> wgpu::TextureView {
            let depth = self
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("JFA witness scene depth"),
                    size: self.output.size(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: DEPTH_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default());
            let occluder = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("JFA witness occluder"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &self.outline.shader,
                    entry_point: Some("fullscreen_vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: None,
                multiview_mask: None,
                cache: None,
            });
            let mut pass = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("JFA witness occluder"),
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(mode.clear_value()),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&occluder);
            let near = if mode == DepthMode::ForwardZ { 0.3 } else { 0.7 };
            pass.set_viewport(0.0, 0.0, WIDTH as f32, HEIGHT as f32, near, near);
            pass.set_scissor_rect(0, 0, WIDTH / 2, HEIGHT);
            pass.draw(0..3, 0..1);
            // A surface between the front and back of the box tests nearest selected depth.
            pass.set_viewport(0.0, 0.0, WIDTH as f32, HEIGHT as f32, 0.5, 0.5);
            pass.set_scissor_rect(WIDTH / 2, 0, WIDTH / 2, HEIGHT);
            pass.draw(0..3, 0..1);
            drop(pass);
            depth
        }

        fn finish(mut self, radius: u32, external: Option<&wgpu::Texture>) -> impl Fn(u32, u32) -> [u8; 4] {
            if let Some(mask) = external {
                self.outline.run_jfa_from_mask(&self.device, &mut self.encoder, mask, radius);
            } else {
                self.outline.run_jfa(&self.device, &self.queue, &mut self.encoder, radius);
            }
            self.outline.composite(
                &self.device,
                &self.queue,
                &mut self.encoder,
                &self.output.create_view(&Default::default()),
                self.output.format(),
            );
            let stride = (self.output.width() * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
            let size = u64::from(stride) * u64::from(self.output.height());
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("JFA witness copy"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.encoder.copy_texture_to_buffer(
                self.output.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(stride),
                        rows_per_image: None,
                    },
                },
                self.output.size(),
            );
            self.queue.submit([self.encoder.finish()]);
            let bytes = crate::testing::read(&self.device, &self.queue, &buffer, size);
            move |x, y| {
                let offset = (y * stride + x * 4) as usize;
                bytes[offset..offset + 4].try_into().unwrap()
            }
        }
    }

    #[test]
    fn exterior_ring_has_euclidean_radius_and_feather() {
        let yellow =
            OutlineStyle { color: [1.0, 0.85, 0.1, 1.0], width_px: 6.0, feather_px: 1.0, ..Default::default() };
        let mut frame = Frame::new(WIDTH, HEIGHT, &[yellow]);
        frame.draw_boxes(&[(0.0, 0)], DepthMode::ForwardZ, None);
        let pixel = frame.finish(7, None);

        for y in 16..48 {
            for x in 48..80 {
                assert_eq!(pixel(x, y), BACKGROUND, "interior at ({x}, {y})");
            }
        }
        for distance in 1..=5 {
            let rgba = pixel(48 - distance, 32);
            assert!(rgba[0] == 255 && rgba[1].abs_diff(217) <= 1 && rgba[2].abs_diff(26) <= 1 && rgba[3] == 255);
        }
        let edge = pixel(42, 32);
        assert!((131..=132).contains(&edge[0]), "half-covered edge: {edge:?}");
        let diagonal = pixel(44, 12);
        let t = (32.0_f32.sqrt() - 5.0) / 2.0;
        let expected_red = (255.0 - 8.0) * (1.0 - t * t * (3.0 - 2.0 * t)) + 8.0;
        assert!((f32::from(diagonal[0]) - expected_red).abs() <= 1.0, "diagonal feather: {diagonal:?}");
        for distance in 9..=48 {
            assert_eq!(pixel(48 - distance, 32), BACKGROUND, "outside at distance {distance}");
        }
        eprintln!(
            "6px ring: interior {:?}, distances 1–5 {:?}, distance 6 {edge:?}, diagonal sqrt(32) {diagonal:?}, distance 9 {:?}",
            pixel(64, 32),
            (1..=5).map(|distance| pixel(48 - distance, 32)).collect::<Vec<_>>(),
            pixel(39, 32),
        );
    }

    #[test]
    fn skinned_mask_uses_the_prepared_joint_pose() {
        let red = OutlineStyle {
            color: [1.0, 0.0, 0.0, 1.0],
            occluded_color: [1.0, 0.0, 0.0, 1.0],
            fill_alpha: 1.0,
            ..Default::default()
        };
        let mut frame = Frame::new(WIDTH, HEIGHT, &[red]);
        let depth = frame.scene_depth(DepthMode::ForwardZ);
        let buffer = |contents, usage| {
            frame.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents, usage })
        };
        let positions = BOX_VERTICES.map(|[x, y, z]| [x * 0.25, y * 0.5, z * 0.1 + 0.5]);
        let vertices = buffer(bytemuck::cast_slice(&positions), wgpu::BufferUsages::VERTEX);
        let indices = buffer(bytemuck::cast_slice(&BOX_INDICES), wgpu::BufferUsages::INDEX);
        let skin = [[0_u32, 0, 1.0_f32.to_bits(), 0, 0, 0]; BOX_VERTICES.len()];
        let skin = buffer(bytemuck::cast_slice(&skin), wgpu::BufferUsages::VERTEX);
        let mut joint = IDENTITY;
        joint[12] = 0.5;
        let joints = buffer(bytemuck::cast_slice(&joint), wgpu::BufferUsages::STORAGE);
        let prepared = frame.outline.prepare_scene_mask(
            &frame.device,
            &MeshMaskParams {
                vertex_stride: 12,
                model: IDENTITY,
                view_proj: IDENTITY,
                style_index: 0,
                depth_mode: DepthMode::ForwardZ,
                scene_depth: Some(&depth),
                depth_bias: 1e-4,
            },
            false,
            Some((0, &joints, 64)),
        );
        {
            let mut pass = frame.outline.begin_mask_pass(&mut frame.encoder);
            pass.set_vertex_buffer(1, skin.slice(..));
            frame.outline.draw_mesh_mask(
                &mut pass,
                &prepared,
                vertices.slice(..),
                Some((indices.slice(..), wgpu::IndexFormat::Uint16)),
                0..36,
            );
        }
        let pixel = frame.finish(1, None);
        assert_eq!(pixel(64, 32), BACKGROUND);
        assert_eq!(pixel(96, 32), [255, 0, 0, 255]);
    }

    #[test]
    fn composite_cuts_off_wide_styles_at_the_requested_radius() {
        let red = OutlineStyle { color: [1.0, 0.0, 0.0, 1.0], width_px: 20.0, ..Default::default() };
        let mut frame = Frame::new(WIDTH, HEIGHT, &[red]);
        frame.draw_boxes(&[(0.0, 0)], DepthMode::ForwardZ, None);
        let pixel = frame.finish(5, None);
        assert_eq!(pixel(43, 32), [255, 0, 0, 255]);
        assert_eq!(pixel(42, 32), BACKGROUND);
        assert_eq!(pixel(44, 12), BACKGROUND);
    }

    #[test]
    fn two_boxes_keep_distinct_colors_and_a_clean_gap() {
        let cyan = OutlineStyle { color: [0.0, 1.0, 1.0, 1.0], width_px: 8.0, ..Default::default() };
        let magenta = OutlineStyle { color: [1.0, 0.0, 1.0, 1.0], ..cyan };
        let mut frame = Frame::new(WIDTH, HEIGHT, &[cyan, magenta]);
        frame.draw_boxes(&[(-0.5, 0), (0.5, 1)], DepthMode::ForwardZ, None);
        let pixel = frame.finish(8, None);

        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let in_ring = |left: u32, right: u32| {
                    let dx = x.abs_diff(x.clamp(left, right));
                    let dy = y.abs_diff(y.clamp(16, 47));
                    let distance_squared = dx * dx + dy * dy;
                    (1..=64).contains(&distance_squared)
                };
                let expected = if in_ring(16, 47) {
                    [0, 255, 255, 255]
                } else if in_ring(80, 111) {
                    [255, 0, 255, 255]
                } else {
                    BACKGROUND
                };
                assert_eq!(pixel(x, y), expected, "pixel at ({x}, {y})");
            }
        }
        eprintln!("two boxes: cyan {:?}, magenta {:?}, gap {:?}", pixel(53, 32), pixel(74, 32), pixel(64, 32));
    }

    #[test]
    fn scene_depth_tags_occluded_edges_in_both_depth_modes() {
        let style = OutlineStyle {
            color: [0.0, 1.0, 0.0, 1.0],
            occluded_color: [0.0, 0.25, 0.375, 1.0],
            width_px: 6.0,
            ..Default::default()
        };
        for mode in [DepthMode::ForwardZ, DepthMode::ReversedZ] {
            let mut frame = Frame::new(WIDTH, HEIGHT, &[style]);
            let depth = frame.scene_depth(mode);
            frame.draw_boxes(&[(0.0, 0)], mode, Some(&depth));
            let pixel = frame.finish(6, None);

            for y in 16..48 {
                assert_eq!(pixel(45, y), [0, 64, 96, 255], "{mode:?} occluded edge at y={y}");
                assert_eq!(pixel(82, y), [0, 255, 0, 255], "{mode:?} visible edge at y={y}");
                assert_eq!(pixel(64, y), BACKGROUND, "{mode:?} interior at y={y}");
            }
            eprintln!("{mode:?}: visible {:?}, occluded {:?}", pixel(82, 32), pixel(45, 32));
        }
    }

    #[test]
    fn external_mask_supports_wide_rings_and_stable_style_ties() {
        const W: u32 = 129;
        const H: u32 = 63;
        let mut styles = [OutlineStyle::default(); OUTLINE_STYLE_CAPACITY];
        styles[0] = OutlineStyle { color: [1.0, 0.0, 0.0, 1.0], width_px: 48.0, ..Default::default() };
        styles[255] = OutlineStyle { occluded_color: [0.0, 1.0, 1.0, 1.0], fill_alpha: 0.5, ..styles[0] };
        let frame = Frame::new(W, H, &styles);
        let mask = frame.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("JFA witness external mask"),
            size: frame.output.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: MASK_FORMAT,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mut packed = vec![0u32; (W * H) as usize];
        packed[(31 * W + 63) as usize] = 1;
        packed[(31 * W + 65) as usize] = 256 | OUTLINE_OCCLUDED_BIT;
        frame.queue.write_texture(
            mask.as_image_copy(),
            bytemuck::cast_slice(&packed),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: None },
            mask.size(),
        );

        let pixel = frame.finish(48, Some(&mask));
        assert_eq!(pixel(15, 31), [255, 0, 0, 255]);
        assert_eq!(pixel(113, 31), [0, 255, 255, 255]);
        assert_eq!(pixel(14, 31), BACKGROUND);
        assert_eq!(pixel(114, 31), BACKGROUND);
        assert_eq!(pixel(64, 31), [255, 0, 0, 255], "equal-distance tie favors the earlier seed");
        let fill = pixel(65, 31);
        for (actual, expected) in fill.into_iter().zip([4, 134, 136, 255]) {
            assert!(actual.abs_diff(expected) <= 1, "external mask fill {fill:?}");
        }
        eprintln!(
            "48px external mask: left {:?}, right {:?}, distance 49 {:?}, tie {:?}, fill {fill:?}",
            pixel(15, 31),
            pixel(113, 31),
            pixel(114, 31),
            pixel(64, 31)
        );
    }
}
