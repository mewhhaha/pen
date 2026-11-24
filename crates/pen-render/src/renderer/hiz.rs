// Hierarchical-Z (Hi-Z) max-depth pyramid built after the opaque scene pass
// for conservative GPU occlusion culling on the next frame (shaders/cull.wgsl).
use std::collections::HashMap;

use pen_protocol::Matrix;

use super::timing::{GpuTimer, Span};
use crate::matrix::IDENTITY;
use crate::packed_texture::mip_levels;

const HIZ_BASE_WGSL: &str = include_str!("../shaders/hiz_base.wgsl");
const HIZ_REDUCE_WGSL: &str = include_str!("../shaders/hiz.wgsl");

const WORKGROUP: u32 = 8;

struct Pyramid {
    _texture: wgpu::Texture,
    size: (u32, u32),
    mips: u32,
    full_view: wgpu::TextureView,
    mip_views: Vec<wgpu::TextureView>,
    reduce_groups: Vec<wgpu::BindGroup>,
    view_id: u64,
}

pub(crate) struct HiZ {
    enabled: bool,
    valid: bool,
    vp: Matrix,
    fallback: wgpu::TextureView,
    base_pipelines: HashMap<u32, wgpu::ComputePipeline>,
    reduce_pipeline: Option<wgpu::ComputePipeline>,
    pyramid: Option<Pyramid>,
    base_group: Option<(u32, u64, wgpu::BindGroup)>,
    depth_serial: u64,
    next_id: u64,
}

impl HiZ {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let fallback_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hiz fallback"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            fallback_tex.as_image_copy(),
            &1.0f32.to_le_bytes(),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let fallback = fallback_tex.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            enabled: true,
            valid: false,
            vp: IDENTITY,
            fallback,
            base_pipelines: HashMap::new(),
            reduce_pipeline: None,
            pyramid: None,
            base_group: None,
            depth_serial: 0,
            next_id: 1,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.valid = false;
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Drops the prior-frame pyramid's validity (e.g., on camera cut or resize)
    /// so the next frame does not cull against stale depth.
    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Marks that the scene's depth texture was recreated so the base bind group rebuilds.
    pub fn reset_target(&mut self) {
        self.valid = false;
        self.base_group = None;
        self.depth_serial += 1;
    }

    /// Returns `(view, view_id, vp, size, mips, active)` for `GpuCulling::dispatch`.
    pub fn source(&self) -> (&wgpu::TextureView, u64, Matrix, (u32, u32), u32, bool) {
        if self.enabled
            && self.valid
            && let Some(pyramid) = &self.pyramid
        {
            return (&pyramid.full_view, pyramid.view_id, self.vp, pyramid.size, pyramid.mips, true);
        }
        (&self.fallback, 0, IDENTITY, (1, 1), 1, false)
    }

    /// Builds the max-depth pyramid from the opaque pass's depth texture.
    #[allow(clippy::too_many_arguments, reason = "explicit render-pass inputs")]
    pub fn build(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_only: &wgpu::TextureView,
        size: (u32, u32),
        samples: u32,
        vp: Matrix,
        cache: Option<&wgpu::PipelineCache>,
        timer: &GpuTimer,
    ) {
        if !self.enabled || size.0 == 0 || size.1 == 0 {
            self.valid = false;
            return;
        }
        let reduce_pipeline = self.reduce_pipeline.get_or_insert_with(|| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("hiz reduce"),
                source: wgpu::ShaderSource::Wgsl(HIZ_REDUCE_WGSL.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hiz reduce"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache,
            })
        });
        let base_pipeline = self.base_pipelines.entry(samples).or_insert_with(|| {
            let depth_ty = if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" };
            let source = HIZ_BASE_WGSL
                .replace("${depth_texture}", depth_ty)
                .replace("${samples}", &samples.to_string());
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("hiz base"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hiz base"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache,
            })
        });

        let needs_pyramid = self.pyramid.as_ref().is_none_or(|p| p.size != size);
        if needs_pyramid {
            let mips = mip_levels(size.0, size.1);
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("hiz pyramid"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: mips,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let full_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mip_views: Vec<_> = (0..mips)
                .map(|mip| {
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        base_mip_level: mip,
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect();
            let reduce_layout = reduce_pipeline.get_bind_group_layout(0);
            let reduce_groups: Vec<_> = (1..mips as usize)
                .map(|mip| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hiz reduce"),
                        layout: &reduce_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&mip_views[mip - 1]),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&mip_views[mip]),
                            },
                        ],
                    })
                })
                .collect();
            let view_id = self.next_id;
            self.next_id += 1;
            self.pyramid = Some(Pyramid {
                _texture: texture,
                size,
                mips,
                full_view,
                mip_views,
                reduce_groups,
                view_id,
            });
            self.base_group = None;
        }

        let pyramid = self.pyramid.as_ref().unwrap();
        let depth_serial = self.depth_serial;
        if self.base_group.as_ref().is_none_or(|(s, ser, _)| *s != samples || *ser != depth_serial) {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hiz base"),
                layout: &base_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_only) },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&pyramid.mip_views[0]),
                    },
                ],
            });
            self.base_group = Some((samples, depth_serial, group));
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hiz"),
            timestamp_writes: timer.compute(Span::Culling),
        });
        pass.set_pipeline(base_pipeline);
        pass.set_bind_group(0, &self.base_group.as_ref().unwrap().2, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(WORKGROUP), size.1.div_ceil(WORKGROUP), 1);

        pass.set_pipeline(reduce_pipeline);
        let (mut w, mut h) = size;
        for group in &pyramid.reduce_groups {
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(w.div_ceil(WORKGROUP), h.div_ceil(WORKGROUP), 1);
        }

        self.vp = vp;
        self.valid = true;
    }
}
