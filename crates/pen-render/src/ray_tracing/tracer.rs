use anyhow::{Result, ensure};

use super::bvh::{BvhNodeGpu, BvhScene, BvhTriGpu, TlasScene, TlasSceneGpu};
use super::{BVH_RAYTRACE_WGSL, encode_copy};
#[cfg(not(target_arch = "wasm32"))]
use super::{TLAS_TRAVERSAL_HW_WGSL, device_supports_hardware_ray_query};

/// Maximum number of dynamic point, spot, and area lights injected into DDGI and RT bounce shading.
pub const MAX_RT_LOCAL_LIGHTS: usize = 64;

/// Compact GPU light record injected into DDGI probe hit shading and RT reflection/GI bounce hits.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RtLocalLight {
    /// xyz = world position, w = reach/range (> 0).
    pub position_range: [f32; 4],
    /// rgb = linear color * intensity, w = outer cone cosine (-1.0 for omnidirectional point light).
    pub color_outer_cos: [f32; 4],
    /// xyz = unit spot or area-light normal direction, w = inner cone cosine.
    pub direction_inner_cos: [f32; 4],
}

/// Ray-tracing pass parameters for sun shadows and hemispherical RTAO.
#[derive(Clone, Copy, Debug)]
pub struct RayTraceParams {
    pub view_projection: [f32; 16],
    pub camera_pos: [f32; 3],
    pub depth_clear: f32,
    pub sun_direction: [f32; 3],
    pub sun_angular_radius: f32,
    pub ao_radius: f32,
    pub ao_samples: u32,
    pub shadow_samples: u32,
    pub frame_seed: f32,
    pub normal_bias: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RayTraceUniform {
    inverse_view_projection: [f32; 16],
    camera_pos_and_clear: [f32; 4],
    sun_dir_and_radius: [f32; 4],
    ao_shadow: [f32; 4],
    bias_params: [f32; 4],
    counts: [u32; 4],
}

/// Portable GPU compute BVH ray tracer outputting `(rt_shadow, rtao, hit_dist, 1.0)` in `Rgba16Float`.
pub struct GpuRayTracer {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    nodes_buf: Option<(u32, wgpu::Buffer)>,
    tris_buf: Option<(u32, wgpu::Buffer)>,
    output: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
    bg: Option<(wgpu::Buffer, wgpu::Buffer, wgpu::TextureView, wgpu::TextureView, wgpu::BindGroup)>,
}

impl GpuRayTracer {
    pub fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bvh_rt_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<RayTraceUniform>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bvh_raytrace"),
            source: wgpu::ShaderSource::Wgsl(BVH_RAYTRACE_WGSL.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bvh_rt_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bvh_rt_pipe"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("raytrace_shadows_ao"),
            compilation_options: Default::default(),
            cache: crate::pipeline_cache::active().as_ref(),
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bvh_rt_u"),
            size: std::mem::size_of::<RayTraceUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, bgl, uniform, nodes_buf: None, tris_buf: None, output: None, bg: None }
    }

    pub fn upload_bvh(&mut self, device: &wgpu::Device, scene: &BvhScene) -> Result<()> {
        let max_storage = device.limits().max_storage_buffer_binding_size;
        let nodes = if scene.nodes().is_empty() {
            vec![BvhNodeGpu { aabb_min: [0.0; 3], left_or_first: 0, aabb_max: [0.0; 3], tri_count: 0 }]
        } else {
            scene.nodes().to_vec()
        };
        let tris = if scene.tris().is_empty() {
            vec![BvhTriGpu { v0: [0.0; 4], v1: [0.0; 4], v2: [0.0; 4] }]
        } else {
            scene.tris().to_vec()
        };
        let n_bytes = std::mem::size_of_val(nodes.as_slice()) as u64;
        let t_bytes = std::mem::size_of_val(tris.as_slice()) as u64;
        ensure!(n_bytes <= max_storage && t_bytes <= max_storage, "BVH buffer exceeds device storage limit");

        let nb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bvh_nodes"),
            size: n_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        nb.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytemuck::cast_slice(&nodes));
        nb.unmap();

        let tb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bvh_tris"),
            size: t_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        tb.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytemuck::cast_slice(&tris));
        tb.unmap();

        self.nodes_buf = Some((scene.nodes().len() as u32, nb));
        self.tris_buf = Some((scene.tris().len() as u32, tb));
        self.bg = None;
        Ok(())
    }

    pub fn output_texture(&self) -> Option<&wgpu::Texture> {
        self.output.as_ref().map(|o| &o.2)
    }

    pub fn output_view(&self) -> Option<&wgpu::TextureView> {
        self.output.as_ref().map(|o| &o.3)
    }

    pub fn trace(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        size: (u32, u32),
        params: &RayTraceParams,
    ) -> Result<()> {
        let max_dim = device.limits().max_texture_dimension_2d;
        ensure!(size.0 >= 1 && size.1 >= 1 && size.0 <= max_dim && size.1 <= max_dim, "invalid raytrace size {size:?}");
        let d_sz = depth_view.texture().size();
        ensure!(
            (d_sz.width, d_sz.height) == size,
            "GpuRayTracer depth_view extent {}x{} does not match requested {size:?}",
            d_sz.width,
            d_sz.height
        );
        ensure!(params.view_projection.iter().all(|v| v.is_finite()), "non-finite raytrace view_projection");
        let (node_count, nodes_buf) =
            self.nodes_buf.as_ref().ok_or_else(|| anyhow::anyhow!("call upload_bvh before trace"))?;
        let (tri_count, tris_buf) = self.tris_buf.as_ref().unwrap();

        if !matches!(&self.output, Some((w, h, _, _)) if *w == size.0 && *h == size.1) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("bvh_rt_output"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.output = Some((size.0, size.1, tex, view));
        }

        let u = RayTraceUniform {
            inverse_view_projection: crate::matrix::invert(&params.view_projection),
            camera_pos_and_clear: [
                params.camera_pos[0],
                params.camera_pos[1],
                params.camera_pos[2],
                params.depth_clear,
            ],
            sun_dir_and_radius: [
                params.sun_direction[0],
                params.sun_direction[1],
                params.sun_direction[2],
                params.sun_angular_radius.clamp(0.0, 0.2),
            ],
            ao_shadow: [
                params.ao_radius.max(0.05),
                params.ao_samples.min(16) as f32,
                params.shadow_samples.clamp(1, 8) as f32,
                params.frame_seed,
            ],
            bias_params: [params.normal_bias.max(1e-5), 0.0, 0.0, 0.0],
            counts: [*node_count, *tri_count, 0, 0],
        };
        encode_copy(device, encoder, &self.uniform, bytemuck::bytes_of(&u));
        let (_, _, _, out_view) = self.output.as_ref().unwrap();
        let hit = matches!(
            &self.bg,
            Some((nb, tb, dv, ov, _)) if nb == nodes_buf && tb == tris_buf && dv == depth_view && ov == out_view
        );
        if !hit {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bvh_rt_bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: nodes_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: tris_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(out_view) },
                ],
            });
            self.bg = Some((nodes_buf.clone(), tris_buf.clone(), depth_view.clone(), out_view.clone(), bg));
        }
        let (_, _, _, _, bg) = self.bg.as_ref().unwrap();
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("bvh_raytrace"), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        Ok(())
    }
}

/// Parameters for [`TlasRayTracer`] covering soft shadows, RTAO, GGX specular reflections, and 1-bounce diffuse GI.
#[derive(Clone, Copy, Debug)]
pub struct TlasRayTraceParams {
    pub view_projection: [f32; 16],
    pub camera_pos: [f32; 3],
    pub depth_clear: f32,
    pub sun_direction: [f32; 3],
    pub sun_angular_radius: f32,
    pub sun_color: [f32; 3],
    pub sky_zenith: [f32; 3],
    pub sky_horizon: [f32; 3],
    pub surface_roughness: f32,
    pub ao_radius: f32,
    pub max_ray_distance: f32,
    pub shadow_samples: u32,
    pub ao_samples: u32,
    pub reflection_samples: u32,
    pub gi_samples: u32,
    pub gi_bounce_scale: f32,
    pub frame_seed: f32,
    pub normal_bias: f32,
    pub visibility_mask: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TlasRayTraceUniform {
    inverse_view_projection: [f32; 16],
    camera_pos_and_clear: [f32; 4],
    sun_dir_and_radius: [f32; 4],
    sun_color_and_rough: [f32; 4],
    sky_zenith: [f32; 4],
    sky_horizon: [f32; 4],
    samples_seed: [f32; 4],
    bias_and_refl: [f32; 4],
    counts: [u32; 4],
    extra_counts: [u32; 4],
}

struct RtOutputs {
    size: (u32, u32),
    shadow_ao: (wgpu::Texture, wgpu::TextureView),
    reflection: (wgpu::Texture, wgpu::TextureView),
    gi: (wgpu::Texture, wgpu::TextureView),
}

impl RtOutputs {
    fn new(device: &wgpu::Device, size: (u32, u32)) -> Self {
        let make = |label: &str| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        Self {
            size,
            shadow_ao: make("tlas_rt_shadow_ao"),
            reflection: make("tlas_rt_reflection"),
            gi: make("tlas_rt_gi"),
        }
    }
}

#[derive(PartialEq)]
struct TlasBgKey {
    hw_slots: Option<usize>,
    tlas_nodes: wgpu::Buffer,
    tlas_instances: wgpu::Buffer,
    blas_nodes: wgpu::Buffer,
    blas_tris: wgpu::Buffer,
    depth: wgpu::TextureView,
    out_sao: wgpu::TextureView,
    ddgi: wgpu::Buffer,
    gbuf_nr: wgpu::TextureView,
    gbuf_am: wgpu::TextureView,
    sharc_keys: wgpu::Buffer,
}

/// Portable 2-level TLAS/BLAS GPU ray tracer (native hardware `wgpu_ray_query` when supported,
/// portable compute BVH on WebGPU/WASM) producing shadows/RTAO, GGX specular reflections,
/// and 1-bounce diffuse GI in a single pass.
pub struct TlasRayTracer {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    #[cfg(not(target_arch = "wasm32"))]
    pipeline_hw: Option<wgpu::ComputePipeline>,
    #[cfg(not(target_arch = "wasm32"))]
    bgl_hw: Option<wgpu::BindGroupLayout>,
    uniform: wgpu::Buffer,
    local_lights_buf: wgpu::Buffer,
    ddgi_dummy_buf: wgpu::Buffer,
    sharc_dummy: crate::sharc::SharcCache,
    gbuf_dummy_tex: (wgpu::Texture, wgpu::TextureView),
    scene: Option<TlasSceneGpu>,
    outputs: Option<RtOutputs>,
    bg: Option<(TlasBgKey, wgpu::BindGroup)>,
}

impl TlasRayTracer {
    #[cfg_attr(not(feature = "render-rt"), allow(dead_code))]
    pub(crate) fn supported(device: &wgpu::Device) -> bool {
        device.limits().max_storage_buffers_per_shader_stage >= 8
    }

    pub fn new(device: &wgpu::Device) -> Self {
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let out_wo = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let tex_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let u_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<TlasRayTraceUniform>() as u64),
            },
            count: None,
        };
        let sharc_u_entry = wgpu::BindGroupLayoutEntry {
            binding: 15,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tlas_rt_bgl"),
            entries: &[
                u_entry,
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_ro(4),
                tex_ro(5),
                out_wo(6),
                out_wo(7),
                out_wo(8),
                storage_ro(9),
                storage_ro(10),
                tex_ro(11),
                tex_ro(12),
                storage_ro(13),
                storage_ro(14),
                sharc_u_entry,
            ],
        });
        let sw_body = crate::deferred_wgsl!("tlas_raytrace.wgsl")
            .replace("// __TLAS_BINDING_1__", "@group(0) @binding(1) var<storage, read> tlas_nodes: array<BvhNode>;");
        let sharc_hash = crate::sharc::SHARC_HASH_WGSL;
        let sharc_ro = crate::sharc::SHARC_QUERY_RO_FN_WGSL;
        let source = format!("{}\n{sharc_hash}\n{sharc_ro}\n{sw_body}", crate::deferred_wgsl!("tlas_traversal.wgsl"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tlas_raytrace"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tlas_rt_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_rt_pipe"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("trace_scene"),
            compilation_options: Default::default(),
            cache: crate::pipeline_cache::active().as_ref(),
        });

        #[cfg(not(target_arch = "wasm32"))]
        let (pipeline_hw, bgl_hw) = if device_supports_hardware_ray_query(device) {
            let bgl_hw = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tlas_rt_bgl_hw"),
                entries: &[
                    u_entry,
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                        count: None,
                    },
                    storage_ro(2),
                    storage_ro(3),
                    storage_ro(4),
                    tex_ro(5),
                    out_wo(6),
                    out_wo(7),
                    out_wo(8),
                    storage_ro(9),
                    storage_ro(10),
                    tex_ro(11),
                    tex_ro(12),
                    storage_ro(13),
                    storage_ro(14),
                    sharc_u_entry,
                ],
            });
            let hw_body = crate::deferred_wgsl!("tlas_raytrace.wgsl")
                .replace("// __TLAS_BINDING_1__", "@group(0) @binding(1) var hw_tlas: acceleration_structure;");
            let hw_src = format!("{TLAS_TRAVERSAL_HW_WGSL}\n{sharc_hash}\n{sharc_ro}\n{hw_body}");
            let hw_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("tlas_raytrace_hw"),
                source: wgpu::ShaderSource::Wgsl(hw_src.into()),
            });
            let pl_hw = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tlas_rt_pl_hw"),
                bind_group_layouts: &[Some(&bgl_hw)],
                immediate_size: 0,
            });
            let p_hw = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tlas_rt_pipe_hw"),
                layout: Some(&pl_hw),
                module: &hw_mod,
                entry_point: Some("trace_scene"),
                compilation_options: Default::default(),
                cache: crate::pipeline_cache::active().as_ref(),
            });
            (Some(p_hw), Some(bgl_hw))
        } else {
            (None, None)
        };

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_rt_u"),
            size: std::mem::size_of::<TlasRayTraceUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let local_lights_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_rt_local_lights"),
            size: (MAX_RT_LOCAL_LIGHTS * std::mem::size_of::<RtLocalLight>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ddgi_dummy_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_rt_ddgi_dummy"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let sharc_dummy = crate::sharc::SharcCache::new_disabled(device).expect("dummy SHaRC cache");
        let gbuf_dummy_tex = {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("tlas_rt_gbuf_dummy"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        Self {
            pipeline,
            bgl,
            #[cfg(not(target_arch = "wasm32"))]
            pipeline_hw,
            #[cfg(not(target_arch = "wasm32"))]
            bgl_hw,
            uniform,
            local_lights_buf,
            ddgi_dummy_buf,
            sharc_dummy,
            gbuf_dummy_tex,
            scene: None,
            outputs: None,
            bg: None,
        }
    }

    pub fn uses_hardware_ray_query(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.pipeline_hw.is_some()
        }
        #[cfg(target_arch = "wasm32")]
        {
            false
        }
    }

    pub fn upload_scene(&mut self, device: &wgpu::Device, scene: &TlasScene) -> Result<()> {
        self.scene = Some(TlasSceneGpu::upload(device, scene)?);
        Ok(())
    }

    pub fn scene_gpu(&self) -> Option<&TlasSceneGpu> {
        self.scene.as_ref()
    }

    pub fn shadow_ao_texture(&self) -> Option<&wgpu::Texture> {
        self.outputs.as_ref().map(|o| &o.shadow_ao.0)
    }

    pub fn shadow_ao_view(&self) -> Option<&wgpu::TextureView> {
        self.outputs.as_ref().map(|o| &o.shadow_ao.1)
    }

    pub fn reflection_texture(&self) -> Option<&wgpu::Texture> {
        self.outputs.as_ref().map(|o| &o.reflection.0)
    }

    pub fn reflection_view(&self) -> Option<&wgpu::TextureView> {
        self.outputs.as_ref().map(|o| &o.reflection.1)
    }

    pub fn gi_texture(&self) -> Option<&wgpu::Texture> {
        self.outputs.as_ref().map(|o| &o.gi.0)
    }

    pub fn gi_view(&self) -> Option<&wgpu::TextureView> {
        self.outputs.as_ref().map(|o| &o.gi.1)
    }

    pub fn trace(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        size: (u32, u32),
        params: &TlasRayTraceParams,
    ) -> Result<()> {
        let scene = self.scene.as_ref().ok_or_else(|| anyhow::anyhow!("call upload_scene before trace"))?.clone();
        self.trace_with_scene_gi(device, encoder, &scene, depth_view, size, params, &[], None, false)
    }

    pub fn trace_with_scene(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &TlasSceneGpu,
        depth_view: &wgpu::TextureView,
        size: (u32, u32),
        params: &TlasRayTraceParams,
    ) -> Result<()> {
        self.trace_with_scene_gi(device, encoder, scene, depth_view, size, params, &[], None, false)
    }

    /// Traces shadows/AO, GGX reflections, and optional 1-bounce diffuse GI at `out_size` (full-res or half-res)
    /// against `depth_view`, injecting `local_lights`, optional live packed DDGI bounce irradiance, and
    /// optional primary TLAS material lookup (`use_material_gbuf`).
    #[allow(clippy::too_many_arguments)]
    pub fn trace_with_scene_gi(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &TlasSceneGpu,
        depth_view: &wgpu::TextureView,
        out_size: (u32, u32),
        params: &TlasRayTraceParams,
        local_lights: &[RtLocalLight],
        ddgi_packed_buf: Option<&wgpu::Buffer>,
        use_material_gbuf: bool,
    ) -> Result<()> {
        self.trace_with_scene_gbuffer(
            device,
            encoder,
            scene,
            depth_view,
            out_size,
            params,
            local_lights,
            ddgi_packed_buf,
            use_material_gbuf,
            None,
        )
    }

    /// Like [`Self::trace_with_scene_gi`], additionally accepting optional Thin G-Buffer views
    /// `(normal_roughness, albedo_metallic)` to supply exact smooth vertex normals and per-pixel
    /// material parameters with zero primary TLAS material-lookup rays.
    #[allow(clippy::too_many_arguments)]
    pub fn trace_with_scene_gbuffer(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &TlasSceneGpu,
        depth_view: &wgpu::TextureView,
        out_size: (u32, u32),
        params: &TlasRayTraceParams,
        local_lights: &[RtLocalLight],
        ddgi_packed_buf: Option<&wgpu::Buffer>,
        use_material_gbuf: bool,
        thin_gbuffer: Option<(&wgpu::TextureView, &wgpu::TextureView)>,
    ) -> Result<()> {
        self.trace_with_scene_gbuffer_sharc(
            device,
            encoder,
            scene,
            depth_view,
            out_size,
            params,
            local_lights,
            ddgi_packed_buf,
            use_material_gbuf,
            thin_gbuffer,
            None,
        )
    }

    /// Like [`Self::trace_with_scene_gbuffer`], additionally binding an optional world-space
    /// [`crate::sharc::SharcCache`] for unbounded multi-bounce diffuse lookup at reflection/GI hits.
    #[allow(clippy::too_many_arguments)]
    pub fn trace_with_scene_gbuffer_sharc(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &TlasSceneGpu,
        depth_view: &wgpu::TextureView,
        out_size: (u32, u32),
        params: &TlasRayTraceParams,
        local_lights: &[RtLocalLight],
        ddgi_packed_buf: Option<&wgpu::Buffer>,
        use_material_gbuf: bool,
        thin_gbuffer: Option<(&wgpu::TextureView, &wgpu::TextureView)>,
        sharc: Option<&crate::sharc::SharcCache>,
    ) -> Result<()> {
        let max_dim = device.limits().max_texture_dimension_2d;
        ensure!(
            out_size.0 >= 1 && out_size.1 >= 1 && out_size.0 <= max_dim && out_size.1 <= max_dim,
            "invalid RT size {out_size:?}"
        );
        let d_sz = depth_view.texture().size();
        ensure!(
            d_sz.width >= out_size.0 && d_sz.height >= out_size.1,
            "TlasRayTracer depth_view {}x{} smaller than output {out_size:?}",
            d_sz.width,
            d_sz.height
        );
        if let Some((nr, am)) = thin_gbuffer {
            let nr_sz = nr.texture().size();
            let am_sz = am.texture().size();
            ensure!(
                nr_sz.width >= out_size.0
                    && nr_sz.height >= out_size.1
                    && am_sz.width == nr_sz.width
                    && am_sz.height == nr_sz.height,
                "TlasRayTracer thin G-buffer dimensions mismatch output {out_size:?}"
            );
        }
        ensure!(params.view_projection.iter().all(|v| v.is_finite()), "non-finite view_projection");

        if !matches!(&self.outputs, Some(o) if o.size == out_size) {
            self.outputs = Some(RtOutputs::new(device, out_size));
            self.bg = None;
        }
        let light_count = local_lights.len().min(MAX_RT_LOCAL_LIGHTS);
        let flags = u32::from(use_material_gbuf)
            | (u32::from(ddgi_packed_buf.is_some()) << 1)
            | (u32::from(thin_gbuffer.is_some()) << 2);
        let u = TlasRayTraceUniform {
            inverse_view_projection: crate::matrix::invert(&params.view_projection),
            camera_pos_and_clear: [
                params.camera_pos[0],
                params.camera_pos[1],
                params.camera_pos[2],
                params.depth_clear,
            ],
            sun_dir_and_radius: [
                params.sun_direction[0],
                params.sun_direction[1],
                params.sun_direction[2],
                params.sun_angular_radius.clamp(0.0, 0.2),
            ],
            sun_color_and_rough: [
                params.sun_color[0].max(0.0),
                params.sun_color[1].max(0.0),
                params.sun_color[2].max(0.0),
                params.surface_roughness.clamp(0.0, 1.0),
            ],
            sky_zenith: [
                params.sky_zenith[0].max(0.0),
                params.sky_zenith[1].max(0.0),
                params.sky_zenith[2].max(0.0),
                params.ao_radius.max(0.05),
            ],
            sky_horizon: [
                params.sky_horizon[0].max(0.0),
                params.sky_horizon[1].max(0.0),
                params.sky_horizon[2].max(0.0),
                params.max_ray_distance.max(1.0),
            ],
            samples_seed: [
                params.shadow_samples.clamp(1, 8) as f32,
                params.ao_samples.min(16) as f32,
                params.gi_samples.min(8) as f32,
                params.frame_seed,
            ],
            bias_and_refl: [
                params.normal_bias.max(1e-5),
                params.reflection_samples.min(8) as f32,
                params.gi_bounce_scale.max(0.0),
                0.0,
            ],
            counts: [scene.tlas_node_count, scene.tlas_instance_count, scene.blas_tri_count, params.visibility_mask],
            extra_counts: [light_count as u32, out_size.0, out_size.1, flags],
        };
        let u_bytes = bytemuck::bytes_of(&u);
        if light_count > 0 {
            let l_bytes: &[u8] = bytemuck::cast_slice(&local_lights[..light_count]);
            let u_len = u_bytes.len() as u64;
            let l_len = l_bytes.len() as u64;
            let stage = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tlas_rt_frame_stage"),
                size: u_len + l_len,
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            stage.slice(0..u_len).get_mapped_range_mut().unwrap().copy_from_slice(u_bytes);
            stage.slice(u_len..u_len + l_len).get_mapped_range_mut().unwrap().copy_from_slice(l_bytes);
            stage.unmap();
            encoder.copy_buffer_to_buffer(&stage, 0, &self.uniform, 0, u_len);
            encoder.copy_buffer_to_buffer(&stage, u_len, &self.local_lights_buf, 0, l_len);
        } else {
            encode_copy(device, encoder, &self.uniform, u_bytes);
        }
        let out = self.outputs.as_ref().unwrap();
        let ddgi_buf = ddgi_packed_buf.unwrap_or(&self.ddgi_dummy_buf);
        let sharc_bind = sharc.unwrap_or(&self.sharc_dummy);
        let (gbuf_nr, gbuf_am) = thin_gbuffer.unwrap_or((&self.gbuf_dummy_tex.1, &self.gbuf_dummy_tex.1));

        #[allow(unused_mut)]
        let mut dispatched_hw = false;
        #[cfg(not(target_arch = "wasm32"))]
        if let (Some(pipe_hw), Some(bgl_hw), Some(hw_tlas)) = (&self.pipeline_hw, &self.bgl_hw, &scene.hw_tlas) {
            let hw_slots = Some(hw_tlas.get().len());
            let hit = matches!(
                &self.bg,
                Some((k, _))
                    if k.hw_slots == hw_slots
                        && k.tlas_nodes == scene.tlas_nodes_buf
                        && k.tlas_instances == scene.tlas_instances_buf
                        && k.blas_nodes == scene.blas_nodes_buf
                        && k.blas_tris == scene.blas_tris_buf
                        && k.depth == *depth_view
                        && k.out_sao == out.shadow_ao.1
                        && k.ddgi == *ddgi_buf
                        && k.gbuf_nr == *gbuf_nr
                        && k.gbuf_am == *gbuf_am
                        && k.sharc_keys == *sharc_bind.keys_buffer()
            );
            if !hit {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("tlas_rt_bg_hw"),
                    layout: bgl_hw,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: hw_tlas.as_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: scene.tlas_instances_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: scene.blas_nodes_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: scene.blas_tris_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(depth_view) },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&out.shadow_ao.1),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(&out.reflection.1),
                        },
                        wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&out.gi.1) },
                        wgpu::BindGroupEntry { binding: 9, resource: self.local_lights_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 10, resource: ddgi_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(gbuf_nr) },
                        wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(gbuf_am) },
                        wgpu::BindGroupEntry { binding: 13, resource: sharc_bind.keys_buffer().as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 14,
                            resource: sharc_bind.resolved_buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry { binding: 15, resource: sharc_bind.uniform_buffer().as_entire_binding() },
                    ],
                });
                self.bg = Some((
                    TlasBgKey {
                        hw_slots,
                        tlas_nodes: scene.tlas_nodes_buf.clone(),
                        tlas_instances: scene.tlas_instances_buf.clone(),
                        blas_nodes: scene.blas_nodes_buf.clone(),
                        blas_tris: scene.blas_tris_buf.clone(),
                        depth: depth_view.clone(),
                        out_sao: out.shadow_ao.1.clone(),
                        ddgi: ddgi_buf.clone(),
                        gbuf_nr: gbuf_nr.clone(),
                        gbuf_am: gbuf_am.clone(),
                        sharc_keys: sharc_bind.keys_buffer().clone(),
                    },
                    bg,
                ));
            }
            let (_, bg) = self.bg.as_ref().unwrap();
            let ts = crate::renderer::timing::scope(crate::renderer::timing::Span::Scene);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tlas_raytrace_hw"),
                timestamp_writes: ts.compute(),
            });
            pass.set_pipeline(pipe_hw);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(out_size.0.div_ceil(8), out_size.1.div_ceil(8), 1);
            dispatched_hw = true;
        }
        if !dispatched_hw {
            let hit = matches!(
                &self.bg,
                Some((k, _))
                    if k.hw_slots.is_none()
                        && k.tlas_nodes == scene.tlas_nodes_buf
                        && k.tlas_instances == scene.tlas_instances_buf
                        && k.blas_nodes == scene.blas_nodes_buf
                        && k.blas_tris == scene.blas_tris_buf
                        && k.depth == *depth_view
                        && k.out_sao == out.shadow_ao.1
                        && k.ddgi == *ddgi_buf
                        && k.gbuf_nr == *gbuf_nr
                        && k.gbuf_am == *gbuf_am
                        && k.sharc_keys == *sharc_bind.keys_buffer()
            );
            if !hit {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("tlas_rt_bg"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: scene.tlas_nodes_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: scene.tlas_instances_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: scene.blas_nodes_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: scene.blas_tris_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(depth_view) },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&out.shadow_ao.1),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(&out.reflection.1),
                        },
                        wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&out.gi.1) },
                        wgpu::BindGroupEntry { binding: 9, resource: self.local_lights_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 10, resource: ddgi_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(gbuf_nr) },
                        wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(gbuf_am) },
                        wgpu::BindGroupEntry { binding: 13, resource: sharc_bind.keys_buffer().as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 14,
                            resource: sharc_bind.resolved_buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry { binding: 15, resource: sharc_bind.uniform_buffer().as_entire_binding() },
                    ],
                });
                self.bg = Some((
                    TlasBgKey {
                        hw_slots: None,
                        tlas_nodes: scene.tlas_nodes_buf.clone(),
                        tlas_instances: scene.tlas_instances_buf.clone(),
                        blas_nodes: scene.blas_nodes_buf.clone(),
                        blas_tris: scene.blas_tris_buf.clone(),
                        depth: depth_view.clone(),
                        out_sao: out.shadow_ao.1.clone(),
                        ddgi: ddgi_buf.clone(),
                        gbuf_nr: gbuf_nr.clone(),
                        gbuf_am: gbuf_am.clone(),
                        sharc_keys: sharc_bind.keys_buffer().clone(),
                    },
                    bg,
                ));
            }
            let (_, bg) = self.bg.as_ref().unwrap();
            let ts = crate::renderer::timing::scope(crate::renderer::timing::Span::Scene);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tlas_raytrace"),
                timestamp_writes: ts.compute(),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(out_size.0.div_ceil(8), out_size.1.div_ceil(8), 1);
        }
        Ok(())
    }
}
