//! Built-in GPU rendering pipelines for outdoor worlds:
//! - CDLOD Heightfield Terrain (`TerrainPipeline`, `TERRAIN_WGSL`)
//! - Analytical 4-Wave Gerstner Water with Beer-Lambert absorption & caustics (`WaterPipeline`, `WATER_WGSL`)
//! - GPU-Instanced Wind-Swaying Foliage with backlit subsurface translucency (`FoliagePipeline`, `FOLIAGE_WGSL`)
//!
//! All three passes use dynamic uniform offset rings, appending storage buffers, and cached bind groups
//! so multiple draws recorded in a single submission never overwrite each other's inputs.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "GPU dimensions and grid counts")]

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, anyhow, ensure};

pub const TERRAIN_WGSL: &str = crate::wgsl!("terrain.wgsl");
pub const WATER_WGSL: &str = crate::wgsl!("water.wgsl");
pub const FOLIAGE_WGSL: &str = crate::wgsl!("foliage.wgsl");

const UNIFORM_RING_SLOTS: usize = 64;
const MAX_GRID_CELLS: u32 = 256;

static NEXT_HEIGHTMAP_ID: AtomicU64 = AtomicU64::new(1);

/// Render-pass target attachment configuration (supports single-sample/MSAA and standard/reversed-Z).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PipelineTarget {
    pub color_format: wgpu::TextureFormat,
    pub depth_format: Option<wgpu::TextureFormat>,
    pub depth_compare: wgpu::CompareFunction,
    pub sample_count: u32,
}

impl PipelineTarget {
    pub fn standard(color_format: wgpu::TextureFormat, depth_format: Option<wgpu::TextureFormat>) -> Self {
        Self { color_format, depth_format, depth_compare: wgpu::CompareFunction::LessEqual, sample_count: 1 }
    }

    pub fn reversed_z(
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        sample_count: u32,
    ) -> Self {
        Self { color_format, depth_format, depth_compare: wgpu::CompareFunction::GreaterEqual, sample_count }
    }
}

/// Uploaded `R32Float` heightfield texture with a stable identity for bind-group caching.
pub struct TerrainHeightmap {
    id: u64,
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

/// One CDLOD terrain patch instance.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainPatch {
    /// `[origin_x, origin_z, patch_world_size, manual_morph (< 0 enables crack-free distance CDLOD)]`
    pub origin_scale_morph: [f32; 4],
    /// `[uv_min_x, uv_min_z, uv_size, grid_cells (even, 2..=256)]`
    pub uv_rect_and_grid: [f32; 4],
}

/// Global terrain uniform parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainUniform {
    pub view_projection: [f32; 16],
    pub camera_pos_and_fog: [f32; 4],
    pub sun_dir_and_height_scale: [f32; 4],
    pub sun_color_and_ambient: [f32; 4],
    /// `[world_size_x, world_size_z, morph_start_dist, morph_end_dist]`
    pub world_bounds: [f32; 4],
    pub layer_heights: [f32; 4],
    pub fog_color: [f32; 4],
}

/// GPU CDLOD terrain renderer over an `R32Float` heightmap texture.
pub struct TerrainPipeline {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    uniform_stride: u32,
    uniform_slot: usize,
    patches: wgpu::Buffer,
    patch_cap: usize,
    patch_cursor: usize,
    storage_gen: u64,
    cached_bg: Option<(u64, u64, wgpu::BindGroup)>,
}

impl TerrainPipeline {
    pub fn new(device: &wgpu::Device, target: PipelineTarget) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<TerrainUniform>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain_shader"),
            source: wgpu::ShaderSource::Wgsl(crate::deferred_wgsl!("terrain.wgsl").into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: target.depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(true),
                depth_compare: Some(target.depth_compare),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: target.sample_count.max(1), ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target.color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let align = device.limits().min_uniform_buffer_offset_alignment.max(256);
        let uniform_stride = (std::mem::size_of::<TerrainUniform>() as u32).next_multiple_of(align);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain_uniform_ring"),
            size: u64::from(uniform_stride) * UNIFORM_RING_SLOTS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let patch_cap = 64;
        let patches = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain_patches"),
            size: (patch_cap * std::mem::size_of::<TerrainPatch>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bgl,
            uniform,
            uniform_stride,
            uniform_slot: 0,
            patches,
            patch_cap,
            patch_cursor: 0,
            storage_gen: 1,
            cached_bg: None,
        }
    }

    pub fn begin_frame(&mut self) {
        self.uniform_slot = 0;
        self.patch_cursor = 0;
    }

    /// Uploads a validated `R32Float` heightfield texture ready for `TerrainPipeline::draw`.
    pub fn upload_heightmap(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        heights: &[f32],
    ) -> Result<TerrainHeightmap> {
        let max_dim = device.limits().max_texture_dimension_2d;
        ensure!(
            width >= 2 && height >= 2 && width <= max_dim && height <= max_dim,
            "heightmap dimensions {width}x{height} out of valid range 2..={max_dim}"
        );
        let expected =
            (width as usize).checked_mul(height as usize).ok_or_else(|| anyhow!("heightmap size overflow"))?;
        ensure!(
            heights.len() == expected,
            "heightmap sample count mismatch: got {}, expected {expected}",
            heights.len()
        );
        ensure!(heights.iter().all(|h| h.is_finite()), "heightmap contains non-finite values");
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_heightmap"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
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
            bytemuck::cast_slice(heights),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width * 4), rows_per_image: Some(height) },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(TerrainHeightmap {
            id: NEXT_HEIGHTMAP_ID.fetch_add(1, Ordering::Relaxed),
            texture: tex,
            view,
            width,
            height,
        })
    }

    /// Batches terrain patches by their validated even grid cell count and draws them in command order.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        uniform: &TerrainUniform,
        patches: &[TerrainPatch],
        heightmap: &TerrainHeightmap,
    ) -> Result<()> {
        if patches.is_empty() {
            return Ok(());
        }
        ensure!(uniform.view_projection.iter().all(|v| v.is_finite()), "non-finite terrain view_projection");
        ensure!(
            uniform.world_bounds[0] > 0.0 && uniform.world_bounds[1] > 0.0,
            "terrain world bounds must be positive"
        );

        // Group patches by validated even grid size so mixed grid sizes never generate stray rows.
        let mut sorted: Vec<(u32, TerrainPatch)> = Vec::with_capacity(patches.len());
        for &p in patches {
            ensure!(
                p.origin_scale_morph.iter().all(|v| v.is_finite()) && p.uv_rect_and_grid.iter().all(|v| v.is_finite()),
                "non-finite TerrainPatch field"
            );
            let cells = p.uv_rect_and_grid[3].round() as u32;
            ensure!(
                (2..=MAX_GRID_CELLS).contains(&cells) && cells.is_multiple_of(2),
                "TerrainPatch grid_cells must be even in 2..={MAX_GRID_CELLS}, got {cells}"
            );
            sorted.push((cells, p));
        }
        crate::sort::sort_by_u64_key(&mut sorted, |&(c, _)| u64::from(c));
        let ordered_patches: Vec<TerrainPatch> = sorted.iter().map(|&(_, p)| p).collect();

        let rec_bytes = std::mem::size_of::<TerrainPatch>();
        let max_storage = device.limits().max_storage_buffer_binding_size;
        let needed =
            self.patch_cursor.checked_add(ordered_patches.len()).ok_or_else(|| anyhow!("patch count overflow"))?;
        if needed > self.patch_cap {
            let new_cap = needed.next_power_of_two().max(64);
            let new_bytes =
                (new_cap as u64).checked_mul(rec_bytes as u64).ok_or_else(|| anyhow!("terrain storage overflow"))?;
            ensure!(new_bytes <= max_storage, "terrain patches buffer {new_bytes} exceeds limit {max_storage}");
            self.patch_cap = new_cap;
            self.patches = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("terrain_patches"),
                size: new_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.patch_cursor = 0;
            self.storage_gen = self.storage_gen.wrapping_add(1);
            self.cached_bg = None;
        }
        let base_instance = self.patch_cursor;
        queue.write_buffer(&self.patches, (base_instance * rec_bytes) as u64, bytemuck::cast_slice(&ordered_patches));
        self.patch_cursor += ordered_patches.len();

        let slot = self.uniform_slot % UNIFORM_RING_SLOTS;
        self.uniform_slot = self.uniform_slot.wrapping_add(1);
        let dyn_offset = (slot as u32) * self.uniform_stride;
        queue.write_buffer(&self.uniform, u64::from(dyn_offset), bytemuck::bytes_of(uniform));

        let cache_hit =
            matches!(&self.cached_bg, Some((hm_id, sgen, _)) if *hm_id == heightmap.id && *sgen == self.storage_gen);
        if !cache_hit {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain_bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.uniform,
                            offset: 0,
                            size: wgpu::BufferSize::new(std::mem::size_of::<TerrainUniform>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: self.patches.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&heightmap.view) },
                ],
            });
            self.cached_bg = Some((heightmap.id, self.storage_gen, bg));
        }
        let (_, _, bg) = self.cached_bg.as_ref().unwrap();
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[dyn_offset]);

        // Emit one instanced draw per distinct even `grid_cells` run.
        let mut run_start = 0usize;
        while run_start < sorted.len() {
            let cells = sorted[run_start].0;
            let mut run_end = run_start + 1;
            while run_end < sorted.len() && sorted[run_end].0 == cells {
                run_end += 1;
            }
            let verts = cells
                .checked_mul(cells)
                .and_then(|v| v.checked_mul(6))
                .ok_or_else(|| anyhow!("terrain vertex count overflow"))?;
            let inst_start = u32::try_from(base_instance + run_start)?;
            let inst_end = u32::try_from(base_instance + run_end)?;
            pass.draw(0..verts, inst_start..inst_end);
            run_start = run_end;
        }
        Ok(())
    }
}

/// Global uniform block for the analytical Gerstner water surface.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterUniform {
    pub view_projection: [f32; 16],
    pub camera_pos_and_time: [f32; 4],
    pub sun_dir_and_level: [f32; 4],
    pub sun_color_and_bed: [f32; 4],
    pub absorption_and_grid: [f32; 4],
    pub deep_and_foam: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub bed_color_and_slope: [f32; 4],
    pub waves: [[f32; 4]; 4],
}

/// GPU 4-wave Gerstner water pipeline with Beer-Lambert depth absorption, Fresnel & caustics.
pub struct WaterPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    uniform_stride: u32,
    uniform_slot: usize,
    bg: wgpu::BindGroup,
}

impl WaterPipeline {
    pub fn new(device: &wgpu::Device, target: PipelineTarget) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<WaterUniform>() as u64),
                },
                count: None,
            }],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water_shader"),
            source: wgpu::ShaderSource::Wgsl(crate::deferred_wgsl!("water.wgsl").into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: target.depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(true),
                depth_compare: Some(target.depth_compare),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: target.sample_count.max(1), ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target.color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let align = device.limits().min_uniform_buffer_offset_alignment.max(256);
        let uniform_stride = (std::mem::size_of::<WaterUniform>() as u32).next_multiple_of(align);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water_uniform_ring"),
            size: u64::from(uniform_stride) * UNIFORM_RING_SLOTS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water_bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<WaterUniform>() as u64),
                }),
            }],
        });
        Self { pipeline, uniform, uniform_stride, uniform_slot: 0, bg }
    }

    pub fn begin_frame(&mut self) {
        self.uniform_slot = 0;
    }

    pub fn draw(&mut self, queue: &wgpu::Queue, pass: &mut wgpu::RenderPass<'_>, uniform: &WaterUniform) -> Result<()> {
        ensure!(uniform.view_projection.iter().all(|v| v.is_finite()), "non-finite water view_projection");
        let raw_cells = uniform.absorption_and_grid[3];
        ensure!(raw_cells.is_finite(), "non-finite water grid cells");
        let cells = raw_cells.round().clamp(2.0, MAX_GRID_CELLS as f32) as u32;
        let verts = cells
            .checked_mul(cells)
            .and_then(|v| v.checked_mul(6))
            .ok_or_else(|| anyhow!("water vertex count overflow"))?;
        let slot = self.uniform_slot % UNIFORM_RING_SLOTS;
        self.uniform_slot = self.uniform_slot.wrapping_add(1);
        let dyn_offset = (slot as u32) * self.uniform_stride;
        queue.write_buffer(&self.uniform, u64::from(dyn_offset), bytemuck::bytes_of(uniform));
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bg, &[dyn_offset]);
        pass.draw(0..verts, 0..1);
        Ok(())
    }
}

/// Global uniform parameters for GPU-instanced wind-swaying foliage.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FoliageUniform {
    pub view_projection: [f32; 16],
    pub camera_pos_and_time: [f32; 4],
    pub sun_dir_and_wind_str: [f32; 4],
    pub sun_color_and_gust: [f32; 4],
    pub wind_dir_and_speed: [f32; 4],
    pub root_color_and_width: [f32; 4],
    pub tip_color_and_height: [f32; 4],
    pub sss_color_and_power: [f32; 4],
}

/// One GPU-instanced foliage blade.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FoliageInstance {
    pub pos_and_scale: [f32; 4],
    pub rot_phase_tint: [f32; 4],
}

/// GPU-instanced wind-swaying foliage renderer with backlit subsurface scattering.
pub struct FoliagePipeline {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    uniform_stride: u32,
    uniform_slot: usize,
    instances: wgpu::Buffer,
    instance_cap: usize,
    instance_cursor: usize,
    cached_bg: wgpu::BindGroup,
}

impl FoliagePipeline {
    pub fn new(device: &wgpu::Device, target: PipelineTarget) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("foliage_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<FoliageUniform>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("foliage_shader"),
            source: wgpu::ShaderSource::Wgsl(crate::deferred_wgsl!("foliage.wgsl").into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("foliage_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("foliage_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: target.depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(true),
                depth_compare: Some(target.depth_compare),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: target.sample_count.max(1), ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target.color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let align = device.limits().min_uniform_buffer_offset_alignment.max(256);
        let uniform_stride = (std::mem::size_of::<FoliageUniform>() as u32).next_multiple_of(align);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foliage_uniform_ring"),
            size: u64::from(uniform_stride) * UNIFORM_RING_SLOTS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance_cap = 256;
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foliage_instances"),
            size: (instance_cap * std::mem::size_of::<FoliageInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cached_bg = Self::make_bg(device, &bgl, &uniform, &instances);
        Self {
            pipeline,
            bgl,
            uniform,
            uniform_stride,
            uniform_slot: 0,
            instances,
            instance_cap,
            instance_cursor: 0,
            cached_bg,
        }
    }

    fn make_bg(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        instances: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("foliage_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: uniform,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<FoliageUniform>() as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: instances.as_entire_binding() },
            ],
        })
    }

    pub fn begin_frame(&mut self) {
        self.uniform_slot = 0;
        self.instance_cursor = 0;
    }

    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        uniform: &FoliageUniform,
        instances: &[FoliageInstance],
    ) -> Result<()> {
        if instances.is_empty() {
            return Ok(());
        }
        ensure!(uniform.view_projection.iter().all(|v| v.is_finite()), "non-finite foliage view_projection");
        let rec_bytes = std::mem::size_of::<FoliageInstance>();
        let max_storage = device.limits().max_storage_buffer_binding_size;
        let needed =
            self.instance_cursor.checked_add(instances.len()).ok_or_else(|| anyhow!("foliage count overflow"))?;
        if needed > self.instance_cap {
            let new_cap = needed.next_power_of_two().max(256);
            let new_bytes =
                (new_cap as u64).checked_mul(rec_bytes as u64).ok_or_else(|| anyhow!("foliage storage overflow"))?;
            ensure!(new_bytes <= max_storage, "foliage instances buffer {new_bytes} exceeds limit {max_storage}");
            self.instance_cap = new_cap;
            self.instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("foliage_instances"),
                size: new_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_cursor = 0;
            self.cached_bg = Self::make_bg(device, &self.bgl, &self.uniform, &self.instances);
        }
        let base_instance = self.instance_cursor;
        queue.write_buffer(&self.instances, (base_instance * rec_bytes) as u64, bytemuck::cast_slice(instances));
        self.instance_cursor += instances.len();

        let slot = self.uniform_slot % UNIFORM_RING_SLOTS;
        self.uniform_slot = self.uniform_slot.wrapping_add(1);
        let dyn_offset = (slot as u32) * self.uniform_stride;
        queue.write_buffer(&self.uniform, u64::from(dyn_offset), bytemuck::bytes_of(uniform));

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.cached_bg, &[dyn_offset]);
        let start = u32::try_from(base_instance)?;
        let end = u32::try_from(base_instance + instances.len())?;
        pass.draw(0..21, start..end);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn look_at_rh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [f32; 16] {
        let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let cross = |a: [f32; 3], b: [f32; 3]| {
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        };
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let norm = |a: [f32; 3]| {
            let l = dot(a, a).sqrt().max(1e-8);
            [a[0] / l, a[1] / l, a[2] / l]
        };
        let f = norm(sub(target, eye));
        let s = norm(cross(f, up));
        let u = cross(s, f);
        [
            s[0],
            u[0],
            -f[0],
            0.0,
            s[1],
            u[1],
            -f[1],
            0.0,
            s[2],
            u[2],
            -f[2],
            0.0,
            -dot(s, eye),
            -dot(u, eye),
            dot(f, eye),
            1.0,
        ]
    }

    fn perspective_rh_zo(fov_y: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
        let f = 1.0 / (fov_y * 0.5).tan();
        [
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            far / (near - far),
            -1.0,
            0.0,
            0.0,
            (far * near) / (near - far),
            0.0,
        ]
    }

    fn mul4(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        for c in 0..4 {
            for r in 0..4 {
                out[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
            }
        }
        out
    }

    fn render_and_read_rgba8(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        draw: impl FnOnce(&mut wgpu::RenderPass<'_>),
    ) -> Vec<[u8; 4]> {
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("env_test_color"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("env_test_depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("env_test_pass"),
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
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            draw(&mut pass);
        }
        let bpr = (w * 4).next_multiple_of(256);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("env_test_readback"),
            size: u64::from(bpr * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let bytes = slice.get_mapped_range().unwrap();
        let mut out = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let o = (y * bpr + x * 4) as usize;
                out.push([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
            }
        }
        drop(bytes);
        buf.unmap();
        out
    }

    fn init_gpu() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter required");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap()
    }

    #[test]
    fn gpu_terrain_cdlod_adjacent_patches_and_splatting_witness() {
        let (device, queue) = init_gpu();
        let target = PipelineTarget::standard(wgpu::TextureFormat::Rgba8Unorm, Some(wgpu::TextureFormat::Depth32Float));
        let mut terrain = TerrainPipeline::new(&device, target);
        let (hm_w, hm_h) = (32u32, 32u32);
        let mut heights = vec![0.0f32; (hm_w * hm_h) as usize];
        for z in 0..hm_h {
            for x in 0..hm_w {
                let fx = x as f32 / (hm_w - 1) as f32;
                let h = if fx < 0.35 {
                    0.22
                } else if fx < 0.65 {
                    0.22 + (fx - 0.35) / 0.30 * 0.78
                } else {
                    1.0
                };
                heights[(z * hm_w + x) as usize] = h;
            }
        }
        let hm = TerrainPipeline::upload_heightmap(&device, &queue, hm_w, hm_h, &heights).unwrap();
        let eye = [0.0, 26.0, 14.0];
        let vp = mul4(
            perspective_rh_zo(std::f32::consts::FRAC_PI_3, 1.0, 0.5, 200.0),
            look_at_rh(eye, [0.0, 4.0, 0.0], [0.0, 1.0, 0.0]),
        );
        let uniform = TerrainUniform {
            view_projection: vp,
            camera_pos_and_fog: [eye[0], eye[1], eye[2], 0.0],
            sun_dir_and_height_scale: [-0.4, 0.85, 0.3, 12.0],
            sun_color_and_ambient: [2.2, 2.1, 1.9, 0.6],
            world_bounds: [20.0, 20.0, 15.0, 40.0], // Distance-based CDLOD morph across 15..40m!
            layer_heights: [1.0, 6.0, 10.0, 0.20],
            fog_color: [0.6, 0.7, 0.85, 0.5],
        };
        // Two adjacent patches sharing x = 0 seam with distance-based CDLOD morph (-1.0) and mixed even grid sizes.
        let patches = [
            TerrainPatch { origin_scale_morph: [-10.0, -10.0, 10.0, -1.0], uv_rect_and_grid: [0.0, 0.0, 0.5, 16.0] },
            TerrainPatch { origin_scale_morph: [0.0, -10.0, 10.0, -1.0], uv_rect_and_grid: [0.5, 0.0, 0.5, 16.0] },
            TerrainPatch { origin_scale_morph: [-10.0, 0.0, 10.0, -1.0], uv_rect_and_grid: [0.0, 0.5, 0.5, 16.0] },
            TerrainPatch { origin_scale_morph: [0.0, 0.0, 10.0, -1.0], uv_rect_and_grid: [0.5, 0.5, 0.5, 16.0] },
        ];
        let img = render_and_read_rgba8(&device, &queue, 64, 64, |pass| {
            terrain.draw(&device, &queue, pass, &uniform, &patches, &hm).unwrap();
        });
        let meadow = img[32 * 64 + 12];
        let seam_center = img[32 * 64 + 32];
        let summit = img[32 * 64 + 52];
        assert!(meadow[1] > meadow[0] + 15 && meadow[1] > meadow[2] + 15, "meadow must be green: {meadow:?}");
        assert!(
            seam_center[0] > 10 || seam_center[1] > 10,
            "adjacent CDLOD seam must have no black crack: {seam_center:?}"
        );
        assert!(
            summit[0] > 160 && summit[1] > 160 && summit[2] > 160,
            "high snow plateau must be bright white: {summit:?}"
        );
    }

    #[test]
    fn gpu_water_gerstner_beer_lambert_dry_discard_and_multi_draw_witness() {
        let (device, queue) = init_gpu();
        let target = PipelineTarget::standard(wgpu::TextureFormat::Rgba8Unorm, Some(wgpu::TextureFormat::Depth32Float));
        let mut water = WaterPipeline::new(&device, target);
        let eye = [0.0, 12.0, 8.0];
        let vp = mul4(
            perspective_rh_zo(std::f32::consts::FRAC_PI_3, 1.0, 0.5, 200.0),
            look_at_rh(eye, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        );
        let uniform = WaterUniform {
            view_projection: vp,
            camera_pos_and_time: [eye[0], eye[1], eye[2], 1.5],
            sun_dir_and_level: [0.2, 0.9, 0.3, 0.0],
            sun_color_and_bed: [2.0, 1.95, 1.85, -3.0],
            absorption_and_grid: [0.65, 0.22, 0.08, 24.0],
            deep_and_foam: [0.02, 0.10, 0.22, 0.45],
            sky_horizon: [0.65, 0.78, 0.92, 10.0],
            sky_zenith: [0.18, 0.38, 0.78, 0.7],
            bed_color_and_slope: [0.85, 0.70, 0.45, -0.45],
            waves: [[1.0, 0.2, 0.14, 5.0], [0.4, 0.9, 0.10, 3.1], [-0.6, 0.8, 0.06, 1.8], [0.8, -0.5, 0.04, 1.1]],
        };
        let img = render_and_read_rgba8(&device, &queue, 64, 64, |pass| {
            water.draw(&queue, pass, &uniform).unwrap();
        });
        let shallow = img[32 * 64 + 14];
        let deep = img[32 * 64 + 50];
        assert!(shallow[0] > deep[0] + 20, "shallow water must transmit warm bed: shallow={shallow:?} deep={deep:?}");
        assert!(deep[2] > deep[0] + 15, "deep water must absorb red faster than blue: deep={deep:?}");

        // Dry-cell discard: seabed above water level (+2.0 > 0.0) with calm waves must discard (stay black).
        let dry_u = WaterUniform {
            sun_color_and_bed: [2.0, 1.95, 1.85, 2.0],
            bed_color_and_slope: [0.8, 0.7, 0.5, 0.0],
            waves: [[0.0; 4]; 4],
            ..uniform
        };
        let dry_img = render_and_read_rgba8(&device, &queue, 32, 32, |pass| {
            water.draw(&queue, pass, &dry_u).unwrap();
        });
        assert!(dry_img.iter().all(|p| p[0] == 0 && p[1] == 0 && p[2] == 0), "dry water cells must discard cleanly");

        // Two-viewport single-submission draw: left half warm shallow bed, right half deep blue water.
        water.begin_frame();
        let left_u = WaterUniform {
            sun_color_and_bed: [2.0, 1.95, 1.85, -0.4],
            bed_color_and_slope: [0.95, 0.20, 0.10, 0.0],
            waves: [[0.0; 4]; 4],
            ..uniform
        };
        let right_u = WaterUniform {
            sun_color_and_bed: [2.0, 1.95, 1.85, -20.0],
            deep_and_foam: [0.02, 0.05, 0.75, 0.9],
            waves: [[0.0; 4]; 4],
            ..uniform
        };
        let split = render_and_read_rgba8(&device, &queue, 64, 32, |pass| {
            pass.set_viewport(0.0, 0.0, 32.0, 32.0, 0.0, 1.0);
            water.draw(&queue, pass, &left_u).unwrap();
            pass.set_viewport(32.0, 0.0, 32.0, 32.0, 0.0, 1.0);
            water.draw(&queue, pass, &right_u).unwrap();
        });
        let left_px = split[16 * 64 + 16];
        let right_px = split[16 * 64 + 48];
        assert!(
            left_px[0] > right_px[0] + 30 && right_px[2] > right_px[0] + 30,
            "two draws in one submission must preserve distinct uniforms: left={left_px:?} right={right_px:?}"
        );
    }

    #[test]
    fn gpu_foliage_wind_sway_outward_normal_and_backlit_sss_witness() {
        let (device, queue) = init_gpu();
        let target = PipelineTarget::standard(wgpu::TextureFormat::Rgba8Unorm, Some(wgpu::TextureFormat::Depth32Float));
        let mut foliage = FoliagePipeline::new(&device, target);
        let eye = [0.0, 1.0, 4.0];
        let vp = mul4(
            perspective_rh_zo(std::f32::consts::FRAC_PI_3, 1.0, 0.2, 100.0),
            look_at_rh(eye, [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        );
        let instances = [FoliageInstance { pos_and_scale: [0.0, 0.0, 0.0, 1.0], rot_phase_tint: [0.0, 0.0, 0.0, 0.5] }];

        // 1. Calm wind vs strong +X wind: blade tips bend right (+X); also verify zero wind vector is safe (no NaN).
        let base_u = FoliageUniform {
            view_projection: vp,
            camera_pos_and_time: [eye[0], eye[1], eye[2], 0.0],
            sun_dir_and_wind_str: [0.0, 0.3, -0.95, 0.0], // Sun behind blade (-Z) -> backlit!
            sun_color_and_gust: [2.5, 2.3, 1.8, 1.0],
            wind_dir_and_speed: [0.0, 0.05, 0.0, 2.0], // Zero wind direction tests degenerate guard!
            root_color_and_width: [0.12, 0.30, 0.08, 0.35],
            tip_color_and_height: [0.30, 0.62, 0.14, 2.2],
            sss_color_and_power: [0.55, 0.90, 0.18, 2.0],
        };
        let calm = render_and_read_rgba8(&device, &queue, 64, 64, |pass| {
            foliage.draw(&device, &queue, pass, &base_u, &instances).unwrap();
        });
        let windy_u = FoliageUniform {
            sun_dir_and_wind_str: [0.0, 0.3, -0.95, 0.85],
            wind_dir_and_speed: [1.0, 0.05, 0.0, 2.0],
            ..base_u
        };
        let windy = render_and_read_rgba8(&device, &queue, 64, 64, |pass| {
            foliage.draw(&device, &queue, pass, &windy_u, &instances).unwrap();
        });
        let centroid_x = |img: &[[u8; 4]]| -> (f32, usize) {
            let (mut sx, mut count) = (0.0f32, 0usize);
            for y in 0..64 {
                for x in 0..64 {
                    let p = img[y * 64 + x];
                    if p[1] > 10 {
                        sx += x as f32;
                        count += 1;
                    }
                }
            }
            (sx / count.max(1) as f32, count)
        };
        let (cx_calm, n_calm) = centroid_x(&calm);
        let (cx_wind, n_wind) = centroid_x(&windy);
        assert!(n_calm > 20 && n_wind > 20, "blade must rasterize: calm={n_calm} wind={n_wind}");
        assert!(cx_wind > cx_calm + 2.5, "wind along +X must bend blade right: calm={cx_calm} windy={cx_wind}");

        // 2. Front-lit normal orientation: sun on +Z (front of yaw=0 blade) lights front brighter than sun on -Z with SSS off.
        let sum_g = |img: &[[u8; 4]]| img.iter().map(|p| u32::from(p[1])).sum::<u32>();
        let no_sss_back = render_and_read_rgba8(&device, &queue, 64, 64, |pass| {
            let u = FoliageUniform { sss_color_and_power: [0.0, 0.0, 0.0, 2.0], ..base_u };
            foliage.draw(&device, &queue, pass, &u, &instances).unwrap();
        });
        let no_sss_front = render_and_read_rgba8(&device, &queue, 64, 64, |pass| {
            let u = FoliageUniform {
                sun_dir_and_wind_str: [0.0, 0.3, 0.95, 0.0], // Sun in front (+Z)
                sss_color_and_power: [0.0, 0.0, 0.0, 2.0],
                ..base_u
            };
            foliage.draw(&device, &queue, pass, &u, &instances).unwrap();
        });
        assert!(
            sum_g(&no_sss_front) > sum_g(&no_sss_back) * 2,
            "outward front normal (+Z) must light brighter from front sun than back sun: front={} back={}",
            sum_g(&no_sss_front),
            sum_g(&no_sss_back)
        );
        // 3. Backlit subsurface scattering brightens backlit blade dramatically over pure shadow.
        assert!(
            sum_g(&calm) > sum_g(&no_sss_back) * 2,
            "backlit subsurface translucency must brighten blade: sss={} no_sss={}",
            sum_g(&calm),
            sum_g(&no_sss_back)
        );
    }
}
