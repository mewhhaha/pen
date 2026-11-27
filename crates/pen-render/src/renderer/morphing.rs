// GPU compute blendshape / morph-target evaluation.
// Each morphed draw writes deformed vertices (position + re-normalized normal +
// unchanged UV/attributes) into a transient per-frame vertex buffer before
// shadow, reflection, scene and inset passes run, so every mesh shader (plain,
// skinned, translucent, inset and shadow) reads morphed vertices transparently.
use pen_protocol::RenderFrame;

use super::{GpuAsset, MeshAsset, Scratch, State};
use crate::asset_store::{AssetStore, MESH_VERTEX_FLOATS};

const MORPH_SHADER: &str = crate::wgsl!("morph.wgsl");

struct MorphJob {
    mesh_slot: usize,
    vertex_count: u32,
    params_offset: u32,
}

pub(super) struct Morphing {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    params_alignment: usize,
    params_data: Vec<u8>,
    weights_data: Vec<f32>,
    jobs: Vec<MorphJob>,
    out_vertices: u32,
    params_buffer: Option<wgpu::Buffer>,
    weights_buffer: Option<wgpu::Buffer>,
    output_buffer: Option<wgpu::Buffer>,
    bg_cache: std::sync::Mutex<Vec<(wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup)>>,
}

impl Morphing {
    pub fn new(device: &wgpu::Device, cache: Option<&wgpu::PipelineCache>) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("morph"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(40),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(24),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(4),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(40),
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("morph"),
            source: wgpu::ShaderSource::Wgsl(MORPH_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("morph"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("morph"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache,
        });
        let params_alignment = (device.limits().min_uniform_buffer_offset_alignment as usize).max(16);
        Self {
            layout,
            pipeline,
            params_alignment,
            params_data: Vec::new(),
            weights_data: Vec::new(),
            jobs: Vec::new(),
            out_vertices: 0,
            params_buffer: None,
            weights_buffer: None,
            output_buffer: None,
            bg_cache: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn begin(&mut self) {
        self.params_data.clear();
        self.weights_data.clear();
        self.jobs.clear();
        self.out_vertices = 0;
    }

    /// Schedules GPU morphing for one draw and returns `(byte_start, byte_end)`
    /// into the frame's morphed vertex buffer, or `None` if all valid weights
    /// are zero.
    pub fn schedule(
        &mut self,
        mesh_slot: usize,
        vertex_count: u32,
        target_count: u32,
        entries: impl Iterator<Item = (u32, f32)>,
    ) -> Option<(u64, u64)> {
        if vertex_count == 0 || target_count == 0 {
            return None;
        }
        let weight_offset = u32::try_from(self.weights_data.len()).ok()?;
        let start = self.weights_data.len();
        self.weights_data.resize(start + target_count as usize, 0.0);
        let mut active = false;
        for (target, weight) in entries {
            if target < target_count && weight.is_finite() {
                self.weights_data[start + target as usize] += weight;
                if self.weights_data[start + target as usize] != 0.0 {
                    active = true;
                }
            }
        }
        if !active {
            self.weights_data.truncate(start);
            return None;
        }
        let out_offset = self.out_vertices;
        self.out_vertices = self.out_vertices.checked_add(vertex_count)?;
        let params_offset = u32::try_from(self.params_data.len()).ok()?;
        let params = [vertex_count, target_count, weight_offset, out_offset];
        self.params_data.extend_from_slice(bytemuck::cast_slice(&params));
        let padded = self.params_data.len().next_multiple_of(self.params_alignment);
        self.params_data.resize(padded, 0);
        self.jobs.push(MorphJob { mesh_slot, vertex_count, params_offset });
        let stride = (MESH_VERTEX_FLOATS * 4) as u64;
        let byte_start = u64::from(out_offset) * stride;
        let byte_end = byte_start + u64::from(vertex_count) * stride;
        Some((byte_start, byte_end))
    }

    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.jobs.is_empty() {
            return;
        }
        let mut reallocated = false;
        let params_size = (self.params_data.len() as u64).max(16);
        if self.params_buffer.as_ref().is_none_or(|b| b.size() < params_size) {
            self.params_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("morph params"),
                size: params_size.next_power_of_two(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            reallocated = true;
        }
        queue.write_buffer(self.params_buffer.as_ref().unwrap(), 0, &self.params_data);

        let weights_bytes: &[u8] = bytemuck::cast_slice(&self.weights_data);
        let weights_size = (weights_bytes.len() as u64).max(4);
        if self.weights_buffer.as_ref().is_none_or(|b| b.size() < weights_size) {
            self.weights_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("morph weights"),
                size: weights_size.next_power_of_two(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            reallocated = true;
        }
        queue.write_buffer(self.weights_buffer.as_ref().unwrap(), 0, weights_bytes);

        let out_size = (u64::from(self.out_vertices) * (MESH_VERTEX_FLOATS * 4) as u64).max(40);
        if self.output_buffer.as_ref().is_none_or(|b| b.size() < out_size) {
            self.output_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("morph output"),
                size: out_size.next_power_of_two(),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
            reallocated = true;
        }
        if reallocated {
            self.bg_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clear();
        }
    }

    pub fn output(&self) -> Option<&wgpu::Buffer> {
        if self.jobs.is_empty() { None } else { self.output_buffer.as_ref() }
    }

    pub fn dispatch(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, assets: &AssetStore<GpuAsset>) {
        if self.jobs.is_empty() {
            return;
        }
        let (Some(params_buf), Some(weights_buf), Some(output_buf)) =
            (&self.params_buffer, &self.weights_buffer, &self.output_buffer)
        else {
            return;
        };
        let mut cache = self.bg_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if cache.len() > 64 {
            cache.clear();
        }
        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("morph"), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        for job in &self.jobs {
            let Some(GpuAsset::Mesh(MeshAsset { vertices, morphs: Some(morphs), .. })) = assets.current(job.mesh_slot)
            else {
                continue;
            };
            let idx = if let Some(idx) = cache.iter().position(|(v, d, _)| v == vertices && d == &morphs.deltas) {
                idx
            } else {
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("morph"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: params_buf,
                                offset: 0,
                                size: wgpu::BufferSize::new(16),
                            }),
                        },
                        wgpu::BindGroupEntry { binding: 1, resource: vertices.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: morphs.deltas.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: weights_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: output_buf.as_entire_binding() },
                    ],
                });
                cache.push((vertices.clone(), morphs.deltas.clone(), group));
                cache.len() - 1
            };
            pass.set_bind_group(0, &cache[idx].2, &[job.params_offset]);
            pass.dispatch_workgroups(job.vertex_count.div_ceil(64), 1, 1);
        }
    }
}

impl Scratch {
    /// Sorts the frame's morph target weights by target draw index (scene
    /// draws first, then each inset's draws in order).
    pub(super) fn group_morphs(&mut self, frame: &RenderFrame) {
        self.morph_first.clear();
        self.morph_order.clear();
        if frame.morph_weights.is_empty() {
            return;
        }
        let total = self.total_draw_targets();
        self.morph_first.resize(total + 1, 0);
        for weight in &frame.morph_weights {
            if let Some(target) = self.draw_target_index(weight.inset, weight.draw) {
                self.morph_first[target + 1] += 1;
            }
        }
        for index in 1..self.morph_first.len() {
            self.morph_first[index] += self.morph_first[index - 1];
        }
        self.group_cursor.clear();
        self.group_cursor.extend_from_slice(&self.morph_first[..total]);
        self.morph_order.resize(frame.morph_weights.len(), 0);
        for (index, weight) in frame.morph_weights.iter().enumerate() {
            if let Some(target) = self.draw_target_index(weight.inset, weight.draw) {
                let slot = &mut self.group_cursor[target];
                self.morph_order[*slot as usize] = u32::try_from(index).unwrap_or(u32::MAX);
                *slot += 1;
            }
        }
    }

    pub(super) fn morphed(&self, target: usize) -> bool {
        self.morph_first.get(target + 1).is_some_and(|&end| end > self.morph_first[target])
    }

    pub(super) fn morph_entries<'a>(
        &'a self,
        frame: &'a RenderFrame,
        target: usize,
    ) -> impl Iterator<Item = (u32, f32)> + 'a {
        let range = match self.morph_first.get(target + 1) {
            Some(&end) => self.morph_first[target] as usize..end as usize,
            None => 0..0,
        };
        self.morph_order[range].iter().map(move |&index| {
            let m = &frame.morph_weights[index as usize];
            (m.target, m.weight)
        })
    }
}

impl State {
    pub(super) fn morph_draw(
        &mut self,
        frame: &RenderFrame,
        scratch: &Scratch,
        target: usize,
        mesh_slot: usize,
    ) -> Option<(u64, u64)> {
        let (vertex_count, target_count) = match self.assets.current(mesh_slot) {
            Some(GpuAsset::Mesh(MeshAsset { vertex_count, morphs: Some(morphs), .. })) => {
                (*vertex_count, morphs.targets)
            }
            _ => return None,
        };
        self.morphing.schedule(mesh_slot, vertex_count, target_count, scratch.morph_entries(frame, target))
    }
}
