//! Stateful GPU Mesh Particles:
//! - Race-free 3-pass compute pipeline (`simulate_step` -> `apply_spawns` -> `compact_alive`)
//! - 3D divergence-free curl-noise turbulence
//! - Swept world-space depth-buffer bounce collisions with camera-oriented normals
//! - GPU-resident dead-slot reclamation & atomic sub-emitter child spark bursts
//! - GPU-compacted `draw_indexed_indirect` 3D mesh particle rendering.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "GPU buffer capacities and indices")]

use anyhow::{Result, ensure};

use crate::environment_passes::PipelineTarget;

pub const GPU_PARTICLES_SIM_WGSL: &str = include_str!("shaders/gpu_particles_sim.wgsl");
pub const GPU_PARTICLES_DRAW_WGSL: &str = include_str!("shaders/gpu_particles_draw.wgsl");

/// GPU-side 64-byte stateful particle record matching `gpu_particles_sim.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticle {
    pub pos_age: [f32; 4],
    pub vel_life: [f32; 4],
    pub color: [f32; 4],
    pub size_rot_gen: [f32; 4],
}

impl GpuParticle {
    pub fn primary(pos: [f32; 3], vel: [f32; 3], lifetime: f32, size: f32, color: [f32; 4]) -> Self {
        Self {
            pos_age: [pos[0], pos[1], pos[2], 0.0],
            vel_life: [vel[0], vel[1], vel[2], lifetime.max(0.01)],
            color,
            size_rot_gen: [size.max(0.005), 0.0, 2.0, 0.0],
        }
    }
}

/// Vertex of an arbitrary 3D mesh instanced per alive GPU particle.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticleMeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Simulation step configuration for `GpuParticleSystem`.
#[derive(Clone, Copy, Debug)]
pub struct GpuParticleSimParams {
    pub view_projection: [f32; 16],
    pub camera_pos: [f32; 3],
    /// Clear depth representing sky (1.0 standard-Z, 0.0 reversed-Z).
    pub depth_clear: f32,
    pub gravity: [f32; 3],
    pub dt: f32,
    pub drag: f32,
    pub curl_strength: f32,
    pub curl_frequency: f32,
    pub time: f32,
    pub restitution: f32,
    pub friction: f32,
    /// World-space surface collision slab thickness (meters).
    pub depth_thickness: f32,
    pub reversed_z: bool,
    pub depth_collide: bool,
    pub sub_emit_on_bounce: bool,
    pub sub_emit_on_death: bool,
    pub sub_emit_count: u32,
    pub sub_emit_speed: f32,
    pub sub_emit_life: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SimUniform {
    view_projection: [f32; 16],
    inverse_view_projection: [f32; 16],
    camera_pos_and_clear: [f32; 4],
    gravity_dt: [f32; 4],
    curl_drag: [f32; 4],
    collision: [f32; 4],
    sub_emit: [f32; 4],
    flags: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawUniform {
    view_projection: [f32; 16],
    sun_dir_and_ambient: [f32; 4],
    sun_color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct IndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleSimCounters {
    pub spawn_queued: u32,
    pub spawn_consumed: u32,
    pub sub_emitted: u32,
    pub bounces: u32,
}

fn encode_copy(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    dst: &wgpu::Buffer,
    dst_offset: u64,
    bytes: &[u8],
) {
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_particles_stage"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    stage.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytes);
    stage.unmap();
    encoder.copy_buffer_to_buffer(&stage, 0, dst, dst_offset, bytes.len() as u64);
}

/// Generates a unit cube 3D mesh suitable for solid 3D mesh particle instancing.
pub fn unit_cube_particle_mesh() -> (Vec<GpuParticleMeshVertex>, Vec<u32>) {
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 0.0, 1.0], [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]),
        ([0.0, 0.0, -1.0], [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]]),
        ([1.0, 0.0, 0.0], [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]]),
        ([-1.0, 0.0, 0.0], [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]]),
        ([0.0, 1.0, 0.0], [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]]),
        ([0.0, -1.0, 0.0], [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]]),
    ];
    let mut verts = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (n, quad) in faces {
        let base = verts.len() as u32;
        for pos in quad {
            verts.push(GpuParticleMeshVertex { position: pos, normal: n });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (verts, indices)
}

/// Stateful GPU compute-simulated + indirect-rendered 3D mesh particle system.
pub struct GpuParticleSystem {
    max_particles: u32,
    particles: wgpu::Buffer,
    spawn_queue: wgpu::Buffer,
    alive_indices: wgpu::Buffer,
    indirect: wgpu::Buffer,
    counters: wgpu::Buffer,
    sim_uniform: wgpu::Buffer,
    sim_bgl: wgpu::BindGroupLayout,
    pipe_simulate: wgpu::ComputePipeline,
    pipe_apply_spawns: wgpu::ComputePipeline,
    pipe_compact: wgpu::ComputePipeline,
    draw_uniform: wgpu::Buffer,
    draw_bg: wgpu::BindGroup,
    draw_module: wgpu::ShaderModule,
    draw_layout: wgpu::PipelineLayout,
    draw_pipelines: Vec<(PipelineTarget, wgpu::RenderPipeline)>,
    mesh_vb: wgpu::Buffer,
    mesh_ib: wgpu::Buffer,
    mesh_index_count: u32,
}

impl GpuParticleSystem {
    pub fn new(
        device: &wgpu::Device,
        max_particles: u32,
        mesh_vertices: &[GpuParticleMeshVertex],
        mesh_indices: &[u32],
        target: PipelineTarget,
    ) -> Result<Self> {
        let limits = device.limits();
        ensure!(
            limits.max_storage_buffers_per_shader_stage >= 5,
            "GpuParticleSystem requires >= 5 storage buffers per shader stage, device has {}",
            limits.max_storage_buffers_per_shader_stage
        );
        ensure!((1..=1_048_576).contains(&max_particles), "invalid max_particles {max_particles}");
        ensure!(
            max_particles.div_ceil(64) <= limits.max_compute_workgroups_per_dimension,
            "max_particles dispatch exceeds max_compute_workgroups_per_dimension"
        );
        ensure!(
            !mesh_vertices.is_empty() && !mesh_indices.is_empty() && mesh_indices.len().is_multiple_of(3),
            "invalid particle mesh"
        );
        let p_bytes = u64::from(max_particles) * std::mem::size_of::<GpuParticle>() as u64;
        let vb_bytes = std::mem::size_of_val(mesh_vertices) as u64;
        let ib_bytes = std::mem::size_of_val(mesh_indices) as u64;
        ensure!(
            p_bytes <= limits.max_storage_buffer_binding_size && p_bytes <= limits.max_buffer_size,
            "particle buffer size {p_bytes} exceeds device buffer limits"
        );
        ensure!(
            vb_bytes <= limits.max_buffer_size && ib_bytes <= limits.max_buffer_size,
            "particle mesh exceeds max_buffer_size"
        );

        let particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_particles"),
            size: p_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let spawn_queue = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_particles_spawn_queue"),
            size: p_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let alive_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_particles_alive"),
            size: u64::from(max_particles) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let indirect = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_particles_indirect"),
            size: std::mem::size_of::<IndirectArgs>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let counters = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_particles_counters"),
            size: std::mem::size_of::<ParticleSimCounters>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sim_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_particles_sim_u"),
            size: std::mem::size_of::<SimUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_particles_draw_u"),
            size: std::mem::size_of::<DrawUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mesh_vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_particle_mesh_vb"),
            size: vb_bytes,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: true,
        });
        mesh_vb.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytemuck::cast_slice(mesh_vertices));
        mesh_vb.unmap();
        let mesh_ib = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_particle_mesh_ib"),
            size: ib_bytes,
            usage: wgpu::BufferUsages::INDEX,
            mapped_at_creation: true,
        });
        mesh_ib.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytemuck::cast_slice(mesh_indices));
        mesh_ib.unmap();

        let rw_storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let sim_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_particles_sim_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<SimUniform>() as u64),
                    },
                    count: None,
                },
                rw_storage(1),
                rw_storage(2),
                rw_storage(3),
                rw_storage(4),
                rw_storage(5),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let sim_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_particles_sim"),
            source: wgpu::ShaderSource::Wgsl(GPU_PARTICLES_SIM_WGSL.into()),
        });
        let sim_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_particles_sim_pl"),
            bind_group_layouts: &[Some(&sim_bgl)],
            immediate_size: 0,
        });
        let make_compute = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&sim_pl),
                module: &sim_mod,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipe_simulate = make_compute("simulate_step");
        let pipe_apply_spawns = make_compute("apply_spawns");
        let pipe_compact = make_compute("compact_alive");

        let draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_particles_draw_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<DrawUniform>() as u64),
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
        let draw_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_particles_draw_bg"),
            layout: &draw_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: draw_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particles.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: alive_indices.as_entire_binding() },
            ],
        });
        let draw_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_particles_draw"),
            source: wgpu::ShaderSource::Wgsl(GPU_PARTICLES_DRAW_WGSL.into()),
        });
        let draw_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_particles_draw_pl"),
            bind_group_layouts: &[Some(&draw_bgl)],
            immediate_size: 0,
        });
        let draw_pipe = Self::draw_pipeline(device, &draw_mod, &draw_pl, target);

        Ok(Self {
            max_particles,
            particles,
            spawn_queue,
            alive_indices,
            indirect,
            counters,
            sim_uniform,
            sim_bgl,
            pipe_simulate,
            pipe_apply_spawns,
            pipe_compact,
            draw_uniform,
            draw_bg,
            draw_module: draw_mod,
            draw_layout: draw_pl,
            draw_pipelines: vec![(target, draw_pipe)],
            mesh_vb,
            mesh_ib,
            mesh_index_count: mesh_indices.len() as u32,
        })
    }

    fn draw_pipeline(
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        target: PipelineTarget,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gpu_particles_draw_pipe"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuParticleMeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                    ],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
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
                module,
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
        })
    }

    pub(crate) fn select_draw_target(&mut self, device: &wgpu::Device, target: PipelineTarget) {
        if let Some(index) = self.draw_pipelines.iter().position(|(key, _)| *key == target) {
            self.draw_pipelines.swap(0, index);
        } else {
            let pipeline = Self::draw_pipeline(device, &self.draw_module, &self.draw_layout, target);
            self.draw_pipelines.push((target, pipeline));
            let index = self.draw_pipelines.len() - 1;
            self.draw_pipelines.swap(0, index);
        }
    }

    /// Uploads an initial batch of particles into slots `0..initial.len()` in command order;
    /// remaining slots are zeroed (dead/available for GPU-resident sub-emitter allocation).
    pub fn seed(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        initial: &[GpuParticle],
    ) -> Result<()> {
        ensure!(initial.len() <= self.max_particles as usize, "initial batch exceeds max_particles");
        let zeros = vec![
            GpuParticle { pos_age: [0.0; 4], vel_life: [0.0; 4], color: [0.0; 4], size_rot_gen: [0.0; 4] };
            self.max_particles as usize
        ];
        encode_copy(device, encoder, &self.particles, 0, bytemuck::cast_slice(&zeros));
        if !initial.is_empty() {
            encode_copy(device, encoder, &self.particles, 0, bytemuck::cast_slice(initial));
        }
        Ok(())
    }

    /// Advances all particles on the GPU via 3 barrier-separated compute passes:
    /// 1. `simulate_step`: curl turbulence, swept depth collision, queues child spawns
    /// 2. `apply_spawns`: recycles dead slots on the GPU to materialize queued child particles
    /// 3. `compact_alive`: builds compacted `alive_indices` and exact `DrawIndexedIndirect.instance_count`.
    pub fn step(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        params: &GpuParticleSimParams,
    ) -> Result<()> {
        ensure!(params.view_projection.iter().all(|v| v.is_finite()), "non-finite particle view_projection");
        let reset_indirect = IndirectArgs {
            index_count: self.mesh_index_count,
            instance_count: 0,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        };
        let reset_counters = ParticleSimCounters::default();
        encode_copy(device, encoder, &self.indirect, 0, bytemuck::bytes_of(&reset_indirect));
        encode_copy(device, encoder, &self.counters, 0, bytemuck::bytes_of(&reset_counters));

        let u = SimUniform {
            view_projection: params.view_projection,
            inverse_view_projection: crate::matrix::invert(&params.view_projection),
            camera_pos_and_clear: [
                params.camera_pos[0],
                params.camera_pos[1],
                params.camera_pos[2],
                params.depth_clear,
            ],
            gravity_dt: [params.gravity[0], params.gravity[1], params.gravity[2], params.dt.clamp(0.0, 0.1)],
            curl_drag: [params.drag.max(0.0), params.curl_strength, params.curl_frequency.max(0.05), params.time],
            collision: [
                params.restitution.clamp(0.0, 1.0),
                params.friction.clamp(0.0, 1.0),
                params.depth_thickness.clamp(0.01, 20.0),
                if params.reversed_z { 1.0 } else { 0.0 },
            ],
            sub_emit: [
                self.max_particles as f32,
                params.sub_emit_count.min(8) as f32,
                params.sub_emit_speed.max(0.5),
                params.sub_emit_life.max(0.15),
            ],
            flags: [
                if params.depth_collide { 1.0 } else { 0.0 },
                if params.sub_emit_on_bounce { 1.0 } else { 0.0 },
                if params.sub_emit_on_death { 1.0 } else { 0.0 },
                self.mesh_index_count as f32,
            ],
        };
        encode_copy(device, encoder, &self.sim_uniform, 0, bytemuck::bytes_of(&u));

        let sim_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_particles_sim_bg"),
            layout: &self.sim_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.sim_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.particles.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.spawn_queue.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.alive_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.indirect.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.counters.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(depth_view) },
            ],
        });
        let wg = self.max_particles.div_ceil(64);
        // Separate compute passes insert guaranteed storage barriers between simulate -> apply_spawns -> compact!
        for (label, pipe) in [
            ("gpu_particles_simulate", &self.pipe_simulate),
            ("gpu_particles_apply_spawns", &self.pipe_apply_spawns),
            ("gpu_particles_compact", &self.pipe_compact),
        ] {
            let mut pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &sim_bg, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_draw(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        view_projection: [f32; 16],
        sun_dir: [f32; 3],
        sun_color: [f32; 3],
        ambient: f32,
        velocity_align: bool,
    ) {
        let u = DrawUniform {
            view_projection,
            sun_dir_and_ambient: [sun_dir[0], sun_dir[1], sun_dir[2], ambient.max(0.0)],
            sun_color: [sun_color[0], sun_color[1], sun_color[2], if velocity_align { 1.0 } else { 0.0 }],
        };
        encode_copy(device, encoder, &self.draw_uniform, 0, bytemuck::bytes_of(&u));
    }

    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.draw_pipelines[0].1);
        pass.set_bind_group(0, &self.draw_bg, &[]);
        pass.set_vertex_buffer(0, self.mesh_vb.slice(..));
        pass.set_index_buffer(self.mesh_ib.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed_indirect(&self.indirect, 0);
    }

    pub fn particles_buffer(&self) -> &wgpu::Buffer {
        &self.particles
    }
    pub fn indirect_buffer(&self) -> &wgpu::Buffer {
        &self.indirect
    }
    pub fn counters_buffer(&self) -> &wgpu::Buffer {
        &self.counters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_particles_race_free_3pass_curl_bounce_subemit_and_slot_recycling_witness() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter required");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

        let (verts, indices) = unit_cube_particle_mesh();
        let target = PipelineTarget::standard(wgpu::TextureFormat::Rgba8Unorm, Some(wgpu::TextureFormat::Depth32Float));
        let sys = GpuParticleSystem::new(&device, 32, &verts, &indices, target).unwrap();

        let vp: [f32; 16] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -0.25, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0];
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("particle_depth_floor"),
            size: wgpu::Extent3d { width: 32, height: 32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &depth_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&vec![0.5f32; 32 * 32]),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(32 * 4), rows_per_image: Some(32) },
            wgpu::Extent3d { width: 32, height: 32, depth_or_array_layers: 1 },
        );
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // 4 falling primaries + 1 rising particle already exiting the surface (must NOT bounce!).
        let initial = [
            GpuParticle::primary([-0.6, 0.12, -0.4], [0.0, -4.0, 0.0], 5.0, 0.25, [1.0, 0.5, 0.1, 1.0]),
            GpuParticle::primary([0.6, 0.12, -0.4], [0.0, -4.0, 0.0], 5.0, 0.25, [1.0, 0.5, 0.1, 1.0]),
            GpuParticle::primary([-0.6, 0.12, 0.4], [0.0, -4.0, 0.0], 5.0, 0.25, [1.0, 0.5, 0.1, 1.0]),
            GpuParticle::primary([0.6, 0.12, 0.4], [0.0, -4.0, 0.0], 5.0, 0.25, [1.0, 0.5, 0.1, 1.0]),
            GpuParticle::primary([0.0, 0.05, 0.0], [0.0, 3.0, 0.0], 5.0, 0.20, [0.2, 1.0, 0.2, 1.0]),
        ];
        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("particle_color"),
            size: wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let render_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("particle_render_depth"),
            size: wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let render_depth_view = render_depth.create_view(&wgpu::TextureViewDescriptor::default());

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        sys.seed(&device, &mut enc, &initial).unwrap();
        let params = GpuParticleSimParams {
            view_projection: vp,
            camera_pos: [0.0, 4.0, 0.0],
            depth_clear: 1.0,
            gravity: [0.0, -9.81, 0.0],
            dt: 0.05,
            drag: 0.02,
            curl_strength: 6.0,
            curl_frequency: 1.5,
            time: 0.5,
            restitution: 0.8,
            friction: 0.1,
            depth_thickness: 0.5,
            reversed_z: false,
            depth_collide: true,
            sub_emit_on_bounce: true,
            sub_emit_on_death: false,
            sub_emit_count: 3,
            sub_emit_speed: 2.0,
            sub_emit_life: 1.0,
        };
        // Step 1: 4 falling particles bounce + spawn 12 children = 17 alive; exiting particle #4 does NOT bounce!
        sys.step(&device, &mut enc, &depth_view, &params).unwrap();
        sys.prepare_draw(&device, &mut enc, vp, [0.3, 0.9, 0.2], [1.2, 1.1, 1.0], 0.35, true);
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("draw_gpu_particles"),
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
                    view: &render_depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            sys.draw(&mut rp);
        }

        let p_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("p_rb"),
            size: (32 * std::mem::size_of::<GpuParticle>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let ind_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ind_rb"),
            size: std::mem::size_of::<IndirectArgs>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let cnt_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cnt_rb"),
            size: std::mem::size_of::<ParticleSimCounters>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bpr = (64u32 * 4).next_multiple_of(256);
        let img_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("img_rb"),
            size: u64::from(bpr * 64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(sys.particles_buffer(), 0, &p_rb, 0, p_rb.size());
        enc.copy_buffer_to_buffer(sys.indirect_buffer(), 0, &ind_rb, 0, ind_rb.size());
        enc.copy_buffer_to_buffer(sys.counters_buffer(), 0, &cnt_rb, 0, cnt_rb.size());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &img_rb,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(64) },
            },
            wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));

        let map_buf = |b: &wgpu::Buffer| -> Vec<u8> {
            let s = b.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            s.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv().unwrap().unwrap();
            let v = s.get_mapped_range().unwrap().to_vec();
            b.unmap();
            v
        };
        let p_bytes = map_buf(&p_rb);
        let ind_bytes = map_buf(&ind_rb);
        let cnt_bytes = map_buf(&cnt_rb);
        let img_bytes = map_buf(&img_rb);

        let parts: &[GpuParticle] = bytemuck::cast_slice(&p_bytes);
        let ind: &IndirectArgs = bytemuck::from_bytes(&ind_bytes);
        let cnt: &ParticleSimCounters = bytemuck::from_bytes(&cnt_bytes);

        assert_eq!(cnt.bounces, 4, "only the 4 falling particles should bounce (exiting #4 ignored): {cnt:?}");
        assert_eq!(cnt.sub_emitted, 12, "3 child sparks per bounce = 12: {cnt:?}");
        assert_eq!(ind.instance_count, 17, "5 primaries + 12 children = 17 exact compacted alive instances");
        for p in &parts[..4] {
            assert!(p.vel_life[1] > 1.5, "bounced particle must rebound upward: {:?}", p.vel_life);
            let lateral = (p.vel_life[0].powi(2) + p.vel_life[2].powi(2)).sqrt();
            assert!(lateral > 0.01, "3D curl noise must deflect laterally: {:?}", p.vel_life);
        }
        let lit_pixels = (0..64)
            .flat_map(|y| (0..64).map(move |x| (x, y)))
            .filter(|&(x, y)| img_bytes[(y * bpr + x * 4) as usize] > 30)
            .count();
        assert!(lit_pixels > 40, "indirect 3D mesh draw must rasterize particles, got {lit_pixels}");
    }
}
