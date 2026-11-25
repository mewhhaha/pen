use super::super::State;
use super::{GPU_PARTICLES_EXT_TAG, WiredPasses};
use crate::built_in::BuiltIns;
use crate::camera_precision::DepthMode;
use crate::environment_passes::PipelineTarget;
use crate::gpu_particles::{
    GpuParticle, GpuParticleMeshVertex, GpuParticleSimParams, GpuParticleSystem, unit_cube_particle_mesh,
};
use crate::matrix::{invert, multiply};
use crate::post::SCENE_FORMAT;
use anyhow::Result;
use pen_protocol::{Matrix, RenderFrame};
use std::hash::{Hash, Hasher};

/// Configuration and live control state for GPU compute-simulated 3D mesh particles rendered inside
/// `Renderer::draw` (curl-noise turbulence, depth-buffer bounce collisions, and GPU sub-emitters).
#[derive(Clone, Debug)]
pub struct GpuParticleScene {
    pub max_particles: u32,
    pub mesh_vertices: Vec<GpuParticleMeshVertex>,
    pub mesh_indices: Vec<u32>,
    pub gravity: [f32; 3],
    /// Simulation timestep in seconds (`<= 0.0` pauses stepping while keeping existing particles rendered).
    pub dt: f32,
    pub drag: f32,
    pub curl_strength: f32,
    pub curl_frequency: f32,
    pub restitution: f32,
    pub friction: f32,
    pub depth_thickness: f32,
    pub depth_collide: bool,
    pub sub_emit_on_bounce: bool,
    pub sub_emit_on_death: bool,
    pub sub_emit_count: u32,
    pub sub_emit_speed: f32,
    pub sub_emit_life: f32,
    pub velocity_align: bool,
    pub ambient: f32,
    /// Optional particle batch seeded into GPU memory on the next frame, then consumed.
    pub pending_seed: Option<Vec<GpuParticle>>,
}

impl Default for GpuParticleScene {
    fn default() -> Self {
        let (mesh_vertices, mesh_indices) = unit_cube_particle_mesh();
        Self {
            max_particles: 4096,
            mesh_vertices,
            mesh_indices,
            gravity: [0.0, -9.81, 0.0],
            dt: 1.0 / 60.0,
            drag: 0.15,
            curl_strength: 1.5,
            curl_frequency: 0.45,
            restitution: 0.45,
            friction: 0.2,
            depth_thickness: 0.35,
            depth_collide: true,
            sub_emit_on_bounce: true,
            sub_emit_on_death: false,
            sub_emit_count: 3,
            sub_emit_speed: 4.0,
            sub_emit_life: 0.6,
            velocity_align: true,
            ambient: 0.35,
            pending_seed: None,
        }
    }
}

#[derive(Default)]
pub(in crate::renderer) struct GpuParticleWiredState {
    pub scene: Option<GpuParticleScene>,
    pub system: Option<GpuParticleSystem>,
    pub config_key: Option<(u32, u32, bool, u64)>,
    pub sim_time: f32,
    pub last_ext_seed_hash: u64,
    pub last_seed_snapshot: Vec<GpuParticle>,
}

fn mesh_hash(verts: &[GpuParticleMeshVertex], indices: &[u32]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    verts.len().hash(&mut h);
    for v in verts {
        for f in v.position {
            f.to_bits().hash(&mut h);
        }
        for f in v.normal {
            f.to_bits().hash(&mut h);
        }
    }
    indices.hash(&mut h);
    h.finish()
}

fn decode_ext_gpu_particles(frame: &RenderFrame) -> Option<(GpuParticleScene, bool, u64)> {
    let ext = frame.extensions.iter().find(|e| e.tag == GPU_PARTICLES_EXT_TAG && e.values.len() >= 16)?;
    let v = &ext.values;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let max_particles = (v[0].round().clamp(1.0, 262_144.0)) as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let flags = v[11].max(0.0).round() as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let sub_emit_count = (v[12].round().clamp(0.0, 8.0)) as u32;
    let reseed_flag = (flags & 16) != 0;
    let particle_floats = v.len() - 16;
    let count = (particle_floats / 16).min(max_particles as usize);
    let mut seed = Vec::with_capacity(count);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    count.hash(&mut hasher);
    for i in 0..count {
        let o = 16 + i * 16;
        for &f in &v[o..o + 16] {
            f.to_bits().hash(&mut hasher);
        }
        seed.push(GpuParticle {
            pos_age: [v[o], v[o + 1], v[o + 2], v[o + 3]],
            vel_life: [v[o + 4], v[o + 5], v[o + 6], v[o + 7]],
            color: [v[o + 8], v[o + 9], v[o + 10], v[o + 11]],
            size_rot_gen: [v[o + 12], v[o + 13], v[o + 14], v[o + 15]],
        });
    }
    let seed_hash = hasher.finish();
    let scene = GpuParticleScene {
        max_particles,
        dt: v[1].clamp(0.0, 0.1),
        gravity: [v[2], v[3], v[4]],
        drag: v[5].max(0.0),
        curl_strength: v[6],
        curl_frequency: v[7].max(0.05),
        restitution: v[8].clamp(0.0, 1.0),
        friction: v[9].clamp(0.0, 1.0),
        depth_thickness: v[10].clamp(0.01, 20.0),
        depth_collide: (flags & 1) != 0,
        sub_emit_on_bounce: (flags & 2) != 0,
        sub_emit_on_death: (flags & 4) != 0,
        velocity_align: (flags & 8) != 0,
        sub_emit_count,
        sub_emit_speed: v[13].max(0.5),
        sub_emit_life: v[14].max(0.15),
        ambient: v[15].max(0.0),
        pending_seed: (!seed.is_empty()).then_some(seed),
        ..GpuParticleScene::default()
    };
    Some((scene, reseed_flag, seed_hash))
}

impl State {
    pub(in crate::renderer) fn has_active_gpu_particles(&self, frame: &RenderFrame) -> bool {
        if self.device.limits().max_storage_buffers_per_shader_stage < 5 {
            return false;
        }
        self.wired.gpu_particles.scene.is_some()
            || frame.extensions.iter().any(|e| e.tag == GPU_PARTICLES_EXT_TAG && e.values.len() >= 16)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_gpu_particles_wired(
        wired: &mut WiredPasses,
        built_ins: &mut BuiltIns,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        active_proj: &Matrix,
        color_attachment: wgpu::RenderPassColorAttachment<'_>,
        depth_view: &wgpu::TextureView,
        depth_only: &wgpu::TextureView,
        size: (u32, u32),
        samples: u32,
        depth_mode: DepthMode,
    ) -> Result<()> {
        let ext_decoded = if wired.gpu_particles.scene.is_none() { decode_ext_gpu_particles(frame) } else { None };
        let (scene_cfg, pending_seed) = if let Some(scene) = wired.gpu_particles.scene.as_mut() {
            let seed = scene.pending_seed.take();
            (scene.clone(), seed)
        } else if let Some((mut ext_scene, force_reseed, seed_hash)) = ext_decoded {
            let need_seed = force_reseed
                || wired.gpu_particles.system.is_none()
                || (seed_hash != wired.gpu_particles.last_ext_seed_hash && ext_scene.pending_seed.is_some());
            let seed = if need_seed {
                wired.gpu_particles.last_ext_seed_hash = seed_hash;
                ext_scene.pending_seed.take()
            } else {
                None
            };
            (ext_scene, seed)
        } else {
            return Ok(());
        };

        let samples = samples.max(1);
        let reversed_z = depth_mode == DepthMode::ReversedZ;
        let mhash = mesh_hash(&scene_cfg.mesh_vertices, &scene_cfg.mesh_indices);
        let key = (scene_cfg.max_particles, samples, reversed_z, mhash);
        let recreated = wired.gpu_particles.system.is_none() || wired.gpu_particles.config_key != Some(key);
        if recreated {
            let target = match depth_mode {
                DepthMode::ForwardZ => PipelineTarget {
                    color_format: SCENE_FORMAT,
                    depth_format: Some(wgpu::TextureFormat::Depth24PlusStencil8),
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    sample_count: samples,
                },
                DepthMode::ReversedZ => {
                    PipelineTarget::reversed_z(SCENE_FORMAT, Some(wgpu::TextureFormat::Depth24PlusStencil8), samples)
                }
            };
            let sys = GpuParticleSystem::new(
                device,
                scene_cfg.max_particles,
                &scene_cfg.mesh_vertices,
                &scene_cfg.mesh_indices,
                target,
            )?;
            wired.gpu_particles.system = Some(sys);
            wired.gpu_particles.config_key = Some(key);
        }

        if let Some(seed) = pending_seed {
            wired.gpu_particles.last_seed_snapshot = seed.clone();
            if let Some(sys) = wired.gpu_particles.system.as_ref() {
                let capped = seed.len().min(scene_cfg.max_particles as usize);
                sys.seed(device, encoder, &seed[..capped])?;
            }
        } else if recreated
            && !wired.gpu_particles.last_seed_snapshot.is_empty()
            && let Some(sys) = wired.gpu_particles.system.as_ref()
        {
            let snap = &wired.gpu_particles.last_seed_snapshot;
            let capped = snap.len().min(scene_cfg.max_particles as usize);
            sys.seed(device, encoder, &snap[..capped])?;
        }

        let depth_r32 = wired.resolve_msaa_depth(device, built_ins, encoder, depth_only, size, samples);
        let active_vp = multiply(active_proj, &frame.view);
        let eye = invert(&frame.view);
        let cam_pos = [eye[12], eye[13], eye[14]];
        let sun_raw = [-frame.light_direction.x, -frame.light_direction.y, -frame.light_direction.z];
        let sl = (sun_raw[0] * sun_raw[0] + sun_raw[1] * sun_raw[1] + sun_raw[2] * sun_raw[2]).sqrt();
        let sun_dir = if sl > 1e-6 { [sun_raw[0] / sl, sun_raw[1] / sl, sun_raw[2] / sl] } else { [0.0, 1.0, 0.0] };
        let sun_col = [frame.light_color.x.max(0.0), frame.light_color.y.max(0.0), frame.light_color.z.max(0.0)];

        let dt = scene_cfg.dt.clamp(0.0, 0.1);
        wired.gpu_particles.sim_time += dt;
        let sim_params = GpuParticleSimParams {
            view_projection: active_vp,
            camera_pos: cam_pos,
            depth_clear: depth_mode.clear_value(),
            gravity: scene_cfg.gravity,
            dt,
            drag: scene_cfg.drag,
            curl_strength: scene_cfg.curl_strength,
            curl_frequency: scene_cfg.curl_frequency,
            time: wired.gpu_particles.sim_time,
            restitution: scene_cfg.restitution,
            friction: scene_cfg.friction,
            depth_thickness: scene_cfg.depth_thickness,
            reversed_z,
            depth_collide: scene_cfg.depth_collide,
            sub_emit_on_bounce: scene_cfg.sub_emit_on_bounce,
            sub_emit_on_death: scene_cfg.sub_emit_on_death,
            sub_emit_count: scene_cfg.sub_emit_count,
            sub_emit_speed: scene_cfg.sub_emit_speed,
            sub_emit_life: scene_cfg.sub_emit_life,
        };

        let sys = wired.gpu_particles.system.as_ref().unwrap();
        sys.step(device, encoder, &depth_r32, &sim_params)?;
        sys.prepare_draw(device, encoder, active_vp, sun_dir, sun_col, scene_cfg.ambient, scene_cfg.velocity_align);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gpu_particles"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
            }),
            ..Default::default()
        });
        sys.draw(&mut pass);
        Ok(())
    }
}
