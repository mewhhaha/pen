#![cfg_attr(not(feature = "render-meshlets"), allow(unused_imports, dead_code))]
use super::super::{GpuAsset, MeshletIndirectSlot, Scratch, Shape, Source, State, hiz, timing};
use super::{DagLodSettings, NANITE_EXT_TAG, OutlineScenes, PlannedDagDraw, PlannedMeshletDraw, WiredPasses, outlined};
use crate::camera_precision::DepthMode;
use crate::matrix::{invert, multiply};
use crate::meshlet_dag::{MeshletDagBatchEntry, MeshletDagCullParams, MeshletDagCuller, MeshletDagTwoPhaseOutput};
use crate::meshlets::{
    MeshletBatchEntry, MeshletCullParams, MeshletCuller, MeshletIndirectDraw, MeshletTwoPhaseBatchEntry,
};
use anyhow::Result;
use pen_protocol::{Matrix, RenderFrame};

fn resolve_nanite_override(frame: &RenderFrame, defaults: DagLodSettings) -> (bool, bool, f32) {
    let Some(ext) = frame.extensions.iter().rev().find(|e| e.tag == NANITE_EXT_TAG && !e.values.is_empty()) else {
        return (true, defaults.enabled, defaults.pixel_error_threshold);
    };
    let mode = ext.values[0];
    if !mode.is_finite() || mode < 0.5 {
        return (false, false, defaults.pixel_error_threshold);
    }
    if mode < 1.5 {
        return (true, false, defaults.pixel_error_threshold);
    }
    let threshold = ext
        .values
        .get(1)
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .map_or(defaults.pixel_error_threshold, |v| v.clamp(0.25, 16.0));
    (true, true, threshold)
}

fn perspective_near_plane(proj: &Matrix, depth_mode: DepthMode) -> Option<f32> {
    if !proj.iter().all(|v| v.is_finite()) || proj[11] >= -0.5 || proj[15].abs() > 0.5 || proj[5].abs() <= 1e-5 {
        return None;
    }
    let near = match depth_mode {
        DepthMode::ReversedZ => {
            let denom = 1.0 + proj[10];
            if denom.abs() > 1e-6 { proj[14] / denom } else { return None }
        }
        DepthMode::ForwardZ => {
            if proj[10].abs() > 1e-6 {
                proj[14] / proj[10]
            } else {
                return None;
            }
        }
    };
    (near.is_finite() && near > 1e-5).then_some(near)
}

impl State {
    /// Phase 1 of two-phase GPU meshlet cluster culling (local frustum + shear-safe normal-cone
    /// backface + prior-frame Hi-Z in both Forward-Z and Reversed-Z) plus continuous meshlet DAG
    /// screen-space error LOD cut selection for opaque LOD0 mesh draws.
    /// When prior-frame Hi-Z is active and matches `depth_mode`, records `phase2_flags` so Phase 2
    /// can recover newly disoccluded clusters on the exact same frame (0-frame lag).
    #[cfg(not(feature = "render-meshlets"))]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_meshlet_culling_phase1(
        &mut self,
        _device: &wgpu::Device,
        _encoder: &mut wgpu::CommandEncoder,
        _frame: &RenderFrame,
        _scratch: &mut Scratch,
        _outline_scenes: &OutlineScenes,
        _active_proj: &Matrix,
        _viewport_height: f32,
        _depth_mode: DepthMode,
    ) -> Result<()> {
        self.wired.last_meshlet_cull_draws = 0;
        self.wired.last_meshlet_dag_draws = 0;
        self.wired.meshlet_phase1_used_hiz = false;
        self.wired.dag_phase1_used_hiz = false;
        self.wired.last_meshlet_phase2_active = false;
        self.wired.meshlet_plan.clear();
        self.wired.dag_plan.clear();
        Ok(())
    }

    #[cfg(feature = "render-meshlets")]
    fn ensure_meshlet_gpu_assets(
        assets: &mut crate::asset_store::AssetStore<GpuAsset>,
        mesh_cache: &crate::mesh_cache::MeshCache,
        device: &wgpu::Device,
        mesh_slot: usize,
        want_meshlets: bool,
        want_dag: bool,
    ) {
        use wgpu::util::DeviceExt;
        let Some(GpuAsset::Mesh(mesh)) = assets.current_mut(mesh_slot) else {
            return;
        };
        if mesh.skin.is_some() || mesh.morphs.is_some() || mesh.instances.is_some() {
            return;
        }
        if want_dag && mesh.meshlet_dag.is_none() && !mesh.dag_attempted {
            mesh.dag_attempted = true;
            if let Some(baked_dag) = mesh.baked_meshlet_dag.take() {
                mesh.meshlet_dag = crate::meshlet_dag::MeshletDagGpu::upload(device, &baked_dag).ok();
            } else if let Some(src) = &mesh.rt_source
                && let Some(indices) = &src.indices
            {
                let (ml, dag) = mesh_cache.load_or_build(&src.vertices, indices, true);
                if let Some(dag) = dag {
                    mesh.meshlet_dag = crate::meshlet_dag::MeshletDagGpu::upload(device, &dag).ok();
                }
                if mesh.meshlets.is_none() && mesh.baked_meshlets.is_none() {
                    mesh.baked_meshlets = ml.map(Box::new);
                }
            }
        }
        let need_ml = want_meshlets && !(want_dag && mesh.meshlet_dag.is_some());
        if need_ml && mesh.meshlets.is_none() && !mesh.meshlets_attempted {
            mesh.meshlets_attempted = true;
            let ml = mesh.baked_meshlets.take().or_else(|| {
                let src = mesh.rt_source.as_ref()?;
                let indices = src.indices.as_ref()?;
                let (ml, dag) = mesh_cache.load_or_build(&src.vertices, indices, false);
                if mesh.meshlet_dag.is_none() && mesh.baked_meshlet_dag.is_none() {
                    mesh.baked_meshlet_dag = dag.map(Box::new);
                }
                ml.map(Box::new)
            });
            if let Some(ml) = ml {
                let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("meshlet_indices"),
                    contents: bytemuck::cast_slice(&ml.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                let descriptors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("meshlet_descriptors"),
                    contents: bytemuck::cast_slice(&ml.meshlets),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                mesh.meshlets = Some(super::super::MeshletGpuAsset {
                    indices,
                    descriptors,
                    count: u32::try_from(ml.meshlets.len()).unwrap_or(0),
                });
            }
        }
    }

    #[cfg(feature = "render-meshlets")]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_meshlet_culling_phase1(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scratch: &mut Scratch,
        outline_scenes: &OutlineScenes,
        active_proj: &Matrix,
        viewport_height: f32,
        depth_mode: DepthMode,
    ) -> Result<()> {
        self.wired.last_meshlet_cull_draws = 0;
        self.wired.last_meshlet_dag_draws = 0;
        self.wired.meshlet_phase1_used_hiz = false;
        self.wired.dag_phase1_used_hiz = false;
        self.wired.last_meshlet_phase2_active = false;
        self.wired.meshlet_plan.clear();
        self.wired.dag_plan.clear();

        let (allow_meshlets, allow_dag, pixel_error) = resolve_nanite_override(frame, self.wired.dag_lod);
        self.wired.dag_lod.frame_pixel_error = pixel_error;
        if !allow_meshlets && !allow_dag {
            return Ok(());
        }

        let active_vp = multiply(active_proj, &frame.view);
        let planes = frustum_planes_from_vp(&active_vp);
        let eye = invert(&frame.view);
        let camera_pos = [eye[12], eye[13], eye[14]];
        let reversed_z = depth_mode == DepthMode::ReversedZ;
        let (hiz_view_ref, hiz_view_id, hiz_vp, _, hiz_mips, hiz_active) = self.hiz.source();
        let use_hiz = hiz_active && self.hiz.depth_mode() == depth_mode;
        let hiz_opt = use_hiz.then_some((hiz_view_ref, hiz_view_id));
        let has_mdi = device.features().contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
        let perspective_near = perspective_near_plane(active_proj, depth_mode);

        let mut slot_cursor = 0usize;
        let mut dag_cursor = 0usize;
        for &ready_idx in &scratch.opaque {
            let entry = &scratch.ready[ready_idx];
            if entry.skin.is_some()
                || entry.morph.is_some()
                || matches!(self.assets.current(entry.shader), Some(GpuAsset::Shader(shader)) if shader.margin.is_some_and(|margin| margin > 0.0))
            {
                continue;
            }
            let item = &scratch.items[entry.item];
            // Outlined draws use the same discrete LOD for the scene and silhouette mask.
            if item.lod != 0
                || !matches!(item.shape, Shape::Mesh(_))
                || matches!(item.source, Source::Draw(index) if outlined(outline_scenes, index))
            {
                continue;
            }
            let Some(mesh_slot) = entry.mesh else {
                continue;
            };
            Self::ensure_meshlet_gpu_assets(
                &mut self.assets,
                &self.loader.mesh_cache,
                device,
                mesh_slot,
                allow_meshlets,
                allow_dag && perspective_near.is_some(),
            );
            let Some(GpuAsset::Mesh(mesh)) = self.assets.current(mesh_slot) else {
                continue;
            };
            if mesh.instances.is_some() {
                continue;
            }
            let model = *item.transform(frame);
            let det3 = model[0] * (model[5] * model[10] - model[6] * model[9])
                - model[4] * (model[1] * model[10] - model[2] * model[9])
                + model[8] * (model[1] * model[6] - model[2] * model[5]);
            if allow_dag
                && perspective_near.is_some()
                && let Some(dag) = &mesh.meshlet_dag
                && dag.cluster_count > 1
                && (has_mdi || dag.cluster_count <= 64)
            {
                if dag_cursor >= self.meshlet_dag_outputs.len() {
                    self.meshlet_dag_outputs.push(MeshletDagTwoPhaseOutput::new(device, dag));
                } else if self.meshlet_dag_outputs[dag_cursor].capacity() < dag.cluster_count {
                    self.meshlet_dag_outputs[dag_cursor] = MeshletDagTwoPhaseOutput::new(device, dag);
                }
                self.wired.dag_plan.push(PlannedDagDraw {
                    slot: dag_cursor,
                    mesh_slot,
                    model,
                    backface_cull: det3 > 0.0,
                });
                scratch.ready[ready_idx].dag_slot = Some(dag_cursor);
                dag_cursor += 1;
                continue;
            }
            if !allow_meshlets {
                continue;
            }
            let Some(ml) = &mesh.meshlets else {
                continue;
            };
            if ml.count <= 1 || (!has_mdi && ml.count > 64) {
                continue;
            }
            let need_draw_bytes = u64::from(ml.count) * size_of::<MeshletIndirectDraw>() as u64;
            let need_flag_bytes = u64::from(ml.count) * size_of::<u32>() as u64;
            let make_slot = |serial: u64, draw_cap: u64, flag_cap: u64| MeshletIndirectSlot {
                count: ml.count,
                generation: serial,
                draws_phase1: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("meshlet_indirect_draws_p1"),
                    size: draw_cap,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                    mapped_at_creation: false,
                }),
                draws_phase2: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("meshlet_indirect_draws_p2"),
                    size: draw_cap,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                    mapped_at_creation: false,
                }),
                phase2_flags: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("meshlet_phase2_flags"),
                    size: flag_cap,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }),
            };
            if slot_cursor >= self.meshlet_indirect.len() {
                self.wired.meshlet_indirect_serial += 1;
                self.meshlet_indirect.push(make_slot(
                    self.wired.meshlet_indirect_serial,
                    need_draw_bytes.max(1280),
                    need_flag_bytes.max(256),
                ));
            } else if self.meshlet_indirect[slot_cursor].draws_phase1.size() < need_draw_bytes
                || self.meshlet_indirect[slot_cursor].phase2_flags.size() < need_flag_bytes
            {
                self.wired.meshlet_indirect_serial += 1;
                self.meshlet_indirect[slot_cursor] = make_slot(
                    self.wired.meshlet_indirect_serial,
                    need_draw_bytes.next_power_of_two(),
                    need_flag_bytes.next_power_of_two(),
                );
            } else {
                self.meshlet_indirect[slot_cursor].count = ml.count;
            }
            let rev = self.assets.slots[mesh_slot].revision;
            let mesh_key = ((mesh_slot as u64) << 32) ^ rev;
            self.wired.meshlet_plan.push(PlannedMeshletDraw {
                slot: slot_cursor,
                mesh_slot,
                mesh_key,
                ml_count: ml.count,
                model,
                backface_cull: det3 > 0.0,
            });
            scratch.ready[ready_idx].meshlet_slot = Some(slot_cursor);
            slot_cursor += 1;
        }
        if let Some(near_plane) = perspective_near
            && !self.wired.dag_plan.is_empty()
            && crate::deferred_shaders::ready_all(&["meshlet_dag_cut.wgsl"])
        {
            let dag_culler = self.wired.meshlet_dag_culler.get_or_insert_with(|| MeshletDagCuller::new(device));
            let dag_two_phase = use_hiz && device.limits().max_storage_buffers_per_shader_stage >= 5;
            let mut dag_batch = Vec::with_capacity(self.wired.dag_plan.len());
            for p in &self.wired.dag_plan {
                let Some(GpuAsset::Mesh(mesh)) = self.assets.current(p.mesh_slot) else { continue };
                let Some(dag) = &mesh.meshlet_dag else { continue };
                dag_batch.push(MeshletDagBatchEntry {
                    mesh: dag,
                    output: &self.meshlet_dag_outputs[p.slot],
                    params: MeshletDagCullParams {
                        cull: MeshletCullParams {
                            model: p.model,
                            hiz_vp: if dag_two_phase { hiz_vp } else { active_vp },
                            frustum_planes: planes,
                            camera_pos,
                            hiz_enabled: dag_two_phase,
                            hiz_mips: if dag_two_phase { hiz_mips } else { 0 },
                            backface_cull: p.backface_cull,
                            reversed_z,
                        },
                        proj: *active_proj,
                        viewport_height: viewport_height.max(1.0),
                        near_plane,
                        pixel_error_threshold: self.wired.dag_lod.frame_pixel_error,
                    },
                });
            }
            if dag_two_phase {
                dag_culler.cull_batch_phase1(device, &self.queue, encoder, &dag_batch, Some(hiz_view_ref))?;
            } else {
                dag_culler.cull_batch_single(device, &self.queue, encoder, &dag_batch, None)?;
            }
            self.wired.dag_phase1_used_hiz = dag_two_phase;
        }
        self.wired.last_meshlet_dag_draws = u32::try_from(dag_cursor).unwrap_or(u32::MAX);
        if !self.wired.meshlet_plan.is_empty() && crate::deferred_shaders::ready_all(&["meshlet_cull.wgsl"]) {
            let culler = self.wired.meshlet_culler.get_or_insert_with(|| MeshletCuller::new(device));
            if use_hiz {
                let mut batch = Vec::with_capacity(self.wired.meshlet_plan.len());
                for p in &self.wired.meshlet_plan {
                    let Some(GpuAsset::Mesh(mesh)) = self.assets.current(p.mesh_slot) else { continue };
                    let Some(ml) = &mesh.meshlets else { continue };
                    let slot_data = &self.meshlet_indirect[p.slot];
                    batch.push(MeshletTwoPhaseBatchEntry {
                        mesh_key: p.mesh_key,
                        draws_gen: slot_data.generation,
                        meshlets_buf: &ml.descriptors,
                        draws_phase1_buf: &slot_data.draws_phase1,
                        draws_phase2_buf: &slot_data.draws_phase2,
                        phase2_flags_buf: &slot_data.phase2_flags,
                        meshlet_count: p.ml_count,
                        params: MeshletCullParams {
                            model: p.model,
                            hiz_vp,
                            frustum_planes: planes,
                            camera_pos,
                            hiz_enabled: true,
                            hiz_mips,
                            backface_cull: p.backface_cull,
                            reversed_z,
                        },
                    });
                }
                culler.cull_batch_phase1(
                    device,
                    &self.queue,
                    encoder,
                    hiz_opt,
                    &batch,
                    self.timer.compute(timing::Span::Culling),
                )?;
                self.wired.meshlet_phase1_used_hiz = true;
            } else {
                let mut batch = Vec::with_capacity(self.wired.meshlet_plan.len());
                for p in &self.wired.meshlet_plan {
                    let Some(GpuAsset::Mesh(mesh)) = self.assets.current(p.mesh_slot) else { continue };
                    let Some(ml) = &mesh.meshlets else { continue };
                    let slot_data = &self.meshlet_indirect[p.slot];
                    batch.push(MeshletBatchEntry {
                        mesh_key: p.mesh_key,
                        draws_gen: slot_data.generation,
                        meshlets_buf: &ml.descriptors,
                        draws_buf: &slot_data.draws_phase1,
                        meshlet_count: p.ml_count,
                        params: MeshletCullParams {
                            model: p.model,
                            hiz_vp: active_vp,
                            frustum_planes: planes,
                            camera_pos,
                            hiz_enabled: false,
                            hiz_mips: 0,
                            backface_cull: p.backface_cull,
                            reversed_z,
                        },
                    });
                }
                culler.cull_batch(
                    device,
                    &self.queue,
                    encoder,
                    None,
                    &batch,
                    self.timer.compute(timing::Span::Culling),
                )?;
            }
        }
        self.wired.last_meshlet_cull_draws = u32::try_from(slot_cursor).unwrap_or(u32::MAX);
        Ok(())
    }

    /// Phase 2 of two-phase GPU meshlet and continuous DAG occlusion culling: re-tests ONLY
    /// `phase2_flags == 1` candidates against the fresh current-frame Hi-Z built from Phase 1
    /// opaque depth so newly disoccluded clusters render in the exact same frame with zero overlap.
    #[cfg(not(feature = "render-meshlets"))]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_meshlet_culling_phase2(
        _wired: &mut WiredPasses,
        _hiz: &hiz::HiZ,
        _assets: &crate::asset_store::AssetStore<GpuAsset>,
        _meshlet_indirect: &[MeshletIndirectSlot],
        _meshlet_dag_outputs: &[MeshletDagTwoPhaseOutput],
        _timer: &timing::GpuTimer,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        _frame: &RenderFrame,
        _active_proj: &Matrix,
        _viewport_height: f32,
        _depth_mode: DepthMode,
    ) -> Result<()> {
        Ok(())
    }

    #[cfg(feature = "render-meshlets")]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_meshlet_culling_phase2(
        wired: &mut WiredPasses,
        hiz: &hiz::HiZ,
        assets: &crate::asset_store::AssetStore<GpuAsset>,
        meshlet_indirect: &[MeshletIndirectSlot],
        meshlet_dag_outputs: &[MeshletDagTwoPhaseOutput],
        timer: &timing::GpuTimer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        active_proj: &Matrix,
        viewport_height: f32,
        depth_mode: DepthMode,
    ) -> Result<()> {
        let run_meshlets = wired.meshlet_phase1_used_hiz && !wired.meshlet_plan.is_empty();
        let run_dag = wired.dag_phase1_used_hiz && !wired.dag_plan.is_empty();
        if !run_meshlets && !run_dag {
            return Ok(());
        }
        let active_vp = multiply(active_proj, &frame.view);
        let planes = frustum_planes_from_vp(&active_vp);
        let eye = invert(&frame.view);
        let camera_pos = [eye[12], eye[13], eye[14]];
        let reversed_z = depth_mode == DepthMode::ReversedZ;
        let (hiz_view_ref, hiz_view_id, hiz_vp, _, hiz_mips, hiz_active) = hiz.source();
        let use_hiz = hiz_active && hiz.depth_mode() == depth_mode;
        let hiz_opt = use_hiz.then_some((hiz_view_ref, hiz_view_id));

        if run_meshlets && let Some(culler) = wired.meshlet_culler.as_mut() {
            let mut batch = Vec::with_capacity(wired.meshlet_plan.len());
            for p in &wired.meshlet_plan {
                let Some(GpuAsset::Mesh(mesh)) = assets.current(p.mesh_slot) else { continue };
                let Some(ml) = &mesh.meshlets else { continue };
                let slot_data = &meshlet_indirect[p.slot];
                batch.push(MeshletTwoPhaseBatchEntry {
                    mesh_key: p.mesh_key,
                    draws_gen: slot_data.generation,
                    meshlets_buf: &ml.descriptors,
                    draws_phase1_buf: &slot_data.draws_phase1,
                    draws_phase2_buf: &slot_data.draws_phase2,
                    phase2_flags_buf: &slot_data.phase2_flags,
                    meshlet_count: p.ml_count,
                    params: MeshletCullParams {
                        model: p.model,
                        hiz_vp: if use_hiz { hiz_vp } else { active_vp },
                        frustum_planes: planes,
                        camera_pos,
                        hiz_enabled: use_hiz,
                        hiz_mips,
                        backface_cull: p.backface_cull,
                        reversed_z,
                    },
                });
            }
            culler.cull_batch_phase2(device, queue, encoder, hiz_opt, &batch, timer.compute(timing::Span::Culling))?;
            wired.last_meshlet_phase2_active = true;
        }
        if run_dag
            && let Some(near_plane) = perspective_near_plane(active_proj, depth_mode)
            && let Some(dag_culler) = wired.meshlet_dag_culler.as_mut()
        {
            let mut dag_batch = Vec::with_capacity(wired.dag_plan.len());
            for p in &wired.dag_plan {
                let Some(GpuAsset::Mesh(mesh)) = assets.current(p.mesh_slot) else { continue };
                let Some(dag) = &mesh.meshlet_dag else { continue };
                dag_batch.push(MeshletDagBatchEntry {
                    mesh: dag,
                    output: &meshlet_dag_outputs[p.slot],
                    params: MeshletDagCullParams {
                        cull: MeshletCullParams {
                            model: p.model,
                            hiz_vp: if use_hiz { hiz_vp } else { active_vp },
                            frustum_planes: planes,
                            camera_pos,
                            hiz_enabled: use_hiz,
                            hiz_mips: if use_hiz { hiz_mips } else { 0 },
                            backface_cull: p.backface_cull,
                            reversed_z,
                        },
                        proj: *active_proj,
                        viewport_height: viewport_height.max(1.0),
                        near_plane,
                        pixel_error_threshold: wired.dag_lod.frame_pixel_error,
                    },
                });
            }
            dag_culler.cull_batch_phase2(device, queue, encoder, &dag_batch, use_hiz.then_some(hiz_view_ref))?;
            wired.last_meshlet_phase2_active = true;
        }
        Ok(())
    }
}

/// Extracts six normalized inward-facing frustum planes from a column-major WebGPU `[0, 1]`
/// view-projection matrix (valid for both Forward-Z and Reversed-Z projections).
fn frustum_planes_from_vp(vp: &Matrix) -> [[f32; 4]; 6] {
    let row = |r: usize| [vp[r], vp[4 + r], vp[8 + r], vp[12 + r]];
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);
    let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
    let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
    let norm = |p: [f32; 4]| {
        let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if l > 1e-7 { [p[0] / l, p[1] / l, p[2] / l, p[3] / l] } else { [0.0; 4] }
    };
    [
        norm(add(r3, r0)), // Left:   x + w >= 0
        norm(sub(r3, r0)), // Right:  w - x >= 0
        norm(add(r3, r1)), // Bottom: y + w >= 0
        norm(sub(r3, r1)), // Top:    w - y >= 0
        norm(r2),          // Near/Far z >= 0
        norm(sub(r3, r2)), // Far/Near w - z >= 0
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_protocol::Extension;

    #[test]
    fn nanite_extension_overrides_meshlet_and_dag_modes() {
        let defaults = DagLodSettings::default();
        let plain = RenderFrame::default();
        assert_eq!(resolve_nanite_override(&plain, defaults), (true, true, 1.5));

        let off = RenderFrame {
            extensions: vec![Extension { tag: NANITE_EXT_TAG, values: vec![0.0] }],
            ..Default::default()
        };
        assert_eq!(resolve_nanite_override(&off, defaults), (false, false, 1.5));

        let meshlets_only = RenderFrame {
            extensions: vec![Extension { tag: NANITE_EXT_TAG, values: vec![1.0] }],
            ..Default::default()
        };
        assert_eq!(resolve_nanite_override(&meshlets_only, defaults), (true, false, 1.5));

        let dag_quality = RenderFrame {
            extensions: vec![Extension { tag: NANITE_EXT_TAG, values: vec![2.0, 0.75] }],
            ..Default::default()
        };
        assert_eq!(resolve_nanite_override(&dag_quality, defaults), (true, true, 0.75));

        let dag_clamped = RenderFrame {
            extensions: vec![Extension { tag: NANITE_EXT_TAG, values: vec![2.0, 99.0] }],
            ..Default::default()
        };
        assert_eq!(resolve_nanite_override(&dag_clamped, defaults), (true, true, 16.0));
    }
}
