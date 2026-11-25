use super::super::{GpuAsset, RtMeshSource, Scratch, Shape, Source, State};
use crate::ddgi::{DDGI_PACKED_HEADER_VEC4S, DdgiFrameParams, DdgiGridConfig, DdgiVolume};
use crate::matrix::{IDENTITY, invert};
use crate::ray_tracing::{GpuBlasPool, MAX_RT_LOCAL_LIGHTS, RtLocalLight, TlasInstance};
use crate::restir::RESTIR_MAX_LIGHTS;
use anyhow::Result;
use pen_protocol::RenderFrame;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Resolves sky zenith/horizon colors for DDGI and RT miss rays.
/// Falls back to `frame.ambient` (and subtle sky gradient when `physical_sky` is active) whenever
/// `frame.sky` was left at default zeros, matching `environment.rs::sky_floats`.
pub(super) fn resolve_rt_sky_colors(frame: &RenderFrame) -> ([f32; 3], [f32; 3]) {
    let z = [frame.sky.zenith.x.max(0.0), frame.sky.zenith.y.max(0.0), frame.sky.zenith.z.max(0.0)];
    let h = [frame.sky.horizon.x.max(0.0), frame.sky.horizon.y.max(0.0), frame.sky.horizon.z.max(0.0)];
    if z[0] + z[1] + z[2] + h[0] + h[1] + h[2] > 1e-4 {
        return (z, h);
    }
    let a = [frame.ambient.x.max(0.0), frame.ambient.y.max(0.0), frame.ambient.z.max(0.0)];
    let a_sum = a[0] + a[1] + a[2];
    if frame.quality.physical_sky && a_sum > 1e-4 {
        let zenith = [a[0] * 0.85, a[1] * 1.05, a[2] * 1.35];
        let horizon = [a[0] * 1.15, a[1] * 1.12, a[2] * 1.05];
        return (zenith, horizon);
    }
    (a, a)
}

/// Whether `frame` contains at least one active point or spot light suitable for ReSTIR DI.
pub(in crate::renderer) fn has_active_point_spot_lights(frame: &RenderFrame) -> bool {
    frame.lights.iter().any(|lt| lt.radius > 1e-4)
}

/// Packs ONLY dynamic point and spot lights (`frame.lights`) up to `RESTIR_MAX_LIGHTS`
/// so Tier-3 ReSTIR DI shades them with per-pixel RT shadows without double-lighting LTC area lights.
pub(super) fn pack_rt_point_spot_lights(frame: &RenderFrame) -> Vec<RtLocalLight> {
    let mut out = Vec::with_capacity(frame.lights.len().min(RESTIR_MAX_LIGHTS));
    for lt in &frame.lights {
        if out.len() >= RESTIR_MAX_LIGHTS {
            break;
        }
        let reach = lt.radius.max(0.0);
        if reach <= 1e-4 {
            continue;
        }
        let dir_raw = [lt.direction.x, lt.direction.y, lt.direction.z];
        let dlen = (dir_raw[0] * dir_raw[0] + dir_raw[1] * dir_raw[1] + dir_raw[2] * dir_raw[2]).sqrt();
        let dir =
            if dlen > 1e-6 { [dir_raw[0] / dlen, dir_raw[1] / dlen, dir_raw[2] / dlen] } else { [0.0, -1.0, 0.0] };
        let outer = if lt.outer > 0.0 && dlen > 1e-6 { lt.outer } else { -1.0 };
        let inner = if outer > -1.0 { lt.inner.max(outer + 1e-4) } else { -1.0 };
        out.push(RtLocalLight {
            position_range: [lt.position.x, lt.position.y, lt.position.z, reach],
            color_outer_cos: [lt.color.x.max(0.0), lt.color.y.max(0.0), lt.color.z.max(0.0), outer],
            direction_inner_cos: [dir[0], dir[1], dir[2], inner],
        });
    }
    out
}

/// Packs dynamic point, spot, and area lights from `RenderFrame` into compact `RtLocalLight` records
/// injected into DDGI probe hit shading and RT reflection/GI bounce hits.
pub(super) fn pack_rt_local_lights(frame: &RenderFrame) -> Vec<RtLocalLight> {
    let mut out = Vec::with_capacity((frame.lights.len() + frame.area_lights.len()).min(MAX_RT_LOCAL_LIGHTS));
    for lt in &frame.lights {
        if out.len() >= MAX_RT_LOCAL_LIGHTS {
            break;
        }
        let reach = lt.radius.max(0.0);
        if reach <= 1e-4 {
            continue;
        }
        let dir_raw = [lt.direction.x, lt.direction.y, lt.direction.z];
        let dlen = (dir_raw[0] * dir_raw[0] + dir_raw[1] * dir_raw[1] + dir_raw[2] * dir_raw[2]).sqrt();
        let dir =
            if dlen > 1e-6 { [dir_raw[0] / dlen, dir_raw[1] / dlen, dir_raw[2] / dlen] } else { [0.0, -1.0, 0.0] };
        let outer = if lt.outer > 0.0 && dlen > 1e-6 { lt.outer } else { -1.0 };
        let inner = if outer > -1.0 { lt.inner.max(outer + 1e-4) } else { -1.0 };
        out.push(RtLocalLight {
            position_range: [lt.position.x, lt.position.y, lt.position.z, reach],
            color_outer_cos: [lt.color.x.max(0.0), lt.color.y.max(0.0), lt.color.z.max(0.0), outer],
            direction_inner_cos: [dir[0], dir[1], dir[2], inner],
        });
    }
    for al in &frame.area_lights {
        if out.len() >= MAX_RT_LOCAL_LIGHTS {
            break;
        }
        let reach = al.range.max(0.0);
        let gain = al.intensity.max(0.0);
        if reach <= 1e-4 || gain <= 1e-5 {
            continue;
        }
        let area_scale = (al.width.max(0.1) * al.height.max(0.1)).clamp(0.25, 8.0) * 0.5;
        let rx = al.right.x;
        let ry = al.right.y;
        let rz = al.right.z;
        let ux = al.up.x;
        let uy = al.up.y;
        let uz = al.up.z;
        let nx = ry * uz - rz * uy;
        let ny = rz * ux - rx * uz;
        let nz = rx * uy - ry * ux;
        let nlen = (nx * nx + ny * ny + nz * nz).sqrt();
        let n = if nlen > 1e-6 { [nx / nlen, ny / nlen, nz / nlen] } else { [0.0, -1.0, 0.0] };
        let (outer, inner) = if al.two_sided || al.kind == 2 { (-1.0, -1.0) } else { (0.0, 0.25) };
        out.push(RtLocalLight {
            position_range: [al.position.x, al.position.y, al.position.z, reach],
            color_outer_cos: [
                al.color.x.max(0.0) * gain * area_scale,
                al.color.y.max(0.0) * gain * area_scale,
                al.color.z.max(0.0) * gain * area_scale,
                outer,
            ],
            direction_inner_cos: [n[0], n[1], n[2], inner],
        });
    }
    out
}

impl State {
    /// Ensures every opaque mesh has a resident BLAS in the persistent `GpuBlasPool` (building &
    /// uploading each unique mesh ID once) and rebuilds ONLY the lightweight TLAS + hardware
    /// `wgpu::Tlas` when instance transforms or material properties change.
    pub(in crate::renderer) fn prepare_tlas_scene(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scratch: &Scratch,
    ) -> Result<bool> {
        self.wired.ddgi_active_this_frame = false;
        if (frame.quality.ray_tracing == 0 || !crate::ray_tracing::TlasRayTracer::supported(&self.device))
            && !(frame.quality.ddgi && DdgiVolume::supported(&self.device))
        {
            return Ok(false);
        }
        let device = &self.device;
        let pool = self.wired.blas_pool.get_or_insert_with(|| GpuBlasPool::new(device));
        let mut instances: Vec<TlasInstance> = Vec::new();
        let mut world_min = [f32::INFINITY; 3];
        let mut world_max = [f32::NEG_INFINITY; 3];

        let mut push_inst = |pool: &mut GpuBlasPool,
                             source: &Arc<RtMeshSource>,
                             transform: [f32; 16],
                             m: &crate::asset_store::Material,
                             draw_params: Option<[f32; 4]>|
         -> Result<()> {
            let Some(bi) = pool.ensure_blas(device, encoder, source.id, &source.triangles)? else {
                return Ok(());
            };
            let inv = invert(&transform);
            if !inv.iter().all(|v| v.is_finite()) {
                return Ok(());
            }
            let (bmin, bmax) = pool.blas_bounds(bi);
            for &x in &[bmin[0], bmax[0]] {
                for &y in &[bmin[1], bmax[1]] {
                    for &z in &[bmin[2], bmax[2]] {
                        let wx = transform[0] * x + transform[4] * y + transform[8] * z + transform[12];
                        let wy = transform[1] * x + transform[5] * y + transform[9] * z + transform[13];
                        let wz = transform[2] * x + transform[6] * y + transform[10] * z + transform[14];
                        world_min[0] = world_min[0].min(wx);
                        world_min[1] = world_min[1].min(wy);
                        world_min[2] = world_min[2].min(wz);
                        world_max[0] = world_max[0].max(wx);
                        world_max[1] = world_max[1].max(wy);
                        world_max[2] = world_max[2].max(wz);
                    }
                }
            }
            #[allow(clippy::cast_possible_truncation, reason = "clamped finite material color")]
            let albedo = [
                m.color[0].clamp(0.0, 10.0) as f32,
                m.color[1].clamp(0.0, 10.0) as f32,
                m.color[2].clamp(0.0, 10.0) as f32,
            ];
            let emissive = if !m.lit {
                let s = m.surface.emissive_strength.max(1.0);
                [albedo[0] * s, albedo[1] * s, albedo[2] * s]
            } else if m.surface.emissive_strength > 1.0 {
                let s = m.surface.emissive_strength - 1.0;
                [albedo[0] * s, albedo[1] * s, albedo[2] * s]
            } else {
                [0.0; 3]
            };
            #[allow(clippy::cast_possible_truncation, reason = "finite material parameters")]
            let (mut metallic, mut roughness) = match m.instance_parameters {
                Some(p) => {
                    let met = (p[0] as f32).clamp(0.0, 1.0);
                    let rgh = if p[1] > 0.0 { (p[1] as f32).clamp(0.04, 1.0) } else { 0.5 };
                    (met, rgh)
                }
                None => (0.0, 0.5),
            };
            if let Some(p) = draw_params {
                if p[0].is_finite() {
                    metallic = p[0].clamp(0.0, 1.0);
                }
                if p[1].is_finite() && p[1] > 0.0 {
                    roughness = p[1].clamp(0.04, 1.0);
                }
            }
            if m.surface.clearcoat > 0.0 {
                roughness = roughness.min(m.surface.clearcoat_roughness.clamp(0.04, 0.35));
            }
            let mut inst = TlasInstance::new(bi, transform);
            inst.albedo = albedo;
            inst.emissive = emissive;
            inst.roughness = roughness;
            inst.metallic = metallic;
            instances.push(inst);
            Ok(())
        };

        for &ready_idx in &scratch.opaque {
            let entry = &scratch.ready[ready_idx];
            let item = &scratch.items[entry.item];
            let mat = &entry.material.material;
            match item.shape {
                Shape::Mesh(_) => {
                    if let Some(GpuAsset::Mesh(mesh)) = entry.mesh.and_then(|slot| self.assets.current(slot))
                        && let Some(src) = &mesh.rt_source
                    {
                        let draw_params = match item.source {
                            Source::Draw(idx) => frame.draws.get(idx).and_then(|d| d.parameters),
                            Source::Inset(inset_idx, idx) => frame
                                .insets
                                .get(inset_idx)
                                .and_then(|ins| ins.draws.get(idx))
                                .and_then(|d| d.parameters),
                            Source::World => None,
                        };
                        push_inst(pool, src, *item.transform(frame), mat, draw_params)?;
                    }
                }
                Shape::Solid(idx) => {
                    if let Some(solid) = self.solid_meshes.get(idx)
                        && let Some(src) = &solid.rt_source
                    {
                        push_inst(pool, src, IDENTITY, mat, None)?;
                    }
                }
                Shape::Batch(batch_idx) => {
                    let batch = &scratch.batches[batch_idx];
                    if let Some(GpuAsset::Mesh(mesh)) = self.assets.current(batch.mesh)
                        && let Some(src) = &mesh.rt_source
                    {
                        for (draw_idx, placement) in scratch.placements.iter().enumerate() {
                            if placement.batch as usize == batch_idx
                                && let Some(draw) = frame.draws.get(draw_idx)
                            {
                                push_inst(pool, src, draw.transform, mat, draw.parameters)?;
                            }
                        }
                    }
                }
            }
        }

        if instances.is_empty() {
            self.wired.tlas_gpu = None;
            self.wired.tlas_bounds = None;
            return Ok(false);
        }

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        instances.len().hash(&mut hasher);
        for inst in &instances {
            inst.blas_index.hash(&mut hasher);
            for v in inst.transform {
                v.to_bits().hash(&mut hasher);
            }
            for v in inst.albedo {
                v.to_bits().hash(&mut hasher);
            }
            for v in inst.emissive {
                v.to_bits().hash(&mut hasher);
            }
            inst.roughness.to_bits().hash(&mut hasher);
            inst.metallic.to_bits().hash(&mut hasher);
        }
        let scene_hash = hasher.finish();
        if self.wired.tlas_gpu.is_none() || self.wired.tlas_hash != scene_hash {
            pool.update_tlas(device, encoder, &instances)?;
            self.wired.tlas_gpu = Some(pool.snapshot_scene());
            self.wired.tlas_hash = scene_hash;
        }
        self.wired.tlas_bounds = Some((world_min, world_max));
        Ok(true)
    }

    /// Updates live GPU DDGI probe cascade(s) BEFORE the forward PBR scene, reflection-probe
    /// capture, and volumetric fog passes, and packs them into `environment.probe_data_buffer()`
    /// so every lit surface receives albedo-modulated, normal-mapped, Fresnel-correct multi-bounce GI
    /// without consuming extra sampled textures or double-counting ambient.
    pub(in crate::renderer) fn encode_ddgi_prepass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
    ) -> Result<()> {
        let Some(tlas) = self.wired.tlas_gpu.as_ref() else {
            return Ok(());
        };
        let eye = invert(&frame.view);
        let camera_pos = [eye[12], eye[13], eye[14]];
        let sun_raw = [-frame.light_direction.x, -frame.light_direction.y, -frame.light_direction.z];
        let sun_len = (sun_raw[0] * sun_raw[0] + sun_raw[1] * sun_raw[1] + sun_raw[2] * sun_raw[2]).sqrt();
        let sun_direction = if sun_len > 1e-6 {
            [sun_raw[0] / sun_len, sun_raw[1] / sun_len, sun_raw[2] / sun_len]
        } else {
            [0.0, 1.0, 0.0]
        };
        let sun_color = [frame.light_color.x.max(0.0), frame.light_color.y.max(0.0), frame.light_color.z.max(0.0)];
        let (sky_zenith, sky_horizon) = resolve_rt_sky_colors(frame);
        let local_lights = pack_rt_local_lights(frame);

        let ddgi_params = DdgiFrameParams {
            sun_direction,
            sun_color,
            sky_zenith,
            sky_horizon,
            visibility_mask: u32::MAX,
            reset_history: frame.exposure.cut,
        };

        let pv = frame.probe_volume;
        let explicit_pv = pv.nx > 0 && pv.ny > 0 && pv.nz > 0 && pv.spacing > 0.0;
        let use_dual_cascade = !explicit_pv
            && self.wired.tlas_bounds.is_some_and(|(bmin, bmax)| (bmax[0] - bmin[0]).max(bmax[2] - bmin[2]) > 16.0);

        if explicit_pv {
            let dims = [pv.nx.clamp(2, 8), pv.ny.clamp(2, 4), pv.nz.clamp(2, 8)];
            let origin = [pv.origin.x, pv.origin.y, pv.origin.z];
            let sp = pv.spacing.max(0.25);
            let desired = DdgiGridConfig {
                origin,
                spacing: [sp, sp, sp],
                dims,
                rays_per_probe: 96,
                hysteresis: 0.85,
                ..Default::default()
            };
            let need_recreate = match &self.wired.ddgi_near {
                None => true,
                Some(existing) => {
                    let c = existing.config();
                    c.dims != desired.dims
                        || (0..3).any(|k| (c.origin[k] - desired.origin[k]).abs() > 1e-2)
                        || (0..3).any(|k| (c.spacing[k] - desired.spacing[k]).abs() > 1e-2)
                }
            };
            if need_recreate {
                self.wired.ddgi_near = Some(DdgiVolume::new(&self.device, desired)?);
            }
            self.wired.ddgi_far = None;
        } else if use_dual_cascade {
            let (bmin, bmax) = self.wired.tlas_bounds.unwrap();
            let near_dims = [6u32, 3, 6];
            let near_sp = [2.5f32, 2.0, 2.5];
            let cam_y = camera_pos[1].clamp(bmin[1] + 0.5, bmax[1].max(bmin[1] + 1.0));
            let target_center = [camera_pos[0], cam_y, camera_pos[2]];
            let need_near = !matches!(&self.wired.ddgi_near, Some(v) if v.config().dims == near_dims
                && (v.config().spacing[0] - near_sp[0]).abs() < 1e-2
                && (v.config().spacing[1] - near_sp[1]).abs() < 1e-2);
            if need_near {
                let origin = [
                    target_center[0] - 0.5 * (near_dims[0] - 1) as f32 * near_sp[0],
                    target_center[1] - 0.5 * (near_dims[1] - 1) as f32 * near_sp[1],
                    target_center[2] - 0.5 * (near_dims[2] - 1) as f32 * near_sp[2],
                ];
                let cfg = DdgiGridConfig {
                    origin,
                    spacing: near_sp,
                    dims: near_dims,
                    rays_per_probe: 64,
                    hysteresis: 0.88,
                    ..Default::default()
                };
                self.wired.ddgi_near = Some(DdgiVolume::new(&self.device, cfg)?);
            } else if let Some(near) = &mut self.wired.ddgi_near {
                near.scroll_to_center(target_center);
            }

            let far_dims = [5u32, 3, 5];
            let y_span = (bmax[1] - bmin[1]).max(1.0);
            let pad =
                [((bmax[0] - bmin[0]) * 0.08).max(0.5), (y_span * 0.1).max(0.4), ((bmax[2] - bmin[2]) * 0.08).max(0.5)];
            let lo = [bmin[0] - pad[0], bmin[1] + (y_span * 0.1).min(0.55), bmin[2] - pad[2]];
            let hi = [bmax[0] + pad[0], bmax[1] + pad[1], bmax[2] + pad[2]];
            let far_sp = [
                ((hi[0] - lo[0]) / (far_dims[0] - 1) as f32).max(1.0),
                ((hi[1] - lo[1]) / (far_dims[1] - 1) as f32).max(0.75),
                ((hi[2] - lo[2]) / (far_dims[2] - 1) as f32).max(1.0),
            ];
            let need_far = match &self.wired.ddgi_far {
                None => true,
                Some(existing) => {
                    let c = existing.config();
                    c.dims != far_dims
                        || (0..3).any(|k| (c.origin[k] - lo[k]).abs() > 2.0)
                        || (0..3).any(|k| (c.spacing[k] - far_sp[k]).abs() > 1.0)
                }
            };
            if need_far {
                let cfg = DdgiGridConfig {
                    origin: lo,
                    spacing: far_sp,
                    dims: far_dims,
                    rays_per_probe: 64,
                    hysteresis: 0.90,
                    ..Default::default()
                };
                self.wired.ddgi_far = Some(DdgiVolume::new(&self.device, cfg)?);
            }
        } else {
            let desired = if let Some((bmin, bmax)) = self.wired.tlas_bounds {
                let dims = [8u32, 4, 8];
                let y_span = (bmax[1] - bmin[1]).max(1.0);
                let pad_x = ((bmax[0] - bmin[0]) * 0.04).max(0.25);
                let pad_z = ((bmax[2] - bmin[2]) * 0.04).max(0.25);
                let pad_top = (y_span * 0.08).max(0.35);
                // Keep bottom probe layer safely above solid floor slabs rather than underground.
                let lo_y = bmin[1] + (y_span * 0.13).min(0.55);
                let lo = [bmin[0] - pad_x, lo_y, bmin[2] - pad_z];
                let hi = [bmax[0] + pad_x, bmax[1] + pad_top, bmax[2] + pad_z];
                let spacing = [
                    ((hi[0] - lo[0]) / (dims[0] - 1) as f32).max(0.45),
                    ((hi[1] - lo[1]) / (dims[1] - 1) as f32).max(0.45),
                    ((hi[2] - lo[2]) / (dims[2] - 1) as f32).max(0.45),
                ];
                DdgiGridConfig {
                    origin: lo,
                    spacing,
                    dims,
                    rays_per_probe: 128,
                    hysteresis: 0.85,
                    ..Default::default()
                }
            } else {
                DdgiGridConfig { rays_per_probe: 128, hysteresis: 0.85, ..Default::default() }
            };
            let need_recreate = match &self.wired.ddgi_near {
                None => true,
                Some(existing) => {
                    let c = existing.config();
                    c.dims != desired.dims
                        || (0..3).any(|k| (c.origin[k] - desired.origin[k]).abs() > 0.5)
                        || (0..3).any(|k| (c.spacing[k] - desired.spacing[k]).abs() > 0.25)
                }
            };
            if need_recreate {
                self.wired.ddgi_near = Some(DdgiVolume::new(&self.device, desired)?);
            }
            self.wired.ddgi_far = None;
        }

        let probe_buf = self.environment.probe_data_buffer();
        let cap_vec4s = self.environment.probe_data_capacity_vec4s();
        let cascade_count = if self.wired.ddgi_far.is_some() { 2u32 } else { 1u32 };

        let mut cursor = DDGI_PACKED_HEADER_VEC4S;
        if let Some(near) = &mut self.wired.ddgi_near {
            near.update_with_lights(&self.device, encoder, tlas, &ddgi_params, &local_lights);
            let (irr_n, vis_n) = near.packed_data_vec4_counts();
            let off_n = near.packed_probe_offset_vec4_count();
            if cursor + irr_n + vis_n + off_n <= cap_vec4s {
                near.pack_cascade(&self.device, encoder, probe_buf, 0, cascade_count, cursor, cursor + irr_n, 1.0);
                cursor += irr_n + vis_n + off_n;
            }
        }
        if let Some(far) = &mut self.wired.ddgi_far {
            far.update_with_lights(&self.device, encoder, tlas, &ddgi_params, &local_lights);
            let (irr_f, vis_f) = far.packed_data_vec4_counts();
            let off_f = far.packed_probe_offset_vec4_count();
            if cursor + irr_f + vis_f + off_f <= cap_vec4s {
                far.pack_cascade(&self.device, encoder, probe_buf, 1, cascade_count, cursor, cursor + irr_f, 1.0);
            }
        }

        self.environment.set_ddgi_active(&self.queue, true);
        self.wired.ddgi_active_this_frame = true;
        Ok(())
    }
}
