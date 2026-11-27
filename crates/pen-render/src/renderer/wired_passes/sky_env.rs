#![cfg_attr(not(all(feature = "render-atmosphere", feature = "render-env")), allow(unused_imports, dead_code))]
use super::super::State;
use super::{ENV_FOLIAGE_EXT_TAG, ENV_TERRAIN_EXT_TAG, ENV_WATER_EXT_TAG, WiredPasses};
use crate::atmosphere_lut::{
    AtmosphereParams, BASE_CAMERA_ALTITUDE_KM, PhysicalAtmosphere, WORLD_METERS_TO_KM, physical_camera_km,
};
use crate::camera_precision::DepthMode;
use crate::environment_passes::{
    FoliageInstance, FoliagePipeline, FoliageUniform, PipelineTarget, TerrainPatch, TerrainPipeline, TerrainUniform,
    WaterPipeline, WaterUniform,
};
use crate::matrix::{invert, multiply};
use crate::post::{PhysicalSkyBinding, SCENE_FORMAT};
use anyhow::Result;
use pen_protocol::{Matrix, RenderFrame};
use std::hash::{Hash, Hasher};

#[cfg(feature = "render-env")]
pub(in crate::renderer) struct EnvPipelines {
    pub terrain: TerrainPipeline,
    pub water: WaterPipeline,
    pub foliage: FoliagePipeline,
}

#[cfg(feature = "render-env")]
impl EnvPipelines {
    fn new(device: &wgpu::Device, samples: u32, depth_mode: DepthMode) -> Self {
        let target = match depth_mode {
            DepthMode::ForwardZ => PipelineTarget {
                color_format: SCENE_FORMAT,
                depth_format: Some(wgpu::TextureFormat::Depth24PlusStencil8),
                depth_compare: wgpu::CompareFunction::LessEqual,
                sample_count: samples.max(1),
            },
            DepthMode::ReversedZ => {
                PipelineTarget::reversed_z(SCENE_FORMAT, Some(wgpu::TextureFormat::Depth24PlusStencil8), samples.max(1))
            }
        };
        Self {
            terrain: TerrainPipeline::new(device, target),
            water: WaterPipeline::new(device, target),
            foliage: FoliagePipeline::new(device, target),
        }
    }
}

impl State {
    /// Ensures physical atmosphere LUTs & optional cloud target exist before `Post::begin`
    /// builds the pass graph, returning their views and generation key for `BuiltIn::Atmosphere`.
    #[cfg(not(feature = "render-atmosphere"))]
    pub(in crate::renderer) fn prepare_physical_sky(
        &mut self,
        _frame: &RenderFrame,
        _scene_size: (u32, u32),
    ) -> Result<Option<PhysicalSkyBinding>> {
        Ok(None)
    }

    /// Ensures physical atmosphere LUTs & optional cloud target exist before `Post::begin`
    /// builds the pass graph, returning their views and generation key for `BuiltIn::Atmosphere`.
    #[cfg(feature = "render-atmosphere")]
    pub(in crate::renderer) fn prepare_physical_sky(
        &mut self,
        frame: &RenderFrame,
        scene_size: (u32, u32),
    ) -> Result<Option<PhysicalSkyBinding>> {
        if !frame.quality.physical_sky {
            return Ok(None);
        }
        let device = self.device.clone();
        let atmo = self.wired.physical_atmosphere.get_or_insert_with(|| PhysicalAtmosphere::new(&device));
        let wants_clouds = frame.clouds.cover > 0.001;
        let cloud_serial = if wants_clouds {
            let cw = (scene_size.0 / 2).clamp(16, 1024);
            let ch = (scene_size.1 / 2).clamp(16, 1024);
            atmo.ensure_clouds(&device, (cw, ch))?
        } else {
            0
        };
        let key = (wants_clouds, cloud_serial);
        if self.wired.physical_sky_serial == 0 || self.wired.physical_sky_last_key != key {
            self.wired.physical_sky_serial += 1;
            self.wired.physical_sky_last_key = key;
        }
        Ok(Some(PhysicalSkyBinding {
            serial: self.wired.physical_sky_serial,
            sky_view: atmo.sky_view_view().clone(),
            transmittance: atmo.transmittance_view().clone(),
            clouds: if wants_clouds { atmo.cloud_view().cloned() } else { None },
        }))
    }

    /// Updates Hillaire physical atmosphere LUTs & volumetric clouds when `quality.physical_sky` is on.
    /// Uses the active (potentially TAA-jittered) projection and unified world-meters-to-km camera coordinates.
    #[cfg(not(feature = "render-atmosphere"))]
    pub(in crate::renderer) fn encode_physical_sky(
        &mut self,
        _encoder: &mut wgpu::CommandEncoder,
        _frame: &RenderFrame,
        _active_projection: &Matrix,
        _scene_size: (u32, u32),
    ) -> Result<()> {
        Ok(())
    }

    /// Updates Hillaire physical atmosphere LUTs & volumetric clouds when `quality.physical_sky` is on.
    /// Uses the active (potentially TAA-jittered) projection and unified world-meters-to-km camera coordinates.
    #[cfg(feature = "render-atmosphere")]
    pub(in crate::renderer) fn encode_physical_sky(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        active_projection: &Matrix,
        scene_size: (u32, u32),
    ) -> Result<()> {
        if !frame.quality.physical_sky {
            return Ok(());
        }
        let device = self.device.clone();
        let atmo = self.wired.physical_atmosphere.get_or_insert_with(|| PhysicalAtmosphere::new(&device));
        let eye = invert(&frame.view);
        let world_eye = [eye[12], eye[13], eye[14]];
        let cam_km = physical_camera_km(world_eye);
        let mut params = AtmosphereParams {
            camera_altitude_km: cam_km[1],
            sun_direction: [-frame.light_direction.x, -frame.light_direction.y, -frame.light_direction.z],
            sun_irradiance: [frame.light_color.x, frame.light_color.y, frame.light_color.z],
            cloud_coverage: frame.clouds.cover.clamp(0.0, 1.0),
            cloud_wind_km: [frame.clouds.drift.x * 0.01, 0.0, frame.clouds.drift.z * 0.01],
            ..Default::default()
        };
        if frame.clouds.cover <= 0.001 {
            params.cloud_density = 0.0;
        }
        atmo.update_luts(&device, encoder, &params);
        if frame.clouds.cover > 0.001 {
            let vp = multiply(active_projection, &frame.view);
            let inv_vp = invert(&vp);
            let cw = (scene_size.0 / 2).clamp(16, 1024);
            let ch = (scene_size.1 / 2).clamp(16, 1024);
            atmo.render_clouds_world(
                &device,
                encoder,
                &params,
                inv_vp,
                world_eye,
                WORLD_METERS_TO_KM,
                BASE_CAMERA_ALTITUDE_KM,
                (cw, ch),
            )?;
        }
        Ok(())
    }

    /// Encodes outdoor GPU environment passes (CDLOD terrain, Gerstner water, GPU foliage)
    /// driven by either [`Renderer::set_environment_scene`] or optional Blot `RenderFrame::extensions`.
    #[cfg(not(feature = "render-env"))]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_environment_passes(
        _wired: &mut WiredPasses,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _pass: &mut wgpu::RenderPass<'_>,
        _frame: &RenderFrame,
        _active_proj: &Matrix,
        _samples: u32,
        _depth_mode: DepthMode,
    ) -> Result<()> {
        Ok(())
    }

    /// Encodes outdoor GPU environment passes (CDLOD terrain, Gerstner water, GPU foliage)
    /// driven by either [`Renderer::set_environment_scene`] or optional Blot `RenderFrame::extensions`.
    #[cfg(feature = "render-env")]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_environment_passes(
        wired: &mut WiredPasses,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        frame: &RenderFrame,
        active_proj: &Matrix,
        samples: u32,
        depth_mode: DepthMode,
    ) -> Result<()> {
        let has_ext_env = frame.extensions.iter().any(|e| {
            matches!(e.tag, ENV_TERRAIN_EXT_TAG | ENV_WATER_EXT_TAG | ENV_FOLIAGE_EXT_TAG) && !e.values.is_empty()
        });
        if wired.env_scene.is_none() && !has_ext_env {
            return Ok(());
        }
        let key = (samples.max(1), depth_mode == DepthMode::ReversedZ);
        let pipes = wired.env_pipelines.entry(key).or_insert_with(|| EnvPipelines::new(device, samples, depth_mode));
        pipes.terrain.begin_frame();
        pipes.water.begin_frame();
        pipes.foliage.begin_frame();

        let active_vp = multiply(active_proj, &frame.view);
        let eye = invert(&frame.view);
        let cam_pos = [eye[12], eye[13], eye[14]];
        let sun_raw = [-frame.light_direction.x, -frame.light_direction.y, -frame.light_direction.z];
        let sl = (sun_raw[0] * sun_raw[0] + sun_raw[1] * sun_raw[1] + sun_raw[2] * sun_raw[2]).sqrt();
        let sun_dir = if sl > 1e-6 { [sun_raw[0] / sl, sun_raw[1] / sl, sun_raw[2] / sl] } else { [0.0, 1.0, 0.0] };
        let sun_col = [frame.light_color.x.max(0.0), frame.light_color.y.max(0.0), frame.light_color.z.max(0.0)];

        if let Some(scene) = &wired.env_scene {
            if let Some(terrain) = &scene.terrain {
                let mut u = terrain.uniform;
                if u.view_projection.iter().all(|v| *v == 0.0) {
                    u.view_projection = active_vp;
                    u.camera_pos_and_fog = [cam_pos[0], cam_pos[1], cam_pos[2], u.camera_pos_and_fog[3]];
                }
                pipes.terrain.draw(device, queue, pass, &u, &terrain.patches, &terrain.heightmap)?;
            }
            if let Some(fol) = &scene.foliage {
                let mut u = fol.uniform;
                if u.view_projection.iter().all(|v| *v == 0.0) {
                    u.view_projection = active_vp;
                    u.camera_pos_and_time = [cam_pos[0], cam_pos[1], cam_pos[2], u.camera_pos_and_time[3]];
                }
                pipes.foliage.draw(device, queue, pass, &u, &fol.instances)?;
            }
            for &wu in &scene.water {
                let mut u = wu;
                if u.view_projection.iter().all(|v| *v == 0.0) {
                    u.view_projection = active_vp;
                    u.camera_pos_and_time = [cam_pos[0], cam_pos[1], cam_pos[2], u.camera_pos_and_time[3]];
                }
                pipes.water.draw(queue, pass, &u)?;
            }
        }

        // Decode optional Blot environment extension sections.
        for ext in &frame.extensions {
            match ext.tag {
                ENV_TERRAIN_EXT_TAG if ext.values.len() >= 12 => {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let gw = (ext.values[0].round().clamp(2.0, 512.0)) as u32;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let gh = (ext.values[1].round().clamp(2.0, 512.0)) as u32;
                    let expected = (gw as usize) * (gh as usize);
                    if ext.values.len() >= 8 + expected {
                        let heights = &ext.values[8..8 + expected];
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        gw.hash(&mut hasher);
                        gh.hash(&mut hasher);
                        for &h in heights {
                            h.to_bits().hash(&mut hasher);
                        }
                        let h_hash = hasher.finish();
                        if !matches!(&wired.ext_terrain_hm, Some((prev, _)) if *prev == h_hash) {
                            let hm = TerrainPipeline::upload_heightmap(device, queue, gw, gh, heights)?;
                            wired.ext_terrain_hm = Some((h_hash, hm));
                        }
                        let (_, hm) = wired.ext_terrain_hm.as_ref().unwrap();
                        let wsx = ext.values[2].max(1.0);
                        let wsz = ext.values[3].max(1.0);
                        let ox = ext.values[4];
                        let oz = ext.values[5];
                        let hscale = ext.values[6].max(0.01);
                        let grid = (ext.values[7].round().clamp(2.0, 128.0) / 2.0).round() * 2.0;
                        let tu = TerrainUniform {
                            view_projection: active_vp,
                            camera_pos_and_fog: [
                                cam_pos[0],
                                cam_pos[1],
                                cam_pos[2],
                                frame.fog.density.clamp(0.0, 0.05),
                            ],
                            sun_dir_and_height_scale: [sun_dir[0], sun_dir[1], sun_dir[2], hscale],
                            sun_color_and_ambient: [sun_col[0], sun_col[1], sun_col[2], 0.35],
                            world_bounds: [wsx, wsz, wsx * 0.6, wsx * 1.6],
                            layer_heights: [hscale * 0.12, hscale * 0.55, hscale * 0.85, 0.2],
                            fog_color: [frame.sky.horizon.x, frame.sky.horizon.y, frame.sky.horizon.z, 0.45],
                        };
                        let patch = TerrainPatch {
                            origin_scale_morph: [ox, oz, wsx.max(wsz), 0.0],
                            uv_rect_and_grid: [0.0, 0.0, 1.0, grid.max(2.0)],
                        };
                        pipes.terrain.draw(device, queue, pass, &tu, &[patch], hm)?;
                    }
                }
                ENV_WATER_EXT_TAG if ext.values.len() >= 16 => {
                    let v = &ext.values;
                    let amp = v[15].clamp(0.0, 2.0);
                    let half_extent = if v.len() >= 17 && v[16].is_finite() && v[16] > 0.1 { v[16] } else { 12.0 };
                    let wu = WaterUniform {
                        view_projection: active_vp,
                        camera_pos_and_time: [cam_pos[0], cam_pos[1], cam_pos[2], v[4]],
                        sun_dir_and_level: [sun_dir[0], sun_dir[1], sun_dir[2], v[0]],
                        sun_color_and_bed: [sun_col[0], sun_col[1], sun_col[2], v[1]],
                        absorption_and_grid: [v[5].max(0.0), v[6].max(0.0), v[7].max(0.0), v[3].clamp(2.0, 128.0)],
                        deep_and_foam: [v[8].max(0.0), v[9].max(0.0), v[10].max(0.0), v[11].clamp(0.0, 1.0)],
                        sky_horizon: [frame.sky.horizon.x, frame.sky.horizon.y, frame.sky.horizon.z, half_extent],
                        sky_zenith: [frame.sky.zenith.x, frame.sky.zenith.y, frame.sky.zenith.z, 0.65],
                        bed_color_and_slope: [v[12].max(0.0), v[13].max(0.0), v[14].max(0.0), v[2]],
                        waves: [
                            [1.0, 0.2, 0.12 * amp, 5.0],
                            [0.4, 0.9, 0.08 * amp, 3.1],
                            [-0.6, 0.8, 0.05 * amp, 1.8],
                            [0.8, -0.5, 0.03 * amp, 1.1],
                        ],
                    };
                    pipes.water.draw(queue, pass, &wu)?;
                }
                ENV_FOLIAGE_EXT_TAG if ext.values.len() >= 26 => {
                    let v = &ext.values;
                    let count = (v.len() - 18) / 8;
                    if count > 0 {
                        let mut insts = Vec::with_capacity(count);
                        for i in 0..count {
                            let o = 18 + i * 8;
                            insts.push(FoliageInstance {
                                pos_and_scale: [v[o], v[o + 1], v[o + 2], v[o + 3]],
                                rot_phase_tint: [v[o + 4], v[o + 5], v[o + 6], v[o + 7]],
                            });
                        }
                        let fu = FoliageUniform {
                            view_projection: active_vp,
                            camera_pos_and_time: [cam_pos[0], cam_pos[1], cam_pos[2], v[0]],
                            sun_dir_and_wind_str: [sun_dir[0], sun_dir[1], sun_dir[2], v[3]],
                            sun_color_and_gust: [sun_col[0], sun_col[1], sun_col[2], v[5]],
                            wind_dir_and_speed: [v[1], 0.05, v[2], v[4]],
                            root_color_and_width: [v[8], v[9], v[10], v[6]],
                            tip_color_and_height: [v[11], v[12], v[13], v[7]],
                            sss_color_and_power: [v[14], v[15], v[16], v[17]],
                        };
                        pipes.foliage.draw(device, queue, pass, &fu, &insts)?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}
