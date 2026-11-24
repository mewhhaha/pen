//! End-to-end wiring of high-fidelity rendering passes into `Renderer::draw`:
//! - Hillaire physical atmosphere LUTs & volumetric clouds (`quality.physical_sky`)
//! - Full PBR deferred decals (`frame.pbr_decals`)
//! - AMD FidelityFX RCAS sharpening & Post-FX polish (`grade.{sharpness, grain, flare, outline_strength}`)
//! - Render-to-texture & split-screen secondary cameras (`frame.camera_targets`)

use std::collections::HashMap;

use anyhow::Result;
use pen_protocol::{Catalog, Matrix, RenderFrame};

use super::{GpuAsset, State, TextureRef, ViewTemporalState};
use crate::asset_store::AssetKind;
use crate::atmosphere_lut::{
    AtmosphereParams, BASE_CAMERA_ALTITUDE_KM, PhysicalAtmosphere, WORLD_METERS_TO_KM, physical_camera_km,
};
use crate::built_in::{BuiltIn, BuiltIns};
use crate::decals::{PbrDecal, PbrDecalPass, PbrDecalView};
use crate::lights::MAX_AREA_COOKIES;
use crate::matrix::{invert, multiply};
use crate::post::{PhysicalSkyBinding, SCENE_FORMAT};
use crate::post_fx::{PostFxParams, PostFxPass, RcasSharpener};
use crate::render_targets::{ColorTransfer, OffscreenRenderTarget, SplitScreenCompositor, ViewportRect};
use crate::RenderTarget;

type CookieUpload = (u32, wgpu::TextureView, u64);
type PreparedCookies = (Vec<Option<u32>>, Vec<CookieUpload>);

pub(super) struct PreparedPbrDecalBatch {
    pub albedo: wgpu::TextureView,
    pub normal: wgpu::TextureView,
    pub orm: wgpu::TextureView,
    pub decals: Vec<PbrDecal>,
}

const COPY_BLIT_WGSL: &str = r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;

@vertex fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
  let y = 1.0 - f32(idx & 2u) * 2.0;
  return vec4f(x, y, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(src_tex, 0));
  let uv = pos.xy / max(dims, vec2f(1.0));
  return textureSampleLevel(src_tex, src_samp, uv, 0.0);
}
"#;

struct BlitShared {
    module: wgpu::ShaderModule,
    bgl: wgpu::BindGroupLayout,
    pl: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
}

pub(super) struct SplitViewSlot {
    pub target: OffscreenRenderTarget,
    pub temporal: ViewTemporalState,
}

#[derive(Default)]
pub(super) struct WiredPasses {
    pub physical_atmosphere: Option<PhysicalAtmosphere>,
    pub physical_sky_serial: u64,
    pub physical_sky_last_key: (bool, u64),
    pub pbr_decals: HashMap<u32, PbrDecalPass>,
    pub rcas: Option<RcasSharpener>,
    pub post_fx: Option<PostFxPass>,
    pub split_screen: Option<(wgpu::TextureFormat, SplitScreenCompositor)>,
    pub split_views: Vec<SplitViewSlot>,
    pub polish_input: Option<(u32, u32, wgpu::TextureFormat, wgpu::Texture, wgpu::TextureView)>,
    pub msaa_depth_resolve: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
    pub dummy_depth: Option<(wgpu::Texture, wgpu::TextureView)>,
    blit_shared: Option<BlitShared>,
    blit_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    pub in_secondary_camera: bool,
    pub bypass_display_output: bool,
}

/// Builds a lightweight secondary-camera `RenderFrame` cloning only 3D scene fields and
/// omitting audio, UI, services, haptics, video streams, camera targets, sprites, and insets.
pub(super) fn secondary_frame(frame: &RenderFrame, view: Matrix, projection: Matrix) -> RenderFrame {
    RenderFrame {
        view,
        projection,
        clear: frame.clear,
        ambient: frame.ambient,
        light_direction: frame.light_direction,
        light_color: frame.light_color,
        aberration: frame.aberration,
        quality: frame.quality,
        sky: frame.sky,
        probe_volume: frame.probe_volume,
        fog: frame.fog,
        shadow: frame.shadow,
        blur: frame.blur,
        clouds: frame.clouds,
        exposure: frame.exposure,
        grade: frame.grade,
        reflection: frame.reflection,
        motion_blur: frame.motion_blur,
        dof: frame.dof,
        draws: frame.draws.clone(),
        solids: frame.solids.clone(),
        decals: frame.decals.clone(),
        poses: frame.poses.clone(),
        emitters: frame.emitters.clone(),
        lights: frame.lights.clone(),
        cascades: frame.cascades.clone(),
        probes: frame.probes.clone(),
        reflection_probes: frame.reflection_probes.clone(),
        fog_volumes: frame.fog_volumes.clone(),
        bone_matrices: frame.bone_matrices.clone(),
        morph_weights: frame.morph_weights.clone(),
        motions: frame.motions.clone(),
        ribbons: frame.ribbons.clone(),
        area_lights: frame.area_lights.clone(),
        pbr_decals: frame.pbr_decals.clone(),
        ..RenderFrame::default()
    }
}

impl WiredPasses {
    pub fn wants_post_polish(frame: &RenderFrame) -> bool {
        frame.grade.sharpness > 0.0
            || frame.quality.upscale
            || frame.grade.grain > 0.0
            || frame.grade.flare > 0.0
            || frame.grade.outline_strength > 0.0
    }

    pub fn polish_target(
        &mut self,
        device: &wgpu::Device,
        size: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        let size = (size.0.max(1), size.1.max(1));
        if !matches!(&self.polish_input, Some((w, h, fmt, _, _)) if *w == size.0 && *h == size.1 && *fmt == format) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("post_polish_input"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.polish_input = Some((size.0, size.1, format, tex, view));
        }
        self.polish_input.as_ref().unwrap().4.clone()
    }

    pub fn dummy_depth_view(&mut self, device: &wgpu::Device) -> wgpu::TextureView {
        if self.dummy_depth.is_none() {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("wired_dummy_depth"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.dummy_depth = Some((tex, view));
        }
        self.dummy_depth.as_ref().unwrap().1.clone()
    }

    /// Resolves a multisampled depth view into a cached 1x `R32Float` texture via `BuiltIn::OpaqueDepth`
    /// so screen-space outlines work identically under MSAA.
    pub fn resolve_msaa_depth(
        &mut self,
        device: &wgpu::Device,
        built_ins: &mut BuiltIns,
        encoder: &mut wgpu::CommandEncoder,
        depth_only: &wgpu::TextureView,
        size: (u32, u32),
        samples: u32,
    ) -> wgpu::TextureView {
        let size = (size.0.max(1), size.1.max(1));
        if !matches!(&self.msaa_depth_resolve, Some((w, h, _, _)) if *w == size.0 && *h == size.1) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("wired_msaa_depth_resolve"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.msaa_depth_resolve = Some((size.0, size.1, tex, view));
        }
        let dst = self.msaa_depth_resolve.as_ref().unwrap().3.clone();
        let pipeline = built_ins.get(BuiltIn::OpaqueDepth(samples));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wired_msaa_depth_resolve_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_only) }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("wired_msaa_depth_resolve"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
        dst
    }

    fn ensure_blit(&mut self, device: &wgpu::Device, target_format: wgpu::TextureFormat) {
        let shared = self.blit_shared.get_or_insert_with(|| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("wired_copy_blit"),
                source: wgpu::ShaderSource::Wgsl(COPY_BLIT_WGSL.into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("wired_copy_blit_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("wired_copy_blit_pl"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("wired_copy_blit_samp"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            BlitShared { module, bgl, pl, sampler }
        });
        self.blit_pipelines.entry(target_format).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("wired_copy_blit"),
                layout: Some(&shared.pl),
                vertex: wgpu::VertexState {
                    module: &shared.module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shared.module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
    }

    pub fn blit(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
    ) {
        self.ensure_blit(device, target_format);
        let shared = self.blit_shared.as_ref().unwrap();
        let pipeline = self.blit_pipelines.get(&target_format).unwrap();
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wired_copy_blit_bg"),
            layout: &shared.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shared.sampler) },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("wired_copy_blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

impl State {
    /// Executes any non-split-screen RTT `frame.camera_targets` before the main pass so materials
    /// referencing those dynamic textures read fresh images without recursion.
    pub(super) fn run_rtt_cameras(&mut self, frame: &RenderFrame, catalog: &Catalog) -> Result<()> {
        if self.wired.in_secondary_camera || frame.camera_targets.is_empty() {
            return Ok(());
        }
        self.wired.in_secondary_camera = true;
        let result = (|| -> Result<()> {
            for cam in &frame.camera_targets {
                if cam.split_screen {
                    continue;
                }
                let Some(mat) = self.material(catalog, cam.material)? else { continue };
                let Some(key) = mat.material.texture.clone() else { continue };
                let w = if cam.width == 0 { 256 } else { cam.width.clamp(16, 4096) };
                let h = if cam.height == 0 { 256 } else { cam.height.clamp(16, 4096) };
                let sub = secondary_frame(frame, cam.view, cam.projection);
                self.render_camera_to_texture(&key, &sub, catalog, (w, h))?;
            }
            Ok(())
        })();
        self.wired.in_secondary_camera = false;
        result
    }

    /// Ensures physical atmosphere LUTs & optional cloud target exist before `Post::begin`
    /// builds the pass graph, returning their views and generation key for `BuiltIn::Atmosphere`.
    pub(super) fn prepare_physical_sky(
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
    pub(super) fn encode_physical_sky(
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

    /// Resolves cookie textures named by `frame.area_lights[i].cookie_material` (1-based catalog material
    /// index) into packed storage slots `0..MAX_AREA_COOKIES`.
    pub(super) fn prepare_area_cookies(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
    ) -> Result<PreparedCookies> {
        if frame.area_lights.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut slots = Vec::with_capacity(frame.area_lights.len().min(64));
        let mut uploads: Vec<CookieUpload> = Vec::new();
        let mut view_to_slot: HashMap<u64, u32> = HashMap::new();
        for spec in frame.area_lights.iter().take(64) {
            if spec.kind != 3 || spec.cookie_material == 0 {
                slots.push(None);
                continue;
            }
            let mat_idx = spec.cookie_material - 1;
            let Some(mat) = self.material(catalog, mat_idx)? else {
                slots.push(None);
                continue;
            };
            let Some(tex_key) = &mat.material.texture else {
                slots.push(None);
                continue;
            };
            let tex_ref = if let Some(&vid) = self.video_ids.get(tex_key.as_str()) {
                Some(TextureRef::Video(vid))
            } else if let Some(&cid) = self.camera_target_ids.get(tex_key.as_str()) {
                Some(TextureRef::CameraTarget(cid))
            } else if !self.reserved_videos.contains(tex_key.as_str()) {
                self.loaded_texture(&mat.material.texture)?.map(TextureRef::Slot)
            } else {
                None
            };
            let Some(tref) = tex_ref else {
                slots.push(None);
                continue;
            };
            let tex = self.texture(tref);
            let view_id = tex.view_id;
            if let Some(&existing) = view_to_slot.get(&view_id) {
                slots.push(Some(existing));
            } else if uploads.len() < MAX_AREA_COOKIES {
                let slot = u32::try_from(uploads.len()).unwrap_or(0);
                view_to_slot.insert(view_id, slot);
                uploads.push((slot, tex.view.clone(), view_id));
                slots.push(Some(slot));
            } else {
                slots.push(None);
            }
        }
        Ok((slots, uploads))
    }

    /// Resolves PBR decal materials/textures ahead of constructing `Drawing` so the render pass
    /// can borrow `self.wired` disjointly from `self.assets`.
    pub(super) fn prepare_pbr_decals(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
    ) -> Result<Vec<PreparedPbrDecalBatch>> {
        if frame.pbr_decals.is_empty() {
            return Ok(Vec::new());
        }
        let mut by_mat: HashMap<u32, Vec<PbrDecal>> = HashMap::new();
        for spec in &frame.pbr_decals {
            let mut d = PbrDecal::from_transform(
                spec.transform,
                [spec.albedo.r, spec.albedo.g, spec.albedo.b, spec.albedo.a],
                [spec.emissive_strength.max(0.0); 3],
                spec.normal_strength.clamp(0.0, 4.0),
                spec.roughness.clamp(0.04, 1.0),
                spec.metallic.clamp(0.0, 1.0),
            );
            d.pbr_params[3] = spec.fade.clamp(0.0, 1.0);
            by_mat.entry(spec.material).or_default().push(d);
        }

        let mut prepared = Vec::with_capacity(by_mat.len());
        for (mat_idx, decals) in by_mat {
            let Some(mat) = self.material(catalog, mat_idx)? else { continue };
            let tex_ref = |state: &mut State, key: &Option<String>, fallback: TextureRef| -> Result<TextureRef> {
                match key {
                    Some(k) => {
                        let slot = state.resolve(AssetKind::Texture, k)?;
                        Ok(if matches!(state.assets.current(slot), Some(GpuAsset::Texture(_))) {
                            TextureRef::Slot(slot)
                        } else {
                            fallback
                        })
                    }
                    None => Ok(fallback),
                }
            };
            let a_ref = tex_ref(self, &mat.material.texture, TextureRef::White)?;
            let n_ref = tex_ref(self, &mat.material.normal_texture, TextureRef::FlatNormal)?;
            let o_ref = tex_ref(self, &mat.material.orm_texture, TextureRef::White)?;
            prepared.push(PreparedPbrDecalBatch {
                albedo: self.texture(a_ref).view.clone(),
                normal: self.texture(n_ref).linear.clone(),
                orm: self.texture(o_ref).linear.clone(),
                decals,
            });
        }
        Ok(prepared)
    }

    /// Encodes prepared PBR decal batches via `PbrDecalPass` onto the opaque HDR scene target.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_pbr_decals_prepared(
        wired: &mut WiredPasses,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        batches: &[PreparedPbrDecalBatch],
        color_attachment: wgpu::RenderPassColorAttachment<'_>,
        depth_view: &wgpu::TextureView,
        depth_only: &wgpu::TextureView,
        samples: u32,
        scene_size: (u32, u32),
    ) -> Result<()> {
        if batches.is_empty() {
            return Ok(());
        }
        if let std::collections::hash_map::Entry::Vacant(e) = wired.pbr_decals.entry(samples) {
            let pass = PbrDecalPass::new(
                device,
                samples,
                SCENE_FORMAT,
                wgpu::TextureFormat::Depth24PlusStencil8,
                true,
            )?;
            e.insert(pass);
        }
        let pass_obj = wired.pbr_decals.get_mut(&samples).unwrap();
        pass_obj.begin_frame();

        let vp = multiply(&frame.projection, &frame.view);
        let eye = invert(&frame.view);
        let decal_view = PbrDecalView {
            view_projection: vp,
            camera_pos: [eye[12], eye[13], eye[14]],
            sun_direction: [-frame.light_direction.x, -frame.light_direction.y, -frame.light_direction.z],
            sun_color: [frame.light_color.x, frame.light_color.y, frame.light_color.z],
            ambient_intensity: 1.0,
            sky_ambient: [frame.sky.zenith.x, frame.sky.zenith.y, frame.sky.zenith.z],
            ground_ambient: [frame.sky.ground.x, frame.sky.ground.y, frame.sky.ground.z],
            depth_clear: 1.0,
            viewport: Some([0.0, 0.0, scene_size.0 as f32, scene_size.1 as f32]),
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pbr_decals"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: None,
                stencil_ops: None,
            }),
            ..Default::default()
        });
        for batch in batches {
            pass_obj.draw(
                device,
                queue,
                &mut render_pass,
                &decal_view,
                &batch.decals,
                depth_only,
                &batch.albedo,
                &batch.normal,
                &batch.orm,
            )?;
        }
        Ok(())
    }

    /// Runs AMD FidelityFX RCAS sharpening and/or Post-FX polish (film grain, anamorphic lens flare,
    /// screen-space outlines) on `polish_src` and writes the final result into `dst_output`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_post_polish_wired(
        wired: &mut WiredPasses,
        built_ins: &mut BuiltIns,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        frames_count: u64,
        target_format: wgpu::TextureFormat,
        polish_src: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        depth_size: (u32, u32),
        dst_output: &wgpu::TextureView,
        size: (u32, u32),
        samples: u32,
    ) -> Result<()> {
        let mut current = polish_src.clone();

        let sharp = if frame.grade.sharpness > 0.0 {
            frame.grade.sharpness.clamp(0.0, 1.0)
        } else if frame.quality.upscale {
            0.35
        } else {
            0.0
        };
        if sharp > 0.0 {
            let rcas = wired.rcas.get_or_insert_with(|| RcasSharpener::new(device));
            rcas.sharpen(device, encoder, &current, size, sharp)?;
            if let Some(view) = rcas.output_view() {
                current = view.clone();
            }
        }

        if frame.grade.grain > 0.0 || frame.grade.flare > 0.0 || frame.grade.outline_strength > 0.0 {
            let outline = frame.grade.outline_strength.clamp(0.0, 1.0);
            let depth_for_fx = if samples > 1 {
                if outline > 0.0 {
                    wired.resolve_msaa_depth(device, built_ins, encoder, depth_view, depth_size, samples)
                } else {
                    wired.dummy_depth_view(device)
                }
            } else {
                depth_view.clone()
            };
            let fx = wired.post_fx.get_or_insert_with(|| PostFxPass::new(device));
            let params = PostFxParams {
                grain_strength: frame.grade.grain.clamp(0.0, 1.0),
                frame_seed: (frames_count % 4096) as f32 * 0.618_034,
                flare_strength: frame.grade.flare.clamp(0.0, 2.0),
                outline_strength: outline,
                inverse_projection: Some(invert(&frame.projection)),
                ..Default::default()
            };
            fx.dispatch(device, encoder, &current, &depth_for_fx, size, &params)?;
            if let Some(view) = fx.output_view() {
                current = view.clone();
            }
        }

        wired.blit(device, encoder, &current, dst_output, target_format);
        Ok(())
    }

    /// Pre-validates split-screen pixel rectangles and renders each `split_screen == true` camera
    /// into its own temporally-isolated offscreen target in presentation format (`self.context.format`),
    /// bypassing display decode and global accessibility filters so those run once on the composite.
    pub(super) fn render_split_screen_views(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        output_size: (u32, u32),
    ) -> Result<(Vec<ViewportRect>, bool, crate::RenderStats)> {
        let split_cams: Vec<_> = frame.camera_targets.iter().filter(|c| c.split_screen).cloned().collect();
        anyhow::ensure!(
            output_size.0 > 0 && output_size.1 > 0,
            "split-screen output target must be non-empty"
        );
        let mut rects: Vec<ViewportRect> = Vec::with_capacity(split_cams.len());
        for cam in &split_cams {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "validated normalized [0, 1] viewport"
            )]
            let (x0, y0, x1, y1) = {
                let ow = output_size.0 as f32;
                let oh = output_size.1 as f32;
                let x0 = (cam.viewport_x * ow).round().clamp(0.0, ow) as u32;
                let y0 = (cam.viewport_y * oh).round().clamp(0.0, oh) as u32;
                let x1 = ((cam.viewport_x + cam.viewport_w) * ow).round().clamp(0.0, ow) as u32;
                let y1 = ((cam.viewport_y + cam.viewport_h) * oh).round().clamp(0.0, oh) as u32;
                (x0, y0, x1, y1)
            };
            anyhow::ensure!(
                x0 < output_size.0 && y0 < output_size.1 && x1 > x0 && y1 > y0,
                "split-screen viewport ({}, {}, {}, {}) produces an empty or out-of-bounds rectangle on {}x{}",
                cam.viewport_x,
                cam.viewport_y,
                cam.viewport_w,
                cam.viewport_h,
                output_size.0,
                output_size.1
            );
            rects.push(ViewportRect { x: x0, y: y0, width: x1 - x0, height: y1 - y0 });
        }

        let format = self.context.format;
        let mut views = std::mem::take(&mut self.wired.split_views);
        let mut any_waiting = false;
        let mut combined = crate::RenderStats::default();
        let mut scale_sum = 0.0f32;
        let mut rendered_count = 0u32;

        let res = (|| -> Result<()> {
            for (idx, (cam, rect)) in split_cams.iter().zip(&rects).enumerate() {
                let size = (rect.width, rect.height);
                if idx >= views.len() {
                    let target = OffscreenRenderTarget::new(&self.device, size.0, size.1, format)?;
                    let temporal = ViewTemporalState::new(&self.device, &self.queue);
                    views.push(SplitViewSlot { target, temporal });
                } else if views[idx].target.size() != size || views[idx].target.format() != format {
                    views[idx].target = OffscreenRenderTarget::new(&self.device, size.0, size.1, format)?;
                }
                let sub = secondary_frame(frame, cam.view, cam.projection);
                let slot = &mut views[idx];
                let (child_waiting, child_stats) =
                    self.submit_secondary_view(&sub, catalog, &mut slot.target, &mut slot.temporal)?;
                any_waiting |= child_waiting;
                combined.draw_calls += child_stats.draw_calls;
                combined.instanced_draw_calls += child_stats.instanced_draw_calls;
                combined.instances += child_stats.instances;
                combined.instance_upload_bytes += child_stats.instance_upload_bytes;
                combined.instance_upload_calls += child_stats.instance_upload_calls;
                combined.triangles += child_stats.triangles;
                combined.culled += child_stats.culled;
                combined.shadow_culled += child_stats.shadow_culled;
                combined.local_shadow_casters += child_stats.local_shadow_casters;
                combined.local_shadow_faces += child_stats.local_shadow_faces;
                combined.local_shadow_kept += child_stats.local_shadow_kept;
                combined.volume_froxels += child_stats.volume_froxels;
                combined.volume_history |= child_stats.volume_history;
                if combined.exposure.is_none() {
                    combined.exposure = child_stats.exposure;
                }
                scale_sum += child_stats.scale;
                rendered_count += 1;
            }
            views.truncate(split_cams.len());
            Ok(())
        })();
        self.wired.split_views = views;
        res?;
        if rendered_count > 0 {
            combined.scale = scale_sum / rendered_count as f32;
        }
        Ok((rects, any_waiting, combined))
    }

    /// Composites already-rendered `split_views` onto `output`.
    pub(super) fn composite_split_screen(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        output_size: (u32, u32),
        rects: Vec<ViewportRect>,
    ) -> Result<()> {
        let format = self.context.format;
        let device = self.device.clone();
        if !matches!(&self.wired.split_screen, Some((fmt, _)) if *fmt == format) {
            self.wired.split_screen =
                Some((format, SplitScreenCompositor::new(&device, format, ColorTransfer::Passthrough)));
        }
        let (_, comp) = self.wired.split_screen.as_mut().unwrap();
        let views: Vec<&wgpu::TextureView> =
            self.wired.split_views.iter().take(rects.len()).map(|s| s.target.render_view()).collect();
        let layout = crate::render_targets::SplitScreenLayout {
            output_size,
            viewports: rects,
            divider_px: 0,
        };
        comp.composite(&device, encoder, output, &layout, &views, wgpu::Color::BLACK)?;
        Ok(())
    }
}
