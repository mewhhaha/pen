#![cfg_attr(not(all(feature = "render-pbr-decals", feature = "render-oit")), allow(unused_imports, dead_code))]
use super::super::{GpuAsset, Scratch, State, TextureRef};
use super::WiredPasses;
use crate::asset_store::AssetKind;
use crate::camera_precision::DepthMode;
use crate::decals::{PbrDecal, PbrDecalPass, PbrDecalView};
use crate::lights::MAX_AREA_COOKIES;
use crate::matrix::{invert, multiply};
use crate::post::SCENE_FORMAT;
use crate::shader::{self, Stage};
use anyhow::Result;
use pen_protocol::{Catalog, RenderFrame};
use std::collections::HashMap;

type CookieUpload = (u32, wgpu::TextureView, u64);
type PreparedCookies = (Vec<Option<u32>>, Vec<CookieUpload>);

pub(in crate::renderer) struct PreparedPbrDecalBatch {
    pub albedo: wgpu::TextureView,
    pub normal: wgpu::TextureView,
    pub orm: wgpu::TextureView,
    pub decals: Vec<PbrDecal>,
}

impl State {
    /// Resolves cookie textures named by `frame.area_lights[i].cookie_material` (1-based catalog material
    /// index) into packed storage slots `0..MAX_AREA_COOKIES`.
    pub(in crate::renderer) fn prepare_area_cookies(
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
    #[cfg(not(feature = "render-pbr-decals"))]
    pub(in crate::renderer) fn prepare_pbr_decals(
        &mut self,
        _frame: &RenderFrame,
        _catalog: &Catalog,
    ) -> Result<Vec<super::PreparedPbrDecalBatch>> {
        Ok(Vec::new())
    }

    /// Resolves PBR decal materials/textures ahead of constructing `Drawing` so the render pass
    /// can borrow `self.wired` disjointly from `self.assets`.
    #[cfg(feature = "render-pbr-decals")]
    pub(in crate::renderer) fn prepare_pbr_decals(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
    ) -> Result<Vec<super::PreparedPbrDecalBatch>> {
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
    #[cfg(not(feature = "render-pbr-decals"))]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_pbr_decals_prepared(
        _wired: &mut WiredPasses,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        _frame: &RenderFrame,
        _batches: &[PreparedPbrDecalBatch],
        _color_attachment: wgpu::RenderPassColorAttachment<'_>,
        _depth_view: &wgpu::TextureView,
        _depth_only: &wgpu::TextureView,
        _samples: u32,
        _scene_size: (u32, u32),
        _depth_mode: DepthMode,
    ) -> Result<()> {
        Ok(())
    }

    /// Encodes prepared PBR decal batches via `PbrDecalPass` onto the opaque HDR scene target.
    #[cfg(feature = "render-pbr-decals")]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_pbr_decals_prepared(
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
        depth_mode: DepthMode,
    ) -> Result<()> {
        if batches.is_empty() || !crate::deferred_shaders::ready_all(&["decal_pbr.wgsl"]) {
            return Ok(());
        }
        if let std::collections::hash_map::Entry::Vacant(e) = wired.pbr_decals.entry(samples) {
            let pass =
                PbrDecalPass::new(device, samples, SCENE_FORMAT, wgpu::TextureFormat::Depth24PlusStencil8, true)?;
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
            depth_clear: depth_mode.clear_value(),
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

    /// Prepares (lazily compiling and caching) WBOIT render pipelines for every translucent entry
    /// when `frame.quality.oit` is enabled.
    #[cfg(not(feature = "render-oit"))]
    pub(in crate::renderer) fn prepare_oit_pipelines(
        &mut self,
        _frame: &RenderFrame,
        _scratch: &Scratch,
        _samples: u32,
    ) -> Vec<Option<wgpu::RenderPipeline>> {
        Vec::new()
    }

    /// Prepares (lazily compiling and caching) WBOIT render pipelines for every translucent entry
    /// when `frame.quality.oit` is enabled.
    #[cfg(feature = "render-oit")]
    pub(in crate::renderer) fn prepare_oit_pipelines(
        &mut self,
        frame: &RenderFrame,
        scratch: &Scratch,
        samples: u32,
    ) -> Vec<Option<wgpu::RenderPipeline>> {
        if !frame.quality.oit || scratch.translucent.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(scratch.translucent.len());
        for &ready_idx in &scratch.translucent {
            let entry = &scratch.ready[ready_idx];
            let skinned = entry.skin.is_some();
            let Some(GpuAsset::Shader(sh)) = self.assets.current(entry.shader) else {
                out.push(None);
                continue;
            };
            let key = (entry.shader, sh.revision, samples, skinned);
            if let Some(cached) = self.wired.oit_pipelines.get(&key) {
                out.push(cached.clone());
                continue;
            }
            let code = sh.code.clone();
            let stage = if skinned { Stage::Skinned } else { Stage::Plain };
            let compiled = crate::oit::derive_oit_wgsl(&code).and_then(|oit_wgsl| {
                let (pipe, mut check) = shader::scoped(&self.device, || {
                    let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("oit_derived_shader"),
                        source: wgpu::ShaderSource::Wgsl(oit_wgsl.into()),
                    });
                    shader::oit_pipeline(&self.context, &module, samples, stage)
                });
                match shader::settled(&mut check) {
                    Some(Ok(())) | None => Some(pipe),
                    Some(Err(_)) => None,
                }
            });
            self.wired.oit_pipelines.insert(key, compiled.clone());
            out.push(compiled);
        }
        out
    }
}
