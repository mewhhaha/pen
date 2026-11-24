//! End-to-end wiring of high-fidelity rendering passes into `Renderer::draw`:
//! - Hillaire physical atmosphere LUTs & volumetric clouds (`quality.physical_sky`)
//! - LTC area lights & cookie projectors (`frame.area_lights`)
//! - Full PBR deferred decals (`frame.pbr_decals`)
//! - AMD FidelityFX RCAS sharpening & Post-FX polish (`grade.{sharpness, grain, flare, outline_strength}`)
//! - Weighted Blended OIT state (`quality.oit`)
//! - Render-to-texture & split-screen secondary cameras (`frame.camera_targets`)

use std::collections::HashMap;

use anyhow::Result;
use pen_protocol::{Catalog, RenderFrame};

use super::{GpuAsset, State, TextureRef};
use crate::area_lights::{AreaLight, AreaLightResources};
use crate::asset_store::AssetKind;
use crate::atmosphere_lut::{AtmosphereParams, PhysicalAtmosphere};
use crate::ddgi::{DdgiGridConfig, DdgiVolume};
use crate::decals::{PbrDecal, PbrDecalPass, PbrDecalView};
use crate::matrix::{invert, multiply};
use crate::oit::OitPass;
use crate::post::SCENE_FORMAT;
use crate::post_fx::{PostFxParams, PostFxPass, RcasSharpener};
use crate::render_targets::{ColorTransfer, OffscreenRenderTarget, SplitScreenCompositor, ViewportRect};

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

#[derive(Default)]
pub(super) struct WiredPasses {
    pub physical_atmosphere: Option<PhysicalAtmosphere>,
    pub pbr_decals: HashMap<u32, PbrDecalPass>,
    pub area_lights: Option<AreaLightResources>,
    pub area_light_buffer: Option<wgpu::Buffer>,
    pub area_light_count: u32,
    pub rcas: Option<RcasSharpener>,
    pub post_fx: Option<PostFxPass>,
    pub oit: Option<OitPass>,
    pub ddgi: Option<DdgiVolume>,
    pub split_screen: Option<SplitScreenCompositor>,
    pub polish_input: Option<(u32, u32, wgpu::TextureFormat, wgpu::Texture, wgpu::TextureView)>,
    pub dummy_depth: Option<(wgpu::Texture, wgpu::TextureView)>,
    pub blit_pipeline: Option<(wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Sampler)>,
    pub in_secondary_camera: bool,
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

    fn ensure_blit(&mut self, device: &wgpu::Device, target_format: wgpu::TextureFormat) {
        if self.blit_pipeline.is_some() {
            return;
        }
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
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("wired_copy_blit"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
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
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("wired_copy_blit_samp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        self.blit_pipeline = Some((pipeline, bgl, sampler));
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
        let (pipeline, bgl, sampler) = self.blit_pipeline.as_ref().unwrap();
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wired_copy_blit_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
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
                let mut sub = frame.clone();
                sub.view = cam.view;
                sub.projection = cam.projection;
                sub.camera_targets.clear();
                self.render_camera_to_texture(&key, &sub, catalog, (w, h))?;
            }
            Ok(())
        })();
        self.wired.in_secondary_camera = false;
        result
    }

    /// Updates Hillaire physical atmosphere LUTs & volumetric clouds when `quality.physical_sky` is on.
    pub(super) fn encode_physical_sky(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scene_size: (u32, u32),
    ) -> Result<()> {
        if frame.quality.oit {
            let device = self.device.clone();
            let oit = self.wired.oit.get_or_insert_with(|| OitPass::new(&device, SCENE_FORMAT, None));
            oit.resize(&device, scene_size);
        }
        if frame.quality.ddgi
            && self.wired.ddgi.is_none()
            && let Ok(vol) = DdgiVolume::new(&self.device, DdgiGridConfig::default())
        {
            self.wired.ddgi = Some(vol);
        }
        if !frame.quality.physical_sky {
            return Ok(());
        }
        let device = self.device.clone();
        let atmo = self.wired.physical_atmosphere.get_or_insert_with(|| PhysicalAtmosphere::new(&device));
        let mut params = AtmosphereParams {
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
            let vp = multiply(&frame.projection, &frame.view);
            let inv_vp = invert(&vp);
            let eye = invert(&frame.view);
            let cw = (scene_size.0 / 2).clamp(16, 1024);
            let ch = (scene_size.1 / 2).clamp(16, 1024);
            atmo.render_clouds(&device, encoder, &params, inv_vp, [eye[12], eye[13], eye[14]], (cw, ch))?;
        }
        Ok(())
    }

    /// Uploads `frame.area_lights` (LTC rect/disk/tube/cookie lights) into GPU storage.
    pub(super) fn upload_area_lights(&mut self, frame: &RenderFrame) {
        if frame.area_lights.is_empty() {
            self.wired.area_light_count = 0;
            return;
        }
        if self.wired.area_lights.is_none() {
            self.wired.area_lights = Some(AreaLightResources::new(&self.device, &self.queue));
        }
        let lights: Vec<AreaLight> = frame
            .area_lights
            .iter()
            .map(|spec| {
                let pos = [spec.position.x, spec.position.y, spec.position.z];
                let right = [spec.right.x, spec.right.y, spec.right.z];
                let up = [spec.up.x, spec.up.y, spec.up.z];
                let gain = spec.intensity.max(0.0);
                let col = [spec.color.x * gain, spec.color.y * gain, spec.color.z * gain];
                let reach = spec.range.max(0.1);
                match spec.kind {
                    1 => AreaLight::disk(pos, right, up, (spec.width * 0.5).max(0.01), col, reach, spec.two_sided),
                    2 => AreaLight::tube(
                        pos,
                        right,
                        up,
                        (spec.width * 0.5).max(0.01),
                        spec.height.max(0.01),
                        col,
                        reach,
                    ),
                    3 => AreaLight::cookie_projector(
                        pos,
                        right,
                        up,
                        spec.width.clamp(0.05, 4.0),
                        (spec.width / spec.height.max(0.01)).clamp(0.1, 10.0),
                        col,
                        reach,
                        0.05,
                    ),
                    _ => AreaLight::rect(
                        pos,
                        right,
                        up,
                        (spec.width * 0.5).max(0.01),
                        (spec.height * 0.5).max(0.01),
                        col,
                        reach,
                        spec.two_sided,
                    ),
                }
            })
            .collect();
        let bytes = bytemuck::cast_slice(&lights);
        let needed = bytes.len() as u64;
        if self.wired.area_light_buffer.as_ref().is_none_or(|b| b.size() < needed) {
            self.wired.area_light_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wired_area_lights"),
                size: needed.next_power_of_two().max(256),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        self.queue.write_buffer(self.wired.area_light_buffer.as_ref().unwrap(), 0, bytes);
        #[allow(clippy::cast_possible_truncation, reason = "bounded by MAX_AREA_LIGHTS")]
        {
            self.wired.area_light_count = lights.len() as u32;
        }
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
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        frames_count: u64,
        target_format: wgpu::TextureFormat,
        polish_src: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
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
            let depth_for_fx = if samples > 1 { wired.dummy_depth_view(device) } else { depth_view.clone() };
            let fx = wired.post_fx.get_or_insert_with(|| PostFxPass::new(device));
            let params = PostFxParams {
                grain_strength: frame.grade.grain.clamp(0.0, 1.0),
                frame_seed: (frames_count % 4096) as f32 * 0.618_034,
                flare_strength: frame.grade.flare.clamp(0.0, 2.0),
                outline_strength: if samples > 1 { 0.0 } else { frame.grade.outline_strength.clamp(0.0, 1.0) },
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

    /// Renders and composites any `split_screen == true` entries in `frame.camera_targets` onto `output`.
    pub(super) fn composite_split_screen(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        output: &wgpu::TextureView,
        output_size: (u32, u32),
    ) -> Result<()> {
        if self.wired.in_secondary_camera {
            return Ok(());
        }
        let split_cams: Vec<_> = frame.camera_targets.iter().filter(|c| c.split_screen).cloned().collect();
        if split_cams.is_empty() {
            return Ok(());
        }
        self.wired.in_secondary_camera = true;
        let format = self.context.format;
        let res = (|| -> Result<()> {
            let mut rendered: Vec<(OffscreenRenderTarget, ViewportRect)> = Vec::with_capacity(split_cams.len());
            for cam in &split_cams {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped normalized [0,1] viewport")]
                let (x, y, w, h) = {
                    let x = (cam.viewport_x.clamp(0.0, 1.0) * output_size.0 as f32).round() as u32;
                    let y = (cam.viewport_y.clamp(0.0, 1.0) * output_size.1 as f32).round() as u32;
                    let w = ((cam.viewport_w.clamp(0.0, 1.0) * output_size.0 as f32).round() as u32)
                        .clamp(1, output_size.0.saturating_sub(x).max(1));
                    let h = ((cam.viewport_h.clamp(0.0, 1.0) * output_size.1 as f32).round() as u32)
                        .clamp(1, output_size.1.saturating_sub(y).max(1));
                    (x, y, w, h)
                };
                let mut rt = OffscreenRenderTarget::new(&self.device, w, h, format)?;
                let mut sub = frame.clone();
                sub.view = cam.view;
                sub.projection = cam.projection;
                sub.camera_targets.clear();
                self.submit(&sub, catalog, &mut rt)?;
                rendered.push((rt, ViewportRect { x, y, width: w, height: h }));
            }
            let device = self.device.clone();
            let comp = self
                .wired
                .split_screen
                .get_or_insert_with(|| SplitScreenCompositor::new(&device, format, ColorTransfer::Passthrough));
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("split_screen_composite"),
            });
            let views: Vec<&wgpu::TextureView> = rendered.iter().map(|(rt, _)| rt.render_view()).collect();
            let rects: Vec<ViewportRect> = rendered.iter().map(|(_, r)| *r).collect();
            let layout = crate::render_targets::SplitScreenLayout {
                output_size,
                viewports: rects,
                divider_px: 0,
            };
            comp.composite(&device, &mut encoder, output, &layout, &views, wgpu::Color::BLACK)?;
            self.queue.submit([encoder.finish()]);
            Ok(())
        })();
        self.wired.in_secondary_camera = false;
        res
    }
}
