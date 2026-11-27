#![cfg_attr(not(feature = "render-post-fx"), allow(unused_imports, dead_code))]
use super::super::{State, ViewTemporalState};
use super::{WiredPasses, secondary_frame};
use crate::RenderTarget;
use crate::built_in::BuiltIns;
use crate::matrix::invert;
use crate::post_fx::{PostFxParams, PostFxPass, RcasSharpener};
use crate::render_targets::{ColorTransfer, OffscreenRenderTarget, SplitScreenCompositor, ViewportRect};
use anyhow::Result;
use pen_protocol::{Catalog, RenderFrame};

pub(in crate::renderer) struct SplitViewSlot {
    pub target: OffscreenRenderTarget,
    pub temporal: ViewTemporalState,
}

impl State {
    /// Executes any non-split-screen RTT `frame.camera_targets` before the main pass so materials
    /// referencing those dynamic textures read fresh images without recursion.
    pub(in crate::renderer) fn run_rtt_cameras(&mut self, frame: &RenderFrame, catalog: &Catalog) -> Result<()> {
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

    /// Runs AMD FidelityFX RCAS sharpening and/or Post-FX polish (film grain, anamorphic lens flare,
    /// screen-space outlines) on `polish_src` and writes the final result into `dst_output`.
    #[cfg(not(feature = "render-post-fx"))]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_post_polish_wired(
        _wired: &mut WiredPasses,
        _built_ins: &mut BuiltIns,
        _device: &wgpu::Device,
        _encoder: &mut wgpu::CommandEncoder,
        _frame: &RenderFrame,
        _frames_count: u64,
        _target_format: wgpu::TextureFormat,
        _polish_src: &wgpu::TextureView,
        _depth_view: &wgpu::TextureView,
        _depth_size: (u32, u32),
        _dst_output: &wgpu::TextureView,
        _size: (u32, u32),
        _samples: u32,
    ) -> Result<()> {
        Ok(())
    }

    /// Runs AMD FidelityFX RCAS sharpening and/or Post-FX polish (film grain, anamorphic lens flare,
    /// screen-space outlines) on `polish_src` and writes the final result into `dst_output`.
    #[cfg(feature = "render-post-fx")]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_post_polish_wired(
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
            crate::post_fx::FsrQualityPreset::from_scale(frame.quality.scale).default_sharpness()
        } else {
            0.0
        };
        if sharp > 0.0 {
            let rcas = wired.rcas.get_or_insert_with(|| RcasSharpener::new(device));
            // `polish_src` is already display-tonemapped for SDR targets (skip Reinhard distortion),
            // while HDR Rgba16Float targets keep the HDR-safe Reinhard compression knee.
            let hdr_knee = if target_format == wgpu::TextureFormat::Rgba16Float { 1.0 } else { 0.0 };
            rcas.sharpen_with_knee(device, encoder, &current, size, sharp, hdr_knee)?;
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
    pub(in crate::renderer) fn render_split_screen_views(
        &mut self,
        frame: &RenderFrame,
        catalog: &Catalog,
        output_size: (u32, u32),
    ) -> Result<(Vec<ViewportRect>, bool, crate::RenderStats)> {
        let split_cams: Vec<_> = frame.camera_targets.iter().filter(|c| c.split_screen).cloned().collect();
        anyhow::ensure!(output_size.0 > 0 && output_size.1 > 0, "split-screen output target must be non-empty");
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
                let mut sub = secondary_frame(frame, cam.view, cam.projection);
                sub.extensions.extend(
                    frame
                        .extensions
                        .iter()
                        .filter(|ext| {
                            matches!(
                                ext.tag,
                                super::GPU_PARTICLES_EXT_TAG
                                    | super::LINES_EXT_TAG
                                    | super::JFA_OUTLINE_EXT_TAG
                                    | super::NANITE_EXT_TAG
                            )
                        })
                        .cloned(),
                );
                let slot = &mut views[idx];
                self.wired.split_child = true;
                let result = self.submit_secondary_view(&sub, catalog, &mut slot.target, &mut slot.temporal);
                self.wired.split_child = false;
                let (child_waiting, child_stats) = result?;
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
    pub(in crate::renderer) fn composite_split_screen(
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
        let layout = crate::render_targets::SplitScreenLayout { output_size, viewports: rects, divider_px: 0 };
        comp.composite(&device, encoder, output, &layout, &views, wgpu::Color::BLACK)?;
        Ok(())
    }
}
