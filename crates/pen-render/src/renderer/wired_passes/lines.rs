#![cfg_attr(not(feature = "render-lines"), allow(unused_imports, dead_code))]
use super::super::State;
use super::{LINES_EXT_TAG, WiredPasses};
use crate::camera_precision::DepthMode;
use crate::environment_passes::PipelineTarget;
use crate::lines::{LineBatch, LineFrame, LineLayer, LineRenderer, LineStyle, XRayStyle};
use crate::matrix::multiply;
use crate::post::SCENE_FORMAT;
use anyhow::Result;
use pen_protocol::{Matrix, RenderFrame};

/// 3D & 2D antialiased line/polyline debug scene attached via [`crate::Renderer::set_line_scene`].
#[derive(Clone, Default)]
pub struct LineScene {
    pub batch: LineBatch,
    pub xray: XRayStyle,
}

#[derive(Default)]
pub(in crate::renderer) struct LineWiredState {
    pub scene: Option<LineScene>,
    pub devtools_scene: Option<LineScene>,
    #[cfg(feature = "render-lines")]
    pub renderer_3d: Option<LineRenderer>,
    #[cfg(feature = "render-lines")]
    pub renderer_2d: Option<LineRenderer>,
}

fn decode_ext_lines(frame: &RenderFrame, batch: &mut LineBatch) -> Option<XRayStyle> {
    let mut xray: Option<XRayStyle> = None;
    for ext in &frame.extensions {
        if ext.tag != LINES_EXT_TAG || ext.values.len() < 18 {
            continue;
        }
        let v = &ext.values;
        xray =
            Some(XRayStyle { alpha: v[0].clamp(0.0, 1.0), desaturate: v[1].clamp(0.0, 1.0), solid_line_dash_px: None });
        let segs = (v.len() - 2) / 16;
        let mut previous = None;
        for i in 0..segs {
            let o = 2 + i * 16;
            let p0 = [v[o], v[o + 1], v[o + 2]];
            let width_px = v[o + 3];
            let p1 = [v[o + 4], v[o + 5], v[o + 6]];
            let dash_len_px = v[o + 7];
            let color = [v[o + 8], v[o + 9], v[o + 10], v[o + 11]];
            let gap_len_px = v[o + 12];
            let feather_px = v[o + 13].max(0.0);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let layer = match v[o + 14].round().clamp(0.0, 2.0) as u32 {
                1 => LineLayer::XRay,
                2 => LineLayer::OnTop,
                _ => LineLayer::DepthTested,
            };
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "line flags")]
            let flags = v[o + 15].round() as u32;
            let is_2d = flags & 1 != 0;
            if !(width_px.is_finite() && width_px > 0.0) {
                previous = None;
                continue;
            }
            let style = LineStyle { width_px, dash_len_px, gap_len_px, dash_offset_px: 0.0, feather_px, layer };
            let point = |[x, y, z]: [f32; 3]| [x, y, if is_2d { 0.0 } else { z }];
            let (p0, p1) = (point(p0), point(p1));
            let continues = flags & 2 != 0 && previous == Some((p0, is_2d, style));
            batch.push(p0, p1, [color; 2], style, is_2d, continues);
            previous = Some((p1, is_2d, style));
        }
    }
    xray
}

impl State {
    #[cfg(not(feature = "render-lines"))]
    pub(in crate::renderer) fn gather_line_scene(&self, _frame: &RenderFrame) -> Option<LineScene> {
        None
    }

    #[cfg(feature = "render-lines")]
    pub(in crate::renderer) fn gather_line_scene(&self, frame: &RenderFrame) -> Option<LineScene> {
        let mut batch = LineBatch::new();
        let mut xray = XRayStyle::default();
        if let Some(scene) = &self.wired.lines.scene {
            batch.extend_from(&scene.batch);
            xray = scene.xray;
        }
        if let Some(ext_xray) = decode_ext_lines(frame, &mut batch)
            && self.wired.lines.scene.is_none()
        {
            xray = ext_xray;
        }
        if let Some(scene) = &self.wired.lines.devtools_scene {
            if batch.is_empty() && self.wired.lines.scene.is_none() {
                xray = scene.xray;
            }
            batch.extend_from(&scene.batch);
        }
        if self.wired.split_child {
            batch = batch.only_3d();
        }
        (!batch.is_empty()).then_some(LineScene { batch, xray })
    }

    #[cfg(not(feature = "render-lines"))]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_lines_3d_wired(
        _wired: &mut WiredPasses,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _pass: &mut wgpu::RenderPass<'_>,
        _scene: &LineScene,
        _view: &Matrix,
        _active_proj: &Matrix,
        _size: (u32, u32),
        _samples: u32,
        _depth_mode: DepthMode,
    ) -> Result<()> {
        Ok(())
    }

    #[cfg(feature = "render-lines")]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_lines_3d_wired(
        wired: &mut WiredPasses,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        scene: &LineScene,
        view: &Matrix,
        active_proj: &Matrix,
        size: (u32, u32),
        samples: u32,
        depth_mode: DepthMode,
    ) -> Result<()> {
        if !scene.batch.has_3d() {
            return Ok(());
        }
        let batch_3d = scene.batch.only_3d();
        let renderer = wired.lines.renderer_3d.get_or_insert_with(|| LineRenderer::new(device));
        let frame = LineFrame {
            view_proj: multiply(active_proj, view),
            viewport: [size.0.max(1) as f32, size.1.max(1) as f32],
            world_target: Some(PipelineTarget {
                color_format: SCENE_FORMAT,
                depth_format: Some(wgpu::TextureFormat::Depth24PlusStencil8),
                depth_compare: depth_mode.compare_nearer_or_equal(),
                sample_count: samples.max(1),
            }),
            screen_target: None,
            xray: scene.xray,
        };
        let prepared = renderer.prepare(device, queue, frame, &batch_3d)?;
        prepared.render_3d(pass);
        Ok(())
    }

    #[cfg(not(feature = "render-lines"))]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_lines_2d_wired(
        _wired: &mut WiredPasses,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        _scene: &LineScene,
        _target_view: &wgpu::TextureView,
        _target_format: wgpu::TextureFormat,
        _size: (u32, u32),
    ) -> Result<()> {
        Ok(())
    }

    #[cfg(feature = "render-lines")]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_lines_2d_wired(
        wired: &mut WiredPasses,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &LineScene,
        target_view: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
        size: (u32, u32),
    ) -> Result<()> {
        if !scene.batch.has_2d() {
            return Ok(());
        }
        let batch_2d = scene.batch.only_2d();
        let renderer = wired.lines.renderer_2d.get_or_insert_with(|| LineRenderer::new(device));
        let frame = LineFrame {
            view_proj: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            viewport: [size.0.max(1) as f32, size.1.max(1) as f32],
            world_target: None,
            screen_target: Some(PipelineTarget {
                color_format: target_format,
                depth_format: None,
                depth_compare: wgpu::CompareFunction::Always,
                sample_count: 1,
            }),
            xray: scene.xray,
        };
        let prepared = renderer.prepare(device, queue, frame, &batch_2d)?;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lines_2d"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        prepared.render_2d(&mut pass);
        Ok(())
    }
}
