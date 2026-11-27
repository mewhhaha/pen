#![cfg_attr(not(feature = "render-jfa-outline"), allow(unused_imports, dead_code))]
use super::super::{Drawing, GpuAsset, ReadyEntry, Source, State};
use super::{JFA_OUTLINE_EXT_TAG, WiredPasses};
use crate::asset_store::MESH_VERTEX_FLOATS;
use crate::built_in::BuiltIns;
use crate::camera_precision::DepthMode;
use crate::culling::CAMERA;
use crate::jfa_outline::{JfaOutlinePass, MeshMaskParams, OUTLINE_STYLE_CAPACITY, OutlineStyle};
use crate::matrix::multiply;
use crate::post::SCENE_FORMAT;
use anyhow::Result;
use pen_protocol::{Matrix, RenderFrame};
use std::collections::BTreeMap;

// Each source has a bounded budget, leaving capacity for the other layers.
const MASK_DRAW_BUDGET: usize = 2048;
pub(in crate::renderer) type OutlineScenes = [Option<JfaOutlineScene>; 3];

pub(in crate::renderer) fn outlined(scenes: &OutlineScenes, index: usize) -> bool {
    scenes.iter().flatten().any(|scene| scene.targets.binary_search_by_key(&index, |target| target.draw_index).is_ok())
}

/// Selects a `RenderFrame::draws` entry for Jump-Flood silhouette outlining.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JfaOutlineTarget {
    /// Index into `RenderFrame::draws`.
    pub draw_index: usize,
    /// Palette index (`0..256`) into [`JfaOutlineScene::styles`].
    pub style_index: usize,
}

/// Multi-entity Jump-Flood silhouette outline scene attached via [`crate::Renderer::set_jfa_outlines`].
#[derive(Clone, Debug, Default)]
pub struct JfaOutlineScene {
    pub styles: Vec<OutlineStyle>,
    pub targets: Vec<JfaOutlineTarget>,
    /// Maximum JFA flood radius in pixels (`0` derives automatically from the widest style).
    pub max_radius_px: u32,
    /// Depth bias in device depth units when tagging occluded outline edges.
    pub depth_bias: f32,
}

#[derive(Default)]
pub(in crate::renderer) struct JfaOutlineWiredState {
    pub scene: Option<JfaOutlineScene>,
    pub devtools_scene: Option<JfaOutlineScene>,
    #[cfg(feature = "render-jfa-outline")]
    pub pass: Option<JfaOutlinePass>,
}

fn decode_ext_jfa(frame: &RenderFrame) -> Option<JfaOutlineScene> {
    let ext = frame.extensions.iter().find(|e| e.tag == JFA_OUTLINE_EXT_TAG && e.values.len() >= 3)?;
    let v = &ext.values;
    if !v[0].is_finite() || !(1.0..=OUTLINE_STYLE_CAPACITY as f32).contains(&v[0]) {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let style_count = v[0].round() as usize;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let max_radius_px = v[1].round().clamp(0.0, 64.0) as u32;
    let depth_bias = v[2].max(0.0);
    let styles_end = 3 + style_count * 11;
    if style_count == 0 || v.len() < styles_end + 2 {
        return None;
    }
    let mut styles = Vec::with_capacity(style_count);
    for i in 0..style_count {
        let o = 3 + i * 11;
        styles.push(OutlineStyle {
            color: [v[o], v[o + 1], v[o + 2], v[o + 3]],
            occluded_color: [v[o + 4], v[o + 5], v[o + 6], v[o + 7]],
            width_px: v[o + 8],
            feather_px: v[o + 9],
            fill_alpha: v[o + 10],
        });
    }
    let target_pairs = (v.len() - styles_end) / 2;
    let mut targets = BTreeMap::new();
    for i in 0..target_pairs {
        let o = styles_end + i * 2;
        if !v[o].is_finite() || !v[o + 1].is_finite() || v[o] < 0.0 || v[o + 1] < 0.0 {
            continue;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let draw_index = v[o].round() as usize;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let style_index = (v[o + 1].round() as usize).min(style_count - 1);
        if draw_index < frame.draws.len() && (targets.len() < MASK_DRAW_BUDGET || targets.contains_key(&draw_index)) {
            targets.insert(draw_index, JfaOutlineTarget { draw_index, style_index });
        }
    }
    (!targets.is_empty()).then(|| JfaOutlineScene {
        styles,
        targets: targets.into_values().collect(),
        max_radius_px,
        depth_bias,
    })
}

impl State {
    #[cfg(not(feature = "render-jfa-outline"))]
    pub(in crate::renderer) fn gather_jfa_outlines(&self, _frame: &RenderFrame) -> OutlineScenes {
        Default::default()
    }

    #[cfg(feature = "render-jfa-outline")]
    pub(in crate::renderer) fn gather_jfa_outlines(&self, frame: &RenderFrame) -> OutlineScenes {
        let normalize = |mut scene: JfaOutlineScene| {
            let mut unique = BTreeMap::new();
            for target in scene.targets {
                if target.draw_index < frame.draws.len()
                    && target.style_index < scene.styles.len().min(OUTLINE_STYLE_CAPACITY)
                    && (unique.len() < MASK_DRAW_BUDGET || unique.contains_key(&target.draw_index))
                {
                    unique.insert(target.draw_index, target);
                }
            }
            scene.targets = unique.into_values().collect();
            (!scene.targets.is_empty()).then_some(scene)
        };
        [
            self.wired.jfa_outline.scene.clone().and_then(normalize),
            decode_ext_jfa(frame),
            self.wired.jfa_outline.devtools_scene.clone().and_then(normalize),
        ]
    }

    #[cfg(not(feature = "render-jfa-outline"))]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_jfa_outlines_wired(
        _wired: &mut WiredPasses,
        _built_ins: &mut BuiltIns,
        _drawing: &Drawing<'_>,
        _ready: &[ReadyEntry],
        _joints: Option<(&wgpu::Buffer, u64)>,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        _frame: &RenderFrame,
        _outline_scene: &JfaOutlineScene,
        _active_proj: &Matrix,
        _scene_view: &wgpu::TextureView,
        _depth_only: &wgpu::TextureView,
        _size: (u32, u32),
        _samples: u32,
        _depth_mode: DepthMode,
    ) -> Result<()> {
        Ok(())
    }

    #[cfg(feature = "render-jfa-outline")]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_jfa_outlines_wired(
        wired: &mut WiredPasses,
        built_ins: &mut BuiltIns,
        drawing: &Drawing<'_>,
        ready: &[ReadyEntry],
        joints: Option<(&wgpu::Buffer, u64)>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        outline_scene: &JfaOutlineScene,
        active_proj: &Matrix,
        scene_view: &wgpu::TextureView,
        depth_only: &wgpu::TextureView,
        size: (u32, u32),
        samples: u32,
        depth_mode: DepthMode,
    ) -> Result<()> {
        let (w, h) = (size.0.max(1), size.1.max(1));
        let seed_bytes = u64::from(w) * u64::from(h) * 16;
        let limits = device.limits();
        if seed_bytes > limits.max_storage_buffer_binding_size.min(limits.max_buffer_size) {
            return Ok(());
        }
        let active_vp = multiply(active_proj, &frame.view);
        let bias = if outline_scene.depth_bias > 0.0 { outline_scene.depth_bias } else { 1e-4 };
        let resolved_r32 =
            (samples > 1).then(|| wired.resolve_msaa_depth(device, built_ins, encoder, depth_only, (w, h), samples));
        if !crate::deferred_shaders::ready_all(&["jfa_outline.wgsl"]) {
            return Ok(());
        }
        let pass_obj = wired.jfa_outline.pass.get_or_insert_with(|| JfaOutlinePass::new(device));
        pass_obj.resize(device, w, h);
        pass_obj.begin_scene_masks();
        let style_slice = &outline_scene.styles[..outline_scene.styles.len().min(OUTLINE_STYLE_CAPACITY)];

        let vertex_stride = (MESH_VERTEX_FLOATS * size_of::<f32>()) as u64;
        let mut queued = Vec::new();
        for entry in ready {
            let item = &drawing.items[entry.item];
            let Source::Draw(index) = item.source else { continue };
            if item.views & CAMERA == 0 || entry.persistent {
                continue;
            }
            let Ok(target) = outline_scene.targets.binary_search_by_key(&index, |target| target.draw_index) else {
                continue;
            };
            let Some(GpuAsset::Mesh(mesh)) = entry.mesh.and_then(|slot| drawing.assets.current(slot)) else {
                continue;
            };
            let lod = mesh.lods.get(usize::from(item.lod)).unwrap_or(&mesh.lods[0]);
            if lod.count == 0 {
                continue;
            }
            let params = MeshMaskParams {
                vertex_stride,
                model: *item.transform(frame),
                view_proj: active_vp,
                style_index: outline_scene.targets[target].style_index,
                depth_mode,
                scene_depth: Some(resolved_r32.as_ref().unwrap_or(depth_only)),
                depth_bias: bias,
            };
            let skin = entry.skin.zip(mesh.skin.as_ref()).zip(joints);
            let prepared = pass_obj.prepare_scene_mask_queued(
                device,
                encoder,
                &params,
                resolved_r32.is_some(),
                skin.map(|((at, _), (buffer, size))| (at, buffer, size)),
            );
            let vertices = match (entry.morph, drawing.morphed) {
                (Some((start, end)), Some(morphed)) => morphed.slice(start..end),
                _ => mesh.vertices.slice(..),
            };
            queued.push((mesh, lod, vertices, skin.map(|((_, skin), _)| skin.vertices.slice(..)), prepared));
        }
        if queued.is_empty() {
            return Ok(());
        }
        {
            let mut mask_pass = pass_obj.begin_mask_pass_with_depth_mode(encoder, depth_mode);
            for (mesh, lod, vertices, skin, prepared) in queued {
                if let Some(skin) = skin {
                    mask_pass.set_vertex_buffer(1, skin);
                }
                pass_obj.draw_mesh_mask(
                    &mut mask_pass,
                    &prepared,
                    vertices,
                    Some((mesh.indices.slice(..), wgpu::IndexFormat::Uint32)),
                    lod.first..lod.first + lod.count,
                );
            }
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped nonnegative pixel radius")]
        let auto_radius = style_slice
            .iter()
            .map(|s| (s.width_px.clamp(0.0, 64.0) + s.feather_px.clamp(0.0, 64.0)).ceil() as u32)
            .max()
            .unwrap_or(1);
        let radius =
            if outline_scene.max_radius_px > 0 { outline_scene.max_radius_px } else { auto_radius }.clamp(1, 64);
        pass_obj.encode_styles(device, encoder, style_slice);
        pass_obj.run_jfa(device, queue, encoder, radius);
        pass_obj.composite(device, queue, encoder, scene_view, SCENE_FORMAT);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_protocol::{Draw, Extension};

    #[test]
    fn outline_extension_rejects_excess_styles_and_deduplicates_targets() {
        let mut values = vec![0.0; 3 + 22];
        values[0] = 2.0;
        values.extend([0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        let mut frame = RenderFrame {
            draws: vec![Draw::default(); 2],
            extensions: vec![Extension { tag: JFA_OUTLINE_EXT_TAG, values }],
            ..Default::default()
        };
        let scene = decode_ext_jfa(&frame).unwrap();
        assert_eq!(
            scene.targets,
            [JfaOutlineTarget { draw_index: 0, style_index: 1 }, JfaOutlineTarget { draw_index: 1, style_index: 0 },]
        );
        let values = &mut frame.extensions[0].values;
        values.resize(3 + 257 * 11 + 2, 0.0);
        values[0] = 257.0;
        assert!(decode_ext_jfa(&frame).is_none());
    }
}
