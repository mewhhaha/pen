use super::super::{GpuAsset, State};
use super::{JFA_OUTLINE_EXT_TAG, WiredPasses};
use crate::asset_store::{AssetKind, MESH_VERTEX_FLOATS};
use crate::built_in::BuiltIns;
use crate::camera_precision::DepthMode;
use crate::jfa_outline::{JfaOutlinePass, MeshMaskParams, OUTLINE_STYLE_CAPACITY, OutlineStyle};
use crate::matrix::multiply;
use crate::post::SCENE_FORMAT;
use anyhow::Result;
use pen_protocol::{Catalog, Matrix, RenderFrame};

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
    pub pass: Option<JfaOutlinePass>,
}

fn decode_ext_jfa(frame: &RenderFrame) -> Option<JfaOutlineScene> {
    let ext = frame.extensions.iter().find(|e| e.tag == JFA_OUTLINE_EXT_TAG && e.values.len() >= 3)?;
    let v = &ext.values;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let style_count = (v[0].round().clamp(0.0, OUTLINE_STYLE_CAPACITY as f32)) as usize;
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
    let mut targets = Vec::with_capacity(target_pairs);
    for i in 0..target_pairs {
        let o = styles_end + i * 2;
        if v[o] < 0.0 || v[o + 1] < 0.0 {
            continue;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let draw_index = v[o].round() as usize;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let style_index = (v[o + 1].round() as usize).min(style_count - 1);
        targets.push(JfaOutlineTarget { draw_index, style_index });
    }
    (!targets.is_empty()).then_some(JfaOutlineScene { styles, targets, max_radius_px, depth_bias })
}

impl State {
    pub(in crate::renderer) fn gather_jfa_outlines(&self, frame: &RenderFrame) -> Option<JfaOutlineScene> {
        let mut combined = self.wired.jfa_outline.scene.clone().unwrap_or_default();
        if let Some(ext) = decode_ext_jfa(frame) {
            let base = combined.styles.len();
            for s in ext.styles {
                if combined.styles.len() < OUTLINE_STYLE_CAPACITY {
                    combined.styles.push(s);
                }
            }
            for t in ext.targets {
                let idx = base + t.style_index;
                if idx < combined.styles.len() {
                    combined.targets.push(JfaOutlineTarget { draw_index: t.draw_index, style_index: idx });
                }
            }
            combined.max_radius_px = combined.max_radius_px.max(ext.max_radius_px);
            if combined.depth_bias <= 0.0 {
                combined.depth_bias = ext.depth_bias;
            }
        }
        (!combined.styles.is_empty() && !combined.targets.is_empty()).then_some(combined)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn encode_jfa_outlines_wired(
        wired: &mut WiredPasses,
        built_ins: &mut BuiltIns,
        assets: &crate::asset_store::AssetStore<GpuAsset>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        catalog: &Catalog,
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
        let pass_obj = wired.jfa_outline.pass.get_or_insert_with(|| JfaOutlinePass::new(device));
        pass_obj.resize(device, w, h);
        let style_slice = &outline_scene.styles[..outline_scene.styles.len().min(OUTLINE_STYLE_CAPACITY)];
        pass_obj.set_styles(queue, style_slice);

        let vertex_stride = (MESH_VERTEX_FLOATS * size_of::<f32>()) as u64;
        struct QueuedMeshDraw {
            mesh_slot: usize,
            first_index: u32,
            index_count: u32,
            prepared: crate::jfa_outline::PreparedMeshMask,
        }
        let mut queued = Vec::new();
        for target in &outline_scene.targets {
            if target.style_index >= style_slice.len() {
                continue;
            }
            let Some(draw) = frame.draws.get(target.draw_index) else { continue };
            let Some(mesh_key) = catalog.meshes.get(draw.mesh as usize) else { continue };
            let Some(mesh_slot) = assets.id(AssetKind::Mesh, mesh_key) else { continue };
            let Some(GpuAsset::Mesh(mesh)) = assets.current(mesh_slot) else { continue };
            let Some(lod0) = mesh.lods.first() else { continue };
            if lod0.count == 0 {
                continue;
            }
            let params = MeshMaskParams {
                vertex_stride,
                model: draw.transform,
                view_proj: active_vp,
                style_index: target.style_index,
                depth_mode,
                scene_depth: Some(resolved_r32.as_ref().unwrap_or(depth_only)),
                depth_bias: bias,
            };
            let prepared = if resolved_r32.is_some() {
                pass_obj.prepare_mesh_mask_r32(device, &params)
            } else {
                pass_obj.prepare_mesh_mask(device, &params)
            };
            queued.push(QueuedMeshDraw { mesh_slot, first_index: lod0.first, index_count: lod0.count, prepared });
        }
        if queued.is_empty() {
            return Ok(());
        }
        {
            let mut mask_pass = pass_obj.begin_mask_pass_with_depth_mode(encoder, depth_mode);
            for item in &queued {
                let Some(GpuAsset::Mesh(mesh)) = assets.current(item.mesh_slot) else { continue };
                pass_obj.draw_mesh_mask(
                    &mut mask_pass,
                    &item.prepared,
                    mesh.vertices.slice(..),
                    Some((mesh.indices.slice(..), wgpu::IndexFormat::Uint32)),
                    item.first_index..item.first_index + item.index_count,
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
        pass_obj.run_jfa(device, queue, encoder, radius);
        pass_obj.composite(device, queue, encoder, scene_view, SCENE_FORMAT);
        Ok(())
    }
}
