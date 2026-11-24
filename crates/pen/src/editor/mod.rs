//! Interactive `pen edit` 3D Viewport Scene Editor & Property Inspector.
//!
//! Renders `.pen-scene.json` scenes live through Pen's own GPU `Renderer` with an integrated
//! HUD overlay (hierarchy tree, property inspector, timeline scrubber, asset browser, and
//! 3D-projected Translate/Rotate/Scale gizmos), full Undo/Redo history, Play-in-Editor
//! snapshot/restore, and live visual ShaderGraph compilation.

pub mod app;
pub mod camera;
pub mod gizmo;
pub mod history;
pub mod hud;
pub mod picking;

use std::path::{Path, PathBuf};

use anyhow::Result;
use pen_protocol::{Catalog, RenderFrame};
use pen_render::{CompiledShaderGraph, ShaderGraph};
use pen_scene::{
    Component, FsPrefabResolver, LightKind, NodeId, NullPrefabResolver, ResolvedScene, Scene, SceneNode, Sequence,
    SequencePlayer, Transform,
};

use self::camera::EditorCamera;
use self::gizmo::{GizmoDrag, GizmoMode, GizmoSpace, gizmo_axes};
use self::history::{EditCommand, UndoHistory};
use self::picking::{pick_gizmo_axis, pick_node};

/// Complete stateful model of the interactive scene editor (testable headlessly without a window).
pub struct EditorState {
    pub scene_path: PathBuf,
    pub assets_dir: PathBuf,
    pub catalog: Catalog,
    pub scene: Scene,
    pub resolved: ResolvedScene,
    pub camera: EditorCamera,
    pub selected: Option<NodeId>,
    pub gizmo_mode: GizmoMode,
    pub gizmo_space: GizmoSpace,
    pub active_drag: Option<GizmoDrag>,
    pub history: UndoHistory,
    pub sequence: Sequence,
    pub sequencer: SequencePlayer,
    pub play_snapshot: Option<Scene>,
    pub asset_files: Vec<String>,
    pub shader_graph_status: Option<String>,
    pub dirty: bool,
}

impl EditorState {
    /// Opens an existing `.pen-scene.json` or creates a starter studio scene beside `assets_dir`.
    pub fn open_or_create(scene_path: Option<PathBuf>, assets_dir: PathBuf) -> Result<Self> {
        let catalog = pen_protocol::boundary::read_catalog(&assets_dir).unwrap_or_else(|_| Catalog {
            meshes: Vec::new(),
            materials: Vec::new(),
            sounds: Vec::new(),
        });
        let path = scene_path.unwrap_or_else(|| PathBuf::from("untitled.pen-scene.json"));
        let scene = if path.exists() {
            Scene::load(&path)?
        } else {
            default_editor_scene(&catalog)
        };
        let resolver = FsPrefabResolver::new(&assets_dir);
        let resolved = scene.resolve(&resolver).or_else(|_| scene.resolve(&NullPrefabResolver))?;
        let selected = scene.nodes.first().map(|n| n.id);
        let asset_files = scan_asset_files(&assets_dir);
        Ok(Self {
            scene_path: path,
            assets_dir,
            catalog,
            scene,
            resolved,
            camera: EditorCamera::default(),
            selected,
            gizmo_mode: GizmoMode::Translate,
            gizmo_space: GizmoSpace::World,
            active_drag: None,
            history: UndoHistory::default(),
            sequence: Sequence { name: "preview".into(), duration: 4.0, fps: 30.0, looping: true, tracks: Vec::new() },
            sequencer: SequencePlayer::default(),
            play_snapshot: None,
            asset_files,
            shader_graph_status: None,
            dirty: false,
        })
    }

    pub fn refresh_resolved(&mut self) {
        let resolver = FsPrefabResolver::new(&self.assets_dir);
        if let Ok(r) = self.scene.resolve(&resolver).or_else(|_| self.scene.resolve(&NullPrefabResolver)) {
            self.resolved = r;
        }
    }

    pub fn build_render_frame(&self, size: (u32, u32)) -> RenderFrame {
        let aspect = (size.0.max(1) as f32) / (size.1.max(1) as f32);
        let cam = if self.is_playing() {
            None
        } else {
            Some((self.camera.view_matrix(), self.camera.projection_matrix(aspect)))
        };
        self.resolved.to_render_frame(&self.catalog, aspect, cam)
    }

    pub fn is_playing(&self) -> bool {
        self.play_snapshot.is_some()
    }

    /// Toggles Play-in-Editor mode: snapshots the authored scene before entering Play and
    /// restores it verbatim when exiting Play so live playback edits never corrupt authored data.
    pub fn toggle_play(&mut self) {
        if let Some(snapshot) = self.play_snapshot.take() {
            self.scene = snapshot;
            self.sequencer.playing = false;
            self.refresh_resolved();
        } else {
            self.play_snapshot = Some(self.scene.clone());
            self.sequencer.playing = true;
        }
    }

    pub fn tick(&mut self, dt: f32) {
        if self.is_playing() {
            let _ = self.sequencer.advance(&self.sequence, dt);
            self.sequencer.apply_to_scene(&self.sequence, &mut self.scene);
            self.refresh_resolved();
        }
    }

    pub fn scrub_timeline(&mut self, time: f32) {
        self.sequencer.scrub(&self.sequence, time);
        self.sequencer.apply_to_scene(&self.sequence, &mut self.scene);
        self.refresh_resolved();
    }

    pub fn pick_at(&mut self, px: f32, py: f32, size: (u32, u32)) -> Option<NodeId> {
        let (orig, dir) = self.camera.screen_ray(px, py, size.0 as f32, size.1 as f32);
        let hit = pick_node(&self.resolved, orig, dir);
        self.selected = hit;
        hit
    }

    pub fn begin_gizmo_drag(&mut self, px: f32, py: f32, size: (u32, u32)) -> bool {
        let Some(sel) = self.selected else { return false };
        let Some(rnode) = self.resolved.nodes.iter().find(|n| n.id == sel) else { return false };
        let Some(snode) = self.scene.find_node(sel) else { return false };
        let pivot = [rnode.world_matrix[12], rnode.world_matrix[13], rnode.world_matrix[14]];
        let arm_len = (self.camera.distance * 0.18).clamp(0.6, 5.0);
        let axes = gizmo_axes(&snode.transform, self.gizmo_space);
        let (orig, dir) = self.camera.screen_ray(px, py, size.0 as f32, size.1 as f32);
        let Some(axis) = pick_gizmo_axis(orig, dir, pivot, axes, arm_len) else { return false };
        let axis_dir = match axis {
            gizmo::GizmoAxis::X => axes[0],
            gizmo::GizmoAxis::Y => axes[1],
            gizmo::GizmoAxis::Z => axes[2],
        };
        self.active_drag = GizmoDrag::begin(axis, axis_dir, pivot, orig, dir, snode.transform);
        self.active_drag.is_some()
    }

    pub fn update_gizmo_drag(&mut self, px: f32, py: f32, size: (u32, u32), snap: bool) {
        let (Some(sel), Some(drag)) = (self.selected, self.active_drag) else { return };
        let (orig, dir) = self.camera.screen_ray(px, py, size.0 as f32, size.1 as f32);
        let updated = drag.evaluate(self.gizmo_mode, orig, dir, snap);
        if let Some(node) = self.scene.find_node_mut(sel) {
            node.transform = updated;
        }
        self.dirty = true;
        self.refresh_resolved();
    }

    pub fn end_gizmo_drag(&mut self) {
        if let (Some(sel), Some(drag)) = (self.selected, self.active_drag.take())
            && let Some(node) = self.scene.find_node(sel)
            && node.transform != drag.initial
        {
            self.history.push_already_applied(EditCommand::SetTransform {
                node: sel,
                before: drag.initial,
                after: node.transform,
            });
            self.dirty = true;
        }
    }

    pub fn undo(&mut self) -> bool {
        let ok = self.history.undo(&mut self.scene);
        if ok {
            self.dirty = true;
            self.refresh_resolved();
        }
        ok
    }

    pub fn redo(&mut self) -> bool {
        let ok = self.history.redo(&mut self.scene);
        if ok {
            self.dirty = true;
            self.refresh_resolved();
        }
        ok
    }

    pub fn focus_selected(&mut self) {
        if let Some(sel) = self.selected
            && let Some(rnode) = self.resolved.nodes.iter().find(|n| n.id == sel)
        {
            let p = [rnode.world_matrix[12], rnode.world_matrix[13], rnode.world_matrix[14]];
            self.camera.focus_on(p, 1.5);
        }
    }

    pub fn cycle_tonemap(&mut self) {
        let before = self.scene.environment.clone();
        let mut after = before.clone();
        after.tonemap = (after.tonemap + 1) % 4;
        self.history.execute(&mut self.scene, EditCommand::SetEnvironment { before, after });
        self.dirty = true;
        self.refresh_resolved();
    }

    pub fn toggle_physical_sky(&mut self) {
        let before = self.scene.environment.clone();
        let mut after = before.clone();
        after.physical_sky = !after.physical_sky;
        self.history.execute(&mut self.scene, EditCommand::SetEnvironment { before, after });
        self.dirty = true;
        self.refresh_resolved();
    }

    pub fn toggle_ddgi(&mut self) {
        let before = self.scene.environment.clone();
        let mut after = before.clone();
        after.ddgi = !after.ddgi;
        self.history.execute(&mut self.scene, EditCommand::SetEnvironment { before, after });
        self.dirty = true;
        self.refresh_resolved();
    }

    pub fn add_cube_node(&mut self) -> NodeId {
        let ord = u32::try_from(self.scene.nodes.len()).unwrap_or(0);
        let mut node = SceneNode::new(format!("Node_{ord}"), None, ord);
        node.transform = Transform::from_position(self.camera.target);
        if let (Some(mesh), Some(mat)) = (self.catalog.meshes.first(), self.catalog.materials.first()) {
            node.components.push(Component::MeshRenderer {
                mesh: mesh.clone(),
                material: mat.clone(),
                cast_shadows: true,
                receive_shadows: true,
                lod_group: Vec::new(),
                parameters: None,
            });
        }
        let id = node.id;
        self.history.execute(&mut self.scene, EditCommand::AddNode { node });
        self.selected = Some(id);
        self.dirty = true;
        self.refresh_resolved();
        id
    }

    pub fn delete_selected(&mut self) {
        let Some(sel) = self.selected else { return };
        let Some(index) = self.scene.nodes.iter().position(|n| n.id == sel) else { return };
        let node = self.scene.nodes[index].clone();
        self.history.execute(&mut self.scene, EditCommand::RemoveNode { index, node });
        self.selected = self.scene.nodes.first().map(|n| n.id);
        self.dirty = true;
        self.refresh_resolved();
    }

    pub fn toggle_selected_visible(&mut self) {
        let Some(sel) = self.selected else { return };
        let Some(before) = self.scene.find_node(sel).map(|n| n.visible) else { return };
        self.history.execute(&mut self.scene, EditCommand::SetVisible { node: sel, before, after: !before });
        self.dirty = true;
        self.refresh_resolved();
    }

    pub fn adjust_selected_light_intensity(&mut self, delta: f32) {
        let Some(sel) = self.selected else { return };
        let before = self.scene.find_node(sel).and_then(|n| {
            n.components.iter().find_map(|c| match c {
                Component::Light { intensity, .. } => Some(*intensity),
                _ => None,
            })
        });
        if let Some(before) = before {
            let after = (before + delta).max(0.0);
            self.history.execute(&mut self.scene, EditCommand::SetLightIntensity { node: sel, before, after });
            self.dirty = true;
            self.refresh_resolved();
        }
    }

    /// Compiles a visual material graph live inside the editor and updates inspector status.
    pub fn compile_shader_graph(&mut self, graph: &ShaderGraph) -> Result<CompiledShaderGraph> {
        match graph.compile_wgsl() {
            Ok(compiled) => {
                self.shader_graph_status = Some(format!("MATGRAPH OK: {} ({} NODES)", compiled.name, compiled.node_count));
                Ok(compiled)
            }
            Err(err) => {
                self.shader_graph_status = Some(format!("MATGRAPH ERR: {err}"));
                Err(err)
            }
        }
    }

    pub fn save(&mut self) -> Result<()> {
        if let Some(snapshot) = &self.play_snapshot {
            snapshot.save(&self.scene_path)?;
        } else {
            self.scene.save(&self.scene_path)?;
        }
        self.dirty = false;
        Ok(())
    }
}

fn default_editor_scene(catalog: &Catalog) -> Scene {
    let mut scene = Scene::new("studio");
    let mut hero = SceneNode::new("HeroMesh", None, 0);
    hero.transform = Transform::from_position([0.0, 0.5, 0.0]);
    if let (Some(mesh), Some(mat)) = (catalog.meshes.first(), catalog.materials.first()) {
        hero.components.push(Component::MeshRenderer {
            mesh: mesh.clone(),
            material: mat.clone(),
            cast_shadows: true,
            receive_shadows: true,
            lod_group: Vec::new(),
            parameters: None,
        });
    }
    let mut sun = SceneNode::new("SunLight", None, 1);
    sun.transform = Transform::from_position([0.0, 8.0, 0.0]);
    sun.components.push(Component::Light {
        kind: LightKind::Directional,
        color: [1.0, 0.96, 0.90],
        intensity: 2.0,
        range: 100.0,
        inner_cone: 0.0,
        outer_cone: 0.0,
        width: 1.0,
        height: 1.0,
        cast_shadows: true,
    });
    scene.nodes.push(hero);
    scene.nodes.push(sun);
    scene
}

fn scan_asset_files(dir: &Path) -> Vec<String> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut files: Vec<String> = rd
        .filter_map(Result::ok)
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.ends_with(".mesh.json")
                || name.ends_with(".material.json")
                || name.ends_with(".matgraph.json")
                || name.ends_with(".prefab.json")
                || name.ends_with(".pen-scene.json")
            {
                Some(name)
            } else {
                None
            }
        })
        .collect();
    files.sort();
    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::hud::EditorHud;
    use pen_render::{Display, NodeRef, RenderTarget, Renderer, RendererOptions, ShaderNode, ShaderNodeKind};
    use pen_scene::{CurveInterp, SequenceTrack, Vec3Key};

    struct OffscreenTarget(wgpu::Texture);
    impl RenderTarget for OffscreenTarget {
        fn size(&self) -> (u32, u32) {
            (self.0.width(), self.0.height())
        }
        fn view(&mut self) -> Option<wgpu::TextureView> {
            Some(self.0.create_view(&Default::default()))
        }
        fn present(&mut self) {}
    }

    #[test]
    fn gpu_editor_picking_gizmo_undo_redo_play_restore_shader_graph_and_hud_render() {
        let tmp = std::env::temp_dir().join(format!("pen_editor_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let scene_file = tmp.join("test.pen-scene.json");

        let mut editor = EditorState::open_or_create(Some(scene_file.clone()), tmp.clone()).unwrap();
        let hero_id = editor.scene.nodes[0].id;

        // 1. Ray pick hero node at center of viewport (camera targets (0, 0.5, 0) where HeroMesh sits).
        editor.selected = None;
        let picked = editor.pick_at(320.0, 180.0, (640, 360));
        assert_eq!(picked, Some(hero_id), "raycast at viewport center must pick HeroMesh");

        // 2. Transform edit via UndoHistory + Undo + Redo.
        let orig = editor.scene.find_node(hero_id).unwrap().transform;
        let moved = Transform::from_position([2.5, 0.5, -1.0]);
        editor.history.execute(
            &mut editor.scene,
            EditCommand::SetTransform { node: hero_id, before: orig, after: moved },
        );
        editor.refresh_resolved();
        assert!((editor.scene.find_node(hero_id).unwrap().transform.position[0] - 2.5).abs() < 1e-5);
        assert!(editor.undo());
        assert!((editor.scene.find_node(hero_id).unwrap().transform.position[0] - 0.0).abs() < 1e-5);
        assert!(editor.redo());
        assert!((editor.scene.find_node(hero_id).unwrap().transform.position[0] - 2.5).abs() < 1e-5);

        // 3. Play-in-Editor timeline playback + exact pre-play scene snapshot restore!
        editor.sequence.tracks.push(SequenceTrack::NodePosition {
            node: hero_id,
            keys: vec![
                Vec3Key { time: 0.0, value: [2.5, 0.5, -1.0], interp: CurveInterp::Linear },
                Vec3Key { time: 2.0, value: [12.5, 0.5, -1.0], interp: CurveInterp::Linear },
            ],
        });
        editor.toggle_play();
        assert!(editor.is_playing());
        editor.tick(1.0);
        assert!((editor.scene.find_node(hero_id).unwrap().transform.position[0] - 7.5).abs() < 0.05);
        editor.toggle_play();
        assert!(!editor.is_playing());
        assert!(
            (editor.scene.find_node(hero_id).unwrap().transform.position[0] - 2.5).abs() < 1e-5,
            "exiting Play-in-Editor must restore pre-play authored transform"
        );

        // 4. Live ShaderGraph compilation inside editor inspector.
        let graph = ShaderGraph {
            name: "editor_mat".into(),
            nodes: vec![
                ShaderNode { id: "c".into(), ui_pos: [0.0, 0.0], kind: ShaderNodeKind::Vec3Const { value: [0.8, 0.2, 0.1] } },
                ShaderNode {
                    id: "out".into(),
                    ui_pos: [120.0, 0.0],
                    kind: ShaderNodeKind::StandardSurfaceOutput {
                        base_color: Some(NodeRef::of("c")),
                        metallic: None,
                        roughness: None,
                        normal: None,
                        emissive: None,
                        ao: None,
                        alpha: None,
                        clearcoat: None,
                        iridescence: None,
                    },
                },
            ],
        };
        let compiled = editor.compile_shader_graph(&graph).unwrap();
        assert!(compiled.wgsl.contains("fs_main"));
        assert!(editor.shader_graph_status.as_deref().unwrap().starts_with("MATGRAPH OK:"));

        // 5. Save & reload `.pen-scene.json`.
        editor.save().unwrap();
        assert!(scene_file.exists());
        let reloaded = Scene::load(&scene_file).unwrap();
        assert_eq!(reloaded.nodes.len(), editor.scene.nodes.len());

        // 6. Headless GPU render: Pen 3D viewport + EditorHud (toolbar, hierarchy, inspector, gizmos)!
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter required");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let format = wgpu::TextureFormat::Bgra8Unorm;
        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererOptions { format, display: Display::SDR, assets: tmp.clone(), veil: None, pipeline_cache: None },
        )
        .unwrap();
        let mut hud = EditorHud::new(&device, &queue, format);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("editor_test_out"),
            size: wgpu::Extent3d { width: 320, height: 180, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let mut target = OffscreenTarget(tex);
        let frame = editor.build_render_frame((320, 180));
        renderer.submit(&frame, &editor.catalog, &mut target).unwrap();
        let mut enc = device.create_command_encoder(&Default::default());
        let zones = hud.render(&device, &queue, &mut enc, &view, (320, 180), &editor);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        assert_eq!(zones.hierarchy_rows.len(), 2, "HUD must lay out clickable rows for both scene nodes");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
