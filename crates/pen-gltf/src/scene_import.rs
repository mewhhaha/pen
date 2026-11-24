//! Full glTF 2.0 Scene/Prefab Importer (`pen import-scene`).
//!
//! Imports a multi-node `.gltf` / `.glb` file into:
//! - Unpacked Pen meshes (with `TEXCOORD_1`, `COLOR_0`, tangents, skins, morphs, PBR materials & images)
//! - A hierarchical `.pen-scene.json` (or `.prefab.json`) preserving parent-child transforms,
//!   cameras, `KHR_lights_punctual` lights, animation clips, Godot/Unity collider naming
//!   conventions (`-col`, `-colonly`, `-convcol`), and `_LOD0..3` sibling grouping
//! - A `<name>.import-scene.json` re-import sidecar preset so subsequent re-imports preserve settings.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use pen_scene::{
    AssetId, BodyMode, ColliderShape, Component, LightKind, LodLevel, NodeId, Scene, SceneNode, Transform,
};
use serde::{Deserialize, Serialize};

use crate::{Imported, Options, import};

/// Options controlling full glTF scene/prefab import.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneImportOptions {
    /// Base asset prefix (defaults to source file stem).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Shader key assigned to imported materials (default `"scene.wgsl"`).
    #[serde(default = "default_shader")]
    pub shader: String,
    /// Optional explicit output scene path (defaults to `<out_dir>/<name>.pen-scene.json`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scene_path: Option<PathBuf>,
    /// Enable vertex-cache + fetch optimization on imported meshes.
    #[serde(default)]
    pub optimize_vertices: bool,
    /// Global uniform scale applied to root nodes (default `1.0`).
    #[serde(default = "default_one")]
    pub root_scale: f32,
}

fn default_shader() -> String {
    "scene.wgsl".to_owned()
}
fn default_one() -> f32 {
    1.0
}

impl Default for SceneImportOptions {
    fn default() -> Self {
        Self {
            name: None,
            shader: default_shader(),
            scene_path: None,
            optimize_vertices: false,
            root_scale: 1.0,
        }
    }
}

/// Sidecar re-import preset saved beside the imported scene (`<name>.import-scene.json`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneImportPreset {
    pub source: String,
    pub scene_file: String,
    pub options: SceneImportOptions,
    pub node_count: usize,
    pub lod_groups: usize,
    pub colliders: usize,
}

/// Summary returned by [`import_scene`].
#[derive(Debug)]
pub struct ImportedScene {
    pub assets: Imported,
    pub scene: Scene,
    pub scene_path: PathBuf,
    pub preset_path: PathBuf,
    pub lod_groups: usize,
    pub colliders: usize,
}

fn parse_lod_suffix(name: &str) -> Option<(&str, u32)> {
    for (suffix, level) in [("_LOD0", 0), ("_LOD1", 1), ("_LOD2", 2), ("_LOD3", 3), ("-lod0", 0), ("-lod1", 1), ("-lod2", 2), ("-lod3", 3)] {
        if let Some(base) = name.strip_suffix(suffix)
            && !base.is_empty()
        {
            return Some((base, level));
        }
    }
    None
}

fn lod_threshold(level: u32) -> f32 {
    match level {
        0 => 1.0,
        1 => 0.5,
        2 => 0.25,
        _ => 0.125,
    }
}

fn read_node_local_bounds(doc: &gltf::Document, node: &gltf::Node, buffers: &[Vec<u8>]) -> ([f32; 3], Vec<[f32; 3]>) {
    let mut min_p = [f32::INFINITY; 3];
    let mut max_p = [f32::NEG_INFINITY; 3];
    let mut hull_pts = Vec::new();
    let Some(mesh) = node.mesh() else {
        return ([0.5, 0.5, 0.5], hull_pts);
    };
    for prim in mesh.primitives() {
        let reader = prim.reader(|b| buffers.get(b.index()).map(Vec::as_slice));
        if let Some(iter) = reader.read_positions() {
            for p in iter {
                for k in 0..3 {
                    min_p[k] = min_p[k].min(p[k]);
                    max_p[k] = max_p[k].max(p[k]);
                }
                if hull_pts.len() < 256 {
                    hull_pts.push(p);
                }
            }
        }
    }
    let _ = doc;
    if !min_p[0].is_finite() {
        return ([0.5, 0.5, 0.5], hull_pts);
    }
    let half = [
        ((max_p[0] - min_p[0]) * 0.5).max(0.01),
        ((max_p[1] - min_p[1]) * 0.5).max(0.01),
        ((max_p[2] - min_p[2]) * 0.5).max(0.01),
    ];
    (half, hull_pts)
}

/// Imports a full glTF 2.0 scene/prefab from `source` into `out_assets`, writing unpacked assets,
/// `<name>.pen-scene.json`, and `<name>.import-scene.json`.
pub fn import_scene(source: &Path, out_assets: &Path, options: &SceneImportOptions) -> Result<ImportedScene> {
    let bytes = std::fs::read(source).with_context(|| format!("reading {}", source.display()))?;
    let stem = source.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "scene".to_owned());
    let base_name = options.name.clone().unwrap_or(stem);

    // 1. Run standard asset import (meshes, materials, textures, skins, morphs, clips).
    let asset_opts = Options {
        name: Some(base_name.clone()),
        shader: options.shader.clone(),
        vat: None,
        optimize_vertices: options.optimize_vertices,
    };
    let assets = import(source, out_assets, &asset_opts)?;

    // 2. Parse glTF hierarchy directly to preserve full node tree, cameras, lights, colliders, and LODs.
    let gltf::Gltf { document, blob } = gltf::Gltf::from_slice_without_validation(&bytes)
        .with_context(|| format!("parsing {} as glTF", source.display()))?;
    let dir = source.parent().unwrap_or_else(|| Path::new("."));
    let buffers: Vec<Vec<u8>> = document
        .buffers()
        .map(|b| match b.source() {
            gltf::buffer::Source::Bin => blob.clone().context("missing GLB binary chunk"),
            gltf::buffer::Source::Uri(uri) => {
                if let Some(rest) = uri.strip_prefix("data:") {
                    let (_, data) = rest.split_once(";base64,").context("unsupported data URI")?;
                    decode_base64(data)
                } else {
                    std::fs::read(dir.join(uri)).with_context(|| format!("reading glTF buffer {uri}"))
                }
            }
        })
        .collect::<Result<_>>()?;

    let gltf_scene = document.default_scene().or_else(|| document.scenes().next()).context("glTF has no scene")?;

    // Map each mesh-bearing node in DFS order to the imported mesh/material entries created by `import`.
    let mut node_mesh_map: HashMap<usize, Vec<(String, String)>> = HashMap::new();
    let mut imported_cursor = 0usize;
    let mut dfs_stack: Vec<gltf::Node> = gltf_scene.nodes().collect();
    dfs_stack.reverse();
    while let Some(node) = dfs_stack.pop() {
        let mut children: Vec<_> = node.children().collect();
        children.reverse();
        dfs_stack.extend(children);
        if let Some(mesh) = node.mesh() {
            let mut entries = Vec::new();
            for prim in mesh.primitives() {
                if prim.mode() == gltf::mesh::Mode::Triangles && imported_cursor < assets.meshes.len() {
                    let m = &assets.meshes[imported_cursor];
                    entries.push((m.mesh.clone(), m.material.clone()));
                    imported_cursor += 1;
                }
            }
            if !entries.is_empty() {
                node_mesh_map.insert(node.index(), entries);
            }
        }
    }

    // Build Scene hierarchy preserving parent-child links.
    let mut scene = Scene::new(base_name.clone());
    scene.id = AssetId::from_path(&format!("{base_name}.pen-scene.json"));
    let mut colliders = 0usize;

    type LodEntry = (String, String, Transform);
    type LodBuckets = BTreeMap<(Option<NodeId>, String), BTreeMap<u32, LodEntry>>;
    // Track LOD siblings keyed by `(parent_id, base_lod_name)` -> `BTreeMap<lod_level, (mesh, material, transform)>`.
    let mut lod_buckets: LodBuckets = BTreeMap::new();

    // Traverse top-down so parent NodeIds exist before children.
    let mut walk: Vec<(gltf::Node, Option<NodeId>, u32)> =
        gltf_scene.nodes().enumerate().map(|(ord, n)| (n, None, u32::try_from(ord).unwrap_or(u32::MAX))).collect();
    walk.reverse();

    while let Some((node, parent_id, ordinal)) = walk.pop() {
        let raw_name = node.name().map_or_else(|| format!("node_{}", node.index()), str::to_owned);
        let (t, r, mut s) = node.transform().decomposed();
        if parent_id.is_none() && (options.root_scale - 1.0).abs() > 1e-6 {
            s = [s[0] * options.root_scale, s[1] * options.root_scale, s[2] * options.root_scale];
        }
        let local_xform = Transform { position: t, rotation: r, scale: s };

        // Detect _LOD0..3 sibling meshes when the node has no children/camera/light of its own.
        if let Some((lod_base, lod_level)) = parse_lod_suffix(&raw_name)
            && node.children().count() == 0
            && node.camera().is_none()
            && node.light().is_none()
            && let Some(prims) = node_mesh_map.get(&node.index())
            && let Some((mesh_key, mat_key)) = prims.first()
        {
            lod_buckets
                .entry((parent_id, lod_base.to_owned()))
                .or_default()
                .insert(lod_level, (mesh_key.clone(), mat_key.clone(), local_xform));
            continue;
        }

        let is_colonly = raw_name.ends_with("-colonly");
        let is_convcol = raw_name.ends_with("-convcol");
        let is_col = raw_name.ends_with("-col");
        let clean_name = raw_name
            .strip_suffix("-colonly")
            .or_else(|| raw_name.strip_suffix("-convcol"))
            .or_else(|| raw_name.strip_suffix("-col"))
            .unwrap_or(&raw_name)
            .to_owned();

        let mut snode = SceneNode::new(clean_name, parent_id, ordinal);
        snode.transform = local_xform;

        // MeshRenderer (omitted for -colonly invisible collision volumes).
        if !is_colonly
            && let Some(prims) = node_mesh_map.get(&node.index())
        {
            for (mesh_key, mat_key) in prims {
                snode.components.push(Component::MeshRenderer {
                    mesh: mesh_key.clone(),
                    material: mat_key.clone(),
                    cast_shadows: true,
                    receive_shadows: true,
                    lod_group: Vec::new(),
                    parameters: None,
                });
            }
        }

        // Collider convention extraction (-col, -colonly, -convcol).
        if is_col || is_colonly || is_convcol {
            let (half_extents, hull_pts) = read_node_local_bounds(&document, &node, &buffers);
            let shape = if is_convcol && !hull_pts.is_empty() {
                ColliderShape::ConvexHull { points: hull_pts }
            } else if is_col
                && let Some(prims) = node_mesh_map.get(&node.index())
                && let Some((mesh_key, _)) = prims.first()
            {
                ColliderShape::TriMesh { mesh: mesh_key.clone() }
            } else {
                ColliderShape::Box { half_extents }
            };
            snode.components.push(Component::Collider { shape, is_trigger: false, layer: 1 });
            snode.components.push(Component::RigidBody {
                mode: BodyMode::Static,
                mass: 0.0,
                linear_damping: 0.0,
                angular_damping: 0.0,
            });
            colliders += 1;
        }

        // Cameras.
        if let Some(cam) = node.camera() {
            match cam.projection() {
                gltf::camera::Projection::Perspective(p) => {
                    snode.components.push(Component::Camera {
                        fov_deg: p.yfov().to_degrees(),
                        near: p.znear(),
                        far: p.zfar().unwrap_or(500.0),
                        active: true,
                        clear_color: [0.08, 0.10, 0.14, 1.0],
                        target_material: None,
                        viewport: None,
                    });
                }
                gltf::camera::Projection::Orthographic(o) => {
                    snode.components.push(Component::Camera {
                        fov_deg: 45.0,
                        near: o.znear(),
                        far: o.zfar(),
                        active: false,
                        clear_color: [0.08, 0.10, 0.14, 1.0],
                        target_material: None,
                        viewport: None,
                    });
                }
            }
        }

        // KHR_lights_punctual attached directly to hierarchy node!
        if let Some(light) = node.light() {
            let (kind, inner_cone, outer_cone) = match light.kind() {
                gltf::khr_lights_punctual::Kind::Directional => (LightKind::Directional, 0.0, 0.0),
                gltf::khr_lights_punctual::Kind::Point => (LightKind::Point, 0.0, 0.0),
                gltf::khr_lights_punctual::Kind::Spot { inner_cone_angle, outer_cone_angle } => {
                    (LightKind::Spot, inner_cone_angle, outer_cone_angle)
                }
            };
            snode.components.push(Component::Light {
                kind,
                color: light.color(),
                intensity: light.intensity(),
                range: light.range().unwrap_or(20.0),
                inner_cone,
                outer_cone,
                width: 1.0,
                height: 1.0,
                cast_shadows: matches!(kind, LightKind::Directional | LightKind::Spot),
            });
        }

        // Attach AnimationPlayer to skinned root or first mesh node when animations exist.
        if !assets.clips.is_empty() && node.skin().is_some() {
            let clip_names: Vec<String> = assets.clips.iter().map(|c| c.name.clone()).collect();
            let active_clip = clip_names.first().cloned();
            snode.components.push(Component::AnimationPlayer {
                clips: clip_names,
                active_clip,
                speed: 1.0,
                looping: true,
            });
        }

        let this_id = snode.id;
        scene.nodes.push(snode);

        let mut children: Vec<(gltf::Node, Option<NodeId>, u32)> = node
            .children()
            .enumerate()
            .map(|(c_ord, c)| (c, Some(this_id), u32::try_from(c_ord).unwrap_or(u32::MAX)))
            .collect();
        children.reverse();
        walk.extend(children);
    }

    // Materialize grouped _LOD0..3 sibling nodes.
    let lod_groups = lod_buckets.len();
    for ((parent_id, lod_base), levels) in lod_buckets {
        let Some((_, (mesh0, mat0, xform0))) = levels.iter().next() else { continue };
        let mut snode = SceneNode::new(lod_base, parent_id, u32::try_from(scene.nodes.len()).unwrap_or(u32::MAX));
        snode.transform = *xform0;
        let lod_group: Vec<LodLevel> = levels
            .iter()
            .map(|(&lvl, (m, mat, _))| LodLevel {
                mesh: m.clone(),
                material: mat.clone(),
                screen_threshold: lod_threshold(lvl),
            })
            .collect();
        snode.components.push(Component::MeshRenderer {
            mesh: mesh0.clone(),
            material: mat0.clone(),
            cast_shadows: true,
            receive_shadows: true,
            lod_group,
            parameters: None,
        });
        scene.nodes.push(snode);
    }

    let scene_path = options
        .scene_path
        .clone()
        .unwrap_or_else(|| out_assets.join(format!("{base_name}.pen-scene.json")));
    scene.save(&scene_path)?;

    let preset_path = out_assets.join(format!("{base_name}.import-scene.json"));
    let preset = SceneImportPreset {
        source: source.display().to_string(),
        scene_file: scene_path.file_name().map_or_else(|| format!("{base_name}.pen-scene.json"), |f| f.to_string_lossy().into_owned()),
        options: options.clone(),
        node_count: scene.nodes.len(),
        lod_groups,
        colliders,
    };
    std::fs::write(&preset_path, serde_json::to_string_pretty(&preset)?)?;

    Ok(ImportedScene { assets, scene, scene_path, preset_path, lod_groups, colliders })
}

fn decode_base64(input: &str) -> Result<Vec<u8>> {
    const TABLE: [i8; 128] = [
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
    ];
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for b in input.bytes() {
        if b == b'=' || b.is_ascii_whitespace() {
            continue;
        }
        let digit = TABLE[b as usize];
        ensure!(digit >= 0, "invalid base64 byte");
        buf = (buf << 6) | u32::from(digit.cast_unsigned());
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xff) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_scene::NullPrefabResolver;

    fn b64_encode(bytes: &[u8]) -> String {
        const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b0 = u32::from(chunk[0]);
            let b1 = if chunk.len() > 1 { u32::from(chunk[1]) } else { 0 };
            let b2 = if chunk.len() > 2 { u32::from(chunk[2]) } else { 0 };
            let n = (b0 << 16) | (b1 << 8) | b2;
            out.push(CHARS[((n >> 18) & 63) as usize] as char);
            out.push(CHARS[((n >> 12) & 63) as usize] as char);
            out.push(if chunk.len() > 1 { CHARS[((n >> 6) & 63) as usize] as char } else { '=' });
            out.push(if chunk.len() > 2 { CHARS[(n & 63) as usize] as char } else { '=' });
        }
        out
    }

    #[test]
    fn imports_hierarchy_camera_punctual_light_colliders_and_lod_groups() {
        // Minimal 1-triangle mesh buffer (3 positions + 3 normals + 3 indices).
        let mut bin = Vec::new();
        let positions: [[f32; 3]; 3] = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        for p in positions {
            for v in p {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        let normals: [[f32; 3]; 3] = [[0.0, 0.0, 1.0]; 3];
        for n in normals {
            for v in n {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        let indices: [u16; 3] = [0, 1, 2];
        for i in indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let uri = format!("data:application/octet-stream;base64,{}", b64_encode(&bin));
        let gltf_json = serde_json::json!({
            "asset": { "version": "2.0" },
            "extensionsUsed": ["KHR_lights_punctual"],
            "extensions": {
                "KHR_lights_punctual": {
                    "lights": [{
                        "name": "Lamp",
                        "type": "spot",
                        "color": [1.0, 0.9, 0.7],
                        "intensity": 12.0,
                        "range": 15.0,
                        "spot": { "innerConeAngle": 0.2, "outerConeAngle": 0.6 }
                    }]
                }
            },
            "cameras": [{
                "name": "MainCam",
                "type": "perspective",
                "perspective": { "yfov": 1.0, "znear": 0.1, "zfar": 200.0 }
            }],
            "buffers": [{ "uri": uri, "byteLength": bin.len() }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 72, "byteLength": 6 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [-1.0, 0.0, 0.0], "max": [1.0, 2.0, 0.0] },
                { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ],
            "materials": [{ "name": "Stone", "pbrMetallicRoughness": { "baseColorFactor": [0.7, 0.7, 0.7, 1.0] } }],
            "meshes": [{
                "name": "Tri",
                "primitives": [{ "attributes": { "POSITION": 0, "NORMAL": 1 }, "indices": 2, "material": 0 }]
            }],
            "nodes": [
                { "name": "Root", "translation": [5.0, 0.0, 0.0], "children": [1, 2, 3, 4, 5, 6] },
                { "name": "MainCam", "camera": 0, "translation": [0.0, 2.0, 5.0] },
                { "name": "LampNode", "extensions": { "KHR_lights_punctual": { "light": 0 } }, "translation": [0.0, 4.0, 0.0] },
                { "name": "Wall-colonly", "mesh": 0, "translation": [1.0, 0.0, 0.0] },
                { "name": "Rock-convcol", "mesh": 0, "translation": [-1.0, 0.0, 0.0] },
                { "name": "Tree_LOD0", "mesh": 0, "translation": [2.0, 0.0, 0.0] },
                { "name": "Tree_LOD1", "mesh": 0, "translation": [2.0, 0.0, 0.0] }
            ],
            "scenes": [{ "nodes": [0] }],
            "scene": 0
        });

        let tmp = std::env::temp_dir().join(format!("pen_scene_import_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let src = tmp.join("demo.gltf");
        std::fs::write(&src, serde_json::to_string_pretty(&gltf_json).unwrap()).unwrap();
        let out_dir = tmp.join("assets");

        let imported = import_scene(&src, &out_dir, &SceneImportOptions::default()).unwrap();
        assert!(imported.scene_path.exists());
        assert!(imported.preset_path.exists());
        assert_eq!(imported.lod_groups, 1, "Tree_LOD0 + Tree_LOD1 must group into 1 LOD node");
        assert_eq!(imported.colliders, 2, "Wall-colonly + Rock-convcol must create 2 colliders");

        let resolved = imported.scene.resolve(&NullPrefabResolver).unwrap();
        let wall = resolved.nodes.iter().find(|n| n.name == "Wall").expect("Wall-colonly node");
        assert!(
            !wall.components.iter().any(|c| matches!(c, Component::MeshRenderer { .. })),
            "-colonly must omit MeshRenderer"
        );
        assert!(wall.components.iter().any(|c| matches!(c, Component::Collider { shape: ColliderShape::Box { .. }, .. })));
        assert!((wall.world_matrix[12] - 6.0).abs() < 1e-5, "parent Root(5) + Wall(1) = 6");

        let rock = resolved.nodes.iter().find(|n| n.name == "Rock").expect("Rock-convcol node");
        assert!(rock.components.iter().any(|c| matches!(c, Component::Collider { shape: ColliderShape::ConvexHull { .. }, .. })));

        let tree = resolved.nodes.iter().find(|n| n.name == "Tree").expect("merged Tree LOD node");
        let lod_count = tree
            .components
            .iter()
            .find_map(|c| match c {
                Component::MeshRenderer { lod_group, .. } => Some(lod_group.len()),
                _ => None,
            })
            .unwrap_or(0);
        assert_eq!(lod_count, 2, "Tree must carry LOD0 and LOD1");

        let lamp = resolved.nodes.iter().find(|n| n.name == "LampNode").expect("LampNode");
        assert!(lamp.components.iter().any(|c| matches!(c, Component::Light { kind: LightKind::Spot, .. })));

        let cam = resolved.nodes.iter().find(|n| n.name == "MainCam").expect("MainCam");
        assert!(cam.components.iter().any(|c| matches!(c, Component::Camera { .. })));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
