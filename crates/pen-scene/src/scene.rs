//! Hierarchical scene and prefab formats, nested prefab expansion with sparse overrides,
//! asset dependency manifests, and deterministic flattening to `pen_protocol::RenderFrame`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use pen_protocol::{
    AreaLightSpec, CameraTargetSpec, Catalog, Color, Draw, Emitter, Fog, Grade, Light, Quality, ReflectionProbe,
    RenderFrame, ReverbZone, Vector,
};
use serde::{Deserialize, Serialize};

use crate::components::{Component, LightKind};
use crate::id::{AssetId, NodeId};
use crate::transform::{IDENTITY_MAT4, Transform, TransformOverride, camera_world_to_view, mul_mat4, normalize3, perspective};

pub const SCENE_FORMAT_VERSION: u32 = 1;

fn default_version() -> u32 {
    SCENE_FORMAT_VERSION
}
fn default_true() -> bool {
    true
}
fn default_one() -> f32 {
    1.0
}
fn default_ambient() -> [f32; 3] {
    [0.18, 0.20, 0.24]
}
fn default_sun_dir() -> [f32; 3] {
    [-0.4, -0.8, -0.45]
}
fn default_sun_color() -> [f32; 3] {
    [1.0, 0.96, 0.90]
}

/// Sparse property override targeting either the prefab instance root (`target_path == ""`)
/// or a nested child path such as `"Turret/Barrel"`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PropertyOverride {
    #[serde(default)]
    pub target_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<TransformOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub replace_components: Vec<Component>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub remove_component_types: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub custom: BTreeMap<String, serde_json::Value>,
}

/// Reference to a `.prefab.json` / `.pen-scene.json` asset instantiated under a node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrefabRef {
    pub prefab: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overrides: Vec<PropertyOverride>,
}

/// One node in a hierarchical `.pen-scene.json` or `.prefab.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneNode {
    pub id: NodeId,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<NodeId>,
    #[serde(default)]
    pub transform: Transform,
    #[serde(default = "default_true")]
    pub visible: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefab_instance: Option<PrefabRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub components: Vec<Component>,
}

impl SceneNode {
    pub fn new(name: impl Into<String>, parent: Option<NodeId>, ordinal: u32) -> Self {
        let name = name.into();
        let id = NodeId::from_seed(&name, parent, ordinal);
        Self {
            id,
            name,
            parent,
            transform: Transform::default(),
            visible: true,
            tags: Vec::new(),
            prefab_instance: None,
            components: Vec::new(),
        }
    }
}

/// Global rendering/environment settings authored with the scene.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneEnvironment {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hdri: Option<String>,
    #[serde(default = "default_one")]
    pub hdri_intensity: f32,
    #[serde(default = "default_one_u32")]
    pub tonemap: u32,
    #[serde(default)]
    pub physical_sky: bool,
    #[serde(default)]
    pub ddgi: bool,
    #[serde(default)]
    pub oit: bool,
    #[serde(default)]
    pub fog_density: f32,
    #[serde(default = "default_ambient")]
    pub ambient: [f32; 3],
    #[serde(default = "default_sun_dir")]
    pub sun_direction: [f32; 3],
    #[serde(default = "default_sun_color")]
    pub sun_color: [f32; 3],
    #[serde(default)]
    pub sharpness: f32,
    #[serde(default)]
    pub grain: f32,
    #[serde(default)]
    pub flare: f32,
}

fn default_one_u32() -> u32 {
    1 // AgX filmic default
}

impl Default for SceneEnvironment {
    fn default() -> Self {
        Self {
            hdri: None,
            hdri_intensity: 1.0,
            tonemap: 1,
            physical_sky: false,
            ddgi: false,
            oit: false,
            fog_density: 0.0,
            ambient: default_ambient(),
            sun_direction: default_sun_dir(),
            sun_color: default_sun_color(),
            sharpness: 0.0,
            grain: 0.0,
            flare: 0.0,
        }
    }
}

/// Top-level scene or reusable prefab document (`.pen-scene.json` / `.prefab.json`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    #[serde(default = "default_version")]
    pub version: u32,
    pub id: AssetId,
    pub name: String,
    #[serde(default)]
    pub environment: SceneEnvironment,
    #[serde(default)]
    pub nodes: Vec<SceneNode>,
}

/// Loads referenced prefab documents during scene expansion.
pub trait PrefabResolver {
    fn load_prefab(&self, key: &str) -> Result<Scene>;
}

/// Empty resolver when scenes have no external prefabs.
pub struct NullPrefabResolver;

impl PrefabResolver for NullPrefabResolver {
    fn load_prefab(&self, key: &str) -> Result<Scene> {
        bail!("prefab '{key}' cannot be resolved with NullPrefabResolver")
    }
}

/// Filesystem-backed prefab loader relative to an asset/scene root directory.
pub struct FsPrefabResolver {
    root: PathBuf,
}

impl FsPrefabResolver {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl PrefabResolver for FsPrefabResolver {
    fn load_prefab(&self, key: &str) -> Result<Scene> {
        let path = self.root.join(key);
        Scene::load(&path)
    }
}

/// In-memory prefab registry for tests and tooling.
#[derive(Default)]
pub struct MemoryPrefabResolver {
    prefabs: HashMap<String, Scene>,
}

impl MemoryPrefabResolver {
    pub fn insert(&mut self, key: impl Into<String>, scene: Scene) {
        self.prefabs.insert(key.into(), scene);
    }
}

impl PrefabResolver for MemoryPrefabResolver {
    fn load_prefab(&self, key: &str) -> Result<Scene> {
        self.prefabs.get(key).cloned().with_context(|| format!("missing in-memory prefab '{key}'"))
    }
}

/// Deduplicated asset dependency manifest extracted from a resolved scene for streaming & packaging.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetManifest {
    pub meshes: Vec<String>,
    pub materials: Vec<String>,
    pub lightmaps: Vec<String>,
    pub audio: Vec<String>,
    pub prefabs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hdri: Option<String>,
}

/// Fully expanded node with final world transform and inherited visibility.
#[derive(Clone, Debug)]
pub struct ResolvedNode {
    pub id: NodeId,
    pub name: String,
    pub path: String,
    pub parent: Option<NodeId>,
    pub local_transform: Transform,
    pub world_matrix: [f32; 16],
    pub visible: bool,
    pub tags: Vec<String>,
    pub components: Vec<Component>,
}

/// Resolved flat hierarchy ready for rendering, physics extraction, or editor inspection.
#[derive(Clone, Debug)]
pub struct ResolvedScene {
    pub name: String,
    pub environment: SceneEnvironment,
    pub nodes: Vec<ResolvedNode>,
    pub prefabs_used: Vec<String>,
}

impl Scene {
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let id = AssetId::from_path(&name);
        Self { version: SCENE_FORMAT_VERSION, id, name, environment: SceneEnvironment::default(), nodes: Vec::new() }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading scene {}", path.display()))?;
        let scene: Self = serde_json::from_str(&text).with_context(|| format!("parsing scene {}", path.display()))?;
        ensure!(scene.version == SCENE_FORMAT_VERSION, "unsupported scene format version {}", scene.version);
        Ok(scene)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text).with_context(|| format!("writing scene {}", path.display()))?;
        Ok(())
    }

    pub fn find_node(&self, id: NodeId) -> Option<&SceneNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn find_node_mut(&mut self, id: NodeId) -> Option<&mut SceneNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    /// Expands all nested prefabs (with cycle detection), applies sparse property overrides,
    /// and computes exact hierarchical world transforms and inherited visibility.
    pub fn resolve(&self, resolver: &dyn PrefabResolver) -> Result<ResolvedScene> {
        let mut expanded_nodes: Vec<SceneNode> = Vec::with_capacity(self.nodes.len());
        let mut prefabs_used = BTreeSet::new();
        let mut stack = Vec::new();
        self.expand_into(None, "", &[], resolver, &mut stack, &mut prefabs_used, &mut expanded_nodes)?;

        // Build topological evaluation order (parents before children), detecting any node cycle.
        let id_to_idx: HashMap<NodeId, usize> = expanded_nodes.iter().enumerate().map(|(i, n)| (n.id, i)).collect();
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); expanded_nodes.len()];
        let mut roots = Vec::new();
        for (i, n) in expanded_nodes.iter().enumerate() {
            if let Some(pid) = n.parent {
                if let Some(&pi) = id_to_idx.get(&pid) {
                    children[pi].push(i);
                } else {
                    roots.push(i);
                }
            } else {
                roots.push(i);
            }
        }

        let mut resolved = Vec::with_capacity(expanded_nodes.len());
        let mut visited = vec![false; expanded_nodes.len()];
        let mut queue: Vec<(usize, [f32; 16], bool, String)> =
            roots.into_iter().rev().map(|r| (r, IDENTITY_MAT4, true, String::new())).collect();

        while let Some((idx, parent_world, parent_vis, parent_path)) = queue.pop() {
            ensure!(!visited[idx], "scene node hierarchy contains a cycle at node {}", expanded_nodes[idx].name);
            visited[idx] = true;
            let node = &expanded_nodes[idx];
            let local_mat = node.transform.to_matrix();
            let world_matrix = mul_mat4(&parent_world, &local_mat);
            let visible = parent_vis && node.visible;
            let path = if parent_path.is_empty() { node.name.clone() } else { format!("{parent_path}/{}", node.name) };
            for &ci in children[idx].iter().rev() {
                queue.push((ci, world_matrix, visible, path.clone()));
            }
            resolved.push(ResolvedNode {
                id: node.id,
                name: node.name.clone(),
                path,
                parent: node.parent,
                local_transform: node.transform,
                world_matrix,
                visible,
                tags: node.tags.clone(),
                components: node.components.clone(),
            });
        }
        ensure!(resolved.len() == expanded_nodes.len(), "scene node hierarchy contains a disconnected cycle");

        Ok(ResolvedScene {
            name: self.name.clone(),
            environment: self.environment.clone(),
            nodes: resolved,
            prefabs_used: prefabs_used.into_iter().collect(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_into(
        &self,
        scope_root: Option<NodeId>,
        path_prefix: &str,
        active_overrides: &[PropertyOverride],
        resolver: &dyn PrefabResolver,
        prefab_stack: &mut Vec<String>,
        prefabs_used: &mut BTreeSet<String>,
        out: &mut Vec<SceneNode>,
    ) -> Result<()> {
        let mut path_by_id: HashMap<NodeId, String> = HashMap::new();
        let id_map: HashMap<NodeId, &SceneNode> = self.nodes.iter().map(|n| (n.id, n)).collect();

        // Compute relative paths within this scene/prefab document.
        fn rel_path(id: NodeId, id_map: &HashMap<NodeId, &SceneNode>, cache: &mut HashMap<NodeId, String>, depth: u32) -> Result<String> {
            ensure!(depth < 64, "node hierarchy exceeds depth limit 64");
            if let Some(p) = cache.get(&id) {
                return Ok(p.clone());
            }
            let node = id_map.get(&id).copied().context("dangling node id")?;
            let path = if let Some(pid) = node.parent
                && id_map.contains_key(&pid)
            {
                let pp = rel_path(pid, id_map, cache, depth + 1)?;
                if pp.is_empty() { node.name.clone() } else { format!("{pp}/{}", node.name) }
            } else {
                node.name.clone()
            };
            cache.insert(id, path.clone());
            Ok(path)
        }

        let roots_count = self.nodes.iter().filter(|n| n.parent.is_none()).count();
        for node in &self.nodes {
            let rp = rel_path(node.id, &id_map, &mut path_by_id, 0)?;
            // If a prefab has a single root node, empty target_path "" targets that root!
            let target_rel = if scope_root.is_some() && node.parent.is_none() && roots_count == 1 {
                String::new()
            } else if path_prefix.is_empty() {
                rp.clone()
            } else {
                format!("{path_prefix}/{rp}")
            };

            let mapped_id = match scope_root {
                Some(root_id) => NodeId::scoped(root_id, node.id),
                None => node.id,
            };
            let mapped_parent = match (scope_root, node.parent) {
                (Some(root_id), Some(pid)) => Some(NodeId::scoped(root_id, pid)),
                (Some(root_id), None) => Some(root_id),
                (None, p) => p,
            };

            let mut cloned = SceneNode {
                id: mapped_id,
                name: node.name.clone(),
                parent: mapped_parent,
                transform: node.transform,
                visible: node.visible,
                tags: node.tags.clone(),
                prefab_instance: None,
                components: node.components.clone(),
            };

            // Apply matching sparse property overrides.
            for ov in active_overrides {
                if ov.target_path == target_rel || (!rp.is_empty() && ov.target_path == rp) {
                    apply_override(&mut cloned, ov);
                }
            }

            out.push(cloned);

            if let Some(prefab_ref) = &node.prefab_instance {
                ensure!(
                    !prefab_stack.iter().any(|k| k == &prefab_ref.prefab),
                    "cyclic prefab reference detected: '{}' -> '{}'",
                    prefab_stack.join(" -> "),
                    prefab_ref.prefab
                );
                prefabs_used.insert(prefab_ref.prefab.clone());
                let prefab_doc = resolver.load_prefab(&prefab_ref.prefab)?;
                prefab_stack.push(prefab_ref.prefab.clone());

                // Combine outer overrides targeting inside this prefab with this prefab instance's own overrides.
                let mut child_overrides = prefab_ref.overrides.clone();
                let prefixes: Vec<String> = [target_rel.as_str(), rp.as_str()]
                    .into_iter()
                    .filter(|p| !p.is_empty())
                    .map(|p| format!("{p}/"))
                    .collect();
                for ov in active_overrides {
                    if let Some(stripped) = prefixes.iter().find_map(|pfx| ov.target_path.strip_prefix(pfx.as_str())) {
                        let mut forwarded = ov.clone();
                        forwarded.target_path = stripped.to_owned();
                        child_overrides.push(forwarded);
                    } else if target_rel.is_empty() && !ov.target_path.is_empty() {
                        // Also allow short paths directly naming children of a single-root prefab.
                        child_overrides.push(ov.clone());
                    }
                }
                prefab_doc.expand_into(
                    Some(mapped_id),
                    "",
                    &child_overrides,
                    resolver,
                    prefab_stack,
                    prefabs_used,
                    out,
                )?;
                prefab_stack.pop();
            }
        }
        Ok(())
    }

    /// Computes the deduplicated asset dependency manifest for streaming/packaging.
    pub fn asset_manifest(&self, resolver: &dyn PrefabResolver) -> Result<AssetManifest> {
        let resolved = self.resolve(resolver)?;
        Ok(resolved.asset_manifest())
    }
}

fn apply_override(node: &mut SceneNode, ov: &PropertyOverride) {
    if let Some(tov) = &ov.transform {
        node.transform.apply_override(tov);
    }
    if let Some(vis) = ov.visible {
        node.visible = vis;
    }
    if !ov.remove_component_types.is_empty() {
        node.components.retain(|c| !ov.remove_component_types.iter().any(|r| r == c.type_name()));
    }
    for replacement in &ov.replace_components {
        if let Some(slot) = node.components.iter_mut().find(|c| c.type_name() == replacement.type_name()) {
            *slot = replacement.clone();
        } else {
            node.components.push(replacement.clone());
        }
    }
    if !ov.custom.is_empty() {
        if let Some(Component::Custom { properties, .. }) =
            node.components.iter_mut().find(|c| matches!(c, Component::Custom { .. }))
        {
            for (k, v) in &ov.custom {
                properties.insert(k.clone(), v.clone());
            }
        } else {
            node.components.push(Component::Custom { kind: "override".into(), properties: ov.custom.clone() });
        }
    }
}

impl ResolvedScene {
    pub fn asset_manifest(&self) -> AssetManifest {
        let mut meshes = BTreeSet::new();
        let mut materials = BTreeSet::new();
        let mut lightmaps = BTreeSet::new();
        let mut audio = BTreeSet::new();

        for node in &self.nodes {
            for comp in &node.components {
                match comp {
                    Component::MeshRenderer { mesh, material, lod_group, .. } => {
                        meshes.insert(mesh.clone());
                        materials.insert(material.clone());
                        for lod in lod_group {
                            meshes.insert(lod.mesh.clone());
                            materials.insert(lod.material.clone());
                        }
                    }
                    Component::Camera { target_material: Some(mat), .. } => {
                        materials.insert(mat.clone());
                    }
                    Component::LightmapRef { irradiance_tex, directional_tex, shadowmask_tex, .. } => {
                        lightmaps.insert(irradiance_tex.clone());
                        if let Some(d) = directional_tex {
                            lightmaps.insert(d.clone());
                        }
                        if let Some(s) = shadowmask_tex {
                            lightmaps.insert(s.clone());
                        }
                    }
                    Component::Collider { shape: crate::components::ColliderShape::TriMesh { mesh }, .. } => {
                        meshes.insert(mesh.clone());
                    }
                    Component::AudioSource { clip, .. } => {
                        audio.insert(clip.clone());
                    }
                    Component::ParticleEmitter { material, .. } => {
                        materials.insert(material.clone());
                    }
                    _ => {}
                }
            }
        }

        AssetManifest {
            meshes: meshes.into_iter().collect(),
            materials: materials.into_iter().collect(),
            lightmaps: lightmaps.into_iter().collect(),
            audio: audio.into_iter().collect(),
            prefabs: self.prefabs_used.clone(),
            hdri: self.environment.hdri.clone(),
        }
    }

    /// Deterministically flattens this resolved scene into a [`RenderFrame`] referencing `catalog`.
    /// Passing `camera_override` (`Some((view, projection))`) lets tools like `pen edit` drive
    /// the viewport camera directly while keeping the scene's own camera nodes intact.
    pub fn to_render_frame(
        &self,
        catalog: &Catalog,
        aspect: f32,
        camera_override: Option<([f32; 16], [f32; 16])>,
    ) -> RenderFrame {
        let find_mesh = |key: &str| catalog.meshes.iter().position(|k| k == key).and_then(|i| u32::try_from(i).ok());
        let find_mat = |key: &str| catalog.materials.iter().position(|k| k == key).and_then(|i| u32::try_from(i).ok());

        let mut frame = RenderFrame {
            ambient: Vector {
                x: self.environment.ambient[0],
                y: self.environment.ambient[1],
                z: self.environment.ambient[2],
            },
            light_direction: Vector {
                x: self.environment.sun_direction[0],
                y: self.environment.sun_direction[1],
                z: self.environment.sun_direction[2],
            },
            light_color: Vector {
                x: self.environment.sun_color[0],
                y: self.environment.sun_color[1],
                z: self.environment.sun_color[2],
            },
            clear: Color { r: 0.08, g: 0.10, b: 0.14, a: 1.0 },
            quality: Quality {
                msaa: 4,
                fxaa: true,
                scale: 1.0,
                filtering: 4,
                occlusion: true,
                sky: true,
                effects: true,
                bloom: true,
                oit: self.environment.oit,
                physical_sky: self.environment.physical_sky,
                ddgi: self.environment.ddgi,
                ..Quality::default()
            },
            grade: Grade {
                tonemap: self.environment.tonemap,
                sharpness: self.environment.sharpness,
                grain: self.environment.grain,
                flare: self.environment.flare,
                ..Grade::default()
            },
            fog: Fog { density: self.environment.fog_density, ..Fog::default() },
            ..RenderFrame::default()
        };

        let mut found_main_camera = false;
        let mut light_id_counter = 1u32;
        let mut reverb_id_counter = 1u32;

        for node in &self.nodes {
            if !node.visible {
                continue;
            }
            let pos = [node.world_matrix[12], node.world_matrix[13], node.world_matrix[14]];
            let right = normalize3([node.world_matrix[0], node.world_matrix[1], node.world_matrix[2]]);
            let up = normalize3([node.world_matrix[4], node.world_matrix[5], node.world_matrix[6]]);
            let forward = normalize3([-node.world_matrix[8], -node.world_matrix[9], -node.world_matrix[10]]);

            for comp in &node.components {
                match comp {
                    Component::MeshRenderer { mesh, material, parameters, .. } => {
                        if let (Some(mi), Some(mati)) = (find_mesh(mesh), find_mat(material)) {
                            frame.draws.push(Draw {
                                mesh: mi,
                                material: mati,
                                transform: node.world_matrix,
                                parameters: *parameters,
                                extra: None,
                            });
                        }
                    }
                    Component::Light {
                        kind,
                        color,
                        intensity,
                        range,
                        inner_cone,
                        outer_cone,
                        width,
                        height,
                        cast_shadows,
                    } => match kind {
                        LightKind::Directional => {
                            frame.light_direction = Vector { x: forward[0], y: forward[1], z: forward[2] };
                            frame.light_color = Vector {
                                x: color[0] * intensity,
                                y: color[1] * intensity,
                                z: color[2] * intensity,
                            };
                        }
                        LightKind::Point | LightKind::Spot => {
                            let is_spot = matches!(kind, LightKind::Spot);
                            frame.lights.push(Light {
                                position: Vector { x: pos[0], y: pos[1], z: pos[2] },
                                radius: if *range > 0.0 { *range } else { 10.0 },
                                color: Vector {
                                    x: color[0] * intensity,
                                    y: color[1] * intensity,
                                    z: color[2] * intensity,
                                },
                                direction: Vector { x: forward[0], y: forward[1], z: forward[2] },
                                inner: if is_spot { inner_cone.cos() } else { 1.0 },
                                outer: if is_spot { outer_cone.cos() } else { -1.0 },
                                shadow: *cast_shadows,
                                id: light_id_counter,
                            });
                            light_id_counter = light_id_counter.saturating_add(1);
                        }
                        LightKind::AreaRect | LightKind::AreaDisk | LightKind::AreaTube => {
                            let ak = match kind {
                                LightKind::AreaRect => 0,
                                LightKind::AreaDisk => 1,
                                _ => 2,
                            };
                            frame.area_lights.push(AreaLightSpec {
                                kind: ak,
                                position: Vector { x: pos[0], y: pos[1], z: pos[2] },
                                right: Vector { x: right[0], y: right[1], z: right[2] },
                                up: Vector { x: up[0], y: up[1], z: up[2] },
                                color: Vector { x: color[0], y: color[1], z: color[2] },
                                intensity: *intensity,
                                width: *width,
                                height: *height,
                                range: if *range > 0.0 { *range } else { 20.0 },
                                cookie_material: 0,
                                two_sided: false,
                            });
                        }
                    },
                    Component::Camera { fov_deg, near, far, active, clear_color, target_material, viewport } => {
                        let view = camera_world_to_view(&node.world_matrix);
                        let proj = perspective(*fov_deg, aspect, *near, *far);
                        if let Some(mat_key) = target_material {
                            if let Some(mi) = find_mat(mat_key) {
                                let vp = viewport.unwrap_or([0.0, 0.0, 1.0, 1.0]);
                                frame.camera_targets.push(CameraTargetSpec {
                                    material: mi,
                                    width: 512,
                                    height: 512,
                                    viewport_x: vp[0],
                                    viewport_y: vp[1],
                                    viewport_w: vp[2],
                                    viewport_h: vp[3],
                                    split_screen: false,
                                    view,
                                    projection: proj,
                                });
                            }
                        } else if *active && !found_main_camera {
                            frame.view = view;
                            frame.projection = proj;
                            frame.clear = Color {
                                r: clear_color[0],
                                g: clear_color[1],
                                b: clear_color[2],
                                a: clear_color[3],
                            };
                            found_main_camera = true;
                        }
                    }
                    Component::ReflectionProbe { id, extent, blend, priority } => {
                        frame.reflection_probes.push(ReflectionProbe {
                            id: *id,
                            revision: 1,
                            priority: *priority,
                            capture: false,
                            position: Vector { x: pos[0], y: pos[1], z: pos[2] },
                            center: Vector { x: pos[0], y: pos[1], z: pos[2] },
                            extent: Vector { x: extent[0], y: extent[1], z: extent[2] },
                            blend: *blend,
                            parallax_center: Vector { x: pos[0], y: pos[1], z: pos[2] },
                            parallax_extent: Vector { x: extent[0], y: extent[1], z: extent[2] },
                            far: (extent[0] + extent[1] + extent[2]).max(10.0),
                        });
                    }
                    Component::ReverbZone { half_extents, blend, decay, damping, wet } => {
                        frame.reverb_zones.push(ReverbZone {
                            id: reverb_id_counter,
                            position: Vector { x: pos[0], y: pos[1], z: pos[2] },
                            half_extents: Vector { x: half_extents[0], y: half_extents[1], z: half_extents[2] },
                            blend: *blend,
                            decay: *decay,
                            damping: *damping,
                            wet: *wet,
                            portal_attenuation: 0.0,
                        });
                        reverb_id_counter = reverb_id_counter.saturating_add(1);
                    }
                    Component::ParticleEmitter {
                        material,
                        rate,
                        lifetime,
                        radius,
                        velocity,
                        gravity,
                        size_start,
                        size_end,
                        color_start,
                        color_end,
                        additive,
                    } => {
                        if let Some(mi) = find_mat(material) {
                            frame.emitters.push(Emitter {
                                material: mi,
                                transform: node.world_matrix,
                                seed: (node.id.0 & 0xffff) as u32,
                                time: 0.0,
                                duration: 0.0,
                                rate: *rate,
                                lifetime: *lifetime,
                                radius: *radius,
                                velocity: Vector { x: velocity[0], y: velocity[1], z: velocity[2] },
                                spread: 0.2,
                                gravity: Vector { x: gravity[0], y: gravity[1], z: gravity[2] },
                                drag: 0.0,
                                size_start: *size_start,
                                size_end: *size_end,
                                color_start: Color {
                                    r: color_start[0],
                                    g: color_start[1],
                                    b: color_start[2],
                                    a: color_start[3],
                                },
                                color_end: Color {
                                    r: color_end[0],
                                    g: color_end[1],
                                    b: color_end[2],
                                    a: color_end[3],
                                },
                                additive: *additive,
                                ..Emitter::default()
                            });
                        }
                    }
                    _ => {}
                }
            }
        }

        if let Some((view, proj)) = camera_override {
            frame.view = view;
            frame.projection = proj;
        } else if !found_main_camera {
            frame.view = crate::transform::look_at([5.0, 3.5, 6.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]);
            frame.projection = perspective(55.0, aspect, 0.1, 500.0);
        }
        frame
    }
}
