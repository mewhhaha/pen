//! `pen-scene`: Hierarchical scene & prefab system with stable 64-bit GUIDs,
//! nested prefab expansion with sparse property overrides, cycle detection,
//! asset dependency manifests, and deterministic `RenderFrame` compilation.

pub mod components;
pub mod id;
pub mod scene;
pub mod sequencer;
pub mod transform;

pub use components::{BodyMode, ColliderShape, Component, LightKind, LodLevel};
pub use id::{AssetId, NodeId, fnv1a64};
pub use scene::{
    AssetManifest, FsPrefabResolver, MemoryPrefabResolver, NullPrefabResolver, PrefabRef, PrefabResolver,
    PropertyOverride, ResolvedNode, ResolvedScene, SCENE_FORMAT_VERSION, Scene, SceneEnvironment, SceneNode,
};
pub use sequencer::{
    CurveInterp, EventKey, FiredEvent, QuatKey, ScalarKey, Sequence, SequencePlayer, SequenceTrack, Vec3Key,
    eval_bezier_ease, sample_quat, sample_scalar, sample_vec3,
};
pub use transform::{
    IDENTITY_MAT4, Transform, TransformOverride, camera_world_to_view, look_at, mat4_to_trs, mul_mat4, normalize3,
    normalize_quat, perspective,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_prefabs_sparse_overrides_cycle_detection_and_render_frame_flattening() {
        // Inner prefab: Barrel mesh at local +Z = 2.0
        let mut barrel_prefab = Scene::new("barrel.prefab.json");
        let mut barrel_node = SceneNode::new("Barrel", None, 0);
        barrel_node.transform = Transform::from_position([0.0, 0.0, 2.0]);
        barrel_node.components.push(Component::MeshRenderer {
            mesh: "barrel.mesh.json".into(),
            material: "steel.material.json".into(),
            cast_shadows: true,
            receive_shadows: true,
            lod_group: Vec::new(),
            parameters: None,
        });
        barrel_prefab.nodes.push(barrel_node);

        // Outer prefab: Turret at +Y = 1.0 instancing barrel_prefab
        let mut turret_prefab = Scene::new("turret.prefab.json");
        let mut turret_root = SceneNode::new("Turret", None, 0);
        turret_root.transform = Transform::from_position([0.0, 1.0, 0.0]);
        turret_root.prefab_instance = Some(PrefabRef { prefab: "barrel.prefab.json".into(), overrides: Vec::new() });
        turret_prefab.nodes.push(turret_root);

        // Main scene: places Tank at X = 10.0, instantiates turret_prefab with a sparse material override on Barrel!
        let mut main_scene = Scene::new("level01.pen-scene.json");
        main_scene.environment.tonemap = 1;
        main_scene.environment.physical_sky = true;

        let mut tank = SceneNode::new("Tank", None, 0);
        tank.transform = Transform::from_position([10.0, 0.0, 0.0]);
        tank.prefab_instance = Some(PrefabRef {
            prefab: "turret.prefab.json".into(),
            overrides: vec![PropertyOverride {
                target_path: "Turret/Barrel".into(),
                transform: Some(TransformOverride { position: Some([0.0, 0.0, 4.0]), ..Default::default() }),
                replace_components: vec![Component::MeshRenderer {
                    mesh: "barrel.mesh.json".into(),
                    material: "gold.material.json".into(),
                    cast_shadows: true,
                    receive_shadows: true,
                    lod_group: Vec::new(),
                    parameters: None,
                }],
                ..Default::default()
            }],
        });
        main_scene.nodes.push(tank);

        let mut resolver = MemoryPrefabResolver::default();
        resolver.insert("barrel.prefab.json", barrel_prefab);
        resolver.insert("turret.prefab.json", turret_prefab);

        // JSON round-trip check.
        let json = serde_json::to_string_pretty(&main_scene).unwrap();
        let reloaded: Scene = serde_json::from_str(&json).unwrap();
        assert_eq!(reloaded, main_scene);

        let resolved = reloaded.resolve(&resolver).unwrap();
        let barrel = resolved.nodes.iter().find(|n| n.name == "Barrel").expect("nested Barrel expanded");
        // World pos = Tank(10,0,0) + Turret(0,1,0) + Overridden Barrel(0,0,4) = (10, 1, 4).
        assert!((barrel.world_matrix[12] - 10.0).abs() < 1e-5);
        assert!((barrel.world_matrix[13] - 1.0).abs() < 1e-5);
        assert!((barrel.world_matrix[14] - 4.0).abs() < 1e-5);

        let manifest = resolved.asset_manifest();
        assert_eq!(manifest.meshes, vec!["barrel.mesh.json"]);
        assert_eq!(manifest.materials, vec!["gold.material.json"]);
        assert_eq!(manifest.prefabs, vec!["barrel.prefab.json", "turret.prefab.json"]);

        let catalog = pen_protocol::Catalog {
            meshes: vec!["barrel.mesh.json".into()],
            materials: vec!["steel.material.json".into(), "gold.material.json".into()],
            sounds: Vec::new(),
        };
        let frame = resolved.to_render_frame(&catalog, 16.0 / 9.0, None);
        assert_eq!(frame.draws.len(), 1);
        assert_eq!(frame.draws[0].material, 1, "sparse override must switch material to gold (index 1)");
        assert_eq!(frame.grade.tonemap, 1);
        assert!(frame.quality.physical_sky);

        // Cycle detection check: A -> B -> A must error cleanly.
        let mut cyc_a = Scene::new("a.prefab.json");
        let mut na = SceneNode::new("A", None, 0);
        na.prefab_instance = Some(PrefabRef { prefab: "b.prefab.json".into(), overrides: Vec::new() });
        cyc_a.nodes.push(na);
        let mut cyc_b = Scene::new("b.prefab.json");
        let mut nb = SceneNode::new("B", None, 0);
        nb.prefab_instance = Some(PrefabRef { prefab: "a.prefab.json".into(), overrides: Vec::new() });
        cyc_b.nodes.push(nb);
        resolver.insert("a.prefab.json", cyc_a.clone());
        resolver.insert("b.prefab.json", cyc_b);
        assert!(cyc_a.resolve(&resolver).is_err(), "cyclic prefab reference must be rejected");
    }
}
