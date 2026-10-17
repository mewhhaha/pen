//! A glTF file imported as the renderer's assets: meshes placed by the
//! scene, materials and their images, skinned meshes with their rig and the
//! file's animations as clips, vertex-animation textures; the same from a
//! `.gltf` and a `.glb`, and nothing rewritten when nothing changed.
mod common;

use common::*;
use pen_gltf::{Options, VatOptions, import};

#[test]
fn imports_meshes_materials_skins_and_clips() {
    let directory = directory("import");
    let out = directory.join("assets");
    let imported = import(&arm_gltf(&directory), &out, &Options::default()).unwrap();
    assert!(imported.notes.is_empty(), "{:?}", imported.notes);
    let meshes: Vec<(&str, &str, usize)> =
        imported.meshes.iter().map(|mesh| (mesh.mesh.as_str(), mesh.material.as_str(), mesh.joints)).collect();
    assert_eq!(
        meshes,
        [
            ("arm_column.mesh.json", "arm_red.material.json", 2),
            ("arm_plinth.mesh.json", "arm_default.material.json", 0)
        ]
    );
    assert_eq!(imported.images, ["arm_white.png"]);
    let clips: Vec<(&str, f32)> = imported.clips.iter().map(|clip| (clip.name.as_str(), clip.duration)).collect();
    assert_eq!(clips, [("bend", 1.0), ("lift", 2.0)]);

    // The column: its vertices in their bind pose, bound to the skin's two
    // joints, its rig the joints and the armature above them.
    let column = read_json(&out.join("arm_column.mesh.json"));
    assert_eq!(column["positions"].as_array().unwrap().len(), 30);
    assert_eq!(column["positions"][28], 2.0, "the top stays where the file binds it");
    assert_eq!(column["joints"].as_array().unwrap().len(), 40);
    let skeleton: Vec<&str> =
        column["skeleton"].as_array().unwrap().iter().map(|node| node["name"].as_str().unwrap()).collect();
    assert_eq!(skeleton, ["armature", "base", "elbow"]);
    assert_eq!(column["skeleton"][2]["parent"], 1);
    assert_eq!(column["skin"].as_array().unwrap().len(), 2);
    assert_eq!(column["skin"][1]["node"], 2);
    assert_eq!(column["skin"][1]["inverseBind"][13], -1.0);
    assert_eq!(column["clips"][1]["channels"][0]["interpolation"], "step");
    // The renderer reads it, rig and all.
    let packed = pen_render::pack_mesh(&std::fs::read(out.join("arm_column.mesh.json")).unwrap()).unwrap();
    assert_eq!(packed[5] & 4, 4, "the binary mesh carries the skin");

    // The plinth is static: placed 2 along x, with normals worked out.
    let plinth = read_json(&out.join("arm_plinth.mesh.json"));
    assert_eq!(plinth["positions"][0], 2.0);
    assert_eq!(plinth["normals"].as_array().unwrap()[..3], [0.0, 0.0, 1.0]);
    assert!(plinth.get("skin").is_none());

    // Materials: the glTF's factors and texture, and a default one.
    let red = read_json(&out.join("arm_red.material.json"));
    assert_eq!(red["shader"], "scene.wgsl");
    assert_eq!(red["color"], serde_json::json!([1.0, 0.0, 0.0, 1.0]));
    assert_eq!(red["instanceParameters"], serde_json::json!([0.5, 0.25, 0.0, 0.0]));
    assert_eq!(red["texture"], "arm_white.png");
    assert!(std::fs::read(out.join("arm_white.png")).unwrap().starts_with(b"\x89PNG"));
    assert_eq!(read_json(&out.join("arm_default.material.json"))["lighting"], "lit");

    // What it wrote, for the game and for people.
    let manifest = read_json(&out.join("arm.import.json"));
    assert_eq!(manifest["clips"][0]["name"], "bend");
    assert_eq!(manifest["meshes"][0]["mesh"], "arm_column.mesh.json");

    // Again: nothing changed, nothing written.
    let again = import(&arm_gltf(&directory), &out, &Options::default()).unwrap();
    assert!(again.changed.is_empty(), "{:?}", again.changed);

    // The same model as a .glb imports to the same files.
    let binary = directory.join("binary");
    import(&arm_glb(&directory), &binary, &Options::default()).unwrap();
    for key in ["arm_column.mesh.json", "arm_plinth.mesh.json", "arm_red.material.json", "arm_white.png"] {
        assert_eq!(std::fs::read(out.join(key)).unwrap(), std::fs::read(binary.join(key)).unwrap(), "{key}");
    }
}

#[test]
fn bakes_skinned_meshes_into_vertex_animation_textures() {
    let directory = directory("vat");
    let out = directory.join("assets");
    let options = Options {
        name: Some("crowd".to_owned()),
        vat: Some(VatOptions { fps: 30.0, shader: "vat.wgsl".to_owned() }),
        ..Options::default()
    };
    let imported = import(&arm_gltf(&directory), &out, &options).unwrap();
    assert_eq!(imported.meshes[0].vat.as_deref(), Some("crowd_column_vat.material.json"));
    assert_eq!(imported.meshes[1].vat, None, "a static mesh has nothing to bake");
    let material = read_json(&out.join("crowd_column_vat.material.json"));
    assert_eq!(material["shader"], "vat.wgsl");
    assert_eq!(material["layerTexture"], "crowd_column_vat.gtex");
    assert_eq!(material["texture"], "crowd_white.png");
    let texture = std::fs::read(out.join("crowd_column_vat.gtex")).unwrap();
    // A raw four-channel packed texture of one level.
    assert_eq!(&texture[..8], b"GTEX\x02\x04\x05\x01");
    // The shader that reads it, written as the directory had none; a
    // game's own is left alone.
    assert_eq!(std::fs::read_to_string(out.join("vat.wgsl")).unwrap(), pen_gltf::VAT_WGSL);
    std::fs::write(out.join("vat.wgsl"), "// the game's own").unwrap();
    import(&arm_gltf(&directory), &out, &options).unwrap();
    assert_eq!(std::fs::read_to_string(out.join("vat.wgsl")).unwrap(), "// the game's own");
}

#[test]
fn a_new_import_removes_what_the_last_made_and_it_does_not() {
    let directory = directory("stale");
    let out = directory.join("assets");
    import(&arm_gltf(&directory), &out, &Options::default()).unwrap();
    assert!(out.join("arm_plinth.mesh.json").exists());
    // The plinth's mesh renamed: its old file goes, the column's stays.
    let (mut document, bytes) = arm();
    document["meshes"][1]["name"] = "stand".into();
    let imported =
        import(&gltf_file(&directory.join("arm.gltf"), document, &bytes), &out, &Options::default()).unwrap();
    assert_eq!(imported.removed, ["arm_plinth.mesh.json"]);
    assert!(!out.join("arm_plinth.mesh.json").exists() && out.join("arm_stand.mesh.json").exists());
    assert!(out.join("arm_column.mesh.json").exists());
    let files = read_json(&out.join("arm.import.json"))["files"].clone();
    assert!(files.as_array().unwrap().contains(&"arm_stand.mesh.json".into()));
    // Files no import of this name made are never touched.
    std::fs::write(out.join("arm_other.png"), b"mine").unwrap();
    import(&arm_gltf(&directory), &out, &Options::default()).unwrap();
    assert!(out.join("arm_other.png").exists() && out.join("arm_plinth.mesh.json").exists());
    assert!(!out.join("arm_stand.mesh.json").exists());
}

#[test]
fn refuses_what_is_not_a_model() {
    let directory = directory("refuse");
    let not_gltf = directory.join("not.gltf");
    std::fs::write(&not_gltf, "{}").unwrap();
    assert!(import(&not_gltf, &directory.join("assets"), &Options::default()).is_err());
    let empty = directory.join("empty.gltf");
    std::fs::write(&empty, r#"{"asset":{"version":"2.0"},"scenes":[{"nodes":[]}]}"#).unwrap();
    let error = import(&empty, &directory.join("assets"), &Options::default()).unwrap_err();
    assert!(format!("{error:#}").contains("no meshes"), "{error:#}");
}
