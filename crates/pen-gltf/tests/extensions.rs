//! The material extensions the import reads, as factors, into a material's
//! `surface` block (docs/materials.md): what each gives for what the file says,
//! the defaults for what it does not, the textures it leaves out reported once
//! a kind, and the extensions it does not read reported too.
mod common;

use common::*;
use pen_gltf::{Options, import};
use serde_json::{Value, json};

/// The material JSON the import of a file with this glTF material wrote, and its notes.
fn imported(name: &str, material: Value, extensions: &[&str]) -> (Value, Vec<String>) {
    let directory = directory(name);
    let out = directory.join("assets");
    let file = model_file(&directory, name, model(&square(), material, extensions, json!({})));
    let imported = import(&file, &out, &Options::default()).unwrap();
    assert_eq!(imported.materials.len(), 1);
    (read_json(&out.join(&imported.materials[0])), imported.notes)
}

#[test]
fn a_material_without_extensions_has_no_surface_block() {
    let (material, notes) = imported("plain", json!({"name": "plain"}), &[]);
    assert!(material.get("surface").is_none(), "{material}");
    assert!(notes.is_empty(), "{notes:?}");
}

#[test]
fn clear_coat_sheen_anisotropy_and_transmission_take_the_numbers_the_file_gives() {
    let (material, notes) = imported(
        "lobes",
        json!({"name": "lobes", "extensions": {
            "KHR_materials_clearcoat": {"clearcoatFactor": 0.8, "clearcoatRoughnessFactor": 0.25},
            "KHR_materials_sheen": {"sheenColorFactor": [0.2, 0.4, 0.6], "sheenRoughnessFactor": 0.3},
            "KHR_materials_anisotropy": {"anisotropyStrength": 0.7, "anisotropyRotation": 1.5707964},
            "KHR_materials_transmission": {"transmissionFactor": 0.9},
        }}),
        &["KHR_materials_clearcoat", "KHR_materials_sheen", "KHR_materials_anisotropy", "KHR_materials_transmission"],
    );
    assert_eq!(
        material["surface"],
        json!({
            "clearcoat": 0.8, "clearcoatRoughness": 0.25,
            "sheenColor": [0.2, 0.4, 0.6], "sheenRoughness": 0.3,
            "anisotropy": 0.7, "anisotropyRotation": 1.5707964,
            "transmission": 0.9,
        })
    );
    assert!(notes.is_empty(), "{notes:?}");
    // A transmissive material says nothing of blending: the renderer draws it
    // among the translucent ones from its transmission.
    assert!(material.get("blend").is_none());
    assert_eq!(material["color"][3], 1.0);
}

#[test]
fn volume_gives_its_thickness_and_attenuation_and_leaves_an_infinite_distance_out() {
    let (material, _) = imported(
        "volume",
        json!({"extensions": {"KHR_materials_volume": {
            "thicknessFactor": 2.5, "attenuationDistance": 4.0, "attenuationColor": [0.9, 0.5, 0.1],
        }}}),
        &["KHR_materials_volume"],
    );
    assert_eq!(
        material["surface"],
        json!({"thickness": 2.5, "attenuationDistance": 4.0, "attenuationColor": [0.9, 0.5, 0.1]})
    );
    // No attenuation distance is glTF's infinity, which the surface block holds as 0 by leaving it out.
    let (material, _) = imported(
        "thin",
        json!({"extensions": {"KHR_materials_volume": {"thicknessFactor": 0.5}}}),
        &["KHR_materials_volume"],
    );
    assert_eq!(material["surface"], json!({"thickness": 0.5}));
    // A volume that says nothing says nothing.
    let (material, _) =
        imported("empty-volume", json!({"extensions": {"KHR_materials_volume": {}}}), &["KHR_materials_volume"]);
    assert!(material.get("surface").is_none());
}

#[test]
fn ior_and_specular_take_the_numbers_the_file_gives_and_hold_to_their_ranges() {
    let (material, _) = imported(
        "ior",
        json!({"extensions": {
            "KHR_materials_ior": {"ior": 1.33},
            "KHR_materials_specular": {"specularFactor": 0.5, "specularColorFactor": [1.2, 1.0, 0.8]},
        }}),
        &["KHR_materials_ior", "KHR_materials_specular"],
    );
    assert_eq!(material["surface"], json!({"ior": 1.33, "specular": 0.5, "specularColor": [1.2, 1.0, 0.8]}));
    // The index of refraction is at least 1, the colours and factors at least 0, clear coat 0 to 1.
    let (material, _) = imported(
        "ranges",
        json!({"extensions": {
            "KHR_materials_ior": {"ior": 0.5},
            "KHR_materials_specular": {"specularFactor": -1.0, "specularColorFactor": [-1.0, 0.0, 2.0]},
            "KHR_materials_clearcoat": {"clearcoatFactor": 3.0},
        }}),
        &["KHR_materials_ior", "KHR_materials_specular", "KHR_materials_clearcoat"],
    );
    assert_eq!(
        material["surface"],
        json!({"ior": 1.0, "specular": 0.0, "specularColor": [0.0, 0.0, 2.0], "clearcoat": 1.0})
    );
}

#[test]
fn emissive_strength_multiplies_the_glow_and_is_the_surfaces_too() {
    let (material, _) = imported(
        "emissive",
        json!({
            "emissiveFactor": [1.0, 0.5, 0.25],
            "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 4.0}},
        }),
        &["KHR_materials_emissive_strength"],
    );
    // The glow the existing import gives (the strongest channel times the strength) stays;
    // glTF's default roughness is 1 and its default metal 1.
    assert_eq!(material["instanceParameters"], json!([0.0, 1.0, 4.0, 0.0]));
    assert_eq!(material["surface"], json!({"emissiveStrength": 4.0}));
    // Without the extension, the glow is the factor's and there is no block.
    let (material, _) = imported("emissive-plain", json!({"emissiveFactor": [1.0, 0.5, 0.25]}), &[]);
    assert_eq!(material["instanceParameters"], json!([0.0, 1.0, 1.0, 0.0]));
    assert!(material.get("surface").is_none());
}

#[test]
fn the_textures_of_the_extensions_are_reported_once_a_kind_and_left_out() {
    let texture = json!({"index": 0});
    let (material, notes) = imported(
        "textures",
        json!({"extensions": {
            "KHR_materials_clearcoat": {
                "clearcoatFactor": 1.0, "clearcoatTexture": texture, "clearcoatRoughnessTexture": texture,
                "clearcoatNormalTexture": texture,
            },
            "KHR_materials_sheen": {"sheenColorTexture": texture, "sheenRoughnessTexture": texture},
            "KHR_materials_anisotropy": {"anisotropyTexture": texture},
            "KHR_materials_transmission": {"transmissionTexture": texture},
            "KHR_materials_volume": {"thicknessTexture": texture},
            "KHR_materials_specular": {"specularTexture": texture, "specularColorTexture": texture},
        }}),
        &[
            "KHR_materials_clearcoat",
            "KHR_materials_sheen",
            "KHR_materials_anisotropy",
            "KHR_materials_transmission",
            "KHR_materials_volume",
            "KHR_materials_specular",
        ],
    );
    // Their factors are in, the textures are not named anywhere.
    assert_eq!(material["surface"], json!({"clearcoat": 1.0}));
    assert!(!material.to_string().contains("Texture"), "{material}");
    assert_eq!(
        notes,
        [
            "KHR_materials_anisotropy's textures (anisotropyTexture) are not imported; its factors are",
            "KHR_materials_clearcoat's textures (clearcoatNormalTexture, clearcoatRoughnessTexture, clearcoatTexture) \
             are not imported; its factors are",
            "KHR_materials_sheen's textures (sheenColorTexture, sheenRoughnessTexture) are not imported; its factors are",
            "KHR_materials_specular's textures (specularColorTexture, specularTexture) are not imported; its factors are",
            "KHR_materials_transmission's textures (transmissionTexture) are not imported; its factors are",
            "KHR_materials_volume's textures (thicknessTexture) are not imported; its factors are",
        ]
    );
}

#[test]
fn a_kind_is_reported_once_however_many_materials_use_it() {
    let directory = directory("twice");
    let out = directory.join("assets");
    let texture = json!({"index": 0});
    let (mut document, bytes) = model(&square(), json!({"name": "a"}), &["KHR_materials_transmission"], json!({}));
    let glass = |factor: f64| {
        json!({"name": format!("glass{factor}"), "extensions": {
            "KHR_materials_transmission": {"transmissionFactor": factor, "transmissionTexture": texture},
        }})
    };
    document["materials"] = json!([glass(0.25), glass(0.75)]);
    document["meshes"][0]["primitives"] = json!([
        document["meshes"][0]["primitives"][0].clone(),
        {
            "attributes": document["meshes"][0]["primitives"][0]["attributes"].clone(),
            "indices": document["meshes"][0]["primitives"][0]["indices"].clone(),
            "material": 1,
        },
    ]);
    let file = gltf_file(&directory.join("twice.gltf"), document, &bytes);
    let imported = import(&file, &out, &Options::default()).unwrap();
    assert_eq!(imported.materials.len(), 2);
    assert_eq!(imported.notes.len(), 1, "{:?}", imported.notes);
    let surfaces: Vec<Value> =
        imported.materials.iter().map(|key| read_json(&out.join(key))["surface"].clone()).collect();
    assert_eq!(surfaces, [json!({"transmission": 0.25}), json!({"transmission": 0.75})]);
}

#[test]
fn an_extension_the_import_does_not_read_is_reported_and_one_it_does_is_not() {
    let (material, notes) = imported(
        "unknown",
        json!({"extensions": {
            "KHR_materials_unlit": {},
            "KHR_materials_iridescence": {"iridescenceFactor": 1.0},
            "KHR_materials_clearcoat": {"clearcoatFactor": 0.5},
        }}),
        &["KHR_materials_unlit", "KHR_materials_iridescence", "KHR_materials_clearcoat", "KHR_texture_transform"],
    );
    assert_eq!(material["lighting"], "unlit");
    assert_eq!(material["surface"], json!({"clearcoat": 0.5}));
    assert_eq!(
        notes,
        [
            "the file uses KHR_materials_iridescence, which the import does not read",
            "the file uses KHR_texture_transform, which the import does not read",
        ]
    );
}

#[test]
fn a_file_that_requires_what_the_import_cannot_read_is_refused() {
    let directory = directory("required");
    let (mut document, bytes) = model(&square(), json!({}), &["KHR_materials_iridescence"], json!({}));
    document["extensionsRequired"] = json!(["KHR_materials_iridescence"]);
    let file = gltf_file(&directory.join("required.gltf"), document, &bytes);
    let error = import(&file, &directory.join("assets"), &Options::default()).unwrap_err();
    assert!(format!("{error:#}").contains("KHR_materials_iridescence"), "{error:#}");
}

#[test]
fn every_extension_at_once_gives_every_key_and_what_is_left_out() {
    let texture = json!({"index": 0});
    let (material, notes) = imported(
        "everything",
        json!({
            "name": "everything",
            "pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.25, 1.0, 1.0], "metallicFactor": 0.5, "roughnessFactor": 0.25},
            "emissiveFactor": [0.0, 0.5, 0.0],
            "extensions": {
                "KHR_materials_clearcoat": {"clearcoatFactor": 0.75, "clearcoatRoughnessFactor": 0.125, "clearcoatTexture": texture},
                "KHR_materials_sheen": {"sheenColorFactor": [0.25, 0.5, 0.75], "sheenRoughnessFactor": 0.5},
                "KHR_materials_anisotropy": {"anisotropyStrength": 0.5, "anisotropyRotation": -0.5, "anisotropyTexture": texture},
                "KHR_materials_transmission": {"transmissionFactor": 1.0},
                "KHR_materials_volume": {"thicknessFactor": 1.5, "attenuationDistance": 2.0, "attenuationColor": [1.0, 0.5, 0.25], "thicknessTexture": texture},
                "KHR_materials_ior": {"ior": 1.45},
                "KHR_materials_specular": {"specularFactor": 0.75, "specularColorFactor": [0.5, 1.0, 1.5]},
                "KHR_materials_emissive_strength": {"emissiveStrength": 2.0},
                "KHR_materials_iridescence": {},
            },
        }),
        &[
            "KHR_materials_clearcoat",
            "KHR_materials_sheen",
            "KHR_materials_anisotropy",
            "KHR_materials_transmission",
            "KHR_materials_volume",
            "KHR_materials_ior",
            "KHR_materials_specular",
            "KHR_materials_emissive_strength",
            "KHR_materials_iridescence",
        ],
    );
    assert_eq!(
        material["surface"],
        json!({
            "clearcoat": 0.75, "clearcoatRoughness": 0.125,
            "sheenColor": [0.25, 0.5, 0.75], "sheenRoughness": 0.5,
            "anisotropy": 0.5, "anisotropyRotation": -0.5,
            "transmission": 1.0,
            "thickness": 1.5, "attenuationDistance": 2.0, "attenuationColor": [1.0, 0.5, 0.25],
            "ior": 1.45,
            "specular": 0.75, "specularColor": [0.5, 1.0, 1.5],
            "emissiveStrength": 2.0,
        })
    );
    // The factors the import has always read stay.
    assert_eq!(material["color"], json!([0.5, 0.25, 1.0, 1.0]));
    assert_eq!(material["instanceParameters"], json!([0.75, 0.5, 1.0, 0.0]));
    assert_eq!(notes.len(), 4, "{notes:#?}");
    assert!(notes.iter().any(|note| note.contains("KHR_materials_iridescence")));
}
