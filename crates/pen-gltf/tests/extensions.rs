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
fn the_textures_of_the_extensions_are_imported_when_backing_images_exist() {
    let directory = directory("surface_textures");
    let out = directory.join("assets");
    let texture = json!({"index": 0});
    let (mut document, mut bytes) = model(
        &square(),
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
        json!({}),
    );
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    let img = png([200, 128, 64, 255]);
    let view_idx = document["bufferViews"].as_array().unwrap().len();
    document["bufferViews"]
        .as_array_mut()
        .unwrap()
        .push(json!({"buffer": 0, "byteOffset": bytes.len(), "byteLength": img.len()}));
    bytes.extend_from_slice(&img);
    document["images"] = json!([{"bufferView": view_idx, "mimeType": "image/png", "name": "lobe"}]);
    document["textures"] = json!([{"source": 0}]);
    document["buffers"][0]["byteLength"] = bytes.len().into();
    let file = gltf_file(&directory.join("surface_textures.gltf"), document, &bytes);
    let imported = import(&file, &out, &Options::default()).unwrap();
    assert!(imported.notes.is_empty(), "{:?}", imported.notes);
    let material = read_json(&out.join(&imported.materials[0]));
    let surface = &material["surface"];
    assert_eq!(surface["clearcoat"], 1.0);
    for key in [
        "clearcoatTexture",
        "clearcoatRoughnessTexture",
        "clearcoatNormalTexture",
        "sheenColorTexture",
        "sheenRoughnessTexture",
        "anisotropyTexture",
        "transmissionTexture",
        "thicknessTexture",
        "specularTexture",
        "specularColorTexture",
    ] {
        assert!(surface[key].as_str().is_some_and(|s| s.ends_with(".png")), "missing {key}: {surface}");
    }
}

#[test]
fn punctual_lights_and_gpu_instancing_and_extra_attributes_are_imported() {
    let directory = directory("lights_instances_attrs");
    let out = directory.join("assets");
    let (mut document, mut bytes) = model(
        &square(),
        json!({"name": "inst"}),
        &["KHR_lights_punctual", "EXT_mesh_gpu_instancing"],
        json!({}),
    );
    let mut push_f32_accessor = |doc: &mut Value, data: &[f32], acc_type: &str, count: usize| -> usize {
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        let offset = bytes.len();
        for &v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let bv = doc["bufferViews"].as_array().unwrap().len();
        doc["bufferViews"].as_array_mut().unwrap().push(json!({
            "buffer": 0, "byteOffset": offset, "byteLength": data.len() * 4,
        }));
        let acc = doc["accessors"].as_array().unwrap().len();
        doc["accessors"].as_array_mut().unwrap().push(json!({
            "bufferView": bv, "componentType": 5126, "count": count, "type": acc_type,
        }));
        acc
    };
    let uv1_acc = push_f32_accessor(&mut document, &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], "VEC2", 4);
    let col_acc = push_f32_accessor(
        &mut document,
        &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "VEC4",
        4,
    );
    let inst_t = push_f32_accessor(&mut document, &[1.0, 2.0, 3.0, -4.0, 5.0, 6.0], "VEC3", 2);
    document["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_1"] = uv1_acc.into();
    document["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"] = col_acc.into();
    document["nodes"][0]["extensions"] = json!({
        "EXT_mesh_gpu_instancing": { "attributes": { "TRANSLATION": inst_t } }
    });
    document["extensions"] = json!({
        "KHR_lights_punctual": {
            "lights": [{
                "name": "lamp",
                "type": "spot",
                "color": [1.0, 0.8, 0.5],
                "intensity": 12.0,
                "range": 25.0,
                "spot": { "innerConeAngle": 0.2, "outerConeAngle": 0.5 }
            }]
        }
    });
    document["nodes"].as_array_mut().unwrap().push(json!({
        "name": "lamp_node",
        "translation": [3.0, 4.0, 5.0],
        "extensions": { "KHR_lights_punctual": { "light": 0 } }
    }));
    document["scenes"][0]["nodes"].as_array_mut().unwrap().push(json!(1));
    document["buffers"][0]["byteLength"] = bytes.len().into();
    let file = gltf_file(&directory.join("lights_instances_attrs.gltf"), document, &bytes);
    let imported = import(&file, &out, &Options::default()).unwrap();
    assert!(imported.notes.is_empty(), "{:?}", imported.notes);
    assert_eq!(imported.lights.len(), 1);
    assert_eq!(imported.lights[0].kind, "spot");
    assert_eq!(imported.lights[0].position, [3.0, 4.0, 5.0]);
    assert_eq!(imported.lights[0].intensity, 12.0);
    assert_eq!(imported.meshes[0].instances.len(), 2);
    assert_eq!(imported.meshes[0].instances[1].translation, [-4.0, 5.0, 6.0]);
    let mesh = read_json(&out.join(&imported.meshes[0].mesh));
    assert_eq!(mesh["uvs1"].as_array().unwrap().len(), 8);
    assert_eq!(mesh["colors"].as_array().unwrap().len(), 16);
    assert_eq!(mesh["instances"]["data"].as_array().unwrap().len(), 40, "2 instances * 20 floats");
}

#[test]
fn instancing_applies_node_world_times_trs_and_validates_normalized_rotations_and_required_exts() {
    let directory = directory("inst_order");
    let out = directory.join("assets");
    // Parent node rotated 90° around +Z: (x, y, z) -> (-y, x, z).
    // Instance translated by (10, 0, 0) in node-local space -> world translation MUST be (0, 10, 0)!
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let (mut document, mut bytes) = model(
        &square(),
        json!({"name": "inst_order"}),
        &["KHR_lights_punctual", "EXT_mesh_gpu_instancing"],
        json!({}),
    );
    document["extensionsRequired"] = json!(["KHR_lights_punctual", "EXT_mesh_gpu_instancing"]);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    let t_off = bytes.len();
    for &v in &[10.0f32, 0.0, 0.0] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let t_bv = document["bufferViews"].as_array().unwrap().len();
    document["bufferViews"].as_array_mut().unwrap().push(json!({"buffer": 0, "byteOffset": t_off, "byteLength": 12}));
    let t_acc = document["accessors"].as_array().unwrap().len();
    document["accessors"].as_array_mut().unwrap().push(json!({
        "bufferView": t_bv, "componentType": 5126, "count": 1, "type": "VEC3"
    }));
    // Normalized BYTE (5120) identity rotation [0, 0, 0, 127].
    let r_off = bytes.len();
    bytes.extend_from_slice(&[0u8, 0, 0, 127]);
    let r_bv = document["bufferViews"].as_array().unwrap().len();
    document["bufferViews"].as_array_mut().unwrap().push(json!({"buffer": 0, "byteOffset": r_off, "byteLength": 4}));
    let r_acc = document["accessors"].as_array().unwrap().len();
    document["accessors"].as_array_mut().unwrap().push(json!({
        "bufferView": r_bv, "componentType": 5120, "normalized": true, "count": 1, "type": "VEC4"
    }));
    document["nodes"][0]["rotation"] = json!([0.0, 0.0, s, s]);
    document["nodes"][0]["extensions"] = json!({
        "EXT_mesh_gpu_instancing": { "attributes": { "TRANSLATION": t_acc, "ROTATION": r_acc } }
    });
    // Also test spot: {} default cone angles & tiny scale orientation preservation.
    document["extensions"] = json!({
        "KHR_lights_punctual": { "lights": [{ "type": "spot", "spot": {} }] }
    });
    document["nodes"].as_array_mut().unwrap().push(json!({
        "name": "tiny_spot",
        "rotation": [0.0, s, 0.0, s], // 90° around Y: local -Z -> world -X
        "scale": [1e-10, 1e-10, 1e-10],
        "extensions": { "KHR_lights_punctual": { "light": 0 } }
    }));
    document["scenes"][0]["nodes"].as_array_mut().unwrap().push(json!(1));
    document["buffers"][0]["byteLength"] = bytes.len().into();
    let file = gltf_file(&directory.join("inst_order.gltf"), document, &bytes);
    let imported = import(&file, &out, &Options::default()).unwrap();
    let mesh = read_json(&out.join(&imported.meshes[0].mesh));
    #[allow(clippy::cast_possible_truncation, reason = "test JSON f32")]
    let data: Vec<f32> = mesh["instances"]["data"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    // Column 3 of composed matrix (indices 12, 13, 14) must be (0, 10, 0) from Rz(90°) * (10, 0, 0).
    assert!(data[12].abs() < 1e-4 && (data[13] - 10.0).abs() < 1e-4 && data[14].abs() < 1e-4, "{data:?}");
    // Spot defaults: inner = 0.0, outer = PI/4; direction = (-1, 0, 0) despite 1e-10 node scale!
    assert_eq!(imported.lights[0].inner_cone_angle, Some(0.0));
    assert!((imported.lights[0].outer_cone_angle.unwrap() - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
    assert!((imported.lights[0].direction[0] + 1.0).abs() < 1e-4, "{:?}", imported.lights[0].direction);
}

#[test]
fn unskinned_morph_weight_animations_are_preserved_per_mesh_node() {
    let directory = directory("unskinned_morph_anim");
    let out = directory.join("assets");
    let (mut document, mut bytes) = model(&square(), json!({"name": "face"}), &[], json!({}));
    let mut push_f32 = |doc: &mut Value, data: &[f32], acc_type: &str, count: usize| -> usize {
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        let off = bytes.len();
        for &v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let bv = doc["bufferViews"].as_array().unwrap().len();
        doc["bufferViews"].as_array_mut().unwrap().push(json!({"buffer": 0, "byteOffset": off, "byteLength": data.len() * 4}));
        let acc = doc["accessors"].as_array().unwrap().len();
        doc["accessors"].as_array_mut().unwrap().push(json!({
            "bufferView": bv, "componentType": 5126, "count": count, "type": acc_type,
            "min": if acc_type == "SCALAR" { json!([data[0]]) } else { Value::Null },
            "max": if acc_type == "SCALAR" { json!([*data.last().unwrap()]) } else { Value::Null },
        }));
        if acc_type != "SCALAR" {
            doc["accessors"][acc].as_object_mut().unwrap().remove("min");
            doc["accessors"][acc].as_object_mut().unwrap().remove("max");
        }
        acc
    };
    let delta_pos = push_f32(&mut document, &[0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0], "VEC3", 4);
    document["meshes"][0]["primitives"][0]["targets"] = json!([{ "POSITION": delta_pos }]);
    let times = push_f32(&mut document, &[0.0, 1.0], "SCALAR", 2);
    let weights_a = push_f32(&mut document, &[0.0, 1.0], "SCALAR", 2);
    let weights_b = push_f32(&mut document, &[1.0, 0.0], "SCALAR", 2);
    // Add a second mesh node animated by the SAME clip with different weights.
    document["nodes"].as_array_mut().unwrap().push(json!({"name": "face_b", "mesh": 0}));
    document["scenes"][0]["nodes"].as_array_mut().unwrap().push(json!(1));
    document["animations"] = json!([{
        "name": "smile",
        "samplers": [
            { "input": times, "output": weights_a, "interpolation": "LINEAR" },
            { "input": times, "output": weights_b, "interpolation": "LINEAR" }
        ],
        "channels": [
            { "sampler": 0, "target": { "node": 0, "path": "weights" } },
            { "sampler": 1, "target": { "node": 1, "path": "weights" } }
        ]
    }]);
    document["buffers"][0]["byteLength"] = bytes.len().into();
    let file = gltf_file(&directory.join("unskinned_morph_anim.gltf"), document, &bytes);
    let imported = import(&file, &out, &Options::default()).unwrap();
    assert_eq!(imported.meshes.len(), 2);
    let m0 = read_json(&out.join(&imported.meshes[0].mesh));
    let m1 = read_json(&out.join(&imported.meshes[1].mesh));
    // Each unskinned mesh has its own rig containing ONLY its own morph channel (not summed together!).
    assert_eq!(m0["clips"][0]["channels"].as_array().unwrap().len(), 1);
    assert_eq!(m1["clips"][0]["channels"].as_array().unwrap().len(), 1);
    assert_eq!(m0["clips"][0]["channels"][0]["path"], "weights");
    assert_eq!(m0["clips"][0]["channels"][0]["values"], json!([0.0, 1.0]));
    assert_eq!(m1["clips"][0]["channels"][0]["values"], json!([1.0, 0.0]));
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
        &["KHR_materials_unlit", "KHR_materials_iridescence", "KHR_materials_clearcoat", "KHR_materials_dispersion"],
    );
    assert_eq!(material["lighting"], "unlit");
    assert_eq!(material["surface"], json!({"clearcoat": 0.5}));
    assert_eq!(
        notes,
        [
            "the file uses KHR_materials_iridescence, which the import does not read",
            "the file uses KHR_materials_dispersion, which the import does not read",
        ]
    );
}

#[test]
fn orm_emissive_and_texture_transform_are_imported_without_unread_notes() {
    let directory = directory("orm_uv");
    let out = directory.join("assets");
    let (mut document, mut bytes) = model(
        &square(),
        json!({
            "name": "pbr",
            "pbrMetallicRoughness": {
                "baseColorTexture": {
                    "index": 0,
                    "extensions": {
                        "KHR_texture_transform": {
                            "offset": [0.25, 0.5],
                            "scale": [2.0, 3.0],
                            "rotation": 0.5,
                        }
                    }
                },
                "metallicRoughnessTexture": {"index": 1},
            },
            "occlusionTexture": {"index": 2},
            "emissiveTexture": {"index": 0},
            "emissiveFactor": [1.0, 0.5, 0.25],
        }),
        &["KHR_texture_transform"],
        json!({}),
    );
    let append_png = |document: &mut Value, bytes: &mut Vec<u8>, name: &str, color: [u8; 4]| {
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        let img = png(color);
        let view_idx = document["bufferViews"].as_array().unwrap().len();
        document["bufferViews"]
            .as_array_mut()
            .unwrap()
            .push(json!({"buffer": 0, "byteOffset": bytes.len(), "byteLength": img.len()}));
        bytes.extend_from_slice(&img);
        let img_idx = document["images"].as_array().unwrap().len();
        document["images"]
            .as_array_mut()
            .unwrap()
            .push(json!({"name": name, "bufferView": view_idx, "mimeType": "image/png"}));
        document["textures"].as_array_mut().unwrap().push(json!({"source": img_idx}));
    };
    // Texture 1: metallicRoughness (G=roughness 128, B=metallic 200)
    append_png(&mut document, &mut bytes, "mr", [10, 128, 200, 255]);
    // Texture 2: occlusion (R=AO 90)
    append_png(&mut document, &mut bytes, "occ", [90, 20, 30, 255]);
    document["buffers"][0]["byteLength"] = json!(bytes.len());

    let file = gltf_file(&directory.join("orm_uv.gltf"), document, &bytes);
    let imported = import(&file, &out, &Options::default()).unwrap();
    assert!(imported.notes.is_empty(), "{:?}", imported.notes);
    let material = read_json(&out.join(&imported.materials[0]));
    assert_eq!(material["uvTransform"], json!({"offset": [0.25, 0.5], "scale": [2.0, 3.0], "rotation": 0.5}));
    assert_eq!(material["emissiveTexture"], material["texture"]);
    let orm_key = material["ormTexture"].as_str().expect("packed ormTexture");
    let orm_img = image::open(out.join(orm_key)).unwrap().to_rgba8();
    assert_eq!(orm_img.get_pixel(0, 0).0, [90, 128, 200, 255]);
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
            "clearcoat": 0.75, "clearcoatRoughness": 0.125, "clearcoatTexture": "everything_white.png",
            "sheenColor": [0.25, 0.5, 0.75], "sheenRoughness": 0.5,
            "anisotropy": 0.5, "anisotropyRotation": -0.5, "anisotropyTexture": "everything_white.png",
            "transmission": 1.0,
            "thickness": 1.5, "thicknessTexture": "everything_white.png", "attenuationDistance": 2.0, "attenuationColor": [1.0, 0.5, 0.25],
            "ior": 1.45,
            "specular": 0.75, "specularColor": [0.5, 1.0, 1.5],
            "emissiveStrength": 2.0,
        })
    );
    // The factors the import has always read stay.
    assert_eq!(material["color"], json!([0.5, 0.25, 1.0, 1.0]));
    assert_eq!(material["instanceParameters"], json!([0.75, 0.5, 1.0, 0.0]));
    assert_eq!(notes.len(), 1, "{notes:#?}");
    assert!(notes.iter().any(|note| note.contains("KHR_materials_iridescence")));
}
