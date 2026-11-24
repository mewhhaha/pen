//! glTF 2.0 import for pen's asset pipeline. A `.gltf` or `.glb` file
//! becomes the renderer's assets, written into an asset directory as any of
//! a checkout's recipes writes them (`pen gltf`, which a Justfile recipe
//! runs and `pipeline.json` reruns as the file changes):
//!
//! - a mesh JSON for each triangle primitive the file's scene places: static
//!   ones where the scene puts them, skinned ones in their bind pose with
//!   their four joints and weights a vertex and their rig (the skeleton, the
//!   joints and every animation of the file as a clip, see
//!   `pen_render::animation`), which posed draws move; and, for a primitive
//!   with a normal map or an anisotropic material, its `tangents`: the file's
//!   `TANGENT` attribute or, where it has none, MikkTSpace tangents ([`tangents`]);
//! - a material JSON for each material those use (base color and its
//!   texture, normal map, PBR factors as `instanceParameters` [gloss, metal,
//!   glow, 0], unlit, blend and cutoff, and the factors of the extensions
//!   below as its `surface` block), naming the shader the import is told;
//! - the PNG and JPEG images those materials name;
//! - with vertex-animation textures asked for, each skinned primitive's clips
//!   baked into one (vat.rs) and a material that draws with it, and the
//!   shader that does ([`VAT_WGSL`]) if the directory has none;
//! - and `NAME.import.json`, what it wrote: each mesh's material, and the
//!   clips by name in the order poses name them.
//!
//! Material extensions read: `KHR_materials_unlit`, `KHR_materials_emissive_strength`
//! (multiplying the emissive `glow` parameter, and as the surface's `emissiveStrength`),
//! and, as factors only, `KHR_materials_clearcoat`, `_sheen`, `_anisotropy`,
//! `_transmission`, `_volume`, `_ior` and `_specular`, into the material's
//! `surface` block (docs/materials.md). The textures of those (clear coat,
//! sheen colour and roughness, anisotropy, transmission, thickness, specular
//! and specular colour) are not imported: each kind with one is reported in
//! [`Imported::notes`], as is any other extension the file uses, which the import
//! does not read.
//!
//! Files whose content is unchanged are left alone, so rerunning an import
//! sets nothing else off; files the last import of the same name made and
//! this one does not are removed.
use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use pen_render::animation::{Channel, Clip, Interpolation, Joint, Node, Path as Property, Rig, RigJson, Transform};
use serde::Serialize;
use serde_json::{Value, json};

mod gtex;
pub mod optimize;
pub mod scene_import;
pub mod tangents;
pub mod vat;

pub use scene_import::{
    ImportedScene, SceneCameraData, SceneColliderData, SceneImportOptions, SceneImportPreset, SceneLightData,
    SceneLodLevel, SceneManifest, SceneMeshRef, SceneNodeData, SceneTransform, import_scene,
};

/// A shader that draws with vertex-animation textures, which an import
/// asked to bake them writes as their materials' shader when the asset
/// directory has none of that name.
pub const VAT_WGSL: &str = include_str!("vat.wgsl");

/// How to import.
#[derive(Clone, Debug)]
pub struct Options {
    /// What the assets' names start with; by default the file's name.
    pub name: Option<String>,
    /// The shader the materials name.
    pub shader: String,
    /// Vertex-animation textures for skinned primitives.
    pub vat: Option<VatOptions>,
    /// Reorder vertex buffers in first-use fetch order in addition to vertex-cache index sorting.
    pub optimize_vertices: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { name: None, shader: "scene.wgsl".to_owned(), vat: None, optimize_vertices: false }
    }
}

#[derive(Clone, Debug)]
pub struct VatOptions {
    /// Frames a second of each clip.
    pub fps: f32,
    /// The shader the textures' materials name, which reads them.
    pub shader: String,
}

/// What an import wrote, by asset key; `NAME.import.json` holds it.
#[derive(Debug, Default, Serialize)]
pub struct Imported {
    /// The file it came from, as given.
    pub source: String,
    pub meshes: Vec<ImportedMesh>,
    pub materials: Vec<String>,
    pub images: Vec<String>,
    /// The file's animations, which are every skinned mesh's clips.
    pub clips: Vec<ImportedClip>,
    /// Punctual lights (`KHR_lights_punctual`) placed in the scene.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lights: Vec<ImportedLight>,
    /// Every asset it made, written or found unchanged: what the next import
    /// of the same name removes if it no longer makes it.
    pub files: Vec<String>,
    /// The keys whose files it wrote, their content having changed.
    #[serde(skip)]
    pub changed: Vec<String>,
    /// The keys an earlier import made that this one no longer does, which
    /// it removed.
    #[serde(skip)]
    pub removed: Vec<String>,
    /// What the file has that the import leaves out.
    #[serde(skip)]
    pub notes: Vec<String>,
}

/// One `KHR_lights_punctual` light placed in the scene's world space.
/// Per `KHR_lights_punctual`, `intensity` is in candela (`lm/sr`) for `"point"`
/// and `"spot"` lights and in lux (`lm/m^2`) for `"directional"` lights.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ImportedLight {
    pub name: String,
    pub kind: String,
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inner_cone_angle: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outer_cone_angle: Option<f32>,
}

/// One `EXT_mesh_gpu_instancing` TRS transform on a mesh node.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ImportedInstance {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

#[derive(Debug, Serialize)]
pub struct ImportedMesh {
    pub mesh: String,
    pub material: String,
    /// Its joints, 0 for a static mesh.
    pub joints: usize,
    /// Its morph targets, omitted when 0.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub morph_targets: usize,
    /// Its vertex-animation texture's material, if it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vat: Option<String>,
    /// GPU instances from `EXT_mesh_gpu_instancing`, if present on the node.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub instances: Vec<ImportedInstance>,
}

fn is_zero(v: &usize) -> bool {
    *v == 0
}

#[derive(Debug, Serialize)]
pub struct ImportedClip {
    pub name: String,
    pub duration: f32,
}

/// One morph target's per-vertex position and normal deltas.
pub struct MorphTarget {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
}

/// A primitive's vertices and triangles.
pub struct Geometry {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    /// Optional secondary UV set (`TEXCOORD_1`) for lightmaps or detail textures.
    pub uvs1: Vec<[f32; 2]>,
    /// Optional RGBA linear vertex colors (`COLOR_0`).
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    /// A skinned primitive's four joints and weights a vertex.
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    /// Each vertex's tangent (xyz and a handedness of -1 or 1), or none for
    /// the primitive.
    pub tangents: Vec<[f32; 4]>,
    /// Morph targets (blendshapes), if any.
    pub morphs: Vec<MorphTarget>,
}

#[derive(Serialize)]
struct MorphTargetFile {
    positions: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    normals: Option<Vec<f32>>,
}

#[derive(Serialize)]
struct MeshInstancesFile {
    data: Vec<f32>,
    min: [f32; 3],
    max: [f32; 3],
}

/// Mesh JSON as the renderer reads it (`pen_render::pack_mesh`).
#[derive(Serialize)]
struct MeshFile<'a> {
    positions: Vec<f32>,
    normals: Vec<f32>,
    uvs: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    uvs1: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    colors: Option<Vec<f32>>,
    indices: &'a [u32],
    #[serde(skip_serializing_if = "Option::is_none")]
    tangents: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    morphs: Option<Vec<MorphTargetFile>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instances: Option<MeshInstancesFile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    joints: Option<Vec<u16>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    weights: Option<Vec<f32>>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    rig: Option<RigJson>,
}

/// Imports the glTF file `source` into the asset directory `out`.
pub fn import(source: &Path, out: &Path, options: &Options) -> Result<Imported> {
    let bytes = std::fs::read(source).with_context(|| format!("reading {}", source.display()))?;
    let stem = source.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_default();
    let name = sanitized(options.name.as_deref().unwrap_or(&stem));
    let gltf::Gltf { document, blob } = gltf::Gltf::from_slice_without_validation(&bytes)
        .with_context(|| format!("reading {} as glTF", source.display()))?;
    for required in document.extensions_required() {
        ensure!(
            READ_EXTENSIONS.contains(&required),
            "reading {} as glTF: unsupported required extension {required}",
            source.display()
        );
    }
    let directory = source.parent().unwrap_or(Path::new("."));
    let buffers = document
        .buffers()
        .map(|buffer| buffer_data(&buffer, blob.as_deref(), directory))
        .collect::<Result<Vec<_>>>()?;
    let scene = document.default_scene().or_else(|| document.scenes().next()).context("the file has no scene")?;
    std::fs::create_dir_all(out).with_context(|| format!("making {}", out.display()))?;

    // Every node's parent, and the scene's nodes, parents first, each with
    // its place in the scene.
    let mut parents = vec![None; document.nodes().len()];
    for node in document.nodes() {
        for child in node.children() {
            parents[child.index()] = Some(node.index());
        }
    }
    let mut placed: Vec<(gltf::Node, [f64; 16])> = Vec::new();
    let mut stack: Vec<(gltf::Node, [f64; 16])> = scene.nodes().map(|node| (node, IDENTITY)).collect();
    stack.reverse();
    let mut seen = HashSet::new();
    while let Some((node, above)) = stack.pop() {
        ensure!(seen.insert(node.index()), "the scene reaches node {} twice", node.index());
        let world = multiply(&above, &columns(node.transform().matrix()));
        stack.extend(node.children().collect::<Vec<_>>().into_iter().rev().map(|child| (child, world)));
        placed.push((node, world));
    }
    let order: Vec<usize> = placed.iter().map(|(node, _)| node.index()).collect();

    let mut writer = Writer {
        out,
        name: name.clone(),
        imported: Imported { source: source.display().to_string(), ..Default::default() },
        taken: HashSet::new(),
        materials: HashMap::new(),
        images: HashMap::new(),
    };
    writer.imported.clips = document
        .animations()
        .enumerate()
        .map(|(index, animation)| {
            let name = animation.name().map_or_else(|| format!("clip{index}"), str::to_owned);
            Ok(ImportedClip { name, duration: duration(&animation, &buffers)? })
        })
        .collect::<Result<_>>()?;
    // Import KHR_lights_punctual scene lights placed at node world transforms.
    for (node, world) in &placed {
        let Some(light) = node.light() else { continue };
        let (kind, inner_cone_angle, outer_cone_angle) = match light.kind() {
            gltf::khr_lights_punctual::Kind::Directional => ("directional".to_owned(), None, None),
            gltf::khr_lights_punctual::Kind::Point => ("point".to_owned(), None, None),
            gltf::khr_lights_punctual::Kind::Spot { inner_cone_angle, outer_cone_angle } => {
                ("spot".to_owned(), Some(inner_cone_angle), Some(outer_cone_angle))
            }
        };
        let lname = light
            .name()
            .or_else(|| node.name())
            .map_or_else(|| format!("light{}", light.index()), str::to_owned);
        let color = light.color().map(|c| if c.is_finite() { c.max(0.0) } else { 1.0 });
        let intensity = if light.intensity().is_finite() { light.intensity().max(0.0) } else { 1.0 };
        let range = light.range().filter(|r| r.is_finite() && *r > 0.0);
        let position = [narrow(world[12]), narrow(world[13]), narrow(world[14])];
        // glTF lights point along local -Z; scale-normalize so tiny node scales keep orientation.
        let dir_raw = [-world[8], -world[9], -world[10]];
        let scale_max = dir_raw[0].abs().max(dir_raw[1].abs()).max(dir_raw[2].abs());
        let direction = if scale_max > 0.0 && scale_max.is_finite() {
            let s = dir_raw.map(|v| v / scale_max);
            let len = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
            s.map(|v| narrow(v / len))
        } else {
            [0.0, 0.0, -1.0]
        };
        writer.imported.lights.push(ImportedLight {
            name: lname,
            kind,
            position,
            direction,
            color,
            intensity,
            range,
            inner_cone_angle,
            outer_cone_angle,
        });
    }

    let parts: Vec<(&gltf::Node, &[f64; 16], gltf::Mesh)> =
        placed.iter().filter_map(|(node, world)| node.mesh().map(|mesh| (node, world, mesh))).collect();
    let primitives: usize = parts.iter().map(|(_, _, mesh)| mesh.primitives().len()).sum();
    ensure!(primitives > 0, "the file's scene has no meshes");
    let mut rigs: HashMap<(Option<usize>, Option<usize>), Rig> = HashMap::new();
    for (node, world, mesh) in parts {
        let gpu_instances = read_gpu_instances(&document, node, &buffers)?;
        let has_morph_anim = node_has_morph_animation(&document, node.index());
        let several = mesh.primitives().len() > 1;
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                writer.note(format!("mesh {} has a primitive of {:?}, not triangles", mesh.index(), primitive.mode()));
                continue;
            }
            let part = if primitives == 1 {
                name.clone()
            } else {
                let base = mesh.name().or(node.name()).map_or_else(|| format!("mesh{}", mesh.index()), sanitized);
                if several { format!("{name}_{base}_{}", primitive.index()) } else { format!("{name}_{base}") }
            };
            let skin = node.skin().filter(|_| primitive.get(&gltf::Semantic::Joints(0)).is_some());
            let morph_only = skin.is_none() && primitive.morph_targets().len() > 0 && has_morph_anim && gpu_instances.is_empty();
            let rig = if let Some(skin) = &skin {
                let key = (Some(skin.index()), has_morph_anim.then_some(node.index()));
                Some(&*match rigs.entry(key) {
                    std::collections::hash_map::Entry::Occupied(known) => known.into_mut(),
                    std::collections::hash_map::Entry::Vacant(slot) => slot.insert(rig(
                        &document,
                        Some(skin),
                        has_morph_anim.then_some(node.index()),
                        &parents,
                        &order,
                        &buffers,
                        &mut writer.imported.notes,
                    )?),
                })
            } else if morph_only {
                let key = (None, Some(node.index()));
                Some(&*match rigs.entry(key) {
                    std::collections::hash_map::Entry::Occupied(known) => known.into_mut(),
                    std::collections::hash_map::Entry::Vacant(slot) => slot.insert(rig(
                        &document,
                        None,
                        Some(node.index()),
                        &parents,
                        &order,
                        &buffers,
                        &mut writer.imported.notes,
                    )?),
                })
            } else {
                None
            };
            // A skinned mesh stays in its bind pose; an instanced mesh stays in mesh-local space so
            // `node_world * T * R * S` applies in the right order.
            let wants = wants_tangents(&primitive.material());
            let bake_world = (rig.is_none() && gpu_instances.is_empty()).then_some(world);
            let mut geometry = geometry(&primitive, &buffers, skin.is_some(), bake_world, wants)
                .with_context(|| format!("mesh {} primitive {}", mesh.index(), primitive.index()))?;
            if morph_only {
                let vcount = geometry.positions.len();
                geometry.joints = vec![[0, 0, 0, 0]; vcount];
                geometry.weights = vec![[1.0, 0.0, 0.0, 0.0]; vcount];
            }
            if options.optimize_vertices {
                optimize::optimize_geometry(&mut geometry);
            } else if geometry.indices.len() >= 9 {
                geometry.indices = optimize::optimize_vertex_cache(&geometry.indices, geometry.positions.len());
            }
            if wants && geometry.tangents.is_empty() {
                writer.note(format!(
                    "mesh {} has a normal map or anisotropy but no texture coordinates to make tangents from",
                    mesh.index()
                ));
            }
            if primitive.get(&gltf::Semantic::Joints(1)).is_some() {
                writer.note(format!(
                    "mesh {} binds vertices to more than four joints; the four first count",
                    mesh.index()
                ));
            }
            let mesh_instances = (!gpu_instances.is_empty() && rig.is_none())
                .then(|| build_mesh_instances(&geometry.positions, world, &gpu_instances));
            let (material, material_json) = writer.material(&document, &primitive.material(), &buffers, options)?;
            let mesh_key = writer.unique(&part, ".mesh.json");
            writer.write_mesh(&mesh_key, &geometry, rig, mesh_instances)?;
            let vat = match (&options.vat, rig) {
                (Some(vat), Some(rig)) => writer.vat(&part, &geometry, rig, vat, &material_json)?,
                _ => None,
            };
            let joints = rig.map_or(0, |rig| rig.joints.len());
            let morph_targets = geometry.morphs.len();
            writer.imported.meshes.push(ImportedMesh {
                mesh: mesh_key,
                material,
                joints,
                morph_targets,
                vat,
                instances: gpu_instances.clone(),
            });
        }
    }
    for name in document.extensions_used() {
        if !READ_EXTENSIONS.contains(&name) {
            writer.note(format!("the file uses {name}, which the import does not read"));
        }
    }
    if let Some(vat) = &options.vat
        && writer.imported.meshes.iter().any(|mesh| mesh.vat.is_some())
    {
        ensure!(
            !vat.shader.is_empty() && !vat.shader.starts_with('/') && !vat.shader.split('/').any(|part| part == ".."),
            "the vertex-animation shader must be a key inside the asset directory"
        );
        if !out.join(&vat.shader).exists() {
            writer.write(&vat.shader, VAT_WGSL.as_bytes())?;
        }
    }
    // What the last import of this name made and this one does not (a part
    // renamed, a material gone) goes.
    let key = format!("{name}.import.json");
    let mut files: Vec<String> = writer.taken.iter().cloned().collect();
    files.sort();
    #[derive(serde::Deserialize)]
    struct Earlier {
        #[serde(default)]
        files: Vec<String>,
    }
    let earlier = std::fs::read(out.join(&key)).ok().and_then(|bytes| serde_json::from_slice::<Earlier>(&bytes).ok());
    for stale in earlier.map(|earlier| earlier.files).unwrap_or_default() {
        let inside = !stale.is_empty() && !stale.starts_with('/') && !stale.split(['/', '\\']).any(|part| part == "..");
        if inside && !writer.taken.contains(&stale) && std::fs::remove_file(out.join(&stale)).is_ok() {
            writer.imported.removed.push(stale);
        }
    }
    writer.imported.files = files;
    let manifest = serde_json::to_string_pretty(&writer.imported)? + "\n";
    writer.write(&key, manifest.as_bytes())?;
    Ok(writer.imported)
}

/// What an asset's name may hold: letters, digits and underscores.
fn sanitized(name: &str) -> String {
    let name: String = name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    if name.is_empty() { "model".to_owned() } else { name }
}

/// The material extensions whose factors become a material's `surface`
/// block, each with the textures it can name, which are left out.
const SURFACE_EXTENSIONS: [(&str, &[&str]); 7] = [
    ("KHR_materials_clearcoat", &["clearcoatTexture", "clearcoatRoughnessTexture", "clearcoatNormalTexture"]),
    ("KHR_materials_sheen", &["sheenColorTexture", "sheenRoughnessTexture"]),
    ("KHR_materials_anisotropy", &["anisotropyTexture"]),
    ("KHR_materials_transmission", &["transmissionTexture"]),
    ("KHR_materials_volume", &["thicknessTexture"]),
    ("KHR_materials_ior", &[]),
    ("KHR_materials_specular", &["specularTexture", "specularColorTexture"]),
];

/// The extensions a file may use or require that the import actually reads.
const READ_EXTENSIONS: [&str; 12] = [
    "KHR_materials_unlit",
    "KHR_materials_emissive_strength",
    "KHR_materials_clearcoat",
    "KHR_materials_sheen",
    "KHR_materials_anisotropy",
    "KHR_materials_transmission",
    "KHR_materials_volume",
    "KHR_materials_ior",
    "KHR_materials_specular",
    "KHR_texture_transform",
    "KHR_lights_punctual",
    "EXT_mesh_gpu_instancing",
];

const IDENTITY: [f64; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

fn columns(matrix: [[f32; 4]; 4]) -> [f64; 16] {
    let mut flat = [0.0; 16];
    for (column, values) in matrix.iter().enumerate() {
        for (row, value) in values.iter().enumerate() {
            flat[column * 4 + row] = f64::from(*value);
        }
    }
    flat
}

fn multiply(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut out = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            out[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
        }
    }
    out
}

#[allow(clippy::cast_possible_truncation, reason = "narrowing f64 to the f32 assets hold")]
fn narrow(value: f64) -> f32 {
    value as f32
}

/// A material's number as people would write it: 0.2 rather than the f32
/// nearest it, 0.19999998807907104.
fn tidy(value: f32) -> f64 {
    (f64::from(value) * 1e6).round() / 1e6
}

/// A surface number as people would write it, too small or large as it may be:
/// the shortest decimal of the f32 nearest it, within what an f32 holds.
fn short(value: f64) -> f64 {
    let most = f64::from(f32::MAX);
    format!("{}", narrow(value.clamp(-most, most))).parse().unwrap_or(0.0)
}

/// Converts a glTF `KHR_texture_transform` extension object into the renderer's
/// `uvTransform` block, omitting identity transforms.
fn uv_transform(extension: Option<&Value>) -> Option<Value> {
    let object = extension?.as_object()?;
    let pair = |key: &str| -> Option<[f64; 2]> {
        let arr = object.get(key)?.as_array()?;
        let x = arr.first()?.as_f64().filter(|v| v.is_finite())?;
        let y = arr.get(1)?.as_f64().filter(|v| v.is_finite())?;
        Some([short(x), short(y)])
    };
    let offset = pair("offset");
    let scale = pair("scale");
    let rotation = object.get("rotation").and_then(Value::as_f64).filter(|v| v.is_finite()).map(short);
    let mut map = serde_json::Map::new();
    if let Some(o) = offset
        && o != [0.0, 0.0]
    {
        map.insert("offset".to_owned(), json!(o));
    }
    if let Some(s) = scale
        && s != [1.0, 1.0]
    {
        map.insert("scale".to_owned(), json!(s));
    }
    if let Some(r) = rotation
        && r != 0.0
    {
        map.insert("rotation".to_owned(), json!(r));
    }
    (!map.is_empty()).then_some(Value::Object(map))
}

/// A buffer's bytes: the GLB's binary chunk, a data URI's, or a file's
/// beside the source.
fn buffer_data(buffer: &gltf::Buffer, blob: Option<&[u8]>, directory: &Path) -> Result<Vec<u8>> {
    let data = match buffer.source() {
        gltf::buffer::Source::Bin => blob.context("a buffer names the binary chunk, which the file lacks")?.to_vec(),
        gltf::buffer::Source::Uri(uri) => uri_bytes(uri, directory)?,
    };
    ensure!(data.len() >= buffer.length(), "buffer {} is shorter than it says", buffer.index());
    Ok(data)
}

fn uri_bytes(uri: &str, directory: &Path) -> Result<Vec<u8>> {
    if let Some(data) = uri.strip_prefix("data:") {
        let (kind, payload) = data.split_once(',').context("a data URI without data")?;
        ensure!(kind.ends_with(";base64"), "data URIs must be base64");
        return base64(payload);
    }
    let path = percent_decoded(uri)?;
    ensure!(!path.contains("://"), "{uri}: only files beside the source and data URIs are read");
    let path = directory.join(path);
    std::fs::read(&path).with_context(|| format!("reading {}", path.display()))
}

fn percent_decoded(uri: &str) -> Result<String> {
    let bytes = uri.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let hex = uri.get(index + 1..index + 3).context("a URI's % escape is cut short")?;
            out.push(u8::from_str_radix(hex, 16).context("a URI's % escape is not hexadecimal")?);
            index += 3;
        } else {
            out.push(bytes[index]);
            index += 1;
        }
    }
    Ok(String::from_utf8(out)?)
}

fn base64(text: &str) -> Result<Vec<u8>> {
    let value = |byte: u8| -> Result<u32> {
        Ok(u32::from(match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => bail!("a data URI's base64 has a stray character"),
        }))
    };
    let digits: Vec<u8> = text.bytes().filter(|byte| !byte.is_ascii_whitespace() && *byte != b'=').collect();
    let mut out = Vec::with_capacity(digits.len() * 3 / 4);
    for chunk in digits.chunks(4) {
        let mut word = 0;
        for (index, &digit) in chunk.iter().enumerate() {
            word |= value(digit)? << (18 - 6 * index);
        }
        let bytes = word.to_be_bytes();
        out.extend_from_slice(&bytes[1..chunk.len()]);
    }
    Ok(out)
}

/// The animation's length: its last key.
fn duration(animation: &gltf::Animation, buffers: &[Vec<u8>]) -> Result<f32> {
    let mut longest = 0.0f32;
    for channel in animation.channels() {
        let reader = channel.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));
        let times = reader.read_inputs().context("an animation channel has no times")?;
        longest = times.fold(longest, f32::max);
    }
    Ok(longest)
}

/// The skin's rig: its joints and the nodes above them, parents first, and
fn node_has_morph_animation(document: &gltf::Document, node_index: usize) -> bool {
    document.animations().any(|anim| {
        anim.channels().any(|ch| {
            ch.target().node().index() == node_index
                && ch.target().property() == gltf::animation::Property::MorphTargetWeights
        })
    })
}

/// Builds a rig containing a skin's joints (if skinned) plus `morph_node` (when
/// the mesh has morph weight animations targeting its node) and their ancestors.
fn rig(
    document: &gltf::Document,
    skin: Option<&gltf::Skin>,
    morph_node: Option<usize>,
    parents: &[Option<usize>],
    order: &[usize],
    buffers: &[Vec<u8>],
    _notes: &mut Vec<String>,
) -> Result<Rig> {
    let get = |buffer: gltf::Buffer| buffers.get(buffer.index()).map(Vec::as_slice);
    let mut needed = HashSet::new();
    let mut add_chain = |start: usize| {
        let mut node = Some(start);
        while let Some(index) = node {
            if !needed.insert(index) {
                break;
            }
            node = parents[index];
        }
    };
    if let Some(mnode) = morph_node {
        add_chain(mnode);
    }
    if let Some(skin) = skin {
        for joint in skin.joints() {
            add_chain(joint.index());
        }
    }
    let nodes_in_order: Vec<usize> = order.iter().copied().filter(|index| needed.contains(index)).collect();
    ensure!(nodes_in_order.len() == needed.len(), "rig references nodes outside the scene");
    let place: HashMap<usize, u32> = nodes_in_order
        .iter()
        .enumerate()
        .map(|(place, &index)| Ok((index, u32::try_from(place)?)))
        .collect::<Result<_>>()?;
    let nodes = nodes_in_order
        .iter()
        .map(|&index| {
            let node = document.nodes().nth(index).unwrap();
            let (translation, rotation, scale) = node.transform().decomposed();
            Node {
                name: node.name().map_or_else(|| format!("node{index}"), str::to_owned),
                parent: parents[index].map(|parent| place[&parent]),
                rest: Transform { translation, rotation, scale },
            }
        })
        .collect();
    let joints = if let Some(skin) = skin {
        let inverse_binds: Vec<[[f32; 4]; 4]> = match skin.reader(get).read_inverse_bind_matrices() {
            Some(matrices) => matrices.collect(),
            None => vec![
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
                skin.joints().len()
            ],
        };
        ensure!(inverse_binds.len() >= skin.joints().len(), "skin {} lacks inverse bind matrices", skin.index());
        skin.joints()
            .zip(inverse_binds)
            .map(|(joint, inverse)| Joint { node: place[&joint.index()], inverse_bind: columns(inverse).map(narrow) })
            .collect()
    } else {
        let mnode = morph_node.context("unskinned morph rig requires morph_node")?;
        vec![Joint { node: place[&mnode], inverse_bind: IDENTITY.map(narrow) }]
    };
    let mut clips = Vec::new();
    for (index, animation) in document.animations().enumerate() {
        let mut channels = Vec::new();
        for channel in animation.channels() {
            let target = channel.target();
            let target_idx = target.node().index();
            let Some(&node) = place.get(&target_idx) else { continue };
            let reader = channel.reader(get);
            let times: Vec<f32> = reader.read_inputs().context("an animation channel has no times")?.collect();
            let (path, values): (Property, Vec<f32>) = match reader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::Translations(values)) => {
                    (Property::Translation, values.flatten().collect())
                }
                Some(gltf::animation::util::ReadOutputs::Rotations(values)) => {
                    (Property::Rotation, values.into_f32().flatten().collect())
                }
                Some(gltf::animation::util::ReadOutputs::Scales(values)) => {
                    (Property::Scale, values.flatten().collect())
                }
                Some(gltf::animation::util::ReadOutputs::MorphTargetWeights(values)) => {
                    // Only bind morph-weight channels belonging to this mesh node.
                    if morph_node != Some(target_idx) {
                        continue;
                    }
                    (Property::Weights, values.into_f32().collect())
                }
                None => bail!("an animation channel has no values"),
            };
            let interpolation = match channel.sampler().interpolation() {
                gltf::animation::Interpolation::Linear => Interpolation::Linear,
                gltf::animation::Interpolation::Step => Interpolation::Step,
                gltf::animation::Interpolation::CubicSpline => Interpolation::Cubic,
            };
            channels.push(Channel { node, path, interpolation, times, values });
        }
        let name = animation.name().map_or_else(|| format!("clip{index}"), str::to_owned);
        clips.push(Clip { name, duration: duration(&animation, buffers)?, channels });
    }
    let rig = Rig { nodes, joints, clips };
    rig.validate().with_context(|| format!("rig for {:?}", morph_node))?;
    Ok(rig)
}

/// The primitive's vertices and triangles; `world` places a static one.
fn geometry(
    primitive: &gltf::Primitive,
    buffers: &[Vec<u8>],
    skinned: bool,
    world: Option<&[f64; 16]>,
    generate_tangents: bool,
) -> Result<Geometry> {
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));
    let mut positions: Vec<[f32; 3]> = reader.read_positions().context("a primitive has no positions")?.collect();
    let count = positions.len();
    ensure!(count > 0, "a primitive has no vertices");
    let mut indices: Vec<u32> = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect(),
        None => (0..u32::try_from(count)?).collect(),
    };
    ensure!(
        !indices.is_empty() && indices.len().is_multiple_of(3) && indices.iter().all(|&index| (index as usize) < count),
        "a primitive's triangles must name its vertices"
    );
    let textured = reader.read_tex_coords(0).is_some();
    let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
        Some(coordinates) => coordinates.into_f32().collect(),
        None => vec![[0.0; 2]; count],
    };
    let uvs1: Vec<[f32; 2]> = reader.read_tex_coords(1).map(|c| c.into_f32().collect()).unwrap_or_default();
    let colors: Vec<[f32; 4]> = reader.read_colors(0).map(|c| c.into_rgba_f32().collect()).unwrap_or_default();
    ensure!(
        (uvs1.is_empty() || (uvs1.len() == count && uvs1.iter().flatten().all(|v| v.is_finite())))
            && (colors.is_empty() || (colors.len() == count && colors.iter().flatten().all(|v| v.is_finite()))),
        "a primitive's secondary UVs and vertex colors must be finite and match the vertex count"
    );
    let mut tangents: Vec<[f32; 4]> = reader.read_tangents().map(Iterator::collect).unwrap_or_default();
    ensure!(
        tangents.is_empty() || tangents.iter().flatten().all(|value| value.is_finite()),
        "a primitive's tangents must be finite"
    );
    let mut normals: Vec<[f32; 3]> = match reader.read_normals() {
        Some(normals) => normals.collect(),
        None => smooth_normals(&positions, &indices),
    };
    let (mut joints, mut weights) = (Vec::new(), Vec::new());
    if skinned {
        joints = reader.read_joints(0).context("a skinned primitive has no joints")?.into_u16().collect();
        weights = reader.read_weights(0).context("a skinned primitive has no weights")?.into_f32().collect();
        // Weights that add up to 1, and a vertex bound to nothing bound to
        // the first joint.
        for (joints, weights) in joints.iter_mut().zip(weights.iter_mut()) {
            let total: f32 = weights.iter().map(|weight| weight.max(0.0)).sum();
            if total > 0.0 && total.is_finite() {
                *weights = weights.map(|weight| weight.max(0.0) / total);
            } else {
                *joints = [joints[0], 0, 0, 0];
                *weights = [1.0, 0.0, 0.0, 0.0];
            }
        }
    }
    let mut morphs: Vec<MorphTarget> = Vec::new();
    for (pos_iter, norm_iter, _) in reader.read_morph_targets() {
        let m_positions: Vec<[f32; 3]> = match pos_iter {
            Some(iter) => iter.collect(),
            None => vec![[0.0; 3]; count],
        };
        let m_normals: Vec<[f32; 3]> = match norm_iter {
            Some(iter) => iter.collect(),
            None => vec![[0.0; 3]; count],
        };
        ensure!(
            m_positions.len() == count
                && m_normals.len() == count
                && m_positions.iter().flatten().all(|v| v.is_finite())
                && m_normals.iter().flatten().all(|v| v.is_finite()),
            "a morph target's position and normal deltas must be finite and match the vertex count"
        );
        morphs.push(MorphTarget { positions: m_positions, normals: m_normals });
    }
    ensure!(
        uvs.len() == count
            && normals.len() == count
            && (tangents.is_empty() || tangents.len() == count)
            && (!skinned || (joints.len() == count && weights.len() == count)),
        "a primitive's attributes must have a value for each vertex"
    );
    if let Some(world) = world {
        let normal = normal_matrix(world);
        for position in &mut positions {
            let [x, y, z] = position.map(f64::from);
            *position =
                [0, 1, 2].map(|row| narrow(world[row] * x + world[4 + row] * y + world[8 + row] * z + world[12 + row]));
        }
        for vector in &mut normals {
            let [x, y, z] = vector.map(f64::from);
            *vector =
                [0, 1, 2].map(|row| narrow(normal[row * 3] * x + normal[row * 3 + 1] * y + normal[row * 3 + 2] * z));
        }
        for target in &mut morphs {
            for delta in &mut target.positions {
                let [x, y, z] = delta.map(f64::from);
                *delta = [0, 1, 2].map(|row| narrow(world[row] * x + world[4 + row] * y + world[8 + row] * z));
            }
            for delta in &mut target.normals {
                let [x, y, z] = delta.map(f64::from);
                *delta = [0, 1, 2]
                    .map(|row| narrow(normal[row * 3] * x + normal[row * 3 + 1] * y + normal[row * 3 + 2] * z));
            }
        }
        // A mirroring placement turns triangles inside out: turn them back.
        let mirrored = determinant(world) < 0.0;
        if mirrored {
            for triangle in indices.as_chunks_mut::<3>().0 {
                triangle.swap(1, 2);
            }
        }
        // A tangent goes where the surface does, and a mirror turns the
        // frame it makes with the normal and the bitangent inside out.
        for tangent in &mut tangents {
            let [x, y, z] = [tangent[0], tangent[1], tangent[2]].map(f64::from);
            let moved = [0, 1, 2].map(|row| world[row] * x + world[4 + row] * y + world[8 + row] * z);
            let length = moved.iter().map(|value| value * value).sum::<f64>().sqrt();
            // One long, but for what is long enough already (so that a
            // placement that moves it alone leaves it as it was).
            let moved =
                if length > 1e-8 && (length - 1.0).abs() > 1e-6 { moved.map(|value| value / length) } else { moved };
            *tangent =
                [narrow(moved[0]), narrow(moved[1]), narrow(moved[2]), if mirrored { -tangent[3] } else { tangent[3] }];
        }
    }
    for vector in &mut normals {
        let length = vector.iter().map(|value| f64::from(*value).powi(2)).sum::<f64>().sqrt();
        *vector = if length > 1e-8 && length.is_finite() {
            vector.map(|value| narrow(f64::from(value) / length))
        } else {
            [0.0, 1.0, 0.0]
        };
    }
    let mut geometry = Geometry { positions, normals, uvs, uvs1, colors, indices, joints, weights, tangents, morphs };
    // The tangents a normal map or an anisotropic material wants, made of
    // the geometry as it stands in the mesh (placed, and wound as it draws).
    if geometry.tangents.is_empty() && generate_tangents && textured {
        let corners = tangents::corners(&geometry.positions, &geometry.normals, &geometry.uvs, &geometry.indices);
        tangents::split(&mut geometry, &corners);
    }
    Ok(geometry)
}

/// Composes `T * R * S` into a 4x4 column-major matrix in `f64`.
fn trs_matrix(t: [f32; 3], r: [f32; 4], s: [f32; 3]) -> [f64; 16] {
    let [qx, qy, qz, qw] = r.map(f64::from);
    let qlen = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt();
    let (x, y, z, w) = if qlen > 0.0 && qlen.is_finite() {
        (qx / qlen, qy / qlen, qz / qlen, qw / qlen)
    } else {
        (0.0, 0.0, 0.0, 1.0)
    };
    let [sx, sy, sz] = s.map(f64::from);
    let [tx, ty, tz] = t.map(f64::from);
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    let (wx, wy, wz) = (w * x, w * y, w * z);
    [
        (1.0 - 2.0 * (yy + zz)) * sx,
        (2.0 * (xy + wz)) * sx,
        (2.0 * (xz - wy)) * sx,
        0.0,
        (2.0 * (xy - wz)) * sy,
        (1.0 - 2.0 * (xx + zz)) * sy,
        (2.0 * (yz + wx)) * sy,
        0.0,
        (2.0 * (xz + wy)) * sz,
        (2.0 * (yz - wx)) * sz,
        (1.0 - 2.0 * (xx + yy)) * sz,
        0.0,
        tx,
        ty,
        tz,
        1.0,
    ]
}

/// Converts `EXT_mesh_gpu_instancing` TRS records into renderer-ready persistent
/// instance records (`node_world * T * R * S`, 20 floats each) and world bounds.
fn build_mesh_instances(
    positions: &[[f32; 3]],
    node_world: &[f64; 16],
    instances: &[ImportedInstance],
) -> MeshInstancesFile {
    let mut local_min = [f64::INFINITY; 3];
    let mut local_max = [f64::NEG_INFINITY; 3];
    for p in positions {
        for a in 0..3 {
            local_min[a] = local_min[a].min(f64::from(p[a]));
            local_max[a] = local_max[a].max(f64::from(p[a]));
        }
    }
    let corners = [
        [local_min[0], local_min[1], local_min[2]],
        [local_max[0], local_min[1], local_min[2]],
        [local_min[0], local_max[1], local_min[2]],
        [local_max[0], local_max[1], local_min[2]],
        [local_min[0], local_min[1], local_max[2]],
        [local_max[0], local_min[1], local_max[2]],
        [local_min[0], local_max[1], local_max[2]],
        [local_max[0], local_max[1], local_max[2]],
    ];
    let mut world_min = [f64::INFINITY; 3];
    let mut world_max = [f64::NEG_INFINITY; 3];
    let mut data = Vec::with_capacity(instances.len() * 20);
    for inst in instances {
        // Per EXT_mesh_gpu_instancing spec: world = node_world * (T * R * S).
        let local = trs_matrix(inst.translation, inst.rotation, inst.scale);
        let m = multiply(node_world, &local);
        for &v in &m {
            data.push(narrow(v));
        }
        data.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        for c in &corners {
            let x = m[0] * c[0] + m[4] * c[1] + m[8] * c[2] + m[12];
            let y = m[1] * c[0] + m[5] * c[1] + m[9] * c[2] + m[13];
            let z = m[2] * c[0] + m[6] * c[1] + m[10] * c[2] + m[14];
            world_min[0] = world_min[0].min(x);
            world_min[1] = world_min[1].min(y);
            world_min[2] = world_min[2].min(z);
            world_max[0] = world_max[0].max(x);
            world_max[1] = world_max[1].max(y);
            world_max[2] = world_max[2].max(z);
        }
    }
    MeshInstancesFile { data, min: world_min.map(narrow), max: world_max.map(narrow) }
}

/// Reads validated per-instance TRS transforms from `EXT_mesh_gpu_instancing` on `node`.
/// Enforces strict buffer-view bounds before allocation, validates matching accessor
/// counts, dimensions, strides, and decodes `FLOAT`, normalized `BYTE`, and normalized
/// `SHORT` quaternion rotations per the `EXT_mesh_gpu_instancing` specification.
fn read_gpu_instances(
    document: &gltf::Document,
    node: &gltf::Node,
    buffers: &[Vec<u8>],
) -> Result<Vec<ImportedInstance>> {
    let Some(attrs) = node
        .extension_value("EXT_mesh_gpu_instancing")
        .and_then(|ext| ext.get("attributes"))
        .and_then(Value::as_object)
    else {
        return Ok(Vec::new());
    };
    let read_vec = |key: &str, expected_dims: gltf::accessor::Dimensions, allow_norm_int: bool| -> Result<Option<(usize, Vec<f32>)>> {
        let Some(idx) = attrs.get(key).and_then(Value::as_u64).and_then(|i| usize::try_from(i).ok()) else {
            return Ok(None);
        };
        let acc = document.accessors().nth(idx).with_context(|| format!("missing instancing accessor {idx}"))?;
        ensure!(acc.dimensions() == expected_dims, "instancing {key} accessor has wrong dimensions");
        let comps = match expected_dims {
            gltf::accessor::Dimensions::Vec3 => 3usize,
            gltf::accessor::Dimensions::Vec4 => 4usize,
            _ => bail!("unsupported instancing accessor dimensions"),
        };
        let dtype = acc.data_type();
        let comp_bytes = match dtype {
            gltf::accessor::DataType::F32 => 4usize,
            gltf::accessor::DataType::I8 if allow_norm_int && acc.normalized() => 1usize,
            gltf::accessor::DataType::I16 if allow_norm_int && acc.normalized() => 2usize,
            _ => bail!("unsupported instancing {key} component type {dtype:?} (normalized={})", acc.normalized()),
        };
        let elem_bytes = comps * comp_bytes;
        let view = acc.view().context("instancing accessor lacks buffer view")?;
        let buf = buffers.get(view.buffer().index()).context("instancing buffer missing")?;
        let view_end = view
            .offset()
            .checked_add(view.length())
            .filter(|&end| end <= buf.len())
            .context("instancing buffer view out of buffer bounds")?;
        let step = view.stride().unwrap_or(elem_bytes);
        ensure!(step >= elem_bytes, "instancing buffer view stride {step} < element size {elem_bytes}");
        let base = view.offset().checked_add(acc.offset()).context("instancing accessor offset overflow")?;
        let count = acc.count();
        if count > 0 {
            let last_start = count
                .checked_sub(1)
                .and_then(|n| n.checked_mul(step))
                .and_then(|span| base.checked_add(span))
                .context("instancing accessor span overflow")?;
            let last_end = last_start.checked_add(elem_bytes).context("instancing accessor end overflow")?;
            ensure!(last_end <= view_end, "instancing {key} accessor exceeds its buffer view bounds");
        }
        let mut out = Vec::with_capacity(count.checked_mul(comps).context("instancing allocation overflow")?);
        for i in 0..count {
            let at = base + i * step;
            for c in 0..comps {
                let v = match dtype {
                    gltf::accessor::DataType::F32 => {
                        let b = &buf[at + c * 4..at + c * 4 + 4];
                        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                    }
                    gltf::accessor::DataType::I8 => {
                        (f32::from(buf[at + c].cast_signed()) / 127.0).max(-1.0)
                    }
                    gltf::accessor::DataType::I16 => {
                        let b = &buf[at + c * 2..at + c * 2 + 2];
                        (f32::from(i16::from_le_bytes([b[0], b[1]])) / 32767.0).max(-1.0)
                    }
                    _ => unreachable!(),
                };
                ensure!(v.is_finite(), "non-finite instancing {key} value");
                out.push(v);
            }
        }
        Ok(Some((count, out)))
    };
    let translations = read_vec("TRANSLATION", gltf::accessor::Dimensions::Vec3, false)?;
    let rotations = read_vec("ROTATION", gltf::accessor::Dimensions::Vec4, true)?;
    let scales = read_vec("SCALE", gltf::accessor::Dimensions::Vec3, false)?;
    let counts: Vec<usize> = [
        translations.as_ref().map(|(c, _)| *c),
        rotations.as_ref().map(|(c, _)| *c),
        scales.as_ref().map(|(c, _)| *c),
    ]
    .into_iter()
    .flatten()
    .collect();
    let Some(&count) = counts.first() else {
        return Ok(Vec::new());
    };
    ensure!(counts.iter().all(|&c| c == count), "instancing attributes have mismatched instance counts");
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let t = translations.as_ref().map_or([0.0, 0.0, 0.0], |(_, v)| [v[i * 3], v[i * 3 + 1], v[i * 3 + 2]]);
        let r = rotations
            .as_ref()
            .map_or([0.0, 0.0, 0.0, 1.0], |(_, v)| [v[i * 4], v[i * 4 + 1], v[i * 4 + 2], v[i * 4 + 3]]);
        let s = scales.as_ref().map_or([1.0, 1.0, 1.0], |(_, v)| [v[i * 3], v[i * 3 + 1], v[i * 3 + 2]]);
        out.push(ImportedInstance { translation: t, rotation: r, scale: s });
    }
    Ok(out)
}

/// Whether a primitive of `material` reads a tangent frame: a normal map, or
/// anisotropy.
fn wants_tangents(material: &gltf::Material) -> bool {
    let anisotropic = material
        .extension_value("KHR_materials_anisotropy")
        .and_then(|extension| extension.get("anisotropyStrength"))
        .and_then(Value::as_f64)
        .is_some_and(|strength| strength != 0.0);
    material.normal_texture().is_some() || anisotropic
}

fn determinant(m: &[f64; 16]) -> f64 {
    m[0] * (m[5] * m[10] - m[6] * m[9]) - m[4] * (m[1] * m[10] - m[2] * m[9]) + m[8] * (m[1] * m[6] - m[2] * m[5])
}

/// The inverse transpose of `m`'s 3 x 3, row by row (its cofactors over
/// its determinant): what turns normals.
fn normal_matrix(m: &[f64; 16]) -> [f64; 9] {
    let (a, b, c, d, e, f, g, h, i) = (m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]);
    let det = determinant(m);
    let det = if det.abs() > 1e-12 { det } else { 1.0 };
    [
        e * i - f * h,
        c * h - b * i,
        b * f - c * e,
        f * g - d * i,
        a * i - c * g,
        c * d - a * f,
        d * h - e * g,
        b * g - a * h,
        a * e - b * d,
    ]
    .map(|cofactor| cofactor / det)
}

/// Each vertex's normal: the area-weighted mean of its triangles'.
fn smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut sums = vec![[0.0f64; 3]; positions.len()];
    for triangle in indices.as_chunks::<3>().0 {
        let [a, b, c] = triangle.map(|corner| positions[corner as usize].map(f64::from));
        let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
        let cross = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        for &corner in triangle {
            for axis in 0..3 {
                sums[corner as usize][axis] += cross[axis];
            }
        }
    }
    sums.into_iter().map(|sum| sum.map(narrow)).collect()
}

/// Writes assets, each once, as their content changes.
struct Writer<'a> {
    out: &'a Path,
    name: String,
    imported: Imported,
    /// Keys given out.
    taken: HashSet<String>,
    /// Each glTF material's key and JSON, by index (None the default).
    materials: HashMap<Option<usize>, (String, Value)>,
    /// Each image's key, or None for one the renderer cannot read.
    images: HashMap<usize, Option<String>>,
}

impl Writer<'_> {
    fn note(&mut self, note: String) {
        if !self.imported.notes.contains(&note) {
            self.imported.notes.push(note);
        }
    }

    /// `stem` and `suffix`, numbered if another asset of this import took it.
    fn unique(&mut self, stem: &str, suffix: &str) -> String {
        let mut key = format!("{stem}{suffix}");
        let mut number = 2;
        while !self.taken.insert(key.clone()) {
            key = format!("{stem}_{number}{suffix}");
            number += 1;
        }
        key
    }

    /// Writes `bytes` as `key`, unless the file holds them already, in one
    /// rename so a reader never sees half of it.
    fn write(&mut self, key: &str, bytes: &[u8]) -> Result<()> {
        let path = self.out.join(key);
        if std::fs::read(&path).is_ok_and(|old| old == bytes) {
            return Ok(());
        }
        let part = self.out.join(format!("{key}.part"));
        std::fs::write(&part, bytes).with_context(|| format!("writing {}", part.display()))?;
        std::fs::rename(&part, &path).with_context(|| format!("writing {}", path.display()))?;
        self.imported.changed.push(key.to_owned());
        Ok(())
    }

    fn write_mesh(
        &mut self,
        key: &str,
        geometry: &Geometry,
        rig: Option<&Rig>,
        instances: Option<MeshInstancesFile>,
    ) -> Result<()> {
        let morphs = (!geometry.morphs.is_empty()).then(|| {
            geometry
                .morphs
                .iter()
                .map(|target| {
                    let has_normals = target.normals.iter().flatten().any(|&v| v != 0.0);
                    MorphTargetFile {
                        positions: target.positions.iter().flatten().copied().collect(),
                        normals: has_normals.then(|| target.normals.iter().flatten().copied().collect()),
                    }
                })
                .collect()
        });
        let file = MeshFile {
            positions: geometry.positions.iter().flatten().copied().collect(),
            normals: geometry.normals.iter().flatten().copied().collect(),
            uvs: geometry.uvs.iter().flatten().copied().collect(),
            uvs1: (!geometry.uvs1.is_empty()).then(|| geometry.uvs1.iter().flatten().copied().collect()),
            colors: (!geometry.colors.is_empty()).then(|| geometry.colors.iter().flatten().copied().collect()),
            indices: &geometry.indices,
            tangents: (!geometry.tangents.is_empty()).then(|| geometry.tangents.iter().flatten().copied().collect()),
            morphs,
            instances,
            joints: rig.map(|_| geometry.joints.iter().flatten().copied().collect()),
            weights: rig.map(|_| geometry.weights.iter().flatten().copied().collect()),
            rig: rig.map(Rig::json),
        };
        let json = serde_json::to_string(&file)? + "\n";
        // What the renderer would refuse is refused here.
        pen_render::pack_mesh(json.as_bytes()).with_context(|| format!("{key} as the renderer reads it"))?;
        self.write(key, json.as_bytes())
    }

    /// The key and JSON of a glTF material, written the first time.
    fn material(
        &mut self,
        document: &gltf::Document,
        material: &gltf::Material,
        buffers: &[Vec<u8>],
        options: &Options,
    ) -> Result<(String, Value)> {
        if let Some(known) = self.materials.get(&material.index()) {
            return Ok(known.clone());
        }
        let label = match (material.index(), material.name()) {
            (_, Some(name)) => sanitized(name),
            (Some(index), None) => format!("material{index}"),
            (None, None) => "default".to_owned(),
        };
        let stem = format!("{}_{label}", self.name);
        let key = self.unique(&stem, ".material.json");
        let pbr = material.pbr_metallic_roughness();
        let [red, green, blue, alpha] = pbr.base_color_factor().map(|channel| channel.clamp(0.0, 1.0));
        let blended = material.alpha_mode() == gltf::material::AlphaMode::Blend;
        // An opaque material's alpha counts for nothing; the renderer needs
        // some.
        let alpha = if blended { alpha.max(1.0 / 255.0) } else { 1.0 };
        let emission =
            material.emissive_factor().into_iter().fold(0.0f32, f32::max) * material.emissive_strength().unwrap_or(1.0);
        let gloss = 1.0 - pbr.roughness_factor().clamp(0.0, 1.0);
        let color = [red, green, blue, alpha].map(tidy);
        let parameters = [gloss, pbr.metallic_factor().clamp(0.0, 1.0), emission, 0.0].map(tidy);
        let mut json = json!({
            "shader": options.shader,
            "color": color,
            "lighting": if material.unlit() { "unlit" } else { "lit" },
            "instanceParameters": parameters,
        });
        if let Some(info) = pbr.base_color_texture() {
            if info.tex_coord() != 0 {
                self.note(format!(
                    "material {label} reads its texture through a second set of coordinates; the first is used"
                ));
            }
            if let Some(image) = self.image(&info.texture().source(), buffers)? {
                json["texture"] = image.into();
            }
            if let Some(transform) = uv_transform(info.extension_value("KHR_texture_transform")) {
                json["uvTransform"] = transform;
            }
        }
        if let Some(normal) = material.normal_texture() {
            if let Some(image) = self.image(&normal.texture().source(), buffers)? {
                json["normalTexture"] = image.into();
            }
            if json.get("uvTransform").is_none()
                && let Some(transform) = uv_transform(normal.extension_value("KHR_texture_transform"))
            {
                json["uvTransform"] = transform;
            }
        }
        if let Some(mr) = pbr.metallic_roughness_texture()
            && json.get("uvTransform").is_none()
            && let Some(transform) = uv_transform(mr.extension_value("KHR_texture_transform"))
        {
            json["uvTransform"] = transform;
        }
        if let Some(orm_key) = self.orm_texture(
            &stem,
            material.occlusion_texture().map(|t| t.texture().source()),
            pbr.metallic_roughness_texture().map(|t| t.texture().source()),
            buffers,
        )? {
            json["ormTexture"] = orm_key.into();
        }
        if let Some(emissive) = material.emissive_texture() {
            if let Some(image) = self.image(&emissive.texture().source(), buffers)? {
                json["emissiveTexture"] = image.into();
            }
            if json.get("uvTransform").is_none()
                && let Some(transform) = uv_transform(emissive.extension_value("KHR_texture_transform"))
            {
                json["uvTransform"] = transform;
            }
        }
        if let Some(surface) = self.surface(document, material, buffers)? {
            json["surface"] = surface;
        }
        match material.alpha_mode() {
            gltf::material::AlphaMode::Blend => json["blend"] = true.into(),
            gltf::material::AlphaMode::Mask => {
                json["alphaCutoff"] = tidy(material.alpha_cutoff().unwrap_or(0.5).clamp(0.0, 1.0)).into();
            }
            gltf::material::AlphaMode::Opaque => {}
        }
        let text = serde_json::to_string_pretty(&json)? + "\n";
        // What the renderer would refuse is refused here.
        pen_render::inspect_material(text.as_bytes()).with_context(|| format!("{key} as the renderer reads it"))?;
        self.write(&key, text.as_bytes())?;
        self.imported.materials.push(key.clone());
        self.materials.insert(material.index(), (key.clone(), json.clone()));
        Ok((key, json))
    }

    /// A material's `surface` block, from both the scalar factors and textures of
    /// its `KHR_materials_*` extensions.
    fn surface(
        &mut self,
        document: &gltf::Document,
        material: &gltf::Material,
        buffers: &[Vec<u8>],
    ) -> Result<Option<Value>> {
        let mut surface = serde_json::Map::new();
        let number = |extension: &Value, key: &str, least: f64, most: f64| -> Option<Value> {
            let value = extension.get(key)?.as_f64().filter(|value| value.is_finite())?;
            Some(json!(short(value.clamp(least, most))))
        };
        let channels = |extension: &Value, key: &str, most: f64| -> Option<Value> {
            let values = extension.get(key)?.as_array()?;
            let channel = |at: usize| -> Option<f64> {
                values.get(at)?.as_f64().filter(|value| value.is_finite()).map(|value| short(value.clamp(0.0, most)))
            };
            Some(json!([channel(0)?, channel(1)?, channel(2)?]))
        };
        let textures_all: Vec<gltf::Texture> = document.textures().collect();
        for (name, textures) in SURFACE_EXTENSIONS {
            let Some(extension) = material.extension_value(name).filter(|extension| extension.is_object()) else {
                continue;
            };
            for &tex_field in textures {
                if let Some(tex_info) = extension.get(tex_field)
                    && let Some(tex_idx) =
                        tex_info.get("index").and_then(Value::as_u64).and_then(|i| usize::try_from(i).ok())
                    && let Some(tex) = textures_all.get(tex_idx)
                    && let Some(image_key) = self.image(&tex.source(), buffers)?
                {
                    if tex_info.get("texCoord").and_then(Value::as_u64).is_some_and(|c| c != 0) {
                        self.note(format!("{tex_field} uses a secondary texCoord; UV0 is assumed"));
                    }
                    if tex_info.get("extensions").and_then(|e| e.get("KHR_texture_transform")).is_some() {
                        self.note(format!("{tex_field} has a per-texture KHR_texture_transform which is not applied separately"));
                    }
                    surface.insert(tex_field.to_owned(), json!(image_key));
                }
            }
            let mut set = |key: &str, value: Option<Value>| {
                if let Some(value) = value {
                    surface.insert(key.to_owned(), value);
                }
            };
            match name {
                "KHR_materials_clearcoat" => {
                    set("clearcoat", number(extension, "clearcoatFactor", 0.0, 1.0));
                    set("clearcoatRoughness", number(extension, "clearcoatRoughnessFactor", 0.0, 1.0));
                }
                "KHR_materials_sheen" => {
                    set("sheenColor", channels(extension, "sheenColorFactor", 1.0));
                    set("sheenRoughness", number(extension, "sheenRoughnessFactor", 0.0, 1.0));
                }
                "KHR_materials_anisotropy" => {
                    set("anisotropy", number(extension, "anisotropyStrength", -1.0, 1.0));
                    set("anisotropyRotation", number(extension, "anisotropyRotation", f64::MIN, f64::MAX));
                }
                "KHR_materials_transmission" => set("transmission", number(extension, "transmissionFactor", 0.0, 1.0)),
                "KHR_materials_volume" => {
                    set("thickness", number(extension, "thicknessFactor", 0.0, f64::MAX));
                    // No distance, or none above 0, is the glTF's infinity: 0.
                    set("attenuationDistance", number(extension, "attenuationDistance", 0.0, f64::MAX));
                    set("attenuationColor", channels(extension, "attenuationColor", 1.0));
                }
                "KHR_materials_ior" => set("ior", number(extension, "ior", 1.0, f64::MAX)),
                _ => {
                    set("specular", number(extension, "specularFactor", 0.0, f64::MAX));
                    set("specularColor", channels(extension, "specularColorFactor", f64::MAX));
                }
            }
        }
        if let Some(strength) = material.emissive_strength() {
            surface.insert("emissiveStrength".to_owned(), json!(short(f64::from(strength).max(0.0))));
        }
        Ok((!surface.is_empty()).then_some(Value::Object(surface)))
    }

    /// Raw encoded bytes of a glTF image from its buffer view or URI.
    fn image_bytes(&self, image: &gltf::Image, buffers: &[Vec<u8>]) -> Result<Vec<u8>> {
        match image.source() {
            gltf::image::Source::View { view, .. } => {
                let buffer = buffers.get(view.buffer().index()).context("an image names a missing buffer")?;
                Ok(buffer
                    .get(view.offset()..view.offset() + view.length())
                    .context("an image lies outside its buffer")?
                    .to_vec())
            }
            gltf::image::Source::Uri { uri, .. } => {
                uri_bytes(uri, Path::new(&self.imported.source).parent().unwrap_or(Path::new(".")))
            }
        }
    }

    /// Packs optional `occlusionTexture` (R) and `metallicRoughnessTexture` (G=roughness, B=metallic)
    /// into one linear RGBA PNG (`R=AO, G=roughness, B=metallic, A=255`).
    fn orm_texture(
        &mut self,
        stem: &str,
        occlusion: Option<gltf::Image>,
        metallic_roughness: Option<gltf::Image>,
        buffers: &[Vec<u8>],
    ) -> Result<Option<String>> {
        if occlusion.is_none() && metallic_roughness.is_none() {
            return Ok(None);
        }
        // When both point to the exact same PNG/JPEG image, reuse it directly.
        if let (Some(occ), Some(mr)) = (&occlusion, &metallic_roughness)
            && occ.index() == mr.index()
        {
            return self.image(occ, buffers);
        }
        let decode = |img: &gltf::Image| -> Result<Option<image::RgbaImage>> {
            let bytes = self.image_bytes(img, buffers)?;
            if !(bytes.starts_with(b"\x89PNG") || bytes.starts_with(&[0xFF, 0xD8])) {
                return Ok(None);
            }
            Ok(Some(image::load_from_memory(&bytes).context("decoding ORM source image")?.to_rgba8()))
        };
        let occ_img = occlusion.as_ref().map(decode).transpose()?.flatten();
        let mr_img = metallic_roughness.as_ref().map(decode).transpose()?.flatten();
        let (width, height) = match (&occ_img, &mr_img) {
            (Some(a), Some(b)) => (a.width().max(b.width()), a.height().max(b.height())),
            (Some(a), None) => (a.width(), a.height()),
            (None, Some(b)) => (b.width(), b.height()),
            (None, None) => return Ok(None),
        };
        let mut packed = image::RgbaImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let sample = |img: &image::RgbaImage| {
                    let sx = (x * img.width() / width).min(img.width() - 1);
                    let sy = (y * img.height() / height).min(img.height() - 1);
                    img.get_pixel(sx, sy).0
                };
                let ao = occ_img.as_ref().map_or(255, |img| sample(img)[0]);
                let (rough, metal) = mr_img.as_ref().map_or((255, 255), |img| {
                    let p = sample(img);
                    (p[1], p[2])
                });
                packed.put_pixel(x, y, image::Rgba([ao, rough, metal, 255]));
            }
        }
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgba8(packed)
            .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
            .context("encoding packed ORM PNG")?;
        let key = self.unique(&format!("{stem}_orm"), ".png");
        self.write(&key, &png_bytes)?;
        self.imported.images.push(key.clone());
        Ok(Some(key))
    }

    /// An image's key, written the first time; None for one that is not a
    /// PNG or JPEG, which the renderer cannot read.
    fn image(&mut self, image: &gltf::Image, buffers: &[Vec<u8>]) -> Result<Option<String>> {
        if let Some(known) = self.images.get(&image.index()) {
            return Ok(known.clone());
        }
        let bytes = self.image_bytes(image, buffers)?;
        let extension = if bytes.starts_with(b"\x89PNG") {
            Some("png")
        } else if bytes.starts_with(&[0xFF, 0xD8]) {
            Some("jpg")
        } else {
            None
        };
        let key = match extension {
            Some(extension) => {
                let label = image.name().map_or_else(|| format!("image{}", image.index()), sanitized);
                let stem = format!("{}_{label}", self.name);
                let key = self.unique(&stem, &format!(".{extension}"));
                self.write(&key, &bytes)?;
                self.imported.images.push(key.clone());
                Some(key)
            }
            None => {
                self.note(format!(
                    "image {} is neither PNG nor JPEG, which the renderer reads; left out",
                    image.index()
                ));
                None
            }
        };
        self.images.insert(image.index(), key.clone());
        Ok(key)
    }

    /// The primitive's clips baked into a vertex-animation texture, and the
    /// material that draws with it: `material`'s, with the texture's shader.
    fn vat(
        &mut self,
        part: &str,
        geometry: &Geometry,
        rig: &Rig,
        options: &VatOptions,
        material: &Value,
    ) -> Result<Option<String>> {
        if rig.clips.is_empty() {
            self.note(format!("{part} has no clips to bake"));
            return Ok(None);
        }
        let baked = vat::bake(geometry, rig, options.fps).with_context(|| format!("baking {part}'s clips"))?;
        let texture = self.unique(&format!("{part}_vat"), ".gtex");
        self.write(&texture, &baked.texture)?;
        let mut json = material.clone();
        json["shader"] = options.shader.clone().into();
        json["layerTexture"] = texture.into();
        // The shader's parameters: which clip, and how far into it.
        json["instanceParameters"] = json!([0.0, 0.0, 0.0, 0.0]);
        let key = self.unique(&format!("{part}_vat"), ".material.json");
        self.write(&key, (serde_json::to_string_pretty(&json)? + "\n").as_bytes())?;
        Ok(Some(key))
    }
}
