// Mesh and material assets, and the store that loads assets by kind and key:
// on demand for what a frame draws, ahead of need for the catalog (a few at a
// time, behind anything asked for), each once, keeping what it last loaded.
// Files are read and decoded by the loader's threads (loader.rs); what they
// bring is made into GPU assets on the thread that submits frames.
use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::{Context, Result, anyhow, bail, ensure};
use pen_protocol::GameEvent;
use serde::Deserialize;
use serde_json::Value;

use crate::animation::{Clip, JointJson, NodeJson, Rig, RigJson, read_rig, write_rig};
use crate::culling::Sphere;
use crate::loader::{Job, JobQueue, Priority};
use crate::shader::Variants;

mod instances;
mod packed;
pub(crate) use instances::MeshInstances;

/// A sprite shader is WGSL like a shader, compiled for the sprite pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum AssetKind {
    Mesh,
    Material,
    Shader,
    SpriteShader,
    Texture,
}

impl AssetKind {
    const ALL: [AssetKind; 5] =
        [AssetKind::Mesh, AssetKind::Material, AssetKind::Shader, AssetKind::SpriteShader, AssetKind::Texture];

    /// The kind called `name`, as [`AssetKind::name`] says them.
    pub fn named(name: &str) -> Option<AssetKind> {
        AssetKind::ALL.into_iter().find(|kind| kind.name() == name)
    }

    pub fn name(self) -> &'static str {
        match self {
            AssetKind::Mesh => "mesh",
            AssetKind::Material => "material",
            AssetKind::Shader => "shader",
            AssetKind::SpriteShader => "spriteShader",
            AssetKind::Texture => "texture",
        }
    }
}

pub(crate) struct Mesh {
    pub vertices: Vec<f32>,
    /// Its triangles, then each simpler level of detail's.
    pub indices: Vec<u32>,
    /// A sphere holding its vertices, which frames cull its draws by.
    pub bounds: Sphere,
    /// Its levels of detail, the whole mesh first: simpler triangles over
    /// the same vertices, for draws far enough away that the difference is
    /// smaller than a pixel.
    pub lods: Vec<Lod>,
    /// An explicit JSON list (even empty), or an already packed mesh.
    /// Shipping must preserve its authored detail instead of simplifying it.
    pub authored_lods: bool,
    /// A skinned mesh's joints and weights and its rig.
    pub skin: Option<Skin>,
    /// Immutable, shader-owned records for repeated geometry.
    pub instances: Option<MeshInstances>,
}

/// Up to four joints (indices into the rig's joints) per vertex, and how
/// much each moves it; the weights of a vertex add up to 1.
pub(crate) struct Skin {
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub rig: Rig,
}

/// A level of detail: its triangles' indices in the mesh's index array, and
/// how far (in the mesh's units) its surface may stray from the whole mesh's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Lod {
    pub first: u32,
    pub count: u32,
    pub error: f32,
}

/// The most levels of detail a mesh has, the whole mesh among them.
pub(crate) const MAX_LODS: usize = 4;

// position3, normal3, UV2, optional shader attributes2 (location 8).
pub(crate) const MESH_VERTEX_FLOATS: usize = 10;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Material {
    pub shader: String,
    pub texture: Option<String>,
    /// Optional normal map, sampled as linear RGB data.
    pub normal_texture: Option<String>,
    /// Optional third image the shader reads at binding 4: an impostor
    /// outline atlas, an indexed atlas's palette or a surface's detail map.
    pub layer_texture: Option<String>,
    /// Indexed color/coverage palette with octahedral two-channel normals
    /// (atlasEncoding "indexed-oct"), rather than "rgba".
    pub indexed: bool,
    /// Shader data that may vary per instance without splitting a draw batch.
    pub instance_parameters: Option<[f64; 4]>,
    pub color: [f64; 4],
    /// "lit" rather than "unlit".
    pub lit: bool,
    /// Blend shader-provided coverage even when the material color is opaque.
    pub blend: bool,
    /// Discard coverage below this value and write surviving pixels to depth.
    pub alpha_cutoff: f64,
    /// Overlay materials draw over the scene: at full strength where nothing
    /// hides them and dimmed where the scene is in front, so outlines read in
    /// depth.
    pub overlay: bool,
    /// The frame's aberration splits this material's colors on screen.
    pub aberration: bool,
    /// The frame's decals lie on this material's surfaces.
    pub decals: bool,
    /// A lit, opaque material casts a shadow unless this is false.
    pub shadow: bool,
}

impl Material {
    /// Everything but the instance parameters, which may differ between
    /// draws that share one instanced batch.
    pub fn state_key(&self) -> String {
        let mut state = self.clone();
        state.instance_parameters = None;
        format!("{state:?}")
    }

    /// Opaque: not an overlay, not blended and with an opaque color.
    pub fn opaque(&self) -> bool {
        !self.overlay && !self.blend && self.color[3] == 1.0
    }

    pub fn casts_shadow(&self) -> bool {
        self.opaque() && self.lit && self.shadow
    }

    /// Which of its shader's pipelines it draws with: opaque, alone,
    /// instanced or skinned; translucent; or as an overlay, dimmed and at
    /// full strength.
    pub fn variants(&self) -> Variants {
        if self.overlay {
            Variants::OCCLUDED | Variants::VISIBLE
        } else if self.opaque() {
            Variants::OPAQUE | Variants::INSTANCED | Variants::SKINNED | Variants::PERSISTENT
        } else {
            Variants::TRANSLUCENT
        }
    }
}

/// The binary mesh's first bytes; mesh JSON starts with `{` or whitespace.
const MESH_MAGIC: &[u8; 4] = b"PMSH";
const MESH_VERSION: u8 = 1;
// Flags: per-vertex shader attributes follow the UVs; indices are 16 bits;
// simpler levels of detail follow the indices.
const MESH_ATTRIBUTES: u8 = 1;
const MESH_SHORT_INDICES: u8 = 2;
// Flags: a skin follows the indices (and any levels of detail), each
// vertex's joints (four u16s) and weights (four f32s), then its rig
// (`animation::write_rig`); simpler levels of detail follow the indices.
const MESH_SKIN: u8 = 4;
const MESH_LODS: u8 = 8;
const MESH_INSTANCES: u8 = 16;
const MESH_HEADER: usize = 16;

/// A mesh as its JSON has it: flat arrays, three or two numbers a vertex,
/// and any simpler levels of detail over the same vertices.
#[derive(Deserialize)]
struct MeshArrays {
    positions: Vec<f32>,
    normals: Vec<f32>,
    uvs: Vec<f32>,
    #[serde(default)]
    attributes: Option<Vec<f32>>,
    indices: Vec<u32>,
    /// A skinned mesh's four joints and weights per vertex, and its rig.
    #[serde(default)]
    joints: Option<Vec<u16>>,
    #[serde(default)]
    weights: Option<Vec<f32>>,
    #[serde(default)]
    skeleton: Option<Vec<NodeJson>>,
    #[serde(default)]
    skin: Option<Vec<JointJson>>,
    #[serde(default)]
    clips: Option<Vec<Clip>>,
    #[serde(default)]
    lods: Option<Vec<LodArrays>>,
    #[serde(default)]
    instances: Option<MeshInstances>,
}

#[derive(Deserialize)]
struct LodArrays {
    error: f32,
    indices: Vec<u32>,
}

/// A mesh asset: its JSON, or the binary form `pack_mesh` makes of it.
pub(crate) fn parse_mesh(bytes: &[u8]) -> Result<Mesh> {
    if bytes.starts_with(MESH_MAGIC) {
        return unpack_mesh(bytes);
    }
    let arrays: MeshArrays = serde_json::from_slice(bytes).map_err(|error| anyhow!("mesh JSON: {error}"))?;
    let MeshArrays {
        positions,
        normals,
        uvs,
        attributes,
        indices,
        lods,
        joints,
        weights,
        skeleton,
        skin,
        clips,
        instances,
    } = arrays;
    let count = positions.len() / 3;
    ensure!(
        positions.len() % 3 == 0
            && normals.len() == positions.len()
            && uvs.len() == count * 2
            && attributes.as_ref().is_none_or(|attributes| attributes.len() == count * 2),
        "mesh requires matching position/normal/UV vertices and triangle indices"
    );
    let mut vertices = Vec::with_capacity(count * MESH_VERTEX_FLOATS);
    for index in 0..count {
        vertices.extend_from_slice(&positions[index * 3..index * 3 + 3]);
        vertices.extend_from_slice(&normals[index * 3..index * 3 + 3]);
        vertices.extend_from_slice(&uvs[index * 2..index * 2 + 2]);
        match &attributes {
            Some(attributes) => vertices.extend_from_slice(&attributes[index * 2..index * 2 + 2]),
            None => vertices.extend([0.0, 0.0]),
        }
    }
    let skin = match (joints, weights, skeleton, skin) {
        (None, None, None, None) => None,
        (Some(joints), Some(weights), Some(skeleton), Some(skin)) => {
            ensure!(
                joints.len() == count * 4 && weights.len() == count * 4,
                "a skinned mesh needs four joints and weights a vertex"
            );
            let rig = Rig::from_json(RigJson { skeleton, skin, clips: clips.unwrap_or_default() })?;
            Some(Skin { joints: joints.as_chunks::<4>().0.to_vec(), weights: weights.as_chunks::<4>().0.to_vec(), rig })
        }
        _ => bail!("a skinned mesh needs joints, weights, a skeleton and a skin"),
    };
    let authored_lods = lods.is_some();
    let lods: Vec<(f32, Vec<u32>)> = lods.unwrap_or_default().into_iter().map(|lod| (lod.error, lod.indices)).collect();
    let mut mesh = checked_mesh(vertices, indices, &lods, skin)?;
    mesh.authored_lods = authored_lods;
    mesh.set_instances(instances)?;
    Ok(mesh)
}

/// A mesh file's bounds (JSON or packed): its vertices' least and greatest
/// coordinates.
pub fn mesh_bounds(bytes: &[u8]) -> Result<([f32; 3], [f32; 3])> {
    let mesh = parse_mesh(bytes)?;
    if let Some(instances) = &mesh.instances {
        return Ok((instances.min, instances.max));
    }
    let mut bounds = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for vertex in mesh.vertices.as_chunks::<MESH_VERTEX_FLOATS>().0 {
        for (axis, &coordinate) in vertex[..3].iter().enumerate() {
            bounds.0[axis] = bounds.0[axis].min(coordinate);
            bounds.1[axis] = bounds.1[axis].max(coordinate);
        }
    }
    Ok(bounds)
}

/// The whole mesh's level of detail, then each of `simpler`'s (its error
/// and indices), their indices after the mesh's in `indices`.
fn levels(indices: &mut Vec<u32>, simpler: &[(f32, Vec<u32>)]) -> Result<Vec<Lod>> {
    let count = |indices: &[u32]| u32::try_from(indices.len()).context("mesh has more indices than a u32 counts");
    let mut lods = vec![Lod { first: 0, count: count(indices)?, error: 0.0 }];
    for (error, lod) in simpler {
        lods.push(Lod { first: count(indices)?, count: count(lod)?, error: *error });
        indices.extend_from_slice(lod);
    }
    count(indices)?;
    Ok(lods)
}

/// The mesh, if its vertices and triangles are whole: some vertices, every
/// index one of them, nonzero normals and coordinates float32 can hold; if
/// its simpler levels of detail (their errors and indices) are triangles of
/// its vertices too, at most `MAX_LODS` in all, their errors finite and
/// growing; and if a skin's joints are its rig's and each vertex's weights
/// add up to something.
pub(crate) fn checked_mesh(
    vertices: Vec<f32>,
    mut indices: Vec<u32>,
    simpler: &[(f32, Vec<u32>)],
    skin: Option<Skin>,
) -> Result<Mesh> {
    ensure!(
        !indices.is_empty() && indices.len().is_multiple_of(3),
        "mesh requires matching position/normal/UV vertices and triangle indices"
    );
    ensure!(simpler.len() < MAX_LODS, "mesh has more than {} simpler levels of detail", MAX_LODS - 1);
    let mut least = 0.0;
    for (error, lod) in simpler {
        ensure!(
            error.is_finite() && *error >= least && !lod.is_empty() && lod.len().is_multiple_of(3),
            "mesh level of detail requires triangles and a finite error no smaller than the last's"
        );
        least = *error;
    }
    let lods = levels(&mut indices, simpler)?;
    let bounds = Sphere::around(&vertices, MESH_VERTEX_FLOATS);
    let mut mesh = Mesh { vertices, indices, bounds, lods, authored_lods: !simpler.is_empty(), skin, instances: None };
    let count = mesh.vertices.len() / MESH_VERTEX_FLOATS;
    ensure!(count != 0, "mesh requires matching position/normal/UV vertices and triangle indices");
    ensure!(mesh.indices.iter().all(|&index| (index as usize) < count), "mesh index is outside its vertex array");
    ensure!(mesh.vertices.iter().all(|value| value.is_finite()), "mesh coordinates exceed float32 range");
    for vertex in mesh.vertices.as_chunks::<MESH_VERTEX_FLOATS>().0 {
        let normal = &vertex[3..6];
        let length = normal.iter().map(|&value| f64::from(value) * f64::from(value)).sum::<f64>().sqrt();
        ensure!(length >= 1e-8, "mesh normal must be nonzero");
    }
    if let Some(skin) = &mut mesh.skin {
        let joints = skin.rig.joints.len();
        ensure!(
            skin.joints.iter().flatten().all(|&joint| usize::from(joint) < joints),
            "a vertex's joint must be one of its skin's"
        );
        // Each vertex's weights, made to add up to 1 if they are not near it
        // (so that packing a mesh again changes nothing).
        for weights in &mut skin.weights {
            let total: f32 = weights.iter().sum();
            ensure!(
                weights.iter().all(|weight| weight.is_finite() && *weight >= 0.0) && total > 0.0,
                "a vertex's weights must be finite, not negative, and some of them more than 0"
            );
            if (total - 1.0).abs() > 1e-5 {
                for weight in weights.iter_mut() {
                    *weight /= total;
                }
            }
        }
    }
    Ok(mesh)
}

/// The compact binary form of a mesh asset (JSON, or already binary), which
/// the renderer reads in place of its JSON under the same key: "PMSH",
/// version 1 (or its losslessly compressed version-2 wrapper), flags (1: shader attributes, 2: 16-bit indices, 4: a skin,
/// 8: simpler levels of detail, 16: persistent instances), two zero bytes, then the vertex and index
/// counts (u32s), each vertex's position, normal, UV and, flagged,
/// attributes (f32s), and the indices; flagged, then the count of simpler
/// levels (u32), each one's index count (u32) and error (f32), and their
/// indices; flagged, a skinned mesh's each vertex's joints (u16s) and
/// weights (f32s) and its rig; or persistent instance count, min/max bounds,
/// and 20 f32s per record. All little-endian.
pub fn pack_mesh(bytes: &[u8]) -> Result<Vec<u8>> {
    pack(&parse_mesh(bytes)?)
}

pub(crate) fn pack(mesh: &Mesh) -> Result<Vec<u8>> {
    packed::encode(pack_uncompressed(mesh)?)
}

fn pack_uncompressed(mesh: &Mesh) -> Result<Vec<u8>> {
    let count = mesh.vertices.len() / MESH_VERTEX_FLOATS;
    let vertices = mesh.vertices.as_chunks::<MESH_VERTEX_FLOATS>().0;
    let attributes = vertices.iter().any(|vertex| vertex[8] != 0.0 || vertex[9] != 0.0);
    let short = count <= usize::from(u16::MAX) + 1;
    let lods = mesh.lods.len() > 1;
    let floats = if attributes { MESH_VERTEX_FLOATS } else { MESH_VERTEX_FLOATS - 2 };
    let mut packed =
        Vec::with_capacity(MESH_HEADER + count * floats * 4 + mesh.indices.len() * if short { 2 } else { 4 });
    packed.extend_from_slice(MESH_MAGIC);
    packed.extend([
        MESH_VERSION,
        if attributes { MESH_ATTRIBUTES } else { 0 }
            | if short { MESH_SHORT_INDICES } else { 0 }
            | if mesh.skin.is_some() { MESH_SKIN } else { 0 }
            | if lods { MESH_LODS } else { 0 }
            | if mesh.instances.is_some() { MESH_INSTANCES } else { 0 },
        0,
        0,
    ]);
    packed.extend(u32::try_from(count).context("mesh has more vertices than a u32 counts")?.to_le_bytes());
    packed.extend(mesh.lods[0].count.to_le_bytes());
    for vertex in vertices {
        for value in &vertex[..floats] {
            packed.extend(value.to_le_bytes());
        }
    }
    let write = |packed: &mut Vec<u8>, indices: &[u32]| -> Result<()> {
        for &index in indices {
            if short {
                packed.extend(u16::try_from(index)?.to_le_bytes());
            } else {
                packed.extend(index.to_le_bytes());
            }
        }
        Ok(())
    };
    let whole = mesh.lods[0];
    write(&mut packed, &mesh.indices[whole.first as usize..(whole.first + whole.count) as usize])?;
    if lods {
        packed.extend(u32::try_from(mesh.lods.len() - 1)?.to_le_bytes());
        for lod in &mesh.lods[1..] {
            packed.extend(lod.count.to_le_bytes());
            packed.extend(lod.error.to_le_bytes());
        }
        for lod in &mesh.lods[1..] {
            write(&mut packed, &mesh.indices[lod.first as usize..(lod.first + lod.count) as usize])?;
        }
    }
    if let Some(skin) = &mesh.skin {
        for (joints, weights) in skin.joints.iter().zip(&skin.weights) {
            packed.extend(joints.iter().flat_map(|joint| joint.to_le_bytes()));
            packed.extend(weights.iter().flat_map(|weight| weight.to_le_bytes()));
        }
        write_rig(&skin.rig, &mut packed);
    }
    if let Some(instances) = &mesh.instances {
        instances.write(&mut packed)?;
    }
    Ok(packed)
}

fn unpack_mesh(bytes: &[u8]) -> Result<Mesh> {
    let decoded = packed::decode(bytes)?;
    let bytes = decoded.as_ref();
    ensure!(bytes.len() >= MESH_HEADER, "binary mesh ends in its header");
    ensure!(bytes[4] == MESH_VERSION, "unsupported binary mesh version");
    let flags = bytes[5];
    ensure!(
        flags & !(MESH_ATTRIBUTES | MESH_SHORT_INDICES | MESH_SKIN | MESH_LODS | MESH_INSTANCES) == 0,
        "unsupported binary mesh flags"
    );
    ensure!(
        flags & (MESH_SKIN | MESH_INSTANCES) != (MESH_SKIN | MESH_INSTANCES),
        "a mesh cannot have both a skin and persistent instances"
    );
    let word = |at: usize| -> Result<usize> {
        let word = bytes.get(at..at + 4).context("binary mesh size does not match its counts")?;
        Ok(u32::from_le_bytes(word.try_into().unwrap()) as usize)
    };
    let (count, index_count) = (word(8)?, word(12)?);
    let floats = if flags & MESH_ATTRIBUTES != 0 { MESH_VERTEX_FLOATS } else { MESH_VERTEX_FLOATS - 2 };
    let index_bytes = if flags & MESH_SHORT_INDICES != 0 { 2 } else { 4 };
    let vertex_end = count.checked_mul(floats * 4).and_then(|size| size.checked_add(MESH_HEADER));
    let end = vertex_end.and_then(|end| index_count.checked_mul(index_bytes).and_then(|size| size.checked_add(end)));
    let (Some(vertex_end), Some(end)) = (vertex_end, end) else { bail!("binary mesh size does not match its counts") };
    ensure!(end <= bytes.len(), "binary mesh size does not match its counts");
    let values = bytes[MESH_HEADER..vertex_end].as_chunks::<4>().0.iter().map(|&value| f32::from_le_bytes(value));
    let vertices = if floats == MESH_VERTEX_FLOATS {
        values.collect()
    } else {
        let values: Vec<f32> = values.collect();
        let mut vertices = Vec::with_capacity(count * MESH_VERTEX_FLOATS);
        for vertex in values.chunks_exact(floats) {
            vertices.extend_from_slice(vertex);
            vertices.extend([0.0, 0.0]);
        }
        vertices
    };
    let read_indices = |from: usize, to: usize| -> Vec<u32> {
        bytes[from..to]
            .chunks_exact(index_bytes)
            .map(|index| match index {
                [a, b] => u32::from(u16::from_le_bytes([*a, *b])),
                _ => u32::from_le_bytes(index.try_into().unwrap()),
            })
            .collect()
    };
    let indices = read_indices(vertex_end, end);
    // Any simpler levels of detail, then any skin.
    let mut simpler = Vec::new();
    let mut at = end;
    if flags & MESH_LODS != 0 {
        let levels = word(end)?;
        ensure!(levels < MAX_LODS, "mesh has more than {} simpler levels of detail", MAX_LODS - 1);
        at = end + 4 + levels * 8;
        for level in 0..levels {
            let table = end + 4 + level * 8;
            let lod_count = word(table)?;
            let error =
                f32::from_le_bytes(bytes.get(table + 4..table + 8).context("binary mesh ends early")?.try_into()?);
            let to = lod_count.checked_mul(index_bytes).and_then(|size| size.checked_add(at));
            let to = to.filter(|&to| to <= bytes.len()).context("binary mesh size does not match its counts")?;
            simpler.push((error, read_indices(at, to)));
            at = to;
        }
    }
    // A skin's joints and weights, 24 bytes a vertex, then its rig.
    let skin = if flags & MESH_SKIN != 0 {
        let skin_end = count.checked_mul(24).and_then(|size| size.checked_add(at));
        let skin_end = skin_end.filter(|&skin_end| skin_end <= bytes.len()).context("binary mesh ends in its skin")?;
        let (mut joints, mut weights) = (Vec::with_capacity(count), Vec::with_capacity(count));
        for vertex in bytes[at..skin_end].as_chunks::<24>().0 {
            let half = |at: usize| u16::from_le_bytes([vertex[at], vertex[at + 1]]);
            let float = |at: usize| f32::from_le_bytes(vertex[at..at + 4].try_into().unwrap());
            joints.push([half(0), half(2), half(4), half(6)]);
            weights.push([float(8), float(12), float(16), float(20)]);
        }
        Some(Skin { joints, weights, rig: read_rig(&bytes[skin_end..])? })
    } else {
        ensure!(flags & MESH_INSTANCES != 0 || at == bytes.len(), "binary mesh size does not match its counts");
        None
    };
    let mut mesh = checked_mesh(vertices, indices, &simpler, skin)?;
    mesh.authored_lods = true;
    if flags & MESH_INSTANCES != 0 {
        mesh.set_instances(Some(MeshInstances::read(&bytes[at..])?))?;
    }
    Ok(mesh)
}

pub(crate) fn parse_material(bytes: &[u8]) -> Result<Material> {
    let material = json_record(bytes)?;
    let optional_key = |field: &str| -> Result<Option<String>> {
        match material.get(field) {
            None | Some(Value::Null) => Ok(None),
            value => asset_key(value, field).map(Some),
        }
    };
    let shader = asset_key(material.get("shader"), "shader")?;
    let texture = optional_key("texture")?;
    let normal_texture = optional_key("normalTexture")?;
    let layer_texture = optional_key("layerTexture")?;
    let color = numbers(material.get("color"), "color")?;
    let instance_parameters = match material.get("instanceParameters") {
        None | Some(Value::Null) => None,
        value => {
            let parameters = numbers(value, "instanceParameters")?;
            ensure!(
                parameters.len() == 4 && parameters.iter().all(|&value| value.abs() <= f64::from(f32::MAX)),
                "material instanceParameters must be four finite float32 values"
            );
            Some([parameters[0], parameters[1], parameters[2], parameters[3]])
        }
    };
    let indexed = match material.get("atlasEncoding") {
        None => false,
        Some(Value::String(encoding)) if encoding == "rgba" => false,
        Some(Value::String(encoding)) if encoding == "indexed-oct" => true,
        Some(_) => bail!("material atlasEncoding must be rgba or indexed-oct"),
    };
    ensure!(
        !indexed || (texture.is_some() && normal_texture.is_some() && layer_texture.is_some()),
        "indexed-oct material requires index, normal, and palette textures"
    );
    // Alpha below 1 makes a translucent material, blended over the scene.
    ensure!(
        color.len() == 4 && color.iter().all(|channel| (0.0..=1.0).contains(channel)) && color[3] != 0.0,
        "material color must be four 0..1 channels with a nonzero alpha"
    );
    let lit = match material.get("lighting") {
        Some(Value::String(lighting)) if lighting == "lit" => true,
        Some(Value::String(lighting)) if lighting == "unlit" => false,
        _ => bail!("material lighting must be lit or unlit"),
    };
    let flag = |field: &str, missing: bool| -> Result<bool> {
        match material.get(field) {
            None => Ok(missing),
            Some(Value::Bool(value)) => Ok(*value),
            Some(_) => bail!("material {field} must be a boolean"),
        }
    };
    let overlay = flag("overlay", false)?;
    let blend = flag("blend", false)?;
    let aberration = flag("aberration", false)?;
    let decals = flag("decals", false)?;
    let shadow = flag("shadow", true)?;
    let alpha_cutoff = match material.get("alphaCutoff") {
        None => 0.0,
        Some(Value::Number(number)) if number.as_f64().is_some_and(|value| (0.0..=1.0).contains(&value)) => {
            number.as_f64().unwrap()
        }
        Some(_) => bail!("material alphaCutoff must be a finite 0..1 number"),
    };
    Ok(Material {
        shader,
        texture,
        normal_texture,
        layer_texture,
        indexed,
        instance_parameters,
        color: [color[0], color[1], color[2], color[3]],
        lit,
        blend,
        alpha_cutoff,
        overlay,
        aberration,
        decals,
        shadow,
    })
}

fn json_record(bytes: &[u8]) -> Result<serde_json::Map<String, Value>> {
    let text = std::str::from_utf8(bytes)?;
    match serde_json::from_str(text)? {
        Value::Object(record) => Ok(record),
        _ => bail!("asset JSON must be an object"),
    }
}

fn numbers(value: Option<&Value>, field: &str) -> Result<Vec<f64>> {
    let invalid = || anyhow!("asset {field} must be an array of finite numbers");
    let entries = value.and_then(Value::as_array).ok_or_else(invalid)?;
    entries.iter().map(|entry| entry.as_f64().filter(|value| value.is_finite()).ok_or_else(invalid)).collect()
}

fn asset_key(value: Option<&Value>, field: &str) -> Result<String> {
    match value {
        Some(Value::String(key)) if !key.is_empty() => Ok(key.clone()),
        _ => bail!("material {field} must be an asset key"),
    }
}

/// Asset keys are relative paths that cannot leave the asset root.
pub fn validate_asset_key(key: &str) -> Result<()> {
    ensure!(
        !(key.is_empty()
            || key.starts_with('/')
            || key.contains('\\')
            || key.split('/').any(|part| part.is_empty() || part == "..")),
        "asset key must stay inside assets: {key}"
    );
    Ok(())
}

// Assets fetched ahead of need, a few at a time, so they never hold up what a
// frame asks for.
const PREFETCHING: usize = 4;

pub(crate) struct Slot<T> {
    pub kind: AssetKind,
    pub key: String,
    pub revision: u64,
    pub current: Option<T>,
    /// Waiting in the prefetch queue.
    queued: bool,
    /// Loads of it under way.
    loading: u32,
    /// Its last load failed, and it has nothing to show.
    failed: bool,
}

/// The assets by kind and key; each slot's index is its ID, which stays.
pub(crate) struct AssetStore<T> {
    pub slots: Vec<Slot<T>>,
    /// Slot IDs by kind (`AssetKind as usize`) and key.
    ids: [HashMap<String, usize>; 5],
    queue: VecDeque<usize>,
    ahead: usize,
    jobs: JobQueue,
    /// AssetReady and AssetFailed events for the game.
    pub events: Vec<GameEvent>,
    /// When (renderer seconds) an asset last arrived.
    pub arrived: f64,
}

impl<T> AssetStore<T> {
    pub fn new(jobs: JobQueue) -> Self {
        Self {
            slots: Vec::new(),
            ids: Default::default(),
            queue: VecDeque::new(),
            ahead: 0,
            jobs,
            events: Vec::new(),
            arrived: 0.0,
        }
    }

    fn slot_for(&mut self, kind: AssetKind, key: &str) -> Result<usize> {
        if let Some(&id) = self.ids[kind as usize].get(key) {
            return Ok(id);
        }
        validate_asset_key(key)?;
        let id = self.slots.len();
        self.slots.push(Slot {
            kind,
            key: key.to_owned(),
            revision: 0,
            current: None,
            queued: false,
            loading: 0,
            failed: false,
        });
        self.ids[kind as usize].insert(key.to_owned(), id);
        Ok(id)
    }

    fn load(&mut self, id: usize, early: bool) {
        let slot = &mut self.slots[id];
        slot.revision += 1;
        slot.loading += 1;
        let job = Job::Load { slot: id, kind: slot.kind, key: slot.key.clone(), revision: slot.revision, early };
        self.jobs.push(job, if early { Priority::Background } else { Priority::Urgent });
    }

    // Starts queued prefetches while fewer than `PREFETCHING` are under way.
    fn pump(&mut self) {
        while self.ahead < PREFETCHING {
            let Some(id) = self.queue.pop_front() else { break };
            if !self.slots[id].queued {
                continue;
            }
            self.slots[id].queued = false;
            self.ahead += 1;
            self.load(id, true);
        }
    }

    /// The asset's slot, loading it at once if it is not yet.
    pub fn get(&mut self, kind: AssetKind, key: &str) -> Result<usize> {
        let id = self.slot_for(kind, key)?;
        let slot = &mut self.slots[id];
        if slot.revision == 0 || slot.queued {
            slot.queued = false;
            self.load(id, false);
        }
        Ok(id)
    }

    /// The slot of the asset of `kind` and `key`, if anything asked for it.
    pub fn id(&self, kind: AssetKind, key: &str) -> Option<usize> {
        self.ids[kind as usize].get(key).copied()
    }

    pub fn current(&self, id: usize) -> Option<&T> {
        self.slots[id].current.as_ref()
    }

    pub fn current_mut(&mut self, id: usize) -> Option<&mut T> {
        self.slots[id].current.as_mut()
    }

    /// Loads the asset ahead of need, behind anything asked for with `get`.
    pub fn prefetch(&mut self, kind: AssetKind, key: &str) -> Result<()> {
        let id = self.slot_for(kind, key)?;
        let slot = &mut self.slots[id];
        if slot.revision > 0 || slot.queued {
            return Ok(());
        }
        slot.queued = true;
        self.queue.push_back(id);
        self.pump();
        Ok(())
    }

    /// Whether the asset has nothing to show yet but is on its way.
    pub fn loading(&self, id: usize) -> bool {
        let slot = &self.slots[id];
        slot.current.is_none() && !slot.failed && (slot.queued || slot.loading > 0 || slot.revision == 0)
    }

    /// Where the asset of `kind` and `key` stands, as the dev tools say it
    /// (`pen_protocol::devtools_asset_state`), and its revision.
    pub fn standing(&self, kind: AssetKind, key: &str) -> (u32, u64) {
        use pen_protocol::devtools_asset_state::{FAILED, LOADING, READY, UNUSED};
        let Some(&id) = self.ids[kind as usize].get(key) else { return (UNUSED, 0) };
        let slot = &self.slots[id];
        let state = if slot.current.is_some() {
            READY
        } else if slot.failed {
            FAILED
        } else if slot.revision == 0 && !slot.queued {
            UNUSED
        } else {
            LOADING
        };
        (state, slot.revision)
    }

    /// Loads under way (prefetches still queued aside).
    pub fn in_flight(&self) -> usize {
        self.slots.iter().map(|slot| slot.loading as usize).sum()
    }

    /// Whether the asset of `kind` and `key` is loading (false if nothing asked for it).
    pub fn loading_key(&self, kind: AssetKind, key: &str) -> bool {
        self.ids[kind as usize].get(key).is_some_and(|&id| self.loading(id))
    }

    /// Loads again every asset of the changed `keys`, of whatever kind, that
    /// was loaded or asked for; one still queued to prefetch reads the new
    /// file when its turn comes. Each keeps what it has until its new
    /// revision is ready, and keeps it if that fails.
    pub fn reload(&mut self, keys: &[String]) {
        // A KTX2 variant's file stands for the texture's key.
        let variants: Vec<String> = keys.iter().filter_map(|key| pen_texture::ktx2::variant_of(key)).collect();
        let changed: HashSet<&str> = keys.iter().chain(&variants).map(String::as_str).collect();
        for key in changed {
            for kind in AssetKind::ALL {
                let Some(&id) = self.ids[kind as usize].get(key) else { continue };
                let slot = &self.slots[id];
                if slot.revision > 0 && !slot.queued {
                    self.load(id, false);
                }
            }
        }
    }

    pub fn reload_all(&mut self) {
        for id in 0..self.slots.len() {
            let slot = &self.slots[id];
            if slot.revision > 0 && !slot.queued {
                self.load(id, false);
            }
        }
    }

    /// Ends load `revision` of slot `id`: publishes what it prepared, or
    /// reports why it failed. Returns the asset a newer revision replaced,
    /// for the caller to let go of what depends on it.
    #[must_use]
    pub fn finish(
        &mut self,
        id: usize,
        revision: u64,
        early: bool,
        prepared: Result<T, String>,
        now: f64,
    ) -> Option<T> {
        let mut previous = None;
        let slot = &mut self.slots[id];
        match prepared {
            Err(message) => {
                if revision == slot.revision {
                    slot.failed = slot.current.is_none();
                    self.events.push(GameEvent::AssetFailed {
                        key: slot.key.clone(),
                        kind: slot.kind.name().to_owned(),
                        revision,
                        message,
                    });
                }
            }
            Ok(prepared) => {
                // A load a newer one overtook is dropped.
                if revision == slot.revision {
                    previous = slot.current.replace(prepared);
                    slot.failed = false;
                    self.events.push(GameEvent::AssetReady {
                        key: slot.key.clone(),
                        kind: slot.kind.name().to_owned(),
                        revision,
                    });
                    self.arrived = now;
                }
            }
        }
        slot.loading -= 1;
        if early {
            self.ahead -= 1;
            self.pump();
        }
        previous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meshes_and_materials_reject_malformed_assets() {
        let mesh =
            br#"{"positions":[0,0,0,1,0,0,0,1,0],"normals":[0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,0,1],"indices":[0,1,2]}"#;
        let parsed = parse_mesh(mesh).unwrap();
        assert_eq!(parsed.vertices.len(), 3 * MESH_VERTEX_FLOATS);
        assert_eq!(parsed.indices, [0, 1, 2]);
        let outside =
            br#"{"positions":[0,0,0,1,0,0,0,1,0],"normals":[0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,0,1],"indices":[0,1,3]}"#;
        assert!(parse_mesh(outside).is_err());
        assert!(parse_mesh(b"[]").is_err());
        assert!(parse_mesh(br#"{"positions":[0,0,0],"normals":[0,0,1],"uvs":[0,0],"indices":[0,0,0.5]}"#).is_err());

        let material = parse_material(br#"{"shader":"scene.wgsl","color":[1,1,1,1],"lighting":"lit"}"#).unwrap();
        assert!(material.opaque() && material.shadow && material.lit);
        assert!(parse_material(br#"{"shader":"scene.wgsl","color":[1,1,1,0],"lighting":"lit"}"#).is_err());
        assert!(parse_material(br#"{"shader":"scene.wgsl","color":[1,1,1,1],"lighting":"glow"}"#).is_err());
        assert!(
            parse_material(br#"{"shader":"s.wgsl","texture":"a","color":[1,1,1,1],"lighting":"lit","atlasEncoding":"indexed-oct"}"#)
                .is_err()
        );
        assert!(validate_asset_key("a/../b").is_err());
        assert!(validate_asset_key("a//b").is_err());
        assert!(validate_asset_key("a/b.png").is_ok());
    }

    #[test]
    fn packed_meshes_read_as_their_json() {
        let plain =
            br#"{"positions":[0,0,0,1,0,0,0,1,0],"normals":[0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,0,1],"indices":[0,1,2]}"#;
        let shaded = br#"{"positions":[0,0,0,1,0,0,0,1,0],"normals":[0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,0,1],
            "attributes":[0.25,1,0,0,3,-2],"indices":[2,1,0]}"#;
        for json in [&plain[..], &shaded[..]] {
            let packed = pack_mesh(json).unwrap();
            assert!(packed.starts_with(MESH_MAGIC));
            let (from_json, from_packed) = (parse_mesh(json).unwrap(), parse_mesh(&packed).unwrap());
            assert_eq!(from_json.vertices, from_packed.vertices);
            assert_eq!(from_json.indices, from_packed.indices);
            // Packing is idempotent.
            assert_eq!(pack_mesh(&packed).unwrap(), packed);
            assert!(parse_mesh(&packed[..packed.len() - 1]).is_err());
        }
        assert_eq!(pack_mesh(plain).unwrap()[5], MESH_SHORT_INDICES);
        assert_eq!(pack_mesh(shaded).unwrap()[5], MESH_ATTRIBUTES | MESH_SHORT_INDICES);
        // An index past the vertices, and an unknown version.
        let mut packed = pack_uncompressed(&parse_mesh(plain).unwrap()).unwrap();
        let last = packed.len() - 2;
        packed[last] = 3;
        assert!(parse_mesh(&packed).is_err());
        let mut packed = pack_mesh(plain).unwrap();
        packed[4] = 255;
        assert!(parse_mesh(&packed).is_err());
    }

    /// A triangle bound to the arm rig's two joints: JSON and its binary
    /// form read alike, rig and all.
    pub(crate) fn skinned_triangle() -> String {
        let rig = serde_json::to_value(crate::animation::tests::arm().json()).unwrap();
        let mut mesh: serde_json::Map<String, Value> = serde_json::from_str(
            r#"{"positions":[0,0,0,1,0,0,0,2,0],"normals":[0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,0,1],"indices":[0,1,2],
                "joints":[0,0,0,0, 0,1,0,0, 1,0,0,0],"weights":[1,0,0,0, 1,1,0,0, 1,0,0,0]}"#,
        )
        .unwrap();
        mesh.extend(rig.as_object().unwrap().clone());
        serde_json::to_string(&mesh).unwrap()
    }

    #[test]
    fn skinned_meshes_pack_with_their_rig() {
        let json = skinned_triangle();
        let parsed = parse_mesh(json.as_bytes()).unwrap();
        let skin = parsed.skin.as_ref().unwrap();
        assert_eq!(skin.joints[1], [0, 1, 0, 0]);
        assert_eq!(skin.weights[1], [0.5, 0.5, 0.0, 0.0], "weights are made to add up to 1");
        assert_eq!(skin.rig.clips.len(), 2);
        let packed = pack_mesh(json.as_bytes()).unwrap();
        assert_eq!(packed[5] & MESH_SKIN, MESH_SKIN);
        let unpacked = parse_mesh(&packed).unwrap();
        let unpacked_skin = unpacked.skin.as_ref().unwrap();
        assert_eq!((&unpacked_skin.joints, &unpacked_skin.weights), (&skin.joints, &skin.weights));
        assert_eq!(unpacked_skin.rig, skin.rig);
        assert_eq!(pack_mesh(&packed).unwrap(), packed);
        assert!(parse_mesh(&packed[..packed.len() - 3]).is_err());

        // A joint past the skin, half a skin, and weights that add to 0.
        let mut broken: Value = serde_json::from_str(&json).unwrap();
        broken["joints"][0] = 2.into();
        assert!(parse_mesh(broken.to_string().as_bytes()).is_err());
        let mut broken: Value = serde_json::from_str(&json).unwrap();
        broken.as_object_mut().unwrap().remove("skin");
        assert!(parse_mesh(broken.to_string().as_bytes()).is_err());
        let mut broken: Value = serde_json::from_str(&json).unwrap();
        broken["weights"][0] = 0.into();
        assert!(parse_mesh(broken.to_string().as_bytes()).is_err());
    }

    #[test]
    fn a_meshs_bounds_hold_its_vertices() {
        let mesh =
            br#"{"positions":[0,-2,0,1,0,3,0,1,0],"normals":[0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,0,1],"indices":[0,1,2]}"#;
        let bounds = ([0.0, -2.0, 0.0], [1.0, 1.0, 3.0]);
        assert_eq!(mesh_bounds(mesh).unwrap(), bounds);
        assert_eq!(mesh_bounds(&pack_mesh(mesh).unwrap()).unwrap(), bounds);
        assert!(mesh_bounds(b"{}").is_err());
    }

    /// A skin follows the levels of detail in the binary form, and both read
    /// back as they were.
    #[test]
    fn skins_and_levels_of_detail_pack_together() {
        let rig = serde_json::to_value(crate::animation::tests::arm().json()).unwrap();
        let mut mesh: serde_json::Map<String, Value> = serde_json::from_str(
            r#"{"positions":[0,0,0,1,0,0,1,1,0,0,1,0],"normals":[0,0,1,0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,1,1,0,1],
                "indices":[0,1,2,0,2,3],"lods":[{"error":0.5,"indices":[0,1,2]}],
                "joints":[0,0,0,0, 0,1,0,0, 1,0,0,0, 1,0,0,0],"weights":[1,0,0,0, 1,1,0,0, 1,0,0,0, 1,0,0,0]}"#,
        )
        .unwrap();
        mesh.extend(rig.as_object().unwrap().clone());
        let json = serde_json::to_string(&mesh).unwrap();
        let parsed = parse_mesh(json.as_bytes()).unwrap();
        let packed = pack_mesh(json.as_bytes()).unwrap();
        assert_eq!(packed[5], MESH_SHORT_INDICES | MESH_SKIN | MESH_LODS);
        let unpacked = parse_mesh(&packed).unwrap();
        assert_eq!((&unpacked.indices, &unpacked.lods), (&parsed.indices, &parsed.lods));
        let (skin, unpacked_skin) = (parsed.skin.as_ref().unwrap(), unpacked.skin.as_ref().unwrap());
        assert_eq!((&unpacked_skin.joints, &unpacked_skin.weights), (&skin.joints, &skin.weights));
        assert_eq!(unpacked_skin.rig, skin.rig);
        assert_eq!(pack_mesh(&packed).unwrap(), packed);
        assert!(parse_mesh(&packed[..packed.len() - 3]).is_err());
    }

    #[test]
    fn levels_of_detail_follow_the_mesh_and_pack_with_it() {
        let quad = br#"{"positions":[0,0,0,1,0,0,1,1,0,0,1,0],"normals":[0,0,1,0,0,1,0,0,1,0,0,1],
            "uvs":[0,0,1,0,1,1,0,1],"indices":[0,1,2,0,2,3],"lods":[{"error":0.5,"indices":[0,1,2]}]}"#;
        let mesh = parse_mesh(quad).unwrap();
        assert_eq!(mesh.indices, [0, 1, 2, 0, 2, 3, 0, 1, 2]);
        assert_eq!(mesh.lods, [Lod { first: 0, count: 6, error: 0.0 }, Lod { first: 6, count: 3, error: 0.5 }]);
        assert_eq!(mesh.bounds.center, [0.5, 0.5, 0.0]);
        let packed = pack_mesh(quad).unwrap();
        assert_eq!(packed[5], MESH_SHORT_INDICES | MESH_LODS);
        let unpacked = parse_mesh(&packed).unwrap();
        assert_eq!((unpacked.indices, unpacked.lods), (mesh.indices, mesh.lods));
        assert_eq!(pack_mesh(&packed).unwrap(), packed);
        assert!(parse_mesh(&packed[..packed.len() - 2]).is_err());
        let mut longer = packed.clone();
        longer.extend([0, 0]);
        assert!(parse_mesh(&longer).is_err());
        // A level's indices must be triangles of the mesh's vertices, and
        // its error no smaller than the last's.
        let bad = |lods: &str| {
            let json = format!(
                r#"{{"positions":[0,0,0,1,0,0,1,1,0],"normals":[0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,1,1],
                "indices":[0,1,2],"lods":{lods}}}"#
            );
            parse_mesh(json.as_bytes()).is_err()
        };
        assert!(bad(r#"[{"error":0.1,"indices":[0,1,3]}]"#));
        assert!(bad(r#"[{"error":0.1,"indices":[0,1]}]"#));
        assert!(bad(r#"[{"error":0.2,"indices":[0,1,2]},{"error":0.1,"indices":[0,1,2]}]"#));
        assert!(bad(r#"[{"error":-1,"indices":[0,1,2]}]"#));
        let four = r#"{"error":1,"indices":[0,1,2]}"#;
        assert!(bad(&format!("[{four},{four},{four},{four}]")));
        assert!(!bad(&format!("[{four},{four},{four}]")));
    }

    /// How long the game's meshes take to parse (docs/render-performance.md).
    #[test]
    #[ignore = "measures dist/gdev/assets; run with --ignored --nocapture in release"]
    fn mesh_parse_time() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dist/gdev/assets");
        let mut files: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.to_string_lossy().ends_with(".mesh.json"))
            .collect();
        files.sort();
        let meshes: Vec<Vec<u8>> = files.iter().map(|path| std::fs::read(path).unwrap()).collect();
        let packed: Vec<Vec<u8>> = meshes.iter().map(|mesh| pack_mesh(mesh).unwrap()).collect();
        for (name, meshes) in [("JSON", &meshes), ("binary", &packed)] {
            let bytes: usize = meshes.iter().map(Vec::len).sum();
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let started = std::time::Instant::now();
                for mesh in meshes {
                    parse_mesh(mesh).unwrap();
                }
                best = best.min(started.elapsed().as_secs_f64());
            }
            eprintln!(
                "{} {name} meshes, {:.1} MB: parsed in {:.1} ms",
                meshes.len(),
                bytes as f64 / 1e6,
                best * 1000.0
            );
        }
    }
}
