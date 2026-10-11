// Mesh and material assets, and the store that loads assets by kind and key:
// on demand for what a frame draws, ahead of need for the catalog (a few at a
// time, behind anything asked for), each once, keeping what it last loaded.
// Files are read and decoded by the loader's threads (loader.rs); what they
// bring is made into GPU assets on the thread that submits frames.
use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::{Result, anyhow, bail, ensure};
use pen_protocol::GameEvent;
use serde_json::Value;

use crate::loader::{Job, JobQueue, Priority};

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
    pub indices: Vec<u32>,
}

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
    /// draws that share one instanced batch (the TS's `JSON.stringify(state)`).
    pub fn state_key(&self) -> String {
        let mut state = self.clone();
        state.instance_parameters = None;
        format!("{state:?}")
    }

    /// Opaque: not an overlay, not blended and with an opaque color.
    pub fn opaque(&self) -> bool {
        !self.overlay && !self.blend && self.color[3] == 1.0
    }
}

pub(crate) fn parse_mesh(bytes: &[u8]) -> Result<Mesh> {
    let mesh = json_record(bytes)?;
    let positions = numbers(mesh.get("positions"), "positions")?;
    let normals = numbers(mesh.get("normals"), "normals")?;
    let uvs = numbers(mesh.get("uvs"), "uvs")?;
    let attributes = match mesh.get("attributes") {
        None => None,
        Some(value) => Some(numbers(Some(value), "attributes")?),
    };
    let indices = numbers(mesh.get("indices"), "indices")?;
    let count = positions.len() / 3;
    ensure!(
        positions.len() % 3 == 0
            && count != 0
            && normals.len() == positions.len()
            && uvs.len() == count * 2
            && attributes.as_ref().is_none_or(|attributes| attributes.len() == count * 2)
            && !indices.is_empty()
            && indices.len() % 3 == 0,
        "mesh requires matching position/normal/UV vertices and triangle indices"
    );
    ensure!(
        indices.iter().all(|&index| index.fract() == 0.0 && index >= 0.0 && index < count as f64),
        "mesh index is outside its vertex array"
    );
    let mut vertices = Vec::with_capacity(count * MESH_VERTEX_FLOATS);
    for index in 0..count {
        let normal = &normals[index * 3..index * 3 + 3];
        ensure!(normal.iter().map(|value| value * value).sum::<f64>().sqrt() >= 1e-8, "mesh normal must be nonzero");
        vertices.extend_from_slice(&positions[index * 3..index * 3 + 3]);
        vertices.extend_from_slice(normal);
        vertices.extend_from_slice(&uvs[index * 2..index * 2 + 2]);
        match &attributes {
            Some(attributes) => vertices.extend_from_slice(&attributes[index * 2..index * 2 + 2]),
            None => vertices.extend([0.0, 0.0]),
        }
    }
    let vertices: Vec<f32> = vertices.into_iter().map(|value| value as f32).collect();
    ensure!(vertices.iter().all(|value| value.is_finite()), "mesh coordinates exceed float32 range");
    Ok(Mesh { vertices, indices: indices.into_iter().map(|index| index as u32).collect() })
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
                parameters.len() == 4 && parameters.iter().all(|&value| (value as f32).is_finite()),
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
    entries
        .iter()
        .map(|entry| entry.as_f64().filter(|value| value.is_finite()).ok_or_else(invalid))
        .collect()
}

fn asset_key(value: Option<&Value>, field: &str) -> Result<String> {
    match value {
        Some(Value::String(key)) if !key.is_empty() => Ok(key.clone()),
        _ => bail!("material {field} must be an asset key"),
    }
}

/// Asset keys are relative paths that cannot leave the asset root.
pub(crate) fn validate_asset_key(key: &str) -> Result<()> {
    ensure!(
        !(key.is_empty() || key.starts_with('/') || key.contains('\\') || key.split('/').any(|part| part.is_empty() || part == "..")),
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
        Self { slots: Vec::new(), ids: Default::default(), queue: VecDeque::new(), ahead: 0, jobs, events: Vec::new(), arrived: 0.0 }
    }

    fn slot_for(&mut self, kind: AssetKind, key: &str) -> Result<usize> {
        if let Some(&id) = self.ids[kind as usize].get(key) {
            return Ok(id);
        }
        validate_asset_key(key)?;
        let id = self.slots.len();
        self.slots.push(Slot { kind, key: key.to_owned(), revision: 0, current: None, queued: false, loading: 0, failed: false });
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
        let changed: HashSet<&str> = keys.iter().map(String::as_str).collect();
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

    /// Ends load `revision` of slot `id`: publishes what it prepared, or
    /// reports why it failed. Returns the asset a newer revision replaced,
    /// for the caller to let go of what depends on it.
    #[must_use]
    pub fn finish(&mut self, id: usize, revision: u64, early: bool, prepared: Result<T, String>, now: f64) -> Option<T> {
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
                    self.events.push(GameEvent::AssetReady { key: slot.key.clone(), kind: slot.kind.name().to_owned(), revision });
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
        let mesh = br#"{"positions":[0,0,0,1,0,0,0,1,0],"normals":[0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,0,1],"indices":[0,1,2]}"#;
        let parsed = parse_mesh(mesh).unwrap();
        assert_eq!(parsed.vertices.len(), 3 * MESH_VERTEX_FLOATS);
        assert_eq!(parsed.indices, [0, 1, 2]);
        let outside = br#"{"positions":[0,0,0,1,0,0,0,1,0],"normals":[0,0,1,0,0,1,0,0,1],"uvs":[0,0,1,0,0,1],"indices":[0,1,3]}"#;
        assert!(parse_mesh(outside).is_err());
        assert!(parse_mesh(b"[]").is_err());

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
}
