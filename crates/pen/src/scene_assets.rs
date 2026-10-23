//! What a scene draws first, worked out when a game is packaged, so that
//! starting it loads those assets before anything else.
//!
//! The game is a pure function of its input, so its first frames from its
//! starting scene are known before it ships: `pen build` (and `pen build
//! --web`, `pen export`) runs the program for the scene's first second
//! without rendering, as `pen packets` does, and writes `scenes.json` beside
//! the catalog: for each scene file, the assets those frames draw (meshes,
//! materials, and the shaders and textures the materials name) in the order
//! they first need them, and the sounds they play.
//!
//! A run that starts from that scene reads and decodes them on threads of
//! their own from its first moment ([`warm_up`], `pen_render::Prefetch`),
//! while the program loads and the GPU opens; asks the renderer for them
//! all at once as soon as it has one (`Renderer::preload_scene`), with their
//! pipelines compiled for the sample count the scene draws with, before the
//! program has made its first frame; and has the sounds decoded first. The
//! rest of the catalog streams in once they are in. Without the file, or
//! for another scene, it loads as before.
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};
use blot_host::GuestModule;
use pen_protocol::boundary::{decode_frame, encode_frame, read_catalog};
use pen_protocol::{Catalog, Draw, Frame, RenderFrame, Viewport};
use serde::{Deserialize, Serialize};

use crate::program::{Operation, Program};
use crate::snapshot;

/// The file, in the asset directory.
pub const FILE: &str = "scenes.json";
/// A second of frames: the menu or editor a player first sees, and the
/// loading veil's time over it.
pub const FRAMES: usize = 60;

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Scenes {
    /// Each scene's, by its file's name.
    pub scenes: BTreeMap<String, SceneAssets>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct SceneAssets {
    /// How many of its first frames were looked at.
    pub frames: usize,
    /// Each asset as its kind (as the renderer names them: mesh, material,
    /// shader, spriteShader, texture) and key, in the order first needed.
    pub assets: Vec<(String, String)>,
    /// The sound clips they play, in the same order.
    pub sounds: Vec<String>,
    /// The samples per pixel its first frame asks for (0: not known).
    #[serde(default)]
    pub samples: u32,
}

/// Collects what frames need, each once, in order.
#[derive(Default)]
struct Needs<'a> {
    catalog: Option<&'a Catalog>,
    seen: HashSet<(String, String)>,
    scene: SceneAssets,
    sprites: HashSet<u32>,
}

impl Needs<'_> {
    fn add(&mut self, kind: &str, key: &str) {
        if self.seen.insert((kind.to_owned(), key.to_owned())) {
            self.scene.assets.push((kind.to_owned(), key.to_owned()));
        }
    }

    fn draws(&mut self, draws: &[Draw]) {
        let catalog = self.catalog.unwrap();
        for draw in draws {
            self.add("material", catalog.material(draw.material));
            self.add("mesh", catalog.mesh(draw.mesh));
        }
    }

    fn frame(&mut self, frame: &RenderFrame) {
        let catalog = self.catalog.unwrap();
        if self.scene.samples == 0 {
            self.scene.samples = frame.quality.msaa;
        }
        self.draws(&frame.draws);
        for solid in &frame.solids {
            for face in &solid.faces {
                self.add("material", catalog.material(face.material));
            }
        }
        for decal in &frame.decals {
            self.add("material", catalog.material(decal.material));
        }
        for field in &frame.fields {
            self.add("material", catalog.material(field.material));
        }
        for inset in &frame.insets {
            self.draws(&inset.draws);
        }
        for sprite in &frame.sprites {
            self.sprites.insert(sprite.material);
            self.add("material", catalog.material(sprite.material));
        }
        let clips = frame.sounds.iter().map(|sound| sound.clip).chain(frame.voices.iter().map(|voice| voice.clip));
        for clip in clips {
            let key = catalog.sound(clip).to_owned();
            if !self.scene.sounds.contains(&key) {
                self.scene.sounds.push(key);
            }
        }
    }

    /// Each material followed by its shader and textures.
    fn expand(mut self, assets: &Path) -> SceneAssets {
        let catalog = self.catalog.unwrap();
        let sprite_materials: HashSet<&str> = self.sprites.iter().map(|&index| catalog.material(index)).collect();
        let mut expanded = SceneAssets {
            frames: self.scene.frames,
            sounds: std::mem::take(&mut self.scene.sounds),
            samples: self.scene.samples,
            ..Default::default()
        };
        let mut seen = HashSet::new();
        let mut push = |kind: &str, key: &str| {
            if seen.insert((kind.to_owned(), key.to_owned())) {
                expanded.assets.push((kind.to_owned(), key.to_owned()));
            }
        };
        for (kind, key) in &self.scene.assets {
            push(kind, key);
            if kind != "material" {
                continue;
            }
            let Ok(bytes) = std::fs::read(assets.join(key)) else { continue };
            let Ok((shader, textures)) = pen_render::inspect_material(&bytes) else { continue };
            push(if sprite_materials.contains(key.as_str()) { "spriteShader" } else { "shader" }, &shader);
            for texture in textures {
                push("texture", &texture);
            }
        }
        expanded
    }
}

/// What the first `frames` frames of the program in `wasm` draw, starting
/// from `scene` (when it is one), with the assets in `assets`.
pub fn compute(wasm: &[u8], assets: &Path, scene: &Path, frames: usize) -> Result<SceneAssets> {
    let catalog = read_catalog(assets)?;
    let engine = blot_host::engine()?;
    let cache = crate::cache_dir().map(|directory| directory.join("wasm"));
    let module =
        GuestModule::new(&engine, wasm, cache.as_deref()).context("compiling the game to see what it draws first")?;
    let mut program = Program::start(&engine, &module)?;
    program.request(Operation::Create, &[])?;
    program.accept();
    if let Ok(saved) = snapshot::read(scene)
        && saved.schema == program.schema
    {
        program.request(Operation::LoadScene, &saved.state)?;
        program.accept();
    }
    let viewport = Viewport { width: 1920, height: 1280 };
    let mut needs = Needs { catalog: Some(&catalog), ..Needs::default() };
    for index in 0..frames {
        let frame = Frame { timestamp: index as f64 / 60.0, viewport, events: Vec::new() };
        let input = encode_frame(&frame, if index == 0 { 0.0 } else { 1.0 / 60.0 }, &[])?;
        let packet = program.request(Operation::Frame, &input)?;
        program.accept();
        let decoded = decode_frame(&packet, program.schema, &catalog).with_context(|| format!("frame {index}"))?;
        needs.frame(&decoded.render);
    }
    program.close();
    needs.scene.frames = frames;
    Ok(needs.expand(assets))
}

/// Works out what the scene draws first and writes it into `out`, the
/// shipped asset directory, saying how much of the catalog that is.
pub fn write(wasm: &[u8], assets: &Path, scene: &Path, out: &Path) -> Result<()> {
    let name = scene.file_name().context("the scene has no file name")?.to_string_lossy().into_owned();
    let found = compute(wasm, assets, scene, FRAMES)?;
    let catalog = read_catalog(assets)?;
    let count = |kind: &str| found.assets.iter().filter(|(of, _)| of == kind).count();
    eprintln!(
        "{name} draws {} of {} meshes and {} of {} materials first ({} shaders, {} textures, {} sounds), which \
         load first",
        count("mesh"),
        catalog.meshes.len(),
        count("material"),
        catalog.materials.len(),
        count("shader") + count("spriteShader"),
        count("texture"),
        found.sounds.len(),
    );
    let scenes = Scenes { scenes: BTreeMap::from([(name, found)]) };
    std::fs::write(out.join(FILE), serde_json::to_string_pretty(&scenes)? + "\n")?;
    Ok(())
}

/// The assets being read ahead, and from where.
static WARMING: Mutex<Option<(PathBuf, pen_render::Prefetch)>> = Mutex::new(None);

/// Starts reading and decoding what the scene at `scene` draws first, if
/// the assets say, for the renderer [`warmed`] gives it to.
pub fn warm_up(assets: &Path, scene: &Path) {
    if let Some(found) = read(assets, scene) {
        let prefetch = pen_render::Prefetch::start(assets.to_owned(), &found.assets);
        *WARMING.lock().unwrap() = Some((assets.to_owned(), prefetch));
    }
}

/// What [`warm_up`] began for the assets in `assets`, once.
pub fn warmed(assets: &Path) -> Option<pen_render::Prefetch> {
    let mut warming = WARMING.lock().unwrap();
    match warming.take() {
        Some((root, prefetch)) if root == assets => Some(prefetch),
        other => {
            *warming = other;
            None
        }
    }
}

/// What the scene at `scene` draws first, if the assets say.
pub fn read(assets: &Path, scene: &Path) -> Option<SceneAssets> {
    let name = scene.file_name()?.to_string_lossy();
    let bytes = std::fs::read(assets.join(FILE)).ok()?;
    let mut scenes: Scenes = serde_json::from_slice(&bytes).map_err(|error| eprintln!("{FILE}: {error}")).ok()?;
    scenes.scenes.remove(&*name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_protocol::{Inset, Sprite};

    #[test]
    fn frames_name_their_assets_once_in_order() {
        let root = crate::store::scratch("scene-assets");
        let material = |shader: &str, texture: &str| {
            format!(r#"{{"shader": "{shader}", "texture": "{texture}", "color": [1,1,1,1], "lighting": "lit"}}"#)
        };
        std::fs::write(root.join("rock.material.json"), material("scene.wgsl", "rock.png")).unwrap();
        std::fs::write(root.join("font.material.json"), material("text.wgsl", "font.gtex")).unwrap();
        std::fs::write(root.join("grass.material.json"), material("scene.wgsl", "grass.png")).unwrap();
        let catalog = Catalog {
            meshes: vec!["rock.mesh.json".into(), "tree.mesh.json".into()],
            materials: vec!["rock.material.json".into(), "font.material.json".into(), "grass.material.json".into()],
            sounds: vec!["music.ogg".into()],
        };
        let draw = |mesh: u32, material: u32| Draw { mesh, material, ..Draw::default() };
        let mut first = RenderFrame { draws: vec![draw(0, 0), draw(0, 0)], ..RenderFrame::default() };
        first.sprites.push(Sprite { material: 1, ..Sprite::default() });
        let mut second = RenderFrame {
            insets: vec![Inset { draws: vec![draw(1, 2)], ..Inset::default() }],
            ..RenderFrame::default()
        };
        second.voices.push(pen_protocol::Voice { clip: 0, ..pen_protocol::Voice::default() });
        let mut needs = Needs { catalog: Some(&catalog), ..Needs::default() };
        needs.frame(&first);
        needs.frame(&second);
        let found = needs.expand(&root);
        let assets: Vec<(&str, &str)> = found.assets.iter().map(|(kind, key)| (kind.as_str(), key.as_str())).collect();
        assert_eq!(
            assets,
            [
                ("material", "rock.material.json"),
                ("shader", "scene.wgsl"),
                ("texture", "rock.png"),
                ("mesh", "rock.mesh.json"),
                ("material", "font.material.json"),
                ("spriteShader", "text.wgsl"),
                ("texture", "font.gtex"),
                ("material", "grass.material.json"),
                ("texture", "grass.png"),
                ("mesh", "tree.mesh.json"),
            ]
        );
        assert_eq!(found.sounds, ["music.ogg"]);
        std::fs::remove_dir_all(root).unwrap();
    }
}
