//! Asset lints, which `pen check` runs over `src/assets` after compiling the
//! game:
//!
//! - **format** (an error): a file the runtime would refuse: an image,
//!   mesh, material or shader that does not read (shaders are parsed and
//!   validated as the GPU's compiler would), a sound pen cannot play (WAV
//!   other than PCM, Ogg other than Opus), JSON that does not parse; or a
//!   file of a kind nothing loads (a warning).
//! - **missing** (an error): a key the catalog or a material names that has
//!   no file.
//! - **unused**: a file nothing names, neither the catalog, a material nor
//!   the program, which `pen build` leaves out.
//! - **oversized**: a texture wider or taller than `maxTextureSize` (4096),
//!   or any file larger than `maxFileSize` (16 MB); past 8192 a side, which
//!   WebGPU need not load, an error.
//! - **mips**: a texture larger than 256 a side, drawn on meshes, that has
//!   one level and gets no others, so it shimmers in the distance. Images get
//!   theirs on the GPU, and a packed texture's `exact` and `raw` kinds are
//!   single on purpose (indices, palettes and data are read, not averaged);
//!   a block-compressed texture (KTX2) must carry its own.
//!
//! A checkout's `pen.json` tunes them under `assets.lints`: the two limits,
//! `allow` (each lint's globs of asset keys it lets be) and `deny` (lints
//! whose warnings fail the check too).
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use crate::pipeline::matches;
use crate::project::{asset_keys, used_assets};

/// What `pen.json`'s `assets.lints` may say.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Settings {
    #[serde(default = "default_texture_size")]
    pub max_texture_size: u32,
    #[serde(default = "default_file_size")]
    pub max_file_size: u64,
    #[serde(default)]
    pub allow: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub deny: Vec<String>,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            max_texture_size: default_texture_size(),
            max_file_size: default_file_size(),
            allow: BTreeMap::new(),
            deny: Vec::new(),
        }
    }
}

fn default_texture_size() -> u32 {
    4096
}

fn default_file_size() -> u64 {
    16_000_000
}

const LINTS: [&str; 5] = ["format", "missing", "unused", "oversized", "mips"];
/// The largest texture side WebGPU promises every adapter loads.
const WEB_TEXTURE_SIZE: u32 = 8192;
/// Single-level textures up to this size are not worth a warning.
const MIPLESS_SIZE: u32 = 256;

impl Settings {
    /// The checkout's, from its `pen.json`.
    pub fn of_checkout(root: &Path) -> Result<Settings> {
        let file = root.join("pen.json");
        let text = std::fs::read_to_string(&file).with_context(|| format!("reading {}", file.display()))?;
        let config: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("reading {}", file.display()))?;
        let Some(lints) = config.get("assets").and_then(|assets| assets.get("lints")) else {
            return Ok(Settings::default());
        };
        let settings: Settings =
            serde_json::from_value(lints.clone()).with_context(|| format!("{}: assets.lints", file.display()))?;
        for lint in settings.allow.keys().chain(&settings.deny) {
            ensure!(
                LINTS.contains(&lint.as_str()),
                "{}: assets.lints has no lint {lint} ({})",
                file.display(),
                LINTS.join(", ")
            );
        }
        Ok(settings)
    }
}

#[derive(Debug, PartialEq)]
pub struct Finding {
    pub lint: &'static str,
    pub key: String,
    pub message: String,
    pub error: bool,
}

/// The assets in `assets` and what is wrong with them, for the game whose
/// program is `wasm`.
pub fn lint(assets: &Path, wasm: &[u8], settings: &Settings) -> Result<Vec<Finding>> {
    let keys = asset_keys(assets)?;
    let present: BTreeSet<&str> = keys.iter().map(String::as_str).collect();
    let used = used_assets(assets, &keys, wasm);
    let mut findings = Vec::new();
    let mut find = |lint: &'static str, key: &str, message: String, error: bool| {
        if settings.allow.get(lint).is_some_and(|globs| globs.iter().any(|glob| matches(glob, key))) {
            return;
        }
        let error = error || settings.deny.iter().any(|denied| denied == lint);
        findings.push(Finding { lint, key: key.to_owned(), message, error });
    };

    // What the catalog and the materials name, and which textures meshes draw.
    let mut named: Vec<(String, String)> = Vec::new();
    let mut on_meshes: HashMap<String, String> = HashMap::new();
    if let Some(catalog) = read_json(assets, "catalog.json") {
        for kind in ["meshes", "materials", "sounds"] {
            for key in
                catalog.get(kind).and_then(|keys| keys.as_array()).into_iter().flatten().filter_map(|key| key.as_str())
            {
                named.push(("catalog.json".to_owned(), key.to_owned()));
            }
        }
    }
    for key in keys.iter().filter(|key| key.ends_with(".material.json")) {
        let Ok((shader, textures)) = std::fs::read(assets.join(key))
            .map_err(anyhow::Error::from)
            .and_then(|bytes| pen_render::inspect_material(&bytes))
        else {
            continue;
        };
        named.push((key.clone(), shader.clone()));
        let sprites = std::fs::read_to_string(assets.join(&shader))
            .ok()
            .and_then(|code| pen_render::shader_draws_sprites(&code))
            .unwrap_or(false);
        for texture in textures {
            named.push((key.clone(), texture.clone()));
            if !sprites {
                on_meshes.entry(texture).or_insert_with(|| key.clone());
            }
        }
    }
    for (by, key) in &named {
        if !present.contains(key.as_str()) {
            find("missing", key, format!("{by} names it, and there is no such file"), true);
        }
    }

    for key in &keys {
        let bytes = std::fs::read(assets.join(key)).with_context(|| format!("reading {key}"))?;
        let megabytes = bytes.len() as f64 / 1e6;
        if !used.contains(key) {
            find(
                "unused",
                key,
                format!("nothing names it (the catalog, a material or the program): {megabytes:.1} MB"),
                false,
            );
        }
        if bytes.len() as u64 > settings.max_file_size {
            let most = settings.max_file_size as f64 / 1e6;
            find("oversized", key, format!("{megabytes:.1} MB, over {most:.1} MB (maxFileSize)"), false);
        }
        match inspect(key, &bytes) {
            Inspected::Texture(info) => {
                let side = info.width.max(info.height);
                let dimensions = format!("{} x {} {}", info.width, info.height, info.format);
                if side > WEB_TEXTURE_SIZE {
                    find(
                        "oversized",
                        key,
                        format!("{dimensions}: WebGPU need not load more than {WEB_TEXTURE_SIZE} a side"),
                        true,
                    );
                } else if side > settings.max_texture_size {
                    find(
                        "oversized",
                        key,
                        format!("{dimensions}, over {} a side (maxTextureSize)", settings.max_texture_size),
                        false,
                    );
                }
                let deliberate = ["gtex exact", "gtex raw"].contains(&info.format.as_str());
                if let Some(material) = on_meshes.get(key)
                    && info.sampled_levels == 1
                    && side > MIPLESS_SIZE
                    && !deliberate
                {
                    find(
                        "mips",
                        key,
                        format!(
                            "{dimensions} has one level, and {material} draws it on meshes: it shimmers when small"
                        ),
                        false,
                    );
                }
            }
            Inspected::Fine => {}
            Inspected::Unknown => find("format", key, "no loader reads a file of this kind".to_owned(), false),
            Inspected::Refused(problem) => find("format", key, problem, true),
        }
    }
    Ok(findings)
}

enum Inspected {
    Texture(pen_render::TextureInfo),
    Fine,
    Unknown,
    Refused(String),
}

/// Reads an asset as what its name says it is.
fn inspect(key: &str, bytes: &[u8]) -> Inspected {
    let refused = |error: anyhow::Error| Inspected::Refused(format!("{error:#}"));
    let fine = |result: Result<()>| result.map_or_else(refused, |()| Inspected::Fine);
    let lower = key.to_ascii_lowercase();
    if [".png", ".jpg", ".jpeg", ".gtex", ".ktx2"].iter().any(|extension| lower.ends_with(extension)) {
        return pen_render::inspect_texture(bytes).map_or_else(refused, Inspected::Texture);
    }
    if lower.ends_with(".mesh.json") {
        return fine(pen_render::inspect_mesh(bytes).map(drop));
    }
    if lower.ends_with(".material.json") {
        return fine(pen_render::inspect_material(bytes).map(drop));
    }
    if lower.ends_with(".json") {
        return fine(serde_json::from_slice::<serde_json::Value>(bytes).map(drop).map_err(Into::into));
    }
    if lower.ends_with(".wgsl") {
        return fine(validate_wgsl(bytes));
    }
    if lower.ends_with(".wav") {
        return fine(wav(bytes));
    }
    if lower.ends_with(".ogg") {
        return fine(ogg(bytes));
    }
    Inspected::Unknown
}

/// Parses and validates WGSL as wgpu does before compiling it.
fn validate_wgsl(bytes: &[u8]) -> Result<()> {
    let code = std::str::from_utf8(bytes)?;
    let module = naga::front::wgsl::parse_str(code).map_err(|error| anyhow::anyhow!(error.emit_to_string(code)))?;
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .map_err(|error| anyhow::anyhow!(error.emit_to_string(code)))?;
    Ok(())
}

/// A WAV file pen plays: PCM or float samples, one or two channels.
fn wav(bytes: &[u8]) -> Result<()> {
    ensure!(bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE", "not a WAV file");
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let length = usize::try_from(u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()))?;
        if &bytes[at..at + 4] == b"fmt " {
            ensure!(at + 8 + 16 <= bytes.len(), "WAV format chunk ends early");
            let chunk = &bytes[at + 8..];
            let format = u16::from_le_bytes([chunk[0], chunk[1]]);
            let channels = u16::from_le_bytes([chunk[2], chunk[3]]);
            ensure!(matches!(format, 1 | 3 | 0xfffe), "WAV of format {format}: pen plays PCM and float samples");
            ensure!(matches!(channels, 1 | 2), "WAV of {channels} channels: pen plays mono and stereo");
            return Ok(());
        }
        at += 8 + length + (length & 1);
    }
    anyhow::bail!("WAV has no format chunk")
}

/// An Ogg file pen plays: Opus.
fn ogg(bytes: &[u8]) -> Result<()> {
    ensure!(bytes.starts_with(b"OggS"), "not an Ogg file");
    ensure!(bytes.len() >= 27, "Ogg page ends early");
    let segments = usize::from(bytes[26]);
    let start = 27 + segments;
    ensure!(bytes.len() >= start + 8, "Ogg page ends early");
    let packet = &bytes[start..];
    if packet.starts_with(b"\x01vorbis") {
        anyhow::bail!("Ogg Vorbis: pen plays Ogg Opus");
    }
    ensure!(packet.starts_with(b"OpusHead"), "Ogg of an unknown codec: pen plays Ogg Opus");
    Ok(())
}

fn read_json(assets: &Path, key: &str) -> Option<serde_json::Value> {
    serde_json::from_slice(&std::fs::read(assets.join(key)).ok()?).ok()
}

/// Lints the checkout's assets for the program in `wasm`, saying what it
/// found; fails when anything is an error.
pub fn check(root: &Path, assets: &Path, wasm: &[u8]) -> Result<()> {
    let settings = Settings::of_checkout(root)?;
    let started = std::time::Instant::now();
    let findings = lint(assets, wasm, &settings)?;
    let milliseconds = started.elapsed().as_secs_f64() * 1000.0;
    for lint in LINTS {
        for finding in findings.iter().filter(|finding| finding.lint == lint) {
            let level = if finding.error { "error" } else { "warning" };
            eprintln!("assets: {level}: {lint}: {}: {}", finding.key, finding.message);
        }
    }
    let errors = findings.iter().filter(|finding| finding.error).count();
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for finding in &findings {
        *counts.entry(finding.lint).or_default() += 1;
    }
    let by_lint: Vec<String> = counts.iter().map(|(lint, count)| format!("{count} {lint}")).collect();
    let files = asset_keys(assets)?.len();
    if findings.is_empty() {
        eprintln!("assets: {files} files linted in {milliseconds:.0} ms, nothing to say");
    } else {
        eprintln!(
            "assets: {files} files linted in {milliseconds:.0} ms: {} ({errors} errors); pen.json's assets.lints \
             allows or denies them",
            by_lint.join(", ")
        );
    }
    ensure!(errors == 0, "{errors} asset problems would stop the game loading them");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        image::ImageEncoder::write_image(
            image::codecs::png::PngEncoder::new(&mut bytes),
            &vec![255; (width * height * 4) as usize],
            width,
            height,
            image::ExtendedColorType::Rgba8,
        )
        .unwrap();
        bytes
    }

    /// How long linting the game's assets takes (docs/assets.md).
    #[test]
    #[ignore = "lints dist/gdev; run with --ignored --nocapture in release"]
    fn lint_time() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dist/gdev");
        let wasm = std::fs::read(root.join("game.wasm")).unwrap();
        let started = std::time::Instant::now();
        let findings = lint(&root.join("assets"), &wasm, &Settings::default()).unwrap();
        eprintln!("{} findings in {:.0} ms", findings.len(), started.elapsed().as_secs_f64() * 1000.0);
    }

    #[test]
    fn lints_find_what_the_runtime_would_refuse_or_waste() {
        let root = crate::store::scratch("asset-lint");
        let assets = root.join("assets");
        std::fs::create_dir_all(&assets).unwrap();
        let catalog = r#"{"meshes": ["cube.mesh.json"], "materials": ["rock.material.json", "hud.material.json"], "sounds": ["gone.wav", "song.ogg"]}"#;
        std::fs::write(assets.join("catalog.json"), catalog).unwrap();
        std::fs::write(
            assets.join("cube.mesh.json"),
            r#"{"positions": [0,0,0], "normals": [0,0,1], "uvs": [0,0], "indices": [0,0]}"#,
        )
        .unwrap();
        std::fs::write(
            assets.join("rock.material.json"),
            r#"{"shader": "scene.wgsl", "texture": "big.gtex", "normalTexture": "rock.png", "layerTexture": "detail.ktx2", "color": [1,1,1,1], "lighting": "lit"}"#,
        )
        .unwrap();
        std::fs::write(
            assets.join("hud.material.json"),
            r#"{"shader": "hud.wgsl", "texture": "rock.png", "color": [1,1,1,1], "lighting": "unlit"}"#,
        )
        .unwrap();
        std::fs::write(
            assets.join("scene.wgsl"),
            "@vertex fn vertex(@location(0) p: vec3f) -> @builtin(position) vec4f { return vec4f(p, 1.0); }",
        )
        .unwrap();
        std::fs::write(assets.join("hud.wgsl"), "@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32(i)); }\nfn broken() -> f32 { return true; }").unwrap();
        std::fs::write(assets.join("rock.png"), png(8, 8)).unwrap();
        // A 512 x 512 exact packed texture: one level, every level inflates.
        let mut gtex = b"GTEX\x02\x01\x00\x01".to_vec();
        gtex.extend(512u32.to_le_bytes());
        gtex.extend(512u32.to_le_bytes());
        let mut deflated = Vec::new();
        {
            use std::io::Write;
            let mut encoder = flate2::write::ZlibEncoder::new(&mut deflated, flate2::Compression::fast());
            encoder.write_all(&vec![0; 512 * 512]).unwrap();
            encoder.finish().unwrap();
        }
        gtex.extend(u32::try_from(deflated.len()).unwrap().to_le_bytes());
        gtex.extend(&deflated);
        std::fs::write(assets.join("big.gtex"), gtex).unwrap();
        // A block-compressed 512 x 512 texture of one level, drawn on meshes.
        let mut ktx = b"\xabKTX 20\xbb\r\n\x1a\n".to_vec();
        ktx.resize(80, 0);
        ktx[12..16].copy_from_slice(&145u32.to_le_bytes());
        ktx[20..24].copy_from_slice(&512u32.to_le_bytes());
        ktx[24..28].copy_from_slice(&512u32.to_le_bytes());
        ktx[40..44].copy_from_slice(&1u32.to_le_bytes());
        std::fs::write(assets.join("detail.ktx2"), ktx).unwrap();
        let mut ogg = b"OggS".to_vec();
        ogg.resize(26, 0);
        ogg.push(1);
        ogg.push(30);
        ogg.extend(b"\x01vorbis");
        ogg.resize(64, 0);
        std::fs::write(assets.join("song.ogg"), ogg).unwrap();
        std::fs::write(assets.join("poster.png"), png(4, 4)).unwrap();
        std::fs::write(assets.join("notes.txt"), "hi").unwrap();

        let settings = Settings { max_texture_size: 256, ..Settings::default() };
        let findings = lint(&assets, b"", &settings).unwrap();
        let said: Vec<(&str, &str, bool)> =
            findings.iter().map(|finding| (finding.lint, finding.key.as_str(), finding.error)).collect();
        assert!(said.contains(&("missing", "gone.wav", true)), "{findings:#?}");
        assert!(said.contains(&("format", "cube.mesh.json", true)), "{findings:#?}");
        assert!(said.contains(&("format", "hud.wgsl", true)), "{findings:#?}");
        assert!(said.contains(&("format", "song.ogg", true)), "{findings:#?}");
        assert!(said.contains(&("format", "notes.txt", false)), "{findings:#?}");
        assert!(said.contains(&("unused", "poster.png", false)), "{findings:#?}");
        assert!(said.contains(&("oversized", "big.gtex", false)), "{findings:#?}");
        assert!(said.contains(&("mips", "detail.ktx2", false)), "{findings:#?}");
        // An exact packed texture is single on purpose.
        assert!(!said.contains(&("mips", "big.gtex", false)), "{findings:#?}");
        // The scene shader is fine, and the PNG gets its mips on the GPU.
        assert!(
            !said.iter().any(|(_, key, _)| ["scene.wgsl", "rock.png", "catalog.json"].contains(key)),
            "{findings:#?}"
        );
        let message = &findings.iter().find(|finding| finding.lint == "mips").unwrap().message;
        assert!(message.contains("rock.material.json"), "{message}");

        // Allowed by glob, or denied into errors.
        let settings: Settings =
            serde_json::from_value(serde_json::json!({ "allow": { "unused": ["poster*"] }, "deny": ["mips"] }))
                .unwrap();
        let findings = lint(&assets, b"", &settings).unwrap();
        assert!(!findings.iter().any(|finding| finding.lint == "unused"));
        assert!(findings.iter().any(|finding| finding.lint == "mips" && finding.error));
        std::fs::remove_dir_all(root).unwrap();
    }
}
