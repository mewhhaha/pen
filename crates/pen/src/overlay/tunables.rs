//! Live-tunable constants (gdev's README, "Tunables"): numbers a designer
//! changes in the running game without a Blot build. The checkout declares
//! them once, in `src/tunables.json` (name, default, range, and optionally a
//! step and a doc line); the game reads that file as it compiles, and this
//! reads it for the overlay's Tunables panel, its `/tune` command and the
//! `tune=NAME=VALUE` script step, as gdev's `tunables.ts` does for the
//! browser. A tunable reaches the game as a `GameEvent::Tune` whose `index`
//! is the tunable's id, a hash of its name, and what the game holds is game
//! state, in a saved section this reads back (`tuned_values`). Baking writes
//! the game's values into the file as defaults, through the checkout's
//! `scripts/bake_tunables.ts`.
// The player runtime (no `devtools`) only names tunables, for scripts.
#![cfg_attr(not(feature = "devtools"), allow(dead_code))]
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// The saved section holding what the host set tunables to (gdev's
/// `packages/tune.blot` codec key): `id, value` pairs.
const TUNED_SECTION: f32 = 11.0;
/// A saved state's first lane (gdev's `packages/persist.blot`).
const STATE_MAGIC: f32 = 5678.0;

/// A tunable as `src/tunables.json` declares it.
#[derive(Clone, Debug, PartialEq)]
pub struct Declared {
    pub name: String,
    /// `tunable_id(name)`, which Tune events carry.
    pub id: u32,
    pub default: f32,
    pub min: f32,
    pub max: f32,
    pub step: Option<f32>,
    pub doc: String,
}

/// A tunable's id: FNV-1a over its name, folded to the 24 bits a float lane
/// holds exactly (gdev's `tunableId`).
pub fn tunable_id(name: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in name.bytes() {
        hash = (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193);
    }
    ((hash >> 24) ^ hash) & 0x00ff_ffff
}

/// The checkout's declarations file.
pub fn path(root: &Path) -> PathBuf {
    root.join("src").join("tunables.json")
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.split('.').all(|word| {
            word.chars().next().is_some_and(|first| first.is_ascii_lowercase())
                && word.chars().all(|char| char.is_ascii_lowercase() || char.is_ascii_digit() || char == '_')
        })
}

/// The tunables `text` declares, checked as gdev's `parseTunables` checks
/// them; the error says what is wrong.
pub fn parse(text: &str) -> Result<Vec<Declared>, String> {
    let file: serde_json::Value = serde_json::from_str(text).map_err(|error| format!("tunables.json: {error}"))?;
    let Some(list) = file.get("tunables").and_then(|list| list.as_array()) else {
        return Err("tunables.json: expected { \"tunables\": [...] }".into());
    };
    let mut declared: Vec<Declared> = Vec::new();
    for (index, entry) in list.iter().enumerate() {
        let Some(object) = entry.as_object() else {
            return Err(format!("tunables.json: tunable {index} is not an object"));
        };
        let name = object.get("name").and_then(|name| name.as_str()).filter(|name| valid_name(name)).ok_or_else(|| {
            format!("tunables.json: tunable {index} needs a name of lowercase words joined by dots")
        })?;
        let fail = |problem: String| format!("tunables.json: {name}: {problem}");
        if let Some(key) = object.keys().find(|key| !["name", "default", "min", "max", "step", "doc"].contains(&key.as_str())) {
            return Err(fail(format!("unknown {key}")));
        }
        let number = |key: &str| -> Result<Option<f32>, String> {
            match object.get(key) {
                None => Ok(None),
                Some(value) => value
                    .as_f64()
                    .map(|value| value as f32)
                    .filter(|value| value.is_finite())
                    .map(Some)
                    .ok_or_else(|| fail(format!("{key} must be a finite number"))),
            }
        };
        let required = |key: &str| number(key)?.ok_or_else(|| fail(format!("{key} must be a finite number")));
        let (default, min, max, step) = (required("default")?, required("min")?, required("max")?, number("step")?);
        if min >= max {
            return Err(fail("min must be below max".into()));
        }
        if !(min..=max).contains(&default) {
            return Err(fail(format!("its default {default} is outside {min}..{max}")));
        }
        if step.is_some_and(|step| step <= 0.0) {
            return Err(fail("step must be positive".into()));
        }
        let doc = match object.get("doc") {
            None => String::new(),
            Some(doc) => doc.as_str().ok_or_else(|| fail("doc must be text".into()))?.to_owned(),
        };
        let id = tunable_id(name);
        if let Some(other) = declared.iter().find(|other| other.name.replace('.', "_") == name.replace('.', "_")) {
            return Err(fail(format!("reads as {}, as {} does; rename one", name.replace('.', "_"), other.name)));
        }
        if let Some(other) = declared.iter().find(|other| other.id == id) {
            return Err(fail(format!("its id {id} is {}'s too; rename one", other.name)));
        }
        declared.push(Declared { name: name.to_owned(), id, default, min, max, step, doc });
    }
    Ok(declared)
}

/// The checkout's tunables: none when it has no file, or why they do not
/// load.
pub fn load(root: &Path) -> Result<Vec<Declared>, String> {
    match std::fs::read_to_string(path(root)) {
        Ok(text) => parse(&text),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(format!("{}: {error}", path(root).display())),
    }
}

/// What a saved state holds for each tunable the host set, by id (gdev's
/// `tunedValues`): persist.blot's framing, `magic, schema, count`, then
/// each section's `key, version, width, length` and lanes.
pub fn tuned_values(state: &[f32]) -> HashMap<u32, f32> {
    let mut tuned = HashMap::new();
    if state.len() < 3 || state[0] != STATE_MAGIC {
        return tuned;
    }
    let mut offset = 3;
    for _ in 0..state[2] as usize {
        let Some(&[key, _, _, length]) = state.get(offset..offset + 4).and_then(|header| <&[f32; 4]>::try_from(header).ok())
        else {
            break;
        };
        let start = offset + 4;
        let end = start + length as usize;
        if key == TUNED_SECTION
            && let Some(lanes) = state.get(start..end)
        {
            for &[id, value] in lanes.as_chunks::<2>().0 {
                tuned.insert(id as u32, value);
            }
        }
        offset = end;
    }
    tuned
}

impl Declared {
    /// The value the game reads (gdev's `tune.value`): as set, or its
    /// default, within its range.
    pub fn live(&self, tuned: &HashMap<u32, f32>) -> f32 {
        tuned.get(&self.id).copied().unwrap_or(self.default).clamp(self.min, self.max)
    }
}

/// Writes `values` (by name) into the checkout's `src/tunables.json` as the
/// defaults, with its `scripts/bake_tunables.ts`; returns what it says
/// changed. Blocks while Deno runs.
pub fn bake(root: &Path, values: &[(String, f32)]) -> Result<String> {
    let script = root.join("scripts").join("bake_tunables.ts");
    if !script.is_file() {
        bail!("{} is missing; this checkout cannot bake", script.display());
    }
    let output = std::process::Command::new("deno")
        .args(["run", "--allow-read", "--allow-write", "--no-lock", "--config"])
        .arg(root.join("deno.json"))
        .arg(&script)
        .args(values.iter().map(|(name, value)| format!("{name}={value}")))
        .current_dir(root)
        .output()
        .context("running deno, which bakes the tunables")?;
    let said = |bytes: &[u8]| String::from_utf8_lossy(bytes).trim().to_owned();
    if !output.status.success() {
        bail!("baking failed: {}", said(&output.stderr));
    }
    Ok(said(&output.stdout))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_match_gdevs() {
        // gdev's tunables_test.ts asserts the same.
        assert_eq!(tunable_id("camera.zoom_speed"), 11_175_412);
        assert!(tunable_id("a.much.longer.name_with_digits_123") <= 0x00ff_ffff);
    }

    #[test]
    fn parses_and_checks_declarations() {
        let text = r#"{ "about": "x", "tunables": [
            { "name": "camera.zoom_speed", "default": 0.001, "min": 0.0002, "max": 0.005, "step": 0.0001, "doc": "zoom" },
            { "name": "gnorps.haul_speed", "default": 1.6, "min": 0.2, "max": 8 }
        ] }"#;
        let declared = parse(text).unwrap();
        assert_eq!(declared.len(), 2);
        assert_eq!(declared[0].id, 11_175_412);
        assert_eq!((declared[0].step, declared[0].doc.as_str()), (Some(0.0001), "zoom"));
        assert_eq!((declared[1].default, declared[1].step), (1.6, None));
        let reject = |entry: &str, problem: &str| {
            let error = parse(&format!(r#"{{ "tunables": [{entry}] }}"#)).unwrap_err();
            assert!(error.contains(problem), "{error}");
        };
        reject(r#"{ "name": "Camera", "default": 1, "min": 0, "max": 2 }"#, "lowercase words");
        reject(r#"{ "name": "a.b", "defualt": 1, "min": 0, "max": 2 }"#, "unknown defualt");
        reject(r#"{ "name": "a.b", "default": 3, "min": 0, "max": 2 }"#, "outside");
        reject(r#"{ "name": "a.b", "default": 1, "min": 2, "max": 2 }"#, "min must be below max");
        reject(r#"{ "name": "a.b", "default": 1, "min": 0, "max": 2, "step": 0 }"#, "step must be positive");
        reject(
            r#"{ "name": "a.b", "default": 1, "min": 0, "max": 2 }, { "name": "a_b", "default": 1, "min": 0, "max": 2 }"#,
            "reads as a_b",
        );
        assert!(parse("[]").is_err());
    }

    #[test]
    fn reads_the_tuned_section_of_a_state() {
        let haul = tunable_id("gnorps.haul_speed") as f32;
        let state = [5678.0, 26.0, 2.0, 1.0, 1.0, 0.0, 2.0, 7.0, 8.0, 11.0, 1.0, 0.0, 4.0, haul, 3.5, 99.0, 1.0];
        let tuned = tuned_values(&state);
        assert_eq!(tuned.len(), 2);
        assert_eq!(tuned[&(haul as u32)], 3.5);
        let declared =
            Declared { name: "x".into(), id: 99, default: 0.5, min: 0.0, max: 0.75, step: None, doc: String::new() };
        assert_eq!(declared.live(&tuned), 0.75);
        assert_eq!(declared.live(&HashMap::new()), 0.5);
        assert!(tuned_values(&[1.0, 2.0, 3.0]).is_empty());
        // A truncated state reads as far as it goes.
        assert!(tuned_values(&state[..12]).is_empty());
    }

    #[test]
    fn reads_and_bakes_the_checkout() {
        // The sibling gdev checkout, when this repo has one and it has tunables.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../gdev");
        if !path(&root).is_file() || !root.join("scripts/bake_tunables.ts").is_file() {
            return;
        }
        let declared = load(&root).unwrap();
        assert!(!declared.is_empty());
        // Baking a copy (with the script's --file): only the default's number changes.
        let directory = std::env::temp_dir().join(format!("pen-bake-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let copy = directory.join("tunables.json");
        let original = std::fs::read_to_string(path(&root)).unwrap();
        std::fs::write(&copy, &original).unwrap();
        let first = &declared[0];
        let value = (first.min + first.max) * 0.5;
        let output = std::process::Command::new("deno")
            .args(["run", "--allow-read", "--allow-write", "--no-lock", "--config"])
            .arg(root.join("deno.json"))
            .arg(root.join("scripts/bake_tunables.ts"))
            .arg("--file")
            .arg(&copy)
            .arg(format!("{}={value}", first.name))
            .output();
        let Ok(output) = output else { return };
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let baked = parse(&std::fs::read_to_string(&copy).unwrap()).unwrap();
        assert_eq!(baked[0].default, value);
        assert_eq!(baked[1..], declared[1..]);
        let _ = std::fs::remove_dir_all(&directory);
    }
}
