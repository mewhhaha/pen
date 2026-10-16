//! Live-tunable constants (the checkout's README, "Tunables"): numbers a
//! designer changes in the running game without a Blot build. The checkout
//! declares them once, in `src/tunables.json` (name, default, range, and
//! optionally a step and a doc line); the game reads that file as it
//! compiles, and this reads it for the `tune=NAME=VALUE` script step, the
//! dev tools' `tuned` answer and baking, and the browser host's bake. A tunable reaches the game as a `GameEvent::Tune` whose `index`
//! is the tunable's id, a hash of its name, and what the game holds is game
//! state, in a saved section this reads back (`tuned_values`). Baking writes
//! the game's values into the file as defaults, preserving its formatting.
// The player runtime (no `devtools`) only names tunables, for scripts.
#![cfg_attr(not(feature = "devtools"), allow(dead_code))]
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

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
/// holds exactly, as the game's build of `tunables.json` gives it.
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

/// The tunables `text` declares, checked as the game's build checks them;
/// the error says what is wrong.
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
        let name =
            object.get("name").and_then(|name| name.as_str()).filter(|name| valid_name(name)).ok_or_else(|| {
                format!("tunables.json: tunable {index} needs a name of lowercase words joined by dots")
            })?;
        let fail = |problem: String| format!("tunables.json: {name}: {problem}");
        if let Some(key) =
            object.keys().find(|key| !["name", "default", "min", "max", "step", "doc"].contains(&key.as_str()))
        {
            return Err(fail(format!("unknown {key}")));
        }
        let number = |key: &str| -> Result<Option<f32>, String> {
            match object.get(key) {
                None => Ok(None),
                Some(value) => value
                    .as_f64()
                    .map(crate::input::narrow)
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

/// What a saved state holds for each tunable the host set, by id:
/// persist.blot's framing, `magic, schema, count`, then
/// each section's `key, version, width, length` and lanes.
#[cfg(any(feature = "devtools", test))]
pub use pen_runtime::tunables::tuned_values;

impl Declared {
    /// The value the game reads (Ink's `tune.value`): as set, or its
    /// default, within its range.
    pub fn live(&self, tuned: &HashMap<u32, f32>) -> f32 {
        tuned.get(&self.id).copied().unwrap_or(self.default).clamp(self.min, self.max)
    }
}

/// Changes only the selected defaults' number tokens. Borrowed raw JSON
/// values locate those tokens without rewriting any surrounding bytes.
fn bake_text(text: &str, values: &[(String, f32)]) -> Result<(String, Vec<String>)> {
    #[derive(serde::Deserialize)]
    struct File<'a> {
        #[serde(borrow)]
        tunables: Vec<Entry<'a>>,
    }
    #[derive(serde::Deserialize)]
    struct Entry<'a> {
        name: String,
        #[serde(borrow)]
        default: &'a serde_json::value::RawValue,
    }

    let declared = parse(text).map_err(anyhow::Error::msg)?;
    let mut requested = HashMap::new();
    for (name, value) in values {
        let tunable =
            declared.iter().find(|entry| entry.name == *name).with_context(|| format!("no tunable {name}"))?;
        if !value.is_finite() || !(tunable.min..=tunable.max).contains(value) {
            bail!("{name}: {value} is outside {}..{}", tunable.min, tunable.max);
        }
        if requested.insert(name.as_str(), *value).is_some() {
            bail!("{name} is given twice");
        }
    }
    let file: File<'_> = serde_json::from_str(text)?;
    let mut edits = Vec::new();
    let mut changed = Vec::new();
    for (entry, tunable) in file.tunables.iter().zip(&declared) {
        let Some(&value) = requested.get(entry.name.as_str()) else { continue };
        if value == tunable.default {
            continue;
        }
        let token = entry.default.get();
        let start = token.as_ptr() as usize - text.as_ptr() as usize;
        let to = value.to_string();
        edits.push((start, start + token.len(), to.clone()));
        changed.push(format!("{}: {} -> {to}", entry.name, tunable.default));
    }
    let mut baked = text.to_owned();
    for (start, end, to) in edits.into_iter().rev() {
        baked.replace_range(start..end, &to);
    }
    Ok((baked, changed))
}

/// Bakes in one rename so the watcher sees a complete declarations file.
pub fn bake(root: &Path, values: &[(String, f32)]) -> Result<String> {
    let path = path(root);
    let original = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let (baked, changed) = bake_text(&original, values)?;
    if changed.is_empty() {
        return Ok("nothing to bake: each value is its default already".into());
    }
    let temporary = path.with_file_name(format!(
        ".tunables.{}-{}.tmp",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()
    ));
    let result = (|| {
        std::fs::write(&temporary, baked)?;
        std::fs::rename(&temporary, &path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result.with_context(|| format!("baking {}", path.display()))?;
    Ok(changed.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_match_gdevs() {
        // The game's tests/tunables.blot asserts the same.
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
        let id = tunable_id("gnorps.haul_speed");
        let haul = id as f32;
        let state = [5678.0, 26.0, 2.0, 1.0, 1.0, 0.0, 2.0, 7.0, 8.0, 11.0, 1.0, 0.0, 4.0, haul, 3.5, 99.0, 1.0];
        let tuned = tuned_values(&state);
        assert_eq!(tuned.len(), 2);
        assert_eq!(tuned[&id], 3.5);
        let declared =
            Declared { name: "x".into(), id: 99, default: 0.5, min: 0.0, max: 0.75, step: None, doc: String::new() };
        assert_eq!(declared.live(&tuned), 0.75);
        assert_eq!(declared.live(&HashMap::new()), 0.5);
        assert!(tuned_values(&[1.0, 2.0, 3.0]).is_empty());
        // A truncated state reads as far as it goes.
        assert!(tuned_values(&state[..12]).is_empty());
        // As does one whose counts are past any state.
        assert!(tuned_values(&[5678.0, 26.0, 1e30, 11.0, 0.0, 0.0, 1e30, haul, 1.0]).is_empty());
    }

    #[test]
    fn bakes_only_number_tokens_and_rejects_without_editing() {
        let original = "{\n  \"about\": \"default: 999\",\n  \"tunables\": [\n    {\"default\": 1e0, \"name\": \"camera.zoom\", \"min\": 0, \"max\": 4},\n    {\"name\": \"speed\", \"default\": 2.00, \"min\": 0, \"max\": 8}\n  ]\n}\n";
        let values = [("camera.zoom".into(), 3.5), ("speed".into(), 2.0)];
        let (baked, changed) = bake_text(original, &values).unwrap();
        assert_eq!(baked, original.replacen("1e0", "3.5", 1));
        assert_eq!(changed, ["camera.zoom: 1 -> 3.5"]);
        assert_eq!(bake_text(&baked, &values).unwrap(), (baked, Vec::new()));
        for value in [f32::NAN, f32::INFINITY, -1.0, 5.0] {
            assert!(bake_text(original, &[("camera.zoom".into(), value)]).is_err());
        }
        assert!(bake_text(original, &[("unknown".into(), 1.0)]).is_err());
        assert!(bake_text(original, &[("speed".into(), 1.0), ("speed".into(), 2.0)]).is_err());
    }
}
