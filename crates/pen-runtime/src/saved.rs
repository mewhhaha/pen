//! Typed save-state migrations a game declares (docs/saved-types.md):
//! `saved.json` beside its asset catalog names its saved types' layouts,
//! version by version, as typed fields, and the host brings a section saved
//! with an older layout to the build's when the game's own code cannot.
//!
//! A state is the save format of ink's `persist.blot` (gdev's README,
//! "Saved state"): `magic, schema, section_count`, then per saved type `key,
//! version, width, length` and its lanes. Restoring, the game says what it
//! made of each section (the restore report, `game.rs`); a section it had
//! to reset, saved with a version its `migrate` does not read, is the host's
//! to migrate when both versions are declared: each field of the build's
//! version takes the lanes of the saved field of its name (or the name it
//! `was`), a field the saved version lacks takes its default, and fields the
//! build dropped go. The game then restores the migrated state as its own.
use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow, bail, ensure};
use pen_protocol::boundary::whole;

/// What a state starts with (ink's `persist.magic`).
pub const STATE_MAGIC: f32 = 5678.0;
/// The file a game declares its saved types in, beside its catalog.
pub const SAVED_TYPES_FILE: &str = "saved.json";
/// That file for a game that declares none, as the browser host is served
/// it: the page asks for the file whatever, and a missing one would be an
/// error in its console.
pub const NO_SAVED_TYPES: &str = "{\"types\": {}}\n";

/// How a saved type's section holds its values (ink's persist layers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// One value's lanes.
    Resource,
    /// `current_count`, the current value's lanes, the previous value's.
    SnapshottedResource,
    /// A `flag, lanes...` record per entity, `width` lanes each.
    Component,
    /// The same, current records and then as many previous ones.
    SnapshottedComponent,
}

/// A field of a version's layout: its name, its lanes and, for a field a
/// migration adds, its lanes' default.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub name: String,
    pub lanes: usize,
    /// Another name an older version gave it.
    pub was: Option<String>,
    pub default: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SavedType {
    pub name: String,
    pub key: u32,
    pub kind: Kind,
    /// Each declared version's fields, in lane order.
    pub versions: BTreeMap<u32, Vec<Field>>,
}

/// A game's declared saved types, by key.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SavedTypes {
    pub types: BTreeMap<u32, SavedType>,
}

/// A number from JSON as the lane a save holds it in.
#[allow(clippy::cast_possible_truncation, reason = "a save's lanes are f32")]
fn lane(value: f64) -> f32 {
    value as f32
}

/// The lanes a field's type takes.
fn lanes_of(field: &serde_json::Value) -> Result<usize> {
    if let Some(lanes) = field.get("lanes") {
        let lanes = lanes.as_u64().filter(|lanes| (1..=4096).contains(lanes)).context("lanes runs from 1 to 4096")?;
        return Ok(usize::try_from(lanes)?);
    }
    let kind = field.get("type").and_then(serde_json::Value::as_str).context("a field needs a type or lanes")?;
    Ok(match kind {
        "f32" | "u32" | "bool" | "id" => 1,
        "vec2" => 2,
        "vec3" => 3,
        "vec4" | "quat" | "color" => 4,
        "mat4" => 16,
        other => {
            bail!("unknown field type {other} (f32, u32, bool, id, vec2, vec3, vec4, quat, color, mat4, or lanes)")
        }
    })
}

impl SavedTypes {
    /// Parses `saved.json`: `{"types": {"camera": {"key": 101, "kind":
    /// "resource", "versions": {"1": [{"name": "position", "type": "vec3"}],
    /// "2": [...]}}}}`.
    pub fn parse(text: &str) -> Result<SavedTypes> {
        let json: serde_json::Value = serde_json::from_str(text)?;
        let declared = json.get("types").and_then(serde_json::Value::as_object).context("saved types need `types`")?;
        let mut types = BTreeMap::new();
        for (name, declaration) in declared {
            let saved = SavedType::parse(name, declaration).with_context(|| format!("saved type {name}"))?;
            let key = saved.key;
            ensure!(types.insert(key, saved).is_none(), "two saved types have key {key}");
        }
        Ok(SavedTypes { types })
    }
}

impl SavedType {
    fn parse(name: &str, declaration: &serde_json::Value) -> Result<SavedType> {
        let key = declaration.get("key").and_then(serde_json::Value::as_u64).context("a key")?;
        let key =
            u32::try_from(key).ok().filter(|key| *key > 0 && *key < 1 << 24).context("a key from 1 below 2^24")?;
        let kind = match declaration.get("kind").and_then(serde_json::Value::as_str) {
            Some("resource") => Kind::Resource,
            Some("snapshotted resource") => Kind::SnapshottedResource,
            Some("component") => Kind::Component,
            Some("snapshotted component") => Kind::SnapshottedComponent,
            _ => bail!("a kind: resource, snapshotted resource, component or snapshotted component"),
        };
        let mut versions = BTreeMap::new();
        let declared = declaration.get("versions").and_then(serde_json::Value::as_object).context("versions")?;
        for (version, fields) in declared {
            let version: u32 = version.parse().with_context(|| format!("version {version}"))?;
            let fields = fields.as_array().with_context(|| format!("version {version}'s fields"))?;
            let fields = fields
                .iter()
                .map(|field| {
                    let name = field.get("name").and_then(serde_json::Value::as_str).context("a field's name")?;
                    let lanes = lanes_of(field).with_context(|| format!("field {name}"))?;
                    let was = field.get("was").and_then(serde_json::Value::as_str).map(str::to_owned);
                    let default = match field.get("default") {
                        None => vec![0.0; lanes],
                        Some(serde_json::Value::Number(number)) => vec![lane(number.as_f64().unwrap_or(0.0)); lanes],
                        Some(serde_json::Value::Array(values)) => values
                            .iter()
                            .map(|value| value.as_f64().map(lane))
                            .collect::<Option<_>>()
                            .context("a default is numbers")?,
                        Some(_) => bail!("field {name}'s default is a number or numbers"),
                    };
                    ensure!(default.len() == lanes, "field {name}'s default has {} lanes, not {lanes}", default.len());
                    Ok(Field { name: name.to_owned(), lanes, was, default })
                })
                .collect::<Result<Vec<_>>>()
                .with_context(|| format!("version {version}"))?;
            versions.insert(version, fields);
        }
        Ok(SavedType { name: name.to_owned(), key, kind, versions })
    }

    /// The lanes a value of `version` takes.
    fn width(&self, version: u32) -> Option<usize> {
        Some(self.versions.get(&version)?.iter().map(|field| field.lanes).sum())
    }

    /// How to bring a value's lanes from version `from` to `to`, if both
    /// are declared: where each lane of the new layout comes from.
    fn conversion(&self, from: u32, to: u32) -> Result<Vec<Lane>> {
        let old = self.versions.get(&from).with_context(|| format!("version {from} is not declared"))?;
        let new = self.versions.get(&to).with_context(|| format!("version {to} is not declared"))?;
        let mut lanes = Vec::new();
        for field in new {
            let mut start = 0;
            let mut found = None;
            for saved in old {
                if saved.name == field.name || field.was.as_deref() == Some(saved.name.as_str()) {
                    found = Some((start, saved.lanes));
                }
                start += saved.lanes;
            }
            match found {
                Some((start, count)) => {
                    ensure!(
                        count == field.lanes,
                        "field {} takes {} lanes in version {to} and {count} in {from}",
                        field.name,
                        field.lanes
                    );
                    lanes.extend((start..start + count).map(Lane::Saved));
                }
                None => lanes.extend(field.default.iter().copied().map(Lane::Default)),
            }
        }
        Ok(lanes)
    }
}

/// Where a migrated lane comes from.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lane {
    Saved(usize),
    Default(f32),
}

fn convert(lanes: &[Lane], value: &[f32]) -> Vec<f32> {
    lanes
        .iter()
        .map(|lane| match lane {
            Lane::Saved(at) => value[*at],
            Lane::Default(default) => *default,
        })
        .collect()
}

/// A section of a state: where its header is, and its fields.
#[derive(Clone, Copy, Debug)]
struct Section {
    start: usize,
    key: u32,
    version: u32,
    width: usize,
    length: usize,
}

/// A state's sections, in order.
fn sections(state: &[f32]) -> Result<Vec<Section>> {
    ensure!(state.len() >= 3 && state[0] == STATE_MAGIC, "not a saved state");
    let count = whole(state[2]).context("a state's section count")?;
    let mut sections = Vec::new();
    let mut at = 3;
    for _ in 0..count {
        let header = state.get(at..at + 4).context("a state cut short")?;
        let field = |lane: f32| whole(lane).context("a section header that is no whole number");
        let section = Section {
            start: at,
            key: field(header[0])?,
            version: field(header[1])?,
            width: field(header[2])? as usize,
            length: field(header[3])? as usize,
        };
        at += 4 + section.length;
        ensure!(at <= state.len(), "a section past the state's end");
        sections.push(section);
    }
    ensure!(at == state.len(), "lanes after the state's sections");
    Ok(sections)
}

/// A section's lanes brought from its version to `to`.
fn migrate_section(saved: &SavedType, section: &Section, lanes: &[f32], to: u32) -> Result<(usize, Vec<f32>)> {
    let conversion = saved.conversion(section.version, to)?;
    let (from_width, to_width) = (saved.width(section.version).unwrap_or(0), conversion.len());
    let values = |count: usize, block: &[f32]| -> Result<Vec<f32>> {
        ensure!(block.len() == count * from_width, "a value of version {} is not {from_width} lanes", section.version);
        Ok(block.chunks(from_width.max(1)).take(count).flat_map(|value| convert(&conversion, value)).collect())
    };
    Ok(match saved.kind {
        Kind::Resource => (0, values(1, lanes)?),
        Kind::SnapshottedResource => {
            let current = lanes.first().copied().and_then(whole).context("a snapshotted resource without its count")?;
            ensure!(current as usize == from_width, "a snapshotted resource's current value is not {from_width} lanes");
            let mut migrated = vec![to_width as f32];
            migrated.extend(values(2, &lanes[1..])?);
            (0, migrated)
        }
        Kind::Component | Kind::SnapshottedComponent => {
            ensure!(section.width == from_width, "a component saved {} lanes wide, not {from_width}", section.width);
            let stride = from_width + 1;
            ensure!(lanes.len().is_multiple_of(stride), "a component's records are cut short");
            let mut migrated = Vec::with_capacity(lanes.len() / stride * (to_width + 1));
            for record in lanes.chunks(stride) {
                if record[0] == 1.0 {
                    migrated.push(1.0);
                    migrated.extend(convert(&conversion, &record[1..]));
                } else {
                    migrated.extend(std::iter::repeat_n(0.0, to_width + 1));
                }
            }
            (to_width, migrated)
        }
    })
}

/// `state` with the sections `targets` names (key and the build's version)
/// brought to those versions by their declarations; the keys it migrated.
pub fn migrate(state: &[f32], types: &SavedTypes, targets: &[(u32, u32)]) -> Result<(Vec<f32>, Vec<u32>)> {
    let mut migrated = state[..3].to_vec();
    let mut keys = Vec::new();
    for section in sections(state)? {
        let lanes = &state[section.start + 4..section.start + 4 + section.length];
        let target = targets.iter().find(|(key, _)| *key == section.key).map(|(_, version)| *version);
        let declared = types.types.get(&section.key);
        match (target, declared) {
            (Some(to), Some(saved)) if to != section.version => {
                let (width, lanes) = migrate_section(saved, &section, lanes, to).with_context(|| {
                    format!(
                        "migrating {} (section {}) from version {} to {to}",
                        saved.name, section.key, section.version
                    )
                })?;
                migrated.extend([section.key as f32, to as f32, width as f32, lanes.len() as f32]);
                migrated.extend(lanes);
                keys.push(section.key);
            }
            _ => migrated.extend_from_slice(&state[section.start..section.start + 4 + section.length]),
        }
    }
    Ok((migrated, keys))
}

/// The sections a restore reset for want of a migration that `types`
/// declares the way to: their keys and the build's versions.
pub fn declared_resets(types: &SavedTypes, outcomes: &[crate::game::SectionOutcome]) -> Vec<(u32, u32)> {
    outcomes
        .iter()
        .filter(|outcome| outcome.fate == crate::game::RESET)
        .filter(|outcome| {
            types.types.get(&outcome.key).is_some_and(|saved| saved.conversion(outcome.saved, outcome.version).is_ok())
        })
        .map(|outcome| (outcome.key, outcome.version))
        .collect()
}

/// Why `types` cannot migrate between two of its versions, if it cannot.
pub fn check(types: &SavedTypes) -> Result<()> {
    for saved in types.types.values() {
        let versions: Vec<u32> = saved.versions.keys().copied().collect();
        for (index, from) in versions.iter().enumerate() {
            for to in &versions[index + 1..] {
                saved.conversion(*from, *to).map_err(|error| anyhow!("{}: {error:#}", saved.name))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::SectionOutcome;

    const DECLARED: &str = r#"{"types": {
        "score": {"key": 100, "kind": "resource", "versions": {
            "1": [{"name": "points", "type": "f32"}],
            "2": [{"name": "points", "type": "f32"}, {"name": "bonus", "type": "f32", "default": 0.5}]}},
        "camera": {"key": 101, "kind": "snapshotted resource", "versions": {
            "1": [{"name": "position", "type": "vec3"}, {"name": "spin", "type": "f32"}],
            "2": [{"name": "rotation", "type": "f32", "was": "spin"}, {"name": "position", "type": "vec3"}]}},
        "health": {"key": 102, "kind": "component", "versions": {
            "3": [{"name": "hp", "type": "f32"}, {"name": "armor", "type": "f32"}],
            "4": [{"name": "hp", "type": "f32"}]}}}}"#;

    /// A state of sections: key, version, width and lanes each.
    fn state(sections: &[(u32, u32, usize, &[f32])]) -> Vec<f32> {
        let mut state = vec![STATE_MAGIC, 1.0, sections.len() as f32];
        for (key, version, width, lanes) in sections {
            state.extend([*key as f32, *version as f32, *width as f32, lanes.len() as f32]);
            state.extend_from_slice(lanes);
        }
        state
    }

    #[test]
    fn declarations_parse_and_check() {
        let types = SavedTypes::parse(DECLARED).unwrap();
        assert_eq!(types.types[&101].kind, Kind::SnapshottedResource);
        assert_eq!(types.types[&100].versions[&2][1].default, [0.5]);
        assert!(check(&types).is_ok());
        assert!(SavedTypes::parse(NO_SAVED_TYPES).unwrap().types.is_empty());
        let retyped = r#"{"types": {"t": {"key": 7, "kind": "resource", "versions": {
            "1": [{"name": "a", "type": "f32"}], "2": [{"name": "a", "type": "vec2"}]}}}}"#;
        let error = format!("{:#}", check(&SavedTypes::parse(retyped).unwrap()).unwrap_err());
        assert!(error.contains("field a takes 2 lanes in version 2 and 1 in 1"), "{error}");
        assert!(SavedTypes::parse(r#"{"types": {"t": {"key": 7, "kind": "file", "versions": {}}}}"#).is_err());
        assert!(SavedTypes::parse(r#"{"types": {"t": {"key": 7, "kind": "resource", "versions": {"1": [{"name": "a", "type": "f32", "default": [1, 2]}]}}}}"#).is_err());
    }

    #[test]
    fn sections_migrate_by_field_names() {
        let types = SavedTypes::parse(DECLARED).unwrap();
        let saved = state(&[
            (0, 1, 0, &[1.0, 1.0]),
            (100, 1, 0, &[42.0]),
            (101, 1, 0, &[4.0, 1.0, 2.0, 3.0, 0.25, 5.0, 6.0, 7.0, 0.5]),
            (102, 3, 2, &[1.0, 10.0, 2.0, 0.0, 0.0, 0.0]),
            (103, 9, 0, &[9.0]),
        ]);
        let (migrated, keys) = migrate(&saved, &types, &[(100, 2), (101, 2), (102, 4), (103, 10)]).unwrap();
        assert_eq!(keys, [100, 101, 102], "an undeclared type is left as it is");
        let expected = state(&[
            (0, 1, 0, &[1.0, 1.0]),
            (100, 2, 0, &[42.0, 0.5]),
            (101, 2, 0, &[4.0, 0.25, 1.0, 2.0, 3.0, 0.5, 5.0, 6.0, 7.0]),
            (102, 4, 1, &[1.0, 10.0, 0.0, 0.0]),
            (103, 9, 0, &[9.0]),
        ]);
        assert_eq!(migrated, expected);
        // Only the sections asked for move.
        assert_eq!(migrate(&saved, &types, &[]).unwrap(), (saved.clone(), Vec::new()));
        assert!(migrate(&saved[..saved.len() - 1], &types, &[]).is_err(), "a state cut short");
    }

    #[test]
    fn only_resets_with_a_declared_way_are_migrated() {
        let types = SavedTypes::parse(DECLARED).unwrap();
        let outcome = |key, fate, saved, version| SectionOutcome { key, fate, saved, version };
        let outcomes = [
            outcome(100, crate::game::RESET, 1, 2),
            outcome(101, 1, 1, 2),
            outcome(102, crate::game::RESET, 3, 5),
            outcome(104, crate::game::RESET, 1, 2),
        ];
        assert_eq!(
            declared_resets(&types, &outcomes),
            [(100, 2)],
            "the game migrated 101; 102's 5 and 104 are undeclared"
        );
    }
}
