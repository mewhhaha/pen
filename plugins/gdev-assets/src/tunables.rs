//! gdev's live-tunable constants as its tunables.ts reads src/tunables.json:
//! each named in lowercase words joined by dots, with finite float32 numbers,
//! `min` below `max`, the default between them and a positive step; no two
//! may share a field or an id.
use crate::json::{self, Json};
use crate::number;

/// A tunable as tunables.json declares it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Tunable {
    /// Words of lowercase letters, digits and underscores joined by dots,
    /// such as "camera.zoom_speed".
    pub(crate) name: String,
    /// The field the game reads it by: the name with underscores for dots.
    pub(crate) field: String,
    /// Which tunable a Tune event sets ([`id`]).
    pub(crate) id: u32,
    pub(crate) default: f64,
    pub(crate) min: f64,
    pub(crate) max: f64,
    /// What it does, in a line.
    pub(crate) doc: Option<String>,
}

/// The fields a tunable may have.
const FIELDS: [&str; 6] = ["name", "default", "min", "max", "step", "doc"];

/// The tunables `text` declares, checked, or what is wrong with them.
pub(crate) fn parse(text: &str) -> Result<Vec<Tunable>, String> {
    let file = json::parse(text)?;
    let Some(Json::Array(list)) = file.get("tunables") else {
        return Err(r#"tunables.json: expected { "tunables": [...] }"#.to_string());
    };
    let mut fields: Vec<(String, String)> = Vec::new();
    let mut ids: Vec<(u32, String)> = Vec::new();
    let mut tunables = Vec::with_capacity(list.len());
    for (index, entry) in list.iter().enumerate() {
        let Json::Object(declared) = entry else {
            return Err(format!("tunables.json: tunable {index} is not an object"));
        };
        let name = match entry.get("name") {
            Some(Json::String(name)) if is_name(name) => name,
            _ => {
                return Err(format!(
                    r#"tunables.json: tunable {index} needs a name of lowercase words joined by dots, such as "camera.zoom_speed""#
                ));
            }
        };
        let fail = |problem: String| format!("tunables.json: {name}: {problem}");
        let unknown: Vec<&str> =
            declared.iter().map(|(key, _)| key.as_str()).filter(|key| !FIELDS.contains(key)).collect();
        if !unknown.is_empty() {
            return Err(fail(format!("unknown {} (a tunable has {})", unknown.join(", "), FIELDS.join(", "))));
        }
        let number = |key: &str, optional: bool| match entry.get(key) {
            None if optional => Ok(None),
            Some(Json::Number(value)) if (*value as f32).is_finite() => Ok(Some(*value)),
            _ => Err(fail(format!("{key} must be a finite number"))),
        };
        let default = number("default", false)?.unwrap_or_default();
        let min = number("min", false)?.unwrap_or_default();
        let max = number("max", false)?.unwrap_or_default();
        let step = number("step", true)?;
        if (min as f32) >= (max as f32) {
            return Err(fail("min must be below max".to_string()));
        }
        if (default as f32) < (min as f32) || (default as f32) > (max as f32) {
            let [default, min, max] = [default, min, max].map(number::javascript);
            return Err(fail(format!("its default {default} is outside {min}..{max}")));
        }
        if step.is_some_and(|step| step <= 0.0) {
            return Err(fail("step must be positive".to_string()));
        }
        let doc = match entry.get("doc") {
            None => None,
            Some(Json::String(doc)) => Some(doc.clone()),
            Some(_) => return Err(fail("doc must be text".to_string())),
        };
        let field = name.replace('.', "_");
        if let Some((_, taken)) = fields.iter().find(|(other, _)| *other == field) {
            return Err(fail(format!("reads as {field}, as {taken} does; rename one")));
        }
        fields.push((field.clone(), name.clone()));
        let id = id(name);
        if let Some((_, shared)) = ids.iter().find(|(other, _)| *other == id) {
            return Err(fail(format!("its id {id} is {shared}'s too; rename one")));
        }
        ids.push((id, name.clone()));
        tunables.push(Tunable { name: name.clone(), field, id, default, min, max, doc });
    }
    Ok(tunables)
}

/// Whether `name` is lowercase words of letters, digits and underscores,
/// each starting with a letter, joined by dots.
fn is_name(name: &str) -> bool {
    name.split('.').all(|word| {
        let mut characters = word.chars();
        characters.next().is_some_and(|first| first.is_ascii_lowercase())
            && characters
                .all(|character| character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_')
    })
}

/// A tunable's id: FNV-1a over its name's UTF-8 bytes, folded to the 24 bits
/// a float32 lane holds exactly.
pub(crate) fn id(name: &str) -> u32 {
    let hash = name.bytes().fold(0x811c_9dc5_u32, |hash, byte| (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193));
    ((hash >> 24) ^ hash) & 0xff_ffff
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_match_gdev() {
        // From gdev's src/tunables.json, as its tunables.ts numbers them.
        assert_eq!(id("camera.zoom_speed"), 11175412);
        assert_eq!(id("wind.most_wait"), 6096487);
    }

    #[test]
    fn names_are_lowercase_words() {
        for name in ["a", "camera.zoom_speed", "a1.b_2"] {
            assert!(is_name(name), "{name}");
        }
        for name in ["", "A", "1a", "a.", ".a", "a..b", "a-b", "_a", "a.B"] {
            assert!(!is_name(name), "{name}");
        }
    }
}
