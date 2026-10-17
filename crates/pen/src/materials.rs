//! A material's numbers as the dev tools' material panel edits them: the
//! color and the instance parameters its `.material.json` sets (with its
//! shader and texture, to show). An edit writes the one number token into
//! the checkout's file, keeping the rest of it byte for byte, and the file
//! then reloads like any asset edit.
#![cfg_attr(not(feature = "devtools"), allow(dead_code))]
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::value::RawValue;

/// The numbers the panel edits, in its order: the color's red, green, blue
/// and alpha, then the four instance parameters.
pub const FIELDS: usize = 8;
const COLOR: &str = "color";
const PARAMETERS: &str = "instanceParameters";
/// What a material that sets neither draws with.
const DEFAULTS: [f32; FIELDS] = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];

/// What a material's file says of the panel's numbers.
#[derive(Clone, Debug, PartialEq)]
pub struct Numbers {
    /// The color and the instance parameters (defaults where unset).
    pub values: [f32; FIELDS],
    pub color: bool,
    pub parameters: bool,
    pub shader: String,
    pub texture: String,
}

#[derive(serde::Deserialize)]
struct File<'a> {
    #[serde(default)]
    shader: String,
    #[serde(default)]
    texture: Option<String>,
    #[serde(borrow, default)]
    color: Option<&'a RawValue>,
    #[serde(borrow, default, rename = "instanceParameters")]
    parameters: Option<&'a RawValue>,
}

/// An array of four numbers, from its raw JSON.
fn four(raw: &RawValue, key: &str) -> Result<[f32; 4]> {
    let values: Vec<f64> =
        serde_json::from_str(raw.get()).with_context(|| format!("{key} is not a list of numbers"))?;
    ensure!(values.len() == 4, "{key} must have four numbers");
    Ok([0, 1, 2, 3].map(|at| crate::input::narrow(values[at])))
}

/// The panel's numbers in a material file's `text`.
pub fn numbers(text: &str) -> Result<Numbers> {
    let file: File<'_> = serde_json::from_str(text)?;
    let mut values = DEFAULTS;
    if let Some(color) = file.color {
        values[..4].copy_from_slice(&four(color, COLOR)?);
    }
    if let Some(parameters) = file.parameters {
        values[4..].copy_from_slice(&four(parameters, PARAMETERS)?);
    }
    Ok(Numbers {
        values,
        color: file.color.is_some(),
        parameters: file.parameters.is_some(),
        shader: file.shader,
        texture: file.texture.unwrap_or_default(),
    })
}

/// Where each number token of a raw array of numbers lies in it.
fn tokens(raw: &str) -> Vec<std::ops::Range<usize>> {
    let mut found = Vec::new();
    let mut start = None;
    for (at, byte) in raw.bytes().enumerate() {
        let number = byte.is_ascii_digit() || matches!(byte, b'-' | b'+' | b'.' | b'e' | b'E');
        match (number, start) {
            (true, None) => start = Some(at),
            (false, Some(from)) => {
                found.push(from..at);
                start = None;
            }
            _ => {}
        }
    }
    found
}

/// `text` with number `field` (see [`FIELDS`]) set to `value`: the token
/// replaced in place, or, for an array the file lacks, the array added with
/// the defaults it stood for.
pub fn set(text: &str, field: usize, value: f32) -> Result<String> {
    ensure!(field < FIELDS, "a material has {FIELDS} numbers to set");
    ensure!(value.is_finite(), "{value} is not a number");
    let file: File<'_> = serde_json::from_str(text)?;
    let (key, raw, index) =
        if field < 4 { (COLOR, file.color, field) } else { (PARAMETERS, file.parameters, field - 4) };
    let Some(raw) = raw else {
        let mut whole: serde_json::Map<String, serde_json::Value> = serde_json::from_str(text)?;
        let mut array = if field < 4 { DEFAULTS[..4].to_vec() } else { DEFAULTS[4..].to_vec() };
        array[index] = value;
        whole.insert(key.to_owned(), serde_json::json!(array));
        return Ok(format!("{}\n", serde_json::to_string_pretty(&whole)?));
    };
    four(raw, key)?;
    let start = raw.get().as_ptr() as usize - text.as_ptr() as usize;
    let Some(token) = tokens(raw.get()).into_iter().nth(index) else { bail!("{key} must have four numbers") };
    let mut edited = text.to_owned();
    edited.replace_range(start + token.start..start + token.end, &value.to_string());
    Ok(edited)
}

/// Sets number `field` of the material file at `path` to `value`, in one
/// rename.
pub fn write(path: &Path, field: usize, value: f32) -> Result<()> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let edited = set(&text, field, value).with_context(|| format!("editing {}", path.display()))?;
    crate::tunables::replace_file(path, &edited).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_materials_numbers() {
        let text = r#"{ "shader": "scene.wgsl", "texture": "checker.png", "color": [1, 0.5, 0.25, 1],
            "instanceParameters": [0.37, 0.0, 0.0, 0.0] }"#;
        let read = numbers(text).unwrap();
        assert_eq!(read.values, [1.0, 0.5, 0.25, 1.0, 0.37, 0.0, 0.0, 0.0]);
        assert!(read.color && read.parameters);
        assert_eq!((read.shader.as_str(), read.texture.as_str()), ("scene.wgsl", "checker.png"));
        let bare = numbers(r#"{ "shader": "sprite.wgsl" }"#).unwrap();
        assert_eq!((bare.values, bare.color, bare.texture.as_str()), (DEFAULTS, false, ""));
        assert!(numbers(r#"{ "color": [1, 2] }"#).is_err());
    }

    #[test]
    fn sets_one_number_token_and_keeps_the_rest() {
        let text = "{\n  \"shader\": \"scene.wgsl\",\n  \"color\": [1, 0.5, 2.5e-1, 1],\n  \"lighting\": \"lit\"\n}\n";
        let edited = set(text, 2, 0.75).unwrap();
        assert_eq!(edited, text.replace("2.5e-1", "0.75"));
        assert_eq!(numbers(&edited).unwrap().values[2], 0.75);
        assert_eq!(set(&edited, 0, -1.0).unwrap(), edited.replacen("[1,", "[-1,", 1));
        // An array the file lacks is added with the defaults it stood for.
        let added = set(text, 5, 2.0).unwrap();
        let read = numbers(&added).unwrap();
        assert_eq!(read.values[4..], [0.0, 2.0, 0.0, 0.0]);
        assert_eq!(read.values[..4], [1.0, 0.5, 0.25, 1.0]);
        assert!(set(text, 8, 1.0).is_err() && set(text, 0, f32::NAN).is_err());
    }
}
