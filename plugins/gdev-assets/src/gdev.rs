//! gdev's asset_parsers.ts: which files a program imports, from where, and
//! what its parsers make of the catalog, the words and the tunables (the
//! shaders' parser is shader.rs).
use crate::files::Tracked;
use crate::json::{self, Json};
use crate::module::{Description, Record, Type, Value};
use crate::{paths, tunables};

/// Which parser reads an imported file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Parser {
    Catalog,
    Text,
    Tunables,
    Shader,
}

/// A file a program imports: its path from the entry's directory
/// (`./text.json`, `../src/text.json`) and its parser.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Import {
    pub(crate) name: String,
    pub(crate) parser: Parser,
}

/// The directory of the game's files that a program imports, for its entry:
/// the entry's own directory when it has the catalog (src/main.blot), else
/// the checkout's src/ beside it (devtools/main.blot).
pub(crate) fn game_source(entry: &str, files: &Tracked) -> String {
    let directory = paths::dirname(entry);
    let catalogued = |path: &str| files.exists(&paths::resolve(path, "assets/catalog.json"));
    if catalogued(&directory) {
        return directory;
    }
    let beside = paths::resolve(&directory, "../src");
    if catalogued(&beside) { beside } else { directory }
}

/// The files a program in `from` imports: the catalog, words, tunables and
/// shaders of `source`, and when `from` is elsewhere, those of `from` too,
/// each by its path from `from`.
pub(crate) fn game_assets(source: &str, from: &str, files: &Tracked) -> Vec<Import> {
    let directories = if source == from { vec![source] } else { vec![source, from] };
    let mut imports = Vec::new();
    for directory in directories {
        for (path, parser) in importable(directory, files) {
            imports.push(Import { name: import_path(from, &paths::resolve(directory, &path)), parser });
        }
    }
    imports
}

/// The files there are to import from `source`, by their paths from it: its
/// catalog, words and tunables, if it has them, and its shaders, in order.
fn importable(source: &str, files: &Tracked) -> Vec<(String, Parser)> {
    let mut found = Vec::new();
    for (path, parser) in [
        ("./assets/catalog.json", Parser::Catalog),
        ("./text.json", Parser::Text),
        ("./tunables.json", Parser::Tunables),
    ] {
        if files.exists(&paths::resolve(source, path)) {
            found.push((path.to_string(), parser));
        }
    }
    let mut shaders: Vec<String> = files
        .read_directory(&paths::resolve(source, "assets"))
        .unwrap_or_default()
        .into_iter()
        .filter(|name| name.ends_with(".wgsl"))
        .map(|name| format!("./assets/{name}"))
        .collect();
    // JavaScript's sort, by UTF-16 code units.
    shaders.sort_by(|left, right| left.encode_utf16().cmp(right.encode_utf16()));
    found.extend(shaders.into_iter().map(|path| (path, Parser::Shader)));
    found
}

/// `file` as an import from a module in `from`: "./text.json" beside it,
/// "../src/text.json" elsewhere.
fn import_path(from: &str, file: &str) -> String {
    let path = paths::relative(from, file);
    if path.starts_with("../") { path } else { format!("./{path}") }
}

/// The catalog's IDs by name: `meshes`, `materials` and `sounds`, records
/// with a U32 field per file, its index in the catalog's list.
pub(crate) fn catalog(text: &str) -> Result<Description, String> {
    let catalog = json::parse(text)?;
    if catalog == Json::Null {
        return Err("catalog: expected an object of lists".to_string());
    }
    let mut values = Vec::new();
    for (list, name) in [("meshes", "Meshes"), ("materials", "Materials"), ("sounds", "Sounds")] {
        let files = match catalog.get(list) {
            None | Some(Json::Null) => Vec::new(),
            Some(Json::Array(files)) => files
                .iter()
                .map(|file| match file {
                    Json::String(file) => Ok(file.as_str()),
                    _ => Err(format!("catalog {list}: expected a list of file names")),
                })
                .collect::<Result<Vec<_>, _>>()?,
            Some(_) => return Err(format!("catalog {list}: expected a list of file names")),
        };
        let mut fields: Vec<(String, u32)> = Vec::with_capacity(files.len());
        for (index, file) in files.iter().enumerate() {
            let field = field_of(file);
            if fields.iter().any(|(other, _)| *other == field) {
                return Err(format!("catalog {list}: two files share a name"));
            }
            fields.push((field, index as u32));
        }
        json::order_keys(&mut fields);
        let record = Record {
            name: name.to_string(),
            fields: fields.iter().map(|(field, _)| (field.clone(), Type::U32)).collect(),
        };
        let value = Value::Record(fields.into_iter().map(|(_, index)| Value::U32(index)).collect());
        values.push((list.to_string(), Type::Record(record), value));
    }
    Ok(Description { types: Vec::new(), values })
}

/// A catalog file's name as a Blot field: without its kind's suffix
/// (`.mesh.json`, `.material.json`, `.wav`, `.ogg`, `.json`), each other
/// character that is not an ASCII letter, digit or underscore an underscore
/// (two for one beyond the Basic Multilingual Plane, as JavaScript's regular
/// expressions count it).
fn field_of(file: &str) -> String {
    let stem = [".mesh.json", ".material.json"].iter().find_map(|suffix| file.strip_suffix(suffix)).unwrap_or(file);
    let stem = [".wav", ".ogg", ".json"].iter().find_map(|suffix| stem.strip_suffix(suffix)).unwrap_or(stem);
    stem.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c.to_string() } else { "_".repeat(c.len_utf16()) })
        .collect()
}

/// The characters the font has (gdev's models/font/font.py): printable ASCII
/// and these.
const FONT_EXTRA: &str = "åäöÅÄÖéüÉ‹›•…×–—";

fn in_font(character: char) -> bool {
    (' '..='~').contains(&character) || FONT_EXTRA.contains(character)
}

/// Text as Blot draws it, code points, or which characters the font lacks.
fn points(text: &str, name: &str) -> Result<Value, String> {
    let mut missing: Vec<char> = Vec::new();
    for character in text.chars().filter(|character| !in_font(*character)) {
        if !missing.contains(&character) {
            missing.push(character);
        }
    }
    if !missing.is_empty() {
        return Err(format!("text {name}: the font has no {}", missing.into_iter().collect::<String>()));
    }
    Ok(Value::Array(text.chars().map(|character| Value::U32(character as u32)).collect()))
}

/// Text to show as code points, a character the font lacks as a question
/// mark.
fn shown(text: &str) -> Value {
    Value::Array(
        text.chars().map(|character| Value::U32(if in_font(character) { character as u32 } else { 63 })).collect(),
    )
}

/// The game's words, `text`: each entry its code points, or an array of them
/// for a list.
pub(crate) fn text(text: &str) -> Result<Description, String> {
    let Json::Object(words) = json::parse(text)? else {
        return Err("text: expected an object of words".to_string());
    };
    let line = Type::Array(Box::new(Type::U32));
    let mut fields = Vec::with_capacity(words.len());
    let mut value = Vec::with_capacity(words.len());
    for (name, words) in &words {
        let not_words = || format!("text {name}: expected words or a list of words");
        match words {
            Json::String(words) => {
                fields.push((name.clone(), line.clone()));
                value.push(points(words, name)?);
            }
            Json::Array(lines) => {
                fields.push((name.clone(), Type::Array(Box::new(line.clone()))));
                let lines = lines.iter().map(|line| match line {
                    Json::String(line) => points(line, name),
                    _ => Err(not_words()),
                });
                value.push(Value::Array(lines.collect::<Result<_, _>>()?));
            }
            _ => return Err(not_words()),
        }
    }
    let record = Record { name: "Text".to_string(), fields };
    Ok(Description {
        types: Vec::new(),
        values: vec![("text".to_string(), Type::Record(record), Value::Record(value))],
    })
}

/// The live-tunable constants: `tunables`, a record with a field per
/// tunable, `(id, default, min, max)`; and `declarations`, each in the file's
/// order with its name and doc as code points, its id, default and range.
pub(crate) fn tunables(text: &str) -> Result<Description, String> {
    let declared = tunables::parse(text)?;
    let tunable = Type::Tuple(vec![Type::U32, Type::F32, Type::F32, Type::F32]);
    let codes = || Type::Array(Box::new(Type::U32));
    let declaration = Record {
        name: "Declaration".to_string(),
        fields: vec![
            ("name".to_string(), codes()),
            ("id".to_string(), Type::U32),
            ("initial".to_string(), Type::F32),
            ("least".to_string(), Type::F32),
            ("most".to_string(), Type::F32),
            ("doc".to_string(), codes()),
        ],
    };
    let record = Record {
        name: "Tunables".to_string(),
        fields: declared.iter().map(|t| (t.field.clone(), tunable.clone())).collect(),
    };
    let values = declared
        .iter()
        .map(|t| Value::Tuple(vec![Value::U32(t.id), Value::F32(t.default), Value::F32(t.min), Value::F32(t.max)]))
        .collect();
    let declarations = declared
        .iter()
        .map(|t| {
            Value::Record(vec![
                shown(&t.name),
                Value::U32(t.id),
                Value::F32(t.default),
                Value::F32(t.min),
                Value::F32(t.max),
                shown(t.doc.as_deref().unwrap_or("")),
            ])
        })
        .collect();
    Ok(Description {
        types: Vec::new(),
        values: vec![
            ("tunables".to_string(), Type::Record(record), Value::Record(values)),
            ("declarations".to_string(), Type::Array(Box::new(Type::Record(declaration))), Value::Array(declarations)),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_are_named_after_files() {
        assert_eq!(field_of("rock_menhir.mesh.json"), "rock_menhir");
        assert_eq!(field_of("brass.png.material.json"), "brass_png");
        assert_eq!(field_of("a-b.c.json"), "a_b_c");
        assert_eq!(field_of("x.mesh.json.json"), "x_mesh_json");
        assert_eq!(field_of("blip.wav"), "blip");
        assert_eq!(field_of("wind😀.ogg"), "wind__");
    }

    #[test]
    fn imports_are_paths_from_the_entry() {
        assert_eq!(import_path("/g/src", "/g/src/text.json"), "./text.json");
        assert_eq!(import_path("/g/devtools", "/g/src/assets/a.wgsl"), "../src/assets/a.wgsl");
    }
}
