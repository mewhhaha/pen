//! pen's gdev assets plugin: a component of wit/pen-plugin.wit's
//! `assets-plugin` world that makes the virtual Blot modules gdev's programs
//! import their assets through, as gdev's `asset_parsers.ts` and blot2's
//! `compiler/assets.ts` did on Deno. pen asks for them before each build and
//! hands them to the compiler as overlays. The work is plain Rust over
//! [`Files`], which `cargo test` checks natively; only the component's
//! bindings (`component.rs`) are Wasm-only.
//!
//! A program imports, by their paths from its entry:
//!
//! - `./assets/catalog.json`: the asset catalog's IDs by name, `meshes`,
//!   `materials` and `sounds`, records with a U32 field per file, named after
//!   it (`rock_menhir.mesh.json` is `meshes.rock_menhir`, its index in the
//!   catalog's list).
//! - `./text.json`: the game's words, `text`, each entry its code points (an
//!   `Array U32`), or an array of them for a list. Every character must be in
//!   the font.
//! - `./tunables.json`: the live-tunable constants, `tunables`, a record with
//!   a field per tunable, `(id, default, min, max)`; and `declarations`, each
//!   with its name and doc as code points, its id, default and range.
//! - `./assets/NAME.wgsl`, each shader: its `Parameters` struct, if it has
//!   one, as a record type of the same name (up to four f32s), and its scalar
//!   constants (f32, u32 and bool). A shader that does not parse fails, its
//!   line in the error.
//!
//! The files come from the directory of the game's files: the entry's own when
//! it has the catalog (`src/main.blot`), else the checkout's `src/` beside it
//! (`devtools/main.blot`). An entry elsewhere imports those by their paths
//! from it (`../src/text.json`), and the same files of its own directory too
//! (`./text.json`). Each module is at its import's path with `.blot` added,
//! beside the entry (`src/text.json.blot`). Paths are POSIX ones, resolved by
//! their text alone: unlike assets.ts, the plugin cannot follow symbolic
//! links.
//!
//! A [`Generator`] keeps what it made of each file and makes it again only
//! when the file's bytes change, as assets.ts's compiler did between builds;
//! the component keeps one for as long as pen keeps it.
//!
//! The plugin takes no settings.
use std::collections::{HashMap, HashSet};
use std::fmt;

mod files;
mod gdev;
mod json;
mod module;
mod number;
mod paths;
mod shader;
mod tunables;

#[cfg(target_arch = "wasm32")]
mod component;

use files::Tracked;
use gdev::Parser;

/// Reading the checkout's files, as wit/pen-plugin.wit's `files` does: paths
/// are absolute, and one outside what the checkout grants is refused.
pub trait Files {
    fn read(&self, path: &str) -> Result<Vec<u8>, String>;
    /// The names in a directory (files and directories), sorted.
    fn read_directory(&self, directory: &str) -> Result<Vec<String>, String>;
    /// Whether a file or directory is there.
    fn exists(&self, path: &str) -> bool;
}

/// A program of the checkout's: its root, its entry file (absolute) and its
/// settings, as JSON text ("{}" when none).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub root: String,
    pub entry: String,
    pub settings: String,
}

/// A generated module: the absolute path the program imports it from, and
/// its Blot source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Module {
    pub path: String,
    pub source: String,
}

/// A program's asset modules, sorted by path, and what they depend on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generated {
    pub modules: Vec<Module>,
    /// The files (absolute) the modules were made from, and the directories
    /// whose new or removed files would change them, sorted.
    pub dependencies: Vec<String>,
}

/// Why a program's asset modules cannot be made: the file at fault (the
/// entry when it is none of the inputs) and what is wrong with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub file: String,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.file, self.message)
    }
}

impl std::error::Error for Error {}

/// The most an input or a generated module may be (assets.ts's limit).
const LIMIT: usize = 16 * 1024 * 1024;
/// The most all inputs, or all generated modules, may be together.
const TOTAL_LIMIT: usize = 64 * 1024 * 1024;
/// The most modules there may be.
const COUNT_LIMIT: usize = 4096;

/// The asset modules of `program`, made from the checkout's `files`.
pub fn generate(program: &Program, files: &dyn Files) -> Result<Generated, Error> {
    Generator::default().generate(program, files)
}

/// Makes programs' asset modules again and again, reusing what it made of
/// each file whose bytes have not changed.
#[derive(Default)]
pub struct Generator {
    /// What each file (by its path) was made into, or why it could not be.
    made: HashMap<String, Made>,
}

struct Made {
    parser: Parser,
    bytes: Vec<u8>,
    source: Result<String, String>,
}

impl Generator {
    /// The asset modules of `program`, made from the checkout's `files`.
    pub fn generate(&mut self, program: &Program, files: &dyn Files) -> Result<Generated, Error> {
        let entry = paths::resolve("/", &program.entry);
        let fail = |file: &str, message: String| Error { file: file.to_string(), message };
        check_settings(&program.settings).map_err(|message| fail(&entry, message))?;
        let files = Tracked::new(files, paths::resolve("/", &program.root));
        let directory = paths::dirname(&entry);
        let source = gdev::game_source(&entry, &files);
        let imports = gdev::game_assets(&source, &directory, &files);

        // What assets.ts's withAssetImports does before a build: each module
        // beside the entry, at its import's path with `.blot` added.
        if imports.len() > COUNT_LIMIT {
            return Err(fail(&entry, "Too many asset modules".to_string()));
        }
        let mut definitions = Vec::with_capacity(imports.len());
        for import in imports {
            let name =
                if import.name.ends_with(".blot") { import.name.clone() } else { format!("{}.blot", import.name) };
            let module = paths::resolve(&directory, &name);
            if module == entry || definitions.iter().any(|(other, _, _)| *other == module) {
                return Err(fail(
                    &entry,
                    "Asset modules must have distinct paths, separate from the entry".to_string(),
                ));
            }
            definitions.push((module, paths::resolve(&directory, &import.name), import.parser));
        }
        definitions.sort_by(|(left, _, _), (right, _, _)| left.cmp(right));

        let mut modules = Vec::with_capacity(definitions.len());
        let mut input_bytes = 0;
        let mut generated_bytes = 0;
        for (path, input, parser) in &definitions {
            let bytes = files.read(input).map_err(|message| fail(input, message))?;
            if bytes.len() > LIMIT {
                return Err(fail(input, "Asset input exceeds 16 MiB".to_string()));
            }
            input_bytes += bytes.len();
            if input_bytes > TOTAL_LIMIT {
                return Err(fail(input, "Asset input budget exceeded".to_string()));
            }
            let unchanged = self.made.get(input).is_some_and(|made| made.parser == *parser && made.bytes == bytes);
            if !unchanged {
                let source = make(*parser, &bytes);
                self.made.insert(input.clone(), Made { parser: *parser, bytes, source });
            }
            let source = self.made[input].source.clone().map_err(|message| fail(input, message))?;
            generated_bytes += source.len();
            modules.push(Module { path: path.clone(), source });
        }
        if generated_bytes > TOTAL_LIMIT {
            let last = definitions.last().map_or(entry.as_str(), |(_, input, _)| input.as_str());
            return Err(fail(last, "Generated asset sources exceed 64 MiB".to_string()));
        }
        // Forget the files the program no longer imports.
        let used: HashSet<&String> = definitions.iter().map(|(_, input, _)| input).collect();
        self.made.retain(|input, _| used.contains(input));
        Ok(Generated { modules, dependencies: files.dependencies() })
    }
}

/// The module `parser` makes of a file's `bytes`, or why it cannot.
fn make(parser: Parser, bytes: &[u8]) -> Result<String, String> {
    let text = decode(bytes);
    let described = match parser {
        Parser::Catalog => gdev::catalog(&text),
        Parser::Text => gdev::text(&text),
        Parser::Tunables => gdev::tunables(&text),
        Parser::Shader => shader::reflect(&text),
    }?;
    module::lower(&described)
}

/// The plugin takes no settings: anything but an empty object is a mistake.
fn check_settings(settings: &str) -> Result<(), String> {
    match json::parse(settings) {
        Ok(json::Json::Object(fields)) if fields.is_empty() => Ok(()),
        _ => Err(format!("the gdev assets plugin takes no settings, but has {settings}")),
    }
}

/// A file's bytes as text, as JavaScript's `TextDecoder` reads them: UTF-8,
/// with a replacement character for each invalid sequence and without a
/// leading byte order mark.
fn decode(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    text.strip_prefix('\u{feff}').unwrap_or(&text).to_string()
}
