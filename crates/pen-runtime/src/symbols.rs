//! Where a guest's code came from, for traps and profiles: function names
//! from the module's standard `name` custom section (and its exports' for
//! functions that has no name for), and source positions
//! from a source map embedded in its `sourceMappingURL` section
//! (docs/source-maps.md). Native pen (blot-host) and the browser read them
//! alike and describe a trap's frames the same way. Debug information only
//! adds words: a module without it runs the same, its frames shown by
//! function index, and one with unreadable debug information runs too, the
//! problem said once ([`Symbols::problems`]).
use std::collections::HashMap;
use std::fmt;

use anyhow::{Result, anyhow, bail, ensure};

use crate::abi::{Reader, sections};

/// The custom section holding the module's debug names.
pub const NAME_SECTION: &str = "name";
/// The custom section holding the source map's URL; a `data:` URL holds the
/// map itself.
pub const SOURCE_MAP_SECTION: &str = "sourceMappingURL";

/// One frame of a guest's stack: the function it was in (in the module's
/// function index space, imports first) and, when the engine says, the
/// byte offset in the module of the instruction it was at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    pub function: u32,
    pub offset: Option<usize>,
}

/// A place in a guest's source: the file as the source map names it, and
/// its line and column, counting from 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Location<'a> {
    pub file: &'a str,
    pub line: u32,
    pub column: u32,
}

impl fmt::Display for Location<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// A guest module's debug information.
#[derive(Clone, Debug, Default)]
pub struct Symbols {
    /// The module's own name, if its name section gives one.
    pub module: Option<String>,
    functions: HashMap<u32, String>,
    lines: Option<SourceMap>,
    /// What was there but could not be read.
    pub problems: Vec<String>,
}

impl Symbols {
    /// Reads a module's debug information. Malformed debug information is
    /// left out and said in [`Symbols::problems`], as the Wasm specification
    /// asks of a malformed name section.
    pub fn read(wasm: &[u8]) -> Symbols {
        let mut symbols = Symbols::default();
        let sections = match sections(wasm) {
            Ok(sections) => sections,
            Err(error) => {
                symbols.problems.push(format!("reading the module's sections: {error:#}"));
                return symbols;
            }
        };
        let mut exports = HashMap::new();
        for (id, contents) in sections {
            let (section, read) = match id {
                0 => {
                    let mut reader = Reader { bytes: contents, at: 0 };
                    let Ok(section) = name(&mut reader) else {
                        symbols.problems.push("a custom section's name is malformed".into());
                        continue;
                    };
                    let payload = &contents[reader.at..];
                    let read = match section.as_str() {
                        NAME_SECTION => symbols.read_names(payload),
                        SOURCE_MAP_SECTION => symbols.read_source_map_url(payload),
                        _ => Ok(()),
                    };
                    (section, read)
                }
                EXPORT_SECTION => ("export".to_owned(), read_exports(contents, &mut exports)),
                _ => continue,
            };
            if let Err(error) = read {
                symbols.problems.push(format!("the {section} section: {error:#}"));
            }
        }
        for (index, name) in exports {
            symbols.functions.entry(index).or_insert(name);
        }
        symbols
    }

    /// The name section: its module name (subsection 0) and function names
    /// (1); the others (local names and the like) are skipped.
    fn read_names(&mut self, payload: &[u8]) -> Result<()> {
        let mut reader = Reader { bytes: payload, at: 0 };
        while reader.at < payload.len() {
            let id = reader.byte()?;
            let size = reader.leb(false)? as usize;
            let end = reader.at.checked_add(size).filter(|end| *end <= payload.len());
            let end = end.ok_or_else(|| anyhow!("truncated subsection {id}"))?;
            let mut subsection = Reader { bytes: &payload[..end], at: reader.at };
            match id {
                0 => self.module = Some(name(&mut subsection)?),
                1 => {
                    for _ in 0..subsection.leb(false)? {
                        let index = subsection.leb(false)?;
                        let name = name(&mut subsection)?;
                        self.functions.insert(index, name);
                    }
                }
                _ => {}
            }
            reader.at = end;
        }
        Ok(())
    }

    fn read_source_map_url(&mut self, payload: &[u8]) -> Result<()> {
        let mut reader = Reader { bytes: payload, at: 0 };
        let url = name(&mut reader)?;
        ensure!(reader.at == payload.len(), "trailing bytes after the URL");
        let json = embedded_map(&url)?;
        self.lines = Some(SourceMap::parse(&json)?);
        Ok(())
    }

    /// Whether the module has any: names or source positions.
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty() && self.lines.is_none()
    }

    /// Whether the module maps its code to source positions.
    pub fn has_source_map(&self) -> bool {
        self.lines.is_some()
    }

    /// The name of function `index`, if the module gives it one.
    pub fn function(&self, index: u32) -> Option<&str> {
        self.functions.get(&index).map(String::as_str)
    }

    /// Function `index`'s name, or `<wasm function N>` without one, for
    /// backtraces and profiles.
    pub fn label(&self, index: u32) -> String {
        match self.function(index) {
            Some(name) => name.to_owned(),
            None => format!("<wasm function {index}>"),
        }
    }

    /// The source position of the instruction at `offset` in the module.
    pub fn location(&self, offset: usize) -> Option<Location<'_>> {
        self.lines.as_ref()?.location(offset)
    }

    /// `frames`, innermost first, as a guest backtrace: each frame's offset
    /// and function, then its source position when the module has one.
    pub fn backtrace(&self, frames: &[Frame]) -> Backtrace {
        let mut text = String::from("guest backtrace:");
        for (index, frame) in frames.iter().enumerate() {
            text.push_str(&format!("\n  {index:>3}: "));
            if let Some(offset) = frame.offset {
                text.push_str(&format!("{offset:#8x} - "));
            }
            text.push_str(&self.label(frame.function));
            if let Some(location) = frame.offset.and_then(|offset| self.location(offset)) {
                text.push_str(&format!("\n                at {location}"));
            }
        }
        Backtrace(text)
    }
}

/// The export section's id.
const EXPORT_SECTION: u8 = 7;

/// The functions a module exports, by index: the first name each has.
fn read_exports(contents: &[u8], exports: &mut HashMap<u32, String>) -> Result<()> {
    let mut reader = Reader { bytes: contents, at: 0 };
    for _ in 0..reader.leb(false)? {
        let name = name(&mut reader)?;
        let kind = reader.byte()?;
        let index = reader.leb(false)?;
        if kind == 0 {
            exports.entry(index).or_insert(name);
        }
    }
    Ok(())
}

/// A Wasm name: its length, then UTF-8.
fn name(reader: &mut Reader) -> Result<String> {
    let length = reader.leb(false)? as usize;
    reader.text(length)
}

/// Where a guest stopped, as [`Symbols::backtrace`] describes it: the
/// context both hosts give an error from a trapped guest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Backtrace(pub String);

impl fmt::Display for Backtrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The frames of a JavaScript engine's stack trace that are in Wasm,
/// innermost first: V8's `at NAME (wasm://wasm/HASH:wasm-function[12]:0x1a2b)`
/// and Firefox's `NAME@URL:wasm-function[12]:0x1a2b`, whose offsets are in
/// the module, as wasmtime's are.
pub fn js_frames(stack: &str) -> Vec<Frame> {
    stack
        .lines()
        .filter_map(|line| {
            let (_, rest) = line.split_once("wasm-function[")?;
            let (index, rest) = rest.split_once(']')?;
            let function = index.parse().ok()?;
            let offset = rest.strip_prefix(":0x").and_then(|rest| {
                let digits = rest.find(|character: char| !character.is_ascii_hexdigit()).unwrap_or(rest.len());
                usize::from_str_radix(&rest[..digits], 16).ok()
            });
            Some(Frame { function, offset })
        })
        .collect()
}

/// The source map a `data:` URL holds. A map in a file of its own (any
/// other URL) is for browser developer tools; pen reads only embedded ones,
/// which travel inside the module through every build and package.
fn embedded_map(url: &str) -> Result<String> {
    let Some(rest) = url.strip_prefix("data:") else {
        bail!("the source map is a file of its own ({url}); pen reads only one embedded as a data: URL");
    };
    let (header, data) = rest.split_once(',').ok_or_else(|| anyhow!("a data: URL without a comma"))?;
    let mut parts = header.split(';');
    ensure!(parts.next() == Some("application/json"), "the source map must be application/json, not {header}");
    ensure!(parts.next_back() == Some("base64"), "the embedded source map must be base64");
    let bytes = base64(data)?;
    String::from_utf8(bytes).map_err(|_| anyhow!("the embedded source map is not UTF-8"))
}

/// A base64 digit's value.
fn digit(character: u8) -> Option<u8> {
    Some(match character {
        b'A'..=b'Z' => character - b'A',
        b'a'..=b'z' => character - b'a' + 26,
        b'0'..=b'9' => character - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => return None,
    })
}

/// Standard base64, padded or not.
fn base64(text: &str) -> Result<Vec<u8>> {
    let text = text.trim_end_matches('=');
    let mut bytes = Vec::with_capacity(text.len() * 3 / 4);
    let (mut bits, mut count) = (0u32, 0);
    for character in text.bytes() {
        let value = digit(character).ok_or_else(|| anyhow!("invalid base64"))?;
        bits = (bits << 6) | u32::from(value);
        count += 6;
        if count >= 8 {
            count -= 8;
            bytes.push(u8::try_from(bits >> count).unwrap_or_default());
            bits &= (1 << count) - 1;
        }
    }
    Ok(bytes)
}

/// Where each stretch of a module's code came from (Source Map revision 3,
/// with Wasm's convention: one generated line, whose columns are byte
/// offsets in the module).
#[derive(Clone, Debug, Default)]
struct SourceMap {
    sources: Vec<String>,
    /// Ascending by offset: each runs to the next.
    segments: Vec<Segment>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Segment {
    offset: u32,
    /// From here, nowhere in the source (a one-field segment).
    unmapped: bool,
    source: u32,
    line: u32,
    column: u32,
}

impl SourceMap {
    fn parse(json: &str) -> Result<SourceMap> {
        let map: serde_json::Value = serde_json::from_str(json)?;
        ensure!(map.get("version").and_then(serde_json::Value::as_u64) == Some(3), "not a version 3 source map");
        let root = map.get("sourceRoot").and_then(serde_json::Value::as_str).unwrap_or("");
        let sources = map
            .get("sources")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow!("the source map has no sources"))?
            .iter()
            .map(|source| {
                let source = source.as_str().unwrap_or("");
                match root {
                    "" => source.to_owned(),
                    root => format!("{}/{source}", root.trim_end_matches('/')),
                }
            })
            .collect::<Vec<_>>();
        let mappings = map
            .get("mappings")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow!("the source map has no mappings"))?;
        // Wasm's code is all on the first generated line.
        let line = mappings.split(';').next().unwrap_or("");
        let mut segments = Vec::new();
        // Each field after the first counts on from the last segment's.
        let (mut offset, mut source, mut source_line, mut column) = (0i64, 0i64, 0i64, 0i64);
        for text in line.split(',').filter(|text| !text.is_empty()) {
            let fields = vlq(text)?;
            ensure!(matches!(fields.len(), 1 | 4 | 5), "a mapping segment has {} fields", fields.len());
            offset += fields[0];
            let unmapped = fields.len() == 1;
            if !unmapped {
                source += fields[1];
                source_line += fields[2];
                column += fields[3];
                ensure!(
                    usize::try_from(source).is_ok_and(|source| source < sources.len()),
                    "a mapping names source {source} of {}",
                    sources.len()
                );
            }
            let whole = |value: i64| u32::try_from(value).map_err(|_| anyhow!("a mapping is out of range"));
            segments.push(Segment {
                offset: whole(offset)?,
                unmapped,
                source: whole(source)?,
                line: whole(source_line)?,
                column: whole(column)?,
            });
        }
        segments.sort_by_key(|segment| segment.offset);
        Ok(SourceMap { sources, segments })
    }

    fn location(&self, offset: usize) -> Option<Location<'_>> {
        let offset = u32::try_from(offset).ok()?;
        let after = self.segments.partition_point(|segment| segment.offset <= offset);
        let segment = self.segments.get(after.checked_sub(1)?)?;
        if segment.unmapped {
            return None;
        }
        Some(Location {
            file: &self.sources[segment.source as usize],
            line: segment.line + 1,
            column: segment.column + 1,
        })
    }
}

/// A segment's base64 VLQ fields.
fn vlq(text: &str) -> Result<Vec<i64>> {
    let mut fields = Vec::new();
    let (mut value, mut shift) = (0i64, 0);
    for character in text.bytes() {
        let digit = i64::from(digit(character).ok_or_else(|| anyhow!("invalid base64 VLQ {text}"))?);
        ensure!(shift < 32, "a VLQ field is too long");
        value |= (digit & 0x1f) << shift;
        if digit & 0x20 != 0 {
            shift += 5;
            continue;
        }
        fields.push(if value & 1 == 1 { -(value >> 1) } else { value >> 1 });
        (value, shift) = (0, 0);
    }
    ensure!(shift == 0, "a VLQ field is cut off");
    Ok(fields)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A Wasm name: its length, then its bytes.
    pub(crate) fn wasm_name(text: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        leb(&mut bytes, text.len());
        bytes.extend(text.as_bytes());
        bytes
    }

    pub(crate) fn leb(bytes: &mut Vec<u8>, mut value: usize) {
        loop {
            let byte = u8::try_from(value & 0x7f).unwrap();
            value >>= 7;
            if value == 0 {
                bytes.push(byte);
                return;
            }
            bytes.push(byte | 0x80);
        }
    }

    /// A custom section.
    pub(crate) fn custom(name: &str, payload: &[u8]) -> Vec<u8> {
        let contents = [wasm_name(name), payload.to_vec()].concat();
        let mut section = vec![0];
        leb(&mut section, contents.len());
        section.extend(contents);
        section
    }

    /// A name section naming the module and functions.
    pub(crate) fn names(module: &str, functions: &[(u32, &str)]) -> Vec<u8> {
        let mut payload = Vec::new();
        let module = wasm_name(module);
        payload.push(0);
        leb(&mut payload, module.len());
        payload.extend(module);
        let mut map = Vec::new();
        leb(&mut map, functions.len());
        for (index, name) in functions {
            leb(&mut map, usize::try_from(*index).unwrap());
            map.extend(wasm_name(name));
        }
        payload.push(1);
        leb(&mut payload, map.len());
        payload.extend(map);
        custom(NAME_SECTION, &payload)
    }

    /// Standard base64, padded.
    pub(crate) fn encode_base64(bytes: &[u8]) -> String {
        const DIGITS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut text = String::new();
        for chunk in bytes.chunks(3) {
            let word =
                chunk.iter().enumerate().fold(0u32, |word, (index, byte)| word | u32::from(*byte) << (16 - 8 * index));
            for index in 0..4 {
                text.push(if index <= chunk.len() {
                    char::from(DIGITS[((word >> (18 - 6 * index)) & 63) as usize])
                } else {
                    '='
                });
            }
        }
        text
    }

    /// A `sourceMappingURL` section embedding `map`.
    pub(crate) fn source_map(map: &str) -> Vec<u8> {
        let url = format!("data:application/json;base64,{}", encode_base64(map.as_bytes()));
        custom(SOURCE_MAP_SECTION, &wasm_name(&url))
    }

    fn module(sections: &[Vec<u8>]) -> Vec<u8> {
        [b"\0asm\x01\0\0\0".to_vec(), sections.concat()].concat()
    }

    #[test]
    fn names_come_from_the_name_section() {
        let symbols = Symbols::read(&module(&[names("main", &[(0, "blot:host.call"), (3, "game.update")])]));
        assert_eq!(symbols.module.as_deref(), Some("main"));
        assert_eq!(symbols.function(3), Some("game.update"));
        assert_eq!(symbols.label(0), "blot:host.call");
        assert_eq!(symbols.label(4), "<wasm function 4>");
        assert!(symbols.problems.is_empty() && !symbols.is_empty() && !symbols.has_source_map());
        assert!(Symbols::read(&module(&[])).is_empty(), "a module without them");
        // Exports name the functions the name section does not.
        let exports = [vec![2], wasm_name("main"), vec![0, 3], wasm_name("tick"), vec![0, 5]].concat();
        let exports = [vec![EXPORT_SECTION, u8::try_from(exports.len()).unwrap()], exports].concat();
        let symbols = Symbols::read(&module(&[exports, names("main", &[(3, "game.main")])]));
        assert_eq!((symbols.label(3), symbols.label(5)), ("game.main".into(), "tick".into()));
        assert!(symbols.problems.is_empty(), "{:?}", symbols.problems);
        // A malformed name section is left out, and said.
        let broken = Symbols::read(&module(&[custom(NAME_SECTION, &[1, 9, 1])]));
        assert!(broken.is_empty() && broken.problems.len() == 1, "{:?}", broken.problems);
    }

    #[test]
    fn source_maps_place_offsets_in_files() {
        // Offsets 0x40 and 0x48 in src/game.blot lines 12 and 14; from 0x50
        // in src/ink.blot line 3; from 0x60 nowhere.
        let map = r#"{"version":3,"sources":["src/game.blot","src/ink.blot"],"names":[],
            "mappings":"gEAWI,QAEC,QCXH,gB"}"#;
        let symbols = Symbols::read(&module(&[source_map(map)]));
        assert!(symbols.problems.is_empty(), "{:?}", symbols.problems);
        let at = |offset| symbols.location(offset).map(|location| location.to_string());
        assert_eq!(at(0x3f), None, "before the first mapping");
        assert_eq!(at(0x40).as_deref(), Some("src/game.blot:12:5"));
        assert_eq!(at(0x47).as_deref(), Some("src/game.blot:12:5"));
        assert_eq!(at(0x48).as_deref(), Some("src/game.blot:14:6"));
        assert_eq!(at(0x50).as_deref(), Some("src/ink.blot:3:3"));
        assert_eq!(at(0x60), None, "unmapped from there");
    }

    #[test]
    fn vlq_and_base64_decode() {
        assert_eq!(vlq("AAAA").unwrap(), [0, 0, 0, 0]);
        assert_eq!(vlq("gEAWI").unwrap(), [64, 0, 11, 4]);
        assert_eq!(vlq("D").unwrap(), [-1]);
        assert_eq!(vlq("2H").unwrap(), [123]);
        assert!(vlq("g").is_err(), "cut off");
        assert_eq!(base64(&encode_base64(b"pen")).unwrap(), b"pen");
        assert_eq!(base64(&encode_base64(b"blot2")).unwrap(), b"blot2");
        assert!(base64("a$").is_err());
    }

    #[test]
    fn only_embedded_maps_are_read() {
        let symbols = Symbols::read(&module(&[custom(SOURCE_MAP_SECTION, &wasm_name("game.wasm.map"))]));
        assert!(!symbols.has_source_map());
        assert!(symbols.problems[0].contains("embedded"), "{:?}", symbols.problems);
        let bad = Symbols::read(&module(&[source_map(r#"{"version":3,"sources":[],"mappings":"AAAA"}"#)]));
        assert!(!bad.has_source_map() && bad.problems[0].contains("source 0 of 0"), "{:?}", bad.problems);
    }

    #[test]
    fn backtraces_name_functions_and_places() {
        let map = r#"{"version":3,"sources":["src/game.blot"],"names":[],"mappings":"gEAWI"}"#;
        let symbols = Symbols::read(&module(&[names("main", &[(3, "game.update")]), source_map(map)]));
        let frames = [Frame { function: 3, offset: Some(0x42) }, Frame { function: 7, offset: None }];
        assert_eq!(
            symbols.backtrace(&frames).to_string(),
            "guest backtrace:\n    0:     0x42 - game.update\n                at src/game.blot:12:5\n    1: <wasm function 7>"
        );
    }

    /// The symbols of a real module: `PEN_SYMBOLS_WASM=FILE cargo test -p
    /// pen-runtime symbols_of_a_module -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads the module PEN_SYMBOLS_WASM names"]
    fn symbols_of_a_module() {
        let path = std::env::var("PEN_SYMBOLS_WASM").expect("PEN_SYMBOLS_WASM names a module");
        let symbols = Symbols::read(&std::fs::read(&path).unwrap());
        assert!(symbols.problems.is_empty(), "{:?}", symbols.problems);
        let mut functions: Vec<_> = symbols.functions.iter().collect();
        functions.sort();
        println!(
            "{path}: module {:?}, {} functions named, source map: {}",
            symbols.module,
            functions.len(),
            symbols.has_source_map()
        );
        for (index, name) in functions.iter().take(8) {
            println!("  {index}: {name}");
        }
    }

    #[test]
    fn javascript_stacks_give_their_wasm_frames() {
        // As V8 (node 24, crates/pen-web/tests/guest.mjs) gives it, then an
        // unnamed frame and a JavaScript one.
        let v8 = "RuntimeError: unreachable\n    at game.explode (wasm://wasm/18c8452e:wasm-function[6]:0x140)\n    \\n                  at game.main (wasm://wasm/18c8452e:wasm-function[1]:0xfe)\n    \\n                  at wasm://wasm/18c8452e:wasm-function[12]:0x1a2b\n    at async next (http://localhost/guest.js:40:5)";
        assert_eq!(
            js_frames(v8),
            [
                Frame { function: 6, offset: Some(0x140) },
                Frame { function: 1, offset: Some(0xfe) },
                Frame { function: 12, offset: Some(0x1a2b) }
            ]
        );
        let firefox = "game.update@http://localhost/pen_web.js line 2 > WebAssembly.instantiate:wasm-function[3]:0x42\n\\n                       next@http://localhost/guest.js:40:5";
        assert_eq!(js_frames(firefox), [Frame { function: 3, offset: Some(0x42) }]);
        assert_eq!(js_frames("at wasm-function[9]"), [Frame { function: 9, offset: None }]);
    }
}
