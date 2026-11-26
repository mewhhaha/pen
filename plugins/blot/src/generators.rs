//! Stage-0 Blot virtual module generators: compile plain `.blot` generators to
//! sandboxed ABI 8 Wasm (with automatic `#entry` signature adaptation), cache
//! compiled Stage-0 Wasm per generator source graph and generated Blot source
//! per input digest, and emit virtual `.blot` overlays for Stage 1.
use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::session::Session;
use crate::settings::{self, GeneratorInputMode, Settings};
use crate::{Build, Child, Diagnostic, Files, GuestCallError, GuestRunner, Overlay, Program};

/// Maximum size of one raw input file or one generated virtual module (16 MiB).
const MAX_MODULE_BYTES: usize = 16 * 1024 * 1024;
/// Maximum combined generated virtual module bytes across one build (64 MiB).
const MAX_TOTAL_GENERATED_BYTES: usize = 64 * 1024 * 1024;
/// Maximum number of expanded virtual modules across all rules.
const MAX_GENERATED_MODULES: usize = 4096;

/// Stage-0 provenance attached to diagnostics for agent and developer output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    pub phase: &'static str,
    pub module: String,
    pub input: String,
    pub generator: String,
}

impl Provenance {
    fn to_value(&self) -> Value {
        json!({
            "stage": "stage0",
            "phase": self.phase,
            "module": self.module,
            "input": self.input,
            "generator": self.generator,
        })
    }
}

/// Prepared Stage-0 output for one Stage-1 build.
pub struct Prepared {
    /// Virtual `.blot` overlays to send to Stage 1 (`<target>.blot`).
    pub overlays: Vec<Overlay>,
    /// Virtual module path -> `(generated_source, provenance)` for Stage-1 error enrichment.
    pub virtual_modules: HashMap<String, (String, Provenance)>,
}

#[derive(Clone)]
struct CompiledGenerator {
    wasm: Vec<u8>,
    adapter_name: String,
    record: bool,
    result: bool,
    graph_key: String,
    entry_canonical: String,
    entry_source: String,
    watched: Vec<String>,
}

struct CachedOutput {
    compiled_key: String,
    input_path: String,
    input_sha256: String,
    source: String,
}

/// Two-level Stage-0 cache held across builds in a [`Session`]:
/// - Level 1: compiled Stage-0 Wasm per `(generator, export_name, input_mode)`
///   invalidated whenever the generator or any transitive import/prelude changes.
/// - Level 2: generated virtual Blot source per target module, invalidated only
///   when its compiled generator or input bytes digest changes.
#[derive(Default)]
pub struct GeneratorCache {
    compiled: HashMap<String, CompiledGenerator>,
    outputs: HashMap<String, CachedOutput>,
}

struct ExpandedModule {
    target: String,
    input: String,
    generator: String,
    export_name: String,
    input_mode: GeneratorInputMode,
}

impl GeneratorCache {
    /// Expands generator rules, compiles/caches Stage-0 generators, executes
    /// only invalidated inputs, and produces virtual overlays for Stage 1.
    ///
    /// Returns `Ok(Ok(prepared))` on success, `Ok(Err(diagnostics))` when a
    /// generator/input problem should reject the build without killing the
    /// session, or `Err(fatal)` on toolchain failure.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare<C, F, G, S>(
        &mut self,
        root: &str,
        settings: &Settings,
        compiler_identity: &str,
        cache_directory: &str,
        watched: &mut Vec<String>,
        files: &mut F,
        guest: &mut G,
        spawn: &mut S,
    ) -> Result<Result<Prepared, Vec<Diagnostic>>, String>
    where
        C: Child,
        F: Files,
        G: GuestRunner,
        S: FnMut(&str, &[String], &[(String, String)]) -> Result<C, String>,
    {
        let mut push_watched = |path: &str| {
            if !watched.iter().any(|existing| existing == path) {
                watched.push(path.to_owned());
            }
        };

        let canonical_root = match files.canonical(root) {
            Ok(path) => path,
            Err(error) => {
                return Ok(Err(vec![asset_diagnostic(
                    root,
                    0,
                    0,
                    format!("Could not resolve project root {root}: {error}"),
                    None,
                )]));
            }
        };

        // 1. Expand all configured rules (re-evaluated on every build so `match`
        //    picks up newly added or removed asset files automatically).
        let mut expanded: Vec<ExpandedModule> = Vec::new();
        let mut seen_targets: HashSet<String> = HashSet::new();

        for rule in &settings.generators {
            push_watched(&rule.generator);
            let generator_canonical = match files.canonical(&rule.generator) {
                Ok(path) => {
                    push_watched(&path);
                    path
                }
                Err(error) => {
                    let input_hint = rule
                        .input
                        .as_deref()
                        .or_else(|| rule.match_glob.as_ref().map(|(dir, _)| dir.as_str()))
                        .unwrap_or(&rule.generator);
                    return Ok(Err(vec![asset_diagnostic(
                        &rule.generator,
                        0,
                        0,
                        format!("Could not find Blot generator {}: {error}", rule.generator),
                        Some(Provenance {
                            phase: "compile",
                            module: rule.module.clone().unwrap_or_else(|| input_hint.to_owned()),
                            input: input_hint.to_owned(),
                            generator: rule.generator.clone(),
                        }),
                    )]));
                }
            };

            if let Some((dir, suffix)) = &rule.match_glob {
                push_watched(dir);
                let dir_canonical = match files.canonical(dir) {
                    Ok(path) => path,
                    Err(error) => {
                        return Ok(Err(vec![asset_diagnostic(
                            dir,
                            0,
                            0,
                            format!("Could not read generator match directory {dir}: {error}"),
                            Some(Provenance {
                                phase: "compile",
                                module: dir.clone(),
                                input: dir.clone(),
                                generator: generator_canonical,
                            }),
                        )]));
                    }
                };
                if !inside_root(&canonical_root, &dir_canonical, true) {
                    return Ok(Err(vec![asset_diagnostic(
                        dir,
                        0,
                        0,
                        format!("Generator match directory escapes the project: {dir}"),
                        None,
                    )]));
                }
                push_watched(&dir_canonical);
                let mut entries = match files.read_directory(&dir_canonical) {
                    Ok(names) => names,
                    Err(error) => {
                        return Ok(Err(vec![asset_diagnostic(
                            &dir_canonical,
                            0,
                            0,
                            format!("Could not list generator match directory {dir_canonical}: {error}"),
                            None,
                        )]));
                    }
                };
                entries.sort();
                for name in entries {
                    if !name.ends_with(suffix) {
                        continue;
                    }
                    let joined = settings::resolve(&dir_canonical, &name);
                    if !files.is_file(&joined) {
                        continue;
                    }
                    let input_canonical = match files.canonical(&joined) {
                        Ok(path) => path,
                        Err(error) => {
                            return Ok(Err(vec![asset_diagnostic(
                                &joined,
                                0,
                                0,
                                format!("Could not resolve generator input {joined}: {error}"),
                                None,
                            )]));
                        }
                    };
                    if !inside_root(&canonical_root, &input_canonical, false) {
                        return Ok(Err(vec![asset_diagnostic(
                            &joined,
                            0,
                            0,
                            format!("Generator input escapes the project: {joined}"),
                            None,
                        )]));
                    }
                    let target = format!("{joined}.blot");
                    if !seen_targets.insert(target.clone()) {
                        return Ok(Err(vec![asset_diagnostic(
                            &joined,
                            0,
                            0,
                            format!("Duplicate generator module: {joined}"),
                            None,
                        )]));
                    }
                    if expanded.len() >= MAX_GENERATED_MODULES {
                        return Ok(Err(vec![asset_diagnostic(
                            &joined,
                            0,
                            0,
                            format!("Too many generated modules (limit {MAX_GENERATED_MODULES})"),
                            None,
                        )]));
                    }
                    push_watched(&input_canonical);
                    expanded.push(ExpandedModule {
                        target,
                        input: input_canonical,
                        generator: generator_canonical.clone(),
                        export_name: rule.export_name.clone(),
                        input_mode: rule.input_mode,
                    });
                }
            } else {
                let raw_module = rule.module.as_deref().expect("validated explicit module");
                let raw_input = rule.input.as_deref().expect("validated explicit input");
                push_watched(raw_input);
                let target_raw =
                    if raw_module.ends_with(".blot") { raw_module.to_owned() } else { format!("{raw_module}.blot") };
                let target = match canonicalize_virtual_path(&canonical_root, &target_raw, files) {
                    Ok(path) => path,
                    Err(message) => {
                        return Ok(Err(vec![asset_diagnostic(raw_input, 0, 0, message, None)]));
                    }
                };
                let input_canonical = match files.canonical(raw_input) {
                    Ok(path) => path,
                    Err(error) => {
                        return Ok(Err(vec![asset_diagnostic(
                            raw_input,
                            0,
                            0,
                            format!("Could not read generator input {raw_input}: {error}"),
                            Some(Provenance {
                                phase: "execute",
                                module: target,
                                input: raw_input.to_owned(),
                                generator: generator_canonical,
                            }),
                        )]));
                    }
                };
                if !inside_root(&canonical_root, &input_canonical, false) {
                    return Ok(Err(vec![asset_diagnostic(
                        raw_input,
                        0,
                        0,
                        format!("Generator input escapes the project: {raw_input}"),
                        None,
                    )]));
                }
                if !seen_targets.insert(target.clone()) {
                    return Ok(Err(vec![asset_diagnostic(
                        raw_input,
                        0,
                        0,
                        format!("Duplicate generator module: {raw_module}"),
                        None,
                    )]));
                }
                if expanded.len() >= MAX_GENERATED_MODULES {
                    return Ok(Err(vec![asset_diagnostic(
                        raw_input,
                        0,
                        0,
                        format!("Too many generated modules (limit {MAX_GENERATED_MODULES})"),
                        None,
                    )]));
                }
                push_watched(&input_canonical);
                expanded.push(ExpandedModule {
                    target,
                    input: input_canonical,
                    generator: generator_canonical,
                    export_name: rule.export_name.clone(),
                    input_mode: rule.input_mode,
                });
            }
        }

        // Drop Level-2 cache entries for modules no longer in the expanded set.
        self.outputs.retain(|target, _| seen_targets.contains(target));

        // 2. Snapshot & compile each distinct generator once (Level-1 cache),
        //    then execute only inputs whose digest or compiled generator changed
        //    (Level-2 cache).
        let aliases = sorted_aliases(settings);
        let mut overlays = Vec::with_capacity(expanded.len());
        let mut virtual_modules = HashMap::with_capacity(expanded.len());
        let mut total_generated_bytes: usize = 0;
        let mut checked_this_build: HashSet<String> = HashSet::new();

        for item in &expanded {
            let config_key = format!("{}\0{}\0{}", item.generator, item.export_name, item.input_mode.as_str());
            if checked_this_build.insert(config_key.clone()) {
                let snap = match snapshot_generator(&item.generator, settings.prelude.as_deref(), &aliases, files) {
                    Ok(snap) => snap,
                    Err(message) => {
                        return Ok(Err(vec![asset_diagnostic(
                            &item.generator,
                            0,
                            0,
                            message,
                            Some(Provenance {
                                phase: "compile",
                                module: item.target.clone(),
                                input: item.input.clone(),
                                generator: item.generator.clone(),
                            }),
                        )]));
                    }
                };
                for path in &snap.watched {
                    push_watched(path);
                }
                let graph_key =
                    format!("{}\0{}\0{}\0{}", snap.key, compiler_identity, item.export_name, item.input_mode.as_str());
                let cache_hit = self.compiled.get(&config_key).is_some_and(|c| c.graph_key == graph_key);
                if !cache_hit {
                    let compiled = compile_generator(
                        root,
                        settings,
                        cache_directory,
                        &snap,
                        &item.export_name,
                        item.input_mode,
                        &graph_key,
                        &item.target,
                        &item.input,
                        spawn,
                    )?;
                    match compiled {
                        Ok(compiled) => {
                            self.compiled.insert(config_key.clone(), compiled);
                        }
                        Err(diagnostics) => return Ok(Err(diagnostics)),
                    }
                }
            }

            let compiled = self.compiled.get(&config_key).expect("compiled or cached above");
            for path in &compiled.watched {
                push_watched(path);
            }

            let bytes = match files.read(&item.input) {
                Ok(bytes) => bytes,
                Err(error) => {
                    return Ok(Err(vec![asset_diagnostic(
                        &item.input,
                        0,
                        0,
                        format!("Could not read generator input {}: {error}", item.input),
                        Some(Provenance {
                            phase: "execute",
                            module: item.target.clone(),
                            input: item.input.clone(),
                            generator: item.generator.clone(),
                        }),
                    )]));
                }
            };
            if bytes.len() > MAX_MODULE_BYTES {
                return Ok(Err(vec![asset_diagnostic(
                    &item.input,
                    0,
                    0,
                    format!("Generator input {} exceeds 16 MiB", item.input),
                    Some(Provenance {
                        phase: "execute",
                        module: item.target.clone(),
                        input: item.input.clone(),
                        generator: item.generator.clone(),
                    }),
                )]));
            }
            let input_sha256 = sha256_hex(&bytes);

            let generated = match self.outputs.get(&item.target) {
                Some(cached)
                    if cached.compiled_key == compiled.graph_key
                        && cached.input_path == item.input
                        && cached.input_sha256 == input_sha256 =>
                {
                    cached.source.clone()
                }
                _ => {
                    let source = match execute_generator(
                        compiled,
                        &item.target,
                        &item.input,
                        &bytes,
                        &item.export_name,
                        item.input_mode,
                        guest,
                    ) {
                        Ok(source) => source,
                        Err(diagnostic) => return Ok(Err(vec![diagnostic])),
                    };
                    self.outputs.insert(
                        item.target.clone(),
                        CachedOutput {
                            compiled_key: compiled.graph_key.clone(),
                            input_path: item.input.clone(),
                            input_sha256,
                            source: source.clone(),
                        },
                    );
                    source
                }
            };

            total_generated_bytes = total_generated_bytes.saturating_add(generated.len());
            if total_generated_bytes > MAX_TOTAL_GENERATED_BYTES {
                return Ok(Err(vec![asset_diagnostic(
                    &item.input,
                    0,
                    0,
                    "Total generated virtual module source exceeds 64 MiB".into(),
                    Some(Provenance {
                        phase: "result",
                        module: item.target.clone(),
                        input: item.input.clone(),
                        generator: item.generator.clone(),
                    }),
                )]));
            }

            overlays.push(Overlay { path: item.target.clone(), contents: Some(generated.clone()) });
            virtual_modules.insert(
                item.target.clone(),
                (
                    generated,
                    Provenance {
                        phase: "generated_source",
                        module: item.target.clone(),
                        input: item.input.clone(),
                        generator: item.generator.clone(),
                    },
                ),
            );
        }

        Ok(Ok(Prepared { overlays, virtual_modules }))
    }
}

/// Enriches Stage-1 diagnostics pointing into a generated virtual module with
/// `virtual_source` and Stage-0 `provenance` so `plugin_build::describe` can
/// render a caret excerpt and trace back to the asset input and generator.
pub fn enrich_stage1_diagnostics(
    diagnostics: &mut [Diagnostic],
    virtual_modules: &HashMap<String, (String, Provenance)>,
) {
    if virtual_modules.is_empty() {
        return;
    }
    for diag in diagnostics {
        let Some((source, provenance)) = virtual_modules.get(&diag.file) else {
            continue;
        };
        let mut details = match diag.details.as_deref() {
            Some(raw) => match serde_json::from_str::<Value>(raw) {
                Ok(Value::Object(map)) => map,
                Ok(other) => {
                    let mut map = Map::new();
                    map.insert("detail".into(), other);
                    map
                }
                Err(_) => {
                    let mut map = Map::new();
                    map.insert("detail".into(), Value::String(raw.to_owned()));
                    map
                }
            },
            None => Map::new(),
        };
        details.insert("virtual_source".into(), Value::String(source.clone()));
        details.insert("provenance".into(), provenance.to_value());
        diag.details = Some(Value::Object(details).to_string());
    }
}

fn asset_diagnostic(file: &str, start: u32, end: u32, message: String, provenance: Option<Provenance>) -> Diagnostic {
    let details = provenance.map(|prov| json!({ "provenance": prov.to_value() }).to_string());
    Diagnostic {
        file: file.to_owned(),
        stage: "asset".to_owned(),
        code: "asset_import".to_owned(),
        start,
        end,
        message,
        details,
    }
}

fn enrich_compile_diagnostics(mut diagnostics: Vec<Diagnostic>, provenance: &Provenance) -> Vec<Diagnostic> {
    for diag in &mut diagnostics {
        let mut details = match diag.details.as_deref() {
            Some(raw) => match serde_json::from_str::<Value>(raw) {
                Ok(Value::Object(map)) => map,
                _ => Map::new(),
            },
            None => Map::new(),
        };
        details.insert("provenance".into(), provenance.to_value());
        diag.details = Some(Value::Object(details).to_string());
    }
    diagnostics
}

fn inside_root(canonical_root: &str, candidate: &str, allow_equal: bool) -> bool {
    let root = canonical_root.trim_end_matches('/');
    if candidate == root || candidate == canonical_root {
        return allow_equal;
    }
    if root.is_empty() {
        return candidate.starts_with('/');
    }
    candidate.starts_with(root) && candidate.as_bytes().get(root.len()) == Some(&b'/')
}

/// Resolves a virtual target path (which need not exist on disk) by
/// canonicalizing its deepest existing ancestor and verifying containment
/// inside `canonical_root`.
fn canonicalize_virtual_path(canonical_root: &str, target: &str, files: &mut impl Files) -> Result<String, String> {
    let mut current = target.to_owned();
    let mut tail: Vec<String> = Vec::new();
    loop {
        if let Ok(canonical) = files.canonical(&current) {
            if !inside_root(canonical_root, &canonical, true) {
                return Err(format!("Generator module path escapes the project: {target}"));
            }
            let mut joined = canonical;
            for part in tail.into_iter().rev() {
                joined = settings::resolve(&joined, &part);
            }
            if !inside_root(canonical_root, &joined, false) {
                return Err(format!("Generator module path escapes the project: {target}"));
            }
            return Ok(joined);
        }
        let parent = settings::parent(&current);
        if parent == current {
            return Err(format!("Generator module path escapes the project: {target}"));
        }
        let name = current.rsplit('/').next().unwrap_or("").to_owned();
        tail.push(name);
        current = parent;
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sorted_aliases(settings: &Settings) -> Vec<(String, String)> {
    let mut aliases = settings.imports.clone();
    if let Some(std) = &settings.std {
        aliases.push(("std/".to_owned(), std.clone()));
    }
    aliases.sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    aliases
}

struct Snapshot {
    entry: String,
    sources: BTreeMap<String, String>,
    watched: Vec<String>,
    key: String,
}

fn snapshot_generator(
    generator: &str,
    prelude: Option<&str>,
    aliases: &[(String, String)],
    files: &mut impl Files,
) -> Result<Snapshot, String> {
    let mut sources: BTreeMap<String, String> = BTreeMap::new();
    let mut hashes: BTreeMap<String, String> = BTreeMap::new();
    let mut watched: Vec<String> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();

    fn visit(
        path: &str,
        required: bool,
        aliases: &[(String, String)],
        files: &mut impl Files,
        sources: &mut BTreeMap<String, String>,
        hashes: &mut BTreeMap<String, String>,
        watched: &mut Vec<String>,
        visited: &mut HashSet<String>,
    ) -> Result<Option<String>, String> {
        if !watched.iter().any(|existing| existing == path) {
            watched.push(path.to_owned());
        }
        let canonical = match files.canonical(path) {
            Ok(path) => path,
            Err(error) => {
                if required {
                    return Err(format!("Could not read generator source {path}: {error}"));
                }
                return Ok(None);
            }
        };
        if !watched.iter().any(|existing| existing == &canonical) {
            watched.push(canonical.clone());
        }
        if !visited.insert(canonical.clone()) {
            return Ok(Some(canonical));
        }
        let bytes = match files.read(&canonical) {
            Ok(bytes) => bytes,
            Err(error) => {
                if required {
                    return Err(format!("Could not read generator source {canonical}: {error}"));
                }
                return Ok(None);
            }
        };
        let text = String::from_utf8(bytes).map_err(|_| format!("Generator source {canonical} is not valid UTF-8"))?;
        let hash = sha256_hex(text.as_bytes());
        let imports = import_paths(&text);
        sources.insert(canonical.clone(), text);
        hashes.insert(canonical.clone(), hash);
        for specifier in imports {
            if let Some(imported) = resolve_import(&canonical, &specifier, aliases) {
                visit(&imported, false, aliases, files, sources, hashes, watched, visited)?;
            }
        }
        Ok(Some(canonical))
    }

    let entry = visit(generator, true, aliases, files, &mut sources, &mut hashes, &mut watched, &mut visited)?
        .expect("required entry returns canonical path");
    if let Some(prelude) = prelude {
        visit(prelude, false, aliases, files, &mut sources, &mut hashes, &mut watched, &mut visited)?;
    }

    let mut key_hash = Sha256::new();
    for (path, hash) in &hashes {
        key_hash.update(path.as_bytes());
        key_hash.update(b"\0");
        key_hash.update(hash.as_bytes());
        key_hash.update(b"\n");
    }
    let key = format!("{:x}", key_hash.finalize());
    Ok(Snapshot { entry, sources, watched, key })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceToken<'a> {
    text: &'a str,
    start: usize,
}

/// Scans Blot source into lightweight tokens while skipping `//` comments and
/// keeping `"..."` string literals intact (so embedded Blot inside strings is
/// never mistaken for generator imports or `#entry` markers).
fn scan(source: &str) -> Vec<SourceToken<'_>> {
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut tokens = Vec::new();
    while i < len {
        let b = bytes[i];
        if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
            i += 1;
            continue;
        }
        // Line comment `// ...`
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            i += 2;
            while i < len && bytes[i] != b'\n' && bytes[i] != b'\r' {
                i += 1;
            }
            continue;
        }
        // Quoted string literal
        if b == b'"' {
            let start = i;
            i += 1;
            while i < len {
                let c = bytes[i];
                if c == b'\n' || c == b'\r' {
                    break;
                }
                if c == b'\\' && i + 1 < len && bytes[i + 1] != b'\n' && bytes[i + 1] != b'\r' {
                    i += 2;
                    continue;
                }
                i += 1;
                if c == b'"' {
                    break;
                }
            }
            tokens.push(SourceToken { text: &source[start..i], start });
            continue;
        }
        // Identifier
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            i += 1;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            tokens.push(SourceToken { text: &source[start..i], start });
            continue;
        }
        // Any single UTF-8 scalar punctuation token
        let ch_len = source[i..].chars().next().map_or(1, char::len_utf8);
        tokens.push(SourceToken { text: &source[i..i + ch_len], start: i });
        i += ch_len;
    }
    tokens
}

/// Extracts relative or aliased import/re-export paths from top-level lines.
fn import_paths(source: &str) -> Vec<String> {
    let tokens = scan(source);
    let bytes = source.as_bytes();
    let mut paths = Vec::new();
    let mut index = 0;
    while index < tokens.len() {
        let token = &tokens[index];
        if token.start > 0 && !matches!(bytes[token.start - 1], b'\n' | b'\r') {
            index += 1;
            continue;
        }
        let shape = tokens.get(index + 1).map(|t| t.text);
        let from = if matches!(token.text, "import" | "export") && shape == Some("{") {
            match tokens[index + 2..].iter().position(|t| t.text == "}") {
                Some(offset) => index + 2 + offset + 1,
                None => {
                    index += 1;
                    continue;
                }
            }
        } else if token.text == "import" && shape == Some("*") {
            index + 4 // import * as alias from "path"
        } else {
            index += 1;
            continue;
        };
        if tokens.get(from).map(|t| t.text) == Some("from")
            && let Some(literal) = tokens.get(from + 1).map(|t| t.text)
            && let Some(unescaped) = unescape_import_literal(literal)
        {
            paths.push(unescaped);
            index = from + 2;
            continue;
        }
        index += 1;
    }
    paths
}

fn unescape_import_literal(literal: &str) -> Option<String> {
    let inner = literal.strip_prefix('"')?.strip_suffix('"')?;
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next()? {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                _ => return None,
            }
        } else {
            out.push(ch);
        }
    }
    Some(out)
}

fn resolve_import(file: &str, specifier: &str, aliases: &[(String, String)]) -> Option<String> {
    if specifier.contains(['?', '#', '\0']) || specifier.to_ascii_lowercase().contains("%2f") {
        return None;
    }
    let local = specifier.starts_with("./") || specifier.starts_with("../");
    let (prefix_dir, raw_tail) = if local {
        (settings::parent(file), specifier)
    } else {
        let (prefix, dir) = aliases.iter().find(|(prefix, _)| specifier.starts_with(prefix.as_str()))?;
        (dir.clone(), &specifier[prefix.len()..])
    };
    let tail = percent_decode(raw_tail)?;
    if tail.starts_with('/') || tail.contains('\0') {
        return None;
    }
    let with_ext = if tail.ends_with(".blot") { tail } else { format!("{tail}.blot") };
    Some(settings::resolve(&prefix_dir, &with_ext))
}

fn percent_decode(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return None;
            }
            let hi = (bytes[i + 1] as char).to_digit(16)?;
            let lo = (bytes[i + 2] as char).to_digit(16)?;
            out.push(((hi << 4) | lo) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// Validates that `wasm` is a zero-import Blot ABI 8 module and checks whether
/// `export_name` exists with `Array U32 -> Array U32` signature.
/// - `Ok(true)`: valid Stage-0 generator module.
/// - `Ok(false)`: export exists with another ABI-8 signature (try adapter / report unsupported).
/// - `Err(message)`: invalid Wasm, missing export, or non-empty host imports.
fn validate_stage0_wasm(wasm: &[u8], export_name: &str, generator_path: &str) -> Result<bool, String> {
    if wasm.len() < 8 || &wasm[..8] != b"\0asm\x01\0\0\0" {
        return Err(format!("Stage-0 compiler output for {generator_path} is not a Wasm module"));
    }
    let mut at = 8;
    let mut abi_payload: Option<&[u8]> = None;
    while at < wasm.len() {
        let id = wasm[at];
        at += 1;
        let size = read_leb_u32(wasm, &mut at)? as usize;
        let end = at.checked_add(size).filter(|&end| end <= wasm.len()).ok_or("truncated Wasm section")?;
        let section = &wasm[at..end];
        at = end;
        if id == 2 {
            // Import section: count must be 0.
            let mut pos = 0;
            let count = read_leb_u32(section, &mut pos)?;
            if count > 0 {
                return Err(format!(
                    "Generator {generator_path} requires Wasm host imports; Stage 0 grants no capabilities"
                ));
            }
        } else if id == 0 {
            let mut pos = 0;
            let name_len = read_leb_u32(section, &mut pos)? as usize;
            let name_end = pos.checked_add(name_len).filter(|&e| e <= section.len()).ok_or("truncated custom name")?;
            if &section[pos..name_end] == b"blot:abi" {
                abi_payload = Some(&section[name_end..]);
            }
        }
    }
    let payload = abi_payload.ok_or_else(|| format!("Generator {generator_path} is missing `blot:abi`"))?;
    let mut pos = 0;
    let version = read_leb_u32(payload, &mut pos)?;
    if version != 8 {
        return Ok(false);
    }
    let func_count = read_leb_u32(payload, &mut pos)?;
    let mut found = None;
    for _ in 0..func_count {
        let name_len = read_leb_u32(payload, &mut pos)? as usize;
        let name_end = pos.checked_add(name_len).filter(|&e| e <= payload.len()).ok_or("truncated abi export name")?;
        let name = std::str::from_utf8(&payload[pos..name_end]).map_err(|_| "non-UTF-8 abi export name")?;
        pos = name_end;
        let param_tag = *payload.get(pos).ok_or("truncated abi param tag")?;
        pos += 1;
        let is_callback = param_tag == 4 || param_tag == 13;
        if is_callback {
            pos = pos.checked_add(2).filter(|&p| p <= payload.len()).ok_or("truncated abi callback")?;
        }
        let result_tag = *payload.get(pos).ok_or("truncated abi result tag")?;
        pos += 1;
        if name == export_name {
            found = Some((!is_callback && param_tag == 5, result_tag == 5));
        }
    }
    match found {
        None => Err(format!("Missing #entry function `{export_name}` in generator {generator_path}")),
        Some((true, true)) => Ok(true),
        Some(_) => Ok(false),
    }
}

fn read_leb_u32(bytes: &[u8], at: &mut usize) -> Result<u32, String> {
    let mut value = 0u32;
    for shift in (0..35).step_by(7) {
        let byte = *bytes.get(*at).ok_or("truncated LEB128")?;
        *at += 1;
        if shift == 28 && (byte & 0xf0) != 0 {
            return Err("LEB128 exceeds u32".into());
        }
        value |= u32::from(byte & 0x7f) << shift;
        if (byte & 0x80) == 0 {
            return Ok(value);
        }
    }
    Err("LEB128 too long".into())
}

#[allow(clippy::too_many_arguments)]
fn compile_generator<C, S>(
    root: &str,
    settings: &Settings,
    cache_directory: &str,
    snapshot: &Snapshot,
    export_name: &str,
    input_mode: GeneratorInputMode,
    graph_key: &str,
    target_module: &str,
    input_path: &str,
    spawn: &mut S,
) -> Result<Result<CompiledGenerator, Vec<Diagnostic>>, String>
where
    C: Child,
    S: FnMut(&str, &[String], &[(String, String)]) -> Result<C, String>,
{
    let entry = &snapshot.entry;
    let entry_source =
        snapshot.sources.get(entry).ok_or_else(|| format!("missing entry source snapshot for {entry}"))?;
    let provenance = Provenance {
        phase: "compile",
        module: target_module.to_owned(),
        input: input_path.to_owned(),
        generator: entry.clone(),
    };
    let field = input_mode.field_name();
    let unsupported = format!(
        "Unsupported signature for generator `{export_name}` in {entry}. Expected #[U32] or {{ path: #[U32], {field}: #[U32] }} to #[U32] or Result [#[U32], #[U32]]."
    );

    let stage0_program = Program { root: root.to_owned(), entry: entry.clone(), settings: settings.for_stage0_json() };
    let mut session = Session::open(&stage0_program, cache_directory, |name, args, env| spawn(name, args, env))?;

    let base_overlays: Vec<Overlay> = snapshot
        .sources
        .iter()
        .map(|(path, contents)| Overlay { path: path.clone(), contents: Some(contents.clone()) })
        .collect();

    let initial = session.build(&base_overlays)?;
    match initial {
        Build::Built(wasm) => {
            let _ = session.close();
            return match validate_stage0_wasm(&wasm, export_name, entry) {
                Ok(true) => Ok(Ok(CompiledGenerator {
                    wasm,
                    adapter_name: export_name.to_owned(),
                    record: false,
                    result: false,
                    graph_key: graph_key.to_owned(),
                    entry_canonical: entry.clone(),
                    entry_source: entry_source.clone(),
                    watched: snapshot.watched.clone(),
                })),
                Ok(false) => Ok(Err(vec![asset_diagnostic(entry, 0, 0, unsupported, Some(provenance))])),
                Err(message) => Ok(Err(vec![asset_diagnostic(entry, 0, 0, message, Some(provenance))])),
            };
        }
        Build::Rejected(initial_diags) => {
            if initial_diags.is_empty() || initial_diags.iter().any(|d| d.code != "entry_type") {
                let _ = session.close();
                return Ok(Err(enrich_compile_diagnostics(initial_diags, &provenance)));
            }

            // Strip `#entry` from `export_name` and adapt rich Blot signatures
            // (`Text`, `{ path, bytes/text }`, `Result [Text, Text]`) to ABI 8
            // `Array U32 -> Array U32`, matching `blot2/host/generators.ts`.
            let tokens = scan(entry_source);
            let bytes = entry_source.as_bytes();
            let marker = tokens.iter().enumerate().position(|(i, token)| {
                token.text == "#"
                    && (token.start == 0 || matches!(bytes[token.start - 1], b'\n' | b'\r'))
                    && tokens.get(i + 1).map(|t| t.text) == Some("entry")
                    && {
                        let name_index =
                            if tokens.get(i + 2).map(|t| t.text) == Some("private") { i + 4 } else { i + 3 };
                        tokens.get(name_index).map(|t| t.text) == Some(export_name)
                    }
            });
            let Some(marker) = marker else {
                let _ = session.close();
                return Ok(Err(vec![asset_diagnostic(
                    entry,
                    0,
                    0,
                    format!("Missing #entry function `{export_name}` in generator {entry}"),
                    Some(provenance),
                )]));
            };
            let start = tokens[marker].start;
            let end = tokens[marker + 2].start;
            let stripped = format!("{}{}", &entry_source[..start], &entry_source[end..]);
            let used: HashSet<&str> = tokens.iter().map(|t| t.text).collect();
            let mut adapter = "_blot_stage0".to_owned();
            while used.contains(adapter.as_str()) {
                adapter.push('_');
            }

            let record_first = initial_diags[0].message.contains("parameters of type record");
            let record_order = if record_first { [true, false] } else { [false, true] };
            let mut lowering_failure: Option<Vec<Diagnostic>> = None;

            for record in record_order {
                let result_order: &[bool] = if record { &[true, false] } else { &[true] };
                for &result in result_order {
                    let annotation = if result { ": Result [#[U32], #[U32]]" } else { "" };
                    let unpack = if record {
                        format!(
                            "  let path_size = packet[0]\n\
                             \x20 let offset = @u32.add path_size 1\n\
                             \x20 let path = @array.generate path_size (func i => return packet[(@u32.add i 1)])\n\
                             \x20 let input_values = @array.generate (@u32.sub packet.length offset) (func i => return packet[(@u32.add i offset)])\n\
                             \x20 let generated{annotation} = {export_name} {{ path: path, {field}: input_values }}\n"
                        )
                    } else {
                        format!("  let generated{annotation} = {export_name} packet\n")
                    };
                    let pack = if result {
                        "  return case generated of\n\
                         \x20   $Ok source => @array.concat #[0] source\n\
                         \x20   $Err message => @array.concat #[1] message\n"
                    } else {
                        "  return generated\n"
                    };
                    let wrapped =
                        format!("{stripped}\n#entry func {adapter} (packet: #[U32]): #[U32] =>\n{unpack}{pack}");
                    let adapted_overlays: Vec<Overlay> = snapshot
                        .sources
                        .iter()
                        .map(|(path, contents)| Overlay {
                            path: path.clone(),
                            contents: Some(if path == entry { wrapped.clone() } else { contents.clone() }),
                        })
                        .collect();
                    match session.build(&adapted_overlays)? {
                        Build::Built(wasm) => {
                            if let Ok(true) = validate_stage0_wasm(&wasm, &adapter, entry) {
                                let _ = session.close();
                                return Ok(Ok(CompiledGenerator {
                                    wasm,
                                    adapter_name: adapter,
                                    record,
                                    result,
                                    graph_key: graph_key.to_owned(),
                                    entry_canonical: entry.clone(),
                                    entry_source: entry_source.clone(),
                                    watched: snapshot.watched.clone(),
                                }));
                            }
                        }
                        Build::Rejected(diags) => {
                            if lowering_failure.is_none()
                                && !diags.is_empty()
                                && diags.iter().all(|d| d.stage != "parse" && d.stage != "check")
                            {
                                lowering_failure = Some(diags);
                            }
                        }
                    }
                }
            }
            let _ = session.close();
            if let Some(diags) = lowering_failure {
                return Ok(Err(enrich_compile_diagnostics(diags, &provenance)));
            }
            let summary = initial_diags
                .iter()
                .map(|d| format!("{} [{}] {}", d.file, d.code, d.message))
                .collect::<Vec<_>>()
                .join("\n");
            Ok(Err(vec![asset_diagnostic(entry, 0, 0, format!("{unsupported}\n{summary}"), Some(provenance))]))
        }
    }
}

fn execute_generator(
    compiled: &CompiledGenerator,
    target_module: &str,
    input_path: &str,
    bytes: &[u8],
    export_name: &str,
    input_mode: GeneratorInputMode,
    guest: &mut impl GuestRunner,
) -> Result<String, Diagnostic> {
    let exec_prov = || Provenance {
        phase: "execute",
        module: target_module.to_owned(),
        input: input_path.to_owned(),
        generator: compiled.entry_canonical.clone(),
    };
    let result_prov = || Provenance {
        phase: "result",
        module: target_module.to_owned(),
        input: input_path.to_owned(),
        generator: compiled.entry_canonical.clone(),
    };

    let data: Vec<u32> = match input_mode {
        GeneratorInputMode::Text => {
            let text = std::str::from_utf8(bytes).map_err(|_| {
                asset_diagnostic(
                    input_path,
                    0,
                    0,
                    format!("Expected UTF-8 text for generator input {input_path}"),
                    Some(exec_prov()),
                )
            })?;
            text.chars().map(u32::from).collect()
        }
        GeneratorInputMode::Bytes => bytes.iter().map(|&b| u32::from(b)).collect(),
    };

    let packet = if compiled.record {
        let path_words: Vec<u32> = input_path.chars().map(u32::from).collect();
        let path_len = u32::try_from(path_words.len()).map_err(|_| {
            asset_diagnostic(input_path, 0, 0, "Generator input path exceeds U32 length".into(), Some(exec_prov()))
        })?;
        let mut combined = Vec::with_capacity(1 + path_words.len() + data.len());
        combined.push(path_len);
        combined.extend_from_slice(&path_words);
        combined.extend_from_slice(&data);
        combined
    } else {
        data
    };

    let output = match guest.run_u32_packet(&compiled.wasm, &compiled.adapter_name, &packet) {
        Ok(words) => words,
        Err(GuestCallError::Timeout) => {
            return Err(asset_diagnostic(
                &compiled.entry_canonical,
                0,
                0,
                format!("Blot generator `{export_name}` timed out for {input_path}"),
                Some(exec_prov()),
            ));
        }
        Err(GuestCallError::Invalid(message)) => {
            return Err(asset_diagnostic(
                &compiled.entry_canonical,
                0,
                0,
                format!("Blot generator `{export_name}` failed for {input_path}: {message}"),
                Some(exec_prov()),
            ));
        }
        Err(GuestCallError::Trapped(message, trap)) => {
            let mut file = input_path;
            let mut start = 0u32;
            let mut end = 0u32;
            if let Some(info) = &trap
                && trap_span_matches(&compiled.entry_source, info)
            {
                file = &compiled.entry_canonical;
                start = info.start;
                end = info.end;
            }
            return Err(asset_diagnostic(
                file,
                start,
                end,
                format!("Blot generator `{export_name}` trapped for {input_path}: {message}"),
                Some(exec_prov()),
            ));
        }
    };

    let (is_err, payload) = if compiled.result {
        let Some((&tag, rest)) = output.split_first() else {
            return Err(asset_diagnostic(
                input_path,
                0,
                0,
                "Generator returned an empty Result packet".into(),
                Some(result_prov()),
            ));
        };
        if tag != 0 && tag != 1 {
            return Err(asset_diagnostic(
                input_path,
                0,
                0,
                format!("Generator returned an invalid Result tag ({tag})"),
                Some(result_prov()),
            ));
        }
        (tag == 1, rest)
    } else {
        (false, output.as_slice())
    };

    let mut text = String::with_capacity(payload.len());
    for &scalar in payload {
        let Some(ch) = char::from_u32(scalar) else {
            return Err(asset_diagnostic(
                input_path,
                0,
                0,
                format!("Generator returned an invalid Unicode scalar value: 0x{scalar:x}"),
                Some(result_prov()),
            ));
        };
        text.push(ch);
    }

    if is_err {
        return Err(asset_diagnostic(input_path, 0, 0, text, Some(result_prov())));
    }
    if text.contains('\0') {
        return Err(asset_diagnostic(
            input_path,
            0,
            0,
            "Generated Blot source contains a NUL byte".into(),
            Some(result_prov()),
        ));
    }
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    if text.len() > MAX_MODULE_BYTES {
        return Err(asset_diagnostic(
            input_path,
            0,
            0,
            "Generated virtual module exceeds 16 MiB".into(),
            Some(result_prov()),
        ));
    }
    Ok(text)
}

fn trap_span_matches(source: &str, trap: &crate::GuestTrapInfo) -> bool {
    let start = trap.start as usize;
    let end = trap.end as usize;
    if end <= start || end > source.len() || !source.is_char_boundary(start) || !source.is_char_boundary(end) {
        return false;
    }
    let snippet = &source[start..end];
    match trap.kind.as_str() {
        "panic" => {
            let has_panic = snippet.contains("panic");
            let message_ok = trap.message.is_empty()
                || trap.message.chars().any(|ch| matches!(ch, '\\' | '"' | '\n' | '\r' | '\t'))
                || snippet.contains(&trap.message);
            has_panic && message_ok
        }
        "bounds" => {
            (snippet.contains('[') && snippet.contains(']'))
                || [".get", ".set", ".at", ".pop", ".slice", ".remove", ".swap_remove"]
                    .iter()
                    .any(|op| snippet.contains(op))
        }
        "unreachable" => snippet.contains("unreachable"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scans_imports_without_matching_comments_or_embedded_strings() {
        let source = r#"
// import { fake } from "./commented"
import * as text from "std/text"
import { decimal, digit } from "./literals"
export { Foo } from "../shared/foo.blot"
const generated = "import { inner } from \"./not_an_import\"\n"
"#;
        assert_eq!(import_paths(source), ["std/text", "./literals", "../shared/foo.blot"]);
    }

    #[test]
    fn resolves_local_and_aliased_imports() {
        let aliases = vec![("std/".to_owned(), "/blot2/std".to_owned()), ("ink/".to_owned(), "/ink/lib".to_owned())];
        assert_eq!(
            resolve_import("/proj/tools/wgsl_gen.blot", "./literals", &aliases).as_deref(),
            Some("/proj/tools/literals.blot")
        );
        assert_eq!(
            resolve_import("/proj/tools/wgsl_gen.blot", "std/text", &aliases).as_deref(),
            Some("/blot2/std/text.blot")
        );
        assert_eq!(resolve_import("/proj/tools/wgsl_gen.blot", "unknown/mod", &aliases), None);
        assert_eq!(resolve_import("/proj/tools/wgsl_gen.blot", "./bad?query", &aliases), None);
    }

    #[test]
    fn validates_abi8_zero_import_wasm_sections() {
        fn section(id: u8, body: &[u8]) -> Vec<u8> {
            let mut out = vec![id, u8::try_from(body.len()).unwrap()];
            out.extend_from_slice(body);
            out
        }
        fn abi_custom(name: &str, param_tag: u8, result_tag: u8) -> Vec<u8> {
            let mut payload = vec![8, 1, u8::try_from(name.len()).unwrap()];
            payload.extend_from_slice(name.as_bytes());
            payload.push(param_tag);
            payload.push(result_tag);
            payload.push(0);
            let mut custom = vec![8];
            custom.extend_from_slice(b"blot:abi");
            custom.extend_from_slice(&payload);
            section(0, &custom)
        }
        let valid = [b"\0asm\x01\0\0\0".as_slice(), &abi_custom("generate", 5, 5)].concat();
        assert_eq!(validate_stage0_wasm(&valid, "generate", "gen.blot"), Ok(true));
        assert!(validate_stage0_wasm(&valid, "other", "gen.blot").unwrap_err().contains("Missing #entry"));

        let wrong_sig = [b"\0asm\x01\0\0\0".as_slice(), &abi_custom("generate", 1, 5)].concat();
        assert_eq!(validate_stage0_wasm(&wrong_sig, "generate", "gen.blot"), Ok(false));

        let with_import = [b"\0asm\x01\0\0\0".as_slice(), &section(2, &[1, 0]), &abi_custom("generate", 5, 5)].concat();
        assert!(validate_stage0_wasm(&with_import, "generate", "gen.blot").unwrap_err().contains("no capabilities"));
    }
}
