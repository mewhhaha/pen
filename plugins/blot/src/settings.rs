//! A program's settings for the plugin (the crate's documentation has their
//! schema), and the paths in them, resolved against the program's root.
use serde_json::Value;

/// The name a checkout grants the compiler under, unless `program` says
/// otherwise.
pub const PROGRAM: &str = "blotc";

/// How much the compiler optimizes what it builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodegenTier {
    Optimized,
    /// Skips the optional scalar and vector passes, for quicker builds.
    Development,
}

impl CodegenTier {
    /// Its name in the protocol and in the settings.
    pub fn name(self) -> &'static str {
        match self {
            CodegenTier::Optimized => "optimized",
            CodegenTier::Development => "development",
        }
    }
}

/// A program's settings, with every path absolute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Settings {
    /// The name the compiler is granted under.
    pub program: String,
    /// Each import prefix (ending in `/`) and the directory behind it.
    pub imports: Vec<(String, String)>,
    /// The directory behind `std/`; none without one.
    pub std: Option<String>,
    /// The module every module sees; none without one.
    pub prelude: Option<String>,
    /// A `.blotdep` bundle of `blot dependencies`.
    pub dependencies: Option<String>,
    /// The compiler's own default (optimized) when none.
    pub codegen_tier: Option<CodegenTier>,
    /// Semantic workers (1 to 16); the compiler works serially when none.
    pub semantic_workers: Option<u32>,
    /// The only compiler identity (64 lowercase hex digits) to accept.
    pub expected_identity: Option<String>,
    /// Native readers owned by blotc. Paths are resolved against the project.
    pub native_assets: Option<Value>,
}

impl Settings {
    /// Reads `json` (a program's settings, "{}" for none), resolving the
    /// relative paths in it against `root`.
    pub fn parse(json: &str, root: &str) -> Result<Settings, String> {
        let value: Value = serde_json::from_str(json).map_err(|error| format!("the settings are not JSON: {error}"))?;
        let Value::Object(fields) = value else {
            return Err("the settings are not a JSON object".into());
        };
        const KNOWN: [&str; 9] = [
            "program",
            "imports",
            "std",
            "prelude",
            "dependencies",
            "codegenTier",
            "semanticWorkers",
            "expectedIdentity",
            "nativeAssets",
        ];
        if let Some(unknown) = fields.keys().find(|key| !KNOWN.contains(&key.as_str())) {
            return Err(format!("the settings have no `{unknown}` (they have {})", KNOWN.join(", ")));
        }
        let path = |name: &str| -> Result<Option<String>, String> {
            match fields.get(name) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::String(path)) => checked_path(path, name).map(|path| Some(resolve(root, path))),
                Some(_) => Err(format!("the settings' `{name}` is not a path")),
            }
        };
        let program = match fields.get("program") {
            None => PROGRAM.to_string(),
            Some(Value::String(name)) if !name.is_empty() => name.clone(),
            Some(_) => return Err("the settings' `program` is not a name".into()),
        };
        let std = path("std")?;
        // As blot2's `createCompiler` has it: the standard library's prelude
        // unless the settings name another, or none (null).
        let prelude = match fields.get("prelude") {
            None => std.as_deref().map(|std| resolve(std, "prelude.blot")),
            Some(_) => path("prelude")?,
        };
        let codegen_tier = match fields.get("codegenTier") {
            None | Some(Value::Null) => None,
            Some(Value::String(tier)) if tier == "optimized" => Some(CodegenTier::Optimized),
            Some(Value::String(tier)) if tier == "development" => Some(CodegenTier::Development),
            Some(_) => return Err("the settings' `codegenTier` is not \"optimized\" or \"development\"".into()),
        };
        let semantic_workers = match fields.get("semanticWorkers") {
            None | Some(Value::Null) => None,
            Some(workers) => match workers.as_u64() {
                Some(workers @ 1..=16) => Some(workers as u32),
                _ => return Err("the settings' `semanticWorkers` is not a whole number from 1 to 16".into()),
            },
        };
        let expected_identity = match fields.get("expectedIdentity") {
            None | Some(Value::Null) => None,
            Some(Value::String(identity)) if is_identity(identity) => Some(identity.clone()),
            Some(_) => return Err("the settings' `expectedIdentity` is not 64 lowercase hex digits".into()),
        };
        Ok(Settings {
            program,
            imports: imports(fields.get("imports"), root)?,
            std,
            prelude,
            dependencies: path("dependencies")?,
            codegen_tier,
            semantic_workers,
            expected_identity,
            native_assets: native_assets(fields.get("nativeAssets"), root)?,
        })
    }

    /// The compiler's arguments.
    pub fn args(&self) -> Vec<String> {
        let mut args = vec!["serve-project".to_string()];
        if let Some(workers) = self.semantic_workers {
            args.extend(["--semantic-workers".to_string(), workers.to_string()]);
        }
        args
    }

    /// The paths whose changes need a new build of the program at `entry`:
    /// its directory, every import's directory, the standard library's and
    /// the prelude, each once.
    pub fn watched(&self, entry: &str) -> Vec<String> {
        let mut watched: Vec<String> = Vec::new();
        let paths = std::iter::once(parent(entry))
            .chain(self.imports.iter().map(|(_, root)| root.clone()))
            .chain(self.std.clone())
            .chain(self.prelude.clone());
        for path in paths {
            if !watched.contains(&path) {
                watched.push(path);
            }
        }
        if let Some(assets) = &self.native_assets {
            for root in assets["roots"].as_array().into_iter().flatten().filter_map(Value::as_str) {
                let root = root.to_owned();
                if !watched.contains(&root) {
                    watched.push(root);
                }
            }
            for import in assets["imports"].as_array().into_iter().flatten() {
                if let Some(path) = import["path"].as_str() {
                    watched.push(path.to_owned());
                }
            }
        }
        watched
    }
}

fn native_assets(value: Option<&Value>, root: &str) -> Result<Option<Value>, String> {
    let mut assets = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(assets)) => assets.clone(),
        _ => return Err("nativeAssets must be an object".into()),
    };
    if assets.keys().any(|key| !["roots", "imports"].contains(&key.as_str())) {
        return Err("nativeAssets accepts roots and imports".into());
    }
    for field in ["roots", "imports"] {
        assets.entry(field.to_owned()).or_insert_with(|| Value::Array(Vec::new()));
    }
    let roots = assets.get_mut("roots").and_then(Value::as_array_mut).ok_or("nativeAssets.roots must be an array")?;
    for path in roots {
        let text = path.as_str().ok_or("nativeAssets root must be a path")?;
        *path = Value::String(resolve(root, checked_path(text, "nativeAssets.roots")?));
    }
    let imports =
        assets.get_mut("imports").and_then(Value::as_array_mut).ok_or("nativeAssets.imports must be an array")?;
    for import in imports {
        let fields = import.as_object_mut().ok_or("nativeAssets import must be an object")?;
        for field in ["module", "path"] {
            let path = fields
                .get(field)
                .and_then(Value::as_str)
                .ok_or_else(|| format!("nativeAssets import needs {field}"))?;
            let path = resolve(root, checked_path(path, field)?);
            fields.insert(field.into(), Value::String(path));
        }
    }
    Ok(Some(Value::Object(assets)))
}

/// The settings' `imports`: an object of prefixes, each ending in `/`, and
/// the directories behind them.
fn imports(value: Option<&Value>, root: &str) -> Result<Vec<(String, String)>, String> {
    let fields = match value {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Object(fields)) => fields,
        Some(_) => return Err("the settings' `imports` is not an object of prefixes and directories".into()),
    };
    let import = |(prefix, directory): (&String, &Value)| {
        if !prefix.ends_with('/') || prefix.contains('\0') {
            return Err(format!("the import prefix `{prefix}` does not end in /"));
        }
        let Value::String(directory) = directory else {
            return Err(format!("the import `{prefix}` is not a directory"));
        };
        Ok((prefix.clone(), resolve(root, checked_path(directory, prefix)?)))
    };
    fields.iter().map(import).collect()
}

/// `path` when it can be one: not empty, and no NUL in it.
fn checked_path<'a>(path: &'a str, name: &str) -> Result<&'a str, String> {
    if path.is_empty() || path.contains('\0') {
        return Err(format!("the settings' `{name}` is not a path"));
    }
    Ok(path)
}

/// Whether `identity` has the form of a compiler identity.
pub(crate) fn is_identity(identity: &str) -> bool {
    identity.len() == 64 && identity.bytes().all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// Whether `path` is absolute: from `/`, or a Windows drive's or share's.
pub(crate) fn is_absolute(path: &str) -> bool {
    path.starts_with('/') || path.starts_with("\\\\") || drive(path)
}

/// Whether `path` starts at a Windows drive (`C:\` or `C:/`).
fn drive(path: &str) -> bool {
    let bytes = path.as_bytes();
    bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' && matches!(bytes[2], b'/' | b'\\')
}

/// `path`, absolute: as it is when it is, else from `directory`, without
/// its `.` and `..` (as Node's `path.resolve` has it). A Windows path (from
/// a drive or a share) takes either separator. Protocol paths use `/`.
pub(crate) fn resolve(directory: &str, path: &str) -> String {
    let joined = if is_absolute(path) {
        path.to_string()
    } else {
        format!("{}/{path}", directory.trim_end_matches(['/', '\\']))
    };
    let windows = drive(&joined) || (joined.starts_with("\\\\") || joined.starts_with("//"));
    let separators: &[char] = if windows { &['/', '\\'] } else { &['/'] };
    // What `..` never climbs above: `/`, `C:\` or `\\server\share\`.
    let (prefix, rest) = if drive(&joined) {
        (joined[..2].to_string() + "/", &joined[3..])
    } else if windows {
        let mut parts = joined[2..].splitn(3, separators);
        let (server, share) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
        (format!("//{server}/{share}/"), parts.next().unwrap_or(""))
    } else {
        ("/".to_string(), joined.trim_start_matches('/'))
    };
    let mut parts: Vec<&str> = Vec::new();
    for part in rest.split(separators) {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            part => parts.push(part),
        }
    }
    prefix + &parts.join("/")
}

/// The directory the file at `path` (absolute) is in; the root's is itself.
pub(crate) fn parent(path: &str) -> String {
    resolve(path, "..")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn resolves_relative_paths_against_the_root() {
        assert_eq!(resolve("/a/b", "c/d"), "/a/b/c/d");
        assert_eq!(resolve("/a/b", "../c/./d/"), "/a/c/d");
        assert_eq!(resolve("/a/b", "/x/../y"), "/y");
        assert_eq!(resolve("/a", "../../.."), "/");
        assert_eq!(resolve("C:\\games\\gdev", "../blot2/std"), "C:/games/blot2/std");
        assert_eq!(resolve("C:/games/gdev", "packages/"), "C:/games/gdev/packages");
        assert_eq!(resolve("\\\\host\\share\\gdev", "..\\..\\x"), "//host/share/x");
        assert_eq!(resolve("/a", "D:\\b"), "D:/b");
    }

    #[test]
    fn finds_a_files_directory() {
        assert_eq!(parent("/a/b/main.blot"), "/a/b");
        assert_eq!(parent("/main.blot"), "/");
        assert_eq!(parent("/"), "/");
        assert_eq!(parent("C:\\gdev\\src\\main.blot"), "C:/gdev/src");
        assert_eq!(parent("C:\\main.blot"), "C:/");
    }

    #[test]
    fn reads_every_setting() {
        let json = json!({
            "program": "blotc-dev",
            "imports": { "gdev/": "packages/", "abs/": "/elsewhere" },
            "std": "../blot2/std",
            "prelude": "../blot2/std/prelude.blot",
            "dependencies": "deps.blotdep",
            "codegenTier": "development",
            "semanticWorkers": 4,
            "expectedIdentity": "0123456789abcdef".repeat(4),
        });
        let settings = Settings::parse(&json.to_string(), "/src/gdev").unwrap();
        assert_eq!(settings.program, "blotc-dev");
        assert_eq!(
            settings.imports,
            [("abs/".to_string(), "/elsewhere".to_string()), ("gdev/".to_string(), "/src/gdev/packages".to_string())]
        );
        assert_eq!(settings.std.as_deref(), Some("/src/blot2/std"));
        assert_eq!(settings.prelude.as_deref(), Some("/src/blot2/std/prelude.blot"));
        assert_eq!(settings.dependencies.as_deref(), Some("/src/gdev/deps.blotdep"));
        assert_eq!(settings.codegen_tier, Some(CodegenTier::Development));
        assert_eq!(settings.args(), ["serve-project", "--semantic-workers", "4"]);
        assert_eq!(
            settings.watched("/src/gdev/src/main.blot"),
            ["/src/gdev/src", "/elsewhere", "/src/gdev/packages", "/src/blot2/std", "/src/blot2/std/prelude.blot"]
        );
    }

    #[test]
    fn defaults_to_no_std_and_the_stds_prelude() {
        let none = Settings::parse("{}", "/p").unwrap();
        assert_eq!(none.program, PROGRAM);
        assert_eq!((&none.std, &none.prelude), (&None, &None));
        assert_eq!(none.args(), ["serve-project"]);
        let std = Settings::parse(r#"{"std": "std"}"#, "/p").unwrap();
        assert_eq!(std.prelude.as_deref(), Some("/p/std/prelude.blot"));
        let without = Settings::parse(r#"{"std": "std", "prelude": null}"#, "/p").unwrap();
        assert_eq!(without.prelude, None);
    }

    #[test]
    fn refuses_what_it_does_not_know() {
        for (json, refusal) in [
            ("[]", "not a JSON object"),
            (r#"{"stdRoot": "x"}"#, "no `stdRoot`"),
            (r#"{"std": 3}"#, "`std` is not a path"),
            (r#"{"std": ""}"#, "`std` is not a path"),
            (r#"{"imports": {"gdev": "packages"}}"#, "does not end in /"),
            (r#"{"codegenTier": "fast"}"#, "`codegenTier`"),
            (r#"{"semanticWorkers": 17}"#, "`semanticWorkers`"),
            (r#"{"expectedIdentity": "ABC"}"#, "`expectedIdentity`"),
            (r#"{"program": ""}"#, "`program`"),
        ] {
            let error = Settings::parse(json, "/p").unwrap_err();
            assert!(error.contains(refusal), "{json}: {error}");
        }
    }
}
