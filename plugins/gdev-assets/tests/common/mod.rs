//! What the tests share: checkouts held in memory or read from disk, and the
//! fixtures tests/fixtures/expected.ts made with today's TypeScript.
#![allow(dead_code)]
use std::collections::{BTreeMap, BTreeSet};

use pen_plugin_gdev_assets::{Files, Generated, Module, Program, generate};
use serde_json::Value;

/// A checkout in memory: files by absolute path, and the directories they
/// are in.
pub struct Memory {
    files: BTreeMap<String, Vec<u8>>,
    directories: BTreeSet<String>,
}

impl Memory {
    /// The files `relative` (paths from `root`) holds, under `root`.
    pub fn new<'a>(root: &str, relative: impl IntoIterator<Item = (&'a str, &'a str)>) -> Self {
        let mut memory = Memory { files: BTreeMap::new(), directories: BTreeSet::new() };
        for (path, contents) in relative {
            let path = format!("{root}/{path}");
            let mut directory = path.as_str();
            while let Some(end) = directory.rfind('/') {
                directory = &directory[..end];
                memory.directories.insert(if directory.is_empty() { "/".to_string() } else { directory.to_string() });
            }
            memory.files.insert(path, contents.as_bytes().to_vec());
        }
        memory
    }
}

impl Files for Memory {
    fn read(&self, path: &str) -> Result<Vec<u8>, String> {
        self.files.get(path).cloned().ok_or_else(|| format!("No such file: {path}"))
    }

    fn read_directory(&self, directory: &str) -> Result<Vec<String>, String> {
        if !self.directories.contains(directory) {
            return Err(format!("No such directory: {directory}"));
        }
        let prefix = if directory == "/" { "/".to_string() } else { format!("{directory}/") };
        let names: BTreeSet<String> = self
            .files
            .keys()
            .chain(&self.directories)
            .filter_map(|path| path.strip_prefix(&prefix))
            .filter_map(|rest| rest.split('/').next())
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .collect();
        Ok(names.into_iter().collect())
    }

    fn exists(&self, path: &str) -> bool {
        self.files.contains_key(path) || self.directories.contains(path)
    }
}

/// The checkout on disk.
pub struct Disk;

impl Files for Disk {
    fn read(&self, path: &str) -> Result<Vec<u8>, String> {
        std::fs::read(path).map_err(|error| format!("{path}: {error}"))
    }

    fn read_directory(&self, directory: &str) -> Result<Vec<String>, String> {
        let mut names = std::fs::read_dir(directory)
            .and_then(|entries| {
                entries
                    .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
                    .collect::<Result<Vec<String>, _>>()
            })
            .map_err(|error| format!("{directory}: {error}"))?;
        names.sort();
        Ok(names)
    }

    fn exists(&self, path: &str) -> bool {
        std::path::Path::new(path).exists()
    }
}

/// A program with no settings.
pub fn program(root: &str, entry: &str) -> Program {
    Program { root: root.to_string(), entry: format!("{root}/{entry}"), settings: "{}".to_string() }
}

/// The modules of the program at `entry` in a checkout of `files`, under
/// `/checkout`.
pub fn modules(entry: &str, files: &[(&str, &str)]) -> Result<Generated, pen_plugin_gdev_assets::Error> {
    let memory = Memory::new("/checkout", files.iter().copied());
    generate(&program("/checkout", entry), &memory)
}

/// The source of the module at `path`.
pub fn source<'a>(generated: &'a Generated, path: &str) -> &'a str {
    &generated.modules.iter().find(|module| module.path == path).unwrap_or_else(|| panic!("no module {path}")).source
}

/// A fixture of tests/fixtures.
pub fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{path}: {error}")))
        .unwrap_or_else(|error| panic!("{path}: {error}"))
}

/// A fixture's expected modules.
pub fn expected_modules(expected: &Value) -> Vec<Module> {
    expected["modules"]
        .as_array()
        .expect("modules")
        .iter()
        .map(|module| Module {
            path: module["path"].as_str().unwrap().to_string(),
            source: module["source"].as_str().unwrap().to_string(),
        })
        .collect()
}

/// A fixture's expected dependencies (the files assets.ts read).
pub fn expected_dependencies(expected: &Value) -> Vec<String> {
    expected["dependencies"].as_array().expect("dependencies").iter().map(|d| d.as_str().unwrap().to_string()).collect()
}
