//! The plugin against today's TypeScript, gdev's asset_parsers.ts over
//! blot2's compiler/assets.ts, as tests/fixtures/expected.ts captured it:
//! the same module paths and byte-identical sources for gdev's two programs,
//! and for each made-up checkout of cases.json the same modules or the same
//! error. When gdev's files or the TypeScript change, regenerate the fixtures
//! with the command in expected.ts, then run these again (and the ignored
//! `gdev_on_disk_matches_fixtures` against the checkout itself).
mod common;

use common::{Disk, Memory, expected_dependencies, expected_modules, fixture};
use pen_plugin_gdev_assets::{Files, Module, Program, generate};
use serde_json::Value;

/// Panics with the first difference between `actual` and `expected`.
fn same_modules(what: &str, actual: &[Module], expected: &[Module]) {
    let paths = |modules: &[Module]| modules.iter().map(|module| module.path.clone()).collect::<Vec<_>>();
    assert_eq!(paths(actual), paths(expected), "{what}: module paths");
    for (actual, expected) in actual.iter().zip(expected) {
        if actual.source != expected.source {
            let line = actual.source.lines().zip(expected.source.lines()).position(|(a, e)| a != e);
            panic!(
                "{what}: {} differs at line {}:\n  plugin:     {:?}\n  TypeScript: {:?}",
                actual.path,
                line.map_or(0, |line| line + 1),
                line.and_then(|line| actual.source.lines().nth(line)).unwrap_or(&actual.source),
                line.and_then(|line| expected.source.lines().nth(line)).unwrap_or(&expected.source),
            );
        }
    }
}

/// gdev's checkout as programs.json holds it.
fn snapshot(fixture: &Value) -> Memory {
    let root = fixture["root"].as_str().unwrap();
    let files = fixture["files"].as_object().unwrap();
    Memory::new(root, files.iter().map(|(path, contents)| (path.as_str(), contents.as_str().unwrap())))
}

/// Checks each of gdev's programs against the TypeScript's modules, and its
/// dependencies: the files it read and the directories whose new files would
/// change it.
fn check_programs(fixture: &Value, files: &dyn Files) {
    let root = fixture["root"].as_str().unwrap();
    let programs = fixture["programs"].as_array().unwrap();
    assert_eq!(programs.len(), 2);
    for expected in programs {
        let name = expected["name"].as_str().unwrap();
        let program = Program {
            root: root.to_string(),
            entry: expected["entry"].as_str().unwrap().to_string(),
            settings: "{}".to_string(),
        };
        let generated = generate(&program, files).unwrap_or_else(|error| panic!("{name}: {error}"));
        same_modules(name, &generated.modules, &expected_modules(expected));
        let mut dependencies = expected_dependencies(expected);
        // The directories of shaders, and for devtools, its own, where a
        // catalog, words, tunables or shaders of its own would appear.
        dependencies.push(format!("{root}/src/assets"));
        if name == "devtools/main.blot" {
            dependencies.push(format!("{root}/devtools"));
        }
        dependencies.sort();
        assert_eq!(generated.dependencies, dependencies, "{name}: dependencies");
    }
}

#[test]
fn gdev_programs_match_typescript() {
    let fixture = fixture("programs.json");
    check_programs(&fixture, &snapshot(&fixture));
}

/// The same against gdev's checkout itself, which another session may be
/// changing: regenerate the fixtures first.
#[test]
#[ignore = "reads ~/src/gdev; regenerate the fixtures first"]
fn gdev_on_disk_matches_fixtures() {
    check_programs(&fixture("programs.json"), &Disk);
}

#[test]
fn cases_match_typescript() {
    let cases = fixture("cases.json");
    let mut compared = 0;
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let root = case["root"].as_str().unwrap();
        let files = case["files"].as_object().unwrap();
        let memory =
            Memory::new(root, files.iter().map(|(path, contents)| (path.as_str(), contents.as_str().unwrap())));
        let program = Program {
            root: root.to_string(),
            entry: case["entry"].as_str().unwrap().to_string(),
            settings: "{}".to_string(),
        };
        let generated = generate(&program, &memory);
        let differs = case.get("differs").is_some();
        if let Some(error) = case.get("error") {
            let file = error["file"].as_str().unwrap();
            let message = error["message"].as_str().unwrap();
            // The shader parser's message began with its file's path.
            let message = message.strip_prefix(&format!("{file}: ")).unwrap_or(message);
            let actual = generated.expect_err(name);
            assert_eq!(actual.file, file, "{name}: file");
            if !differs {
                assert_eq!(actual.message, message, "{name}: message");
                compared += 1;
            }
        } else if !differs {
            let generated = generated.unwrap_or_else(|error| panic!("{name}: {error}"));
            same_modules(name, &generated.modules, &expected_modules(case));
            for dependency in expected_dependencies(case) {
                assert!(generated.dependencies.contains(&dependency), "{name}: depends on {dependency}");
            }
            compared += 1;
        }
        // Where the plugin's modules differ, tests/parsers.rs says what they
        // are.
    }
    assert!(compared >= 40, "only {compared} cases compared");
}
