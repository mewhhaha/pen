//! A checkout's Blot tests (gdev's `tests/README.md`): which modules there
//! are, how each compiles, and what its tests must do. pen builds each and
//! runs its tests as Wasm. Every module compiles as if it stood at
//! `test_fixture.blot` beside the program's entry, so that it imports what
//! the program's own sources do; its directory says what its tests must do:
//!
//! - `tests/*.blot`: every `test_…` entry returns the report of its checks.
//! - `tests/traps/*.blot`: every `test_…` entry must stop at `@panic`, which
//!   is Wasm's `unreachable` (a decoder refusing bad input).
//! - `tests/rejected/*.blot`: the module must not compile, and the
//!   compiler's diagnostics must contain the text of its `// rejects: TEXT`
//!   line, among the comment lines it opens with.
//!
//! `settings.tests.entry` overrides the shared virtual entry, and
//! `settings.tests.files` overrides discovery when no CLI paths are given.
//! A library can keep relative imports beside its tests without a game entry.
//!
//! Never the other directories under `tests/`: `tests/blocked/` (tests
//! waiting on compiler fixes), and the modules the tests import.
use crate::settings::{Settings, parent, resolve};
use crate::{Overlay, Program};

/// Where the test modules are, in the checkout.
const TESTS: &str = "tests";
/// The directories under it whose modules' tests must trap, and whose
/// modules must not compile.
const TRAPS: &str = "traps";
const REJECTED: &str = "rejected";
/// What a module's file name ends with.
const EXTENSION: &str = ".blot";
/// What each module compiles as, beside the program's entry.
const FIXTURE: &str = "test_fixture.blot";
/// What the names of the entries that are tests start with.
pub const PREFIX: &str = "test_";
/// What the line of a module that must not compile starts with, before the
/// text its diagnostics must contain.
const REJECTS: &str = "// rejects:";

/// The checkout's files, as the plugin reads them: through pen's `files`
/// in the component, from the disk natively.
pub trait Files {
    fn read(&mut self, path: &str) -> Result<Vec<u8>, String>;
    /// The names in a directory, sorted.
    fn read_directory(&mut self, directory: &str) -> Result<Vec<String>, String>;
    fn is_file(&mut self, path: &str) -> bool;
}

/// What a module's tests must do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expectation {
    /// Each returns a report whose checks pass.
    Checks,
    /// Each traps at `unreachable`.
    Traps,
    /// The module does not compile, with diagnostics that contain this.
    Rejected(String),
}

/// A test module: what it builds as, and what its tests must do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestModule {
    /// Its file, absolute.
    pub path: String,
    /// The entry it compiles as.
    pub entry: String,
    /// Its source, standing in at the entry.
    pub overlays: Vec<Overlay>,
    /// What its tests' names start with.
    pub prefix: String,
    pub expect: Expectation,
}

/// The test modules of `program`'s checkout to run, in order: those at
/// `paths`, or every one in `tests/`, then `tests/traps/` and
/// `tests/rejected/` (each directory's sorted); less those none of whose
/// tests can be named by `filter`. A module's tests' names are in its
/// source, and one that must not compile is named for its file. One that
/// must not compile has to say what its diagnostics contain.
pub fn find(
    program: &Program,
    paths: &[String],
    filter: Option<&str>,
    files: &mut impl Files,
) -> Result<Vec<TestModule>, String> {
    let settings = Settings::parse(&program.settings, &program.root)?;
    let paths = if paths.is_empty() {
        match settings.tests.files {
            Some(paths) => paths,
            None => modules(&program.root, files)?,
        }
    } else {
        paths.iter().map(|path| resolve(&program.root, path)).collect()
    };
    let entry =
        settings.tests.entry.unwrap_or_else(|| resolve(&parent(&resolve(&program.root, &program.entry)), FIXTURE));
    let mut found = Vec::new();
    for path in paths {
        let source = files.read(&path).map_err(|error| format!("reading {path}: {error}"))?;
        let source = String::from_utf8(source).map_err(|_| format!("{path} is not UTF-8"))?;
        let named = match kind(&path) {
            Kind::Rejected => stem(&path).to_owned(),
            Kind::Checks | Kind::Traps => source.clone(),
        };
        if filter.is_some_and(|filter| !named.contains(filter)) {
            continue;
        }
        let expect = match kind(&path) {
            Kind::Checks => Expectation::Checks,
            Kind::Traps => Expectation::Traps,
            Kind::Rejected => match rejects(&source) {
                Some(text) => Expectation::Rejected(text.to_owned()),
                None => {
                    let shown = path.strip_prefix(&format!("{}/", program.root.trim_end_matches('/'))).unwrap_or(&path);
                    return Err(format!(
                        "{shown} must not compile, but has no `{REJECTS} TEXT` line among the comment lines it opens \
                         with, to say what the compiler's diagnostics must contain"
                    ));
                }
            },
        };
        // A module may keep expected panics beside its ordinary checks.
        // Pen runs each export in a fresh instance, including after a trap.
        let inline_traps = kind(&path) == Kind::Checks
            && source.lines().any(|line| {
                line.trim_start().strip_prefix("entry const ").is_some_and(|name| name.starts_with("trap_"))
            });
        let overlays = vec![Overlay { path: entry.clone(), contents: Some(source) }];
        found.push(TestModule {
            path: path.clone(),
            entry: entry.clone(),
            overlays: overlays.clone(),
            prefix: PREFIX.to_owned(),
            expect,
        });
        if inline_traps {
            found.push(TestModule {
                path,
                entry: entry.clone(),
                overlays,
                prefix: "trap_".to_owned(),
                expect: Expectation::Traps,
            });
        }
    }
    Ok(found)
}

/// What a module's tests must do, by its directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Checks,
    Traps,
    Rejected,
}

/// A module's kind, by its directory's name.
fn kind(path: &str) -> Kind {
    match parent(path).rsplit('/').next() {
        Some(TRAPS) => Kind::Traps,
        Some(REJECTED) => Kind::Rejected,
        _ => Kind::Checks,
    }
}

/// A file's name without its extension.
fn stem(path: &str) -> &str {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.strip_suffix(EXTENSION).unwrap_or(name)
}

/// The checkout's modules: `tests/*.blot`, then `tests/traps/*.blot` and
/// `tests/rejected/*.blot`, each directory's sorted.
fn modules(root: &str, files: &mut impl Files) -> Result<Vec<String>, String> {
    let tests = resolve(root, TESTS);
    let mut found = blot_files(&tests, files).map_err(|error| format!("reading {tests}: {error}"))?;
    for directory in [TRAPS, REJECTED].map(|name| resolve(&tests, name)) {
        // Either may be missing.
        found.extend(blot_files(&directory, files).unwrap_or_default());
    }
    Ok(found)
}

/// The `.blot` files in `directory`, sorted.
fn blot_files(directory: &str, files: &mut impl Files) -> Result<Vec<String>, String> {
    let mut found: Vec<String> = files
        .read_directory(directory)?
        .into_iter()
        .filter(|name| name.ends_with(EXTENSION))
        .map(|name| resolve(directory, &name))
        .filter(|path| files.is_file(path))
        .collect();
    found.sort();
    Ok(found)
}

/// What a module that must not compile has its diagnostics contain: the
/// text of its `// rejects: TEXT` line, among the comment lines it opens
/// with.
fn rejects(source: &str) -> Option<&str> {
    source
        .lines()
        .take_while(|line| line.starts_with("//"))
        .find_map(|line| line.strip_prefix(REJECTS))
        .map(str::trim)
        .filter(|text| !text.is_empty())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// A checkout's files in memory.
    struct Memory(BTreeMap<String, String>);

    impl Files for Memory {
        fn read(&mut self, path: &str) -> Result<Vec<u8>, String> {
            self.0.get(path).map(|source| source.clone().into_bytes()).ok_or_else(|| "not found".to_owned())
        }

        fn read_directory(&mut self, directory: &str) -> Result<Vec<String>, String> {
            let inside = format!("{directory}/");
            let mut names: Vec<String> = self
                .0
                .keys()
                .filter_map(|path| path.strip_prefix(&inside))
                .map(|rest| rest.split('/').next().unwrap_or(rest).to_owned())
                .collect();
            names.dedup();
            if names.is_empty() { Err("not found".to_owned()) } else { Ok(names) }
        }

        fn is_file(&mut self, path: &str) -> bool {
            self.0.contains_key(path)
        }
    }

    fn checkout(files: &[(&str, &str)]) -> Memory {
        Memory(files.iter().map(|(path, source)| (format!("/gdev/{path}"), source.to_string())).collect())
    }

    fn program() -> Program {
        Program { root: "/gdev".into(), entry: "/gdev/src/main.blot".into(), settings: "{}".into() }
    }

    const SAYS: &str = "// rejects: what test_d says\nentry const test_d = func () => return 0\n";
    const TEST: &str = "entry const test_d = func () => return 0\n";

    fn layout() -> Memory {
        checkout(&[
            ("tests/b.blot", TEST),
            ("tests/a.blot", TEST),
            ("tests/notes.md", TEST),
            ("tests/traps/z.blot", TEST),
            ("tests/traps/c.blot", TEST),
            ("tests/rejected/r.blot", SAYS),
            ("tests/blocked/x.blot", TEST),
            ("tests/codec_support/s.blot", TEST),
            ("tests/support/check.blot", TEST),
        ])
    }

    fn paths(modules: &[TestModule]) -> Vec<&str> {
        modules.iter().map(|module| module.path.strip_prefix("/gdev/tests/").unwrap()).collect()
    }

    #[test]
    fn finds_the_checkouts_modules_in_order() {
        let modules = find(&program(), &[], None, &mut layout()).unwrap();
        // Each directory's sorted, in this order; never blocked/ nor the
        // modules the tests import.
        assert_eq!(paths(&modules), ["a.blot", "b.blot", "traps/c.blot", "traps/z.blot", "rejected/r.blot"]);
        let expected: Vec<&Expectation> = modules.iter().map(|module| &module.expect).collect();
        let rejected = Expectation::Rejected("what test_d says".into());
        assert_eq!(
            expected,
            [&Expectation::Checks, &Expectation::Checks, &Expectation::Traps, &Expectation::Traps, &rejected]
        );
        // Each compiles beside the program's entry, its source standing in.
        let a = &modules[0];
        assert_eq!((a.entry.as_str(), a.prefix.as_str()), ("/gdev/src/test_fixture.blot", PREFIX));
        assert_eq!(a.overlays, [Overlay { path: a.entry.clone(), contents: Some(TEST.into()) }]);
        // Without traps/ and rejected/.
        let mut only = checkout(&[("tests/a.blot", TEST)]);
        assert_eq!(paths(&find(&program(), &[], None, &mut only).unwrap()), ["a.blot"]);
        let error = find(&program(), &[], None, &mut checkout(&[])).unwrap_err();
        assert!(error.starts_with("reading /gdev/tests: "), "{error}");
    }

    #[test]
    fn library_layout_selects_modules_and_preserves_a_shared_entry() {
        let mut program = program();
        program.settings = r#"{"tests": {
            "entry": "tests/.fixture.blot",
            "files": ["tests/b.blot", "tests/a.blot"]
        }}"#
        .into();
        let mut files = layout();
        let modules = find(&program, &[], None, &mut files).unwrap();
        assert_eq!(paths(&modules), ["b.blot", "a.blot"]);
        for module in &modules {
            assert_eq!(module.entry, "/gdev/tests/.fixture.blot");
            assert_eq!(module.overlays, [Overlay { path: module.entry.clone(), contents: Some(TEST.into()) }]);
        }
        let given = ["tests/traps/c.blot".to_owned()];
        let modules = find(&program, &given, Some("test_d"), &mut files).unwrap();
        assert_eq!(paths(&modules), ["traps/c.blot"]);
        assert_eq!(modules[0].expect, Expectation::Traps);
        assert!(find(&program, &given, Some("missing"), &mut files).unwrap().is_empty());
        // An explicit empty suite is not the discovery default.
        program.settings = r#"{"tests":{"files":[]}}"#.into();
        assert!(find(&program, &[], None, &mut files).unwrap().is_empty());
    }

    #[test]
    fn expected_panics_can_share_the_check_fixture() {
        let mut files = checkout(&[(
            "tests/a.blot",
            "entry const test_ok = func () => return #[1.0,0.0]\nentry const trap_bad = func () => return @panic \"bad\"\n",
        )]);
        let found = find(&program(), &[], None, &mut files).unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].expect, Expectation::Checks);
        assert_eq!(found[1].expect, Expectation::Traps);
        assert_eq!(found[1].prefix, "trap_");
        assert_eq!(found[0].entry, found[1].entry);
        assert_eq!(found[0].overlays, found[1].overlays);
    }

    #[test]
    fn finds_the_modules_it_is_given() {
        let given = ["tests/rejected/r.blot".to_owned(), "/gdev/tests/traps/z.blot".to_owned()];
        let modules = find(&program(), &given, None, &mut layout()).unwrap();
        assert_eq!(paths(&modules), ["rejected/r.blot", "traps/z.blot"]);
        let error = find(&program(), &["tests/gone.blot".to_owned()], None, &mut layout()).unwrap_err();
        assert_eq!(error, "reading /gdev/tests/gone.blot: not found");
    }

    #[test]
    fn filters_by_the_names_of_its_tests() {
        // The names of a module's tests are in its source, and one that must
        // not compile is named for its file.
        let mut files = checkout(&[
            ("tests/a.blot", TEST),
            ("tests/b.blot", "entry const test_e = func () => return 0\n"),
            ("tests/rejected/twice.blot", SAYS),
        ]);
        assert_eq!(paths(&find(&program(), &[], Some("test_d"), &mut files).unwrap()), ["a.blot"]);
        assert_eq!(paths(&find(&program(), &[], Some("twice"), &mut files).unwrap()), ["rejected/twice.blot"]);
        assert!(find(&program(), &[], Some("nothing"), &mut files).unwrap().is_empty());
    }

    #[test]
    fn needs_a_module_that_must_not_compile_to_say_why() {
        let mut files = checkout(&[("tests/rejected/unsaid.blot", TEST)]);
        let error = find(&program(), &[], None, &mut files).unwrap_err();
        assert!(error.starts_with("tests/rejected/unsaid.blot must not compile, but has no `// rejects: TEXT` line"));
        // Unless the filter leaves it out.
        assert!(find(&program(), &[], Some("test_d"), &mut files).unwrap().is_empty());
    }

    #[test]
    fn reads_what_a_module_must_be_rejected_for() {
        let ecs = "// rejects: component type already registered\n// A component type registered twice\nimport * as ecs from \"gdev/ecs\"\n";
        assert_eq!(rejects(ecs), Some("component type already registered"));
        assert_eq!(rejects("// A fixed update\r\n// rejects: type_mismatch \r\n"), Some("type_mismatch"));
        assert_eq!(rejects("// Why it does not compile\n// rejects: x y\nconst a = 1\n"), Some("x y"));
        // Only among the comment lines it opens with, and never empty.
        assert_eq!(rejects("// A note\nconst a = 1\n// rejects: x\n"), None);
        assert_eq!(rejects("\n// rejects: x\n"), None);
        assert_eq!(rejects("// rejects:\n"), None);
        assert_eq!(rejects("// rejects:  \n// rejects: x\n"), None);
        assert_eq!(rejects("// rejected: x\n"), None);
        assert_eq!(rejects(""), None);
    }

    #[test]
    fn tells_kinds_by_their_directories() {
        assert_eq!(kind("/gdev/tests/a.blot"), Kind::Checks);
        assert_eq!(kind("/gdev/tests/traps/a.blot"), Kind::Traps);
        assert_eq!(kind("/gdev/tests/rejected/a.blot"), Kind::Rejected);
        assert_eq!(kind("/elsewhere/rejected/a.blot"), Kind::Rejected);
        assert_eq!(stem("/gdev/tests/rejected/ecs_twice.blot"), "ecs_twice");
    }
}
