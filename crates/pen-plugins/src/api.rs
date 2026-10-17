//! The plugin API's version: `wit/pen-plugin.wit` is the WIT package
//! `pen:plugin@VERSION`, and a component names that version in every
//! interface it imports from pen or exports to it. [`check`] refuses one
//! built against a version this pen does not host before linking it, saying
//! which; `wit/CHANGELOG.md` records each version, what changed, and the
//! shape of its interfaces ([`shape`]), which a test holds the WIT to.
use std::collections::BTreeSet;
use std::fmt;

use anyhow::{Result, bail};
use sha2::{Digest, Sha256};
use wasmtime::Engine;
use wasmtime::component::Component;

/// The package's name.
pub const PACKAGE: &str = "pen:plugin";
/// The version this pen hosts: `wit/pen-plugin.wit`'s, which the bindings
/// are made from.
pub const VERSION: Version = Version { major: 0, minor: 2, patch: 0 };

/// A semantic version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

impl Version {
    /// `MAJOR.MINOR.PATCH`, without a pre-release or build.
    pub fn parse(text: &str) -> Option<Version> {
        let mut parts = text.split('.').map(|part| part.parse::<u64>().ok());
        let version = Version { major: parts.next()??, minor: parts.next()??, patch: parts.next()?? };
        parts.next().is_none().then_some(version)
    }

    /// Whether a plugin built against this version runs on a pen hosting
    /// `host`: semver's compatible versions (the same major version, or
    /// below 1.0 the same minor one, as wasmtime links them), and none newer
    /// than the host's, whose plugins may ask for what it lacks.
    pub fn runs_on(self, host: Version) -> bool {
        let track = |version: Version| if version.major == 0 { (0, version.minor) } else { (version.major, 0) };
        track(self) == track(host) && self <= host
    }
}

impl fmt::Display for Version {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// The versions of `pen:plugin` among a component's import and export
/// names (`pen:plugin/log@0.2.0`), and whether any name has none.
fn versions<'a>(names: impl IntoIterator<Item = &'a str>) -> (BTreeSet<Version>, bool) {
    let mut found = BTreeSet::new();
    let mut unversioned = false;
    for name in names {
        let Some(rest) = name.strip_prefix(PACKAGE).and_then(|rest| rest.strip_prefix('/')) else { continue };
        match rest.split_once('@').and_then(|(_, version)| Version::parse(version)) {
            Some(version) => {
                found.insert(version);
            }
            None => unversioned = true,
        }
    }
    (found, unversioned)
}

/// Whether a component whose imports and exports are `names` runs on this
/// pen, or why not; `name` is the plugin's, for the message.
fn check_names<'a>(name: &str, names: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let (found, unversioned) = versions(names);
    let rebuild = "rebuild it against wit/pen-plugin.wit (wit/CHANGELOG.md says what changed)";
    if unversioned {
        bail!("plugin {name} names {PACKAGE} without a version: {rebuild}");
    }
    let Some(&built) = found.last() else {
        bail!("plugin {name} is not a pen plugin: it imports and exports nothing of {PACKAGE}");
    };
    if found.len() > 1 {
        let all: Vec<String> = found.iter().map(Version::to_string).collect();
        bail!("plugin {name} was built against several versions of {PACKAGE} ({}): {rebuild}", all.join(", "));
    }
    if built.runs_on(VERSION) {
        return Ok(());
    }
    if built > VERSION {
        bail!(
            "plugin {name} was built for {PACKAGE}@{built}, newer than this pen's {VERSION}: update pen, or {rebuild}"
        );
    }
    bail!("plugin {name} was built for {PACKAGE}@{built}, and this pen hosts {VERSION}: {rebuild}")
}

/// Refuses `component` (plugin `name`'s) unless it was built against a
/// version of the API this pen hosts.
pub fn check(name: &str, engine: &Engine, component: &Component) -> Result<()> {
    let ty = component.component_type();
    let imports: Vec<String> = ty.imports(engine).map(|(name, _)| name.to_owned()).collect();
    let exports: Vec<String> = ty.exports(engine).map(|(name, _)| name.to_owned()).collect();
    check_names(name, imports.iter().chain(&exports).map(String::as_str))
}

/// The shape of a WIT document: its text without comments or blank lines,
/// each line trimmed, hashed (SHA-256, hex). Words in comments change no
/// plugin; anything else may.
pub fn shape(wit: &str) -> String {
    let mut hash = Sha256::new();
    for line in wit.lines().map(str::trim).filter(|line| !line.is_empty() && !line.starts_with("//")) {
        hash.update(line.as_bytes());
        hash.update(b"\n");
    }
    hash.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIT: &str = include_str!("../../../wit/pen-plugin.wit");
    const CHANGELOG: &str = include_str!("../../../wit/CHANGELOG.md");

    #[test]
    fn versions_run_on_compatible_hosts() {
        let v = |text| Version::parse(text).unwrap();
        assert_eq!(Version::parse("0.2.0.1"), None);
        assert_eq!(Version::parse("0.2"), None);
        assert_eq!(Version::parse("0.2.0-rc1"), None);
        assert!(v("0.2.0").runs_on(v("0.2.0")));
        assert!(v("0.2.0").runs_on(v("0.2.3")));
        assert!(!v("0.2.4").runs_on(v("0.2.3")), "newer than the host");
        assert!(!v("0.1.0").runs_on(v("0.2.0")), "below 1.0, a minor version breaks");
        assert!(v("1.1.0").runs_on(v("1.3.0")));
        assert!(!v("1.3.0").runs_on(v("1.1.0")));
        assert!(!v("1.0.0").runs_on(v("2.0.0")));
    }

    #[test]
    fn components_name_the_version_they_were_built_for() {
        let host = format!("{PACKAGE}/log@{VERSION}");
        let names = |list: &[&str]| list.iter().map(|name| name.to_string()).collect::<Vec<_>>();
        assert!(check_names("blot", ["wasi:io/streams@0.2.6", host.as_str()]).is_ok());
        let error =
            |list: &[&str]| check_names("blot", names(list).iter().map(String::as_str)).unwrap_err().to_string();
        let old = error(&["pen:plugin/log@0.1.0", "pen:plugin/compiler@0.1.0"]);
        assert_eq!(
            old,
            format!(
                "plugin blot was built for pen:plugin@0.1.0, and this pen hosts {VERSION}: rebuild it against \
                 wit/pen-plugin.wit (wit/CHANGELOG.md says what changed)"
            )
        );
        assert!(error(&["pen:plugin/log@9.0.0"]).contains("newer than this pen's"));
        assert!(error(&["wasi:io/streams@0.2.6"]).contains("is not a pen plugin"));
        assert!(error(&["pen:plugin/log"]).contains("without a version"));
        assert!(error(&["pen:plugin/log@0.1.0", "pen:plugin/compiler@0.2.0"]).contains("several versions"));
    }

    /// A component that imports `name`, an instance with nothing in it.
    fn importing(name: &str) -> Vec<u8> {
        use wasm_encoder::{ComponentImportSection, ComponentTypeRef, ComponentTypeSection, InstanceType};
        let mut component = wasm_encoder::Component::new();
        let mut types = ComponentTypeSection::new();
        types.instance(&InstanceType::new());
        component.section(&types);
        let mut imports = ComponentImportSection::new();
        imports.import(name, ComponentTypeRef::Instance(0));
        component.section(&imports);
        component.finish()
    }

    #[test]
    fn refuses_a_component_built_for_another_version() {
        let mut config = wasmtime::Config::new();
        config.wasm_component_model(true);
        let engine = Engine::new(&config).unwrap();
        let load = |name: &str| Component::new(&engine, importing(name)).unwrap();
        let refused = check("old", &engine, &load("pen:plugin/log@0.1.0")).unwrap_err().to_string();
        assert!(refused.starts_with("plugin old was built for pen:plugin@0.1.0, and this pen hosts"), "{refused}");
        check("current", &engine, &load(&format!("pen:plugin/log@{VERSION}"))).unwrap();
    }

    #[test]
    fn the_wit_is_the_version_its_changelog_records() {
        let package = WIT
            .lines()
            .find_map(|line| line.trim().strip_prefix("package "))
            .and_then(|rest| rest.strip_suffix(';'))
            .expect("the WIT names its package");
        assert_eq!(package, format!("{PACKAGE}@{VERSION}"), "api::VERSION is the WIT's");
        // The changelog's newest entry, its first heading that starts with
        // a version, is this version, with this shape.
        fn entry(line: &str) -> Option<&str> {
            line.strip_prefix("## ").filter(|title| title.starts_with(char::is_numeric))
        }
        let newest = CHANGELOG.lines().find_map(entry).expect("a version in the changelog");
        assert!(
            newest.starts_with(&format!("{VERSION} ")),
            "wit/CHANGELOG.md's newest entry is {newest}, not {VERSION}"
        );
        let recorded = CHANGELOG
            .lines()
            .skip_while(|line| entry(line).is_none())
            .find_map(|line| line.strip_prefix("Shape: `").and_then(|rest| rest.strip_suffix('`')))
            .expect("the newest entry records its shape");
        let shape = shape(WIT);
        assert_eq!(
            recorded, shape,
            "wit/pen-plugin.wit's interfaces changed but its version did not: give it a new version (a new minor \
             one below 1.0), and an entry in wit/CHANGELOG.md with what changed and `Shape: `{shape}``"
        );
    }

    #[test]
    fn shapes_ignore_comments() {
        assert_eq!(shape("/// docs\ninterface a {\n  f: func();\n}\n"), shape("interface a {\n    f: func();\n}"));
        assert_ne!(shape("interface a { f: func(); }"), shape("interface a { g: func(); }"));
    }
}
