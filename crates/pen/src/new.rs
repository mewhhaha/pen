//! `pen new DIR`: a game to start from, which compiles and runs as it is. It
//! has what pen needs of any game and nothing of gdev's: `pen.json` (its
//! name, and the Blot compiler and standard library it builds with), the
//! program (`src/main.blot`, `src/game.blot`), a catalog with a mesh, a
//! material and its shader, a test, and `pen/`, the bindings `pen codegen`
//! generates. The templates are `templates/new/`.
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

use crate::project::Identity;

/// The scaffold's files by their paths in the checkout; `${TITLE}` in them
/// is the game's title.
const FILES: [(&str, &str); 9] = [
    (".gitignore", include_str!("../templates/new/.gitignore")),
    ("README.md", include_str!("../templates/new/README.md")),
    ("src/main.blot", include_str!("../templates/new/src/main.blot")),
    ("src/game.blot", include_str!("../templates/new/src/game.blot")),
    ("src/assets/catalog.json", include_str!("../templates/new/src/assets/catalog.json")),
    ("src/assets/cube.mesh.json", include_str!("../templates/new/src/assets/cube.mesh.json")),
    ("src/assets/cube.material.json", include_str!("../templates/new/src/assets/cube.material.json")),
    ("src/assets/lit.wgsl", include_str!("../templates/new/src/assets/lit.wgsl")),
    ("tests/game.blot", include_str!("../templates/new/tests/game.blot")),
];

/// Whether `directory` is a Blot checkout a game can build with: its
/// standard library and its packed compiler.
fn is_blot(directory: &Path) -> bool {
    directory.join("std/prelude.blot").is_file() && directory.join("generated/compiler/blotc.gz").is_file()
}

/// The Blot checkout a game at `root` builds with: `given`, else
/// `PEN_BLOT`, else a `blot2` beside `root` or beside one of the
/// directories above it, else the one beside the pen checkout this was
/// built from.
fn find_blot(root: &Path, given: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(given) = given.or_else(|| std::env::var_os("PEN_BLOT").map(PathBuf::from)) {
        ensure!(
            is_blot(&given),
            "{} is not a Blot checkout with std/prelude.blot and generated/compiler/blotc.gz",
            given.display()
        );
        return absolute(&given);
    }
    let mut looked = Vec::new();
    let built_beside = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../blot2");
    for candidate in root.ancestors().skip(1).map(|directory| directory.join("blot2")).chain([built_beside]) {
        if is_blot(&candidate) {
            return absolute(&candidate);
        }
        looked.push(candidate.display().to_string());
    }
    bail!(
        "found no Blot checkout (blot2) to build with: pass --blot DIR or set PEN_BLOT (looked at {})",
        looked.join(", ")
    )
}

/// `path` from the root, its `.` and `..` worked out from its words alone,
/// so that a link in it stays as it was named.
fn absolute(path: &Path) -> Result<PathBuf> {
    let mut whole = PathBuf::new();
    for component in std::path::absolute(path)?.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                whole.pop();
            }
            other => whole.push(other),
        }
    }
    Ok(whole)
}

/// `to` from the directory `from`, both absolute, with `/` between its
/// parts; `to` itself when they share no root (two Windows drives).
fn relative(from: &Path, to: &Path) -> String {
    let from: Vec<Component> = from.components().collect();
    let to: Vec<Component> = to.components().collect();
    let shared = from.iter().zip(&to).take_while(|(a, b)| a == b).count();
    if shared == 0 {
        return to.iter().collect::<PathBuf>().to_string_lossy().replace('\\', "/");
    }
    let mut parts: Vec<String> = from[shared..].iter().map(|_| "..".to_owned()).collect();
    parts.extend(to[shared..].iter().map(|part| part.as_os_str().to_string_lossy().into_owned()));
    if parts.is_empty() { ".".to_owned() } else { parts.join("/") }
}

/// The checkout's `pen.json`: its game, and the Blot plugin with the
/// compiler and standard library at `blot` (a path from the checkout).
fn config(identity: &Identity, blot: &str) -> String {
    let text = |value: &str| serde_json::Value::String(value.to_owned()).to_string();
    format!(
        r#"{{
  "game": {{ "name": {name}, "title": {title} }},
  "plugins": {{
    "blot": {{
      "read": [{std}],
      "run": {{ "blotc": {blotc} }}
    }}
  }},
  "compiler": {{
    "plugin": "blot",
    "settings": {{
      "std": {std},
      "imports": {{ "pen/": "pen/" }},
      "nativeAssets": {{ "roots": ["src"] }}
    }}
  }}
}}
"#,
        name = text(&identity.name),
        title = text(&identity.title),
        std = text(&format!("{blot}/std")),
        blotc = text(&format!("{blot}/generated/compiler/blotc.gz")),
    )
}

/// Makes the game in `directory`, which must be empty or not yet there,
/// named after it, building with the Blot checkout `blot` (else one found
/// as [`find_blot`] says); the checkout's root.
pub fn create(directory: &Path, blot: Option<PathBuf>) -> Result<PathBuf> {
    if directory.exists() {
        let mut entries = std::fs::read_dir(directory).with_context(|| format!("reading {}", directory.display()))?;
        ensure!(
            entries.next().is_none(),
            "{} is not empty: pen new makes a game in a new directory",
            directory.display()
        );
    }
    let root = absolute(directory)?;
    let name = root.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
    let identity = Identity::named(&name);
    identity.check()?;
    let blot = find_blot(&root, blot)?;
    std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
    std::fs::write(root.join("pen.json"), config(&identity, &relative(&root, &blot)))?;
    for (path, contents) in FILES {
        let target = root.join(path);
        std::fs::create_dir_all(target.parent().context("a template has no directory")?)?;
        std::fs::write(&target, contents.replace("${TITLE}", &identity.title))?;
    }
    pen_codegen::generate(&root, None, false)?;
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let directory = crate::project::temporary_dir().unwrap().join(format!("new-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    /// A stand-in Blot checkout: the two files `is_blot` looks for.
    fn fake_blot(at: &Path) -> PathBuf {
        let blot = at.join("blot2");
        std::fs::create_dir_all(blot.join("std")).unwrap();
        std::fs::create_dir_all(blot.join("generated/compiler")).unwrap();
        std::fs::write(blot.join("std/prelude.blot"), "").unwrap();
        std::fs::write(blot.join("generated/compiler/blotc.gz"), "").unwrap();
        blot
    }

    #[test]
    fn paths_between_directories() {
        assert_eq!(absolute(Path::new("/a/b/../c/./d")).unwrap(), Path::new("/a/c/d"));
        assert_eq!(relative(Path::new("/a/b/game"), Path::new("/a/blot2")), "../../blot2");
        assert_eq!(relative(Path::new("/a/game"), Path::new("/a/game/blot2")), "blot2");
        assert_eq!(relative(Path::new("/a"), Path::new("/a")), ".");
    }

    #[test]
    fn scaffolds_a_checkout_beside_its_blot() {
        let scratch = scratch("scaffold");
        fake_blot(&scratch);
        let root = create(&scratch.join("Rolling Ball"), None).unwrap();
        // pen finds it as a checkout, named after its directory.
        let project = crate::project::Project::find(Some(root.join("src"))).unwrap();
        assert_eq!(project.root, root.canonicalize().unwrap());
        assert_eq!((project.identity.name.as_str(), project.identity.title.as_str()), ("Rolling Ball", "Rolling Ball"));
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("pen.json")).unwrap()).unwrap();
        assert_eq!(config["compiler"]["settings"]["std"], "../blot2/std");
        assert_eq!(config["plugins"]["blot"]["run"]["blotc"], "../blot2/generated/compiler/blotc.gz");
        for (path, _) in FILES {
            assert!(root.join(path).is_file(), "{path}");
        }
        assert!(std::fs::read_to_string(root.join("README.md")).unwrap().starts_with("# Rolling Ball\n"));
        for binding in ["program.blot", "frame.blot", "check.blot", "protocol.blot", "types.blot"] {
            assert!(root.join("pen").join(binding).is_file(), "{binding}");
        }
        // Its bindings are fresh, and it has no declarations of its own.
        pen_codegen::generate(&root, None, true).unwrap();
        assert!(!root.join("pen/extensions.blot").exists());
        // Its catalog reads as the renderer reads it.
        let catalog = pen_protocol::boundary::read_catalog(&root.join("src/assets")).unwrap();
        assert_eq!((catalog.meshes.len(), catalog.materials.len()), (1, 1));
        // Not over a game that is there.
        let refused = create(&root, None).unwrap_err().to_string();
        assert!(refused.contains("is not empty"), "{refused}");
        // Nor with a Blot checkout that is not one.
        let refused = create(&scratch.join("other"), Some(scratch.join("nowhere"))).unwrap_err().to_string();
        assert!(refused.contains("is not a Blot checkout"), "{refused}");
        std::fs::remove_dir_all(&scratch).unwrap();
    }

    #[test]
    fn a_name_that_cannot_name_a_game_is_refused() {
        let scratch = scratch("names");
        fake_blot(&scratch);
        // Identity::named makes it a file name; a reserved one is refused.
        assert!(create(&scratch.join("con"), None).is_err());
        std::fs::remove_dir_all(&scratch).unwrap();
    }
}
