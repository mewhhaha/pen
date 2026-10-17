//! Where a package's files go. Beside the runtime on Linux and Windows:
//! `gdev`, Steam's library and `game/` in one folder. On macOS, an
//! application bundle, since Finder, code signing and notarization want one:
//!
//! ```text
//! gdev.app/Contents/
//! ├─ Info.plist
//! ├─ MacOS/gdev, libsteam_api.dylib   (build.rs sets the rpath @executable_path)
//! └─ Resources/game/, LICENSE, THIRD_PARTY_NOTICES.md
//! ```
//!
//! The game lies in `Resources`, because anything but code in
//! `Contents/MacOS` breaks the bundle's signature; the runtime looks for it
//! there ([`packaged_game`]). Signing and notarizing the bundle are
//! `scripts/macos-sign.sh`'s, with the owner's Developer ID.
use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};

use crate::project::Identity;

/// The oldest macOS the bundle says it runs on: Big Sur, the first with
/// Apple silicon, and Rust's own minimum there.
const MINIMUM_MACOS: &str = "11.0";

/// The paths `pen build` writes a package's parts to.
#[derive(Debug, PartialEq)]
pub struct Layout {
    /// The runtime, named after the game.
    pub executable: PathBuf,
    /// Where the libraries the runtime links (Steam's) go: beside it.
    pub libraries: PathBuf,
    /// The bundle: game.cwasm, game.wasm, assets, the scene and manifest.
    pub game: PathBuf,
    /// The licence and notices.
    pub notices: PathBuf,
    /// A macOS application bundle's Info.plist and its text.
    pub info: Option<(PathBuf, String)>,
}

impl Layout {
    /// The layout under `out` of a package of `identity` for `target`
    /// (a Rust triple). `bundle_id` is a macOS bundle's CFBundleIdentifier,
    /// `pen.NAME` when not given.
    pub fn new(out: &Path, identity: &Identity, target: &str, bundle_id: Option<&str>) -> Result<Layout> {
        let name = &identity.name;
        if !target.contains("apple") {
            let executable = if target.contains("windows") { format!("{name}.exe") } else { name.clone() };
            return Ok(Layout {
                executable: out.join(executable),
                libraries: out.to_owned(),
                game: out.join("game"),
                notices: out.to_owned(),
                info: None,
            });
        }
        let contents = out.join(format!("{name}.app")).join("Contents");
        let identifier = bundle_identifier(name, bundle_id)?;
        Ok(Layout {
            executable: contents.join("MacOS").join(name),
            libraries: contents.join("MacOS"),
            game: contents.join("Resources/game"),
            notices: contents.join("Resources"),
            info: Some((contents.join("Info.plist"), info_plist(identity, &identifier))),
        })
    }

    /// Makes the directories, and writes Info.plist when there is one.
    pub fn create(&self) -> Result<()> {
        for directory in [&self.libraries, &self.game, &self.notices] {
            std::fs::create_dir_all(directory)?;
        }
        if let Some((path, text)) = &self.info {
            std::fs::write(path, text)?;
        }
        Ok(())
    }
}

/// A bundle identifier: `given`, which must be one (letters, digits, `-`
/// and `.`, as Apple allows), else `pen.` and the game's name made one.
pub fn bundle_identifier(name: &str, given: Option<&str>) -> Result<String> {
    let allowed = |character: char| character.is_ascii_alphanumeric() || character == '-' || character == '.';
    if let Some(given) = given {
        ensure!(
            !given.is_empty() && !given.starts_with('.') && !given.ends_with('.') && given.chars().all(allowed),
            "--bundle-id {given} is not a bundle identifier: use letters, digits, `-` and `.`, as in com.studio.game"
        );
        return Ok(given.to_owned());
    }
    let name: String = name.chars().map(|character| if allowed(character) { character } else { '-' }).collect();
    Ok(format!("pen.{}", name.trim_matches('.')))
}

/// `text` safe in an XML element.
fn escaped(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// What macOS reads of the bundle: its executable, names and identifier,
/// that it is an application and a game, the oldest system it runs on, and
/// that it draws at the display's full resolution (without this, a Retina
/// display would show it scaled up from half).
fn info_plist(identity: &Identity, identifier: &str) -> String {
    let name = escaped(&identity.name);
    let title = escaped(&identity.title);
    let identifier = escaped(identifier);
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleDisplayName</key>
	<string>{title}</string>
	<key>CFBundleExecutable</key>
	<string>{name}</string>
	<key>CFBundleIdentifier</key>
	<string>{identifier}</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleName</key>
	<string>{name}</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>1.0</string>
	<key>CFBundleVersion</key>
	<string>1</string>
	<key>LSApplicationCategoryType</key>
	<string>public.app-category.games</string>
	<key>LSMinimumSystemVersion</key>
	<string>{MINIMUM_MACOS}</string>
	<key>NSHighResolutionCapable</key>
	<true/>
</dict>
</plist>
"#
    )
}

/// The packaged game for the runtime at `exe`: `game/` beside it, or in a
/// macOS application bundle `Contents/Resources/game`.
pub fn packaged_game(exe: &Path) -> Option<PathBuf> {
    let beside = exe.parent()?;
    [beside.join("game"), beside.join("../Resources/game")]
        .into_iter()
        .find(|directory| directory.join("game.wasm").exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Identity {
        Identity { name: "gdev".into(), title: "gdev — Blot 3D & <friends>".into(), players: 1 }
    }

    #[test]
    fn packages_lay_out_by_platform() {
        let out = Path::new("dist/package");
        let linux = Layout::new(out, &identity(), "x86_64-unknown-linux-gnu", None).unwrap();
        assert_eq!((linux.executable, linux.game, linux.info), (out.join("gdev"), out.join("game"), None));
        let windows = Layout::new(out, &identity(), "x86_64-pc-windows-gnu", None).unwrap();
        assert_eq!((windows.executable, windows.libraries), (out.join("gdev.exe"), out.to_owned()));
        let mac = Layout::new(out, &identity(), "aarch64-apple-darwin", Some("com.studio.gdev")).unwrap();
        let contents = out.join("gdev.app/Contents");
        assert_eq!(mac.executable, contents.join("MacOS/gdev"));
        assert_eq!(mac.libraries, contents.join("MacOS"));
        assert_eq!(mac.game, contents.join("Resources/game"));
        assert_eq!(mac.notices, contents.join("Resources"));
        let (path, text) = mac.info.unwrap();
        assert_eq!(path, contents.join("Info.plist"));
        for line in [
            "<key>CFBundleExecutable</key>\n\t<string>gdev</string>",
            "<key>CFBundleIdentifier</key>\n\t<string>com.studio.gdev</string>",
            "<string>gdev — Blot 3D &amp; &lt;friends&gt;</string>",
            "<key>NSHighResolutionCapable</key>\n\t<true/>",
            "<key>LSMinimumSystemVersion</key>\n\t<string>11.0</string>",
        ] {
            assert!(text.contains(line), "{line}\n{text}");
        }
    }

    #[test]
    fn bundle_identifiers_are_checked_or_made() {
        assert_eq!(bundle_identifier("gdev", None).unwrap(), "pen.gdev");
        assert_eq!(bundle_identifier("salt game+", None).unwrap(), "pen.salt-game-");
        assert_eq!(bundle_identifier("x", Some("com.studio.x-1")).unwrap(), "com.studio.x-1");
        for bad in ["", ".com", "com.", "com studio", "com/studio", "côm"] {
            assert!(bundle_identifier("x", Some(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_game_is_found_beside_the_runtime_or_in_its_bundle() {
        let root = crate::project::temporary_dir().unwrap().join(format!("app-bundle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let folder = root.join("linux");
        std::fs::create_dir_all(folder.join("game")).unwrap();
        assert_eq!(packaged_game(&folder.join("gdev")), None);
        std::fs::write(folder.join("game/game.wasm"), b"").unwrap();
        assert_eq!(packaged_game(&folder.join("gdev")), Some(folder.join("game")));
        let layout = Layout::new(&root.join("mac"), &identity(), "x86_64-apple-darwin", None).unwrap();
        layout.create().unwrap();
        assert!(layout.info.as_ref().is_some_and(|(path, _)| path.is_file()));
        assert_eq!(packaged_game(&layout.executable), None);
        std::fs::write(layout.game.join("game.wasm"), b"").unwrap();
        let found = packaged_game(&layout.executable).unwrap();
        assert_eq!(found.canonicalize().unwrap(), layout.game.canonicalize().unwrap());
        std::fs::remove_dir_all(root).unwrap();
    }
}
