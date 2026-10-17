//! itch.io builds: `butler push` of a package to its platform's channel of
//! the game (`pen.json`'s `itch.target`, `user/game`). butler uploads a
//! patch against the channel's last build (itch.io's wharf diffs), never
//! the whole package again, and with `--if-changed` nothing when it is
//! the same; itch.io tags a channel's platform by its name (`linux`,
//! `windows`, `osx`; `html5` for a browser build).
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Result, bail, ensure};
use serde::Deserialize;

use super::Platform;

/// `pen.json`'s `itch` section.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The game on itch.io, `user/game`.
    pub target: String,
    /// A channel for a platform other than its own name: `{"web":
    /// "html5-beta"}`.
    #[serde(default)]
    pub channels: BTreeMap<String, String>,
}

/// Who pushes, from the environment: butler (PEN_BUTLER, else on PATH),
/// and whether BUTLER_API_KEY holds its key, which butler reads itself.
pub struct Account {
    pub butler: PathBuf,
    pub key: bool,
}

/// The channel a platform's builds go to.
fn channel(config: &Config, platform: Platform) -> String {
    let default = match platform {
        Platform::Linux => "linux",
        Platform::Windows => "windows",
        Platform::Macos => "osx",
        Platform::Web => "html5",
    };
    config.channels.get(platform.name()).cloned().unwrap_or_else(|| default.to_owned())
}

fn name(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|character| character.is_ascii_alphanumeric() || "-_".contains(character))
}

/// The butler push of the package `out`: refused without an API key when
/// it is to run.
pub fn push(
    config: &Config,
    out: &Path,
    platform: Platform,
    label: Option<&str>,
    account: &Account,
    run: bool,
) -> Result<Command> {
    let parts: Vec<&str> = config.target.split('/').collect();
    ensure!(
        parts.len() == 2 && parts.iter().all(|part| name(part)),
        "itch.target {:?} is not user/game, as the game's page address has them",
        config.target
    );
    let channel = channel(config, platform);
    ensure!(name(&channel), "itch.channels: {channel:?} is not a channel's name");
    if run && !account.key {
        bail!("--upload needs BUTLER_API_KEY, an itch.io API key for butler (pen never runs butler login)");
    }
    let mut command = Command::new(&account.butler);
    command.arg("push").arg(out).arg(format!("{}:{channel}", config.target)).arg("--if-changed");
    if let Some(label) = label {
        command.args(["--userversion", label]);
    }
    Ok(command)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(text: &str) -> Config {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn packages_push_to_their_platforms_channel() {
        let config = config(r#"{"target": "studio/salt", "channels": {"web": "html5-beta"}}"#);
        let account = Account { butler: "butler".into(), key: false };
        let shown = |platform, label| {
            super::super::shown(&push(&config, Path::new("dist/linux"), platform, label, &account, false).unwrap())
        };
        assert_eq!(shown(Platform::Linux, None), "butler push dist/linux studio/salt:linux --if-changed");
        assert_eq!(
            shown(Platform::Windows, Some("0.3.1")),
            "butler push dist/linux studio/salt:windows --if-changed --userversion 0.3.1"
        );
        assert!(shown(Platform::Macos, None).contains("studio/salt:osx"));
        assert!(shown(Platform::Web, None).contains("studio/salt:html5-beta"));
    }

    #[test]
    fn butler_runs_only_when_asked_with_a_key() {
        let out = Path::new("dist/linux");
        let config = config(r#"{"target": "studio/salt"}"#);
        let keyless = Account { butler: "butler".into(), key: false };
        let refused = push(&config, out, Platform::Linux, None, &keyless, true).unwrap_err().to_string();
        assert!(refused.contains("BUTLER_API_KEY"), "{refused}");
        for target in ["salt", "studio/salt/extra", "studio/my salt", "/salt"] {
            let wrong = Config { target: target.into(), channels: BTreeMap::new() };
            assert!(push(&wrong, out, Platform::Linux, None, &keyless, false).is_err(), "{target}");
        }
        let odd = Config { target: "studio/salt".into(), channels: [("linux".into(), "a:b".into())].into() };
        assert!(push(&odd, out, Platform::Linux, None, &keyless, false).is_err());
        assert!(serde_json::from_str::<Config>(r#"{"target": "a/b", "key": "secret"}"#).is_err());
        // With a key and --upload: what runs is a stand-in that notes its
        // arguments (no test runs butler).
        #[cfg(unix)]
        {
            let directory =
                crate::project::temporary_dir().unwrap().join(format!("publish-itch-{}", std::process::id()));
            let (fake, arguments) = super::super::tests::fake_tool(&directory, "butler");
            let account = Account { butler: fake, key: true };
            let command = push(&config, out, Platform::Linux, Some("7"), &account, true).unwrap();
            super::super::run(command, true, "butler").unwrap();
            let ran = std::fs::read_to_string(arguments).unwrap();
            assert_eq!(ran, "push\ndist/linux\nstudio/salt:linux\n--if-changed\n--userversion\n7\n");
            std::fs::remove_dir_all(directory).unwrap();
        }
    }
}
