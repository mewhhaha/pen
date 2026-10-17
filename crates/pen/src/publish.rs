//! Store builds of a package `pen build` made: Steam's (`steam`) and
//! itch.io's (`itch`), from the checkout's `pen.json`:
//!
//! ```json
//! "steam": {"app": 1234560, "depots": {"linux": 1234561, "windows": 1234562}, "branch": "beta"},
//! "itch": {"target": "studio/game", "channels": {"web": "html5"}}
//! ```
//!
//! `pen build --steam` writes steamcmd's build scripts beside the package,
//! and `--itch` says the butler push it would make; both run their tool
//! only with `--upload` and the account's credentials in the environment,
//! never otherwise. pen never logs in: steamcmd uses the login it cached
//! after the account's first interactive one (Steam Guard), and butler the
//! API key in BUTLER_API_KEY.
mod itch;
mod steam;

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};
use serde::de::DeserializeOwned;

use crate::project::Project;

/// What `pen build` was asked to publish.
pub struct Request {
    pub steam: bool,
    pub itch: bool,
    /// Run the uploads, rather than say what they would run.
    pub upload: bool,
    /// The build's description on Steam and version on itch.io.
    pub label: Option<String>,
}

/// What a package runs on, as stores sort builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform {
    Linux,
    Windows,
    Macos,
    Web,
}

impl Platform {
    pub fn name(self) -> &'static str {
        match self {
            Platform::Linux => "linux",
            Platform::Windows => "windows",
            Platform::Macos => "macos",
            Platform::Web => "web",
        }
    }

    /// The platform of the package in `out`, from its runtime's header: the
    /// executable named after the game.
    fn of_package(out: &Path, name: &str, web: bool) -> Result<Platform> {
        if web {
            return Ok(Platform::Web);
        }
        let windows = out.join(format!("{name}.exe"));
        if windows.is_file() {
            return Ok(Platform::Windows);
        }
        // A macOS package is an application bundle.
        if out.join(format!("{name}.app/Contents/MacOS/{name}")).is_file() {
            return Ok(Platform::Macos);
        }
        let executable = out.join(name);
        let header = std::fs::read(&executable).with_context(|| format!("reading {}", executable.display()))?;
        match crate::project::executable_target(&header) {
            Some(target) if target.contains("linux") => Ok(Platform::Linux),
            Some(target) if target.contains("apple") => Ok(Platform::Macos),
            _ => bail!("cannot tell what {} runs on", executable.display()),
        }
    }
}

/// The `key` section of the checkout's `pen.json`, if it has one.
fn config<T: DeserializeOwned>(root: &Path, key: &str) -> Result<Option<T>> {
    let file = root.join("pen.json");
    let text = std::fs::read_to_string(&file).with_context(|| format!("reading {}", file.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("reading {}", file.display()))?;
    config
        .get(key)
        .map(|section| serde_json::from_value(section.clone()))
        .transpose()
        .with_context(|| format!("{}: {key}", file.display()))
}

/// A tool to run: `variable`'s path, else `name` on PATH.
fn tool(variable: &str, name: &str) -> PathBuf {
    std::env::var_os(variable).map_or_else(|| PathBuf::from(name), PathBuf::from)
}

/// A command as a shell would take it, for a dry run to show.
fn shown(command: &Command) -> String {
    std::iter::once(command.get_program())
        .chain(command.get_args())
        .map(|word| {
            let word = word.to_string_lossy();
            if !word.is_empty()
                && word.chars().all(|character| character.is_ascii_alphanumeric() || "-_./:+=@$".contains(character))
            {
                word.into_owned()
            } else {
                format!("'{}'", word.replace('\'', r"'\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Runs an upload, or with `upload` false says what it would run.
fn run(mut command: Command, upload: bool, what: &str) -> Result<()> {
    if !upload {
        eprintln!("{what}, a dry run (--upload runs it): {}", shown(&command));
        return Ok(());
    }
    eprintln!("{what}: {}", shown(&command));
    let status = command.status().with_context(|| format!("running {}", command.get_program().to_string_lossy()))?;
    ensure!(status.success(), "{what} failed ({status})");
    Ok(())
}

impl Request {
    /// The stores' sections of `pen.json` this asks for, read and checked
    /// before the package is built.
    pub fn plan(self, project: &Project) -> Result<Option<Plan>> {
        if !self.steam && !self.itch {
            ensure!(!self.upload && self.label.is_none(), "--upload and --label are for --steam and --itch");
            return Ok(None);
        }
        let steam = if self.steam {
            let config: steam::Config = config(&project.root, "steam")?
                .context("pen.json has no \"steam\" section: {\"app\": ID, \"depots\": {\"linux\": ID, ...}}")?;
            steam::check(&config)?;
            Some(config)
        } else {
            None
        };
        let itch = if self.itch {
            Some(
                config(&project.root, "itch")?
                    .context("pen.json has no \"itch\" section: {\"target\": \"user/game\"}")?,
            )
        } else {
            None
        };
        // Before a build that takes minutes, not after it.
        if self.upload {
            ensure!(
                steam.is_none() || std::env::var_os("STEAM_BUILD_USER").is_some(),
                "--upload needs STEAM_BUILD_USER, the Steamworks build account steamcmd has logged in as before"
            );
            ensure!(itch.is_none() || std::env::var_os("BUTLER_API_KEY").is_some(), "--upload needs BUTLER_API_KEY");
        }
        Ok(Some(Plan { steam, itch, upload: self.upload, label: self.label }))
    }
}

/// The store builds to make of a package.
pub struct Plan {
    steam: Option<steam::Config>,
    itch: Option<itch::Config>,
    upload: bool,
    label: Option<String>,
}

impl Plan {
    /// After `pen build` made the package `out` (`web`, a browser export):
    /// Steam's build scripts and upload, and itch.io's push.
    pub fn publish(&self, project: &Project, out: &Path, web: bool) -> Result<()> {
        let platform = Platform::of_package(out, &project.identity.name, web)?;
        let label = self.label.as_deref();
        if let Some(config) = &self.steam {
            let scripts = steam::write_scripts(config, out, platform, label)?;
            eprintln!(
                "steam: {} for this package, and {} for depots {:?}",
                scripts.depot.display(),
                scripts.app.display(),
                scripts.depots
            );
            let account = steam::Account {
                steamcmd: tool("PEN_STEAMCMD", "steamcmd"),
                user: std::env::var("STEAM_BUILD_USER").ok(),
            };
            run(steam::upload(config, &scripts, &account, self.upload)?, self.upload, "steamcmd")?;
        }
        if let Some(config) = &self.itch {
            let account = itch::Account {
                butler: tool("PEN_BUTLER", "butler"),
                key: std::env::var_os("BUTLER_API_KEY").is_some(),
            };
            run(itch::push(config, out, platform, label, &account, self.upload)?, self.upload, "butler")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in for a store's tool, which writes its arguments to a file
    /// beside it and succeeds: what the tests run instead of steamcmd or
    /// butler, which they never run.
    #[cfg(unix)]
    pub(super) fn fake_tool(directory: &Path, name: &str) -> (PathBuf, PathBuf) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::create_dir_all(directory).unwrap();
        let script = directory.join(name);
        let arguments = directory.join(format!("{name}.args"));
        std::fs::write(&script, format!("#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\n", arguments.display())).unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        (script, arguments)
    }

    #[test]
    fn dry_runs_show_their_commands() {
        let mut command = Command::new("butler");
        command.args(["push", "dist/my game", "studio/game:linux", "--userversion", "it's 1"]);
        assert_eq!(shown(&command), r"butler push 'dist/my game' studio/game:linux --userversion 'it'\''s 1'");
        assert!(run(Command::new("/nonexistent/tool"), false, "a tool").is_ok());
        assert!(run(Command::new("/nonexistent/tool"), true, "a tool").is_err());
    }

    #[test]
    fn packages_say_what_they_run_on() {
        let directory =
            crate::project::temporary_dir().unwrap().join(format!("publish-platform-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        assert_eq!(Platform::of_package(&directory, "game", true).unwrap(), Platform::Web);
        assert!(Platform::of_package(&directory, "game", false).is_err());
        let mut elf = vec![0u8; 20];
        elf[..4].copy_from_slice(b"\x7fELF");
        elf[18..20].copy_from_slice(&0x3eu16.to_le_bytes());
        std::fs::write(directory.join("game"), elf).unwrap();
        assert_eq!(Platform::of_package(&directory, "game", false).unwrap(), Platform::Linux);
        std::fs::write(directory.join("game.exe"), "MZ").unwrap();
        assert_eq!(Platform::of_package(&directory, "game", false).unwrap(), Platform::Windows);
        let mac = directory.join("mac");
        std::fs::create_dir_all(mac.join("game.app/Contents/MacOS")).unwrap();
        std::fs::write(mac.join("game.app/Contents/MacOS/game"), "").unwrap();
        assert_eq!(Platform::of_package(&mac, "game", false).unwrap(), Platform::Macos);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
