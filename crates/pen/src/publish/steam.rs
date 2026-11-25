//! Steam builds: steamcmd's app and depot build scripts (VDF) for a
//! package, in `steam/` beside it, and the steamcmd run that uploads them.
//! Each package is one depot, by its platform (`pen.json`'s
//! `steam.depots`); the app build names every depot whose script is there,
//! so packaging Linux and Windows into `dist/linux` and `dist/windows`
//! makes one build of both in `dist/steam/`.
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

use super::Platform;

/// Spacewar, Valve's shared test app: never a build's destination.
const SPACEWAR: u32 = 480;

/// `pen.json`'s `steam` section.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The app's ID on Steamworks.
    pub app: u32,
    /// A depot for each platform (`linux`, `windows`, `macos`).
    pub depots: BTreeMap<String, u32>,
    /// The branch a build goes live on once uploaded; none leaves it for
    /// Steamworks' builds page. Never `default`, which Steamworks asks a
    /// person to set live.
    #[serde(default)]
    pub branch: Option<String>,
}

/// The build scripts written for a package.
#[derive(Debug)]
pub struct Scripts {
    pub app: PathBuf,
    pub depot: PathBuf,
    /// Every depot the app build names.
    pub depots: Vec<u32>,
}

/// Who uploads, from the environment: steamcmd (PEN_STEAMCMD, else on
/// PATH) and the build account's name (STEAM_BUILD_USER).
pub struct Account {
    pub steamcmd: PathBuf,
    pub user: Option<String>,
}

/// `text` as a VDF string: quoted, with backslashes and quotes escaped.
fn quoted(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', r"\\").replace('"', "\\\""))
}

fn absolute(path: &Path) -> Result<PathBuf> {
    std::fs::canonicalize(path).with_context(|| format!("{} does not exist", path.display()))
}

/// The depot script: the whole package, into the depot's root.
fn depot_script(depot: u32, content: &Path) -> String {
    let mut text = String::new();
    let _ = writeln!(text, "\"DepotBuild\"\n{{");
    let _ = writeln!(text, "\t\"DepotID\" \"{depot}\"");
    let _ = writeln!(text, "\t\"ContentRoot\" {}", quoted(&content.to_string_lossy()));
    let _ = writeln!(text, "\t\"FileMapping\"\n\t{{");
    let _ = writeln!(text, "\t\t\"LocalPath\" \"*\"\n\t\t\"DepotPath\" \".\"\n\t\t\"Recursive\" \"1\"");
    let _ = writeln!(text, "\t}}\n}}");
    text
}

/// The app script: every depot, its script by name beside this one.
fn app_script(config: &Config, depots: &[u32], description: &str, output: &Path) -> String {
    let mut text = String::new();
    let _ = writeln!(text, "\"AppBuild\"\n{{");
    let _ = writeln!(text, "\t\"AppID\" \"{}\"", config.app);
    let _ = writeln!(text, "\t\"Desc\" {}", quoted(description));
    let _ = writeln!(text, "\t\"BuildOutput\" {}", quoted(&output.to_string_lossy()));
    let _ = writeln!(text, "\t\"SetLive\" {}", quoted(config.branch.as_deref().unwrap_or_default()));
    let _ = writeln!(text, "\t\"Preview\" \"0\"");
    let _ = writeln!(text, "\t\"Depots\"\n\t{{");
    for depot in depots {
        let _ = writeln!(text, "\t\t\"{depot}\" \"depot_build_{depot}.vdf\"");
    }
    let _ = writeln!(text, "\t}}\n}}");
    text
}

/// Refuses Spacewar, the default branch and IDs or names Steam would not take.
pub fn check(config: &Config) -> Result<()> {
    ensure!(config.app != SPACEWAR, "app 480 is Spacewar, Valve's test app: give pen.json the game's own app ID");
    ensure!(config.app != 0 && config.depots.values().all(|depot| *depot != 0), "Steam IDs are not 0");
    if let Some(branch) = &config.branch {
        ensure!(
            !branch.eq_ignore_ascii_case("default"),
            "steam.branch: a build goes live on the default branch from Steamworks' builds page, not from a script"
        );
        ensure!(!branch.is_empty() && !branch.contains(['"', '\\']), "steam.branch {branch:?} is not a branch's name");
    }
    for platform in config.depots.keys() {
        ensure!(
            ["linux", "windows", "macos"].contains(&platform.as_str()),
            "steam.depots: {platform} is not linux, windows or macos"
        );
    }
    Ok(())
}

/// Writes the depot script for the package `out` (for `platform`) and the
/// app script for every depot with one, in `steam/` beside `out`.
pub fn write_scripts(config: &Config, out: &Path, platform: Platform, label: Option<&str>) -> Result<Scripts> {
    check(config)?;
    let Some(&depot) = config.depots.get(platform.name()) else {
        bail!("pen.json's steam.depots has no depot for {}", platform.name());
    };
    let content = absolute(out)?;
    let directory = content.parent().context("the package has no parent directory")?.join("steam");
    std::fs::create_dir_all(&directory)?;
    let depot_path = directory.join(format!("depot_build_{depot}.vdf"));
    std::fs::write(&depot_path, depot_script(depot, &content))?;
    let depots: Vec<u32> = config
        .depots
        .values()
        .copied()
        .filter(|depot| directory.join(format!("depot_build_{depot}.vdf")).is_file())
        .collect();
    let description = label.map_or_else(|| format!("{} ({})", content.display(), platform.name()), str::to_owned);
    let app = directory.join(format!("app_build_{}.vdf", config.app));
    std::fs::write(&app, app_script(config, &depots, &description, &directory.join("output")))?;
    Ok(Scripts { app, depot: depot_path, depots })
}

/// The steamcmd run that uploads `scripts`' build: refused without the
/// build account's name when it is to run.
pub fn upload(config: &Config, scripts: &Scripts, account: &Account, run: bool) -> Result<Command> {
    check(config)?;
    let user = match (&account.user, run) {
        (Some(user), _) if !user.is_empty() && !user.starts_with('+') => user.clone(),
        (_, true) => bail!(
            "--upload needs STEAM_BUILD_USER, the Steamworks build account steamcmd has logged in as before \
             (pen never logs in or takes a password)"
        ),
        (_, false) => "$STEAM_BUILD_USER".to_owned(),
    };
    let mut command = Command::new(&account.steamcmd);
    command.args(["+login", &user, "+run_app_build"]).arg(&scripts.app).arg("+quit");
    Ok(command)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(text: &str) -> Config {
        serde_json::from_str(text).unwrap()
    }

    fn directory(name: &str) -> PathBuf {
        let directory =
            crate::project::temporary_dir().unwrap().join(format!("publish-steam-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    #[test]
    fn packages_make_one_build_of_every_depot() {
        let root = directory("scripts");
        let config = config(r#"{"app": 1234560, "depots": {"linux": 1234561, "windows": 1234562}, "branch": "beta"}"#);
        let (linux, windows) = (root.join("linux"), root.join("windows"));
        std::fs::create_dir_all(&linux).unwrap();
        std::fs::create_dir_all(&windows).unwrap();
        let scripts = write_scripts(&config, &linux, Platform::Linux, Some("0.3 \"rc\"")).unwrap();
        let root = std::fs::canonicalize(&root).unwrap();
        assert_eq!(scripts.depot, root.join("steam/depot_build_1234561.vdf"));
        let depot = std::fs::read_to_string(&scripts.depot).unwrap();
        let expected = format!(
            "\"DepotBuild\"\n{{\n\t\"DepotID\" \"1234561\"\n\t\"ContentRoot\" \"{}\"\n\t\"FileMapping\"\n\t{{\n\t\t\
             \"LocalPath\" \"*\"\n\t\t\"DepotPath\" \".\"\n\t\t\"Recursive\" \"1\"\n\t}}\n}}\n",
            root.join("linux").display()
        );
        assert_eq!(depot, expected);
        let app = std::fs::read_to_string(&scripts.app).unwrap();
        assert!(app.contains("\t\"AppID\" \"1234560\"\n\t\"Desc\" \"0.3 \\\"rc\\\"\"\n"), "{app}");
        assert!(app.contains("\t\"SetLive\" \"beta\"\n") && app.contains("\t\"Preview\" \"0\"\n"), "{app}");
        assert!(app.contains("\t\t\"1234561\" \"depot_build_1234561.vdf\"\n") && !app.contains("1234562"), "{app}");
        // Windows next: the same app build now has both.
        let scripts = write_scripts(&config, &windows, Platform::Windows, None).unwrap();
        let app = std::fs::read_to_string(&scripts.app).unwrap();
        assert!(app.contains("depot_build_1234561.vdf") && app.contains("depot_build_1234562.vdf"), "{app}");
        assert!(app.contains(&format!("\"Desc\" \"{} (windows)\"", root.join("windows").display())), "{app}");
        // No depot for the platform.
        assert!(write_scripts(&config, &linux, Platform::Macos, None).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn spacewar_the_default_branch_and_other_mistakes_are_refused() {
        let root = directory("refused");
        for (text, why) in [
            (r#"{"app": 480, "depots": {"linux": 481}}"#, "Spacewar"),
            (r#"{"app": 1234560, "depots": {"linux": 1}, "branch": "Default"}"#, "default branch"),
            (r#"{"app": 1234560, "depots": {"linux": 1}, "branch": "a\"b"}"#, "not a branch"),
            (r#"{"app": 1234560, "depots": {"android": 1}}"#, "not linux"),
            (r#"{"app": 1234560, "depots": {"linux": 0}}"#, "not 0"),
        ] {
            let error = write_scripts(&config(text), &root, Platform::Linux, None).unwrap_err().to_string();
            assert!(error.contains(why), "{text}: {error}");
        }
        assert!(serde_json::from_str::<Config>(r#"{"app": 1, "depots": {}, "password": "x"}"#).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn steamcmd_runs_only_when_asked_with_an_account() {
        let root = directory("upload");
        let config = config(r#"{"app": 1234560, "depots": {"linux": 1234561}}"#);
        let package = root.join("package");
        std::fs::create_dir_all(&package).unwrap();
        let scripts = write_scripts(&config, &package, Platform::Linux, None).unwrap();
        let app = scripts.app.to_string_lossy().into_owned();
        let nobody = Account { steamcmd: "steamcmd".into(), user: None };
        // A dry run shows the command, the account left to the environment.
        let dry = upload(&config, &scripts, &nobody, false).unwrap();
        assert_eq!(super::super::shown(&dry), format!("steamcmd +login $STEAM_BUILD_USER +run_app_build {app} +quit"));
        let refused = upload(&config, &scripts, &nobody, true).unwrap_err().to_string();
        assert!(refused.contains("STEAM_BUILD_USER"), "{refused}");
        let sneaky = Account { steamcmd: "steamcmd".into(), user: Some("+quit".into()) };
        assert!(upload(&config, &scripts, &sneaky, true).is_err());
        let spacewar = Config { app: 480, depots: config.depots.clone(), branch: None };
        let builder = Account { steamcmd: "steamcmd".into(), user: Some("builder".into()) };
        assert!(upload(&spacewar, &scripts, &builder, true).is_err());
        // With an account and --upload: what runs is a stand-in that notes
        // its arguments (no test runs steamcmd).
        #[cfg(unix)]
        {
            let (fake, arguments) = super::super::tests::fake_tool(&root.join("tools"), "steamcmd");
            let account = Account { steamcmd: fake, user: Some("builder".into()) };
            super::super::run(upload(&config, &scripts, &account, true).unwrap(), true, "steamcmd").unwrap();
            let ran = std::fs::read_to_string(arguments).unwrap();
            assert_eq!(ran, format!("+login\nbuilder\n+run_app_build\n{app}\n+quit\n"));
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
