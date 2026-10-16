//! Compile and watch checkout programs through Wasm components.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

use anyhow::{Result, bail};
use pen_plugins::{Diagnostic, Level, Outcome, Plugins};

use crate::dev::watch;
use crate::devlog::{self, Source};

/// The checkout's plugins, if its `pen.json` has a compiler.
pub fn load(root: &Path) -> Result<Option<Arc<Plugins>>> {
    let log: pen_plugins::Logger = Arc::new(|level, message| match level {
        Level::Info => devlog::info(Source::Build, message),
        Level::Error => devlog::error(Source::Build, message),
    });
    Ok(Plugins::load(root, log)?.filter(|plugins| plugins.compiles()))
}

/// Writes `bytes` to `path` in one rename, unless it holds them already;
/// whether it changed.
fn write(path: &Path, bytes: &[u8]) -> Result<bool> {
    if std::fs::read(path).is_ok_and(|old| old == bytes) {
        return Ok(false);
    }
    let name = path.file_name().map_or_else(|| "build".into(), |name| name.to_string_lossy().into_owned());
    let temporary = path.with_file_name(format!(".{name}.{}.tmp", std::process::id()));
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(&temporary, path)?;
    Ok(true)
}

/// A diagnostic as `file:line:column: code: message`, its place counted in
/// the file when it can be read (or in `standing_in`'s source, when it is
/// that path's: a file that is only an overlay), else its byte offset.
pub fn describe(diagnostic: &Diagnostic, standing_in: Option<(&str, &str)>) -> String {
    let source = match standing_in {
        Some((path, source)) if path == diagnostic.file => Some(source.as_bytes().to_vec()),
        _ => std::fs::read(&diagnostic.file).ok(),
    };
    let place = source
        .filter(|bytes| diagnostic.start as usize <= bytes.len())
        .map(|bytes| {
            let before = &bytes[..diagnostic.start as usize];
            let line = before.iter().filter(|byte| **byte == b'\n').count() + 1;
            let column = String::from_utf8_lossy(
                &before[before.iter().rposition(|byte| *byte == b'\n').map_or(0, |at| at + 1)..],
            )
            .chars()
            .count()
                + 1;
            format!("{line}:{column}")
        })
        .unwrap_or_else(|| format!("@{}", diagnostic.start));
    let mut description = format!("{}:{place}: {}: {}", diagnostic.file, diagnostic.code, diagnostic.message);
    if let Some(details) = &diagnostic.details {
        description.push('\n');
        description.push_str(details);
    }
    description
}

/// Says what a build of `what` came to, and writes its Wasm to `out`;
/// whether it built.
fn take(what: &str, out: &Path, outcome: Outcome) -> bool {
    match outcome {
        Outcome::Built(built) => match write(out, &built.wasm) {
            Ok(changed) => {
                let unchanged = if changed { "" } else { ", unchanged" };
                devlog::info(Source::Build, format!("plugins: {what} compiled ({:.2} s{unchanged})", built.seconds));
                true
            }
            Err(error) => {
                devlog::error(Source::Build, format!("plugins: writing {}: {error:#}", out.display()));
                false
            }
        },
        Outcome::Rejected(diagnostics) => {
            devlog::error(Source::Build, format!("plugins: {what} build failed"));
            for diagnostic in &diagnostics {
                devlog::error(Source::Build, describe(diagnostic, None));
            }
            false
        }
        Outcome::Failed(error) => {
            devlog::error(Source::Build, format!("plugins: {what} build failed: {error}"));
            false
        }
    }
}

/// Builds `what`, the program at `entry`, into `out` once.
pub fn compile(plugins: Arc<Plugins>, what: &str, entry: &Path, out: &Path) -> Result<()> {
    if !take(what, out, pen_plugins::build(plugins, entry)) {
        bail!("the {what} did not compile");
    }
    Ok(())
}

/// Builds `what` into `out`, and again as its sources change, until the
/// result is dropped; the receiver hears whether each build built. The
/// sources are watched through the session's one file watcher
/// (`dev::watch`), as each build says which they are.
pub fn watch(
    plugins: Arc<Plugins>,
    what: &str,
    entry: &Path,
    out: &Path,
    ignored: Vec<PathBuf>,
) -> Result<(pen_plugins::Watching, Receiver<bool>)> {
    let (sender, built) = channel();
    let (changed, changes) = channel();
    let subscription = watch::subscribe(&[], move |event| {
        if let Ok(event) = event
            && !matches!(event.kind, notify::EventKind::Access(_))
        {
            let _ = changed.send(event.paths);
        }
    })?;
    let (what, out) = (what.to_owned(), out.to_owned());
    let watching = pen_plugins::watch_changes(plugins, entry, ignored, changes, move |outcome, sources| {
        // Before the outcome is told: an edit made on seeing it must count.
        if let Err(error) = subscription.extend(&roots(sources)) {
            devlog::error(Source::Build, format!("plugins: watching the {what}'s sources: {error:#}"));
        }
        let _ = sender.send(take(&what, &out, outcome));
    })?;
    Ok((watching, built))
}

/// What to watch for a build's sources: a directory and everything under
/// it, or a file through its directory (editors replace files they save,
/// and a file may not be there yet).
fn roots(sources: &BTreeSet<PathBuf>) -> Vec<(&Path, bool)> {
    let mut roots: Vec<(&Path, bool)> = Vec::new();
    for source in sources {
        let root = if source.is_dir() {
            Some((source.as_path(), true))
        } else {
            source.parent().filter(|parent| parent.is_dir()).map(|parent| (parent, false))
        };
        if let Some(root) = root.filter(|root| !roots.contains(root)) {
            roots.push(root);
        }
    }
    roots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watches_directories_whole_and_files_through_theirs() {
        let directory = std::env::temp_dir().join(format!("pen-plugin-roots-{}", std::process::id()));
        std::fs::create_dir_all(directory.join("src/lib")).unwrap();
        let sources: BTreeSet<PathBuf> = [
            directory.join("src/lib"),
            directory.join("src/main.blot"),
            directory.join("src/later.blot"),
            directory.join("gone/away.blot"),
        ]
        .into();
        let src = directory.join("src");
        let lib = directory.join("src/lib");
        assert_eq!(roots(&sources), [(src.as_path(), false), (lib.as_path(), true)]);
        let _ = std::fs::remove_dir_all(&directory);
    }
}
