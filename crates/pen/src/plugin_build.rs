//! Building a checkout's programs through its Wasm plugins (`pen.json`,
//! `pen-plugins`) instead of Deno: the game's `game.wasm` and the dev tools'
//! `devtools.wasm`, written whole in one rename as the Deno build did, once
//! or again as their sources change. What the plugins and the builds say
//! goes to the log, so to stderr and the dev tools' console.
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

use anyhow::{Result, bail};
use pen_plugins::{Diagnostic, Level, Outcome, Plugins};

use crate::overlay::{self, Source};

/// The checkout's plugins, if its `pen.json` has a compiler.
pub fn load(root: &Path) -> Result<Option<Arc<Plugins>>> {
    let log: pen_plugins::Logger = Arc::new(|level, message| match level {
        Level::Info => overlay::info(Source::Build, message),
        Level::Error => overlay::error(Source::Build, message),
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
            let column = String::from_utf8_lossy(&before[before.iter().rposition(|byte| *byte == b'\n').map_or(0, |at| at + 1)..])
                .chars()
                .count()
                + 1;
            format!("{line}:{column}")
        })
        .unwrap_or_else(|| format!("@{}", diagnostic.start));
    format!("{}:{place}: {}: {}", diagnostic.file, diagnostic.code, diagnostic.message)
}

/// Says what a build of `what` came to, and writes its Wasm to `out`;
/// whether it built.
fn take(what: &str, out: &Path, outcome: Outcome) -> bool {
    match outcome {
        Outcome::Built(built) => match write(out, &built.wasm) {
            Ok(changed) => {
                let unchanged = if changed { "" } else { ", unchanged" };
                overlay::info(Source::Build, format!("plugins: {what} compiled ({:.2} s{unchanged})", built.seconds));
                true
            }
            Err(error) => {
                overlay::error(Source::Build, format!("plugins: writing {}: {error:#}", out.display()));
                false
            }
        },
        Outcome::Rejected(diagnostics) => {
            overlay::error(Source::Build, format!("plugins: {what} build failed"));
            for diagnostic in &diagnostics {
                overlay::error(Source::Build, describe(diagnostic, None));
            }
            false
        }
        Outcome::Failed(error) => {
            overlay::error(Source::Build, format!("plugins: {what} build failed: {error}"));
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
/// result is dropped; the receiver hears whether each build built.
pub fn watch(
    plugins: Arc<Plugins>,
    what: &str,
    entry: &Path,
    out: &Path,
    ignored: Vec<PathBuf>,
) -> Result<(pen_plugins::Watching, Receiver<bool>)> {
    let (sender, built) = channel();
    let (what, out) = (what.to_owned(), out.to_owned());
    let watching = pen_plugins::watch(plugins, entry, ignored, move |outcome| {
        let _ = sender.send(take(&what, &out, outcome));
    })?;
    Ok((watching, built))
}
