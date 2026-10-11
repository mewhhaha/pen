//! What the dev overlay shows besides its own lines: the game's reports, the
//! Blot compiler's output, new builds, asset reloads and the asset pipeline's
//! recipe runs. Each line still goes to stderr as before; builds with the
//! `devtools` feature also keep the last ones for the overlay to read.
use std::io::{BufRead, BufReader, Read};

/// Where a line came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The game runtime's reports (`Host::report`).
    Game,
    /// The Blot compiler watching the checkout (`pen dev`).
    Build,
    /// New builds of the game: compiled, reloaded or refused.
    Reload,
    /// Asset files reloading, and the renderer's word on them.
    Asset,
    /// The checkout's asset recipes (`pipeline.json`).
    Pipeline,
    /// The session: time travel, input recordings, saves and loads.
    Session,
}

/// A line kept for the overlay.
#[cfg(feature = "devtools")]
#[derive(Clone, Debug)]
pub struct Line {
    /// Counts up from 0 over the process's lines.
    pub serial: u64,
    pub source: Source,
    pub error: bool,
    pub text: String,
    pub at: std::time::Instant,
}

pub fn info(source: Source, text: impl Into<String>) {
    say(source, false, text.into());
}

pub fn error(source: Source, text: impl Into<String>) {
    say(source, true, text.into());
}

/// Keeps a line for the overlay that has gone to stderr already.
pub fn note(source: Source, text: impl Into<String>) {
    keep(source, false, text.into());
}

fn say(source: Source, error: bool, text: String) {
    eprintln!("{text}");
    keep(source, error, text);
}

/// The Blot compiler's stderr, line by line, to ours and the overlay's, on a
/// thread of its own until the compiler closes it. A failed build's
/// diagnostics follow its `blot: Game build failed` line, up to the next
/// `blot:` line.
pub fn forward_build_output(stderr: impl Read + Send + 'static) {
    let forwarding = std::thread::Builder::new().name("pen-build-output".into()).spawn(move || {
        let mut failing = false;
        for line in BufReader::new(stderr).split(b'\n').map_while(Result::ok) {
            let line = String::from_utf8_lossy(&line).trim_end().to_owned();
            if line.starts_with("blot: ") {
                failing = line.contains("build failed");
            }
            say(Source::Build, failing, line);
        }
    });
    if let Err(error) = forwarding {
        eprintln!("cannot forward the Blot compiler's output: {error}");
    }
}

#[cfg(feature = "devtools")]
mod kept {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use super::{Line, Source};

    /// Lines the overlay can still read; older ones are dropped.
    const KEEP: usize = 2000;

    struct Kept {
        lines: VecDeque<Line>,
        next: u64,
    }

    static KEPT: Mutex<Kept> = Mutex::new(Kept { lines: VecDeque::new(), next: 0 });

    pub fn keep(source: Source, error: bool, text: String) {
        let text = without_escapes(&text);
        let mut kept = KEPT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let serial = kept.next;
        kept.next += 1;
        kept.lines.push_back(Line { serial, source, error, text, at: std::time::Instant::now() });
        if kept.lines.len() > KEEP {
            kept.lines.pop_front();
        }
    }

    /// The lines from `serial` on that are still kept, oldest first.
    pub fn since(serial: u64) -> Vec<Line> {
        let kept = KEPT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let skip = kept.lines.front().map_or(0, |first| serial.saturating_sub(first.serial) as usize);
        kept.lines.iter().skip(skip).cloned().collect()
    }

    /// The text without terminal colour codes (`ESC [ ... letter`).
    fn without_escapes(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut chars = text.chars();
        while let Some(char) = chars.next() {
            if char == '\u{1b}' {
                if chars.next() == Some('[') {
                    for code in chars.by_ref() {
                        if code.is_ascii_alphabetic() {
                            break;
                        }
                    }
                }
            } else {
                out.push(char);
            }
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn strips_colour_codes() {
            assert_eq!(without_escapes("\u{1b}[31merror\u{1b}[0m: no"), "error: no");
            assert_eq!(without_escapes("plain"), "plain");
        }
    }
}

#[cfg(feature = "devtools")]
use kept::keep;
#[cfg(feature = "devtools")]
pub use kept::since;

#[cfg(not(feature = "devtools"))]
fn keep(_source: Source, _error: bool, _text: String) {}
