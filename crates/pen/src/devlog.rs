//! The development log, which the dev tools show in their console: the
//! game's reports, the Blot compiler's output, new builds, asset reloads,
//! the asset pipeline's recipe runs and the session's time travel. Each line
//! goes to stderr; builds with the `devtools` feature (the player runtime
//! has none) also keep the last ones for the dev tools to read.

/// Where a line came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The game runtime's reports (`Host::report`).
    Game,
    /// The Blot compiler watching the checkout (`pen dev`).
    #[cfg_attr(not(feature = "compiler"), allow(dead_code))]
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

/// A line kept for the dev tools.
#[cfg(feature = "devtools")]
#[derive(Clone, Debug)]
pub struct Line {
    /// Counts up from 0 over the process's lines.
    pub serial: u64,
    pub source: Source,
    pub error: bool,
    pub text: String,
}

pub fn info(source: Source, text: impl Into<String>) {
    say(source, false, text.into());
}

pub fn error(source: Source, text: impl Into<String>) {
    say(source, true, text.into());
}

/// Keeps a line for the dev tools that has gone to stderr already.
pub fn note(source: Source, text: impl Into<String>) {
    keep(source, false, text.into());
}

fn say(source: Source, error: bool, text: String) {
    eprintln!("{text}");
    keep(source, error, text);
}

#[cfg(feature = "devtools")]
mod kept {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use super::{Line, Source};

    /// Lines the dev tools can still read; older ones are dropped.
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
        kept.lines.push_back(Line { serial, source, error, text });
        if kept.lines.len() > KEEP {
            kept.lines.pop_front();
        }
    }

    /// The lines from `serial` on that are still kept, oldest first.
    pub fn since(serial: u64) -> Vec<Line> {
        let kept = KEPT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let skip = kept
            .lines
            .front()
            .map_or(0, |first| usize::try_from(serial.saturating_sub(first.serial)).unwrap_or(usize::MAX));
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
