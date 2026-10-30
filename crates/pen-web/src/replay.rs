//! Shareable replays. A page plays the input recording its link names:
//! `#replay=DATA`, the recording itself in the link (`pen_runtime::
//! recording::fragment`), or `?replay=URL`, a `.penrec` beside the page or
//! wherever a fetch from it may read. The dev tools' `/record` records the
//! page's input and hands the recording over as a file and as such a link.
//! A recording replays exactly on the build that made it, native or here.
use anyhow::{Context, Result};
use pen_runtime::game::Moment;
use pen_runtime::recording;
use pen_runtime::timeline::{Recorded, Recording};

/// Where the page's link has a recording, if it has one.
pub fn linked() -> Option<String> {
    let location = web_sys::window().unwrap().location();
    if let Some(data) = location.hash().ok()?.strip_prefix("$replay=").filter(|data| !data.is_empty()) {
        return Some(format!("$replay={data}"));
    }
    let query = web_sys::UrlSearchParams::new_with_str(&location.search().ok()?).ok()?;
    query.get("replay").filter(|url| !url.is_empty())
}

/// The recording `source` (what [`linked`] found) holds, and a name for it.
pub async fn fetch(source: &str) -> Result<(Recording, String)> {
    let (bytes, name) = match source.strip_prefix("$replay=") {
        Some(data) => (recording::from_fragment(data)?, "the link".to_owned()),
        None => (crate::fetch(source).await?, source.to_owned()),
    };
    let decoded = recording::decode(&bytes).with_context(|| format!("reading the recording in {name}"))?;
    Ok((decoded.recording, name))
}

/// A link to this page that plays `bytes`, a recording.
pub fn link(bytes: &[u8]) -> String {
    let location = web_sys::window().unwrap().location();
    let page = format!("{}{}", location.origin().unwrap_or_default(), location.pathname().unwrap_or_default());
    format!("{page}#replay={}", recording::fragment(bytes))
}

/// A recording of the page's input being made (`/record`).
pub struct Recorder {
    /// The position it starts at.
    pub first: u64,
    start: Moment,
    frames: Vec<Recorded>,
}

impl Recorder {
    pub fn new(first: u64, start: Moment) -> Recorder {
        Recorder { first, start, frames: Vec::new() }
    }

    pub fn frame(&mut self, recorded: &Recorded) {
        self.frames.push(recorded.clone());
    }

    /// Forgets the frames from `position` on: the game went back to it.
    pub fn rewind(&mut self, position: u64) {
        let kept = usize::try_from(position.saturating_sub(self.first)).unwrap_or(usize::MAX);
        self.frames.truncate(kept);
    }

    pub fn frames(&self) -> usize {
        self.frames.len()
    }

    /// The recording, as a `.penrec` holds it.
    pub fn finish(self) -> Vec<u8> {
        recording::encode(&Recording { start: self.start, frames: self.frames })
    }
}
