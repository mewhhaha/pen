//! Every player's input, frame by frame: known from frame 0 up to where
//! its peer's messages have reached, and past that predicted. A player's
//! input is events, and the game keeps what they leave (a key held, where
//! the pointer is), so the prediction is that nothing new happens: no
//! events. It proved wrong when a frame that ran on it turns out to have
//! had some.
use std::collections::VecDeque;

use anyhow::{Result, ensure};
use pen_protocol::boundary::{EVENT_FLOATS, FRAME_HEADER_FLOATS, encode_frame, event_tags, whole};
use pen_protocol::{Frame, GameEvent, Viewport};

/// Events a peer's host makes rather than its player: the renderer's
/// assets, stored preferences, tunables and service answers. Only the
/// session host's reach the game (`records`).
const SESSION_TAGS: [u32; 5] = [
    event_tags::ASSET_CHANGED,
    event_tags::PREFERENCE,
    event_tags::TUNE,
    event_tags::SERVICE_ANSWER,
    event_tags::SERVICE_DATA,
];

/// Events whose lanes are page positions in the viewport.
const POSITIONED_TAGS: [u32; 4] =
    [event_tags::POINTER_DOWN, event_tags::POINTER_UP, event_tags::POINTER_MOVE, event_tags::WHEEL];

fn tag(record: &[f32]) -> Option<u32> {
    whole(record[0])
}

/// A peer's frame of events as its player's input: its records, the
/// session host's own events kept only on the host, and positions moved
/// from the peer's viewport into the session's.
pub fn records(events: &[GameEvent], local: Viewport, session: Viewport, host: bool) -> Result<Vec<f32>> {
    let frame = Frame { timestamp: 0.0, viewport: session, events: events.to_vec() };
    let lanes = encode_frame(&frame, 0.0, &[])?;
    let (x, y) = (session.width as f32 / local.width as f32, session.height as f32 / local.height as f32);
    let mut records = Vec::with_capacity(lanes.len() - FRAME_HEADER_FLOATS);
    for record in lanes[FRAME_HEADER_FLOATS..].as_chunks::<EVENT_FLOATS>().0 {
        let tag = tag(record);
        if !host && tag.is_some_and(|tag| SESSION_TAGS.contains(&tag)) {
            continue;
        }
        let at = records.len();
        records.extend_from_slice(record);
        if tag.is_some_and(|tag| POSITIONED_TAGS.contains(&tag)) {
            records[at + 2] *= x;
            records[at + 3] *= y;
        }
    }
    Ok(records)
}

/// Checks a frame of `player`'s records from its peer: whole tags, never a
/// Player marker (the merge writes those, so a peer cannot speak for
/// another player), and the session host's events only from the host.
pub fn check(player: usize, records: &[f32]) -> Result<()> {
    ensure!(records.len().is_multiple_of(EVENT_FLOATS), "a frame of partial events");
    for record in records.as_chunks::<EVENT_FLOATS>().0 {
        let tag = tag(record).filter(|tag| *tag >= 1);
        ensure!(
            tag.is_some_and(|tag| tag != event_tags::PLAYER && (player == 0 || !SESSION_TAGS.contains(&tag))),
            "player {player} sent an event it cannot send (tag {})",
            record[0]
        );
    }
    Ok(())
}

/// One player's known frames.
#[derive(Debug, Default)]
struct Track {
    /// Frames from `base` on.
    known: VecDeque<Vec<f32>>,
    base: u64,
}

impl Track {
    fn end(&self) -> u64 {
        self.base + self.known.len() as u64
    }

    fn get(&self, frame: u64) -> Option<&[f32]> {
        let index = usize::try_from(frame.checked_sub(self.base)?).ok()?;
        self.known.get(index).map(Vec::as_slice)
    }

    fn forget_before(&mut self, frame: u64) {
        while self.base < frame && !self.known.is_empty() {
            self.known.pop_front();
            self.base += 1;
        }
    }
}

#[derive(Debug)]
pub struct Inputs {
    tracks: Vec<Track>,
    /// Frames before it ran, each on what was known of every input then.
    next: u64,
    /// The earliest frame that ran on a prediction that proved wrong.
    wrong: Option<u64>,
}

impl Inputs {
    /// `players` players' input, of which the first `delay` frames are
    /// known to be empty: nobody's input reaches them.
    pub fn new(players: usize, delay: u32) -> Inputs {
        let empty = || Track { known: (0..delay).map(|_| Vec::new()).collect(), base: 0 };
        Inputs { tracks: (0..players).map(|_| empty()).collect(), next: 0, wrong: None }
    }

    pub fn players(&self) -> usize {
        self.tracks.len()
    }

    /// Frames before it have every player's input known.
    pub fn confirmed(&self) -> u64 {
        self.tracks.iter().map(Track::end).min().unwrap_or(0)
    }

    /// Frames before it have `player`'s input known.
    pub fn known(&self, player: usize) -> u64 {
        self.tracks[player].end()
    }

    /// `player`'s known input for `frame`.
    pub fn get(&self, player: usize, frame: u64) -> Option<&[f32]> {
        self.tracks[player].get(frame)
    }

    /// The next frame of `player`'s input, which its own peer knows.
    pub fn push(&mut self, player: usize, records: Vec<f32>) {
        let frame = self.tracks[player].end();
        if frame < self.next && !records.is_empty() {
            self.wrong = Some(self.wrong.map_or(frame, |wrong| wrong.min(frame)));
        }
        self.tracks[player].known.push_back(records);
    }

    /// A peer's frames of its player's input from `first` on: those known
    /// already are skipped, and frames past a gap wait to be sent again.
    /// Returns how many were new.
    pub fn receive(&mut self, player: usize, first: u64, frames: Vec<Vec<f32>>) -> usize {
        let end = self.tracks[player].end();
        let Some(skip) = end.checked_sub(first) else { return 0 };
        let mut new = 0;
        for records in frames.into_iter().skip(usize::try_from(skip).unwrap_or(usize::MAX)) {
            self.push(player, records);
            new += 1;
        }
        new
    }

    /// The input frame `frame` runs on: `elapsed` and the viewport, then
    /// each player in order, its Player marker and its events (none where
    /// they are not known yet).
    pub fn frame(&self, frame: u64, elapsed: f32, viewport: Viewport) -> Vec<f32> {
        let mut lanes = vec![elapsed, viewport.width as f32, viewport.height as f32, 0.0, 0.0];
        let mut events = 0;
        for (player, track) in self.tracks.iter().enumerate() {
            lanes.extend_from_slice(&[event_tags::PLAYER as f32, player as f32, 0.0, 0.0, 0.0]);
            let records = track.get(frame).unwrap_or_default();
            lanes.extend_from_slice(records);
            events += 1 + records.len() / EVENT_FLOATS;
        }
        lanes[3] = events as f32;
        lanes
    }

    /// The frame at `frame` ran (or ran again) on what is known now.
    pub fn ran(&mut self, frame: u64) {
        self.next = self.next.max(frame + 1);
    }

    /// The next frame to run.
    pub fn next(&self) -> u64 {
        self.next
    }

    /// The earliest frame that ran on a wrong prediction, once.
    pub fn take_wrong(&mut self) -> Option<u64> {
        self.wrong.take()
    }

    /// Forgets `player`'s frames before `frame`, which nothing needs again.
    pub fn forget_before(&mut self, player: usize, frame: u64) {
        let confirmed = self.confirmed();
        self.tracks[player].forget_before(frame.min(confirmed));
    }

    /// `player`'s known frames from `from` on, at most `count` of them.
    pub fn frames_from(&self, player: usize, from: u64, count: usize) -> impl Iterator<Item = &[f32]> {
        let track = &self.tracks[player];
        (from.max(track.base)..track.end()).take(count).filter_map(move |frame| track.get(frame))
    }

    /// Where `player`'s frames held start.
    pub fn base(&self, player: usize) -> u64 {
        self.tracks[player].base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SESSION: Viewport = Viewport { width: 960, height: 640 };

    fn key(name: &str) -> GameEvent {
        GameEvent::KeyDown(name.to_owned())
    }

    #[test]
    fn a_peers_events_become_its_players_records() {
        let events = [
            key("a"),
            GameEvent::PointerMove { x: 100.0, y: 50.0 },
            GameEvent::AssetReady { key: "rock".into(), kind: "mesh".into(), revision: 1 },
            GameEvent::ServiceAnswer { request: 3, status: 0, value: 1.0, length: 0 },
        ];
        // A window twice the session's size: positions halve.
        let local = Viewport { width: 1920, height: 1280 };
        let host = records(&events, local, SESSION, true).unwrap();
        assert_eq!(host.len(), 4 * EVENT_FLOATS);
        assert_eq!(&host[5..10], &[5.0, 0.0, 50.0, 25.0, 0.0]);
        // Another peer's host events stay with it.
        let guest = records(&events, local, SESSION, false).unwrap();
        assert_eq!(guest, host[..10]);
        assert!(check(1, &guest).is_ok() && check(0, &host).is_ok());
        assert!(check(1, &host).is_err(), "only the host sends the session's events");
        assert!(check(1, &[26.0, 0.0, 0.0, 0.0, 0.0]).is_err(), "nobody sends a Player marker");
        assert!(check(1, &[1.5, 0.0, 0.0, 0.0, 0.0]).is_err() && check(1, &[1.0, 0.0]).is_err());
    }

    #[test]
    fn frames_list_every_player_in_order() {
        let mut inputs = Inputs::new(2, 2);
        assert_eq!((inputs.confirmed(), inputs.known(0), inputs.known(1)), (2, 2, 2));
        let frame = inputs.frame(0, 0.5, SESSION);
        assert_eq!(frame, [0.5, 960.0, 640.0, 2.0, 0.0, 26.0, 0.0, 0.0, 0.0, 0.0, 26.0, 1.0, 0.0, 0.0, 0.0]);
        inputs.push(1, vec![1.0, 65.0, 0.0, 0.0, 0.0]);
        inputs.push(0, Vec::new());
        assert_eq!(inputs.confirmed(), 3);
        let frame = inputs.frame(2, 0.5, SESSION);
        assert_eq!(frame[3], 3.0);
        assert_eq!(&frame[10..20], &[26.0, 1.0, 0.0, 0.0, 0.0, 1.0, 65.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn frames_that_ran_on_a_wrong_prediction_are_found() {
        let mut inputs = Inputs::new(2, 1);
        // Player 0 is local: known ahead. Player 1's input lags.
        for _ in 0..6 {
            inputs.push(0, Vec::new());
        }
        for frame in 0..5 {
            inputs.ran(frame);
        }
        assert_eq!(inputs.next(), 5);
        // Frames 1 and 2 arrive empty: as predicted.
        assert_eq!(inputs.receive(1, 1, vec![Vec::new(), Vec::new()]), 2);
        assert_eq!(inputs.take_wrong(), None);
        // A repeat, then frames 2 to 4 with a key in 4 and 3: 3 was wrong.
        assert_eq!(inputs.receive(1, 1, vec![Vec::new()]), 0);
        let key = vec![1.0, 65.0, 0.0, 0.0, 0.0];
        assert_eq!(inputs.receive(1, 2, vec![Vec::new(), key.clone(), key.clone()]), 2);
        assert_eq!(inputs.take_wrong(), Some(3));
        assert_eq!(inputs.take_wrong(), None);
        // A frame that has not run yet was never predicted.
        assert_eq!(inputs.receive(1, 5, vec![key]), 1);
        assert_eq!(inputs.take_wrong(), None);
        // A gap waits for the frames before it.
        assert_eq!(inputs.receive(1, 9, vec![Vec::new()]), 0);
        assert_eq!(inputs.confirmed(), 6);
        inputs.forget_before(1, 4);
        assert_eq!((inputs.base(1), inputs.known(1)), (4, 6));
        assert_eq!(inputs.frames_from(1, 0, 8).count(), 2);
        assert_eq!(inputs.get(1, 3), None);
    }
}
