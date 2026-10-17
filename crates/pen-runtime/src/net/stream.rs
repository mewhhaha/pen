//! Spectating: a session's final frames streamed to viewers, who play them
//! as a replay plays a recording. The game is a pure function of its input,
//! so a stream is a moment and then every frame's input, like a `.penrec`;
//! a viewer that joins late starts from the latest moment kept at or before
//! the last final frame, and catches up on the frames since. A frame is
//! final once nothing will run it again: in a net session once every
//! player's input for it is known, elsewhere once it has run. Messages go
//! over any ordered, reliable transport (native pen's is a WebSocket),
//! little-endian, each starting with `MAGIC`, the version and its kind.
use anyhow::{Result, bail, ensure};

use super::wire;
use crate::game::Moment;
use crate::snapshot::SavedState;
use crate::timeline::{Load, Recorded, Timeline};

const MAGIC: [u8; 4] = *b"PSTR";
const VERSION: u8 = 1;
const START: u8 = 1;
const FRAMES: u8 = 2;
const END: u8 = 3;
/// The most frames one message carries.
pub const MAX_FRAMES: usize = 600;
/// A frame's lanes at most: its header and 100 000 events of five lanes.
const MAX_INPUT_FLOATS: usize = 5 + 5 * 100_000;

#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    /// Play from `moment`, which is at `position` of the session.
    Start { position: u64, moment: Moment },
    /// The frames from `first` on.
    Frames { first: u64, frames: Vec<Recorded> },
    /// The session ended: there are no more frames.
    End,
}

fn header(kind: u8) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&[VERSION, kind]);
    out
}

fn put_u32(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&u32::try_from(value).expect("counts the protocol bounds").to_le_bytes());
}

fn put_lanes(out: &mut Vec<u8>, lanes: &[f32]) {
    put_u32(out, lanes.len());
    for lane in lanes {
        out.extend_from_slice(&lane.to_le_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, count: usize) -> Result<&[u8]> {
        let Some(bytes) = self.bytes.get(self.at..self.at + count) else { bail!("stream message ends early") };
        self.at += count;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn lanes(&mut self, most: usize) -> Result<Vec<f32>> {
        let count = self.u32()? as usize;
        ensure!(count <= most, "stream message holds an oversized array");
        let lanes: Vec<f32> =
            self.take(count * 4)?.as_chunks::<4>().0.iter().map(|lane| f32::from_le_bytes(*lane)).collect();
        ensure!(lanes.iter().all(|lane| lane.is_finite()), "stream message holds a nonfinite number");
        Ok(lanes)
    }
}

impl Message {
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Message::Start { position, moment } => {
                let mut out = header(START);
                out.extend_from_slice(&position.to_le_bytes());
                out.extend_from_slice(&wire::encode_moment(moment));
                out
            }
            Message::Frames { first, frames } => {
                let mut out = header(FRAMES);
                out.extend_from_slice(&first.to_le_bytes());
                put_u32(&mut out, frames.len());
                for recorded in frames {
                    put_lanes(&mut out, &recorded.input);
                    out.push(u8::from(recorded.restored));
                    put_u32(&mut out, recorded.loads.len());
                    for load in &recorded.loads {
                        let (kind, saved) = match load {
                            Load::Save(saved) => (0, saved),
                            Load::Scene(saved) => (1, saved),
                        };
                        out.push(kind);
                        put_u32(&mut out, saved.schema as usize);
                        put_lanes(&mut out, &saved.state);
                    }
                }
                out
            }
            Message::End => header(END),
        }
    }

    pub fn decode(bytes: &[u8]) -> Result<Message> {
        let mut input = Reader { bytes, at: 0 };
        ensure!(input.take(4)? == MAGIC, "not a pen stream message");
        ensure!(input.u8()? == VERSION, "a pen stream message of another version");
        let message = match input.u8()? {
            START => {
                let position = input.u64()?;
                return Ok(Message::Start { position, moment: wire::decode_moment(&bytes[input.at..])? });
            }
            FRAMES => {
                let first = input.u64()?;
                let count = input.u32()? as usize;
                ensure!(count <= MAX_FRAMES, "stream message holds too many frames");
                let mut frames = Vec::with_capacity(count);
                for _ in 0..count {
                    let input_lanes = input.lanes(MAX_INPUT_FLOATS)?;
                    ensure!(input_lanes.len() >= 5, "stream message holds an invalid frame");
                    let restored = match input.u8()? {
                        0 => false,
                        1 => true,
                        _ => bail!("stream message holds unknown frame flags"),
                    };
                    let mut loads = Vec::new();
                    for _ in 0..input.u32()? {
                        let kind = input.u8()?;
                        let schema = pen_protocol::boundary::validate_schema(input.u32()? as f32)?;
                        let state = input.lanes(pen_protocol::boundary::MAX_STATE_FLOATS)?;
                        let saved = SavedState { schema, state };
                        loads.push(match kind {
                            0 => Load::Save(saved),
                            1 => Load::Scene(saved),
                            _ => bail!("stream message holds an unknown load"),
                        });
                    }
                    frames.push(Recorded { input: input_lanes, loads, restored });
                }
                Message::Frames { first, frames }
            }
            END => Message::End,
            kind => bail!("unknown pen stream message {kind}"),
        };
        ensure!(input.at == bytes.len(), "stream message runs on past its end");
        Ok(message)
    }
}

/// What a session streams: to each viewer joining, where to start; and to
/// every viewer, each frame once it is final.
#[derive(Debug, Default)]
pub struct Broadcast {
    /// Frames before it went to every viewer.
    sent: Option<u64>,
}

/// The frames from `first` to `end` of `timeline`, in messages of at most
/// `MAX_FRAMES`.
fn frames(timeline: &Timeline, first: u64, end: u64) -> Vec<Message> {
    let mut messages = Vec::new();
    let mut at = first;
    while at < end {
        let frames: Vec<Recorded> =
            (at..end).take(MAX_FRAMES).map_while(|position| timeline.recorded(position).cloned()).collect();
        if frames.is_empty() {
            break;
        }
        let count = frames.len() as u64;
        messages.push(Message::Frames { first: at, frames });
        at += count;
    }
    messages
}

impl Broadcast {
    /// A viewer joining when frames before `end` are final: the latest
    /// moment at or before it, and the frames from there to it. None while
    /// the timeline keeps no moment (it keeps no history).
    pub fn join(&self, timeline: &Timeline, end: u64) -> Option<Vec<Message>> {
        let (at, moment) = timeline.moment_at(end)?;
        let mut messages = vec![Message::Start { position: at, moment: moment.clone() }];
        messages.extend(frames(timeline, at, end));
        Some(messages)
    }

    /// For every viewer: the frames that became final since the last call
    /// (frames before `end` are). When the session went back past what was
    /// sent (time travel), viewers start again from where it is.
    pub fn advance(&mut self, timeline: &Timeline, end: u64) -> Vec<Message> {
        let sent = *self.sent.get_or_insert(end);
        if end < sent {
            self.sent = Some(end);
            return self.join(timeline, end).unwrap_or_default();
        }
        self.sent = Some(end);
        frames(timeline, sent, end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Edits;
    use crate::timeline::Recording;

    fn moment(value: f32) -> Moment {
        Moment { saved: SavedState { schema: 1, state: vec![value] }, edits: Edits::default() }
    }

    fn frame(key: f32) -> Recorded {
        Recorded { input: vec![1.0 / 60.0, 64.0, 64.0, 1.0, 0.0, 1.0, key, 0.0, 0.0, 0.0], ..Recorded::default() }
    }

    #[test]
    fn messages_round_trip() {
        let loaded = Recorded {
            loads: vec![
                Load::Save(SavedState { schema: 3, state: vec![1.0] }),
                Load::Scene(SavedState { schema: 3, state: vec![] }),
            ],
            restored: true,
            ..frame(9.0)
        };
        for message in [
            Message::Start { position: 300, moment: moment(4.0) },
            Message::Frames { first: 300, frames: vec![frame(1.0), loaded, frame(2.0)] },
            Message::End,
        ] {
            let bytes = message.encode();
            assert_eq!(Message::decode(&bytes).unwrap(), message);
            assert!(Message::decode(&bytes[..bytes.len() - 1]).is_err());
        }
        assert!(Message::decode(b"PSTR\x01\x09").is_err());
    }

    #[test]
    fn viewers_start_at_a_moment_and_get_each_final_frame_once() {
        let mut timeline = Timeline::default();
        timeline.keep_history(None);
        for position in 0..100 {
            if timeline.wants_moment() {
                timeline.keep_moment(moment(position as f32));
            }
            timeline.ran(frame(position as f32));
        }
        let mut broadcast = Broadcast::default();
        assert!(broadcast.advance(&timeline, 70).is_empty(), "the first call only says where it stands");
        // A viewer joining at 70 starts at the moment kept at 60.
        let joined = broadcast.join(&timeline, 70).unwrap();
        assert_eq!(joined[0], Message::Start { position: 60, moment: moment(60.0) });
        assert!(matches!(&joined[1], Message::Frames { first: 60, frames } if frames.len() == 10));
        // Then every frame once, as it becomes final.
        let next = broadcast.advance(&timeline, 95);
        assert!(matches!(&next[..], [Message::Frames { first: 70, frames }] if frames.len() == 25));
        assert!(broadcast.advance(&timeline, 95).is_empty());
        // Back in time: viewers start again.
        let again = broadcast.advance(&timeline, 40);
        assert_eq!(again[0], Message::Start { position: 30, moment: moment(30.0) });
        // What a viewer makes of it: a recording from the moment.
        let mut viewer = Timeline::default();
        let Message::Start { position, moment } = joined[0].clone() else { unreachable!() };
        viewer.load_at(Recording { start: moment, frames: Vec::new() }, position);
        for message in joined.into_iter().skip(1).chain(next) {
            let Message::Frames { first, frames } = message else { unreachable!() };
            assert_eq!(first, viewer.head());
            viewer.extend(frames);
        }
        assert_eq!((viewer.position(), viewer.head()), (60, 95));
        assert_eq!(viewer.next(), timeline.recorded(60));
        assert_eq!(Timeline::default().moment_at(10), None);
    }
}
