//! What peers send each other: one message a datagram, little-endian, each
//! starting with `MAGIC`, the version and its kind. A datagram that does
//! not decode is dropped; nothing in one is trusted before it is checked.
use std::sync::Arc;

use anyhow::{Result, bail, ensure};
use pen_protocol::boundary::{MAX_STATE_FLOATS, validate_schema, validate_state};

use crate::game::{Edits, Moment};
use crate::snapshot::SavedState;

const MAGIC: [u8; 4] = *b"PNET";
const VERSION: u8 = 1;

/// The size a datagram keeps to, below common path MTUs: inputs past it
/// wait for the next one. A single frame larger than it goes alone.
pub const DATAGRAM_BYTES: usize = 1200;
/// The most frames one input message carries.
pub const MAX_FRAMES: usize = 64;
/// The most bytes a moment's chunk carries.
pub const CHUNK_BYTES: usize = 1024;
/// The largest moment a peer takes from the session's host.
pub const MAX_MOMENT_BYTES: usize = 64 << 20;
/// The most events one player's frame carries over the wire.
pub const MAX_FRAME_EVENTS: usize = 4096;
/// Lanes an event record has (`EVENT_FLOATS`).
const RECORD: usize = pen_protocol::boundary::EVENT_FLOATS;
/// The edit history keeps at most 200 scenes.
const MAX_SCENES: usize = 200;

#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    Hello(Hello),
    Input(Inputs),
    /// Part of the session host's starting moment, for a peer that started
    /// from another one.
    Moment {
        session: u32,
        total: u32,
        offset: u32,
        bytes: Vec<u8>,
    },
    /// The sender leaves the session.
    Bye {
        session: u32,
        player: u8,
    },
}

/// Who calls, and on what terms: sent until every peer has answered. The
/// session host's terms (player 0's) are the session's.
#[derive(Clone, Debug, PartialEq)]
pub struct Hello {
    /// The host's nonce for this session; 0 from a peer that has not heard
    /// the host yet.
    pub session: u32,
    pub player: u8,
    pub players: u8,
    /// Frames between a player's input and the frame it plays in.
    pub delay: u8,
    /// Frames a peer may run past the last frame every input is known for.
    pub window: u8,
    pub fps: f32,
    /// The viewport every peer's game runs with (the host's window).
    pub width: u32,
    pub height: u32,
    /// The game's build: peers run the same one, or they would part ways.
    pub build: u64,
    /// The moment the sender starts from (a hash of it).
    pub start: u64,
    /// The players whose Hello the sender has, as bits.
    pub heard: u32,
}

/// A player's input from frame `first` on, with what the sender knows of
/// the receiver.
#[derive(Clone, Debug, PartialEq)]
pub struct Inputs {
    pub session: u32,
    pub player: u8,
    /// The sender's next frame to run.
    pub frame: u32,
    /// The frames of the receiver's input the sender holds, from 0.
    pub ack: u32,
    /// The sender's clock (milliseconds), the last stamp it had from the
    /// receiver, and how long it held that one, for the round trip.
    pub stamp: u32,
    pub echo: u32,
    pub held: u32,
    /// A frame every input of which is known, and its packet's hash: peers
    /// compare these to catch a game that parted ways (0, 0 for none yet).
    pub checked: u32,
    pub hash: u64,
    pub first: u32,
    /// Each frame's event records, five lanes each.
    pub frames: Vec<Vec<f32>>,
}

struct Writer(Vec<u8>);

impl Writer {
    fn new(kind: u8) -> Writer {
        let mut bytes = Vec::with_capacity(DATAGRAM_BYTES);
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&[VERSION, kind]);
        Writer(bytes)
    }

    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, value: f32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, count: usize) -> Result<&[u8]> {
        let Some(bytes) = self.bytes.get(self.at..self.at + count) else { bail!("datagram ends early") };
        self.at += count;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn f32(&mut self) -> Result<f32> {
        let value = f32::from_le_bytes(self.take(4)?.try_into()?);
        ensure!(value.is_finite(), "datagram holds a nonfinite number");
        Ok(value)
    }

    fn done(&self) -> Result<()> {
        ensure!(self.at == self.bytes.len(), "datagram runs on past its message");
        Ok(())
    }
}

const HELLO: u8 = 1;
const INPUT: u8 = 2;
const MOMENT: u8 = 3;
const BYE: u8 = 4;

/// The bytes an input frame takes on the wire.
pub fn frame_bytes(records: &[f32]) -> usize {
    2 + records.len() * 4
}

impl Message {
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Message::Hello(hello) => {
                let mut out = Writer::new(HELLO);
                out.u32(hello.session);
                out.u8(hello.player);
                out.u8(hello.players);
                out.u8(hello.delay);
                out.u8(hello.window);
                out.f32(hello.fps);
                out.u32(hello.width);
                out.u32(hello.height);
                out.u64(hello.build);
                out.u64(hello.start);
                out.u32(hello.heard);
                out.0
            }
            Message::Input(inputs) => {
                let mut out = Writer::new(INPUT);
                out.u32(inputs.session);
                out.u8(inputs.player);
                out.u32(inputs.frame);
                out.u32(inputs.ack);
                out.u32(inputs.stamp);
                out.u32(inputs.echo);
                out.u32(inputs.held);
                out.u32(inputs.checked);
                out.u64(inputs.hash);
                out.u32(inputs.first);
                out.u8(u8::try_from(inputs.frames.len()).expect("at most MAX_FRAMES frames"));
                for records in &inputs.frames {
                    out.u16(u16::try_from(records.len() / RECORD).expect("at most MAX_FRAME_EVENTS events"));
                    for lane in records {
                        out.f32(*lane);
                    }
                }
                out.0
            }
            Message::Moment { session, total, offset, bytes } => {
                let mut out = Writer::new(MOMENT);
                out.u32(*session);
                out.u32(*total);
                out.u32(*offset);
                out.u16(u16::try_from(bytes.len()).expect("a chunk of at most CHUNK_BYTES"));
                out.0.extend_from_slice(bytes);
                out.0
            }
            Message::Bye { session, player } => {
                let mut out = Writer::new(BYE);
                out.u32(*session);
                out.u8(*player);
                out.0
            }
        }
    }

    pub fn decode(bytes: &[u8]) -> Result<Message> {
        let mut input = Reader { bytes, at: 0 };
        ensure!(input.take(4)? == MAGIC, "not a pen net datagram");
        ensure!(input.u8()? == VERSION, "a pen net datagram of another version");
        let message = match input.u8()? {
            HELLO => Message::Hello(Hello {
                session: input.u32()?,
                player: input.u8()?,
                players: input.u8()?,
                delay: input.u8()?,
                window: input.u8()?,
                fps: input.f32()?,
                width: input.u32()?,
                height: input.u32()?,
                build: input.u64()?,
                start: input.u64()?,
                heard: input.u32()?,
            }),
            INPUT => {
                let (session, player, frame, ack) = (input.u32()?, input.u8()?, input.u32()?, input.u32()?);
                let (stamp, echo, held) = (input.u32()?, input.u32()?, input.u32()?);
                let (checked, hash) = (input.u32()?, input.u64()?);
                let first = input.u32()?;
                let count = usize::from(input.u8()?);
                ensure!(count <= MAX_FRAMES, "datagram holds too many frames");
                let mut frames = Vec::with_capacity(count);
                for _ in 0..count {
                    let events = usize::from(input.u16()?);
                    ensure!(events <= MAX_FRAME_EVENTS, "datagram holds too many events");
                    frames.push((0..events * RECORD).map(|_| input.f32()).collect::<Result<Vec<f32>>>()?);
                }
                Message::Input(Inputs { session, player, frame, ack, stamp, echo, held, checked, hash, first, frames })
            }
            MOMENT => {
                let (session, total, offset) = (input.u32()?, input.u32()?, input.u32()?);
                let length = usize::from(input.u16()?);
                ensure!(length <= CHUNK_BYTES, "datagram holds an oversized chunk");
                let bytes = input.take(length)?.to_vec();
                let (whole, end) = (total as usize, (offset as usize).checked_add(length));
                ensure!(
                    whole <= MAX_MOMENT_BYTES && end.is_some_and(|end| end <= whole),
                    "datagram holds a chunk outside its moment"
                );
                Message::Moment { session, total, offset, bytes }
            }
            BYE => Message::Bye { session: input.u32()?, player: input.u8()? },
            kind => bail!("unknown pen net message {kind}"),
        };
        input.done()?;
        Ok(message)
    }
}

/// A moment as the session host sends it to a peer that started from
/// another: the state's schema and lanes, then the edit history's scenes
/// and which is current (+1, 0 for none).
pub fn encode_moment(moment: &Moment) -> Vec<u8> {
    let lanes = |out: &mut Writer, lanes: &[f32]| {
        out.u32(u32::try_from(lanes.len()).expect("states are bounded below 2^24 lanes"));
        for lane in lanes {
            out.f32(*lane);
        }
    };
    let mut out = Writer::new(MOMENT);
    out.u32(moment.saved.schema);
    lanes(&mut out, &moment.saved.state);
    out.u32(u32::try_from(moment.edits.scenes.len()).expect("the edit history is bounded"));
    out.u32(moment.edits.step.map_or(0, |step| u32::try_from(step + 1).expect("the edit history is bounded")));
    for scene in &moment.edits.scenes {
        lanes(&mut out, scene);
    }
    out.0
}

pub fn decode_moment(bytes: &[u8]) -> Result<Moment> {
    let mut input = Reader { bytes, at: 0 };
    ensure!(input.take(4)? == MAGIC && input.u8()? == VERSION && input.u8()? == MOMENT, "not a pen net moment");
    let lanes = |input: &mut Reader| -> Result<Vec<f32>> {
        let count = input.u32()? as usize;
        ensure!(count <= MAX_STATE_FLOATS, "moment holds an oversized state");
        let lanes = (0..count).map(|_| input.f32()).collect::<Result<Vec<f32>>>()?;
        validate_state(&lanes)?;
        Ok(lanes)
    };
    let schema = validate_schema(input.u32()? as f32)?;
    let state = lanes(&mut input)?;
    let (count, step) = (input.u32()? as usize, input.u32()? as usize);
    ensure!(count <= MAX_SCENES && step <= count, "moment holds an invalid edit history");
    let scenes = (0..count).map(|_| lanes(&mut input).map(Arc::from)).collect::<Result<Vec<Arc<[f32]>>>>()?;
    input.done()?;
    Ok(Moment { saved: SavedState { schema, state }, edits: Edits { scenes, step: step.checked_sub(1) } })
}

/// FNV-1a over bytes: a moment's or a build's hash, the same on every peer.
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hello() -> Hello {
        Hello {
            session: 7,
            player: 1,
            players: 2,
            delay: 2,
            window: 8,
            fps: 60.0,
            width: 960,
            height: 640,
            build: 0x0123_4567_89ab_cdef,
            start: 42,
            heard: 0b11,
        }
    }

    #[test]
    fn messages_round_trip() {
        let inputs = Inputs {
            session: 7,
            player: 1,
            frame: 30,
            ack: 28,
            stamp: 1000,
            echo: 990,
            held: 3,
            checked: 0,
            hash: 0,
            first: 25,
            frames: vec![Vec::new(), vec![1.0, 65.0, 0.0, 0.0, 0.0, 5.0, 0.0, 10.5, 20.25, 0.0], Vec::new()],
        };
        for message in [
            Message::Hello(hello()),
            Message::Input(inputs),
            Message::Moment { session: 7, total: 3000, offset: 2048, bytes: vec![9; 952] },
            Message::Bye { session: 7, player: 1 },
        ] {
            let bytes = message.encode();
            assert_eq!(Message::decode(&bytes).unwrap(), message);
            // Cut short or run on, it is no message.
            assert!(Message::decode(&bytes[..bytes.len() - 1]).is_err());
            let mut longer = bytes.clone();
            longer.push(0);
            assert!(Message::decode(&longer).is_err());
        }
    }

    #[test]
    fn strangers_and_lies_are_refused() {
        let mut bytes = Message::Hello(hello()).encode();
        bytes[0] = b'X';
        assert!(Message::decode(&bytes).is_err(), "another magic");
        let mut bytes = Message::Hello(hello()).encode();
        bytes[4] = VERSION + 1;
        assert!(Message::decode(&bytes).is_err(), "another version");
        assert!(Message::decode(b"PNET\x01\x09").is_err(), "an unknown kind");
        let chunk = Message::Moment { session: 1, total: 10, offset: 8, bytes: vec![0; 4] }.encode();
        assert!(Message::decode(&chunk).is_err(), "a chunk past its moment's end");
        let mut nan = Message::Input(Inputs {
            session: 1,
            player: 0,
            frame: 0,
            ack: 0,
            stamp: 0,
            echo: 0,
            held: 0,
            checked: 0,
            hash: 0,
            first: 0,
            frames: vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]],
        })
        .encode();
        let lane = nan.len() - 4;
        nan[lane..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(Message::decode(&nan).is_err(), "a nonfinite lane");
    }

    #[test]
    fn moments_round_trip() {
        let scenes: Vec<Arc<[f32]>> = vec![vec![4.0].into(), vec![5.0, 6.0].into()];
        let moment = Moment {
            saved: SavedState { schema: 26, state: vec![1.0, -2.5, 3.25] },
            edits: Edits { scenes, step: Some(1) },
        };
        let bytes = encode_moment(&moment);
        assert_eq!(decode_moment(&bytes).unwrap(), moment);
        assert!(decode_moment(&bytes[..bytes.len() - 2]).is_err());
        let empty = Moment { saved: SavedState { schema: 1, state: Vec::new() }, edits: Edits::default() };
        assert_eq!(decode_moment(&encode_moment(&empty)).unwrap(), empty);
        assert_ne!(hash_bytes(&bytes), hash_bytes(&encode_moment(&empty)));
    }
}
