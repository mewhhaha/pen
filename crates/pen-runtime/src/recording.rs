//! Input recordings (`.penrec`): the moment a session started from, then
//! every frame's input exactly as the game received it, so a replay gives
//! the same packets bit for bit. Native pen writes them as frames run
//! (`--record-input`, the dev tools' `/record`) and plays them
//! (`--replay`); the browser plays one from a link. All little-endian:
//!
//! - "PENREC\0\0", version 1 (u32);
//! - the moment: schema (u32), the state (lanes), the edit history's scene
//!   count (u32), current scene + 1 (u32, 0 for none), each scene (lanes);
//! - each frame until the end of the file: its `encode_frame` payload
//!   (lanes), flags (u32: 1 when the game was restored from its own state
//!   just before it, as a rewind does), the count of states it loaded
//!   (u32), each as its kind (u32: 0 quicksave, 1 scene), schema (u32) and
//!   state (lanes).
//!
//! Lanes are a count (u32) and that many f32s. A recording cut short (a
//! crash while it was written) plays up to its last whole frame.
//!
//! A link carries a recording in its fragment (`#replay=...`, see
//! [`fragment`]): deflated and in URL-safe base64, so a page can play it
//! without the recording being hosted anywhere.
use std::sync::Arc;

use anyhow::{Result, bail, ensure};
use pen_protocol::boundary::{MAX_STATE_FLOATS, validate_schema, validate_state, whole};

use crate::game::{Edits, Moment};
use crate::snapshot::SavedState;
use crate::timeline::{Load, Recorded, Recording};

const MAGIC: &[u8; 8] = b"PENREC\0\0";
const VERSION: u32 = 1;
/// A frame's events are five lanes each, and at most 100 000 of them.
const MAX_INPUT_FLOATS: usize = 5 + 5 * 100_000;
/// The edit history keeps at most 200 scenes.
const MAX_SCENES: usize = 200;
/// The largest recording a link may unpack to.
const MAX_LINKED_BYTES: usize = 256 << 20;

/// Whether `bytes` start as a recording does.
pub fn is_recording(bytes: &[u8]) -> bool {
    bytes.starts_with(MAGIC)
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// A count of lanes, scenes or loads, all bounded far below 2^32.
fn put_count(out: &mut Vec<u8>, count: usize) {
    put_u32(out, u32::try_from(count).expect("a count the protocol bounds"));
}

fn put_lanes(out: &mut Vec<u8>, lanes: &[f32]) {
    put_count(out, lanes.len());
    for lane in lanes {
        out.extend_from_slice(&lane.to_le_bytes());
    }
}

fn put_saved(out: &mut Vec<u8>, saved: &SavedState) {
    put_u32(out, saved.schema);
    put_lanes(out, &saved.state);
}

/// The start of a recording: its header and the moment it starts from.
pub fn encode_moment(moment: &Moment) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    put_u32(&mut out, VERSION);
    put_saved(&mut out, &moment.saved);
    put_count(&mut out, moment.edits.scenes.len());
    put_count(&mut out, moment.edits.step.map_or(0, |step| step + 1));
    for scene in &moment.edits.scenes {
        put_lanes(&mut out, scene);
    }
    out
}

/// One frame of a recording, which follows its moment and the frames before.
pub fn encode_frame(recorded: &Recorded) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + recorded.input.len() * 4);
    put_lanes(&mut out, &recorded.input);
    put_u32(&mut out, u32::from(recorded.restored));
    put_count(&mut out, recorded.loads.len());
    for load in &recorded.loads {
        let (kind, saved) = match load {
            Load::Save(saved) => (0, saved),
            Load::Scene(saved) => (1, saved),
        };
        put_u32(&mut out, kind);
        put_saved(&mut out, saved);
    }
    out
}

/// A whole recording, as a `.penrec` file holds it.
pub fn encode(recording: &Recording) -> Vec<u8> {
    let mut out = encode_moment(&recording.start);
    for frame in &recording.frames {
        out.extend(encode_frame(frame));
    }
    out
}

/// Reads a recording's fields, failing past its end.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    /// A read ran past the end.
    short: bool,
}

impl Reader<'_> {
    fn take(&mut self, count: usize) -> Result<&[u8]> {
        let Some(bytes) = self.at.checked_add(count).and_then(|end| self.bytes.get(self.at..end)) else {
            self.short = true;
            bail!("recording ends early");
        };
        self.at += count;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn count(&mut self) -> Result<usize> {
        Ok(usize::try_from(self.u32()?)?)
    }

    fn lanes(&mut self, most: usize) -> Result<Vec<f32>> {
        let count = self.count()?;
        ensure!(count <= most, "recording holds an oversized array");
        Ok(self.take(count * 4)?.as_chunks::<4>().0.iter().map(|lane| f32::from_le_bytes(*lane)).collect())
    }

    fn saved(&mut self) -> Result<SavedState> {
        let schema = validate_schema(self.u32()? as f32)?;
        let state = self.lanes(MAX_STATE_FLOATS)?;
        validate_state(&state)?;
        Ok(SavedState { schema, state })
    }

    fn frame(&mut self) -> Result<Recorded> {
        let input = self.lanes(MAX_INPUT_FLOATS)?;
        ensure!(input.len() >= 5 && input.iter().all(|lane| lane.is_finite()), "recording holds an invalid frame");
        let flags = self.u32()?;
        ensure!(flags <= 1, "recording holds unknown frame flags");
        let count = self.u32()?;
        let mut loads = Vec::new();
        for _ in 0..count {
            loads.push(match self.u32()? {
                0 => Load::Save(self.saved()?),
                1 => Load::Scene(self.saved()?),
                kind => bail!("recording holds an unknown load kind {kind}"),
            });
        }
        Ok(Recorded { input, loads, restored: flags == 1 })
    }
}

/// A recording read back.
#[derive(Debug)]
pub struct Decoded {
    pub recording: Recording,
    /// Its last frame was cut short (it was being written when the
    /// recording stopped), and is left out.
    pub cut: bool,
}

pub fn decode(bytes: &[u8]) -> Result<Decoded> {
    ensure!(is_recording(bytes), "not a pen input recording");
    let mut reader = Reader { bytes, at: MAGIC.len(), short: false };
    ensure!(reader.u32()? == VERSION, "unsupported pen input recording version");
    let saved = reader.saved()?;
    let count = reader.count()?;
    let step = reader.count()?;
    ensure!(count <= MAX_SCENES && step <= count, "recording holds an invalid edit history");
    let mut scenes = Vec::with_capacity(count);
    for _ in 0..count {
        let scene = reader.lanes(MAX_STATE_FLOATS)?;
        validate_state(&scene)?;
        scenes.push(Arc::<[f32]>::from(scene));
    }
    let start = Moment { saved, edits: Edits { scenes, step: step.checked_sub(1) } };
    let mut frames = Vec::new();
    let mut cut = false;
    while reader.at < bytes.len() {
        match reader.frame() {
            Ok(frame) => frames.push(frame),
            Err(_) if reader.short => {
                cut = true;
                break;
            }
            Err(error) => return Err(error.context(format!("frame {} of the recording", frames.len()))),
        }
    }
    Ok(Decoded { recording: Recording { start, frames }, cut })
}

/// The viewport the recording's first frame had, in pixels.
pub fn viewport(recording: &Recording) -> Option<(u32, u32)> {
    recording.frames.first().and_then(|frame| whole(frame.input[1]).zip(whole(frame.input[2])))
}

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// A link's fragment value for a recording's bytes: deflated, then in
/// URL-safe base64 without padding.
pub fn fragment(bytes: &[u8]) -> String {
    let deflated = miniz_oxide::deflate::compress_to_vec(bytes, 9);
    let mut out = String::with_capacity(deflated.len().div_ceil(3) * 4);
    for chunk in deflated.chunks(3) {
        let group =
            chunk.iter().enumerate().fold(0u32, |group, (index, byte)| group | u32::from(*byte) << (16 - 8 * index));
        for index in 0..=chunk.len() {
            out.push(char::from(BASE64[usize::try_from(group >> (18 - 6 * index) & 63).unwrap()]));
        }
    }
    out
}

/// A recording's bytes from a link's fragment value.
pub fn from_fragment(text: &str) -> Result<Vec<u8>> {
    let mut deflated = Vec::with_capacity(text.len() / 4 * 3 + 2);
    let mut group = 0u32;
    let mut bits = 0;
    for byte in text.bytes() {
        let Some(value) = BASE64.iter().position(|digit| *digit == byte) else {
            bail!("the link's recording is not URL-safe base64");
        };
        group = group << 6 | u32::try_from(value)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            deflated.push(u8::try_from(group >> bits & 0xff)?);
        }
    }
    miniz_oxide::inflate::decompress_to_vec_with_limit(&deflated, MAX_LINKED_BYTES)
        .map_err(|error| anyhow::anyhow!("the link's recording does not unpack: {:?}", error.status))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Recording {
        let saved = SavedState { schema: 7, state: vec![1.0, -2.5, 3.25] };
        let scenes: Vec<Arc<[f32]>> = vec![vec![4.0].into(), vec![5.0, 6.0].into()];
        Recording {
            start: Moment { saved: saved.clone(), edits: Edits { scenes, step: Some(1) } },
            frames: vec![
                Recorded { input: vec![0.0, 960.0, 640.0, 0.0, 0.0], ..Recorded::default() },
                Recorded {
                    input: vec![0.016, 960.0, 640.0, 1.0, 0.0, 1.0, 9.0, 0.0, 0.0, 0.0],
                    loads: vec![Load::Save(saved.clone()), Load::Scene(SavedState { schema: 7, state: vec![] })],
                    restored: true,
                },
                Recorded { input: vec![0.017, 960.0, 640.0, 0.0, 0.0], ..Recorded::default() },
            ],
        }
    }

    #[test]
    fn round_trips_and_plays_what_was_cut_short_whole() {
        let recording = sample();
        let bytes = encode(&recording);
        let decoded = decode(&bytes).unwrap();
        assert_eq!((decoded.recording, decoded.cut), (recording.clone(), false));
        let cut = decode(&bytes[..bytes.len() - 3]).unwrap();
        assert_eq!((cut.recording.frames.len(), cut.cut), (2, true));
        assert!(decode(&bytes[..10]).is_err());
        assert!(decode(b"PENREC\0\0\x02\0\0\0").is_err());
        assert_eq!(viewport(&recording), Some((960, 640)));
    }

    #[test]
    fn links_carry_recordings() {
        let bytes = encode(&sample());
        let text = fragment(&bytes);
        assert!(text.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_'), "{text}");
        assert_eq!(from_fragment(&text).unwrap(), bytes);
        // Every length of the last base64 group.
        for length in 0..8 {
            let bytes: Vec<u8> = (0..length).map(|index| index * 37).collect();
            assert_eq!(from_fragment(&fragment(&bytes)).unwrap(), bytes);
        }
        assert!(from_fragment("not base64!").is_err());
        assert!(from_fragment("AAAA").is_err());
    }
}
