//! Input recordings (`.penrec`): the moment a session started from, then
//! every frame's input exactly as the game received it, so a replay gives
//! the same packets bit for bit. All little-endian:
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
//! Lanes are a count (u32) and that many f32s. Frames are written as they
//! run, so a recording cut short (a crash) still plays up to its last whole
//! frame; a rewind while recording cuts the file back to the frame rewound to.
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
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

fn encode_moment(moment: &Moment) -> Vec<u8> {
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

fn encode_frame(recorded: &Recorded) -> Vec<u8> {
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

/// Reads a recording's fields, failing past its end.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    /// A read ran past the end.
    short: bool,
}

impl Reader<'_> {
    fn take(&mut self, count: usize) -> Result<&[u8]> {
        let Some(bytes) = self.bytes.get(self.at..self.at + count) else {
            self.short = true;
            bail!("recording ends early");
        };
        self.at += count;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn lanes(&mut self, most: usize) -> Result<Vec<f32>> {
        let count = self.u32()? as usize;
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

pub fn decode(bytes: &[u8]) -> Result<Recording> {
    ensure!(bytes.starts_with(MAGIC), "not a pen input recording");
    let mut reader = Reader { bytes, at: MAGIC.len(), short: false };
    ensure!(reader.u32()? == VERSION, "unsupported pen input recording version");
    let saved = reader.saved()?;
    let count = reader.u32()? as usize;
    let step = reader.u32()? as usize;
    ensure!(count <= MAX_SCENES && step <= count, "recording holds an invalid edit history");
    let mut scenes = Vec::with_capacity(count);
    for _ in 0..count {
        let scene = reader.lanes(MAX_STATE_FLOATS)?;
        validate_state(&scene)?;
        scenes.push(Arc::<[f32]>::from(scene));
    }
    let start = Moment { saved, edits: Edits { scenes, step: step.checked_sub(1) } };
    let mut frames = Vec::new();
    while reader.at < bytes.len() {
        match reader.frame() {
            Ok(frame) => frames.push(frame),
            // The last frame was being written when the recording stopped.
            Err(_) if reader.short => {
                eprintln!("the recording ends partway through a frame; it plays {} whole frames", frames.len());
                break;
            }
            Err(error) => return Err(error.context(format!("frame {} of the recording", frames.len()))),
        }
    }
    Ok(Recording { start, frames })
}

pub fn read(path: &Path) -> Result<Recording> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    decode(&bytes).with_context(|| format!("reading {}", path.display()))
}

/// The viewport a recording's first frame had, in pixels.
pub fn viewport(path: &Path) -> Result<Option<(u32, u32)>> {
    let recording = read(path)?;
    Ok(recording.frames.first().and_then(|frame| whole(frame.input[1]).zip(whole(frame.input[2]))))
}

/// Writes a recording as frames run.
pub struct Writer {
    path: PathBuf,
    file: BufWriter<File>,
    /// The position the recording starts at.
    first: u64,
    /// Where each frame starts in the file.
    starts: Vec<u64>,
    length: u64,
}

impl Writer {
    /// Starts a recording at `position`, from `start`.
    pub fn create(path: &Path, position: u64, start: &Moment) -> Result<Writer> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let header = encode_moment(start);
        let mut file = BufWriter::new(File::create(path).with_context(|| format!("writing {}", path.display()))?);
        file.write_all(&header)?;
        Ok(Writer { path: path.to_owned(), file, first: position, starts: Vec::new(), length: header.len() as u64 })
    }

    pub fn frame(&mut self, recorded: &Recorded) -> Result<()> {
        let bytes = encode_frame(recorded);
        self.starts.push(self.length);
        self.file.write_all(&bytes).with_context(|| format!("writing {}", self.path.display()))?;
        self.length += bytes.len() as u64;
        Ok(())
    }

    /// Forgets the frames from `position` on: the game went back to it.
    pub fn rewind(&mut self, position: u64) -> Result<()> {
        ensure!(position >= self.first, "cannot rewind before the input recording started");
        let Ok(index) = usize::try_from(position - self.first) else { return Ok(()) };
        let Some(&start) = self.starts.get(index) else { return Ok(()) };
        self.file.flush()?;
        self.file.get_mut().set_len(start)?;
        self.file.seek(SeekFrom::Start(start))?;
        self.starts.truncate(index);
        self.length = start;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.file.flush().with_context(|| format!("writing {}", self.path.display()))?;
        eprintln!("recorded {} frames of input to {}", self.starts.len(), self.path.display());
        Ok(())
    }
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

    fn written(recording: &Recording, path: &Path) -> Writer {
        let mut writer = Writer::create(path, 10, &recording.start).unwrap();
        for frame in &recording.frames {
            writer.frame(frame).unwrap();
        }
        writer
    }

    #[test]
    fn round_trips_and_rewinds() {
        let directory = std::env::temp_dir().join(format!("pen-recording-{}", std::process::id()));
        let path = directory.join("session.penrec");
        let recording = sample();
        written(&recording, &path).finish().unwrap();
        assert_eq!(read(&path).unwrap(), recording);

        // Back to the second frame, then on differently.
        let mut writer = written(&recording, &path);
        writer.rewind(11).unwrap();
        let other = Recorded { input: vec![0.5, 1.0, 1.0, 0.0, 0.0], ..Recorded::default() };
        writer.frame(&other).unwrap();
        writer.finish().unwrap();
        let mut expected = recording.clone();
        expected.frames.truncate(1);
        expected.frames.push(other);
        assert_eq!(read(&path).unwrap(), expected);
        assert!(written(&recording, &path).rewind(9).is_err());

        // Cut short mid-frame, it plays its whole frames.
        let bytes = std::fs::read(&path).unwrap();
        written(&recording, &path).finish().unwrap();
        let full = std::fs::read(&path).unwrap();
        assert_eq!(decode(&full[..full.len() - 3]).unwrap().frames.len(), 2);
        assert!(decode(&bytes[..10]).is_err());
        let _ = std::fs::remove_dir_all(directory);
    }
}
