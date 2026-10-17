//! Input recordings (`.penrec`) on disk, in `pen_runtime::recording`'s
//! format, which the browser plays too. Frames are written as they run, so
//! a recording cut short (a crash) still plays up to its last whole frame;
//! a rewind while recording cuts the file back to the frame rewound to.
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use pen_runtime::recording::{encode_frame, encode_moment};

use crate::game::Moment;
use crate::timeline::{Recorded, Recording};

/// A recording's bytes (a bug report's too), saying when it was cut short.
pub(crate) fn decode(bytes: &[u8]) -> Result<Recording> {
    let decoded = pen_runtime::recording::decode(bytes)?;
    if decoded.cut {
        eprintln!(
            "the recording ends partway through a frame; it plays {} whole frames",
            decoded.recording.frames.len()
        );
    }
    Ok(decoded.recording)
}

/// A recording, its moment and all its frames, as a file holds it.
pub fn encode(recording: &Recording) -> Vec<u8> {
    let mut bytes = encode_moment(&recording.start);
    for frame in &recording.frames {
        bytes.extend(encode_frame(frame));
    }
    bytes
}

/// Reads a recording, or the one a bug report (`bugreport`) holds.
pub fn read(path: &Path) -> Result<Recording> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let bytes = if crate::bugreport::is_report(&bytes) {
        crate::bugreport::recording_bytes(&bytes).with_context(|| format!("reading {}", path.display()))?
    } else {
        bytes
    };
    decode(&bytes).with_context(|| format!("reading {}", path.display()))
}

/// The viewport a recording's first frame had, in pixels.
pub fn viewport(path: &Path) -> Result<Option<(u32, u32)>> {
    Ok(pen_runtime::recording::viewport(&read(path)?))
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
    use crate::snapshot::SavedState;
    use crate::timeline::Load;
    use pen_runtime::game::Edits;
    use std::sync::Arc;

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
        assert_eq!(decode(&encode(&recording)).unwrap(), recording);

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
