//! What a sound clip's header says of it before any of it is decoded: its
//! rate, channels and length, for PCM WAV and Ogg Opus, the formats the
//! game's sounds come in. Both hosts decide by it which clips stream
//! ([`streams`]): those too long to hold decoded play as they decode.

/// Clips longer than this many seconds stream: only their bytes stay in
/// memory, and they decode as they play.
pub const STREAM_SECONDS: f64 = 12.0;

/// A clip as its header has it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Header {
    /// Its length in frames at its own rate.
    pub frames: u64,
    pub rate: u32,
    pub channels: u16,
}

impl Header {
    pub fn seconds(&self) -> f64 {
        self.frames as f64 / f64::from(self.rate)
    }
}

fn u16_at(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

/// A RIFF WAVE file's `fmt ` and `data` chunks.
fn wav(bytes: &[u8]) -> Option<Header> {
    if bytes.get(..4)? != b"RIFF" || bytes.get(8..12)? != b"WAVE" {
        return None;
    }
    let (mut at, mut format, mut data) = (12, None, None);
    while let (Some(id), Some(size)) = (bytes.get(at..at + 4), u32_at(bytes, at + 4)) {
        let size = usize::try_from(size).ok()?;
        match id {
            b"fmt " => format = Some((u16_at(bytes, at + 10)?, u32_at(bytes, at + 12)?, u16_at(bytes, at + 20)?)),
            // A data chunk may claim more than the file holds.
            b"data" => data = Some(size.min(bytes.len().saturating_sub(at + 8))),
            _ => {}
        }
        at += 8 + size + size % 2;
    }
    let ((channels, rate, block), data) = (format?, data?);
    (block > 0 && rate > 0).then(|| Header { frames: (data / usize::from(block)) as u64, rate, channels })
}

/// An Ogg Opus stream's head, and the granule position of its last page.
fn opus(bytes: &[u8]) -> Option<Header> {
    if bytes.get(..4)? != b"OggS" {
        return None;
    }
    let (mut at, mut head, mut granule) = (0, None, None);
    while let Some(page) = bytes.get(at..at + 27).filter(|page| page.starts_with(b"OggS")) {
        let segments = usize::from(page[26]);
        let table = bytes.get(at + 27..at + 27 + segments)?;
        let body = at + 27 + segments;
        if head.is_none() && bytes.get(body..body + 8) == Some(b"OpusHead") {
            head = Some((u16::from(*bytes.get(body + 9)?), u16_at(bytes, body + 10)?));
        }
        let position = u64::from_le_bytes(page[6..14].try_into().ok()?);
        if position != u64::MAX {
            granule = Some(position);
        }
        at = body + table.iter().map(|&size| usize::from(size)).sum::<usize>();
    }
    let (channels, pre_skip) = head?;
    Some(Header { frames: granule?.checked_sub(u64::from(pre_skip))?, rate: 48_000, channels })
}

/// The clip's header, if it is a PCM WAV or Ogg Opus clip.
pub fn header(bytes: &[u8]) -> Option<Header> {
    wav(bytes).or_else(|| opus(bytes))
}

/// Whether the clip streams: it is longer than STREAM_SECONDS.
pub fn streams(bytes: &[u8]) -> bool {
    header(bytes).is_some_and(|header| header.seconds() > STREAM_SECONDS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav_bytes(rate: u32, channels: u16, frames: u32, extra_chunk: bool) -> Vec<u8> {
        let block = channels * 2;
        let size = frames * u32::from(block);
        let mut bytes = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
        bytes.extend_from_slice(&block.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        if extra_chunk {
            bytes.extend_from_slice(b"LIST\x03\0\0\0abc\0");
        }
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.resize(bytes.len() + size as usize, 0);
        bytes
    }

    #[test]
    fn wav_headers_say_how_long() {
        let header = super::header(&wav_bytes(22_050, 2, 1000, true)).unwrap();
        assert_eq!(header, Header { frames: 1000, rate: 22_050, channels: 2 });
        assert!(!streams(&wav_bytes(22_050, 1, 22_050 * 12, false)));
        assert!(streams(&wav_bytes(22_050, 1, 22_050 * 12 + 1, false)));
        // A data chunk claiming more than the file holds is as long as it is.
        let mut cut = wav_bytes(48_000, 1, 100, false);
        cut.truncate(cut.len() - 50);
        assert_eq!(super::header(&cut).unwrap().frames, 75);
        assert_eq!(super::header(b"not a clip"), None);
    }

    /// A quarter second of Ogg Opus: its head's channels, and its last
    /// granule less its pre-skip.
    #[test]
    fn opus_headers_say_how_long() {
        let header = super::header(include_bytes!("../../pen/src/audio/testdata/tone.opus")).unwrap();
        assert_eq!(header, Header { frames: 12_000, rate: 48_000, channels: 2 });
        assert!(!streams(include_bytes!("../../pen/src/audio/testdata/tone.opus")));
    }
}
