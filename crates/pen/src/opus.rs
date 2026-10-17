//! Ogg Opus clips (RFC 7845), as the game's music comes. Decoding uses the
//! reference libopus (opusic-sys: vendored, built with CMake), in floating
//! point, a packet at a time ([`OpusSource`]): a clip decodes whole by
//! running it to the end, or streams as it plays (`audio::stream`).
use std::io::Cursor;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use opusic_sys as sys;

use crate::audio::source::Source;

/// Opus always decodes at 48 kHz here; the mixer resamples it to the
/// device's rate.
pub const RATE: i32 = 48_000;
/// The longest Opus packet: 120 ms.
const MAX_PACKET_SAMPLES: usize = 5760;

/// libopus's decoder state, freed on drop.
struct Decoder(*mut sys::OpusDecoder);

// SAFETY: the decoder state is plain memory that libopus only touches
// through the calls below, which take `&mut self`: one thread at a time.
unsafe impl Send for Decoder {}

impl Decoder {
    fn new(channels: usize) -> Result<Decoder> {
        let mut error = 0;
        let channels = i32::try_from(channels)?;
        // SAFETY: arguments are a supported rate and channel count; libopus
        // returns null and sets `error` on failure.
        let state = unsafe { sys::opus_decoder_create(RATE, channels, &mut error) };
        ensure!(!state.is_null() && error == sys::OPUS_OK, "creating an Opus decoder failed ({error})");
        Ok(Decoder(state))
    }

    /// Decodes one packet into `pcm`; returns samples per channel.
    fn decode(&mut self, packet: &[u8], pcm: &mut [f32], channels: usize) -> Result<usize> {
        let length = i32::try_from(packet.len()).context("an Opus packet is longer than libopus reads")?;
        let frames = i32::try_from(pcm.len() / channels)?;
        // SAFETY: `pcm` holds MAX_PACKET_SAMPLES frames of `channels`
        // samples, as the frame size passed says.
        let count = unsafe { sys::opus_decode_float(self.0, packet.as_ptr(), length, pcm.as_mut_ptr(), frames, 0) };
        usize::try_from(count).ok().with_context(|| format!("decoding Opus failed ({count})"))
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: created by opus_decoder_create and freed once.
        unsafe { sys::opus_decoder_destroy(self.0) }
    }
}

pub fn is_ogg_opus(bytes: &[u8]) -> bool {
    bytes.starts_with(b"OggS") && bytes.windows(8).take(256).any(|window| window == b"OpusHead")
}

/// What an Ogg Opus stream's head and last page say of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Head {
    pub channels: usize,
    pub pre_skip: usize,
    /// The output gain, linear.
    pub gain: f32,
    /// The samples it holds past the pre-skip, by its last page's granule
    /// position.
    pub frames: usize,
}

/// The granule position of the stream's last page that has one: the
/// samples it holds, the pre-skip included.
fn last_granule(bytes: &[u8]) -> Option<u64> {
    let mut at = 0;
    let mut last = None;
    while let Some(header) = bytes.get(at..at + 27) {
        if !header.starts_with(b"OggS") {
            break;
        }
        let segments = usize::from(header[26]);
        let table = bytes.get(at + 27..at + 27 + segments)?;
        let granule = u64::from_le_bytes(header[6..14].try_into().ok()?);
        if granule != u64::MAX {
            last = Some(granule);
        }
        at += 27 + segments + table.iter().map(|&size| usize::from(size)).sum::<usize>();
    }
    last
}

pub fn head(bytes: &[u8]) -> Result<Head> {
    let mut reader = ogg::PacketReader::new(Cursor::new(bytes));
    let head = reader.read_packet()?.context("empty Ogg stream")?;
    ensure!(head.data.len() >= 19 && head.data.starts_with(b"OpusHead"), "not an Ogg Opus stream");
    let channels = usize::from(head.data[9]);
    ensure!(channels == 1 || channels == 2, "Opus with {channels} channels is not supported");
    ensure!(head.data[18] == 0, "multistream Opus is not supported");
    let pre_skip = usize::from(u16::from_le_bytes([head.data[10], head.data[11]]));
    // Output gain in Q7.8 dB.
    let gain_db = f32::from(i16::from_le_bytes([head.data[16], head.data[17]])) / 256.0;
    let granule = last_granule(bytes).context("Ogg Opus stream has no granule position")?;
    let total = usize::try_from(granule).unwrap_or(usize::MAX);
    if pre_skip >= total {
        bail!("Ogg Opus stream is shorter than its pre-skip");
    }
    Ok(Head { channels, pre_skip, gain: 10f32.powf(gain_db / 20.0), frames: total - pre_skip })
}

/// An Ogg Opus stream decoded a packet at a time: its pre-skip left out,
/// and nothing past the samples its last page says it holds.
pub struct OpusSource {
    reader: ogg::PacketReader<Cursor<Arc<[u8]>>>,
    decoder: Decoder,
    head: Head,
    pcm: Vec<f32>,
    /// Pre-skip samples still to leave out.
    skip: usize,
    /// Samples still to give.
    left: usize,
}

impl OpusSource {
    pub fn new(bytes: Arc<[u8]>) -> Result<OpusSource> {
        let head = head(&bytes)?;
        let mut reader = ogg::PacketReader::new(Cursor::new(bytes));
        reader.read_packet()?.context("empty Ogg stream")?;
        let tags = reader.read_packet()?.context("Ogg Opus stream has no tags")?;
        ensure!(tags.data.starts_with(b"OpusTags"), "Ogg Opus stream has no tags");
        Ok(OpusSource {
            reader,
            decoder: Decoder::new(head.channels)?,
            pcm: vec![0f32; MAX_PACKET_SAMPLES * head.channels],
            skip: head.pre_skip,
            left: head.frames,
            head,
        })
    }
}

impl Source for OpusSource {
    fn next(&mut self, out: &mut Vec<[f32; 2]>) -> Result<bool> {
        let channels = self.head.channels;
        while self.left > 0 {
            let Some(packet) = self.reader.read_packet()? else { return Ok(false) };
            let count = self.decoder.decode(&packet.data, &mut self.pcm, channels)?;
            let skipped = self.skip.min(count);
            self.skip -= skipped;
            let given = (count - skipped).min(self.left);
            self.left -= given;
            let gain = self.head.gain;
            out.extend(
                self.pcm[skipped * channels..(skipped + given) * channels]
                    .chunks_exact(channels)
                    .map(|frame| [frame[0] * gain, frame[channels - 1] * gain]),
            );
            if given > 0 {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// A whole clip, decoded.
#[cfg(test)]
pub fn decode(bytes: &[u8]) -> Result<crate::audio::mixer::Clip> {
    let mut source = OpusSource::new(bytes.into())?;
    let mut frames = Vec::new();
    while source.next(&mut frames)? {}
    Ok(crate::audio::mixer::Clip::new(frames.into(), f64::from(RATE)))
}

#[cfg(test)]
mod tests {
    use crate::audio::source::Source;

    /// A quarter second of Ogg Opus (64 kb/s, as ffmpeg's libopus encodes
    /// it): a 440 Hz sine at 0.5 on the left, 660 Hz at 0.25 on the right.
    const TONE: &[u8] = include_bytes!("audio/testdata/tone.opus");

    /// Its head says how long it is before decoding anything, and it
    /// decodes a packet at a time to just that long.
    #[test]
    fn opus_streams_a_packet_at_a_time() {
        let head = super::head(TONE).unwrap();
        assert_eq!((head.channels, head.frames), (2, 12_000));
        let mut source = super::OpusSource::new(TONE.into()).unwrap();
        let mut frames = Vec::new();
        let mut packets = 0;
        while source.next(&mut frames).unwrap() {
            packets += 1;
        }
        assert!(packets > 10, "{packets}");
        assert_eq!(frames.len(), head.frames);
    }

    /// The stream decodes at 48 kHz to exactly as long as it says (its
    /// pre-skip and end padding cut), and to the sines it was made of, to
    /// within what the codec loses.
    #[test]
    fn opus_decodes_its_samples() {
        assert!(super::is_ogg_opus(TONE));
        let clip = super::decode(TONE).unwrap();
        assert_eq!((clip.rate, clip.mono, clip.frames.len()), (48_000.0, false, 12_000));
        for (side, (amplitude, frequency)) in [(0.5, 440.0), (0.25, 660.0)].into_iter().enumerate() {
            let (mut signal, mut noise) = (0.0, 0.0);
            // Past the codec's first frames, which fade in.
            for (index, frame) in clip.frames.iter().enumerate().skip(960) {
                let expected = amplitude * (std::f64::consts::TAU * frequency * index as f64 / 48_000.0).sin();
                signal += expected * expected;
                noise += (f64::from(frame[side]) - expected).powi(2);
            }
            let snr = 10.0 * (signal / noise).log10();
            assert!(snr > 30.0, "side {side}: {snr:.1} dB");
        }
    }

    /// Writes the bundle's music as raw f32 stereo, for comparing with ffmpeg.
    #[test]
    #[ignore]
    fn dump_music() {
        let assets = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dist/gdev/assets");
        let data = super::decode(&std::fs::read(assets.join("music.ogg")).unwrap()).unwrap();
        let out = std::env::var("OPUS_DUMP").unwrap();
        let bytes: Vec<u8> = data.frames.iter().flatten().flat_map(|sample| sample.to_le_bytes()).collect();
        std::fs::write(out, bytes).unwrap();
    }
}
