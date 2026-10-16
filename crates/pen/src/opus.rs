//! Ogg Opus clips (RFC 7845), as the game's music comes. Decoding uses the
//! reference libopus (opusic-sys: vendored, built with CMake), in floating
//! point.
use std::io::Cursor;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use opusic_sys as sys;

use crate::audio::mixer::Clip;

/// Opus always decodes at 48 kHz here; the mixer resamples it to the
/// device's rate.
const RATE: i32 = 48_000;
/// The longest Opus packet: 120 ms.
const MAX_PACKET_SAMPLES: usize = 5760;

/// libopus's decoder state, freed on drop.
struct Decoder(*mut sys::OpusDecoder);

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

pub fn decode(bytes: &[u8]) -> Result<Clip> {
    let mut reader = ogg::PacketReader::new(Cursor::new(bytes));
    let head = reader.read_packet()?.context("empty Ogg stream")?;
    ensure!(head.data.len() >= 19 && head.data.starts_with(b"OpusHead"), "not an Ogg Opus stream");
    let channels = usize::from(head.data[9]);
    ensure!(channels == 1 || channels == 2, "Opus with {channels} channels is not supported");
    ensure!(head.data[18] == 0, "multistream Opus is not supported");
    let pre_skip = usize::from(u16::from_le_bytes([head.data[10], head.data[11]]));
    // Output gain in Q7.8 dB.
    let gain_db = f32::from(i16::from_le_bytes([head.data[16], head.data[17]])) / 256.0;
    let gain = 10f32.powf(gain_db / 20.0);
    let tags = reader.read_packet()?.context("Ogg Opus stream has no tags")?;
    ensure!(tags.data.starts_with(b"OpusTags"), "Ogg Opus stream has no tags");

    let mut decoder = Decoder::new(channels)?;
    let mut pcm = vec![0f32; MAX_PACKET_SAMPLES * channels];
    let mut samples: Vec<f32> = Vec::new();
    let mut end_granule = None;
    while let Some(packet) = reader.read_packet()? {
        let count = decoder.decode(&packet.data, &mut pcm, channels)?;
        samples.extend_from_slice(&pcm[..count * channels]);
        if packet.last_in_stream() {
            end_granule = Some(usize::try_from(packet.absgp_page()).unwrap_or(usize::MAX));
        }
    }
    let decoded = samples.len() / channels;
    // The last page's granule position counts the samples the stream holds,
    // including the pre-skip; anything decoded past it is padding.
    let end = end_granule.map_or(decoded, |granule| granule.min(decoded));
    if pre_skip >= end {
        bail!("Ogg Opus stream is shorter than its pre-skip");
    }
    let frames: Arc<[[f32; 2]]> = (pre_skip..end)
        .map(|index| [samples[index * channels] * gain, samples[index * channels + channels - 1] * gain])
        .collect();
    Ok(Clip::new(frames, f64::from(RATE)))
}

#[cfg(test)]
mod tests {
    /// A quarter second of Ogg Opus (64 kb/s, as ffmpeg's libopus encodes
    /// it): a 440 Hz sine at 0.5 on the left, 660 Hz at 0.25 on the right.
    const TONE: &[u8] = include_bytes!("audio/testdata/tone.opus");

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
