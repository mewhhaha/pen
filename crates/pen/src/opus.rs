//! Ogg Opus clips (RFC 7845), which kira's decoders do not read: gdev's
//! music is Opus, as browsers decode it. Decoding uses the reference libopus
//! (vendored, built with CMake).
use std::io::Cursor;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use kira::Frame;
use kira::sound::static_sound::{StaticSoundData, StaticSoundSettings};
use opus_head_sys as sys;

/// Opus always decodes at 48 kHz here; kira resamples to the device.
const RATE: u32 = 48_000;
/// The longest Opus packet: 120 ms.
const MAX_PACKET_SAMPLES: usize = 5760;

/// libopus's decoder state, freed on drop.
struct Decoder(*mut sys::OpusDecoder);

impl Decoder {
    fn new(channels: usize) -> Result<Decoder> {
        let mut error = 0;
        // SAFETY: arguments are a supported rate and channel count; libopus
        // returns null and sets `error` on failure.
        let state = unsafe { sys::opus_decoder_create(RATE as i32, channels as i32, &mut error) };
        ensure!(!state.is_null() && error == sys::OPUS_OK as i32, "creating an Opus decoder failed ({error})");
        Ok(Decoder(state))
    }

    /// Decodes one packet into `pcm`; returns samples per channel.
    fn decode(&mut self, packet: &[u8], pcm: &mut [f32], channels: usize) -> Result<usize> {
        // SAFETY: `pcm` holds MAX_PACKET_SAMPLES frames of `channels`
        // samples, as the frame size passed says.
        let count = unsafe {
            sys::opus_decode_float(
                self.0,
                packet.as_ptr(),
                packet.len() as i32,
                pcm.as_mut_ptr(),
                (pcm.len() / channels) as i32,
                0,
            )
        };
        ensure!(count >= 0, "decoding Opus failed ({count})");
        Ok(count as usize)
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

pub fn decode(bytes: &[u8]) -> Result<StaticSoundData> {
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
            end_granule = Some(packet.absgp_page() as usize);
        }
    }
    let decoded = samples.len() / channels;
    // The last page's granule position counts the samples the stream holds,
    // including the pre-skip; anything decoded past it is padding.
    let end = end_granule.map_or(decoded, |granule| granule.min(decoded));
    if pre_skip >= end {
        bail!("Ogg Opus stream is shorter than its pre-skip");
    }
    let frames: Arc<[Frame]> = (pre_skip..end)
        .map(|index| {
            let left = samples[index * channels] * gain;
            let right = samples[index * channels + channels - 1] * gain;
            Frame { left, right }
        })
        .collect();
    Ok(StaticSoundData { sample_rate: RATE, frames, settings: StaticSoundSettings::default(), slice: None })
}

#[cfg(test)]
mod tests {
    /// Writes the bundle's music as raw f32 stereo, for comparing with ffmpeg.
    #[test]
    #[ignore]
    fn dump_music() {
        let assets = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dist/gdev/assets");
        let data = super::decode(&std::fs::read(assets.join("music.ogg")).unwrap()).unwrap();
        let out = std::env::var("OPUS_DUMP").unwrap();
        let bytes: Vec<u8> = data
            .frames
            .iter()
            .flat_map(|frame| [frame.left, frame.right])
            .flat_map(f32::to_le_bytes)
            .collect();
        std::fs::write(out, bytes).unwrap();
    }
}
