//! A clip's encoded bytes and how to decode them a piece at a time: PCM WAV
//! through symphonia, Ogg Opus through libopus (`opus.rs`). A clip decodes
//! whole by running its source to the end; one too long to hold decoded
//! (`pen_runtime::clip::streams`) keeps only its bytes, and streams as it
//! plays (`stream.rs`).
use std::io::Cursor;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use symphonia::core::codecs::CodecParameters;
use symphonia::core::codecs::audio::AudioDecoder;
use symphonia::core::formats::{FormatReader, TrackType};
use symphonia::core::io::MediaSourceStream;

use super::mixer::Clip;
use crate::opus::{self, OpusSource};

/// Decodes a clip a piece at a time, as (left, right) frames at its own
/// rate (a mono clip's on both sides).
pub trait Source: Send {
    /// Adds the next frames to `out`; false at the end, adding none.
    fn next(&mut self, out: &mut Vec<[f32; 2]>) -> Result<bool>;
}

/// A PCM WAV clip, decoded a packet at a time, its samples as f32 (an i16
/// `s` as `s / 32768`, as WebAudio's decoding gives it).
pub struct WavSource {
    reader: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track: u32,
    samples: Vec<f32>,
}

/// What a WAV clip's header says: its sample rate, channels and frames.
fn wav_header(reader: &dyn FormatReader) -> Result<(u32, f64, usize, usize)> {
    let track = reader.default_track(TrackType::Audio).context("the clip has no audio")?;
    let Some(CodecParameters::Audio(parameters)) = &track.codec_params else { bail!("the clip has no audio") };
    let rate = parameters.sample_rate.context("the clip has no sample rate")?;
    // The rates a WebAudio buffer may have; at 0 Hz, a clip would resample
    // to no end.
    ensure!((3_000..=768_000).contains(&rate), "the clip's sample rate, {rate} Hz, is outside 3 to 768 kHz");
    let channels = parameters.channels.as_ref().map_or(0, |channels| channels.count());
    ensure!(channels == 1 || channels == 2, "clips with {channels} channels are not supported");
    let frames = track.num_frames.and_then(|frames| usize::try_from(frames).ok()).unwrap_or(0);
    Ok((track.id, f64::from(rate), channels, frames))
}

fn wav_reader(bytes: Arc<[u8]>) -> Result<Box<dyn FormatReader>> {
    let source = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    symphonia::default::get_probe()
        .probe(&Default::default(), source, Default::default(), Default::default())
        .context("not a WAV or Ogg Opus clip")
}

impl WavSource {
    pub fn new(bytes: Arc<[u8]>) -> Result<WavSource> {
        let reader = wav_reader(bytes)?;
        let (track, ..) = wav_header(reader.as_ref())?;
        let track_data = reader.default_track(TrackType::Audio).context("the clip has no audio")?;
        let Some(CodecParameters::Audio(parameters)) = &track_data.codec_params else { bail!("the clip has no audio") };
        let decoder = symphonia::default::get_codecs().make_audio_decoder(parameters, &Default::default())?;
        Ok(WavSource { reader, decoder, track, samples: Vec::new() })
    }
}

impl Source for WavSource {
    fn next(&mut self, out: &mut Vec<[f32; 2]>) -> Result<bool> {
        while let Some(packet) = self.reader.next_packet()? {
            if packet.track_id != self.track {
                continue;
            }
            let buffer = self.decoder.decode(&packet)?;
            buffer.copy_to_vec_interleaved(&mut self.samples);
            match buffer.num_planes() {
                1 => out.extend(self.samples.iter().map(|&value| [value, value])),
                2 => out.extend_from_slice(self.samples.as_chunks::<2>().0),
                channels => bail!("clips with {channels} channels are not supported"),
            }
            if !self.samples.is_empty() {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Wav,
    Opus,
}

/// A clip as its file holds it: its bytes, and what their header says of
/// it before any is decoded.
pub struct Encoded {
    bytes: Arc<[u8]>,
    format: Format,
    /// Its length in frames at its own rate.
    pub frames: usize,
    pub rate: f64,
    pub channels: usize,
}

impl Encoded {
    /// A WAV or Ogg Opus clip's bytes.
    pub fn new(bytes: Arc<[u8]>) -> Result<Encoded> {
        if opus::is_ogg_opus(&bytes) {
            let head = opus::head(&bytes)?;
            return Ok(Encoded {
                bytes,
                format: Format::Opus,
                frames: head.frames,
                rate: f64::from(opus::RATE),
                channels: head.channels,
            });
        }
        let (_, rate, channels, frames) = wav_header(wav_reader(bytes.clone())?.as_ref())?;
        Ok(Encoded { bytes, format: Format::Wav, frames, rate, channels })
    }

    /// Its frames from the first, a piece at a time.
    pub fn source(&self) -> Result<Box<dyn Source>> {
        Ok(match self.format {
            Format::Wav => Box::new(WavSource::new(self.bytes.clone())?),
            Format::Opus => Box::new(OpusSource::new(self.bytes.clone())?),
        })
    }

    /// The whole clip, decoded.
    pub fn decode(&self) -> Result<Clip> {
        let mut source = self.source()?;
        let mut frames = Vec::with_capacity(self.frames);
        while source.next(&mut frames)? {}
        Ok(Clip::new(frames.into(), self.rate))
    }
}
