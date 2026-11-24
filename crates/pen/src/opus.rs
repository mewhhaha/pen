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

/// Real-time VoIP mono Opus encoder at 48 kHz (wrapping vendored libopus).
#[allow(dead_code, reason = "voice chat codec API wired through runtime voice transport")]
pub struct VoiceEncoder(*mut sys::OpusEncoder);

// SAFETY: encoder state is plain memory accessed exclusively through `&mut self`.
unsafe impl Send for VoiceEncoder {}

#[allow(dead_code, reason = "voice chat codec API wired through runtime voice transport")]
impl VoiceEncoder {
    pub fn new_voip(bitrate_bps: i32) -> Result<Self> {
        let mut error = 0;
        // SAFETY: valid 48 kHz mono VoIP application; libopus returns null and sets error on failure.
        let state = unsafe { sys::opus_encoder_create(RATE, 1, sys::OPUS_APPLICATION_VOIP, &mut error) };
        ensure!(!state.is_null() && error == sys::OPUS_OK, "creating an Opus VoIP encoder failed ({error})");
        let mut enc = Self(state);
        enc.set_bitrate(bitrate_bps)?;
        enc.set_inband_fec(true)?;
        Ok(enc)
    }

    pub fn set_bitrate(&mut self, bps: i32) -> Result<()> {
        let bps = bps.clamp(6_000, 128_000);
        // SAFETY: `self.0` is a live OpusEncoder; OPUS_SET_BITRATE_REQUEST takes one i32 vararg.
        let err = unsafe { sys::opus_encoder_ctl(self.0, sys::OPUS_SET_BITRATE_REQUEST, bps) };
        ensure!(err == sys::OPUS_OK, "OPUS_SET_BITRATE failed ({err})");
        Ok(())
    }

    pub fn set_inband_fec(&mut self, enabled: bool) -> Result<()> {
        let val: i32 = i32::from(enabled);
        // SAFETY: `self.0` is a live OpusEncoder; OPUS_SET_INBAND_FEC_REQUEST takes one i32 vararg.
        let err = unsafe { sys::opus_encoder_ctl(self.0, sys::OPUS_SET_INBAND_FEC_REQUEST, val) };
        ensure!(err == sys::OPUS_OK, "OPUS_SET_INBAND_FEC failed ({err})");
        Ok(())
    }

    pub fn set_dtx(&mut self, enabled: bool) -> Result<()> {
        let val: i32 = i32::from(enabled);
        // SAFETY: `self.0` is a live OpusEncoder; OPUS_SET_DTX_REQUEST takes one i32 vararg.
        let err = unsafe { sys::opus_encoder_ctl(self.0, sys::OPUS_SET_DTX_REQUEST, val) };
        ensure!(err == sys::OPUS_OK, "OPUS_SET_DTX failed ({err})");
        Ok(())
    }

    /// Encodes one 20 ms mono 48 kHz PCM frame (960 samples) into a non-empty Opus VoIP packet.
    pub fn encode_frame(&mut self, pcm: &[f32]) -> Result<Vec<u8>> {
        ensure!(
            pcm.len() == pen_runtime::voice::VOICE_FRAME_SAMPLES,
            "Opus VoIP frame length must be {} samples (20 ms at 48 kHz), got {}",
            pen_runtime::voice::VOICE_FRAME_SAMPLES,
            pcm.len()
        );
        let mut out = vec![0u8; pen_runtime::voice::MAX_OPUS_PAYLOAD];
        let frame_size = i32::try_from(pcm.len())?;
        let max_bytes = i32::try_from(out.len())?;
        // SAFETY: `pcm` has `frame_size` mono floats and `out` holds `max_bytes` bytes.
        let len = unsafe { sys::opus_encode_float(self.0, pcm.as_ptr(), frame_size, out.as_mut_ptr(), max_bytes) };
        let len = usize::try_from(len).ok().with_context(|| format!("encoding Opus VoIP frame failed ({len})"))?;
        out.truncate(len);
        Ok(out)
    }
}

impl Drop for VoiceEncoder {
    fn drop(&mut self) {
        // SAFETY: created by opus_encoder_create and freed once.
        unsafe { sys::opus_encoder_destroy(self.0) }
    }
}

/// Real-time VoIP mono Opus decoder with Packet Loss Concealment (PLC).
#[allow(dead_code, reason = "voice chat codec API wired through runtime voice transport")]
pub struct VoiceDecoder(Decoder);

#[allow(dead_code, reason = "voice chat codec API wired through runtime voice transport")]
impl VoiceDecoder {
    pub fn new_mono() -> Result<Self> {
        Ok(Self(Decoder::new(1)?))
    }

    /// Decodes one non-empty incoming Opus packet into `out` (returns decoded sample count).
    /// Empty packets are rejected so PLC is only invoked explicitly via [`Self::conceal_loss`].
    pub fn decode_frame(&mut self, packet: &[u8], out: &mut [f32]) -> Result<usize> {
        ensure!(!packet.is_empty(), "empty Opus packet passed to decode_frame; call conceal_loss for PLC");
        self.0.decode(packet, out, 1)
    }

    /// Generates `frame_samples` of Packet Loss Concealment audio when a packet was dropped.
    pub fn conceal_loss(&mut self, out: &mut [f32]) -> Result<usize> {
        let frames = i32::try_from(out.len())?;
        // SAFETY: libopus accepts a null packet pointer with length 0 to invoke PLC for `frames` samples.
        let count = unsafe { sys::opus_decode_float(self.0.0, std::ptr::null(), 0, out.as_mut_ptr(), frames, 0) };
        usize::try_from(count).ok().with_context(|| format!("Opus PLC failed ({count})"))
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

    #[test]
    fn voip_voice_chat_pipeline_encodes_reorders_conceals_loss_and_decodes() {
        use pen_runtime::voice::{
            FLAG_SPEAKING, JitterBuffer, PopAction, VOICE_FRAME_SAMPLES, VoiceActivityDetector, VoicePacket,
            decode_packet, encode_packet,
        };
        let mut vad = VoiceActivityDetector::default();
        let mut enc = super::VoiceEncoder::new_voip(32_000).unwrap();
        enc.set_dtx(false).unwrap();
        let mut dec = super::VoiceDecoder::new_mono().unwrap();
        let mut jb = JitterBuffer::new(2);

        // Generate 8 frames (160 ms) of a 350 Hz speech-band tone.
        let mut packets = Vec::new();
        for f in 0..8u16 {
            let pcm: Vec<f32> = (0..VOICE_FRAME_SAMPLES)
                .map(|i| {
                    let t = (usize::from(f) * VOICE_FRAME_SAMPLES + i) as f32 / 48_000.0;
                    0.35 * (std::f32::consts::TAU * 350.0 * t).sin()
                })
                .collect();
            assert!(vad.analyze(&pcm), "frame {f} must trigger VAD");
            let payload = enc.encode_frame(&pcm).unwrap();
            assert!(!payload.is_empty());
            let wire = encode_packet(&VoicePacket {
                speaker_id: 11,
                seq: f,
                timestamp: u32::from(f) * u32::try_from(VOICE_FRAME_SAMPLES).unwrap(),
                flags: FLAG_SPEAKING,
                payload,
            })
            .unwrap();
            packets.push(decode_packet(&wire).unwrap());
        }
        // Deliver out-of-order and drop packet #4 to exercise JitterBuffer reorder + Opus PLC.
        for &idx in &[1usize, 0, 3, 2, 5, 6, 7] {
            let p = &packets[idx];
            jb.push(p.seq, p.payload.clone());
        }
        let mut pcm_out = vec![0.0f32; VOICE_FRAME_SAMPLES];
        let mut played = 0usize;
        let mut concealed = 0usize;
        for _ in 0..8 {
            match jb.pop() {
                PopAction::Play(data) => {
                    let n = dec.decode_frame(&data, &mut pcm_out).unwrap();
                    assert_eq!(n, VOICE_FRAME_SAMPLES);
                    assert!(pcm_out.iter().all(|v| v.is_finite()));
                    played += 1;
                }
                PopAction::Conceal => {
                    let n = dec.conceal_loss(&mut pcm_out).unwrap();
                    assert_eq!(n, VOICE_FRAME_SAMPLES);
                    assert!(pcm_out.iter().all(|v| v.is_finite()));
                    let rms = (pcm_out.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / n as f64).sqrt();
                    assert!(rms > 0.02, "PLC must synthesize continuous signal, got rms={rms}");
                    concealed += 1;
                }
                PopAction::Silence => {}
            }
        }
        assert_eq!((played, concealed), (7, 1));
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
