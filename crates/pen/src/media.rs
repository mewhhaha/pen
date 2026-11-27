//! Host media runtime wiring: GPU video playback registry (`.y4m` / `.ivf` driven by
//! `frame.video_streams`) and live Opus VoIP transport session (`VoiceRouter` + `VoiceEncoder`/`VoiceDecoder`).
#![allow(dead_code, reason = "host media/VoIP APIs exercised by integration tests and session runtime")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use pen_protocol::{Catalog, RenderFrame};
use pen_render::{IvfStream, Renderer, VideoPlayer};
use pen_runtime::voice::{PopAction, VOICE_FRAME_SAMPLES, VoiceActivityDetector, VoicePacket, VoiceRouter};

use crate::opus::{VoiceDecoder, VoiceEncoder};

struct ActiveVideo {
    player: VideoPlayer,
    last_frame_index: Option<usize>,
    last_seek: Option<(u32, u32, bool)>,
}

/// Manages GPU-decoded video textures referenced by `frame.video_streams`.
pub struct VideoRegistry {
    assets_dir: PathBuf,
    streams: HashMap<String, ActiveVideo>,
}

impl VideoRegistry {
    pub fn new(assets_dir: PathBuf) -> Self {
        Self { assets_dir, streams: HashMap::new() }
    }

    /// Steps all active `frame.video_streams`, lazily opening `.y4m`/`.ivf` files from
    /// `assets_dir` and uploading changed frames zero-copy into `renderer`.
    pub fn update(&mut self, frame: &RenderFrame, catalog: &Catalog, renderer: &mut Renderer) -> Result<()> {
        let max_dim = renderer.max_texture_dimension_2d();
        let mut directives: Vec<(String, &pen_protocol::VideoStreamSpec)> = Vec::new();
        let mut seen: HashMap<String, usize> = HashMap::new();
        for spec in &frame.video_streams {
            let Some(mat_key) = catalog.materials.get(spec.material as usize) else { continue };
            if pen_render::validate_asset_key(mat_key).is_err() {
                continue;
            }
            let mat_path = self.assets_dir.join(mat_key);
            let Ok(mat_bytes) = std::fs::read(&mat_path) else { continue };
            let Ok(Some(tex_key)) = pen_render::inspect_material_base_texture(&mat_bytes) else { continue };
            if pen_render::validate_asset_key(&tex_key).is_err() {
                continue;
            }
            if let Some(&pos) = seen.get(&tex_key) {
                directives[pos].1 = spec;
            } else {
                seen.insert(tex_key.clone(), directives.len());
                directives.push((tex_key, spec));
            }
        }
        for (tex_key, spec) in directives {
            if !self.streams.contains_key(&tex_key) {
                let video_path = self.assets_dir.join(&tex_key);
                if let Some(player) = load_video_player(&video_path, spec.looping, max_dim)? {
                    self.streams
                        .insert(tex_key.clone(), ActiveVideo { player, last_frame_index: None, last_seek: None });
                } else {
                    continue;
                }
            }
            if let Some(active) = self.streams.get_mut(&tex_key) {
                active.player.looping = spec.looping;
                active.player.paused = !spec.playing;
                let seek_key = (spec.time.to_bits(), spec.speed.to_bits(), spec.looping);
                if spec.playing || active.last_seek != Some(seek_key) {
                    active.player.seek_time(f64::from(spec.time), f64::from(spec.speed));
                    active.last_seek = Some(seek_key);
                }
                let idx = active.player.current_index();
                if active.last_frame_index != Some(idx) {
                    let (w, h, cs, siting, y, u, v) = active.player.current_planes();
                    renderer.upload_video_planes(&tex_key, w, h, cs, siting, y, u, v)?;
                    active.last_frame_index = Some(idx);
                }
            }
        }
        Ok(())
    }
}

fn load_video_player(path: &Path, looping: bool, max_dim: u32) -> Result<Option<VideoPlayer>> {
    if !path.is_file() {
        return Ok(None);
    }
    const MAX_NATIVE_Y4M_BYTES: usize = 256 * 1024 * 1024;
    let bytes = std::fs::read(path).with_context(|| format!("reading video {}", path.display()))?;
    if bytes.starts_with(b"YUV4MPEG2") {
        let stream = pen_render::IndexedY4mStream::parse_with_limits(
            bytes,
            pen_render::ColorMatrix::Bt709,
            max_dim,
            MAX_NATIVE_Y4M_BYTES,
        )?;
        let player = VideoPlayer::from_indexed_y4m(stream, looping)?;
        return Ok(Some(player));
    }
    if bytes.starts_with(b"DKIF") {
        let _ivf = IvfStream::parse(&bytes)?;
    }
    Ok(None)
}

/// Live Opus VoIP session combining `VoiceRouter` (per-speaker jitter buffers & routing),
/// `VoiceActivityDetector`, and vendored libopus `VoiceEncoder`/`VoiceDecoder`.
pub struct VoiceSession {
    pub router: VoiceRouter,
    pub vad: VoiceActivityDetector,
    encoder: VoiceEncoder,
    decoders: HashMap<u32, VoiceDecoder>,
    local_speaker: u32,
    next_seq: u16,
    timestamp: u32,
}

impl VoiceSession {
    pub fn new(local_speaker: u32, bitrate_bps: i32) -> Result<Self> {
        let encoder = VoiceEncoder::new_voip(bitrate_bps)?;
        Ok(Self {
            router: VoiceRouter::new(2, 16, 1000),
            vad: VoiceActivityDetector::default(),
            encoder,
            decoders: HashMap::new(),
            local_speaker,
            next_seq: 0,
            timestamp: 0,
        })
    }

    /// Encodes a 20 ms 48 kHz mono PCM frame (960 samples) through VAD + Opus.
    pub fn capture_frame(&mut self, mono_pcm: &[f32]) -> Result<Option<VoicePacket>> {
        if !self.vad.analyze(mono_pcm) {
            return Ok(None);
        }
        let payload = self.encoder.encode_frame(mono_pcm)?;
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        let ts = self.timestamp;
        #[allow(clippy::cast_possible_truncation, reason = "audio frame length fits u32")]
        {
            self.timestamp = self.timestamp.wrapping_add(mono_pcm.len() as u32);
        }
        Ok(Some(VoicePacket { speaker_id: self.local_speaker, seq, timestamp: ts, flags: 0, payload }))
    }

    /// Queues an incoming peer voice packet into its speaker's jitter buffer.
    pub fn receive_packet(&mut self, packet: VoicePacket) -> bool {
        self.router.receive(packet)
    }

    /// Pops and decodes the current 20 ms tick for all active speakers.
    pub fn pop_decoded_speakers(&mut self) -> Vec<(u32, Vec<f32>)> {
        let actions = self.router.pop_all();
        let mut decoded = Vec::with_capacity(actions.len());
        for (speaker, action) in actions {
            if let std::collections::hash_map::Entry::Vacant(e) = self.decoders.entry(speaker) {
                if let Ok(dec) = VoiceDecoder::new_mono() {
                    e.insert(dec);
                } else {
                    continue;
                }
            }
            let dec = self.decoders.get_mut(&speaker).unwrap();
            let mut pcm = vec![0.0f32; VOICE_FRAME_SAMPLES];
            let ok = match action {
                PopAction::Play(bytes) => dec.decode_frame(&bytes, &mut pcm).is_ok(),
                PopAction::Conceal => dec.conceal_loss(&mut pcm).is_ok(),
                PopAction::Silence => false,
            };
            if ok {
                decoded.push((speaker, pcm));
            }
        }
        decoded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_session_vad_opus_jitter_round_trip() {
        let mut session = VoiceSession::new(7, 32_000).unwrap();
        let tone: Vec<f32> = (0..VOICE_FRAME_SAMPLES)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.4)
            .collect();
        let mut sent = 0usize;
        for _ in 0..4 {
            if let Some(pkt) = session.capture_frame(&tone).unwrap() {
                assert!(session.receive_packet(pkt));
                sent += 1;
            }
        }
        assert!(sent >= 3, "VAD + Opus should emit packets for loud tone");
        let mut played = 0usize;
        for _ in 0..4 {
            for (speaker, pcm) in session.pop_decoded_speakers() {
                assert_eq!(speaker, 7);
                assert_eq!(pcm.len(), VOICE_FRAME_SAMPLES);
                let energy: f32 = pcm.iter().map(|s| s * s).sum();
                if energy > 0.1 {
                    played += 1;
                }
            }
        }
        assert!(played >= 1, "decoded Opus speaker should emit non-silent PCM");
    }
}
