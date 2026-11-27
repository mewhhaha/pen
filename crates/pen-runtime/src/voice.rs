//! Voice-chat transport layer: bounded Opus voice-packet wire codec, adaptive
//! DC-free energy + zero-crossing Voice Activity Detection (VAD), wrapping-sequence
//! jitter buffer with Packet Loss Concealment (PLC) scheduling, and spatial
//! proximity / team-radio speaker routing.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "audio & wire framing math")]

use std::collections::{BTreeMap, HashMap, HashSet};

use pen_protocol::boundary::MAX_VOICE_ID;
use pen_protocol::{Panner, Vector, distance_model};

/// Magic header (`"VO"`) identifying Pen voice-chat datagrams.
pub const VOICE_MAGIC: u16 = 0x564F;
pub const VOICE_VERSION: u8 = 1;
pub const VOICE_HEADER_BYTES: usize = 16;
/// RFC 6716 maximum single Opus frame payload length.
pub const MAX_OPUS_PAYLOAD: usize = 1275;
/// Standard 20 ms voice frame at 48 kHz (960 samples).
pub const VOICE_FRAME_SAMPLES: usize = 960;
/// Idle ticks (20 ms each -> 3 seconds) before an inactive speaker buffer is reclaimed.
pub const SPEAKER_IDLE_EXPIRY_TICKS: u32 = 150;

pub const FLAG_SPEAKING: u8 = 1 << 0;
pub const FLAG_SPATIAL: u8 = 1 << 1;
pub const FLAG_DTX: u8 = 1 << 2;

/// One framed voice-chat packet carrying an encoded 20 ms Opus frame (or an
/// explicit empty DTX comfort-silence marker when `FLAG_DTX` is set).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoicePacket {
    pub speaker_id: u32,
    pub seq: u16,
    pub timestamp: u32,
    pub flags: u8,
    pub payload: Vec<u8>,
}

/// Wire framing errors for voice packets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoicePacketError {
    TooShort,
    BadMagic,
    UnsupportedVersion,
    EmptyPayload,
    PayloadTooLarge,
    LengthMismatch,
}

/// Encodes a `VoicePacket` into a bounded network datagram. Empty payloads are
/// permitted only when `FLAG_DTX` is explicitly set.
pub fn encode_packet(packet: &VoicePacket) -> Result<Vec<u8>, VoicePacketError> {
    if packet.payload.len() > MAX_OPUS_PAYLOAD {
        return Err(VoicePacketError::PayloadTooLarge);
    }
    if packet.payload.is_empty() && (packet.flags & FLAG_DTX) == 0 {
        return Err(VoicePacketError::EmptyPayload);
    }
    let mut out = Vec::with_capacity(VOICE_HEADER_BYTES + packet.payload.len());
    out.extend_from_slice(&VOICE_MAGIC.to_be_bytes());
    out.push(VOICE_VERSION);
    out.push(packet.flags);
    out.extend_from_slice(&packet.speaker_id.to_be_bytes());
    out.extend_from_slice(&packet.seq.to_be_bytes());
    out.extend_from_slice(&packet.timestamp.to_be_bytes());
    out.extend_from_slice(&(packet.payload.len() as u16).to_be_bytes());
    out.extend_from_slice(&packet.payload);
    Ok(out)
}

/// Decodes and strictly validates a voice datagram (exact length, no trailing bytes).
pub fn decode_packet(bytes: &[u8]) -> Result<VoicePacket, VoicePacketError> {
    if bytes.len() < VOICE_HEADER_BYTES {
        return Err(VoicePacketError::TooShort);
    }
    let magic = u16::from_be_bytes([bytes[0], bytes[1]]);
    if magic != VOICE_MAGIC {
        return Err(VoicePacketError::BadMagic);
    }
    if bytes[2] != VOICE_VERSION {
        return Err(VoicePacketError::UnsupportedVersion);
    }
    let flags = bytes[3];
    let speaker_id = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let seq = u16::from_be_bytes([bytes[8], bytes[9]]);
    let timestamp = u32::from_be_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]);
    let payload_len = usize::from(u16::from_be_bytes([bytes[14], bytes[15]]));
    if payload_len > MAX_OPUS_PAYLOAD {
        return Err(VoicePacketError::PayloadTooLarge);
    }
    if payload_len == 0 && (flags & FLAG_DTX) == 0 {
        return Err(VoicePacketError::EmptyPayload);
    }
    if bytes.len() != VOICE_HEADER_BYTES + payload_len {
        return Err(VoicePacketError::LengthMismatch);
    }
    Ok(VoicePacket { speaker_id, seq, timestamp, flags, payload: bytes[VOICE_HEADER_BYTES..].to_vec() })
}

/// Adaptive DC-removed RMS energy + zero-crossing Voice Activity Detector (VAD)
/// with onset/hangover hysteresis and steady-noise floor recovery.
#[derive(Clone, Debug)]
pub struct VoiceActivityDetector {
    /// Minimum AC RMS threshold above silence (linear amplitude).
    pub min_rms: f32,
    /// SNR margin above the adaptive background noise floor (linear ratio).
    pub margin: f32,
    /// Consecutive quiet frames kept active after speech stops to avoid clipping word tails.
    pub hangover_frames: u32,
    noise_floor: f32,
    hangover_left: u32,
    steady_frames: u32,
    last_rms: f32,
    speaking: bool,
}

impl Default for VoiceActivityDetector {
    fn default() -> Self {
        Self::new(0.015, 2.5, 8)
    }
}

impl VoiceActivityDetector {
    pub fn new(min_rms: f32, margin: f32, hangover_frames: u32) -> Self {
        Self {
            min_rms: min_rms.clamp(1e-5, 0.5),
            margin: margin.clamp(1.1, 20.0),
            hangover_frames,
            noise_floor: min_rms * 0.5,
            hangover_left: 0,
            steady_frames: 0,
            last_rms: 0.0,
            speaking: false,
        }
    }

    pub fn is_speaking(&self) -> bool {
        self.speaking
    }

    pub fn noise_floor(&self) -> f32 {
        self.noise_floor
    }

    /// Evaluates one PCM mono frame and returns `true` when voice should be transmitted.
    pub fn analyze(&mut self, samples: &[f32]) -> bool {
        if samples.is_empty() {
            self.speaking = false;
            return false;
        }
        // Compute and subtract frame DC offset so constant bias never triggers VAD.
        let mut sum = 0.0f64;
        for &s in samples {
            let v = if s.is_finite() { f64::from(s.clamp(-4.0, 4.0)) } else { 0.0 };
            sum += v;
        }
        let mean = sum / samples.len() as f64;
        let mut sq = 0.0f64;
        let mut zcr = 0usize;
        let mut prev = 0.0f64;
        for (i, &s) in samples.iter().enumerate() {
            let v = (if s.is_finite() { f64::from(s.clamp(-4.0, 4.0)) } else { 0.0 }) - mean;
            sq += v * v;
            if i > 0 && ((prev < 0.0 && v >= 0.0) || (prev >= 0.0 && v < 0.0)) {
                zcr += 1;
            }
            prev = v;
        }
        let rms = (sq / samples.len() as f64).sqrt() as f32;
        let zcr_rate = zcr as f32 / samples.len() as f32;
        let speech_like = (0.002..=0.55).contains(&zcr_rate);
        let threshold = self.min_rms.max(self.noise_floor * self.margin);
        if rms >= threshold && speech_like {
            // Detect prolonged constant-envelope stationary background noise (e.g. fan hum)
            // and allow the noise floor to adapt upward so the gate can recover.
            if (rms - self.last_rms).abs() < rms * 0.08 {
                self.steady_frames = self.steady_frames.saturating_add(1);
                if self.steady_frames > 40 {
                    self.noise_floor = (self.noise_floor * 0.92 + rms * 0.08).clamp(1e-5, 0.5);
                }
            } else {
                self.steady_frames = 0;
            }
            self.last_rms = rms;
            let adapted_threshold = self.min_rms.max(self.noise_floor * self.margin);
            if rms >= adapted_threshold {
                self.speaking = true;
                self.hangover_left = self.hangover_frames;
            } else if self.hangover_left > 0 {
                self.hangover_left -= 1;
                self.speaking = true;
            } else {
                self.speaking = false;
            }
        } else {
            self.steady_frames = 0;
            self.last_rms = rms;
            self.noise_floor = (self.noise_floor * 0.95 + rms * 0.05).clamp(1e-5, 0.25);
            if self.hangover_left > 0 {
                self.hangover_left -= 1;
                self.speaking = true;
            } else {
                self.speaking = false;
            }
        }
        self.speaking
    }
}

/// Playout instruction emitted on each 20 ms audio tick by [`JitterBuffer`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PopAction {
    /// Decode and play this non-empty 20 ms Opus packet payload.
    Play(Vec<u8>),
    /// Packet missing while talkspurt is active: run Opus PLC (`decode(null)`).
    Conceal,
    /// Buffer priming, explicit DTX silence, or speaker idle: emit zero samples.
    Silence,
}

/// Signed wrapping 16-bit sequence delta (`a - b`).
fn seq_delta(a: u16, b: u16) -> i16 {
    a.wrapping_sub(b).cast_signed()
}

/// Signed wrapping 32-bit timestamp delta (`a - b`).
fn ts_delta(a: u32, b: u32) -> i32 {
    a.wrapping_sub(b).cast_signed()
}

#[derive(Clone, Debug)]
struct QueuedFrame {
    payload: Vec<u8>,
    dtx: bool,
}

/// Bounded wrapping-sequence jitter buffer for one remote speaker.
#[derive(Clone, Debug)]
pub struct JitterBuffer {
    target_delay: usize,
    max_buffered: usize,
    max_plc: usize,
    initialized: bool,
    primed: bool,
    prime_wait_ticks: usize,
    consecutive_plc: usize,
    idle_ticks: u32,
    /// Keyed by monotonic unwrapped sequence index (`u64`) to handle `u16` wrap-around cleanly.
    queue: BTreeMap<u64, QueuedFrame>,
    /// Next sequence expected for playout (or lowest anchor while unprimed).
    base_unwrapped: u64,
    base_seq: u16,
    /// Highest playout watermark emitted so late duplicates after idle never re-trigger playout.
    watermark_unwrapped: Option<u64>,
    last_timestamp: Option<u32>,
    late_dropped: u64,
    duplicate_dropped: u64,
    plc_frames: u64,
}

impl JitterBuffer {
    pub fn new(target_delay_frames: usize) -> Self {
        Self {
            target_delay: target_delay_frames.clamp(1, 16),
            max_buffered: 32,
            max_plc: 5,
            initialized: false,
            primed: false,
            prime_wait_ticks: 0,
            consecutive_plc: 0,
            idle_ticks: 0,
            queue: BTreeMap::new(),
            base_unwrapped: 1 << 20,
            base_seq: 0,
            watermark_unwrapped: None,
            last_timestamp: None,
            late_dropped: 0,
            duplicate_dropped: 0,
            plc_frames: 0,
        }
    }

    /// Explicitly resets stream synchronization (e.g. on peer reconnect).
    pub fn reset(&mut self) {
        self.initialized = false;
        self.primed = false;
        self.prime_wait_ticks = 0;
        self.consecutive_plc = 0;
        self.idle_ticks = 0;
        self.queue.clear();
        self.base_unwrapped = 1 << 20;
        self.base_seq = 0;
        self.watermark_unwrapped = None;
        self.last_timestamp = None;
    }

    pub fn buffered(&self) -> usize {
        self.queue.len()
    }

    pub fn idle_ticks(&self) -> u32 {
        self.idle_ticks
    }

    pub fn late_dropped(&self) -> u64 {
        self.late_dropped
    }

    pub fn duplicate_dropped(&self) -> u64 {
        self.duplicate_dropped
    }

    pub fn plc_frames(&self) -> u64 {
        self.plc_frames
    }

    fn unwrap_seq(&self, seq: u16) -> Option<u64> {
        let d = seq_delta(seq, self.base_seq);
        if d >= 0 {
            Some(self.base_unwrapped + u64::from(d as u16))
        } else {
            self.base_unwrapped.checked_sub(u64::from(d.unsigned_abs()))
        }
    }

    /// Inserts an incoming voice packet using default non-DTX flags.
    pub fn push(&mut self, seq: u16, payload: Vec<u8>) {
        self.push_packet(seq, None, 0, payload);
    }

    /// Inserts an incoming voice packet with optional RTP-style sample timestamp and flags.
    pub fn push_packet(&mut self, seq: u16, timestamp: Option<u32>, flags: u8, payload: Vec<u8>) {
        let dtx = (flags & FLAG_DTX) != 0;
        if payload.len() > MAX_OPUS_PAYLOAD || (payload.is_empty() && !dtx) {
            return;
        }
        self.idle_ticks = 0;
        if !self.initialized {
            self.initialized = true;
            self.base_seq = seq;
            self.base_unwrapped = 1 << 20;
            self.primed = false;
            self.prime_wait_ticks = 0;
            self.consecutive_plc = 0;
            self.last_timestamp = timestamp;
            self.queue.insert(self.base_unwrapped, QueuedFrame { payload, dtx });
            return;
        }
        let delta = seq_delta(seq, self.base_seq);
        if delta.abs() > 1000 {
            // Only re-anchor on large sequence discontinuities when timestamp strictly advances
            // (preventing a heavily delayed stale packet from rewinding playout).
            let forward_reset = match (timestamp, self.last_timestamp) {
                (Some(ts), Some(prev)) => ts_delta(ts, prev) > 0,
                _ => delta > 1000,
            };
            if forward_reset {
                self.queue.clear();
                self.base_seq = seq;
                self.base_unwrapped = self.watermark_unwrapped.map_or(1 << 20, |w| w + 1024);
                self.primed = false;
                self.prime_wait_ticks = 0;
                self.consecutive_plc = 0;
                self.last_timestamp = timestamp;
                self.queue.insert(self.base_unwrapped, QueuedFrame { payload, dtx });
            } else {
                self.late_dropped += 1;
            }
            return;
        }
        let Some(key) = self.unwrap_seq(seq) else {
            self.late_dropped += 1;
            return;
        };
        // Reject packets at or behind the already-emitted playout watermark.
        if let Some(wm) = self.watermark_unwrapped
            && key <= wm
        {
            self.late_dropped += 1;
            return;
        }
        if self.primed && delta < 0 {
            self.late_dropped += 1;
            return;
        }
        if self.queue.contains_key(&key) {
            self.duplicate_dropped += 1;
            return;
        }
        if let Some(ts) = timestamp
            && self.last_timestamp.is_none_or(|prev| ts_delta(ts, prev) > 0)
        {
            self.last_timestamp = Some(ts);
        }
        self.queue.insert(key, QueuedFrame { payload, dtx });
        while self.queue.len() > self.max_buffered {
            if let Some((&oldest_key, _)) = self.queue.iter().next() {
                self.queue.remove(&oldest_key);
                if oldest_key >= self.base_unwrapped {
                    let advance = oldest_key - self.base_unwrapped + 1;
                    self.base_unwrapped += advance;
                    self.base_seq = self.base_seq.wrapping_add(advance as u16);
                }
            }
        }
    }

    /// Pops the next playout frame for the current 20 ms tick.
    pub fn pop(&mut self) -> PopAction {
        if !self.initialized {
            self.idle_ticks = self.idle_ticks.saturating_add(1);
            return PopAction::Silence;
        }
        if !self.primed {
            if self.queue.is_empty() {
                self.prime_wait_ticks = 0;
                self.idle_ticks = self.idle_ticks.saturating_add(1);
                return PopAction::Silence;
            }
            // Wait up to `target_delay` ticks for buffer fill; a partial burst still plays
            // out once the priming deadline expires so a lone packet never stalls forever.
            if self.queue.len() < self.target_delay && self.prime_wait_ticks < self.target_delay {
                self.prime_wait_ticks += 1;
                return PopAction::Silence;
            }
            if let Some(&first_key) = self.queue.keys().next()
                && first_key != self.base_unwrapped
            {
                let diff = first_key as i128 - self.base_unwrapped as i128;
                self.base_seq = self.base_seq.wrapping_add(diff as i16 as u16);
                self.base_unwrapped = first_key;
            }
            self.primed = true;
            self.prime_wait_ticks = 0;
            self.consecutive_plc = 0;
        }
        let expected = self.base_unwrapped;
        self.base_unwrapped += 1;
        self.base_seq = self.base_seq.wrapping_add(1);
        self.watermark_unwrapped = Some(self.watermark_unwrapped.map_or(expected, |w| w.max(expected)));
        if let Some(frame) = self.queue.remove(&expected) {
            self.consecutive_plc = 0;
            self.idle_ticks = 0;
            if frame.dtx || frame.payload.is_empty() { PopAction::Silence } else { PopAction::Play(frame.payload) }
        } else if self.consecutive_plc < self.max_plc {
            self.consecutive_plc += 1;
            self.plc_frames += 1;
            PopAction::Conceal
        } else {
            // Talkspurt ended: re-prime cleanly with a fresh PLC budget on the next burst,
            // preserving `watermark_unwrapped` so stale duplicates never re-trigger playout.
            self.primed = false;
            self.prime_wait_ticks = 0;
            self.consecutive_plc = 0;
            self.idle_ticks = self.idle_ticks.saturating_add(1);
            if let Some(&k) = self.queue.keys().next() {
                let diff = k as i128 - self.base_unwrapped as i128;
                self.base_seq = self.base_seq.wrapping_add(diff as i16 as u16);
                self.base_unwrapped = k;
            }
            PopAction::Silence
        }
    }
}

/// Spatial or team-radio voice routing state for a remote speaker.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeakerRoute {
    pub speaker_id: u32,
    pub voice_id: u32,
    pub bus: u32,
    pub gain: f32,
    pub spatial: bool,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub max_distance: f32,
    pub occlusion: f32,
    pub hrtf: bool,
}

impl SpeakerRoute {
    /// Builds a Pen protocol [`Panner`] for spatial proximity voice playback.
    pub fn panner(&self) -> Option<Panner> {
        if !self.spatial {
            return None;
        }
        Some(Panner {
            voice: self.voice_id,
            sound: 0,
            position: Vector { x: self.position[0], y: self.position[1], z: self.position[2] },
            orientation: Vector { x: 0.0, y: 0.0, z: 0.0 },
            distance_model: distance_model::LINEAR,
            ref_distance: 2.0,
            max_distance: self.max_distance.max(2.5),
            rolloff: 1.0,
            cone_inner: 360.0,
            cone_outer: 360.0,
            cone_outer_gain: 1.0,
            hrtf: self.hrtf,
            velocity: Vector { x: self.velocity[0], y: self.velocity[1], z: self.velocity[2] },
            doppler: 1.0,
            occlusion: self.occlusion.clamp(0.0, 1.0),
            zone: 0,
        })
    }
}

/// Coordinates per-speaker jitter buffers, mute/deafen policy, idle speaker
/// eviction, and spatial voice routing.
pub struct VoiceRouter {
    target_delay: usize,
    max_talkers: usize,
    base_voice_id: u32,
    next_alloc_offset: u32,
    deafened: bool,
    muted: HashSet<u32>,
    buffers: HashMap<u32, JitterBuffer>,
    routes: HashMap<u32, SpeakerRoute>,
}

impl VoiceRouter {
    pub fn new(target_delay_frames: usize, max_talkers: usize, base_voice_id: u32) -> Self {
        let max_id = u32::try_from(MAX_VOICE_ID).unwrap_or(16_777_215);
        Self {
            target_delay: target_delay_frames.clamp(1, 16),
            max_talkers: max_talkers.clamp(1, 64),
            base_voice_id: base_voice_id.clamp(1, max_id),
            next_alloc_offset: 0,
            deafened: false,
            muted: HashSet::new(),
            buffers: HashMap::new(),
            routes: HashMap::new(),
        }
    }

    pub fn active_speakers(&self) -> usize {
        self.buffers.len()
    }

    pub fn set_deafened(&mut self, deafened: bool) {
        self.deafened = deafened;
        if deafened {
            // Drop queued audio so un-deafening never plays stale buffered speech.
            self.buffers.clear();
        }
    }

    pub fn set_muted(&mut self, speaker_id: u32, muted: bool) {
        if muted {
            self.muted.insert(speaker_id);
            self.buffers.remove(&speaker_id);
        } else {
            self.muted.remove(&speaker_id);
        }
    }

    /// Explicitly removes a speaker's playout buffer and route (e.g. on peer leave).
    pub fn remove_speaker(&mut self, speaker_id: u32) {
        self.buffers.remove(&speaker_id);
        self.routes.remove(&speaker_id);
        self.muted.remove(&speaker_id);
    }

    fn allocate_voice_id(&mut self) -> Option<u32> {
        let max_id = u32::try_from(MAX_VOICE_ID).unwrap_or(16_777_215);
        let span = max_id.saturating_sub(self.base_voice_id).saturating_add(1).min(65_536);
        for _ in 0..span {
            let candidate = self.base_voice_id + (self.next_alloc_offset % span);
            self.next_alloc_offset = self.next_alloc_offset.wrapping_add(1);
            if candidate >= 1 && candidate <= max_id && !self.routes.values().any(|r| r.voice_id == candidate) {
                return Some(candidate);
            }
        }
        None
    }

    /// Registers or updates a speaker's spatial/radio route. Preserves any
    /// previously assigned automatic `voice_id` when updating with `voice_id == 0`.
    pub fn set_speaker_route(&mut self, mut route: SpeakerRoute) -> bool {
        let max_id = u32::try_from(MAX_VOICE_ID).unwrap_or(16_777_215);
        if route.voice_id == 0 {
            if let Some(existing) = self.routes.get(&route.speaker_id) {
                route.voice_id = existing.voice_id;
            } else if let Some(id) = self.allocate_voice_id() {
                route.voice_id = id;
            } else {
                return false;
            }
        } else if route.voice_id > max_id {
            return false;
        }
        self.routes.insert(route.speaker_id, route);
        true
    }

    pub fn route(&self, speaker_id: u32) -> Option<SpeakerRoute> {
        self.routes.get(&speaker_id).copied()
    }

    /// Feeds a decoded incoming voice datagram into its speaker's jitter buffer.
    /// Reclaims idle buffers when at `max_talkers` so new speakers are never permanently blocked.
    pub fn receive(&mut self, packet: VoicePacket) -> bool {
        if self.deafened || self.muted.contains(&packet.speaker_id) {
            return false;
        }
        if !self.buffers.contains_key(&packet.speaker_id) && self.buffers.len() >= self.max_talkers {
            // Evict the most-idle silent speaker if one exists.
            let victim = self
                .buffers
                .iter()
                .filter(|(_, jb)| jb.buffered() == 0 && jb.idle_ticks() > 0)
                .max_by_key(|(_, jb)| jb.idle_ticks())
                .map(|(&id, _)| id);
            if let Some(id) = victim {
                self.buffers.remove(&id);
            } else {
                return false;
            }
        }
        let delay = self.target_delay;
        self.buffers.entry(packet.speaker_id).or_insert_with(|| JitterBuffer::new(delay)).push_packet(
            packet.seq,
            Some(packet.timestamp),
            packet.flags,
            packet.payload,
        );
        true
    }

    /// Pops the next frame for every active speaker (in deterministic speaker-ID order)
    /// and reaps speakers who have remained idle past [`SPEAKER_IDLE_EXPIRY_TICKS`].
    pub fn pop_all(&mut self) -> Vec<(u32, PopAction)> {
        if self.deafened {
            return Vec::new();
        }
        let mut ids: Vec<u32> = self.buffers.keys().copied().collect();
        crate::sort::sort_by_u64_key(&mut ids, |&id| u64::from(id));
        let mut out = Vec::with_capacity(ids.len());
        let mut expired = Vec::new();
        for id in ids {
            if self.muted.contains(&id) {
                expired.push(id);
                continue;
            }
            if let Some(jb) = self.buffers.get_mut(&id) {
                let action = jb.pop();
                if action != PopAction::Silence {
                    out.push((id, action));
                } else if jb.buffered() == 0 && jb.idle_ticks() >= SPEAKER_IDLE_EXPIRY_TICKS {
                    expired.push(id);
                }
            }
        }
        for id in expired {
            self.buffers.remove(&id);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_packet_round_trips_and_rejects_malformed_inputs() {
        let pkt = VoicePacket {
            speaker_id: 42,
            seq: 65534,
            timestamp: 960 * 100,
            flags: FLAG_SPEAKING | FLAG_SPATIAL,
            payload: vec![0xAA, 0xBB, 0xCC, 0xDD],
        };
        let wire = encode_packet(&pkt).unwrap();
        assert_eq!(decode_packet(&wire).unwrap(), pkt);

        // Non-DTX empty payload rejected; explicit DTX empty payload accepted.
        let empty_non_dtx = VoicePacket { payload: Vec::new(), flags: FLAG_SPEAKING, ..pkt.clone() };
        assert_eq!(encode_packet(&empty_non_dtx), Err(VoicePacketError::EmptyPayload));
        let empty_dtx = VoicePacket { payload: Vec::new(), flags: FLAG_DTX, ..pkt.clone() };
        let dtx_wire = encode_packet(&empty_dtx).unwrap();
        assert_eq!(decode_packet(&dtx_wire).unwrap(), empty_dtx);

        // Trailing garbage & truncated & oversized rejected.
        let mut trailing = wire.clone();
        trailing.push(0);
        assert_eq!(decode_packet(&trailing), Err(VoicePacketError::LengthMismatch));
        assert_eq!(decode_packet(&wire[..wire.len() - 1]), Err(VoicePacketError::LengthMismatch));
        assert_eq!(decode_packet(&wire[..10]), Err(VoicePacketError::TooShort));
        let huge = VoicePacket { payload: vec![0; MAX_OPUS_PAYLOAD + 1], ..pkt };
        assert_eq!(encode_packet(&huge), Err(VoicePacketError::PayloadTooLarge));
    }

    #[test]
    fn vad_rejects_dc_gates_silence_detects_speech_and_applies_hangover() {
        let mut vad = VoiceActivityDetector::new(0.02, 2.5, 3);
        let dc_bias = vec![0.25f32; VOICE_FRAME_SAMPLES];
        assert!(!vad.analyze(&dc_bias), "constant DC bias must not trigger VAD");
        let silence = vec![0.001f32; VOICE_FRAME_SAMPLES];
        let speech: Vec<f32> = (0..VOICE_FRAME_SAMPLES)
            .map(|i| 0.2 * (std::f32::consts::TAU * 300.0 * i as f32 / 48_000.0).sin())
            .collect();
        assert!(!vad.analyze(&silence));
        assert!(vad.analyze(&speech));
        // Three hangover frames stay open, fourth closes.
        assert!(vad.analyze(&silence));
        assert!(vad.analyze(&silence));
        assert!(vad.analyze(&silence));
        assert!(!vad.analyze(&silence));
    }

    #[test]
    fn jitter_buffer_reorders_wraps_u16_recovers_after_loss_burst_and_ignores_stale_rewinds() {
        let mut jb = JitterBuffer::new(2);
        // Push packets across u16 wraparound out of order: 65535, 65534, 1 (skipping 0!).
        jb.push(65535, vec![2]);
        assert_eq!(jb.pop(), PopAction::Silence); // not yet primed
        jb.push(65534, vec![1]);
        jb.push(65534, vec![99]); // duplicate must be ignored
        jb.push(1, vec![4]);
        assert_eq!(jb.duplicate_dropped(), 1);
        assert_eq!(jb.pop(), PopAction::Play(vec![1]));
        assert_eq!(jb.pop(), PopAction::Play(vec![2]));
        assert_eq!(jb.pop(), PopAction::Conceal); // seq 0 missing -> PLC!
        assert_eq!(jb.pop(), PopAction::Play(vec![4]));
        assert_eq!(jb.plc_frames(), 1);

        // Exhaust all 5 PLC frames to enter idle re-prime state.
        for _ in 0..5 {
            assert_eq!(jb.pop(), PopAction::Conceal);
        }
        assert_eq!(jb.pop(), PopAction::Silence);

        // A delayed duplicate (seq 1) or heavily delayed packet (-1500 behind) must NOT rewind or play!
        jb.push_packet(1, Some(960), FLAG_SPEAKING, vec![9]);
        jb.push_packet(64000, Some(10), FLAG_SPEAKING, vec![8]);
        assert_eq!(jb.buffered(), 0);

        // Second talkspurt with a single packet followed by gap must play out (deadline) and have fresh PLC budget.
        jb.push_packet(20, Some(20 * 960), FLAG_SPEAKING, vec![20]);
        let mut got_20 = false;
        for _ in 0..4 {
            if jb.pop() == PopAction::Play(vec![20]) {
                got_20 = true;
                break;
            }
        }
        assert!(got_20, "lone packet in second talkspurt must not stall indefinitely");
        assert_eq!(jb.pop(), PopAction::Conceal, "second talkspurt must have fresh PLC budget");
    }

    #[test]
    fn voice_router_frees_idle_slots_clears_on_mute_and_allocates_unique_route_ids() {
        let mut router = VoiceRouter::new(1, 2, 9000);
        let route_for = |id| SpeakerRoute {
            speaker_id: id,
            voice_id: 0,
            bus: 2,
            gain: 1.0,
            spatial: true,
            position: [5.0, 1.5, -3.0],
            velocity: [1.0, 0.0, 0.0],
            max_distance: 30.0,
            occlusion: 0.4,
            hrtf: true,
        };
        assert!(router.set_speaker_route(route_for(1)));
        assert!(router.set_speaker_route(route_for(2)));
        let id1_first = router.route(1).unwrap().voice_id;
        // Updating speaker 1 with voice_id=0 preserves its ID and never collides with speaker 3.
        assert!(router.set_speaker_route(route_for(1)));
        assert_eq!(router.route(1).unwrap().voice_id, id1_first);
        assert!(router.set_speaker_route(route_for(3)));
        assert_ne!(router.route(3).unwrap().voice_id, router.route(1).unwrap().voice_id);
        assert_ne!(router.route(3).unwrap().voice_id, router.route(2).unwrap().voice_id);

        // Two talkers fill slots; after speaker 1 goes idle, speaker 3 can join.
        assert!(router.receive(VoicePacket {
            speaker_id: 1,
            seq: 1,
            timestamp: 960,
            flags: FLAG_SPEAKING,
            payload: vec![1]
        }));
        assert!(router.receive(VoicePacket {
            speaker_id: 2,
            seq: 1,
            timestamp: 960,
            flags: FLAG_SPEAKING,
            payload: vec![2]
        }));
        for _ in 0..10 {
            let _ = router.pop_all();
        }
        assert!(router.receive(VoicePacket {
            speaker_id: 3,
            seq: 1,
            timestamp: 960,
            flags: FLAG_SPEAKING,
            payload: vec![3]
        }));
        // Muting clears buffered packets so unmuting never plays stale audio.
        router.set_muted(3, true);
        router.set_muted(3, false);
        assert!(router.pop_all().is_empty());
    }
}
