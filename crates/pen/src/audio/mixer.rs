//! The game's mix, as the browser host's WebAudio graph of nodes plays it:
//! the frame's buses, voices and sounds, sample by sample, a render quantum
//! (128 samples) at a time. The live output and video recordings both play
//! this, so they sound alike, and like the browser.
//!
//! Per voice, as its nodes do in the browser: the clip (AudioBufferSourceNode,
//! linearly interpolated, its pitch a k-rate playbackRate; a long clip
//! streams, `stream.rs`, sample for sample as it would decoded whole), a
//! lowpass (BiquadFilterNode, Butterworth), its fade (a GainNode ramped
//! linearly), its gain (a GainNode), its pan (StereoPannerNode, mono or
//! stereo as the clip is), or for a voice the frame places in the world its
//! PannerNode (equal-power, `pen_runtime::panner`, placed in the listener's
//! frame and heard from WebAudio's default listener), then into its bus, and
//! its reverb send (a GainNode) into the bus's send. A bus filters and
//! scales both alike, passes the first to the master and the second, with
//! `reverb` of the first, to the shared reverb (ConvolverNode, `reverb.rs`),
//! which returns into the master at the master's `reverb`. Values that change glide as setTargetAtTime does, with
//! AUDIO_GLIDE_MS for its time constant; a frame's changes land at the start
//! of the next quantum, where the browser's `currentTime` is.
//!
//! For the dev tools, the mixer meters each bus's output (the master's after
//! the reverb's return), and the developer may mute and solo buses
//! (`pen_runtime::mix::Monitor`): their gains glide to 0 as a frame's would.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "audio sample math")]
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex};

use pen_protocol::boundary::{AUDIO_FADE_MS, AUDIO_GLIDE_MS, AUDIO_OPEN, MAX_BUSES};
use pen_runtime::mix::{Meter, Monitor};
use pen_runtime::panner::{Pan, Placement};
use pen_runtime::synth::{SynthEngine, SynthSpec};

use super::resample::{Resampler, resample_stereo};
use super::reverb::{AcousticProfile, Convolver};
use super::source::Encoded;
use super::stream::{Reader, Streams};

mod dsp;
use dsp::*;

/// WebAudio's render quantum.
pub const QUANTUM: usize = 128;
/// A lowpass cutoff at which the filter is open.
pub const OPEN: f32 = AUDIO_OPEN as f32;
const GLIDE_SECONDS: f64 = AUDIO_GLIDE_MS as f64 / 1000.0;
const FADE_SECONDS: f64 = AUDIO_FADE_MS as f64 / 1000.0;

/// A decoded clip: its frames (left, right) at its own rate, and whether it
/// is mono (both channels the same), which WebAudio pans with the mono law.
/// One that streams has no frames, but its bytes, decoded as it plays.
pub struct Clip {
    pub frames: Arc<[[f32; 2]]>,
    pub rate: f64,
    pub mono: bool,
    pub streamed: Option<Arc<Encoded>>,
}

impl Clip {
    pub fn new(frames: Arc<[[f32; 2]]>, rate: f64) -> Clip {
        let mono = frames.iter().all(|[left, right]| left.to_bits() == right.to_bits());
        Clip { frames, rate, mono, streamed: None }
    }

    /// A clip that streams from its bytes, at their rate.
    pub fn streamed(encoded: Encoded) -> Clip {
        Clip {
            frames: Arc::new([]),
            rate: encoded.rate,
            mono: encoded.channels == 1,
            streamed: Some(Arc::new(encoded)),
        }
    }

    /// How many frames it lasts at its rate.
    pub fn len(&self) -> usize {
        match &self.streamed {
            Some(encoded) if (encoded.rate - self.rate).abs() >= 0.5 => {
                Resampler::output_length(encoded.frames, encoded.rate, self.rate)
            }
            Some(encoded) => encoded.frames,
            None => self.frames.len(),
        }
    }

    /// The clip at `rate`, resampled as the browser's decoding resamples it
    /// (`resample.rs`), or as it is if it is at that rate already. The mixer
    /// plays a clip at any rate, but one at its own as the browser does.
    pub fn at_rate(self: Arc<Clip>, rate: f64) -> Arc<Clip> {
        if let Some(encoded) = &self.streamed {
            // Streamed, it is resampled as it plays.
            return Arc::new(Clip { frames: Arc::new([]), rate, mono: self.mono, streamed: Some(encoded.clone()) });
        }
        if (self.rate - rate).abs() < 0.5 || self.frames.is_empty() {
            return self;
        }
        let frames: Arc<[[f32; 2]]> = resample_stereo(&self.frames, self.mono, self.rate, rate).into();
        Arc::new(Clip { frames, rate, mono: self.mono, streamed: None })
    }
}

/// A bus as a frame declares it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BusSpec {
    pub gain: f32,
    pub lowpass: f32,
    pub reverb: f32,
    pub comp_threshold: f32,
    pub comp_ratio: f32,
    pub duck_bus: usize,
    pub duck_amount: f32,
    pub eq_low_db: f32,
    pub eq_mid_db: f32,
    pub eq_high_db: f32,
}

impl Default for BusSpec {
    fn default() -> Self {
        PLAIN
    }
}

/// What a frame without buses mixes through: an open master at gain 1,
/// returning the reverb whole, and buses it lacks open and dry at gain 1.
const MASTER: BusSpec = BusSpec {
    gain: 1.0,
    lowpass: OPEN,
    reverb: 1.0,
    comp_threshold: 0.0,
    comp_ratio: 1.0,
    duck_bus: 0,
    duck_amount: 0.0,
    eq_low_db: 0.0,
    eq_mid_db: 0.0,
    eq_high_db: 0.0,
};
const PLAIN: BusSpec = BusSpec {
    gain: 1.0,
    lowpass: OPEN,
    reverb: 0.0,
    comp_threshold: 0.0,
    comp_ratio: 1.0,
    duck_bus: 0,
    duck_amount: 0.0,
    eq_low_db: 0.0,
    eq_mid_db: 0.0,
    eq_high_db: 0.0,
};

/// A voice as a frame declares it.
#[derive(Clone)]
pub struct VoiceSpec {
    pub id: u32,
    pub clip: Arc<Clip>,
    pub bus: usize,
    pub gain: f32,
    pub pan: f32,
    pub pitch: f32,
    pub lowpass: f32,
    pub reverb: f32,
    pub looping: bool,
    pub start: f32,
    pub fade_in: f32,
    pub fade_out: f32,
    /// Where the frame places it, if it does.
    pub panner: Option<Placement>,
}

/// A sound to play once.
#[derive(Clone)]
pub struct SoundSpec {
    pub clip: Arc<Clip>,
    pub bus: usize,
    pub gain: f32,
    pub pan: f32,
    pub pitch: f32,
    pub lowpass: f32,
    pub reverb: f32,
    pub panner: Option<Placement>,
}

/// One frame's sound: its buses, voices, sounds, procedural/granular synths,
/// and optional reverb room or ray-traced acoustic profile.
#[derive(Clone, Default)]
pub struct Mix {
    pub buses: Vec<BusSpec>,
    pub voices: Vec<VoiceSpec>,
    pub sounds: Vec<SoundSpec>,
    pub synths: Vec<SynthSpec>,
    pub reverb_decay: f32,
    pub reverb_damping: f32,
    pub acoustics: Option<AcousticProfile>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::large_enum_variant, reason = "inline Copy avoids heap allocation/deallocation on the RT audio thread")]
enum RoomSpec {
    Plain { decay: f32, damping: f32 },
    Traced(AcousticProfile),
}

impl RoomSpec {
    fn from_mix(mix: &Mix) -> Self {
        if let Some(profile) = mix.acoustics {
            return Self::Traced(profile);
        }
        let decay = if mix.reverb_decay.is_finite() { mix.reverb_decay.max(0.0) } else { 0.0 };
        let damping = if mix.reverb_damping.is_finite() { mix.reverb_damping.clamp(0.0, 1.0) } else { 0.0 };
        Self::Plain { decay, damping }
    }

    fn differs_significantly(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Plain { decay: d1, damping: m1 }, Self::Plain { decay: d2, damping: m2 }) => {
                (d1 - d2).abs() > 0.08 || (m1 - m2).abs() > 0.05 || ((*d1 > 0.0) != (*d2 > 0.0))
            }
            (Self::Traced(a), Self::Traced(b)) => a.differs_significantly(b),
            _ => true,
        }
    }

    fn build_convolver(&self, rate: f64) -> Convolver {
        match self {
            Self::Plain { decay, damping } => Convolver::with_room(rate, f64::from(*decay), f64::from(*damping)),
            Self::Traced(profile) => Convolver::with_acoustics(rate, profile),
        }
    }
}

/// Off-thread reverb IR synthesizer and garbage collector for `Mixer::live` so the
/// real-time audio callback never runs FFT/IR synthesis or frees old convolvers.
struct ReverbWorker {
    pending: Arc<(Mutex<Option<RoomSpec>>, Condvar)>,
    shutdown: Arc<AtomicBool>,
    done_rx: Receiver<(RoomSpec, Convolver)>,
    retire_tx: SyncSender<Convolver>,
}

impl ReverbWorker {
    fn spawn(rate: f64) -> Self {
        let pending = Arc::new((Mutex::new(None::<RoomSpec>), Condvar::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let (done_tx, done_rx) = sync_channel::<(RoomSpec, Convolver)>(4);
        let (retire_tx, retire_rx) = sync_channel::<Convolver>(8);
        let p_bg = pending.clone();
        let s_bg = shutdown.clone();
        let _ = std::thread::Builder::new().name("pen-audio-reverb".into()).spawn(move || {
            loop {
                while retire_rx.try_recv().is_ok() {}
                let next = {
                    let (lock, cvar) = &*p_bg;
                    let mut guard = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                    while guard.is_none() && !s_bg.load(Ordering::Relaxed) {
                        guard = cvar.wait(guard).unwrap_or_else(std::sync::PoisonError::into_inner);
                    }
                    if s_bg.load(Ordering::Relaxed) {
                        return;
                    }
                    guard.take()
                };
                if let Some(room) = next {
                    let conv = room.build_convolver(rate);
                    if done_tx.send((room, conv)).is_err() {
                        return;
                    }
                }
            }
        });
        Self { pending, shutdown, done_rx, retire_tx }
    }

    /// Non-blocking latest-wins submission from the audio thread.
    /// Returns `true` if handed off, or `false` if the worker briefly held the slot lock.
    fn submit(&self, room: RoomSpec) -> bool {
        let (lock, cvar) = &*self.pending;
        match lock.try_lock() {
            Ok(mut guard) => {
                *guard = Some(room);
                drop(guard);
                cvar.notify_one();
                true
            }
            Err(_) => false,
        }
    }

    /// Non-blocking poll for the most recently built convolver (discarding superseded ones off-thread).
    fn poll_latest(&self) -> Option<(RoomSpec, Convolver)> {
        let mut latest = None;
        while let Ok(built) = self.done_rx.try_recv() {
            if let Some((_, old)) = latest.replace(built) {
                let _ = self.retire_tx.try_send(old);
            }
        }
        latest
    }

    /// Hands an expired convolver back to the worker thread to drop off the audio callback.
    fn retire(&self, conv: Convolver) {
        let _ = self.retire_tx.try_send(conv);
    }
}

impl Drop for ReverbWorker {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.pending.1.notify_all();
    }
}

struct Voice {
    /// None for a sound played once, or a voice its id has moved on from.
    id: Option<u32>,
    clip: Arc<Clip>,
    bus: usize,
    /// Where it is in the clip, in the clip's frames.
    position: f64,
    looping: bool,
    pitch: Glide,
    gain: Glide,
    pan: Glide,
    send: Glide,
    lowpass: Lowpass,
    fade: Ramp,
    fade_out: f32,
    leaving: bool,
    ended: bool,
    /// Declared by the latest frame.
    seen: bool,
    /// The pan its gains are for, and they.
    panned: f32,
    gains: Pan,
    /// Its PannerNode, in place of its pan, if the frame placed it as it
    /// started.
    spatial: Option<Spatial>,
    /// A streamed clip's frames, until it ends, and where in them the voice
    /// started (`first`, a frame of its clip) and how often it went round.
    stream: Option<Reader>,
    first: usize,
    passes: u64,
}

impl Voice {
    /// Starts at sample `now`, `start` seconds into `clip`, its stream from
    /// `streams` if it streams.
    #[allow(clippy::too_many_arguments, reason = "a voice's start")]
    fn new(
        id: Option<u32>,
        clip: Arc<Clip>,
        bus: usize,
        gain: f32,
        pan: f32,
        pitch: f32,
        lowpass: f32,
        reverb: f32,
        panner: Option<Placement>,
        looping: bool,
        start: f32,
        fade_in: f32,
        now: u64,
        rate: f64,
        streams: &mut Streams,
    ) -> Voice {
        let length = clip.len() as f64;
        let offset = f64::from(start) * clip.rate;
        let position = if looping && length > 0.0 { offset.rem_euclid(length) } else { offset.max(0.0) };
        let fade = if fade_in > 0.0 { Ramp::new(0.0, 1.0, now, f64::from(fade_in) * rate) } else { Ramp::hold(1.0) };
        let first = position as usize;
        let stream = match &clip.streamed {
            Some(encoded) if position < length => streams.open(encoded, clip.rate, first),
            _ => None,
        };
        Voice {
            id,
            // A clip that streams but finds no stream free plays nothing.
            ended: position >= length || (clip.streamed.is_some() && stream.is_none()),
            clip,
            bus,
            position,
            looping,
            pitch: Glide::at(pitch.max(0.0)),
            gain: Glide::at(gain),
            pan: Glide::at(pan.clamp(-1.0, 1.0)),
            send: Glide::at(reverb),
            lowpass: Lowpass::new(
                cutoff(panner.map_or(lowpass, |p| lowpass.min(p.occlusion_lowpass(OPEN))), rate),
                rate,
            ),
            fade,
            fade_out: 0.0,
            leaving: false,
            seen: true,
            panned: f32::NAN,
            gains: Pan { left: 1.0, right: 1.0, rightward: false },
            spatial: panner.map(Spatial::new),
            stream,
            first,
            passes: 0,
        }
    }

    /// Fades out from where it is, then goes.
    fn leave(&mut self, now: u64, rate: f64) {
        let current = self.fade.at(now);
        let seconds = f64::from(self.fade_out).max(FADE_SECONDS);
        self.fade = Ramp::new(current, 0.0, now, seconds * rate);
        self.leaving = true;
    }

    fn gone(&self, now: u64) -> bool {
        (self.leaving && self.fade.done(now)) || (self.ended && (self.leaving || self.id.is_none()))
    }

    /// One quantum into its bus's dry and send inputs.
    fn render(
        &mut self,
        now: u64,
        rate: f64,
        glide: f64,
        quantum_glide: f64,
        dry: &mut [[f32; 2]; QUANTUM],
        send: &mut [[f32; 2]; QUANTUM],
    ) {
        if self.ended {
            return;
        }
        // A stream not begun yet is waited for.
        if let Some(stream) = &mut self.stream {
            if !stream.begun() {
                return;
            }
            stream.begin(self.looping);
        }
        let frames = &self.clip.frames;
        let length = self.clip.len();
        let end = length as f64;
        // playbackRate is k-rate: its value at the quantum's start, modulated by smoothed Doppler shift.
        let doppler = self.spatial.as_mut().map_or(1.0, |s| f64::from(s.next_doppler(quantum_glide)));
        let wet_scale = self.spatial.as_ref().map_or(1.0, Spatial::wet_scale);
        let step = f64::from(self.pitch.next(quantum_glide)) * doppler * self.clip.rate / rate;
        let mono = self.clip.mono;
        // Fast path: non-streamed, non-spatial, settled 1:1 playback with open lowpass inside clip bounds.
        if self.stream.is_none()
            && self.spatial.is_none()
            && self.lowpass.is_open_settled()
            && self.gain.is_settled()
            && self.pan.is_settled()
            && self.send.is_settled()
            && step == 1.0
            && let Some(fade) = self.fade.settled_at(now, now + QUANTUM as u64)
            && self.position >= 0.0
            && self.position == (self.position as usize) as f64
        {
            let start_idx = self.position as usize;
            if start_idx + QUANTUM <= length {
                let pan = self.pan.current();
                if pan.to_bits() != self.panned.to_bits() {
                    self.panned = pan;
                    let (left, right) = if mono { mono_gains(pan) } else { pan_gains(pan) };
                    self.gains = Pan { left, right, rightward: pan > 0.0 };
                }
                let gains = self.gains;
                let voice_gain = self.gain.current();
                let rev = self.send.current() * wet_scale;
                let src = &frames[start_idx..start_idx + QUANTUM];
                if mono {
                    if rev == 0.0 {
                        for i in 0..QUANTUM {
                            let x = src[i][0] * fade * voice_gain;
                            dry[i][0] += x * gains.left;
                            dry[i][1] += x * gains.right;
                        }
                    } else {
                        for i in 0..QUANTUM {
                            let x = src[i][0] * fade * voice_gain;
                            let ol = x * gains.left;
                            let or = x * gains.right;
                            dry[i][0] += ol;
                            dry[i][1] += or;
                            send[i][0] += ol * rev;
                            send[i][1] += or * rev;
                        }
                    }
                    self.lowpass.seed_open_pair(0, src[QUANTUM - 2][0], src[QUANTUM - 1][0]);
                } else {
                    if gains.rightward {
                        if rev == 0.0 {
                            for i in 0..QUANTUM {
                                let l = src[i][0] * fade * voice_gain;
                                let r = src[i][1] * fade * voice_gain;
                                dry[i][0] += l * gains.left;
                                dry[i][1] += r + l * gains.right;
                            }
                        } else {
                            for i in 0..QUANTUM {
                                let l = src[i][0] * fade * voice_gain;
                                let r = src[i][1] * fade * voice_gain;
                                let ol = l * gains.left;
                                let or = r + l * gains.right;
                                dry[i][0] += ol;
                                dry[i][1] += or;
                                send[i][0] += ol * rev;
                                send[i][1] += or * rev;
                            }
                        }
                    } else if rev == 0.0 {
                        for i in 0..QUANTUM {
                            let l = src[i][0] * fade * voice_gain;
                            let r = src[i][1] * fade * voice_gain;
                            dry[i][0] += l + r * gains.left;
                            dry[i][1] += r * gains.right;
                        }
                    } else {
                        for i in 0..QUANTUM {
                            let l = src[i][0] * fade * voice_gain;
                            let r = src[i][1] * fade * voice_gain;
                            let ol = l + r * gains.left;
                            let or = r * gains.right;
                            dry[i][0] += ol;
                            dry[i][1] += or;
                            send[i][0] += ol * rev;
                            send[i][1] += or * rev;
                        }
                    }
                    self.lowpass.seed_open_pair(0, src[QUANTUM - 2][0], src[QUANTUM - 1][0]);
                    self.lowpass.seed_open_pair(1, src[QUANTUM - 2][1], src[QUANTUM - 1][1]);
                }
                self.position = (start_idx + QUANTUM) as f64;
                return;
            }
        }
        // Frame `index` of a streamed clip, by its place in the stream.
        let (first, whole) = (self.first as u64, length as u64);
        for i in 0..QUANTUM {
            if self.position >= end {
                if self.looping && length > 0 {
                    self.passes += (self.position / end) as u64;
                    self.position %= end;
                } else {
                    self.ended = true;
                    // Done with its stream: another voice may have it.
                    self.stream = None;
                    return;
                }
            }
            let index = self.position as usize;
            let fraction = self.position - index as f64;
            let next = if index + 1 < length {
                index + 1
            } else if self.looping {
                0
            } else {
                index
            };
            let (a, b) = match &mut self.stream {
                Some(stream) => {
                    let at = self.passes * whole + index as u64 - first;
                    (stream.frame(at), stream.frame(at + u64::from(next != index)))
                }
                None => (frames[index], frames[next]),
            };
            let left = (f64::from(a[0]) * (1.0 - fraction) + f64::from(b[0]) * fraction) as f32;
            let right =
                if mono { left } else { (f64::from(a[1]) * (1.0 - fraction) + f64::from(b[1]) * fraction) as f32 };
            self.position += step;
            self.lowpass.step(glide, rate);
            let fade = self.fade.at(now + i as u64) as f32;
            let voice_gain = self.gain.next(glide);
            let pan = self.pan.next(glide);
            let reverb = self.send.next(glide);
            let (pan, direct_gain, spatial_send_gain) = match &mut self.spatial {
                Some(spatial) => {
                    let (pan, placed_direct, placed_send) = spatial.next(glide, mono, rate, i == 0);
                    (pan, voice_gain * placed_direct, Some(voice_gain * placed_send))
                }
                None => {
                    if pan.to_bits() != self.panned.to_bits() {
                        self.panned = pan;
                        let (left, right) = if mono { mono_gains(pan) } else { pan_gains(pan) };
                        self.gains = Pan { left, right, rightward: pan > 0.0 };
                    }
                    (self.gains, voice_gain, None)
                }
            };
            let (mut out_left, mut out_right) = if mono {
                let x = self.lowpass.run(0, left) * fade * direct_gain;
                (x * pan.left, x * pan.right)
            } else {
                let l = self.lowpass.run(0, left) * fade * direct_gain;
                let r = self.lowpass.run(1, right) * fade * direct_gain;
                if pan.rightward { (l * pan.left, r + l * pan.right) } else { (l + r * pan.left, r * pan.right) }
            };
            if let Some(spatial) = &mut self.spatial {
                (out_left, out_right) = spatial.filter_binaural(out_left, out_right);
            }
            dry[i][0] += out_left;
            dry[i][1] += out_right;
            let rev = reverb * wet_scale;
            if let Some(send_gain) = spatial_send_gain {
                // Feed pre-pan, pre-binaural, pre-occlusion mono equally into both stereo convolver
                // inputs so hard-panned or occluded sources properly excite L/R early reflections
                // and room reverberation around obstacles without double-binauralization.
                let raw_mono = if mono { left } else { 0.5 * (left + right) };
                let s = raw_mono * fade * send_gain * std::f32::consts::FRAC_1_SQRT_2 * rev;
                send[i][0] += s;
                send[i][1] += s;
            } else {
                send[i][0] += out_left * rev;
                send[i][1] += out_right * rev;
            }
        }
        if let Some(stream) = &mut self.stream {
            stream.end(self.passes * whole + self.position as u64 - first);
        }
    }
}

#[derive(Clone, Copy)]
enum EqKind {
    LowShelf,
    Peaking,
    HighShelf,
}

/// One stereo Audio EQ Cookbook biquad band (bypassed cleanly at 0 dB).
struct EqBand {
    kind: EqKind,
    freq: f64,
    gain_db: f32,
    active: bool,
    b: [f64; 3],
    a: [f64; 2],
    state: [[f64; 4]; 2],
}

impl EqBand {
    fn new(kind: EqKind, freq: f64) -> Self {
        Self { kind, freq, gain_db: 0.0, active: false, b: [1.0, 0.0, 0.0], a: [0.0, 0.0], state: [[0.0; 4]; 2] }
    }

    fn set_gain(&mut self, gain_db: f32, rate: f64) {
        let gain_db = if gain_db.is_finite() { gain_db.clamp(-24.0, 24.0) } else { 0.0 };
        if gain_db.to_bits() == self.gain_db.to_bits() {
            return;
        }
        self.gain_db = gain_db;
        if gain_db.abs() < 1e-4 {
            self.active = false;
            self.b = [1.0, 0.0, 0.0];
            self.a = [0.0, 0.0];
            return;
        }
        self.active = true;
        let w0 = 2.0 * PI * (self.freq / rate).clamp(1e-4, 0.49);
        let cos_w = w0.cos();
        let sin_w = w0.sin();
        let amp = 10f64.powf(f64::from(gain_db) / 40.0);
        let (b0, b1, b2, a0, a1, a2) = match self.kind {
            EqKind::LowShelf => {
                let alpha = sin_w * 0.5 * 2f64.sqrt();
                let t = 2.0 * amp.sqrt() * alpha;
                (
                    amp * ((amp + 1.0) - (amp - 1.0) * cos_w + t),
                    2.0 * amp * ((amp - 1.0) - (amp + 1.0) * cos_w),
                    amp * ((amp + 1.0) - (amp - 1.0) * cos_w - t),
                    (amp + 1.0) + (amp - 1.0) * cos_w + t,
                    -2.0 * ((amp - 1.0) + (amp + 1.0) * cos_w),
                    (amp + 1.0) + (amp - 1.0) * cos_w - t,
                )
            }
            EqKind::Peaking => {
                let alpha = sin_w * 0.5; // Q = 1.0
                (1.0 + alpha * amp, -2.0 * cos_w, 1.0 - alpha * amp, 1.0 + alpha / amp, -2.0 * cos_w, 1.0 - alpha / amp)
            }
            EqKind::HighShelf => {
                let alpha = sin_w * 0.5 * 2f64.sqrt();
                let t = 2.0 * amp.sqrt() * alpha;
                (
                    amp * ((amp + 1.0) + (amp - 1.0) * cos_w + t),
                    -2.0 * amp * ((amp - 1.0) + (amp + 1.0) * cos_w),
                    amp * ((amp + 1.0) + (amp - 1.0) * cos_w - t),
                    (amp + 1.0) - (amp - 1.0) * cos_w + t,
                    2.0 * ((amp - 1.0) - (amp + 1.0) * cos_w),
                    (amp + 1.0) - (amp - 1.0) * cos_w - t,
                )
            }
        };
        self.b = [b0 / a0, b1 / a0, b2 / a0];
        self.a = [a1 / a0, a2 / a0];
    }

    fn run(&mut self, channel: usize, input: f32) -> f32 {
        if !self.active {
            return input;
        }
        let [x1, x2, y1, y2] = self.state[channel];
        let x = f64::from(input);
        let y = self.b[0] * x + self.b[1] * x1 + self.b[2] * x2 - self.a[0] * y1 - self.a[1] * y2;
        let y = if y.abs() < 1e-30 { 0.0 } else { y };
        self.state[channel] = [x, x1, y, y1];
        y as f32
    }
}

struct Bus {
    active: bool,
    /// Its gain as the frame has it, before the developer's mute and solo.
    level: f32,
    gain: Glide,
    reverb: Glide,
    /// Channels 0 and 1 filter the dry input, 2 and 3 the send.
    lowpass: Lowpass,
    eq_low: EqBand,
    eq_mid: EqBand,
    eq_high: EqBand,
    eq_any: bool,
    comp_threshold: f64,
    inv_comp_threshold: f64,
    comp_exp: f64,
    comp_on: bool,
    duck_bus: usize,
    duck_amount: f64,
    duck_on: bool,
    dyn_attack: f64,
    dyn_release: f64,
    env: f64,
    dyn_gain: f64,
    last_rms: f64,
    dry: [[f32; 2]; QUANTUM],
    send: [[f32; 2]; QUANTUM],
}

impl Bus {
    fn new(rate: f64) -> Bus {
        Bus {
            active: false,
            level: 1.0,
            gain: Glide::at(1.0),
            reverb: Glide::at(0.0),
            lowpass: Lowpass::new(OPEN, rate),
            eq_low: EqBand::new(EqKind::LowShelf, 200.0),
            eq_mid: EqBand::new(EqKind::Peaking, 1000.0),
            eq_high: EqBand::new(EqKind::HighShelf, 4000.0),
            eq_any: false,
            comp_threshold: 0.0,
            inv_comp_threshold: 0.0,
            comp_exp: 0.0,
            comp_on: false,
            duck_bus: 0,
            duck_amount: 0.0,
            duck_on: false,
            dyn_attack: 1.0 - (-1.0 / (0.010 * rate)).exp(),
            dyn_release: 1.0 - (-1.0 / (0.120 * rate)).exp(),
            env: 0.0,
            dyn_gain: 1.0,
            last_rms: 0.0,
            dry: [[0.0; 2]; QUANTUM],
            send: [[0.0; 2]; QUANTUM],
        }
    }

    fn apply_processing(&mut self, spec: BusSpec, rate: f64) {
        let thresh = if spec.comp_threshold.is_finite() { f64::from(spec.comp_threshold.clamp(0.0, 4.0)) } else { 0.0 };
        let ratio = if spec.comp_ratio.is_finite() { f64::from(spec.comp_ratio.max(1.0)) } else { 1.0 };
        self.comp_threshold = thresh;
        self.comp_on = thresh > 0.0 && ratio > 1.0;
        self.inv_comp_threshold = if self.comp_on { 1.0 / thresh } else { 0.0 };
        self.comp_exp = if self.comp_on { 1.0 / ratio - 1.0 } else { 0.0 };
        self.duck_bus = spec.duck_bus.min(MAX_BUSES - 1);
        self.duck_amount = if spec.duck_amount.is_finite() { f64::from(spec.duck_amount.clamp(0.0, 1.0)) } else { 0.0 };
        self.duck_on = self.duck_amount > 0.0;
        self.eq_low.set_gain(spec.eq_low_db, rate);
        self.eq_mid.set_gain(spec.eq_mid_db, rate);
        self.eq_high.set_gain(spec.eq_high_db, rate);
        self.eq_any = self.eq_low.active || self.eq_mid.active || self.eq_high.active;
    }

    /// Starts it at `spec` at once, as a new bus's nodes do, silent if it
    /// is not `audible`.
    fn start(&mut self, spec: BusSpec, rate: f64, audible: bool) {
        self.active = true;
        self.level = spec.gain;
        self.gain = Glide::at(if audible { spec.gain } else { 0.0 });
        self.reverb = Glide::at(spec.reverb);
        self.lowpass = Lowpass::new(cutoff(spec.lowpass, rate), rate);
        self.apply_processing(spec, rate);
    }

    fn glide_to(&mut self, spec: BusSpec, rate: f64, audible: bool) {
        self.level = spec.gain;
        self.gain.set(if audible { spec.gain } else { 0.0 });
        self.reverb.set(spec.reverb);
        self.lowpass.cutoff.set(cutoff(spec.lowpass, rate));
        self.apply_processing(spec, rate);
    }

    #[inline]
    fn run_eq(&mut self, left: f32, right: f32) -> (f32, f32) {
        if !self.eq_any {
            return (left, right);
        }
        let l = self.eq_high.run(0, self.eq_mid.run(0, self.eq_low.run(0, left)));
        let r = self.eq_high.run(1, self.eq_mid.run(1, self.eq_low.run(1, right)));
        (l, r)
    }

    #[inline]
    fn step_dynamics(&mut self, peak: f64, sidechain_rms: f64) -> f32 {
        if !self.comp_on && !self.duck_on {
            self.dyn_gain = 1.0;
            return 1.0;
        }
        let coeff = if peak > self.env { self.dyn_attack } else { self.dyn_release };
        self.env += (peak - self.env) * coeff;
        let mut target = 1.0;
        if self.comp_on && self.env > self.comp_threshold {
            let over = self.env * self.inv_comp_threshold;
            target *= over.powf(self.comp_exp);
        }
        if self.duck_on {
            let duck = (1.0 - self.duck_amount * (sidechain_rms * 2.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);
            target *= duck;
        }
        let g_coeff = if target < self.dyn_gain { self.dyn_attack } else { self.dyn_release };
        self.dyn_gain += (target - self.dyn_gain) * g_coeff;
        self.dyn_gain as f32
    }

    #[inline]
    fn is_plain_settled(&self) -> bool {
        !self.comp_on
            && !self.duck_on
            && !self.eq_any
            && self.lowpass.is_open_settled()
            && self.gain.is_settled()
            && self.reverb.is_settled()
    }
}

/// The mix, as WebAudio would play it, at `rate` samples a second.
pub struct Mixer {
    rate: f64,
    glide: f64,
    quantum_glide: f64,
    /// The sample the next quantum starts at.
    now: u64,
    voices: Vec<Voice>,
    /// O(1) voice reconciliation map (`voice.id -> index in voices`).
    voice_by_id: HashMap<u32, usize>,
    synths: SynthEngine,
    buses: Vec<Bus>,
    reverb: Convolver,
    reverb_outgoing: Option<Convolver>,
    reverb_fade: f64,
    reverb_fade_step: f64,
    reverb_room: RoomSpec,
    reverb_pending: Option<RoomSpec>,
    reverb_worker: Option<ReverbWorker>,
    reverb_requested: Option<RoomSpec>,
    master_in: [[f32; 2]; QUANTUM],
    reverb_in: [[f32; 2]; QUANTUM],
    reverb_out: [[f32; 2]; QUANTUM],
    reverb_out_prev: [[f32; 2]; QUANTUM],
    out: [[f32; 2]; QUANTUM],
    /// How much of `out` has been handed out.
    taken: usize,
    /// Each bus's output since the meters were last taken.
    meters: [Meter; MAX_BUSES],
    monitor: Monitor,
    streams: Streams,
}

impl Mixer {
    /// A mixer whose streams decode on its own thread, before each quantum:
    /// for recordings and tests, which render faster than time passes.
    pub fn new(rate: f64) -> Mixer {
        Mixer::with_streams(rate, Streams::offline(), false)
    }

    /// A mixer for the output device, whose streams decode and whose reverb IRs
    /// synthesize on background threads so the live audio callback stays glitch-free.
    pub fn live(rate: f64) -> Mixer {
        Mixer::with_streams(rate, Streams::live(), true)
    }

    fn with_streams(rate: f64, streams: Streams, async_reverb: bool) -> Mixer {
        let mut buses: Vec<Bus> = (0..MAX_BUSES).map(|_| Bus::new(rate)).collect();
        buses[0].start(MASTER, rate, true);
        Mixer {
            rate,
            glide: 1.0 - (-1.0 / (GLIDE_SECONDS * rate)).exp(),
            quantum_glide: 1.0 - (-(QUANTUM as f64) / (GLIDE_SECONDS * rate)).exp(),
            now: 0,
            voices: Vec::with_capacity(1024),
            voice_by_id: HashMap::with_capacity(1024),
            synths: SynthEngine::new(),
            buses,
            reverb: Convolver::new(rate),
            reverb_outgoing: None,
            reverb_fade: 1.0,
            reverb_fade_step: 1.0 / (0.25 * rate).max(1.0),
            reverb_room: RoomSpec::Plain { decay: 0.0, damping: 0.0 },
            reverb_pending: None,
            reverb_worker: async_reverb.then(|| ReverbWorker::spawn(rate)),
            reverb_requested: None,
            master_in: [[0.0; 2]; QUANTUM],
            reverb_in: [[0.0; 2]; QUANTUM],
            reverb_out: [[0.0; 2]; QUANTUM],
            reverb_out_prev: [[0.0; 2]; QUANTUM],
            out: [[0.0; 2]; QUANTUM],
            taken: QUANTUM,
            meters: [Meter::default(); MAX_BUSES],
            monitor: Monitor::default(),
            streams,
        }
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Voices, sounds, and active synth voices playing, those fading out included.
    pub fn playing(&self) -> usize {
        self.voices.len() + self.synths.active_count()
    }

    /// Those of them streaming their clip.
    pub fn streaming(&self) -> usize {
        self.voices.iter().filter(|voice| voice.stream.is_some()).count()
    }

    /// How often a stream ran dry so far.
    pub fn underruns(&self) -> u64 {
        self.streams.underruns()
    }

    /// The developer's mute and solo, from the next quantum on: the buses
    /// they silence or bring back glide there.
    pub fn set_monitor(&mut self, monitor: Monitor) {
        if monitor == self.monitor {
            return;
        }
        self.monitor = monitor;
        for (index, bus) in self.buses.iter_mut().enumerate().filter(|(_, bus)| bus.active) {
            bus.gain.set(if monitor.audible(index) { bus.level } else { 0.0 });
        }
    }

    /// Each bus's meter since the last time, and which buses are in use;
    /// the meters start again.
    pub fn take_meters(&mut self) -> ([Meter; MAX_BUSES], [bool; MAX_BUSES]) {
        let used = std::array::from_fn(|index| self.buses[index].active);
        (std::mem::take(&mut self.meters), used)
    }

    /// Bus `index`, and those before it, started plain if they are new, as
    /// the browser makes them.
    fn bus(&mut self, index: usize) -> usize {
        let index = index.min(MAX_BUSES - 1);
        let monitor = self.monitor;
        for (at, bus) in self.buses[..=index].iter_mut().enumerate().filter(|(_, bus)| !bus.active) {
            bus.start(PLAIN, self.rate, monitor.audible(at));
        }
        index
    }

    fn install_reverb_convolver(&mut self, room: RoomSpec, next: Convolver) {
        let prev = std::mem::replace(&mut self.reverb, next);
        if let Some(old_out) = self.reverb_outgoing.replace(prev)
            && let Some(worker) = &self.reverb_worker
        {
            worker.retire(old_out);
        }
        self.reverb_fade = 0.0;
        self.reverb_room = room;
    }

    fn request_reverb_room(&mut self, room: RoomSpec) {
        if let Some(worker) = &self.reverb_worker {
            if self.reverb_requested.as_ref().is_some_and(|req| !room.differs_significantly(req)) {
                return;
            }
            if worker.submit(room) {
                self.reverb_requested = Some(room);
                self.reverb_pending = None;
            } else {
                self.reverb_pending = Some(room);
            }
        } else if self.reverb_outgoing.is_none() {
            let next = room.build_convolver(self.rate);
            self.install_reverb_convolver(room, next);
            self.reverb_pending = None;
        } else {
            self.reverb_pending = Some(room);
        }
    }

    fn poll_reverb_worker(&mut self) {
        if self.reverb_worker.is_none() {
            return;
        }
        if let Some(pending) = self.reverb_pending.take() {
            self.request_reverb_room(pending);
        }
        if self.reverb_outgoing.is_none()
            && let Some(worker) = &self.reverb_worker
            && let Some((room, conv)) = worker.poll_latest()
        {
            self.reverb_requested = None;
            self.install_reverb_convolver(room, conv);
        }
    }

    /// A frame's sound, from the next quantum on: buses glide to theirs (those
    /// it lacks to plain), voices start, return, change or leave by id, and
    /// its sounds start.
    pub fn apply(&mut self, mix: &Mix) {
        let (now, rate) = (self.now, self.rate);
        self.poll_reverb_worker();
        let next_room = RoomSpec::from_mix(mix);
        // Quantized room hysteresis: avoids rebuilding IRs every frame while walking across a zone blend
        // or acoustic ray-trace sweep, crossfading persistent convolvers smoothly without tail cuts.
        if next_room.differs_significantly(&self.reverb_room) {
            self.request_reverb_room(next_room);
        }
        for (index, bus) in self.buses.iter_mut().enumerate() {
            let spec = mix.buses.get(index).copied();
            let audible = self.monitor.audible(index);
            match (spec, bus.active) {
                (Some(spec), true) => bus.glide_to(spec, rate, audible),
                (Some(spec), false) => bus.start(spec, rate, audible),
                (None, true) => bus.glide_to(if index == 0 { MASTER } else { PLAIN }, rate, audible),
                (None, false) => {}
            }
        }
        for voice in &mut self.voices {
            voice.seen = false;
        }
        for spec in &mix.voices {
            let bus = self.bus(spec.bus);
            let found = self.voice_by_id.get(&spec.id).copied();
            if let Some(index) = found {
                let voice = &mut self.voices[index];
                if voice.seen {
                    // A second voice with an id this frame already has.
                    continue;
                }
                if Arc::ptr_eq(&voice.clip, &spec.clip)
                    && voice.bus == bus
                    && voice.spatial.is_some() == spec.panner.is_some()
                {
                    voice.seen = true;
                    if let (Some(spatial), Some(placement)) = (&mut voice.spatial, spec.panner) {
                        spatial.set(placement);
                    }
                    voice.gain.set(spec.gain);
                    voice.pan.set(spec.pan.clamp(-1.0, 1.0));
                    voice.pitch.set(spec.pitch.max(0.0));
                    voice.send.set(spec.reverb);
                    let eff_lp = spec.panner.map_or(spec.lowpass, |p| spec.lowpass.min(p.occlusion_lowpass(OPEN)));
                    voice.lowpass.cutoff.set(cutoff(eff_lp, rate));
                    voice.looping = spec.looping;
                    voice.fade_out = spec.fade_out;
                    if voice.leaving {
                        let current = voice.fade.at(now);
                        let seconds = f64::from(spec.fade_in).max(FADE_SECONDS);
                        voice.fade = Ramp::new(current, 1.0, now, seconds * rate);
                        voice.leaving = false;
                    }
                    continue;
                }
                // Another clip or bus, or placed or no longer: the old
                // fades out as the new starts.
                voice.id = None;
                voice.fade_out = spec.fade_out;
                voice.leave(now, rate);
            }
            let mut voice = Voice::new(
                Some(spec.id),
                spec.clip.clone(),
                bus,
                spec.gain,
                spec.pan,
                spec.pitch,
                spec.lowpass,
                spec.reverb,
                spec.panner,
                spec.looping,
                spec.start,
                spec.fade_in,
                now,
                rate,
                &mut self.streams,
            );
            voice.fade_out = spec.fade_out;
            let new_index = self.voices.len();
            self.voices.push(voice);
            self.voice_by_id.insert(spec.id, new_index);
        }
        for voice in &mut self.voices {
            if voice.id.is_some() && !voice.seen && !voice.leaving {
                voice.leave(now, rate);
            }
        }
        for sound in &mix.sounds {
            let bus = self.bus(sound.bus);
            self.voices.push(Voice::new(
                None,
                sound.clip.clone(),
                bus,
                sound.gain,
                sound.pan,
                sound.pitch,
                sound.lowpass,
                sound.reverb,
                sound.panner,
                false,
                0.0,
                0.0,
                now,
                rate,
                &mut self.streams,
            ));
        }
        for synth in &mix.synths {
            let _ = self.bus(synth.bus);
        }
        self.synths.apply(&mix.synths);
    }

    /// The next quantum, into `out`.
    fn quantum(&mut self) {
        let (now, rate, glide) = (self.now, self.rate, self.glide);
        self.streams.pump();
        for bus in self.buses.iter_mut().filter(|bus| bus.active) {
            bus.dry = [[0.0; 2]; QUANTUM];
            bus.send = [[0.0; 2]; QUANTUM];
        }
        for voice in &mut self.voices {
            let bus = &mut self.buses[voice.bus];
            voice.render(now, rate, glide, self.quantum_glide, &mut bus.dry, &mut bus.send);
        }
        let buses_ref = &mut self.buses;
        self.synths.render(rate, QUANTUM, |bus_idx, i, l, r, rev| {
            let bus = &mut buses_ref[bus_idx.min(MAX_BUSES - 1)];
            bus.dry[i][0] += l;
            bus.dry[i][1] += r;
            bus.send[i][0] += l * rev;
            bus.send[i][1] += r * rev;
        });
        let end = now + QUANTUM as u64;
        let mut i = 0;
        while i < self.voices.len() {
            if self.voices[i].gone(end) {
                let removed = self.voices.swap_remove(i);
                if let Some(id) = removed.id
                    && self.voice_by_id.get(&id) == Some(&i)
                {
                    self.voice_by_id.remove(&id);
                }
                if let Some(moved) = self.voices.get(i)
                    && let Some(id) = moved.id
                {
                    self.voice_by_id.insert(id, i);
                }
            } else {
                i += 1;
            }
        }
        let prev_rms: [f64; MAX_BUSES] = std::array::from_fn(|i| self.buses[i].last_rms);
        // The master's own voices go straight to it and to the reverb.
        {
            let (master, buses) = self.buses.split_first_mut().expect("a master bus");
            self.master_in = master.dry;
            self.reverb_in = master.send;
            for (bus, meter) in buses.iter_mut().zip(&mut self.meters[1..]).filter(|(bus, _)| bus.active) {
                let mut sq_sum = 0.0f64;
                if bus.is_plain_settled() {
                    bus.dyn_gain = 1.0;
                    let gain = bus.gain.current();
                    let reverb = bus.reverb.current();
                    for i in 0..QUANTUM {
                        let left = bus.dry[i][0] * gain;
                        let right = bus.dry[i][1] * gain;
                        let wet_left = bus.send[i][0] * gain;
                        let wet_right = bus.send[i][1] * gain;
                        sq_sum += f64::from(left) * f64::from(left) + f64::from(right) * f64::from(right);
                        meter.add([left, right]);
                        self.master_in[i][0] += left;
                        self.master_in[i][1] += right;
                        self.reverb_in[i][0] += wet_left + left * reverb;
                        self.reverb_in[i][1] += wet_right + right * reverb;
                    }
                    bus.lowpass.seed_open_pair(0, bus.dry[QUANTUM - 2][0], bus.dry[QUANTUM - 1][0]);
                    bus.lowpass.seed_open_pair(1, bus.dry[QUANTUM - 2][1], bus.dry[QUANTUM - 1][1]);
                    bus.lowpass.seed_open_pair(2, bus.send[QUANTUM - 2][0], bus.send[QUANTUM - 1][0]);
                    bus.lowpass.seed_open_pair(3, bus.send[QUANTUM - 2][1], bus.send[QUANTUM - 1][1]);
                } else {
                    let sc_rms = prev_rms[bus.duck_bus];
                    for i in 0..QUANTUM {
                        bus.lowpass.step(glide, rate);
                        let gain = bus.gain.next(glide);
                        let reverb = bus.reverb.next(glide);
                        let [dry_left, dry_right] = bus.dry[i];
                        let [send_left, send_right] = bus.send[i];
                        let lp_l = bus.lowpass.run(0, dry_left);
                        let lp_r = bus.lowpass.run(1, dry_right);
                        let (eq_l, eq_r) = bus.run_eq(lp_l, lp_r);
                        let peak = f64::from(eq_l.abs().max(eq_r.abs())) * f64::from(gain.abs());
                        let dyn_gain = bus.step_dynamics(peak, sc_rms);
                        let left = eq_l * gain * dyn_gain;
                        let right = eq_r * gain * dyn_gain;
                        let wet_left = bus.lowpass.run(2, send_left) * gain;
                        let wet_right = bus.lowpass.run(3, send_right) * gain;
                        sq_sum += f64::from(left) * f64::from(left) + f64::from(right) * f64::from(right);
                        meter.add([left, right]);
                        self.master_in[i][0] += left;
                        self.master_in[i][1] += right;
                        self.reverb_in[i][0] += wet_left + left * reverb;
                        self.reverb_in[i][1] += wet_right + right * reverb;
                    }
                }
                bus.last_rms = (sq_sum / (2 * QUANTUM) as f64).sqrt();
            }
        }
        self.reverb_out = [[0.0; 2]; QUANTUM];
        self.reverb.process(&self.reverb_in, &mut self.reverb_out);
        if let Some(outgoing) = &mut self.reverb_outgoing {
            self.reverb_out_prev = [[0.0; 2]; QUANTUM];
            let silence = [[0.0f32; 2]; QUANTUM];
            outgoing.process(&silence, &mut self.reverb_out_prev);
            for i in 0..QUANTUM {
                let w_in = self.reverb_fade.clamp(0.0, 1.0) as f32;
                let w_out = 1.0 - w_in;
                self.reverb_out[i][0] = self.reverb_out[i][0] * w_in + self.reverb_out_prev[i][0] * w_out;
                self.reverb_out[i][1] = self.reverb_out[i][1] * w_in + self.reverb_out_prev[i][1] * w_out;
                self.reverb_fade = (self.reverb_fade + self.reverb_fade_step).min(1.0);
            }
            if self.reverb_fade >= 1.0 {
                if let Some(old_out) = self.reverb_outgoing.take()
                    && let Some(worker) = &self.reverb_worker
                {
                    worker.retire(old_out);
                }
                if let Some(pending) = self.reverb_pending.take()
                    && pending.differs_significantly(&self.reverb_room)
                {
                    self.request_reverb_room(pending);
                }
            }
        }
        self.poll_reverb_worker();
        let master = &mut self.buses[0];
        let mut master_sq = 0.0f64;
        if master.is_plain_settled() {
            master.dyn_gain = 1.0;
            let back = master.reverb.current();
            let gain = master.gain.current();
            let mut prev_l = 0.0f32;
            let mut prev_r = 0.0f32;
            let mut last_l = 0.0f32;
            let mut last_r = 0.0f32;
            for i in 0..QUANTUM {
                let left = self.master_in[i][0] + self.reverb_out[i][0] * back;
                let right = self.master_in[i][1] + self.reverb_out[i][1] * back;
                if i == QUANTUM - 2 {
                    prev_l = left;
                    prev_r = right;
                } else if i == QUANTUM - 1 {
                    last_l = left;
                    last_r = right;
                }
                let out_l = left * gain;
                let out_r = right * gain;
                master_sq += f64::from(out_l) * f64::from(out_l) + f64::from(out_r) * f64::from(out_r);
                self.out[i] = [out_l, out_r];
                self.meters[0].add(self.out[i]);
            }
            master.lowpass.seed_open_pair(0, prev_l, last_l);
            master.lowpass.seed_open_pair(1, prev_r, last_r);
        } else {
            let master_sc = prev_rms[master.duck_bus];
            for i in 0..QUANTUM {
                let back = master.reverb.next(glide);
                master.lowpass.step(glide, rate);
                let gain = master.gain.next(glide);
                let left = self.master_in[i][0] + self.reverb_out[i][0] * back;
                let right = self.master_in[i][1] + self.reverb_out[i][1] * back;
                let lp_l = master.lowpass.run(0, left);
                let lp_r = master.lowpass.run(1, right);
                let (eq_l, eq_r) = master.run_eq(lp_l, lp_r);
                let peak = f64::from(eq_l.abs().max(eq_r.abs())) * f64::from(gain.abs());
                let dyn_gain = master.step_dynamics(peak, master_sc);
                let out_l = eq_l * gain * dyn_gain;
                let out_r = eq_r * gain * dyn_gain;
                master_sq += f64::from(out_l) * f64::from(out_l) + f64::from(out_r) * f64::from(out_r);
                self.out[i] = [out_l, out_r];
                self.meters[0].add(self.out[i]);
            }
        }
        master.last_rms = (master_sq / (2 * QUANTUM) as f64).sqrt();
        self.now = end;
    }

    /// The next `out.len()` samples of the mix.
    pub fn render(&mut self, out: &mut [[f32; 2]]) {
        let mut written = 0;
        while written < out.len() {
            if self.taken == QUANTUM {
                self.quantum();
                self.taken = 0;
            }
            let count = (QUANTUM - self.taken).min(out.len() - written);
            out[written..written + count].copy_from_slice(&self.out[self.taken..self.taken + count]);
            self.taken += count;
            written += count;
        }
    }
}

#[cfg(test)]
mod tests;
