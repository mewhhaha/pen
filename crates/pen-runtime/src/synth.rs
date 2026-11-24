//! Real-time procedural and granular DSP synthesizer shared by native and browser
//! hosts. Supports PolyBLEP/PolyBLAMP anti-aliased waveforms (sine, saw, square,
//! triangle), deterministic white noise, and Hann-windowed multi-grain granular
//! clouds with Nyquist-safe harmonic rolloff, routed through a click-free ADSR
//! envelope, an unconditionally stable trapezoidal zero-delay-feedback (Cytomic
//! ZDF) resonant lowpass filter, and equal-power stereo panning.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "audio sample math")]

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use pen_protocol::SynthVoice;
use pen_protocol::boundary::{AUDIO_OPEN, MAX_BUSES, MAX_SYNTHS};

/// Synthesis oscillator / engine mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynthMode {
    Sine = 0,
    Saw = 1,
    Square = 2,
    Triangle = 3,
    Noise = 4,
    Granular = 5,
}

impl SynthMode {
    pub fn from_u32(value: u32) -> Self {
        match value {
            1 => Self::Saw,
            2 => Self::Square,
            3 => Self::Triangle,
            4 => Self::Noise,
            5 => Self::Granular,
            _ => Self::Sine,
        }
    }
}

/// Sanitized real-time synth voice parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynthSpec {
    pub id: u32,
    pub mode: SynthMode,
    pub bus: usize,
    pub frequency: f32,
    pub gain: f32,
    pub pan: f32,
    pub detune_cents: f32,
    pub attack: f32,
    pub decay: f32,
    pub sustain: f32,
    pub release: f32,
    pub cutoff: f32,
    pub resonance: f32,
    pub grain_size: f32,
    pub grain_pitch: f32,
    pub reverb: f32,
}

fn finite_clamp(v: f32, default: f32, lo: f32, hi: f32) -> f32 {
    if v.is_finite() { v.clamp(lo, hi) } else { default }
}

impl SynthSpec {
    pub fn from_wire(v: &SynthVoice) -> Self {
        let cutoff = if v.cutoff.is_finite() && v.cutoff > 0.0 {
            v.cutoff.clamp(20.0, AUDIO_OPEN as f32)
        } else {
            AUDIO_OPEN as f32
        };
        let resonance = if v.resonance.is_finite() && v.resonance > 0.0 {
            v.resonance.clamp(0.5, 20.0)
        } else {
            std::f32::consts::FRAC_1_SQRT_2
        };
        let grain_size = if v.grain_size.is_finite() && v.grain_size > 0.0 {
            v.grain_size.clamp(0.01, 0.5)
        } else {
            0.06
        };
        let grain_pitch = if v.grain_pitch.is_finite() && v.grain_pitch > 0.0 {
            v.grain_pitch.clamp(0.25, 4.0)
        } else {
            1.0
        };
        Self {
            id: v.id,
            mode: SynthMode::from_u32(v.mode),
            bus: (v.bus as usize).min(MAX_BUSES - 1),
            frequency: finite_clamp(v.frequency, 440.0, 10.0, 20_000.0),
            gain: finite_clamp(v.gain, 0.0, 0.0, 4.0),
            pan: finite_clamp(v.pan, 0.0, -1.0, 1.0),
            detune_cents: finite_clamp(v.detune, 0.0, -2400.0, 2400.0),
            attack: finite_clamp(v.attack, 0.005, 0.0, 10.0),
            decay: finite_clamp(v.decay, 0.05, 0.0, 10.0),
            sustain: finite_clamp(v.sustain, 1.0, 0.0, 1.0),
            release: finite_clamp(v.release, 0.05, 0.005, 10.0),
            cutoff,
            resonance,
            grain_size,
            grain_pitch,
            reverb: finite_clamp(v.reverb, 0.0, 0.0, 2.0),
        }
    }
}

/// Deterministic xorshift32 PRNG for noise and granular clouds.
#[derive(Clone, Copy, Debug)]
struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed ^ 0x9e37_79b9 | 1)
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Uniform in `[-1.0, 1.0]`.
    fn bipolar(&mut self) -> f64 {
        f64::from(self.next_u32()) / 2_147_483_648.0 - 1.0
    }
}

/// PolyBLEP residual for anti-aliased step discontinuities (saw, square).
fn poly_blep(t: f64, dt: f64) -> f64 {
    if dt <= 0.0 || dt >= 0.5 {
        return 0.0;
    }
    if t < dt {
        let x = t / dt;
        x + x - x * x - 1.0
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

/// PolyBLAMP residual for anti-aliased slope discontinuities (triangle peaks).
fn poly_blamp(t: f64, dt: f64) -> f64 {
    if dt <= 0.0 || dt >= 0.5 {
        return 0.0;
    }
    if t < dt {
        let x = t / dt - 1.0;
        -(1.0 / 3.0) * x * x * x
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt + 1.0;
        (1.0 / 3.0) * x * x * x
    } else {
        0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EnvStage {
    Attack,
    Decay,
    Sustain,
    Release,
    Done,
}

#[derive(Clone, Copy, Debug)]
struct Adsr {
    stage: EnvStage,
    level: f64,
    release_start: f64,
    release_pos: f64,
    /// Minimum de-click attack duration enforced when retriggering from a live release.
    retrigger_min_attack: f64,
}

impl Adsr {
    fn new(attack: f32) -> Self {
        if attack <= 1e-4 {
            Self { stage: EnvStage::Decay, level: 1.0, release_start: 1.0, release_pos: 0.0, retrigger_min_attack: 0.0 }
        } else {
            Self { stage: EnvStage::Attack, level: 0.0, release_start: 0.0, release_pos: 0.0, retrigger_min_attack: 0.0 }
        }
    }

    fn trigger_release(&mut self) {
        if !matches!(self.stage, EnvStage::Release | EnvStage::Done) {
            self.stage = EnvStage::Release;
            self.release_start = self.level;
            self.release_pos = 0.0;
        }
    }

    /// Smoothly retriggers from the current envelope level without clicking even when `attack == 0`.
    fn retrigger(&mut self, attack: f32) {
        if matches!(self.stage, EnvStage::Release | EnvStage::Done) {
            self.stage = EnvStage::Attack;
            self.retrigger_min_attack = if attack <= 1e-4 { 0.003 } else { 0.0 };
        }
    }

    fn step(&mut self, spec: &SynthSpec, rate: f64) -> f64 {
        match self.stage {
            EnvStage::Attack => {
                let atk_sec = f64::from(spec.attack).max(self.retrigger_min_attack);
                let dur = (atk_sec * rate).max(1.0);
                self.level += 1.0 / dur;
                if self.level >= 1.0 {
                    self.level = 1.0;
                    self.retrigger_min_attack = 0.0;
                    self.stage = EnvStage::Decay;
                }
            }
            EnvStage::Decay => {
                let sustain = f64::from(spec.sustain);
                if spec.decay <= 1e-4 {
                    self.level = sustain;
                    self.stage = EnvStage::Sustain;
                } else {
                    let dur = (f64::from(spec.decay) * rate).max(1.0);
                    self.level -= (1.0 - sustain) / dur;
                    if self.level <= sustain {
                        self.level = sustain;
                        self.stage = EnvStage::Sustain;
                    }
                }
            }
            EnvStage::Sustain => {
                let target = f64::from(spec.sustain);
                self.level += (target - self.level) * (1.0 - (-1.0 / (0.015 * rate)).exp());
            }
            EnvStage::Release => {
                let dur = (f64::from(spec.release.max(0.005)) * rate).max(1.0);
                self.release_pos += 1.0 / dur;
                if self.release_pos >= 1.0 {
                    self.level = 0.0;
                    self.stage = EnvStage::Done;
                } else {
                    self.level = self.release_start * (1.0 - self.release_pos);
                }
            }
            EnvStage::Done => {
                self.level = 0.0;
            }
        }
        self.level
    }
}

/// Cytomic zero-delay-feedback trapezoidal SVF lowpass (unconditionally stable
/// across all frequencies up to 0.48 * fs and Q up to 20; always runs so
/// integrators track cleanly without bypass-switching clicks).
#[derive(Clone, Copy, Debug, Default)]
struct ZdfLowpass {
    ic1eq: [f64; 2],
    ic2eq: [f64; 2],
}

impl ZdfLowpass {
    fn process(&mut self, ch: usize, input: f64, cutoff: f64, resonance: f64, rate: f64) -> f64 {
        let norm = (cutoff / rate).clamp(1e-4, 0.48);
        let g = (PI * norm).tan();
        let k = 1.0 / resonance.clamp(0.5, 20.0);
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        let v3 = input - self.ic2eq[ch];
        let v1 = a1 * self.ic1eq[ch] + a2 * v3;
        let v2 = self.ic2eq[ch] + a2 * self.ic1eq[ch] + a3 * v3;
        let next1 = 2.0 * v1 - self.ic1eq[ch];
        let next2 = 2.0 * v2 - self.ic2eq[ch];
        self.ic1eq[ch] = if next1.abs() < 1e-30 { 0.0 } else { next1 };
        self.ic2eq[ch] = if next2.abs() < 1e-30 { 0.0 } else { next2 };
        v2
    }
}

const MAX_GRAINS: usize = 8;

#[derive(Clone, Copy, Debug, Default)]
struct Grain {
    active: bool,
    age: u32,
    length: u32,
    phase: f64,
    phase_inc: f64,
    h2_weight: f64,
    pan_l: f64,
    pan_r: f64,
}

struct ActiveSynth {
    spec: SynthSpec,
    phase: f64,
    rng: Rng,
    adsr: Adsr,
    filter: ZdfLowpass,
    grains: [Grain; MAX_GRAINS],
    grain_cooldown: u32,
    seen: bool,
    filter_enabled: bool,
}

impl ActiveSynth {
    fn new(spec: SynthSpec) -> Self {
        Self::with_filter(spec, true)
    }

    fn with_filter(spec: SynthSpec, filter_enabled: bool) -> Self {
        Self {
            adsr: Adsr::new(spec.attack),
            rng: Rng::new(spec.id.wrapping_mul(2_654_435_761) ^ 0x517c_c1b7),
            phase: 0.0,
            filter: ZdfLowpass::default(),
            grains: [Grain::default(); MAX_GRAINS],
            grain_cooldown: 0,
            seen: true,
            filter_enabled,
            spec,
        }
    }

    fn update(&mut self, spec: SynthSpec) {
        self.seen = true;
        self.spec = spec;
        self.adsr.retrigger(spec.attack);
    }

    fn is_releasing(&self) -> bool {
        matches!(self.adsr.stage, EnvStage::Release | EnvStage::Done)
    }

    fn done(&self) -> bool {
        self.adsr.stage == EnvStage::Done
    }

    fn spawn_grain(&mut self, rate: f64) {
        let Some(slot) = self.grains.iter_mut().find(|g| !g.active) else { return };
        let len = (f64::from(self.spec.grain_size) * rate).round().clamp(64.0, rate * 0.5) as u32;
        let spread = f64::from(self.spec.detune_cents.abs().max(15.0));
        let cents = self.rng.bipolar() * spread;
        let freq = f64::from(self.spec.frequency)
            * f64::from(self.spec.grain_pitch)
            * 2f64.powf(cents / 1200.0);
        let freq = freq.clamp(20.0, rate * 0.45);
        // Smoothly roll off the 2nd harmonic before it reaches Nyquist so grains never fold back.
        let h2_weight = ((0.46 * rate - 2.0 * freq) / (0.08 * rate)).clamp(0.0, 1.0) * 0.25;
        let pan = (f64::from(self.spec.pan) + self.rng.bipolar() * 0.35).clamp(-1.0, 1.0);
        let theta = (pan + 1.0) * 0.5 * FRAC_PI_2;
        *slot = Grain {
            active: true,
            age: 0,
            length: len,
            phase: self.rng.bipolar() * 0.5 + 0.5,
            phase_inc: freq / rate,
            h2_weight,
            pan_l: theta.cos(),
            pan_r: theta.sin(),
        };
    }

    fn next_sample(&mut self, rate: f64) -> (f32, f32, f32) {
        let env = self.adsr.step(&self.spec, rate);
        if env <= 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let detune_ratio = 2f64.powf(f64::from(self.spec.detune_cents) / 1200.0);
        let freq = (f64::from(self.spec.frequency) * detune_ratio).clamp(10.0, rate * 0.48);
        let dt = freq / rate;
        let (raw_l, raw_r) = match self.spec.mode {
            SynthMode::Granular => {
                if self.grain_cooldown == 0 {
                    self.spawn_grain(rate);
                    let hop = (f64::from(self.spec.grain_size) * rate * 0.25).round().clamp(32.0, rate * 0.25) as u32;
                    self.grain_cooldown = hop;
                } else {
                    self.grain_cooldown -= 1;
                }
                let mut sum_l = 0.0;
                let mut sum_r = 0.0;
                for g in &mut self.grains {
                    if !g.active {
                        continue;
                    }
                    let u = f64::from(g.age) / f64::from(g.length.max(1));
                    let win = 0.5 * (1.0 - (TAU * u).cos());
                    let osc = (1.0 - g.h2_weight) * (TAU * g.phase).sin()
                        + g.h2_weight * (2.0 * TAU * g.phase).sin();
                    let sample = osc * win * 0.5;
                    sum_l += sample * g.pan_l;
                    sum_r += sample * g.pan_r;
                    g.phase = (g.phase + g.phase_inc) % 1.0;
                    g.age += 1;
                    if g.age >= g.length {
                        g.active = false;
                    }
                }
                (sum_l, sum_r)
            }
            mode => {
                let mono = match mode {
                    SynthMode::Sine => (TAU * self.phase).sin(),
                    SynthMode::Saw => (2.0 * self.phase - 1.0) - poly_blep(self.phase, dt),
                    SynthMode::Square => {
                        let base = if self.phase < 0.5 { 1.0 } else { -1.0 };
                        base + poly_blep(self.phase, dt) - poly_blep((self.phase + 0.5) % 1.0, dt)
                    }
                    SynthMode::Triangle => {
                        // PolyBLAMP-smoothed triangle: slope changes by +8 at phase=0 and -8 at phase=0.5.
                        let naive = 1.0 - 4.0 * (self.phase - 0.5).abs();
                        let t0 = poly_blamp(self.phase, dt);
                        let t1 = poly_blamp((self.phase + 0.5) % 1.0, dt);
                        (naive + 4.0 * dt * (t0 - t1)).clamp(-1.0, 1.0)
                    }
                    SynthMode::Noise => self.rng.bipolar(),
                    SynthMode::Granular => 0.0,
                };
                self.phase = (self.phase + dt) % 1.0;
                let theta = (f64::from(self.spec.pan) + 1.0) * 0.5 * FRAC_PI_2;
                (mono * theta.cos(), mono * theta.sin())
            }
        };
        let (fl, fr) = if self.filter_enabled {
            let cutoff = f64::from(self.spec.cutoff);
            let q = f64::from(self.spec.resonance);
            (self.filter.process(0, raw_l, cutoff, q, rate), self.filter.process(1, raw_r, cutoff, q, rate))
        } else {
            (raw_l, raw_r)
        };
        let scale = env * f64::from(self.spec.gain);
        let out_l = (fl * scale) as f32;
        let out_r = (fr * scale) as f32;
        (out_l, out_r, self.spec.reverb)
    }
}

/// Collection of active procedural/granular synth voices mixed into audio buses.
#[derive(Default)]
pub struct SynthEngine {
    voices: Vec<ActiveSynth>,
}

impl SynthEngine {
    pub fn new() -> Self {
        Self { voices: Vec::with_capacity(MAX_SYNTHS * 2) }
    }

    pub fn active_count(&self) -> usize {
        self.voices.len()
    }

    /// Reconciles the current frame's synth voices against playing voices.
    /// Voices omitted from `specs` enter their ADSR release phase instead of
    /// stopping abruptly; retiring voices are evicted quietest-first if capacity
    /// is full so fresh active voices up to `MAX_SYNTHS` are never starved.
    pub fn apply(&mut self, specs: &[SynthSpec]) {
        for v in &mut self.voices {
            v.seen = false;
        }
        for &spec in specs.iter().take(MAX_SYNTHS) {
            if spec.id == 0 {
                continue;
            }
            if let Some(existing) = self.voices.iter_mut().find(|v| v.spec.id == spec.id) {
                if existing.seen {
                    continue;
                }
                existing.update(spec);
            } else {
                if self.voices.len() >= MAX_SYNTHS * 2 {
                    // Reclaim the quietest retiring voice so active slots are always guaranteed.
                    if let Some((idx, _)) = self
                        .voices
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| !v.seen || v.is_releasing())
                        .min_by(|(_, a), (_, b)| a.adsr.level.total_cmp(&b.adsr.level))
                    {
                        self.voices.swap_remove(idx);
                    }
                }
                if self.voices.len() < MAX_SYNTHS * 2 {
                    self.voices.push(ActiveSynth::new(spec));
                }
            }
        }
        for v in &mut self.voices {
            if !v.seen {
                v.adsr.trigger_release();
            }
        }
    }

    /// Renders `frames` samples per active synth voice into the caller-supplied
    /// per-bus dry & reverb-send accumulators.
    pub fn render<F>(&mut self, rate: f64, frames: usize, mut accumulate: F)
    where
        F: FnMut(usize, usize, f32, f32, f32),
    {
        for voice in &mut self.voices {
            let bus = voice.spec.bus;
            for i in 0..frames {
                let (l, r, rev) = voice.next_sample(rate);
                accumulate(bus, i, l, r, rev);
            }
        }
        self.voices.retain(|v| !v.done());
    }
}

/// Renders a seamlessly loopable, **dry, unfiltered, centered** stereo granular
/// cloud buffer of `frames` samples at `rate` Hz for WebAudio's `AudioBufferSourceNode`
/// so browser filter cutoff, resonance, and stereo pan nodes apply cleanly once
/// without double-filtering or baking global pan.
pub fn render_granular_loop(spec: SynthSpec, rate: f64, frames: usize) -> [Vec<f32>; 2] {
    let frames = frames.clamp(256, 96_000);
    let mut steady = spec;
    steady.mode = SynthMode::Granular;
    steady.attack = 0.0;
    steady.decay = 0.0;
    steady.sustain = 1.0;
    steady.gain = 1.0;
    steady.pan = 0.0;
    let mut synth = ActiveSynth::with_filter(steady, false);
    let warmup = (f64::from(steady.grain_size) * rate).round() as usize;
    for _ in 0..warmup.min(16_384) {
        let _ = synth.next_sample(rate);
    }
    let fade = (frames / 16).clamp(32, 1024);
    let mut left = vec![0.0f32; frames];
    let mut right = vec![0.0f32; frames];
    for i in 0..frames {
        let (l, r, _) = synth.next_sample(rate);
        left[i] = l;
        right[i] = r;
    }
    for i in 0..fade {
        let (tl, tr, _) = synth.next_sample(rate);
        let alpha = i as f32 / fade as f32;
        left[i] = left[i] * alpha + tl * (1.0 - alpha);
        right[i] = right[i] * alpha + tr * (1.0 - alpha);
    }
    [left, right]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_spec(id: u32, mode: SynthMode, freq: f32) -> SynthSpec {
        SynthSpec {
            id,
            mode,
            bus: 0,
            frequency: freq,
            gain: 0.5,
            pan: 0.0,
            detune_cents: 0.0,
            attack: 0.0,
            decay: 0.0,
            sustain: 1.0,
            release: 0.02,
            cutoff: AUDIO_OPEN as f32,
            resonance: std::f32::consts::FRAC_1_SQRT_2,
            grain_size: 0.05,
            grain_pitch: 1.0,
            reverb: 0.0,
        }
    }

    #[test]
    fn sine_frequency_and_zero_attack_retrigger_stay_click_free() {
        let rate = 48_000.0;
        let mut engine = SynthEngine::new();
        let spec = base_spec(1, SynthMode::Sine, 440.0);
        engine.apply(&[spec]);
        let mut samples = vec![[0.0f32; 2]; 4800];
        engine.render(rate, samples.len(), |_, i, l, r, _| {
            samples[i][0] += l;
            samples[i][1] += r;
        });
        let crossings = samples.windows(2).filter(|w| w[0][0] <= 0.0 && w[1][0] > 0.0).count();
        assert!((43..=45).contains(&crossings), "expected ~44 crossings for 440 Hz, got {crossings}");
        // Omit voice -> enters release, then retrigger mid-release with attack=0: must ramp smoothly without step jump.
        engine.apply(&[]);
        let mut release_part = vec![[0.0f32; 2]; 600];
        engine.render(rate, release_part.len(), |_, i, l, r, _| {
            release_part[i][0] = l;
            release_part[i][1] = r;
        });
        engine.apply(&[spec]);
        let mut retrigger_part = vec![[0.0f32; 2]; 600];
        engine.render(rate, retrigger_part.len(), |_, i, l, r, _| {
            retrigger_part[i][0] = l;
            retrigger_part[i][1] = r;
        });
        let boundary_jump = (retrigger_part[0][0] - release_part[599][0]).abs();
        assert!(boundary_jump < 0.08, "zero-attack retrigger must not click: jump={boundary_jump}");
        // Full release reaps voice.
        engine.apply(&[]);
        engine.render(rate, 2000, |_, _, _, _, _| {});
        assert_eq!(engine.active_count(), 0, "released synth voice must be reaped");
    }

    #[test]
    fn retiring_voice_churn_never_starves_fresh_active_voices() {
        let mut engine = SynthEngine::new();
        // Churn two full 64-voice batches with 5s releases, then start a 3rd batch.
        for batch in 0..3u32 {
            let specs: Vec<SynthSpec> = (1..=MAX_SYNTHS as u32)
                .map(|i| {
                    let mut s = base_spec(batch * 1000 + i, SynthMode::Sine, 440.0);
                    s.release = 5.0;
                    s
                })
                .collect();
            engine.apply(&specs);
            engine.render(48_000.0, 128, |_, _, _, _, _| {});
        }
        assert!(engine.active_count() <= MAX_SYNTHS * 2);
        // Every voice of batch 2 (ids 2001..=2064) must be present and active.
        assert!(engine.voices.iter().filter(|v| v.spec.id > 2000 && !v.is_releasing()).count() == MAX_SYNTHS);
    }

    #[test]
    fn zdf_lowpass_attenuates_saw_harmonics_and_stays_finite_at_extremes() {
        let rate = 48_000.0;
        let render_rms = |cutoff: f32, q: f32| {
            let mut eng = SynthEngine::new();
            let mut s = base_spec(2, SynthMode::Saw, 880.0);
            s.cutoff = cutoff;
            s.resonance = q;
            eng.apply(&[s]);
            let mut buf = vec![0.0f32; 4096];
            eng.render(rate, buf.len(), |_, i, l, _, _| buf[i] = l);
            assert!(buf.iter().all(|v| v.is_finite()), "filter must stay finite at cutoff={cutoff} q={q}");
            let tail = &buf[1024..];
            (tail.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / tail.len() as f64).sqrt()
        };
        let open = render_rms(20_000.0, 0.707);
        let muffled = render_rms(250.0, 0.707);
        assert!(open > muffled * 3.0, "250 Hz lowpass must strongly attenuate 880 Hz saw: open={open} muffled={muffled}");
        let extreme = render_rms(23_900.0, 20.0);
        assert!(extreme.is_finite() && extreme < 10.0, "extreme ZDF SVF must remain bounded: {extreme}");
    }

    #[test]
    fn all_modes_including_granular_produce_bounded_finite_audio_and_dry_loop() {
        let rate = 48_000.0;
        for mode in [
            SynthMode::Sine,
            SynthMode::Saw,
            SynthMode::Square,
            SynthMode::Triangle,
            SynthMode::Noise,
            SynthMode::Granular,
        ] {
            let mut eng = SynthEngine::new();
            let mut s = base_spec(10 + mode as u32, mode, 330.0);
            s.detune_cents = 25.0;
            s.grain_pitch = 1.5;
            eng.apply(&[s]);
            let mut peak = 0.0f32;
            let mut energy = 0.0f64;
            eng.render(rate, 4800, |_, _, l, r, _| {
                assert!(l.is_finite() && r.is_finite());
                peak = peak.max(l.abs()).max(r.abs());
                energy += f64::from(l) * f64::from(l) + f64::from(r) * f64::from(r);
            });
            assert!(energy > 1.0 && peak <= 3.0, "mode {mode:?}: energy={energy} peak={peak}");
        }
        // Granular loop must NOT bake cutoff or pan into the returned loop.
        let mut hard_panned_low_cut = base_spec(77, SynthMode::Granular, 2000.0);
        hard_panned_low_cut.pan = 1.0;
        hard_panned_low_cut.cutoff = 100.0;
        let [l, r] = render_granular_loop(hard_panned_low_cut, rate, 12_000);
        let rms_l = (l.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / l.len() as f64).sqrt();
        let rms_r = (r.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / r.len() as f64).sqrt();
        assert!(rms_l > 0.02 && rms_r > 0.02, "dry granular loop must remain unfiltered and centered: {rms_l} {rms_r}");
    }
}
