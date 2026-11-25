#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "audio sample math")]
use std::f64::consts::TAU;

/// The response's length, in seconds.
pub const SECONDS: f64 = 1.6;
/// How long its tail takes to fall 60 dB.
const DECAY: f64 = 1.5;
/// Silence before it starts, and how fast it comes in.
const PREDELAY: f64 = 0.012;
const ONSET: f64 = 0.004;
/// The last of it fades to nothing over this many seconds.
const TAPER: f64 = 0.08;
/// Each channel's response has this much energy (sum of squares).
const ENERGY: f64 = 1.0;
/// ln(0.001): a decay of 60 dB.
const SIXTY_DB: f64 = -6.907_755_278_982_137;
/// The two channels' noise seeds.
const SEEDS: [u32; 2] = [0x2545_f491, 0x9e37_79b9];

/// The impulse response at `rate` samples a second, left and right: noise
/// whose highs die away faster than its lows (a one-pole lowpass closing from
/// 8.2 kHz to 1.2 kHz), decaying 60 dB in `DECAY` seconds after a short
/// predelay, normalized so each channel's energy is `ENERGY`. Generated in
/// f64 with deterministic samples shared by the native and browser hosts.
pub fn impulse_response(rate: f64) -> [Vec<f32>; 2] {
    impulse_response_for(rate, 0.0, 0.0)
}

/// Parameterized room impulse response at `rate` Hz:
/// - `decay`: RT60 decay seconds (0 or non-finite uses the default 1.5 s room; clamped to 0.15..6.0 s).
/// - `damping`: high-frequency damping in 0..1 (0 or non-finite uses the default bright/dark decay curve;
///   higher values lower the tail's lowpass cutoff and shorten HF decay).
pub fn impulse_response_for(rate: f64, decay: f64, damping: f64) -> [Vec<f32>; 2] {
    let custom_decay = decay.is_finite() && decay > 0.0;
    let custom_damping = damping.is_finite() && damping > 0.0;
    let decay = if custom_decay { decay.clamp(0.15, 6.0) } else { DECAY };
    let seconds = if custom_decay { (decay + 0.1).clamp(0.25, 6.2) } else { SECONDS };
    let damp = if custom_damping { damping.clamp(0.0, 1.0) } else { -1.0 };
    let length = (seconds * rate).round().max(1.0) as usize;
    let mut channels = [vec![0.0f64; length], vec![0.0f64; length]];
    let mut energy = 0.0f64;
    for (channel, seed) in channels.iter_mut().zip(SEEDS) {
        let mut state = seed;
        let mut low = 0.0f64;
        for (index, value) in channel.iter_mut().enumerate() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = f64::from(state) / 4_294_967_296.0 * 2.0 - 1.0;
            let t = index as f64 / rate;
            let cutoff = if damp < 0.0 {
                1200.0 + 7000.0 * (-t / 0.3).exp()
            } else {
                // High damping closes down faster toward 400 Hz; low damping stays bright up to 11 kHz.
                let floor_hz = 2400.0 - 2000.0 * damp;
                let span_hz = 9000.0 * (1.0 - 0.7 * damp);
                let tau_hf = 0.55 - 0.42 * damp;
                floor_hz + span_hz * (-t / tau_hf).exp()
            };
            low += (1.0 - (-TAU * cutoff / rate).exp()) * (noise - low);
            let since = t - PREDELAY;
            let envelope = if since < 0.0 {
                0.0
            } else {
                (SIXTY_DB * since / decay).exp() * (since / ONSET).min(1.0) * ((seconds - t) / TAPER).min(1.0)
            };
            *value = low * envelope;
            energy += *value * *value;
        }
    }
    let scale = if energy > 0.0 { (ENERGY / (energy / 2.0)).sqrt() } else { 1.0 };
    channels.map(|channel| channel.into_iter().map(|value| (value * scale) as f32).collect())
}

/// Extension tag (`render.Extension.tag`) carrying ray-traced acoustic room parameters and
/// up to [`MAX_ACOUSTIC_TAPS`] stereo 3-band early reflection taps into native and WebAudio hosts.
///
/// Layout:
/// - `[0]` decay RT60 seconds (`0.15..6.0`, `0` = default)
/// - `[1]` high-frequency damping (`0..1`)
/// - `[2]` late tail predelay seconds (`0.002..0.15`)
/// - `[3]` early reflections linear gain (`0..2`)
/// - `[4]` late diffuse tail linear gain (`0..2`)
/// - `[5]` outdoor sky-escape factor (`0..1`: `0` enclosed room, `1` open outdoor field)
/// - `[6]` early reflection tap count (`0..=16`)
/// - `[7 + i * 5 ..]` per tap: `(delay_s, pan, gain_low, gain_mid, gain_high)`
pub const ACOUSTICS_EXT_TAG: u32 = 65_004;

/// Maximum ray-traced early reflection taps synthesized into the stereo convolution IR.
pub const MAX_ACOUSTIC_TAPS: usize = 16;

/// One ray-traced early acoustic reflection tap with 3-band surface coloration (`~250 Hz`, `~1 kHz`, `~4 kHz`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AcousticTap {
    /// Arrival delay in seconds (`0.0005..=0.25`).
    pub delay_s: f32,
    /// Stereo azimuth pan in `-1.0` (hard left) .. `+1.0` (hard right).
    pub pan: f32,
    /// Low-band linear reflection amplitude (`0.0..=2.0`, `< 450 Hz`).
    pub gain_low: f32,
    /// Mid-band linear reflection amplitude (`0.0..=2.0`, `450..2400 Hz`).
    pub gain_mid: f32,
    /// High-band linear reflection amplitude (`0.0..=2.0`, `> 2400 Hz`).
    pub gain_high: f32,
}

impl AcousticTap {
    pub fn sanitized(self) -> Self {
        let f = |v: f32, lo: f32, hi: f32, def: f32| if v.is_finite() { v.clamp(lo, hi) } else { def };
        Self {
            delay_s: f(self.delay_s, 0.0005, 0.25, 0.015),
            pan: f(self.pan, -1.0, 1.0, 0.0),
            gain_low: f(self.gain_low, 0.0, 2.0, 0.0),
            gain_mid: f(self.gain_mid, 0.0, 2.0, 0.0),
            gain_high: f(self.gain_high, 0.0, 2.0, 0.0),
        }
    }
}

/// Ray-traced acoustic room profile carrying geometric early reflections and Eyring/Sabine late reverb tail.
/// Fixed-capacity inline tap storage keeps `AcousticProfile: Copy` so real-time audio threads never allocate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcousticProfile {
    /// Late reverberation RT60 decay in seconds (`0.15..=6.0`, `0.0` = default 1.5 s).
    pub decay: f32,
    /// High-frequency damping factor (`0.0..=1.0`).
    pub damping: f32,
    /// Late diffuse tail onset predelay in seconds (`0.002..=0.15`).
    pub predelay_s: f32,
    /// Early reflections wet mix multiplier (`0.0..=2.0`).
    pub early_gain: f32,
    /// Late diffuse tail wet mix multiplier (`0.0..=2.0`).
    pub late_gain: f32,
    /// Open-sky escape fraction (`0.0` enclosed room .. `1.0` open outdoor field).
    pub outdoor: f32,
    /// Up to [`MAX_ACOUSTIC_TAPS`] discrete stereo early reflections (active prefix `..tap_count`).
    pub taps: [AcousticTap; MAX_ACOUSTIC_TAPS],
    /// Number of valid early reflection taps stored in `taps`.
    pub tap_count: u8,
}

impl AcousticProfile {
    /// Active early reflection taps.
    pub fn taps(&self) -> &[AcousticTap] {
        &self.taps[..usize::from(self.tap_count).min(MAX_ACOUSTIC_TAPS)]
    }

    /// Packs up to [`MAX_ACOUSTIC_TAPS`] taps into inline storage.
    pub fn pack_taps(slice: &[AcousticTap]) -> ([AcousticTap; MAX_ACOUSTIC_TAPS], u8) {
        let mut taps = [AcousticTap::default(); MAX_ACOUSTIC_TAPS];
        let count = slice.len().min(MAX_ACOUSTIC_TAPS);
        taps[..count].copy_from_slice(&slice[..count]);
        (taps, count as u8)
    }

    /// Parses and sanitizes the last [`ACOUSTICS_EXT_TAG`] payload from `frame.extensions`.
    pub fn from_extensions(extensions: &[pen_protocol::Extension]) -> Option<Self> {
        let ext = extensions.iter().rev().find(|e| e.tag == ACOUSTICS_EXT_TAG && e.values.len() >= 7)?;
        let vals = &ext.values;
        let f = |idx: usize, lo: f32, hi: f32, def: f32| {
            let v = vals.get(idx).copied().unwrap_or(def);
            if v.is_finite() { v.clamp(lo, hi) } else { def }
        };
        let raw_decay = vals[0];
        let decay = if raw_decay.is_finite() && raw_decay > 0.0 { raw_decay.clamp(0.15, 6.0) } else { 0.0 };
        let damping = f(1, 0.0, 1.0, 0.35);
        let predelay_s = f(2, 0.002, 0.15, PREDELAY as f32);
        let early_gain = f(3, 0.0, 2.0, 0.7);
        let late_gain = f(4, 0.0, 2.0, 1.0);
        let outdoor = f(5, 0.0, 1.0, 0.0);
        let declared =
            if vals[6].is_finite() && vals[6] > 0.0 { (vals[6].round() as usize).min(MAX_ACOUSTIC_TAPS) } else { 0 };
        let available = (vals.len().saturating_sub(7)) / 5;
        let count = declared.min(available);
        let mut taps = [AcousticTap::default(); MAX_ACOUSTIC_TAPS];
        for (i, slot) in taps.iter_mut().take(count).enumerate() {
            let base = 7 + i * 5;
            *slot = AcousticTap {
                delay_s: vals[base],
                pan: vals[base + 1],
                gain_low: vals[base + 2],
                gain_mid: vals[base + 3],
                gain_high: vals[base + 4],
            }
            .sanitized();
        }
        Some(Self { decay, damping, predelay_s, early_gain, late_gain, outdoor, taps, tap_count: count as u8 })
    }

    /// Hysteresis predicate deciding whether two ray-traced acoustic profiles differ enough to
    /// regenerate the stereo convolution IR and crossfade convolvers without per-frame thrashing.
    pub fn differs_significantly(&self, other: &Self) -> bool {
        let self_taps = self.taps();
        let other_taps = other.taps();
        if (self.decay - other.decay).abs() > 0.08
            || (self.damping - other.damping).abs() > 0.05
            || (self.predelay_s - other.predelay_s).abs() > 0.006
            || (self.early_gain - other.early_gain).abs() > 0.08
            || (self.late_gain - other.late_gain).abs() > 0.08
            || (self.outdoor - other.outdoor).abs() > 0.08
            || self_taps.len().abs_diff(other_taps.len()) > 2
        {
            return true;
        }
        let tap_summary = |taps: &[AcousticTap]| -> (f32, f32, f32, f32, f32) {
            if taps.is_empty() {
                return (0.0, 0.0, 0.0, 0.0, 0.0);
            }
            let mut energy = 0.0f32;
            let mut w_delay = 0.0f32;
            let mut w_pan = 0.0f32;
            let mut w_abs_pan = 0.0f32;
            let mut sum_low = 0.0f32;
            let mut sum_high = 0.0f32;
            for t in taps {
                let g = (t.gain_low + t.gain_mid + t.gain_high) / 3.0;
                energy += g;
                w_delay += t.delay_s * g;
                w_pan += t.pan * g;
                w_abs_pan += t.pan.abs() * g;
                sum_low += t.gain_low;
                sum_high += t.gain_high;
            }
            let inv = if energy > 1e-4 { 1.0 / energy } else { 0.0 };
            let hf_ratio = sum_high / (sum_low + 1e-4);
            (energy, w_delay * inv, w_pan * inv, w_abs_pan * inv, hf_ratio)
        };
        let (e_a, d_a, p_a, w_a, hf_a) = tap_summary(self_taps);
        let (e_b, d_b, p_b, w_b, hf_b) = tap_summary(other_taps);
        (e_a - e_b).abs() > 0.12
            || (d_a - d_b).abs() > 0.006
            || (p_a - p_b).abs() > 0.18
            || (w_a - w_b).abs() > 0.16
            || (hf_a - hf_b).abs() > 0.18
    }
}

/// Synthesizes a deterministic stereo room impulse response at `rate` Hz directly from a ray-traced
/// [`AcousticProfile`], combining:
/// 1. **3-Band Early Reflections**: each [`AcousticTap`] is placed with fractional-sample delay,
///    Woodworth interaural time difference (`±0.65 ms` from `pan`), equal-power stereo panning,
///    and unity-reconstructing 3-band crossover coloration (`< 450 Hz`, `450..2400 Hz`, `> 2400 Hz`).
/// 2. **Physical Late Diffuse Tail**: Eyring/Sabine RT60 decay & HF damping starting at `predelay_s`,
///    attenuated continuously by open-sky `outdoor` escape factor.
///
/// Shared identically by Pen's native partitioned convolver and browser WebAudio `ConvolverNode`.
pub fn impulse_response_from_acoustics(rate: f64, profile: &AcousticProfile) -> [Vec<f32>; 2] {
    let custom_decay = profile.decay.is_finite() && profile.decay > 0.0;
    let decay = if custom_decay { f64::from(profile.decay.clamp(0.15, 6.0)) } else { DECAY };
    let damp = if profile.damping.is_finite() { f64::from(profile.damping.clamp(0.0, 1.0)) } else { 0.35 };
    let predelay =
        if profile.predelay_s.is_finite() { f64::from(profile.predelay_s.clamp(0.002, 0.15)) } else { PREDELAY };
    let early_gain = if profile.early_gain.is_finite() { f64::from(profile.early_gain.clamp(0.0, 2.0)) } else { 0.7 };
    let late_gain = if profile.late_gain.is_finite() { f64::from(profile.late_gain.clamp(0.0, 2.0)) } else { 1.0 };
    let outdoor = if profile.outdoor.is_finite() { f64::from(profile.outdoor.clamp(0.0, 1.0)) } else { 0.0 };

    let active_taps = profile.taps();
    let max_tap_s = active_taps.iter().map(|t| f64::from(t.sanitized().delay_s)).fold(0.0f64, f64::max);
    let tail_seconds = (predelay + decay + 0.12).clamp(0.25, 6.2);
    let seconds = tail_seconds.max(max_tap_s + 0.08).clamp(0.25, 6.2);
    let length = (seconds * rate).round().max(1.0) as usize;

    // Stage 1: Late diffuse tail (normalized to unit channel energy before scaling by `late_scale`).
    // Respect `profile.decay` directly (ray tracers already bake open-sky absorption into Eyring RT60)
    // and apply a single clean open-sky late-level rolloff so outdoor spaces are never double-damped.
    let mut late = [vec![0.0f64; length], vec![0.0f64; length]];
    let mut late_energy = 0.0f64;
    let effective_decay = decay.max(0.12);
    for (channel, seed) in late.iter_mut().zip(SEEDS) {
        let mut state = seed;
        let mut low = 0.0f64;
        for (index, value) in channel.iter_mut().enumerate() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = f64::from(state) / 4_294_967_296.0 * 2.0 - 1.0;
            let t = index as f64 / rate;
            let floor_hz = 2400.0 - 2000.0 * damp;
            let span_hz = 9000.0 * (1.0 - 0.7 * damp);
            let tau_hf = 0.55 - 0.42 * damp;
            let cutoff = floor_hz + span_hz * (-t / tau_hf).exp();
            low += (1.0 - (-TAU * cutoff / rate).exp()) * (noise - low);
            let since = t - predelay;
            let envelope = if since < 0.0 {
                0.0
            } else {
                (SIXTY_DB * since / effective_decay).exp() * (since / 0.008).min(1.0) * ((seconds - t) / TAPER).min(1.0)
            };
            *value = low * envelope;
            late_energy += *value * *value;
        }
    }
    let late_norm = if late_energy > 0.0 { (ENERGY / (late_energy / 2.0)).sqrt() } else { 1.0 };
    let late_scale = late_norm * late_gain * (1.0 - 0.78 * outdoor);

    // Stage 2: Ray-traced 3-band early reflections with Woodworth ITD (±0.65 ms) & equal-power panning.
    // Unity-reconstructing 3-band split: LP(450 Hz) + [LP(2400 Hz) - LP(450 Hz)] + [x - LP(2400 Hz)] == x!
    let mut er_lo = [vec![0.0f64; length], vec![0.0f64; length]];
    let mut er_md = [vec![0.0f64; length], vec![0.0f64; length]];
    let mut er_hi = [vec![0.0f64; length], vec![0.0f64; length]];
    if early_gain > 0.0 && !active_taps.is_empty() {
        let add_frac = |buf: &mut [f64], pos: f64, amp: f64| {
            if amp == 0.0 || !pos.is_finite() || pos < 0.0 {
                return;
            }
            let idx = pos.floor() as usize;
            let frac = pos - pos.floor();
            if idx < buf.len() {
                buf[idx] += amp * (1.0 - frac);
            }
            if idx + 1 < buf.len() {
                buf[idx + 1] += amp * frac;
            }
        };
        for raw_tap in active_taps {
            let tap = raw_tap.sanitized();
            let pan = f64::from(tap.pan);
            let angle = (pan + 1.0) * 0.5 * std::f64::consts::FRAC_PI_2;
            let pan_l = angle.cos();
            let pan_r = angle.sin();
            // Woodworth spherical head ITD: sound from +1 (right) hits right ear up to 0.65 ms earlier.
            let itd_s = 0.000_65 * pan;
            let delay_l = (f64::from(tap.delay_s) + itd_s.max(0.0)).max(0.0005) * rate;
            let delay_r = (f64::from(tap.delay_s) + (-itd_s).max(0.0)).max(0.0005) * rate;
            let gl = f64::from(tap.gain_low) * early_gain;
            let gm = f64::from(tap.gain_mid) * early_gain;
            let gh = f64::from(tap.gain_high) * early_gain;
            add_frac(&mut er_lo[0], delay_l, gl * pan_l);
            add_frac(&mut er_lo[1], delay_r, gl * pan_r);
            add_frac(&mut er_md[0], delay_l, gm * pan_l);
            add_frac(&mut er_md[1], delay_r, gm * pan_r);
            add_frac(&mut er_hi[0], delay_l, gh * pan_l);
            add_frac(&mut er_hi[1], delay_r, gh * pan_r);
        }
    }

    let a_lo = 1.0 - (-TAU * 450.0 / rate).exp();
    let a_hi = 1.0 - (-TAU * 2400.0 / rate).exp();
    let mut out = [vec![0.0f64; length], vec![0.0f64; length]];
    let mut er_energy = 0.0f64;
    for ch in 0..2 {
        let mut s_lo = 0.0f64;
        let mut s_md_lo = 0.0f64;
        let mut s_md_hi = 0.0f64;
        let mut s_hi_lp = 0.0f64;
        for i in 0..length {
            s_lo += a_lo * (er_lo[ch][i] - s_lo);
            s_md_lo += a_lo * (er_md[ch][i] - s_md_lo);
            s_md_hi += a_hi * (er_md[ch][i] - s_md_hi);
            s_hi_lp += a_hi * (er_hi[ch][i] - s_hi_lp);
            let er_sample = s_lo + (s_md_hi - s_md_lo) + (er_hi[ch][i] - s_hi_lp);
            out[ch][i] = er_sample;
            er_energy += er_sample * er_sample;
        }
    }
    // Cap early-reflection RMS energy cleanly so dense close walls never clip the bus.
    let max_er_energy = 2.0 * (early_gain.max(0.25)).powi(2);
    let er_cap = if er_energy > max_er_energy { (max_er_energy / er_energy).sqrt() } else { 1.0 };

    let mut result = [vec![0.0f32; length], vec![0.0f32; length]];
    for ch in 0..2 {
        for i in 0..length {
            result[ch][i] = (out[ch][i] * er_cap + late[ch][i] * late_scale) as f32;
        }
    }
    result
}

/// Blended acoustic reverb state for the listener's current world position across
/// any overlapping [`pen_protocol::ReverbZone`] volumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoneMix {
    pub decay: f32,
    pub damping: f32,
    pub wet_scale: f32,
    pub dominant_zone: u32,
}

/// Signed distance from `point` to an axis-aligned box `(center, half_extents)`.
/// Zero inside the box; positive Euclidean distance outside.
fn box_distance(point: [f32; 3], center: [f32; 3], half: [f32; 3]) -> f64 {
    let mut sq = 0.0f64;
    for i in 0..3 {
        let ext = f64::from(half[i]).max(0.0);
        let d = (f64::from(point[i]) - f64::from(center[i])).abs() - ext;
        if d > 0.0 {
            sq += d * d;
        }
    }
    sq.sqrt()
}

/// Smooth weight in `0..=1` for `point` relative to `zone` (1 inside box, smoothstep
/// falloff over `zone.blend` outside).
pub fn zone_weight(point: [f32; 3], zone: &pen_protocol::ReverbZone) -> f64 {
    let pos = [zone.position.x, zone.position.y, zone.position.z];
    let ext = [zone.half_extents.x, zone.half_extents.y, zone.half_extents.z];
    if !pos.iter().chain(&ext).all(|v| v.is_finite()) {
        return 0.0;
    }
    let dist = box_distance(point, pos, ext);
    if dist <= 0.0 {
        return 1.0;
    }
    let blend = if zone.blend.is_finite() { f64::from(zone.blend).max(0.0) } else { 0.0 };
    if blend <= 1e-6 || dist >= blend {
        return 0.0;
    }
    let t = 1.0 - (dist / blend).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Smoothly blends spatial [`pen_protocol::ReverbZone`] rooms around `listener_pos`,
/// falling back cleanly toward `(base_decay, base_damping)` outside all zones.
pub fn blend_zones(
    listener_pos: [f32; 3],
    base_decay: f32,
    base_damping: f32,
    zones: &[pen_protocol::ReverbZone],
) -> ZoneMix {
    let fallback_decay = if base_decay.is_finite() && base_decay > 0.0 { base_decay } else { 0.0 };
    let fallback_damp = if base_damping.is_finite() { base_damping.clamp(0.0, 1.0) } else { 0.0 };
    if zones.is_empty() {
        return ZoneMix { decay: fallback_decay, damping: fallback_damp, wet_scale: 1.0, dominant_zone: 0 };
    }
    let mut sum_w = 0.0f64;
    let mut max_w = 0.0f64;
    let mut dominant_zone = 0u32;
    let mut decay_acc = 0.0f64;
    let mut damp_acc = 0.0f64;
    let mut wet_acc = 0.0f64;
    for zone in zones {
        let w = zone_weight(listener_pos, zone);
        if w <= 0.0 {
            continue;
        }
        let z_decay = if zone.decay.is_finite() && zone.decay > 0.0 {
            f64::from(zone.decay.clamp(0.15, 6.0))
        } else {
            f64::from(if fallback_decay > 0.0 { fallback_decay } else { DECAY as f32 })
        };
        let z_damp =
            if zone.damping.is_finite() { f64::from(zone.damping.clamp(0.0, 1.0)) } else { f64::from(fallback_damp) };
        let z_wet = if zone.wet.is_finite() && zone.wet > 0.0 { f64::from(zone.wet.clamp(0.0, 2.0)) } else { 1.0 };
        sum_w += w;
        decay_acc += w * z_decay;
        damp_acc += w * z_damp;
        wet_acc += w * z_wet;
        if w > max_w {
            max_w = w;
            dominant_zone = zone.id;
        }
    }
    if sum_w <= 1e-6 {
        return ZoneMix { decay: fallback_decay, damping: fallback_damp, wet_scale: 1.0, dominant_zone: 0 };
    }
    // When partially in a blend margin (sum_w < 1), smoothly mix with the base room.
    let base_w = (1.0 - sum_w).max(0.0);
    let total_w = sum_w + base_w;
    let base_d = f64::from(if fallback_decay > 0.0 { fallback_decay } else { DECAY as f32 });
    let decay = ((decay_acc + base_w * base_d) / total_w) as f32;
    let damping = ((damp_acc + base_w * f64::from(fallback_damp)) / total_w) as f32;
    let wet_scale = ((wet_acc + base_w * 1.0) / total_w) as f32;
    ZoneMix { decay, damping, wet_scale, dominant_zone }
}

/// Computes smooth cross-room portal attenuation in `0..=1` and the **source's**
/// own reverb wet multiplier based on source and listener membership across `zones`.
/// Both vary continuously through zone blend margins without abrupt edge jumps.
pub fn source_zone_acoustics(
    source_pos: [f32; 3],
    explicit_zone: u32,
    listener_pos: [f32; 3],
    zones: &[pen_protocol::ReverbZone],
) -> (f32, f32) {
    if zones.is_empty() {
        return (0.0, 1.0);
    }
    let mut portal = 0.0f64;
    let mut wet_sum = 0.0f64;
    let mut wet_w = 0.0f64;
    for z in zones {
        let wl = zone_weight(listener_pos, z);
        let ws =
            if explicit_zone != 0 { if z.id == explicit_zone { 1.0 } else { 0.0 } } else { zone_weight(source_pos, z) };
        if ws > 0.0 {
            let z_wet = if z.wet.is_finite() && z.wet > 0.0 { f64::from(z.wet.clamp(0.0, 2.0)) } else { 1.0 };
            wet_sum += ws * z_wet;
            wet_w += ws;
        }
        if z.id != 0 && z.portal_attenuation.is_finite() {
            // Smooth separation: 0 when source & listener have equal membership in `z`,
            // smoothly ramping to 1 when one is inside (`w=1`) and the other is outside (`w=0`).
            let sep = (wl - ws).abs();
            portal = portal.max(sep * f64::from(z.portal_attenuation.clamp(0.0, 1.0)));
        }
    }
    let base_w = (1.0 - wet_w).max(0.0);
    let wet_scale = ((wet_sum + base_w) / (wet_w + base_w)) as f32;
    (portal.clamp(0.0, 1.0) as f32, wet_scale)
}

/// Same as [`source_zone_acoustics`] using precomputed listener weights `listener_weights[i] == zone_weight(listener_pos, &zones[i])`.
pub fn source_zone_acoustics_cached(
    source_pos: [f32; 3],
    explicit_zone: u32,
    listener_weights: &[f64],
    zones: &[pen_protocol::ReverbZone],
) -> (f32, f32) {
    if zones.is_empty() {
        return (0.0, 1.0);
    }
    let mut portal = 0.0f64;
    let mut wet_sum = 0.0f64;
    let mut wet_w = 0.0f64;
    for (z, &wl) in zones.iter().zip(listener_weights) {
        let ws =
            if explicit_zone != 0 { if z.id == explicit_zone { 1.0 } else { 0.0 } } else { zone_weight(source_pos, z) };
        if ws > 0.0 {
            let z_wet = if z.wet.is_finite() && z.wet > 0.0 { f64::from(z.wet.clamp(0.0, 2.0)) } else { 1.0 };
            wet_sum += ws * z_wet;
            wet_w += ws;
        }
        if z.id != 0 && z.portal_attenuation.is_finite() {
            let sep = (wl - ws).abs();
            portal = portal.max(sep * f64::from(z.portal_attenuation.clamp(0.0, 1.0)));
        }
    }
    let base_w = (1.0 - wet_w).max(0.0);
    let wet_scale = ((wet_sum + base_w) / (wet_w + base_w)) as f32;
    (portal.clamp(0.0, 1.0) as f32, wet_scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_decay_and_damping_shape_room_tail() {
        let rate = 24_000.0;
        let short = impulse_response_for(rate, 0.3, 0.2);
        let long = impulse_response_for(rate, 2.5, 0.2);
        assert!(long[0].len() > short[0].len() * 4, "longer decay must produce a longer IR");

        let bright = impulse_response_for(rate, 1.0, 0.05);
        let dark = impulse_response_for(rate, 1.0, 0.95);
        // First-difference RMS measures high-frequency content in the late tail.
        let hf = |ir: &[f32]| -> f64 {
            let tail = &ir[ir.len() / 4..ir.len() * 3 / 4];
            let sum: f64 = tail.windows(2).map(|w| f64::from(w[1] - w[0]).powi(2)).sum();
            (sum / tail.len() as f64).sqrt()
        };
        assert!(hf(&bright[0]) > hf(&dark[0]) * 1.5, "higher damping must attenuate tail highs");
    }

    #[test]
    fn reverb_zones_blend_smoothly_and_apply_continuous_portal_and_source_wet() {
        let v = |x, y, z| pen_protocol::Vector { x, y, z };
        let zones = vec![
            pen_protocol::ReverbZone {
                id: 1,
                position: v(0.0, 0.0, 0.0),
                half_extents: v(5.0, 3.0, 5.0),
                blend: 4.0,
                decay: 0.5,
                damping: 0.7,
                wet: 0.8,
                portal_attenuation: 0.45,
            },
            pen_protocol::ReverbZone {
                id: 2,
                position: v(14.0, 0.0, 0.0),
                half_extents: v(5.0, 3.0, 5.0),
                blend: 4.0,
                decay: 3.5,
                damping: 0.1,
                wet: 1.5,
                portal_attenuation: 0.6,
            },
        ];
        let in_small = blend_zones([0.0, 0.0, 0.0], 1.5, 0.3, &zones);
        assert_eq!(in_small.dominant_zone, 1);
        assert!((in_small.decay - 0.5).abs() < 1e-4);

        // Midpoint x=7 sits 2m outside BOTH boxes (inside both 4m blend margins) -> smooth 50/50 blend.
        let mid = blend_zones([7.0, 0.0, 0.0], 1.5, 0.3, &zones);
        assert!((mid.decay - 2.0).abs() < 0.05, "expected ~2.0s blend, got {}", mid.decay);
        assert!((mid.damping - 0.4).abs() < 0.05, "expected ~0.4 damping, got {}", mid.damping);

        // Same zone -> 0 portal occlusion & source's own wet (0.8).
        let (p_same, w_same) = source_zone_acoustics([1.0, 0.0, 0.0], 0, [0.0, 0.0, 0.0], &zones);
        assert_eq!(p_same, 0.0);
        assert!((w_same - 0.8).abs() < 1e-5);
        // Cross-zone -> portal occlusion 0.6 & source's own zone 2 wet (1.5), NOT listener's (0.8)!
        let (p_cross, w_cross) = source_zone_acoustics([14.0, 0.0, 0.0], 0, [0.0, 0.0, 0.0], &zones);
        assert!((p_cross - 0.6).abs() < 1e-5, "expected 0.6 portal occlusion, got {p_cross}");
        assert!((w_cross - 1.5).abs() < 1e-5, "wet must follow source zone, got {w_cross}");
        // Barely entering blend margin (dist = 3.99 of 4.0) must have tiny portal occlusion, no edge jump!
        let (p_edge, _) = source_zone_acoustics([14.0, 0.0, 0.0], 0, [8.99, 0.0, 0.0], &zones);
        let (p_out, _) = source_zone_acoustics([14.0, 0.0, 0.0], 0, [8.999, 0.0, 0.0], &zones);
        assert!((p_edge - p_out).abs() < 0.01, "portal attenuation must vary continuously at blend edge");
    }

    #[test]
    fn acoustic_extension_parses_synthesizes_itd_3band_early_reflections_and_outdoor_tail() {
        let ext = pen_protocol::Extension {
            tag: ACOUSTICS_EXT_TAG,
            values: vec![
                2.4,   // decay RT60
                0.25,  // damping
                0.040, // predelay_s
                0.9,   // early_gain
                1.0,   // late_gain
                0.0,   // outdoor (enclosed)
                2.0,   // 2 early taps
                0.008, -0.95, 0.8, 0.8, 0.8, // bright left-wall reflection at 8 ms
                0.016, 0.95, 0.8, 0.3, 0.05, // muffled carpeted right-wall reflection at 16 ms
            ],
        };
        let profile = AcousticProfile::from_extensions(&[ext]).expect("valid acoustics extension");
        assert_eq!(profile.taps().len(), 2);
        assert!((profile.decay - 2.4).abs() < 1e-5);

        let rate = 48_000.0;
        let ir = impulse_response_from_acoustics(rate, &profile);
        assert!(ir[0].iter().chain(&ir[1]).all(|v| v.is_finite()));

        // Left tap at 8 ms (before 40 ms late predelay): left ear must be much louder than right ear!
        let win_l = (0.006 * rate) as usize..(0.012 * rate) as usize;
        let e_left_l: f64 = ir[0][win_l.clone()].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let e_left_r: f64 = ir[1][win_l].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        assert!(e_left_l > e_left_r * 10.0, "left-wall reflection must pan strongly left: L={e_left_l}, R={e_left_r}");

        // Compare HF energy (first-difference RMS) of bright stone left tap vs muffled right tap!
        let (bright_taps, bright_count) = AcousticProfile::pack_taps(&[AcousticTap {
            delay_s: 0.010,
            pan: 0.0,
            gain_low: 0.7,
            gain_mid: 0.7,
            gain_high: 0.7,
        }]);
        let bright_prof = AcousticProfile {
            decay: 0.5,
            damping: 0.3,
            predelay_s: 0.08,
            early_gain: 1.0,
            late_gain: 0.0,
            outdoor: 1.0,
            taps: bright_taps,
            tap_count: bright_count,
        };
        let (dark_taps, dark_count) = AcousticProfile::pack_taps(&[AcousticTap {
            delay_s: 0.010,
            pan: 0.0,
            gain_low: 0.7,
            gain_mid: 0.25,
            gain_high: 0.02,
        }]);
        let dark_prof = AcousticProfile { taps: dark_taps, tap_count: dark_count, ..bright_prof };
        let ir_bright = impulse_response_from_acoustics(rate, &bright_prof);
        let ir_dark = impulse_response_from_acoustics(rate, &dark_prof);
        let hf_win = (0.008 * rate) as usize..(0.025 * rate) as usize;
        let hf = |s: &[f32]| -> f64 { s.windows(2).map(|w| f64::from(w[1] - w[0]).powi(2)).sum::<f64>().sqrt() };
        assert!(
            hf(&ir_bright[0][hf_win.clone()]) > hf(&ir_dark[0][hf_win]) * 3.0,
            "3-band early reflection filter must attenuate highs on absorbent surfaces"
        );

        // Outdoor profile must strongly suppress late diffuse tail compared to enclosed room!
        let mut outdoor_prof = profile;
        outdoor_prof.outdoor = 1.0;
        let ir_indoor = impulse_response_from_acoustics(rate, &profile);
        let ir_outdoor = impulse_response_from_acoustics(rate, &outdoor_prof);
        let tail_start = (0.060 * rate) as usize;
        let tail_end = ir_outdoor[0].len().min(ir_indoor[0].len());
        let tail_in: f64 = ir_indoor[0][tail_start..tail_end].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let tail_out: f64 = ir_outdoor[0][tail_start..tail_end].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        assert!(
            tail_in > tail_out * 15.0,
            "open outdoor sky must suppress late reverberant tail: in={tail_in}, out={tail_out}"
        );

        // Hysteresis ignores tiny jitter, triggers on real room/tap shifts.
        let mut jittered = profile;
        jittered.decay += 0.02;
        jittered.outdoor += 0.02;
        assert!(!profile.differs_significantly(&jittered));
        assert!(profile.differs_significantly(&outdoor_prof));
    }

    #[test]
    fn acoustic_hysteresis_detects_symmetric_width_spectral_shift_and_sanitizes_malformed_wire() {
        // Symmetric corridor (L/R taps at ±0.85) vs narrow center taps (±0.05): mean pan is 0 for both!
        let (wide_taps, wide_count) = AcousticProfile::pack_taps(&[
            AcousticTap { delay_s: 0.012, pan: -0.85, gain_low: 0.7, gain_mid: 0.7, gain_high: 0.7 },
            AcousticTap { delay_s: 0.012, pan: 0.85, gain_low: 0.7, gain_mid: 0.7, gain_high: 0.7 },
        ]);
        let wide_corridor = AcousticProfile {
            decay: 1.4,
            damping: 0.3,
            predelay_s: 0.02,
            early_gain: 0.8,
            late_gain: 0.7,
            outdoor: 0.0,
            taps: wide_taps,
            tap_count: wide_count,
        };
        let (narrow_taps, narrow_count) = AcousticProfile::pack_taps(&[
            AcousticTap { delay_s: 0.012, pan: -0.05, gain_low: 0.7, gain_mid: 0.7, gain_high: 0.7 },
            AcousticTap { delay_s: 0.012, pan: 0.05, gain_low: 0.7, gain_mid: 0.7, gain_high: 0.7 },
        ]);
        let narrow_hall = AcousticProfile { taps: narrow_taps, tap_count: narrow_count, ..wide_corridor };
        assert!(
            wide_corridor.differs_significantly(&narrow_hall),
            "hysteresis must detect stereo width change even when mean pan cancels to 0"
        );

        // Same mean energy (0.7) but bright stone vs muffled carpet HF/LF ratio.
        let (stone_taps, stone_count) = AcousticProfile::pack_taps(&[AcousticTap {
            delay_s: 0.012,
            pan: 0.0,
            gain_low: 0.7,
            gain_mid: 0.7,
            gain_high: 0.7,
        }]);
        let stone = AcousticProfile { taps: stone_taps, tap_count: stone_count, ..wide_corridor };
        let (carpet_taps, carpet_count) = AcousticProfile::pack_taps(&[AcousticTap {
            delay_s: 0.012,
            pan: 0.0,
            gain_low: 1.15,
            gain_mid: 0.8,
            gain_high: 0.15,
        }]);
        let carpet = AcousticProfile { taps: carpet_taps, tap_count: carpet_count, ..wide_corridor };
        assert!(
            stone.differs_significantly(&carpet),
            "hysteresis must detect HF/LF material coloration change at identical mean tap energy"
        );

        // Malformed wire payloads (short, NaN/Inf, absurd tap count, truncated tail) never panic.
        let short = pen_protocol::Extension { tag: ACOUSTICS_EXT_TAG, values: vec![1.0, 0.2] };
        assert!(AcousticProfile::from_extensions(&[short]).is_none());

        let hostile = pen_protocol::Extension {
            tag: ACOUSTICS_EXT_TAG,
            values: vec![
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                -5.0,
                99.0,
                f32::NAN,
                1e20, // huge declared tap count with only 1 complete tap + 2 trailing floats
                f32::NAN,
                f32::INFINITY,
                -1.0,
                50.0,
                f32::NAN,
                0.01,
                0.5,
            ],
        };
        let parsed = AcousticProfile::from_extensions(&[hostile]).expect("sanitized fallback");
        assert_eq!(parsed.taps().len(), 1);
        let ir = impulse_response_from_acoustics(48_000.0, &parsed);
        assert!(ir[0].iter().chain(&ir[1]).all(|v| v.is_finite()));
    }
}
