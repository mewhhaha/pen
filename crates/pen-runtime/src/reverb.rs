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
}
