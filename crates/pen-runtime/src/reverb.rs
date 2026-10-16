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
    let length = (SECONDS * rate).round() as usize;
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
            let cutoff = 1200.0 + 7000.0 * (-t / 0.3).exp();
            low += (1.0 - (-TAU * cutoff / rate).exp()) * (noise - low);
            let since = t - PREDELAY;
            let envelope = if since < 0.0 {
                0.0
            } else {
                (SIXTY_DB * since / DECAY).exp() * (since / ONSET).min(1.0) * ((SECONDS - t) / TAPER).min(1.0)
            };
            *value = low * envelope;
            energy += *value * *value;
        }
    }
    let scale = (ENERGY / (energy / 2.0)).sqrt();
    channels.map(|channel| channel.into_iter().map(|value| (value * scale) as f32).collect())
}
