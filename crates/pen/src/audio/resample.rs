//! Clips at the mixer's rate, resampled as WebAudio's decodeAudioData
//! resamples them in Chrome (Blink's SincResampler): a 32-tap
//! Blackman-windowed sinc, cut off at 0.9 of the lower Nyquist frequency,
//! its kernel tabulated at 32 subsample offsets and interpolated linearly
//! between them. The browser plays every clip at its context's rate this way,
//! so the mixer does too, rather than interpolating linearly across rates.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "sample positions and taps"
)]

use std::f64::consts::PI;

const TAPS: usize = 32;
const OFFSETS: usize = 32;

/// The kernel at each of OFFSETS + 1 subsample offsets.
fn kernels(ratio: f64) -> Vec<[f32; TAPS]> {
    let (a0, a1, a2) = (0.42, 0.5, 0.08);
    let scale = if ratio > 1.0 { 1.0 / ratio } else { 1.0 } * 0.9;
    let half = (TAPS / 2) as f64;
    (0..=OFFSETS)
        .map(|offset| {
            let subsample = offset as f64 / OFFSETS as f64;
            let mut kernel = [0.0f32; TAPS];
            for (i, value) in kernel.iter_mut().enumerate() {
                let s = scale * PI * (i as f64 - half - subsample);
                let sinc = if s == 0.0 { 1.0 } else { s.sin() / s } * scale;
                let x = (i as f64 - subsample) / TAPS as f64;
                let window = a0 - a1 * (2.0 * PI * x).cos() + a2 * (4.0 * PI * x).cos();
                *value = (sinc * window) as f32;
            }
            kernel
        })
        .collect()
}

/// `input` at `from` samples a second, at `to`: as long as it lasts there,
/// whole samples (`len * to / from`, truncated).
pub fn resample(input: &[f32], from: f64, to: f64) -> Vec<f32> {
    let ratio = from / to;
    let length = (input.len() as f64 / ratio) as usize;
    let kernels = kernels(ratio);
    let half = (TAPS / 2) as i64;
    let at = |index: i64| if index >= 0 && (index as usize) < input.len() { input[index as usize] } else { 0.0 };
    (0..length)
        .map(|j| {
            let position = j as f64 * ratio;
            let base = position.floor();
            let offset = (position - base) * OFFSETS as f64;
            let which = (offset as usize).min(OFFSETS - 1);
            let blend = offset - which as f64;
            let (first, second) = (&kernels[which], &kernels[which + 1]);
            let start = base as i64 - half;
            let (mut sum1, mut sum2) = (0.0f32, 0.0f32);
            for i in 0..TAPS {
                let x = at(start + i as i64);
                sum1 += x * first[i];
                sum2 += x * second[i];
            }
            ((1.0 - blend) * f64::from(sum1) + blend * f64::from(sum2)) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tone keeps its pitch and level; the length scales with the rate.
    #[test]
    fn a_tone_resamples_cleanly() {
        let (from, to) = (22_050.0, 48_000.0);
        let tone: Vec<f32> =
            (0..22_050).map(|i| (std::f64::consts::TAU * 1000.0 * i as f64 / from).sin() as f32).collect();
        let out = resample(&tone, from, to);
        assert_eq!(out.len(), 48_000);
        let worst = (1000..47_000)
            .map(|j| (out[j] - (std::f64::consts::TAU * 1000.0 * j as f64 / to).sin() as f32).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 2e-3, "{worst}");
        // Down too, its length truncated as Chrome's is (48000 / (48000 / 44100)
        // is a hair under 44100).
        assert_eq!(resample(&out, to, 44_100.0).len(), 44_099);
    }
}
