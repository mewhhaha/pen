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

/// Input samples each output sample reads.
pub const TAPS: usize = 32;
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

/// Chrome's resampler for one ratio: its kernels, and each output sample
/// worked out from the input samples around it.
pub struct Resampler {
    ratio: f64,
    kernels: Vec<[f32; TAPS]>,
}

impl Resampler {
    /// From `from` samples a second to `to`.
    pub fn new(from: f64, to: f64) -> Resampler {
        let ratio = from / to;
        Resampler { ratio, kernels: kernels(ratio) }
    }

    /// How long `input` samples last from `from` Hz to `to` Hz without building
    /// sinc kernels: whole samples, truncated.
    pub fn output_length(input: usize, from: f64, to: f64) -> usize {
        (input as f64 / (from / to)) as usize
    }

    /// How long `input` samples last at the new rate: whole samples,
    /// truncated.
    pub fn length(&self, input: usize) -> usize {
        (input as f64 / self.ratio) as usize
    }

    /// The input samples output `j` reads: TAPS of them from this one.
    pub fn first(&self, j: usize) -> i64 {
        (j as f64 * self.ratio).floor() as i64 - (TAPS / 2) as i64
    }

    /// Output stereo frame `j` from an interleaved window `input` representing
    /// frames `[start .. start + input.len())` of a clip with `total_frames`.
    pub fn sample_stereo_window(
        &self,
        j: usize,
        input: &[[f32; 2]],
        start: usize,
        total_frames: usize,
        mono: bool,
    ) -> [f32; 2] {
        let position = j as f64 * self.ratio;
        let base = position.floor();
        let tap_start = base as i64 - (TAPS / 2) as i64;
        let offset = (position - base) * OFFSETS as f64;
        let which = (offset as usize).min(OFFSETS - 1);
        let blend = offset - which as f64;
        let (first, second) = (&self.kernels[which], &self.kernels[which + 1]);
        let limit = total_frames.min(start + input.len());
        if tap_start >= start as i64 && (tap_start as usize) + TAPS <= limit {
            let rel = tap_start as usize - start;
            let window = &input[rel..rel + TAPS];
            if mono {
                let (mut sum1, mut sum2) = (0.0f32, 0.0f32);
                for i in 0..TAPS {
                    let x = window[i][0];
                    sum1 += x * first[i];
                    sum2 += x * second[i];
                }
                let s = ((1.0 - blend) * f64::from(sum1) + blend * f64::from(sum2)) as f32;
                return [s, s];
            }
            let (mut l1, mut l2, mut r1, mut r2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for i in 0..TAPS {
                let [xl, xr] = window[i];
                let f = first[i];
                let s = second[i];
                l1 += xl * f;
                l2 += xl * s;
                r1 += xr * f;
                r2 += xr * s;
            }
            let left = ((1.0 - blend) * f64::from(l1) + blend * f64::from(l2)) as f32;
            let right = ((1.0 - blend) * f64::from(r1) + blend * f64::from(r2)) as f32;
            return [left, right];
        }
        if mono {
            let (mut sum1, mut sum2) = (0.0f32, 0.0f32);
            for i in 0..TAPS {
                let idx = tap_start + i as i64;
                let x =
                    if idx >= start as i64 && (idx as usize) < limit { input[idx as usize - start][0] } else { 0.0 };
                sum1 += x * first[i];
                sum2 += x * second[i];
            }
            let s = ((1.0 - blend) * f64::from(sum1) + blend * f64::from(sum2)) as f32;
            return [s, s];
        }
        let (mut l1, mut l2, mut r1, mut r2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for i in 0..TAPS {
            let idx = tap_start + i as i64;
            let [xl, xr] =
                if idx >= start as i64 && (idx as usize) < limit { input[idx as usize - start] } else { [0.0, 0.0] };
            let f = first[i];
            let s = second[i];
            l1 += xl * f;
            l2 += xl * s;
            r1 += xr * f;
            r2 += xr * s;
        }
        let left = ((1.0 - blend) * f64::from(l1) + blend * f64::from(l2)) as f32;
        let right = ((1.0 - blend) * f64::from(r1) + blend * f64::from(r2)) as f32;
        [left, right]
    }
}

/// Resamples interleaved stereo frames in a single pass with one kernel table.
pub fn resample_stereo(input: &[[f32; 2]], mono: bool, from: f64, to: f64) -> Vec<[f32; 2]> {
    let resampler = Resampler::new(from, to);
    let len = resampler.length(input.len());
    (0..len).map(|j| resampler.sample_stereo_window(j, input, 0, input.len(), mono)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resample(input: &[f32], from: f64, to: f64) -> Vec<f32> {
        let frames: Vec<[f32; 2]> = input.iter().map(|&v| [v, v]).collect();
        resample_stereo(&frames, true, from, to).into_iter().map(|[l, _]| l).collect()
    }

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
