//! The shared reverb: the impulse response of `pen_runtime::reverb`, which
//! the browser host's ConvolverNode plays too, and a convolution with it
//! that, like the ConvolverNode, adds no latency.
//!
//! Why a convolution: WebAudio has no algorithmic reverb, but its
//! ConvolverNode convolves exactly, so the browser and this mixer agree to
//! float rounding with the same impulse response, which both generate from
//! the sample rate alone (deterministic noise, no file). The convolution is
//! uniformly partitioned in two stages: the response's first 2048 samples in
//! blocks of one render quantum (128), which gives each quantum its output at
//! once, and the rest in blocks of 2048, each computed as its input block
//! completes and played over the next, which is exactly when it is due.

use std::f64::consts::TAU;

use super::mixer::QUANTUM;

#[cfg(test)]
pub use pen_runtime::reverb::impulse_response;
pub use pen_runtime::reverb::{AcousticProfile, impulse_response_for, impulse_response_from_acoustics};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    fn new(re: f32, im: f32) -> Complex {
        Complex { re, im }
    }
    fn conj(self) -> Complex {
        Complex::new(self.re, -self.im)
    }
    fn add(self, other: Complex) -> Complex {
        Complex::new(self.re + other.re, self.im + other.im)
    }
    fn sub(self, other: Complex) -> Complex {
        Complex::new(self.re - other.re, self.im - other.im)
    }
    fn mul(self, other: Complex) -> Complex {
        Complex::new(self.re * other.re - self.im * other.im, self.re * other.im + self.im * other.re)
    }
    fn scale(self, by: f32) -> Complex {
        Complex::new(self.re * by, self.im * by)
    }
}

/// An iterative radix-2 FFT of one power-of-two size.
struct Fft {
    size: usize,
    twiddles: Vec<Complex>,
    reversed: Vec<usize>,
}

impl Fft {
    fn new(size: usize) -> Fft {
        assert!(size.is_power_of_two() && size >= 2);
        let bits = size.trailing_zeros();
        #[allow(clippy::cast_possible_truncation, reason = "f32 twiddles")]
        let twiddles = (0..size / 2)
            .map(|k| {
                let angle = -TAU * k as f64 / size as f64;
                Complex::new(angle.cos() as f32, angle.sin() as f32)
            })
            .collect();
        let reversed = (0..size).map(|index| index.reverse_bits() >> (usize::BITS - bits)).collect();
        Fft { size, twiddles, reversed }
    }

    fn transform(&self, data: &mut [Complex], inverse: bool) {
        debug_assert_eq!(data.len(), self.size);
        for index in 0..self.size {
            let other = self.reversed[index];
            if index < other {
                data.swap(index, other);
            }
        }
        let mut half = 1;
        while half < self.size {
            let step = self.size / (2 * half);
            for start in (0..self.size).step_by(2 * half) {
                for j in 0..half {
                    let twiddle = self.twiddles[j * step];
                    let twiddle = if inverse { twiddle.conj() } else { twiddle };
                    let a = data[start + j];
                    let b = data[start + j + half].mul(twiddle);
                    data[start + j] = a.add(b);
                    data[start + j + half] = a.sub(b);
                }
            }
            half *= 2;
        }
        if inverse {
            let scale = 1.0 / self.size as f32;
            for value in data {
                *value = value.scale(scale);
            }
        }
    }
}

/// One uniformly partitioned overlap-save convolution with a stretch of the
/// response, two channels at once: left in the real part, right in the
/// imaginary, split apart in the frequency domain where each meets its own
/// channel's response, and joined again for one inverse transform.
struct Stage {
    block: usize,
    partitions: usize,
    bins: usize,
    fft: Fft,
    /// Flat slab: `partitions * 2 * bins`, indexed as `(partition * 2 + channel) * bins + bin`.
    filters: Vec<Complex>,
    /// Flat slab ring: `partitions * 2 * bins`, newest at `newest`.
    history: Vec<Complex>,
    newest: usize,
    /// The previous input block, then the current one.
    input: Vec<[f32; 2]>,
    work: Vec<Complex>,
    sums: [Vec<Complex>; 2],
}

impl Stage {
    fn new(block: usize, response: &[Vec<f32>; 2], from: usize, to: usize) -> Stage {
        let size = 2 * block;
        let fft = Fft::new(size);
        let partitions = (to.saturating_sub(from)).div_ceil(block).max(1);
        let bins = block + 1;
        let mut work = vec![Complex::default(); size];
        let mut filters = vec![Complex::default(); partitions * 2 * bins];
        for partition in 0..partitions {
            work.fill(Complex::default());
            let start = from + partition * block;
            for (offset, value) in work.iter_mut().take(block).enumerate() {
                let at = start + offset;
                if at < to {
                    *value = Complex::new(response[0][at], response[1][at]);
                }
            }
            fft.transform(&mut work, false);
            let base = partition * 2 * bins;
            let (left, right) = filters[base..base + 2 * bins].split_at_mut(bins);
            split(&work, left, right);
        }
        Stage {
            block,
            partitions,
            bins,
            fft,
            filters,
            history: vec![Complex::default(); partitions * 2 * bins],
            newest: 0,
            input: vec![[0.0; 2]; size],
            work,
            sums: [vec![Complex::default(); bins], vec![Complex::default(); bins]],
        }
    }

    fn clear_sums(&mut self) {
        self.sums[0].fill(Complex::default());
        self.sums[1].fill(Complex::default());
    }

    /// Accumulates spectral products for `partitions_range` assuming the
    /// upcoming input block will land at `(self.newest + 1) % self.partitions`.
    fn accumulate_partitions(&mut self, start: usize, end: usize, next_newest: usize) {
        let (partitions, bins) = (self.partitions, self.bins);
        for partition in start..end {
            let slot = (next_newest + partitions - partition) % partitions;
            let h_base = slot * 2 * bins;
            let f_base = partition * 2 * bins;
            for channel in 0..2 {
                let off = channel * bins;
                let sum = &mut self.sums[channel];
                let input = &self.history[h_base + off..h_base + off + bins];
                let response = &self.filters[f_base + off..f_base + off + bins];
                for bin in 0..bins {
                    sum[bin] = sum[bin].add(input[bin].mul(response[bin]));
                }
            }
        }
    }

    /// Takes the next input block (in the second half of `input`), finishes
    /// convolution against all partitions, and writes into `output`.
    fn convolve(&mut self, output: &mut [[f32; 2]], preaccumulated_tail: bool) {
        let (block, size, bins, partitions) = (self.block, 2 * self.block, self.bins, self.partitions);
        for (value, &[left, right]) in self.work.iter_mut().zip(self.input.iter()) {
            *value = Complex::new(left, right);
        }
        self.fft.transform(&mut self.work, false);
        self.newest = (self.newest + 1) % partitions;
        let base = self.newest * 2 * bins;
        let (left, right) = self.history[base..base + 2 * bins].split_at_mut(bins);
        split(&self.work, left, right);
        if preaccumulated_tail {
            self.accumulate_partitions(0, 1, self.newest);
        } else {
            self.clear_sums();
            self.accumulate_partitions(0, partitions, self.newest);
        }
        // Both channels' real outputs as one complex signal: left + i right.
        for (bin, value) in self.work.iter_mut().enumerate() {
            let (left, right) = if bin < bins {
                (self.sums[0][bin], self.sums[1][bin])
            } else {
                (self.sums[0][size - bin].conj(), self.sums[1][size - bin].conj())
            };
            *value = Complex::new(left.re - right.im, left.im + right.re);
        }
        self.fft.transform(&mut self.work, true);
        for (out, value) in output.iter_mut().zip(&self.work[block..]) {
            *out = [value.re, value.im];
        }
        self.input.copy_within(block.., 0);
    }
}

/// The left and right spectra, bins 0 to `left.len() - 1`, of two real
/// signals transformed together as left + i right.
fn split(spectrum: &[Complex], left: &mut [Complex], right: &mut [Complex]) {
    let size = spectrum.len();
    for (bin, (left, right)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        let z = spectrum[bin];
        let mirror = spectrum[(size - bin) % size].conj();
        *left = z.add(mirror).scale(0.5);
        // (z - mirror) / 2i
        let difference = z.sub(mirror);
        *right = Complex::new(difference.im * 0.5, -difference.re * 0.5);
    }
}

/// The tail's blocks: 16 quanta each.
const TAIL_BLOCK: usize = 2048;
const TAIL_QUANTA: usize = TAIL_BLOCK / QUANTUM;

/// The reverb: stereo in, stereo out, a quantum at a time.
pub struct Convolver {
    head: Stage,
    tail: Stage,
    /// The tail's output for the block now playing, and how much has played.
    tail_output: Vec<[f32; 2]>,
    played: usize,
    head_output: Vec<[f32; 2]>,
    /// Quanta of silence needed after the last non-zero input until all head
    /// and tail rings and overlap buffers have flushed to exact zero.
    ir_quanta: usize,
    active_quanta: usize,
}

impl Convolver {
    pub fn new(rate: f64) -> Convolver {
        Convolver::with_room(rate, 0.0, 0.0)
    }

    pub fn with_room(rate: f64, decay: f64, damping: f64) -> Convolver {
        Self::from_response(impulse_response_for(rate, decay, damping))
    }

    pub fn with_acoustics(rate: f64, profile: &AcousticProfile) -> Convolver {
        Self::from_response(impulse_response_from_acoustics(rate, profile))
    }

    fn from_response(response: [Vec<f32>; 2]) -> Convolver {
        let length = response[0].len();
        let split_at = TAIL_BLOCK.min(length);
        let tail = Stage::new(TAIL_BLOCK, &response, split_at, length);
        let ir_quanta = (tail.partitions + 3) * TAIL_QUANTA;
        Convolver {
            head: Stage::new(QUANTUM, &response, 0, split_at),
            tail,
            tail_output: vec![[0.0; 2]; TAIL_BLOCK],
            played: 0,
            head_output: vec![[0.0; 2]; QUANTUM],
            ir_quanta,
            active_quanta: 0,
        }
    }

    /// Convolves one quantum of input, adding the output to `output`.
    pub fn process(&mut self, input: &[[f32; 2]; QUANTUM], output: &mut [[f32; 2]; QUANTUM]) {
        let silent = input.iter().all(|&[l, r]| l == 0.0 && r == 0.0);
        if !silent {
            self.active_quanta = self.ir_quanta;
        } else if self.active_quanta == 0 && self.played == 0 {
            return;
        } else {
            self.active_quanta = self.active_quanta.saturating_sub(1);
        }
        // The head: this quantum's own contribution, at once.
        self.head.input[QUANTUM..].copy_from_slice(input);
        self.head.convolve(&mut self.head_output, false);
        // The tail: what earlier blocks have for this quantum, worked out as
        // the block before this one completed.
        let start = self.played;
        let q = start / QUANTUM;
        for ((out, head), tail) in
            output.iter_mut().zip(&self.head_output).zip(&self.tail_output[start..start + QUANTUM])
        {
            out[0] += head[0] + tail[0];
            out[1] += head[1] + tail[1];
        }
        self.tail.input[TAIL_BLOCK + start..TAIL_BLOCK + start + QUANTUM].copy_from_slice(input);
        // Spread partitions `1..tail.partitions` (which only depend on already-
        // completed history blocks) across quanta `0..15` so quantum 15 only
        // runs the FFTs plus partition 0.
        if q == 0 {
            self.tail.clear_sums();
        }
        if q < TAIL_QUANTA - 1 {
            let rest = self.tail.partitions.saturating_sub(1);
            let p_start = 1 + (q * rest) / (TAIL_QUANTA - 1);
            let p_end = 1 + ((q + 1) * rest) / (TAIL_QUANTA - 1);
            if p_start < p_end {
                let next_newest = (self.tail.newest + 1) % self.tail.partitions;
                self.tail.accumulate_partitions(p_start, p_end, next_newest);
            }
        }
        self.played += QUANTUM;
        if self.played == TAIL_BLOCK {
            self.played = 0;
            self.tail.convolve(&mut self.tail_output, true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The straightforward convolution, in f64.
    fn direct(input: &[[f32; 2]], response: &[Vec<f32>; 2]) -> Vec<[f64; 2]> {
        let mut output = vec![[0.0f64; 2]; input.len()];
        for (n, out) in output.iter_mut().enumerate() {
            for channel in 0..2 {
                let mut sum = 0.0f64;
                for k in 0..=n.min(response[channel].len() - 1) {
                    sum += f64::from(input[n - k][channel]) * f64::from(response[channel][k]);
                }
                out[channel] = sum;
            }
        }
        output
    }

    #[test]
    fn fft_round_trips() {
        let fft = Fft::new(16);
        let original: Vec<Complex> = (0..16).map(|i| Complex::new(i as f32, (i * i % 7) as f32)).collect();
        let mut data = original.clone();
        fft.transform(&mut data, false);
        // Bin 0 is the sum.
        assert!((data[0].re - (0..16).sum::<i32>() as f32).abs() < 1e-3);
        fft.transform(&mut data, true);
        for (a, b) in data.iter().zip(&original) {
            assert!((a.re - b.re).abs() < 1e-4 && (a.im - b.im).abs() < 1e-4, "{a:?} {b:?}");
        }
    }

    /// The partitioned convolution is the convolution, with no latency, in
    /// both channels and across the head and the tail.
    #[test]
    fn convolution_matches_direct() {
        let rate = 8000.0;
        let response = impulse_response(rate);
        let mut convolver = Convolver::new(rate);
        let quanta = 120;
        let mut state = 1u32;
        let input: Vec<[f32; 2]> = (0..quanta * QUANTUM)
            .map(|i| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (state >> 8) as f32 / (1 << 24) as f32 - 0.5;
                // An impulse, then noise, then silence.
                if i == 0 {
                    [1.0, 0.5]
                } else if i < 4000 {
                    [noise, -noise * 0.5]
                } else {
                    [0.0, 0.0]
                }
            })
            .collect();
        let mut output = Vec::new();
        for chunk in input.chunks(QUANTUM) {
            let mut out = [[0.0f32; 2]; QUANTUM];
            convolver.process(chunk.try_into().unwrap(), &mut out);
            output.extend_from_slice(&out);
        }
        let expected = direct(&input, &response);
        let peak = expected.iter().flat_map(|pair| pair.iter()).fold(0.0f64, |m, v| m.max(v.abs()));
        let worst = output
            .iter()
            .zip(&expected)
            .flat_map(|(a, b)| (0..2).map(move |c| (f64::from(a[c]) - b[c]).abs()))
            .fold(0.0f64, f64::max);
        assert!(worst < peak * 1e-4, "worst error {worst} against a peak of {peak}");
        // The impulse's response, from sample 0.
        assert!((f64::from(output[100][0]) - expected[100][0]).abs() < 1e-5);
    }

    /// The response is pinned: the browser host's ConvolverNode plays the
    /// same one, so it must not drift.
    #[test]
    fn impulse_response_is_pinned() {
        let response = impulse_response(48_000.0);
        assert_eq!(response[0].len(), 76_800);
        let energy: f64 = response.iter().flatten().map(|v| f64::from(*v) * f64::from(*v)).sum();
        assert!((energy - 2.0).abs() < 1e-4, "{energy}");
        // Silent through the predelay, then the noise.
        assert_eq!(response[0][575], 0.0);
        for (channel, index, expected) in [
            (0, 600, 4.367_595_5e-4),
            (1, 600, 6.779_858_4e-4),
            (0, 24_000, -8.966_267_3e-4),
            (1, 70_000, 5.950_865_7e-6),
        ] {
            let value = response[channel][index];
            assert!((value - expected).abs() <= expected.abs() * 1e-5, "{channel} {index}: {value:e}");
        }
    }
}
