//! Loudness, as EBU R128 measures it (ITU-R BS.1770-4), and the asset
//! pipeline's normalization of the game's sounds to it (`pen loudness`).
//!
//! A clip's integrated loudness, in LUFS: each channel K-weighted (a high
//! shelf, then a high-pass), its mean square over 400 ms blocks every 100
//! ms, the blocks below -70 LUFS left out, and then those 10 LU below the
//! rest; a clip shorter than a block is one block. Its true peak, in dBTP:
//! the highest sample once oversampled four times (twice at 96 kHz), as a
//! windowed-sinc interpolator finds it.
//!
//! Normalizing scales a clip to its target loudness, less if that would
//! take its true peak above the ceiling. A PCM WAV clip's samples are
//! scaled where they are; an Ogg Opus clip's are left as they are encoded,
//! and its header's output gain (RFC 7845) changes instead, which every
//! decoder applies: pen's, and the browsers'.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "samples, gains, block lengths and taps"
)]
use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

use crate::audio::source::Encoded;

/// EBU R128's target: -23 LUFS.
pub const TARGET: f64 = -23.0;
/// The highest a normalized clip's true peak may go, in dBTP.
pub const CEILING: f64 = -1.0;
/// Blocks below this are silence.
const ABSOLUTE_GATE: f64 = -70.0;
/// Blocks this far below the rest are left out.
const RELATIVE_GATE: f64 = -10.0;

/// A second-order filter, as BS.1770 gives K-weighting.
#[derive(Clone, Copy, Debug)]
struct Biquad {
    b: [f64; 3],
    a: [f64; 2],
}

impl Biquad {
    fn run(&self, input: &[f64]) -> Vec<f64> {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        input
            .iter()
            .map(|&x| {
                let y = self.b[0] * x + self.b[1] * x1 + self.b[2] * x2 - self.a[0] * y1 - self.a[1] * y2;
                (x2, x1, y2, y1) = (x1, x, y1, y);
                y
            })
            .collect()
    }
}

/// BS.1770's K-weighting at `rate`: the shelf and the high-pass, from their
/// analog prototypes (as libebur128 designs them), which at 48 kHz are the
/// standard's own coefficients.
fn k_weighting(rate: f64) -> [Biquad; 2] {
    let (f0, gain, q) = (1_681.974_450_955_533, 3.999_843_853_973_347, 0.707_175_236_955_419_6);
    let k = (PI * f0 / rate).tan();
    let vh = 10f64.powf(gain / 20.0);
    let vb = vh.powf(0.499_666_774_154_541_6);
    let a0 = 1.0 + k / q + k * k;
    let shelf = Biquad {
        b: [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0],
        a: [2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
    };
    let (f0, q) = (38.135_470_876_024_44, 0.500_327_037_323_877_3);
    let k = (PI * f0 / rate).tan();
    let a0 = 1.0 + k / q + k * k;
    let high_pass = Biquad { b: [1.0, -2.0, 1.0], a: [2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0] };
    [shelf, high_pass]
}

/// The integrated loudness, in LUFS, of `channels` (each a channel's
/// samples) at `rate`; none for silence.
pub fn integrated(channels: &[Vec<f64>], rate: f64) -> Option<f64> {
    let [shelf, high_pass] = k_weighting(rate);
    let weighted: Vec<Vec<f64>> = channels.iter().map(|channel| high_pass.run(&shelf.run(channel))).collect();
    let length = weighted.first().map_or(0, Vec::len);
    let window = ((0.4 * rate).round() as usize).max(1);
    let step = ((0.1 * rate).round() as usize).max(1);
    let starts: Vec<usize> =
        if length >= window { (0..=(length - window) / step).map(|block| block * step).collect() } else { vec![0] };
    let span = window.min(length).max(1);
    // Each block's mean square on each channel.
    let blocks: Vec<Vec<f64>> = starts
        .iter()
        .map(|&start| {
            weighted
                .iter()
                .map(|channel| {
                    channel[start..(start + span).min(length)].iter().map(|x| x * x).sum::<f64>() / span as f64
                })
                .collect()
        })
        .collect();
    let loudness = |squares: &[f64]| -0.691 + 10.0 * squares.iter().sum::<f64>().log10();
    let mean = |blocks: &[&Vec<f64>]| -> Vec<f64> {
        (0..channels.len()).map(|c| blocks.iter().map(|block| block[c]).sum::<f64>() / blocks.len() as f64).collect()
    };
    let audible: Vec<&Vec<f64>> = blocks.iter().filter(|block| loudness(block) > ABSOLUTE_GATE).collect();
    if audible.is_empty() {
        return None;
    }
    let gate = loudness(&mean(&audible)) + RELATIVE_GATE;
    let gated: Vec<&Vec<f64>> = audible.into_iter().filter(|block| loudness(block) > gate).collect();
    Some(loudness(&mean(&gated)))
}

/// Taps either side of an oversampled point.
const REACH: usize = 8;

/// The highest absolute sample of `channels` at `rate`, oversampled to at
/// least 192 kHz (four times at 48 kHz) by a Blackman-windowed sinc.
pub fn true_peak(channels: &[Vec<f64>], rate: f64) -> f64 {
    let factor = if rate < 96_000.0 {
        4
    } else if rate < 192_000.0 {
        2
    } else {
        1
    };
    // Each fraction's taps, from REACH - 1 samples before it to REACH after.
    let phases: Vec<Vec<f64>> = (1..factor)
        .map(|phase| {
            let fraction = phase as f64 / factor as f64;
            let taps: Vec<f64> = (0..2 * REACH)
                .map(|tap| {
                    let x = tap as f64 - (REACH - 1) as f64 - fraction;
                    let sinc = if x == 0.0 { 1.0 } else { (PI * x).sin() / (PI * x) };
                    let w = (x + REACH as f64) / (2.0 * REACH as f64);
                    sinc * (0.42 - 0.5 * (2.0 * PI * w).cos() + 0.08 * (4.0 * PI * w).cos())
                })
                .collect();
            let sum: f64 = taps.iter().sum();
            taps.into_iter().map(|tap| tap / sum).collect()
        })
        .collect();
    let mut peak = 0.0f64;
    for channel in channels {
        let at =
            |index: isize| usize::try_from(index).ok().and_then(|index| channel.get(index)).copied().unwrap_or(0.0);
        for (index, &sample) in channel.iter().enumerate() {
            peak = peak.max(sample.abs());
            let first = index as isize - (REACH as isize - 1);
            for taps in &phases {
                let value: f64 = taps.iter().enumerate().map(|(tap, weight)| at(first + tap as isize) * weight).sum();
                peak = peak.max(value.abs());
            }
        }
    }
    peak
}

fn decibels(linear: f64) -> f64 {
    20.0 * linear.log10()
}

/// A clip's loudness, true peak and length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measurement {
    /// None for silence.
    pub lufs: Option<f64>,
    pub true_peak_db: f64,
    pub seconds: f64,
}

/// A clip's file, measured.
pub fn measure(bytes: &[u8]) -> Result<Measurement> {
    if bytes.iter().copied().find(|b| !b.is_ascii_whitespace()) == Some(b'{') {
        let recipe = pen_runtime::sound_recipe::Recipe::parse(bytes)?;
        let channels = vec![recipe.render()?.into_iter().map(f64::from).collect::<Vec<_>>()];
        let rate = f64::from(recipe.rate);
        return Ok(Measurement {
            lufs: integrated(&channels, rate),
            true_peak_db: decibels(true_peak(&channels, rate)),
            seconds: channels[0].len() as f64 / rate,
        });
    }
    let encoded = Encoded::new(bytes.to_vec().into())?;
    let clip = encoded.decode()?;
    let channels: Vec<Vec<f64>> =
        (0..encoded.channels).map(|side| clip.frames.iter().map(|frame| f64::from(frame[side])).collect()).collect();
    Ok(Measurement {
        lufs: integrated(&channels, encoded.rate),
        true_peak_db: decibels(true_peak(&channels, encoded.rate)),
        seconds: encoded.frames as f64 / encoded.rate,
    })
}

/// The gain, in dB, that takes `measurement` to `target` LUFS with its
/// true peak at most `ceiling` dBTP, and whether the ceiling held it back.
pub fn gain(measurement: &Measurement, target: f64, ceiling: f64) -> Option<(f64, bool)> {
    let wanted = target - measurement.lufs?;
    let room = ceiling - measurement.true_peak_db;
    Some(if wanted > room { (room, true) } else { (wanted, false) })
}

fn u16_at(bytes: &[u8], at: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(bytes.get(at..at + 2).context("a short WAV header")?.try_into()?))
}

fn u32_at(bytes: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(bytes.get(at..at + 4).context("a short WAV header")?.try_into()?))
}

/// A PCM WAV file with its samples scaled by `gain` (linear) where they
/// are, every other chunk as it was: integers rounded and kept within
/// their range, floats as they come.
pub fn scale_wav(bytes: &[u8], gain: f64) -> Result<Vec<u8>> {
    ensure!(bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE"), "not a WAV file");
    let mut out = bytes.to_vec();
    let (mut at, mut format) = (12, None);
    while at + 8 <= bytes.len() {
        let size = usize::try_from(u32_at(bytes, at + 4)?)?;
        let body = at + 8;
        match &bytes[at..at + 4] {
            b"fmt " => {
                let mut tag = u16_at(bytes, body)?;
                // WAVE_FORMAT_EXTENSIBLE names its format in its subformat.
                if tag == 0xfffe {
                    tag = u16_at(bytes, body + 24)?;
                }
                format = Some((tag, u16_at(bytes, body + 14)?));
            }
            b"data" => {
                let (tag, bits) = format.context("a WAV data chunk before its format")?;
                let end = (body + size).min(bytes.len());
                scale_samples(&mut out[body..end], tag, bits, gain)?;
            }
            _ => {}
        }
        at = body + size + size % 2;
    }
    ensure!(format.is_some(), "a WAV file without a format");
    Ok(out)
}

fn scale_samples(data: &mut [u8], tag: u16, bits: u16, gain: f64) -> Result<()> {
    let scale_int = |value: i64, bits: u32| {
        let limit = (1i64 << (bits - 1)) as f64;
        (value as f64 * gain).round().clamp(-limit, limit - 1.0) as i64
    };
    match (tag, bits) {
        (1, 16) => {
            for sample in data.as_chunks_mut::<2>().0 {
                let value = scale_int(i64::from(i16::from_le_bytes(*sample)), 16);
                sample.copy_from_slice(&i16::try_from(value)?.to_le_bytes());
            }
        }
        (1, 24) => {
            for sample in data.as_chunks_mut::<3>().0 {
                let value = i64::from(i32::from_le_bytes([0, sample[0], sample[1], sample[2]]) >> 8);
                let scaled = i32::try_from(scale_int(value, 24))?;
                sample.copy_from_slice(&scaled.to_le_bytes()[..3]);
            }
        }
        (1, 32) => {
            for sample in data.as_chunks_mut::<4>().0 {
                let value = scale_int(i64::from(i32::from_le_bytes(*sample)), 32);
                sample.copy_from_slice(&i32::try_from(value)?.to_le_bytes());
            }
        }
        (3, 32) => {
            for sample in data.as_chunks_mut::<4>().0 {
                let value = f32::from_le_bytes(*sample);
                sample.copy_from_slice(&((f64::from(value) * gain) as f32).to_le_bytes());
            }
        }
        (1, 8) => {
            for sample in data.iter_mut() {
                let value = scale_int(i64::from(*sample) - 128, 8);
                *sample = u8::try_from(value + 128)?;
            }
        }
        _ => bail!("WAV samples of format {tag} at {bits} bits are not scaled"),
    }
    Ok(())
}

/// The Ogg page checksum: CRC-32 with polynomial 0x04c11db7, unreflected,
/// over the page with its checksum field zero.
fn ogg_crc(page: &[u8]) -> u32 {
    let mut crc = 0u32;
    for &byte in page {
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 { crc << 1 ^ 0x04c1_1db7 } else { crc << 1 };
        }
    }
    crc
}

/// An Ogg Opus file with `gain` dB added to its head's output gain, its
/// first page's checksum made again.
pub fn regain_opus(bytes: &[u8], gain: f64) -> Result<Vec<u8>> {
    ensure!(bytes.starts_with(b"OggS") && bytes.len() >= 27, "not an Ogg stream");
    let segments = usize::from(bytes[26]);
    let body = 27 + segments;
    let length: usize = bytes.get(27..body).context("a short Ogg page")?.iter().map(|&size| usize::from(size)).sum();
    ensure!(bytes.get(body..body + 8) == Some(b"OpusHead") && length >= 19, "the first Ogg page is not an Opus head");
    let mut out = bytes.to_vec();
    let at = body + 16;
    let current = f64::from(i16::from_le_bytes([out[at], out[at + 1]])) / 256.0;
    let wanted = ((current + gain) * 256.0).round();
    ensure!((f64::from(i16::MIN)..=f64::from(i16::MAX)).contains(&wanted), "an output gain of {} dB", wanted / 256.0);
    out[at..at + 2].copy_from_slice(&(wanted as i16).to_le_bytes());
    out[22..26].fill(0);
    let crc = ogg_crc(&out[..body + length]);
    out[22..26].copy_from_slice(&crc.to_le_bytes());
    Ok(out)
}

/// What `pen loudness` was asked: to normalize, the target and ceiling,
/// and targets by file name from a JSON file (`{"target": -23, "peak": -1,
/// "sounds": {"music.ogg": -20}}`).
pub struct Options {
    pub normalize: bool,
    pub target: Option<f64>,
    pub peak: Option<f64>,
    pub targets: Option<PathBuf>,
}

/// Each file's target and ceiling, as the options and the targets file say.
struct Targets {
    target: f64,
    peak: f64,
    by_name: HashMap<String, f64>,
}

impl Targets {
    fn new(options: &Options) -> Result<Targets> {
        let mut targets = Targets { target: TARGET, peak: CEILING, by_name: HashMap::new() };
        if let Some(path) = &options.targets {
            let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
            let value: serde_json::Value =
                serde_json::from_str(&text).with_context(|| format!("{} is not JSON", path.display()))?;
            let number = |key: &str| value.get(key).and_then(serde_json::Value::as_f64);
            targets.target = number("target").unwrap_or(targets.target);
            targets.peak = number("peak").unwrap_or(targets.peak);
            if let Some(sounds) = value.get("sounds").and_then(serde_json::Value::as_object) {
                for (name, target) in sounds {
                    let target = target.as_f64().with_context(|| format!("{name}'s target is not a number"))?;
                    targets.by_name.insert(name.clone(), target);
                }
            }
        }
        targets.target = options.target.unwrap_or(targets.target);
        targets.peak = options.peak.unwrap_or(targets.peak);
        Ok(targets)
    }

    fn of(&self, path: &Path) -> f64 {
        let name = path.file_name().and_then(|name| name.to_str()).unwrap_or_default();
        self.by_name.get(name).copied().unwrap_or(self.target)
    }
}

/// `pen loudness FILE...`: each clip's loudness, true peak and length, and
/// with `normalize` each brought to its target.
pub fn run(files: &[PathBuf], options: &Options) -> Result<()> {
    ensure!(!files.is_empty(), "pen loudness needs the clips to measure");
    let targets = Targets::new(options)?;
    for path in files {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let measured = measure(&bytes).with_context(|| format!("measuring {}", path.display()))?;
        let lufs = measured.lufs.map_or("silent".to_owned(), |lufs| format!("{lufs:.1} LUFS"));
        let mut line =
            format!("{}: {lufs}, {:.1} dBTP, {:.2} s", path.display(), measured.true_peak_db, measured.seconds);
        if options.normalize {
            let target = targets.of(path);
            match gain(&measured, target, targets.peak) {
                Some((db, _)) if db.abs() < 0.1 => line += &format!(", at {target:.1} LUFS already"),
                Some((db, limited)) => {
                    let linear = 10f64.powf(db / 20.0);
                    let normalized = if crate::opus::is_ogg_opus(&bytes) {
                        regain_opus(&bytes, db)?
                    } else if path.to_string_lossy().ends_with(".sound") {
                        pen_runtime::sound_recipe::Recipe::parse(&bytes)?.scaled(linear)?
                    } else {
                        scale_wav(&bytes, linear)?
                    };
                    let staged = path.with_extension("loudness.tmp");
                    std::fs::write(&staged, normalized).with_context(|| format!("writing {}", staged.display()))?;
                    std::fs::rename(&staged, path).with_context(|| format!("replacing {}", path.display()))?;
                    let held = if limited {
                        format!(", held back by the {:.1} dBTP ceiling", targets.peak)
                    } else {
                        String::new()
                    };
                    line += &format!(", {db:+.1} dB toward {target:.1} LUFS{held}");
                }
                None => line += ", left silent",
            }
        }
        println!("{line}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = 48_000.0;

    /// A sine at `frequency` and `dbfs`, `seconds` long.
    fn sine(frequency: f64, dbfs: f64, seconds: f64) -> Vec<f64> {
        let amplitude = 10f64.powf(dbfs / 20.0);
        (0..(seconds * RATE) as usize).map(|i| amplitude * (2.0 * PI * frequency * i as f64 / RATE).sin()).collect()
    }

    /// At 48 kHz the design gives BS.1770's own coefficients.
    #[test]
    fn k_weighting_is_the_standards() {
        let [shelf, high_pass] = k_weighting(RATE);
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(close(shelf.b[0], 1.535_124_859_586_97) && close(shelf.b[1], -2.691_696_189_406_38));
        assert!(close(shelf.b[2], 1.198_392_810_852_85) && close(shelf.a[0], -1.690_659_293_182_41));
        assert!(close(shelf.a[1], 0.732_480_774_215_85));
        assert!(close(high_pass.a[0], -1.990_047_454_833_98) && close(high_pass.a[1], 0.990_072_250_366_21));
    }

    /// EBU Tech 3341's first cases: a 1 kHz stereo sine at -23 and -33 dBFS
    /// is -23 and -33 LUFS; and gating: -36, -23 and -36 dBFS for 10, 60
    /// and 10 s are -23 LUFS, the quiet parts gated out, and so with 10 s
    /// at -72 dBFS either side, below the absolute gate.
    #[test]
    fn loudness_meets_tech_3341() {
        let stereo = |signal: Vec<f64>| vec![signal.clone(), signal];
        let measure = |signal: Vec<f64>| integrated(&stereo(signal), RATE).unwrap();
        assert!((measure(sine(1000.0, -23.0, 20.0)) - -23.0).abs() < 0.1);
        assert!((measure(sine(1000.0, -33.0, 20.0)) - -33.0).abs() < 0.1);
        let parts = |parts: &[(f64, f64)]| {
            parts.iter().flat_map(|&(dbfs, seconds)| sine(1000.0, dbfs, seconds)).collect::<Vec<f64>>()
        };
        assert!((measure(parts(&[(-36.0, 10.0), (-23.0, 60.0), (-36.0, 10.0)])) - -23.0).abs() < 0.1);
        let gated = parts(&[(-72.0, 10.0), (-36.0, 10.0), (-23.0, 60.0), (-36.0, 10.0), (-72.0, 10.0)]);
        assert!((measure(gated) - -23.0).abs() < 0.1);
        // Silence has none; a click shorter than a block is one block.
        assert_eq!(integrated(&[vec![0.0; 48_000]], RATE), None);
        assert!(integrated(&[sine(1000.0, -20.0, 0.1)], RATE).is_some());
    }

    /// A sine at a quarter of the rate, sampled 45° off its peaks, peaks 3
    /// dB above its samples; the true peak finds it. (It fades in and out:
    /// a sine cut off at full swing overshoots between its first samples.)
    #[test]
    fn true_peaks_are_between_samples() {
        let fade = |i: usize| (i.min(48_000 - 1 - i) as f64 / 1000.0).min(1.0);
        let signal: Vec<f64> = (0..48_000).map(|i| 0.5 * fade(i) * (PI * i as f64 / 2.0 + PI / 4.0).sin()).collect();
        let samples = signal.iter().fold(0.0f64, |peak, sample| peak.max(sample.abs()));
        assert!((decibels(samples) - decibels(0.5 * std::f64::consts::FRAC_1_SQRT_2)).abs() < 1e-6);
        let peak = true_peak(&[signal], RATE);
        assert!((decibels(peak) - decibels(0.5)).abs() < 0.1, "{}", decibels(peak));
    }

    #[test]
    fn gains_reach_the_target_under_the_ceiling() {
        let quiet = Measurement { lufs: Some(-30.0), true_peak_db: -12.0, seconds: 1.0 };
        assert_eq!(gain(&quiet, -23.0, -1.0), Some((7.0, false)));
        let peaky = Measurement { lufs: Some(-30.0), true_peak_db: -3.0, seconds: 1.0 };
        assert_eq!(gain(&peaky, -23.0, -1.0), Some((2.0, true)));
        assert_eq!(gain(&Measurement { lufs: None, ..quiet }, -23.0, -1.0), None);
    }

    /// A 16-bit WAV of `samples`, with a chunk of its own before the data.
    fn wav(samples: &[i16]) -> Vec<u8> {
        let size = u32::try_from(samples.len() * 2).unwrap();
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(48 + size).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        for value in [16u32, 1 | 1 << 16, 48_000, 96_000, 2 | 16 << 16] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(b"JUNK\x04\0\0\0abcddata");
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend(samples.iter().flat_map(|sample| sample.to_le_bytes()));
        bytes
    }

    /// Scaling a WAV scales its samples, clipping none past their range,
    /// and leaves the rest of the file as it was; normalized, it measures
    /// its target.
    #[test]
    fn wavs_scale_where_they_are() {
        let samples: Vec<i16> =
            sine(1000.0, -30.0, 2.0).iter().map(|value| (value * 32_768.0).round() as i16).chain([30_000]).collect();
        let bytes = wav(&samples);
        let scaled = scale_wav(&bytes, 2.0).unwrap();
        assert_eq!(scaled.len(), bytes.len());
        assert_eq!(scaled[..56], bytes[..56]);
        let read = |bytes: &[u8], index: usize| i16::from_le_bytes([bytes[56 + 2 * index], bytes[57 + 2 * index]]);
        assert_eq!(read(&scaled, 10), samples[10] * 2);
        assert_eq!(read(&scaled, samples.len() - 1), i16::MAX);
        let clean = wav(&samples[..samples.len() - 1]);
        let before = measure(&clean).unwrap();
        let (db, limited) = gain(&before, -23.0, -1.0).unwrap();
        assert!(!limited);
        let after = measure(&scale_wav(&clean, 10f64.powf(db / 20.0)).unwrap()).unwrap();
        assert!((after.lufs.unwrap() - -23.0).abs() < 0.05, "{after:?}");
    }

    /// An Opus file's head gains the gain, its first page's checksum made
    /// as the encoder made it, and it decodes that much louder.
    #[test]
    fn opus_gains_in_its_head() {
        let tone = include_bytes!("audio/testdata/tone.opus");
        let first = 27
            + usize::from(tone[26])
            + tone[27..27 + usize::from(tone[26])].iter().map(|&size| usize::from(size)).sum::<usize>();
        let mut page = tone[..first].to_vec();
        page[22..26].fill(0);
        assert_eq!(ogg_crc(&page).to_le_bytes(), tone[22..26]);
        let louder = regain_opus(tone, 6.0).unwrap();
        assert_eq!((louder.len(), &louder[first..]), (tone.len(), &tone[first..]));
        let (before, after) = (measure(tone).unwrap(), measure(&louder).unwrap());
        let (before, after) = (before.lufs.unwrap(), after.lufs.unwrap());
        assert!((after - before - 6.0).abs() < 0.05, "{before} {after}");
    }
}
