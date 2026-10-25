//! Plays what the game asks for: each frame's buses, the voices it keeps
//! playing and the sounds it plays once, mixed by `mixer.rs` (WebAudio's
//! graph, sample for sample) with the shared reverb of `reverb.rs`. The same
//! mixer renders video recordings (`record.rs`), so what is recorded is what
//! plays.
//!
//! Live, the mixer runs in the output device's callback (`device.rs`), and
//! each frame's mix reaches it over a channel, landing at the start of its
//! next 128-sample quantum. Clips decode on a background thread; a voice or
//! sound whose clip is not ready yet is left out until it is, as the browser
//! leaves it out while decoding. A clip longer than STREAM_SECONDS keeps
//! only its bytes and decodes as it plays (`stream.rs`).
//!
//! For the dev tools ([`Audio::report`], [`Audio::monitor`]) the device's
//! callback also measures how late the output plays, and the mixer meters
//! its buses and takes the developer's mute and solo, all through
//! [`Shared`].
mod device;
pub mod mixer;
mod resample;
mod reverb;
pub mod source;
mod stream;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender, TrySendError, channel, sync_channel};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use mixer::{BusSpec, Clip, Mix, Mixer, QUANTUM, SoundSpec, VoiceSpec};
use pen_protocol::boundary::MAX_BUSES;
use pen_protocol::{Catalog, RenderFrame};
use pen_runtime::mix::{Meter, Monitor};
use pen_runtime::panner::Placements;
use source::Encoded;

/// Loads a clip, in either format the game's sounds come in: Ogg Opus
/// (`opus.rs`) or PCM WAV. One longer than STREAM_SECONDS streams, as in
/// the browser (`pen_runtime::clip`); the rest decode whole.
pub fn load_clip(path: &Path) -> Result<Clip> {
    let bytes = std::fs::read(path)?;
    if path.to_string_lossy().ends_with(".sound") {
        let recipe = pen_runtime::sound_recipe::Recipe::parse(&bytes)?;
        let samples = recipe.render()?;
        return Ok(Clip::new(samples.into_iter().map(|v| [v, v]).collect(), f64::from(recipe.rate)));
    }
    let streams = pen_runtime::clip::streams(&bytes);
    let encoded = Encoded::new(bytes.into())?;
    if streams {
        return Ok(Clip::streamed(encoded));
    }
    encoded.decode()
}

/// A PCM WAV clip, decoded whole.
#[cfg(test)]
fn decode_wav(bytes: Vec<u8>) -> Result<Clip> {
    Encoded::new(bytes.into())?.decode()
}

/// A clip, decoded for a mixer at `rate`.
pub fn decode(path: &Path, rate: f64) -> Result<Arc<Clip>> {
    Ok(Arc::new(load_clip(path)?).at_rate(rate))
}

/// A frame's sound for the mixer, with the decoded clips it plays; voices
/// and sounds whose clip is not among them are left out.
pub fn mix(frame: &RenderFrame, catalog: &Catalog, clips: &HashMap<String, Arc<Clip>>) -> Mix {
    let clip = |index: u32| clips.get(catalog.sound(index)).cloned();
    let placements = Placements::of(frame);
    Mix {
        buses: frame
            .buses
            .iter()
            .map(|bus| BusSpec { gain: bus.gain, lowpass: bus.lowpass, reverb: bus.reverb })
            .collect(),
        voices: frame
            .voices
            .iter()
            .filter_map(|voice| {
                Some(VoiceSpec {
                    id: voice.id,
                    clip: clip(voice.clip)?,
                    bus: voice.bus as usize,
                    gain: voice.gain,
                    pan: voice.pan,
                    pitch: voice.pitch,
                    lowpass: voice.lowpass,
                    reverb: voice.reverb,
                    looping: voice.looping,
                    start: voice.start,
                    fade_in: voice.fade_in,
                    fade_out: voice.fade_out,
                    panner: placements.voice(voice.id),
                })
            })
            .collect(),
        sounds: frame
            .sounds
            .iter()
            .enumerate()
            .filter_map(|(index, sound)| {
                Some(SoundSpec {
                    clip: clip(sound.clip)?,
                    bus: sound.bus as usize,
                    gain: sound.gain,
                    pan: sound.pan,
                    pitch: sound.pitch,
                    lowpass: sound.lowpass,
                    reverb: sound.reverb,
                    panner: placements.sound(index),
                })
            })
            .collect(),
    }
}

/// The rate the mixer starts at; it starts again at the device's if that
/// differs (or changes), and clips are decoded again for it.
const LIKELY_RATE: f64 = 48_000.0;

/// The output's sample rate, as the device runs, for the decoder.
#[derive(Clone)]
struct Rate(Arc<AtomicU64>);

impl Rate {
    fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }
    fn set(&self, rate: f64) {
        self.0.store(rate.to_bits(), Ordering::Relaxed);
    }
}

/// How late the output plays: from a frame's mix reaching the mixer to its
/// first sample leaving the device, on average. The mix waits for the next
/// callback (half a callback on average) and the next quantum within it
/// (half a quantum), and the callback's samples play `ahead` after it is
/// called, as the device says. Averaged over callbacks, as devices jitter.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Latency {
    /// Seconds; none until a callback said when its samples play.
    pub seconds: Option<f64>,
}

impl Latency {
    /// A callback of `frames` samples at `rate`, whose first plays `ahead`
    /// seconds after it was called, if the device says.
    pub fn measure(&mut self, ahead: Option<f64>, frames: usize, rate: f64) {
        let Some(ahead) = ahead else { return };
        let latency = ahead + (frames as f64 + QUANTUM as f64) / 2.0 / rate;
        self.seconds = Some(self.seconds.map_or(latency, |seconds| seconds + (latency - seconds) * 0.05));
    }
}

/// What the audio thread and the game's share besides the mixes: the
/// developer's mute and solo, the buses' meters and how the output plays.
#[derive(Default)]
#[cfg_attr(not(feature = "devtools"), allow(dead_code, reason = "the dev tools read it"))]
pub struct Shared {
    /// `Monitor::bits`.
    monitor: AtomicU64,
    meters: Mutex<([Meter; MAX_BUSES], [bool; MAX_BUSES])>,
    /// The latency's seconds as f64 bits, 0 until measured.
    latency: AtomicU64,
    /// Samples per callback, as the device last asked for them.
    callback: AtomicU32,
    playing: AtomicUsize,
    streaming: AtomicUsize,
    underruns: AtomicU64,
}

/// The mixer as the output device plays it. Mixes arrive from `Audio::frame`
/// and go back once applied, so that nothing the frames allocated is freed
/// on the audio thread.
struct Live {
    mixer: Mixer,
    rate: Rate,
    mixes: Receiver<Mix>,
    applied: SyncSender<Mix>,
    scratch: [[f32; 2]; QUANTUM],
    shared: Arc<Shared>,
    /// Meters not yet handed over (the game's side held them).
    meters: [Meter; MAX_BUSES],
    latency: Latency,
}

impl Live {
    fn new(rate: Rate, mixes: Receiver<Mix>, applied: SyncSender<Mix>, shared: Arc<Shared>) -> Live {
        Live {
            mixer: Mixer::live(rate.get()),
            rate,
            mixes,
            applied,
            scratch: [[0.0; 2]; QUANTUM],
            shared,
            meters: [Meter::default(); MAX_BUSES],
            latency: Latency::default(),
        }
    }

    /// The device's callback was called for `frames` samples that play
    /// `ahead` seconds from now, if it says.
    fn played(&mut self, ahead: Option<f64>, frames: usize) {
        self.latency.measure(ahead, frames, self.mixer.rate());
        if let Some(seconds) = self.latency.seconds {
            self.shared.latency.store(seconds.to_bits(), Ordering::Relaxed);
        }
        self.shared.callback.store(u32::try_from(frames).unwrap_or(u32::MAX), Ordering::Relaxed);
    }

    /// A device starts playing it at `rate`. At another rate than the
    /// mixer's, the mixer starts afresh: the frames to come declare what
    /// plays, and the clips are decoded again for it.
    fn start(&mut self, rate: f64) {
        if (rate - self.mixer.rate()).abs() > 0.5 {
            self.mixer = Mixer::live(rate);
            self.rate.set(rate);
        }
    }

    /// The device's next samples, `channels` interleaved: the mix within
    /// -1 and 1, both sides averaged on a single channel, and silence on
    /// any past the first two.
    fn fill(&mut self, out: &mut [f32], channels: usize) {
        let channels = channels.max(1);
        self.mixer.set_monitor(Monitor::from_bits(self.shared.monitor.load(Ordering::Relaxed)));
        while let Ok(mix) = self.mixes.try_recv() {
            self.mixer.apply(&mix);
            let _ = self.applied.try_send(mix);
        }
        for chunk in out.chunks_mut(QUANTUM * channels) {
            let scratch = &mut self.scratch[..chunk.len() / channels];
            self.mixer.render(scratch);
            for (frame, &[left, right]) in chunk.chunks_exact_mut(channels).zip(scratch.iter()) {
                let (left, right) = (left.clamp(-1.0, 1.0), right.clamp(-1.0, 1.0));
                match frame {
                    [only] => *only = (left + right) / 2.0,
                    [first, second, rest @ ..] => {
                        *first = left;
                        *second = right;
                        rest.fill(0.0);
                    }
                    [] => {}
                }
            }
        }
        let (meters, used) = self.mixer.take_meters();
        for (kept, meter) in self.meters.iter_mut().zip(&meters) {
            kept.merge(meter);
        }
        if let Ok(mut shared) = self.shared.meters.try_lock() {
            for (shared, kept) in shared.0.iter_mut().zip(&mut self.meters) {
                shared.merge(kept);
                *kept = Meter::default();
            }
            shared.1 = used;
        }
        self.shared.playing.store(self.mixer.playing(), Ordering::Relaxed);
        self.shared.streaming.store(self.mixer.streaming(), Ordering::Relaxed);
        self.shared.underruns.store(self.mixer.underruns(), Ordering::Relaxed);
    }
}

pub struct Audio {
    /// The output device, playing the mixer while it lives.
    _device: device::Device,
    clips: HashMap<String, Arc<Clip>>,
    /// The rate the clips are decoded for, and the output's.
    decoded_at: f64,
    rate: Rate,
    requests: Sender<String>,
    decoded: Receiver<(String, Result<Arc<Clip>, String>)>,
    mixes: SyncSender<Mix>,
    applied: Receiver<Mix>,
    #[cfg_attr(not(feature = "devtools"), allow(dead_code, reason = "the dev tools read it"))]
    shared: Arc<Shared>,
    /// The device's buffer asked for, in samples (`--audio-buffer`).
    #[cfg_attr(not(feature = "devtools"), allow(dead_code, reason = "the dev tools read it"))]
    buffer: Option<u32>,
}

impl Audio {
    /// Starts the output device, with a buffer of `buffer` samples if asked
    /// for (within what it takes), and decodes every clip in the catalog,
    /// those in `first` first.
    pub fn new(assets: PathBuf, catalog: &Catalog, first: &[String], buffer: Option<u32>) -> Result<Audio> {
        let (mixes, receiving) = sync_channel(32);
        let (returning, applied) = sync_channel(64);
        let rate = Rate(Arc::new(AtomicU64::new(LIKELY_RATE.to_bits())));
        let shared = Arc::new(Shared::default());
        let live = Live::new(rate.clone(), receiving, returning, shared.clone());
        let device = device::Device::start(Arc::new(Mutex::new(live)), buffer)?;
        // The device's own, which the clips are decoded for from the start.
        let decoded_at = rate.get();
        let (requests, wanted) = channel::<String>();
        let (sender, decoded) = channel();
        let decoder = rate.clone();
        std::thread::Builder::new().name("pen-audio-decode".into()).spawn(move || {
            for key in wanted {
                let result =
                    decode(&assets.join(&key), decoder.get()).map_err(|error| format!("sound {key}: {error:#}"));
                if sender.send((key, result)).is_err() {
                    return;
                }
            }
        })?;
        let mut audio = Audio {
            _device: device,
            clips: HashMap::new(),
            decoded_at,
            rate,
            requests,
            decoded,
            mixes,
            applied,
            shared,
            buffer,
        };
        // What the scene plays first decodes first.
        for key in first.iter().filter(|key| catalog.sounds.contains(key)) {
            let _ = audio.requests.send(key.clone());
        }
        audio.preload(catalog, first);
        Ok(audio)
    }

    /// Decodes the catalog's clips that are not decoded yet, but those in
    /// `asked` (asked for already).
    pub fn preload(&mut self, catalog: &Catalog, asked: &[String]) {
        for key in &catalog.sounds {
            if !self.clips.contains_key(key) && !asked.contains(key) {
                let _ = self.requests.send(key.clone());
            }
        }
    }

    /// Clips that changed on disk, decoded afresh; each plays its old
    /// version until then, and voices playing it start again with the new
    /// one, crossfading.
    #[cfg(feature = "compiler")]
    pub fn reload(&mut self, keys: &[String]) {
        for key in keys {
            if self.clips.contains_key(key) {
                let _ = self.requests.send(key.clone());
            }
        }
    }

    fn receive(&mut self) {
        // On a device at another rate, every clip again at its rate.
        let rate = self.rate.get();
        if rate != self.decoded_at {
            self.decoded_at = rate;
            for key in self.clips.keys() {
                let _ = self.requests.send(key.clone());
            }
        }
        while let Ok((key, result)) = self.decoded.try_recv() {
            match result {
                Ok(clip) => {
                    self.clips.insert(key, clip);
                }
                Err(message) => eprintln!("{message}"),
            }
        }
        // Mixes the audio thread is done with, freed here.
        while self.applied.try_recv().is_ok() {}
    }

    /// The mix as it plays, for the dev tools; its meters start again.
    #[cfg(feature = "devtools")]
    pub fn report(&mut self) -> pen_runtime::mix::AudioReport {
        let monitor = Monitor::from_bits(self.shared.monitor.load(Ordering::Relaxed));
        let (meters, used) = {
            let mut shared = self.shared.meters.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            (std::mem::take(&mut shared.0), shared.1)
        };
        let latency = f64::from_bits(self.shared.latency.load(Ordering::Relaxed));
        let callback = self.shared.callback.load(Ordering::Relaxed);
        pen_runtime::mix::AudioReport {
            rate: self.rate.get(),
            buffer: self.buffer.or((callback > 0).then_some(callback)),
            voices: self.shared.playing.load(Ordering::Relaxed),
            streams: self.shared.streaming.load(Ordering::Relaxed),
            underruns: self.shared.underruns.load(Ordering::Relaxed),
            buses: pen_runtime::mix::AudioReport::buses(&meters, &used, monitor),
            latency_ms: (latency > 0.0).then_some(latency * 1000.0),
        }
    }

    /// The developer mutes (`mute`) or solos bus `bus`, or stops.
    #[cfg(feature = "devtools")]
    pub fn monitor(&mut self, bus: usize, mute: bool, on: bool) {
        let mut monitor = Monitor::from_bits(self.shared.monitor.load(Ordering::Relaxed));
        monitor.set(bus, mute, on);
        self.shared.monitor.store(monitor.bits(), Ordering::Relaxed);
    }

    pub fn frame(&mut self, frame: &RenderFrame, catalog: &Catalog) {
        self.receive();
        // A full queue means the audio thread is stalled; the next frame
        // declares everything again (only its sounds played once are lost).
        if let Err(TrySendError::Disconnected(_)) = self.mixes.try_send(mix(frame, catalog, &self.clips)) {
            eprintln!("the audio output stopped");
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "test tones and WAV bytes")]
mod tests {
    use super::*;
    use mixer::{BusSpec, OPEN};

    const RATE: f64 = 48_000.0;
    /// A device's callback, in samples.
    const CALLBACK: usize = 512;

    /// The live output, without a device: what a device's callback asks of
    /// it is up to the test.
    fn output() -> (Live, SyncSender<Mix>, Receiver<Mix>) {
        let (mixes, receiving) = sync_channel(32);
        let (returning, applied) = sync_channel(64);
        let rate = Rate(Arc::new(AtomicU64::new(RATE.to_bits())));
        (Live::new(rate, receiving, returning, Arc::new(Shared::default())), mixes, applied)
    }

    /// Plays `frames` (game frames at 60 a second) through the live output as
    /// a stereo device would: CALLBACK samples at a time, while the frames
    /// arrive at their own uneven pace, up to 6 ms early or late, and all at
    /// once after a 120 ms stall at one second.
    fn live(frames: Vec<Mix>) -> Vec<[f32; 2]> {
        let (mut output, mixes, applied) = output();
        let mut random = 7u32;
        let arrivals: Vec<f64> = (0..frames.len())
            .map(|index| {
                random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let at = index as f64 / 60.0 + (f64::from(random >> 8) / f64::from(1u32 << 24) - 0.5) * 0.012;
                if (1.0..1.12).contains(&at) { 1.12 } else { at.max(0.0) }
            })
            .collect();
        let seconds = frames.len() as f64 / 60.0 + 0.5;
        let mut frames = frames.into_iter().zip(arrivals).peekable();
        let mut out = Vec::new();
        let mut buffer = vec![0.0; CALLBACK * 2];
        let mut time = 0.0;
        while time < seconds {
            while let Some((mix, _)) = frames.next_if(|(_, at)| *at <= time) {
                mixes.try_send(mix).unwrap();
            }
            output.fill(&mut buffer, 2);
            out.extend_from_slice(buffer.as_chunks::<2>().0);
            while applied.try_recv().is_ok() {}
            time += CALLBACK as f64 / RATE;
        }
        out
    }

    /// A sine of whole cycles a second, looped without a seam.
    fn sine(frequency: f64, seconds: f64) -> Arc<Clip> {
        let frames: Arc<[[f32; 2]]> = (0..(seconds * RATE) as usize)
            .map(|i| {
                let value = (std::f64::consts::TAU * frequency * i as f64 / RATE).sin() as f32;
                [value, value]
            })
            .collect();
        Arc::new(Clip::new(frames, RATE))
    }

    /// Whatever the frames change, each frame, and however unevenly they
    /// arrive, the live output never steps further between samples than its
    /// sines themselves do: no clicks.
    #[test]
    fn live_output_glides_without_clicks() {
        let (low, middle, high, blip) = (sine(220.0, 1.0), sine(330.0, 1.0), sine(440.0, 1.0), sine(500.0, 0.1));
        let voice = |id: u32, clip: &Arc<Clip>| VoiceSpec {
            id,
            clip: clip.clone(),
            bus: 1,
            gain: 0.6,
            pan: 0.0,
            pitch: 1.0,
            lowpass: OPEN,
            reverb: 0.2,
            looping: true,
            start: 0.0,
            fade_in: 0.3,
            fade_out: 0.5,
            panner: None,
        };
        let frames: Vec<Mix> = (0..240u32)
            .map(|frame| {
                let f = frame as f32;
                let mut first = voice(1, &low);
                first.gain = if frame / 3 % 2 == 0 { 0.8 } else { 0.2 };
                first.pan = (f / 9.0).sin();
                first.lowpass = if frame / 5 % 2 == 0 { 600.0 } else { OPEN };
                first.pitch = 1.0 + 0.2 * (f / 13.0).sin();
                let mut voices = vec![first];
                // On, off (fading), back while it fades, then another clip.
                if (30..90).contains(&frame) || (100..150).contains(&frame) {
                    voices.push(voice(2, &middle));
                } else if frame >= 150 {
                    voices.push(voice(2, &high));
                }
                let sounds = if frame % 20 == 0 {
                    vec![SoundSpec {
                        clip: blip.clone(),
                        bus: 2,
                        gain: 0.5,
                        pan: -0.5,
                        pitch: 1.0,
                        lowpass: 4000.0,
                        reverb: 0.3,
                        panner: None,
                    }]
                } else {
                    Vec::new()
                };
                let buses = vec![
                    BusSpec { gain: 0.9, lowpass: OPEN, reverb: 1.0 },
                    BusSpec { gain: if frame / 7 % 2 == 0 { 1.0 } else { 0.5 }, lowpass: OPEN, reverb: 0.1 },
                    BusSpec { gain: 1.0, lowpass: if frame / 11 % 2 == 0 { 1000.0 } else { OPEN }, reverb: 0.0 },
                ];
                Mix { buses, voices, sounds }
            })
            .collect();
        let out = live(frames);
        let largest = out
            .windows(2)
            .map(|pair| (pair[1][0] - pair[0][0]).abs().max((pair[1][1] - pair[0][1]).abs()))
            .fold(0.0f32, f32::max);
        let loudest = out.iter().map(|[l, r]| l.abs().max(r.abs())).fold(0.0f32, f32::max);
        // The sines at their loudest move at most about 0.03 a sample each;
        // a gain stepping from 0.8 to 0.2 unglided would jump by 0.4.
        eprintln!("live output: largest step {largest}, loudest sample {loudest}");
        assert!(largest < 0.1, "a step of {largest} (loudest sample {loudest})");
        assert!(loudest > 0.3);
    }

    /// A `pen packets` directory's frames played live, as
    /// `live_output_glides_without_clicks` plays its own, into raw f32 stereo
    /// at 48 kHz: PEN_MIX_PACKETS=DIR PEN_MIX_ASSETS=DIR PEN_MIX_OUT=FILE
    /// cargo test --release -p pen live_packets -- --ignored
    #[test]
    #[ignore]
    fn live_packets() {
        let var = |name: &str| PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("{name} is not set")));
        let (packets, assets, out) = (var("PEN_MIX_PACKETS"), var("PEN_MIX_ASSETS"), var("PEN_MIX_OUT"));
        let catalog = pen_protocol::boundary::read_catalog(&assets).unwrap();
        let clips: HashMap<String, Arc<Clip>> =
            catalog.sounds.iter().map(|key| (key.clone(), decode(&assets.join(key), RATE).unwrap())).collect();
        let schema: u32 = std::fs::read_to_string(packets.join("schema.txt")).unwrap().trim().parse().unwrap();
        let mut paths: Vec<PathBuf> =
            std::fs::read_dir(packets.join("output")).unwrap().map(|entry| entry.unwrap().path()).collect();
        paths.sort();
        let frames = paths
            .iter()
            .map(|path| {
                let bytes = std::fs::read(path).unwrap();
                let lanes: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect();
                mix(&pen_protocol::boundary::decode_frame(&lanes, schema, &catalog).unwrap().render, &catalog, &clips)
            })
            .collect();
        let samples = live(frames);
        let bytes: Vec<u8> = samples.iter().flatten().flat_map(|sample| sample.to_le_bytes()).collect();
        std::fs::write(&out, bytes).unwrap();
    }

    /// Every clip in an exported bundle decodes, those that stream too, to
    /// as long as their headers say.
    #[test]
    fn bundle_clips_decode() {
        let Some(bundle) = crate::fixtures::bundle() else { return };
        let assets = bundle.join("assets");
        let catalog = pen_protocol::boundary::read_catalog(&assets).unwrap();
        let decode = |key: &String| -> Result<()> {
            let clip = load_clip(&assets.join(key))?;
            if let Some(encoded) = &clip.streamed {
                let whole = encoded.decode()?;
                anyhow::ensure!(
                    whole.frames.len() == encoded.frames,
                    "{} frames, not {}",
                    whole.frames.len(),
                    encoded.frames
                );
            }
            Ok(())
        };
        let failed: Vec<String> = catalog
            .sounds
            .iter()
            .filter_map(|key| decode(key).err().map(|error| format!("{key}: {error:#}")))
            .collect();
        assert!(failed.is_empty(), "clips that do not decode: {failed:#?}");
    }

    /// A clip played streamed, at another rate, from a start, round and
    /// round with its pitch changing, and played to its end, is sample for
    /// sample the clip played decoded whole; streamed, it keeps only its
    /// bytes.
    #[test]
    fn streams_play_what_whole_clips_play() {
        let samples: Vec<i16> = (0..2 * 33_075)
            .map(|i| ((i as f64 * 0.0131).sin() * 9000.0 + (i as f64 * 0.17).sin() * 3000.0) as i16)
            .collect();
        let clips = [
            wav(22_050, 2, &samples),
            wav(48_000, 1, &samples[..40_000]),
            include_bytes!("audio/testdata/tone.opus").to_vec(),
        ];
        for bytes in clips {
            // The header both hosts decide by agrees with the decoders'.
            let header = pen_runtime::clip::header(&bytes).unwrap();
            let encoded = Encoded::new(bytes.into()).unwrap();
            assert_eq!((header.frames as usize, f64::from(header.rate)), (encoded.frames, encoded.rate));
            let whole = Arc::new(encoded.decode().unwrap()).at_rate(RATE);
            let streamed = Arc::new(Clip::streamed(encoded)).at_rate(RATE);
            assert!(streamed.frames.is_empty() && streamed.len() == whole.frames.len());
            let play = |clip: &Arc<Clip>| {
                let mut mixer = Mixer::new(RATE);
                let mut out = Vec::new();
                for frame in 0..150u32 {
                    let looping = frame < 120;
                    let voice = VoiceSpec {
                        id: 1,
                        clip: clip.clone(),
                        bus: 0,
                        gain: 0.8,
                        pan: 0.3,
                        pitch: 1.0 + 0.3 * (frame as f32 / 9.0).sin(),
                        lowpass: mixer::OPEN,
                        reverb: 0.0,
                        looping,
                        start: 0.37,
                        fade_in: 0.0,
                        fade_out: 0.0,
                        panner: None,
                    };
                    mixer.apply(&Mix { voices: vec![voice], ..Mix::default() });
                    let mut chunk = vec![[0.0f32; 2]; 800];
                    mixer.render(&mut chunk);
                    out.extend(chunk);
                }
                (out, mixer.underruns())
            };
            let ((whole, _), (streamed, underruns)) = (play(&whole), play(&streamed));
            assert_eq!(underruns, 0);
            assert!(whole.iter().any(|[left, _]| left.abs() > 0.1));
            let first = whole.iter().zip(&streamed).position(|(a, b)| a != b);
            assert_eq!(first, None, "streamed differs from sample {first:?}");
        }
    }

    /// A voice of a stereo clip whose sides are a sine and a slower one,
    /// as loud as `gain` makes them.
    fn stereo_voice(gain: f32) -> Mix {
        let frames: Arc<[[f32; 2]]> = (0..48_000)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / RATE;
                [(t * 300.0).sin() as f32, (t * 120.0).sin() as f32 * 0.5]
            })
            .collect();
        let voice = VoiceSpec {
            id: 1,
            clip: Arc::new(Clip::new(frames, RATE)),
            bus: 0,
            gain,
            pan: 0.2,
            pitch: 1.0,
            lowpass: mixer::OPEN,
            reverb: 0.3,
            looping: true,
            start: 0.0,
            fade_in: 0.0,
            fade_out: 0.0,
            panner: None,
        };
        Mix { voices: vec![voice], ..Mix::default() }
    }

    /// Whatever channels a device has, the live output gives it the mix the
    /// mixer renders, sample for sample, within -1 and 1: both sides on the
    /// first two, their average on a single one, silence on the rest.
    #[test]
    fn live_output_fills_any_channels() {
        // Loud enough to be cut at 1, over 1000 samples: not a whole number
        // of quanta.
        let mut mixer = Mixer::new(RATE);
        mixer.apply(&stereo_voice(3.0));
        let mut expected = vec![[0.0f32; 2]; 1000];
        mixer.render(&mut expected);
        assert!(expected.iter().any(|[left, _]| left.abs() > 1.0));
        for channels in [1, 2, 6] {
            let (mut output, mixes, _applied) = output();
            mixes.try_send(stereo_voice(3.0)).unwrap();
            let mut buffer = vec![f32::NAN; 1000 * channels];
            output.fill(&mut buffer, channels);
            for (frame, [left, right]) in buffer.chunks_exact(channels).zip(&expected) {
                let (left, right) = (left.clamp(-1.0, 1.0), right.clamp(-1.0, 1.0));
                if channels == 1 {
                    assert_eq!(frame, [(left + right) / 2.0]);
                } else {
                    assert_eq!(frame[..2], [left, right]);
                    assert!(frame[2..].iter().all(|&sample| sample == 0.0), "{frame:?}");
                }
            }
        }
    }

    /// A device at another rate starts the mixer afresh at it, for the
    /// frames to come to declare what plays and the clips to be decoded
    /// again for; one at the same rate plays on.
    #[test]
    fn live_output_starts_again_at_a_new_rate() {
        let (mut output, mixes, _applied) = output();
        mixes.try_send(stereo_voice(0.5)).unwrap();
        output.fill(&mut [0.0; 512], 2);
        output.start(RATE);
        assert_eq!(output.mixer.playing(), 1);
        output.start(44_100.0);
        assert_eq!((output.mixer.rate(), output.rate.get(), output.mixer.playing()), (44_100.0, 44_100.0, 0));
    }

    /// The live output hands its meters over to the game's side, takes the
    /// developer's mute from it, and says how late it plays.
    #[test]
    fn live_output_meters_mutes_and_times() {
        let (mut output, mixes, _applied) = output();
        let shared = output.shared.clone();
        mixes.try_send(stereo_voice(0.5)).unwrap();
        let mut buffer = vec![0.0; 1024];
        output.played(Some(0.010), 512);
        output.fill(&mut buffer, 2);
        let meters = shared.meters.lock().unwrap().0;
        assert_eq!(meters[0].samples, 512);
        assert!(meters[0].peak[0] > 0.1);
        // 10 ms ahead, then half a callback and half a quantum on average.
        let latency = f64::from_bits(shared.latency.load(Ordering::Relaxed));
        assert!((latency - (0.010 + 320.0 / RATE)).abs() < 1e-9, "{latency}");
        assert_eq!(shared.callback.load(Ordering::Relaxed), 512);
        // Muting the master silences the output once its gain has glided.
        shared.monitor.store(Monitor { muted: 1, soloed: 0 }.bits(), Ordering::Relaxed);
        for _ in 0..20 {
            output.fill(&mut buffer, 2);
        }
        assert!(buffer.iter().all(|sample| sample.abs() < 1e-4));
    }

    /// Latency is averaged over callbacks, and unknown until a device says.
    #[test]
    fn latency_averages_callbacks() {
        let mut latency = Latency::default();
        latency.measure(None, 512, RATE);
        assert_eq!(latency.seconds, None);
        latency.measure(Some(0.02), 832, RATE);
        assert!((latency.seconds.unwrap() - 0.03).abs() < 1e-12);
        latency.measure(Some(0.04), 832, RATE);
        assert!((latency.seconds.unwrap() - 0.031).abs() < 1e-12);
    }

    /// Played live as a device would, in time, a streamed clip's voice
    /// waits for its stream to begin and then never runs dry: the
    /// streamer's thread keeps ahead of it.
    #[test]
    fn live_streams_keep_ahead() {
        let samples: Vec<i16> = (0..48_000 * 3).map(|i| ((i as f64 * 0.05).sin() * 8000.0) as i16).collect();
        let clip = Arc::new(Clip::streamed(Encoded::new(wav(48_000, 1, &samples).into()).unwrap()));
        let (mut output, mixes, _applied) = output();
        let voice = VoiceSpec {
            id: 1,
            clip,
            bus: 0,
            gain: 1.0,
            pan: 0.0,
            pitch: 1.5,
            lowpass: mixer::OPEN,
            reverb: 0.0,
            looping: true,
            start: 2.5,
            fade_in: 0.0,
            fade_out: 0.0,
            panner: None,
        };
        mixes.try_send(Mix { voices: vec![voice], ..Mix::default() }).unwrap();
        let mut buffer = vec![0.0; CALLBACK * 2];
        let mut heard = 0;
        for _ in 0..60 {
            output.fill(&mut buffer, 2);
            heard += buffer.iter().filter(|sample| sample.abs() > 1e-3).count();
            std::thread::sleep(std::time::Duration::from_secs_f64(CALLBACK as f64 / RATE));
        }
        assert_eq!(output.mixer.underruns(), 0);
        assert_eq!(output.mixer.streaming(), 1);
        assert!(heard > 50_000, "{heard}");
    }

    /// A 16-bit PCM WAV file of `samples`, interleaved.
    fn wav(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let size = (samples.len() * 2) as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + size).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * u32::from(channels) * 2).to_le_bytes());
        bytes.extend_from_slice(&(channels * 2).to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend(samples.iter().flat_map(|sample| sample.to_le_bytes()));
        bytes
    }

    /// WAV clips, as the game's sounds come (16-bit PCM), decode at their
    /// own rate, a sample `s` as `s / 32768`, and a mono one is mono.
    #[test]
    fn wav_clips_decode() {
        let mut samples: Vec<i16> = (0..9_998).map(|i| ((i * 7_919) % 65_536 - 32_768) as i16).collect();
        samples.extend([i16::MIN, i16::MAX]);
        let value = |sample: i16| f32::from(sample) / 32_768.0;
        let mono = decode_wav(wav(22_050, 1, &samples)).unwrap();
        assert_eq!((mono.rate, mono.mono, mono.frames.len()), (22_050.0, true, 10_000));
        assert!(mono.frames.iter().zip(&samples).all(|(frame, &sample)| *frame == [value(sample); 2]));
        assert_eq!(mono.frames[9_998..], [[-1.0; 2], [value(i16::MAX); 2]]);
        let stereo = decode_wav(wav(44_100, 2, &samples)).unwrap();
        assert_eq!((stereo.rate, stereo.mono, stereo.frames.len()), (44_100.0, false, 5_000));
        assert!(
            stereo
                .frames
                .iter()
                .zip(samples.as_chunks::<2>().0)
                .all(|(frame, &[left, right])| *frame == [value(left), value(right)])
        );
        assert!(decode_wav(b"not a clip".to_vec()).is_err());
    }
}
