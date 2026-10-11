//! Plays what the game asks for, as gdev's `web/audio.ts` does: each frame's
//! buses, the voices it keeps playing and the sounds it plays once, mixed by
//! `mixer.rs` (WebAudio's graph, sample for sample) with the shared reverb of
//! `reverb.rs`. The same mixer renders video recordings (`record.rs`), so
//! what is recorded is what plays.
//!
//! Live, kira is only the output: the mixer runs inside one endless kira
//! sound on the audio thread, and each frame's mix reaches it over a channel,
//! landing at the start of its next 128-sample quantum. Clips decode on a
//! background thread; a voice or sound whose clip is not ready yet is left
//! out until it is, as the browser leaves it out while decoding.
pub mod mixer;
mod resample;
mod reverb;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender, TrySendError, channel, sync_channel};

use anyhow::Result;
use kira::sound::static_sound::StaticSoundData;
use kira::sound::{Sound, SoundData};
use kira::{AudioManager, AudioManagerSettings, DefaultBackend, Frame};
use pen_protocol::{Catalog, RenderFrame};

use mixer::{BusSpec, Clip, Mix, Mixer, QUANTUM, SoundSpec, VoiceSpec};

/// Decodes a clip: Ogg Opus here, everything else with kira's decoders.
pub fn load_clip(path: &Path) -> Result<StaticSoundData> {
    let bytes = std::fs::read(path)?;
    if crate::opus::is_ogg_opus(&bytes) {
        return crate::opus::decode(&bytes);
    }
    Ok(StaticSoundData::from_cursor(std::io::Cursor::new(bytes))?)
}

/// A clip, decoded for a mixer at `rate`.
pub fn decode(path: &Path, rate: f64) -> Result<Arc<Clip>> {
    let data = load_clip(path)?;
    Ok(Arc::new(Clip::new(data.frames, f64::from(data.sample_rate))).at_rate(rate))
}

/// A frame's sound for the mixer, with the decoded clips it plays; voices
/// and sounds whose clip is not among them are left out.
pub fn mix(frame: &RenderFrame, catalog: &Catalog, clips: &HashMap<String, Arc<Clip>>) -> Mix {
    let clip = |index: u32| clips.get(catalog.sound(index)).cloned();
    Mix {
        buses: frame.buses.iter().map(|bus| BusSpec { gain: bus.gain, lowpass: bus.lowpass, reverb: bus.reverb }).collect(),
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
                })
            })
            .collect(),
        sounds: frame
            .sounds
            .iter()
            .filter_map(|sound| {
                Some(SoundSpec {
                    clip: clip(sound.clip)?,
                    bus: sound.bus as usize,
                    gain: sound.gain,
                    pan: sound.pan,
                    pitch: sound.pitch,
                    lowpass: sound.lowpass,
                    reverb: sound.reverb,
                })
            })
            .collect(),
    }
}

/// The rate the mixer starts at; it starts again at the device's if that
/// differs (or changes), and clips are decoded again for it.
const LIKELY_RATE: f64 = 48_000.0;

/// The output's sample rate, as the audio thread finds it, for the decoder.
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

/// The mixer, as an endless kira sound on the audio thread. Mixes arrive
/// from `Audio::frame` and go back once applied, so that nothing the frames
/// allocated is freed on the audio thread.
struct Output {
    mixer: Mixer,
    rate: Rate,
    mixes: Receiver<Mix>,
    applied: SyncSender<Mix>,
    scratch: [[f32; 2]; QUANTUM],
}

impl Sound for Output {
    fn process(&mut self, out: &mut [Frame], dt: f64, _info: &kira::info::Info) {
        let rate = 1.0 / dt;
        if (rate - self.mixer.rate()).abs() > 0.5 {
            // A new device: the frames to come declare what plays.
            self.mixer = Mixer::new(rate);
            self.rate.set(rate);
        }
        while let Ok(mix) = self.mixes.try_recv() {
            self.mixer.apply(&mix);
            let _ = self.applied.try_send(mix);
        }
        for chunk in out.chunks_mut(QUANTUM) {
            let scratch = &mut self.scratch[..chunk.len()];
            self.mixer.render(scratch);
            for (frame, &[left, right]) in chunk.iter_mut().zip(scratch.iter()) {
                *frame = Frame { left, right };
            }
        }
    }

    fn finished(&self) -> bool {
        false
    }
}

impl SoundData for Output {
    type Error = std::convert::Infallible;
    type Handle = ();

    fn into_sound(self) -> Result<(Box<dyn Sound>, ()), Self::Error> {
        Ok((Box::new(self), ()))
    }
}

pub struct Audio {
    /// The output device, playing the mixer while it lives.
    _output: AudioManager<DefaultBackend>,
    clips: HashMap<String, Arc<Clip>>,
    /// The rate the clips are decoded for, and the output's.
    decoded_at: f64,
    rate: Rate,
    requests: Sender<String>,
    decoded: Receiver<(String, Result<Arc<Clip>, String>)>,
    mixes: SyncSender<Mix>,
    applied: Receiver<Mix>,
}

impl Audio {
    /// Starts the output device and decodes every clip in the catalog.
    pub fn new(assets: PathBuf, catalog: &Catalog) -> Result<Audio> {
        let mut manager = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default())?;
        let (mixes, receiving) = sync_channel(32);
        let (returning, applied) = sync_channel(64);
        let rate = Rate(Arc::new(AtomicU64::new(LIKELY_RATE.to_bits())));
        let output = Output {
            mixer: Mixer::new(LIKELY_RATE),
            rate: rate.clone(),
            mixes: receiving,
            applied: returning,
            scratch: [[0.0; 2]; QUANTUM],
        };
        manager.play(output)?;
        let (requests, wanted) = channel::<String>();
        let (sender, decoded) = channel();
        let decoder = rate.clone();
        std::thread::Builder::new().name("pen-audio-decode".into()).spawn(move || {
            for key in wanted {
                let result = decode(&assets.join(&key), decoder.get()).map_err(|error| format!("sound {key}: {error:#}"));
                if sender.send((key, result)).is_err() {
                    return;
                }
            }
        })?;
        let mut audio =
            Audio { _output: manager, clips: HashMap::new(), decoded_at: LIKELY_RATE, rate, requests, decoded, mixes, applied };
        audio.preload(catalog);
        Ok(audio)
    }

    /// Decodes the catalog's clips that are not decoded yet.
    pub fn preload(&mut self, catalog: &Catalog) {
        for key in &catalog.sounds {
            if !self.clips.contains_key(key) {
                let _ = self.requests.send(key.clone());
            }
        }
    }

    /// Clips that changed on disk, decoded afresh; each plays its old
    /// version until then, and voices playing it start again with the new
    /// one, crossfading.
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
mod tests {
    use super::*;
    use mixer::{BusSpec, OPEN};

    const RATE: f64 = 48_000.0;
    /// A device's callback, in samples.
    const CALLBACK: usize = 512;

    /// Plays `frames` (game frames at 60 a second) through the live output as
    /// a device would: CALLBACK samples at a time, while the frames arrive at
    /// their own uneven pace, up to 6 ms early or late, and all at once after
    /// a 120 ms stall at one second.
    fn live(frames: Vec<Mix>) -> Vec<[f32; 2]> {
        let (mixes, receiving) = sync_channel(32);
        let (returning, applied) = sync_channel(64);
        let mut output = Output {
            mixer: Mixer::new(RATE),
            rate: Rate(Arc::new(AtomicU64::new(RATE.to_bits()))),
            mixes: receiving,
            applied: returning,
            scratch: [[0.0; 2]; QUANTUM],
        };
        let mut random = 7u32;
        let arrivals: Vec<f64> = (0..frames.len())
            .map(|index| {
                random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let at = index as f64 / 60.0 + (f64::from(random >> 8) / f64::from(1u32 << 24) - 0.5) * 0.012;
                if (1.0..1.12).contains(&at) { 1.12 } else { at.max(0.0) }
            })
            .collect();
        let info = kira::info::MockInfoBuilder::new().build();
        let seconds = frames.len() as f64 / 60.0 + 0.5;
        let mut frames = frames.into_iter().zip(arrivals).peekable();
        let mut out = Vec::new();
        let mut buffer = vec![Frame::ZERO; CALLBACK];
        let mut time = 0.0;
        while time < seconds {
            while let Some((mix, _)) = frames.next_if(|(_, at)| *at <= time) {
                mixes.try_send(mix).unwrap();
            }
            output.process(&mut buffer, 1.0 / RATE, &info);
            out.extend(buffer.iter().map(|frame| [frame.left, frame.right]));
            while applied.try_recv().is_ok() {}
            time += CALLBACK as f64 / RATE;
        }
        out
    }

    /// A sine of whole cycles a second, looped without a seam.
    fn sine(frequency: f64, seconds: f64) -> Arc<Clip> {
        let frames: Arc<[Frame]> = (0..(seconds * RATE) as usize)
            .map(|i| {
                let value = (std::f64::consts::TAU * frequency * i as f64 / RATE).sin() as f32;
                Frame { left: value, right: value }
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
                    vec![SoundSpec { clip: blip.clone(), bus: 2, gain: 0.5, pan: -0.5, pitch: 1.0, lowpass: 4000.0, reverb: 0.3 }]
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
        let largest = out.windows(2).map(|pair| (pair[1][0] - pair[0][0]).abs().max((pair[1][1] - pair[0][1]).abs())).fold(0.0f32, f32::max);
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
        let mut paths: Vec<PathBuf> = std::fs::read_dir(packets.join("output")).unwrap().map(|entry| entry.unwrap().path()).collect();
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

    /// Every clip in an exported bundle decodes.
    #[test]
    fn bundle_clips_decode() {
        let assets = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dist/gdev/assets");
        let Ok(catalog) = pen_protocol::boundary::read_catalog(&assets) else { return };
        let failed: Vec<String> = catalog
            .sounds
            .iter()
            .filter_map(|key| load_clip(&assets.join(key)).err().map(|error| format!("{key}: {error:#}")))
            .collect();
        assert!(failed.is_empty(), "clips that do not decode: {failed:#?}");
    }
}
