//! Records frames to a video at a steady rate through ffmpeg, with the game's
//! sound mixed offline in step with them: each frame lasts exactly 1/fps
//! of game time and of audio, however long it took to draw. The mix is the
//! live output's (`audio::mixer`), so the recording sounds as the game does,
//! and the same frames always give the same samples.
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::mpsc::{SyncSender, sync_channel};
use std::thread::JoinHandle;

use anyhow::{Context, Result, bail};
use pen_protocol::{Catalog, RenderFrame};

use crate::audio::mixer::{Clip, Mix, Mixer};

/// The recording's sample rate.
const RATE: f64 = 48_000.0;

/// The catalog's clips, decoded up front at the recording's rate, so a sound
/// plays from the frame that asks for it.
fn decode_clips(assets: &Path, catalog: &Catalog) -> HashMap<String, Arc<Clip>> {
    let mut clips = HashMap::new();
    for key in &catalog.sounds {
        match crate::audio::decode(&assets.join(key), RATE) {
            Ok(clip) => {
                clips.insert(key.clone(), clip);
            }
            Err(error) => eprintln!("recording without sound {key}: {error:#}"),
        }
    }
    clips
}

/// The game's sound, mixed offline into `path` as raw f32 stereo, each
/// frame's samples written as it arrives: the recording's whole sound is
/// never held in memory.
struct Offline {
    clips: HashMap<String, Arc<Clip>>,
    mixer: Mixer,
    path: PathBuf,
    file: BufWriter<File>,
    /// Samples written so far.
    written: usize,
    /// One frame's samples, reused.
    samples: Vec<[f32; 2]>,
}

impl Offline {
    fn create(path: PathBuf, clips: HashMap<String, Arc<Clip>>) -> Result<Offline> {
        let file = File::create(&path).with_context(|| format!("creating {}", path.display()))?;
        Ok(Offline {
            clips,
            mixer: Mixer::new(RATE),
            path,
            file: BufWriter::new(file),
            written: 0,
            samples: Vec::new(),
        })
    }

    /// A frame's sound, then the samples up to `due`.
    fn frame(&mut self, frame: &RenderFrame, catalog: &Catalog, due: usize) -> Result<()> {
        let mix = crate::audio::mix(frame, catalog, &self.clips);
        self.mix(&mix, due)
    }

    fn mix(&mut self, mix: &Mix, due: usize) -> Result<()> {
        self.mixer.apply(mix);
        self.samples.resize(due.saturating_sub(self.written), [0.0; 2]);
        self.mixer.render(&mut self.samples);
        for sample in self.samples.iter().flatten() {
            self.file.write_all(&sample.to_le_bytes())?;
        }
        self.written += self.samples.len();
        Ok(())
    }

    /// The mix's file, complete.
    fn finish(self) -> Result<PathBuf> {
        self.file.into_inner().map_err(|error| error.into_error())?.sync_all()?;
        Ok(self.path)
    }
}

/// Frames waiting for ffmpeg at most: past them, the frame loop waits.
const QUEUED_FRAMES: usize = 2;

pub struct Recorder {
    path: PathBuf,
    /// Where ffmpeg writes the picture: the output itself when silent.
    video: PathBuf,
    ffmpeg: Child,
    /// Frames for the thread that writes them to ffmpeg, and that thread.
    pictures: Option<SyncSender<Vec<u8>>>,
    writer: Option<JoinHandle<std::io::Result<()>>>,
    width: u32,
    height: u32,
    fps: f64,
    /// Frames whose pictures, and whose sounds, have been added.
    frames: u64,
    sounds: u64,
    mixer: Option<Offline>,
}

fn webm(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "webm")
}

impl Recorder {
    /// Starts ffmpeg for a `width` x `height` stream at `fps` of BGRA
    /// pixels (`bgra`) or RGBA ones. With `audio`, the catalog's clips are
    /// decoded now for the offline mix.
    pub fn start(
        path: &Path,
        (width, height): (u32, u32),
        fps: f64,
        bgra: bool,
        audio: Option<(&Path, &Catalog)>,
    ) -> Result<Recorder> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let video = match audio {
            Some(_) => {
                let name = path.file_name().context("recording needs a file name")?.to_string_lossy();
                path.with_file_name(format!(".{name}.video.{}", if webm(path) { "webm" } else { "mkv" }))
            }
            None => path.to_owned(),
        };
        let codec: &[&str] = if webm(path) {
            &["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-row-mt", "1", "-deadline", "good"]
        } else {
            &["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
        };
        let mut ffmpeg = Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", if bgra { "bgra" } else { "rgba" }])
            .args(["-video_size", &format!("{width}x{height}"), "-framerate", &fps.to_string(), "-i", "-"])
            // yuv420p needs even sides.
            .args(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
            .args(codec)
            .args(["-pix_fmt", "yuv420p"])
            .arg(&video)
            .stdin(Stdio::piped())
            .spawn()
            .context("starting ffmpeg, which recording needs")?;
        let stdin = ffmpeg.stdin.take().context("ffmpeg's input")?;
        let mixer = match audio
            .map(|(assets, catalog)| Offline::create(video.with_extension("f32"), decode_clips(assets, catalog)))
        {
            Some(Ok(mixer)) => Some(mixer),
            Some(Err(error)) => {
                let _ = ffmpeg.kill();
                let _ = ffmpeg.wait();
                return Err(error);
            }
            None => None,
        };
        // Encoding takes longer than drawing: frames go to ffmpeg from a
        // thread of their own while the next ones draw.
        let (pictures, queued) = sync_channel::<Vec<u8>>(QUEUED_FRAMES);
        let writer = std::thread::Builder::new().name("pen-record".into()).spawn(move || {
            let mut stdin = stdin;
            for picture in queued {
                stdin.write_all(&picture)?;
            }
            Ok(())
        })?;
        Ok(Recorder {
            path: path.to_owned(),
            video,
            ffmpeg,
            pictures: Some(pictures),
            writer: Some(writer),
            width,
            height,
            fps,
            frames: 0,
            sounds: 0,
            mixer,
        })
    }

    /// Adds 1/fps of a frame's sound, as it is drawn.
    pub fn sound(&mut self, frame: &RenderFrame, catalog: &Catalog) -> Result<()> {
        self.sounds += 1;
        if let Some(mixer) = &mut self.mixer {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "fps is positive (`pen run --fps`)"
            )]
            let due = (self.sounds as f64 * RATE / self.fps).round() as usize;
            mixer.frame(frame, catalog, due).context("writing the recording's sound")?;
        }
        Ok(())
    }

    /// Adds a frame's picture, its pixels with rows packed, once it has
    /// been read back: the frames' pictures come in the order they drew.
    pub fn picture(&mut self, pixels: Vec<u8>, width: u32, height: u32) -> Result<()> {
        if (width, height) != (self.width, self.height) {
            bail!("the recording is {}x{}; the window cannot change size while recording", self.width, self.height);
        }
        let sent = self.pictures.as_ref().context("recording already finished")?.send(pixels);
        if sent.is_err() {
            // The writer stopped: what it said is the trouble.
            self.stop_writing()?;
            bail!("ffmpeg stopped taking frames");
        }
        self.frames += 1;
        Ok(())
    }

    /// Ends the frames for ffmpeg and waits for the writer to hand it the last.
    fn stop_writing(&mut self) -> Result<()> {
        drop(self.pictures.take());
        match self.writer.take().map(JoinHandle::join) {
            Some(Ok(written)) => written.context("writing a frame to ffmpeg"),
            Some(Err(_)) => bail!("writing frames to ffmpeg panicked"),
            None => Ok(()),
        }
    }

    /// Closes the stream and, with sound, muxes the mix into the output.
    pub fn finish(mut self) -> Result<PathBuf> {
        let written = self.stop_writing();
        let status = self.ffmpeg.wait()?;
        written?;
        if !status.success() {
            bail!("ffmpeg failed encoding {}", self.video.display());
        }
        if let Some(mixer) = self.mixer.take() {
            let audio = mixer.finish().context("writing the recording's sound")?;
            // The mix as it is, before it is encoded: for checking it.
            if let Some(path) = std::env::var_os("PEN_RECORD_PCM") {
                std::fs::copy(&audio, &path).with_context(|| format!("writing {}", Path::new(&path).display()))?;
            }
            let codec = if webm(&self.path) { "libopus" } else { "aac" };
            let status = Command::new("ffmpeg")
                .args(["-y", "-loglevel", "error", "-i"])
                .arg(&self.video)
                .args(["-f", "f32le", "-ar", "48000", "-ac", "2", "-i"])
                .arg(&audio)
                .args(["-c:v", "copy", "-c:a", codec, "-b:a", "192k", "-shortest"])
                .arg(&self.path)
                .status()?;
            let _ = std::fs::remove_file(&audio);
            let _ = std::fs::remove_file(&self.video);
            if !status.success() {
                bail!("ffmpeg failed adding sound to {}", self.path.display());
            }
        }
        eprintln!(
            "recorded {} frames ({:.1} s at {} fps) to {}",
            self.frames,
            self.frames as f64 / self.fps,
            self.fps,
            self.path.display()
        );
        Ok(self.path.clone())
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "test tones")]
mod tests {
    use super::*;

    /// A recording holds exactly one frame per tick at the rate asked for,
    /// and its sound lasts as long as its picture.
    #[test]
    fn recordings_are_steady() {
        if Command::new("ffprobe").arg("-version").output().is_err() {
            eprintln!("skipped: no ffmpeg");
            return;
        }
        let directory = std::env::temp_dir().join(format!("pen-record-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("steady.mp4");
        let catalog = Catalog { meshes: vec!["m".into()], materials: vec!["x".into()], sounds: vec![] };
        let mut header = vec![0.0f32; pen_protocol::boundary::PACKET_HEADER_FLOATS];
        header[0] = pen_protocol::boundary::PACKET_MAGIC;
        header[1] = 1.0;
        let frame = pen_protocol::boundary::decode_frame(&header, 1, &catalog).unwrap().render;
        let mut recorder = Recorder::start(&path, (64, 48), 30.0, true, Some((directory.as_path(), &catalog))).unwrap();
        for index in 0..45u8 {
            recorder.sound(&frame, &catalog).unwrap();
            recorder.picture(vec![index * 5; 64 * 48 * 4], 64, 48).unwrap();
        }
        assert!(recorder.picture(vec![0; 4], 1, 1).is_err(), "another size");
        recorder.finish().unwrap();
        let probe = Command::new("ffprobe")
            .args(["-v", "error", "-count_frames", "-show_entries"])
            .args(["stream=codec_type,nb_read_frames,r_frame_rate,duration", "-of", "csv=p=0"])
            .arg(&path)
            .output()
            .unwrap();
        let text = String::from_utf8(probe.stdout).unwrap();
        let _ = std::fs::remove_dir_all(&directory);
        let streams: Vec<Vec<&str>> = text.lines().map(|line| line.split(',').collect()).collect();
        let video = streams.iter().find(|stream| stream[0] == "video").expect(&text);
        let audio = streams.iter().find(|stream| stream[0] == "audio").expect(&text);
        assert_eq!(video[1], "30/1", "{text}");
        assert_eq!(video[3], "45", "{text}");
        let seconds = |stream: &Vec<&str>| stream[2].parse::<f64>().unwrap();
        assert!((seconds(video) - 1.5).abs() < 1e-3, "{text}");
        // AAC frames are 1024 samples, so its length rounds to about 21 ms.
        assert!((seconds(audio) - 1.5).abs() < 0.03, "{text}");
    }

    /// The offline mix goes to its file a frame at a time as the frames
    /// arrive, and holds exactly the samples one render of the same frames
    /// gives: the same frames always mix to the same bytes.
    #[test]
    fn offline_mix_streams_the_same_samples() {
        use crate::audio::mixer::{BusSpec, OPEN, SoundSpec, VoiceSpec};
        let directory = std::env::temp_dir().join(format!("pen-offline-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let sine = |frequency: f64, seconds: f64| {
            let frames: Arc<[[f32; 2]]> = (0..(seconds * RATE) as usize)
                .map(|i| {
                    let value = (std::f64::consts::TAU * frequency * i as f64 / RATE).sin() as f32;
                    [value, value * 0.5]
                })
                .collect();
            Arc::new(Clip::new(frames, RATE))
        };
        let (tone, blip) = (sine(220.0, 1.0), sine(660.0, 0.05));
        let mixes = || {
            (0..90u32).map(|frame| {
                let voice = VoiceSpec {
                    id: 1,
                    clip: tone.clone(),
                    bus: 1,
                    gain: if frame / 10 % 2 == 0 { 0.8 } else { 0.3 },
                    pan: (frame as f32 / 7.0).sin(),
                    pitch: 1.0 + frame as f32 * 0.005,
                    lowpass: if frame / 15 % 2 == 0 { 2000.0 } else { OPEN },
                    reverb: 0.4,
                    looping: true,
                    start: 0.0,
                    fade_in: 0.1,
                    fade_out: 0.2,
                    panner: None,
                };
                let sounds = if frame % 9 == 0 {
                    vec![SoundSpec {
                        clip: blip.clone(),
                        bus: 2,
                        gain: 0.5,
                        pan: -0.3,
                        pitch: 1.5,
                        lowpass: 3000.0,
                        reverb: 0.2,
                        panner: None,
                    }]
                } else {
                    Vec::new()
                };
                let buses = vec![
                    BusSpec { gain: 1.0, lowpass: OPEN, reverb: 1.0 },
                    BusSpec { gain: 0.9, lowpass: OPEN, reverb: 0.3 },
                    BusSpec { gain: 0.7, lowpass: 5000.0, reverb: 0.0 },
                ];
                Mix { buses, voices: if frame < 70 { vec![voice] } else { Vec::new() }, sounds }
            })
        };
        // 24 frames a second: 2000 samples each, not a whole number of quanta.
        let due = |frames: usize| (frames as f64 * RATE / 24.0).round() as usize;
        let mut files = Vec::new();
        for name in ["first.f32", "second.f32"] {
            let mut offline = Offline::create(directory.join(name), HashMap::new()).unwrap();
            for (index, mix) in mixes().enumerate() {
                offline.mix(&mix, due(index + 1)).unwrap();
            }
            files.push(std::fs::read(offline.finish().unwrap()).unwrap());
        }
        let _ = std::fs::remove_dir_all(&directory);
        let mut mixer = Mixer::new(RATE);
        let mut expected = Vec::new();
        for (index, mix) in mixes().enumerate() {
            mixer.apply(&mix);
            let start = expected.len();
            expected.resize(due(index + 1), [0.0f32; 2]);
            mixer.render(&mut expected[start..]);
        }
        assert!(expected.iter().any(|[left, _]| left.abs() > 0.1));
        let expected: Vec<u8> = expected.iter().flatten().flat_map(|sample| sample.to_le_bytes()).collect();
        assert_eq!(files[0].len(), 90 * 2000 * 8);
        assert!(files[0] == expected, "the streamed mix differs from the mix rendered whole");
        assert!(files[1] == files[0], "the same frames mixed to other bytes");
    }

    /// Mixes a `pen packets` directory's frames (60 a second) offline, as a
    /// recording would, into raw f32 stereo at 48 kHz: for comparing with the
    /// browser's mix of the same packets.
    ///
    /// PEN_MIX_PACKETS=DIR PEN_MIX_ASSETS=DIR PEN_MIX_OUT=FILE \
    ///   cargo test --release -p pen mix_packets -- --ignored
    ///
    /// With PEN_MIX_VOICES=ID,ID only those voices play, and no sounds.
    #[test]
    #[ignore]
    fn mix_packets() {
        let var = |name: &str| PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("{name} is not set")));
        let (packets, assets, out) = (var("PEN_MIX_PACKETS"), var("PEN_MIX_ASSETS"), var("PEN_MIX_OUT"));
        let catalog = pen_protocol::boundary::read_catalog(&assets).unwrap();
        let schema: u32 = std::fs::read_to_string(packets.join("schema.txt")).unwrap().trim().parse().unwrap();
        let mut paths: Vec<PathBuf> =
            std::fs::read_dir(packets.join("output")).unwrap().map(|entry| entry.unwrap().path()).collect();
        paths.sort();
        let only: Option<Vec<u32>> = std::env::var("PEN_MIX_VOICES")
            .ok()
            .map(|ids| ids.split(',').map(|id| id.trim().parse().unwrap()).collect());
        let mut offline = Offline::create(out.clone(), decode_clips(&assets, &catalog)).unwrap();
        for (index, path) in paths.iter().enumerate() {
            let bytes = std::fs::read(path).unwrap();
            let lanes: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect();
            let mut frame = pen_protocol::boundary::decode_frame(&lanes, schema, &catalog).unwrap().render;
            if let Some(only) = &only {
                frame.voices.retain(|voice| only.contains(&voice.id));
                frame.sounds.clear();
            }
            offline.frame(&frame, &catalog, (index + 1) * 800).unwrap();
        }
        let samples = offline.written;
        offline.finish().unwrap();
        eprintln!("mixed {} frames into {samples} samples at {}", paths.len(), out.display());
    }
}
