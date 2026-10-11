//! Records frames to a video at a steady rate through ffmpeg, with the game's
//! sound mixed offline in step with them: each frame lasts exactly 1/fps
//! of game time and of audio, however long it took to draw. The mix is the
//! live output's (`audio::mixer`, WebAudio's graph as gdev's web/audio.ts
//! builds it), so the recording sounds as the game does, and the same frames
//! always give the same samples.
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use pen_protocol::{Catalog, RenderFrame};

use crate::audio::mixer::{Clip, Mixer};

/// The recording's sample rate.
const RATE: f64 = 48_000.0;

/// The game's sound, mixed offline: every clip decoded up front, so a sound
/// plays from the frame that asks for it.
struct Offline {
    clips: HashMap<String, Arc<Clip>>,
    mixer: Mixer,
    samples: Vec<[f32; 2]>,
}

impl Offline {
    fn new(assets: &Path, catalog: &Catalog) -> Offline {
        let mut clips = HashMap::new();
        for key in &catalog.sounds {
            match crate::audio::decode(&assets.join(key), RATE) {
                Ok(clip) => {
                    clips.insert(key.clone(), clip);
                }
                Err(error) => eprintln!("recording without sound {key}: {error:#}"),
            }
        }
        Offline { clips, mixer: Mixer::new(RATE), samples: Vec::new() }
    }

    /// A frame's sound, then the samples up to `due`.
    fn frame(&mut self, frame: &RenderFrame, catalog: &Catalog, due: usize) {
        self.mixer.apply(&crate::audio::mix(frame, catalog, &self.clips));
        let start = self.samples.len();
        self.samples.resize(due.max(start), [0.0; 2]);
        self.mixer.render(&mut self.samples[start..]);
    }
}

pub struct Recorder {
    path: PathBuf,
    /// Where ffmpeg writes the picture: the output itself when silent.
    video: PathBuf,
    ffmpeg: Child,
    stdin: Option<ChildStdin>,
    width: u32,
    height: u32,
    fps: f64,
    frames: u64,
    mixer: Option<Offline>,
}

fn webm(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "webm")
}

impl Recorder {
    /// Starts ffmpeg for a `width` x `height` BGRA stream at `fps`. With
    /// `audio`, the catalog's clips are decoded now for the offline mix.
    pub fn start(
        path: &Path,
        width: u32,
        height: u32,
        fps: f64,
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
            .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra"])
            .args(["-video_size", &format!("{width}x{height}"), "-framerate", &fps.to_string(), "-i", "-"])
            // yuv420p needs even sides.
            .args(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
            .args(codec)
            .args(["-pix_fmt", "yuv420p"])
            .arg(&video)
            .stdin(Stdio::piped())
            .spawn()
            .context("starting ffmpeg, which recording needs")?;
        let stdin = ffmpeg.stdin.take();
        Ok(Recorder {
            path: path.to_owned(),
            video,
            ffmpeg,
            stdin,
            width,
            height,
            fps,
            frames: 0,
            mixer: audio.map(|(assets, catalog)| Offline::new(assets, catalog)),
        })
    }

    /// Adds one frame: its BGRA pixels, rows packed, and 1/fps of its sound.
    pub fn frame(&mut self, bgra: &[u8], width: u32, height: u32, frame: &RenderFrame, catalog: &Catalog) -> Result<()> {
        if (width, height) != (self.width, self.height) {
            bail!("the recording is {}x{}; the window cannot change size while recording", self.width, self.height);
        }
        let stdin = self.stdin.as_mut().context("recording already finished")?;
        stdin.write_all(bgra).context("writing a frame to ffmpeg")?;
        self.frames += 1;
        if let Some(mixer) = &mut self.mixer {
            let due = (self.frames as f64 * RATE / self.fps).round() as usize;
            mixer.frame(frame, catalog, due);
        }
        Ok(())
    }

    /// Closes the stream and, with sound, muxes the mix into the output.
    pub fn finish(mut self) -> Result<PathBuf> {
        drop(self.stdin.take());
        let status = self.ffmpeg.wait()?;
        if !status.success() {
            bail!("ffmpeg failed encoding {}", self.video.display());
        }
        if let Some(mixer) = self.mixer.take() {
            let audio = self.video.with_extension("f32");
            let bytes: Vec<u8> = mixer.samples.iter().flatten().flat_map(|sample| sample.to_le_bytes()).collect();
            std::fs::write(&audio, &bytes)?;
            // The mix as it is, before it is encoded: for checking it.
            if let Some(path) = std::env::var_os("PEN_RECORD_PCM") {
                std::fs::write(&path, &bytes).with_context(|| format!("writing {}", Path::new(&path).display()))?;
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
        let mut recorder = Recorder::start(&path, 64, 48, 30.0, Some((directory.as_path(), &catalog))).unwrap();
        for index in 0..45u8 {
            recorder.frame(&vec![index * 5; 64 * 48 * 4], 64, 48, &frame, &catalog).unwrap();
        }
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
        let mut paths: Vec<PathBuf> = std::fs::read_dir(packets.join("output")).unwrap().map(|entry| entry.unwrap().path()).collect();
        paths.sort();
        let only: Option<Vec<u32>> =
            std::env::var("PEN_MIX_VOICES").ok().map(|ids| ids.split(',').map(|id| id.trim().parse().unwrap()).collect());
        let mut offline = Offline::new(&assets, &catalog);
        for (index, path) in paths.iter().enumerate() {
            let bytes = std::fs::read(path).unwrap();
            let lanes: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect();
            let mut frame = pen_protocol::boundary::decode_frame(&lanes, schema, &catalog).unwrap().render;
            if let Some(only) = &only {
                frame.voices.retain(|voice| only.contains(&voice.id));
                frame.sounds.clear();
            }
            offline.frame(&frame, &catalog, (index + 1) * 800);
        }
        let bytes: Vec<u8> = offline.samples.iter().flatten().flat_map(|sample| sample.to_le_bytes()).collect();
        std::fs::write(&out, bytes).unwrap();
        eprintln!("mixed {} frames into {} samples at {}", paths.len(), offline.samples.len(), out.display());
    }
}
