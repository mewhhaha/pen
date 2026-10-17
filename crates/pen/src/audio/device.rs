//! The output device: a cpal stream on the default output device that plays
//! the live mixer (`Live`), kept by a thread of its own. Every half second
//! it checks the stream, and starts it again on the default device when the
//! device went away or, but on macOS (where asking while playing makes
//! artifacts), another became the default or its rate changed; while there
//! is none, it keeps trying.
//!
//! The device's buffer is the system's unless asked for (`--audio-buffer`,
//! `PEN_AUDIO_BUFFER`), within the sizes the device takes: smaller plays
//! sooner after a frame, at more risk of gaps. Each callback tells the mixer
//! when its samples will play, which is how late the output is.
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use super::Live;

/// How often the stream is checked.
const CHECK: Duration = Duration::from_millis(500);

/// The device's thread, playing while this lives.
pub struct Device {
    /// Dropped to stop the thread.
    stop: Option<Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Device {
    /// Starts playing `live` on the default output device, `buffer`
    /// samples a callback if asked; fails if there is none or it does not
    /// start.
    pub fn start(live: Arc<Mutex<Live>>, buffer: Option<u32>) -> Result<Device> {
        let (stop, stopped) = channel::<()>();
        let (started, first) = channel();
        let thread = std::thread::Builder::new().name("pen-audio-device".into()).spawn(move || {
            let stream = match Stream::start(&live, buffer) {
                Ok(stream) => stream,
                Err(error) => {
                    let _ = started.send(Err(error));
                    return;
                }
            };
            let _ = started.send(Ok(()));
            keep(&live, stream, &stopped, buffer);
        })?;
        first.recv().context("the audio device's thread ended")??;
        Ok(Device { stop: Some(stop), thread: Some(thread) })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Keeps a stream playing until `stopped` disconnects.
fn keep(live: &Arc<Mutex<Live>>, stream: Stream, stopped: &Receiver<()>, buffer: Option<u32>) {
    let mut playing = Some(stream);
    let mut reported = false;
    while let Err(RecvTimeoutError::Timeout) = stopped.recv_timeout(CHECK) {
        if playing.as_ref().is_some_and(|stream| !stream.stale()) {
            continue;
        }
        // The old stream ends before another starts.
        playing = None;
        match Stream::start(live, buffer) {
            Ok(stream) => {
                playing = Some(stream);
                reported = false;
            }
            Err(error) if !reported => {
                eprintln!("the audio output stopped: {error:#}");
                reported = true;
            }
            Err(_) => {}
        }
    }
}

/// A stream on the default output device, as it was when it started.
#[cfg_attr(target_os = "macos", allow(dead_code))]
struct Stream {
    _stream: cpal::Stream,
    device: Option<cpal::DeviceId>,
    rate: u32,
    errors: Receiver<cpal::Error>,
}

/// The buffer to ask the device for: `asked` within the sizes it takes, or
/// the system's.
pub fn buffer_size(asked: Option<u32>, supported: &cpal::SupportedBufferSize) -> cpal::BufferSize {
    match (asked, supported) {
        (None, _) => cpal::BufferSize::Default,
        (Some(asked), cpal::SupportedBufferSize::Range { min, max }) => {
            cpal::BufferSize::Fixed(asked.clamp(*min, (*max).max(*min)))
        }
        (Some(asked), cpal::SupportedBufferSize::Unknown) => cpal::BufferSize::Fixed(asked.max(1)),
    }
}

/// How long after the callback its first sample plays, if the device says.
fn ahead(info: &cpal::OutputCallbackInfo) -> Option<f64> {
    let timestamp = info.timestamp();
    timestamp.playback.checked_duration_since(timestamp.callback).map(|ahead| ahead.as_secs_f64())
}

impl Stream {
    fn start(live: &Arc<Mutex<Live>>, buffer: Option<u32>) -> Result<Stream> {
        let device = cpal::default_host().default_output_device().context("there is no audio output device")?;
        let supported = device.default_output_config()?;
        let mut config = supported.config();
        config.buffer_size = buffer_size(buffer, supported.buffer_size());
        let channels = usize::from(config.channels);
        live.lock().unwrap_or_else(PoisonError::into_inner).start(f64::from(config.sample_rate));
        let callback = live.clone();
        let (report, errors) = channel();
        let stream = device.build_output_stream(
            config,
            move |out: &mut [f32], info: &cpal::OutputCallbackInfo| match callback.lock() {
                Ok(mut live) => {
                    live.played(ahead(info), out.len() / channels.max(1));
                    live.fill(out, channels);
                }
                Err(_) => out.fill(0.0),
            },
            move |error| {
                let _ = report.send(error);
            },
            None,
        )?;
        stream.play()?;
        Ok(Stream { _stream: stream, device: device.id().ok(), rate: config.sample_rate, errors })
    }

    /// Whether to start again, on another device or at another rate.
    fn stale(&self) -> bool {
        let lost = self.errors.try_iter().any(|error| {
            matches!(error.kind(), cpal::ErrorKind::DeviceNotAvailable | cpal::ErrorKind::StreamInvalidated)
        });
        if lost {
            return true;
        }
        #[cfg(not(target_os = "macos"))]
        if let Some(device) = cpal::default_host().default_output_device() {
            let rate = device.default_output_config().map(|config| config.sample_rate());
            return device.id().ok() != self.device || rate.is_ok_and(|rate| rate != self.rate);
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A buffer asked for is one the device takes; none asked for is the
    /// system's.
    #[test]
    fn asks_for_a_buffer_the_device_takes() {
        let range = cpal::SupportedBufferSize::Range { min: 64, max: 4096 };
        assert_eq!(buffer_size(None, &range), cpal::BufferSize::Default);
        assert_eq!(buffer_size(Some(256), &range), cpal::BufferSize::Fixed(256));
        assert_eq!(buffer_size(Some(16), &range), cpal::BufferSize::Fixed(64));
        assert_eq!(buffer_size(Some(1 << 20), &range), cpal::BufferSize::Fixed(4096));
        assert_eq!(buffer_size(Some(512), &cpal::SupportedBufferSize::Unknown), cpal::BufferSize::Fixed(512));
    }
}
