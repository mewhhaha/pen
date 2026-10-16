//! The output device: a cpal stream on the default output device that plays
//! the live mixer (`Live`), kept by a thread of its own. Every half second
//! it checks the stream, and starts it again on the default device when the
//! device went away or, but on macOS (where asking while playing makes
//! artifacts), another became the default or its rate changed; while there
//! is none, it keeps trying.
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
    /// Starts playing `live` on the default output device; fails if there
    /// is none or it does not start.
    pub fn start(live: Arc<Mutex<Live>>) -> Result<Device> {
        let (stop, stopped) = channel::<()>();
        let (started, first) = channel();
        let thread = std::thread::Builder::new().name("pen-audio-device".into()).spawn(move || {
            let stream = match Stream::start(&live) {
                Ok(stream) => stream,
                Err(error) => {
                    let _ = started.send(Err(error));
                    return;
                }
            };
            let _ = started.send(Ok(()));
            keep(&live, stream, &stopped);
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
fn keep(live: &Arc<Mutex<Live>>, stream: Stream, stopped: &Receiver<()>) {
    let mut playing = Some(stream);
    let mut reported = false;
    while let Err(RecvTimeoutError::Timeout) = stopped.recv_timeout(CHECK) {
        if playing.as_ref().is_some_and(|stream| !stream.stale()) {
            continue;
        }
        // The old stream ends before another starts.
        playing = None;
        match Stream::start(live) {
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

impl Stream {
    fn start(live: &Arc<Mutex<Live>>) -> Result<Stream> {
        let device = cpal::default_host().default_output_device().context("there is no audio output device")?;
        let config = device.default_output_config()?.config();
        let channels = usize::from(config.channels);
        live.lock().unwrap_or_else(PoisonError::into_inner).start(f64::from(config.sample_rate));
        let callback = live.clone();
        let (report, errors) = channel();
        let stream = device.build_output_stream(
            config,
            move |out: &mut [f32], _: &cpal::OutputCallbackInfo| match callback.lock() {
                Ok(mut live) => live.fill(out, channels),
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
