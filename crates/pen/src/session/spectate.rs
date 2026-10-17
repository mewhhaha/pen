//! A session that streams its final frames to viewers (`--broadcast`), or
//! plays a stream as it comes (`--spectate`), as a replay plays a
//! recording (`crate::spectate`, docs/netplay.md).
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use pen_runtime::net::stream::Message;
use pen_runtime::timeline::Recording;
use pen_runtime::transport::Replaying;
use winit::window::Window;

use super::{KEEP_SECONDS, Quiet, Session, Shell, settle};
use crate::devlog::{self, Source};
use crate::spectate::{Stream, Watch};

/// How long a viewer waits for a stream to start.
const START_WAIT: Duration = Duration::from_secs(30);

impl Session {
    /// Streams to viewers on `address`: a viewer joining starts from a
    /// moment of the history, which streaming keeps.
    pub(super) fn start_stream(&mut self, address: &str) -> Result<()> {
        let stream = Stream::serve(address)?;
        devlog::info(Source::Session, format!("streaming to viewers on ws://{}", stream.address()));
        self.stream = Some(stream);
        self.timeline.keep_history(Some(KEEP_SECONDS));
        Ok(())
    }

    /// Sends viewers the frames that became final: in a net session those
    /// every player's input is known for, otherwise those that ran.
    pub(super) fn stream_final(&mut self) {
        let Some(stream) = &mut self.stream else { return };
        let end = match &self.net {
            Some(net) => net.play.settled().unwrap_or(0),
            None => self.timeline.position(),
        };
        if stream.poll(&self.timeline, end) > 0 {
            devlog::info(Source::Session, format!("a viewer joined the stream at frame {end}"));
        }
    }

    /// The run ends: its last final frames, then the end, to every viewer.
    pub(super) fn finish_stream(&mut self) {
        self.stream_final();
        if let Some(stream) = &mut self.stream {
            stream.end();
        }
    }

    /// Plays the stream at `address`: it waits for the stream's start, and
    /// plays its frames as a replay does, waiting for more as they come.
    pub(super) fn start_watch(&mut self, address: &str, window: Option<&Window>) -> Result<()> {
        // The stream may not be up yet.
        let asked = Instant::now();
        let mut watch = loop {
            match Watch::connect(address) {
                Ok(watch) => break watch,
                Err(_) if asked.elapsed() < START_WAIT => std::thread::sleep(Duration::from_millis(100)),
                Err(error) => return Err(error),
            }
        };
        let Some(Message::Start { position, moment }) = watch.next(START_WAIT)? else {
            bail!("the stream at {address} did not start within {} s", START_WAIT.as_secs());
        };
        self.timeline.keep_history(Some(KEEP_SECONDS));
        self.timeline.load_at(Recording { start: moment.clone(), frames: Vec::new() }, position);
        let shown = {
            let shell = shell!(self, window);
            self.runtime.restore(&mut Quiet(&shell), &moment).with_context(|| format!("watching {address}"))?
        };
        if self.options.settle {
            settle(&mut self.gpu, &shown, self.runtime.catalog())?;
        }
        self.show(shown);
        self.services.forget();
        self.transport.replay(Replaying::File);
        devlog::info(Source::Session, format!("watching {address} from frame {position}"));
        self.watch = Some(watch);
        Ok(())
    }

    /// Headless, the frame about to run is the last of a stream that ended.
    pub(super) fn watched_to_the_end(&self) -> bool {
        self.options.headless
            && self.options.frames == 0
            && self.watch.as_ref().is_some_and(Watch::ended)
            && self.timeline.next().is_some()
            && self.timeline.recorded(self.timeline.position() + 1).is_none()
    }

    /// A viewer's turn before a frame: the frames that arrived, or a new
    /// start. Returns None when a frame can run, else whether the run goes
    /// on: while no frame has come the last one shows, and once the stream
    /// ended and its frames ran, the replay ends.
    pub(super) fn watch_turn(&mut self, window: Option<&Window>) -> Result<Option<bool>> {
        let Some(mut watch) = self.watch.take() else { return Ok(None) };
        let wait = if self.timeline.next().is_some() { Duration::ZERO } else { Duration::from_millis(5) };
        let mut result = Ok(());
        let mut timeout = wait;
        loop {
            match watch.next(timeout) {
                Ok(Some(Message::Frames { first, frames })) if first == self.timeline.head() => {
                    self.timeline.extend(frames);
                }
                Ok(Some(Message::Frames { first, .. })) => {
                    result = Err(anyhow::anyhow!("the stream skipped to frame {first}"));
                    break;
                }
                Ok(Some(Message::Start { position, moment })) => {
                    // The session went back in time: from its moment again.
                    self.timeline.load_at(Recording { start: moment.clone(), frames: Vec::new() }, position);
                    let shell = shell!(self, window);
                    result = self.runtime.restore(&mut Quiet(&shell), &moment).map(drop);
                    if result.is_err() {
                        break;
                    }
                }
                Ok(Some(Message::End) | None) => break,
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
            timeout = Duration::ZERO;
        }
        result?;
        if self.timeline.next().is_some() {
            self.watch = Some(watch);
            return Ok(None);
        }
        if watch.ended() {
            devlog::info(Source::Session, "the stream ended");
            let going = !self.replay_ended();
            if !going && let Some(path) = &self.options.screenshot {
                self.capture.screenshots.push(path.clone());
                self.present_last(window)?;
            }
            return Ok(Some(going));
        }
        self.watch = Some(watch);
        if window.is_some() {
            self.present_last(window)?;
        }
        Ok(Some(true))
    }
}
