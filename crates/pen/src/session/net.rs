//! A session as one peer of a net session (`crate::net`, docs/netplay.md):
//! its frames' input is every player's, it rolls back where a prediction
//! of the others' proved wrong, and what would stop or change the game for
//! this peer alone (time travel, loads, reloads) is off.
use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use pen_protocol::Viewport;
use winit::window::Window;

use super::{KEEP_SECONDS, Quiet, Recorded, Session, Shell, hash_lanes, timeline};
use crate::devlog::{self, Source};
use crate::net::{Net, Spec, Udp};

impl Session {
    /// Joins the net session `spec` names: its frames keep moments to roll
    /// back to, and every packet's hash is compared with the peers'.
    pub(super) fn start_net(&mut self, spec: &Spec) -> Result<()> {
        let wasm = std::fs::read(self.paths.game.join("game.wasm")).context("reading the game for the net session")?;
        let moment = self.runtime.moment()?;
        let viewport = Viewport { width: self.gpu.size.0, height: self.gpu.size.1 };
        let link = Box::new(Udp::bind(spec)?);
        self.net = Some(Net::start(spec, link, &wasm, &moment, viewport)?);
        self.timeline.keep_history(Some(KEEP_SECONDS));
        self.runtime.hash_packets(true);
        self.overlap.enabled = false;
        devlog::info(
            Source::Session,
            format!("net: player {} of {}, waiting for the others", spec.player, spec.addresses.len()),
        );
        Ok(())
    }

    /// Time travel stops the game for this peer only, so a net session
    /// plays on without it.
    pub(super) fn travel_refused(&self) -> bool {
        if self.net.is_some() {
            devlog::info(Source::Session, "net: time travel is off in a net session");
        }
        self.net.is_some()
    }

    /// A net session's turn before a frame: what the peers sent, the
    /// host's moment for a peer that started elsewhere, and a rollback
    /// where a prediction proved wrong. Returns whether the next frame may
    /// run; while it waits, the last frame shows.
    pub(super) fn net_turn(&mut self, window: Option<&Window>) -> Result<bool> {
        let Some(mut net) = self.net.take() else { return Ok(true) };
        let ready = self.net_step(&mut net, window);
        self.net = Some(net);
        let ready = ready?;
        if !ready {
            if window.is_some() {
                self.present_last(window)?;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(ready)
    }

    fn net_step(&mut self, net: &mut Net, window: Option<&Window>) -> Result<bool> {
        net.poll()?;
        if let Some(start) = net.play.take_start() {
            let shown = {
                let shell = shell!(self, window);
                self.runtime.restore(&mut Quiet(&shell), &start)?
            };
            self.show(shown);
            devlog::info(Source::Session, "net: playing from the host's moment");
        }
        let corrected = {
            let shell = shell!(self, window);
            let trace = &mut self.trace;
            let mut unseen = |position: u64, recorded: &Recorded, runtime: &timeline::Runtime| -> Result<()> {
                if let Some(trace) = trace {
                    writeln!(trace, "{position} {:016x} {:016x}", hash_lanes(&recorded.input), runtime.packet_hash())?;
                }
                Ok(())
            };
            net.correct(&mut self.timeline, &mut self.runtime, &shell, &mut unseen)?
        };
        if let Some((shown, at)) = corrected {
            self.show(shown);
            // The input recording takes the frames as they ran again.
            self.rewind_recording(at)?;
            if let Some(recorder) = &mut self.input_recorder {
                for frame in at..self.timeline.position() {
                    recorder.frame(self.timeline.recorded(frame).context("a rolled back frame is gone")?)?;
                }
            }
        }
        net.play.ready()
    }

    /// The run ends: every peer gets this one's last input, and this one
    /// the others', so all end on the same frames; then it says goodbye.
    pub(super) fn finish_net(&mut self) -> Result<()> {
        let Some(mut net) = self.net.take() else { return Ok(()) };
        let frames = net.play.next();
        let waited = Instant::now();
        while !net.play.done(frames) && waited.elapsed() < Duration::from_secs(10) {
            self.net_step(&mut net, None)?;
            std::thread::sleep(Duration::from_millis(1));
        }
        for _ in 0..3 {
            net.play.leave();
        }
        devlog::info(Source::Session, net.report());
        Ok(())
    }
}
