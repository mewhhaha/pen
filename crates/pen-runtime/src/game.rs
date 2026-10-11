//! Runs one long-lived `main(host)`. The program owns its state;
//! snapshots cross the boundary only for persistence, undo history and the
//! moments time travel and replays restore.
//!
//! A frame is sent and received separately, so the guest can compute one
//! frame while the host draws the one before (`Session`'s pipeline).
use std::sync::Arc;
use web_time::Instant;

use anyhow::{Result, bail, ensure};
use pen_protocol::boundary::{decode_frame, encode_frame, validate_state};
use pen_protocol::{Catalog, Frame, RenderFrame, Viewport};

use crate::program::{Backend, Operation, Program};
use crate::snapshot::SavedState;

const HISTORY_LIMIT: usize = 200;

/// What the runtime needs from the application around it.
pub trait Host {
    fn viewport(&self) -> Viewport;
    /// Rejects a frame the renderer could not draw.
    fn validate(&self, frame: &RenderFrame) -> Result<()>;
    /// Draws (and plays) a frame whose asset IDs index `catalog`.
    fn render(&mut self, frame: &RenderFrame, catalog: &Catalog) -> Result<()>;
    fn report(&mut self, message: &str);
    /// The game's world was replaced (a load, scene load, undo, redo or
    /// reload): replay the player's preferences, over whatever ones the
    /// restored state held, and drop what was held for the world replaced.
    fn restored(&mut self) {}
}

/// A host that draws, plays and says nothing: frames run again to reach a
/// moment (a rewind) are computed but never shown. Their preference events
/// are in the recorded frames already, so a restore replays none.
pub struct Quiet<'a>(pub &'a dyn Host);

impl Host for Quiet<'_> {
    fn viewport(&self) -> Viewport {
        self.0.viewport()
    }

    fn validate(&self, frame: &RenderFrame) -> Result<()> {
        self.0.validate(frame)
    }

    fn render(&mut self, _frame: &RenderFrame, _catalog: &Catalog) -> Result<()> {
        Ok(())
    }

    fn report(&mut self, _message: &str) {}
}

/// What became of a new build of the game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReloadResult {
    /// It runs now, with the game's state carried over.
    Activated,
    /// It failed to start or to take the state; the old build keeps running.
    Rejected,
    /// Its state schema differs, so the running state cannot move to it;
    /// only a restart starts the new one.
    Incompatible,
}

/// CPU wall times, in milliseconds. Guest includes simulation, render prep and bridge.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameTimings {
    /// From sending the frame's input to the guest sending its packet.
    pub guest_ms: f64,
    /// How long the host waited for the packet: the guest time that
    /// overlapping frames did not hide (all of it without overlap).
    pub wait_ms: f64,
    pub decode_ms: f64,
    pub validate_ms: f64,
    pub submit_ms: f64,
    pub housekeeping_ms: f64,
    pub total_ms: f64,
    pub packet_bytes: usize,
}

/// The edit history the host keeps for undo: scenes, and which is current.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Edits {
    pub scenes: Vec<Arc<[f32]>>,
    pub step: Option<usize>,
}

/// Everything later frames depend on besides their input: the game's state
/// and the host's edit history (a frame may ask to undo into it).
#[derive(Clone, Debug, PartialEq)]
pub struct Moment {
    pub saved: SavedState,
    pub edits: Edits,
}

struct Game<P: Program> {
    program: P,
    schema: u32,
}

pub struct GameRuntime<B: Backend> {
    engine: B,
    module: B::Module,
    catalog: Catalog,
    active: Game<B::Program>,
    edits: Edits,
    previous_timestamp: Option<f64>,
    measuring: bool,
    last_timings: Option<FrameTimings>,
    /// When the frame in flight was sent.
    sent: Option<Instant>,
    /// Hash each packet, for determinism checks.
    hashing: bool,
    packet_hash: u64,
    /// The last frame received, when hashing: tests say how two differ.
    received: Option<RenderFrame>,
    /// The last frame the runtime drew (or, starting, computed) itself:
    /// starting, loading or reloading, until taken.
    drawn: Option<RenderFrame>,
}

async fn open_game<B: Backend>(engine: &B, module: &B::Module) -> Result<Game<B::Program>> {
    let program = engine.start(module).await?;
    let schema = program.schema();
    Ok(Game { program, schema })
}

async fn state_request<P: Program>(game: &mut Game<P>, operation: Operation, payload: &[f32]) -> Result<Vec<f32>> {
    let state = game.program.request(operation, payload).await?;
    validate_state(&state)?;
    game.program.accept();
    Ok(state)
}

fn validation_frame(host: &dyn Host) -> Frame {
    Frame { timestamp: 0.0, viewport: host.viewport(), events: Vec::new() }
}

/// A frame for checking a program: it runs, but is not accepted until the
/// caller does so.
async fn evaluate<P: Program>(game: &mut Game<P>, catalog: &Catalog, host: &dyn Host) -> Result<RenderFrame> {
    let packet = game.program.request(Operation::Frame, &encode_frame(&validation_frame(host), 0.0, &[])?).await?;
    let next = decode_frame(&packet, game.schema, catalog)?;
    ensure!(next.state.is_empty(), "program frame must keep its state inside the guest");
    host.validate(&next.render)?;
    Ok(next.render)
}

fn ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// What a restore or scene load made of one saved section (gdev's
/// `packages/persist.blot` Fate, by code): 0 kept as saved, 1 migrated from
/// another version, 2 reset for want of a migration, 3 started afresh because
/// the state lacked it, 4 dropped because the build no longer saves it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SectionOutcome {
    /// The section's key, its saved type's from build to build.
    pub key: u32,
    pub fate: u32,
    /// The version the state saved it with; 0 when the state lacked it.
    pub saved: u32,
    /// The build's version; 0 for a section the build does not read.
    pub version: u32,
}

const REPORT_MAGIC: f32 = 5680.0;

/// The report a restore or scene load answers with: `5680, count`, then
/// `key, fate, saved, version` per section. None for any other answer, as
/// from a game whose restore answers with its state.
pub fn restore_report(response: &[f32]) -> Option<Vec<SectionOutcome>> {
    let [magic, count, rest @ ..] = response else {
        return None;
    };
    if *magic != REPORT_MAGIC || count.fract() != 0.0 || *count < 0.0 || rest.len() != *count as usize * 4 {
        return None;
    }
    Some(
        rest.as_chunks::<4>()
            .0
            .iter()
            .map(|&[key, fate, saved, version]| SectionOutcome {
                key: key as u32,
                fate: fate as u32,
                saved: saved as u32,
                version: version as u32,
            })
            .collect(),
    )
}

/// What a restore did other than keep each saved section as it was, for the
/// log (gdev's `describeRestore`), or None if it kept them all or did not say.
pub fn describe_restore(outcomes: Option<&[SectionOutcome]>) -> Option<String> {
    let outcomes = outcomes?;
    let changes: Vec<String> = outcomes
        .iter()
        .filter_map(|&SectionOutcome { key, fate, saved, version }| match fate {
            0 => None,
            1 => Some(format!("migrated section {key} from version {saved} to {version}")),
            2 => Some(format!("reset section {key} (version {saved} has no migration to {version})")),
            3 => Some(format!("started section {key} afresh (the state has none)")),
            4 => Some(format!("dropped section {key} (no longer saved)")),
            _ => Some(format!("section {key}: unknown fate {fate}")),
        })
        .collect();
    if changes.is_empty() {
        return None;
    }
    let read = outcomes.iter().filter(|outcome| outcome.fate != 4).count();
    let kept = outcomes.iter().filter(|outcome| outcome.fate == 0).count();
    let mut parts = vec![format!("kept {kept} of {read} saved sections")];
    parts.extend(changes);
    Some(parts.join("; "))
}

/// `message`, then what the restore answering `response` did not keep as it
/// was, if anything.
fn with_report(message: &str, response: &[f32]) -> String {
    match describe_restore(restore_report(response).as_deref()) {
        Some(changes) => format!("{message}; {changes}"),
        None => message.to_owned(),
    }
}

/// FNV-1a over a packet's lanes, bit for bit.
pub fn hash_lanes(lanes: &[f32]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for lane in lanes {
        for byte in lane.to_bits().to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    hash
}

impl<B: Backend> GameRuntime<B> {
    pub async fn new(
        engine: &B,
        module: B::Module,
        catalog: Catalog,
        host: &mut dyn Host,
        initial_save: Option<SavedState>,
        initial_scene: Option<SavedState>,
    ) -> Result<GameRuntime<B>> {
        let mut active = open_game(engine, &module).await?;
        if let Some(saved) = &initial_save
            && saved.schema != active.schema
        {
            host.report("quicksave state schema changed; starting fresh");
        }
        match initial_save.filter(|saved| saved.schema == active.schema) {
            Some(saved) => {
                let restored = state_request(&mut active, Operation::Restore, &saved.state).await?;
                host.report(&with_report("resuming the saved state", &restored));
            }
            None => {
                state_request(&mut active, Operation::Create, &[]).await?;
                if let Some(scene) = initial_scene {
                    if scene.schema == active.schema {
                        let loaded = state_request(&mut active, Operation::LoadScene, &scene.state).await?;
                        host.report(&with_report("loaded the saved scene", &loaded));
                    } else {
                        host.report("scene state schema changed; seeding from source");
                    }
                }
            }
        }
        let first = evaluate(&mut active, &catalog, host).await?;
        active.program.accept();
        let mut runtime = GameRuntime {
            engine: engine.clone(),
            module,
            catalog,
            active,
            edits: Edits::default(),
            previous_timestamp: None,
            measuring: false,
            last_timings: None,
            sent: None,
            hashing: false,
            packet_hash: 0,
            received: None,
            drawn: Some(first),
        };
        runtime.record().await?;
        Ok(runtime)
    }

    /// Disabled by default; ordinary frames take no profiling timestamps.
    pub fn measure_frames(&mut self, enabled: bool) {
        self.measuring = enabled;
        self.last_timings = None;
    }

    /// The guest, decode and validate times of the last frame received.
    pub fn last_timings(&self) -> Option<FrameTimings> {
        self.last_timings
    }

    /// Hash every packet received (`packet_hash`).
    pub fn hash_packets(&mut self, enabled: bool) {
        self.hashing = enabled;
    }

    /// The hash of the last packet received, when hashing.
    pub fn packet_hash(&self) -> u64 {
        self.packet_hash
    }

    /// The last frame received, when hashing.
    pub fn received(&self) -> Option<&RenderFrame> {
        self.received.as_ref()
    }

    async fn record(&mut self) -> Result<()> {
        if !self.active.program.scenes() {
            return Ok(());
        }
        let scene: Arc<[f32]> = state_request(&mut self.active, Operation::Scene, &[]).await?.into();
        if let Some(step) = self.edits.step
            && self.edits.scenes[step] == scene
        {
            return Ok(());
        }
        let keep = self.edits.step.map_or(0, |step| step + 1);
        self.edits.scenes.truncate(keep);
        self.edits.scenes.push(scene);
        if self.edits.scenes.len() > HISTORY_LIMIT {
            self.edits.scenes.remove(0);
        }
        self.edits.step = Some(self.edits.scenes.len() - 1);
        Ok(())
    }

    /// Restores are validated in another program instance, so malformed
    /// saves and traps cannot poison the running one. The program draws a
    /// frame to prove it works; `commit` keeps what that frame did to the
    /// state (as loads and undo always have), and without it the next
    /// request starts from exactly the restored state. Also returns what the
    /// restore (or the scene load) answered: what it made of each section.
    async fn replace_state(
        &mut self,
        host: &mut dyn Host,
        state: &[f32],
        scene: Option<&[f32]>,
        show: bool,
        commit: bool,
    ) -> Result<(RenderFrame, Vec<f32>)> {
        let mut candidate = open_game(&self.engine, &self.module).await?;
        let mut answer = state_request(&mut candidate, Operation::Restore, state).await?;
        if let Some(scene) = scene {
            answer = state_request(&mut candidate, Operation::LoadScene, scene).await?;
        }
        let rendered = evaluate(&mut candidate, &self.catalog, host).await?;
        if show {
            host.render(&rendered, &self.catalog)?;
        }
        if commit {
            candidate.program.accept();
        }
        let mut previous = std::mem::replace(&mut self.active, candidate);
        self.previous_timestamp = None;
        previous.program.close().await;
        host.restored();
        Ok((rendered, answer))
    }

    async fn travel(&mut self, host: &mut dyn Host, forward: bool) -> Result<()> {
        let to = match (self.edits.step, forward) {
            (Some(step), false) if step > 0 => step - 1,
            (Some(step), true) if step + 1 < self.edits.scenes.len() => step + 1,
            _ => {
                host.report(if forward { "nothing to redo" } else { "nothing to undo" });
                return Ok(());
            }
        };
        let state = state_request(&mut self.active, Operation::Snapshot, &[]).await?;
        let scene = self.edits.scenes[to].clone();
        self.replace_state(host, &state, Some(&scene), false, true).await?;
        self.edits.step = Some(to);
        let message = if forward {
            format!("redid an edit ({} more to redo)", self.edits.scenes.len() - 1 - to)
        } else {
            format!("undid an edit ({to} more to undo)")
        };
        host.report(&message);
        Ok(())
    }

    /// The input a live frame gives the game: the game time since the last
    /// frame (none after a restore), the viewport and the events.
    pub fn input(&mut self, frame: &Frame) -> Result<Vec<f32>> {
        let elapsed = match self.previous_timestamp {
            Some(previous) => (frame.timestamp - previous).max(0.0),
            None => 0.0,
        };
        self.previous_timestamp = Some(frame.timestamp);
        encode_frame(frame, elapsed, &[])
    }

    /// Where the clock stands, so the next live frame's elapsed time runs
    /// on from it (after a rewind, rather than starting over at zero).
    pub fn continue_clock(&mut self, timestamp: f64) {
        self.previous_timestamp = Some(timestamp);
    }

    /// Starts the game on a frame's input (an `input`, or one recorded
    /// earlier); `receive` takes the result.
    pub fn send(&mut self, input: &[f32]) -> Result<()> {
        self.sent = Some(Instant::now());
        self.active.program.send(Operation::Frame, input)
    }

    /// The frame the game computed for the input sent: decoded, validated
    /// and accepted. The host draws it, then `follow` does what it asked.
    pub async fn receive(&mut self, host: &dyn Host) -> Result<RenderFrame> {
        let waiting = Instant::now();
        let packet = self.active.program.response().await?;
        let returned = Instant::now();
        let next = decode_frame(&packet, self.active.schema, &self.catalog)?;
        ensure!(next.state.is_empty(), "program frame must keep its state inside the guest");
        let decoded = Instant::now();
        host.validate(&next.render)?;
        self.active.program.accept();
        if self.hashing {
            self.packet_hash = hash_lanes(&packet);
            {
                self.received = Some(next.render.clone());
            }
        }
        let answered = self.active.program.answered().unwrap_or(returned);
        self.last_timings = self.measuring.then(|| FrameTimings {
            guest_ms: ms(answered.saturating_duration_since(self.sent.unwrap_or(waiting))),
            wait_ms: ms(returned - waiting),
            decode_ms: ms(decoded - returned),
            validate_ms: ms(decoded.elapsed()),
            packet_bytes: packet.len() * 4,
            ..FrameTimings::default()
        });
        Ok(next.render)
    }

    /// The edit history follows any checkpoint, undo or redo a frame asked for.
    pub async fn follow(&mut self, host: &mut dyn Host, rendered: &RenderFrame) -> Result<()> {
        if rendered.checkpoint_requested {
            self.record().await?;
        }
        if rendered.undo_requested {
            self.travel(host, false).await?;
        }
        if rendered.redo_requested {
            self.travel(host, true).await?;
        }
        Ok(())
    }

    /// Moves the running state into a new build: the new program restores a snapshot and
    /// draws a frame before it replaces the old one.
    pub async fn reload(&mut self, host: &mut dyn Host, module: B::Module, catalog: Catalog) -> ReloadResult {
        let result: Result<ReloadResult> = (async {
            let mut prepared = open_game(&self.engine, &module).await?;
            if prepared.schema != self.active.schema {
                host.report(
                    "Blot reload: the state schema changed; restart the application to initialize the new state",
                );
                return Ok(ReloadResult::Incompatible);
            }
            // The new build keeps what it can of each saved section and says
            // what it migrated or started over.
            let state = state_request(&mut self.active, Operation::Snapshot, &[]).await?;
            let restored = state_request(&mut prepared, Operation::Restore, &state).await?;
            let rendered = evaluate(&mut prepared, &catalog, host).await?;
            host.render(&rendered, &catalog)?;
            prepared.program.accept();
            let mut previous = std::mem::replace(&mut self.active, prepared);
            self.module = module;
            self.catalog = catalog;
            self.previous_timestamp = None;
            self.drawn = Some(rendered);
            previous.program.close().await;
            host.restored();
            host.report(&match describe_restore(restore_report(&restored).as_deref()) {
                Some(changes) => format!("Blot reloaded; {changes}"),
                None => "Blot reloaded; game state preserved".to_owned(),
            });
            Ok(ReloadResult::Activated)
        }).await;
        result.unwrap_or_else(|error| {
            host.report(&format!("Blot reload rejected: {error:#}"));
            ReloadResult::Rejected
        })
    }

    /// The last frame the runtime drew itself (starting, loading,
    /// reloading), once.
    pub fn take_drawn(&mut self) -> Option<RenderFrame> {
        self.drawn.take()
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// The running program's linear memory, in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.active.program.memory_bytes()
    }

    /// The running state, as a save holds it.
    pub async fn snapshot(&mut self) -> Result<SavedState> {
        Ok(SavedState {
            schema: self.active.schema,
            state: state_request(&mut self.active, Operation::Snapshot, &[]).await?,
        })
    }

    /// The running state and edit history, to come back to.
    pub async fn moment(&mut self) -> Result<Moment> {
        Ok(Moment { saved: self.snapshot().await?, edits: self.edits.clone() })
    }

    /// Returns to a moment exactly: the next frame starts from its state,
    /// as the frame after it did. Returns the frame the restored program
    /// drew to prove itself, undrawn.
    pub async fn restore(&mut self, host: &mut dyn Host, moment: &Moment) -> Result<RenderFrame> {
        if moment.saved.schema != self.active.schema {
            bail!("the state schema changed since that moment; it cannot be restored");
        }
        let (rendered, _) = self.replace_state(host, &moment.saved.state, None, false, false).await?;
        self.edits = moment.edits.clone();
        Ok(rendered)
    }

    /// Starts a new program from the running state, as a restore does
    /// (what the game keeps outside its state starts over); the clock runs
    /// on. Replays do this where the run they recorded was restored.
    pub async fn reroot(&mut self, host: &dyn Host) -> Result<()> {
        let timestamp = self.previous_timestamp;
        let moment = self.moment().await?;
        self.restore(&mut Quiet(host), &moment).await?;
        self.previous_timestamp = timestamp;
        Ok(())
    }

    /// Replaces the game with a saved state (a quicksave's).
    pub async fn load(&mut self, host: &mut dyn Host, saved: &SavedState) -> Result<()> {
        if saved.schema != self.active.schema {
            bail!("save file state schema changed; cannot load this quicksave");
        }
        let (rendered, answer) = self.replace_state(host, &saved.state, None, true, true).await?;
        if let Some(changes) = describe_restore(restore_report(&answer).as_deref()) {
            host.report(&format!("loaded the save; {changes}"));
        }
        self.drawn = Some(rendered);
        self.record().await
    }

    pub async fn scene(&mut self) -> Result<SavedState> {
        Ok(SavedState { schema: self.active.schema, state: state_request(&mut self.active, Operation::Scene, &[]).await? })
    }

    /// Replaces the game's scene with a saved one, keeping the rest of its state.
    pub async fn load_scene(&mut self, host: &mut dyn Host, scene: &SavedState) -> Result<()> {
        if scene.schema != self.active.schema {
            bail!("scene state schema changed; cannot load this scene");
        }
        let state = state_request(&mut self.active, Operation::Snapshot, &[]).await?;
        let (rendered, answer) = self.replace_state(host, &state, Some(&scene.state), true, true).await?;
        if let Some(changes) = describe_restore(restore_report(&answer).as_deref()) {
            host.report(&format!("loaded the scene; {changes}"));
        }
        self.drawn = Some(rendered);
        self.record().await
    }

    pub async fn close(&mut self) {
        self.active.program.close().await;
    }
}

#[cfg(test)]
mod restore_report_tests {
    use super::*;

    #[test]
    fn reports_parse_and_describe_what_was_not_kept() {
        assert_eq!(restore_report(&[5678.0, 26.0, 0.0]), None, "a state is no report");
        assert_eq!(restore_report(&[5680.0, 1.0, 100.0]), None, "a cut-off report is none");
        let report = restore_report(&[5680.0, 3.0, 0.0, 0.0, 1.0, 1.0, 100.0, 1.0, 1.0, 2.0, 104.0, 4.0, 1.0, 0.0]).unwrap();
        assert_eq!(report[1], SectionOutcome { key: 100, fate: 1, saved: 1, version: 2 });
        assert_eq!(
            describe_restore(Some(&report)).as_deref(),
            Some("kept 1 of 2 saved sections; migrated section 100 from version 1 to 2; dropped section 104 (no longer saved)")
        );
        assert_eq!(describe_restore(Some(&report[..1])), None, "all kept");
        assert_eq!(with_report("loaded", &[5680.0, 1.0, 106.0, 3.0, 0.0, 1.0]), "loaded; kept 0 of 1 saved sections; started section 106 afresh (the state has none)");
    }
}
