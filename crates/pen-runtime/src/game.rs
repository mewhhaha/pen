//! Runs one long-lived `main(host)`. The program owns its state;
//! snapshots cross the boundary only for persistence, undo history and the
//! moments time travel and replays restore.
//!
//! A frame is sent and received separately, so the guest can compute one
//! frame while the host draws the one before (`Session`'s pipeline).
use std::sync::Arc;
use web_time::Instant;

use anyhow::{Result, bail, ensure};
use pen_protocol::boundary::{decode_frame, decode_into, encode_frame, validate_state, whole};
use pen_protocol::{Catalog, Frame, GameEvent, RenderFrame, Viewport};

use crate::program::{Backend, Operation, Program, Unresponsive};
use crate::saved::{SavedTypes, declared_resets};
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
    /// The saved types the game declares (`saved.json` beside its catalog,
    /// saved.rs), which migrate what its own code cannot. Asked only when a
    /// restore reset a section.
    fn saved_types(&self) -> Option<SavedTypes> {
        None
    }
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

    fn saved_types(&self) -> Option<SavedTypes> {
        self.0.saved_types()
    }
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

/// Output rejected before accepting the guest's candidate state.
#[derive(Debug)]
pub struct InvalidFrame;

impl std::fmt::Display for InvalidFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("invalid game frame")
    }
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

impl FrameTimings {
    /// The timings once the host drew the frame: `submit_ms` drawing,
    /// `total_ms` in all, and what the phases leave unaccounted,
    /// housekeeping.
    pub fn completed(mut self, submit_ms: f64, total_ms: f64) -> FrameTimings {
        self.submit_ms = submit_ms;
        self.total_ms = total_ms;
        self.housekeeping_ms = (total_ms - self.wait_ms - self.decode_ms - self.validate_ms - submit_ms).max(0.0);
        self
    }
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
    /// The last packet's length in lanes.
    packet_lanes: usize,
    /// Keep a copy of the last frame received: tests say how two differ.
    keeping: bool,
    received: Option<RenderFrame>,
    /// A frame the host is done with, for the next to be decoded into.
    spare: Option<RenderFrame>,
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

/// Restores `state` into `game`, a new program of `module`. When the game
/// resets a section for want of a migration and the host's declared saved
/// types give one, the state is migrated (saved.rs) and restored again in
/// another program of `module`, which takes `game`'s place. Returns the
/// game's answer, its report marking the sections the host migrated.
async fn restore<B: Backend>(
    engine: &B,
    module: &B::Module,
    game: &mut Game<B::Program>,
    state: &[f32],
    host: &dyn Host,
) -> Result<Vec<f32>> {
    let answer = state_request(game, Operation::Restore, state).await?;
    let Some(outcomes) = restore_report(&answer) else { return Ok(answer) };
    if outcomes.iter().all(|outcome| outcome.fate != RESET) {
        return Ok(answer);
    }
    let Some(types) = host.saved_types() else { return Ok(answer) };
    let targets = declared_resets(&types, &outcomes);
    if targets.is_empty() {
        return Ok(answer);
    }
    let (migrated, keys) = crate::saved::migrate(state, &types, &targets)?;
    let mut fresh = open_game(engine, module).await?;
    let mut answer = state_request(&mut fresh, Operation::Restore, &migrated).await?;
    let mut replaced = std::mem::replace(game, fresh);
    replaced.program.close().await;
    // The report, from the second restore, says it kept those sections.
    if let Some(records) = answer.get_mut(2..) {
        for record in records.as_chunks_mut::<4>().0 {
            let key = whole(record[0]);
            let original = outcomes.iter().find(|outcome| Some(outcome.key) == key && keys.contains(&outcome.key));
            if let Some(original) = original
                && whole(record[1]) == Some(0)
            {
                record[1] = DECLARED as f32;
                record[2] = original.saved as f32;
            }
        }
    }
    Ok(answer)
}

fn validation_frame(host: &dyn Host) -> Frame {
    Frame { timestamp: 0.0, viewport: host.viewport(), events: Vec::new() }
}

/// A frame for checking a program: it runs, but is not accepted until the
/// caller does so.
async fn evaluate<P: Program>(game: &mut Game<P>, catalog: &Catalog, host: &dyn Host) -> Result<RenderFrame> {
    evaluate_with_frame(game, catalog, host, &validation_frame(host)).await
}

async fn evaluate_with_frame<P: Program>(
    game: &mut Game<P>,
    catalog: &Catalog,
    host: &dyn Host,
    frame: &Frame,
) -> Result<RenderFrame> {
    let packet = game.program.request(Operation::Frame, &encode_frame(frame, 0.0, &[])?).await?;
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
/// the state lacked it, 4 dropped because the build no longer saves it; and
/// the host's own, [`DECLARED`]: migrated by the game's declared saved
/// types.
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
/// A section's fate when the game reset it for want of a migration.
pub const RESET: u32 = 2;
/// A section's fate when the host migrated it as the game's saved types
/// declare (saved.rs), after the game could not.
pub const DECLARED: u32 = 5;

/// The report a restore or scene load answers with: `5680, count`, then
/// `key, fate, saved, version` per section. None for any other answer, as
/// from a game whose restore answers with its state.
pub fn restore_report(response: &[f32]) -> Option<Vec<SectionOutcome>> {
    let [magic, count, rest @ ..] = response else {
        return None;
    };
    if *magic != REPORT_MAGIC || whole(*count).and_then(|count| (count as usize).checked_mul(4)) != Some(rest.len()) {
        return None;
    }
    rest.as_chunks::<4>()
        .0
        .iter()
        .map(|&[key, fate, saved, version]| {
            Some(SectionOutcome {
                key: whole(key)?,
                fate: whole(fate)?,
                saved: whole(saved)?,
                version: whole(version)?,
            })
        })
        .collect()
}

/// What a restore did other than keep each saved section as it was, for the
/// log, or None if it kept them all or did not say.
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
            DECLARED => {
                Some(format!("migrated section {key} from version {saved} to {version} as saved.json declares"))
            }
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
        let initial_save = initial_save.filter(|saved| saved.schema == active.schema);
        let restored = u32::from(initial_save.is_some());
        match initial_save {
            Some(saved) => {
                let restored = restore(engine, &module, &mut active, &saved.state, host).await?;
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
        let mut first_input = validation_frame(host);
        first_input.events.push(GameEvent::SessionStarted { restored });
        let first = evaluate_with_frame(&mut active, &catalog, host, &first_input).await?;
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
            packet_lanes: 0,
            keeping: false,
            received: None,
            spare: None,
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

    /// The last packet received's length in lanes.
    pub fn packet_lanes(&self) -> usize {
        self.packet_lanes
    }

    /// Keep a copy of every frame received (`received`).
    pub fn keep_received(&mut self, enabled: bool) {
        self.keeping = enabled;
        self.received = None;
    }

    /// The last frame received, when keeping them.
    pub fn received(&self) -> Option<&RenderFrame> {
        self.received.as_ref()
    }

    /// Takes back a frame the host is done with: the next one is decoded
    /// into its vectors.
    pub fn recycle(&mut self, frame: RenderFrame) {
        self.spare = Some(frame);
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
        let mut answer = restore(&self.engine, &self.module, &mut candidate, state, host).await?;
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
    /// A program that stopped answering rejects the frame as invalid.
    pub async fn receive(&mut self, host: &dyn Host) -> Result<RenderFrame> {
        let waiting = Instant::now();
        let packet = self
            .active
            .program
            .response()
            .await
            .map_err(|error| if error.is::<Unresponsive>() { error.context(InvalidFrame) } else { error })?;
        let returned = Instant::now();
        let mut frame = self.spare.take().unwrap_or_default();
        let state = decode_into(&packet, self.active.schema, &self.catalog, &mut frame).map_err(|error| {
            let details = match packet.iter().position(|value| !value.is_finite()) {
                Some(lane) => format!("nonfinite value {} at frame lane {lane}", packet[lane]),
                None => format!("decoding a {}-lane game frame", packet.len()),
            };
            error.context(details).context(InvalidFrame)
        })?;
        if !state.is_empty() {
            return Err(anyhow::anyhow!("program frame must keep its state inside the guest").context(InvalidFrame));
        }
        let decoded = Instant::now();
        host.validate(&frame).map_err(|error| error.context(InvalidFrame))?;
        self.active.program.accept();
        if self.hashing {
            self.packet_hash = hash_lanes(&packet);
        }
        self.packet_lanes = packet.len();
        if self.keeping {
            self.received = Some(frame.clone());
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
        self.active.program.recycle(packet.into_buffer());
        Ok(frame)
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
            let restored = restore(&self.engine, &module, &mut prepared, &state, host).await?;
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
        })
        .await;
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

    /// The catalog, for the dev tools to draw an entry from another asset
    /// until the next build.
    pub fn catalog_mut(&mut self) -> &mut Catalog {
        &mut self.catalog
    }

    /// The running program, for a host to look into it.
    pub fn program(&self) -> &B::Program {
        &self.active.program
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
        Ok(SavedState {
            schema: self.active.schema,
            state: state_request(&mut self.active, Operation::Scene, &[]).await?,
        })
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
    fn housekeeping_is_what_the_phases_leave() {
        let timings = FrameTimings { wait_ms: 2.0, decode_ms: 1.0, validate_ms: 0.5, ..FrameTimings::default() };
        let done = timings.completed(3.0, 10.0);
        assert_eq!((done.submit_ms, done.total_ms, done.housekeeping_ms), (3.0, 10.0, 3.5));
        assert_eq!(timings.completed(9.0, 10.0).housekeeping_ms, 0.0);
    }

    #[test]
    fn reports_parse_and_describe_what_was_not_kept() {
        assert_eq!(restore_report(&[5678.0, 26.0, 0.0]), None, "a state is no report");
        assert_eq!(restore_report(&[5680.0, 1.0, 100.0]), None, "a cut-off report is none");
        assert_eq!(restore_report(&[5680.0, 1e30, 100.0]), None, "a count past any report is none");
        assert_eq!(restore_report(&[5680.0, 1.0, 100.0, -1.0, 1.0, 1.0]), None, "so is a fate that is no number");
        let report =
            restore_report(&[5680.0, 3.0, 0.0, 0.0, 1.0, 1.0, 100.0, 1.0, 1.0, 2.0, 104.0, 4.0, 1.0, 0.0]).unwrap();
        assert_eq!(report[1], SectionOutcome { key: 100, fate: 1, saved: 1, version: 2 });
        assert_eq!(
            describe_restore(Some(&report)).as_deref(),
            Some(
                "kept 1 of 2 saved sections; migrated section 100 from version 1 to 2; dropped section 104 (no longer saved)"
            )
        );
        assert_eq!(describe_restore(Some(&report[..1])), None, "all kept");
        assert_eq!(
            with_report("loaded", &[5680.0, 1.0, 106.0, 3.0, 0.0, 1.0]),
            "loaded; kept 0 of 1 saved sections; started section 106 afresh (the state has none)"
        );
    }
}

#[cfg(test)]
mod receive_tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use pen_protocol::boundary::{DRAW_FLOATS, PACKET_HEADER_FLOATS, PACKET_MAGIC, lanes};

    use super::*;
    use crate::program::{Response, Unresponsive, response};

    /// What the scripted program answers, and the buffers it took back.
    #[derive(Default)]
    struct Script {
        answers: Vec<Result<Vec<f32>>>,
        recycled: Vec<Vec<f32>>,
        requests: Vec<(Operation, Vec<f32>)>,
    }

    #[derive(Clone, Default)]
    struct Scripted(Rc<RefCell<Script>>);

    struct Answering(Rc<RefCell<Script>>, Option<Operation>);

    impl Scripted {
        fn answer(&self, operations: &[Operation]) {
            self.0.borrow_mut().answers.extend(operations.iter().map(|operation| {
                if *operation == Operation::Frame {
                    return Ok(frame(0));
                }
                Ok(vec![*operation as u8 as f32])
            }));
        }
    }

    impl Backend for Scripted {
        type Module = ();
        type Program = Answering;
        async fn start(&self, _: &()) -> Result<Answering> {
            Ok(Answering(self.0.clone(), None))
        }
    }

    impl Program for Answering {
        fn schema(&self) -> u32 {
            1
        }
        fn scenes(&self) -> bool {
            false
        }
        fn send(&mut self, operation: Operation, payload: &[f32]) -> Result<()> {
            self.0.borrow_mut().requests.push((operation, payload.to_vec()));
            self.1 = Some(operation);
            Ok(())
        }
        async fn response(&mut self) -> Result<Response> {
            let answer = self.0.borrow_mut().answers.remove(0)?;
            response(self.1.take().unwrap(), answer)
        }
        fn recycle(&mut self, buffer: Vec<f32>) {
            self.0.borrow_mut().recycled.push(buffer);
        }
        fn accept(&mut self) {}
        fn answered(&self) -> Option<Instant> {
            None
        }
        fn memory_bytes(&self) -> usize {
            0
        }
        async fn close(&mut self) {}
    }

    struct Viewer;

    impl Host for Viewer {
        fn viewport(&self) -> Viewport {
            Viewport { width: 64, height: 64 }
        }
        fn validate(&self, _: &RenderFrame) -> Result<()> {
            Ok(())
        }
        fn render(&mut self, _: &RenderFrame, _: &Catalog) -> Result<()> {
            Ok(())
        }
        fn report(&mut self, _: &str) {}
    }

    /// The runtime's futures never wait on a scripted program.
    fn ready<T>(future: impl Future<Output = T>) -> T {
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match std::pin::pin!(future).poll(&mut context) {
            std::task::Poll::Ready(value) => value,
            std::task::Poll::Pending => unreachable!("a scripted program answers at once"),
        }
    }

    /// A Frame response: its echoed operation, then a packet of `draws` draws.
    fn frame(draws: usize) -> Vec<f32> {
        let mut packet = vec![0.0; 1 + PACKET_HEADER_FLOATS];
        packet[0] = Operation::Frame as u8 as f32;
        packet[1 + lanes::header::MAGIC] = PACKET_MAGIC;
        packet[1 + lanes::header::SCHEMA] = 1.0;
        packet[1 + lanes::header::DRAW_COUNT] = draws as f32;
        packet.extend(std::iter::repeat_n(0.0, draws * DRAW_FLOATS));
        packet
    }

    #[test]
    fn frames_decode_into_recycled_ones_and_stopped_programs_reject_them() {
        let backend = Scripted::default();
        let catalog = Catalog { meshes: vec!["m".into()], materials: vec!["x".into()], sounds: Vec::new() };
        // Create, then the frame that proves the program works.
        backend.0.borrow_mut().answers = vec![Ok(vec![Operation::Create as u8 as f32]), Ok(frame(0))];
        let mut runtime = ready(GameRuntime::new(&backend, (), catalog, &mut Viewer, None, None)).unwrap();

        let mut shown = RenderFrame::default();
        shown.draws.reserve(64);
        let reserved = shown.draws.capacity();
        runtime.recycle(shown);
        backend.0.borrow_mut().answers.push(Ok(frame(2)));
        runtime.send(&[]).unwrap();
        let received = ready(runtime.receive(&Viewer)).unwrap();
        assert_eq!(received.draws.len(), 2);
        assert_eq!(received.draws.capacity(), reserved, "decoded into the recycled frame");
        let recycled = backend.0.borrow().recycled.last().map(Vec::len);
        assert_eq!(recycled, Some(frame(2).len()), "the packet's buffer goes back, echoed operation and all");

        backend.0.borrow_mut().answers.push(Err(anyhow::anyhow!("trapped").context(Unresponsive)));
        runtime.send(&[]).unwrap();
        let error = ready(runtime.receive(&Viewer)).unwrap_err();
        assert!(error.is::<InvalidFrame>() && error.is::<Unresponsive>(), "{error:#}");
        backend.0.borrow_mut().answers.push(Err(anyhow::anyhow!("trapped")));
        runtime.send(&[]).unwrap();
        assert!(!ready(runtime.receive(&Viewer)).unwrap_err().is::<InvalidFrame>(), "other failures stay fatal");
    }

    #[test]
    fn session_started_follows_initial_restore_or_creation_and_scene_setup() {
        use Operation::{Create, Frame, LoadScene, Restore};
        let cases = [
            (None, None, &[Create, Frame][..], 0),
            (None, Some(1), &[Create, LoadScene, Frame], 0),
            (None, Some(2), &[Create, Frame], 0),
            (Some(1), Some(1), &[Restore, Frame], 1),
            (Some(2), Some(1), &[Create, LoadScene, Frame], 0),
        ];
        for (save_schema, scene_schema, operations, restored) in cases {
            let backend = Scripted::default();
            backend.answer(operations);
            let saved = |schema| SavedState { schema, state: vec![42.0] };
            let runtime = ready(GameRuntime::new(
                &backend,
                (),
                Catalog::default(),
                &mut Viewer,
                save_schema.map(saved),
                scene_schema.map(saved),
            ))
            .unwrap();
            let script = backend.0.borrow();
            assert_eq!(script.requests.iter().map(|(operation, _)| *operation).collect::<Vec<_>>(), operations);
            assert_eq!(
                script.requests.last().unwrap().1,
                [0.0, 64.0, 64.0, 1.0, 0.0, 27.0, restored as f32, 0.0, 0.0, 0.0],
                "save {save_schema:?}, scene {scene_schema:?}"
            );
            assert!(runtime.drawn.is_some(), "startup evaluation was accepted");
        }
    }

    #[test]
    fn session_started_is_absent_from_live_input_recordings_and_state_replacements() {
        use crate::timeline::{Recorded, Timeline};
        use Operation::{Create, Frame, LoadScene, Restore, Snapshot};
        let backend = Scripted::default();
        backend.answer(&[Create, Frame]);
        let catalog = Catalog::default();
        let mut runtime = ready(GameRuntime::new(&backend, (), catalog.clone(), &mut Viewer, None, None)).unwrap();
        backend.0.borrow_mut().requests.clear();
        let moment = Moment { saved: SavedState { schema: 1, state: vec![42.0] }, edits: Edits::default() };

        backend.answer(&[Snapshot, Restore, Frame]);
        assert_eq!(ready(runtime.reload(&mut Viewer, (), catalog)), ReloadResult::Activated);
        backend.answer(&[Restore, Frame]);
        ready(runtime.load(&mut Viewer, &moment.saved)).unwrap(); // F9 takes this path too.
        backend.answer(&[Snapshot, Restore, LoadScene, Frame]);
        ready(runtime.load_scene(&mut Viewer, &moment.saved)).unwrap();
        backend.answer(&[Restore, Frame]);
        ready(runtime.restore(&mut Viewer, &moment)).unwrap();
        backend.answer(&[Snapshot, Restore, Frame]);
        ready(runtime.reroot(&Viewer)).unwrap();

        let mut timeline = Timeline::default();
        timeline.keep_history(None);
        timeline.keep_moment(moment);
        let input = runtime.input(&validation_frame(&Viewer)).unwrap();
        backend.answer(&[Frame]);
        runtime.send(&input).unwrap();
        ready(runtime.receive(&Viewer)).unwrap();
        timeline.ran(Recorded { input, ..Recorded::default() });
        assert_eq!(timeline.recording().unwrap().frames[0].input, [0.0, 64.0, 64.0, 0.0, 0.0]);

        backend.answer(&[Restore, Frame, Frame]);
        ready(timeline.rewind(&mut runtime, &Viewer, 1, &mut |_, _, _| Ok(()))).unwrap();
        backend.answer(&[Restore, Frame]);
        ready(timeline.restart(&mut runtime, &Viewer, 0)).unwrap();

        let script = backend.0.borrow();
        assert!(script.answers.is_empty(), "all lifecycle evaluations ran");
        let frames = script.requests.iter().filter(|(operation, _)| *operation == Frame).collect::<Vec<_>>();
        assert_eq!(frames.len(), 9, "reload, load, scene, restore, reroot, live, rewind validation/replay, restart");
        for (_, input) in frames {
            assert_eq!(input, &[0.0, 64.0, 64.0, 0.0, 0.0], "only startup carries SessionStarted");
        }
    }
}

#[cfg(test)]
mod saved_types_tests {
    use pen_protocol::boundary::{PACKET_HEADER_FLOATS, PACKET_MAGIC, lanes};

    use super::*;
    use crate::program::{Response, response};
    use crate::saved::STATE_MAGIC;

    /// A game saved as ink's persist saves one: the entities, and section
    /// 100, `points` (version 1) or `points, bonus` (version 2, which has no
    /// migration from 1). Each module is a build: its version.
    #[derive(Clone)]
    struct Builds;

    struct Persisting {
        version: u32,
        state: Vec<f32>,
        answer: Option<(Operation, Vec<f32>)>,
    }

    fn state(version: u32, record: &[f32]) -> Vec<f32> {
        let mut state = vec![STATE_MAGIC, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        state.extend([100.0, version as f32, 0.0, record.len() as f32]);
        state.extend_from_slice(record);
        state
    }

    impl Backend for Builds {
        type Module = u32;
        type Program = Persisting;
        async fn start(&self, version: &u32) -> Result<Persisting> {
            Ok(Persisting { version: *version, state: Vec::new(), answer: None })
        }
    }

    impl Program for Persisting {
        fn schema(&self) -> u32 {
            1
        }
        fn scenes(&self) -> bool {
            false
        }
        fn send(&mut self, operation: Operation, payload: &[f32]) -> Result<()> {
            let lanes = match operation {
                Operation::Create => {
                    self.state = if self.version == 1 { state(1, &[42.0]) } else { state(2, &[0.0, 0.25]) };
                    Vec::new()
                }
                Operation::Snapshot => self.state.clone(),
                Operation::Restore => {
                    let saved = whole(payload[9]).unwrap();
                    let fate = if saved == self.version { 0.0 } else { 2.0 };
                    self.state = if fate == 0.0 { payload.to_vec() } else { state(2, &[0.0, 0.25]) };
                    vec![5680.0, 2.0, 0.0, 0.0, 1.0, 1.0, 100.0, fate, saved as f32, self.version as f32]
                }
                _ => {
                    let mut packet = vec![0.0; PACKET_HEADER_FLOATS];
                    packet[lanes::header::MAGIC] = PACKET_MAGIC;
                    packet[lanes::header::SCHEMA] = 1.0;
                    packet
                }
            };
            self.answer = Some((operation, [&[operation as u8 as f32][..], &lanes].concat()));
            Ok(())
        }
        async fn response(&mut self) -> Result<Response> {
            let (operation, answer) = self.answer.take().unwrap();
            response(operation, answer)
        }
        fn accept(&mut self) {}
        fn answered(&self) -> Option<Instant> {
            None
        }
        fn memory_bytes(&self) -> usize {
            0
        }
        async fn close(&mut self) {}
    }

    struct Viewer {
        declared: Option<SavedTypes>,
        reports: Vec<String>,
    }

    impl Host for Viewer {
        fn viewport(&self) -> Viewport {
            Viewport { width: 64, height: 64 }
        }
        fn validate(&self, _: &RenderFrame) -> Result<()> {
            Ok(())
        }
        fn render(&mut self, _: &RenderFrame, _: &Catalog) -> Result<()> {
            Ok(())
        }
        fn report(&mut self, message: &str) {
            self.reports.push(message.to_owned());
        }
        fn saved_types(&self) -> Option<SavedTypes> {
            self.declared.clone()
        }
    }

    fn ready<T>(future: impl Future<Output = T>) -> T {
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match std::pin::pin!(future).poll(&mut context) {
            std::task::Poll::Ready(value) => value,
            std::task::Poll::Pending => unreachable!("these programs answer at once"),
        }
    }

    const DECLARED: &str = r#"{"types": {"score": {"key": 100, "kind": "resource", "versions": {
        "1": [{"name": "points", "type": "f32"}],
        "2": [{"name": "points", "type": "f32"}, {"name": "bonus", "type": "f32", "default": 0.5}]}}}}"#;

    fn reload(declared: Option<SavedTypes>) -> (Viewer, Vec<f32>) {
        let catalog = Catalog { meshes: vec!["m".into()], materials: vec!["x".into()], sounds: Vec::new() };
        let mut viewer = Viewer { declared, reports: Vec::new() };
        let builds = Builds;
        let mut runtime = ready(GameRuntime::new(&builds, 1, catalog.clone(), &mut viewer, None, None)).unwrap();
        assert_eq!(ready(runtime.reload(&mut viewer, 2, catalog)), ReloadResult::Activated);
        let saved = ready(runtime.snapshot()).unwrap();
        (viewer, saved.state)
    }

    #[test]
    fn reloads_migrate_what_the_game_cannot_as_declared() {
        let (viewer, state) = reload(Some(SavedTypes::parse(DECLARED).unwrap()));
        assert_eq!(state[8..], [100.0, 2.0, 0.0, 2.0, 42.0, 0.5], "the points kept, the bonus its default");
        let reported = viewer.reports.last().unwrap();
        assert!(reported.contains("migrated section 100 from version 1 to 2 as saved.json declares"), "{reported}");
        // Without the declaration the game resets the section.
        let (viewer, state) = reload(None);
        assert_eq!(state[8..], [100.0, 2.0, 0.0, 2.0, 0.0, 0.25]);
        assert!(viewer.reports.last().unwrap().contains("reset section 100"), "{:?}", viewer.reports);
    }
}
