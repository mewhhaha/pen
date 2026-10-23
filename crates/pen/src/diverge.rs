//! `pen diverge RECORDING --against DIR`: plays an input recording on two
//! builds of the game side by side, without rendering, and reports the
//! first frame where they part: the fields of its packet that differ, and
//! the lanes of the state after it that differ, by saved section. A
//! recording replays exactly only on the build that made it (docs/
//! determinism.md); this says where, and how, another build departs from
//! it. It fails when they diverge, so `git bisect run` can find the commit
//! that changed what the game does.
use pen_runtime::state::{Section, StateAdapter};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use pen_protocol::boundary::{read_catalog, whole};
use pen_protocol::{Catalog, RenderFrame, Viewport};

use crate::game::{GameRuntime, Host};
use crate::recording;
use crate::snapshot::SavedState;
use crate::timeline::{self, Recorded, Recording, Timeline};

/// Differing lanes a section lists before it says how many more there are.
const SHOWN_LANES: usize = 8;

/// A host that draws nothing: the frames are only compared.
struct Null(Viewport, Arc<dyn StateAdapter>);

impl Host for Null {
    fn state_adapter(&self) -> Arc<dyn StateAdapter> {
        self.1.clone()
    }

    fn viewport(&self) -> Viewport {
        self.0
    }

    fn validate(&self, frame: &RenderFrame) -> Result<()> {
        pen_render::validate_frame(frame)
    }

    fn render(&mut self, _frame: &RenderFrame, _catalog: &Catalog) -> Result<()> {
        Ok(())
    }

    fn report(&mut self, _message: &str) {}
}

/// A game bundle to replay on: the directory with its `game.wasm` (or
/// `game.cwasm`) and its assets.
pub struct Bundle {
    pub directory: PathBuf,
    pub assets: PathBuf,
}

/// What a frame gave: its packet's hash, the frame, and the state after it.
struct Ran {
    packet: u64,
    frame: RenderFrame,
    state: SavedState,
}

/// One build replaying the recording.
struct Replay {
    runtime: GameRuntime,
    timeline: Timeline,
    host: Null,
}

impl Replay {
    fn start(engine: &wasmtime::Engine, bundle: &Bundle, recording: &Recording, host: Null) -> Result<Replay> {
        let module = crate::load_module(engine, &bundle.directory, true, false)?;
        let catalog = read_catalog(&bundle.assets)?;
        let mut host = host;
        let mut runtime = GameRuntime::new(engine, module, catalog, &mut host, None, None)?;
        runtime.hash_packets(true);
        runtime.keep_received(true);
        runtime.restore(&mut host, &recording.start).context("restoring the recording's start")?;
        let mut timeline = Timeline::default();
        timeline.load(recording.clone());
        Ok(Replay { runtime, timeline, host })
    }

    /// Runs the next recorded frame, as a replay runs it.
    fn step(&mut self) -> Result<Ran> {
        self.timeline.prepare(&mut self.runtime, &self.host)?;
        let recorded = self.timeline.next().cloned().context("the recording has ended")?;
        self.runtime.send(&recorded.input)?;
        let rendered = self.runtime.receive(&self.host)?;
        self.runtime.follow(&mut self.host, &rendered)?;
        for load in &recorded.loads {
            timeline::apply(&mut self.runtime, &mut self.host, load)?;
        }
        let restored = self.timeline.restored_here();
        self.timeline.ran(Recorded { restored, ..recorded });
        let frame = self.runtime.received().cloned().unwrap_or(rendered);
        Ok(Ran { packet: self.runtime.packet_hash(), frame, state: self.runtime.snapshot()? })
    }
}

/// Plays `path` on `this` build and on `reference`, frame by frame, and
/// says where they first differ; fails if they do.
pub fn run(path: &Path, this: &Bundle, reference: &Bundle) -> Result<()> {
    let recording = recording::read(path)?;
    let viewport = recording
        .frames
        .first()
        .and_then(|frame| Some(Viewport { width: whole(frame.input[1])?, height: whole(frame.input[2])? }))
        .unwrap_or(Viewport { width: 960, height: 640 });
    let seconds: f64 = recording.frames.iter().map(Recorded::elapsed).sum();
    eprintln!(
        "diverge: {} frames ({seconds:.1} s of game time) of {} on {} and on {}",
        recording.frames.len(),
        path.display(),
        this.directory.display(),
        reference.directory.display()
    );
    let engine = blot_host::engine()?;
    let mut theirs = Replay::start(
        &engine,
        reference,
        &recording,
        Null(viewport, crate::session::load_state_adapter(&reference.assets)),
    )
    .with_context(|| format!("the reference build ({}) cannot replay it", reference.directory.display()))?;
    let mut ours = match Replay::start(
        &engine,
        this,
        &recording,
        Null(viewport, crate::session::load_state_adapter(&this.assets)),
    ) {
        Ok(replay) => replay,
        Err(error) => bail!("the builds diverge at the start: this build cannot replay the recording: {error:#}"),
    };
    let mut time = 0.0;
    for (position, recorded) in recording.frames.iter().enumerate() {
        time += recorded.elapsed();
        let expected = theirs.step().with_context(|| format!("the reference build failed at frame {position}"))?;
        let at = format!("frame {position} ({time:.2} s in)");
        let got = match ours.step() {
            Ok(got) => got,
            Err(error) => bail!("the builds diverge at {at}: this build failed where the reference ran: {error:#}"),
        };
        let packets = expected.packet != got.packet;
        let states = expected.state != got.state;
        if !packets && !states {
            continue;
        }
        eprintln!("\n{at} differs (the reference's values first, then this build's):");
        if packets {
            eprintln!("its packet:\n{}", indent(&difference(&expected.frame, &got.frame)));
        } else {
            eprintln!("its packet is the same");
        }
        if states {
            eprintln!(
                "the state after it:\n{}",
                indent(
                    &state_difference(&expected.state, &got.state, theirs.host.1.as_ref(), ours.host.1.as_ref())
                        .join("\n")
                )
            );
        } else {
            eprintln!("the state after it is the same");
        }
        bail!("the builds diverge at {at}");
    }
    eprintln!("the builds agree: {} frames gave the same packets and states", recording.frames.len());
    Ok(())
}

fn indent(text: &str) -> String {
    text.lines().map(|line| format!("  {line}")).collect::<Vec<_>>().join("\n")
}

/// The entries of `lines` (a `{:#?}` form) indented `indent` spaces: a
/// struct's fields or a list's elements, each with what is nested in it.
fn entries<'a>(lines: &[&'a str], indent: usize) -> Vec<Vec<&'a str>> {
    let mut found: Vec<Vec<&str>> = Vec::new();
    for &line in lines {
        let depth = line.len() - line.trim_start().len();
        if depth == indent && !line.trim_start().starts_with(['}', ']', ')']) {
            found.push(vec![line]);
        } else if depth > indent
            && let Some(last) = found.last_mut()
        {
            last.push(line);
        }
    }
    found
}

/// Where two `{:#?}` forms first differ: the line in each, after the
/// lines that enclose it.
fn first_difference(was: &[&str], now: &[&str]) -> String {
    let at = was.iter().zip(now).position(|(a, b)| a != b).unwrap_or(was.len().min(now.len()));
    let depth = |line: &str| line.len() - line.trim_start().len();
    let line = |lines: &[&str]| lines.get(at).map_or("(none)", |line| line.trim()).to_owned();
    let mut path = Vec::new();
    let mut inside = was.get(at).map_or(usize::MAX, |line| depth(line));
    for enclosing in was[..at.min(was.len())].iter().rev() {
        if depth(enclosing) < inside {
            inside = depth(enclosing);
            path.push(enclosing.trim().trim_end_matches(['{', '[', '(']).trim());
        }
    }
    path.reverse();
    format!("{} > {}\n  again: {}", path.join(" > "), line(was), line(now))
}

/// How two frames differ: each field that does, and in it the first
/// line that does (for a list, of the first element that does).
pub fn difference(first: &RenderFrame, again: &RenderFrame) -> String {
    let (first, again) = (format!("{first:#?}"), format!("{again:#?}"));
    let (first, again): (Vec<&str>, Vec<&str>) = (first.lines().collect(), again.lines().collect());
    let mut said = Vec::new();
    for (was, now) in entries(&first, 4).iter().zip(entries(&again, 4)) {
        if *was == now {
            continue;
        }
        let name = was[0].trim().split(':').next().unwrap_or_default();
        let (was_items, now_items) = (entries(&was[1..], 8), entries(&now[1..], 8));
        if was_items.len() == now_items.len()
            && let Some(at) = was_items.iter().zip(&now_items).position(|(a, b)| a != b)
        {
            said.push(format!("{name}[{at}]: {}", first_difference(&was_items[at], &now_items[at])));
            continue;
        }
        if was_items.len() != now_items.len() {
            said.push(format!("{name}: {} entries, {} again", was_items.len(), now_items.len()));
        }
        said.push(first_difference(was, &now));
    }
    said.join("\n")
}

/// Differing lanes, named by the adapter when available: a few shown and
/// the rest counted.
fn lane_differences(was: &[f32], now: &[f32], label: impl Fn(usize) -> String) -> String {
    let differing: Vec<usize> =
        (0..was.len().min(now.len())).filter(|&lane| was[lane].to_bits() != now[lane].to_bits()).collect();
    let mut said: Vec<String> = differing
        .iter()
        .take(SHOWN_LANES)
        .map(|&lane| format!("{}: {} -> {}", label(lane), was[lane], now[lane]))
        .collect();
    if differing.len() > SHOWN_LANES {
        said.push(format!("and {} more", differing.len() - SHOWN_LANES));
    }
    if was.len() != now.len() {
        said.push(format!("{} lanes, {} here", was.len(), now.len()));
    }
    said.join("; ")
}

/// How two states differ, a line per section that does (or per lane, for
/// a state not laid out in sections).
fn state_difference(
    was: &SavedState,
    now: &SavedState,
    before: &dyn StateAdapter,
    after: &dyn StateAdapter,
) -> Vec<String> {
    let mut said = Vec::new();
    if was.schema != now.schema {
        said.push(format!("state schema {} -> {}", was.schema, now.schema));
    }
    let (Some(was_sections), Some(now_sections)) = (before.sections(&was.state), after.sections(&now.state)) else {
        said.push(lane_differences(&was.state, &now.state, |lane| format!("lane {lane}")));
        return said;
    };
    for section in &was_sections {
        let Some(other) = now_sections.iter().find(|other| other.key == section.key) else {
            said.push(format!("section {}: only in the reference's", section.key));
            continue;
        };
        if (section.version, section.width) != (other.version, other.width) {
            said.push(format!(
                "section {}: version {} width {} -> version {} width {}",
                section.key, section.version, section.width, other.version, other.width
            ));
        } else if section.lanes != other.lanes {
            let lanes = lane_differences(section.lanes, other.lanes, |lane| before.lane_label(section, lane));
            said.push(format!("section {} (version {}): {lanes}", section.key, section.version));
        }
    }
    for section in now_sections.iter().filter(|section| was_sections.iter().all(|was| was.key != section.key)) {
        said.push(format!("section {}: only in this build's", section.key));
    }
    let order = |sections: &[Section]| sections.iter().map(|section| section.key).collect::<Vec<_>>();
    if said.is_empty() && order(&was_sections) != order(&now_sections) {
        said.push(format!("sections in another order: {:?} -> {:?}", order(&was_sections), order(&now_sections)));
    }
    // Adapters can omit headers or other opaque lanes; never hide a raw difference.
    if said.is_empty() && was.state.iter().map(|v| v.to_bits()).ne(now.state.iter().map(|v| v.to_bits())) {
        said.push(lane_differences(&was.state, &now.state, |lane| format!("lane {lane}")));
    }
    said
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state of `sections`: key, version, width and lanes each.
    fn state(sections: &[(u32, u32, u32, &[f32])]) -> SavedState {
        let mut lanes = vec![5678.0, 26.0, sections.len() as f32];
        for (key, version, width, values) in sections {
            lanes.extend([*key as f32, *version as f32, *width as f32, values.len() as f32]);
            lanes.extend_from_slice(values);
        }
        SavedState { schema: 26, state: lanes }
    }

    fn state_difference(was: &SavedState, now: &SavedState) -> Vec<String> {
        super::state_difference(was, now, &pen_ink::Ink::default(), &pen_ink::Ink::default())
    }

    #[test]
    fn state_differences_name_sections_entities_and_lanes() {
        let was = state(&[(1, 1, 0, &[7.0]), (102, 3, 2, &[1.0, 0.5, 0.25, 1.0, 2.0, 3.0]), (104, 2, 0, &[1.0])]);
        let now = state(&[(1, 1, 0, &[8.0]), (102, 3, 2, &[1.0, 0.5, 0.25, 0.0, 2.0, 4.0]), (110, 1, 0, &[])]);
        assert_eq!(
            state_difference(&was, &now),
            [
                "section 1 (version 1): lane 0: 7 -> 8",
                "section 102 (version 3): entity 1's flag: 1 -> 0; entity 1 lane 1: 3 -> 4",
                "section 104: only in the reference's",
                "section 110: only in this build's",
            ]
        );
        let migrated = state(&[(1, 2, 0, &[7.0])]);
        assert_eq!(
            state_difference(&state(&[(1, 1, 0, &[7.0])]), &migrated),
            ["section 1: version 1 width 0 -> version 2 width 0"]
        );
        // Many lanes: a few, then how many more; and a state of no sections.
        let many: Vec<f32> = (0..20).map(|lane| lane as f32).collect();
        let raw = |lanes: &[f32]| SavedState { schema: 1, state: lanes.to_vec() };
        let said = state_difference(&raw(&[9.0; 20]), &raw(&many));
        assert_eq!(said.len(), 1);
        assert!(said[0].starts_with("lane 0: 9 -> 0; lane 1: 9 -> 1;"), "{said:?}");
        assert!(said[0].ends_with("and 11 more"), "{said:?}");
        assert_eq!(state_difference(&raw(&[1.0]), &raw(&[1.0, 2.0])), ["1 lanes, 2 here"]);
    }

    /// The game itself, where `just export` made it: a recording of a few
    /// seconds of play, written and replayed on the bundle against itself,
    /// agrees frame by frame.
    #[test]
    fn a_build_agrees_with_itself() {
        let Some(directory) = crate::fixtures::bundle() else { return };
        let engine = blot_host::engine().unwrap();
        let module = crate::load_module(&engine, &directory, true, false).unwrap();
        let viewport = Viewport { width: 960, height: 640 };
        let mut host = Null(viewport, crate::session::load_state_adapter(&directory.join("assets")));
        let catalog = read_catalog(&directory.join("assets")).unwrap();
        let mut runtime = GameRuntime::new(&engine, module, catalog, &mut host, None, None).unwrap();
        let start = runtime.moment().unwrap();
        let mut script = crate::script::Script::new(
            crate::script::parse(&["wait=10 key=Tab wait=20 console=3,4 wait=60 click=470,150 wait=20".into()])
                .unwrap(),
        );
        let mut frames = Vec::new();
        for index in 0..150 {
            let events = script.tick(viewport).events;
            let input =
                runtime.input(&pen_protocol::Frame { timestamp: f64::from(index) / 60.0, viewport, events }).unwrap();
            runtime.send(&input).unwrap();
            let rendered = runtime.receive(&host).unwrap();
            runtime.follow(&mut host, &rendered).unwrap();
            frames.push(Recorded { input, ..Recorded::default() });
        }
        runtime.close();
        let path = std::env::temp_dir().join(format!("pen-diverge-{}.penrec", std::process::id()));
        std::fs::write(&path, recording::encode(&Recording { start, frames })).unwrap();
        let bundle = || Bundle { directory: directory.clone(), assets: directory.join("assets") };
        run(&path, &bundle(), &bundle()).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn frames_differ_by_their_fields() {
        let first = RenderFrame::default();
        let mut again = first.clone();
        again.blur = 0.5;
        let said = difference(&first, &again);
        assert!(said.contains("blur: 0.0") && said.contains("again: blur: 0.5"), "{said}");
        assert_eq!(difference(&first, &first), "");
    }
}
