//! The determinism contract on the game itself (`dist/gdev`, from `just
//! export`), without a GPU: the same input gives the same packets, replays
//! give the packets of the run they recorded (rewinds and all), and every
//! frame a rewind runs again gives the packets and state it gave the first
//! time, whatever the restored moment held: menus and the keyboard focus
//! open, panels easing in, a button shaking, and service requests in flight to a host
//! that answers them. Skipped when the bundle is missing (`fixtures`).
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "frame positions as indices, and the state's lanes"
)]
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

use pen_protocol::boundary::read_catalog;
use pen_protocol::{Catalog, Frame, UiNode, Viewport, service};

use super::*;
use crate::diverge::difference;
use crate::fixtures::bundle;
use crate::game::hash_lanes;
use crate::input::Input;
use crate::script::{self, Script};
use crate::services::{Answer, Request, Service, Services};
use crate::snapshot;

mod input;

const VIEWPORT: Viewport = Viewport { width: 960, height: 640 };

struct Null;

impl Host for Null {
    fn viewport(&self) -> Viewport {
        VIEWPORT
    }

    fn validate(&self, frame: &RenderFrame) -> Result<()> {
        pen_render::validate_frame(frame)
    }

    fn render(&mut self, _frame: &RenderFrame, _catalog: &Catalog) -> Result<()> {
        Ok(())
    }

    fn report(&mut self, _message: &str) {}
}

/// Answers every request of its service with the text "fake", `delay`
/// frames after it was asked (as a dialog closing would), so requests
/// wait for their answers across moments.
struct Late {
    code: u32,
    delay: u32,
    frame: u32,
    /// The frame each answer is due, and its request.
    due: VecDeque<(u32, u32)>,
}

impl Service<Input> for Late {
    fn code(&self) -> u32 {
        self.code
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        self.due.push_back((self.frame + self.delay, request.id));
        None
    }

    fn poll(&mut self) -> Vec<(u32, Answer)> {
        let mut ready = Vec::new();
        while self.due.front().is_some_and(|(due, _)| *due <= self.frame) {
            let (_, id) = self.due.pop_front().unwrap();
            ready.push((id, Answer::text("fake")));
        }
        ready
    }

    fn frame(&mut self, _input: &mut Input) {
        self.frame += 1;
    }
}

/// A host whose every service answers `delay` frames late.
fn late_services(delay: u32) -> Services {
    let mut services = Services::new();
    for code in [service::STEAM, service::CLIPBOARD, service::URL, service::PLATFORM, service::PATHFINDING] {
        services.register(Box::new(Late { code, delay, frame: 0, due: VecDeque::new() }));
    }
    services
}

/// What each position's frame gave when it last ran: its packet's hash,
/// the frame itself (to say how two differ) and the state after it.
#[derive(Clone, Default)]
struct Pass {
    packets: Vec<u64>,
    frames: Vec<RenderFrame>,
    states: Vec<u64>,
}

fn keep<T>(list: &mut Vec<T>, position: u64, value: T) {
    list.truncate(position as usize);
    list.push(value);
}

impl Pass {
    fn keep(&mut self, position: u64, packet: u64, frame: RenderFrame) {
        keep(&mut self.packets, position, packet);
        keep(&mut self.frames, position, frame);
    }

    /// Panics where this pass first differs from `first`, saying how
    /// the frames differ.
    fn assert_same(&self, first: &Pass, context: &str) {
        assert_eq!(self.packets.len(), first.packets.len(), "{context}: the run ends at another frame");
        if let Some(position) = (0..first.packets.len()).find(|&index| self.packets[index] != first.packets[index]) {
            panic!(
                "{context}: frame {position} gives other packets than the first time\n{}",
                difference(&first.frames[position], &self.frames[position])
            );
        }
        if let Some(position) = (0..first.states.len()).find(|&index| self.states.get(index) != first.states.get(index))
        {
            panic!("{context}: the state after frame {position} differs from the first time");
        }
    }
}

/// A game as a session runs it, minus the GPU: frames from a script,
/// quicksaves kept in memory, a timeline keeping everything, and
/// perhaps services answering live frames' requests.
struct Game {
    runtime: GameRuntime,
    module: blot_host::GuestModule,
    timeline: Timeline,
    /// Quicksaves by name ("" for the game's own F6 and F9).
    saves: HashMap<PathBuf, SavedState>,
    services: Option<Services>,
    pass: Pass,
}

impl Game {
    fn start(directory: &Path, start: Option<&Moment>) -> Game {
        let engine = blot_host::engine().unwrap();
        let wasm = std::fs::read(directory.join("game.wasm")).unwrap();
        let cache = std::env::temp_dir().join("pen-test-wasm");
        let module = blot_host::GuestModule::new(&engine, &wasm, Some(&cache)).unwrap();
        let catalog = read_catalog(&directory.join("assets")).unwrap();
        let scene = snapshot::read(&directory.join("main.scene")).ok();
        let mut runtime = GameRuntime::new(&engine, module.clone(), catalog, &mut Null, None, scene).unwrap();
        runtime.hash_packets(true);
        let mut timeline = Timeline::default();
        timeline.keep_history(None);
        if let Some(start) = start {
            timeline.load(Recording { start: start.clone(), frames: Vec::new() });
            runtime.restore(&mut Null, start).unwrap();
        }
        Game { runtime, module, timeline, saves: HashMap::new(), services: None, pass: Pass::default() }
    }

    /// Notes what the frame at `position` gave, and every tenth frame
    /// the state after it (a snapshot takes as long as the frame).
    fn note(&mut self, position: u64) {
        let state = if position % 10 == 9 { hash_lanes(&self.runtime.snapshot().unwrap().state) } else { 0 };
        let frame = self.runtime.received().unwrap().clone();
        self.pass.keep(position, self.runtime.packet_hash(), frame);
        keep(&mut self.pass.states, position, state);
    }

    /// Runs one frame of input, live or recorded, with its loads.
    fn run(&mut self, input: Vec<f32>, saves: &[PathBuf], loads: &[PathBuf]) {
        self.timeline.prepare(&mut self.runtime, &Null).unwrap();
        let replayed = self.timeline.next().cloned();
        let input = replayed.as_ref().map_or(input, |recorded| recorded.input.clone());
        self.runtime.send(&input).unwrap();
        let rendered = self.runtime.receive(&Null).unwrap();
        self.runtime.follow(&mut Null, &rendered).unwrap();
        let mut done = Vec::new();
        match &replayed {
            Some(recorded) => {
                for load in &recorded.loads {
                    apply(&mut self.runtime, &mut Null, load).unwrap();
                }
                done.clone_from(&recorded.loads);
            }
            None => {
                // Only live frames ask the services.
                if let Some(services) = &mut self.services {
                    services.answer(&rendered.services);
                }
                let mut saving: Vec<PathBuf> = saves.to_vec();
                let mut loading: Vec<PathBuf> = loads.to_vec();
                if rendered.save_requested {
                    saving.insert(0, PathBuf::new());
                }
                if rendered.load_requested {
                    loading.insert(0, PathBuf::new());
                }
                for path in saving {
                    self.saves.insert(path, self.runtime.snapshot().unwrap());
                }
                for path in loading {
                    let saved = self.saves[&path].clone();
                    self.runtime.load(&mut Null, &saved).unwrap();
                    if let Some(services) = &mut self.services {
                        services.forget();
                    }
                    done.push(Load::Save(saved));
                }
            }
        }
        let position = self.timeline.position();
        self.note(position);
        let restored = self.timeline.restored_here();
        self.timeline.ran(Recorded { input, loads: done, restored });
    }

    /// Plays a script on a steady 60 Hz clock, the services' answers
    /// coming in with the frame after.
    fn play(&mut self, steps: &str) {
        let mut script = Script::new(script::parse(&[steps.to_owned()]).unwrap());
        let mut time = 0.0;
        while !script.done() {
            let tick = script.tick(VIEWPORT);
            let mut events = tick.events;
            if let Some(services) = &mut self.services {
                services.frame(&mut Input::default());
                events.extend(services.take_answers());
            }
            let frame = Frame { timestamp: time, viewport: VIEWPORT, events };
            time += 1.0 / 60.0;
            let input = self.runtime.input(&frame).unwrap();
            self.run(input, &tick.saves, &tick.loads);
        }
    }

    /// Goes to `target`, noting what the frames run unseen gave.
    fn rewind(&mut self, target: u64) {
        let mut unseen = Vec::new();
        let mut note = |position: u64, _: &Recorded, runtime: &Runtime| {
            unseen.push((position, runtime.packet_hash(), runtime.received().unwrap().clone()));
            Ok(())
        };
        self.timeline.rewind(&mut self.runtime, &Null, target, &mut note).unwrap();
        if let Some(services) = &mut self.services {
            services.forget();
        }
        for (position, packet, frame) in unseen {
            self.pass.keep(position, packet, frame);
        }
        assert_eq!(self.timeline.position(), target);
    }

    /// Runs recorded frames again up to the end of the record.
    fn catch_up(&mut self) {
        while self.timeline.next().is_some() {
            self.run(Vec::new(), &[], &[]);
        }
    }

    /// Rewinds to `target`, runs the record again to its end, and
    /// checks every frame gave what it did the first time.
    fn travel(&mut self, target: u64, first: &Pass) {
        self.rewind(target);
        self.catch_up();
        self.pass.assert_same(first, &format!("after a rewind to {target}"));
    }

    fn recording(&self) -> Recording {
        self.timeline.recording().unwrap()
    }
}

/// The editor and play, edits (undo history), undo and redo, and
/// quicksaves and loads from the script and the game's own keys.
const STEPS: &str = "wait=30 key=Tab wait=40 click=470,150 wait=20 click=470,152 wait=20 \
    key=Tab wait=30 drag=300,500,600,450 wait=20 drag=200,400,500,350 wait=20 \
    key=KeyU wait=10 key=KeyU wait=10 down=Shift key=KeyU up=Shift wait=10 \
    save=a wait=20 key=F6 wait=20 drag=300,300,400,450 wait=20 load=a wait=20 \
    drag=250,450,450,350 wait=20 key=F9 wait=20 key=Tab wait=30 wheel=-300,480,320 wait=30";

/// Play and the interface, laid out so a moment (every 30 frames)
/// falls in each of these, which `rewinds_into_the_interface_...`
/// checks: the build bar easing in as play starts (frame 20, moment
/// 30); the keyboard focus on the build bar's roll (F10 at 51: moments
/// 60 and 90); a plan in hand and its route asked of pathfinding (104),
/// answered after moment 120; the station's hire button refused for
/// want of salt and shaking (116: 120), the build bar open with the
/// god's hand swaying after the pointer's last move (148: 150), and the
/// bar closing (178: 180); the pause menu flying up (235: 240),
/// its focus moved to the second row (256: 270), its Settings sliding
/// in (295: 300), a preference changed (316: 330) and the menu flying
/// back down (352: 360).
const INTERFACE: &str = "wait=20 key=Tab wait=30 key=F10 wait=40 key=Escape \
    ui=0x20001,activate wait=10 ui=0x20010,activate move=480,420 wait=11 \
    ui=0x30000,activate wait=30 move=500,400 wait=29 ui=0x30000,collapse ui=0x20001,collapse wait=55 \
    key=Escape wait=20 key=ArrowDown wait=38 key=Enter wait=20 key=ArrowRight wait=18 \
    key=Escape wait=16 key=Escape wait=60";

/// Play with a crowd: twelve gnorps hired, salt knocked out of the rock
/// for them to haul, a gust sent, a building placed for them to raise
/// (its route asked of pathfinding first), the pointer left over them.
const CROWD: &str = "wait=10 key=Tab wait=20 console=3,12 wait=30 \
    click=470,150 wait=15 click=470,152 wait=15 click=471,150 wait=15 click=470,151 wait=30 \
    console=6 wait=30 ui=0x20001,activate wait=10 ui=0x20010,activate move=680,140 wait=20 \
    click=680,140 wait=10 ui=0x20001,collapse move=480,330 wait=480";

/// Plays a recording in a new program; returns its packets.
fn replay(directory: &Path, recording: Recording) -> Vec<u64> {
    let mut game = Game::start(directory, Some(&recording.start));
    game.timeline.load(recording);
    game.catch_up();
    game.runtime.close();
    game.pass.packets
}

/// The lanes of the section of `state` with `key` (packages/persist.blot).
fn section(state: &[f32], key: u32) -> Option<Vec<f32>> {
    let mut at = 3;
    for _ in 0..state[2] as usize {
        let length = state[at + 3] as usize;
        if state[at] as u32 == key {
            return Some(state[at + 4..at + 4 + length].to_vec());
        }
        at += 4 + length;
    }
    None
}

fn node(frame: &RenderFrame, id: u32) -> Option<&UiNode> {
    frame.ui.iter().find(|node| node.id == id)
}

#[test]
fn the_same_input_gives_the_same_packets() {
    let Some(directory) = bundle() else { return };
    let mut first = Game::start(&directory, None);
    first.play(STEPS);
    let mut second = Game::start(&directory, None);
    second.play(STEPS);
    let packets = &first.pass.packets;
    assert!(packets.len() > 400);
    assert_eq!(*packets, second.pass.packets);
    // Packets differ from frame to frame, so equal runs are not trivially equal.
    assert!(packets.windows(2).filter(|pair| pair[0] != pair[1]).count() > packets.len() / 2);
    // A recording of it replays the same packets.
    assert_eq!(replay(&directory, first.recording()), *packets);
    first.runtime.close();
    second.runtime.close();
}

#[test]
fn the_last_seconds_replay_on_a_new_build() {
    let Some(directory) = bundle() else { return };
    let mut game = Game::start(&directory, None);
    game.play(STEPS);
    let first = game.pass.clone();
    // A hot reload (of the same build here, so the frames must match).
    let catalog = game.runtime.catalog().clone();
    let module = game.module.clone();
    assert_eq!(game.runtime.reload(&mut Null, module, catalog), crate::game::ReloadResult::Activated);
    game.timeline.rebuilt();
    let end = game.timeline.position();
    let from = game.timeline.seconds_back(2.0);
    let (_, at) = game.timeline.restart(&mut game.runtime, &Null, from).unwrap();
    assert!(at <= from && end - at >= 120, "{at}");
    // The moments after it came from the old build: they go, and come
    // back from the new one as the frames replay.
    assert_eq!(game.timeline.moments().last(), Some(at));
    game.catch_up();
    game.pass.assert_same(&first, "replaying the last seconds after a reload");
    assert!(game.timeline.moments().filter(|position| *position > at).count() >= 3);
    game.runtime.close();
}

#[test]
fn rewinds_give_the_first_packets_and_replay_exactly() {
    let Some(directory) = bundle() else { return };
    let mut game = Game::start(&directory, None);
    game.play(STEPS);
    let first = game.pass.clone();
    let end = game.timeline.position();
    // Into play (frames 30 to 160 play; 110 is between moments), over
    // the undo and loads, to the start and to the last frame.
    for target in [110, end - 200, end - 75, 1, end - 1] {
        game.travel(target, &first);
    }
    // The run with its rewinds, recorded, replays its packets exactly:
    // restores where it was restored included.
    let recording = game.recording();
    assert!(recording.frames.iter().filter(|recorded| recorded.restored).count() >= 4);
    assert_eq!(replay(&directory, recording), game.pass.packets);
    game.runtime.close();
}

#[test]
fn rewinds_into_the_interface_give_the_first_packets() {
    let Some(directory) = bundle() else { return };
    let mut game = Game::start(&directory, None);
    game.services = Some(late_services(40));
    game.play(INTERFACE);
    let first = game.pass.clone();
    let end = game.timeline.position();
    assert_eq!(end, 413);

    // The moments fall where INTERFACE says: each holds what the frame
    // before it left.
    let frames = &first.frames;
    let focused = |at: usize, id: u32| node(&frames[at], id).is_some_and(|node| node.focused);
    let (bar, roll, plan) = (0x20000, 0x20001, 0x20010);
    let (paused, settings, second_row) = (0x40001, 0x40002, 0x40101);
    // Answers come 40 frames late: the game's first requests (what it
    // runs on, Steam's stat and presence) wait over moment 30, the
    // route's over 120.
    assert!(frames[..30].iter().any(|frame| frame.services.len() >= 4));
    let route = frames[100..120]
        .iter()
        .position(|frame| frame.services.iter().any(|request| request.service == service::PATHFINDING));
    assert!(route.is_some_and(|at| at + 100 + 40 > 120), "a route asked before 120, answered after");
    let y = |at: usize| node(&frames[at], bar).map(|node| node.y);
    assert!(y(29).is_some() && y(29) != y(30), "the build bar eases in over moment 30");
    assert!(focused(59, roll) && focused(89, roll), "the focus is on the build bar over 60 and 90");
    let catalog = game.runtime.catalog().clone();
    let button_x = |at: usize| {
        let mut sprites = frames[at].sprites.iter();
        sprites
            .find(|sprite| catalog.material(sprite.material) == "station_button.material.json")
            .map(|sprite| sprite.x)
    };
    assert!(
        button_x(116).is_some() && button_x(119) != button_x(116) && button_x(149) == button_x(116),
        "the station's button shakes, refused, over 120"
    );
    let width = |at: usize| node(&frames[at], bar).map(|node| node.width);
    assert!(
        [119, 149, 179].iter().all(|&at| width(at) > Some(400.0))
            && width(185) < width(179)
            && width(200) < Some(100.0),
        "the build bar is open over 120 and 150 and closes over 180"
    );
    assert!(node(&frames[119], plan).is_some() && node(&frames[179], plan).is_none(), "its plans go with it");
    assert!(node(&frames[239], paused).is_some() && frames[239].blur > 0.0, "the menu flies up over 240");
    assert!(focused(269, second_row), "its second row has the focus over 270");
    assert!(node(&frames[299], settings).is_some(), "its settings show over 300");
    assert!(frames[316..330].iter().any(|frame| frame.preferences.is_some()), "a preference changed before 330");
    assert!(node(&frames[359], paused).is_none() && frames[359].blur > 0.0, "it flies down over 360");

    // A rewind restores the moment before its target: every moment's.
    for moment in (MOMENT_INTERVAL..end).step_by(MOMENT_INTERVAL as usize) {
        game.travel(moment + 1, &first);
    }
    let recording = game.recording();
    assert_eq!(replay(&directory, recording), game.pass.packets);
    game.runtime.close();
}

#[test]
fn rewinds_among_the_crowd_give_the_first_packets() {
    let Some(directory) = bundle() else { return };
    let mut game = Game::start(&directory, None);
    game.services = Some(late_services(40));
    game.play(CROWD);
    let first = game.pass.clone();
    let end = game.timeline.position();
    let catalog = game.runtime.catalog().clone();
    let mesh =
        |at: usize, name: &str| first.frames[at].draws.iter().filter(|draw| catalog.mesh(draw.mesh) == name).count();
    // A building going up draws one of its stage meshes
    // (`building_{kind}_stage{1..5}.mesh.json`) until it is finished.
    let staged = |at: usize| first.frames[at].draws.iter().any(|draw| catalog.mesh(draw.mesh).contains("_stage"));
    let bubble = |at: usize| {
        first.frames[at].sprites.iter().any(|sprite| catalog.material(sprite.material) == "bubble.material.json")
    };
    // What the frames show: the crowd, a gust, a building going up and
    // gnorps speaking, and a route asked.
    assert!((60..end as usize).all(|at| mesh(at, "robot3d.mesh.json") == 12), "twelve gnorps");
    assert!((150..300).any(|at| mesh(at, "wind_strip.mesh.json") > 0), "a gust");
    assert!(staged(end as usize - 1), "a building going up");
    assert!((0..end as usize).filter(|&at| bubble(at)).count() > 60, "gnorps speaking");
    assert!(
        first.frames.iter().any(|frame| frame.services.iter().any(|request| request.service == service::PATHFINDING))
    );
    let state = game.runtime.snapshot().unwrap().state;
    let builders = section(&state, 116).unwrap();
    assert!(builders[0] >= 1.0, "gnorps punching away at it: {builders:?}");

    // Every third moment's: the frames are dearer than the interface's.
    for moment in (MOMENT_INTERVAL..end).step_by(3 * MOMENT_INTERVAL as usize) {
        game.travel(moment + 1, &first);
    }
    let recording = game.recording();
    assert_eq!(replay(&directory, recording), game.pass.packets);
    game.runtime.close();
}
