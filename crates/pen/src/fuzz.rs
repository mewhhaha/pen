//! `pen fuzz`: random input against the game's invariants, without
//! rendering. Each run plays a script of random steps that a seed decides
//! (keys, chords, clicks, drags, the wheel, console commands, tunables, and
//! actions on the nodes of the game's interface as it shows them), with the
//! game's own quicksaves and loads kept in memory and every service answered
//! by a stand-in, and checks after each frame that the game did not trap,
//! its packet decoded and was valid, and (every [`STATE_EVERY`] frames and
//! at the end) that its state is a bounded array of finite lanes.
//!
//! A run that breaks one is cut down: steps are dropped, in halves, then
//! quarters and so on down to single steps, while the same invariant still
//! breaks, and what is left is written as a script that `pen run --headless
//! --script-file` (or `pen fuzz --replay`) plays again.
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use blot_host::GuestModule;
use pen_protocol::boundary::{read_catalog, validate_state};
use pen_protocol::{Catalog, Frame, RenderFrame, Viewport, service};

use crate::game::{GameRuntime, Host};
use crate::input::Input;
use crate::script::{self, Script, Step};
use crate::services::Services;
use crate::snapshot::{self, SavedState};

/// The viewport runs play at: the page script coordinates are measured in.
const VIEWPORT: Viewport = Viewport { width: 960, height: 640 };
/// Frames between the checks of the game's state.
const STATE_EVERY: usize = 30;

/// Keys a step taps or holds: those games read most, the time travel and
/// dev tools chords aside (the host keeps them).
const KEYS: [&str; 34] = [
    "Tab",
    "Escape",
    "Enter",
    "Space",
    "Backspace",
    "Delete",
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "F1",
    "F5",
    "F6",
    "F9",
    "F10",
    "KeyA",
    "KeyB",
    "KeyD",
    "KeyE",
    "KeyG",
    "KeyQ",
    "KeyR",
    "KeyS",
    "KeyU",
    "KeyW",
    "KeyZ",
    "Digit1",
    "Digit2",
    "Digit3",
    "Digit4",
    "Digit5",
    "Home",
    "End",
    "PageDown",
];
/// Keys a step holds down for others to chord with.
const MODIFIERS: [&str; 3] = ["Shift", "Control", "Alt"];

/// splitmix64: small, fast and the same everywhere, so a seed names a run.
#[derive(Clone)]
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number below `bound` (above 0).
    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound.max(1)
    }

    fn index(&mut self, length: usize) -> usize {
        usize::try_from(self.below(length as u64)).expect("below a usize")
    }

    /// A number in `low..high`.
    fn between(&mut self, low: f32, high: f32) -> f32 {
        let unit = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        low + (high - low) * unit
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

/// A point on the page, now and then off its edges.
fn point(random: &mut Random) -> (f32, f32) {
    let (width, height) = (VIEWPORT.width as f32, VIEWPORT.height as f32);
    if random.chance(5) {
        return (random.between(-50.0, width + 50.0), random.between(-50.0, height + 50.0));
    }
    (random.between(0.0, width).round(), random.between(0.0, height).round())
}

/// What a step may use besides the seed: the game's tunables (from the
/// checkout) and the interface its last frame showed.
pub struct Vocabulary {
    /// Each tunable's name and range.
    pub tunables: Vec<(String, f32, f32)>,
}

/// Makes up the next step, as a script word.
fn step(
    random: &mut Random,
    vocabulary: &Vocabulary,
    interface: &[pen_protocol::UiNode],
    held: &mut Vec<&'static str>,
) -> String {
    loop {
        match random.below(100) {
            0..18 => return format!("key={}", KEYS[random.index(KEYS.len())]),
            18..23 => {
                // A modifier down, or one held let go.
                if let Some(index) = (!held.is_empty() && random.chance(60)).then(|| random.index(held.len())) {
                    return format!("up={}", held.swap_remove(index));
                }
                let modifier = MODIFIERS[random.index(MODIFIERS.len())];
                if !held.contains(&modifier) {
                    held.push(modifier);
                    return format!("down={modifier}");
                }
            }
            23..38 => {
                let (x, y) = point(random);
                return format!("click={x},{y}");
            }
            38..46 => {
                let ((x0, y0), (x1, y1)) = (point(random), point(random));
                return format!("drag={x0},{y0},{x1},{y1}");
            }
            46..54 => {
                let (x, y) = point(random);
                return format!("move={x},{y}");
            }
            54..60 => {
                let (x, y) = point(random);
                return format!("wheel={},{x},{y}", random.between(-600.0, 600.0).round());
            }
            60..64 => return format!("console={},{}", random.below(12), random.below(13)),
            64..68 if !vocabulary.tunables.is_empty() => {
                let (name, low, high) = &vocabulary.tunables[random.index(vocabulary.tunables.len())];
                return format!("tune={name}={}", random.between(*low, *high));
            }
            68..82 if !interface.is_empty() => {
                let node = &interface[random.index(interface.len())];
                // An action it says it takes, mostly.
                let actions: Vec<u32> = (0..8).filter(|action| node.actions & (1 << action) != 0).collect();
                let action = if actions.is_empty() || random.chance(10) {
                    random.index(8)
                } else {
                    usize::try_from(actions[random.index(actions.len())]).expect("below 8")
                };
                let value = random.between(node.minimum.min(0.0) - 1.0, node.maximum.max(1.0) + 1.0);
                return format!("ui={:#x},{},{value}", node.id, pen_protocol::ui_action::NAMES[action]);
            }
            82..100 => return format!("wait={}", 1 + random.below(30)),
            _ => {}
        }
    }
}

/// A host that draws nothing, with the game's quicksave in memory.
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

/// Stand-ins for the platform's services, the same every run: nothing is
/// opened, copied or asked of Steam.
fn services() -> Services {
    let mut services = Services::new();
    for code in [service::STEAM, service::CLIPBOARD, service::URL] {
        services.register(Box::new(pen_runtime::services::Unavailable(code)));
    }
    services.register(Box::new(pen_runtime::services::Platform {
        os: "fuzz".into(),
        locale: Some("en-US".into()),
        device: "fuzz".into(),
        handheld: false,
    }));
    services.register(Box::new(pen_runtime::services::Pathfinding));
    services
}

/// An invariant the game broke, and when.
#[derive(Clone, Debug, PartialEq)]
pub struct Broken {
    /// Which invariant: what a cut-down script must still break.
    pub invariant: &'static str,
    pub frame: usize,
    pub message: String,
}

/// Another check of each frame, for tests: says what it found wrong.
pub type Extra<'a> = &'a dyn Fn(&RenderFrame) -> Option<String>;

/// The game to fuzz, started afresh for each run.
pub struct Game {
    pub engine: wasmtime::Engine,
    pub module: GuestModule,
    pub catalog: Catalog,
    pub scene: Option<SavedState>,
}

impl Game {
    pub fn load(engine: wasmtime::Engine, directory: &Path, assets: &Path, scene: &Path) -> Result<Game> {
        let module = crate::load_module(&engine, directory, true, false)?;
        let catalog = read_catalog(assets)?;
        Ok(Game { engine, module, catalog, scene: snapshot::read(scene).ok() })
    }
}

/// What a run did: its steps (as words), and what it broke, if anything.
pub struct Run {
    pub words: Vec<String>,
    pub broken: Option<Broken>,
}

/// Plays `frames` frames: `steps` (`words` as written), then (with
/// `random`) steps made up as the game goes. Stops at the first invariant
/// broken.
fn play(
    game: &Game,
    steps: Vec<Step>,
    mut words: Vec<String>,
    mut random: Option<(Random, &Vocabulary)>,
    frames: usize,
    extra: Option<Extra>,
) -> Result<Run> {
    let mut host = Null;
    let mut runtime =
        GameRuntime::new(&game.engine, game.module.clone(), game.catalog.clone(), &mut host, None, game.scene.clone())?;
    let mut services = services();
    let mut script = Script::new(steps);
    let mut held = Vec::new();
    let mut interface = Vec::new();
    let mut quicksave: Option<SavedState> = None;
    for index in 0..frames {
        if let Some((random, vocabulary)) = &mut random
            && script.idle()
        {
            let word = step(random, vocabulary, &interface, &mut held);
            script.push(script::parse(std::slice::from_ref(&word))?.remove(0));
            words.push(word);
        }
        let tick = script.tick(VIEWPORT);
        let mut events = tick.events;
        services.frame(&mut Input::default());
        events.extend(services.take_answers());
        let frame = Frame { timestamp: index as f64 / 60.0, viewport: VIEWPORT, events };
        let input = runtime.input(&frame)?;
        let outcome = (|| -> Result<RenderFrame> {
            runtime.send(&input)?;
            let rendered = runtime.receive(&host)?;
            runtime.follow(&mut host, &rendered)?;
            Ok(rendered)
        })();
        let rendered = match outcome {
            Ok(rendered) => rendered,
            Err(error) if error.is::<crate::game::InvalidFrame>() => {
                return stop(words, "a valid packet", index, format!("{error:#}"));
            }
            Err(error) => return stop(words, "no trap", index, format!("{error:#}")),
        };
        services.answer(&rendered.services);
        if rendered.save_requested {
            quicksave = Some(runtime.snapshot()?);
        }
        if rendered.load_requested
            && let Some(saved) = &quicksave
        {
            if let Err(error) = runtime.load(&mut host, saved) {
                return stop(words, "no trap", index, format!("loading its quicksave: {error:#}"));
            }
            services.forget();
        }
        if let Some(found) = extra.and_then(|extra| extra(&rendered)) {
            return stop(words, "the extra check", index, found);
        }
        if (index + 1) % STATE_EVERY == 0 || index + 1 == frames {
            match runtime.snapshot() {
                Ok(saved) => {
                    if let Err(error) = validate_state(&saved.state) {
                        return stop(words, "a finite state", index, format!("{error:#}"));
                    }
                }
                Err(error) => return stop(words, "a finite state", index, format!("{error:#}")),
            }
        }
        interface = rendered.ui.iter().filter(|node| !node.hidden && !node.disabled).cloned().collect();
        if rendered.exit_requested {
            break;
        }
    }
    runtime.close();
    Ok(Run { words, broken: None })
}

/// A run that broke `invariant` at `frame`.
fn stop(words: Vec<String>, invariant: &'static str, frame: usize, message: String) -> Result<Run> {
    Ok(Run { words, broken: Some(Broken { invariant, frame, message }) })
}

/// The smallest list of `words` found that still breaks `invariant`, by
/// dropping chunks of them (halves, then quarters, … then single words)
/// while it does; `fails` plays a list.
pub fn minimize(words: Vec<String>, mut fails: impl FnMut(&[String]) -> Result<bool>) -> Result<Vec<String>> {
    let mut words = words;
    let mut chunk = words.len().div_ceil(2).max(1);
    loop {
        let mut start = 0;
        let mut dropped = false;
        while start < words.len() {
            let end = (start + chunk).min(words.len());
            let candidate: Vec<String> = words[..start].iter().chain(&words[end..]).cloned().collect();
            if fails(&candidate)? {
                words = candidate;
                dropped = true;
            } else {
                start = end;
            }
        }
        if chunk == 1 && !dropped {
            return Ok(words);
        }
        if !dropped {
            chunk = chunk.div_ceil(2);
        }
    }
}

pub struct Options {
    pub seed: u64,
    pub runs: u64,
    pub frames: usize,
    /// Where scripts that break the game are written.
    pub out: PathBuf,
    pub vocabulary: Vocabulary,
}

/// Fuzzes `game`: `runs` runs from `seed` on, each `frames` long; writes a
/// cut-down script for each one that breaks an invariant, and fails if any
/// did.
pub fn run(game: &Game, options: &Options, extra: Option<Extra>) -> Result<Vec<PathBuf>> {
    let mut written = Vec::new();
    for seed in options.seed..options.seed + options.runs {
        let random = Random(seed);
        let ran = play(game, Vec::new(), Vec::new(), Some((random, &options.vocabulary)), options.frames, extra)?;
        let Some(broken) = ran.broken else {
            eprintln!("fuzz: seed {seed}: {} steps, {} frames, nothing broken", ran.words.len(), options.frames);
            continue;
        };
        eprintln!(
            "fuzz: seed {seed} broke {} at frame {} after {} steps: {}",
            broken.invariant,
            broken.frame,
            ran.words.len(),
            broken.message.lines().next().unwrap_or_default()
        );
        let frames = broken.frame + 1;
        let mut tries = 0;
        let words = minimize(ran.words, |words| {
            tries += 1;
            let again = play_words(game, words, frames, extra)?;
            Ok(again.broken.is_some_and(|again| again.invariant == broken.invariant))
        })?;
        let replayed =
            play_words(game, &words, frames, extra)?.broken.context("the cut-down script stopped failing")?;
        std::fs::create_dir_all(&options.out)?;
        let path = options.out.join(format!("seed-{seed}.steps"));
        let text = format!(
            "# pen fuzz, seed {seed}: breaks \"{}\" at frame {}\n# {}\n{}\n",
            replayed.invariant,
            replayed.frame,
            replayed.message.lines().next().unwrap_or_default(),
            words.join("\n")
        );
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
        eprintln!(
            "  cut down to {} steps in {tries} runs, written to {}:\n    {}\n  play it: pen run --headless --script-file {}",
            words.len(),
            path.display(),
            words.join(" "),
            path.display()
        );
        written.push(path);
    }
    if !written.is_empty() {
        bail!("fuzzing broke the game's invariants {} times", written.len());
    }
    Ok(written)
}

/// Plays `words`, a script, for `frames` frames.
fn play_words(game: &Game, words: &[String], frames: usize, extra: Option<Extra>) -> Result<Run> {
    play(game, script::parse(words)?, words.to_vec(), None, frames, extra)
}

/// Plays `steps` for `frames` frames, and says what invariant they break.
pub fn check(game: &Game, steps: Vec<Step>, frames: usize) -> Result<Option<Broken>> {
    Ok(play(game, steps, Vec::new(), None, frames, None)?.broken)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::bundle;

    /// Plays a script file (a cut-down one) and says what it breaks.
    fn replay(game: &Game, path: &Path, frames: usize, extra: Option<Extra>) -> Result<Option<Broken>> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let words: Vec<String> = text
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .flat_map(str::split_whitespace)
            .map(str::to_owned)
            .collect();
        Ok(play_words(game, &words, frames, extra)?.broken)
    }

    #[test]
    fn seeds_name_the_same_steps() {
        let vocabulary = Vocabulary { tunables: vec![("camera.zoom_speed".into(), 0.0002, 0.005)] };
        let node = pen_protocol::UiNode { id: 0x20001, actions: 1 << 2, ..pen_protocol::UiNode::default() };
        let words = |seed| {
            let mut random = Random(seed);
            let mut held = Vec::new();
            (0..200).map(|_| step(&mut random, &vocabulary, std::slice::from_ref(&node), &mut held)).collect::<Vec<_>>()
        };
        assert_eq!(words(1), words(1));
        assert_ne!(words(1), words(2));
        // Every one is a step a script takes, and each kind comes up.
        let all = words(3);
        script::parse(&all).unwrap();
        for kind in [
            "key=",
            "down=",
            "click=",
            "drag=",
            "move=",
            "wheel=",
            "console=",
            "tune=camera.zoom_speed=",
            "ui=0x20001,",
            "wait=",
        ] {
            assert!(all.iter().any(|word| word.starts_with(kind)), "{kind}");
        }
    }

    #[test]
    fn minimizing_keeps_only_what_fails() {
        let words: Vec<String> = (0..40).map(|index| format!("w{index}")).collect();
        // It fails when w7 comes before w31.
        let mut runs = 0;
        let fails = |words: &[String]| {
            runs += 1;
            let at = |word: &str| words.iter().position(|w| w == word);
            Ok(matches!((at("w7"), at("w31")), (Some(a), Some(b)) if a < b))
        };
        assert_eq!(minimize(words, fails).unwrap(), ["w7", "w31"]);
        assert!(runs < 120, "{runs} runs");
        // Nothing to drop.
        assert_eq!(minimize(vec!["only".into()], |_| Ok(true)).unwrap(), Vec::<String>::new());
    }

    /// The game itself, where `just export` made it: random runs break
    /// nothing, and a run that trips an extra check is cut down to the
    /// steps that do it.
    #[test]
    fn fuzzes_the_game_and_cuts_failures_down() {
        let Some(directory) = bundle() else { return };
        let engine = blot_host::engine().unwrap();
        let game = Game::load(engine, &directory, &directory.join("assets"), &directory.join("main.scene")).unwrap();
        let out = std::env::temp_dir().join(format!("pen-fuzz-{}", std::process::id()));
        let options = Options {
            seed: 1,
            runs: 2,
            frames: 300,
            out: out.clone(),
            vocabulary: Vocabulary { tunables: Vec::new() },
        };
        assert!(run(&game, &options, None).unwrap().is_empty());
        // A check that breaks once a node of the interface has the keyboard
        // focus, and the first seed whose steps do that.
        let focused =
            |frame: &RenderFrame| frame.ui.iter().any(|node| node.focused).then(|| "a node took the focus".to_owned());
        let vocabulary = Vocabulary { tunables: Vec::new() };
        let seed = (1..=20)
            .find(|&seed| {
                let run = play(&game, Vec::new(), Vec::new(), Some((Random(seed), &vocabulary)), 600, Some(&focused));
                run.unwrap().broken.is_some()
            })
            .expect("a seed whose steps focus a node");
        let options = Options { seed, runs: 1, frames: 600, ..options };
        let error = run(&game, &options, Some(&focused)).unwrap_err().to_string();
        assert_eq!(error, "fuzzing broke the game's invariants 1 times");
        let path = out.join(format!("seed-{seed}.steps"));
        let broken = replay(&game, &path, 600, Some(&focused)).unwrap().unwrap();
        assert_eq!(broken.invariant, "the extra check");
        let steps = std::fs::read_to_string(&path).unwrap();
        let kept = steps.lines().filter(|line| !line.starts_with('#')).count();
        assert!((1..=4).contains(&kept), "{steps}");
        let _ = std::fs::remove_dir_all(out);
    }
}
