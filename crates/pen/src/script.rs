//! Scripted input, a step at a time: `wait=N` frames, `key=KEY` taps a key, `down=KEY` /
//! `up=KEY` hold and release it, `type=TEXT`, `move=X,Y`, `glide=X,Y,FRAMES`, `click=X,Y`,
//! `press=BUTTON` / `release=BUTTON` (left, middle or right),
//! `drag=X,Y,X,Y`, `wheel=DELTA[,X,Y]`, `shot=NAME` (a PNG in the shots
//! directory), `overlay-shot=NAME` (the same with the dev tools over it),
//! `save=FILE` / `load=FILE` (quicksaves), and `quit`.
//!
//! `glide` eases from the current scripted pointer position to its destination
//! over a positive number of frames, then runs the next step. It only moves:
//! `press=left glide=600,300,45 release=left` makes a smooth drag. `move` is instant.
//!
//! `ui=ID,ACTION[,VALUE]` does what assistive technology does to a node of
//! the game's interface (see `accessibility`): ID is the node's id, decimal
//! or 0x hex, and ACTION one of `focus`, `blur`, `activate`, `increment`,
//! `decrement`, `set_value` (to VALUE), `expand` and `collapse`, or its code.
//! It reaches the game as a `UiAction` event, as the window's adapter sends it.
//!
//! `tune=NAME=VALUE` sets a live-tunable constant (the checkout's
//! `src/tunables.json`, `tunables::tunable_id`), as the dev tools' `/tune`
//! does: a Tune event, in the frame it runs in.
//!
//! `report[=FILE]` writes a bug report (`bugreport`), as Ctrl+Shift+B does.
//!
//! `console=CODE[,ARGUMENT]` sends the game a console command by its number
//! (the checkout's `src/commands.blot`), as the console's `/command N [ARG]` does:
//! `console=3,10` hires ten gnorps in gdev.
//!
//! `game=EVENT[,VALUE...]` sends one of the game's own events (its
//! `pen.json`'s `protocol.events`) by name, or by tag, with up to three
//! values: a Game event, as nothing else on the host sends.
//!
//! A key that types a character (`key=a`, `type=hello`) sends it as text
//! too, as a keyboard does. `commit=TEXT` sends text as an input method
//! commits it, and `compose=TEXT[,CURSOR]` its composition so far (the caret
//! CURSOR characters in; an empty TEXT ends it). `motion=DX,DY` moves the
//! mouse as it does while the pointer is locked (a MouseMotion event).
//!
//! `pad=connect` and `pad=disconnect` plug in and pull out a gamepad, and
//! `pad=BUTTON[,VALUE]` and `pad=AXIS,VALUE` move it (plugging it in first):
//! a button of `pen_protocol::gamepad_button`'s (south, left_trigger,
//! dpad_up, ...) to VALUE (1 if left out, 0 lets go), an axis of
//! `gamepad_axis`'s (left_x, ...) to VALUE from -1 to 1, past the same dead
//! zone a real pad's sticks have. `touch=ID,X,Y` puts touch ID down at X,Y,
//! or moves it there, and `lift=ID` lifts it: they make the same touch,
//! pointer, pan and pinch events a touch screen does.
//!
//! Time travel has steps too: `pause` and `resume` the game, `step[=N]`
//! runs N frames (one by default, one per frame of the script, as `wait`
//! waits) and stays paused, `rewind=N` goes back N
//! frames (paused there), and `replay=SECONDS` plays the last SECONDS of
//! recorded input again. While paused, steps still take one call each (a
//! `wait` counts presented frames, a `shot` captures the paused frame), and
//! input waits for the next frame that runs.
//!
//! Coordinates are pixels of a page 960 wide (960 x 640, the window's size
//! when it opens); they scale with the viewport's width, so a 1920 x 1280
//! viewport sees them at twice the size.
use std::collections::VecDeque;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail, ensure};
use pen_protocol::boundary::MAX_UI_ID;
use pen_protocol::{GameEvent, Viewport, gamepad_axis, gamepad_button, touch_phase, ui_action};
use pen_runtime::gamepad::{Gamepads, Pad};
use pen_runtime::text;
use pen_runtime::touch::Touches;

/// The page width script coordinates are measured in.
const PAGE_WIDTH: f32 = 960.0;
/// Frames a drag takes between pressing and releasing.
const DRAG_FRAMES: usize = 8;

#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    Wait(u32),
    Key(String),
    Down(String),
    Up(String),
    Type(String),
    Move(f32, f32),
    Glide {
        x: f32,
        y: f32,
        frames: u32,
    },
    Click(f32, f32),
    Press(u32),
    Release(u32),
    Drag(f32, f32, f32, f32),
    Wheel(f32, Option<(f32, f32)>),
    Shot(String),
    /// A shot with the dev tools over the frame.
    OverlayShot(String),
    Save(PathBuf),
    Load(PathBuf),
    Quit,
    Travel(Travel),
    /// An action on an interface node: its id, the action's code and a value.
    Ui {
        node: u32,
        action: u32,
        value: f32,
    },
    /// A tunable, by name, and its new value.
    Tune(String, f32),
    /// A console command's number and argument.
    Console(f32, f32),
    /// A line typed into the dev tools' console.
    Overlay(String),
    /// A bug report written to this file, or (empty) where Ctrl+Shift+B writes one.
    Report(PathBuf),
    /// One of the game's own events: as named, its values, and once named
    /// ([`name_game_events`]), its tag and lanes.
    Game {
        event: String,
        values: Vec<f32>,
        sent: Option<(u32, [f32; 3])>,
    },
    Pad(PadStep),
    /// The mouse moved this far while locked.
    Motion(f32, f32),
    /// An input method's composition, and its caret in characters.
    Compose(String, Option<usize>),
    /// Text an input method commits.
    Commit(String),
    /// A touch down at, or moved to, a point.
    Touch(u64, f32, f32),
    /// A touch lifted.
    Lift(u64),
}

/// How `game=` steps name the game's events: the tag and lanes of the event
/// called the first argument, given the second's values.
pub type GameEvents<'a> = dyn Fn(&str, &[f32]) -> Result<(u32, Vec<f32>)> + 'a;

/// Gives each `game=` step that names its event the tag and lanes `events`
/// finds for it.
pub fn name_game_events(steps: &mut [Step], events: &GameEvents) -> Result<()> {
    for step in steps {
        if let Step::Game { event, values, sent: sent @ None } = step {
            let (tag, lanes) = events(event, values).with_context(|| format!("game={event}"))?;
            ensure!(lanes.len() <= 3, "game={event}: an event has at most three values");
            let mut three = [0.0; 3];
            three[..lanes.len()].copy_from_slice(&lanes);
            *sent = Some((tag, three));
        }
    }
    Ok(())
}

/// What a `pad=` step does to the script's gamepad.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PadStep {
    Connect,
    Disconnect,
    /// A button (`gamepad_button`) to a value from 0 to 1.
    Button(usize, f32),
    /// An axis (`gamepad_axis`) to a value from -1 to 1.
    Axis(usize, f32),
}

/// A time travel step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Travel {
    Pause,
    Resume,
    /// Run this many frames, paused.
    Step(u32),
    /// Go back this many frames.
    Rewind(u64),
    /// Play the last this many seconds again.
    Replay(f64),
}

/// Playwright's key names as `KeyboardEvent.key` reports them.
fn key_name(name: &str) -> String {
    match name {
        "Space" => " ".to_owned(),
        "Backquote" => "`".to_owned(),
        _ => match (name.strip_prefix("Key"), name.strip_prefix("Digit")) {
            (Some(letter), _) if letter.len() == 1 => letter.to_lowercase(),
            (_, Some(digit)) if digit.len() == 1 => digit.to_owned(),
            _ => name.to_owned(),
        },
    }
}

fn button(name: &str) -> Result<u32> {
    Ok(match name {
        "left" => 0,
        "middle" => 1,
        "right" => 2,
        _ => bail!("button must be left, middle or right, not {name}"),
    })
}

fn numbers(text: &str, count: usize, step: &str) -> Result<Vec<f32>> {
    let values: Vec<f32> = text
        .split(',')
        .map(|part| part.trim().parse::<f32>())
        .collect::<Result<_, _>>()
        .with_context(|| format!("{step} takes numbers, not {text}"))?;
    if values.len() != count {
        bail!("{step} takes {count} numbers, not {text}");
    }
    Ok(values)
}

/// `ui=ID,ACTION[,VALUE]`.
fn ui_step(text: &str) -> Result<Step> {
    let parts: Vec<&str> = text.split(',').map(str::trim).collect();
    let [target, action, rest @ ..] = parts.as_slice() else { bail!("ui takes ID,ACTION[,VALUE], not {text}") };
    ensure!(rest.len() <= 1, "ui takes ID,ACTION[,VALUE], not {text}");
    let node = match target.strip_prefix("0x").or_else(|| target.strip_prefix("0X")) {
        Some(hex) => u32::from_str_radix(hex, 16),
        None => target.parse(),
    }
    .ok()
    .filter(|&node| (1..=MAX_UI_ID).contains(&(node as usize)))
    .with_context(|| format!("ui takes a node id from 1 to {MAX_UI_ID} (or 0x hex), not {target}"))?;
    let action = match ui_action::NAMES.iter().position(|name| name == action) {
        Some(code) => u32::try_from(code)?,
        None => action.parse().ok().filter(|code| (*code as usize) < ui_action::NAMES.len()).with_context(|| {
            format!("ui takes an action ({} or its number), not {action}", ui_action::NAMES.join(", "))
        })?,
    };
    let value = match rest {
        [value] => value
            .parse::<f32>()
            .ok()
            .filter(|value| value.is_finite())
            .with_context(|| format!("ui takes a number as its value, not {value}"))?,
        _ => 0.0,
    };
    Ok(Step::Ui { node, action, value })
}

/// `pad=connect`, `pad=disconnect`, `pad=BUTTON[,VALUE]` or `pad=AXIS,VALUE`.
fn pad_step(text: &str) -> Result<Step> {
    let usage = || {
        format!(
            "pad takes connect, disconnect, BUTTON[,VALUE] or AXIS,VALUE (buttons {}; axes {}), not {text}",
            gamepad_button::NAMES.join(", "),
            gamepad_axis::NAMES.join(", ")
        )
    };
    let (name, value) = match text.split_once(',') {
        Some((name, value)) => {
            (name, Some(value.parse::<f32>().ok().filter(|value| value.is_finite()).with_context(usage)?))
        }
        None => (text, None),
    };
    Ok(Step::Pad(match (name, value) {
        ("connect", None) => PadStep::Connect,
        ("disconnect", None) => PadStep::Disconnect,
        _ => match (gamepad_button::NAMES.iter().position(|button| *button == name), value) {
            (Some(button), value) => {
                let value = value.unwrap_or(1.0);
                ensure!((0.0..=1.0).contains(&value), "pad takes a button's value from 0 to 1, not {value}");
                PadStep::Button(button, value)
            }
            (None, Some(value)) => {
                let axis = gamepad_axis::NAMES.iter().position(|axis| *axis == name).with_context(usage)?;
                ensure!((-1.0..=1.0).contains(&value), "pad takes an axis's value from -1 to 1, not {value}");
                PadStep::Axis(axis, value)
            }
            (None, None) => bail!(usage()),
        },
    }))
}

/// A touch's id in `touch=` and `lift=` steps.
fn touch_id(text: &str, step: &str) -> Result<u64> {
    text.parse().with_context(|| format!("{step} takes a touch's number, not {text}"))
}

/// Parses steps; each argument may hold several, separated by whitespace.
pub fn parse(arguments: &[String]) -> Result<Vec<Step>> {
    let mut steps = Vec::new();
    for word in arguments.iter().flat_map(|argument| argument.split_whitespace()) {
        let (name, value) = word.split_once('=').unwrap_or((word, ""));
        steps.push(match name {
            "wait" => Step::Wait(value.parse().with_context(|| format!("wait takes frames, not {value}"))?),
            "key" => Step::Key(key_name(value)),
            "down" => Step::Down(key_name(value)),
            "up" => Step::Up(key_name(value)),
            "type" => Step::Type(value.to_owned()),
            "move" => {
                let at = numbers(value, 2, "move")?;
                Step::Move(at[0], at[1])
            }
            "glide" => {
                let usage =
                    || format!("glide takes X,Y,FRAMES with finite coordinates and positive whole frames, not {value}");
                let (position, frames) = value.rsplit_once(',').with_context(usage)?;
                let at = numbers(position, 2, "glide")?;
                let frames = frames.trim().parse::<u32>().ok().filter(|frames| *frames > 0).with_context(usage)?;
                ensure!(at.iter().all(|number| number.is_finite()), usage());
                Step::Glide { x: at[0], y: at[1], frames }
            }
            "click" => {
                let at = numbers(value, 2, "click")?;
                Step::Click(at[0], at[1])
            }
            "press" => Step::Press(button(value)?),
            "release" => Step::Release(button(value)?),
            "drag" => {
                let at = numbers(value, 4, "drag")?;
                Step::Drag(at[0], at[1], at[2], at[3])
            }
            "wheel" => {
                let parts = value.split(',').count();
                let at = numbers(value, if parts == 3 { 3 } else { 1 }, "wheel")?;
                Step::Wheel(at[0], (parts == 3).then(|| (at[1], at[2])))
            }
            "shot" if !value.is_empty() => Step::Shot(value.to_owned()),
            "overlay-shot" if !value.is_empty() => Step::OverlayShot(value.to_owned()),
            "save" if !value.is_empty() => Step::Save(PathBuf::from(value)),
            "load" if !value.is_empty() => Step::Load(PathBuf::from(value)),
            "quit" => Step::Quit,
            "ui" => ui_step(value)?,
            "tune" => {
                let usage = || format!("tune takes NAME=VALUE, not {value}");
                let (name, number) = value.rsplit_once('=').with_context(usage)?;
                let number = number.parse::<f32>().ok().filter(|number| number.is_finite()).with_context(usage)?;
                ensure!(!name.is_empty(), usage());
                Step::Tune(name.to_owned(), number)
            }
            "console" => {
                let parts = value.split(',').count();
                let usage = || format!("console takes CODE[,ARGUMENT], not {value}");
                ensure!(parts <= 2, usage());
                let at = numbers(value, parts, "console")?;
                ensure!(at[0] >= 0.0 && at[0].fract() == 0.0, usage());
                Step::Console(at[0], at.get(1).copied().unwrap_or(0.0))
            }
            "overlay" => {
                ensure!(value.starts_with('/'), "overlay takes a console line, /COMMAND[,ARGUMENT...], not {value}");
                Step::Overlay(value.replace(',', " "))
            }
            "report" => Step::Report(PathBuf::from(value)),
            "game" => {
                let mut parts = value.split(',');
                let event = parts.next().unwrap_or_default().trim().to_owned();
                ensure!(!event.is_empty(), "game takes EVENT[,VALUE...], not {value}");
                let values: Vec<f32> = parts
                    .map(|part| part.trim().parse::<f32>().ok().filter(|value| value.is_finite()))
                    .collect::<Option<_>>()
                    .with_context(|| format!("game takes finite numbers after its event, not {value}"))?;
                ensure!(values.len() <= 3, "game takes at most three values, not {value}");
                // By tag, it needs no declaration.
                let sent = event.parse::<u32>().ok().map(|tag| {
                    let mut lanes = [0.0; 3];
                    lanes[..values.len()].copy_from_slice(&values);
                    (tag, lanes)
                });
                Step::Game { event, values, sent }
            }
            "pad" => pad_step(value)?,
            "motion" => {
                let by = numbers(value, 2, "motion")?;
                Step::Motion(by[0], by[1])
            }
            "compose" => match value.rsplit_once(',').and_then(|(text, cursor)| Some((text, cursor.parse().ok()?))) {
                Some((text, cursor)) => Step::Compose(text.to_owned(), Some(cursor)),
                None => Step::Compose(value.to_owned(), None),
            },
            "commit" => Step::Commit(value.to_owned()),
            "touch" => {
                let (id, at) = value.split_once(',').with_context(|| format!("touch takes ID,X,Y, not {value}"))?;
                let at = numbers(at, 2, "touch")?;
                Step::Touch(touch_id(id, "touch")?, at[0], at[1])
            }
            "lift" => Step::Lift(touch_id(value, "lift")?),
            "pause" => Step::Travel(Travel::Pause),
            "resume" => Step::Travel(Travel::Resume),
            "step" => Step::Travel(Travel::Step(if value.is_empty() {
                1
            } else {
                value.parse().with_context(|| format!("step takes frames, not {value}"))?
            })),
            "rewind" => Step::Travel(Travel::Rewind(
                value.parse().with_context(|| format!("rewind takes frames, not {value}"))?,
            )),
            "replay" => {
                let seconds: f64 = value.parse().with_context(|| format!("replay takes seconds, not {value}"))?;
                ensure!(seconds > 0.0, "replay takes a positive number of seconds");
                Step::Travel(Travel::Replay(seconds))
            }
            _ => return Err(anyhow!("unknown script step {word}")),
        });
    }
    Ok(steps)
}

/// Whether steps go back in time, or write a bug report (and so need a
/// history kept).
pub fn travels_back(steps: &[Step]) -> bool {
    steps.iter().any(|step| matches!(step, Step::Travel(Travel::Rewind(_) | Travel::Replay(_)) | Step::Report(_)))
}

/// What one frame of the script does.
#[derive(Debug, Default)]
pub struct Tick {
    pub events: Vec<GameEvent>,
    pub shots: Vec<String>,
    /// Shots with the dev tools over the frame.
    pub overlay_shots: Vec<String>,
    pub saves: Vec<PathBuf>,
    pub loads: Vec<PathBuf>,
    pub quit: bool,
    /// Time travel, before the frame runs.
    pub travel: Option<Travel>,
    /// Lines for the dev tools' console.
    pub overlay: Vec<String>,
    /// Bug reports to write (an empty path: where Ctrl+Shift+B writes one).
    pub reports: Vec<PathBuf>,
}

/// Plays steps one frame at a time: each input step takes a frame, a tap
/// or click releases on the next, and a glide or drag moves over several.
pub struct Script {
    steps: VecDeque<Step>,
    /// Frames of events already worked out, from the next frame on.
    queued: VecDeque<Vec<GameEvent>>,
    waiting: u32,
    pointer: (f32, f32),
    /// The script's gamepad while it is plugged in, and the pads as the
    /// game knows them, through pen-runtime as a real pad's are.
    pad: Option<Pad>,
    gamepads: Gamepads,
    /// The touches down and where, and the touches as the game knows them.
    touched: Vec<(u64, (f32, f32))>,
    touches: Touches,
    /// An input method's composition is open (a commit ends it).
    composing: bool,
    glide: Option<Glide>,
}

/// Work out one position per frame, without queuing a duration's worth of events.
struct Glide {
    from: (f32, f32),
    to: (f32, f32),
    frames: u32,
    elapsed: u32,
}

/// A key pressed: its KeyDown, and the character it types, if it types one.
fn press(key: &str) -> Vec<GameEvent> {
    let mut events = vec![GameEvent::KeyDown(key.to_owned())];
    if key.chars().count() == 1 {
        text::commit(key, &mut events);
    }
    events
}

impl Script {
    pub fn new(steps: Vec<Step>) -> Script {
        Script {
            steps: steps.into(),
            queued: VecDeque::new(),
            waiting: 0,
            pointer: (0.0, 0.0),
            pad: None,
            gamepads: Gamepads::default(),
            touched: Vec::new(),
            touches: Touches::default(),
            composing: false,
            glide: None,
        }
    }

    pub fn done(&self) -> bool {
        self.steps.is_empty() && self.queued.is_empty() && self.waiting == 0 && self.glide.is_none()
    }

    /// Whether every step has been taken, though the last may still play out.
    #[cfg(feature = "compiler")]
    pub fn idle(&self) -> bool {
        self.steps.is_empty()
    }

    /// Adds a step after the others: input fuzzing makes them up as it goes,
    /// each in time for the frame that takes it.
    #[cfg(feature = "compiler")]
    pub fn push(&mut self, step: Step) {
        self.steps.push_back(step);
    }

    fn queue(&mut self, frame: usize, event: GameEvent) {
        while self.queued.len() <= frame {
            self.queued.push_back(Vec::new());
        }
        self.queued[frame].push(event);
    }

    fn advance_glide(&mut self) -> Option<GameEvent> {
        let glide = self.glide.as_mut()?;
        glide.elapsed += 1;
        let t = glide.elapsed as f32 / glide.frames as f32;
        let eased = t * t * (3.0 - 2.0 * t);
        self.pointer =
            (glide.from.0 * (1.0 - eased) + glide.to.0 * eased, glide.from.1 * (1.0 - eased) + glide.to.1 * eased);
        if glide.elapsed == glide.frames {
            self.pointer = glide.to;
            self.glide = None;
        }
        let (x, y) = self.pointer;
        Some(GameEvent::PointerMove { x, y })
    }

    pub fn tick(&mut self, viewport: Viewport) -> Tick {
        let scale = viewport.width as f32 / PAGE_WIDTH;
        let mut tick = Tick { events: self.queued.pop_front().unwrap_or_default(), ..Tick::default() };
        if !self.queued.is_empty() {
            return tick;
        }
        if let Some(event) = self.advance_glide() {
            tick.events.push(event);
            return tick;
        }
        if self.waiting > 0 {
            self.waiting -= 1;
            return tick;
        }
        let Some(step) = self.steps.pop_front() else { return tick };
        let at = |x: f32, y: f32| (x * scale, y * scale);
        match step {
            Step::Wait(frames) => self.waiting = frames.saturating_sub(1),
            Step::Key(key) => {
                tick.events.extend(press(&key));
                self.queue(0, GameEvent::KeyUp(key));
            }
            Step::Down(key) => tick.events.extend(press(&key)),
            Step::Up(key) => tick.events.push(GameEvent::KeyUp(key)),
            Step::Type(text) => {
                for (index, character) in text.chars().enumerate() {
                    let key = character.to_string();
                    for event in press(&key) {
                        if index == 0 {
                            tick.events.push(event);
                        } else {
                            self.queue(index - 1, event);
                        }
                    }
                    self.queue(index, GameEvent::KeyUp(key));
                }
            }
            Step::Move(x, y) => {
                self.pointer = at(x, y);
                let (x, y) = self.pointer;
                tick.events.push(GameEvent::PointerMove { x, y });
            }
            Step::Glide { x, y, frames } => {
                self.glide = Some(Glide { from: self.pointer, to: at(x, y), frames, elapsed: 0 });
                tick.events.extend(self.advance_glide());
            }
            Step::Click(x, y) => {
                self.pointer = at(x, y);
                let (x, y) = self.pointer;
                tick.events.push(GameEvent::PointerMove { x, y });
                tick.events.push(GameEvent::PointerDown { button: 0, x, y });
                self.queue(0, GameEvent::PointerUp { button: 0, x, y });
            }
            Step::Press(button) => {
                let (x, y) = self.pointer;
                tick.events.push(GameEvent::PointerDown { button, x, y });
            }
            Step::Release(button) => {
                let (x, y) = self.pointer;
                tick.events.push(GameEvent::PointerUp { button, x, y });
            }
            Step::Drag(x0, y0, x1, y1) => {
                let (from, to) = (at(x0, y0), at(x1, y1));
                tick.events.push(GameEvent::PointerMove { x: from.0, y: from.1 });
                tick.events.push(GameEvent::PointerDown { button: 0, x: from.0, y: from.1 });
                for frame in 1..=DRAG_FRAMES {
                    let t = frame as f32 / DRAG_FRAMES as f32;
                    let (x, y) = (from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t);
                    self.queue(frame - 1, GameEvent::PointerMove { x, y });
                }
                self.queue(DRAG_FRAMES, GameEvent::PointerUp { button: 0, x: to.0, y: to.1 });
                self.pointer = to;
            }
            Step::Wheel(delta, place) => {
                if let Some((x, y)) = place {
                    self.pointer = at(x, y);
                }
                let (x, y) = self.pointer;
                tick.events.push(GameEvent::Wheel { x, y, delta });
            }
            Step::Shot(name) => tick.shots.push(name),
            Step::OverlayShot(name) => tick.overlay_shots.push(name),
            Step::Save(path) => tick.saves.push(path),
            Step::Load(path) => tick.loads.push(path),
            Step::Quit => tick.quit = true,
            Step::Ui { node, action, value } => tick.events.push(GameEvent::UiAction { node, action, value }),
            Step::Tune(name, value) => {
                tick.events.push(GameEvent::Tune { index: crate::tunables::tunable_id(&name), value });
            }
            Step::Console(command, argument) => tick.events.push(GameEvent::Console { command, argument }),
            Step::Overlay(line) => tick.overlay.push(line),
            Step::Report(path) => tick.reports.push(path),
            // A name no declaration gave a tag sends nothing (name_game_events refuses it first).
            Step::Game { sent, .. } => {
                if let Some((event, [a, b, c])) = sent {
                    tick.events.push(GameEvent::Game { event, a, b, c });
                }
            }
            Step::Pad(step) => {
                let pad = self.pad.get_or_insert_with(Pad::default);
                match step {
                    PadStep::Connect => {}
                    PadStep::Disconnect => self.pad = None,
                    PadStep::Button(button, value) => pad.buttons[button] = value,
                    PadStep::Axis(axis, value) => pad.axes[axis] = value,
                }
                match &self.pad {
                    Some(pad) => self.gamepads.update(0, pad, &mut tick.events),
                    None => self.gamepads.disconnect(0, &mut tick.events),
                }
            }
            Step::Motion(dx, dy) => tick.events.push(GameEvent::MouseMotion { dx, dy }),
            Step::Compose(composed, cursor) => {
                self.composing = !composed.is_empty();
                text::compose(&composed, cursor, &mut tick.events);
            }
            Step::Commit(committed) => {
                // As an input method does, a commit ends the composition.
                if std::mem::take(&mut self.composing) {
                    text::compose("", None, &mut tick.events);
                }
                text::commit(&committed, &mut tick.events);
            }
            Step::Touch(id, x, y) => {
                let (x, y) = at(x, y);
                let phase = match self.touched.iter_mut().find(|(touched, _)| *touched == id) {
                    Some((_, place)) => {
                        *place = (x, y);
                        touch_phase::MOVE
                    }
                    None => {
                        self.touched.push((id, (x, y)));
                        touch_phase::START
                    }
                };
                self.touches.touch(id, phase, x, y, &mut tick.events);
            }
            Step::Lift(id) => {
                if let Some(index) = self.touched.iter().position(|(touched, _)| *touched == id) {
                    let (_, (x, y)) = self.touched.swap_remove(index);
                    self.touches.touch(id, touch_phase::END, x, y, &mut tick.events);
                }
            }
            Step::Travel(travel) => {
                // Stepping takes a frame per frame stepped, as waiting does.
                if let Travel::Step(frames) = travel {
                    self.waiting = frames.saturating_sub(1);
                }
                tick.travel = Some(travel);
            }
        }
        tick
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VIEW: Viewport = Viewport { width: 1920, height: 1280 };

    #[test]
    fn steps_play_one_frame_each() {
        let steps = parse(&["key=Tab wait=2 click=480,320".into(), "shot=start quit".into()]).unwrap();
        let mut script = Script::new(steps);
        let frames: Vec<Tick> = (0..6).map(|_| script.tick(VIEW)).collect();
        // A release shares its frame with the next step; a wait counts that frame.
        assert_eq!(frames[0].events, [GameEvent::KeyDown("Tab".into())]);
        assert_eq!(frames[1].events, [GameEvent::KeyUp("Tab".into())]);
        assert!(frames[2].events.is_empty());
        assert_eq!(
            frames[3].events,
            [GameEvent::PointerMove { x: 960.0, y: 640.0 }, GameEvent::PointerDown { button: 0, x: 960.0, y: 640.0 }]
        );
        assert_eq!(frames[4].events, [GameEvent::PointerUp { button: 0, x: 960.0, y: 640.0 }]);
        assert_eq!(frames[4].shots, ["start"]);
        assert!(frames[5].quit && script.done());
    }

    #[test]
    fn parses_glides_with_positive_whole_frame_counts() {
        assert_eq!(parse(&["glide=-12.5,320,45".into()]).unwrap(), [Step::Glide { x: -12.5, y: 320.0, frames: 45 }]);
        for wrong in [
            "glide=1,2",
            "glide=1,2,0",
            "glide=1,2,-1",
            "glide=1,2,1.5",
            "glide=1,2,NaN",
            "glide=NaN,2,30",
            "glide=1,inf,30",
            "glide=1,2,4294967296",
            "glide=1,2,3,4",
        ] {
            assert!(parse(&[wrong.into()]).is_err(), "{wrong}");
        }
    }

    #[test]
    fn glides_ease_over_exact_frames_and_release_after_arrival() {
        let mut script =
            Script::new(parse(&["move=100,200 press=left glide=500,400,4 release=left shot=arrived".into()]).unwrap());
        assert_eq!(script.tick(VIEW).events, [GameEvent::PointerMove { x: 200.0, y: 400.0 }]);
        assert_eq!(script.tick(VIEW).events, [GameEvent::PointerDown { button: 0, x: 200.0, y: 400.0 }]);
        // Smaller distances at the ends, a larger one through the middle.
        for (x, y) in [(325.0, 462.5), (600.0, 600.0), (875.0, 737.5), (1000.0, 800.0)] {
            let frame = script.tick(VIEW);
            assert_eq!(frame.events, [GameEvent::PointerMove { x, y }]);
            assert!(frame.shots.is_empty());
            assert!(!script.done());
        }
        assert_eq!(script.tick(VIEW).events, [GameEvent::PointerUp { button: 0, x: 1000.0, y: 800.0 }]);
        assert_eq!(script.tick(VIEW).shots, ["arrived"]);
        assert!(script.done());
    }

    #[test]
    fn glides_follow_clicks_wheels_and_other_glides() {
        let mut script = Script::new(
            parse(&["click=100,200 glide=200,300,2 wheel=1,300,400 glide=500,600,2 glide=500,600,1".into()]).unwrap(),
        );
        script.tick(VIEW);
        assert_eq!(
            script.tick(VIEW).events,
            [GameEvent::PointerUp { button: 0, x: 200.0, y: 400.0 }, GameEvent::PointerMove { x: 300.0, y: 500.0 }]
        );
        assert_eq!(script.tick(VIEW).events, [GameEvent::PointerMove { x: 400.0, y: 600.0 }]);
        assert_eq!(script.tick(VIEW).events, [GameEvent::Wheel { delta: 1.0, x: 600.0, y: 800.0 }]);
        assert_eq!(script.tick(VIEW).events, [GameEvent::PointerMove { x: 800.0, y: 1000.0 }]);
        assert_eq!(script.tick(VIEW).events, [GameEvent::PointerMove { x: 1000.0, y: 1200.0 }]);
        assert!(!script.done());
        assert_eq!(script.tick(VIEW).events, [GameEvent::PointerMove { x: 1000.0, y: 1200.0 }]);
        assert!(script.done());
    }

    #[test]
    fn a_final_glide_finishes_only_when_it_reaches_its_destination() {
        for viewport in [VIEW, Viewport { width: 960, height: 640 }] {
            let mut script = Script::new(parse(&["glide=480,320,2".into()]).unwrap());
            assert_eq!(
                script.tick(viewport).events,
                [GameEvent::PointerMove { x: viewport.width as f32 * 0.25, y: viewport.height as f32 * 0.25 }]
            );
            assert!(!script.done());
            assert_eq!(
                script.tick(viewport).events,
                [GameEvent::PointerMove { x: viewport.width as f32 * 0.5, y: viewport.height as f32 * 0.5 }]
            );
            assert!(script.done());
            assert!(script.tick(viewport).events.is_empty());
        }
    }

    #[test]
    fn glides_between_distant_offscreen_positions_without_overflowing() {
        let viewport = Viewport { width: 960, height: 640 };
        let mut script = Script::new(parse(&["move=-3e38,0 glide=3e38,0,2".into()]).unwrap());
        script.tick(viewport);
        assert_eq!(script.tick(viewport).events, [GameEvent::PointerMove { x: 0.0, y: 0.0 }]);
        assert_eq!(script.tick(viewport).events, [GameEvent::PointerMove { x: 3e38, y: 0.0 }]);
        assert!(script.done());
    }

    #[test]
    fn parses_time_travel() {
        assert_eq!(
            parse(&["pause step step=3 rewind=30 replay=2.5 resume".into()]).unwrap(),
            [
                Step::Travel(Travel::Pause),
                Step::Travel(Travel::Step(1)),
                Step::Travel(Travel::Step(3)),
                Step::Travel(Travel::Rewind(30)),
                Step::Travel(Travel::Replay(2.5)),
                Step::Travel(Travel::Resume),
            ]
        );
        assert!(parse(&["rewind=back".into()]).is_err());
        assert!(parse(&["replay=0".into()]).is_err());
        let mut script = Script::new(parse(&["pause step=3 rewind=5".into()]).unwrap());
        assert_eq!(script.tick(VIEW).travel, Some(Travel::Pause));
        assert_eq!(script.tick(VIEW).travel, Some(Travel::Step(3)));
        assert_eq!(script.tick(VIEW).travel, None);
        assert_eq!(script.tick(VIEW).travel, None);
        assert_eq!(script.tick(VIEW).travel, Some(Travel::Rewind(5)));
    }

    #[test]
    fn parses_interface_actions() {
        assert_eq!(
            parse(&["ui=12,activate ui=0x1f,set_value,0.25 ui=7,4 ui=3,expand,".into()]).unwrap_err().to_string(),
            "ui takes a number as its value, not "
        );
        assert_eq!(
            parse(&["ui=12,activate ui=0x1f,set_value,0.25 ui=7,4 ui=0XFFFFFF,collapse".into()]).unwrap(),
            [
                Step::Ui { node: 12, action: ui_action::ACTIVATE, value: 0.0 },
                Step::Ui { node: 31, action: ui_action::SET_VALUE, value: 0.25 },
                Step::Ui { node: 7, action: ui_action::DECREMENT, value: 0.0 },
                Step::Ui { node: 0xffffff, action: ui_action::COLLAPSE, value: 0.0 },
            ]
        );
        let wrong = [
            "ui=12",
            "ui=0,focus",
            "ui=0x1000000,focus",
            "ui=abc,focus",
            "ui=12,poke",
            "ui=12,8",
            "ui=12,focus,1,2",
            "ui=12,set_value,loud",
            "ui=12,set_value,NaN",
        ];
        for wrong in wrong {
            assert!(parse(&[wrong.into()]).is_err(), "{wrong}");
        }
        // It takes a frame, as a key does.
        let mut script = Script::new(parse(&["ui=5,focus wait=1 ui=5,increment".into()]).unwrap());
        assert_eq!(script.tick(VIEW).events, [GameEvent::UiAction { node: 5, action: ui_action::FOCUS, value: 0.0 }]);
        assert!(script.tick(VIEW).events.is_empty());
        assert_eq!(
            script.tick(VIEW).events,
            [GameEvent::UiAction { node: 5, action: ui_action::INCREMENT, value: 0.0 }]
        );
        assert!(script.done());
    }

    #[test]
    fn rejects_unknown_steps() {
        assert!(parse(&["view=0,0,4,0,0.6,24".into()]).is_err());
        assert!(parse(&["press=sideways".into()]).is_err());
        assert_eq!(parse(&["key=KeyA key=Space".into()]).unwrap(), [Step::Key("a".into()), Step::Key(" ".into())]);
    }

    #[test]
    fn sends_the_games_own_events() {
        let mut steps = parse(&["game=Kick,1.5,-2 game=7 game=kick".into()]).unwrap();
        assert_eq!(steps[1], Step::Game { event: "7".into(), values: vec![], sent: Some((7, [0.0; 3])) });
        for wrong in ["game=", "game=Kick,fast", "game=Kick,1,2,3,4", "game=Kick,inf"] {
            assert!(parse(&[wrong.into()]).is_err(), "{wrong}");
        }
        // Names are the checkout's to give.
        let declared = |name: &str, values: &[f32]| -> Result<(u32, Vec<f32>)> {
            ensure!(name.eq_ignore_ascii_case("kick"), "no event {name}");
            Ok((3, values.to_vec()))
        };
        name_game_events(&mut steps, &declared).unwrap();
        let mut script = Script::new(steps);
        assert_eq!(script.tick(VIEW).events, [GameEvent::Game { event: 3, a: 1.5, b: -2.0, c: 0.0 }]);
        assert_eq!(script.tick(VIEW).events, [GameEvent::Game { event: 7, a: 0.0, b: 0.0, c: 0.0 }]);
        assert_eq!(script.tick(VIEW).events, [GameEvent::Game { event: 3, a: 0.0, b: 0.0, c: 0.0 }]);
        let mut unknown = parse(&["game=Jump".into()]).unwrap();
        let refused = format!("{:#}", name_game_events(&mut unknown, &declared).unwrap_err());
        assert_eq!(refused, "game=Jump: no event Jump");
    }

    #[test]
    fn parses_console_commands() {
        assert_eq!(
            parse(&["console=3,10 console=6".into()]).unwrap(),
            [Step::Console(3.0, 10.0), Step::Console(6.0, 0.0)]
        );
        for wrong in ["console=", "console=a", "console=1.5", "console=-1", "console=1,2,3"] {
            assert!(parse(&[wrong.into()]).is_err(), "{wrong}");
        }
        let mut script = Script::new(parse(&["console=3,10".into()]).unwrap());
        assert_eq!(script.tick(VIEW).events, [GameEvent::Console { command: 3.0, argument: 10.0 }]);
    }

    #[test]
    fn parses_overlay_lines() {
        let steps = parse(&["overlay=/timeline overlay=/diagnostics,high overlay-shot=panel".into()]).unwrap();
        assert_eq!(steps[..2], [Step::Overlay("/timeline".into()), Step::Overlay("/diagnostics high".into())]);
        assert!(parse(&["overlay=timeline".into()]).is_err() && parse(&["overlay-shot=".into()]).is_err());
        let mut script = Script::new(steps);
        assert_eq!(script.tick(VIEW).overlay, ["/timeline"]);
        assert_eq!(script.tick(VIEW).overlay, ["/diagnostics high"]);
        assert_eq!(script.tick(VIEW).overlay_shots, ["panel"]);
    }

    #[test]
    fn parses_bug_reports() {
        let steps = parse(&["report report=bug.penbug".into()]).unwrap();
        assert_eq!(steps, [Step::Report(PathBuf::new()), Step::Report(PathBuf::from("bug.penbug"))]);
        assert!(travels_back(&steps));
        let mut script = Script::new(steps);
        assert_eq!(script.tick(VIEW).reports, [PathBuf::new()]);
    }

    #[test]
    fn parses_tunes() {
        assert_eq!(
            parse(&["tune=camera.zoom_speed=0.004 tune=wind.least_wait=-2".into()]).unwrap(),
            [Step::Tune("camera.zoom_speed".into(), 0.004), Step::Tune("wind.least_wait".into(), -2.0)]
        );
        for wrong in ["tune=camera.zoom_speed", "tune==1", "tune=a=fast", "tune=a=inf"] {
            assert!(parse(&[wrong.into()]).is_err(), "{wrong}");
        }
        // The game's id for the name, in a frame of its own.
        let mut script = Script::new(parse(&["tune=camera.zoom_speed=0.004 wait=1".into()]).unwrap());
        assert_eq!(script.tick(VIEW).events, [GameEvent::Tune { index: 11_175_412, value: 0.004 }]);
        assert!(script.tick(VIEW).events.is_empty());
    }

    #[test]
    fn keys_that_type_send_their_text() {
        let mut script = Script::new(parse(&["key=KeyA type=é! key=Enter".into()]).unwrap());
        let text = |character: char| GameEvent::TextInput { character: u32::from(character), composing: 0 };
        assert_eq!(script.tick(VIEW).events, [GameEvent::KeyDown("a".into()), text('a')]);
        assert_eq!(script.tick(VIEW).events, [GameEvent::KeyUp("a".into()), GameEvent::KeyDown("é".into()), text('é')]);
        assert_eq!(script.tick(VIEW).events, [GameEvent::KeyUp("é".into()), GameEvent::KeyDown("!".into()), text('!')]);
        assert_eq!(script.tick(VIEW).events, [GameEvent::KeyUp("!".into()), GameEvent::KeyDown("Enter".into())]);
    }

    #[test]
    fn input_method_and_mouse_motion_steps() {
        assert_eq!(
            parse(&["compose=かん,1 compose= commit=漢字 motion=3,-2.5".into()]).unwrap(),
            [
                Step::Compose("かん".into(), Some(1)),
                Step::Compose(String::new(), None),
                Step::Commit("漢字".into()),
                Step::Motion(3.0, -2.5),
            ]
        );
        let mut script = Script::new(parse(&["compose=かん,1 commit=漢 motion=3,-2.5".into()]).unwrap());
        assert_eq!(
            script.tick(VIEW).events,
            [
                GameEvent::Composition { length: 2, cursor: 1 },
                GameEvent::TextInput { character: u32::from('か'), composing: 1 },
                GameEvent::TextInput { character: u32::from('ん'), composing: 1 },
            ]
        );
        assert_eq!(
            script.tick(VIEW).events,
            [
                GameEvent::Composition { length: 0, cursor: 0 },
                GameEvent::TextInput { character: u32::from('漢'), composing: 0 }
            ]
        );
        assert_eq!(script.tick(VIEW).events, [GameEvent::MouseMotion { dx: 3.0, dy: -2.5 }]);
        assert!(parse(&["motion=1".into()]).is_err());
    }

    #[test]
    fn a_scripted_gamepad_plugs_in_moves_and_leaves() {
        let steps = parse(&["pad=south pad=left_x,0.1 pad=left_x,-1 pad=south,0 pad=disconnect".into()]).unwrap();
        assert_eq!(steps[0], Step::Pad(PadStep::Button(gamepad_button::SOUTH as usize, 1.0)));
        let mut script = Script::new(steps);
        assert_eq!(
            script.tick(VIEW).events,
            [
                GameEvent::GamepadConnected { pad: 0, standard: 1 },
                GameEvent::GamepadButton { button: gamepad_button::SOUTH, pad: 0, value: 1.0, pressed: 1 },
            ]
        );
        // Inside the dead zone: nothing.
        assert!(script.tick(VIEW).events.is_empty());
        assert_eq!(
            script.tick(VIEW).events,
            [GameEvent::GamepadAxis { axis: gamepad_axis::LEFT_X, pad: 0, value: -1.0 }]
        );
        assert_eq!(
            script.tick(VIEW).events,
            [GameEvent::GamepadButton { button: gamepad_button::SOUTH, pad: 0, value: 0.0, pressed: 0 }]
        );
        assert_eq!(
            script.tick(VIEW).events,
            [
                GameEvent::GamepadAxis { axis: gamepad_axis::LEFT_X, pad: 0, value: 0.0 },
                GameEvent::GamepadDisconnected { pad: 0 },
            ]
        );
        for wrong in ["pad=", "pad=south,2", "pad=left_x", "pad=left_x,-1.5", "pad=wheel,1", "pad=connect,1"] {
            assert!(parse(&[wrong.into()]).is_err(), "{wrong}");
        }
    }

    #[test]
    fn scripted_touches_point_pan_and_pinch() {
        let steps =
            parse(&["touch=7,100,100 touch=8,200,100 touch=7,80,100 touch=8,220,100 lift=7 lift=8".into()]).unwrap();
        let mut script = Script::new(steps);
        // Page pixels, scaled to the viewport as clicks are.
        assert_eq!(
            script.tick(VIEW).events,
            [
                GameEvent::Touch { id: 0, x: 200.0, y: 200.0, phase: touch_phase::START },
                GameEvent::PointerMove { x: 200.0, y: 200.0 },
                GameEvent::PointerDown { button: 0, x: 200.0, y: 200.0 },
            ]
        );
        assert_eq!(
            script.tick(VIEW).events,
            [
                GameEvent::Touch { id: 1, x: 400.0, y: 200.0, phase: touch_phase::START },
                GameEvent::PointerUp { button: 0, x: 200.0, y: 200.0 },
            ]
        );
        // One touch moving out moves their centre and spreads them: a pan
        // and a pinch.
        assert_eq!(
            script.tick(VIEW).events,
            [
                GameEvent::Touch { id: 0, x: 160.0, y: 200.0, phase: touch_phase::MOVE },
                GameEvent::Pan { touches: 2, dx: -20.0, dy: 0.0 },
                GameEvent::Pinch { x: 280.0, y: 200.0, scale: 1.2 },
            ]
        );
        let spreading = script.tick(VIEW).events;
        assert_eq!(spreading[1], GameEvent::Pan { touches: 2, dx: 20.0, dy: 0.0 });
        assert!(
            matches!(spreading[2], GameEvent::Pinch { x: 300.0, scale, .. } if (scale - 140.0 / 120.0).abs() < 1e-6)
        );
        assert_eq!(script.tick(VIEW).events, [GameEvent::Touch { id: 0, x: 160.0, y: 200.0, phase: touch_phase::END }]);
        assert_eq!(script.tick(VIEW).events, [GameEvent::Touch { id: 1, x: 440.0, y: 200.0, phase: touch_phase::END }]);
        assert!(parse(&["touch=1,2".into()]).is_err());
        assert!(parse(&["lift=x".into()]).is_err());
    }
}
