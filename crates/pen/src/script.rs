//! Scripted input, in the steps gdev's browser tooling takes
//! (`scripts/page.ts`): `wait=N` frames, `key=KEY` taps a key, `down=KEY` /
//! `up=KEY` hold and release it, `type=TEXT`, `move=X,Y`, `click=X,Y`,
//! `press=BUTTON` / `release=BUTTON` (left, middle or right),
//! `drag=X,Y,X,Y`, `wheel=DELTA[,X,Y]`, `shot=NAME` (a PNG in the shots
//! directory), `save=FILE` / `load=FILE` (quicksaves), and `quit`.
//!
//! `ui=ID,ACTION[,VALUE]` does what assistive technology does to a node of
//! the game's interface (see `accessibility`): ID is the node's id, decimal
//! or 0x hex, and ACTION one of `focus`, `blur`, `activate`, `increment`,
//! `decrement`, `set_value` (to VALUE), `expand` and `collapse`, or its code.
//! It reaches the game as a `UiAction` event, as the window's adapter sends it.
//!
//! `tune=NAME=VALUE` sets a live-tunable constant (the checkout's
//! `src/tunables.json`, `overlay::tunable_id`), as the overlay's Tunables
//! panel and `/tune` do: a Tune event, in the frame it runs in.
//!
//! `console=CODE[,ARGUMENT]` sends the game a console command by its number
//! (the checkout's `commands.ts`), as the console's `/command N [ARG]` does:
//! `console=3,10` hires ten gnorps in gdev.
//!
//! Time travel has steps too: `pause` and `resume` the game, `step[=N]`
//! runs N frames (one by default, one per frame of the script, as `wait`
//! waits) and stays paused, `rewind=N` goes back N
//! frames (paused there), and `replay=SECONDS` plays the last SECONDS of
//! recorded input again. While paused, steps still take one call each (a
//! `wait` counts presented frames, a `shot` captures the paused frame), and
//! input waits for the next frame that runs.
//!
//! Coordinates are page pixels of a page 960 wide, as the browser tooling's
//! 960 x 640 page has them; they scale with the viewport's width, so a
//! 1920 x 1280 viewport sees them at twice the size.
use std::collections::VecDeque;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail, ensure};
use pen_protocol::boundary::MAX_UI_ID;
use pen_protocol::{GameEvent, Viewport, ui_action};

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
    Click(f32, f32),
    Press(u32),
    Release(u32),
    Drag(f32, f32, f32, f32),
    Wheel(f32, Option<(f32, f32)>),
    Shot(String),
    Save(PathBuf),
    Load(PathBuf),
    Quit,
    Travel(Travel),
    /// An action on an interface node: its id, the action's code and a value.
    Ui { node: u32, action: u32, value: f32 },
    /// A tunable, by name, and its new value.
    Tune(String, f32),
    /// A console command's number and argument.
    Console(f32, f32),
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
    .filter(|node| (1..=MAX_UI_ID as u32).contains(node))
    .with_context(|| format!("ui takes a node id from 1 to {MAX_UI_ID} (or 0x hex), not {target}"))?;
    let action = match ui_action::NAMES.iter().position(|name| name == action) {
        Some(code) => code as u32,
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
            "pause" => Step::Travel(Travel::Pause),
            "resume" => Step::Travel(Travel::Resume),
            "step" => Step::Travel(Travel::Step(if value.is_empty() {
                1
            } else {
                value.parse().with_context(|| format!("step takes frames, not {value}"))?
            })),
            "rewind" => {
                Step::Travel(Travel::Rewind(value.parse().with_context(|| format!("rewind takes frames, not {value}"))?))
            }
            "replay" => {
                let seconds: f64 = value.parse().with_context(|| format!("replay takes seconds, not {value}"))?;
                ensure!(seconds > 0.0, "replay takes a positive number of seconds");
                Step::Travel(Travel::Replay(seconds))
            }
            "view" | "over" | "reload" | "storage" => {
                return Err(anyhow!("{name}= needs the browser tooling (`just drive`)"));
            }
            _ => return Err(anyhow!("unknown script step {word}")),
        });
    }
    Ok(steps)
}

/// Whether steps go back in time (and so need a history kept).
pub fn travels_back(steps: &[Step]) -> bool {
    steps.iter().any(|step| matches!(step, Step::Travel(Travel::Rewind(_) | Travel::Replay(_))))
}

/// What one frame of the script does.
#[derive(Debug, Default)]
pub struct Tick {
    pub events: Vec<GameEvent>,
    pub shots: Vec<String>,
    pub saves: Vec<PathBuf>,
    pub loads: Vec<PathBuf>,
    pub quit: bool,
    /// Time travel, before the frame runs.
    pub travel: Option<Travel>,
}

/// Plays steps one frame at a time: each input step takes a frame, a tap
/// or click releases on the next, and a drag moves over several.
pub struct Script {
    steps: VecDeque<Step>,
    /// Frames of events already worked out, from the next frame on.
    queued: VecDeque<Vec<GameEvent>>,
    waiting: u32,
    pointer: (f32, f32),
}

impl Script {
    pub fn new(steps: Vec<Step>) -> Script {
        Script { steps: steps.into(), queued: VecDeque::new(), waiting: 0, pointer: (0.0, 0.0) }
    }

    pub fn done(&self) -> bool {
        self.steps.is_empty() && self.queued.is_empty() && self.waiting == 0
    }

    fn queue(&mut self, frame: usize, event: GameEvent) {
        while self.queued.len() <= frame {
            self.queued.push_back(Vec::new());
        }
        self.queued[frame].push(event);
    }

    pub fn tick(&mut self, viewport: Viewport) -> Tick {
        let scale = viewport.width as f32 / PAGE_WIDTH;
        let mut tick = Tick { events: self.queued.pop_front().unwrap_or_default(), ..Tick::default() };
        if !self.queued.is_empty() {
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
                tick.events.push(GameEvent::KeyDown(key.clone()));
                self.queue(0, GameEvent::KeyUp(key));
            }
            Step::Down(key) => tick.events.push(GameEvent::KeyDown(key)),
            Step::Up(key) => tick.events.push(GameEvent::KeyUp(key)),
            Step::Type(text) => {
                for (index, character) in text.chars().enumerate() {
                    let key = character.to_string();
                    if index == 0 {
                        tick.events.push(GameEvent::KeyDown(key.clone()));
                    } else {
                        self.queue(index - 1, GameEvent::KeyDown(key.clone()));
                    }
                    self.queue(index, GameEvent::KeyUp(key));
                }
            }
            Step::Move(x, y) => {
                self.pointer = at(x, y);
                let (x, y) = self.pointer;
                tick.events.push(GameEvent::PointerMove { x, y });
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
            Step::Save(path) => tick.saves.push(path),
            Step::Load(path) => tick.loads.push(path),
            Step::Quit => tick.quit = true,
            Step::Ui { node, action, value } => tick.events.push(GameEvent::UiAction { node, action, value }),
            Step::Tune(name, value) => {
                tick.events.push(GameEvent::Tune { index: crate::overlay::tunable_id(&name), value });
            }
            Step::Console(command, argument) => tick.events.push(GameEvent::Console { command, argument }),
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
            [
                GameEvent::PointerMove { x: 960.0, y: 640.0 },
                GameEvent::PointerDown { button: 0, x: 960.0, y: 640.0 }
            ]
        );
        assert_eq!(frames[4].events, [GameEvent::PointerUp { button: 0, x: 960.0, y: 640.0 }]);
        assert_eq!(frames[4].shots, ["start"]);
        assert!(frames[5].quit && script.done());
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
    fn rejects_browser_only_steps() {
        assert!(parse(&["view=0,0,4,0,0.6,24".into()]).is_err());
        assert!(parse(&["press=sideways".into()]).is_err());
        assert_eq!(parse(&["key=KeyA key=Space".into()]).unwrap(), [Step::Key("a".into()), Step::Key(" ".into())]);
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
}
