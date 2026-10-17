//! The input lane's events on the game itself (`dist/gdev`): a gamepad,
//! touches (with the pointer they drive, their pans and pinches), text and
//! an input method's composition, the locked mouse's motion, and actions
//! the host maps from keys and the pad, sent as a session sends them. They
//! are input like any other: the same input gives the same packets, a
//! rewind among them runs again exactly, and a recording replays them.
use pen_protocol::boundary::{EVENT_FLOATS, FRAME_HEADER_FLOATS, event_tags, whole};
use pen_protocol::{GameEvent, input_device};
use pen_runtime::actions::Actions;

use super::*;

/// Every kind of event, in play, with moments (every 30 frames) among them.
const DEVICES: &str = "wait=20 key=Tab wait=20 pad=connect pad=south wait=2 pad=south,0 \
    pad=left_x,0.8 wait=10 pad=left_x,0 touch=1,400,300 touch=2,560,300 touch=1,360,300 \
    touch=2,600,300 lift=1 lift=2 wait=5 compose=かん,1 commit=漢 type=gnorp key=Space \
    motion=12,-4 pad=disconnect wait=30";

/// Jump on space or the south button, and move.x on the left stick, as
/// the player's bindings keep them.
fn actions() -> Actions {
    Actions::load(
        r#"{"actions": [
            {"index": 0, "name": "jump", "kind": "button", "defaults": [[0, 32, 1], [2, 0, 1]]},
            {"index": 1, "name": "move.x", "kind": "axis", "defaults": [[3, 0, 1]]}
        ]}"#,
    )
}

impl Game {
    /// Plays a script as `play` does, `actions` mapping each frame's
    /// events as a session's do; returns the events sent.
    fn play_mapped(&mut self, steps: &str, actions: &mut Actions) -> Vec<GameEvent> {
        let mut script = Script::new(script::parse(&[steps.to_owned()]).unwrap());
        let mut sent = Vec::new();
        let mut time = 0.0;
        while !script.done() {
            let tick = script.tick(VIEWPORT);
            let mut events = tick.events;
            actions.map(&mut events);
            sent.extend(events.iter().cloned());
            let frame = Frame { timestamp: time, viewport: VIEWPORT, events };
            time += 1.0 / 60.0;
            let input = self.runtime.input(&frame).unwrap();
            self.run(input, &tick.saves, &tick.loads);
        }
        sent
    }
}

/// The tags of the events a recording's frames carry.
fn recorded_tags(recording: &Recording) -> Vec<u32> {
    let mut tags = Vec::new();
    for frame in &recording.frames {
        let count = whole(frame.input[3]).unwrap() as usize;
        let events = frame.input[FRAME_HEADER_FLOATS..].as_chunks::<EVENT_FLOATS>().0;
        tags.extend(events.iter().take(count).map(|event| whole(event[0]).unwrap()));
    }
    tags
}

#[test]
fn gamepads_touches_text_and_actions_record_and_replay() {
    let Some(directory) = bundle() else { return };
    let mut game = Game::start(&directory, None);
    let sent = game.play_mapped(DEVICES, &mut actions());
    let has = |wanted: &dyn Fn(&GameEvent) -> bool, what: &str| assert!(sent.iter().any(wanted), "no {what}");
    has(&|event| matches!(event, GameEvent::GamepadConnected { pad: 0, standard: 1 }), "pad connecting");
    has(&|event| matches!(event, GameEvent::GamepadButton { pressed: 1, .. }), "pad button");
    has(&|event| matches!(event, GameEvent::GamepadAxis { value, .. } if *value > 0.7), "stick");
    has(&|event| matches!(event, GameEvent::GamepadDisconnected { pad: 0 }), "pad leaving");
    has(&|event| matches!(event, GameEvent::Touch { id: 1, .. }), "second touch");
    has(&|event| matches!(event, GameEvent::Pan { touches: 2, .. }), "pan");
    has(&|event| matches!(event, GameEvent::Pinch { scale, .. } if *scale > 1.0), "pinch");
    has(&|event| matches!(event, GameEvent::Composition { length: 2, cursor: 1 }), "composition");
    has(&|event| matches!(event, GameEvent::TextInput { character: 0x6f22, composing: 0 }), "committed text");
    has(&|event| matches!(event, GameEvent::TextInput { character: 0x67, composing: 0 }), "typed text");
    has(&|event| matches!(event, GameEvent::MouseMotion { dx: 12.0, dy: -4.0 }), "mouse motion");
    let jumped = |device| move |event: &GameEvent| matches!(event, GameEvent::Action { action: 0, value: 1.0, device: by } if *by == device);
    has(&jumped(input_device::GAMEPAD), "jump from the pad");
    has(&jumped(input_device::KEYBOARD), "jump from space");
    has(&|event| matches!(event, GameEvent::Action { action: 1, device: input_device::GAMEPAD, .. }), "move.x");
    let first = game.pass.clone();
    assert!(first.packets.len() > 60);
    // The same input gives the same packets.
    let mut second = Game::start(&directory, None);
    second.play_mapped(DEVICES, &mut actions());
    assert_eq!(second.pass.packets, first.packets);
    second.runtime.close();
    // The recording holds them, and rewinds among them run again exactly.
    let tags = recorded_tags(&game.recording());
    for tag in
        [event_tags::GAMEPAD_BUTTON, event_tags::TOUCH, event_tags::PINCH, event_tags::COMPOSITION, event_tags::ACTION]
    {
        assert!(tags.contains(&tag), "the recording lacks tag {tag}");
    }
    for moment in [MOMENT_INTERVAL + 1, 2 * MOMENT_INTERVAL + 5] {
        game.travel(moment, &first);
    }
    // A recording replays the same packets.
    assert_eq!(replay(&directory, game.recording()), game.pass.packets);
    game.runtime.close();
}
