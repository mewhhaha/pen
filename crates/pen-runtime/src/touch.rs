//! Touches as the game's events, worked out the same way on every host.
//! A host passes each touch as it gets it (winit's Touch natively, pointer
//! events whose pointerType is touch in a browser, a script's steps
//! headless), by its own id; [`Touches`] numbers them from 0, the lowest
//! free number first, and sends Touch events. The first touch down also
//! moves and presses the pointer (button 0), so a game that only takes the
//! pointer plays on a touch screen; a second touch lets the pointer go until
//! every touch is lifted. Two or more touches moving together are gestures:
//! Pan, as their centre moves, and Pinch, as they spread apart or close.
//! Each starts once it has moved [`SLOP`] pixels, and then reports every
//! change, the first report included what moved before it started.
use pen_protocol::boundary::MAX_TOUCHES;
use pen_protocol::{GameEvent, touch_phase};

/// How far touches' centre moves, or how much their spread changes, in
/// window pixels, before they pan or pinch: less is fingers settling.
pub const SLOP: f32 = 8.0;

/// A touch down: the host's id for it and where it is.
#[derive(Clone, Copy, Debug)]
struct Down {
    key: u64,
    x: f32,
    y: f32,
}

/// Two or more touches: where their centre and spread were when last
/// reported, where they started, and whether they pan or pinch yet.
#[derive(Clone, Copy, Debug)]
struct Gesture {
    centre: (f32, f32),
    spread: f32,
    from: (f32, f32),
    spread_from: f32,
    panning: bool,
    pinching: bool,
}

#[derive(Debug, Default)]
pub struct Touches {
    slots: Vec<Option<Down>>,
    /// The number of the touch that drives the pointer.
    pointer: Option<usize>,
    /// A second touch let the pointer go: it waits until every touch is up.
    held_back: bool,
    gesture: Option<Gesture>,
}

/// The touches' centre and their mean distance from it.
fn centre_and_spread(downs: &[Down]) -> ((f32, f32), f32) {
    let count = downs.len() as f32;
    let x = downs.iter().map(|down| down.x).sum::<f32>() / count;
    let y = downs.iter().map(|down| down.y).sum::<f32>() / count;
    let spread = downs.iter().map(|down| (down.x - x).hypot(down.y - y)).sum::<f32>() / count;
    ((x, y), spread)
}

fn number(slot: usize) -> u32 {
    u32::try_from(slot).expect("fewer than MAX_TOUCHES touches")
}

impl Touches {
    fn slot(&self, key: u64) -> Option<usize> {
        self.slots.iter().position(|slot| slot.is_some_and(|down| down.key == key))
    }

    fn downs(&self) -> Vec<Down> {
        self.slots.iter().flatten().copied().collect()
    }

    /// Touch `key` is at (`x`, `y`) in window pixels, in `phase`
    /// (`pen_protocol::touch_phase`): its events, and the pointer's and the
    /// gestures' it makes. A position that is not a number leaves it out.
    pub fn touch(&mut self, key: u64, phase: u32, x: f32, y: f32, events: &mut Vec<GameEvent>) {
        if !(x.is_finite() && y.is_finite()) {
            return;
        }
        match (phase, self.slot(key)) {
            (touch_phase::START, None) => self.start(key, x, y, events),
            (touch_phase::START | touch_phase::MOVE, Some(slot)) => self.moved(slot, x, y, events),
            (touch_phase::END | touch_phase::CANCEL, Some(slot)) => self.lifted(slot, phase, x, y, events),
            _ => {}
        }
    }

    /// Every touch is taken away (the window lost the focus).
    pub fn cancel(&mut self, events: &mut Vec<GameEvent>) {
        // Lifting the last empties the slots.
        for slot in 0..self.slots.len() {
            if let Some(down) = self.slots.get(slot).copied().flatten() {
                self.lifted(slot, touch_phase::CANCEL, down.x, down.y, events);
            }
        }
    }

    fn start(&mut self, key: u64, x: f32, y: f32, events: &mut Vec<GameEvent>) {
        let slot = self.slots.iter().position(Option::is_none).unwrap_or(self.slots.len());
        if slot >= MAX_TOUCHES {
            return;
        }
        if slot == self.slots.len() {
            self.slots.push(None);
        }
        let first = self.slots.iter().all(Option::is_none);
        self.slots[slot] = Some(Down { key, x, y });
        events.push(GameEvent::Touch { id: number(slot), x, y, phase: touch_phase::START });
        if first && !self.held_back {
            self.pointer = Some(slot);
            events.push(GameEvent::PointerMove { x, y });
            events.push(GameEvent::PointerDown { button: 0, x, y });
        } else if let Some(pointer) = self.pointer.take() {
            let down = self.slots[pointer].expect("the pointer's touch");
            events.push(GameEvent::PointerUp { button: 0, x: down.x, y: down.y });
            self.held_back = true;
        }
        self.regroup();
    }

    fn moved(&mut self, slot: usize, x: f32, y: f32, events: &mut Vec<GameEvent>) {
        let down = self.slots[slot].as_mut().expect("a touch down");
        if (down.x, down.y) == (x, y) {
            return;
        }
        (down.x, down.y) = (x, y);
        events.push(GameEvent::Touch { id: number(slot), x, y, phase: touch_phase::MOVE });
        if self.pointer == Some(slot) {
            events.push(GameEvent::PointerMove { x, y });
        }
        let downs = self.downs();
        let Some(gesture) = &mut self.gesture else { return };
        let (centre, spread) = centre_and_spread(&downs);
        let travelled = (centre.0 - gesture.from.0).hypot(centre.1 - gesture.from.1);
        gesture.panning |= travelled >= SLOP;
        gesture.pinching |= (spread - gesture.spread_from).abs() >= SLOP;
        if gesture.panning && centre != gesture.centre {
            events.push(GameEvent::Pan {
                touches: u32::try_from(downs.len()).expect("fewer than MAX_TOUCHES touches"),
                dx: centre.0 - gesture.centre.0,
                dy: centre.1 - gesture.centre.1,
            });
            gesture.centre = centre;
        }
        if gesture.pinching && spread != gesture.spread && gesture.spread > 0.0 {
            events.push(GameEvent::Pinch { x: centre.0, y: centre.1, scale: spread / gesture.spread });
            gesture.spread = spread;
        }
    }

    fn lifted(&mut self, slot: usize, phase: u32, x: f32, y: f32, events: &mut Vec<GameEvent>) {
        self.slots[slot] = None;
        events.push(GameEvent::Touch { id: number(slot), x, y, phase });
        if self.pointer == Some(slot) {
            self.pointer = None;
            events.push(GameEvent::PointerUp { button: 0, x, y });
        }
        if self.slots.iter().all(Option::is_none) {
            self.slots.clear();
            self.held_back = false;
        }
        self.regroup();
    }

    /// The touches down changed: two or more are measured from where they
    /// are now, keeping whether they already pan or pinch.
    fn regroup(&mut self) {
        let downs = self.downs();
        if downs.len() < 2 {
            self.gesture = None;
            return;
        }
        let (centre, spread) = centre_and_spread(&downs);
        let (panning, pinching) = self.gesture.map_or((false, false), |gesture| (gesture.panning, gesture.pinching));
        self.gesture = Some(Gesture { centre, spread, from: centre, spread_from: spread, panning, pinching });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use touch_phase::{CANCEL, END, MOVE, START};

    #[test]
    fn one_touch_is_the_pointer() {
        let mut touches = Touches::default();
        let mut events = Vec::new();
        touches.touch(900, START, 10.0, 20.0, &mut events);
        touches.touch(900, MOVE, 12.0, 20.0, &mut events);
        touches.touch(900, MOVE, 12.0, 20.0, &mut events);
        touches.touch(900, END, 12.0, 21.0, &mut events);
        assert_eq!(
            events,
            [
                GameEvent::Touch { id: 0, x: 10.0, y: 20.0, phase: START },
                GameEvent::PointerMove { x: 10.0, y: 20.0 },
                GameEvent::PointerDown { button: 0, x: 10.0, y: 20.0 },
                GameEvent::Touch { id: 0, x: 12.0, y: 20.0, phase: MOVE },
                GameEvent::PointerMove { x: 12.0, y: 20.0 },
                GameEvent::Touch { id: 0, x: 12.0, y: 21.0, phase: END },
                GameEvent::PointerUp { button: 0, x: 12.0, y: 21.0 },
            ]
        );
        // A touch the game never heard of, or that is not a number, is left out.
        events.clear();
        touches.touch(5, MOVE, 1.0, 1.0, &mut events);
        touches.touch(5, START, f32::NAN, 1.0, &mut events);
        assert!(events.is_empty());
    }

    #[test]
    fn a_second_touch_lets_the_pointer_go_until_all_are_lifted() {
        let mut touches = Touches::default();
        let mut events = Vec::new();
        touches.touch(1, START, 0.0, 0.0, &mut events);
        events.clear();
        touches.touch(2, START, 100.0, 0.0, &mut events);
        assert_eq!(
            events,
            [
                GameEvent::Touch { id: 1, x: 100.0, y: 0.0, phase: START },
                GameEvent::PointerUp { button: 0, x: 0.0, y: 0.0 }
            ]
        );
        events.clear();
        // Lifting the first leaves the second as touch 1, and no pointer.
        touches.touch(1, END, 0.0, 0.0, &mut events);
        touches.touch(3, START, 50.0, 50.0, &mut events);
        assert_eq!(
            events,
            [
                GameEvent::Touch { id: 0, x: 0.0, y: 0.0, phase: END },
                GameEvent::Touch { id: 0, x: 50.0, y: 50.0, phase: START },
            ]
        );
        events.clear();
        touches.cancel(&mut events);
        assert_eq!(
            events,
            [
                GameEvent::Touch { id: 0, x: 50.0, y: 50.0, phase: CANCEL },
                GameEvent::Touch { id: 1, x: 100.0, y: 0.0, phase: CANCEL },
            ]
        );
        // All lifted: the next first touch is the pointer again.
        events.clear();
        touches.touch(4, START, 5.0, 5.0, &mut events);
        assert_eq!(events.len(), 3);
        assert_eq!(events[2], GameEvent::PointerDown { button: 0, x: 5.0, y: 5.0 });
    }

    #[test]
    fn two_touches_pan_once_past_the_slop() {
        let mut touches = Touches::default();
        let mut events = Vec::new();
        touches.touch(1, START, 100.0, 100.0, &mut events);
        touches.touch(2, START, 200.0, 100.0, &mut events);
        events.clear();
        // Both move 3 pixels down: the centre moves 1.5 then 3, under the slop.
        touches.touch(1, MOVE, 100.0, 103.0, &mut events);
        touches.touch(2, MOVE, 200.0, 103.0, &mut events);
        assert!(events.iter().all(|event| matches!(event, GameEvent::Touch { .. })));
        events.clear();
        // Past it, the pan reports all it moved, and then each move.
        touches.touch(1, MOVE, 100.0, 113.0, &mut events);
        touches.touch(2, MOVE, 200.0, 113.0, &mut events);
        let pans: Vec<_> = events.iter().filter(|event| matches!(event, GameEvent::Pan { .. })).collect();
        assert_eq!(
            pans,
            [&GameEvent::Pan { touches: 2, dx: 0.0, dy: 8.0 }, &GameEvent::Pan { touches: 2, dx: 0.0, dy: 5.0 }]
        );
        assert!(!events.iter().any(|event| matches!(event, GameEvent::Pinch { .. })));
    }

    #[test]
    fn two_touches_spreading_pinch() {
        let mut touches = Touches::default();
        let mut events = Vec::new();
        touches.touch(1, START, 100.0, 100.0, &mut events);
        touches.touch(2, START, 140.0, 100.0, &mut events);
        events.clear();
        // Each moves 10 pixels out: the spread grows from 20 to 25, then 30.
        touches.touch(1, MOVE, 90.0, 100.0, &mut events);
        touches.touch(2, MOVE, 150.0, 100.0, &mut events);
        let pinches: Vec<_> = events.iter().filter(|event| matches!(event, GameEvent::Pinch { .. })).cloned().collect();
        assert_eq!(pinches, [GameEvent::Pinch { x: 120.0, y: 100.0, scale: 1.5 }]);
        // The centre moved 5 then back: under the slop, no pan.
        assert!(!events.iter().any(|event| matches!(event, GameEvent::Pan { .. })));
        events.clear();
        touches.touch(1, MOVE, 105.0, 100.0, &mut events);
        let Some(GameEvent::Pinch { scale, .. }) = events.iter().find(|event| matches!(event, GameEvent::Pinch { .. }))
        else {
            panic!("no pinch in {events:?}")
        };
        assert!((scale - 22.5 / 30.0).abs() < 1e-6);
    }

    #[test]
    fn cancelling_lifts_every_touch_whatever_their_numbers() {
        let mut touches = Touches::default();
        let mut events = Vec::new();
        touches.touch(1, START, 0.0, 0.0, &mut events);
        touches.touch(2, START, 1.0, 0.0, &mut events);
        touches.touch(2, END, 1.0, 0.0, &mut events);
        events.clear();
        touches.cancel(&mut events);
        assert_eq!(events, [GameEvent::Touch { id: 0, x: 0.0, y: 0.0, phase: CANCEL }]);
    }

    #[test]
    fn touches_past_the_limit_are_left_out() {
        let mut touches = Touches::default();
        let mut events = Vec::new();
        for key in 0..=MAX_TOUCHES as u64 {
            touches.touch(key, START, 1.0, 1.0, &mut events);
        }
        let started = events.iter().filter(|event| matches!(event, GameEvent::Touch { .. })).count();
        assert_eq!(started, MAX_TOUCHES);
    }
}
