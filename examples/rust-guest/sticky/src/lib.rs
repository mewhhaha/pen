//! Sticky arrows, a pen mod (pen's docs/mods.md) for games steered with
//! the arrow keys: a tapped arrow stays held until it is tapped again or
//! another arrow is, so a player who cannot hold a key can still steer.
//!
//! It sees each frame's events (`observe`) and adds keys (`input`): when
//! the player lets go of the held arrow, it presses it again after them;
//! when another arrow goes down, it lets go of the held one. The game gets
//! these as it gets the player's, and input recordings keep them.
//!
//! Unlike a game's `main`, a mod's `frame` takes and returns arrays, which
//! Rust can export directly: the module needs no adapter, only the
//! `blot:abi` manifest below and its memory exported as `blot:memory`
//! (build.rs).
use pen_protocol::boundary::{EVENT_FLOATS, event_tags, key_code};

/// The guest ABI's manifest: version 2, one function, `frame: Array F32 ->
/// Array F32`, no constants.
#[cfg(target_arch = "wasm32")]
#[unsafe(link_section = "blot:abi")]
#[used]
static MANIFEST: [u8; 11] = [0x02, 0x01, 0x05, b'f', b'r', b'a', b'm', b'e', 0x06, 0x06, 0x00];

const ARROWS: [&str; 4] = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"];

#[derive(Debug, Default)]
pub struct Sticky {
    /// The arrow it holds down, by its key code.
    held: Option<f32>,
}

impl Sticky {
    /// The events to add after a frame's own (`events`, five lanes each).
    pub fn frame(&mut self, events: &[f32]) -> Vec<[f32; EVENT_FLOATS]> {
        let key = |tag: u32, code: f32| [tag as f32, code, 0.0, 0.0, 0.0];
        let mut added = Vec::new();
        for event in events.chunks_exact(EVENT_FLOATS) {
            let (tag, code) = (event[0] as u32, event[1]);
            if !ARROWS.iter().any(|arrow| key_code(arrow) == code) {
                continue;
            }
            match tag {
                // Tapped again, it goes when the player lets go.
                event_tags::KEY_DOWN if self.held == Some(code) => self.held = None,
                event_tags::KEY_DOWN => {
                    if let Some(other) = self.held.replace(code) {
                        added.push(key(event_tags::KEY_UP, other));
                    }
                }
                event_tags::KEY_UP if self.held == Some(code) => added.push(key(event_tags::KEY_DOWN, code)),
                _ => {}
            }
        }
        added
    }
}

/// The answer to a request (the protocol's lanes: version, frame, elapsed,
/// width, height, event count, events).
pub fn answer(sticky: &mut Sticky, request: &[f32]) -> Vec<f32> {
    let count = request.get(5).map_or(0, |count| *count as usize);
    let events = request.get(6..6 + count * EVENT_FLOATS).unwrap_or(&[]);
    let added = sticky.frame(events);
    let mut answer = vec![1.0, added.len() as f32];
    answer.extend(added.iter().flatten());
    answer
}

#[cfg(target_arch = "wasm32")]
mod exports {
    use std::cell::UnsafeCell;

    /// The request the host writes, the answer it reads, and the mod's own
    /// state: one thread, one call at a time.
    struct Cell<T>(UnsafeCell<T>);
    // SAFETY: a wasm32-unknown-unknown module has one thread.
    unsafe impl<T> Sync for Cell<T> {}

    static REQUEST: Cell<Vec<u32>> = Cell(UnsafeCell::new(Vec::new()));
    static ANSWER: Cell<Vec<u32>> = Cell(UnsafeCell::new(Vec::new()));
    static STICKY: Cell<super::Sticky> = Cell(UnsafeCell::new(super::Sticky { held: None }));

    #[unsafe(export_name = "blot:allocate")]
    extern "C" fn allocate(bytes: u32) -> *mut u32 {
        // SAFETY: the host writes here before it calls frame, which reads it.
        let request = unsafe { &mut *REQUEST.0.get() };
        request.clear();
        request.resize((bytes as usize).div_ceil(4), 0);
        request.as_mut_ptr()
    }

    #[unsafe(export_name = "blot:reset")]
    extern "C" fn reset(_: u32) -> u32 {
        0
    }

    #[unsafe(export_name = "frame")]
    extern "C" fn frame(request: *const u32) -> *const u32 {
        // SAFETY: `request` is the array the host wrote in allocate's space;
        // the answer stays until the next call, after the host copied it.
        unsafe {
            let lanes = std::slice::from_raw_parts(request.add(1), *request as usize);
            let lanes: Vec<f32> = lanes.iter().map(|word| f32::from_bits(*word)).collect();
            let answer = super::answer(&mut *STICKY.0.get(), &lanes);
            let words = &mut *ANSWER.0.get();
            words.clear();
            words.push(answer.len() as u32);
            words.extend(answer.iter().map(|lane| lane.to_bits()));
            words.as_ptr()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(tag: u32, name: &str) -> [f32; EVENT_FLOATS] {
        [tag as f32, key_code(name), 0.0, 0.0, 0.0]
    }

    #[test]
    fn a_tapped_arrow_stays_held_until_tapped_again() {
        let mut sticky = Sticky::default();
        let tap = [key(event_tags::KEY_DOWN, "ArrowRight"), key(event_tags::KEY_UP, "ArrowRight")].concat();
        assert_eq!(sticky.frame(&tap), [key(event_tags::KEY_DOWN, "ArrowRight")], "held after the tap");
        assert!(sticky.frame(&[]).is_empty(), "and nothing more while it is");
        let other = [key(event_tags::KEY_DOWN, "ArrowUp")].concat();
        assert_eq!(sticky.frame(&other), [key(event_tags::KEY_UP, "ArrowRight")], "another arrow lets it go");
        assert_eq!(sticky.frame(&key(event_tags::KEY_UP, "ArrowUp")), [key(event_tags::KEY_DOWN, "ArrowUp")]);
        assert!(sticky.frame(&tap_of("ArrowUp")).is_empty(), "tapped again, it goes");
        assert!(sticky.frame(&tap_of("a")).is_empty(), "other keys are the player's");
    }

    fn tap_of(name: &str) -> Vec<f32> {
        [key(event_tags::KEY_DOWN, name), key(event_tags::KEY_UP, name)].concat()
    }

    #[test]
    fn answers_speak_the_mod_protocol() {
        let mut sticky = Sticky::default();
        let request = [&[1.0, 0.0, 1.0 / 60.0, 640.0, 480.0, 2.0][..], &tap_of("ArrowLeft")].concat();
        let answer = answer(&mut sticky, &request);
        assert_eq!(answer, [&[1.0, 1.0][..], &key(event_tags::KEY_DOWN, "ArrowLeft")].concat());
    }
}
