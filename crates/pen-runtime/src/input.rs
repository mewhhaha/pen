/// Platform-independent time travel shortcuts, consumed before game input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hotkey {
    Pause,
    Step,
    Back,
    Replay,
}
