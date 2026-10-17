//! Where a run's startup goes. With `PEN_STARTUP=1` each phase prints the
//! time since the process started, once: the program loaded, the GPU ready,
//! the game's first frame, the first frame drawn with everything it asked
//! for (in a window the loading veil starts to clear then), and settling.
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Instant;

static STARTED: OnceLock<Instant> = OnceLock::new();
static SAID: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PEN_STARTUP").is_some_and(|value| value != "0"))
}

/// The process starts: phases count from here.
pub fn begin() {
    STARTED.get_or_init(Instant::now);
}

/// Milliseconds since the process started.
pub fn elapsed() -> f64 {
    STARTED.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

/// Says that `phase` has been reached, the first time it is.
pub fn mark(phase: &'static str) {
    if !enabled() {
        return;
    }
    let mut said = SAID.lock().unwrap();
    if said.contains(&phase) {
        return;
    }
    said.push(phase);
    eprintln!("startup: {phase} at {:.0} ms", elapsed());
}

/// After a frame is drawn: the first one that waited on nothing is the
/// first a player sees whole.
pub fn drawn(renderer: &pen_render::Renderer) {
    if !renderer.waiting() {
        mark("first complete frame");
    }
}
