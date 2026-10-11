//! What the dev tools (`devtools`, the checkout's own program for them) read
//! of the session besides its frames: the log of the game's reports, the
//! Blot compiler's output, new builds, asset reloads and recipe runs
//! (`log`), and the checkout's live-tunable constants (`tunables`). The
//! player runtime leaves the dev tools out (the `devtools` feature); the log
//! is always there, and without the feature only prints.
mod log;
pub mod tunables;

#[cfg(feature = "devtools")]
pub use log::{Line, since};
pub use log::{Source, error, info, note};
pub use tunables::{load as load_tunables, tunable_id};
