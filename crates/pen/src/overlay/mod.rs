//! The dev overlay: the native counterpart of the browser build's console
//! and diagnostics (gdev's `web/client.ts` and `web/diagnostics.ts`), drawn
//! with egui over the window in `pen dev` and `pen run`. The backquote key
//! shows and hides it, and the game never sees that key; `--overlay` (or
//! PEN_OVERLAY=1) shows it from the start. While a panel has the keyboard
//! (the console's input) or the pointer is over one, the game gets neither;
//! a drag the game started stays the game's.
//!
//! - **Console**: the game's commands, read from the checkout's
//!   `commands.ts` (`commands`), with completion and history, and the
//!   browser's `/help` and `/diagnostics`; `/command N ARG` sends any by
//!   number. It prints the game's reports, the Blot compiler's output,
//!   reloads and time travel (`log`).
//! - **Profiler**: frame interval and CPU work with a graph, the frame's
//!   phases (guest, decode, validate, render+present, housekeeping), the
//!   renderer's draws and instances, guest memory and the GPU. Frames are
//!   timed only while it shows.
//! - **Timeline**: play and pause, step, rewind (a frame, a second, or to a
//!   frame on the scrub slider), replay the last seconds, and record the
//!   input to a `.penrec` (`session::Session`'s time travel).
//! - **Hot reload**: the Blot build's status and errors, the running
//!   build's reloads, and asset and recipe events.
//! - **Tunables**: the checkout's live-tunable constants (`src/tunables.json`,
//!   `tunables`): a slider each, a reset, and Bake, which writes the game's
//!   values back as the defaults. They reach the game as Tune events, the
//!   console's `/tune NAME VALUE` too, and the panel follows the game's
//!   state; the file is watched, so new declarations show at once.
//!
//! It draws over the swapchain image after the game's frame and before
//! present (`session::Overlay`), never into captures, recordings or headless
//! output. `--overlay-shot PNG` and the console's `/shot` save what the
//! window shows, overlay included. The player runtime leaves it out (the
//! `devtools` feature); the log it reads is always there, and without the
//! feature only prints.
mod log;
mod tunables;

pub use log::{Source, error, forward_build_output, info, note};
pub use tunables::{load as load_tunables, tunable_id};

#[cfg(feature = "devtools")]
mod commands;
#[cfg(feature = "devtools")]
mod panels;
#[cfg(feature = "devtools")]
mod ui;

#[cfg(feature = "devtools")]
pub use ui::{DevOverlay, Settings, requested};
