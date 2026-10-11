//! pen's Blot compiler plugin: a component of wit/pen-plugin.wit's
//! `compiler-plugin` world that compiles a program with the Blot compiler,
//! `blotc serve-project` (blot2's `zig-native/PROJECT_PROTOCOL.md`). Each
//! session is one compiler process, which keeps the program open between
//! builds and reuses what did not change; a protocol failure kills it, and
//! pen opens a new session. The client is plain Rust ([`Session`], over any
//! [`Child`]), which `cargo test` checks natively against a real compiler;
//! only the component's bindings (`component.rs`) are Wasm-only.
//!
//! The compiler runs as the program the checkout grants the plugin under
//! `blotc`, with only `BLOT_CACHE_DIR` in its environment (the plugin's cache
//! directory, where it keeps its restart cache).
//!
//! # Settings
//!
//! A program's `settings` in the checkout's `pen.json`, all of them
//! optional. Relative paths are from the program's root.
//!
//! ```json
//! {
//!   "program": "blotc",
//!   "imports": { "gdev/": "packages/" },
//!   "std": "../blot2/std",
//!   "prelude": "../blot2/std/prelude.blot",
//!   "dependencies": null,
//!   "codegenTier": "optimized",
//!   "semanticWorkers": 4,
//!   "expectedIdentity": "<64 lowercase hex digits>"
//! }
//! ```
//!
//! - `program`: the name the compiler is granted under (`blotc`).
//! - `imports`: import prefixes, each ending in `/`, and the directories
//!   behind them.
//! - `std`: the directory behind `std/`. Without it there is no `std/`.
//! - `prelude`: the module every module sees: `prelude.blot` in `std` unless
//!   given, and none when null.
//! - `dependencies`: a `.blotdep` bundle of `blot dependencies`, read once
//!   when the session opens.
//! - `codegenTier`: `optimized` (the compiler's default) or `development`,
//!   which skips the optional scalar and vector passes.
//! - `semanticWorkers`: 1 to 16 workers for semantic jobs; serial without.
//! - `expectedIdentity`: the only compiler identity to accept, from its
//!   hello.
//!
//! A session watches the entry's directory, every import's directory, the
//! standard library's and the prelude.
mod session;
mod settings;

#[cfg(target_arch = "wasm32")]
mod component;
#[cfg(not(target_arch = "wasm32"))]
pub mod native;

pub use session::{Child, ReadError, Session};
pub use settings::{CodegenTier, PROGRAM, Settings};

/// A program of the checkout's: its root, its entry file (absolute) and its
/// settings, as JSON text ("{}" when none).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub root: String,
    pub entry: String,
    pub settings: String,
}

/// A source file for one build: its absolute path and its contents, or none
/// to hide the file on disk for this build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Overlay {
    pub path: String,
    pub contents: Option<String>,
}

/// A problem a build found in a program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    /// The absolute path of the file it is in, as the compiler has it (with
    /// symbolic links resolved).
    pub file: String,
    /// Where the compiler found it: `parse`, `check`, `emit`...
    pub stage: String,
    /// A stable name for the problem.
    pub code: String,
    /// UTF-8 byte offsets in `file`; both 0 when it has no place.
    pub start: u32,
    pub end: u32,
    pub message: String,
}

/// What a build made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Build {
    /// The program's Wasm module.
    Built(Vec<u8>),
    /// Why it does not compile: at least one diagnostic.
    Rejected(Vec<Diagnostic>),
}
