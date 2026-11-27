# Gather: a pen game in Rust

Steer the square with the arrow keys (or WASD) and gather the gems. A tiny
game demonstrating pen's language-agnostic, `externref`-free `pen:guest/1` ABI:
pen runs the raw `cargo build --target wasm32-unknown-unknown` output unchanged
natively (`pen --game`, `pen build --game`) and in the browser (`pen build
--web --web-compact --game`), with time travel, recordings, replays, and mods.

- **The ABI (`pen:guest/1`).** Imports `pen:guest/1::exchange(ptr: i32) -> i32`,
  exports `pen:main() -> ()`, `pen:allocate(bytes: i32) -> i32`, and standard
  Wasm `memory`. No `externref`, no custom ABI section, and no post-build
  bytecode adapter are needed.
- **The program protocol:** session protocol v2 — the handshake (magic, version,
  state schema, no scenes), then Create, Frame, Snapshot, Restore and Stop
  ([`src/lib.rs`](src/lib.rs), `run`).
- **The packets** of pen's `protocol/schema.json`, through `pen-protocol`'s own
  constants and key codes: input frames in, render packets (sprites) out.
- **Mods (`sticky/`):** exports `frame(ptr: i32) -> i32`, `pen:allocate`, and
  standard `memory` — no custom manifest or linker flags needed.

```sh
cargo test --workspace                                   # the game, its protocol (against pen's decoder) and the mod
cargo build --release --workspace --lib --target wasm32-unknown-unknown
cargo run --release --bin bundle -- target/wasm32-unknown-unknown/release/rust_guest.wasm ../../dist/rust-guest
pen --game ../../dist/rust-guest --headless --size 960x640 --shots shots \
  --script "down=ArrowRight wait=50 up=ArrowRight shot=moved quit"
pen build --game ../../dist/rust-guest --out ../../dist/rust-package
pen build --web --web-compact --game ../../dist/rust-guest --out ../../dist/rust-web
```

`just rust-guest` in pen builds, bundles, and plays it headlessly with the
sticky-arrows mod. The game's state is 14 lanes (a version, the player, the
score, the gem generator's seed, the gems, the keys held, the time), so a
snapshot restores it exactly: a rewind of 30 frames and 30 steps forward again
draws the same pixels.
