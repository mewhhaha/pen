# Gather: a pen game in Rust

Steer the square with the arrow keys (or WASD) and gather the gems. A tiny
game, here to show that pen's guest boundary is not Blot's alone: pen runs
it unchanged, natively (`pen --game`, `pen build --game`) as it would in the
browser, with time travel, recordings and replays, because it speaks what a
Blot game speaks.

- **The ABI.** Its module has the `blot:abi` manifest of a Blot game (ABI 2,
  `main: (Array F32 -> Array F32 ! {Foreign}) -> Unit`), the array arena
  (`blot:memory`, `blot:allocate`, `blot:reset`) and nothing else
  (blot2's `compiler/guest-abi.md`).
- **The program protocol** of ink's `lib/pen/program.blot`: the handshake
  (magic, version, state schema, no scenes), then Create, Frame, Snapshot,
  Restore and Stop, each request saying whether the host accepted the
  world the last one made ([`src/lib.rs`](src/lib.rs), `run`).
- **The packets** of pen's `protocol/schema.json`, through pen-protocol's
  own constants and key codes: input frames in, render packets (sprites)
  out.

Rust cannot name `externref`, the opaque host reference `main` takes, so
the game builds as a module that imports `pen:guest/1` `exchange` (an
array's address in, the host's array's out) and exports `pen:main`
([`src/guest.rs`](src/guest.rs)). [`src/adapt.rs`](src/adapt.rs) then gives
it the guest ABI's shape: the `blot:host/1` import first, an `externref`
global for the host, `exchange` defined as a call of the import, `main`
around `pen:main`, the ABI's exports and the manifest. Every other function
keeps its index, and the name section follows, so traps name Rust's
functions. A language that can name `externref` needs no adapter.

```sh
cargo test                                   # the game, its protocol (against pen's decoder) and the adapter
cargo build --release --lib --target wasm32-unknown-unknown
cargo run --release --bin bundle -- target/wasm32-unknown-unknown/release/rust_guest.wasm ../../dist/rust-guest
pen --game ../../dist/rust-guest --headless --size 960x640 --shots shots \
  --script "down=ArrowRight wait=50 up=ArrowRight shot=moved quit"
pen build --game ../../dist/rust-guest --out ../../dist/rust-package
```

`just rust-guest` in pen does the first three and plays it headlessly. The
game's state is 14 lanes (a version, the player, the score, the gem
generator's seed, the gems, the keys held, the time), so a snapshot restores
it exactly: a rewind of 30 frames and 30 steps forward again draws the same
pixels.
