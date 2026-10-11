# pen

Runs [gdev](../gdev), a game written in [Blot](../blot2) and compiled to
WebAssembly, as a native program: a winit window, a wgpu port of gdev's
WebGPU renderer, and the game's Blot program on wasmtime, with no JavaScript
engine at run time (Deno only compiles the Blot program while developing).
One `pen` binary develops a gdev checkout with hot reload, its own dev
tools and live tunables; tests it with golden images, renderer parity and an engine
conformance check; records and replays sessions bit for bit; and packages the
game with a smaller player runtime for Linux, the Steam Deck and Windows. The
same Rust renderer also builds for the browser, where gdev's web build uses it
behind `?renderer=rust`.

```
pen (the runtime)                           game/ (a package's bundle)
├─ winit window, input, pointer lock        ├─ game.cwasm  the Blot program, compiled ahead of time
├─ wgpu renderer     (pen-render)           ├─ game.wasm   its Wasm, for the guest ABI manifest
├─ wasmtime guest    (blot-host)            ├─ assets/     meshes, materials, WGSL, textures, sounds
├─ packets           (pen-protocol)         └─ main.scene  the starting level
├─ audio: gdev's WebAudio mix on kira's output (+ libopus for Opus clips)
├─ AccessKit, and services: Steam, clipboard, url, platform, pathfinding
└─ the checkout's dev tools program (not in the player runtime)
```

The crates:

- **blot-host**: Blot's guest ABI 2 on wasmtime: the `blot:abi` manifest,
  copied numeric arrays, capability tokens as `externref`. Where the
  JavaScript adapter needs JSPI to suspend a guest inside a host call, a
  blocking callback on the guest's own thread does the same here. It makes
  three engines: `engine()` (Cranelift with inlining; compiled modules are
  cached in the platform cache directory), `dev_engine()` (no inlining, and an
  in-memory incremental cache for hot reload) and `portable_engine()` (a
  package's `game.cwasm`, for another target and a CPU baseline).
- **pen-protocol**: the frame and render packets (gdev's `boundary.ts` and
  `protocol.ts`), generated from gdev's `protocol/schema.ts` ("The boundary").
- **pen-render**: the wgpu port of gdev's renderer (`renderer.ts`, `post.ts`
  and friends), natively and in the browser. See its
  [PORTING.md](crates/pen-render/PORTING.md);
  `cargo run --release -p pen-render --example render_packet` renders packets
  from `just packets` to PNGs.
- **pen**: the `pen` command. `game.rs` and
  `program.rs` port gdev's `game_runtime.ts` and `program_runtime.ts`,
  `project.rs` compiles and packages a checkout, and `session.rs` is the frame
  loop that windowed and headless runs share.

## Quick start

`just install` builds the development runtime with Steam and installs it as
`~/.local/bin/pen`, a link to `~/.local/lib/pen-runtime/pen` beside Steam's
library. In a gdev checkout (or with `--project DIR`):

| | |
|---|---|
| `pen dev` | Compile the Blot program and play, reloading the program, shaders and assets as they change, with the game's state kept, and rerunning the asset recipes whose sources change |
| `pen web` | gdev's browser build (`deno task web`): its console, diagnostics and the browser's devtools, with hot reload |
| `pen run` | Compile once and play |
| `pen build [--out DIR]` | A shippable folder: the player runtime, its Steam library and the game, precompiled (default `.pen/package`) |
| `pen build --web [--out DIR]` | The browser build as static files for any web host (default `.pen/web-export`) |
| `pen test [FILE]` | Golden-image tests (default `tests/visual/scenarios.json`) |
| `pen test-blot [FILE...]` | The checkout's Blot tests (default `tests/*.blot`; `--filter TEXT` runs the tests whose names hold it) |
| `pen parity` | The native renderer against the checkout's `renderer.ts` on the same frames |
| `pen bench` | Guest time per frame, without rendering |
| `pen packets DIR` | Each frame's input and render packet, without rendering |

Without a command, `pen` plays a packaged game: `game/` beside the
executable, or `--game DIR`. `pen --help` lists every option and environment
variable.

The checkout needs nothing new. The runtime compiles its Blot program with the
checkout's own compiler setup (`blot_runtime.ts`) through Deno, from a small
script embedded in the binary, into `.pen/runtime/`. The compiler's temporary
files go under the cache directory (`~/.cache/pen/tmp`), not `/tmp`. Assets and
the editor's scene are used in place from `src/`. gdev's own `just dev` and
`just run` are `pen dev` and `pen run`, passing their arguments on; pen replaced
the Deno Desktop host there on 2026-10-11.

This repository's Justfile runs the same commands on `../gdev`
(`just gdev=PATH ...` for another checkout):

| | |
|---|---|
| `just dev`, `just run`, `just headless ARGS` | Play the checkout (`just dev --mute`, `just headless --frames 300 --record out.mp4`) |
| `just web` | gdev's browser build |
| `just test` | Everything in "Testing": `check`, `visual`, `parity`, `conformance`, `determinism` and `blot-tests` |
| `just bench` | Guest time on wasmtime, then the same frames on V8 |
| `just export`, `just packets` | The bundle `dist/gdev` and the packets `dist/packets` that some tests and examples read |
| `just sync-protocol` | Regenerate pen-protocol's `packets.rs` from the checkout's schema |
| `just package` | `pen build` into `dist/package` |
| `just player`, `just windows`, `just install-player`, `just ship` | The player runtimes, and packages with them ("Shipping") |

Saves, preferences and crash reports live in the platform data directory
(`~/.local/share/gdev` on Linux), or in `--data DIR`.

## Developing

### Hot reload

`pen dev` keeps the checkout's Blot compiler open, and it writes each new
build to `.pen/runtime/game.wasm` in one rename. The runtime compiles the new
build on a background thread, then moves the running state into it between
frames (gdev's `game_runtime.ts` reload): the new program restores a snapshot
of the old one's state and draws a frame before it replaces it. A build that
fails to compile, start or take the state leaves the old one running, and the
console says why.

The state survives most edits. gdev saves it in keyed, versioned sections, one
per saved type (gdev's README, "Saved state"). A new build keeps each section
saved with its type's current version, migrates one saved with another
version through the type's `migrate`, starts afresh a type the state lacks or
cannot migrate, and drops sections no type reads any more. The runtime reports
what it did not simply keep:

```
Blot reloaded; kept 1 of 2 saved sections; migrated section 100 from version 1 to 2; dropped section 104 (no longer saved)
```

Starting from a saved state, loading a quicksave and loading the editor's
scene go through the same restore and report the same way. Only a build with
another `state_schema`, the save format's own version, is refused until a
restart.

Hot reload compiles with blot-host's development engine: Cranelift without
inlining, and an in-memory incremental cache that reuses each unchanged
function's machine code, so a small edit recompiles only what it touched.
Without inlining, gdev's first compile drops from about 360 to 105 ms and a
reload's from about 100 to 50 ms, for slower guest code ("Performance");
`BLOT_INLINING=1` turns inlining back on. Every other command compiles with
inlining. From a new `game.wasm` to the new build running, the runtime's side
of a reload takes about 90 ms. The Blot compiler's build comes before that
and prints its own time (`blot: Game compiled (1.05 s)`); gdev's README puts a
build at 1 to 2 s.

Asset files reload in the renderer and audio once they have been quiet for
15 ms, and the old version draws until the new one is ready. A changed catalog
arrives with a new build, and a deleted asset keeps its last version until it
comes back. After a reload, Ctrl+Shift+R replays the last 5 s of input on the
new build ("Recording, replay and time travel").

### Wasm plugins

pen's contract is with Wasm, not with one language: what is specific to a
language or toolchain lives in Wasm plugins the checkout chooses, not in pen.
A plugin is a WebAssembly component implementing a world of
[wit/pen-plugin.wit](wit/pen-plugin.wit):

- **compiler**: a session per program that builds it (with overlays over the
  files on disk) into its Wasm module or diagnostics, and names the paths
  whose changes need a new build. `plugins/blot` drives the Blot compiler,
  `blotc serve-project`, as blot2's `zig-native/PROJECT_PROTOCOL.md`
  specifies it.
- **assets**: the virtual modules a program imports its assets through (its
  catalog, words, shaders), made from the checkout's files, which pen passes
  to the compiler as overlays. `plugins/gdev-assets` makes gdev's.

A checkout chooses them in a `pen.json` at its root:

```json
{
  "plugins": {
    "blot": { "read": ["../blot2/std"], "run": { "blotc": "../blot2/generated/compiler/blotc.gz" } },
    "gdev-assets": {}
  },
  "compiler": { "plugin": "blot", "settings": { "imports": { "gdev/": "packages/" }, "std": "../blot2/std" } },
  "assets": { "plugin": "gdev-assets" }
}
```

With a compiler chosen, `pen dev`, `pen run` and the dev tools' build go
through the plugins instead of Deno: each program builds in a session kept
open, so an edit rebuilds incrementally, and its watched paths are watched;
a build that compiles is written to `.pen/runtime/` as the Deno build's was
and reloads the same way. Without a `pen.json`, Deno builds as before.

A plugin has no ambient access to the machine. WASI gives it clocks,
randomness and output (to pen's log); pen's own interfaces give it what
`pen.json` grants: reading under the checkout and its `read` directories,
and starting its `run` programs by name with piped input and output (a `.gz`
one is unpacked into pen's cache first). Plugins are found by name in
`PEN_PLUGINS`, in `plugins/` beside the pen executable (`just install` puts
them there), or in this repository's `plugins/target`, or at a `pen.json`
plugin's `wasm` path. `just plugins` builds them (Rust for `wasm32-wasip2`)
and `just plugin-tests` tests them natively. The player runtime has no
plugins (the `plugins` feature).

### The asset pipeline

gdev's `pipeline.json` maps the files its assets are built from (model
scripts, `.blend` files, fonts, other recipes' outputs, as globs) to the
Justfile recipes that build them. `pen dev` reruns a recipe once its sources
have been quiet for 250 ms, and what the recipe writes into `src/assets`
reloads like any asset edit. Recipes run one at a time, in the file's order,
so one that reads another's output runs after it. A change that arrives while
a recipe runs queues it again, and a file counts as changed only when its
content does, so rewriting an identical output sets nothing else off. Each
line a recipe prints comes after its name; a recipe that fails is reported,
and the game keeps the assets it has. Recipes that need a program that is not
installed (`requires`: Blender, threejs, ffmpeg) are named once at startup and
skipped. `pen dev --no-pipeline` turns the pipeline off; gdev's README
("Rebuilding assets as you edit") describes the recipes.

### The browser

`pen web` runs gdev's own browser build (`deno task web`, on
<http://localhost:8000>) with its hot reload, and `pen build --web` exports it
as static files (gdev's `scripts/export_web.ts`). The Rust renderer builds for
the browser too: `crates/pen-render/web` is a wasm-bindgen package shaped
like `renderer.ts`'s `createRenderer`, and gdev's `scripts/rust_renderer.ts`
builds it into `.gdev/rust-renderer`, which the dev server and the export
serve for `?renderer=rust` (PORTING.md, "In the browser").

### The dev tools

The dev tools are the checkout's own: a second Blot program beside the game
(gdev's `devtools/main.blot`, its console and diagnostics, written with the
game's interface packages), which `pen dev` and `pen run` build and run
while the game runs, drawing its sprites over the game's frame. pen hosts it
and draws nothing of its own; gdev's README ("Dev tools") says what it does.
A checkout without one runs without dev tools.

It is a program of its own, with its own state, outside the game's saves,
recordings and timeline: it keeps running while the game is paused,
rewinding, failing to build or stopped on an error, and what it is told
(frame timings, the log) never reaches the game, so determinism and
recordings are untouched. Each frame (`crates/pen/src/devtools.rs`):

- **Input.** It gets every window event before the game, key repeats
  included, and the key left of 1 always as the backquote, which the game
  never sees. Its interface says what it takes from the game: the keyboard
  while one of its nodes has the focus (its console's input; the game gets
  FocusLost then, so held keys let go), and the pointer and wheel over its
  dialogs. A drag the game started stays the game's.
- **What it asks.** The `devtools` service (gdev's `protocol/schema.ts`,
  DevtoolsOperation): the log (the game's reports, the Blot compiler's
  output, reloads, assets, recipes, time travel), the last frame's numbers
  (DevtoolsFrame: frame interval, the frame's phases when it asks for
  timing, draws, meshes, memory, size, the timeline's state) and the GPU's
  name come back in its next frame; game console commands, pause, play,
  step, rewind, replay, keeping history, input recordings, shots, tunable
  values and bake are carried out.
- **Drawing.** Its sprites, through a renderer of its own (so its assets'
  events never reach the game), over the window's image only, never into
  screenshots, recordings or headless output; its interface hangs from the
  window's accessibility tree.

Its build (`.pen/runtime/devtools.wasm`) compiles beside the game's (lines
in the log name its entry); a new one is picked up with its state carried
over, as the game's is. Its state is kept in the checkout's
`.pen/devtools.state` between windowed sessions (the console's history, the
diagnostics' level). `--overlay` or `PEN_OVERLAY=1` opens its console from
the start. `--overlay-shot PNG` and its `/shot` save the frame with the dev
tools over it, headless too (where a run waits for its build and its
assets). A script's `overlay=/COMMAND[,ARG...]` step types a line into its
console (opening it first), so

```sh
pen run --headless --frames 300 --script "overlay=/diagnostics,high" --overlay-shot shot.png
```

shows the diagnostics without a window; a headless run keeps no state. They
are the `devtools` cargo feature, on by default; the player runtime leaves
them out.

Not in the program yet: the timeline, tunables and builds panels of pen's
earlier egui overlay (`/timeline`, `/tunables` and `/builds` say so), and
Bake, which the service carries out but nothing asks for yet. Time travel
works from the console (`/pause`, `/play`, `/step`, `/rewind`, `/replay`,
`/history`, `/record`) and its keys ("Recording, replay and time travel").

### Live tunables

A number to get right by feel, such as how fast the camera zooms or the gnorps
walk, can be a live-tunable constant, declared in the checkout's
`src/tunables.json` (gdev's README, "Tunables"). Setting one needs no build:
it reaches the game as a Tune input event in the next frame the game runs, so
recordings, replays and rewinds keep it. `/tune` in the dev tools' console
lists them with the values the game holds, and `/tune NAME VALUE` sets one,
as the script step `tune=NAME=VALUE` does. Baking the game's values into the
file as the defaults (through the checkout's `scripts/bake_tunables.ts`) is
the devtools service's `bake`, for the dev tools' coming Tunables panel; the
game then rebuilds, reloads and keeps them, so nothing moves.

### Headless runs, scripts, screenshots and videos

```sh
# The editor, then play, a shot, three strikes on the rock and another shot.
pen run --headless --shots shots --script "key=Tab wait=60 shot=playing click=470,150 click=470,152 wait=30 shot=struck quit"
# Ten seconds of 1080p60 video with the game's sound, at a steady clock.
pen run --headless --size 1920x1080 --fps 60 --frames 600 --record clip.mp4
# The same while watching it in a window.
pen run --size 1920x1080 --record clip.mp4 --frames 600
```

- `--headless` draws offscreen, as fast as the machine allows, on a steady
  clock: every frame is exactly `1/fps` of game time (60 unless `--fps`), so
  runs and recordings are repeatable whatever the frame rate. It needs
  `--frames`, a script or a replay to know when to stop.
- Before the clock starts, headless runs and recordings settle: they draw the
  first frame until every asset has loaded and faded in and the loading veil
  is gone (`--no-settle` skips it). After that a frame depends only on game
  time and input, so the same script gives the same pixels.
- `--record FILE` (`.mp4`, `.mkv`, `.webm`) is steady by construction: frame N
  of the video is game time N/fps, ffmpeg receives a constant-rate stream, and
  the game's sound is mixed offline so that exactly N x 48000/fps samples
  accompany N frames, by the same mixer the window plays ("Sound"). However
  slow a frame is to draw, the video does not stutter or drift. In a window
  the frames are also paced to the wall clock, so the game runs at its real
  speed while you play and record; `--size` pins the frame size whatever size
  the window gets.
- `--script` (or `--script-file`) takes the steps of gdev's browser tooling
  (`scripts/page.ts`): `wait key down up type move click press release drag
  wheel shot save load`, plus `quit`, `tune=NAME=VALUE` (above),
  `console=CODE[,ARG]` (a console command by number: `console=3,12` hires
  twelve gnorps), `overlay=/COMMAND[,ARG...]` (a line typed into the dev
  tools' console, above), the time travel steps ("Recording, replay and
  time travel") and `ui=NODE,ACTION[,VALUE]` ("Accessibility"). Coordinates are page pixels of a
  page 960 wide, scaled to the viewport, so `looks.json` coordinates land in
  the same places at 1920x1280 (the headless default). `view=`, `over=`,
  `reload` and `storage=` need the browser tooling.
- `--screenshot PNG` writes the last frame; `--frames N` stops after N.

## Testing

`just test` runs all of these but gdev's own TypeScript tests.

- **Unit and GPU tests** (`just check`): it checks that pen-protocol's
  generated `packets.rs` matches the checkout's schema, then runs clippy,
  `cargo test --workspace`, and the renderer's tests again in release
  ("Toolchain note"). The tests cover the guest ABI manifest, packet encoding,
  scripts, saves, input recordings and time travel, video recordings (exact
  frame counts and durations, through ffprobe), the mixer and reverb, the
  accessibility tree, services, the pipeline, the dev tools' host and
  tunables, and the renderer against real assets, hot reload included. Tests
  that read the exported bundle `dist/gdev` (`just export`) skip themselves
  without it.
- **Golden images** (`pen test`, `just visual`): the six scenarios in
  `tests/visual/scenarios.json` run headless at 960x640 after settling, each
  from a fresh game with its own data directory, and every `shot=` is compared
  with `tests/visual/golden/`. A shot fails above 1% RMSE and leaves the image
  and a diff (differences in red) in `.pen/visual/` of the checkout. Runs are
  pixel-identical on one machine; another GPU or driver may differ slightly,
  which the threshold allows. When the game's look changes on purpose,
  `just visual --update` accepts the new images, and `just visual --only menu`
  runs one scenario.
- **Renderer parity** (`pen parity`, `just parity`): the native renderer is a
  port of the checkout's `renderer.ts`, so a renderer change needs its twin in
  `crates/pen-render`. Parity runs a script on the game, draws frames with
  both renderers (the TypeScript one on Deno's WebGPU) and requires them to
  match within 0.2% RMSE; they agree to about 0.013% (PORTING.md). Deno's
  WebGPU cannot allocate a full-screen inset's 4K, 4x MSAA targets, so frames
  with one are compared without their insets, and the report says so.
- **Engine conformance** (`just conformance`): `pen packets` runs
  `tests/conformance.steps` on wasmtime and records each frame's input and
  render packet; `tools/conformance_v8.ts` replays the inputs through gdev's
  own TypeScript runtime on V8 and requires every packet to match bit for bit
  (367 frames on 2026-10-11). This checks the Blot ABI port and both engines.
- **Determinism** (`just determinism`): `cargo test -p pen contract`,
  then `tests/determinism.steps` headless twice, and once with `--overlap`,
  for the same packets; the first run's recording replayed, without the
  quicksave it loaded, for the same inputs and packets; and a run that
  rewound, went live from the past and replayed its last second, whose
  recording must replay the packets of its final timeline. The rewinds land in
  the build bar, the interface focus, the station card and the pause menu, and
  every frame run again must give the packet it gave the first time (on the
  game before its interface state was saved, 213 of 460 such frames did not).
  The contract tests check the same from every moment, with a host that
  answers service requests 40 frames late, and name the first packet field
  that differs. See [docs/determinism.md](docs/determinism.md), "Checks".
- **Blot tests** (`pen test-blot`, `just blot-tests`): gdev's game logic,
  tested in Blot. Each module in the checkout's `tests/*.blot` is an entry
  file, compiled through the checkout's compiler plugin as if it stood at
  `src/test_fixture.blot`, with the game's asset modules; each worker keeps
  one compiler session from module to module ("Wasm plugins"). Each `test_…`
  entry, Unit to Array F32, runs in a fresh instance and returns its checks'
  report (gdev's `packages/check.blot`); a failed check prints what it
  expected and got. `tests/traps/` must trap, and `tests/rejected/` must not
  compile. gdev's whole suite, 472 tests in 54 modules, takes about a minute.
- gdev's own TypeScript tests (`deno task test` in the checkout) cover the
  game logic not yet moved into Blot tests.

### CI

`.github/workflows/ci.yml` runs what needs only this repository: the build,
clippy (`-D warnings`, with and without the compiler) and the tests on Linux,
with Mesa's lavapipe as the GPU (`PEN_SOFTWARE=1`), and the two player
runtimes (`just player`, `just windows`) as artifacts, with their unstripped
builds. The game's checkouts are private siblings (`../gdev`, `../blot2`); a
third job (the tests that read `dist/gdev`, engine conformance, a package, and
golden images on lavapipe, which only report) runs when the repository
variables `GDEV_REPOSITORY` and `BLOT2_REPOSITORY` name them and the secret
`CHECKOUTS_TOKEN` can read both. The repository has no git remote yet, so the
workflow has never run.

`--software` (or `PEN_SOFTWARE=1`) asks wgpu for a fallback adapter: lavapipe
on Linux, WARP on Windows. Lavapipe is not installed on the development
machine, so that mode has not run here.

### Toolchain note

Rust 1.99's release optimizer miscompiled the sampler setup in `pen-render`
(`if filtering > 1 { filtering.min(16) } else { 1 }` became
`filtering.min(16)`, an invalid 0), which only debug builds hid. It is written
as `filtering.clamp(1, 16)` now; keep `cargo test --release` in the loop when
changing the renderer, and set `PEN_GPU_BACKTRACE=1` to see where a GPU
validation error came from.

## Recording, replay and time travel

gdev is a pure function of its input: from the same state, the same frames of
input give the same render packets, bit for bit. The runtime builds on that.
gdev saves all the state later frames read, its interface's included, so a
rewind runs each frame again exactly as it first ran; a moment is 2–6 KB (a
pathfinding request in flight adds about 20 KB).
[docs/determinism.md](docs/determinism.md) has the contract, what keeps runs
inside it, what breaks it, and what it costs.

| | |
|---|---|
| `--record-input FILE` | Records the session's starting moment, then every frame's input, flags and loaded states, to a `.penrec` |
| `--replay FILE` | Plays a recording in place of the player's input and clock: the same frames, bit for bit. Headless, the run ends with it; in a window it pauses there, and Ctrl+Shift+P plays on, live |
| `--trace FILE` | A line per frame: its number, and the hashes of its input and packet |
| `--history` | Keeps the last 10 s to rewind and replay (always on in `pen dev`, in replays, and in scripts that go back) |

In a window, time travel is on keys the game never sees:

| Keys | |
|---|---|
| Ctrl+Shift+P | Pause or play |
| Ctrl+Shift+period | Step one frame |
| Ctrl+Shift+comma | Step back one frame |
| Ctrl+Shift+R | Replay the last 5 s (after a reload, on the new build) |

Scripts have the same as steps: `pause`, `resume`, `step[=N]`, `rewind=N`
(frames) and `replay=SECONDS`. The dev tools' console has them too, and
`/record [FILE]` starts or stops an input recording in the middle of a
session.

**Edit and replay.** In `pen dev`, pause when something goes wrong
(Ctrl+Shift+P), so the moment stays in the history while you edit. Save the
Blot fix. When the reload is in, press Ctrl+Shift+R: the last 5 s of input
play again on the new build, starting from the state before them, and then the
game plays on live.

A recording replays exactly only on the build that made it, and it does not
record which build that was (docs/determinism.md, "What breaks it").

## The boundary

### Protocol

The host and the game exchange two packets a frame. The input carries the
elapsed game time, the viewport and the events: keys, pointer, wheel, focus
lost, console commands, preferences, Tune, UiAction, assets ready or failed,
and service answers. The render packet carries what to draw, the sound, the
interface's nodes, service requests, and requests to checkpoint, undo, redo,
save, load or exit. pen-protocol's `packets.rs` is generated from gdev's
`protocol/schema.ts`, the packets' one description, as gdev's TypeScript and
Blot sides are (gdev's README, "Host boundary"). After a schema change,
`just sync-protocol` regenerates it, and `just check` fails while it is
stale.

### Services

The game reaches the platform and native libraries through one boundary,
gdev's `packages/services.blot`: it asks in its render packet's `services`
section (a request id, a service and operation code, four numbers and values,
text as code points), and each request is answered once, as ServiceAnswer and
ServiceData events in a later frame's input. Answers are input, so an input
recording holds them: a replay, a rewind or a step through recorded frames
plays them back and calls no service (`crates/pen/src/services/`). The
runtime's services:

| Service | What it does |
|---|---|
| `steam` | Achievements, stats and rich presence through the Steam client; `unavailable` without one |
| `clipboard` | Copies text to the system clipboard or reads it (arboard: X11 and XWayland, Windows, macOS) |
| `url` | Opens an http or https address in the player's browser |
| `platform` | The operating system, the player's locale, and whether this is a Steam Deck |
| `pathfinding` | Native compute: the shortest path over a grid the game sends (the build bar's route preview) |

`PEN_SERVICES_LOG=1` prints a line for each request and what it was answered.
A native library becomes a service by implementing the `Service` trait (its
code, and an answer for each request, now or later through `poll`) and
registering it in the session's `Services`; the service and its operations get
codes in gdev's `protocol/schema.ts` (`just sync-protocol`), and the game asks
through `services.blot`. `services/pathfinding.rs` is the example.

### Sound

gdev declares its sound every frame, whole: buses, the voices it keeps playing
by id, and sounds played once (gdev's README, "Sound"). The browser plays it
with WebAudio (`web/mixer.ts`). Here `audio/mixer.rs` emulates the same graph
sample for sample: each node's arithmetic, `setTargetAtTime` glides, linear
fades, the k-rate playback rate, Butterworth lowpasses with Chrome's
coefficients, StereoPannerNode's mono and stereo laws, and clips resampled to
the output rate as Chrome's decodeAudioData resamples them (`resample.rs`).
`audio/reverb.rs` is the shared reverb: the same generated impulse response as
the browser's ConvolverNode, convolved in two uniformly partitioned stages
(128-sample blocks for its first 2048 samples, 2048 for the rest), which, like
the ConvolverNode, adds no latency. Live, the mixer runs inside one kira sound
on the audio thread, and each frame's mix lands at the start of its next
128-sample quantum. Ogg Opus clips decode through libopus, the rest through
kira's decoders. Recordings run the same mixer offline at 48 kHz.

Mixing a run's packets both ways (`PEN_MIX_PACKETS=DIR PEN_MIX_ASSETS=DIR
PEN_MIX_OUT=FILE cargo test --release -p pen mix_packets -- --ignored`
here, gdev's `web/mixer.ts` in Chromium's OfflineAudioContext there) gave the
same sound to about -64 dB, and the same 100 ms envelopes to 0.01 dB.
`PEN_RECORD_PCM=FILE` also writes a recording's mix as raw f32 stereo, before
it is encoded.

### Accessibility

The game declares its interface as a tree of nodes every frame (gdev's README,
"Accessibility"): the HUD, the build bar, the station, the menus and settings,
and the level editor's toolbar. The runtime shows it to assistive technology
through [AccessKit](https://accesskit.dev) (`accessibility.rs`, with
`accesskit_winit`: AT-SPI on Linux, UI Automation on Windows, NSAccessibility
on macOS), in the player runtime too, and the dev tools' own tree (their
program's interface) is grafted under it while it shows. Each node has its role, name, description,
value, states, range and the actions it takes, and lies where the game draws
it, scaled to the window. The game drives the keyboard focus itself (F10, Tab,
the arrows, Enter: the same in the browser); the tree's focus follows it, or
the dev tools' while one of their nodes has the keyboard. An assistive
technology's action on a game node reaches the game as a `UiAction` input
event with the next frame, so input recordings capture it and replays repeat
it. The tree is rebuilt only when the interface changed (otherwise a
comparison of the frame's nodes, about 2 µs), and sent only while an assistive
technology listens.

`PEN_A11Y_DUMP=FILE` writes the tree as text whenever it changes, in a window
or headless (the frame, then each node's role, `#id`, words, states, bounds and
actions), and the `ui=NODE,ACTION[,VALUE]` script step acts on a node as
assistive technology does: NODE is its id, decimal or `0x` hex, and ACTION
`focus`, `blur`, `activate`, `increment`, `decrement`, `set_value`, `expand`
or `collapse`.

```sh
PEN_A11Y_DUMP=a11y.txt pen run --headless --script "key=Tab wait=30 ui=0x20001,activate wait=30 key=F10 key=Tab key=Enter wait=10 quit"
```

## Shipping

`pen build` makes a shippable folder (default `.pen/package`; `just package`
writes `dist/package`): the runtime, named after the game (`gdev` or
`gdev.exe`), Steam's library beside it when the runtime has Steam
(`libsteam_api.so`, `steam_api64.dll`), and `game/`, the bundle in the diagram
at the top.

It ships the **player runtime**: pen built without its default `compiler`
and `devtools` features (`--no-default-features --features steam`), so without
Cranelift or the dev tools. It only loads the `game.cwasm` its package was
precompiled with, and refuses every command that compiles a checkout.
`pen build` ships the runtime in `player/` beside itself, else itself (and
says so); `--runtime EXE` ships another. `game.cwasm` is compiled for the
runtime's platform (read from its executable header, or `--target TRIPLE`)
and for a CPU baseline (`--cpu PRESET`, default `x86-64-v2`), not the
packaging machine's CPU.

| | |
|---|---|
| `just player` | The Linux player runtime, built with cargo-zigbuild against glibc 2.28, which every Steam Linux Runtime container and SteamOS have; `just glibc` checks the result |
| `just windows` | The Windows player runtime, cross-compiled as `x86_64-pc-windows-gnu` with zig: no Windows SDK or MSVC |
| `just install-player` | Installs `just player`'s runtime in `~/.local/lib/pen-runtime/player/`, where `pen build` looks for it |
| `just ship` | Both player runtimes, and the checkout's game packaged with each in `dist/linux` and `dist/windows` |

Both use the `dist` profile (fat LTO, one codegen unit, `panic = "abort"`) and
ship a stripped copy from `target/<target>/dist/player/`. The unstripped build
beside it in `target/<target>/dist/` resolves crash reports, so keep one per
release. [docs/shipping.md](docs/shipping.md) has the details: the CPU
baseline, the build profiles, glibc and the Steam Linux Runtime, the Windows
build, what macOS would take, crash reports, Steam and Steam Auto-Cloud.

### Steam

`just install`, `just player` and `just windows` build with Steam, and
`pen build` then ships Steam's library beside the runtime. Steam starts when
a packaged game is played in a window, or with `PEN_STEAM=1`, and the game
plays on without it when there is no Steam client. The Steam overlay hooks
the swapchain the renderer presents, so it needs no code. While developing,
put `steam_appid.txt` containing `480` beside the executable to test against
Spacewar, and `PEN_STEAM_DEBUG` runs achievement and stat steps
(docs/shipping.md, "Steam"); `PEN_STEAM_APP_ID` makes a store build relaunch
through Steam when started outside it. The game unlocks its achievements
(`FIRST_BRICK`, `FIRST_BUILDING`), raises its stat (`MOST_SALT`) and sets rich
presence (`status`) through its services; without Steam they are answered
`unavailable`. None of this has run against a live Steam client ("Limits").

### Crash reports

A panic on any thread writes a report to `crashes/` in the data directory and
prints its path: what panicked where, the backtrace, the build, the system,
the GPU and the last 30 frame times, at most three reports a run.
`PEN_CRASH_TEST=N` panics at frame N, to try it on a platform, and
`PEN_BUILD` set while building (a commit, say) goes into the version line.
Shipped runtimes are stripped, so a backtrace has addresses; `addr2line`
against the unstripped build turns them into functions and lines
(docs/shipping.md, "Crash reports").

### Sizes

On 2026-10-11, on Linux:

| | Size |
|---|---|
| Development build (`cargo build --release -p pen`, without Steam) | 33.7 MB |
| Linux player runtime (`just player`, stripped) | 12.7 MB, and 0.4 MB of `libsteam_api.so` |
| Windows player runtime (`just windows`, stripped) | 9.7 MB, and 0.3 MB of `steam_api64.dll` |
| Their unstripped builds, kept for crash reports | 64.1 MB and 45.4 MB |
| The game's assets | 110.1 MB |

docs/shipping.md ("Sizes", "Build profiles") measured what wasmtime's
features, Cranelift and each profile setting cost, earlier the same day, when
the Linux player was 10.2 MB. The assets are most of a package:
[docs/texture-compression.md](docs/texture-compression.md) looks at shrinking
them.

## Performance

Measured on the development machine: AMD Ryzen 7 3800X (Zen 2, 16 threads),
NVIDIA RTX 2070 Super, CachyOS (Linux, Wayland, Vulkan).

**Guest time.** The same 600 frames (gdev's default scene, 1280x720, one
60 Hz step each, into play at frame 60), median and p95 after 120 warm-up
frames, with `pen bench --game dist/gdev` and `tools/bench_v8.ts` on the same
`game.wasm`. Three runs each on 2026-10-11, load average about 1:

| Engine | Median | p95 |
|---|---|---|
| V8 (Deno, JSPI) | 0.57–0.60 ms | 0.61–0.64 ms |
| wasmtime, Cranelift without inlining (as `pen dev` compiles) | 0.36–0.38 ms | 0.37–0.39 ms |
| wasmtime, Cranelift with inlining (every other command) | 0.32 ms | 0.34–0.35 ms |

**A frame.** Medians over 600 frames with `PEN_PROFILE_FRAMES=600`, gdev in
play, from docs/determinism.md ("Overlapped frames", 2026-10-11): in a window at
1440x960 with `PEN_PRESENT_MODE=immediate`, a frame takes about 4.5 ms, of
which about 0.7 ms is the guest and 3.6 to 3.7 ms rendering and presenting; at
640x427, about 2.6 ms; headless at 1440x960, about 3 ms.

**Overlapped frames.** `--overlap` computes the next frame in the guest while
this one draws and presents. It hides the guest's time entirely, but rendering
and presenting grow by about as much while the guest runs alongside them, so
the frame interval does not improve today, and headless it gets worse. It
stays opt-in for that reason and because it adds a frame of input latency; it
may pay off when the guest takes several milliseconds (docs/determinism.md,
"Overlapped frames").

**Hot reload.** The development engine compiles gdev's first build in about
105 ms and a reload in about 50 ms (blot-host's `dev_engine`; the date was
not recorded). The runtime's side of a reload, from a new `game.wasm` to the
new build running, takes about 90 ms; the Blot compiler's build before it,
1 to 2 s (gdev's README).

The player runtime's build profiles cost no measurable frame time
(docs/shipping.md, "Build profiles", measured 2026-10-11 on a machine loaded
by other builds).

## Limits

The runtime is developed and tested on one machine: Linux (Wayland), Vulkan,
NVIDIA. These have not run, or have known gaps:

- **Windows** has never run. `just windows` builds and `pen build --runtime
  …/pen.exe` packages, but there is no Windows machine or wine here;
  docs/shipping.md lists what to check on a first run.
- **macOS** is not built: there is no macOS SDK here. Only docs/shipping.md
  ("macOS") covers what it would take.
- **Steam** has never run against a live Steam client; only the way without
  one has. Achievements, stats, rich presence, the Steam overlay and
  Auto-Cloud are untested.
- **The Steam Deck and the Steam Linux Runtime containers**: the player
  runtime needs only glibc 2.28, but it has not run in a container or on a
  Deck. gdev has no gamepad support, so on the Deck Steam Input's keyboard and
  mouse template is what makes it playable.
- **`--software` and lavapipe** have not run here (lavapipe is not
  installed), and CI depends on them.
- **CI** has never run: the repository has no git remote yet.
- **Function names in traps**: the runtime's wasmtime names guest functions in
  a trap's backtrace from the module's name section, but Blot does not emit
  one yet. That waits on a pending blot2 patch; until then traps show function
  indices.
- **A recording is tied to its build**: it replays exactly only on the build
  that made it.
- **Renderer parity skips full-screen insets** where Deno's WebGPU cannot
  allocate their targets ("Testing").
- **No texture compression** yet; docs/texture-compression.md is the
  investigation.
- **The data directory is named `gdev`**; a store build should use the game's
  own name (docs/shipping.md, "Steam Auto-Cloud").
