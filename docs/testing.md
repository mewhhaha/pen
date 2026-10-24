# Testing tools

Beyond the suites `just test` runs (README, "Testing"), pen has tools for
what goes wrong between builds and in players' hands: a review page for the
golden images, replays that say where another build departs, input fuzzing,
property tests between the Rust and Blot codecs, bug reports, and benchmarks
compared from commit to commit. None of them needs a window; only the golden
images and recording a session need the GPU.

## Reviewing golden images

Every `pen test` run writes `.pen/visual/review.html` (or `--out DIR`'s): each
shot beside its golden image (copied to `SCENARIO/SHOT.golden.png`, so the
directory stands alone, as CI's `visual-diffs` artifact) and their
difference, four times as bright, with the RMSE, the largest difference and
the share of pixels changed. Shots that did not pass come first in a filter.

`pen test --review [--port N]` (`just visual --review`) then serves the page
on the loopback address until stopped, with a button on each shot that makes
it the golden image (`tests/visual/golden/SCENARIO/SHOT.png`, or `--golden
DIR`'s). The server takes only the run's own shots, only as JSON from its own
page (Host and Origin are checked, as `pen web`'s Bake is). `pen test
--review --last` serves the last run's page again (its shots are listed in
`review.json` beside it) without running or drawing anything, so the GPU is
needed only while the shots are taken. Opened as a file,
the page has no buttons and says how to accept from the command line (`pen
test --update --only SCENARIO`).

## Replaying on another build

A recording replays exactly only on the build that made it. `pen diverge
REC.penrec --against OLD` plays it on this build (the checkout's, or `--game
DIR`) and on another (a `pen export` bundle, or a package's `game/`) side by
side, without rendering, and compares each frame's packet and the state
after it. At the first frame either differs it says which fields of the
packet differ, and which lanes of the state, by saved section (gdev's
`ink/persist` layout: `key, version, width, length`, then lanes; a
component's are records `width + 1` lanes wide, its flag first), and fails:

```
frame 61 (1.02 s in) differs (the reference's values first, then this build's):
its packet:
  draws[0]: Draw > transform: > 0.256,
    again: 0.3,
the state after it:
  section 4 (version 1): entity 1 lane 1: 0.256 -> 0.3; entity 2 lane 1: 0.256 -> 0.3; … and 16 more
  section 117 (version 1): lane 7: 0.256 -> 0.3; lane 11: 0.256 -> 0.3; … and 4 more
Error: the builds diverge at frame 61 (1.02 s in)
```

(That is a recording of gnorps being hired, replayed on a build whose gnorps
walk at another height.) Sections only one build saves, or saves at another
version, are named as such. Because it fails on a difference, it bisects:

```sh
pen export --project ../gdev --out /var/tmp/good    # at the commit the recording was made on
git -C ../gdev bisect start HEAD GOOD_COMMIT
git -C ../gdev bisect run pen diverge session.penrec --project ../gdev --against /var/tmp/good
```

A bug report (below) is a recording too: `pen diverge report.penbug ...`.

## Input fuzzing

`pen fuzz [--seed N] [--runs N] [--frames N]` (`just fuzz`) plays runs of
random steps, each run's steps decided by its seed (splitmix64): keys and
held modifiers, clicks, drags, moves and the wheel anywhere on the page (now
and then off it), console commands, the checkout's tunables within their
ranges, and actions on the nodes the game's interface shows at the time
(mostly ones a node says it takes). It runs the program as `pen packets`
does, without rendering, with the game's own quicksaves kept in memory and
every service answered by a stand-in (nothing is opened, copied or asked of
Steam). After each frame it checks the game did not trap, its packet decoded
and validated, and every 30 frames that its state is a bounded array of
finite lanes.

A run that breaks one is cut down: chunks of its steps are dropped, halves
first and single steps last, while the same invariant still breaks. What is
left is written to `.pen/fuzz/seed-N.steps`; `pen fuzz --script-file FILE`
checks it again and `pen run --headless --script-file FILE` plays it to
watch. Unit tests check that a seed names the same steps every time and the
cutting down; a test on `dist/gdev` fuzzes two runs and cuts a run that
trips an extra check (a node taking the keyboard focus) down to the one
step that does it. Time travel steps and checks that rewinds give the same
packets are not fuzzed yet.

## The Rust and Blot codecs agree

pen-codegen generates the Rust codecs (`encode_frame`, `decode_frame`) and the
game's Blot ones (`decode_input`, `encode_render`) from `protocol/schema.json`.
A round trip through one side's encoder and decoder could not catch the two
sides laying a field out differently, so the test compares values across
languages through a canonical walk, also generated: every field in schema
order, arrays as a count and their elements, options as a flag and the value,
events as their tag and fields. Rust gets `canonical_input` and
`canonical_render` (in `packets.rs`), Blot `canonical_input` and
`render_from_canonical` (`pen/protocol_canonical.blot`, which only tests
import).

`cargo test -p pen codec_round_trip` compiles a harness
(`crates/pen/src/codec_round_trip.blot`) with the checkout's compiler, then
for 2000 seeded random frames each way checks that Rust's `encode_frame` read
by the game's `decode_input` walks as Rust walks the frame, and that a random
render frame built by the game from its walk, encoded by the game and
decoded by Rust is the frame. A failure names the seed
(`PEN_CODEC_SEED`, `PEN_CODEC_CASES` run one alone), the lane and the field.
Swapping two lanes in either side's encoder, decoder or walk fails it.

## Bug reports

Ctrl+Shift+B in a window (`report[=FILE]` in a script) writes a `.penbug` to
`reports/` in the data directory, the latest 20 kept: the last 30 s of input
as a recording, the game's state, the session's log (the dev tools' kept
lines; the player runtime keeps none) and what crash reports say of the
build, the system and the GPU, with the game's `game.wasm` fingerprint. To
have the 30 s, windows keep a history (a moment every 30 frames, as `pen
dev` always has) and every history now keeps 30 s rather than 10.

It is a ustar archive (`tar -xf` opens it): `report.txt`, `log.txt`,
`recording.penrec` and `state.bin`. `pen report FILE` prints the report, the
log's end and what the recording holds (`--out DIR` unpacks it), and
`--replay FILE` takes the report as its recording. A history kept across a
hot reload spans two builds, and such a report's recording replays on
neither alone.

## Benchmarks per commit

`pen bench --json` prints its results as JSON: the guest's median, mean and
p95 frame time past the warm-up, the median snapshot (taken every 30
frames, as a history's moments are), and loading the module (compiled, not
from the cache), with pen's version and runtime mark and the game.wasm's
fingerprint. `--runs N` runs the frames N times and keeps each metric's
best. `--compare BASE.json` (again for the best of several) compares a run
with another build's and fails when a metric is slower than its threshold
(the median and mean 10%, p95 and snapshots 20%, loading 25%; `--threshold
PCT` for all), ignoring changes under 5 µs.

`.github/workflows/bench.yml` does this for every push and pull request:
it builds this commit's pen and the base commit's (a pull request's base,
else the commit before) on the same runner, benches the game with each
(best of 5 runs), compares, and keeps both results as the artifact
`bench-SHA`. It needs the same checkouts and variables as CI's game job. A
base from before `--json` leaves nothing to compare, with a notice.

## Blot tests' speed

`pen test-blot` says how long each test that takes a second or more ran, and
how many passes the guest runtime's cycle collector made meanwhile (read from
the guest's exported `blot:cycle_passes` counter, `Guest::runtime_counters`).
gdev's `tests/codec_saved_state.blot` is the slow module: its 19 tests start
from `created ()`, the tests' level with a crowd of 10,000 gnorps created
and saved, and several restore and save it again.

Measured on 2026-10-16 and 17 with the sibling blot2's compiler of that day,
on a machine shared with other builds (load 15 to 50 on 16 threads):

- Creating the sandbox alone (`runtime.create sandbox state_schema`) took
  196 s with the development engine and 195 s with the optimizing one
  (Cranelift with inlining, no epoch checks): the engine does not matter.
  Alone at a load of about 19 it took 248 s, 245 s of it CPU time.
- Sampling the guest's stack every 100 ms (an epoch callback taking a
  backtrace) put 3074 of 3078 samples, 99.9%, in the Blot runtime's cycle
  collection pass and what it calls; 0.1% were the game's code. The pass
  (blot2's `zig-native/SHARED_OWNERSHIP.md`, "Complete cycle strategy")
  is trial deletion over the union of the strong closures of every queued
  candidate. It runs whenever candidates are queued and 64 KiB have been
  allocated (or 65,536 candidates queued) since the last, and a candidate
  that reaches the world makes the pass walk all of it.
- Creating it made 20,249 passes. With the crowd at 2,500 gnorps it took
  1.9 s (1,507 passes), at 5,000 37.6 s (5,136 passes), at 10,000 248 s
  (20,249 passes): passes grow about fourfold and the time 7 to 20 fold
  for each doubling of the world.
- The module before (one test at a time, as pen ran a module's tests): the
  five tests `--filter crowd` matches took 10,940 s, one of them stopped
  after 7,205 s. After (a module's tests in parallel): all 19 took 1,744 s
  of wall time (6,822 s of CPU), 7 passing and 12 stopped by the 600 s
  deadline.

The collector is blot2's, generated into every guest, which neither pen nor
gdev can change: its passes need to scale with what changed rather than
with what candidates reach (for example, a trigger that grows with the
heap, or not walking into structures already traced live in an earlier
pass). Until then the codec module costs minutes per test. A smaller crowd
for the tests that do not need 10,000 gnorps would cut them to seconds, but
the crowd is what those tests are about, so that is gdev's call.

What pen does: a module's tests run in parallel (up to `--jobs` at once
over all modules, each in its own instance on its own thread), so a module
of slow tests takes about as long as its slowest few rather than all of them
in turn; the deadline a test may run is 600 s (it was 120 s), and
`--deadline SECONDS` sets another, since the time a test waits for a core
counts.

Set `PEN_TEST_BUNDLE` to an isolated `pen export` directory when qualifying
concurrent work. An explicitly selected missing bundle fails, even without
`PEN_REQUIRE_FIXTURES`; it never falls back to a stale `dist/gdev`.

The native/browser conformance fixture compares every rendered packet and the
final save file byte for byte. `pen packets` writes `native.snapshot` using the
same snapshot container as the browser; this witness is outside benchmark timing
samples. `just conformance` serves isolated responses so `PEN_WEB_WITHOUT=jspi`
can exercise the worker fallback as well.
