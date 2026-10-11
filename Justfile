# The gdev checkout the recipes work on (override with `just gdev=PATH ...`).
gdev := "../gdev"
# The runtime, built with `cargo build --release` (`just install` puts a
# Steam-enabled one on PATH as `pen`).
pen := "cargo run --release -q -p pen --"

# Build the runtime with Steam and install it as ~/.local/bin/pen, beside
# the Steam library it links against.
# Installs the Steam-enabled runtime as ~/.local/bin/pen.
install:
  cargo build --release -p pen --features steam
  mkdir -p ~/.local/lib/pen-runtime
  cp target/release/pen ~/.local/lib/pen-runtime/pen
  cp "$(ls -t target/release/build/steamworks-sys-*/out/libsteam_api.so | head -1)" ~/.local/lib/pen-runtime/
  ln -sf ~/.local/lib/pen-runtime/pen ~/.local/bin/pen
  @echo "installed: pen dev | web | run | build | test | parity | bench | packets (in a gdev checkout)"

# Develop natively: Blot, shader and asset edits reload into the running
# game with its state kept (`just dev --mute`, ...).
# Develops natively with hot reload.
dev *args:
  {{pen}} dev --project {{gdev}} {{args}}

# Develop in the browser: gdev's own hot-reloading web build.
web:
  cd {{gdev}} && just web

# Compile once and play natively.
run *args:
  {{pen}} run --project {{gdev}} {{args}}

# Run headless: `just headless --frames 300 --record out.mp4`, or with a --script.
headless *args:
  {{pen}} run --project {{gdev}} --headless {{args}}

# A shippable folder in dist/package; with Steam from `just install`'s build.
package:
  {{pen}} build --project {{gdev}} --out dist/package

# Guest time per frame on wasmtime, and the same frames on V8 for comparison.
bench:
  {{pen}} bench --project {{gdev}} --frames 600
  deno run -A --no-lock --config {{gdev}}/deno.json tools/bench_v8.ts {{gdev}} 600

# Golden-image tests (tests/visual): `just visual --update` accepts what the
# scenarios show now; `just visual --only menu` runs one.
# Golden-image tests; `--update` accepts what they show now.
visual *args:
  {{pen}} test --project {{gdev}} tests/visual/scenarios.json {{args}}

# The native renderer against the checkout's renderer.ts on the same frames.
parity *args:
  {{pen}} parity --project {{gdev}} {{args}}

# Engine conformance: the same scripted input on wasmtime and on V8 (Deno)
# must give bit-identical render packets.
# Wasmtime and V8 give bit-identical packets on the same input.
conformance:
  {{pen}} packets dist/conformance --project {{gdev}} --script-file tests/conformance.steps
  deno run -A --no-lock --config {{gdev}}/deno.json tools/conformance_v8.ts {{gdev}} dist/conformance

# The asset bundle some renderer tests and examples read (dist/gdev).
export:
  deno run -A --no-lock --config {{gdev}}/deno.json tools/export_gdev.ts {{gdev}} dist/gdev

# The packets pen-protocol's real-packet test and the render_packet example
# read: dist/packets holds NNNN.f32 and schema.txt (the assets in dist/gdev),
# the first 120 frames of a headless run.
# Writes dist/packets, the real packets the protocol tests read.
packets: export
  {{pen}} packets dist/packets-run --project {{gdev}} --frames 120
  rm -rf dist/packets && mkdir -p dist/packets
  cp dist/packets-run/schema.txt dist/packets/
  for path in dist/packets-run/output/*.f32; do name=$(basename "$path" .f32); cp "$path" "dist/packets/${name#0}.f32"; done
  rm -rf dist/packets-run

# Regenerate crates/pen-protocol/src/packets.rs from the checkout's
# protocol/schema.ts, the packets' one description (gdev's README, "Host
# boundary"); `just check` fails while it is stale.
# Regenerates the Rust protocol from gdev's protocol/schema.ts.
sync-protocol:
  deno run --allow-read --allow-write --allow-env --no-lock --config {{gdev}}/deno.json {{gdev}}/scripts/generate_protocol.ts --only rust --rust crates/pen-protocol/src/packets.rs

# Everything: clippy, unit and GPU tests, golden images, renderer parity,
# engine conformance and the determinism contract.
# Runs every check.
test: check visual parity conformance determinism

# The generated protocol is fresh, clippy and tests; the renderer's GPU tests
# also in release, where the optimizer once broke what debug builds hid.
# The protocol is fresh, clippy is clean and the tests pass.
check:
  deno run --allow-read --allow-env --no-lock --config {{gdev}}/deno.json {{gdev}}/scripts/generate_protocol.ts --only rust --rust crates/pen-protocol/src/packets.rs --check
  cargo clippy --workspace --all-targets --features pen/steam
  cargo test --workspace
  cargo test --release -p pen-render

# No Wasm compiler (it loads the game.cwasm its package was precompiled
# with), Steam, the `dist` profile, and glibc 2.28 and later (every Steam
# Linux Runtime container: soldier, which the default scout runtime runs in,
# and sniper; SteamOS; Debian 10, Ubuntu 18.10 on); needs zig and
# cargo-zigbuild. Linking allows the host's newer libasound and libdbus,
# which the player's system replaces with its own. What ships is
# target/x86_64-unknown-linux-gnu/dist/player/; see docs/shipping.md.
# The Linux player runtime `pen build` ships.
player:
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=-Wl,--allow-shlib-undefined" \
    cargo zigbuild --profile dist -p pen --no-default-features --features steam --target x86_64-unknown-linux-gnu.2.28
  just glibc target/x86_64-unknown-linux-gnu/dist/pen
  just stripped target/x86_64-unknown-linux-gnu/dist pen libsteam_api.so

# Beside `just install`'s pen, in player/, where `pen build` looks for the
# runtime to ship.
# Installs `just player`'s runtime for `pen build`.
install-player: player
  mkdir -p ~/.local/lib/pen-runtime/player
  cp target/x86_64-unknown-linux-gnu/dist/player/* ~/.local/lib/pen-runtime/player/

# As x86_64-pc-windows-gnu, with no Windows SDK or MSVC; libopus builds
# through packaging/zig's CMake toolchain, which drops the MSVC-only flag
# opus-head-sys passes. What ships is target/x86_64-pc-windows-gnu/dist/player/.
# The Windows player runtime, cross-compiled with zig.
windows:
  CMAKE_TOOLCHAIN_FILE_x86_64_pc_windows_gnu="{{justfile_directory()}}/packaging/zig/windows-gnu.cmake" \
    cargo zigbuild --profile dist -p pen --no-default-features --features steam --target x86_64-pc-windows-gnu
  just stripped target/x86_64-pc-windows-gnu/dist pen.exe steam_api64.dll

# The checkout's game with the player runtimes: dist/linux (Steam Deck
# included) and dist/windows.
# Shippable folders for Linux and Windows.
ship: player windows
  {{pen}} build --project {{gdev}} --runtime target/x86_64-unknown-linux-gnu/dist/player/pen --out dist/linux
  {{pen}} build --project {{gdev}} --runtime target/x86_64-pc-windows-gnu/dist/player/pen.exe --out dist/windows

# DIR keeps the unstripped build, whose symbols and line tables resolve its
# crash reports (keep one per release); the copy without them ships.
# A dist build's runtime, stripped, and Steam's library, into DIR/player/.
[private]
stripped dir exe library:
  mkdir -p {{dir}}/player
  objcopy --strip-all {{dir}}/{{exe}} {{dir}}/player/{{exe}}
  cp "$(ls -t {{dir}}/build/steamworks-sys-*/out/{{library}} | head -1)" {{dir}}/player/

# The newest glibc an executable needs; fails above 2.28 (the soldier container's).
glibc exe="target/x86_64-unknown-linux-gnu/dist/pen":
  #!/usr/bin/env sh
  newest=$(objdump -T "{{exe}}" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -n 1)
  echo "{{exe}} needs glibc ${newest#GLIBC_}"
  [ "$(printf '%s\n' "$newest" GLIBC_2.28 | sort -V | tail -n 1)" = GLIBC_2.28 ]

# The determinism contract (docs/determinism.md): the same script twice, and
# with overlapped frames, gives the same packets; a recording of it replays
# them bit for bit (without the quicksave it loaded); a run that rewinds,
# steps, goes live from the past and replays its last second gives every
# frame it runs again the packet it gave the first time; and its recording
# replays the packets of its final timeline. Exports first: its contract
# tests read dist/gdev, which a protocol change leaves stale.
# The determinism contract: replays and rewinds give the same packets.
determinism: export
  #!/usr/bin/env bash
  set -euo pipefail
  cargo test -q -p pen contract
  out=dist/determinism
  rm -rf "$out" && mkdir -p "$out"
  play() { name=$1; shift; {{pen}} run --project {{gdev}} --headless --data "$out/data-$name" --trace "$out/$name.trace" "$@" 2>"$out/$name.log" || { cat "$out/$name.log"; exit 1; }; }
  packets() { cut -d' ' -f1,3 "$1"; }
  play first --script-file tests/determinism.steps --record-input "$out/first.penrec"
  frames=$(wc -l < "$out/first.trace")
  play second --script-file tests/determinism.steps
  cmp <(packets "$out/first.trace") <(packets "$out/second.trace")
  echo "the same script twice: $frames frames, the same packets"
  play overlapped --script-file tests/determinism.steps --overlap
  cmp <(packets "$out/first.trace") <(packets "$out/overlapped.trace")
  echo "overlapped frames: the same packets"
  rm "$out/quick.bin"
  play replay --replay "$out/first.penrec"
  cmp "$out/first.trace" "$out/replay.trace"
  echo "its recording replayed: the same inputs and packets"
  play travel --script-file tests/determinism.steps --script-file tests/rewind.steps --record-input "$out/travel.penrec"
  # A frame that runs again after a rewind or replay (the trace goes back)
  # must give the packet it gave before, as long as every frame since the
  # restore ran with the input it had. Going live from the past starts a new
  # timeline at the first input that differs, and the frames after it on the
  # old one are gone.
  rerun() { awk '
    NR > 1 && $1 <= last { same = 1 }
    $1 in input && same && input[$1] == $2 { again++; if (packet[$1] != $3) { print "frame " $1 " gave another packet when it ran again"; wrong = 1 } }
    $1 in input && input[$1] != $2 { same = 0; for (frame in input) if (frame + 0 > $1 + 0) { delete input[frame]; delete packet[frame] } }
    { input[$1] = $2; packet[$1] = $3; last = $1 }
    END { if (!wrong) print again; exit wrong }' "$1"; }
  again=$(rerun "$out/travel.trace") || { echo "$again"; exit 1; }
  echo "time travel: $again frames ran again after rewinds, steps and a replay, each giving the packet it gave before"
  play travel-replay --replay "$out/travel.penrec"
  # The final timeline: each frame as it last ran, up to the run's last frame.
  final() { awk '{ last[$1] = $0; end = $1 } END { for (frame in last) if (frame + 0 <= end) print last[frame] }' "$1" | sort -n; }
  cmp <(final "$out/travel.trace") <(sort -n "$out/travel-replay.trace")
  echo "its recording replayed: the packets of its final timeline"
