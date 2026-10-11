# Override with `just gdev=PATH ...`.
gdev := "../gdev"
pen := "cargo run --release -q -p pen --"

# Install the native CLI, compiler component and browser host.
install: plugins web-host
  cargo build --release -p pen --features steam
  mkdir -p ~/.local/lib/pen-runtime/plugins ~/.local/lib/pen-runtime/web
  cp target/release/pen ~/.local/lib/pen-runtime/pen.new
  mv -f ~/.local/lib/pen-runtime/pen.new ~/.local/lib/pen-runtime/pen
  cp "$(ls -t target/release/build/steamworks-sys-*/out/libsteam_api.so | head -1)" ~/.local/lib/pen-runtime/libsteam_api.so.new
  mv -f ~/.local/lib/pen-runtime/libsteam_api.so.new ~/.local/lib/pen-runtime/libsteam_api.so
  cp plugins/target/wasm32-wasip2/release/pen_plugin_blot.wasm ~/.local/lib/pen-runtime/plugins/blot.wasm.new
  mv -f ~/.local/lib/pen-runtime/plugins/blot.wasm.new ~/.local/lib/pen-runtime/plugins/blot.wasm
  cp crates/pen-web/pkg/{pen_web.js,pen_web_bg.wasm,guest.js,index.html} ~/.local/lib/pen-runtime/web/
  cp LICENSE THIRD_PARTY_NOTICES.md ~/.local/lib/pen-runtime/
  ln -sf ~/.local/lib/pen-runtime/pen ~/.local/bin/pen

plugins:
  cd plugins && cargo build --release --target wasm32-wasip2

plugin-tests:
  cd plugins && cargo test

web-host:
  sh crates/pen-web/build.sh

dev *args:
  {{pen}} dev --project {{gdev}} {{args}}

web *args:
  {{pen}} web --project {{gdev}} {{args}}

run *args:
  {{pen}} run --project {{gdev}} {{args}}

headless *args:
  {{pen}} run --project {{gdev}} --headless {{args}}

package:
  {{pen}} build --project {{gdev}} --out dist/package

bench:
  {{pen}} bench --project {{gdev}} --frames 600

visual *args:
  {{pen}} test --project {{gdev}} tests/visual/scenarios.json {{args}}

blot-tests *args:
  {{pen}} test-blot --project {{gdev}} {{args}}

# Cache browser tools outside the source tree; an existing module can be supplied.
[private]
browser-tools:
  #!/usr/bin/env bash
  set -euo pipefail
  if [ -n "${PLAYWRIGHT_MODULE:-}" ]; then exit 0; fi
  if [ ! -f target/browser-tools/node_modules/playwright-core/index.mjs ]; then
    mkdir -p target/browser-tools
    printf '{"private":true}\n' > target/browser-tools/package.json
    pnpm --dir target/browser-tools add playwright-core@latest
  fi
  pnpm --dir target/browser-tools exec playwright-core install chromium

conformance: browser-tools
  #!/usr/bin/env bash
  set -euo pipefail
  {{pen}} build --web --project {{gdev}} --out dist/web
  {{pen}} packets target/browser-conformance --project {{gdev}} --script-file tests/conformance.steps
  export PLAYWRIGHT_MODULE="${PLAYWRIGHT_MODULE:-$PWD/target/browser-tools/node_modules/playwright-core/index.mjs}"
  browser_port=${PEN_CONFORMANCE_PORT:-8046}
  python3 -m http.server --bind 127.0.0.1 --directory dist/web "$browser_port" > target/browser-server.log 2>&1 &
  browser_server=$!
  trap 'kill "$browser_server"' EXIT
  PEN_CONFORMANCE_URL="http://127.0.0.1:$browser_port/qualification.html" node crates/pen-web/tests/conformance.mjs

# Game/scene/assets consumed by protocol, renderer and determinism tests.
export:
  {{pen}} export --project {{gdev}} --out dist/gdev

packets: export
  {{pen}} packets dist/packets-run --project {{gdev}} --frames 120
  rm -rf dist/packets && mkdir -p dist/packets
  cp dist/packets-run/schema.txt dist/packets/
  for path in dist/packets-run/output/*.f32; do name=$(basename "$path" .f32); cp "$path" "dist/packets/${name#0}.f32"; done
  rm -rf dist/packets-run

sync-protocol:
  cargo run -q -p pen-codegen -- --project {{gdev}} --rust crates/pen-protocol/src/packets.rs

# Includes the full Blot suite, not only smoke fixtures.
test: check visual determinism blot-tests conformance

check: plugins plugin-tests packets
  cargo run -q -p pen-codegen -- --project {{gdev}} --rust crates/pen-protocol/src/packets.rs --check
  cargo clippy --workspace --all-targets --features pen/steam -- -D warnings
  cargo clippy -p pen --all-targets --no-default-features --features steam -- -D warnings
  cargo test --workspace -- --test-threads=1
  cargo test --release -p pen-render -- --test-threads=1
  cargo clippy --manifest-path crates/pen-web/Cargo.toml --target wasm32-unknown-unknown -- -D warnings

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
  cp LICENSE THIRD_PARTY_NOTICES.md {{dir}}/player/

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
