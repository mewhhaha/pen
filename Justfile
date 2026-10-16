# Override with `just gdev=PATH ...`.
gdev := "../gdev"
pen := "cargo run --release -q -p pen --"
# The browser tests' tools, pinned (CI installs them with `just browser-tools`
# too): pnpm, run through npx, and playwright-core, which picks Chromium.
pnpm := "npx --yes pnpm@12.8.2"
playwright := "1.63.0"

# Then removes what earlier installs left in plugins/ and web/ that this one
# no longer builds, saying what it removes.
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
  find ~/.local/lib/pen-runtime/plugins -mindepth 1 -maxdepth 1 ! -name blot.wasm \
    -exec echo removing stale {} \; -exec rm -rf {} +
  find ~/.local/lib/pen-runtime/web -mindepth 1 -maxdepth 1 ! -name pen_web.js ! -name pen_web_bg.wasm ! -name guest.js ! -name index.html \
    -exec echo removing stale {} \; -exec rm -rf {} +
  cp LICENSE THIRD_PARTY_NOTICES.md ~/.local/lib/pen-runtime/
  ln -sf ~/.local/lib/pen-runtime/pen ~/.local/bin/pen

plugins:
  cd plugins && cargo build --release --target wasm32-wasip2

plugin-tests:
  cd plugins && cargo test

# The browser host `just install` ships, optimized by wasm-opt.
web-host:
  @command -v wasm-opt > /dev/null || { echo "web-host: wasm-opt is not on PATH; install binaryen (https://github.com/WebAssembly/binaryen/releases)" >&2; exit 1; }
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

# Render dist/packets headlessly; time loading and each frame.
render-packets *args:
  cargo run --release -p pen-render --example render_packet -- --sequence --veil {{args}}

visual *args:
  {{pen}} test --project {{gdev}} tests/visual/scenarios.json {{args}}

# Compile times of the release and development engines, first and again.
compile-time: export
  cargo run --release -q -p blot-host --example compile_time dist/gdev/game.wasm dist/gdev/game.wasm

blot-tests *args:
  {{pen}} test-blot --project {{gdev}} {{args}}

# Cache browser tools outside the source tree; an existing module can be supplied.
[private]
browser-tools:
  #!/usr/bin/env bash
  set -euo pipefail
  if [ -n "${PLAYWRIGHT_MODULE:-}" ]; then exit 0; fi
  tools=target/browser-tools
  if [ "$(node -p "require('./$tools/node_modules/playwright-core/package.json').version" 2>/dev/null)" != "{{playwright}}" ]; then
    mkdir -p "$tools"
    [ -f "$tools/package.json" ] || printf '{"private":true}\n' > "$tools/package.json"
    {{pnpm}} --dir "$tools" add --save-exact "playwright-core@{{playwright}}"
  fi
  node "$tools/node_modules/playwright-core/cli.js" install chromium

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

# The exported game in headless Chromium on the GPU (Vulkan): the game, the
# backquote console, diagnostics, the timeline, tunables and builds panels,
# a resize, and the autosave a reload restores, shot into
# target/browser-shots. PEN_WEB_GPU=software uses SwiftShader, which a full
# scene can outlast.
# The browser game, console, panels and autosave, captured.
browser: browser-tools
  #!/usr/bin/env bash
  set -euo pipefail
  {{pen}} build --web --project {{gdev}} --out dist/web
  export PLAYWRIGHT_MODULE="${PLAYWRIGHT_MODULE:-$PWD/target/browser-tools/node_modules/playwright-core/index.mjs}"
  browser_port=${PEN_BROWSER_PORT:-8043}
  python3 -m http.server --bind 127.0.0.1 --directory dist/web "$browser_port" > target/browser-server.log 2>&1 &
  browser_server=$!
  trap 'kill "$browser_server"' EXIT
  PEN_WEB_URL="http://127.0.0.1:$browser_port/" node crates/pen-web/tests/browser.mjs

# `pen web` in headless Chromium on the GPU (Vulkan): its Bake endpoint and
# validation, and a touched FILE of the game (its timestamp is restored)
# reloading the game, dev tools, catalog, shaders and textures; then the
# static export served from a subdirectory.
# `pen web`'s reload and Bake, and a static export beneath a directory.
browser-dev file="src/assets/catalog.json": browser-tools
  #!/usr/bin/env bash
  set -euo pipefail
  {{pen}} build --web --project {{gdev}} --out dist/web
  export PLAYWRIGHT_MODULE="${PLAYWRIGHT_MODULE:-$PWD/target/browser-tools/node_modules/playwright-core/index.mjs}"
  dev_port=${PEN_DEVELOPMENT_PORT:-8045}
  static_port=$((dev_port + 1))
  {{pen}} web --project {{gdev}} --port "$dev_port" > target/pen-web.log 2>&1 &
  dev_server=$!
  python3 -m http.server --bind 127.0.0.1 --directory dist "$static_port" > target/browser-server.log 2>&1 &
  static_server=$!
  trap 'kill "$dev_server" "$static_server"' EXIT
  until curl -fs "http://127.0.0.1:$dev_port/__pen/revision" > /dev/null; do
    kill -0 "$dev_server" 2> /dev/null || { cat target/pen-web.log; exit 1; }
    sleep 1
  done
  PEN_DEVELOPMENT_URL="http://127.0.0.1:$dev_port/?dev" PEN_RELOAD_FILE="{{gdev}}/{{file}}" \
    PEN_STATIC_URL="http://127.0.0.1:$static_port/web/" node crates/pen-web/tests/development.mjs

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
  {{pen}} codegen --project {{gdev}} --rust crates/pen-protocol/src/packets.rs

# Includes the full Blot suite, not only smoke fixtures.
test: check visual determinism blot-tests conformance

# Clippy lints pen in every combination of its features (the default one
# and the player runtime's among them) through cargo-hack.
# Protocol freshness, clippy and the Cargo tests.
# rustfmt.toml formats both workspaces; packets.rs is generated and skipped.
check: plugins plugin-tests packets
  {{pen}} codegen --project {{gdev}} --rust crates/pen-protocol/src/packets.rs --check
  cargo fmt --all --check
  cd plugins && cargo fmt --all --check
  cargo clippy --workspace --all-targets --features pen/steam -- -D warnings
  cargo hack clippy -p pen --feature-powerset --exclude-features default --all-targets -- -D warnings
  cargo test --workspace -- --test-threads=1
  cargo test --release -p pen-render -- --test-threads=1
  cargo clippy -p pen-web --target wasm32-unknown-unknown -- -D warnings

# Needs cargo-deny; the policy is deny.toml.
# Licenses, advisories, bans and sources of both lockfiles.
deny:
  for manifest in Cargo.toml plugins/Cargo.toml; do cargo deny --manifest-path "$manifest" check; done

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

# As x86_64-pc-windows-gnu, with no Windows SDK or MSVC: zig compiles
# libopus (opusic-sys, through CMake) as it links. What ships is
# target/x86_64-pc-windows-gnu/dist/player/.
# The Windows player runtime, cross-compiled with zig.
windows:
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
