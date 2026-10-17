# Shipping gdev

How `pen build` makes a package, what it runs on, and what a crash or Steam
needs. Measured on the development machine: AMD Ryzen 7 3800X (Zen 2, the
Steam Deck's CPU family), RTX 2070 Super, CachyOS (glibc 2.44).

## The player runtime

`pen build` ships a runtime beside the game:

```
package/
├─ gdev (gdev.exe)        the player runtime, named after the game
├─ libsteam_api.so        (steam_api64.dll, libsteam_api.dylib) when built with Steam
└─ game/
   ├─ game.cwasm          the Blot program, compiled to machine code ahead of time
   ├─ game.wasm           its Wasm, for the guest ABI manifest
   ├─ assets/
   └─ main.scene
```

The executable is named after the game (pen.json's `game.name`, kept in `game/game.json`,
which also names the data directory), whatever the runtime it copies is called (`pen`,
`pen.exe`); the runtime finds `game/` beside itself under any name.

The runtime it ships is the **player runtime**: pen built without the
`compiler` feature (`--no-default-features`), so without Cranelift. It only
loads the `game.cwasm` its package was precompiled with, and is less than
half the size of the development build (10.2 against 23.9 MB). Everything that compiles a checkout
(`dev`, `run`, `build`, `test`, `bench` and `packets` on a
checkout) refuses in it and says to use a full build.

- `just player` builds it for Linux (below) and `just windows` for Windows,
  each with the `dist` profile into `target/<target>/dist/`, then puts a
  stripped copy and Steam's library in `dist/player/`: that is what ships.
  `just install-player` installs the Linux one beside `just install`'s pen,
  in `~/.local/lib/pen-runtime/player/`, where `pen build` looks for the
  runtime to ship. Without one there it ships itself, and says so.
- `pen build --runtime EXE` ships another runtime. Its platform comes from
  its executable header (ELF, PE or Mach-O, and the machine), and
  `game.cwasm` is compiled for that platform; `--target TRIPLE` overrides it
  (cargo-zigbuild's `.2.28` suffix may stay on). The Steam library it links
  is found beside it or in cargo's `steamworks-sys` build output.
- `just ship` builds both player runtimes and packages the checkout's game
  for each, in `dist/linux` and `dist/windows`.
- `pen build --game DIR` packages a bundle compiled already (`pen export`'s,
  or a package's `game/`) instead of a checkout: no checkout, compiler
  plugin or Blot compiler needed, only a pen with Cranelift for the
  runtime's architecture. CI packages the game job's bundle this way for the
  Windows player, and on a Mac for the macOS one.

### A CPU baseline for game.cwasm

`game.cwasm` used to be compiled for the packaging machine's exact CPU, so
it could fail on a player's older one. It is now compiled for a portable
baseline, **x86-64-v2** (SSE4.2, POPCNT, CMPXCHG16B): every x86-64 CPU since
about 2009 has it, Windows 11 requires it, and it keeps Wasm's `floor`,
`ceil`, `trunc` and `nearest` single instructions (SSE4.1) rather than calls
into the host. `pen build --cpu PRESET` picks another Cranelift preset
(`x86-64-v3` for AVX2 and FMA, `znver2`, `baseline` for plain x86-64) or
`native`. This is `blot_host::portable_engine`: `Config::target` with the
triple, which starts Cranelift from the architecture's baseline instead of
detecting this CPU's features, plus the preset's ISA flags.

Loading checks the player's CPU against every feature the module was
compiled with, and wasmtime's other settings (inlining, tunables, Wasm
features) against the runtime's. The player runtime refuses what does not
match, with the reason:

```
Error: …/game/game.cwasm is missing, and this runtime has no compiler: it plays games packaged by `pen build`

Error: the precompiled game does not suit this runtime (it was made by another pen version, or for CPU features this machine lacks), and this runtime has no compiler to compile it again
Caused by:
    0: compilation settings of module incompatible with native host
    1: compilation setting "has_avx512dq" is enabled, but not available on the host
```

(The second is a `game.cwasm` compiled with `--cpu x86-64-v4` on this Zen 2,
which has no AVX-512.) A development build still compiles `game.wasm` again
when `game.cwasm` does not suit it.

For the two builds to agree, the workspace's wasmtime dependency has only
the features running a precompiled guest needs (`runtime`, `std`, `gc`,
`gc-drc` for `externref`, `backtrace` for named guest frames in traps,
`anyhow`), and blot-host's `compiler` feature adds only `cranelift`,
`incremental-cache` and `parallel-compilation`: a feature that changed the
engine's Wasm features or tunables on one side only would make every
`game.cwasm` incompatible.

What the baseline costs: guest time a frame (`pen bench`, 600 frames of
the default scene, the same player runtime, only `game.cwasm` differing),
median and minimum over eight interleaved rounds. The machine was shared
with other builds (load average about 40 on 16 threads), so differences of a
few percent are within the noise:

| `game.cwasm` compiled for | Guest median | Lowest median |
|---|---|---|
| this machine (`native`, Zen 2) | 0.51 ms | 0.47 ms |
| `x86-64-v3` | 0.49 ms | 0.46 ms |
| `x86-64-v2` (the default) | 0.52 ms | 0.49 ms |
| `baseline` (SSE2) | 0.53 ms | 0.50 ms |

## Sizes

The runtime, with Steam, on Linux:

| | Size |
|---|---|
| Before: `cargo build --release`, wasmtime's default features (what shipped) | 27.4 MB |
| Development build now (`just install`): wasmtime with only what gdev uses | 23.9 MB (30.2 MB with the egui dev overlay added later, since replaced by the checkout's own dev tools program; the player leaves the dev tools out) |
| Player runtime, `release` profile (no compiler) | 13.5 MB |
| **Player runtime, `dist` profile (`just player`, what ships)** | **10.2 MB** |
| Windows player runtime (`just windows`) | 9.2 MB |

These were measured before AccessKit, the clipboard and the other services
were added. Later the same day (2026-10-11) `just player` gave 12.7 MB and
`just windows` 9.7 MB, and the development build (`cargo build --release`,
without Steam) 33.7 MB.

Built from the same source, wasmtime's default features (the component
model, WAT, threads, the cache, profiling, coredumps, DWARF and more) are
3.9 MB of the development build (27.7 against 23.9 MB), and Cranelift with
its compile cache is 10.3 MB (23.9 against 13.5 MB for the player). A package of the current game shrinks from
135 MB to 124 MB: the runtime 10.2 MB (+ 0.4 MB Steam), `game.cwasm`
4.3 MB, `game.wasm` 0.9 MB and the assets 108.4 MB. The assets are the rest
of the story: see [texture-compression.md](texture-compression.md) (about
18 MB of them look unused, and mesh JSON is half of them).

## Build profiles

`[profile.release]` stays what development builds use (thin LTO, 4 codegen
units: an edit relinks in about a minute). Shipped runtimes use a new
`[profile.dist]`: fat LTO, one codegen unit and `panic = "abort"`. Every
variant below was built from one snapshot of the tree; frame times are
medians (and the lowest median) over six interleaved rounds of `pen bench`
(guest time) and of a headless 1440×960 run with `PEN_PROFILE_FRAMES=600`
(every phase of a frame), all with the same `game.cwasm`, on a shared,
loaded machine:

| Player runtime profile | Size | Build | Guest | Render + present | Frame |
|---|---|---|---|---|---|
| release (thin LTO, 4 units) | 13.53 MB | 26 s | 0.52 (0.49) ms | 1.82 (1.40) ms | 2.63 (1.95) ms |
| release + `panic = "abort"` | 12.14 MB | 25 s | 0.53 (0.50) ms | 1.99 (1.44) ms | 2.48 (1.97) ms |
| fat LTO, 1 unit | 11.37 MB | 169 s | 0.54 (0.49) ms | 1.79 (1.44) ms | 2.70 (2.36) ms |
| **fat LTO, 1 unit, abort (`dist`)** | **10.23 MB** | 138 s | 0.55 (0.49) ms | 1.67 (1.28) ms | 2.59 (2.14) ms |
| `dist` + `opt-level = "s"` | 7.93 MB | 175 s | 0.59 (0.56) ms | 1.84 (1.57) ms | 2.56 (2.36) ms |
| `dist` keeping its symbol table | 12.01 MB | 139 s | | | |

(Full builds, with the compiler: 23.87, 21.92, 21.04 and 19.06 MB in the
same order.) None of fat LTO, one codegen unit or `abort` costs measurable
frame time. `opt-level = "s"` would save another 2.3 MB but its guest and
render times rise about 5% against `dist` measured alongside it (lowest
medians 0.56 against 0.53 ms, 1.57 against 1.51 ms), so `dist` keeps
`opt-level = 3`. `panic = "abort"` also means a panic on a side thread (an
asset loader, the audio callback) ends the game with a crash report
instead of leaving it running without that thread. The guest's own code is
Cranelift's either way: the profiles only change the host around it.

## Store builds: Steam and itch.io

`pen build --steam` and `pen build --itch` (`publish.rs`) make a store build
of the package they build, from sections of the checkout's `pen.json`:

```json
"steam": {"app": 1234560, "depots": {"linux": 1234561, "windows": 1234562}, "branch": "beta"},
"itch": {"target": "studio/gdev", "channels": {"web": "html5"}}
```

- `--steam` writes steamcmd's build scripts in `steam/` beside the package:
  `depot_build_<depot>.vdf` maps the whole package into the depot of its
  platform (from its runtime's header: Linux, Windows, or a macOS
  application bundle), and `app_build_<app>.vdf` names every depot that has
  a script there, so `just ship --steam` (`dist/linux` and `dist/windows`)
  makes one build of both in `dist/steam/`. `branch` sets the build live on
  that branch once uploaded; `default` is refused, as Steamworks wants a
  person to set the default branch live, and so is app 480 (Spacewar).
- `--itch` pushes with butler to the platform's channel (`linux`,
  `windows`, `osx`, `html5` for `--web`; `channels` renames them):
  `butler push DIR studio/gdev:linux --if-changed`. butler uploads a patch
  against the channel's last build, not the package again, and nothing when
  it is unchanged.
- `--label TEXT` is the build's description on Steam and its version
  (`--userversion`) on itch.io.

Without `--upload` nothing is uploaded: each prints the command it would
run. `--upload` runs it, and only with credentials in the environment:
steamcmd as `STEAM_BUILD_USER`, the build account steamcmd has logged in as
before (pen never logs in or sees a password: steamcmd keeps the login of
the account's first interactive one, with Steam Guard), and butler with
`BUTLER_API_KEY`. `PEN_STEAMCMD` and `PEN_BUTLER` name the tools when they
are not on `PATH`. The tests run stand-ins that note their arguments, never
steamcmd or butler; nothing here was uploaded, and the scripts have not met
a real app.

## Release checks

### Signing the Windows player

`scripts/sign-windows.sh EXE...` signs Windows executables with
Authenticode through osslsigncode, timestamped, and checks each signature,
when `WINDOWS_CERTIFICATE` holds a code-signing certificate (a `.pfx` file,
or its base64 as a CI secret) and `WINDOWS_CERTIFICATE_PASSWORD` its
password; without one it says so and leaves the file unsigned. `just ship`
signs the Windows player before packaging it, and CI's player job signs it
before uploading it when the `WINDOWS_CERTIFICATE` and
`WINDOWS_CERTIFICATE_PASSWORD` secrets are set. Tried here with a
self-signed certificate (osslsigncode 2.9 built from source, no timestamp):
signed, verified against that certificate, and refused with a wrong password
or without the certificate to check against. The owner's step: buy an OV or
EV code-signing certificate and set the two secrets (an EV certificate on a
hardware token or in a cloud HSM needs that service's signing tool instead).

### Reproducible packages

The same checkout, runtime and options make the same package, byte for
byte. Its files hold nothing of when or where they were built (no paths, no
times, assets in catalog order); with `SOURCE_DATE_EPOCH` set (seconds since
1970, the convention of reproducible-builds.org) `pen build` gives every
file and directory that modification time, so an archive of the package is
the same bytes too (`reproducible.rs`). `just reproducible`
(`scripts/reproducible.sh`) builds the package twice, each with caches of
its own (the Blot program and `game.cwasm` compiled from nothing both
times), and fails unless every file, mode and time is the same; CI's game
job runs it.

### Budgets

| What | Budget | Measured (2026-10-17) |
|---|---|---|
| Linux player runtime (`just player`, stripped) | 19 MB | 17.15 MB |
| Windows player runtime (`just windows`, stripped) | 15 MB | 13.66 MB |
| The packaged game (`game/`) | 110 MB | 85.65 MB |
| Cold start to the first frame, this machine's GPU (`just cold-start`) | 8 s | 5.4 s (warm 1.6 s) |
| Cold start to the first frame, lavapipe (CI, one run) | 180 s | about 68 s, warm alike |

The times are medians of five runs of the `.2.28` player package on the RTX
2070 Super, and single runs on lavapipe, both while other builds kept the
machine's load average near 40 on its 16 threads, so an idle machine is
faster. Cold, most of the time is the GPU driver compiling pipelines pen has
not cached yet; lavapipe compiles them on the CPU every time.

`just size-budget` (`scripts/size-budget.sh FILE MB...`) fails when a
player runtime outgrows its budget, which CI's player job checks after
building them; the game job checks the packaged game. A change that needs
more raises the budget in the same change, saying why.

The player budgets went from 16 and 13 MB to 19 and 15 MB when the
simulation services (rapier, parry, nalgebra, spade), gamepads (gilrs),
compressed textures, localization and netplay joined the runtime: about
4.3 MB on each player. Leaving the simulation services out behind a feature
would win back most of it, for games that don't call them.

`just cold-start` (`scripts/cold-start.sh EXE MS [RUNS]`) starts a
package's runtime headless five times as a player's first run does, with
none of pen's caches (no compiled GPU pipelines), and times it from start to
exit after its first frame, which headless is drawn once every asset it asks
for has loaded; the median must be within the budget. It times five warm
starts too (the caches the first left), for comparison. CI's game job runs
it on lavapipe with a budget of its own. With a runtime that marks its
phases (`PEN_STARTUP=1`), the first run's are shown.

## Graphics presets

A game can declare graphics presets beside its assets, in
`src/presets.json`, which `pen build` ships as `game/presets.json`: its
preferences' lanes by name, and the values of some of them for each tier
(`presets.rs`; gdev's sets anti-aliasing, render scale, texture filtering,
shadows, ambient occlusion and bloom). While the player has no stored
preferences (the first run, until they change a setting and the game sends
its preferences to be kept), the host puts the GPU it opened in a tier and
plays that tier's preset to the game as Preference events, as it plays
stored preferences:

| Tier | Adapters |
|---|---|
| software | lavapipe, llvmpipe, WARP, SwiftShader (the CPU) |
| handheld | the Steam Deck (SteamOS says so, or its APU's name, VanGogh) |
| low | Intel's UHD and Iris integrated graphics, virtual GPUs |
| medium | Apple silicon, AMD's integrated Radeons, Intel Arc, anything unknown |
| high | discrete GPUs |

A tier the file leaves out takes the next lower one's. Headless runs keep
the game's own defaults, so that tests and golden images draw alike on
every machine, unless `PEN_GRAPHICS_TIER` asks: `auto` for the GPU's tier,
or a tier's name (also in a window, to try one). The browser host does not
pick presets yet.

## Platforms

### Linux and the Steam Deck

The Steam Deck runs SteamOS 3 (Arch-based, a recent glibc), but a native
Linux game on Steam runs inside a Steam Linux Runtime container whose glibc
can be older than the build machine's (2.44 here). By default a native game
gets "Steam Linux Runtime 1.0 (scout)", which Steam now runs inside the
Steam Runtime 2 "soldier" container (Debian 10, glibc 2.28); a game can opt
into 3.0 "sniper" (Debian 11, glibc 2.31) or 4.0 on its Steamworks
Installation → Linux Runtime page. (pressure-vessel uses the host's glibc
when it is newer, as on SteamOS.) `just player` therefore builds with
cargo-zigbuild for `x86_64-unknown-linux-gnu.2.28`, against zig's glibc 2.28
stubs, so the default runtime works too, and `just glibc EXE` checks the
result:

| Build | Newest glibc symbol |
|---|---|
| `cargo build --release` on this machine (before) | GLIBC_2.44 |
| zigbuild for `.2.31` | GLIBC_2.30 (`gettid`) |
| `just player` (zigbuild for `.2.28`) | GLIBC_2.28 (`statx`); `gettid` and `posix_spawn_file_actions_addchdir_np` become weak references that Rust's std checks for at run time |

The 2.31 build needed 2.29 and 2.30 only for newer versions of `exp`, `log`,
`pow` and `gettid`; against 2.28's stubs the math functions bind to their
older versions.

Its shared libraries are `libsteam_api.so` (found through the RUNPATH
`$ORIGIN`, beside it), `libasound.so.2` and `libdbus-1.so.3` (audio: cpal's
ALSA backend, and rtkit for a real-time audio thread), `libudev.so.1`
(gamepads, through gilrs), and glibc's own
(`libc`, `libm`, `libpthread`, `libdl`, `ld-linux-x86-64`), all in the
Steam Linux Runtime and on SteamOS. Linking passes
`--allow-shlib-undefined` because this machine's libasound, libdbus and libudev were
themselves built against glibc 2.44; the player's system has its own
builds. The window system (X11, Wayland, xkbcommon) and Vulkan are loaded
at run time by winit and wgpu, not linked.

Verified here: the final `.2.28` player package loads its x86-64-v2 guest and
renders headlessly on the host and installed soldier/sniper containers, with
identical PNGs. `just slr [PACKAGE]` (scripts/slr.sh) repeats that check: in
each container the runtime first starts with the container's own libraries,
then plays 40 frames with the host's graphics, and every run's last frame
must be the same image; pressure-vessel's copies of the runtimes go into its
output directory, not Steam's. Container-only library probes use glibc 2.31; GPU execution
imports the host's graphics stack and glibc 2.44. The symbol requirement remains
2.28, but an actual glibc 2.28 installation was not exercised. See
[local qualification](qualification.md). A real Steam Deck and Steam Input
were not verified. gdev has no gamepad support, so on the Deck Steam Input's "Keyboard
(WASD) and Mouse" template is what makes it playable until it has.

### Windows

`just windows` cross-compiles the player runtime on Linux with
cargo-zigbuild as `x86_64-pc-windows-gnu`, using zig's MinGW-w64 headers and
libraries: no Windows SDK, no MSVC and no license to accept. One fix makes
it build:

- The player runtime uses the Windows subsystem, so no console window opens
  with the game (the development build keeps its console).

The result imports only system DLLs (kernel32, user32, gdi32, dxgi,
opengl32, ole32 and friends), the Universal CRT (`api-ms-win-crt-*`,
part of Windows 10 and 11, an update on 7 and 8.1) and `steam_api64.dll`,
which `pen build` ships beside it. Steam's import library is MSVC's
`steam_api64.lib`, which zig's linker reads. `pen build --runtime
…/pen.exe` precompiles `game.cwasm` for Windows from Linux (Cranelift
cross-compiles within an architecture).

Not verified here: there is no Windows machine or wine, so the bar was "it
builds and packages". CI is the route to a real run ([ci.md](ci.md)): the
`windows` job builds the player with MSVC on `windows-latest` and runs the
renderer's GPU tests on WARP, and `windows-smoke` plays the shipped package
(this cross-built runtime, packaged from the game job's bundle) for 30
frames headless on WARP, uploads its last frame, and compares its packets
with Linux's bit for bit. A headless run or a command such as `packets` that stops with an error
writes its crash report without the message box, which nobody would
dismiss there. On first run on Windows, check the DX12 and Vulkan
adapters, audio (WASAPI through cpal), saves in `%APPDATA%\gdev`, a crash
report, and the Steam overlay. An MSVC build (`x86_64-pc-windows-msvc`)
works the usual way on a Windows machine or a `windows-latest` CI runner;
cross-compiling it from Linux needs cargo-xwin, which downloads Microsoft's
CRT and SDK under Microsoft's license, so it is not set up here.

### macOS

`pen build` makes an application bundle when the runtime is a Mach-O
executable (or `--target` names an Apple triple): `gdev.app/Contents/` with
`Info.plist` (`--bundle-id ID` names it, `pen.NAME` by default; it asks for
the display's full resolution and macOS 11), the runtime and Steam's
library in `MacOS/`, and the game in `Resources/game`, where the runtime
looks for it (`app_bundle.rs`). `scripts/macos-sign.sh APP` signs it with
the hardened runtime and `packaging/macos/entitlements.plist`, and
notarizes and staples it with the owner's credentials, or signs it ad hoc
without them. CI's `macos` job builds pen and the player on `macos-14`,
runs the renderer tests on Metal, and with the game job's bundle packages,
signs, plays and uploads the app ([ci.md](ci.md)). None of it has run yet:
there is no Mac here, and hosted runs were not started. The layout was
checked here by packaging for `x86_64-apple-darwin` from Linux and playing
from `Contents/MacOS`.

What it took, and what is still to try on a Mac:

- A Mac with Xcode or the Command Line Tools (or osxcross with an SDK
  extracted from Xcode, under Apple's license), and the targets
  `aarch64-apple-darwin` and `x86_64-apple-darwin`; `lipo` makes a universal
  binary. wgpu uses Metal; cpal uses CoreAudio; libopus builds with the
  SDK's clang; steamworks-sys ships `libsteam_api.dylib`, and `build.rs`
  already sets the rpath `@executable_path` for it.
- `game.cwasm` for Apple silicon: Cranelift here only has the host's x86-64
  backend, so `pen build --target aarch64-apple-darwin` fails ("Support for
  this target is disabled") unless the development build enables wasmtime's
  `all-arch` feature; packaging on the Mac itself avoids that. (Intel macOS
  precompiles from Linux today.)
- An app bundle: `Contents/MacOS/gdev`, `Info.plist`. Data files in
  `Contents/MacOS` break code signing, so the game is in
  `Contents/Resources/game`, where `packaged_game()` looks too. It has no
  icon yet (`CFBundleIconFile` and an `.icns` in `Resources`).
- Signing with a Developer ID, the hardened runtime and notarization.
  Loading `game.cwasm` maps machine code as executable at run time: wasmtime
  writes it to read-write memory and `mprotect`s it read-execute, without
  `MAP_JIT` (wasmtime issue 11989), so under the hardened runtime it needs
  the entitlement `com.apple.security.cs.allow-unsigned-executable-memory`;
  `allow-jit` alone covers only `MAP_JIT` memory. The entitlements file
  has it, and the two the Steam overlay's injected library needs
  (`disable-library-validation`, `allow-dyld-environment-variables`). Try
  them before relying on them.

## Crash reports

A panic on any thread writes a report to `crashes/` in the data directory
(`~/.local/share/gdev/crashes/` on Linux, `%APPDATA%\gdev\crashes\` on
Windows, `~/Library/Application Support/gdev/crashes/` on macOS, or under
`--data DIR`) and prints its path. `PEN_CRASH_TEST=N` panics at frame N, to try it on
a platform; this is the `dist` player runtime's, headless:

```
gdev crashed: PEN_CRASH_TEST: a crash at frame 25, as asked
at crates/pen/src/crash.rs:77:9 on thread main, 2026-10-11T01:06:04Z
pen 0.1.0 (optimized, x86_64-linux, features: steam)
system: linux x86_64, CachyOS, 7.2.9-2-cachyos
command: pkg-dist/gdev --headless --frames 60 --mute --data crash-data
GPU: NVIDIA GeForce RTX 2070 SUPER (Vulkan, DiscreteGpu; driver NVIDIA 615.78.08; vendor 0x10de, device 0x1e84)
frames: 25; the last 25 took (ms, oldest first): 6.3 2.1 1.4 1.9 1.8 1.3 1.5 1.8 2.0 1.5 4.4 2.1 …
executable loaded at 0x563bb94f4000
anchor: crash::frame at 0x563bb978d0e0

backtrace:
   0:     0x563bb9a90060 - <unknown>
   …
```

The GPU is the adapter `open_device` opened; the frame times are what each
`Session::step` took (an atomic ring of 30, so noting them costs nothing
measurable); the system line also says when Steam started the game. At most
three reports are written a run. Set `PEN_BUILD` when building (a commit,
say) to have it in the version line.

Shipped runtimes are stripped, so the backtrace has addresses, not names.
Changing `strip` in a profile compiles different code (the `.text` of a
stripped and an unstripped `release` build of the same source differ), so
the names must come from the very build that ships: the `dist` profile keeps
its symbols and line tables, and `just player` and `just windows` ship a
copy stripped with `objcopy --strip-all` from `dist/player/`, leaving the
unstripped build in `dist/` (64 MB on Linux, 45 MB on Windows on
2026-10-11; keep one per release). The load address (Linux) or the anchor (any system: an
address minus the anchor, plus `crash::frame`'s address from `nm`) turns the
report's addresses into offsets in that build:

```sh
addr2line -fiCpe target/x86_64-unknown-linux-gnu/dist/pen 0x<address − load address>
```

which for the report above gives `std::backtrace::Backtrace::force_capture`,
the panic machinery, then `pen::play` at `main.rs` with `crash::frame`
inlined into it. Shipping the symbol table instead would cost 1.8 MB (12.01
against 10.23 MB) and give names without lines.

### Opt-in reports to the developer

Crash and performance reports can reach the game's developer, and only
when both have said so (`reports.rs`):

- The developer gives an address: `pen.json`'s `"reports": {"endpoint":
  "https://..."}` (https, or http to this machine for a test server), which
  `pen build` ships as `game/reports.json`. Without it nothing is ever
  sent, and the reports service answers the game unavailable, so that it
  offers no choice with nothing behind it. A checkout's runs, `pen
  export`'s bundles and the browser have none.
- The player agrees, to crash reports, to performance reports or to both,
  through the game, which asks the reports service (`REPORTS_OP_CONSENT`,
  `SET_CONSENT`, `QUEUED`). Their choice is kept in
  `reports/consent.json` in the data directory; until they make one, it is
  neither.

With consent, each crash report is copied into the local queue,
`reports/queue/` in the data directory, with the player's home directory
taken out of its paths, and a packaged game that ends queues a performance
report: its frame times (median, 95th and 99th percentiles and the worst,
from a histogram every frame adds to), the GPU, the system and pen's
version; nothing that names the player. When a packaged game starts, a
thread of its own sends what is queued, of the kinds the player agreed to,
one POST each through the system's curl (Windows 10 and later, macOS,
SteamOS and desktop Linux have it, so pen links no HTTP client), with
`X-Pen-Report` and `X-Pen-Game` headers; a report that went is deleted, and
one that did not waits for the next start. The queue keeps the latest 20.
`PEN_REPORTS=off` sends nothing whatever was agreed, and the tests send
nothing (they post to a stand-in, and the curl transport refuses in them).
gdev asks on its settings' About page: a Reports row of none, crash
reports, or crash and performance reports, or "Not collected" where the
service is unavailable (gdev's `src/reports.blot`).

## Steam

`steam.rs` (the `steam` feature) starts Steam for a packaged game in a
window, or with `PEN_STEAM=1` or `PEN_STEAM_DEBUG`, and plays on without
it when there is no Steam client, saying why.

### Achievements and stats

`Steam::unlock`, `clear`, `achievements`, `stat`, `stat_float`,
`set_stat`, `set_stat_float`, `add_stat` and `reset` read and change the
player's achievements and stats by their API names on the app's Steamworks
page. Steam fetches the player's stats itself when it starts (Steamworks SDK
1.61 removed `RequestCurrentStats`); until its `UserStatsReceived` arrives
they return an error. Changes are stored with Steam by `Steam::frame`: an
unlocked achievement at once, so its notification shows, stats at most once
a minute (Steam rate-limits storing), and whatever is left when the game
closes. The game reaches them, and rich presence, through its services
(`services/steam.rs`; the README's "Services"), which answer `unavailable`
without Steam.

`PEN_STEAM_DEBUG` runs steps once the stats arrive, to test against
Spacewar (app 480, with `steam_appid.txt` containing `480` beside the
executable):

```sh
PEN_STEAM_DEBUG="list unlock=ACH_WIN_ONE_GAME add=NumGames=1 show=NumGames" pen --game dist/gdev
```

Steps: `list`, `unlock=NAME`, `clear=NAME`, `stat=NAME=N`, `statf=NAME=X`,
`add=NAME=N`, `show=NAME`, `reset`, `overlay` (whether the overlay has
hooked the window) and `cloud` (whether Steam Cloud is on for the account
and the app, its files, and a file written, read back and deleted through
it); a last line counts the steps that failed. Without a Steam client it
prints `Steam is off: …` and that there is nothing to do; that is all that
could be tried here, since Steam does not run on this machine.

`just steam-smoke APP_ID [PACKAGE] [--achievement NAME] [--stat NAME]`
(scripts/steam-smoke.sh) is the live check, for the game's own app: it
refuses 480, copies the package (default `dist/linux`) with
`steam_appid.txt` beside its runtime, needs the Steam client already
running (it never starts it), plays 15 s in a window, since the overlay
hooks a window, and passes when Steam started the game as that app, every
step (achievements, stats, overlay, cloud) passed, and the game's own Steam
service requests were answered rather than `unavailable`. `--dry-run`
prints what it would run. It has not run: it needs the owner's app ID.

### Workshop

`workshop.rs` lets players publish the levels they build in the game's
editor and play the ones they subscribe to, through the Steam service's
workshop operations (`protocol/schema.json`, SteamOperation):

| Operation | What it does |
|---|---|
| `workshop_publish` | Uploads the player's saved scene (`main.scene` in the data directory, what the editor saves) as a new item titled with the text, visible to `a` (0 everyone, 1 friends, 2 only the player, 3 those with the link); with `b` 1, as a new version of the item whose id is the text's first line. Answered once Steam has it, with the item's id as text (ids do not fit a float lane) and value 1 when the player must accept the Workshop's legal agreement |
| `workshop_items` | The installed subscribed items' ids, a space between them, their count the value; the others start downloading |
| `workshop_subscribe`, `workshop_unsubscribe` | An item's id as text; answered once Steam has |
| `workshop_use` | Makes an installed item's scene the player's, for the game to load as the editor's scene (Ctrl+R in gdev), which a recording keeps like any load |

Before uploading, pen refuses what Steam would: an empty or long title (128
characters) or description (8000), an empty content folder, a preview that
is no PNG, JPEG or GIF or is past 1 MB. Uploads are staged in
`workshop-uploads/` in the data directory until Steam has them (keep it out
of Auto-Cloud). The logic is generic over a `Ugc` trait and tested without
Steam (`cargo test -p pen workshop`); `Steamworks` drives `steamworks::UGC`
and compiles with the `steam` feature. It has never run against a live
Steam client: that needs the game's own app ID with the Workshop enabled on
its Steamworks page (not Spacewar's), and gdev does not ask for these
operations yet.

### Steam Auto-Cloud

Saves and preferences live in the platform data directory, `gdev/` under it
(`dirs::data_dir()`):

| File | What |
|---|---|
| `quicksave.bin` | the quick save |
| `main.scene` | the player's edited scene |
| `preferences.json` | settings |
| `crashes/` | crash reports: do not sync |
| `backups/` | the last five versions of each save: do not sync |
| `reports/` | the player's consent to reports, and their queue: do not sync |

The quick save and the player's scene are written stamped (save format
version 2: when, and the identities of the saves each replaced), and each
write keeps its bytes as `backups/NAME.1.EXT`, the older ones moving down to
`.5` (`saves.rs`). When the game starts, each is compared with this device's
last write: a save another device wrote that went on from it (the ordinary
case with Steam Cloud) is taken in quietly; an older one of this device's
own, or one that forked from an older save on another device (both played
since they last synced, and Steam's conflict dialog kept the other), plays,
since the player or Steam chose it, but is said in the log, and this
device's own last save is kept as `backups/NAME.2.EXT`. Either way the save
that came is kept among the backups before anything can overwrite it, so a
conflict resolved the wrong way loses nothing. Version 1 saves still load.

Auto-Cloud settings on the app's Steamworks Cloud page (no code needed),
one root path per file pattern:

| Root | Subdirectory | Pattern | OS |
|---|---|---|---|
| `WinAppDataRoaming` (`%APPDATA%`) | `gdev` | `quicksave.bin` | All OSes |
| `WinAppDataRoaming` | `gdev` | `main.scene` | All OSes |
| `WinAppDataRoaming` | `gdev` | `preferences.json` | All OSes |

with root overrides so that the same files sync between platforms:

| Original root | OS | New root | Path |
|---|---|---|---|
| `WinAppDataRoaming` | Linux + SteamOS | `LinuxXdgDataHome` (`~/.local/share`) | `gdev` |
| `WinAppDataRoaming` | macOS | `MacAppSupport` (`~/Library/Application Support`) | `gdev` |

Overrides only apply when the root's OS is "All OSes". Patterns name the
files, so `crashes/` stays local. Set a byte quota and file count to match
(three files, a few MB). `dirs::data_dir()` honours `$XDG_DATA_HOME`, as
Steam's `LinuxXdgDataHome` does.

The directory is named `gdev` for now; a store build should use the game's
own name, so that it cannot collide with another game's `gdev`.

## CI

[ci.md](ci.md) has every job, what it proves and what the owner sets. In
short: the workflow builds native and browser hosts, runs strict Clippy and tests on
Mesa lavapipe, and cross-builds Linux (glibc 2.28) and Windows players. The game
job checks out sibling repositories at `GDEV_REF` and `BLOT2_REF`, builds blotc
with Zig, checks out Ink at `INK_REF`, compiles the plugin, then runs the full Blot suite, deterministic
replays, required golden images and packaging. `CHECKOUTS_TOKEN` grants access
to private siblings.

Cross-compilation does not establish runtime compatibility. Only this Linux
machine was available for completion: Windows, Steam Deck, live Steam, audible
output and manual window checks still require those environments. The player
runtime can be tested here headlessly without starting Steam.
