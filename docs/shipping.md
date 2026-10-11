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

The executable is named after the game (`GAME` in main.rs, which also names
the data directory), whatever the runtime it copies is called (`pen`,
`pen.exe`); the runtime finds `game/` beside itself under any name.

The runtime it ships is the **player runtime**: pen built without the
`compiler` feature (`--no-default-features`), so without Cranelift. It only
loads the `game.cwasm` its package was precompiled with, and is less than
half the size of the development build (10.2 against 23.9 MB). Everything that compiles a checkout
(`dev`, `run`, `build`, `test`, `parity`, `bench` and `packets` on a
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
ALSA backend, and rtkit for a real-time audio thread), and glibc's own
(`libc`, `libm`, `libpthread`, `libdl`, `ld-linux-x86-64`), all in the
Steam Linux Runtime and on SteamOS. Linking passes
`--allow-shlib-undefined` because this machine's libasound and libdbus were
themselves built against glibc 2.44; the player's system has its own
builds. The window system (X11, Wayland, xkbcommon) and Vulkan are loaded
at run time by winit and wgpu, not linked.

Verified here: the `.2.31` and `.2.28` player runtimes, packaged with
`pen build`, load their x86-64-v2 `game.cwasm` in 6 ms and play (headless,
the same picture as the development build). Not verified: a real Steam
Deck, the Steam Linux Runtime containers, Steam Input. gdev has no gamepad support, so on the Deck Steam Input's "Keyboard
(WASD) and Mouse" template is what makes it playable until it has.

### Windows

`just windows` cross-compiles the player runtime on Linux with
cargo-zigbuild as `x86_64-pc-windows-gnu`, using zig's MinGW-w64 headers and
libraries: no Windows SDK, no MSVC and no license to accept. Two fixes make
it build:

- opus-head-sys passes MSVC's `/guard:cf` (Control Flow Guard) to CMake for
  every Windows target, `-gnu` included, and zig's clang driver takes it for
  a file name. `packaging/zig/windows-gnu.cmake`, which the recipe hands to
  CMake through `CMAKE_TOOLCHAIN_FILE_x86_64_pc_windows_gnu` (cargo-zigbuild
  then leaves its own aside), compiles C through a wrapper that drops it and
  the cmake crate's Rust-style `--target=x86_64-pc-windows-gnu`.
- The player runtime uses the Windows subsystem, so no console window opens
  with the game (the development build keeps its console).

The result imports only system DLLs (kernel32, user32, gdi32, dxgi,
opengl32, ole32 and friends), the Universal CRT (`api-ms-win-crt-*`,
part of Windows 10 and 11, an update on 7 and 8.1) and `steam_api64.dll`,
which `pen build` ships beside it. Steam's import library is MSVC's
`steam_api64.lib`, which zig's linker reads. `pen build --runtime
…/pen.exe` precompiles `game.cwasm` for Windows from Linux (Cranelift
cross-compiles within an architecture).

Not verified: there is no Windows machine or wine here, so the bar was "it
builds and packages". On first run on Windows, check the DX12 and Vulkan
adapters, audio (WASAPI through cpal), saves in `%APPDATA%\gdev`, a crash
report, and the Steam overlay. An MSVC build (`x86_64-pc-windows-msvc`)
works the usual way on a Windows machine or a `windows-latest` CI runner;
cross-compiling it from Linux needs cargo-xwin, which downloads Microsoft's
CRT and SDK under Microsoft's license, so it is not set up here.

### macOS

Not built: there is no macOS SDK here. What it takes:

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
- An app bundle: `Contents/MacOS/gdev`, `Info.plist`, an icon. Data files in
  `Contents/MacOS` break code signing, so the game belongs in
  `Contents/Resources/game`, and the runtime's `packaged_game()` (main.rs)
  must also look there.
- Signing with a Developer ID, the hardened runtime and notarization.
  Loading `game.cwasm` maps machine code as executable at run time: wasmtime
  writes it to read-write memory and `mprotect`s it read-execute, without
  `MAP_JIT` (wasmtime issue 11989), so under the hardened runtime it needs
  the entitlement `com.apple.security.cs.allow-unsigned-executable-memory`;
  `allow-jit` alone covers only `MAP_JIT` memory. Try it before relying on
  it.

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
`add=NAME=N`, `show=NAME`, `reset`. Without a Steam client it prints
`Steam is off: …` and that there is nothing to do; that is all that could be
tried here, since Steam does not run on this machine.

### Steam Auto-Cloud

Saves and preferences live in the platform data directory, `gdev/` under it
(`dirs::data_dir()`):

| File | What |
|---|---|
| `quicksave.bin` | the quick save |
| `main.scene` | the player's edited scene |
| `preferences.json` | settings |
| `crashes/` | crash reports: do not sync |

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

`.github/workflows/ci.yml` runs what needs only this repository: the build,
clippy (`-D warnings`, with and without the compiler) and the tests on
Linux, with Mesa's lavapipe as the GPU, and the two player runtimes
(`just player`, `just windows`) as artifacts. The game's checkouts are
private siblings (`../gdev`, `../blot2`); its job (exporting `dist/gdev`,
the tests that read it, engine conformance, a package, and golden images on
lavapipe, which only report) runs when the repository variables
`GDEV_REPOSITORY` and `BLOT2_REPOSITORY` name them and the secret
`CHECKOUTS_TOKEN` can read both. There is no git remote yet, so the
workflow has not run.

`--software` (or `PEN_SOFTWARE=1`) asks wgpu for a fallback adapter:
lavapipe on Linux, WARP on Windows. Lavapipe is not installed on the
development machine, so that mode is untested here: without it the run
stops with `no software GPU adapter for PEN_SOFTWARE (on Linux: Mesa's
lavapipe, in mesa-vulkan-drivers)`.
