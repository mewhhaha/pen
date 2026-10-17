# Android

Where pen stands on Android (aarch64), checked on 2026-10-16 with
`cargo clippy --target aarch64-linux-android -- -D warnings`; CI's `android`
job runs the same with the runner image's NDK.

## What builds

| Crate | Status |
|---|---|
| pen-protocol | Clean |
| pen-runtime | Clean |
| pen-render | Clean: wgpu on Vulkan (Android 7 and later) or GLES |
| blot-host, without its compiler (the player's) | Clean |
| blot-host, with its compiler | Clean: Cranelift has an aarch64 backend, so a development build could compile on the device |
| pen | Not ported (below) |

Nothing needed fixing in the portable crates. wasmtime's build script
compiles a small C file of helpers (`stdbool.h`, `stdint.h`, `stdlib.h`
only), so even `cargo check` needs a C compiler for the target. CI uses the
NDK the runner image ships:

```sh
rustup target add aarch64-linux-android
ndk=$ANDROID_NDK_LATEST_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin
export CC_aarch64_linux_android=$ndk/aarch64-linux-android24-clang AR_aarch64_linux_android=$ndk/llvm-ar
cargo clippy --target aarch64-linux-android -p pen-protocol -p pen-runtime -p pen-render -p blot-host -- -D warnings
cargo clippy --target aarch64-linux-android -p blot-host --no-default-features -- -D warnings
```

This machine has no NDK (installing one means accepting Google's NDK
terms), so here zig stands in for its clang, with musl's headers, which is
enough for a check that compiles the helpers but never links them:

```sh
#!/bin/sh
# cc: drop cc-rs's --target, which zig would take as Android's (it has no bionic headers)
for argument in "$@"; do
  shift
  case $argument in --target=*) ;; *) set -- "$@" "$argument" ;; esac
done
exec zig cc -target aarch64-linux-musl "$@"
```

with `CC_aarch64_linux_android` naming that script and
`AR_aarch64_linux_android` a script running `zig ar "$@"`.

## What pen itself lacks

`cargo check --keep-going --target aarch64-linux-android -p pen
--no-default-features` (the player) stops at:

- **opusic-sys** builds libopus with CMake and needs the NDK's CMake
  toolchain file (`ANDROID_NDK_HOME`). With the NDK it should build as on
  the other platforms.
- **winit** needs one of its Android activity features
  (`android-native-activity` or `android-game-activity`) and the game an
  `android_main` entry point in a `cdylib`, packaged as an APK (cargo-apk or
  xbuild) with a manifest. The event loop starts from the activity, not
  `main`.
- **arboard** has no Android backend: the clipboard service would answer
  `unavailable` there (a `cfg(not(target_os = "android"))` around it).

Past those, what an Android player needs that the desktop does not have:

- **Files.** `dirs` has no data directory on Android; saves belong in the
  activity's internal storage path. The package's `game/` lives in the APK's
  assets, read through `AAssetManager`, or is unpacked on first run, since
  `game.cwasm` must be mapped from a file.
- **The precompiled game.** `pen build --target aarch64-linux-android`
  needs Cranelift's aarch64 backend in the packaging pen (wasmtime's
  `all-arch` feature) or packaging on an aarch64 machine, as for Apple
  silicon (shipping.md, "macOS").
- **Input.** Touch and the back button, and the gamepads of hill 9; the
  Steam features are desktop only.
- **Audio.** cpal plays through AAudio on Android 8.1 and later.
- **Lifecycle.** The surface goes away when the app is paused; wgpu's
  surface must be dropped and made again on resume.
