# pen's plugin API

[pen-plugin.wit](pen-plugin.wit) is the WIT package `pen:plugin`: what pen
hosts and its Wasm plugins build against. A component names the version it
was built for in every interface it imports from pen or exports to it
(`pen:plugin/log@0.2.0`), and pen reads those names before linking one.

## Compatibility

- A plugin runs on a pen whose version is semver-compatible with the one it
  was built for and no older: below 1.0 the same minor version
  (`0.2.0` runs on `0.2.3`), from 1.0 the same major one. wasmtime links
  semver-compatible names the same way.
- Anything else is refused before linking, with the versions named:
  `plugin blot was built for pen:plugin@0.1.0, and this pen hosts 0.2.0:
  rebuild it against wit/pen-plugin.wit`. One built for a newer version is
  told to update pen.
- Below 1.0, any change to an interface's shape (a function, type, field or
  case added, removed or changed) takes a new minor version; a patch version
  changes words only. From 1.0, removing or changing anything takes a new
  major version, and adding a function pen imports a new minor one.
- Each entry below records its interfaces' shape: the SHA-256 of the WIT
  without comments or blank lines, each line trimmed (`pen_plugins::api::shape`).
  A test holds `pen-plugin.wit` to the newest entry, so a change to its
  interfaces that does not add an entry fails `cargo test -p pen-plugins`,
  saying the shape to record.

## 0.2.0 (2026-10-16, 52a55af)

Shape: `1e7ba4e0252d7a27e5cc83720c87e6d6cedb2363f6fb2fd21a8776fa9b0e7625`

- Removed the `assets` interface and the `assets-plugin` world: pen makes a
  game's asset modules itself (Blot's native typed imports).
- Added the `tests` interface to the `compiler-plugin` world: `find` lays out
  a checkout's test modules (`test-module`: its path, the entry it builds as,
  its overlays, its tests' prefix and its `expectation`: `checks`, `traps` or
  `rejected`), which `pen test-blot` builds and runs.
- Each call into a plugin has a deadline (pen.json's `timeout`): a wait
  without a timeout of its own (`child.read`, `child.wait`) ends there with
  `read-error::timeout`.

## 0.1.0, changed in place (2026-10-11, 84506b6)

Shape: `b8d8d842c84b32938ee886b22b4796fbf5bb21e0c5ec0c396d1b326fc7f60e35`

These changes kept the version 0.1.0, so a plugin built before them named
the same version and failed to link. The shape check above exists so that
does not happen again.

- `diagnostic` gained `details: option<string>`.
- `files` gained `is-file` and `canonical`; paths are `/`-separated on every
  platform.
- `compiler.session` gained `close`.
- `assets.generate` failed with `generation-error` (its message and the
  inputs read) instead of a string.

## 0.1.0 (2026-10-11, 194dcc3)

Shape: `37d28733fdc462070ed49dbbb07d3b13ed15bf601b78bbf6f33354d120c45384`

The first plugin API: the `types`, `log`, `files` and `process` interfaces
pen gives a plugin, and two worlds, `compiler-plugin` (exports `compiler`:
sessions that build a program again as its files change) and
`assets-plugin` (exports `assets`: the virtual modules a program imports its
assets through).
