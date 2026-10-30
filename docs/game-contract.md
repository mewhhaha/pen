# What a game needs

The owned-storage integration is unfinished. Follow
[CONTINUE.md](../CONTINUE.md) for current execution status; existing examples
still need final qualification.

Everything pen needs of a game, from the files of its checkout down to the
numbers in its packets, and what pen still assumes of gdev, the game it was
built for. `pen new` makes the smallest checkout that has all of it.

## The checkout

pen takes a directory as a checkout when it has `pen.json` and `src/main.blot`
(`--project DIR` names one; otherwise the current directory or one above it).

| File                          | Needed | What pen does with it                                                                                                                                                                                                                                                                                                     |
| ----------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pen.json`                    | yes    | `game`: the name (data directory, executable) and title (window, assistive technology), else the directory's name. `plugins` and `compiler`: the compiler plugin that builds the game and what it may read and run. `protocol`: the game's own events and sections ([the game's own events and sections](extensions.md)). |
| `src/main.blot`               | yes    | The program's entry, compiled by the chosen plugin into `.pen/runtime/game.wasm`.                                                                                                                                                                                                                                         |
| `src/assets/catalog.json`     | yes    | The meshes, materials and sounds the game names by index. Each list may be empty. `pen build` ships what the catalog reaches.                                                                                                                                                                                             |
| `src/main.scene`              | no     | A saved scene the game starts from, if the program keeps scenes.                                                                                                                                                                                                                                                          |
| `src/tunables.json`           | no     | Live-tunable constants, for `tune=` steps and the dev tools.                                                                                                                                                                                                                                                              |
| `devtools/main.blot`          | no     | A second program pen runs over the game as its dev tools. Without one there are none.                                                                                                                                                                                                                                     |
| `pipeline.json`               | no     | Asset recipes `pen dev` reruns as their sources change, run through `just` in the checkout.                                                                                                                                                                                                                               |
| `tests/*.blot`                | no     | `pen test-blot`'s modules, as the Blot plugin lays them out.                                                                                                                                                                                                                                                              |
| `tests/visual/scenarios.json` | no     | `pen test`'s golden-image scenarios.                                                                                                                                                                                                                                                                                      |

pen keeps what it makes under `.pen/` (builds, packages, shots, test output). It
writes the checkout's own files only when asked to: `pen
codegen` into `pen/`,
the dev tools' Bake into `src/tunables.json`, `pen test
--update` into the
golden images, and asset recipes where they write.

## The program

A game is a WebAssembly module that speaks Blot's guest ABI (blot2's
`compiler/guest-abi.md`): a `blot:abi` custom section lists its exports, and it
exports `main`, which takes the host, a callback from an `Array F32` to an
`Array F32`, and returns nothing. Blot compiles to this; another language can
write the section and the adapters itself.

`main` runs for as long as the game does. Each call to the host hands it a
response and gets the next request back:

- The first call carries the program's hello: `4670544, 2, schema, scenes`, the
  magic number, the protocol's version, the state schema (1 to 16777215) and 1
  if the program keeps scenes apart from its state.
- A request is `[published, operation, payload...]`. `published` is 1 when the
  host published the preceding output and 0 when it rejected it. Session version
  2 uses this acknowledgement to repair presentation caches; it does not roll
  simulation back. Snapshot/Restore is explicit. A retry must not silently
  advance simulation twice.
- A response is `[operation, lanes...]`.

| Operation     | Payload          | Response lanes                               |
| ------------- | ---------------- | -------------------------------------------- |
| 1, create     |                  | the new world's state                        |
| 2, frame      | the input packet | the render packet                            |
| 3, snapshot   |                  | the world's state                            |
| 4, restore    | a state          | anything (a report, if it says what it kept) |
| 5, scene      |                  | the world's scene                            |
| 6, load scene | a scene          | as restore                                   |
| 0, stop       |                  | (returns from `main`)                        |

A state is at most 16777215 finite numbers; a program answers `[-1]` for one it
could not make. pen keeps states for quicksaves, time travel, replays and hot
reload: a new build restores the old one's state, so a world must come back
whole from what it saved. The state schema says when it cannot: pen restores no
state saved under another schema.

The state payload and restore response are opaque by default. Optional
[state adapters](state-adapters.md) provide format-specific tooling; an Ink
application opts in through `src/assets/saved.json`. Pen does not infer Ink from
a save's contents.

The owned-storage loop in `ink/lib/pen/program.blot` takes explicit execution
and acknowledgement callbacks, access aliases and caller-owned incoming/outgoing
buffers. It encodes before invocation-local owners close and copies numeric
packets at the host boundary. Consult the current generated bindings and source
rather than restoring old `serve`/snapshot-world adapters.

## The packets

pen's protocol is `protocol/schema.json`; `pen codegen` writes it into a
checkout as Blot (`pen/protocol.blot`, `pen/types.blot`,
`pen/protocol_render.blot`) and into pen as Rust.

- **Input**, each frame: the seconds since the last frame, the viewport's width
  and height, the events (keys by code, the pointer, the wheel, focus lost,
  console commands, stored preferences, tunables, interface actions, service
  answers, the game's own events) and no state.
- **Render packet**, each frame: the camera, the light, the sky, fog and shadow
  settings, quality, then sections: the draws, solids, decals, sprites, insets,
  sounds, voices and buses, field patches, the interface's nodes for assistive
  technology, service requests, preferences to store, then the game's own
  sections. Header flags ask pen to save, load, record a checkpoint for undo, or
  stop.

`pen/frame.blot`'s `blank` is a frame that draws, plays and asks nothing, lit
from above; a game sets what it uses. Keys arrive as codes: `pen/protocol.blot`
names the named ones (`key_arrow_left`, `key_f1`...), and a key that types a
character carries its code point (a letter, its capital's).

## Assets

A catalog lists paths under `src/assets`. A mesh is JSON (`positions`,
`normals`, `uvs`, `indices`, optional `attributes`) or the binary form
`pen build` packs it into. A material is JSON naming its shader (a WGSL asset),
its colour, `lit` or `unlit`, and optionally textures, blending, overlay, decals
and shadow casting. A sound is a WAV or Ogg Opus clip. pen has no built-in
shaders: each game supplies its own ([material shaders](materials.md)).

## What pen still assumes of gdev

pen grew up running gdev. These are the places it still leans on gdev or on Ink,
gdev's library, and what that means for another game.

Taken away by this work:

- **Preferences were exactly 14 numbers**, gdev's settings. A game now stores as
  many as it keeps, up to 256.
- **Named keys were the ones gdev used** (F2, F6, F9, F10 but no other function
  keys). F1 to F12 and Insert now travel, and the Blot bindings name every key's
  code, which Ink kept a copy of by hand.
- **The program loop lived in Ink** (`ink/lib/pen/program.blot`), with pen's
  operation numbers written out again. pen now generates it
  (`pen/program.blot`), with a blank frame to start from (`pen/frame.blot`) and
  the checks `pen test-blot` reads (`pen/check.blot`, which gdev keeps its own
  copy of). A game needs neither gdev nor Ink.
- **A packet a game needed meant a change to pen's protocol.** A game now
  declares its own events and sections in its `pen.json`
  ([the game's own events and sections](extensions.md)).
- **Packaging needed `src/main.scene`.** `pen build` and `pen export` now ship
  one only when the checkout has it.
- **A catalog needed a mesh and a material.** Either list may be empty now; a
  frame that names an entry of an empty list is refused.

- **Save tooling assumed Ink's layout.** Numeric inspection, tunable readback,
  restore reports and declared migration fallback now live behind an explicit
  state adapter. `pen-ink` supplies the Ink implementation; ordinary snapshots
  and semantic observations need no adapter.

Still there, for another lane or another day:

- **The dev tools are the checkout's own program.** pen draws none of its own;
  gdev's are a Blot program built on Ink and gdev's font, so a new game has no
  console, diagnostics or timeline panel until it writes or borrows one. Time
  travel's keys, scripts and `pen packets` work without them.
- **Numbers that spell gdev.** The render packet's magic is 0x474446 ("GDF"),
  the program's 0x474450 ("GDP"), and a save file starts `GDEVSAVE`. They are
  only numbers, and changing them would break every save and build.
- **pen's repository tests gdev.** `just` recipes default to `../gdev`;
  `tests/visual/` holds gdev's golden images and `tests/*.steps` its scripts;
  CI's game job checks gdev out; `pen packets` without a script presses Tab
  after a second, gdev's editor key; pen-render's measurements read `dist/gdev`.
  `examples/bounce` and a game `pen new` makes now run beside it
  (`just examples`).
- **Script coordinates are pixels of a page 960 wide**, the size gdev's window
  opens at, scaled to the viewport.
- **Asset recipes run through `just`.** `pipeline.json` names recipes in the
  checkout's Justfile; a game without one has no recipes.
- **The guest ABI is Blot's.** A module must carry Blot's `blot:abi` section;
  another language writes it as Blot's compiler does.
