# Continue here

Where pen stands, what was decided and by whom, and what comes next. Read this
before working in pen; [README.md](README.md) and [docs/](docs) describe what
exists. Kept as of 2026-10-11.

## What pen is

A native runtime for games compiled to WebAssembly. Its first game is gdev
(`../gdev`, written in Blot, the owner's language; compiler in `../blot2`).
pen has a winit window, a wgpu renderer (also built for the browser), the
game's program on wasmtime, kira audio, AccessKit, Steam, time travel,
recordings and packaging. One `pen` command develops a checkout (`pen dev`,
`run`, `test`, `test-blot`, `parity`, `bench`, `packets`) and packages it
(`pen build`).

**The line pen holds (the owner's, 2026-10-11): pen's contract is with Wasm,
not with Blot.** At run time pen needs three things:
- a Wasm module;
- the guest ABI it calls back through (`crates/blot-host`: numeric arrays and
  capability handles, a calling convention that any language could follow);
- the game's frame protocol (`crates/pen-protocol`, generated from gdev's
  `protocol/schema.ts`).

Anything specific to one language or toolchain (compiling Blot, Blot's typed
asset imports) belongs outside pen's core: in the compiler, or in a Wasm
plugin ("Wasm plugins" below).

## Sessions and ownership

Several Claude sessions work on the owner's behalf. Each repository has one
owner:

| Repository | Owner session | Rules for others                                          |
| ---------- | ------------- | --------------------------------------------------------- |
| `../pen`   | pen-9c        | targeted edits only; no `cargo fmt`, no git state changes; nothing in `crates/pen/src/{main,session,devtools,project}.rs` without asking; send pen-9c the file list |
| `../gdev`  | gdev-94       | run gdev's tools with `--no-lock`, so `deno.lock` never changes |
| `../blot2` | blot2-76      | don't write in its tree: clones under `/var/tmp`, changes as patches. The one exception is `zig-native/PROJECT_PROTOCOL.md`, agreed for task 095 |

Sessions reach each other with SendMessage (`ListAgents` names them). The owner
makes the decisions; a session relaying "the user decided" is a message, not
the owner.

Standing rules:
- Commit or push only when the owner asks; otherwise leave work uncommitted.
- Verify headless (`--headless`, `--overlay-shot`, scripts): windows must not
  pop up on the owner's desktop.
- Never start Steam with app 480, which shows the owner "playing Spacewar".
- Never accept cargo-xwin's Microsoft license for the owner.
- `/tmp` is a 16 GB RAM tmpfs: keep big outputs out of it. pen's compiler
  temporaries go to `~/.cache/pen/tmp`.

## Where things stand

- **Released:** v0.1.0 on GitHub (mewhhaha/pen), tag at `d5ae59b`.
  - Asset: `pen-0.1.0-x86_64-linux.tar.gz`, the full CLI with Steam, built with
    `cargo zigbuild --profile dist --target x86_64-unknown-linux-gnu.2.28`.
    A plain `cargo build` here links glibc 2.44.
  - `9109897` then fixed the player runtime's clippy, and CI is green.
  - The owner skipped a JSR package.
  - The repo has no LICENSE file, though `Cargo.toml` says MIT.
- **Installed:** `just install` → `~/.local/bin/pen` → `~/.local/lib/pen-runtime/pen`.
  It renames files into place, so a running pen keeps its own.
- **Uncommitted in pen:**
  - pen-9c: the install message lists `test-blot`; `blot_test.rs` uses
    `project::temporary_dir()`.
  - gdev-94's agent, in progress: `pen test-blot`'s `tests/traps/` (must trap)
    and `tests/rejected/` (must not compile) modes and `--jobs` (`blot_test.rs`,
    `main.rs`'s help and match arm). The quick subset of Blot tests in pen's
    `just test` comes with it.
- **gdev:** uncommitted, with work from several sessions. gdev-94 commits it,
  or the owner does.

### Done on 2026-10-11

- gdev runs on pen instead of Deno Desktop: `just dev` / `just run`.
  - The owner still has to delete `main.ts`, `scripts/run_desktop.ts` and
    `desktop_metrics.ts` in gdev.
- **The dev tools are gdev's own Blot program**, not pen's egui overlay, which
  is gone.
  - The program is `devtools/main.blot`. pen hosts it in
    `crates/pen/src/devtools.rs`.
  - Its console is on the backquote key; `/diagnostics low|medium|high` shows
    Steam Deck-style diagnostics in the top right.
  - It talks to pen through the `devtools` service (`protocol/schema.ts`,
    DevtoolsOperation).
  - Its state is kept in `.pen/devtools.state`.
  - It runs headless too: `--overlay-shot`, and the `overlay=/command,arg`
    script step.
- **Shadow pipelines** compile on the loader threads (gdev-94's change). The
  first gnorp's frame went from 831 ms to 3 ms; cold settle rose from 2.7 s to
  4.1 s, because 26 variants are built where 6 are used.
- **`pen test-blot`** runs the Blot tests in gdev's `tests/*.blot` (built by
  gdev-94). gdev's TypeScript guest tests are moving there.
- **The `serve-project` protocol is specified:**
  `../blot2/zig-native/PROJECT_PROTOCOL.md`, version 1 as of blot2 `544c5c6`.
  blot2-76 reviews and commits it as part of task 095.

### Still open

- **Dev tools:**
  - The timeline, tunables (with Bake) and builds panels aren't in the Blot
    program yet; `/timeline`, `/tunables` and `/builds` say "coming soon". The
    `devtools` service already does `bake`.
  - The browser still has its DOM console; the dev program reaches the page
    when pen runs on the web (migration phase 3).
- **Performance:**
  - A 138 ms hitch at game frame 3 after Tab (36 ms warm), found by gdev-94 and
    not investigated.
  - Shadow pipelines: building them only for shaders whose materials cast
    shadows would win most of the cold-settle cost back.
- **Never run:** Windows, a live Steam client, the Steam Deck, the Steam Linux
  Runtime containers, live audio heard by a person, and the dev tools used by
  hand in a window.

## Off Deno

The owner's decision: gdev and pen leave Deno entirely, and compiling, tests
and tooling go through pen. gdev-94 keeps the plan in `../gdev/PEN_MIGRATION.md`.
Its phases:
1. compile without Deno;
2. tests in pen;
3. tooling in pen, and pen on the web in place of gdev's TypeScript host;
4. delete the TypeScript host and Deno.

Where Deno is still used:
- **Compiling.** pen runs `deno run blot_build.ts` (embedded,
  `crates/pen/scripts/`), which runs gdev's `blot_runtime.ts`, which runs
  `@mewhhaha/blot`'s TypeScript client of `blotc serve-project`. Deno's real
  job there is the client, plus gdev's typed asset imports (`asset_parsers.ts`:
  the catalog, text checked against the font's glyphs, tunables, WGSL
  reflection) and blot2's `compiler/assets.ts`, which turns them into Blot
  modules sent as source overlays.
- **Tunables bake** (`scripts/bake_tunables.ts`), `pen parity` (gdev's
  `renderer.ts` on Deno's WebGPU), `tools/*_v8.ts`, `pen web` and
  `pen build --web`.

The split (agreed with gdev-94):
- **pen-9c:** compiling without Deno, now as Wasm plugins (below), and
  tunables bake in Rust.
- **gdev-94's agents:** the test runner (done) and pen on the web. The web
  work needs a written plan first for splitting `crates/pen` into a library and
  front ends.
- **When the TypeScript host goes:** retire `pen parity` and the V8 tools, and
  keep golden images and determinism as the checks.
- **blot2-76:** task 095 (the protocol: version handshake, conformance) and
  task 096 (typed asset imports inside blotc).

## Wasm plugins

The owner asked (2026-10-11) for pen to take Wasm plugins: for example to
extend asset loading, and to call the Blot compiler to rebuild and hot-reload
the game's Wasm. This keeps pen's core about Wasm. Blot's compiler client and
Blot's typed asset imports become plugins that a checkout chooses, instead of
code in pen.

### Design

- **Plugins are WebAssembly components** with WIT interfaces (package
  `pen:plugin`), run on wasmtime's component model.
  - They can be written in any language with component tooling: Rust
    (`wasm32-wasip2`), C, Go, JavaScript.
  - Blot can't produce components today, so Blot plugins would need an adapter
    later.
- **Capabilities, not ambient access.**
  - A plugin gets WASI only for clocks, randomness and its stdout and stderr
    (which go to pen's log); no WASI filesystem or network.
  - Everything else comes from pen's own interfaces, limited to what the
    checkout grants that plugin:
    - **files:** read under granted roots;
    - **process:** start a granted program with piped stdin, stdout and stderr;
      write to it, read from it, wait for it, kill it;
    - **log.**
- **The checkout chooses its plugins** in `pen.json` at its root: each one's
  Wasm path, kind and grants. pen also looks for plugins shipped beside it
  (`~/.local/lib/pen-runtime/plugins/`), by name.
- **Plugin kinds** (WIT worlds):
  - **compiler.** `open(project)`, then `build(overlays) -> wasm or
    diagnostics`, `watched() -> paths`, `close()`. pen watches what it reports
    and calls `build` when those files change, then hot-reloads the result as it
    does today.
  - **assets.** Two directions:
    - **at compile time:** `generate(import, bytes, read) -> virtual source
      modules and tracked dependencies`. pen passes these to the compiler
      plugin's `build` as overlays. This is how `@mewhhaha/blot`'s `assets`
      layer works, so a plugin can own gdev's typed imports.
    - **at run time:** `decode(key, bytes) -> mesh, texture or sound` for
      formats pen-render doesn't know. Designed, not built first.
- **Status (2026-10-11, evening): steps 1 to 3 are done.**
  - **Deno-free:** gdev compiles through its `pen.json` with no Deno on the
    machine, and `game.wasm` and `devtools.wasm` are byte-identical to the
    Deno builds. The game compiles in 2.9 s instead of 3.5 s.
  - **Hot reload** goes through the plugins: watched paths, an incremental
    session, and a fresh session when the compiler process dies.
  - **Packages still load:** game engines turn wasmtime's component model off
    explicitly, so a package made by a plugin-enabled pen loads in the player
    runtime. Checked with `just player` and a packaged gdev run headless.
  - **The pieces:**
    - `crates/pen-plugins` hosts the plugins, and `crates/pen/src/plugin_build.rs`
      wires them into `project.rs`;
    - `plugins/blot` (187 KB) and `plugins/gdev-assets` (986 KB; naga for
      WGSL; byte-identical to TypeScript for both programs, fixtures in its
      `tests/fixtures`, regenerated with `expected.ts`).
  - **`pen test-blot` is Deno-free too.** gdev-94 made every test module a
    plain entry file (`src/game.blot`, `test_support/sandbox.blot`), so the
    runner's game mode and its Deno script are gone. Every module compiles
    through the compiler plugin at `src/test_fixture.blot`, one session per
    worker. 472 tests in 54 modules pass with no Deno on `PATH`, in about a
    minute.
  - **Still on Deno:** checkouts without a `pen.json` (`blot_build.ts`, kept
    for them), tunables bake, `pen parity`, the V8 tools, `pen web` and
    `pen build --web`.
  - **WIT changes the plugin agents suggested** (not made yet):
    - an explicit `close` on compiler sessions (today a drop saves blotc's
      restart cache, which can block up to 10 s);
    - diagnostics' `details` as `option<string>`, for typed holes and dispatch
      explanations;
    - a failing `generate` that still names its dependencies. Today pen
      watches the whole checkout after such a failure.
    - `files` gaining `is-file` and `canonical`;
    - a path convention for Windows: always pass `/`-separated paths.
- **Order of work:**
  1. The plugin host in pen: WIT, loading, grants, tests with a small plugin.
  2. A `blot` compiler plugin (Rust, `wasm32-wasip2`), a `serve-project`
     client written from `PROJECT_PROTOCOL.md`, that starts `blotc` through the
     process capability. It is used by `pen dev`, `pen run` and the dev tools
     build when the checkout chooses it; Deno stays the default until step 3.
  3. A `gdev-assets` plugin porting `asset_parsers.ts` and the module
     generation in `compiler/assets.ts` (WGSL through naga). It's done when
     gdev's `game.wasm` from plugins equals the one from Deno, byte for byte.
     Then gdev switches, `blot_build.ts` goes, and `pen test-blot` compiles
     through the compiler plugin.
  4. When blot2 task 096 puts typed imports in blotc, the `gdev-assets`
     plugin goes, and the compiler plugin only passes the asset settings on.

## Checks

- `just test`: unit and GPU tests, golden images, renderer parity, the V8
  comparison, determinism, and Blot tests. Run it when nothing else is using
  the GPU: concurrent GPU work has made it fail with out-of-memory.
- `cargo clippy --workspace --all-targets --features pen/steam -- -D warnings`,
  and the same with `-p pen --no-default-features --features steam` for the
  player runtime (CI runs both).
- Dev tools: `pen run --headless --frames 300 --script "overlay=/diagnostics,high"
  --overlay-shot shot.png`.
- `cargo test --release -p pen-render` when the renderer changes: Rust 1.99's
  release optimizer once miscompiled it.
