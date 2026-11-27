# Continue the owned-storage migration

## Current status

Web exports list their startup files in `game.json`, prune unused optional JS
glue (`structured.js`, `kernels.js`, `kernel_worker.js`), externalize heavy
optional 3D WGSL pass shaders into lazy `shaders/*.wgsl` files
(`deferred-shaders`), and load required assets alongside GPU setup. Optional
files absent from the export cause no requests. `pen build --web --web-compact`
supports optional `pen.json` `"web"` config (`mode` / `features`), auto-detects
pure-2D games (omitting `render-3d` and `audio-advanced`), caches tailored
source builds under `~/.cache/pen/web-hosts/`, and otherwise picks the tightest
prebuilt tier (`web-2d`, `web-3d`, `web-compact`, `web`). `crates/pen-web/build.sh`
uses `-Z build-std=std,panic_abort` with `-Cpanic=immediate-abort` when
`rust-src` is installed. The Linux tools bundle publishes all four prebuilt
browser hosts (`web`, `web-compact`, `web-3d`, `web-2d`).

Read `AGENTS.md` and [Blot's shared continuation](../blot2/CONTINUE.md). Use the
existing sibling checkouts and pinned toolchains. This is a greenfield project;
remove superseded APIs and implementations rather than keeping compatibility
paths. Commit coherent checked batches continuously to `main` and verify remote
hashes. Environment publication requires a separate explicit request.

The named-function migration now changes the Blot generator templates and test
entry discovery. Regenerated Gdev bindings and Ink's protocol fixtures preserve
Rust packet output, schema and wire layout. Ten codegen laws, eight suite laws
and both generated-output freshness checks pass. This qualifies generation and
discovery; the new compiler's full release/Main qualification remains pending.
See [the shared source-migration evidence](../blot2/benchmarks/evidence/qualification/2026-11-08-named-source-migration.json).
The frame evidence below qualifies its older compiler/host, not this batch.

Actual Gdev main passes one native frame with the current reviewed compiler and
rebuilt host. All six checks pass; the inspected 640×480 PNG and guest trace
match the original frame byte for byte. The
[current exact result](../blot2/tasks/storage-handoff-evidence/2026-10-31-migrated-frame.json)
records compiler/host/Wasm/source identities and the Mesa driver. Broader
gameplay, browser, recovery and release gates remain open. Recheck actual
integration after semantic changes.

The generated Blot is in the explicit ownership form (17 November 2026): the
packet encoders (`encode_render_views`, `encode_mesh_frame`) take their buffer
with `own`, write it with `:=` and return it; the canonical stream threads one
owned `Stream` value through a function per record; and `program.run` owns the
game's world and rebinds it round the loop (`access := execute (own access) …`),
so a game's `execute` takes the world over and returns it. The template and
`examples/bounce` hold their world as a plain field. No generated file has an `=`
store. Ten generator tests, the codec round trip (2,000 input and 2,000 render
frames), the example's seven tests and a scaffolded game's three pass; Gdev's
packets over 120 frames are byte-identical to before. See
`../blot2/tasks/MEMORY_MODEL.md`, "The generated encoders".

The generator now emits borrowed frame/header records, known projections and
named packet writers. Owning RenderFrame/Extension compatibility wrappers are
removed. The production `encode_render_views` bridge and canonical stream both
exercise 2,000 input and 2,000 render frames, with exact packet equality and
Rust decoding across all 23 sections. All ten generator tests pass; sibling
bindings and provenance are regenerated. Rust packet output and wire layout are
unchanged.

Bounce/new-game fixtures pass eight original entries with eleven assertions.
Readonly diagnostic rows and selector views have their original laws. Real ABI 8
session create/restore/frame/rejected-presentation/checkpoint paths pass focused
CPU tests. Copied-worker and owned-storage fixtures pass; plugins pass nineteen
unit and nine compiler integration checks. Atlas CPU tests pass. These results
do not imply full GPU/browser qualification.

`blot_host::processes::Processes` is an opt-in coordinator for trusted
ABI 8 programs whose `init` and `step` both return. The caller owns the turn
allowance, mailbox and linear-memory caps. The native test runs two isolated
guests on one thread, preserves ticket outcomes across fault and restore, and
replays admissions and applied events through the same Blot caller. Gdev's
frame loop does not start these processes automatically.

## Contracts

- Guest ABI 8 uses owned data stacks and copied numeric host packets, and
  `pen:guest/1` provides an `externref`-free ABI (`pen:guest/1::exchange`,
  `pen:main`, `pen:allocate`, standard `memory`) for non-Blot languages. Native
  and browser readers agree on both. Do not restore counted/COW/RC semantics or
  language-task support. Copied-packet host workers remain.
- Native packages are 0.1.0, compiler-plugin integration `pen:plugin@0.4.0`
  (adding `codegen` and diagnostic `fix`/`doc` fields), and locked Wasmtime
  49.0.3. `pen build --web --game DIR` exports prebuilt bundles directly, and
  `pen.json` supports `compiler.entry` / `compiler.devtools` plus `web`. Build
  matching tools; a pre-existing `pen` on PATH may differ.
- Session protocol 2 preserves explicit Snapshot/Restore. Renderer rejection
  does not roll simulation back. Retries must not advance simulation twice;
  trapped sessions may require recreation from a checkpoint.
- Encode output while invocation-local owners remain alive. No borrowed guest
  descriptor escapes through a host packet or worker. Preserve packet section
  order, retained reset/rejection, variable payloads and full-width IDs.
- Fix schema/templates and regenerate bindings; never patch generated output.
  Keep read-only array/view consumers concrete without adding copies.

## Build and verify

Build the Blot compiler in `../blot2` with `deno task build:compiler` (it is
written in Goose and needs clang, cmake, ninja and Python 3). Build the native
host and plugin workspace using the checked-in Rust toolchain:

```sh
cargo build --release -p pen
just plugins
```

`just install` includes broader browser/player/Steam packaging and is optional
for a native frame through `target/release/pen`. Browser integration
additionally needs `just web-host` and its Wasm/Binaryen tools.

Check protocol freshness through `just sync-protocol` and `pen codegen --check`.
Run affected codegen/host/runtime/plugin checks, plus `cargo fmt --all --check`
in both workspaces. Follow `just packets` before Cargo tests needing real packet
fixtures. Full `just check`/`just test`, browser acknowledgement/recovery and
release qualification remain pending.

From Blot, run actual Gdev main, requiring all checks and an inspected PNG:

```sh
python3 scripts/storage_handoff/run_pen.py --launch
```

Record configured compiler hello identity separately from executable SHA-256,
plus project settings and Wasm hash. Project inference overrides affect hello
identity. Rebuild the host after host/ABI changes. Keep detailed compiler and
fixture frontiers in Blot's shared record rather than duplicate histories here.

Run GPU checks headlessly and sequentially. A local Vulkan software adapter may
be used; SwiftShader paths are machine-specific conveniences. Never open desktop
windows, start Steam app 480 or accept the cargo-xwin license as part of this
work. Preserve unrelated sibling source and assets.
