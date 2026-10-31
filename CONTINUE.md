# Continue the owned-storage migration

## Current status

Read `AGENTS.md` and [Blot's shared continuation](../blot2/CONTINUE.md). Use the
existing sibling checkouts and pinned toolchains. This is a greenfield project;
remove superseded APIs and implementations rather than keeping compatibility
paths. Commit coherent checked batches continuously to `main` and verify remote
hashes. Environment publication requires a separate explicit request.

Actual Gdev main passes one native frame with the current reviewed compiler and
rebuilt host. All six checks pass; the inspected 640×480 PNG and guest trace
match the original frame byte for byte. The
[current exact result](../blot2/tasks/storage-handoff-evidence/2026-10-31-migrated-frame.json)
records compiler/host/Wasm/source identities and the Mesa driver. Broader
gameplay, browser, recovery and release gates remain open. Recheck actual
integration after semantic changes.

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

## Contracts

- Guest ABI 8 uses owned data stacks and copied numeric host packets. Native and
  browser readers must agree with the compiler. Do not restore counted/COW/RC
  semantics or language-task support. Copied-packet host workers remain.
- Native packages are 0.1.0, compiler-plugin integration 0.2.0 and locked
  Wasmtime 49.0.3. Build matching tools; a pre-existing `pen` on PATH may
  differ.
- Session protocol 2 preserves explicit Snapshot/Restore. Renderer rejection
  does not roll simulation back. Retries must not advance simulation twice;
  trapped sessions may require recreation from a checkpoint.
- Encode output while invocation-local owners remain alive. No borrowed guest
  descriptor escapes through a host packet or worker. Preserve packet section
  order, retained reset/rejection, variable payloads and full-width IDs.
- Fix schema/templates and regenerate bindings; never patch generated output.
  Keep read-only array/view consumers concrete without adding copies.

## Build and verify

Use Zig 0.17.0 and `-Dllvm=false` for Blot development. Build the native host
and plugin workspace using the checked-in Rust toolchain:

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
