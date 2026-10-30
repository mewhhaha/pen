# Continue the owned-storage migration

## Status at handoff

The immediate milestone remains **one real Gdev frame in Pen**. No real Gdev
Wasm module or frame has completed during this migration. Compiler and source
integration have advanced through many concrete proof boundaries; passing small
guests, protocol tests or Rust builds does not establish a working game.

Keep the sibling `../blot2`, `../ink` and `../gdev` checkouts at the
corresponding handoff commits. Read their `CONTINUE.md` files and Blot's
`tasks/STORAGE_REWRITE.md` before continuing. The owner requests continuous
checked commits and pushes to `main` across all four repositories. Verify push
success; a local commit is not a remote checkpoint. Preserve all existing
render, assets and runtime work.

## Contracts to preserve

- Guest ABI **8** is the owned data-stack path. Native and browser readers must
  agree with the compiler's manifest, storage layout and copied numeric packets.
  Do not restore immutable/COW/RC semantics or language-task support. Data job
  queues and copied-buffer workers remain intentional application features.
- Pen/native host packages report **0.1.0**, the compiler-plugin integration
  reports **0.2.0**, and `Cargo.lock` pins Wasmtime **49.0.3**. Rebuild these on
  a new machine; a pre-existing `pen` on `PATH` may use another ABI or
  generator.
- Program/session protocol is **2**, distinct from the guest ABI. A presentation
  rejection does not automatically roll simulation back. Keep renderer
  transactions and explicit Snapshot/Restore, including netplay restore/replay.
  A retry must not silently advance simulation twice. A trapped session may need
  recreation from an explicit checkpoint; partial mutations are possible.
- Guest output is encoded while invocation-local frame/payload owners remain
  alive. The host receives independent numeric packet copies. No guest borrowed
  descriptor or frame access bundle may escape through a host packet/worker.
- The code generator emits a concrete render header plus known projections and
  concrete counts, streaming flat collections into a caller-owned packet buffer.
  Keep section order and framing. Change templates/schema, then regenerate;
  never fix generated Blot or Rust by hand.

The render generator's concrete header/projection/count batch previously passed
10 focused tests and regeneration reproducibility. This is historical evidence,
not qualification of the final combined candidate. Historical results do not
qualify the current compiler and guest together.

## Resume on another machine

1. Follow `AGENTS.md` and the checked-in Rust toolchain. Build Blot using Zig
   0.17.0 and its native development backend (`-Dllvm=false`); configure Gdev's
   compiler plugin to that checkout. Do not reuse machine-local absolute paths.
2. From Pen, rebuild native Pen and the plugin workspace:

   ```sh
   cargo build --release -p pen
   just plugins
   ```

   For browser integration, also build `just web-host` with the required Wasm
   target and Binaryen available. `just install` installs a broader local
   runtime/player/browser bundle; it is optional for the first headless native
   frame when invoking `target/release/pen` directly.
3. Regenerate/check the protocol through `just sync-protocol` and the documented
   `pen codegen --check` route. Use the new binary's generated bindings in Gdev.
4. Resume the exact Gdev main compile frontier from its handoff. Record the
   compiler's **configured hello identity**, executable SHA-256, project
   settings and resulting Wasm hash. The raw build fingerprint differs when the
   project overrides the compiler inference budget. Rebuild Pen after host/ABI
   changes.
5. Once main compiles, obtain a real session start and frame headlessly,
   preserving packet diagnostics. Then test native and browser
   acknowledgement/recovery.

## Pending integration and qualification

Start with the actual frame, then a small affected set; do not restart an
exhaustive build/test loop after every source annotation.

- ABI 8 acceptance/rejection, copied packet independence and copied-worker
  responses; stale ABI/engine/budget cache entries must not be reused.
- Native/browser session-v2 acknowledgements, rejected render transaction,
  explicit Snapshot/Restore and restore/replay, trap/checkpoint recovery and no
  duplicate simulation step on retry.
- Generator tests/reproducibility and protocol freshness after final changes;
  render groups, variable payload spans, retained reset/rejection and UI output.
- `cargo fmt --all --check` in both workspaces, plugin tests and affected Cargo
  host/runtime/codegen tests. Follow `just packets` before Cargo tests requiring
  real packet fixtures. Full `just check`/`just test` remain pending until the
  game fixture can actually be built; GPU consumers run sequentially.
- Current sibling compiler qualification, remaining lint findings and the exact
  main-build frontier are recorded in
  [Blot's continuation](../blot2/CONTINUE.md). Native Pen and its plugin build
  successfully; Gdev `pen codegen --check` passes. These checks do not qualify a
  real frame.

Native headless GPU verification can use a locally installed Vulkan software
adapter. SwiftShader is optional: point `VK_ICD_FILENAMES` at **your own**
`vk_swiftshader_icd.json` (for example beneath a local Chromium/Playwright
installation or a separately unpacked SDK). Existing `/var/tmp`, cache or
browser-install paths are machine-local conveniences, not required checkout
inputs. CPU packet/session checks can proceed without SwiftShader. Never open
windows on the owner's desktop; do not start Steam app 480 or accept the
cargo-xwin license as part of this work.
