# Working in pen

- This is a greenfield project. Breaking changes are welcome. Remove superseded
  code, APIs and compatibility scaffolding instead of preserving legacy
  behavior.
- The owner requests continuous checkpoints to `main`: commit and push coherent,
  checked batches as work progresses. Verify each push and report failures.
- Active source/generator migration is checkpointed, and the current reviewed
  compiler and host pass one real Gdev frame with an inspected PNG. Use
  `CONTINUE.md` for exact identities and remaining original-export/backend
  qualification; source inventory success is not whole-runtime qualification.
- Read `CONTINUE.md` and `../blot2/CONTINUE.md` first. Preserve exact candidate
  identities; installed binaries qualify only the revisions actually tested.
- Use the existing sibling Blot, Ink and Gdev checkouts. Build the matching
  compiler, native host and compiler plugin with the pinned Rust toolchain.
  `just install` also builds browser/player/Steam packaging; it is optional for
  the first frame with `target/release/pen`.
- Preserve guest ABI 8, session protocol 2, copied numeric packets and explicit
  checkpoint/restore. Renderer rejection does not imply simulation rollback;
  retries must not advance simulation twice. Borrowed guest views must not
  escape through host or worker packets. Language tasks are retired; isolated
  host workers and data queues remain.
- Verify headlessly; do not open windows on the owner's desktop.
- Never start Steam with app 480 or accept Microsoft's cargo-xwin license.
- Preserve unrelated work in sibling checkouts.
- Keep large artifacts in `~/.cache/pen` or `/var/tmp`; `/tmp` is RAM-backed.
- Run GPU consumers sequentially. `just test` prepares real game/packet fixtures
  before testing; direct Cargo tests need `just packets` first.
- Generate packet code from `protocol/schema.json` with `just sync-protocol`. Do
  not hand-edit generated files. Keep `cargo fmt --all --check` clean in both
  workspaces (rustfmt.toml; CI and `just check` run it); packets.rs is generated
  and skipped.
- Check protocol freshness without rewriting outputs when no generator change is
  intended. Run affected host/plugin/generator tests after a coherent change;
  full game-dependent gates remain pending until a real fixture builds.
- Keep continuation notes current and concise. Older ABI and performance
  documents describe their own candidates, not current integration readiness.
