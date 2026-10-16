# Working in pen

- Commit or push only when the owner asks.
- Verify headlessly; do not open windows on the owner's desktop.
- Never start Steam with app 480 or accept Microsoft's cargo-xwin license.
- Preserve unrelated work in sibling checkouts.
- Keep large artifacts in `~/.cache/pen` or `/var/tmp`; `/tmp` is RAM-backed.
- Run GPU consumers sequentially. `just test` prepares real game/packet
  fixtures before testing; direct Cargo tests need `just packets` first.
- Generate packet code from `protocol/schema.json` with `just sync-protocol`.
  Do not hand-edit generated files. Keep `cargo fmt --all --check` clean in
  both workspaces (rustfmt.toml; CI and `just check` run it); packets.rs is
  generated and skipped.
