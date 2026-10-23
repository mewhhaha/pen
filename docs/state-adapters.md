# State formats and adapters

Pen owns the save-file container (`GDEVSAVE`, its version, state-schema
identifier and optional timestamp/lineage), storage, replay and the
snapshot/restore calls. The game owns the payload: a bounded array of finite
float32 lanes. No section layout is required. Existing save files keep their
bytes and remain readable.

Ink owns its persistence format, typed codecs and guest migrations. Pen's
optional `pen-ink` adapter understands that format for numeric inspection,
tunable readback, restore descriptions and the existing declared migration
fallback. The adapter is a separate Rust crate; `pen-runtime` has no Ink
dependency or Ink wire parser.

## Selecting an adapter

An Ink game opts in with `src/assets/saved.json`:

```json
{ "format": "ink-persist-v1" }
```

`ink-persist-v1` names the adapter's supported framing, independently of the
game's state schema or individual codec versions. Add a `types` object to
use [declared migration fallback](saved-types.md). Existing `saved.json`
declarations must add the `format` property; neither the presence of type
declarations nor magic numbers in a save select an adapter.

No file, or `{ "format": "opaque" }`, leaves state uninterpreted. Saving,
loading, undo, replay and hot reload still use the game's own snapshot and
restore functions. Numeric state queries, lane watches, tunable readback,
baking and preset capture return unavailable when they need an unsupported
view. Tune events and [semantic observations](semantic-inspection.md) remain
independent of the save format. `pen diverge` falls back to raw lane diffs.

`pen check` rejects unknown formats and malformed declarations. Both hosts
cache the selected adapter at startup and refresh it on development reload.
Native asset changes also refresh it immediately. Tooling queries reuse the
same adapter without reading or parsing the file. Missing files select opaque
state; malformed declarations and other read failures remain explicit errors.
They reject startup, restore and build activation, and fail format-dependent
tooling queries until corrected. The current simulation can continue running.
Packaging includes the declaration. Web exports without one receive an explicit
opaque default.

The runtime also shares one snapshot per accepted guest state revision across
live tooling queries. Every accepted frame invalidates it, including zero-delta
frames; successful load, undo, rewind and build activation also invalidate it.
Rejected frames and failed state replacements leave the accepted snapshot intact.
Snapshot requests must run while the guest is idle. Semantic inspection reads
selected published objects and does not request a whole-world snapshot.

## Adapter boundary

`pen_runtime::state::StateAdapter` supplies optional section views, lane
labels, tuned values and restore descriptions. An unsupported or malformed
view is `None`, distinct from a valid empty view. Ink views require a complete
state; truncated saves never become partial tuning or inspection results.

The guest always restores first. An adapter can propose one migration retry
when the guest cannot migrate. Pen validates the proposed payload, starts a
fresh instance and asks it to restore. The adapter describes the second
restore's actual result. Normal guest migrations take priority, and scene
loads do not use the fallback.

Other formats can implement this Rust interface and register selection in
both hosts. The bundled selector currently supports opaque state and Ink.
This is an in-process adapter interface, separate from the Wasm compiler/test
plugin interface in `wit/pen-plugin.wit`. Games can keep serialization and
migrations entirely in guest code without adding a host adapter.

## Verification

`cargo test -p pen-runtime -p pen-ink` checks opaque state, an unrelated flat
state adapter, Ink inspection and migration, and restore/hot-reload behavior.
Run `just packets` before Cargo tests, as for other runtime changes.

The native `semantic_history_preserves_simulation_and_external_service_replay`
fixture and `crates/pen-web/tests/semantic.mjs` exercise explicit Ink selection,
restore, replay and hot reload. The browser harness also runs without a
`saved.json`: format queries are unavailable, semantic observations still work,
and the exported save preserves the guest payload. It runs with both JSPI
and the worker fallback (`PEN_WEB_WITHOUT=jspi`).
