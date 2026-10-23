# Semantic state and recorded decisions

Pen's developer service can inspect application-supplied values and reasons at
positions in the existing input/snapshot timeline. It does not decode arbitrary
game state or infer a reason from a changed number. Native and browser hosts use
the same transport, retention, diff and watch implementation.

This is an opt-in development interface. The game publishes observations; its
Blot/Ink devtools program requests them through `DEVTOOLS_OP.semantic`. The
application owns selection, layout, source navigation and game-specific controls.
Pen owns bounded transport and the existing pause, step and history services.

## Publishing observations

Declare an application extension tag in `pen.json` (for example 42). In a
development build, put one version-1 observation in that extension's `values`:
UTF-8 JSON bytes, each represented exactly as one float32 lane. No reserved tag
or game-specific Rust code is needed. `semantic::Observation::extension` is a
Rust producer helper; Blot can use generated `types.Extension` and
`protocol_render.encode_extensions` after encoding its frame.

```json
{
  "version": 1,
  "schema": {
    "id": "haulers-v1",
    "types": [{
      "id": 1, "label": "Hauler",
      "descriptor": {
        "version": 1,
        "variants": [{
          "tag": 0, "label": "Hauling",
          "fields": [{
            "id": 17, "label": "destination", "type_name": "U32",
            "kind": "unsigned"
          }]
        }]
      }
    }]
  },
  "tick": "51",
  "objects": [{
    "id": { "kind": "Hauler", "words": [7, 10, 2, 3] },
    "type_id": 1, "tag": 0,
    "fields": [{ "id": 17, "value": { "kind": "unsigned", "value": 42 } }]
  }],
  "decisions": [{
    "tick": "51",
    "object": { "kind": "Hauler", "words": [7, 10, 2, 3] },
    "system": "routing", "source": { "file": "src/hauler.blot", "start": 120, "end": 164 },
    "reason": "destination full"
  }],
  "omitted_objects": 0,
  "omitted_decisions": 0
}
```

The application validates domain values before publication. Pen checks structure,
identity uniqueness, descriptor membership, value kinds, finite F32s, bounds,
and decision order. `unsigned` is a full-width U32, `float` a finite F32,
`boolean` a JSON Boolean, and `encoded` an array of exact U32 words. Encoded
custom/nested values are identified by their declared type; Pen never invents
their internal field names. Unknown fields and versions are rejected.

The object list is the publisher's current observed inventory. Account for
unobserved objects with `omitted_objects`; missing identities in an incomplete
inventory are *unobserved*, not destroyed. Fields of each included object must
be complete for its declared variant. Stable field IDs determine correspondence,
not labels or row order. The identity includes an application kind and one to
eight U32 words; using just a reusable array index is insufficient.

Each frame publishes only that frame's new decisions in execution order, with
nondecreasing ticks no later than the snapshot tick. Decisions may name objects
already destroyed. `decisions: null` means instrumentation was disabled;
`decisions: []` means it recorded no decisions. `omitted_decisions` reports events
the producer dropped, including its own ring overflow. Reasons and source spans
are supplied by the producer, not verified causal traces. Source offsets are
half-open UTF-8 byte positions. Ticks and timeline positions are canonical
decimal U64 strings so JavaScript does not round them.

### Mapping Ink's completed contracts

The transport projects the contracts introduced in
[Ink ccac41f](https://github.com/mewhhaha/ink/commit/ccac41fc6a5eaa704821874726323333954fdff4),
documented in [Ink's state mechanisms](https://github.com/mewhhaha/ink/blob/ccac41fc6a5eaa704821874726323333954fdff4/docs/state.md).
Pen has no Ink import, reflection engine or second codec implementation.

| Ink value | Publication |
|---|---|
| Generated `Name_descriptor` | Descriptor version, variant tags, stable field IDs, labels and declared types |
| `codec.inspect descriptor decode words` | Publish only a successful validated `View`; its field kinds map to `unsigned`, `float`, `boolean`, `encoded` |
| `handles.encode_handle handle` | The four exact words: region, birth origin, slot, generation; add the nominal kind name |
| `behavior.State` | Application-described fields such as stable step ID, elapsed time, local state and status |
| Behavior `Waiting reason`, transitions, application decisions | Explicit decision records with the application's reason text, system label and available source span |

Convert Ink's label code points to UTF-8 JSON text. Preserve U32 words exactly;
do not cast handle words or codec words numerically into F32. Use Ink's validated
view for custom fields rather than interpreting its persistence lanes in Pen.
An application may project nested validated values as separately described
objects. A changed selection scope or incompatible descriptor needs a new schema
identity. A restore or divergent allocation history must follow Ink's region and
origin rules; Pen cannot repair reused identities.

## Querying the developer service

Use `service = DEVTOOLS`, `operation = DEVTOOLS_OP.semantic`, and `values` equal
to the JSON request's Unicode code points, as for other text services. The four
numeric request arguments are unused. Answers carry JSON text as code points.
Their service status is `ok`, `failed` for malformed requests, or `unavailable`
for missing instrumentation/history. The JSON includes a reason when unavailable.

Enable host collection with `{"op":"capture","tag":42}`. This also enables the
existing timeline history; it does not turn on instrumentation inside the game.
`{"op":"capture","tag":null}` clears observations and watches. The default is
disabled, including in development. A stripped release game omits the extension.
Native net sessions report this interface unavailable, consistent with their
existing restriction on pausing and time travel.

| Request `op` | Additional fields | Answer |
|---|---|---|
| `status` | None | Capture tag, epoch, retained range/count, retained allocation bytes, eviction count, invalidation reason, watch and pause position |
| `schema` | `position`, optional `first` | Type/variant rows and field descriptors; nullary variants have a row too |
| `objects` | `position`, optional `first` | Identities, type and variant IDs, tick, omission count, restore boundary |
| `values` | `position`, `object`, optional `first` | Named typed fields and values for that exact identity |
| `diff` | `before`, `after`, optional `object`, `first` | Created/destroyed identities, variant changes and named old/new field values |
| `decisions` | `from`, `through`, optional `object`, `first` | Inclusive position range, ordered by position then original event index, with producer omission count |
| `watch` | `watch: {object, field, equals}` | Arm a one-shot transition watch; `field: null` watches the whole object, `equals: null` watches any change |
| `unwatch` | None | Clear the watch and previous pause position |
| `seek` | `position` | Queue the existing timeline seek, paused; reject positions outside retained input history |

Positions identify the boundary **after** a frame: position `2` shows output of
input frame `1`. A schema query is tied to a retained observed position. Paged
answers include `rows`, `total` and `next`; repeat the same query with
`first: next` until `next` is null. A page never truncates a field or event.
Read pages while paused; running can evict the queried position between pages.

For a stalled actor: capture, inspect `values`, request its `decisions`, then
`seek` to the event's position or the preceding one and use the existing `step`
operation. A watch can pause on the destination field changing or transitioning
to a specified value. Native overlap is disabled while a watch is armed, so the
next simulation frame cannot run past that transition. A watch also cancels
remaining requested steps. Seeking itself does not consume a watch, and a
restore is not treated as a simulation transition.

## Retention and lifecycle

Observations use the existing timeline's positions and replayed inputs. They
contain no snapshots that can restore simulation state. Seek recomputes them
from accepted guest outputs; it does not call external services again. Recorded
external answers remain in the normal input recording. Continuing live from a
rewind drops the obsolete observed future along with the input future.

Successful hot reload clears descriptors, observations and watches even if the
game's saved-state schema stayed compatible. Descriptor changes also clear them;
an old field ID never silently acquires a new meaning in retained observations.
Loading a different recording clears them. A save/scene/undo replacement marks
the resulting observation as a restore boundary. Invalid observation JSON creates
an `invalid_observation` gap without rejecting or changing the accepted game
frame. No observation answers are injected into the game's service queue.

Missing data is explicit: `capture_disabled`, `instrumentation_disabled`,
`decisions_disabled`, `not_retained`, `objects_omitted`, or `object_absent`.
Requesting decisions across an unobserved or expired position returns unavailable
instead of silently presenting an incomplete causal history. Input history and
observation retention have independent limits; a position can still be seekable
after its observations expired.

Limits per publication: 65,536 UTF-8 bytes, 16 types, 64 variants, 256 descriptor
fields, 128 objects, 256 object fields in total, 64 decision events, 128 words
per encoded value and 256 UTF-8 bytes per label/path/reason. IDs, versions and
source offsets are U32s. The ring holds at most 3,600 positions (one minute at 60 frames/s) and 2 MiB of
retained allocation capacity, including frame storage, strings, descriptors,
objects and decisions. It shares one descriptor per epoch and unchanged object
values across adjacent samples; wire observations remain complete and independent.
Busy histories evict earlier. `status` reports `retained_bytes`, `max_bytes` and
`max_frames`. Queries accept at most 4,096 UTF-8 bytes and return at most 64 rows
and 10,000 row bytes per page, below the service's 16,384-code-point bound.

Observations do not affect simulation state, rendered records or external
service answers. Host capture on/off leaves guest packets identical. A game
compiled to omit publication produces different raw packet bytes/hashes, as
with other optional extensions; compare simulation snapshots and rendered
records when qualifying instrumented against stripped builds.

## Verification

Prepare real packets with `just packets`, then run:

```sh
cargo test -p pen-runtime
export PEN_SEMANTIC_FIXTURES="$PWD/target/semantic-inspection"
cargo test -p pen semantic_history_preserves_simulation_and_external_service_replay -- --test-threads=1
sh crates/pen-web/build.sh "$PEN_SEMANTIC_FIXTURES/web"
export PLAYWRIGHT_MODULE="$PWD/target/browser-tools/node_modules/playwright-core/index.mjs"
PEN_WEB_HOST="$PEN_SEMANTIC_FIXTURES/web" node crates/pen-web/tests/semantic.mjs
PEN_WEB_WITHOUT=jspi PEN_WEB_HOST="$PEN_SEMANTIC_FIXTURES/web" node crates/pen-web/tests/semantic.mjs
cargo test --release -p pen-runtime semantic_overhead -- --ignored --nocapture
```

Run GPU consumers sequentially and headlessly. The native regression exports
the same observed/stripped guest binaries and expected answers used by both
browser execution modes. It checks destination rejection/selection, reused
slots, rewind, transition pause, identical rendered/state witnesses and no
additional external-service calls during replay. Unit tests cover malformed
payloads, Unicode spans, exact U32s/U64s, overflow, pagination, eviction, schema
invalidation and disabled instrumentation. The opt-in benchmark separates host
capture and query/formatting cost; it excludes application-side encoding and
whole-frame performance.

On 23 October 2026, Rust 1.99.0 on a Ryzen 7 7800X3D, seven alternating-order
release samples of 5,000 iterations used a 554-byte publication with one object
and two primitive fields:

| Host operation | Median |
|---|---:|
| Disabled capture call | 0.0019 µs |
| Enabled capture, validation and bounded retention | 2.988 µs |
| Query and format the object's two named values | 3.692 µs |

These costs exclude producer encoding, packet decoding, rendering and transport
of the service answer. They qualify this small fixture, not a maximum-size
publication or a complete game's frame time. The benchmark writes all samples
to `PEN_SEMANTIC_FIXTURES/timing.json` when that environment variable is set.

The unchanged-object retention witness keeps 3,600 samples in 1,690,171 accounted
bytes, including its decision events. Changing large encoded values still evicts
within the same 2 MiB limit. Ink's producer benchmark measures encoding
separately: 1, 8 and 32 small objects produce 385, 1,155 and 3,839 bytes in median
0.062, 0.187 and 0.609 ms. These are synthetic publications; a game's schema and
decision rate determine its actual cost.
