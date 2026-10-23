# Rejected frame diagnostics

Both hosts report rejected render packets using protocol field names:

```text
insets[1].draws[1].transform[4] at frame lane 296: expected a finite float32, got NaN
```

Indices are zero-based. Matrix indices follow the sixteen column-major wire
lanes. A diagnostic contains the section, outer item index (when present),
full field path, absolute packet lane, expected constraint, and actual value
or missing/length information. Rust callers can downcast the decoder's error
to `pen_protocol::diagnostic::FrameDiagnostic`; the original rejection remains
in its error chain.

The walk is generated from `protocol/schema.json` with `just sync-protocol`.
It runs only after decoding fails, without allocating render records. It
validates sizes before locating sections, checks variable records against
section boundaries, and stops when a count, tag or length no longer permits
a trustworthy interpretation. Application extension payloads are identified
by extension index and lane; Pen does not invent their field names. Existing
fallback choices, such as MSAA, retain their behavior.

The shared decoder handles startup, save restores, hot reload and live frames.
During development, a rejected live frame pauses at the last accepted state;
a valid reload can resume it. The browser keeps the error visible until a
valid frame replaces it, including games without a devtools overlay. A failed
Continue/save restore stays visible until a successful load or reload; normal
menu frames do not clear that explanation.

## Optional application provenance

An application or Ink adapter may opt in to development provenance. None is
generated or sent by default. It describes a packet range according to the
application, not proof of where an invalid number was first calculated. Pen
does not load the named source file or infer game rules from an object ID.

The existing extension block follows the packet and its draw extras:

```text
EXTENSIONS_MARKER, extension_count,
    tag, payload_lane_count, payload...,
    ...
```

Application tags remain 1 through `MAX_EXTENSION_TAG`. Reserved tag
`PROVENANCE_TAG` (0) may appear once. Its payload is UTF-8 JSON bytes, each
encoded as one integer float32 lane from 0 through 255:

```json
{
  "version": 1,
  "entries": [{
    "first": 148,
    "end": 164,
    "object": "building:7:42",
    "system": "construction",
    "source": { "file": "src/buildings.blot", "start": 120, "end": 164 }
  }]
}
```

`first` and `end` delimit a half-open range of absolute packet lanes, before
the extensions marker. Entries must be ordered and disjoint. Object IDs are
opaque strings; include store and generation identity when slots are reused.
Source spans are half-open UTF-8 byte offsets, encoded as JSON integers so
full-width `u32` values survive. `object`, `system` and `source` may be omitted
or null, but at least one must be present. Unknown fields and versions fail.

Schema bounds allow at most 64 entries, 16,384 JSON bytes, and 256 UTF-8 bytes
per object ID, system label or filename. Displayed text is escaped. Malformed
provenance rejects an otherwise valid packet; untrusted metadata is omitted
when explaining another fault. A corrupt layout never triggers a search for
marker-like numbers in arbitrary payloads.

Rust guests and adapters can construct `Provenance` and call
`Provenance::extension(packet_end)` to validate and encode it. Other languages
can append this reserved extension to their encoded development packets.
In Blot, `types.Extension` with `protocol.provenance_tag` and the UTF-8 bytes
as `values` can travel through `protocol_render.encode_extensions`, alongside
any application sections. Older hosts reject tag 0; omit it when targeting
those hosts.
Pen consumes it without adding it to the rendered frame or retaining an
identity map. Each error uses only its packet's metadata, so a reload, reused
object slot or later uninstrumented frame cannot reuse old labels. Raw packet
bytes and recording hashes include the opt-in extension; decoded rendering
and service requests are unchanged.

## Verification

Prepare real game fixtures with `just packets`. The protocol tests cover
nested matrices, solid faces, field patches, services, draw extras, invalid
tags/counts, every fixture truncation, oversized packets, hostile numeric
mutations and bounded provenance. Native recovery checks rejected candidates
are never committed, pause/reload behavior, and overlapped frames.

Compare native and browser diagnostics and recovery with identical packets
and guest binaries:

```sh
export PEN_DIAGNOSTIC_FIXTURES="$PWD/target/packet-diagnostics"
cargo test -p pen-protocol --test diagnostics
cargo test -p pen session::recovery_tests -- --test-threads=1
sh crates/pen-web/build.sh "$PEN_DIAGNOSTIC_FIXTURES/web"
export PLAYWRIGHT_MODULE="$PWD/target/browser-tools/node_modules/playwright-core/index.mjs"
PEN_WEB_HOST="$PEN_DIAGNOSTIC_FIXTURES/web" node crates/pen-web/tests/diagnostics.mjs
```

`just browser-tools` installs the pinned browser dependencies. All browser
checks run headlessly. Run native and browser GPU checks sequentially.
`PEN_WEB_WITHOUT=jspi` exercises the worker guest adapter.

The opt-in release measurement compares the raw decoder with its diagnostic
wrapper, provenance-enabled decoding, and rejection with provenance. It uses
a real game packet, reuses frame allocations, alternates measurement order,
and writes seven samples per mode to `timing.json`:

```sh
cargo test --release -p pen-protocol diagnostic_overhead -- --ignored --nocapture
```

These are host boundary costs; application-side provenance encoding and full
frame times are not included.

On 23 October 2026, the release probe on a Ryzen 7 7800X3D with Rust 1.99.0
used a 1,078-lane gdev packet, 2,000 iterations per sample and seven samples:

| Boundary operation | Median |
|---|---:|
| Raw decoder | 1.375 µs |
| Shared decoder, provenance omitted | 1.369 µs |
| Shared decoder, one provenance entry | 2.081 µs |
| Header NaN, provenance lookup and diagnostic formatting | 4.868 µs |

The raw and normal valid-path samples overlap; this probe found no measurable
wrapper overhead. One opt-in provenance entry added about 0.71 µs to decoding.
The rejection measurement covers an early header fault, not a worst-case walk.
