# Module compilation and cache memory

The 23 October 2026 isolated probe attributes a large native memory peak to
Wasmtime's inlining compilation. It supplies an already compiled game module
directly to `blot-host`; it does not run the Blot compiler, load assets, create a
renderer, or execute a frame. It therefore does not reproduce or attribute all
of the historical application's roughly 5 GiB peak.

Three fresh processes per configuration, in alternating order, on Linux
6.18.55 with a four-CPU cgroup quota and a 32 GiB memory limit:

| Engine | Inlining | Wall seconds | User + system seconds | Peak RSS MiB |
| --- | --- | ---: | ---: | ---: |
| Normal | On | 3.245 | 11.040 | 2,513.9 |
| Normal | Off | 0.580 | 2.127 | 90.7 |
| Development | On | 3.181 | 10.890 | 2,363.8 |
| Development | Off | 0.570 | 2.036 | 105.1 |

These are medians. CPU is accumulated across the child's threads; wall time is
elapsed time. RSS is that process's maximum resident set, not live allocator
bytes or a sum of separate process peaks. OS caches were not flushed.

Loading the populated disk cache takes about 5–6 ms wall time including process
startup and peaks at 22.4 MiB in this probe. That first measurement predates the
cache integrity envelope described below; it must not be used to claim the
envelope has no loading cost.

The module is 2,241,523 bytes, SHA-256
`b597c6c13c0c9199333122f11ff19e46abd34203d4193fe3d18c34837498d46e`.
The executable, all individual samples, machine information, and cache state
are pinned in [the raw report](measurements/module-compile-baseline-2026-10-23.json).
The probe adds no guest instrumentation. It uses Linux child-process accounting
and saves the example's own module timer separately in subsequent reports.

Reproduce with an equivalent frozen module:

```sh
cargo build --release -p blot-host --example compile_time
python3 scripts/profile-compile.py /path/to/game.wasm --out target/module-profile
```

The output directory must be new. Each cold sample runs in its own process;
each disk-cache population/load pair uses a separate directory. `--inline-budget`
is an experimental override of Wasmtime's expansion threshold in the example;
it does not change production engine defaults. A runtime benefit from inlining
needs its own frame measurements before choosing a different release policy.
Development already defaults to inlining off.

## Cache ownership and validation

The development engine's in-memory cache owns two generations of up to 32 MiB
of key/value bytes each. An entry larger than a generation is declined. A
replacement accounts for the replaced bytes before deciding to rotate; a
promotion removes the older copy. These bounds exclude allocator metadata,
returned copies, compiled modules held by callers, and active compiler scratch.
They cannot explain away the much larger inlining peak above.

The optional module disk cache binds each native artifact to the exact Wasm
bytes and Wasmtime compatibility identity. Its envelope checks that key and a
SHA-256 of the complete payload before deserialization. Old, truncated, corrupt,
or mismatched entries fall back to compilation. Reads are bounded, including if
a file grows during the read. Entries above 64 MiB decline disk caching; the
eight-entry policy bounds completed artifacts to approximately 512 MiB plus
headers. Writes use separate temporary files and atomic publication.

This remains a trusted local executable cache. Its checksum detects accidental
damage and misplaced artifacts; it is not authentication against someone who
can replace both a payload and its checksum.

The [integrity-envelope repeat](measurements/module-compile-integrity-2026-10-23.json)
keeps cold compilation near the first measurement. It makes the validation cost
visible: cached inlining builds load in 31–34 ms including process startup,
versus 5–6 ms before validation, and peak at 35–41 MiB. Non-inlining builds load
in 12–13 ms and peak at 22.5 MiB. Reading, hashing and safely deserializing an
owned buffer costs more than mapping an unchecked file. This remains far below
the measured cold compilation cost; no zero-overhead claim is made.

The twelve host tests cover the generation bound, replacement/promotion,
complete-payload byte flips and truncations, wrong keys, old formats, actual
recompilation after corruption, oversized entries, pruning, and existing guest
ownership, traps, probes, snapshots and deadline behavior. A focused writer
regression also checks that a temporary-file collision leaves the other writer’s
file intact.

## Inlining-budget experiment

A [repeat with the threshold set to 128](measurements/module-compile-budget128-2026-10-23.json)
reduces normal-engine cold compilation to 0.947 s wall, 2.870 s CPU and 778.2 MiB
peak RSS. The development engine with inlining explicitly enabled measures
0.790 s wall, 2.450 s CPU and 606.1 MiB. These are again three-process medians.
The much smaller default development path remains available.

These results qualify compilation cost only. Production settings remain
unchanged pending paired guest-frame timings and bit-exact packet/snapshot
checks. The next check compares guest frame distributions and exact output traces before
choosing a production policy.
