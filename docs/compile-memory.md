# Module compilation and cache memory

The 23 October 2026 isolated probe attributes a large native memory peak to
Wasmtime's inlining compilation. It supplies an already compiled game module
directly to `blot-host`; it does not run the Blot compiler, load assets, create
a renderer, or execute a frame. It therefore does not reproduce or attribute all
of the historical application's roughly 5 GiB peak.

Three fresh processes per configuration, in alternating order, on Linux 6.18.55
with a four-CPU cgroup quota and a 32 GiB memory limit:

| Engine      | Inlining | Wall seconds | User + system seconds | Peak RSS MiB |
| ----------- | -------- | -----------: | --------------------: | -----------: |
| Normal      | On       |        3.245 |                11.040 |      2,513.9 |
| Normal      | Off      |        0.580 |                 2.127 |         90.7 |
| Development | On       |        3.181 |                10.890 |      2,363.8 |
| Development | Off      |        0.570 |                 2.036 |        105.1 |

These are medians. CPU is accumulated across the child's threads; wall time is
elapsed time. RSS is that process's maximum resident set, not live allocator
bytes or a sum of separate process peaks. OS caches were not flushed.

Loading the populated disk cache takes about 5–6 ms wall time including process
startup and peaks at 22.4 MiB in this probe. That first measurement predates the
cache integrity envelope described below; it must not be used to claim the
envelope has no loading cost.

The module is 2,241,523 bytes, SHA-256
`b597c6c13c0c9199333122f11ff19e46abd34203d4193fe3d18c34837498d46e`. The
executable, all individual samples, machine information, and cache state are
pinned in
[the raw report](measurements/module-compile-baseline-2026-10-23.json). The
probe adds no guest instrumentation. It uses Linux child-process accounting and
saves the example's own module timer separately in subsequent reports.

Reproduce with an equivalent frozen module:

```sh
cargo build --release -p blot-host --example compile_time
python3 scripts/profile-compile.py /path/to/game.wasm --out target/module-profile
```

The output directory must be new. Each cold sample runs in its own process; each
disk-cache population/load pair uses a separate directory. `--inline-budget` is
an experimental override of Wasmtime's expansion threshold in the example; it
does not change production engine defaults. A runtime benefit from inlining
needs its own frame measurements before choosing a different release policy.
Development already defaults to inlining off.

## Cache ownership and validation

The development engine's in-memory cache owns two generations of up to 32 MiB of
key/value bytes each. An entry larger than a generation is declined. A
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
damage and misplaced artifacts; it is not authentication against someone who can
replace both a payload and its checksum.

The
[integrity-envelope repeat](measurements/module-compile-integrity-2026-10-23.json)
keeps cold compilation near the first measurement. It makes the validation cost
visible: cached inlining builds load in 31–34 ms including process startup,
versus 5–6 ms before validation, and peak at 35–41 MiB. Non-inlining builds load
in 12–13 ms and peak at 22.5 MiB. Reading, hashing and safely deserializing an
owned buffer costs more than mapping an unchecked file. This remains far below
the measured cold compilation cost; no zero-overhead claim is made.

The fourteen host tests cover the generation bound, replacement/promotion,
complete-payload byte flips and truncations, wrong keys, old formats, actual
recompilation after corruption, oversized entries, pruning, and existing guest
ownership, traps, probes, snapshots and deadline behavior. A focused writer
regression also checks that a temporary-file collision leaves the other writer’s
file intact.

## Inlining-budget experiment

A
[repeat with the threshold set to 128](measurements/module-compile-budget128-2026-10-23.json)
reduces normal-engine cold compilation to 0.947 s wall, 2.870 s CPU and 778.2
MiB peak RSS. The development engine with inlining explicitly enabled measures
0.790 s wall, 2.450 s CPU and 606.1 MiB. These are again three-process medians.
The much smaller default development path remains available.

The follow-up
[guest-frame report](measurements/guest-runtime-inline-budgets-2026-10-23.json)
contains seven alternating rounds per configuration, 600 frames per process, and
100 warmup frames excluded from the frame distributions. All 21 executions
produce the same complete packet trace and final snapshot. The per-frame timers
exclude compilation; process RSS and CPU include compilation and execution.

| Inline threshold | Median frame ms | p95 frame ms | Process peak RSS MiB |
| ---------------- | --------------: | -----------: | -------------------: |
| Default (2000)   |          0.4600 |       0.6685 |              2,513.6 |
| 128              |          0.5419 |       0.7771 |                778.3 |
| 256              |          0.5074 |       0.7042 |                891.4 |

Each table cell is the median of the seven process summaries. The median of
paired frame-time ratios is 1.136 for threshold 128 and 1.092 for threshold 256;
that differs from dividing independent medians. Both lose enough runtime
performance to reject as production defaults. The shorter three-pair, 180-frame
pilot understated this loss. Tail samples are noisy; raw chronological frame
durations and every process summary are retained.

There is also a packaging constraint: an artifact compiled with threshold 128 is
rejected by a runtime expecting threshold 2000. The pinned Wasmtime 49 API only
exposes this threshold through a compiler setting; the compiler-free player
cannot set it through that API. A regression test verifies rejection of the
mismatch and successful loading by a matching engine. Artifact headers must not
be patched to bypass the compatibility check.

Production settings remain unchanged. The experiment identifies a real
compile-time/memory tradeoff, not an accepted release optimization. Reproduce:

```sh
cargo build --release -p blot-host --example memory_images
python3 scripts/profile-runtime.py /path/to/game.wasm --out target/frame-profile
```

To compare two compiler outputs, add `--baseline /path/to/baseline.wasm` and use
`--budgets default`. Each round alternates module order and requires the same
complete packet trace and final snapshot. Frame percentiles exclude the
configured warm-up; process CPU and RSS still include module compilation.

## Source export and packaged startup

`scripts/profile-startup.py` separates source export from loading an exported
bundle. It runs cold and warm exports with one isolated XDG cache, then cold and
warm packet runs with a second cache. Every round starts with new cache
directories; operating-system file caches are not flushed.

```sh
cargo build --release -p pen
python3 scripts/profile-startup.py \
  --baseline-project /path/to/baseline-project \
  --candidate-project /path/to/candidate-project \
  --out target/startup-profile --runs 3 --frames 120
```

The project configurations select the two compiler executables. Set
`PEN_PLUGINS` to the built plugin directory when testing local plugins. The
profiler pins both project trees, their standard libraries, compilers, plugins,
Pen executable and its own source. Project-root `.pen`, `target` and `.git`
directories are excluded from source pins because they hold generated output or
repository metadata. Cold and warm exports must produce identical Wasm; all
packet runs must preserve the same schema and packet bytes.

Linux process-tree sampling records each observed process's CPU ticks and
resident size every 50 ms by default, alongside separate completed-command
`wait4` accounting. The sampled RSS sum counts shared pages in each process and
can miss short peaks. It is neither unique physical memory nor live allocator
bytes. Timestamped stdout/stderr and cache file counts, sizes and largest
entries remain in the output directory. Headless packet runs cover bundle
loading and game execution; they exclude rendering and GPU asset upload.

### Measured source export and startup, 23 October 2026

Three alternating pairs compare Blot `93bc8d4` with the numeric/ownership
compiler now published as `4091cb2`, using the same rebuilt Pen host. All 24
commands pass, cold/warm Wasm stays identical per compiler, and every schema and
packet witness matches across variants.

| Phase        | Blot variant | Wall seconds | User + system seconds | wait4 maximum RSS, MiB |
| ------------ | ------------ | -----------: | --------------------: | ---------------------: |
| export-cold  | baseline     |        6.912 |                14.798 |                 2548.9 |
| export-cold  | candidate    |        6.749 |                14.584 |                 2472.2 |
| export-warm  | baseline     |        3.547 |                10.920 |                 2541.2 |
| export-warm  | candidate    |        3.321 |                10.441 |                 2470.1 |
| packets-cold | baseline     |        3.430 |                10.810 |                 2539.1 |
| packets-cold | candidate    |        3.472 |                10.906 |                 2465.4 |
| packets-warm | baseline     |        0.108 |                 0.105 |                  105.3 |
| packets-warm | candidate    |        0.102 |                 0.100 |                  105.3 |

These are medians. The wait4 peak is a completed-command maximum; the raw report
separately records sampled process-tree sums and individual processes. For
example, short warm packet launches reach 105.3 MiB by wait4 while 50 ms
sampling observes less. Do not substitute the sampled value for the true peak.

The largest observed resident process is Pen (up to 2,539 MiB); the Blot child
peaks at 319.4 MiB in these samples. This attributes the observed large headless
peak to the host; it does not reproduce all of the historical 5 GiB observation
or cover rendering/GPU upload. Cache directories have five entries for export
and one for packet startup; sizes are stable within each variant. This is not a
long-edit-session cache-growth test.

Warm export still spends about 3.3 seconds and reaches about 2.4 GiB with the
new compiler. Inspection finds that scene discovery opens the game with
`GuestModule::new(..., None)`, bypassing the host's validated disk cache. Warm
packet startup uses that cache and takes about 0.1 seconds. Routing scene
discovery through the same bounded cache is the next separate experiment.

The [raw report](measurements/application-startup-2026-10-23.json.gz) pins
compiler, host, profiler, plugin, source and asset bytes and includes individual
process samples and timestamped output. Its uncompressed SHA-256 is
`2b85da38dcf998abea198f9b3cd4f64eb1b2379fd72bbbadec7ef1beff1a1a47`.
