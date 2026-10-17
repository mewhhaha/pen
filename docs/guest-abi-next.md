# The guest ABI after 3: what pen hosts, and what it needs

pen reads a guest's `blot:abi` manifest in `pen_runtime::abi` and runs it in
blot-host (native, wasmtime) and pen-web (`guest.js`, JSPI). This page says
which of blot2's guest ABI pen hosts, how it tears guests down, what blot2
already defines beyond it, and what pen asks of the next versions. The ABI
itself is blot2's `compiler/guest-abi.md`, ABI 3's ownership hooks
`zig-native/SHARED_OWNERSHIP.md`, and the reference adapter
`compiler/guest.ts`.

## What pen hosts

| | Native (blot-host) | Browser (guest.js) |
|---|---|---|
| ABI 2: scalars, copied numeric arrays, one host callback per entry | yes | `main(host)` only |
| ABI 3: `blot:ownership_enter` before a call, `blot:ownership_leave(0)` after it returns or `(1)` after it fails, then `blot:reset` | yes | yes |
| ABI 3 teardown: `blot:ownership_leave(2)` when idle, then `blot:reset` | `Guest::dispose`; the game's program after `main` returns | `close()`, after `main` settles |
| Ownership counters (`blot:rc_*`, `blot:cycle_*` globals) | `Guest::ownership` | no |
| Debug names (`name`), source maps (`sourceMappingURL`) | yes (docs/source-maps.md) | yes |
| Structured tasks (`blot:tasks`, tag 13 concurrent callbacks) | read, refused with the reason | read, refused |
| ABI 4 to 7 | refused (unknown version) | refused |

**Teardown.** Blot's adapter disposes of an idle instance with
`ownership_leave(2)`, which releases the module's roots and drains
destruction and cycle work, and resets the arena. pen does the same wherever
an instance outlives its last call: `Guest::dispose` (which then refuses
calls and reads, as `disposed_guest`, while the store stays for its counters
and memory to be read), the game's program thread after `main` returns, and
`guest.js`'s `close`, whose JavaScript instance lives on until the garbage
collector takes it. A failed call leaves with outcome 1 first, so teardown
always starts idle, as the specification requires. Tests: blot-host's
`abi_3_calls_run_between_entering_and_leaving_ownership` (hand-built hooks
that log each outcome, read after teardown through `Guest::ownership`) and
`crates/pen-web/tests/guest.mjs` (a hand-built ABI 3 game under node's
JSPI: `124` after a clean stop, `134` after a trap).

## What blot2 defines beyond 3

`compiler/guest.ts` already reads manifest versions 4 to 7 and several
sections pen does not:

| Feature | What it adds | What pen would do |
|---|---|---|
| ABI 4: resource journals | `blot:resource/1` `reserve`, `register`, `release` imports; per-resource cleanup and root-drop exports | a typed journal per instance, drained in reverse at `finish`, before roots retire |
| ABI 5: resource transfer | `blot:resource/2` `register_owned`, `retire`, `drain` | journals that move between calls |
| ABI 6: public owners | owner results (tags 8, 9), `blot:owner_release`, `blot:owner_token`, `blot:owner_call/NAME`; instance-bound handles | host handles for unique or shared guest values, released on disposal |
| ABI 7: cleanup data | tags 11, 12 and `blot:cleanup_data*` imports with bounded schemas | structured cleanup callbacks |
| `blot:multiplicity` | functions callable once | refuse a second call |
| `blot:tasks` v2, tag 13 | structured tasks and concurrent callbacks | see docs/services-as-tasks.md: blocked natively |

None of these is needed by a game today: gdev is ABI 3 with one callback.
pen adds them as games need them, through the same manifest reader, so both
hosts refuse what they cannot run before it starts.

## What pen asks of ABI 4 and later

1. **Debug information** in the module: the standard `name` section (blot2's
   pending patch) and an embedded source map in the format of
   docs/source-maps.md, so traps name Blot files and lines.
2. **Features named, not implied by the version number.** A host reading
   version N cannot tell which of N's features a module uses unless the
   module says so; blot2 already does for tasks and multiplicity (sections)
   and owners (flags). Keep that for every addition, and let a module that
   uses none of a version's new features keep the older version number, as
   blot2 does for resources, so pen can run newer compilers' ordinary games.
3. **A conformance corpus.** The manifest has test vectors; the rest of the
   ABI is specified by `guest.ts` and its TypeScript tests. Small modules per
   feature with the host trace each must produce (hooks called, values
   returned, traps) would let pen check blot-host and guest.js against the
   reference adapter.
4. **Teardown, said.** Whether `ownership_leave(2)` is valid after a call
   that trapped and left with 1 (pen assumes so, as `guest.ts` does), whether
   a second teardown is an error, and what the counters must read after it
   (pen expects `rc_roots` and `rc_leases` at zero).
5. **State the host can keep.** Time travel restores a game from its own
   snapshot in a new instance. Rewinding without that needs the guest's state
   to be all in memory and globals at a frame boundary: a step-shaped entry
   (`frame: Array F32 -> Array F32`, the world in a module root) or a
   guarantee that `main` keeps nothing in Wasm locals across its host call,
   and every mutable global exported (docs/instant-rewind.md).
6. **Tasks a single-stack host can run** (docs/services-as-tasks.md).
