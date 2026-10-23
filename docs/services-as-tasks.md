# Services as Blot tasks

Pen hosts **synchronous structured tasks natively** and **cooperative structured
tasks in the browser**. These paths preserve Blot's task contexts and ownership;
neither path adds CPU parallelism or persisted Wasm stacks.

## Supported boundary

- Native `blot-host` admits ABI 3 task modules without concurrent callbacks.
  It validates the version-2 task manifest, every import/export signature and
  descriptor, executes children in spawn order at joins/drains/scope exits,
  cancels unstarted work, releases unused results and drains failures. Handles
  are bounded, monotonic identities. Array results move through the compiler's
  adapters. Newer resource ABIs remain outside native admission.
- The browser uses Blot's bundled upstream adapter for task modules. JSPI runs
  their stacks cooperatively. Without JSPI, the existing isolated worker uses
  the synchronous adapter and the SharedArrayBuffer frame exchange.
- The game's frame callback remains serial. `main` with a concurrent callback
  is refused by the browser program boundary; services need a separate recorded
  request/answer channel. Native admission refuses all concurrent callbacks.
- Memory images refuse task instances: their host contexts are not part of
  linear memory. Game save/load still requires ordinary saved data and a fresh
  instance. A suspended Wasm stack must never masquerade as saved state.

The reference adapter is generated from `crates/pen-web/structured-source.ts`
by `deno run -A scripts/bundle-structured.ts`; `--check` detects drift. It ships
as `structured.js`, including Blot's third-party notice.

`blot_host::workers::Workers` is a separate opt-in CPU experiment for independent
value entries. It shares compiled code but creates a fresh Wasm instance per job,
copies every input/result, and returns answers in submission order. Limits are
eight workers, 32 jobs per batch, 16,384 input/output lanes, and 64 MiB of linear
memory per instance. Worker stacks include the configured guest stack allowance.
Jobs cannot receive callback capabilities. Execution deadlines depend on the
supplied engine; this API does not provide cooperative cancellation of a running
job. It is not connected to game scheduling or the structured-task ABI.

Ink's opt-in `task_queue` now implements the explicit cross-frame state machine:
world epochs, ordered publication, saved request/result/continuation values,
cancellation and bounded logical timelines. `services/typed` validates domain
answers over the existing service protocol. These mechanisms do not change Pen's
native admission rules or make persisted Wasm stacks possible.

## Evidence

`cargo test -p blot-host --test structured_tasks` compiles the actual Blot
fixture (select its compiler with `BLOTC`). It exercises collection, nested
scopes, cancellation before execution, unjoined results, owned array results,
repeated failure/reuse and zero live ownership counters after calls/teardown.
The existing native host/ABI tests and browser ownership tests also exercise
legacy modules. The gdev modernization harness exercises task-bearing game
exchanges through JSPI and the isolated worker, including traps and restart.
This is an initial conformance set, not exhaustive malformed-module/OOM proof.

## Cross-frame services remain separate

The existing service protocol submits requests with IDs and returns answers as
recorded input events at frame boundaries. Replay calls no service. A saved
request can be re-issued after restore; a suspended stack cannot.

For task-shaped service calls across frames, keep these requirements:

1. Deliver answers at a recorded frame boundary in deterministic request order.
2. Save an explicit continuation/request record, or finish tasks within a frame.
3. Close the old world's task lifetime before replacing/reloading that world.
4. Keep worker inputs/results copied or serialized; never share guest pointers.

The native embedding still cannot hold several suspended core-Wasm calls into
one Wasmtime store. The next architectural experiment is explicit, saved state
machines over frame-gated service answers. A compiler lowering or a qualified
stack-switching implementation would be required for general source-level
suspension. The synchronous host does not imply that this problem is solved.
