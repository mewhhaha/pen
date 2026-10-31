# Services, data queues and copied-buffer workers

This greenfield rewrite retires language tasks, including their scope,
spawn/join/spark/receive/cancel APIs. Do not add a compatibility facade. Host
workers and saved application data queues remain separate mechanisms. See
[CONTINUE.md](../CONTINUE.md) and the [guest ABI](guest-abi-next.md).

## Frame-gated services

Service requests carry IDs; answers arrive as recorded input events at frame
boundaries. Replay uses those recorded answers. Save explicit request/result and
continuation data; a suspended Wasm stack is not saved state. Restore and replay
must preserve ordering and avoid duplicate simulation advances.

Worker inputs, contexts and results are copied numeric packets. They must not
contain guest pointers, borrowed views or live ECS access records. Applications
own worker lifetime, revision identities and deterministic result publication.
Neither a data queue nor a worker API makes application schedules parallel by
itself. Qualify these boundaries after the real Gdev frame works.

## Independent numeric workers

`blot_host::workers::Workers` is a separate opt-in CPU experiment for
independent value entries. It shares compiled code but creates a fresh Wasm
instance per job, copies every input/result, and returns answers in submission
order. Limits are eight workers, 32 jobs per batch, 16,384 input/output lanes,
and 64 MiB of linear memory per instance. Worker stacks include the configured
guest stack allowance. Jobs cannot receive callback capabilities. Execution
deadlines depend on the supplied engine; this API does not provide cooperative
cancellation of a running job. It is not connected to game scheduling or the
structured-task ABI.

`Workers::set_tracing(true)` records submission, execution start and completion
for the latest accepted batch, including worker index and success/failure.
`timings()` returns at most 32 records in input order. Tracing is off by
default; disabling it clears the history and avoids per-job clock reads. These
host times are diagnostics, never simulation inputs or saved continuations. The
`compare_workers` example captures a separate traced batch after its performance
samples, so tracing overhead does not enter the paired timing comparison.

## Prepared numeric worker sessions

`blot_host::worker_sessions::Workers` retains isolated guests inside a callback
service entry. Construct it with an engine, module, entry and 1–8 workers;
`configure(revision, context)` supplies finite F32 context once, then
`run_prepared(inputs)` submits up to 32 jobs without resending it. Context,
individual input and output are limited to 16,384 lanes; each guest has a 64 MiB
memory limit. The application owns the pool and immutable context. The service
ABI is `(Array F32 -> Array F32 ! { Foreign }) -> Unit`.

A service first calls its host with `[-4670546]`. The host returns
`[1, context…]` to prepare, `[2, input…]` to run a job, or `[0]` to stop.
Responses are `[1]` for prepared context and `[2, output…]` for a job. Reusing
the current revision requires identical context bits, including signed zeros. A
new revision replaces it; old revision numbers are not remembered. `reset`
clears configuration; the next configured batch prepares workers again. Old
worker data stays retained until replaced or closed. No guest pointers or live
ECS worlds cross threads.

Results retain submission order. Traps, malformed responses and early guest exit
fail the affected job, discard that guest, and recreate it for subsequent jobs.
Direct Rust batches return per-job Results. `call(Value)` is an optional client
callback using -4670547 configure/reset/run commands and the existing -4670545
batch framing; it rejects a failed batch as a whole. Ink's optional
`pen/worker_session` adapter supplies both ends of this protocol; Pen has no Ink
or game dependency. Nothing automatically makes application schedules parallel.

`stats` reports starts, prepared contexts, copied context lanes, successful
jobs, and each worker's last observed committed guest memory. `close` joins
workers and is idempotent; Drop also closes. Service code must terminate its
jobs and honor stop: this API has no per-job timeout. Context setup, scheduling
overhead, serialization and extra worker memory belong in application
measurements. The standalone `worker_sessions` host tests cover ordering,
validation, traps, early exits, restart, reset and stable memory over repeated
batches.

The retired language-task conformance fixture and adapter-admission tests have
been removed. `owned_storage` retains the useful numeric results, copied-array
round trips, bounds/trap recovery, 20 reuse cycles, 1,000,000-element capacity
probe and 128/5,000-iteration workloads. Five thousand ordinary iterations now
complete; the old task scheduler's admission limit no longer applies. The
1 MiB sandbox still rejects a module whose planned storage exceeds it. Copied
worker tests retain 32-job batches, submission order, tracing and lane limits.
