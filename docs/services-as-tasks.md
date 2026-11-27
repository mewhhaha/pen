# Services, data queues, `KernelPool` and isolated `Processes`

This greenfield rewrite retires language tasks and the legacy `squid_host::workers` /
`squid_host::worker_sessions` experiments. Host kernels (`#kernel` + `KernelPool`),
isolated stateful processes (`#process` + `Processes`), and frame-gated Pen services
remain clean, separate mechanisms. See `squid-lang/tasks/CONCURRENCY_VNEXT.md`.

## Frame-gated services

Service requests carry IDs; answers arrive as recorded input events at frame
boundaries. Replay uses those recorded answers. Save explicit request/result and
continuation data; a suspended Wasm stack is not saved state. Restore and replay
must preserve ordering and avoid duplicate simulation advances.

Worker inputs, contexts and results are copied numeric packets (`FlatNumeric` or
`#[U32]`/`#[F32]`). They never contain guest pointers, borrowed views or live ECS
access records. Applications own revision identities and deterministic result publication.

## Warm work-stealing `KernelPool` (`squid_host::kernels`)

`squid_host::kernels::KernelPool` executes ABI 8 `#kernel` modules declared in
the `squid:kernels` v1 custom section. Workers are warm, persistent Wasmtime `Store`s
sharing compiled code and reusing prepared context snapshots via `set_context` /
`clear_context`.

- Fixed-output batches (`run_fixed`) write disjointly into a single `DisjointStorage`
  buffer (`1×` exact peak, zero-copy `Box::into_vec`).
- Variable-output batches (`run_variable_chunks` at `1×`, `run_variable` at `<= 2×`)
  use pre-copy Wasm-memory size inspection, deadlock-free frontier eviction, and
  concurrent bisection on the non-trapping `[0xffffffff]` overflow sentinel.
- Deadlines require an epoch-interrupted engine (`bounded_engine`, `dev_engine`, or
  packages built with `pen build --worker-deadlines` shipping `workers.cwasm` +
  `workers.meta` and verified by compiler-free `pen check-workers`).

## Deterministic isolated `Processes` (`squid_host::processes`)

`squid_host::processes::Processes` schedules ABI 8 `#process` actors declared in
the `squid:processes` v1 custom section. Each admitted endpoint owns an isolated
Wasmtime `Store` with persistent staged state across turns, bounded by explicit
`ProcessPolicy` work, memory, mailbox-slot, `mailbox_bytes`, `max_reply_bytes`,
optional Wasmtime fuel (`FuelPolicy`), and optional wall-clock `deadline` limits.
`advance_all_parallel` executes independent endpoints concurrently in serial-equivalent
waves while enforcing Overrun-first precedence and scheduler-wide fencing on overrun.
