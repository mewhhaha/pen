# Guest ABI and owned-storage integration

The current migration uses guest ABI **8**, the owned data-stack path. Read
[CONTINUE.md](../CONTINUE.md) for integration status and
[Blot's guest ABI](../../blot2/compiler/guest-abi.md) for the wire contract. The
guest ABI version is distinct from Pen's numeric session protocol **2**.

## Current boundary

- `crates/pen-runtime/src/abi.rs` reads `blot:abi`. It admits versions 2, 3 and
  8; ABI 8 selects data stacks and rejects language-task callback tag 13.
  Retained older admission is implementation awaiting retirement, not a target
  compatibility requirement for this greenfield rewrite.
- Native `blot-host` and the browser guest adapter must agree on ABI 8, storage
  budgets, copied numeric arrays and invocation lifetime. Rebuild the matching
  host/compiler/plugin after boundary changes; stale compiled module caches are
  not evidence for the new ABI.
- Host and worker packets contain independent numeric copies. Guest owner
  descriptors, borrowed frame views and access records must not cross them.
- Frame output is encoded before invocation-local output and payload owners
  close. Session pools remain separately named persistent owners.
- Language tasks are retired in the owned-storage model. Isolated host workers
  and application data queues remain; see
  [services and workers](services-as-tasks.md).

## Sessions, rejection and recovery

The numeric hello uses session version 2. Publication acknowledgement repairs
presentation caches and does not roll simulation back. Explicit Snapshot and
Restore preserve state; rejected frames and traps must not trigger duplicate
simulation advances. A trapped session can require recreation from an explicit
checkpoint because mutations may already have happened.

See [the game contract](game-contract.md),
[packet diagnostics](packet-diagnostics.md) and the
[active storage tasks](../../blot2/tasks/STORAGE_REWRITE.md).

## Qualification

One real Gdev frame passed native Pen on 2026-10-31; the exact candidate and
inspected image are linked from [CONTINUE.md](../CONTINUE.md). Broader gameplay,
native/browser recovery and release qualification remain pending. Small protocol
fixtures qualify their own candidates and do not requalify a later game build.
