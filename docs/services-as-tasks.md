# Services as Blot tasks

**Status: blocked on blot2's task ABI.** pen reads a task module's manifest and
refuses it with the reason; it cannot host one natively. This page says why,
what the design must keep, and exactly what pen needs from blot2.

## Today

A game reaches a service (Steam, the clipboard, pathfinding...) through its
render packet: a request with an id, a service and an operation, answered once
as `ServiceAnswer` and `ServiceData` input events in a later frame
(`pen_runtime::services`). The game keeps the ids of requests in flight in its
state and looks for their answers in each frame's events. Because answers are
input, recordings hold them, and replays, rewinds and reloads repeat exactly.

## The goal

Game code that reads as it means:

```blot
// In a concurrent task: the path arrives some frames later, and the task
// resumes with it; other tasks and the frame loop run meanwhile.
use path <- pathfinding.route grid from to
```

Blot has the pieces: structured tasks (`spawn`, `join`, `cancel`, groups) and
`concurrent` callbacks, in which tasks suspend side by side (guest-abi.md,
"Structured task extension"; the manifest's callback tag 13 beside a
`blot:tasks` section).

## What the design must keep

1. **Determinism.** An answer may reach a task only at a frame boundary, in
   an order fixed by the requests (not by when services finish), and as input
   a recording keeps. A replay settles the same calls with the same answers in
   the same frames and calls no service.
2. **Time travel and reload.** pen restores a moment by starting a new program
   from the game's snapshot, and moves the state into a new build the same
   way. A suspended task is a Wasm stack, not state: no snapshot holds it. So
   either a task awaiting a service is part of what the game saves (its
   request, re-issued on restore by the game, as gdev re-issues pathfinding
   today) or tasks may not wait across frames. The language should make the
   choice explicit; silently losing in-flight tasks on every rewind would make
   rewinds diverge.
3. **Both hosts.** The browser keeps one suspended Wasm stack per JSPI
   `promising` call, so concurrent callbacks work there. wasmtime does not: a
   core module's call (`Func::call_async`) holds its store for the whole call,
   so one instance has at most one suspended stack. Concurrent calls into one
   store exist only for components (`Store::run_concurrent`). Native pen today
   runs `main` on its own thread and blocks it in the host call: one stack.

## What pen needs from blot2

1. **A lowering a single-stack host can run.** Either
   - tasks scheduled inside the guest: a concurrent callback records its
     request and yields to the guest's own scheduler (Wasm stack switching,
     which wasmtime has behind `Config::wasm_stack_switching` on x86-64 Linux,
     or state machines), and the host sees one call per frame; or
   - frame-gated concurrent callbacks: the requests a frame's tasks made
     leave in that frame's response, their answers arrive in a later request,
     and the compiler resumes the waiting tasks from it. That is pen's service
     protocol with task syntax over it, and needs no new host machinery.
2. **A host-neutral specification of `blot:task/1`** (the twelve imports, the
   task context exports, resource journals moving between tasks) with
   conformance modules and expected traces. Today `compiler/task_guest.ts`,
   `task_runtime.ts` and `resource_journal.ts` are the specification.
3. **What a snapshot keeps of a task** (point 2 above).

## What pen has built

- `pen_runtime::abi` reads the task extension: the `blot:tasks` section
  (version 2, its descriptor count, at most 4096) and concurrent callbacks
  (tag 13), refusing them without ABI 3's ownership or without the section, as
  the reference adapter does. `GuestAbi::check_hostable` refuses a task module
  in both hosts with the reason and this page, before anything runs:
  `unsupported_abi: the module uses Blot's structured tasks (blot:tasks, 3
  descriptors; concurrent callbacks for main), which pen does not host yet`.
  Tested by `task_modules_are_read_and_refused`.
- The determinism half already exists: services answer as recorded input,
  in request order, at frame boundaries, and the same protocol would carry a
  frame-gated lowering unchanged.

## What is left

Hosting tasks natively once blot2 chooses a lowering: with guest-side
scheduling, nothing beyond the manifest (pen runs one call per frame as it
does); with host-side suspension, a wasmtime embedding that keeps several
stacks per instance, which wasmtime's core API does not offer.
