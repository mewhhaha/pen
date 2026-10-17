# Instant rewind: memory images against today's moments

Time travel today keeps a moment every 30 frames: the game's own snapshot (a
Snapshot request; gdev answers with about 3 KB of lanes) and the edit history.
A rewind starts a new program from the nearest earlier moment (instantiate,
run `main` to its handshake, Restore, a frame to prove it works) and runs the
recorded frames from there to the one asked for, unseen. pen-runtime's
`memory_image` and blot-host's `Guest::capture`/`restore` are the other way:
keep the guest's linear memory itself, page by page, and write it back.

## What is built

- `pen_runtime::memory_image::MemoryImage`: a memory as 64 KiB pages, sharing
  every page unchanged since the image before it (so an image a frame costs
  the pages the frame wrote), and restoring by writing back only the pages
  that differ from the memory as it is (zeroing what lies past the image,
  since Wasm memory does not shrink). Platform independent, for both hosts.
- `blot_host::Guest::capture`/`restore`: an instance's image between calls,
  its memory and its mutable globals, put back exactly. A module that keeps a
  mutable global it does not export is refused (`1 of its 2 mutable globals
  are not exported, out of the host's reach`): that state would not come
  back. Tested with hand-built modules (`images_put_idle_guests_back`).
- `cargo run --release -p blot-host --example memory_images
  dist/gdev/game.wasm`: plays gdev's frames on the inlining engine, imaging
  its memory at every frame boundary from inside its host call, beside the
  snapshots the timeline asks for, and times both ways of coming back.

## Why it cannot rewind today's games

A game is a long-lived `main(host)` suspended in its host call between
frames. Part of its state is that suspended Wasm stack (the loop's carry in
Wasm locals), which no wasmtime or browser API lets a host read or restore.
Putting back an older memory under a newer stack leaves the locals pointing at
whatever the old memory holds there. `Guest::capture` can only be called
between calls, so the type system keeps it to sound cases; gdev's 19 of 29
mutable globals that are not exported would refuse it too.

For images to rewind a game, the guest ABI must give the host the whole state
at a frame boundary (docs/guest-abi-next.md): a step-shaped entry
(`frame: Array F32 -> Array F32` with the world in a module root, called once
a frame, as mods already are) or a promise that `main` keeps nothing in Wasm
locals across its host call; and every mutable global exported.

## Measured on gdev

2026-10-16, gdev's `dist/gdev/game.wasm`, 600 frames (play from frame 60),
three runs. The machine was shared with four other builds (load average
about 35 on 16 threads): `pen bench` gave 4.3 ms a frame against the 0.32 ms
measured unloaded, so read the times as ratios.

| | Memory images | Today's moments |
|---|---|---|
| Guest memory | 0.9 MB (14 pages) | |
| Kept per frame | 11.5 of 14 pages change: 735 KB a frame (310 KB even in 256-byte blocks) | nothing between moments |
| Kept per moment (30 frames) | 0.9 MB | 720 lanes, 2.8 KB |
| Taking it | 0.11 ms (median) | a Snapshot request: 0.63 ms (median) |
| Coming back | 12 pages written in 0.075 ms | instantiate 0.05 ms, handshake at 0.11–0.15 ms, restored at 1.3–3.1 ms, first frame at 5–12 ms |
| Then | the frames since the image | up to 29 frames run again, 4.5–7.5 ms each under this load |

gdev's heap is small and its allocator touches most of it every frame (an
arena reset per frame and block reuse across the heap): a third of its bytes
change even counted in 256-byte blocks. So an image every frame, which is
what "instant" needs, costs 300–700 KB a frame (180–440 MB for the 10 s
history), against 2.8 KB for a game snapshot. Images only at moments save
the restore (about 0.1 ms against 1–3 ms and a proving frame) but not the
frames run again, which dominate.

## What would make rewinds instant

1. **A moment every frame, with today's snapshots.** 2.8 KB a frame is 1.7 MB
   for the 10 s history, and a rewind is then one restore with no frames run
   again. Its cost is a Snapshot request a frame (here 0.63 ms against a
   4.5 ms frame under load, about a seventh; it scales with the game's
   state). The timeline could keep one a frame only while paused or
   scrubbing, and every 30 frames otherwise. This needs no ABI change.
2. **Images, once the ABI allows them** (above): for a game whose memory
   changes little per frame. gdev's does not, so its snapshots stay cheaper.
3. **A warm spare program.** Each restore instantiates and runs a program to
   its handshake; keeping one ready would cut that part, which is small
   (0.1–0.2 ms here) beside the Restore and the proving frame.
