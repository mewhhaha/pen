# Determinism

gdev is a pure function of its input: the same starting state and the same
frames of input give the same render packets, bit for bit. That holds
between runs (`just determinism`) and between native and browser engines
(the browser qualification harness). On top of it the runtime records and replays sessions,
travels back in time, and replays the last seconds on a new build after a
hot reload. This page says what the contract is, what keeps runs inside it,
and what breaks it.

## The contract

**A frame's input** is exactly what `encode_frame` sends the game (`pen-protocol::encode_frame`):

- the game time since the last frame (elapsed seconds, as f32);
- the viewport (width and height in pixels);
- the events since the last frame: keys, pointer, wheel, focus lost, console
  commands, stored preferences, live-tunable constants set (Tune: the dev tools'
  `/tune`, `tune=` script steps; what the game holds of them
  is part of its state), assistive technology's actions on the game's
  interface (UiAction: AccessKit's action requests, the browser's
  accessibility layer, `ui=` script steps), assets ready or failed (these
  carry no key, only that one happened), and the answers to the game's
  service requests (ServiceAnswer and ServiceData: Steam, the clipboard,
  the platform, pathfinding; see the README's "Services").

Nothing else reaches the game: no wall clock, no host randomness, and no
file except the states a load reads (below).

**A moment** is what later frames depend on besides their input: the game's
state (the `Snapshot` operation, as a quicksave holds it) and the edit
history the host keeps for undo (the editor's scenes, and which is current),
because a frame can ask to undo into it. The game's state is all of the
game: it keeps nothing a later frame reads outside it (below, "Everything in
the state").

**The contract:** from the same moment, the same frames of input give the
same packets. So a rewind, which restores a moment and runs the recorded
frames from it again, gives every frame the packet it gave the first time. A frame's packet also tells the host to checkpoint, undo,
redo, save or load. The host does these the same way every time, so they
are part of the game's behaviour, not extra input. The exceptions are loads,
which read files that can change, so a recording keeps the states that were
loaded.

## What keeps runs inside it

- **A steady clock.** Headless runs and video recordings advance every frame
  by exactly `1/fps` of game time (60 unless `--fps`). A window uses the real
  clock, so its elapsed times differ from run to run. Recordings capture
  elapsed time, so a replay of a window's session is still exact.
- **Input only through frames.** The player's input, script steps and the
  renderer's asset events reach the game only in a frame's events. A paused
  game keeps what arrives and delivers it with the next frame that runs.
  While recorded frames play, the player's input is dropped, and the
  renderer's asset events wait until play is live again.
- **Settling.** Asset events depend on when loads finish. Headless runs and
  recordings therefore settle before the clock starts: they draw the first
  frame until every asset has loaded and faded in, the veil is gone, and no
  load, shader compile or check is pending. All asset events then arrive in
  frame 0, and none after. (Before the pending check was added, a few events
  could slip into frames 1 to 8 on some runs: the packets stayed the same,
  but the inputs did not.) A window doesn't settle, so its asset events
  arrive when loads finish, and a recording keeps them where they arrived.
- **Services through input.** The session calls a service only for a live
  frame's requests, and its answer goes to the game with the next live
  frame, as input. Recorded frames bring their answers with them: a replay,
  a rewind or a step through recorded frames calls no service, whatever the
  game asks. A frame with requests never overlaps the next (`--overlap`), so
  their answers arrive in the same frame either way. The game's record of
  what it asked is part of its state, ids and all, so a restored moment
  waits for what it waited for and the recorded answers reach the requests
  that asked. A restore drops the answers the session still held for the
  world it replaced: after a load, undo or reload the game asks again, with
  the same ids, for what it still waits on (timed by its fixed steps, a
  second later).
- **Loads from the recording.** A recording keeps every state a frame loaded
  (a quicksave from F9 or `load=FILE`, the editor's scene from Ctrl+R), and a
  replay loads those rather than the files. A replay writes no saves.
- **Overlapped frames** (`--overlap`) change when live input is read: one
  frame earlier, so it shows a frame later. They don't change what a frame
  does with its input. The same script gives the same packets with and
  without overlap.

## Everything in the state

A restore starts a new program from a state, as a quicksave load does, so
anything the game kept only in the running program would start over and
the frames after it would differ from the first time. gdev keeps nothing
there. Each resource a later frame reads is saved, with a codec of its own
(gdev's README, "Saved state"): the game state, and the interface too, its
easing included:

| Saved | Key | |
|---|---|---|
| `input.Input` (version 2) | 3 | now also the last event's pointer motion, claim and pressed keys, which later steps read (the hand sways with the motion) |
| `services.Services` | 12 | the requests made, waiting and answered, and the next id |
| `app.Pause` | 13 | whether the game waits under its menu |
| `preferences.Preferences` | 14 | the player's preferences; after a load the host's stored ones still win, played back as Preference events |
| `ui/tree.Focus` | 15 | the keyboard focus, and the last frame's nodes it moves through |
| `platform.Platform` | 16 | what the platform said, and the requests asking |
| `menu.Menu`, `menu.Flight` | 107, 108 | the pause menu's screen, tab, row and slide-in, and its flight up among the stars |
| `station.Station` | 109 | the station's card, open and how far |
| `build_bar.Building` | 110 | the blueprint roll: open, how far, hovered, risen into view (the bar easing in as play starts), the plan in hand |
| `hover.Hover` | 111 | the gnorp under the pointer and when to look again, the seconds over the station |
| `speech.Speech` | 112 | the bubbles up, the quiet before the next, the dice |
| `wind.Wind` | 113 | the strands blowing, the wait for the next gust, the dice |
| `god_hand.Hand` | 114 | the hand's pose, easing after the pointer |
| `hand.Gesture` | 115 | the time since a strike |
| `robots.Builders` | 116 | who punched on the last step, which frames that run no step draw |
| `trails.Footsteps` | 117 | where each gnorp stood at the last stamp, which the next measures from |
| `routes.Route` | 118 | the route to the plan in hand, and the request asking for it |
| `achievements.Told` | 119 | what was told to Steam, and the stat's request |

All are resources, so states hold them and scenes do not: a scene is the
level's entities. A state from before they were saved restores with each
started afresh (`state_schema` is still 26). Only constants stay unsaved:
the orbit camera's settings and where gnorps may roam, which never change.

Frames after a restore are still marked `restored` in the history and in
recordings, and a replay restores there too: a game that did keep something
outside its state would then replay exactly the packets its run gave,
though its rewinds would not.

## What breaks it

- **Another build.** A recording replays exactly on the build it was made
  with. It doesn't record which build that was. After a hot reload, frames
  run on the new build, so a recording made across a reload can't be
  replayed on either build alone.
- **Another state schema.** A moment or recording from a build with a
  different schema can't be restored, and the runtime says so.
- **A quicksave moved elsewhere.** A state carries what the game knew
  when it was saved, the platform's answers included (its operating system,
  locale and device): a quicksave loaded on another machine shows the one it
  was saved on in Settings' About page.
- **The real clock, unrecorded.** Two windowed sessions with the same key
  presses differ, because their elapsed times differ. Record one with
  `--record-input` to repeat it.
- **Pixels aren't packets.** The renderer keeps state of its own:
  - assets that are still loading, in a window;
  - field patches, which the game sends as they change and in full every so
    often. After a rewind, fields can show newer values until those arrive.
  - the GPU and driver: other hardware can differ slightly, which the
    golden-image threshold allows.

  Same packets mean the same picture once assets have settled, on one
  machine.

## Tools

| | |
|---|---|
| `--record-input FILE` | Writes the session's moment at the start, then each frame's input, flags and loaded states (`.penrec`, see `crates/pen/src/recording.rs`). Frames are written as they run, and a rewind cuts the file back. |
| `--replay FILE` | Restores the recording's moment and plays its frames in place of the player's input and clock. Headless, the run ends with the recording. In a window it pauses there, and Ctrl+Shift+P then plays on, live. The window or image takes the recording's size unless `--size` says otherwise. |
| `--trace FILE` | One line per frame that runs: its number, then the FNV-1a hashes of its input and packet. Frames a rewind runs unseen are included, so the last line for each frame describes the final timeline. |
| `--history` | Keeps the last 10 s (always on in `pen dev`, in replays, and in scripts that rewind or replay). |
| `--overlap` | Overlapped frames (below). |

**Time travel**, in a window:

| Keys | Action |
|---|---|
| Ctrl+Shift+P | Pause or play. A paused window shows its last frame again without drawing it again (it is redrawn when the window's size changes or assets load); its sounds do not play again, and the voices it keeps go on as they were until frames run. |
| Ctrl+Shift+period | Step one frame (it repeats while held). |
| Ctrl+Shift+comma | Step back one frame. |
| Ctrl+Shift+R | Replay the last 5 s. |

The game never sees these keys. In scripts the same actions are `pause`,
`resume`, `step[=N]` (N frames, one per script frame), `rewind=N` and
`replay=SECONDS`. For the dev tools, `Session` has `pause`, `resume`,
`step_frames(n)`, `rewind_to(frame)`, `replay_recent(seconds)` and `timeline()`
(position, the recorded end, the earliest frame reachable, seconds kept,
paused, replaying). `rewind_to` and `replay_recent` take effect on the next
step.

A history keeps each frame's input and a moment every 30 frames. A gdev
moment is about 2 to 6 KB, more while a pathfinding request waits for its
answer (it carries its grid, about 20 KB), and takes 0.05 to 1 ms to
snapshot (a dozen gnorps at work is the 1 ms). A rewind restores the
nearest moment before its target and runs up to 30 frames unseen: 7 to 15
ms. After a rewind, stepping runs the recorded frames again.
Playing on drops them and continues live from there.

The dev tools (README, "The dev tools") put these behind their console's
commands, which can also start and stop an input recording in the
middle of a session: the game is then restored from its own state where the
recording starts, as a replay of it is, so the replay gives the packets the
session did, rewinds within it included.

**Edit and replay.** In `pen dev`, pause when something goes wrong
(Ctrl+Shift+P), so the moment stays in the 10 s history while you edit.
Save the Blot fix. When the reload is in, press Ctrl+Shift+R: the last 5 s
of input play again on the new build, starting from the state before them,
and then the game plays on live. Moments from before the reload are still
valid starting points. The ones after the replay's start are dropped and
taken again from the new build as its frames run.

## Checks

- `just determinism`:
  - runs `tests/determinism.steps` twice, and once with `--overlap`, and
    requires the same packets;
  - replays the first run's recording, with the quicksave file it loaded
    deleted, and requires the same inputs and packets;
  - runs it again followed by `tests/rewind.steps`, which rewinds into the
    interface `tests/determinism.steps` ends in, to moments where the
    keyboard focus goes to the build bar and it rises, the station's card
    opens, and the pause menu flies up, shows its second row and flies
    down, stepping through recorded frames after each; then into play and
    over the undo and loads, plays on live from the past and replays a
    second. Every frame that runs again with the input it had (the trace
    shows each run) must give the packet it gave the first time, and that
    run's recording must replay the packets of its final timeline.
- `cargo test -p pen contract` checks the same things on
  `dist/gdev` without a GPU, and more rewinds:
  - the same input gives the same packets, and a recording replays them;
  - rewinds into play, over the undo and loads, give every frame they run
    again the packet and state it gave the first time;
  - so do rewinds to each moment of a run through the interface (the build
    bar easing in, the focus on it, the station's card opening, open and
    closing, the hand swaying after the pointer, the pause menu flying up,
    its rows, its settings and a preference changed, and it flying down),
    and to every third moment of a run with a crowd of gnorps (hauling,
    speaking, building, a gust of wind), both with a host whose services
    answer 40 frames late, so requests are in flight over moments;
  - a recording with rewinds replays exactly;
  - after a hot reload, replaying the last seconds gives the same packets.

  A frame that differs fails with the first field of its packet that does,
  e.g. `insets[5]: Inset > draws: > Draw > extra: Some > 801.29816` and its
  value again: the way to find what a game keeps outside its state.

## Overlapped frames

Normally each frame runs in order: read input, the guest computes the frame,
then it is decoded, drawn and presented. With `--overlap`, the guest
computes frame N+1 while the main thread draws and presents frame N. Overlap
turns itself off around anything that needs the guest idle or replaces its
state (reloads, loads, undo and redo, saves, replays and time travel) and
comes back after.

Measured on Linux (Wayland, Vulkan, RTX 2070 Super) with
`PEN_PROFILE_FRAMES=600` and gdev in play. Medians, in ms:

| Run | Guest | Waited for guest | Render + present | Frame interval |
|---|---|---|---|---|
| Window 1440x960, `immediate` | 0.71 to 0.75 | 0.73 to 0.77 | 3.59 to 3.73 | 4.53 to 4.60 |
| the same, `--overlap` | 0.45 to 0.77 | 0.00 | 4.43 to 4.51 | 4.52 to 4.62 |
| Window 640x427, `immediate` | 0.67 to 0.74 | 0.69 to 0.76 | 1.78 to 1.82 | 2.59 to 2.64 |
| the same, `--overlap` | 0.62 to 0.66 | 0.00 | 2.52 to 2.54 | 2.61 to 2.62 |
| Headless 1440x960 | 0.66 to 0.73 | 0.67 to 0.76 | 2.09 to 2.16 | 2.95 to 3.20 |
| the same, `--overlap` | 0.66 to 0.74 | 0.00 | 3.72 to 3.74 | 3.80 |

Overlap hides the guest entirely: the host never waits for it. But today's
gdev spends only about 0.7 ms in the guest per frame. Render and present
grow by about as much while the guest runs alongside them, so the frame
interval doesn't improve, and headless it gets worse. Overlap stays opt-in
for that reason, and because it adds a frame of input latency. It may pay
off when the guest step is several milliseconds (busier scenes), and costs
nothing to try.
