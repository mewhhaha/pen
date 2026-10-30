# Net sessions and spectating

Several players, each on their own machine, play one game: a net session. pen
sends nothing but their input. The game is a pure function of its state and
input ([determinism.md](determinism.md)), so peers that start from the same
moment and give every frame the same input compute the same frames, bit for bit,
each on its own. A game declares how many players it takes; that is all the
netcode it has.

The same property streams a session to viewers: a moment, then every frame's
input, as a recording holds them. A viewer plays them as a replay does.

## The game's side

A game declares its players in `pen.json` (and `pen build` keeps it in the
package's `game.json`):

```json
{ "game": { "players": 2 } }
```

A game that declares none takes one player, and pen refuses to start a net
session for it. In a session, every frame's input lists every player in order,
each behind a `Player` event (`protocol/schema.json`, tag 26; Ink's
`#events.Player { index }`), followed by what that player's peer gave it:

```
Player 0, KeyDown "A", PointerMove 480,330, Player 1, PointerDown 0 470,151
```

Outside a session there are no Player events and every event is player 0's, so a
single-player game runs unchanged. gdev declares two players and does not read
the markers yet: both players' hands act on one world.

The input every peer's game hears is the same:

- **Each player's own events**: keys, pointer, wheel, focus lost, UI actions,
  console commands, and whatever other events the player makes (gamepads when
  they land). Positions are moved from the peer's window into the session's
  viewport.
- **The session host's events**: player 0's peer is the host. Its renderer's
  asset events, its stored preferences, tunables and its services' answers
  travel as player 0's. Other peers keep theirs: their services still run (a
  Steam achievement unlocks for each player), but the game hears the host's
  answers.
- **The host's terms**: the viewport (its window's size when the session
  starts), the frame rate (every frame is 1/60 s of game time) and the input
  delay and window below. A peer whose window has another size draws the same
  frame stretched to it.

## How a session runs

```sh
# Two peers on one machine; on a LAN, each machine's address.
pen run --net 0:127.0.0.1:7000,127.0.0.1:7001
pen run --net 1:127.0.0.1:7000,127.0.0.1:7001
```

`--net PLAYER:ADDRESS,...` lists every player's UDP address in order and says
which player this peer is (from 0); a packaged game takes it too. The peers say
hello until each has heard every other and they agree on the game's build (a
hash of `game.wasm`) and the starting moment. A peer that started from another
moment (a quicksave, frames it played alone) takes the host's, sent in chunks,
and restores it before frame 0.

`pen_runtime::net` (`crates/pen-runtime/src/net/`) is the session, platform
independent; native pen carries it over UDP (`crates/pen/src/net.rs`):

- **Input delay.** A player's input plays `delay` frames after it was given (2
  by default). Input that arrives within the delay needs no prediction.
- **Prediction.** Past it a peer does not wait: it predicts that the players it
  has not heard from did nothing new. Events leave their effect in the game's
  state (a key held, the pointer where it was), so "no events" is the game's own
  "the same as before".
- **Rollback.** Before each frame that runs on a prediction the timeline keeps a
  moment (`Timeline::moments_from`). When a player's input arrives and a frame
  ran without events that player had, the frames from that one get their real
  input (`Timeline::amend`) and run again from its moment, unseen
  (`Timeline::rewind`, as time travel does). Sounds of the frames that ran wrong
  have played; the frame drawn next is right. A rollback restores once: the
  frames it runs again lose the marks of earlier rollbacks' restores, which
  peers make at frames of their own.
- **Window.** A peer runs at most `window` frames (8) past the last frame every
  player's input is known for, then waits for the others, showing the last
  frame. That bounds a rollback to 8 frames.
- **The wire.** Each input message repeats every frame of the player's input the
  receiver has not acknowledged (up to 64 frames, 1200 bytes), so a lost or late
  datagram costs nothing; messages carry the sender's frame, a clock stamp for
  the round trip, and a checked frame's hash.
- **Desyncs.** Every 30 frames the peers compare the hash of a frame whose every
  input was known. A difference stops the session with "the game parted ways
  with player N's at frame F": one game kept something outside its state, or ran
  differently.

What would change one peer's game alone is off in a session: time travel (pause,
step, rewind, replay), quicksave and scene loads (they read this peer's files;
saves still write them), hot reloads and overlapped frames. `--record-input`
records the merged input, and the recording replays the session exactly on one
machine. A run with `--frames N` ends once every peer holds the others' input
for those frames, so all end on the same frame, and says how the session went:

```
net: 300 frames, 165 on predictions; 4 rollbacks ran 9 frames again (at most 3); waited 163 times; 10 frames compared with the peers, 153 ms round trip
```

## What a rollback costs

`cargo test --release -p pen rollback_costs -- --ignored --nocapture` plays gdev
with a crowd (12 gnorps hired, hauling and building, the CROWD script of the
contract tests) and rolls back from ten heads: a restore of the moment N frames
back (a new program from it, and its proving frame) and the N frames run again;
and, as today's history keeps a moment every 30 frames, from the latest multiple
of 30 at or before it.

Measured 2026-10-17 on the development machine
([earlier performance results](reference.md#performance)) while other builds ran
(load average 11 to 22 on 16 threads): a frame of this busy scene took 5.6 ms of
guest time, against about 0.3 ms for gdev's default scene on an idle machine, so
read the times as multiples of a frame. Moments were up to 49 KB (a route
request in flight) and took 8.1 ms to snapshot (median). Medians of ten:

| N  | Restore | N frames | Rollback, a moment every frame | Rollback, a moment every 30 frames |
| -- | ------- | -------- | ------------------------------ | ---------------------------------- |
| 1  | 10.4 ms | 5.4 ms   | 15.8 ms                        | 100.2 ms                           |
| 2  | 10.1 ms | 11.4 ms  | 21.4 ms                        | 103.8 ms                           |
| 4  | 9.7 ms  | 23.8 ms  | 33.1 ms                        | 138.8 ms                           |
| 8  | 10.4 ms | 47.1 ms  | 59.1 ms                        | 136.6 ms                           |
| 16 | 10.4 ms | 90.2 ms  | 100.4 ms                       | 198.2 ms                           |
| 30 | 10.6 ms | 194.2 ms | 204.5 ms                       | 322.4 ms                           |

So a rollback of N frames costs a restore, about two frames' worth, and the N
frames: 3 frames' worth for one frame back, 10 for eight. From today's moments,
every 30 frames, it runs 15 frames more on average (100 ms rather than 16 for
one frame back here). A snapshot costs about 1.4 frames, so a peer keeps one
before each frame only while that frame runs on a prediction
(`Timeline::moments_from`): input that arrives within the delay costs nothing,
and a frame on a prediction costs about 2.4 frames' worth. By these ratios (not
measured idle), gdev's default scene on an idle machine would take under 1 ms
for a frame on a prediction and about 3 ms for a rollback of the whole window (8
frames), inside a 16.7 ms frame.

## Checks

- `cargo test -p pen-runtime net`: the wire, input tracks and prediction, and
  sessions of a small deterministic game over a link that delays, loses, repeats
  and reorders datagrams: two and three players end with the same input and
  packets for every frame as a straight run of that input; input inside the
  delay needs no rollback; a peer that started elsewhere takes the host's moment
  and viewport; a game that keeps something outside its state is caught; another
  build is refused.
- `cargo test -p pen net::`: two peers of gdev itself (`dist/gdev`), each on its
  own thread and program, over real UDP sockets on the loopback with 25 ms each
  way: the same input and packets as a straight run, the guest having played 20
  frames alone first and taken the host's moment.
- Two headless processes (`tmp/e2e/net-run.sh` in the lane that built it;
  `pen --game dist/gdev --headless --frames 300 --net ...` twice, each with its
  own `--script` and `--data`): the last line of every frame of their `--trace`s
  agree, their last frames are the same PNG bytes, and the host's
  `--record-input` replays to the same trace.

## Spectating

```sh
pen run --broadcast 127.0.0.1:7200          # serve viewers
pen --game DIR --spectate 127.0.0.1:7200    # watch (or ws://HOST:PORT)
```

`--broadcast ADDRESS` serves viewers over WebSockets. A viewer joining gets the
latest moment the history keeps at or before the last final frame, the frames
from there, and then each frame once it is final: once it ran, or in a net
session once every player's input for it is known (so a viewer never sees a
rollback). `--spectate ADDRESS` plays the stream as a replay plays a recording,
waiting for frames as they come; headless, it ends with the stream. If the
session goes back in time, viewers start again from where it is.
`pen_runtime::net::stream` is the stream (messages and what to send whom),
platform independent; `crates/pen/src/spectate.rs` the native server and viewer,
and `spectate/websocket.rs` the little of WebSocket they need (a browser page
can open the stream with `new WebSocket(url)`).

A stream carries every player's input, keys typed included: broadcast only where
that may go. On a loopback address only native viewers and pages from the
loopback may watch.

Checked by `cargo test -p pen spectate` (the WebSocket handshake and framing,
and a viewer joining gdev's stream at frame 100 that plays every frame from the
moment at 90 to the end with the session's packets) and by two headless
processes: a viewer that joined at frame 630 played the session's 270 frames
with the same packets. Their last frames differed in 3 of 2.46 million pixels,
all within 2%: the renderer's field patches, which a viewer that joined late
only has in full once the game sends them again ("Pixels aren't packets" in
determinism.md).

## What is left

- **The browser.** pen-web plays neither: a net session there needs WebRTC data
  channels (or a relay over WebSockets) as its `Link`, and a viewer the stream
  over `WebSocket`; both reuse `pen_runtime::net` as it is.
- **Reaching peers.** Peers need each other's addresses; there is no
  matchmaking, NAT traversal or relay. Steam's networking (sockets relayed
  through Valve, lobbies) would be the next `Link`.
- **Time sync.** Peers start together (the handshake), run on their own clocks
  at 60 Hz and wait at the window; there is no gentle slowing of a peer that
  runs ahead, as GGPO does.
- **Joining a session under way**, and leaving one: the player count is fixed
  when it starts, and a peer that leaves ends it for the rest once they need its
  input.
- **Cheaper rollbacks.** A restore starts a new program and runs a frame to
  prove it; moments the session took itself could skip the proof.
- **gdev's players.** gdev reads no Player events: one world, both hands.
