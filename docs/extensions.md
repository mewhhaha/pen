# The game's own events and sections

pen's protocol (`protocol/schema.json`) is shared by every game. A game that
needs to tell the host something pen has no section for, or to be told
something pen has no event for, declares it in its own `pen.json`. pen
carries it without a change to its protocol, `pen codegen` writes typed Blot
for it, and pen's tools name it.

```json
"protocol": {
  "events": {
    "Kick": {
      "tag": 1,
      "doc": "Pushes the ball: adds to its velocity along x and z.",
      "fields": { "x": "f32", "z": "f32" }
    }
  },
  "sections": {
    "Score": {
      "tag": 1,
      "doc": "How the ball is doing, every frame.",
      "fields": { "bounces": "integer", "speed": "f32" }
    }
  }
}
```

- A name is a type name (a capital, then letters and digits); a field's is
  lowercase. Tags run from 1 to 65535, each once among the events and once
  among the sections.
- A field is `f32` (any finite number), `integer` (a whole number to
  16777215, a `U32` in Blot) or `flag` (a `Bool`).
- An event has at most three fields. A section has at least one; a frame
  sends a list of its records, none or many.

## On the wire

Both travel in generic carriers that are part of pen's protocol:

- An event is pen's `Game` input event (tag 100): its own tag in the code
  lane, its fields in the three lanes after. Hosts send it only to a game
  that declares it; pen's own Blot bindings leave it out, so a game without
  declarations never sees one.
- A frame's sections follow the whole render packet, after the draws' extra
  lanes: a marker (0x504558), their count, then each one's tag, length and
  lanes. A packet without any ends before the marker, as it always did.
  Hosts read them into `RenderFrame::extensions` (a tag and its lanes) and
  check only their bounds: at most 64 sections a frame, a million lanes each.

Reserved tag 0 carries optional [development provenance](packet-diagnostics.md).
It has tighter bounds and is consumed by the host's diagnostics. It is not an
application section and does not appear in `RenderFrame::extensions`.

Nothing in pen's types or decoders depends on a game's declarations, so a
game adds, renames or drops them without pen changing, and gdev and Ink,
which declare none, build as they did.

## In Blot

With declarations, `pen codegen` writes `pen/extensions.blot` (and removes it
once there are none):

- `Event`, pen's events (`#Pen protocol.Event`) and the game's (`#Kick { x,
  z }`), and `Input`, whose events are those; `decode_input` reads it.
- For each section, its record type (`Score`), its tag (`score_tag`) and
  `encode_score`, which makes a list of records the `types.Extension` the
  frame carries.
- `run`, which serves the game as `pen/program`'s does, with the game's
  `step` returning a third thing: the frame's sections.

```blot
import * as extensions from "pen/extensions"

entry const main = fn host => extensions.run game host

const step = fn (world: World) => fn (input: extensions.Input) => do:
  let taken = array.fold_left take world input.events
  ...
  let score = extensions.encode_score #[#extensions.Score { bounces: next.bounces, speed }]
  return (next, view next hit input, #[score])
```

## In pen's tools

- A script sends an event by name or tag: `game=Kick,9,4` (`--script`,
  scenario files, `pen packets`). Its values are checked against the
  declaration; by tag (`game=1,9,4`) it needs none, as in a package.
- `pen packets DIR` writes each frame's sections to `DIR/extensions.jsonl`,
  by their declarations:

  ```json
  {"frame":39,"sections":[{"section":"Score","records":[{"bounces":1,"speed":8.264}]}]}
  ```

  A section the checkout does not declare is written as its tag and lanes.
- Recordings, replays, time travel and determinism traces carry them like
  every other event and lane.

## What it is for, and what it is not

The game's own sections reach pen's tools, not its renderer or mixer: pen
draws and plays only what its protocol describes. They are for what a game
wants seen outside itself (scores and progress for tests and bots, debugging
views, telemetry a test reads back), and its events for what scripts, tests
and tools tell it. A host that does more with them (a native service, the
dev tools, an embedding of pen-runtime) reads `RenderFrame::extensions` and
sends `GameEvent::Game`.
