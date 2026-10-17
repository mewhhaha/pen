# Mods

A mod is a guest program that runs beside the game in a sandbox of its own,
with only the capabilities its manifest asks for. Each frame the player plays
live, pen shows every mod the frame and adds what the mods may add to the
frame's input, after the player's events. The game cannot tell a mod's input
from the player's, and input recordings keep it, so replays, rewinds and
reloads hold without the mods.

```sh
pen --game dist/rust-guest --mod dist/mods/sticky-arrows
pen run --mod ../my-mods/autopilot --mod ../my-mods/macro      # several
pen packets out --game dist/gdev --mod mods/x --script "wait=60 quit"
```

`--mod DIR` works for `pen` (a package), `pen dev`, `pen run` and `pen packets`.

## A mod's directory

- `mod.json`, its manifest:

  ```json
  {"name": "sticky-arrows", "capabilities": ["observe", "input"]}
  ```

  `name` is up to 64 letters, digits, spaces, `-` and `_`. `capabilities` are
  what it may do; an unknown one refuses the mod.
- `mod.wasm`, a module of Blot's guest ABI (2 or 3; blot2's
  `compiler/guest-abi.md`) that exports `frame: Array F32 -> Array F32` (its
  manifest: `02 01 05 frame 06 06 00`) with the array arena (`blot:memory`,
  `blot:allocate`, `blot:reset`), and imports nothing.

## Capabilities

| | |
|---|---|
| `observe` | See the frame's events: the player's input as the game gets it |
| `input` | Add input events: keys, pointer buttons and moves, the wheel, and console commands (event tags 1 to 6 and 9) |

A mod without `observe` sees the frame's number, time and size, and no
events. Only the host makes the other events (focus, assets, preferences,
tunables, interface actions, service answers): a mod that adds one of them, or
any event without `input`, is stopped.

## The protocol

Each frame, `frame` is called with

```
1 (the protocol), frame number, elapsed seconds, width, height, event count, events...
```

each event five lanes, as the game gets it (tag, code, x, y, delta; keys by
pen-protocol's `key_code`). It answers

```
1, event count (at most 64), events...
```

## The sandbox

- **Nothing to call.** The module may import nothing: the guest ABI admits
  only the host callbacks its manifest declares, and a mod's `frame` takes
  none.
- **Memory**: 64 MiB at most (blot-host's `Guest::sandboxed`): growing past
  it fails, as `memory.grow` does at its limit.
- **Time**: 100 ms a call, on the development engine whose code checks a
  deadline.
- A mod that traps, overruns, answers malformed or asks for what it may not
  is stopped and the log says why; the game plays on.

## Limits

- Mods compile on load, so a runtime without the compiler (the player's)
  refuses them.
- The browser host does not load mods yet; the protocol and its checks
  (`pen_runtime::mods`) are shared and ready for it.
- A mod's own state is its own: rewinding the game does not rewind it.

## Example and tests

`examples/rust-guest/sticky/` is a mod in Rust: an arrow key, once tapped,
stays held until it is tapped again or another arrow is (`just rust-guest`
builds it into `dist/mods/sticky-arrows`). Rust exports `frame` directly; its
`build.rs` names the memory `blot:memory` and a `link_section` static carries
the manifest. With it, a tap moves Gather's player across the field, and a
recording made with the mod replays exactly without it (the traces' input and
packet hashes agree frame for frame).

`pen_runtime::mods`' tests check manifests, what a mod sees with and without
`observe`, which events it may add and the splicing; pen's
`mods_add_what_they_may_and_stop_otherwise` runs hand-built mods (one adding
a key, one adding input without the capability, one looping past its
deadline); blot-host's `sandboxes_keep_guests_to_their_memory` checks the
memory limit.
