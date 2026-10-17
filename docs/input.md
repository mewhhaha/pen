# Input

What the player does reaches the game as input events, in the frame's
packet (`protocol/schema.json`, `input.events`), the same from both hosts:
native pen (`crates/pen/src/input.rs`) and the browser (`crates/pen-web/src/input.rs`).
Whatever both must work out alike lives in pen-runtime: the gamepads'
dead zones (`gamepad`), touches and their gestures (`touch`), text and an
input method's composition (`text`) and the game's actions (`actions`).
Every event is input like any other: an input recording (`--record-input`,
the timeline) keeps it, and a replay, a rewind or a step plays it back
without reading a device again. `pen packets` and `pen run --headless` take
the same events from script steps, through the same pen-runtime code.

| Event | Tag | What |
|---|---|---|
| KeyDown, KeyUp, PointerDown, PointerUp, PointerMove, Wheel, FocusLost | 1–7 | The keyboard and mouse, as before |
| GamepadConnected, GamepadDisconnected | 15, 16 | A pad by number (from 0, the lowest free), and whether its layout is the standard gamepad's |
| GamepadButton | 17 | A button (GAMEPAD_BUTTON_*), its value 0 to 1, and pressed (at half or more) |
| GamepadAxis | 18 | A stick axis (GAMEPAD_AXIS_*), -1 to 1, right and down positive, past the dead zone |
| Action | 19 | An action the game declared: its value and the device that moved it (INPUT_DEVICE_*) |
| TextInput | 20 | A character typed or committed, or (composing 1) one of a composition's |
| Composition | 21 | An input method's composition: its length (its characters follow) and caret |
| MouseMotion | 22 | The mouse's motion while the pointer is locked |
| Touch | 23 | A touch by number, where it is, and its phase (TOUCH_PHASE_*) |
| Pan, Pinch | 24, 25 | Two or more touches dragged together, and spreading or closing |

## Gamepads

Natively gilrs reads the pads (evdev through udev on Linux,
Windows.Gaming.Input on Windows) and maps them to one layout with SDL's
controller database; in the browser, `navigator.getGamepads()` each frame.
Both read each pad into the standard gamepad's layout (`pen_runtime::gamepad::Pad`:
17 buttons, 4 axes, in the W3C order), and `Gamepads` numbers them, applies a
radial dead zone of 0.15 to each stick (rescaled, so the stick rises from 0
at its edge) and sends a change only when it is worth one (1/128, or a value
reaching 0 or ±1). gilrs's own filters are off, so its dead zone does not
stack on pen's. A pad that goes away lets go of what it held first. Only a
window reads pads; a headless run's pads are its script's.

## Actions

A game names what the player does (jump, move, open the map) apart from the
keys, buttons and sticks that do it, and declares each action with its
default bindings through the input service (SERVICE_INPUT, INPUT_OP_*):

- `declare` action `a` (0 to 255): `b` 0 for a button (1 while any binding
  holds it at half or more), 1 for an axis (its bindings summed, -1 to 1);
  the values are its name's `c` code points, then its bindings, three
  numbers each: the source (INPUT_SOURCE_*: key, mouse, pad_button,
  pad_axis), the code (a key's code as KeyDown carries it, a mouse button,
  GAMEPAD_BUTTON_*, GAMEPAD_AXIS_*) and a scale (-1 makes D push move.x
  left, or reads a stick's other half).
- `capture` slot `b`: the next key, mouse button, pad button or stick pushed
  past half becomes that binding, answered when it comes; Escape gives up.
  The press it takes moves no action.
- `bind`, `unbind`, `reset` (every action's with `a` = 256) and `bindings`.

Once a live frame's events are in, `Actions::map` (the session's `plan`, the
page's `step`) puts an Action event right after each event that moved an
action, saying which device moved it, so a game can show the prompts of the
device the player used last. The action events are recorded with the rest,
so a replay maps nothing. The host keeps the declarations and the player's
bindings beside the preferences: `bindings.json` in the data directory,
`pen.bindings` in the page's localStorage. A binding the player chose keeps
with the action's name, so a game may renumber its actions. A game should
declare its actions on every start (the host keeps them, so a replay that
goes on live still has them).

## Steam Input

With the `steam` feature and Steam running, Steam Input reads the game's
actions from the controllers Steam configures (`crates/pen/src/steam_input.rs`),
by name: a button action reads the digital action of its name, or an analog
action's x (a trigger); an axis action the analog action of its name, or
`NAME.x` and `NAME.y` that axis of analog action `NAME`. What Steam reads
joins the action's own bindings, and its Action events say INPUT_DEVICE_STEAM.
The Steam service adds `input_action_set` (activate the set named by the
text, on every controller and those that come later), `input_glyph` (the
path of the glyph image of what is bound first to the action named by the
text) and `input_panel` (Steam's binding panel). The action manifest is the
app's Steamworks configuration; `PEN_STEAM_INPUT_MANIFEST=FILE` loads a local
one while developing. None of this has run against a live Steam client: the
logic is tested against a stand-in (`steam_input::Api`).

## Text and input methods

A key that types a character sends KeyDown and then TextInput (repeating
while it is held; not with Ctrl or the logo key). Natively the text is
winit's (`KeyEvent::text`, dead keys composed); in the browser, a key whose
`key` is one character. An input method (IME) is on only while the game's
focused interface node is a text box (`pen_runtime::text::text_box`), since it
takes keys a game would otherwise get: natively `set_ime_allowed` and
`set_ime_cursor_area` over the text box, in the browser a hidden text field
that takes the focus over it. Its composition comes as a Composition event
(length and caret, in characters) and its characters as TextInput events
with composing 1; length 0 ends it, and what it commits comes as TextInput
events. The dev tools' console types keys, so no input method opens over it.

## Raw mouse motion

Holding the secondary button locks the pointer (as before). While it is
locked, MouseMotion carries the mouse's own motion: winit's
`DeviceEvent::MouseMotion` (unaccelerated where the platform gives it), the
browser's `movementX/Y` in CSS pixels. PointerMove still follows, from the
same motion, for games that only take the pointer.

## Touch

Natively winit's `Touch`; in the browser, pointer events whose pointerType
is touch (the canvas has `touch-action: none`). `Touches` numbers them (from
0, reused once lifted), and the first touch down also moves and presses the
pointer (button 0), so a game that only takes the pointer plays on a touch
screen: on gdev, two taps on the rock strike it exactly as two clicks do,
packet for packet. A second touch lets the pointer go until every touch is
lifted. Two or more touches are gestures: once their centre has moved 8
pixels they Pan (how far it moved since the last Pan, the first one
included), and once their spread has changed by 8 pixels they Pinch (their
centre, and the factor their spread grew by since the last Pinch).

## Script steps

`pad=connect`, `pad=disconnect`, `pad=BUTTON[,VALUE]` and `pad=AXIS,VALUE`
(a gamepad, through the same dead zone), `touch=ID,X,Y` and `lift=ID`,
`motion=DX,DY`, `compose=TEXT[,CURSOR]` and `commit=TEXT`; `key=` and `type=`
send the text of the keys they press. `tests/input.steps` uses them all:

```sh
mkdir -p data && cat > data/bindings.json <<'JSON'
{"actions": [{"index": 0, "name": "jump", "kind": "button", "defaults": [[0, 32, 1], [2, 0, 1]]}]}
JSON
pen run --headless --data data --script-file tests/input.steps --record-input input.penrec --trace first.trace
pen run --headless --data data --replay input.penrec --trace replay.trace
cmp first.trace replay.trace
```

## Tests

- pen-runtime: `gamepad`, `touch`, `text` and `actions` unit tests (dead
  zones, numbering, pans and pinches, carets in bytes and UTF-16, mapping,
  captures, persistence, the service).
- pen: script steps, `session::bindings`, `steam_input` (with a stand-in for
  Steam), and `timeline::contract_tests::input`, which plays every kind of
  event on the exported game (`dist/gdev`) and checks the same input gives
  the same packets, rewinds among them run again exactly, and a recording
  replays them.
- gdev: `tests/protocol.blot` decodes every new event as the host encodes it.
- The browser: `crates/pen-web/tests/input.mjs` (`just browser-input`) drives
  keys, an input method (CDP's `Input.imeSetComposition` and `insertText`),
  the locked mouse, touches (`Input.dispatchTouchEvent`), a stubbed gamepad
  and a declared action in headless Chromium, reads what the game was sent
  from the canvas's `data-events` hook, and checks Ctrl+Shift+R's replay
  brings the game the same events again.

## Left to do

- Gamepads: rumble, more than one player's actions (actions read every pad),
  and pads Steam Input hides behind its own (with Steam Input on, a game
  should use the actions).
- Steam Input: a live run (no Steam client or controller here), generating the
  action manifest from the declared actions, and motion data.
- Input methods: candidate windows place over the text box, not the caret
  inside it (the game does not say where its caret is).
- gdev declares no actions and has no text box of its own yet; its dev tools
  console could take TextInput, which would let an input method type into it.
