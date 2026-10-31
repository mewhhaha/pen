# Your first pen game

This makes a game, plays it, changes it, tests it and packages it. It takes
pen built from this repository (`just install` puts it on your `PATH` as
`pen`) and a Blot checkout, blot2, with its compiler built
(`generated/compiler/blotc.gz`); games are written in Blot, and pen compiles
them through its Blot plugin. Nothing here needs gdev or Ink.

## Make it

```sh
pen new marble
```

```
made marble in /home/you/games/marble: `pen dev` there plays it as you edit it, `pen test-blot` runs its tests
```

`pen new DIR` makes a new directory, and the game is named after it. It
looks for blot2 beside the new directory, beside one of the directories above
it, or beside the pen checkout it was built from; `--blot DIR` (or
`PEN_BLOT`) names it instead, and `pen.json` refers to it by a relative path.
What it makes:

```
marble/
├─ pen.json            the game's name, and the Blot compiler and library it builds with
├─ src/main.blot       the entry: hands the game to pen
├─ src/game.blot       the game
├─ src/assets/         catalog.json, a cube mesh, its material and the material's shader
├─ tests/game.blot     its tests
├─ pen/                bindings `pen codegen` generates: never edit them
└─ README.md, .gitignore
```

## Play it

```sh
cd marble
pen dev
```

A window opens with an orange cube turning. The right arrow turns it faster,
the left one slower. `pen dev` keeps compiling: save a change to any file
under `src/` and the game reloads with its state kept, the cube going on
from where it was.

Without a window (on a server, in CI, or to look at one frame):

```sh
pen run --headless --frames 120 --screenshot shot.png
pen run --headless --script "wait=20 key=ArrowRight wait=60 shot=faster quit" --shots shots
```

A headless run's clock is steady (every frame is 1/60 s of game time), so the
same script gives the same pixels every time.

## Read it

`src/main.blot` calls the game's invocation-scoped adapter:

```blot
import { run } from "./game"

entry const main = func host => return run host
```

`run` creates named world, event, draw, sound and numeric packet pools once.
`program.run` handles session version 2, calling the game's `execute` function
for create, frame, snapshot and restore requests. `execute` updates the world
pool and encodes the response before the next host call. Renderer rejection
keeps that simulation state; restoring an earlier state is an explicit request.

- `World` contains the cube's angle and speed. `create` makes its initial value.
- `step` reads a borrowed `protocol.Input` and returns the next `World`.
- `save` copies the world into numeric lanes; `restore` reads those lanes back.
  `schema` versions that layout.
- `view` fills a caller-owned draw pool and returns the fixed render header.
  `codec.encode_mesh_frame` encodes that header, flat mesh draws and sounds into
  the named packet pool. More complex renderers use `encode_render_views` with
  borrowed source pools and known projections.

The cube's step reads the arrows from the events:

```blot
const step = func (world: World) -> (input: protocol.Input) => return do:
  let speed = world.speed
  for event in input.events:
    if let $protocol.KeyDown code = event:
      if code == protocol.key_arrow_right:
        speed := self + nudge
      if code == protocol.key_arrow_left:
        speed := self - nudge
  return $World { angle: world.angle + speed * input.elapsed, speed }
```

`view` starts from `frame.header schema`, sets the camera and light, clears its
`ResizeArray codec.MeshDraw` parameter, and appends the cube's mesh draw:

```blot
  @storage.truncate draws 0
  @storage.push draws { mesh: meshes.cube, material: materials.cube, transform: turned world.angle }
```

Matrices are column-major (`x`, `y`, `z`, `w` are columns), as WGSL reads
them; the projection maps depth into WebGPU's 0 to 1. `meshes.cube` and
`materials.cube` are IDs: Blot imports `src/assets/catalog.json`, and each
file it lists becomes a name for its index.

## Change it

Put a second cube beside the first, standing still: append another draw in
`view`, after the existing cube, and save.

```blot
  @storage.push draws { mesh: meshes.cube, material: materials.cube, transform: moved 2.6 0.0 (-3.0) }
```

Give it a colour of its own. A material is a JSON file naming its shader,
colour and lighting; add `src/assets/blue.material.json`:

```json
{ "shader": "lit.wgsl", "color": [0.2, 0.45, 0.95, 1], "lighting": "lit" }
```

list it in `src/assets/catalog.json` (`"materials": ["cube.material.json",
"blue.material.json"]`), and draw the second cube with `materials.blue`. The
shader, `lit.wgsl`, is the game's too; [material shaders](materials.md) says
what pen binds for it.

## Test it

```sh
pen test-blot
```

```
tests/game.blot (checks)
  ok      test_the_cube_turns_at_its_speed
  ok      test_the_right_arrow_turns_it_faster
  ok      test_a_saved_world_comes_back
```

A test is an `entry const test_...` that returns `check.report` of its
checks (`pen/check`: `equal`, `near`, `holds`). Tests call the game's
functions directly; a test module compiles as if it were
`src/test_fixture.blot`, so it imports the game as `./game`.

Golden images catch what tests of numbers miss. Write
`tests/visual/scenarios.json`:

```json
{
  "size": [480, 320],
  "scenarios": {
    "faster": { "about": "the cube after the right arrow", "script": "key=ArrowRight wait=60 shot=faster" }
  }
}
```

`pen test --update` records `tests/visual/golden/faster/faster.png`; look at
it, keep it, and from then on `pen test` fails when a frame differs from it by
more than the threshold, leaving the difference beside the shot in
`.pen/visual`.

## Ship it

```sh
pen build
```

makes `.pen/package`: the player runtime named after the game, and `game/`
with the program precompiled for the runtime's platform and the assets the
catalog reaches. `pen build --web` makes a static site instead. The game's
name and title (its data directory, the executable and the window's title)
come from `pen.json`'s `game`.

## Where next

- `examples/bounce` in pen's repository is a little more: held keys, a sound,
  several materials, golden images, and an event and a render section of its
  own ([the game's own events and sections](extensions.md)).
- [What a game needs](game-contract.md) is the whole contract between a game
  and pen, for a game in another language or one that wants more than
  `pen/program` gives.
- The README lists every command, option and script step.
