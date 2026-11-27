# ${TITLE}

A game for pen, written in Squid.

| | |
|---|---|
| `pen dev` | Compile and play, reloading the program, shaders and assets as you edit them |
| `pen run --headless --frames 120 --screenshot shot.png` | A frame, without a window |
| `pen test-squid` | The tests in `tests/` |
| `pen build` | A shippable folder in `.pen/package` (`--web` for a static site) |
| `pen codegen` | Regenerate `pen/`, the bindings pen generates, after pen changes |

- `src/main.sq` hands the game to pen; `src/game.sq` is the game: its
  world, what each frame does to it, what it draws, and how its state is
  saved and restored.
- `src/assets/catalog.json` names the meshes, materials and sounds the game
  uses; Squid imports it, so `meshes.cube` is the cube's ID.
- `pen/` is generated: the packets, the program loop and a frame to start
  from. Do not edit it.
- `pen.json` names the game and the Squid compiler and standard library it
  builds with. It may also declare the game's own events and render sections
  (`protocol`), for which `pen codegen` writes `pen/extensions.sq`.
