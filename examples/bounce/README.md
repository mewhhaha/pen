# Bounce

A second game for pen, beside gdev, and the smallest that uses each part of
the boundary a game needs: a ball in a walled yard that the arrow keys roll.
It bounces off the walls with a blip, and each bounce sets a marker on the
far wall.

It is built on pen's bindings alone (`pen/`, which `pen codegen` writes):
no gdev, no Ink. It adds two things of its own, which its `pen.json`
declares under `protocol` and pen carries without a protocol change:

- the **Kick** event, which pushes the ball (`--script "game=Kick,9,4"`);
- the **Score** section, which says every frame how often the ball has hit
  a wall and how fast it goes (`pen packets DIR` writes it, by name, to
  `DIR/extensions.jsonl`).

| File | What it is |
|---|---|
| `src/main.blot` | The entry: `pen/extensions`' `run` serves the game |
| `src/game.blot` | The world, its events, physics, frame and Score |
| `src/assets/` | The catalog, a box and a ball, four materials, their shader and the blip |
| `tests/bounce.blot` | Its rules, for `pen test-blot` |
| `tests/visual/` | Golden images, for `pen test` |
| `make_assets.py` | Writes the meshes and the sound |

From pen's checkout, with blot2 beside it (`../blot2`):

```sh
cargo run --release -p pen -- run --project examples/bounce
cargo run --release -p pen -- test-blot --project examples/bounce
cargo run --release -p pen -- test --project examples/bounce examples/bounce/tests/visual/scenarios.json
cargo run --release -p pen -- packets target/bounce --project examples/bounce --script "wait=5 game=Kick,9,4 wait=60"
```

`just examples` runs them all, and CI's game job runs `just examples`. The
golden images were made on Vulkan (NVIDIA). The scene is flat-lit geometry
without MSAA, FXAA, shadows or post effects, to keep other adapters close;
it has not run on lavapipe yet, which is not installed here.
