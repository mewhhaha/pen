# pen

pen runs games compiled to WebAssembly, natively (winit, wgpu, wasmtime,
cpal) and in the browser (WebGPU, WebAudio), from one Rust runtime and
renderer. A game is a Wasm program that pen calls once a frame: the frame's
input goes in as a packet of numbers, and a packet comes back that says what
to draw, what to play and what to ask of the platform. pen draws it, plays
it, answers it, and keeps everything else that makes a game shippable:
recording and replay, time travel, hot reload with the game's state kept,
golden-image tests, accessibility, Steam, and packages for Linux, Windows
and the web.

gdev, a 3D game written in Blot, is the game pen was built for, and the
largest one it runs. pen assumes nothing of it: a new game starts with
`pen new`, builds on the bindings pen generates, and can add events and
render sections of its own without changing pen.

- [Your first pen game](first-game.md) makes one and walks through it.
- [What a game needs](game-contract.md) is the whole contract: the files, the
  program protocol, the packets, the catalog, and what pen still assumes.
- [The game's own events and sections](extensions.md) and
  [material shaders](materials.md) cover the two places a game extends pen.
- The rest is how pen works inside: its plugin API, determinism, rendering
  performance and shipping.

The repository's README is the reference for every command and option
(`pen --help` prints them too). `examples/bounce` is a second, small game in
the repository, run in CI beside gdev.
