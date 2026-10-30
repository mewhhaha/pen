# Pen

Pen runs WebAssembly games natively and in the browser. It provides the host,
renderer, audio, services, compiler plugins and development tools. Game logic
lives in applications such as [Gdev](../gdev/README.md); reusable game systems
live in [Ink](../ink/README.md).

The owned-storage integration is **unfinished**. First compile real Gdev and
render one headless frame with the matching sibling Blot compiler and Pen host.
Read [CONTINUE.md](CONTINUE.md) and [AGENTS.md](AGENTS.md) before resuming.

This is a greenfield project. Breaking changes are welcome; remove superseded
implementations and APIs rather than maintaining legacy compatibility.

## Development

Use the checked-in Rust toolchain and the existing sibling checkouts. From Pen:

```sh
cargo build --locked --release -p pen
cd plugins
cargo build --locked --release --target wasm32-wasip2
```

Build Blot's native development compiler as described in its handoff. Once the
real Gdev main compiles, run from `../blot2`:

```sh
python3 scripts/storage_handoff/run_pen.py --launch
```

Require a successful real frame, trace and PNG; inspect the image. GPU checks
run sequentially and headlessly. `just install` prepares a broader Steam,
shipping-player and browser bundle and is optional for this first native frame.

Preserve guest ABI **8**, session protocol **2**, copied numeric packets and
explicit checkpoint/restore. A rejected presentation does not automatically roll
simulation back. Language tasks are retired in the new ABI; copied-buffer host
workers and application data queues remain.

## Documentation

- [Book contents](docs/SUMMARY.md) and [first game](docs/first-game.md).
- [Full tools and runtime reference](docs/reference.md), retained from the
  former README. Some examples still require migration verification.
- [Game contract](docs/game-contract.md), [guest ABI](docs/guest-abi-next.md),
  [services and workers](docs/services-as-tasks.md), and
  [packet diagnostics](docs/packet-diagnostics.md).
- [Testing](docs/testing.md), [qualification](docs/qualification.md),
  [shipping](docs/shipping.md), and [browser export](docs/web-export.md).

Use affected tests during integration. Full game-dependent native/browser/GPU
qualification remains pending until the actual game fixture builds. Never infer
current readiness from earlier release measurements.
