# pen in the browser

`pen-web` is the browser front end: Rust input, frame scheduling, WebAudio,
platform services, IndexedDB saves and the game's Blot dev tools, drawn through
`pen-render` on WebGPU.

`pen-runtime` contains what native pen and the browser share: request
validation and the frame protocol, the guest ABI manifest, transactional
reload and restores, scenes and undo history, snapshots, time travel (the
timeline, the transport and the game clock), the services registry and the
answers both give alike (pathfinding, the platform), and the dev tools
service with its input routing. Native pen supplies a Wasmtime thread
backend; `pen-web` supplies the browser's WebAssembly backend. The native
adapter keeps separate send/receive calls so guest work can overlap drawing.

Browsers suspend the long-lived guest `main(host)` with WebAssembly JS Promise
Integration (JSPI). `guest.js` is the small required JavaScript seam for JSPI,
capability identity and copied array memory. Rust owns the game protocol and
checks the module's `blot:abi` manifest before it runs. A browser without JSPI
reports that requirement explicitly.

The crate is a member of the root workspace (one `Cargo.lock` pins wgpu and
wasm-bindgen for native and browser alike); for any target but
`wasm32-unknown-unknown` it is empty. Build the host with `./build.sh OUTPUT`:
it builds with the workspace's `web` profile (small, no unwinding) into
`target/browser`, installs the wasm-bindgen CLI of the version `Cargo.lock`
pins under `target/wasm-tools`, and writes `pen_web.js`, `pen_web_bg.wasm`,
`guest.js`, `index.html` and bindgen typings. It runs binaryen's
`wasm-opt -O1` when `wasm-opt` is on `PATH` and warns when it is not;
`just web-host` (and so `just install`) requires it. Lint it with
`cargo clippy -p pen-web --target wasm32-unknown-unknown`.

Export a game beside these as `game.wasm`, `devtools.wasm` (optional),
`catalog.json`, `scene.bin` (optional, the same snapshot format as native),
`game.json` (optional: its `title` names the page) and `assets/`. All URLs
are relative, so exports work beneath a directory on static hosting.

```js
import init, { start } from './pen_web.js';
await init();
await start(canvas, {
  game: 'game.wasm', devtools: 'devtools.wasm', catalog: 'catalog.json',
  scene: 'scene.bin', manifest: 'game.json', assets: 'assets',
  development: false, qualify: false,
});
```

The canvas is backed by device pixels: its CSS size times the display's
`devicePixelRatio`, followed as the page moves between screens or zooms. The
game lays out on those pixels and draws its scene at its quality scale of
them, as native pen does with a window's physical size. Pointer positions
are in the same pixels. One frame is in flight: the GPU draws it while the
next is computed, and the next waits for it before it is submitted, which
bounds the queue on slow adapters.

Time travel works as natively: a history keeps each frame's input and a
moment every 30 frames, a rewind restores the nearest moment and runs the
recorded frames again unseen, and Ctrl+Shift+R plays the last seconds again.
Ctrl+Shift+P pauses and plays, Ctrl+Shift+period steps and Ctrl+Shift+comma
goes back a frame. Development keeps a history from the start; a static
export keeps one once the dev tools ask (their timeline).

Development mode (`?dev`, as `pen web` serves the page) asks `/__pen/revision`
every second on a task of its own; a changed opaque revision fetches the new
build, which reloads transactionally when the game is next idle. `POST
/__pen/bake` accepts JSON `[id,value]` pairs; the dev tools' Bake sends the
game's tuned values there and the log says how it went. An invalid frame
pauses the game until a new build, as `pen dev` does. A static export runs
without these endpoints: its Bake service is unavailable and an invalid frame
stops the page. Steam services are unavailable on the web; URL and clipboard
services follow browser permissions, the clipboard answering a later frame.
Audio resumes on the first keyboard or pointer gesture.

Preferences use localStorage, written when they change; saves use IndexedDB
with native snapshot bytes and a key scoped to the page's path. The game's
and the dev tools' states are saved every five seconds when they changed,
written off the frame's path.

The backquote console and diagnostics are the same Blot program as native pen.
They draw as sprites and take input through the game's declared UI tree, and
pen-runtime answers what they ask as it does natively.

With `?qualify` (or `qualify: true`) the canvas carries the hooks the browser
harnesses wait on, of the last frame the GPU finished: `data-frame` (its
position), `data-quality`, `data-pending` and `aria-busy` (assets still
loading). The harnesses add the flag to the URLs they are given.

`tests/conformance.mjs` compares every native `pen packets` output byte with
the browser guest and exercises snapshot storage, pathfinding, unavailable
services and WebAudio. Serve the exported game, set `PEN_PACKETS` to the packet
directory and `PEN_CONFORMANCE_URL` to `qualification.html` beside the export;
the harness supplies that blank page. `PLAYWRIGHT_MODULE` can point to the
installed Playwright ES module. `tests/browser.mjs` captures the actual game,
Blot panels and resize; `PEN_WEB_URL` and `PEN_WEB_SHOTS` select its URL/output.
`tests/fxaa.mjs` executes renderer shader witnesses on browser WebGPU.
It fetches native raw RGBA outputs and asserts all twelve agree within one LSB.
Full game captures use native Vulkan by default; set `PEN_WEB_GPU=software`
explicitly for SwiftShader. The software qualification uses packet replay and
small shader witnesses, since compiling a full scene can exceed its time budget.
`tests/development.mjs` checks the real `pen web` revision/Bake endpoints and
portable static paths; `PEN_RELOAD_FILE` names an asset whose timestamp it touches
and restores, and `PEN_STATIC_URL` selects an optional subdirectory smoke. `just browser`
and `just browser-dev` run `tests/browser.mjs` and `tests/development.mjs`
against a fresh export and a running `pen web`.
