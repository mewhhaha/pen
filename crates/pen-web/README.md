# pen in the browser

`pen-web` is the browser front end: Rust input, frame scheduling, WebAudio,
platform services, IndexedDB saves and the game's Blot dev tools, drawn through
`pen-render` on WebGPU.

`pen-runtime` contains the shared game lifecycle: request validation, frame
protocol, transactional reload and restores, scenes and undo history, snapshots,
service answers and deterministic pathfinding. Native pen supplies a Wasmtime
thread backend; `pen-web` supplies the browser's WebAssembly backend. The native
adapter keeps separate send/receive calls so guest work can overlap drawing.

Browsers suspend the long-lived guest `main(host)` with WebAssembly JS Promise
Integration (JSPI). `guest.js` is the small required JavaScript seam for JSPI,
capability identity and copied array memory. Rust owns the game protocol. A
browser without JSPI reports that requirement explicitly.

Build the host with `./build.sh OUTPUT`. It writes `pen_web.js`,
`pen_web_bg.wasm`, `guest.js`, `index.html` and bindgen typings. Export a game
beside these as `game.wasm`, `devtools.wasm` (optional), `catalog.json`,
`scene.bin` (optional, the same snapshot format as native) and `assets/`.
All URLs are relative, so exports work beneath a directory on static hosting.

```js
import init, { start } from './pen_web.js';
await init();
await start(canvas, {
  game: 'game.wasm', devtools: 'devtools.wasm', catalog: 'catalog.json',
  scene: 'scene.bin', assets: 'assets', development: false,
});
```

Development mode polls `/__pen/revision`; a changed opaque revision reloads the
module transactionally. `POST /__pen/bake` accepts JSON `[id,value]` pairs and returns the completed
tunables bake result as text.
A static export runs without these endpoints. Its Bake service is unavailable.
Steam services are unavailable on the web; URL and clipboard services follow
browser permissions. Audio resumes on the first keyboard or pointer gesture.

Preferences use localStorage; saves use IndexedDB with native snapshot bytes
and a key scoped to the page's path. Rendering waits for submitted GPU work
before scheduling another frame, bounding the queue on slow adapters.

The backquote console and diagnostics are the same Blot program as native pen.
They draw as sprites and take input through the game's declared UI tree.

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
and restores, and `PEN_STATIC_URL` selects an optional subdirectory smoke.
