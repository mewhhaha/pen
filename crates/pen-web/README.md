# pen in the browser

`pen-web` is the browser front end: Rust input, frame scheduling, WebAudio,
platform services, IndexedDB saves and the game's Blot dev tools, drawn through
`pen-render` on WebGPU, or on WebGL2 where the browser has no WebGPU.

A browser without WebGPU (no `navigator.gpu`, or no adapter from it, as in
Firefox on Linux) draws through WebGL2 with wgpu's GL backend. The choice is
made before the canvas has a context, and the diagnostics name the adapter
"WebGL2: …" (or "WebGPU: …"). The device asks for the limits the adapter
has, which include no storage buffers: pen-render then draws sprites and
decals from uniform arrays, a window of them at a time, copies the depth and
the stencil's marks where its passes read them, and draws the scene with one
sample a pixel, FXAA still smoothing it (`crates/pen-render/src/downlevel.rs`).
The host is larger for it: 2.8 MB (1.1 MB gzipped) against 0.8 MB with
WebGPU alone, as wgpu's GL backend and naga come with it. The scene draws in
floating point, so WebGL2 needs `EXT_color_buffer_float`, which desktop
browsers have; without it the page says so. WebGL2 bounds its own queue, so
frames do not wait there for the GPU to finish the one before.

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
(Firefox before 153, Safari before 27, Chrome before 137) runs each guest in a
worker instead, `guest.js` again: the guest blocks at its host call with
`Atomics.wait` on a SharedArrayBuffer until the page has written the next
request into it (in chunks of 4 MiB, the worker asking for each next one), and
posts its packet back. That needs a cross-origin isolated page (COOP and COEP),
which `pen web` serves, a static export's service worker provides (below) and
itch.io gives with its "SharedArrayBuffer support" frame option; without
either, the page says what it needs. Both paths give the same packets: the
conformance harness passes bit for bit with JSPI taken away, in Chromium and in
Firefox. Chromium runs guests as fast in workers; Firefox 155 took 20 ms a
packet against 5.5 with JSPI in the conformance replay.

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
`game.json` (optional: its `title` names the page, its `name` the files the
page hands over) and `assets/`. All URLs are relative, so exports work beneath
a directory on static hosting.

`pen build --web` does that, then prepares the site for hosting
(`crates/pen/src/web_export.rs`): a Brotli (quality 11) and a gzip copy beside
each file that compresses by a tenth or more (`NAME.br`, `NAME.gz`; not PNG,
Ogg or the packed textures), and `service-worker.js`, this crate's template
with the export's file list and version (a hash of its files) filled in. With
`--zip FILE` it also writes the site as a ZIP to upload, and warns where
itch.io would refuse it (over 1000 files, 500 MB, 200 MB a file or 240
characters a path). gdev's export: 482 files, 126 of 229 precompressed (39.4
MB of them as 10.1 MB of Brotli, the host's 2.9 MB Wasm as 0.8 MB), a 76.4
MB ZIP, made in 40 s with Brotli on four threads.

The service worker (registered by `index.html`, except on `?dev` pages, which
drop any an export left at their address) handles the export's own files and
nothing else:

- It fetches a file's Brotli copy where the host serves `.br` files with
  `Content-Encoding: br`, as itch.io does (the first copy asked for tells;
  a host that serves them as they are costs that one download), and the file
  itself otherwise; hosts that negotiate encodings (nginx's `gzip_static`,
  Caddy's `precompressed`) serve the copies for the files' own names.
- It keeps every file it fetched in a cache named for the page's scope and the
  export's version, and drops other versions' once it runs: the second visit
  loads from the cache, and the game plays offline after that.
- It serves them cross-origin isolated (COOP, COEP and CORP), so that a
  browser without JSPI can run the game in workers. The first visit waits up
  to 3 s for the worker to take the page over; without JSPI the page then
  loads once more through it.

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

Input follows native pen's (docs/input.md): keys and their text, an input
method composing into a hidden text field while the game's text box has the
focus, `movementX/Y` while the pointer is locked, touches (pointer events
whose pointerType is touch; the canvas takes `touch-action: none`) and the
Gamepad API, read each frame, all through pen-runtime's `gamepad`, `touch`,
`text` and `actions`.

Preferences use localStorage, written when they change, as do the game's
actions and the player's bindings (`pen.bindings`); saves use IndexedDB
with native snapshot bytes and a key scoped to the page's path. The game's
and the dev tools' states are saved every five seconds when they changed,
written off the frame's path. Saves move as files (`files.rs`): the dev
tools' `/export [NAME]` downloads the game's state in native pen's save
format (`NAME`, or `GAME-20261017-031500.save`), which native pen loads as its
`quicksave.bin`; `/import` opens the browser's file chooser (the Enter that
ran it lets the page open one), and a save chosen there, or dropped onto the
page, becomes the quicksave and loads after the next live frame, as F9 loads
one, so the history and recordings keep it. A new itch.io upload has a new
address, and so its own saves: exporting and importing carries them over.

A page plays the input recording its link names (`replay.rs`): `#replay=DATA`
holds the recording itself, deflated in URL-safe base64
(`pen_runtime::recording::fragment`), and `?replay=URL` fetches a `.penrec`
beside the page or wherever the page may fetch from. A recording dropped onto
the page, or chosen by `/import`, plays too. The game restores the
recording's moment, plays its frames at their game time in place of the
player's input and pauses where they end; Ctrl+Shift+P plays on, live. The dev
tools' `/record` starts recording the page's input (restoring the game from
its moment first, as native pen does, so the replay matches) and stops: the
recording downloads as a `.penrec`, and the link that plays it goes on the
clipboard and into the browser's console. Recordings are native pen's
`.penrec` either way: one made in the browser replays natively with `pen run
--replay`, and one made natively plays in the browser, on the same build.

The backquote console and diagnostics are the same Blot program as native pen.
They draw as sprites and take input through the game's declared UI tree, and
pen-runtime answers what they ask as it does natively.

With `?qualify` (or `qualify: true`) the canvas carries the hooks the browser
harnesses wait on, of the last frame the GPU finished: `data-frame` (its
position), `data-quality`, `data-pending`, `aria-busy` (assets still
loading), `data-packet` (the hash of its packet, which a replay must reach)
and `data-events`, the input events of the last frames that ran, recorded or
live (each its five lanes; assets' and services' left out). The harnesses
add the flag to the URLs they are given.

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
`tests/input.mjs` (`just browser-input`) sends a `pen web` page keys, text, an
input method's composition (CDP), the locked mouse, touches, a stubbed gamepad
and a declared action, checks `data-events`, and checks Ctrl+Shift+R's replay
brings the game the same events again.
`tests/development.mjs` checks the real `pen web` revision/Bake endpoints and
portable static paths; `PEN_RELOAD_FILE` names an asset whose timestamp it touches
and restores, and `PEN_STATIC_URL` selects an optional subdirectory smoke. `just browser`
and `just browser-dev` run `tests/browser.mjs` and `tests/development.mjs`
against a fresh export and a running `pen web`.

`tests/launch.mjs` opens the browser for the harnesses: `PEN_WEB_BROWSER`
picks Chromium (the default) or Firefox (`just browser-tools` fetches it with
`PEN_WEB_BROWSER=firefox`), and `PEN_WEB_WITHOUT=jspi,webgpu` takes those away
from pages (deleting `WebAssembly.Suspending` and `Navigator.prototype.gpu`
before their scripts run, and in Firefox through its preferences), as browsers
without them are. `tests/serve.mjs` serves an export as hosts do: plainly, as
itch.io does (`.br` files with `Content-Encoding: br`), cross-origin isolated,
or gone (offline). `tests/export.mjs` checks the service worker takes the first
visit over, fetches the Brotli copies only where the host decodes them, and
that a third visit plays with the host gone; without JSPI, also that the page
is isolated and the game runs in workers. `tests/files.mjs` records input with
`/record` and checks its link, `?replay=` and the file dropped onto a page each
replay onto the recorded run's last packet (`data-packet`), and that `/import`
and a dropped save each become the game's state byte for byte; given
`PEN_NATIVE_RECORDING` and `PEN_NATIVE_PACKET` (from `pen run --record-input
--trace`), it replays native pen's recording too. `just browser-files` runs
both.
