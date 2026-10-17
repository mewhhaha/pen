# The browser build and itch.io

`pen build --web` makes the game a static site any web host can serve:
the browser host, the game and dev tools, the assets the game uses, a
Brotli and a gzip copy of each file that compresses, and a service worker.
`just itch` builds gdev's into `dist/web` and zips it into `dist/web.zip`
(`pen build --web --zip FILE`). crates/pen-web/README.md describes how the
page, the service worker and the copies work; this is what a host needs.

## Uploading to itch.io

1. `just itch`. The ZIP has `index.html` at its root; the build warns if
   itch.io would refuse it (more than 1000 files, 500 MB, 200 MB in one
   file or 240 characters in a path). gdev's: 482 files, 76.4 MB, made in
   40 s.
2. Upload `dist/web.zip` on the game's page (Kind of project: HTML), tick
   "This file will be played in the browser", or push it with butler:
   `butler push dist/web.zip USER/GAME:html5` (the channel's name is free;
   tick the box after the first push).
3. Embed options: the viewport size you want the game at (960 x 640 is
   what the harnesses use), and the fullscreen button.
4. For browsers without WebAssembly JSPI (Firefox before 153, Safari before
   27, Chrome before 137), tick "SharedArrayBuffer support" under Frame
   options. That serves the game cross-origin isolated, from html.itch.zone
   instead of html-classic.itch.zone, so the game runs in workers there.
   The game's address changes with it, and with it the players' browser
   saves (below).

itch.io serves a `.br` file with `Content-Encoding: br`, so the service
worker fetches the Brotli copies: gdev's 39.4 MB of compressible files
(meshes, Wasm, sounds, JSON) arrive as 10.1 MB. It gzips HTML, JavaScript,
Wasm and WAV itself, but not the binary meshes, which are most of them.

## Other hosts

The site works under any directory of any static host, with HTTPS (or on
localhost): the host serves the files as they are, and the service worker
fetches each file itself where the host does not decode `.br` copies (it
learns which from the first one). Hosts that negotiate encodings (nginx
`gzip_static on; brotli_static on;`, Caddy `file_server { precompressed br
gzip }`) serve the copies for the files' own names. Hosts that send
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` give the isolation without
the service worker; without them, the service worker provides it from the
second load (the page loads once more by itself on a browser without
JSPI). In a private window without service workers, a browser without JSPI
cannot run the game, and the page says so.

## Offline play and updates

Every file the page fetched is kept in the browser, under the export's
version (a hash of its files): from the second visit the game loads from
there, and plays without the network. A new export at the same address
replaces the kept files as it is first loaded. A new itch.io upload has a
new address, and starts afresh.

## Saves

Browser saves live in the page's IndexedDB, under its address: a new
itch.io upload, or the SharedArrayBuffer option, starts with none. The dev
tools' console (the key left of 1) moves them as files: `/export [NAME]`
downloads the game's state in native pen's save format (native pen loads it
as `quicksave.bin` in its data directory), and `/import` asks for a save to
load, as does dropping one onto the page.

## Replays

`/record` in the console starts recording the input, and again stops: the
recording downloads as a `.penrec`, and a link that plays it (`#replay=`,
the recording itself in the address) goes on the clipboard (a 200-frame
recording made a 1.7 KB link). `?replay=NAME.penrec` plays a recording
uploaded beside the game, and one dropped onto the page plays too. A
recording plays only on the build that made it (docs/determinism.md);
recordings made natively with `--record-input` play in the browser and the
other way round, onto the same packets.
