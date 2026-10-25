// The browser host's simulation services, in wasm32 under node: the built
// host (crates/pen-web/build.sh's pkg, or the directory given) answers fixed
// requests to noise, and its answers
// must have the digest native pen's have (pen_simulation::conformance).
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const pkg = resolve(process.argv[2] || new URL('../pkg', import.meta.url).pathname);
const host = await import(pathToFileURL(`${pkg}/pen_web.js`).href);
host.initSync({ module: await readFile(`${pkg}/pen_web_bg.wasm`) });
console.log(`wasm32 ${host.qualify_services()}`);
