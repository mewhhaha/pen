// A static file server for the browser harnesses, which can answer as the
// hosts a web export meets do:
//   isolated: every response carries COOP, COEP and CORP, as `pen web` and
//     itch.io's "SharedArrayBuffer support" send them (the page is then
//     cross-origin isolated);
//   encoded: a `.br` file is served as Brotli and a gzip `.gz` file as gzip
//     (Content-Encoding, and the type of the name without it), as itch.io
//     serves them; otherwise they go out as they are, as most hosts do.
// `files` serves more files, by path, from elsewhere. `requests` lists the
// paths asked for, and `offline` makes every request fail, as a host gone
// away. As a command:
//   node serve.mjs DIR PORT [--isolated] [--encoded]
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const types = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm',
  '.json': 'application/json', '.png': 'image/png', '.ogg': 'audio/ogg', '.wav': 'audio/wav', '.wgsl': 'text/plain',
  '.md': 'text/markdown', '.txt': 'text/plain',
};

export function serve(directory, { port = 0, isolated = false, encoded = false, files = {} } = {}) {
  const requests = [];
  const state = { offline: false };
  const server = createServer(async (request, response) => {
    const path = decodeURIComponent(new URL(request.url, 'http://host').pathname);
    requests.push(path);
    if (state.offline) {
      response.socket.destroy();
      return;
    }
    const relative = normalize(path).replace(/^([/\\])+/, '');
    if (relative.startsWith('..')) {
      response.writeHead(400).end();
      return;
    }
    let file = files[path] ?? join(directory, relative);
    if (path.endsWith('/')) file = join(file, 'index.html');
    let body;
    try {
      body = await readFile(file);
    } catch {
      response.writeHead(404, { 'Content-Type': 'text/plain' }).end('not found');
      return;
    }
    const headers = { 'Cache-Control': 'no-cache' };
    let name = file;
    if (encoded && file.endsWith('.br')) {
      headers['Content-Encoding'] = 'br';
      name = file.slice(0, -3);
    } else if (encoded && file.endsWith('.gz') && body[0] === 0x1f && body[1] === 0x8b) {
      headers['Content-Encoding'] = 'gzip';
      name = file.slice(0, -3);
    }
    headers['Content-Type'] = types[extname(name)] ?? 'application/octet-stream';
    if (isolated) {
      headers['Cross-Origin-Opener-Policy'] = 'same-origin';
      headers['Cross-Origin-Embedder-Policy'] = 'require-corp';
      headers['Cross-Origin-Resource-Policy'] = 'cross-origin';
    }
    response.writeHead(200, headers).end(body);
  });
  return new Promise(resolve => server.listen(port, '127.0.0.1', () => {
    resolve({
      url: `http://127.0.0.1:${server.address().port}/`,
      requests,
      set offline(value) { state.offline = value; },
      close: () => new Promise(done => { server.closeAllConnections(); server.close(done); }),
    });
  }));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [directory, port] = process.argv.slice(2);
  const flags = new Set(process.argv.slice(4));
  const served = await serve(directory, { port: Number(port), isolated: flags.has('--isolated'), encoded: flags.has('--encoded') });
  console.log(`serving ${directory} at ${served.url}`);
}
