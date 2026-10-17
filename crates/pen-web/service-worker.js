// The service worker of a static web export (`pen build --web` writes it,
// filling in EXPORT). For the files of the export, and only those:
// - it fetches a file's Brotli copy (NAME.br) when the host serves those
//   with Content-Encoding (itch.io does), else the file as it is;
// - it keeps what it fetched, so the game loads again offline, and drops
//   other exports' copies once this one runs (each export is a version);
// - it serves them cross-origin isolated (COOP and COEP), which lets a
//   browser without JSPI run the game's programs in workers.
// A development page (`?dev`, served by `pen web`) drops the worker and
// loads again without it.

// { version, files: { path: [encodings] } }, written by `pen build --web`.
const EXPORT = null;

const scope = new URL(self.registration.scope);
const prefix = `pen ${scope.pathname} `;
const cacheName = prefix + EXPORT.version;

self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    for (const name of await caches.keys()) {
      if (name.startsWith(prefix) && name !== cacheName) await caches.delete(name);
    }
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== scope.origin || !url.pathname.startsWith(scope.pathname)) return;
  if (request.mode === 'navigate' && url.searchParams.has('dev')) return;
  const path = decodeURIComponent(url.pathname.slice(scope.pathname.length)) || 'index.html';
  if (!Object.hasOwn(EXPORT.files, path)) return;
  event.respondWith(answer(event, path));
});

async function answer(event, path) {
  const cache = await caches.open(cacheName);
  const key = new URL(path, scope).href;
  let response = await cache.match(key);
  if (!response) {
    response = await download(path);
    // Kept while the page goes on with it.
    if (response.status === 200) event.waitUntil(cache.put(key, response.clone()));
  }
  return isolated(response);
}

// Whether the host serves `.br` files Brotli-encoded (a promise): the first
// one asked for tells, and a host that does not wastes only that download.
let decodes;

async function download(path) {
  if (EXPORT.files[path].includes('br')) {
    let response;
    if (!decodes) {
      const first = brotli(path);
      decodes = first.then(Boolean);
      response = await first;
    } else if (await decodes) {
      response = await brotli(path);
    }
    if (response) return response;
  }
  return fetch(new URL(path, scope), { cache: 'no-cache' });
}

// The file from its Brotli copy, if the host decoded that as it served it.
async function brotli(path) {
  const response = await fetch(new URL(`${path}.br`, scope), { cache: 'no-cache' }).catch(() => undefined);
  if (response?.status !== 200 || !/\bbr\b/.test(response.headers.get('content-encoding') ?? '')) return undefined;
  // The browser has decoded it: it goes on as the file itself.
  return new Response(response.body, { headers: { 'Content-Type': type(path) } });
}

const types = {
  html: 'text/html; charset=utf-8', js: 'text/javascript', wasm: 'application/wasm', json: 'application/json',
  png: 'image/png', ogg: 'audio/ogg', wav: 'audio/wav', md: 'text/markdown; charset=utf-8',
};

function type(path) {
  return types[path.slice(path.lastIndexOf('.') + 1)] ?? 'application/octet-stream';
}

// `response` with the headers that make the page cross-origin isolated.
function isolated(response) {
  if (response.status === 0) return response;
  const headers = new Headers(response.headers);
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  // Cross-origin, as a page that embeds the game cross-origin isolated
  // (itch.io's, with its "SharedArrayBuffer support") requires of it.
  headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}
