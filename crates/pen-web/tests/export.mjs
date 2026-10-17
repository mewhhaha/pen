// A static export (`pen build --web`) as hosts serve it: as itch.io does,
// Brotli copies (NAME.br) decoded through Content-Encoding, then as a plain
// static server does, files as they are. Each time the service worker must
// take the page over on its first visit, fetch the Brotli copies only where
// the host decodes them, and keep the game: a third visit with the host gone
// plays offline. Without JSPI (PEN_WEB_WITHOUT=jspi) the worker must also
// make the page cross-origin isolated, for the game to run in workers.
// PEN_WEB_SITE is the export (default dist/web), PEN_WEB_SHOTS where the
// shots and evidence go; see launch.mjs for the browser.
import { mkdir, writeFile } from 'node:fs/promises';
import { browserName, hide, launch, without } from './launch.mjs';
import { serve } from './serve.mjs';

const site = process.env.PEN_WEB_SITE || 'dist/web';
const shots = process.env.PEN_WEB_SHOTS || 'target/browser-shots';
await mkdir(shots, { recursive: true });
const evidence = [];

// The page shows the game: its status line cleared and its assets loaded.
// It may load itself again first (to become cross-origin isolated).
async function settled(page) {
  for (let attempt = 0; ; attempt++) {
    try {
      await page.waitForFunction(() => document.querySelector('#status')?.textContent === ''
        && document.querySelector('canvas')?.getAttribute('aria-busy') === 'false', null, { timeout: 120000 });
      break;
    } catch (error) {
      if (attempt > 2 || !/context was destroyed|navigat/i.test(String(error))) throw error;
      await page.waitForLoadState();
    }
  }
  return page.evaluate(async () => ({
    controlled: Boolean(navigator.serviceWorker?.controller),
    isolated: crossOriginIsolated,
    jspi: typeof WebAssembly.Suspending === 'function',
    // A canvas drawn through WebGPU has no WebGL2 context to give.
    graphics: document.querySelector('canvas').getContext('webgl2') ? 'WebGL2' : 'WebGPU',
    caches: await Promise.all((await caches.keys()).map(async name => [name, (await (await caches.open(name)).keys()).length])),
  }));
}

for (const encoded of [true, false]) {
  const host = encoded ? 'itch.io-like host' : 'plain host';
  const server = await serve(site, { encoded });
  const browser = await launch();
  const messages = [];
  try {
    const context = await browser.newContext({ viewport: { width: 960, height: 640 } });
    await hide(context);
    const page = await context.newPage();
    page.on('console', message => messages.push({ type: message.type(), text: message.text() }));
    page.on('pageerror', error => messages.push({ type: 'pageerror', text: String(error) }));
    const url = `${server.url}?qualify`;
    await page.goto(url, { waitUntil: 'commit' });
    const first = await settled(page);
    const firstRequests = [...server.requests];
    if (!first.controlled) throw new Error(`${host}: the service worker did not take the first visit over`);
    if (without.has('jspi') && (first.jspi || !first.isolated)) throw new Error(`${host}: without JSPI the page must be cross-origin isolated: ${JSON.stringify(first)}`);
    // Where the host decodes Brotli copies, the worker fetches those of the
    // host's wasm and the game's; otherwise it asks for one, then the files.
    const brotli = firstRequests.filter(path => path.endsWith('.br'));
    if (encoded && !['/pen_web_bg.wasm.br', '/game.wasm.br'].every(path => firstRequests.includes(path))) throw new Error(`${host}: Brotli copies were not fetched: ${firstRequests.join(' ')}`);
    if (encoded && firstRequests.includes('/game.wasm')) throw new Error(`${host}: game.wasm was fetched as it is beside its Brotli copy`);
    if (!encoded && (brotli.length !== 1 || !firstRequests.includes('/game.wasm'))) throw new Error(`${host}: expected one Brotli probe, then the files: ${brotli.join(' ')}`);
    await page.reload({ waitUntil: 'commit' });
    const second = await settled(page);
    server.offline = true;
    const before = server.requests.length;
    await page.reload({ waitUntil: 'commit' });
    const offline = await settled(page);
    await page.screenshot({ path: `${shots}/export-offline-${encoded ? 'encoded' : 'plain'}.png` });
    const errors = messages.filter(message => message.type === 'error' || message.type === 'pageerror');
    if (errors.length) throw new Error(`${host}: ${JSON.stringify(errors.slice(0, 4))}`);
    evidence.push({ host, browser: browserName, without: [...without], first, second, offline, firstRequests: firstRequests.length,
      brotli: brotli.length, offlineRequests: server.requests.length - before });
    console.log(`${host}: the service worker took the first visit over (${brotli.length} Brotli copies fetched), `
      + `and a visit with the host gone played from its cache (${offline.caches.map(([, count]) => count).join('+')} files kept)`
      + `${offline.isolated ? ', cross-origin isolated' : ''}${offline.jspi ? '' : ', the game in workers (no JSPI)'}`
      + `, drawn through ${offline.graphics} in ${browserName}`);
  } catch (error) {
    evidence.push({ host, failure: String(error), messages, requests: server.requests });
    throw error;
  } finally {
    await writeFile(`${shots}/export-evidence.json`, JSON.stringify(evidence, null, 2));
    await browser.close();
    await server.close();
  }
}
