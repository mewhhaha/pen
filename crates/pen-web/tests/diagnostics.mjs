// The Rust tests export cases.json, recovery.json and guest binaries to
// PEN_DIAGNOSTIC_FIXTURES. Both hosts inspect the very same corrupt bytes.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { launch, hide, without } from './launch.mjs';
import { serve } from './serve.mjs';

const fixtures = resolve(process.env.PEN_DIAGNOSTIC_FIXTURES || 'target/packet-diagnostics');
const host = resolve(process.env.PEN_WEB_HOST || 'crates/pen-web/pkg');
const cases = JSON.parse(await readFile(join(fixtures, 'cases.json')));
const recovery = JSON.parse(await readFile(join(fixtures, 'recovery.json')));
const files = Object.fromEntries(['pen_web.js', 'pen_web_bg.wasm', 'guest.js', 'structured.js'].map(name => [`/${name}`, join(host, name)]));
const server = await serve(fixtures, { isolated: true, files });
const browser = await launch();
const results = [];
try {
  const context = await browser.newContext();
  await hide(context);
  const page = await context.newPage();
  await page.route('**/diagnostics.html', route => route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>Packet diagnostics</title>' }));
  await page.goto(`${server.url}diagnostics.html`);
  const actual = await page.evaluate(async cases => {
    const host = await import('./pen_web.js');
    await host.default();
    const catalog = JSON.stringify({ meshes: ['mesh'], materials: ['material'], sounds: ['sound'] });
    return cases.map(test => ({ name: test.name, diagnostic: host.qualify_frame(new Float32Array(new Uint32Array(test.bits).buffer), 1, catalog) }));
  }, cases);
  for (let index = 0; index < cases.length; index++) assert.equal(actual[index].diagnostic, cases[index].diagnostic, cases[index].name);
  results.push({ comparison: `${cases.length} native/browser diagnostics match exactly` });
  await context.close();

  for (const test of [...recovery, ...['startup', 'restore'].map(name => ({ name, diagnostic: recovery.find(test => test.name === 'nonfinite').diagnostic }))]) {
    const context = await browser.newContext({ viewport: { width: 160, height: 120 } });
    await hide(context);
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    let revision = 1;
    let revisionsRead = 0;
    const development = !['startup', 'restore'].includes(test.name);
    await page.route('**/__pen/revision', route => { revisionsRead++; return route.fulfill({ body: String(revision) }); });
    await page.route('**/catalog.json', route => route.fulfill({ json: { meshes: [], materials: [], sounds: [] } }));
    await page.route('**/game.json', route => route.fulfill({ json: { name: 'diagnostic-fixture', title: 'Packet diagnostic fixture' } }));
    await page.route('**/game.wasm', route => route.fulfill({ path: join(fixtures, revision === 1 ? `${test.name}.wasm` : 'good.wasm'), contentType: 'application/wasm' }));
    await page.route('**/diagnostics.html', route => route.fulfill({
      contentType: 'text/html',
      headers: { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' },
      body: `<!doctype html><style>canvas{width:160px;height:120px}#status{white-space:pre-wrap}</style><canvas tabindex="0"></canvas><div id="status">Starting</div><script type="module">
        import init,{start} from './pen_web.js';
        try {
          await init();
          if (${test.name === 'restore'}) {
            const bytes = new Uint8Array(await (await fetch('quicksave.bin')).arrayBuffer());
            const database = await new Promise((resolve,reject) => {
              const request = indexedDB.open('pen',1);
              request.onupgradeneeded = () => request.result.createObjectStore('saves');
              request.onsuccess = () => resolve(request.result);
              request.onerror = () => reject(request.error);
            });
            await new Promise((resolve,reject) => {
              const transaction = database.transaction('saves','readwrite');
              transaction.objectStore('saves').put(bytes,location.pathname+':pen.quicksave');
              transaction.oncomplete = resolve;
              transaction.onabort = () => reject(transaction.error);
            });
            database.close();
          }
          await start(document.querySelector('canvas'),{qualify:true,development:${development}});
        }
        catch(error) { document.querySelector('#status').textContent=String(error); }
      </script>`,
    }));
    await page.goto(`${server.url}diagnostics.html`);
    await page.waitForFunction(expected => document.querySelector('#status').textContent.includes(expected), test.diagnostic, { timeout: 120000 });
    if (development) {
      assert.match(await page.locator('#status').textContent(), /game paused/);
      // Establish the development watcher's baseline before publishing a replacement.
      while (revisionsRead === 0) await page.waitForTimeout(100);
      const position = await page.locator('canvas').getAttribute('data-frame');
      await page.waitForTimeout(100);
      assert.equal(await page.locator('canvas').getAttribute('data-frame'), position, 'rejected frame stays paused');
      revision = 2;
      await page.waitForFunction(() => document.querySelector('#status').textContent === '' && Number(document.querySelector('canvas').dataset.frame) >= 3, null, { timeout: 120000 });
      // The replacement's Restore traps unless it receives the last accepted
      // state, 42. The original traps if its rejected candidate was committed.
      results.push({ recovery: test.name, resumed: true });
    } else {
      if (test.name === 'restore') {
        assert.match(await page.locator('#status').textContent(), /could not load pen.quicksave/);
        const frame = await page.locator('canvas').getAttribute('data-frame');
        await page.waitForFunction(previous => document.querySelector('canvas').dataset.frame !== previous, frame);
        assert.match(await page.locator('#status').textContent(), /could not load pen.quicksave/, 'accepted menu frames do not hide a failed Continue');
      }
      results.push({ [test.name]: 'full structural error visible' });
    }
    assert.deepEqual(errors, [], test.name);
    await context.close();
  }
  await writeFile(join(fixtures, without.has('jspi') ? 'browser-workers.json' : 'browser-jspi.json'), JSON.stringify(results, null, 2));
  console.log(results);
} finally {
  await browser.close();
  await server.close();
}
