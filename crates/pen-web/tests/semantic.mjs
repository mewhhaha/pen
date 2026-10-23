// Runs the same deterministic guest and semantic queries as the native session
// regression. Qualification commands use the browser's real devtools Host.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { launch, hide, without } from './launch.mjs';
import { serve } from './serve.mjs';

const fixtures = resolve(process.env.PEN_SEMANTIC_FIXTURES || 'target/semantic-inspection');
const host = resolve(process.env.PEN_WEB_HOST || 'crates/pen-web/pkg');
const expected = JSON.parse(await readFile(join(fixtures, 'native.json')));
const op = JSON.parse(await readFile(join(fixtures, 'operations.json')));
const id = { kind: 'Hauler', words: [7, 4294967295, 2, 0] };
const files = Object.fromEntries(['pen_web.js', 'pen_web_bg.wasm', 'guest.js', 'structured.js'].map(name => [`/${name}`, join(host, name)]));
const server = await serve(fixtures, { isolated: true, files });
const browser = await launch();
const results = [];
try {
  for (const [observed, opaque] of [[true, true], [false, false], [true, false]]) {
    const context = await browser.newContext({ viewport: { width: 160, height: 120 } });
    await hide(context);
    const page = await context.newPage();
    const errors = [];
    let revision = 1;
    let revisionsRead = 0;
    let adapterReads = 0;
    let adapterConfig = opaque ? null : JSON.stringify({ format: "ink-persist-v1" });
    page.on('pageerror', error => errors.push(String(error)));
    await page.route('**/__pen/revision', route => { revisionsRead++; return route.fulfill({ body: String(revision) }); });
    await page.route('**/assets/saved.json', route => {
      adapterReads++;
      return route.fulfill(adapterConfig === null ? { status: 404, body: 'not found' } : { body: adapterConfig, contentType: 'application/json' });
    });
    await page.route('**/catalog.json', route => route.fulfill({ json: { meshes: [], materials: [], sounds: [] } }));
    await page.route('**/game.json', route => route.fulfill({ json: { name: 'semantic-fixture', title: 'Semantic fixture' } }));
    await page.route('**/game.wasm', route => route.fulfill({ path: join(fixtures, observed ? 'observed.wasm' : 'plain.wasm'), contentType: 'application/wasm' }));
    await page.route('**/semantic.html', route => route.fulfill({
      contentType: 'text/html',
      headers: { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' },
      body: `<!doctype html><style>canvas{width:160px;height:120px}</style><canvas tabindex="0"></canvas><div id="status">Starting</div><script type="module">
        import init,{start} from './pen_web.js';
        const canvas=document.querySelector('canvas');
        canvas.setAttribute('data-devtools-request',JSON.stringify({operation:${op.pause}}));
        try {await init();await start(canvas,{qualify:true,development:true});}
        catch(error){document.querySelector('#status').textContent=String(error);}
      </script>`,
    }));
    await page.goto(`${server.url}semantic.html`);
    await page.waitForFunction(() => document.querySelector('canvas').hasAttribute('data-devtools-answer'), null, { timeout: 120000 });
    const ask = async request => {
      await page.locator('canvas').evaluate((canvas, request) => {
        canvas.removeAttribute('data-devtools-answer');
        canvas.setAttribute('data-devtools-request', JSON.stringify(request));
      }, request);
      await page.waitForFunction(() => document.querySelector('canvas').hasAttribute('data-devtools-answer'));
      return JSON.parse(await page.locator('canvas').getAttribute('data-devtools-answer'));
    };
    const semantic = async text => {
      const answer = await ask({ operation: op.semantic, text });
      return answer.text ? JSON.parse(answer.text) : answer;
    };
    const waitAt = async position => page.waitForFunction(position => document.querySelector('canvas').getAttribute('data-frame') === String(position), position);
    await ask({ operation: op.history });
    if (opaque) {
      for (const operation of [op.state, op.tuned, op.watch]) {
        const answer = await ask({ operation, a: 99, b: 0, c: 0, d: 0 });
        assert.equal(answer.status, 1, 'format tools unavailable for opaque state');
      }
      assert.equal((await semantic({ op: 'capture', tag: 42 })).status, 'ok');
      await ask({ operation: op.step, a: 2 });
      await waitAt(2);
      assert.deepEqual((await semantic({ op: 'objects', position: '2' })).rows[0].id, id);
      const download = page.waitForEvent('download');
      const exported = await ask({ operation: op.export });
      assert.equal(exported.status, 0);
      const bytes = await readFile(await (await download).path());
      assert.equal(bytes.subarray(0, 8).toString(), 'GDEVSAVE');
      const lanes = Array.from({ length: bytes.readUInt32LE(16) }, (_, at) => bytes.readFloatLE(20 + at * 4));
      assert.deepEqual(lanes, [5678, 1, 1, 99, 1, 2, 2, 1, 1]);
      assert.deepEqual(errors, []);
      results.push({ opaque: 'no adapter: format tools unavailable; semantic observations available; save exported unchanged' });
      await context.close();
      continue;
    }
    if (observed) assert.equal((await semantic({ op: 'capture', tag: 42 })).status, 'ok');
    const witnesses = [];
    for (let position = 1; position <= 4; position++) {
      await ask({ operation: op.step, a: 1 });
      await waitAt(position);
      witnesses.push((await ask({ operation: op.state, a: 99 })).values);
    }
    assert.deepEqual(witnesses, [[0, 0], [1, 1], [2, 2], [3, 3]]);
    await semantic({ op: 'seek', position: '2' });
    await waitAt(2);
    assert.deepEqual((await ask({ operation: op.state, a: 99 })).values, witnesses[1]);
    if (observed) {
      await semantic({ op: 'watch', watch: { object: id, field: 17, equals: { kind: 'unsigned', value: 4294967295 } } });
      await ask({ operation: op.step, a: 2 });
      await waitAt(3);
      assert.equal((await semantic({ op: 'status' })).paused_at, '3');
      // A remaining requested step must be cancelled by the transition watch.
      await page.waitForTimeout(150);
      assert.equal(await page.locator('canvas').getAttribute('data-frame'), '3');
      assert.deepEqual((await ask({ operation: op.state, a: 99 })).values, witnesses[2]);
      for (const test of expected) assert.deepEqual(await semantic(test.request), test.answer, JSON.stringify(test.request));
      assert.equal((await semantic({ op: 'seek', position: '999' })).reason, 'outside_retained_history');
      assert.equal((await semantic({ op: 'objects', position: '0' })).reason, 'not_retained');
      await semantic({ op: 'seek', position: '4' });
      await waitAt(4);
      await semantic({ op: 'watch', watch: { object: id, field: null, equals: null } });
      // Exercise the player's existing save-import path with an actual file.
      await page.evaluate(async () => {
        const bytes = await (await fetch('restore.bin')).arrayBuffer();
        const transfer = new DataTransfer();
        transfer.items.add(new File([bytes], 'restore.bin'));
        window.dispatchEvent(new DragEvent('drop', { dataTransfer: transfer }));
      });
      await page.waitForFunction(async () => {
        const database = await new Promise(resolve => {
          const request = indexedDB.open('pen', 1);
          request.onupgradeneeded = () => request.result.createObjectStore('saves');
          request.onsuccess = () => resolve(request.result);
          request.onerror = () => resolve(null);
        });
        if (!database || !database.objectStoreNames.contains('saves')) return false;
        const exists = await new Promise(resolve => {
          const request = database.transaction('saves').objectStore('saves').get(location.pathname + ':pen.quicksave');
          request.onsuccess = () => resolve(Boolean(request.result));
          request.onerror = () => resolve(false);
        });
        database.close();
        return exists;
      });
      await ask({ operation: op.step, a: 1 });
      await waitAt(5);
      assert.deepEqual((await ask({ operation: op.state, a: 99 })).values, witnesses[1]);
      const restored = await semantic({ op: 'objects', position: '5' });
      assert.equal(restored.restored, true);
      assert.deepEqual(restored.rows[0].id, id);
      assert.ok((await semantic({ op: 'status' })).watch);
      await semantic({ op: 'seek', position: '4' });
      await waitAt(4);
      await semantic({ op: 'seek', position: '5' });
      await waitAt(5);
      assert.equal((await semantic({ op: 'objects', position: '5' })).restored, true);
      assert.deepEqual((await ask({ operation: op.state, a: 99 })).values, witnesses[1]);
      while (revisionsRead === 0) await page.waitForTimeout(100);
      revision = 2;
      let status;
      for (let tries = 0; tries < 200; tries++) {
        status = await semantic({ op: 'status' });
        if (status.invalidated === 'build_changed') break;
      }
      assert.equal(status.invalidated, 'build_changed');
      assert.equal(status.retained, 0);
      assert.equal(status.watch, null);
      assert.deepEqual((await ask({ operation: op.state, a: 99 })).values, witnesses[1]);
      results.push({ comparison: `${expected.length} native/browser semantic answers match`, transitionPause: 3 });
      results.push({ restore: 'boundary marked without firing watch', reload: 'observations and watches invalidated' });
      const reads = adapterReads;
      for (let i = 0; i < 8; i++) await ask({ operation: op.state, a: 99 });
      assert.equal(adapterReads, reads, 'tooling reuses the adapter without fetching configuration');
      adapterConfig = '{"format":';
      revision++;
      for (let tries = 0; tries < 200; tries++) {
        if ((await ask({ operation: op.state, a: 99 })).status === 2) break;
      }
      assert.equal((await ask({ operation: op.state, a: 99 })).status, 2, 'bad declaration explicitly fails tooling');
      assert.equal((await semantic({ op: 'status' })).epoch, status.epoch, 'bad adapter rejects build activation');
      adapterConfig = JSON.stringify({ format: 'opaque' });
      revision++;
      for (let tries = 0; tries < 200; tries++) {
        if ((await ask({ operation: op.state, a: 99 })).status === 1) break;
      }
      assert.equal((await ask({ operation: op.state, a: 99 })).status, 1, 'corrected declaration switches to opaque');
      adapterConfig = JSON.stringify({ format: 'ink-persist-v1' });
      revision++;
      for (let tries = 0; tries < 200; tries++) {
        if ((await ask({ operation: op.state, a: 99 })).status === 0) break;
      }
      assert.deepEqual((await ask({ operation: op.state, a: 99 })).values, witnesses[1]);
      results.push({ adapter: 'cached queries; bad reload rejected; corrected selection applied without page restart' });
    }
    assert.deepEqual(errors, []);
    results.push({ observed, state: witnesses });
    await context.close();
  }
  const adapter = without.has('jspi') ? 'workers' : 'jspi';
  await writeFile(join(fixtures, `browser-${adapter}.json`), JSON.stringify(results, null, 2));
  process.stdout.write(JSON.stringify({ adapter, results }, null, 2) + '\n');
} finally {
  await browser.close();
  await server.close();
}
