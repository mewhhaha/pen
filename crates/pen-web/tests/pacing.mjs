// Delayed guest replies must not move canvas submission out of repaint.
// Holding completion notifications must allow two frames, then stop submitting.
// PEN_WEB_SITE selects an export; PEN_WEB_HOST optionally replaces its host files.
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { launch, hide } from './launch.mjs';
import { serve } from './serve.mjs';

const shots = resolve(process.env.PEN_WEB_SHOTS || 'target/browser-pacing');
const schema = JSON.parse(await readFile(new URL('../../../protocol/schema.json', import.meta.url)));
const indices = name => Object.fromEntries(schema.enumerations.find(item => item.name === name).values.map((value, index) => [value, index]));
const operations = indices('DevtoolsOperation');
const fields = indices('DevtoolsFrame');
await mkdir(shots, { recursive: true });
const fixture = join(shots, 'pacing.html');
await writeFile(fixture, `<!doctype html><html><style>html,body{margin:0;width:100%;height:100%}canvas{width:100%;height:100%;display:block}</style><canvas tabindex="0"></canvas><div id="status">Starting</div><script type="module">import init,{start} from './pen_web.js';await init();const canvas=document.querySelector('canvas');await start(canvas,{qualify:true});canvas.focus();</script></html>`);
const files = { '/pacing.html': fixture };
if (process.env.PEN_WEB_HOST) for (const name of ['pen_web.js', 'pen_web_bg.wasm', 'guest.js', 'structured.js']) files[`/${name}`] = join(process.env.PEN_WEB_HOST, name);
const server = await serve(process.env.PEN_WEB_SITE || 'dist/web', { isolated: true, files });
const browser = await launch();
let page;
try {
  page = await browser.newPage({ viewport: { width: 640, height: 480 } });
  await hide(page);
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  await page.route('**/guest.js', async route => {
    const response = await route.fetch();
    const source = (await response.text()).replace('export async function openGuest(', 'async function openGuestNow(');
    await route.fulfill({ response, body: source + `\nexport async function openGuest(...args) { const guest=await openGuestNow(...args); const next=guest.next; guest.next=async function(...args) { const packet=await next.apply(this,args); await new Promise(resolve=>setTimeout(resolve,12)); return packet; }; return guest; }` });
  });
  await page.addInitScript(() => {
    localStorage.setItem('pen.preferences', JSON.stringify([1,1,0.5,0,0,0,1,1,0,0.8,1,0.6,0,0]));
    const probe = globalThis.pacing = { inRepaint: false, draws: [], hold: false, held: [], acquireDelay: 0, frameOperation: 0 };
    const raf = requestAnimationFrame;
    globalThis.requestAnimationFrame = callback => raf.call(globalThis, time => {
      probe.inRepaint = true;
      setTimeout(() => { probe.inRepaint = false; }, 0);
      callback(time);
    });
    const acquire = GPUCanvasContext.prototype.getCurrentTexture;
    GPUCanvasContext.prototype.getCurrentTexture = function(...args) {
      probe.draws.push(probe.inRepaint);
      if (probe.draws.length > 1000) probe.draws.shift();
      if (probe.acquireDelay) {
        const until = performance.now() + probe.acquireDelay;
        probe.acquireDelay = 0;
        while (performance.now() < until) { /* Deliberate surface stall. */ }
        // Read this frame's timings at the start of the next host frame.
        const canvas = document.querySelector('canvas');
        canvas.removeAttribute('data-devtools-answer');
        canvas.setAttribute('data-devtools-request', JSON.stringify({ operation: probe.frameOperation, a: 1 }));
      }
      return acquire.apply(this, args);
    };
    const done = GPUQueue.prototype.onSubmittedWorkDone;
    GPUQueue.prototype.onSubmittedWorkDone = function(...args) {
      return done.apply(this, args).then(() => probe.hold ? new Promise(resolve => probe.held.push(resolve)) : undefined);
    };
  });
  await page.goto(`${server.url}pacing.html`);
  await page.waitForFunction(() => document.querySelector('canvas')?.getAttribute('aria-busy') === 'false', null, { timeout: 120000 });
  const answer = async () => {
    await page.waitForFunction(() => document.querySelector('canvas').hasAttribute('data-devtools-answer'));
    const result = JSON.parse(await page.locator('canvas').getAttribute('data-devtools-answer'));
    assert.equal(result.status, 0);
    return Object.fromEntries(Object.entries(fields).map(([name, index]) => [name, result.values[index]]));
  };
  const request = async () => page.locator('canvas').evaluate((canvas, operation) => {
    canvas.removeAttribute('data-devtools-answer');
    canvas.setAttribute('data-devtools-request', JSON.stringify({ operation, a: 1 }));
  }, operations.frame);
  await request();
  await answer(); // Enable measurement before exercising waits.
  await page.evaluate(() => { pacing.draws.length = 0; });
  await page.waitForFunction(() => pacing.draws.length >= 20);
  const draws = await page.evaluate(() => pacing.draws);
  assert.ok(draws.every(Boolean), `${draws.filter(value => !value).length}/${draws.length} frames acquired the canvas outside repaint`);
  await page.evaluate(() => { pacing.hold = true; });
  await page.waitForTimeout(400);
  const held = await page.evaluate(() => ({ count: pacing.held.length, draws: pacing.draws.length }));
  assert.equal(held.count, 2, 'two frames can be submitted before completion must be acknowledged');
  await page.waitForTimeout(150);
  assert.equal(await page.evaluate(() => pacing.draws.length), held.draws, 'a stalled GPU must not accumulate more frames');
  await request(); // The next loop observes the frame blocked on completion.
  await page.evaluate(() => { pacing.hold = false; for (const done of pacing.held.splice(0)) done(); });
  const delayed = await answer();
  assert.ok(delayed.pace >= 300, `GPU completion wait belongs to pacing: ${JSON.stringify(delayed)}`);
  assert.ok(delayed.render < delayed.pace / 2, 'completion waiting must not inflate render time');
  const sum = value => value.wait + value.decode + value.validate + value.render + value.acquire + value.present + value.pace + value.housekeeping;
  assert.ok(Math.abs(sum(delayed) - delayed.total) < 0.1, 'host phases must account for total exactly once');
  await page.waitForFunction(previous => pacing.draws.length >= previous + 4, held.draws);
  await page.evaluate(operation => {
    document.querySelector('canvas').removeAttribute('data-devtools-answer');
    pacing.frameOperation = operation;
    pacing.acquireDelay = 120;
  }, operations.frame);
  const acquired = await answer();
  assert.ok(acquired.acquire >= 110, `surface stall belongs to acquisition: ${JSON.stringify(acquired)}`);
  assert.ok(acquired.render < acquired.acquire / 2, 'surface waiting must not inflate render time');
  assert.ok(Math.abs(sum(acquired) - acquired.total) < 0.1, 'surface timing must not be counted twice');
  await page.keyboard.press('Control+Shift+P');
  await page.waitForTimeout(500);
  const paused = await page.locator('canvas').getAttribute('data-frame');
  await page.waitForTimeout(200);
  assert.equal(await page.locator('canvas').getAttribute('data-frame'), paused, 'pausing holds the game while repaint continues');
  await page.keyboard.press('Control+Shift+P');
  await page.waitForFunction(previous => document.querySelector('canvas').dataset.frame !== previous, paused);
  await page.setViewportSize({ width: 600, height: 400 });
  await page.waitForFunction(() => document.querySelector('canvas').width === 600);
  await page.screenshot({ path: join(shots, 'pacing.png') });
  assert.deepEqual(errors, []);
  console.log({ delayed, acquired });
  console.log(`Pacing passed: ${draws.length} delayed frames within repaint; bounded at two outstanding frames; waits attributed separately; resume, pause and resize work.`);
} catch (error) {
  await page?.screenshot({ path: join(shots, 'failure.png'), timeout: 5000 }).catch(() => {});
  throw error;
} finally {
  await browser.close();
  await server.close();
}
