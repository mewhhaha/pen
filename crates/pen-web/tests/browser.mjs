// Headless browser verification. Set PLAYWRIGHT_MODULE to its ES-module entry
// and PEN_WEB_URL / PEN_WEB_SHOTS to an exported game's URL and output directory.
import { mkdir, writeFile } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const shots = process.env.PEN_WEB_SHOTS || 'target/browser-shots';
await mkdir(shots, { recursive: true });
let args = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=native', '--use-angle=vulkan', '--disable-vulkan-surface'];
if (process.env.PEN_WEB_GPU === 'software') {
  args = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader',
    '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'];
}
const browser = await chromium.launch({ headless: true, args });
let page;
const messages = [];
try {
  page = await browser.newPage({ viewport: { width: 960, height: 640 } });
  page.on('console', message => { messages.push({ type: message.type(), text: message.text() }); if (message.type() === 'error') console.log(message.text()); });
  page.on('pageerror', error => messages.push({ type: 'pageerror', text: String(error) }));
  // SwiftShader verifies the real shader/host at a bounded software-GPU cost.
  await page.addInitScript(() => {
    if (!localStorage.getItem('pen.preferences')) localStorage.setItem('pen.preferences', JSON.stringify([1,1,0.5,0,0,0,1,1,0,0.8,1,0.6,0,0]));
  });
  page.setDefaultTimeout(60000);
  // ?qualify has the page write the hooks this waits on (aria-busy, data-frame).
  const url = new URL(process.env.PEN_WEB_URL || 'http://127.0.0.1:8043/');
  url.searchParams.set('qualify', '');
  await page.goto(url.href, { waitUntil: 'networkidle' });
  console.log('page loaded');
  await page.waitForFunction(() => document.querySelector('#status')?.textContent === '', null, { timeout: 60000 });
  await page.waitForFunction(() => document.querySelector('canvas')?.getAttribute('aria-busy') === 'false', null, { timeout: 120000 });
  console.log('renderer settled');
  await page.screenshot({ path: `${shots}/game.png` });
  await page.keyboard.press('Backquote');
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `${shots}/console.png` });
  await page.keyboard.type('/diagnostics high');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(2000);
  await page.keyboard.press('Backquote');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: `${shots}/diagnostics.png` });
  for (const panel of ['timeline', 'tunables', 'builds']) {
    await page.keyboard.press('Backquote');
    await page.keyboard.type(`/${panel}`);
    await page.keyboard.press('Enter');
    await page.waitForTimeout(1500);
    await page.waitForFunction(name => Array.from(document.querySelectorAll('[role="dialog"]')).some(node => node.getAttribute('aria-label')?.toLowerCase() === name), panel);
    await page.screenshot({ path: `${shots}/${panel}.png` });
    if (panel === 'timeline') {
      await page.mouse.click(50, 142);
      await page.waitForTimeout(500);
      const paused = await page.locator('canvas').getAttribute('data-frame');
      await page.waitForTimeout(500);
      if (paused !== await page.locator('canvas').getAttribute('data-frame')) throw new Error('Timeline Pause pointer action did not pause the guest');
      await page.screenshot({ path: `${shots}/timeline-action.png` });
      await page.mouse.click(50, 176);
      await page.waitForTimeout(500);
      if (paused === await page.locator('canvas').getAttribute('data-frame')) throw new Error('Timeline Play pointer action did not resume the guest');
    }
    await page.keyboard.press('Backquote');
    await page.keyboard.press('Backquote');
  }
  await page.setViewportSize({ width: 640, height: 480 });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: `${shots}/resize.png` });
  const preferences = await page.evaluate(() => localStorage.getItem('pen.preferences'));
  const beforeReload = messages.length;
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForFunction(() => document.querySelector('canvas')?.getAttribute('aria-busy') === 'false', null, { timeout: 120000 });
  if (!messages.slice(beforeReload).some(message => message.text.includes('resuming the saved state'))) throw new Error('Page reload did not restore the autosave');
  if (preferences !== await page.evaluate(() => localStorage.getItem('pen.preferences'))) throw new Error('Page reload changed saved preferences');
  await page.screenshot({ path: `${shots}/restored.png` });
  // A dense display: the canvas is backed by device pixels.
  const dense = await browser.newPage({ viewport: { width: 480, height: 320 }, deviceScaleFactor: 2 });
  dense.on('console', message => { if (message.type() === 'error') messages.push({ type: 'error', text: `dense: ${message.text()}` }); });
  dense.on('pageerror', error => messages.push({ type: 'pageerror', text: `dense: ${error}` }));
  await dense.goto(url.href, { waitUntil: 'networkidle' });
  await dense.waitForFunction(() => document.querySelector('canvas')?.getAttribute('aria-busy') === 'false', null, { timeout: 120000 });
  const backing = await dense.evaluate(() => [document.querySelector('canvas').width, document.querySelector('canvas').height]);
  if (backing[0] !== 960 || backing[1] !== 640) throw new Error(`devicePixelRatio 2 backs a 480x320 canvas with ${backing.join('x')}`);
  await dense.screenshot({ path: `${shots}/dense.png` });
  await dense.close();
  const status = await page.locator('#status').textContent();
  await writeFile(`${shots}/evidence.json`, JSON.stringify({ status, messages }, null, 2));
  if (status || messages.some(m => m.type === 'pageerror' || m.type === 'error')) {
    throw new Error(`browser verification failed: ${JSON.stringify({ status, errors: messages.filter(m => m.type === 'error' || m.type === 'pageerror').slice(0, 4) })}`);
  }
  console.log(`Browser game, Blot console, diagnostics, resize and a dense display captured in ${shots}`);
} catch (error) {
  const status = await page?.locator('#status').textContent().catch(() => 'unresponsive');
  await writeFile(`${shots}/evidence.json`, JSON.stringify({ status, messages, failure: String(error) }, null, 2));
  await page?.screenshot({ path: `${shots}/failure.png`, timeout: 10000 }).catch(() => {});
  throw error;
} finally { await browser.close(); }
