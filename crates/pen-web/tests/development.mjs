// Real pen web reload and portable static-host smoke; source bytes stay unchanged.
import { mkdir, writeFile, stat, utimes } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
// ?qualify has the page write the hooks this waits on (aria-busy, data-frame).
const qualified = address => {
  const url = new URL(address);
  url.searchParams.set('qualify', '');
  return url.href;
};
const url = qualified(process.env.PEN_DEVELOPMENT_URL || 'http://127.0.0.1:8045/?dev');
const changed = process.env.PEN_RELOAD_FILE;
if (!changed) throw new Error('PEN_RELOAD_FILE must name a watched source asset');
const shots = process.env.PEN_WEB_SHOTS || 'target/browser-shots';
await mkdir(shots, { recursive: true });
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=native', '--use-angle=vulkan', '--disable-vulkan-surface'] });
const stamp = await stat(changed);
const messages = [];
const requests = [];
try {
  const page = await browser.newPage({ viewport: { width: 960, height: 640 } });
  page.on('console', message => messages.push({ type: message.type(), text: message.text() }));
  page.on('pageerror', error => messages.push({ type: 'pageerror', text: String(error) }));
  page.on('request', request => requests.push(new URL(request.url()).pathname));
  await page.addInitScript(() => localStorage.setItem('pen.preferences', JSON.stringify([1,1,0.5,0,0,0,1,1,0,0.8,1,0.6,0,0])));
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => document.querySelector('canvas')?.getAttribute('aria-busy') === 'false');
  const bake = await page.evaluate(async () => {
    const send = async values => {
      const response = await fetch('/__pen/bake', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values) });
      return { status: response.status, text: await response.text() };
    };
    return await send([]);
  });
  const rejected = await fetch(new URL('/__pen/bake', url), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify([[11175412, 999999999]]) });
  const invalid = { status: rejected.status, text: await rejected.text() };
  if (bake.status !== 200 || invalid.status !== 400 || !invalid.text.includes('outside')) throw new Error(`Bake validation failed: ${JSON.stringify({ bake, invalid })}`);
  await page.waitForTimeout(1500);
  const start = requests.length;
  await utimes(changed, stamp.atime, new Date());
  await page.waitForFunction(() => document.querySelector('canvas')?.getAttribute('data-frame') !== null);
  for (let attempt = 0; attempt < 40; attempt++) {
    if (messages.some(message => message.text.includes('Blot reloaded'))) break;
    await page.waitForTimeout(250);
  }
  if (!messages.some(message => message.text.includes('Blot reloaded'))) throw new Error('Revision did not reload the game');
  const refreshed = requests.slice(start);
  for (const path of ['/catalog.json', '/game.wasm', '/devtools.wasm']) {
    if (!refreshed.includes(path)) throw new Error(`Revision did not refresh ${path}`);
  }
  if (!refreshed.some(path => path.endsWith('.wgsl'))) throw new Error('Revision did not refresh requested shaders');
  if (!refreshed.some(path => path.endsWith('.png'))) throw new Error('Revision did not refresh requested textures');
  await page.waitForFunction(() => document.querySelector('canvas')?.getAttribute('aria-busy') === 'false');
  await page.keyboard.press('Backquote');
  await page.keyboard.type('/builds');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${shots}/builds-reloaded.png` });
  if (process.env.PEN_STATIC_URL) {
    await page.goto(qualified(process.env.PEN_STATIC_URL), { waitUntil: 'networkidle' });
    await page.waitForFunction(() => document.querySelector('canvas')?.getAttribute('aria-busy') === 'false');
    await page.screenshot({ path: `${shots}/subdirectory.png` });
  }
  await writeFile(`${shots}/development.json`, JSON.stringify({ bake, invalid, refreshed, messages }, null, 2));
  if (messages.some(message => message.type === 'error' || message.type === 'pageerror')) throw new Error('Development runtime reported errors');
  console.log('pen web Bake validation, game/devtools/catalog/shader/texture revision reload and static subdirectory passed.');
} finally {
  await utimes(changed, stamp.atime, stamp.mtime);
  await browser.close();
}
