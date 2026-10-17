// The browser host's input in headless Chromium on the GPU: keys and their
// text, an input method's composition, the locked mouse's motion, touches
// with their pointer, pans and pinches, a gamepad (navigator.getGamepads,
// stubbed with a pad this drives), and a declared action, as the events the
// game was sent (the canvas's data-events hook, ?qualify); then the text
// field staying out of the dev tools' console, and Ctrl+Shift+R replaying the last
// seconds, whose recorded frames must bring the game the same events again
// (those of the last seconds: the whole run when the machine keeps up).
// Needs a `pen web` page (?dev keeps the history a replay plays from).
import { mkdir, writeFile } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const url = new URL(process.env.PEN_INPUT_URL || 'http://127.0.0.1:8047/?dev');
url.searchParams.set('qualify', '');
const shots = process.env.PEN_WEB_SHOTS || 'target/browser-shots';
await mkdir(shots, { recursive: true });

// Tags (protocol/schema.json) and codes the checks read.
const tag = {
  KeyDown: 1, PointerDown: 3, PointerUp: 4, PointerMove: 5, GamepadConnected: 15, GamepadDisconnected: 16,
  GamepadButton: 17, GamepadAxis: 18, Action: 19, TextInput: 20, Composition: 21, MouseMotion: 22, Touch: 23,
  Pan: 24, Pinch: 25,
};
const input = { tags: new Set(Object.values(tag).filter(value => value >= 15)) };
const SOUTH = 0, LEFT_X = 0, KEYBOARD = 0, GAMEPAD = 1;

const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=native', '--use-angle=vulkan', '--disable-vulkan-surface'] });
const messages = [];
try {
  const context = await browser.newContext({ viewport: { width: 960, height: 640 }, hasTouch: true });
  const page = await context.newPage();
  page.on('console', message => messages.push({ type: message.type(), text: message.text() }));
  page.on('pageerror', error => messages.push({ type: 'pageerror', text: String(error) }));
  await page.addInitScript(() => {
    // Action 0, "jump": space and the south button, as the game declares it
    // through the input service; the player's bindings, as the page keeps them.
    localStorage.setItem('pen.bindings', JSON.stringify({ actions: [
      { index: 0, name: 'jump', kind: 'button', defaults: [[0, 32, 1], [2, 0, 1]] },
    ] }));
    // A pad the harness drives: Gamepad and GamepadButton objects whose
    // fields are their own.
    window.__pad = { connected: false, buttons: new Array(17).fill(0), axes: [0, 0, 0, 0] };
    const button = value => Object.create(GamepadButton.prototype, {
      value: { value }, pressed: { value: value >= 0.5 }, touched: { value: value > 0 },
    });
    navigator.getGamepads = () => {
      const pad = window.__pad;
      if (!pad.connected) return [null, null, null, null];
      return [Object.create(Gamepad.prototype, {
        index: { value: 0 }, id: { value: 'pen test pad' }, connected: { value: true },
        mapping: { value: 'standard' }, timestamp: { value: performance.now() },
        buttons: { value: pad.buttons.map(button) }, axes: { value: pad.axes.slice() },
      }), null, null, null];
    };
    // A locked pointer, without asking the headless page for one.
    window.__locked = false;
    Object.defineProperty(Document.prototype, 'pointerLockElement', {
      get() { return window.__locked ? document.querySelector('canvas') : null; },
    });
  });
  await page.goto(url.href, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => document.querySelector('canvas')?.getAttribute('aria-busy') === 'false', null, { timeout: 300000 });
  await page.focus('canvas');
  const events = async () => JSON.parse(await page.evaluate(() => document.querySelector('canvas').getAttribute('data-events') || '[]'));
  const frames = async count => {
    const at = Number(await page.evaluate(() => document.querySelector('canvas').getAttribute('data-frame')));
    await page.waitForFunction(target => Number(document.querySelector('canvas').getAttribute('data-frame')) >= target, at + count);
  };
  const find = (list, wanted) => list.find(event => event[0] === wanted[0] && wanted.every((lane, index) => lane === null || Math.abs(event[index] - lane) < 1e-4));
  const expect = async (what, wanted) => {
    for (let attempt = 0; attempt < 40; attempt++) {
      if (find(await events(), wanted)) return;
      await page.waitForTimeout(50);
    }
    throw new Error(`the game was not sent ${what} (${JSON.stringify(wanted)}): ${JSON.stringify((await events()).slice(-40))}`);
  };

  // Keys and their text, and the action space moves.
  await page.keyboard.press('Space');
  await page.keyboard.type('hi');
  await expect('the space key', [tag.KeyDown, 32, 0, 0, 0]);
  await expect('jump, pressed by space', [tag.Action, 0, 1, KEYBOARD, 0]);
  await expect('typed text', [tag.TextInput, 'h'.codePointAt(0), 0, 0, 0]);
  await expect('typed text', [tag.TextInput, 'i'.codePointAt(0), 0, 0, 0]);

  // The locked mouse's motion.
  await page.evaluate(() => { window.__locked = true; });
  await page.mouse.move(400, 300);
  await page.mouse.move(410, 295);
  await expect('the locked mouse moving', [tag.MouseMotion, 0, 10, -5, 0]);
  await page.evaluate(() => { window.__locked = false; });

  // Two touches: the first is the pointer until the second lets it go;
  // spreading them pans and pinches.
  const cdp = await context.newCDPSession(page);
  const touch = (type, points) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: points.map(([id, x, y]) => ({ id, x, y })) });
  await touch('touchStart', [[1, 300, 300]]);
  await touch('touchStart', [[1, 300, 300], [2, 400, 300]]);
  await touch('touchMove', [[1, 280, 300], [2, 420, 300]]);
  await touch('touchMove', [[1, 260, 300], [2, 440, 300]]);
  await touch('touchEnd', [[2, 440, 300]]);
  await touch('touchEnd', []);
  await expect('the first touch', [tag.Touch, 0, null, null, 0]);
  await expect('the first touch pressing the pointer', [tag.PointerDown, 0, null, null, 0]);
  await expect('the second touch', [tag.Touch, 1, null, null, 0]);
  await expect('two touches spreading', [tag.Pinch, 0, null, null, null]);
  await expect('the touches lifted', [tag.Touch, 1, null, null, 2]);

  // A gamepad: plugged in, its south button (which jumps) and its left stick.
  await page.evaluate(() => { window.__pad.connected = true; });
  await expect('the pad connecting', [tag.GamepadConnected, 0, 1, 0, 0]);
  await page.evaluate(() => { window.__pad.buttons[0] = 1; window.__pad.axes[0] = -1; });
  await expect('the south button', [tag.GamepadButton, SOUTH, 0, 1, 1]);
  await expect('the left stick', [tag.GamepadAxis, LEFT_X, 0, -1, 0]);
  await expect('jump, pressed by the pad', [tag.Action, 0, 1, GAMEPAD, 0]);
  await page.evaluate(() => { window.__pad.buttons[0] = 0; window.__pad.axes[0] = 0.05; });
  await expect('the stick back inside its dead zone', [tag.GamepadAxis, LEFT_X, 0, 0, 0]);
  await page.evaluate(() => { window.__pad.connected = false; });
  await expect('the pad leaving', [tag.GamepadDisconnected, 0, 0, 0, 0]);

  // An input method composing into the text field (focused here as a game's
  // text box focuses it) and committing.
  await page.focus('textarea');
  await cdp.send('Input.imeSetComposition', { text: 'かん', selectionStart: 1, selectionEnd: 1 });
  await expect('the composition', [tag.Composition, 2, 1, 0, 0]);
  await expect('the composition\'s characters', [tag.TextInput, 'ん'.codePointAt(0), 1, 0, 0]);
  await cdp.send('Input.insertText', { text: '漢' });
  await expect('the committed text', [tag.TextInput, '漢'.codePointAt(0), 0, 0, 0]);
  await expect('the composition ending', [tag.Composition, 0, 0, 0, 0]);
  await page.focus('canvas');
  await frames(5);
  const live = (await events()).filter(event => input.tags.has(event[0]));

  // The last seconds again: the recorded frames bring the same events.
  await page.keyboard.press('Control+Shift+KeyR');
  for (let attempt = 0; attempt < 200; attempt++) {
    if (messages.some(message => message.text.includes('the replay ended'))) break;
    await page.waitForTimeout(100);
  }
  if (!messages.some(message => message.text.includes('the replay ended'))) throw new Error('the replay did not end');
  const all = (await events()).filter(event => input.tags.has(event[0]));
  const replayed = all.slice(live.length);
  const tail = live.slice(live.length - replayed.length);
  const same = replayed.length > 0 && tail.every((event, index) => event.every((lane, at) => lane === replayed[index][at]));
  await writeFile(`${shots}/input.json`, JSON.stringify({ live, replayed, messages }, null, 2));
  if (!same || !find(replayed, [tag.Composition, 2, 1, 0, 0])) {
    throw new Error(`the replay did not bring the same events: ${live.length} live, ${replayed.length} replayed`);
  }

  // The dev tools' console types keys, which an input method would take:
  // the text field stays out of it.
  await page.keyboard.press('Backquote');
  await frames(10);
  await page.keyboard.type('/he');
  await frames(10);
  const focused = await page.evaluate(() => document.activeElement?.tagName);
  await page.screenshot({ path: `${shots}/input-console.png` });
  await page.keyboard.press('Escape');
  if (focused !== 'CANVAS') throw new Error(`the console moved the focus to ${focused}`);
  if (messages.some(message => message.type === 'error' || message.type === 'pageerror')) throw new Error('the page reported errors');
  console.log(`browser input passed: keys, text, composition, mouse motion, touches, pans and pinches, a gamepad and an action (${live.length} events), a replay bringing the last ${replayed.length} again, and the console typing keys with the canvas focused`);
} finally {
  await browser.close();
}
