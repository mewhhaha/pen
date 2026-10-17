// Saves and recordings as files and links, in a static export (`pen build
// --web`) served at PEN_WEB_SITE (default dist/web):
// - the dev tools' `/record` hands over a `.penrec` and a link; the link,
//   `?replay=` naming the file, and the file dropped onto a page each play
//   it, ending paused on the packet the recorded run ended on (its hash);
// - `/export` downloads the game's state as a save; `/import` loads one the
//   file chooser is given, and a save dropped onto the page loads too: the
//   state exported after either is the save, byte for byte.
// The recording, the save and the hashes go to PEN_WEB_SHOTS (with
// files-evidence.json), for native pen to replay the same recording. Given
// PEN_NATIVE_RECORDING (a `.penrec` native pen recorded) and
// PEN_NATIVE_PACKET (the hash of its last packet, from `--trace`), the page
// must replay that onto the same packet too, in a page of another size: the
// canvas keeps the recording's 960x640 for it.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { hide, launch } from './launch.mjs';
import { serve } from './serve.mjs';

const site = process.env.PEN_WEB_SITE || 'dist/web';
const shots = process.env.PEN_WEB_SHOTS || 'target/browser-shots';
await mkdir(shots, { recursive: true });
const recording = `${shots}/files.penrec`;
// The recording, once made, as if it were hosted beside the game.
const native = process.env.PEN_NATIVE_RECORDING;
const server = await serve(site, {
  files: { '/recordings/files.penrec': recording, ...native && { '/recordings/native.penrec': native } },
});
const browser = await launch();
const messages = [];
const evidence = {};
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));

async function open(url, viewport) {
  const page = await context.newPage();
  if (viewport) await page.setViewportSize(viewport);
  page.on('console', message => messages.push({ type: message.type(), text: message.text() }));
  page.on('pageerror', error => messages.push({ type: 'pageerror', text: String(error) }));
  // The page's module runs until the game has started: wait for that below.
  await page.goto(url, { waitUntil: 'commit' });
  await page.waitForFunction(() => document.querySelector('#status')?.textContent === ''
    && document.querySelector('canvas')?.getAttribute('aria-busy') === 'false', null, { timeout: 120000 });
  return page;
}

// Types a line into the dev tools' console, which it opens and closes.
async function command(page, line) {
  await page.keyboard.press('Backquote');
  await wait(300);
  await page.keyboard.type(line);
  await page.keyboard.press('Enter');
  await wait(300);
  await page.keyboard.press('Backquote');
  await wait(300);
}

// The log line that starts with `prefix`, once the page has said it.
async function said(prefix, from = 0) {
  for (let tries = 0; tries < 600; tries++) {
    const found = messages.slice(from).find(message => message.text.startsWith(prefix));
    if (found) return found.text;
    await wait(100);
  }
  throw new Error(`the page never said "${prefix}…"; it said ${JSON.stringify(messages.slice(-12))}`);
}

const hooks = page => page.evaluate(() => {
  const canvas = document.querySelector('canvas');
  return { frame: Number(canvas.getAttribute('data-frame')), packet: canvas.getAttribute('data-packet') };
});

// A replay of `url` ends paused on `expected`'s packet, `frames` frames in.
async function replays(url, expected, frames, name) {
  const from = messages.length;
  const page = await open(url);
  const ended = await said('the replay ended at frame', from);
  await wait(500);
  const shown = await hooks(page);
  await page.screenshot({ path: `${shots}/files-replay-${name}.png` });
  if (shown.packet !== expected.packet || shown.frame !== frames) {
    throw new Error(`replaying ${name}: ended on ${JSON.stringify(shown)}, not ${frames} frames on packet ${expected.packet} (${ended})`);
  }
  evidence[`replay ${name}`] = shown;
  return page;
}

const context = await browser.newContext({ viewport: { width: 960, height: 640 }, acceptDownloads: true });
await hide(context);
try {
  const url = `${server.url}?qualify`;
  const page = await open(url);

  // Record: play into the game, pause, stop.
  await command(page, '/record');
  await said('recording input from frame');
  await page.keyboard.press('Tab');
  await wait(700);
  for (const [x, y] of [[470, 150], [470, 152], [471, 150]]) {
    await page.mouse.click(x, y);
    await wait(350);
  }
  await page.mouse.move(300, 500);
  await page.mouse.down();
  await page.mouse.move(600, 450, { steps: 10 });
  await page.mouse.up();
  await wait(500);
  await command(page, '/pause');
  await wait(500);
  const recorded = await hooks(page);
  const downloading = page.waitForEvent('download');
  const from = messages.length;
  await command(page, '/record');
  const download = await downloading;
  await download.saveAs(recording);
  const stopped = await said('recorded ', from);
  const frames = Number(/recorded (\d+) frames/.exec(stopped)[1]);
  const link = (await said('a link that replays', from)).replace(/^a link that replays \S+: /, '');
  evidence.recorded = { ...recorded, frames, file: download.suggestedFilename(), bytes: (await readFile(recording)).length, link: link.length };
  console.log(`recorded ${frames} frames (${evidence.recorded.bytes} bytes; a link of ${link.length} characters), paused on packet ${recorded.packet}`);

  // The link, the file by URL and the file dropped each replay it.
  await replays(link.replace(/#/, '?qualify#'), recorded, frames, 'link');
  await replays(`${server.url}?qualify&replay=recordings/files.penrec`, recorded, frames, 'url');
  const dropped = await open(url);
  const droppedFrom = messages.length;
  const bytes = [...await readFile(recording)];
  await dropped.evaluate(bytes => {
    const transfer = new DataTransfer();
    transfer.items.add(new File([new Uint8Array(bytes)], 'dropped.penrec'));
    window.dispatchEvent(new DragEvent('drop', { dataTransfer: transfer, cancelable: true }));
  }, bytes);
  await said('the replay ended at frame', droppedFrom);
  await wait(500);
  const droppedShown = await hooks(dropped);
  if (droppedShown.packet !== recorded.packet) throw new Error(`the dropped recording ended on ${JSON.stringify(droppedShown)}`);
  evidence['replay dropped'] = droppedShown;
  console.log(`the link, ?replay=URL and the dropped file each replayed ${frames} frames onto packet ${recorded.packet}`);
  if (native) {
    const expected = { packet: process.env.PEN_NATIVE_PACKET };
    const from = messages.length;
    const replayed = await open(`${server.url}?qualify&replay=recordings/native.penrec`, { width: 1280, height: 720 });
    await said('the replay ended at frame', from);
    await wait(500);
    const shown = await hooks(replayed);
    shown.backing = await replayed.evaluate(() => `${document.querySelector('canvas').width}x${document.querySelector('canvas').height}`);
    await replayed.screenshot({ path: `${shots}/files-replay-native.png` });
    if (shown.packet !== expected.packet) throw new Error(`the native recording ended on ${JSON.stringify(shown)}, not packet ${expected.packet}`);
    if (shown.backing !== '960x640') throw new Error(`a 1280x720 page replayed a 960x640 recording on a ${shown.backing} canvas`);
    evidence['replay native'] = shown;
    console.log(`native pen's recording replayed ${shown.frame} frames onto its last packet ${shown.packet}, on a ${shown.backing} canvas in a 1280x720 page`);
  }

  // Saves: export paused, change the game, import, step once, export again.
  const exportSave = async name => {
    const downloading = page.waitForEvent('download');
    await command(page, `/export ${name}`);
    const download = await downloading;
    await download.saveAs(`${shots}/${name}`);
    return readFile(`${shots}/${name}`);
  };
  const first = await exportSave('first.save');
  if (first.subarray(0, 8).toString() !== 'GDEVSAVE') throw new Error('the export is not a save');
  await command(page, '/play');
  await page.keyboard.press('Tab');
  await wait(1000);
  await command(page, '/pause');
  const changed = await exportSave('changed.save');
  if (changed.equals(first)) throw new Error('playing on did not change the state');
  const choosing = page.waitForEvent('filechooser');
  const importFrom = messages.length;
  await command(page, '/import');
  await (await choosing).setFiles(`${shots}/first.save`);
  await wait(500);
  await command(page, '/step');
  await said('loaded first.save', importFrom);
  const imported = await exportSave('imported.save');
  if (!imported.equals(first)) throw new Error('the imported save did not become the state');
  // Dropped onto the page.
  const dropFrom = messages.length;
  await page.evaluate(bytes => {
    const transfer = new DataTransfer();
    transfer.items.add(new File([new Uint8Array(bytes)], 'changed.save'));
    window.dispatchEvent(new DragEvent('drop', { dataTransfer: transfer, cancelable: true }));
  }, [...changed]);
  await wait(500);
  await command(page, '/step');
  await said('loaded changed.save', dropFrom);
  const redropped = await exportSave('dropped.save');
  if (!redropped.equals(changed)) throw new Error('the dropped save did not become the state');
  evidence.saves = { bytes: first.length, imported: true, dropped: true };
  console.log(`/export saved ${first.length} bytes; /import and a dropped save each became the state byte for byte`);
  const errors = messages.filter(message => message.type === 'error' || message.type === 'pageerror');
  if (errors.length) throw new Error(JSON.stringify(errors.slice(0, 4)));
} catch (error) {
  evidence.failure = String(error);
  evidence.messages = messages.slice(-40);
  throw error;
} finally {
  await writeFile(`${shots}/files-evidence.json`, JSON.stringify(evidence, null, 2));
  await browser.close();
  await server.close();
}
