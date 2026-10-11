// Run after `pen packets DIR --script ...`; serve DIR at packets/ beside export.
import { readFile, writeFile, readdir } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const directory = process.env.PEN_PACKETS || 'target/browser-conformance';
const names = (await readdir(`${directory}/input`)).filter(name => name.endsWith('.f32')).sort();
if (names.length === 0) throw new Error(`No input packets in ${directory}/input`);
// Transfer binary strings: millions of separate byte values make Playwright's
// recursive protocol serializer allocate gigabytes and spend minutes encoding.
const packets = [];
for (const name of names) {
  const bytes = await readFile(`${directory}/input/${name}`);
  if (bytes.byteLength % 4) throw new Error(`Invalid float packet size: ${name}`);
  packets.push(bytes.toString('base64'));
}
const browser = await chromium.launch({ headless: true });
const messages = [];
let stage = 'opening qualification page';
let timer;
const started = performance.now();
try {
  const page = await browser.newPage();
  page.on('console', message => messages.push({ type: message.type(), text: message.text() }));
  page.on('pageerror', error => messages.push({ type: 'pageerror', text: String(error) }));
  await page.exposeFunction('penQualificationStage', value => {
    stage = value;
    console.log(`Browser qualification: ${value}`);
  });
  await page.route('**/qualification.html', route => route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>pen qualification</title>' }));
  await page.goto(process.env.PEN_CONFORMANCE_URL || 'http://127.0.0.1:8044/qualification.html');
  const evaluated = page.evaluate(async packets => {
    const decode = text => new Float32Array(Uint8Array.from(atob(text), byte => byte.charCodeAt(0)).buffer);
    const encode = view => {
      const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
      const chunks = [];
      for (let at = 0; at < bytes.length; at += 16384) chunks.push(String.fromCharCode(...bytes.subarray(at, at + 16384)));
      return btoa(chunks.join(''));
    };
    await window.penQualificationStage('loading Rust host');
    const host = await import('./pen_web.js');
    await host.default();
    await window.penQualificationStage('loading guest and scene');
    const game = new Uint8Array(await (await fetch('game.wasm')).arrayBuffer());
    const scene = new Uint8Array(await (await fetch('scene.bin')).arrayBuffer());
    await window.penQualificationStage(`replaying ${packets.length} packets`);
    const output = await host.qualify(game, scene, packets.map(decode));
    await window.penQualificationStage('checking browser platform adapters');
    const platform = await host.qualify_platform();
    await window.penQualificationStage('transferring binary results');
    return { platform, received: output.map(encode) };
  }, packets);
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`Browser qualification timed out at: ${stage}`)), Number(process.env.PEN_CONFORMANCE_TIMEOUT_MS || 120000));
  });
  const result = await Promise.race([evaluated, timeout]);
  clearTimeout(timer);
  const { received, platform } = result;
  const snapshot = received.pop();
  if (received.length !== names.length) throw new Error(`Expected ${names.length} browser packets; received ${received.length}`);
  const mismatches = [];
  for (let index = 0; index < received.length; index++) {
    const expected = await readFile(`${directory}/output/${names[index]}`);
    const actual = Buffer.from(received[index], 'base64');
    if (!expected.equals(actual)) mismatches.push(names[index]);
  }
  await writeFile(`${directory}/browser.snapshot`, Buffer.from(snapshot, 'base64'));
  await writeFile(`${directory}/browser-conformance.json`, JSON.stringify({ frames: received.length, mismatches, platform, elapsed_ms: performance.now() - started, messages }, null, 2));
  if (mismatches.length) throw new Error(`Native/browser packets differ: ${mismatches.join(', ')}`);
  console.log(`${received.length} native/browser packets agree bit-for-bit; browser snapshot saved.`);
  console.log(platform);
} catch (error) {
  await writeFile(`${directory}/browser-conformance.json`, JSON.stringify({ failure: String(error), stage, elapsed_ms: performance.now() - started, messages }, null, 2));
  throw error;
} finally { clearTimeout(timer); await browser.close(); }
