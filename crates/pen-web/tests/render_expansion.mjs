// End-to-end headless Chromium WebGPU + WebAudio + Browser Accessibility + Locale
// verification against the freshly built `crates/pen-web/pkg` host.
//
// Usage:
//   PLAYWRIGHT_MODULE=.../playwright-core/index.mjs node crates/pen-web/tests/render_expansion.mjs
import assert from 'node:assert/strict';
import { join, resolve } from 'node:path';
import { launch, hide } from './launch.mjs';
import { serve } from './serve.mjs';

const hostDir = resolve(process.env.PEN_WEB_HOST || 'crates/pen-web/pkg');

// --- Minimal Wasm binary helpers (ABI 8 + session protocol 2) ---
const uleb = value => {
  const bytes = [];
  do {
    let byte = value & 0x7f;
    value >>>= 7;
    if (value) byte |= 0x80;
    bytes.push(byte);
  } while (value);
  return bytes;
};
const sleb = value => {
  const bytes = [];
  let more = true;
  while (more) {
    let byte = value & 0x7f;
    value >>= 7;
    if ((value === 0 && (byte & 0x40) === 0) || (value === -1 && (byte & 0x40) !== 0)) {
      more = false;
    } else {
      byte |= 0x80;
    }
    bytes.push(byte);
  }
  return bytes;
};
const text = str => [...uleb(str.length), ...new TextEncoder().encode(str)];
const section = (id, contents) => [id, ...uleb(contents.length), ...contents];
const vector = items => [...uleb(items.length), ...items.flat()];
const custom = (name, payload) => section(0, [...text(name), ...payload]);
const f32Bytes = val => [...new Uint8Array(new Float32Array([val]).buffer)];
const u32Bytes = val => [...new Uint8Array(new Uint32Array([val]).buffer)];
const i32Const = v => [0x41, ...sleb(v)];
const f32Const = v => [0x43, ...f32Bytes(v)];
const i32Load = addr => [...i32Const(addr), 0x28, 2, 0];
const i32Store = (addr, val) => [...i32Const(addr), ...val, 0x36, 2, 0];

function encodeFloatArray(values) {
  const out = [...u32Bytes(values.length)];
  for (const v of values) out.push(...f32Bytes(v));
  return out;
}

function uiNodeLanes({
  id, parent = 0, role, x = 0, y = 0, w = 0, h = 0,
  focusable = 0, focused = 0, disabled = 0, hidden = 0, modal = 0,
  checked = 0, expanded = 0, selected = 0, live = 0,
  value = 0, min = 0, max = 0, step = 0, actions = 0,
  name = '',
}) {
  const cps = [...name].map(c => c.codePointAt(0));
  return [
    id, parent, role, x, y, w, h,
    focusable, focused, disabled, hidden, modal,
    checked, expanded, selected, live,
    value, min, max, step, actions,
    cps.length, cps.length, 0,
    ...cps,
  ];
}

function buildRenderFramePacket({ includeLocaleRequest }) {
  const PACKET_MAGIC = 0x474446;
  const EXTENSIONS_MARKER = 0x504558;
  const header = new Array(266).fill(0);
  header[0] = PACKET_MAGIC;
  header[1] = 1; // schema
  // View = identity (lanes 20..36), ShadowView = identity (lanes 128..144)
  for (const base of [20, 128]) {
    header[base + 0] = 1;
    header[base + 5] = 1;
    header[base + 10] = 1;
    header[base + 15] = 1;
  }
  // Forward-Z perspective projection (fov=90deg, aspect=160/120, near=0.1, far=100) at lanes 36..52
  const aspect = 160 / 120;
  const f = 1.0;
  const near = 0.1;
  const far = 100.0;
  header[36 + 0] = f / aspect;
  header[36 + 5] = f;
  header[36 + 10] = far / (near - far);
  header[36 + 11] = -1.0;
  header[36 + 14] = (far * near) / (near - far);
  header[55] = 1.0; // clear.a
  header[60] = -1.0; // LIGHT_DIRECTION.y (nonzero)
  header[67] = 1.0; // msaa = 1
  header[69] = 1.0; // scale = 1.0
  header[84] = 1.0; // SKY_SUN.y (nonzero)
  header[127] = 1.0; // FOG_HEIGHT > 0
  header[187] = 0.05; // EXPOSURE_LOW
  header[188] = 0.95; // EXPOSURE_HIGH
  header[190] = -8.0; // EXPOSURE_RANGE_LOW
  header[191] = 8.0; // EXPOSURE_RANGE_HIGH
  header[192] = 0.18; // EXPOSURE_TARGET
  header[198] = 1.0; // GRADE_CONTRAST
  header[199] = 0.18; // GRADE_PIVOT
  header[200] = 1.0; // GRADE_SATURATION
  header[205] = 1.0; header[206] = 1.0; header[207] = 1.0; // GRADE_GAMMA
  header[208] = 1.0; header[209] = 1.0; header[210] = 1.0; // GRADE_GAIN
  header[229] = 1.0; // FOG_NOISE_SCALE > 0
  header[247] = 1.0; // REVERB_DECAY
  header[248] = 0.5; // REVERB_DAMPING

  // UI nodes:
  // 0: Button "Play" (checked toggles to TRUE on click; expanded set to TRUE when LocaleService answers RTL!)
  // 1: Slider "Volume" (value starts 50, increments by +10 on ArrowRight!)
  // 2: List "Difficulty" (promoted to role="listbox" by child focusable listitem)
  // 3: ListItem "Hard" (promoted to role="option", aria-selected="true")
  const n0 = uiNodeLanes({
    id: 1, role: 3, x: 0.05, y: 0.05, w: 0.25, h: 0.12,
    focusable: 1, checked: 1, actions: (1 << 0) | (1 << 2), name: 'Play',
  });
  const n1 = uiNodeLanes({
    id: 2, role: 6, x: 0.05, y: 0.22, w: 0.35, h: 0.12,
    focusable: 1, value: 50, min: 0, max: 100, step: 10,
    actions: (1 << 0) | (1 << 3) | (1 << 4), name: 'Volume',
  });
  const n2 = uiNodeLanes({
    id: 3, role: 8, x: 0.05, y: 0.42, w: 0.35, h: 0.35, name: 'Difficulty',
  });
  const n3 = uiNodeLanes({
    id: 4, parent: 3, role: 9, x: 0.06, y: 0.46, w: 0.30, h: 0.12,
    focusable: 1, selected: 2, actions: (1 << 0) | (1 << 2), name: 'Hard',
  });
  const nodes = [
    n0, n1, n2, n3,
    uiNodeLanes({ id: 5, role: 0, name: 'Dialog holder' }),
    uiNodeLanes({ id: 6, parent: 5, role: 10, hidden: 1, modal: 1, name: 'Confirm' }),
    uiNodeLanes({ id: 7, parent: 6, role: 3, focusable: 1, focused: 1, actions: 1 << 2, name: 'Exit' }),
    uiNodeLanes({ id: 8, parent: 6, role: 0, disabled: 1, name: 'Disabled group' }),
    uiNodeLanes({ id: 9, parent: 8, role: 3, focusable: 1, actions: 1 << 2, name: 'Unavailable' }),
    uiNodeLanes({ id: 10, role: 8, name: 'News' }),
    uiNodeLanes({ id: 11, parent: 10, role: 9, focusable: 1, hidden: 1, name: 'Unpublished' }),
    uiNodeLanes({ id: 12, role: 9, focusable: 1, name: 'Orphan' }),
  ];
  const ui = nodes.flat();
  header[159] = nodes.length; // UI_COUNT
  header[160] = ui.length; // UI_LANES

  const sounds = includeLocaleRequest ? [0, 0, 0, 0, 1, 0, 0] : [];
  header[7] = sounds.length ? 1 : 0;
  header[175] = 1; // SHOW_CAPTIONS

  const keyCps = [...'greeting'].map(c => c.codePointAt(0));
  const services = includeLocaleRequest
    ? [1, 12, 0, 0, 0, 0, 0, keyCps.length, ...keyCps] // req=1, service::LOCALE(12), locale_op::TEXT(0)
    : [];
  header[161] = includeLocaleRequest ? 1 : 0; // SERVICE_COUNT
  header[162] = services.length; // SERVICE_LANES

  // Extensions (proves zero collision on 65004 + live WebGPU 65006, 65008, 65009):
  const extAcoustics = [1.0, 0.85, 0.35, 0.25, 0.5, 0.0];
  const extLines = [
    0.3, 0.0, // xray alpha, desaturate
    10.0, 14.0, 0.0, 8.0, // p0=(10,14), width=8px
    150.0, 14.0, 0.0, 0.0, // p1=(150,14), solid
    1.0, 0.95, 0.0, 1.0, // bright yellow
    0.0, 1.0, 2.0, 1.0, // gap, feather, OnTop, is_2d=1
  ];
  const extSdf = [
    112.0, 82.0, 40.0, 30.0, // rect (bottom-right card)
    6.0, 6.0, 6.0, 6.0, // corner_radii
    0.05, 0.35, 1.0, 1.0, // fill_color (bright blue)
    0.05, 0.35, 1.0, 1.0, // fill_color_end
    1.0, 1.0, 1.0, 1.0, // border_color
    0.0, 0.0, 0.0, 0.5, // shadow_color
    16.0, 2.0, 0.0, 0.0, // kind=16 (overlay RoundedRect), border_width=2
    0.0, 2.0, 3.0, 0.0, // shadow_off, blur, clip_r
    0.0, 0.0, 0.0, 0.0, // clip_rect
  ];
  const extWorld2d = [
    0.0, 0.0, 1.0, 0.0, // camera center(0,0), zoom=1, rot=0
    1.0, 0.0, 2.0, 1.0, // 1 tilemap, 0 sprites, 2 RC emitters, flags=1 (opaque RC2D)
    2.0, 4.0, 0.0, 0.0, 1.0, 0.0, // RC2D: spacing=2, dirs=4, auto cascades, ambient=0, mul=1, add=0
    // Tilemap (15 header + 4 cells): white mat=-1, covers [-80..80] x [-60..60]
    -1.0, -80.0, 60.0, 80.0, 60.0, 2.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0.95, 0.95, 0.95, 1.0,
    1.0, 1.0, 1.0, 1.0,
    // RC2D emitters: warm orange light on left (28,66) + opaque shadow wall center (74,24,14,88)
    0.0, 28.0, 66.0, 12.0, 0.0, 6.0, 3.2, 0.4, 1.0,
    1.0, 74.0, 24.0, 14.0, 88.0, 0.0, 0.0, 0.0, 1.0,
  ];
  const extensions = [
    EXTENSIONS_MARKER, 4,
    65004, extAcoustics.length, ...extAcoustics,
    65006, extLines.length, ...extLines,
    65008, extSdf.length, ...extSdf,
    65009, extWorld2d.length, ...extWorld2d,
  ];

  // Offsets within the RenderFrame payload (after the leading protocol op=2 lane):
  const n0CheckedLane = 266 + 12;
  const n0ExpandedLane = 266 + 13;
  const n1ValueLane = 266 + n0.length + 16;

  return {
    packet: [...header, ...sounds, ...ui, ...services, ...extensions],
    n0CheckedLane,
    n0ExpandedLane,
    n1ValueLane,
    modalHiddenLane: 266 + nodes.slice(0, 5).flat().length + 10,
  };
}

function buildGuestWasm() {
  const HANDSHAKE = 2048;
  const CREATE = 2112;
  const SNAPSHOT = 2176;
  const RESTORE = 2240;
  const FRAME_FIRST = 4096;
  const FRAME_STEADY = 12288;

  const first = buildRenderFramePacket({ includeLocaleRequest: true });
  const steady = buildRenderFramePacket({ includeLocaleRequest: false });
  const firstReply = [2.0, ...first.packet];
  const steadyReply = [2.0, ...steady.packet];

  // Byte addresses of mutable lanes inside FRAME_STEADY (header u32 len + op lane = +8 bytes)
  const laneAddr = (base, lane) => base + 8 + lane * 4;
  const checkedAddr = laneAddr(FRAME_STEADY, steady.n0CheckedLane);
  const expandedAddr = laneAddr(FRAME_STEADY, steady.n0ExpandedLane);
  const sliderValAddr = laneAddr(FRAME_STEADY, steady.n1ValueLane);
  const modalHiddenAddr = laneAddr(FRAME_STEADY, steady.modalHiddenLane);
  const exitAddr = laneAddr(FRAME_STEADY, 19);
  const requestArena = 32768;
  const resetRequestArena = [...i32Const(requestArena), 0x24, 0];

  // Scratch slots in low linear memory:
  // 0: req ptr, 4: sent_locale flag, 8: loop i, 12: event_count, 16: event ptr, 20: reply ptr
  // The guest has consumed the previous request when it publishes its reply.
  const exchange = [...resetRequestArena, ...i32Const(0), 0x20, 0, ...i32Load(20), 0x10, 0, 0x36, 2, 0];
  const reqF32 = off => [...i32Load(0), 0x2a, 2, off];
  const reqI32 = off => [...reqF32(off), 0xa8]; // i32.trunc_f32_s

  const main = [
    ...i32Store(20, i32Const(HANDSHAKE)),
    ...exchange,
    // loop
    0x03, 0x40,
      // if op == 0 (Stop): return 0
      ...reqI32(8), 0x45, 0x04, 0x40, ...i32Const(0), 0x0f, 0x0b,
      // op == 1 (Create)
      ...reqI32(8), ...i32Const(1), 0x46, 0x04, 0x40,
        ...i32Store(20, i32Const(CREATE)),
        ...exchange,
        0x0c, 1,
      0x0b,
      // op == 3 (Snapshot)
      ...reqI32(8), ...i32Const(3), 0x46, 0x04, 0x40,
        ...i32Store(20, i32Const(SNAPSHOT)),
        ...exchange,
        0x0c, 1,
      0x0b,
      // op == 4 (Restore)
      ...reqI32(8), ...i32Const(4), 0x46, 0x04, 0x40,
        ...i32Store(20, i32Const(RESTORE)),
        ...exchange,
        0x0c, 1,
      0x0b,
      // op == 2 (Frame): scan events at req + 32 + i * 20
      ...i32Store(8, i32Const(0)),
      ...i32Store(12, reqI32(24)),
      0x02, 0x40, 0x03, 0x40,
        ...i32Load(8), ...i32Load(12), 0x4f, 0x0d, 1, // break if i >= event_count
        // evt_ptr = req + 32 + i * 20
        ...i32Store(16, [...i32Load(0), ...i32Const(32), 0x6a, ...i32Load(8), ...i32Const(20), 0x6c, 0x6a]),
        // tag == 12 (UI_ACTION): [12, action, node, value, 0]
        ...i32Load(16), 0x2a, 2, 0, 0xa8, ...i32Const(12), 0x46, 0x04, 0x40,
          // Play opens the modal; its Exit button requests shutdown.
          ...i32Load(16), 0x2a, 2, 4, 0xa8, ...i32Const(2), 0x46, 0x04, 0x40,
            ...i32Load(16), 0x2a, 2, 8, 0xa8, ...i32Const(1), 0x46, 0x04, 0x40,
              ...i32Const(checkedAddr), ...f32Const(2.0), 0x38, 2, 0,
              ...i32Const(modalHiddenAddr), ...f32Const(0.0), 0x38, 2, 0,
            0x0b,
            ...i32Load(16), 0x2a, 2, 8, 0xa8, ...i32Const(7), 0x46, 0x04, 0x40,
              ...i32Const(exitAddr), ...f32Const(1.0), 0x38, 2, 0,
            0x0b,
          0x0b,
          // action == 3 (INCREMENT) && node == 2 (Volume slider) -> value += 10.0
          ...i32Load(16), 0x2a, 2, 4, 0xa8, ...i32Const(3), 0x46, 0x04, 0x40,
            ...i32Load(16), 0x2a, 2, 8, 0xa8, ...i32Const(2), 0x46, 0x04, 0x40,
              ...i32Const(sliderValAddr),
              ...i32Const(sliderValAddr), 0x2a, 2, 0,
              ...f32Const(10.0), 0x92, // f32.add
              0x38, 2, 0,
            0x0b,
          0x0b,
        0x0b,
        // tag == 13 (SERVICE_ANSWER): [13, status, request, value, length]
        ...i32Load(16), 0x2a, 2, 0, 0xa8, ...i32Const(13), 0x46, 0x04, 0x40,
          ...i32Load(16), 0x2a, 2, 4, 0xa8, 0x45, 0x04, 0x40, // status == 0 (OK)
            ...i32Load(16), 0x2a, 2, 8, 0xa8, ...i32Const(1), 0x46, 0x04, 0x40, // request == 1
              ...i32Load(16), 0x2a, 2, 12, 0xa8, ...i32Const(1), 0x46, 0x04, 0x40, // value == 1 (RTL)
                ...i32Const(expandedAddr), ...f32Const(2.0), 0x38, 2, 0,
              0x0b,
            0x0b,
          0x0b,
        0x0b,
        ...i32Store(8, [...i32Load(8), ...i32Const(1), 0x6a]),
        0x0c, 0,
      0x0b, 0x0b,
      // Reply FRAME_FIRST once when elapsed > 0 and sent_locale == 0, else FRAME_STEADY
      ...i32Store(20, i32Const(FRAME_STEADY)),
      ...i32Load(4), 0x45, 0x04, 0x40,
        ...reqF32(12), ...f32Const(0.0), 0x5e, 0x04, 0x40, // elapsed > 0
          ...i32Store(4, i32Const(1)),
          ...i32Store(20, i32Const(FRAME_FIRST)),
        0x0b,
      0x0b,
      ...exchange,
      0x0c, 0,
    0x0b,
    ...i32Const(0),
  ];

  const types = vector([
    [0x60, 2, 0x6f, 0x7f, 1, 0x7f],
    [0x60, 1, 0x6f, 1, 0x7f],
    [0x60, 1, 0x7f, 1, 0x7f],
  ]);
  const imports = vector([[...text('squid:host/1'), ...text('call_array_f32_array_f32'), 0, 0]]);
  const functions = vector([[1], [2], [2]]);
  const memory = vector([[0, 2]]); // 128 KiB
  const globals = vector([[0x7f, 1, ...i32Const(requestArena), 0x0b]]);
  const exports = vector([
    [...text('main'), 0, 1],
    [...text('squid:memory'), 2, 0],
    [...text('squid:allocate'), 0, 2],
    [...text('squid:reset'), 0, 3],
  ]);
  const body = code => {
    const bytes = [...vector([]), ...code, 0x0b];
    return [...uleb(bytes.length), ...bytes];
  };
  const code = vector([
    body(main),
    // allocate: bump global 0, aligned to 4
    body([0x23, 0, 0x23, 0, 0x20, 0, 0x41, 3, 0x6a, 0x41, ...sleb(-4), 0x71, 0x6a, 0x24, 0]),
    body([...resetRequestArena, ...i32Const(requestArena)]),
  ]);
  const seg = (addr, floats) => {
    const bytes = encodeFloatArray(floats);
    return [0, ...i32Const(addr), 0x0b, ...uleb(bytes.length), ...bytes];
  };
  const data = vector([
    seg(HANDSHAKE, [4670544.0, 2.0, 1.0, 0.0]),
    seg(CREATE, [1.0, 0.0]),
    seg(SNAPSHOT, [3.0, 0.0]),
    seg(RESTORE, [4.0, 0.0]),
    seg(FRAME_FIRST, firstReply),
    seg(FRAME_STEADY, steadyReply),
  ]);
  const manifest = [8, 1, ...text('main'), 4, 6, 6, 0, 0];
  return new Uint8Array([
    0, 0x61, 0x73, 0x6d, 1, 0, 0, 0,
    ...custom('squid:abi', manifest),
    ...section(1, types),
    ...section(2, imports),
    ...section(3, functions),
    ...section(5, memory),
    ...section(6, globals),
    ...section(7, exports),
    ...section(10, code),
    ...section(11, data),
  ]);
}

// Exercise thousands of exchanges without waiting for browser animation frames.
async function verifyRequestArena(bytes) {
  let exchanged = 0;
  const frames = 4096;
  const { instance } = await WebAssembly.instantiate(bytes, {
    'squid:host/1': {
      call_array_f32_array_f32() {
        const memory = instance.exports['squid:memory'].buffer;
        const pointer = instance.exports['squid:allocate'](256);
        const request = new Float32Array(memory, pointer + 4, 63);
        new DataView(memory).setUint32(pointer, request.length, true);
        request.fill(0);
        request[1] = exchanged++ < frames ? 2 : 0;
        return pointer;
      },
    },
  });
  instance.exports.main(null);
  assert.equal(exchanged, frames + 1);
  assert.equal(instance.exports['squid:memory'].buffer.byteLength, 128 * 1024);
}

const guestBytes = buildGuestWasm();
await verifyRequestArena(guestBytes);
const hostFiles = Object.fromEntries(
  ['pen_web.js', 'pen_web_bg.wasm', 'guest.js', 'structured.js', 'kernels.js', 'kernel_worker.js']
    .map(name => [`/${name}`, join(hostDir, name)]),
);
const server = await serve(hostDir, { isolated: true, files: hostFiles });
const browser = await launch({ gpu: true });

try {
  const context = await browser.newContext({ viewport: { width: 160, height: 120 } });
  await hide(context);
  const page = await context.newPage();
  const consoleErrors = [];
  const pageErrors = [];
  const localeWarnings = [];
  page.on('console', msg => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
      console.error('[browser console.error]', msg.text());
    } else if (msg.type() === 'warning' && msg.text().includes("the game's string tables")) {
      localeWarnings.push(msg.text());
    } else if (process.env.PEN_WEB_VERBOSE) {
      console.log(`[browser ${msg.type()}]`, msg.text());
    }
  });
  page.on('pageerror', err => {
    pageErrors.push(String(err));
    console.error('[browser pageerror]', String(err));
  });

  await page.route('**/catalog.json', route => route.fulfill({ json: { meshes: [], materials: [], sounds: ['chime.wav'] } }));
  await page.route('**/game.json', route => route.fulfill({ json: {
    name: 'render-expansion', title: 'Render Expansion Check', web: { saved: true, captions: true, locales: true },
  } }));
  await page.route('**/game.wasm', route => route.fulfill({ body: Buffer.from(guestBytes), contentType: 'application/wasm' }));
  await page.route('**/assets/saved.json', route => route.fulfill({ json: { format: 'opaque' } }));
  await page.route('**/assets/captions.json', route => route.fulfill({
    json: { 'chime.wav': { text: 'Chime', seconds: 120 } },
  }));
  await page.route('**/assets/chime.wav', route => route.fulfill({
    body: Buffer.from('524946462600000057415645666d74201000000001000100401f0000401f00000100080064617461020000008080', 'hex'),
    contentType: 'audio/wav',
  }));
  await page.route('**/text.json', route => route.fulfill({ json: { greeting: 'Hello' } }));
  let revision = 1;
  let expectRevisionPoll;
  const revisionPolled = new Promise(resolve => { expectRevisionPoll = resolve; });
  let arabicAvailable = true;
  await page.route('**/__pen/revision', route => {
    return route.fulfill({ body: String(revision) }).then(expectRevisionPoll);
  });
  await page.route('**/locales/index.json', route => route.fulfill({ json: ['ar', 'sv', 'de'] }));
  await page.route('**/locales/ar.json', route => arabicAvailable
    ? route.fulfill({ json: { greeting: 'مرحبا' } })
    : route.fulfill({ status: 404 }));
  await page.route('**/locales/sv.json', route => route.fulfill({ body: '{' }));
  await page.route('**/locales/de.json', route => route.fulfill({ status: 404 }));
  await page.route('**/test.html*', route => route.fulfill({
    contentType: 'text/html; charset=utf-8',
    headers: { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' },
    body: `<!doctype html>
      <style>html,body{margin:0;padding:0;background:#000}canvas{display:block;width:160px;height:120px}</style>
      <canvas width="160" height="120" tabindex="0"></canvas>
      <div id="status">Starting</div>
      <script type="module">
        import init, { start, qualify_platform } from './pen_web.js';
        const status = document.querySelector('#status');
        try {
          await init();
          window.__platformResult = await qualify_platform();
          const canvas = document.querySelector('canvas');
          window.__startGame = options => start(canvas, { qualify: true, ...options });
          await window.__startGame();
          window.__started = 'ready';
        } catch (err) {
          window.__started = 'ERR:' + String(err);
          throw err;
        }
      </script>`,
  }));

  // Load with ?lang=ar to exercise Arabic locale selection and direction
  await page.goto(`${server.url}test.html?lang=ar`);
  await page.waitForFunction(() => Boolean(window.__started), null, { timeout: 30000 });
  const statusText = await page.evaluate(() => window.__started);
  assert.equal(statusText, 'ready', `page startup failed: ${statusText}`);
  await page.waitForFunction(
    () => {
      const c = document.querySelector('canvas');
      return c?.getAttribute('aria-busy') === 'false' && Number(c?.getAttribute('data-frame') || '0') >= 4;
    },
    null,
    { timeout: 30000 },
  );

  // 1) Verify WebAudio + IndexedDB + Noise conformance (`qualify_platform`)
  const platformMsg = await page.evaluate(() => window.__platformResult);
  assert.match(platformMsg, /WebAudio graph passed/, `unexpected platform result: ${platformMsg}`);

  // 2) Verify LocaleService RTL (`dir="rtl"` on accessibility root + guest received RTL answer!)
  const a11y = page.locator('[aria-label="Game interface"]');
  const a11yDir = await a11y.getAttribute('dir');
  assert.equal(await a11y.getAttribute('lang'), 'ar');
  assert.equal(await page.locator('[aria-label="Captions"]').textContent(), 'Chime');
  for (const path of ['locales/sv.json', 'locales/de.json']) {
    assert.ok(localeWarnings.some(warning => warning.includes(path)), localeWarnings.join('\n'));
  }
  assert.equal(a11yDir, 'rtl', 'Arabic locale must set dir="rtl" on browser accessibility root');
  await page.waitForFunction(
    () => document.querySelector('[role="button"][aria-label="Play"]')?.getAttribute('aria-expanded') === 'true',
    null,
    { timeout: 15000 },
  );

  // 3) Verify ARIA role projection & live DOM action round-trip
  const listbox = page.locator('[role="listbox"][aria-label="Difficulty"]');
  assert.equal(await listbox.count(), 1, 'LIST containing focusable LISTITEM must promote to role="listbox"');
  const option = listbox.locator('[role="option"][aria-label="Hard"]');
  assert.equal(await option.getAttribute('aria-selected'), 'true');

  assert.equal(await page.locator('[aria-label="News"]').getAttribute('role'), 'list');
  assert.equal(await page.locator('[aria-label="Orphan"]').getAttribute('role'), 'listitem');

  const slider = page.locator('[role="slider"][aria-label="Volume"]');
  assert.equal(await slider.getAttribute('aria-valuenow'), '50');
  await slider.evaluate(el => el.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true })));
  await page.waitForFunction(
    () => document.querySelector('[role="slider"][aria-label="Volume"]')?.getAttribute('aria-valuenow') === '60',
    null,
    { timeout: 15000 },
  );

  const button = page.locator('[role="button"][aria-label="Play"]');
  assert.equal(await button.getAttribute('aria-checked'), 'false');
  await button.evaluate(el => {
    el.focus();
    el.click();
  });
  await page.waitForFunction(
    () => document.querySelector('[role="button"][aria-label="Play"]')?.getAttribute('aria-checked') === 'true',
    null,
    { timeout: 15000 },
  );

  const dialog = page.locator('[role="dialog"][aria-label="Confirm"]');
  await dialog.waitFor({ state: 'attached' });
  assert.equal(await page.evaluate(() => document.activeElement?.getAttribute('aria-label')), 'Exit');
  assert.equal(await dialog.getAttribute('aria-modal'), 'true');
  assert.equal(await button.getAttribute('inert'), '');
  const unavailable = dialog.locator('[aria-label="Unavailable"]');
  assert.equal(await unavailable.getAttribute('aria-disabled'), 'true');
  await unavailable.evaluate(el => el.click());
  const blockedAt = await slider.evaluate(el => {
    el.focus();
    el.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    return Number(document.querySelector('canvas').dataset.frame);
  });
  await page.waitForFunction(frame => Number(document.querySelector('canvas').dataset.frame) > frame + 2, blockedAt);
  assert.equal(await slider.getAttribute('aria-valuenow'), '60');
  assert.equal(await page.evaluate(() => document.activeElement?.getAttribute('aria-label')), 'Exit');
  assert.ok(await page.evaluate(() => !JSON.parse(document.querySelector('canvas').dataset.events)
    .some(([tag, , id]) => tag === 12 && id === 9)));

  // 4) Verify REAL WebGPU rendered canvas pixels for 65006 (lines), 65008 (SDF UI), and 65009 (2D World + RC2D GI)!
  //    Wait 2 extra frames so the swapchain presents the latest state cleanly, then sample the PNG screenshot.
  const shotPng = await page.locator('canvas').screenshot({ type: 'png' });
  const pixels = await page.evaluate(async b64 => {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const bmp = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
    const c = new OffscreenCanvas(bmp.width, bmp.height);
    const ctx = c.getContext('2d');
    ctx.drawImage(bmp, 0, 0);
    const img = ctx.getImageData(0, 0, bmp.width, bmp.height);
    const at = (x, y) => {
      const i = (y * bmp.width + x) * 4;
      return [img.data[i], img.data[i + 1], img.data[i + 2], img.data[i + 3]];
    };
    return {
      width: bmp.width,
      height: bmp.height,
      lineYellow: at(80, 14),
      rcLitLeft: at(48, 66),
      rcShadowRight: at(108, 66),
      sdfBlueCard: at(132, 96),
    };
  }, shotPng.toString('base64'));

  assert.ok(
    pixels.lineYellow[0] > 70 && pixels.lineYellow[1] > 70 && pixels.lineYellow[2] < 35 && pixels.lineYellow[0] > pixels.lineYellow[2] * 4,
    `LINES_EXT_TAG (65006) yellow 2D line missing in WebGPU canvas: ${JSON.stringify(pixels)}`,
  );
  assert.ok(
    pixels.rcLitLeft[0] > 70 && pixels.rcShadowRight[0] < 25 && pixels.rcLitLeft[0] > pixels.rcShadowRight[0] * 4,
    `RENDER2D_WORLD_EXT_TAG (65009) 2D RC GI light/shadow missing in WebGPU canvas: ${JSON.stringify(pixels)}`,
  );
  assert.ok(
    pixels.sdfBlueCard[2] > 70 && pixels.sdfBlueCard[0] < 40 && pixels.sdfBlueCard[2] > pixels.sdfBlueCard[0] * 4,
    `RENDER2D_EXT_TAG (65008) blue SDF UI card missing in WebGPU canvas: ${JSON.stringify(pixels)}`,
  );

  // Shutdown removes both DOM regions even if a caller retains an old control.
  await dialog.locator('[aria-label="Exit"]').evaluate(el => {
    window.__oldExit = el;
    el.click();
  });
  await a11y.waitFor({ state: 'detached' });
  assert.equal(await page.locator('[aria-label="Captions"]').count(), 0);
  await page.evaluate(() => window.__startGame({ development: true }));
  await page.locator('#pen-ui-1').waitFor({ state: 'attached' });
  assert.ok(await page.evaluate(() => {
    let bubbled = false;
    window.__oldExit.parentElement.addEventListener('click', () => { bubbled = true; }, { once: true });
    window.__oldExit.click();
    return bubbled;
  }));
  assert.equal(await page.locator('#pen-ui-1').count(), 1);
  assert.equal(await a11y.count(), 1);

  // A new revision resolves the remaining base locale when Arabic is missing.
  await revisionPolled;
  arabicAvailable = false;
  revision += 1;
  await page.waitForFunction(() => document.querySelector('[aria-label="Game interface"]')?.lang === 'en');
  assert.equal(await a11y.getAttribute('dir'), 'ltr');

  const unexpectedConsoleErrors = consoleErrors.filter(
    msg => !msg.includes('Failed to load resource: the server responded with a status of 404'),
  );
  const unexpectedPageErrors = pageErrors.filter(
    err => !err.includes('The play() request was interrupted by a call to pause()'),
  );
  assert.deepEqual(unexpectedPageErrors, [], `unexpected pageerrors: ${JSON.stringify(unexpectedPageErrors)}`);
  assert.deepEqual(unexpectedConsoleErrors, [], `unexpected console errors: ${JSON.stringify(unexpectedConsoleErrors)}`);

  console.log(JSON.stringify({
    platform: platformMsg,
    a11yDir,
    ariaVerified: { listbox: true, buttonActivated: true, sliderValue: 60, localeRtlAnswered: true },
    webgpuPixels: pixels,
  }, null, 2));
} finally {
  await browser.close();
  await server.close();
}
