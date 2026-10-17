// guest.js, the browser's JSPI membrane, against a hand-built ABI 3 game:
// its ownership hooks run around main and tear the instance down on close
// (blot:ownership_leave(2)), and a trap's stack has the Wasm frames, named
// from the module's name section, that pen-runtime's symbols read.
// No GPU: `node --experimental-wasm-jspi crates/pen-web/tests/guest.mjs`.
import assert from 'node:assert/strict';
import { openGuest } from '../guest.js';

const leb = value => {
  const bytes = [];
  do {
    let byte = value & 0x7f;
    value >>>= 7;
    if (value) byte |= 0x80;
    bytes.push(byte);
  } while (value);
  return bytes;
};
const text = string => [...leb(string.length), ...new TextEncoder().encode(string)];
const section = (id, contents) => [id, ...leb(contents.length), ...contents];
const vector = items => [...leb(items.length), ...items.flat()];
const custom = (name, payload) => section(0, [...text(name), ...payload]);
const f32 = value => [...new Uint8Array(new Float32Array([value]).buffer)];
const i32 = value => [...new Uint8Array(new Int32Array([value]).buffer)];

// Functions: 0 the host import, 1 main, 2 blot:allocate, 3 blot:reset,
// 4 blot:ownership_enter, 5 blot:ownership_leave, 6 explode. Global 0 is
// the hooks' log (enter appends 1, leave 2 plus its outcome), 1 the arena's
// bump pointer. main sends the handshake at 16, then stops at a Stop
// request (operation lane 0) and calls explode at any other.
function game(names) {
  const types = vector([
    [0x60, 2, 0x6f, 0x7f, 1, 0x7f],
    [0x60, 1, 0x6f, 1, 0x7f],
    [0x60, 1, 0x7f, 1, 0x7f],
    [0x60, 0, 0],
    [0x60, 1, 0x7f, 0],
  ]);
  const imports = vector([[...text('blot:host/1'), ...text('call_array_f32_array_f32'), 0, 0]]);
  const functions = vector([[1], [2], [2], [3], [4], [3]]);
  const memory = vector([[0, 1]]);
  const globals = vector([[0x7f, 1, 0x41, 0, 0x0b], [0x7f, 1, 0x41, ...leb(1024), 0x0b]]);
  const exports = vector([
    [...text('main'), 0, 1],
    [...text('blot:memory'), 2, 0],
    [...text('blot:allocate'), 0, 2],
    [...text('blot:reset'), 0, 3],
    [...text('blot:ownership_enter'), 0, 4],
    [...text('blot:ownership_leave'), 0, 5],
    [...text('log'), 3, 0],
  ]);
  const body = (code, locals = []) => {
    const bytes = [...vector(locals), ...code, 0x0b];
    return [...leb(bytes.length), ...bytes];
  };
  const appendToLog = digit => [0x23, 0, 0x41, 10, 0x6c, ...digit, 0x6a, 0x24, 0];
  const code = vector([
    // main: the reply's operation lane (after its length and commit lanes).
    body([0x20, 0, 0x41, 16, 0x10, 0, 0x2a, 2, 8, 0x43, ...f32(0), 0x5c, 0x04, 0x40, 0x10, 6, 0x0b, 0x41, 0]),
    // allocate: the bump pointer, moved on by the byte count, rounded to 4.
    body([0x23, 1, 0x23, 1, 0x20, 0, 0x41, 3, 0x6a, 0x41, ...leb(-4 & 0x7f), 0x71, 0x6a, 0x24, 1]),
    body([0x41, ...leb(1024), 0x24, 1, 0x41, ...leb(1024)]),
    body(appendToLog([0x41, 1])),
    body(appendToLog([0x41, 2, 0x20, 0, 0x6a])),
    body([0x00]),
  ]);
  // The handshake: four lanes, the magic number, version 1, schema 0, no scenes.
  const handshake = [...i32(4), ...f32(4670544), ...f32(1), ...f32(0), ...f32(0)];
  const data = vector([[0, 0x41, 16, 0x0b, ...leb(handshake.length), ...handshake]]);
  const manifest = [3, 1, ...text('main'), 4, 6, 6, 0, 0];
  const sections = [
    ...custom('blot:abi', manifest),
    ...section(1, types),
    ...section(2, imports),
    ...section(3, functions),
    ...section(5, memory),
    ...section(6, globals),
    ...section(7, exports),
    ...section(10, code),
    ...section(11, data),
  ];
  if (names) {
    const functionNames = vector([[1, ...text('game.main')], [6, ...text('game.explode')]]);
    sections.push(...custom('name', [1, ...leb(functionNames.length), ...functionNames]));
  }
  return new Uint8Array([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0, ...sections]);
}

// The log as a number, read through a second instance's view of nothing:
// guest.js keeps its exports, so the hooks' log is read off the module's
// export instead, through a patched WebAssembly.instantiate.
let instances = [];
const instantiate = WebAssembly.instantiate;
WebAssembly.instantiate = async (...args) => {
  const instance = await instantiate(...args);
  instances.push(instance);
  return instance;
};
const log = () => instances.at(-1).exports.log.value;

// A game that stops when asked: entered once, left normally, torn down.
{
  const guest = await openGuest(game(false), true);
  assert.deepEqual([...guest.description], [4670544, 1, 0, 0]);
  assert.equal(log(), 1, 'main runs inside ownership');
  await guest.close();
  assert.equal(log(), 124, 'left normally, then torn down');
}

// A game that traps: left failing, then torn down; its stack names the
// functions the name section names, at offsets in the module.
{
  const guest = await openGuest(game(true), true);
  const failure = await guest.next(new Float32Array([1, 2])).then(() => null, error => error);
  assert.ok(failure instanceof WebAssembly.RuntimeError, String(failure));
  const frames = [...failure.stack.matchAll(/wasm-function\[(\d+)\]:0x([0-9a-f]+)/g)].map(match => Number(match[1]));
  assert.deepEqual(frames.slice(0, 2), [6, 1], failure.stack);
  assert.match(failure.stack, /at game\.explode \(wasm:\/\/wasm\/[^)]*wasm-function\[6\]:0x[0-9a-f]+\)/);
  await guest.close();
  assert.equal(log(), 134, 'left failing, then torn down');
}

// An ABI 2 game has no hooks to call.
{
  const guest = await openGuest(game(false), false);
  await guest.close();
  assert.equal(log(), 0);
}
console.log('guest.js: ownership hooks, teardown and trap stacks ok');
