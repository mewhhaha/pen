// The only JS runtime seam: a guest's Wasm stack waits at each host call.
// Rust owns game requests, validation (the blot:abi manifest included),
// commits and every platform operation. `ownership` is the manifest's ABI 3
// flag: main runs between blot:ownership_enter and blot:ownership_leave, and
// closing tears the instance down (blot:ownership_leave(2)).
//
// With WebAssembly JS Promise Integration (JSPI) the guest runs on the page's
// thread, suspended at each host call. Without it, a cross-origin isolated
// page runs the guest in a worker (this module again), which blocks at each
// host call on a SharedArrayBuffer until the page sends the next request.
import { openStructured, runStructuredSync } from './structured.js';

async function hasTasks(bytes) {
  const module = await WebAssembly.compile(bytes);
  return WebAssembly.Module.customSections(module, 'blot:tasks').length !== 0;
}
export async function openGuest(bytes, ownership) {
  if (WebAssembly.Suspending && WebAssembly.promising) {
    return await hasTasks(bytes) ? openStructured(bytes) : openSuspending(bytes, ownership);
  }
  if (globalThis.crossOriginIsolated && typeof SharedArrayBuffer === 'function') return openWorker(bytes, ownership);
  throw new Error('This browser needs WebAssembly JS Promise Integration (JSPI), or a cross-origin isolated page '
    + '(COOP and COEP headers; on itch.io, "SharedArrayBuffer support") to run the game in a worker.');
}

// Instantiates the guest with `exchange` as its host call and checks its
// array arena and ownership hooks.
async function instantiate(bytes, ownership, exchange) {
  const module = await WebAssembly.compile(bytes);
  const instance = await WebAssembly.instantiate(module, { 'blot:host/1': { call_array_f32_array_f32: exchange } });
  const exports = instance.exports;
  if (!(exports['blot:memory'] instanceof WebAssembly.Memory)
    || typeof exports['blot:allocate'] !== 'function' || typeof exports['blot:reset'] !== 'function') throw new Error('missing guest array arena');
  const enter = exports['blot:ownership_enter'];
  const leave = exports['blot:ownership_leave'];
  if (ownership && (typeof enter !== 'function' || enter.length !== 0
    || typeof leave !== 'function' || leave.length !== 1)) throw new Error('missing guest ownership hooks');
  return exports;
}

// The guest's arrays: the packet at `pointer` copied out, and `values`
// written into a new one, whose address is returned.
function arrays(exports) {
  const range = (pointer, length) => {
    const address = pointer >>> 0;
    if (address < 4 || address % 4 || address > exports['blot:memory'].buffer.byteLength - 4
      || length > (exports['blot:memory'].buffer.byteLength - address - 4) / 4) throw new Error('array outside guest arena');
    return address;
  };
  return {
    read(pointer) {
      const address = range(pointer, 0);
      const length = new DataView(exports['blot:memory'].buffer).getUint32(address, true);
      range(address, length);
      return new Float32Array(exports['blot:memory'].buffer, address + 4, length).slice();
    },
    write(values) {
      const target = range(exports['blot:allocate']((values.length + 1) * 4), values.length);
      new DataView(exports['blot:memory'].buffer).setUint32(target, values.length, true);
      new Float32Array(exports['blot:memory'].buffer, target + 4, values.length).set(values);
      return target;
    }
  };
}

async function openSuspending(bytes, ownership) {
  const capability = Object.freeze({});
  let guest;
  let incoming;
  let response = Promise.withResolvers();
  let failure;
  let stopping = false;
  const exchange = new WebAssembly.Suspending(async (reference, pointer) => {
    if (reference !== capability) throw new Error('stale guest capability');
    const packet = guest.read(pointer);
    incoming = Promise.withResolvers();
    response.resolve(packet);
    return guest.write(await incoming.promise);
  });
  const exports = await instantiate(bytes, ownership, exchange);
  guest = arrays(exports);
  if (ownership) exports['blot:ownership_enter']();
  exports['blot:reset'](0);
  const done = WebAssembly.promising(exports.main)(capability).then(() => {
    if (!stopping) throw new Error('game main returned before shutdown');
  });
  void done.catch(error => { failure = error; response.reject(error); });
  const description = await response.promise;
  return {
    description,
    memoryBytes() { return exports['blot:memory'].buffer.byteLength; },
    next(values) {
      if (failure) return Promise.reject(failure);
      response = Promise.withResolvers();
      incoming.resolve(values);
      return response.promise;
    },
    async close() {
      if (stopping) return;
      stopping = true;
      if (!failure) incoming.resolve(new Float32Array([0, 0]));
      let outcome = 0;
      try { await done; } catch { outcome = 1; /* Rust reports the failed request. */ }
      if (ownership) exports['blot:ownership_leave'](outcome);
      exports['blot:reset'](0);
      // Idle teardown, as Blot's adapter disposes of an instance: the module
      // releases its roots now rather than whenever the instance is collected.
      if (ownership) {
        exports['blot:ownership_leave'](2);
        exports['blot:reset'](0);
      }
    }
  };
}

// The page and a guest's worker share a channel: SIGNALS Int32 words, then
// CHUNK floats. The worker waits on the READY word while it is 0; the page
// writes a chunk of the next request (the request's length, the chunk's
// offset and count) and sets it to 1. A request longer than a chunk goes
// over in several, the worker asking for each next one.
const SIGNALS = 8;
const CHUNK = 1 << 20;
const READY = 0;
const TOTAL = 1;
const OFFSET = 2;
const COUNT = 3;
// Set before the last request, which shuts the guest down.
const STOPPING = 4;

async function openWorker(bytes, ownership) {
  const channel = new SharedArrayBuffer((SIGNALS + CHUNK) * 4);
  const signals = new Int32Array(channel, 0, SIGNALS);
  const data = new Float32Array(channel, SIGNALS * 4, CHUNK);
  const worker = new Worker(import.meta.url, { type: 'module', name: 'pen guest' });
  let response = Promise.withResolvers();
  const closed = Promise.withResolvers();
  let failure;
  let stopping = false;
  let memory = 0;
  // The rest of the request being sent: its values and where the next chunk starts.
  let sending;
  const fail = error => {
    failure ??= error;
    response.reject(failure);
    closed.resolve(1);
  };
  // Hands the worker the next chunk of the request.
  const send = () => {
    const { values, offset } = sending;
    const count = Math.min(CHUNK, values.length - offset);
    data.set(values.subarray(offset, offset + count));
    signals[TOTAL] = values.length;
    signals[OFFSET] = offset;
    signals[COUNT] = count;
    sending = { values, offset: offset + count };
    Atomics.store(signals, READY, 1);
    Atomics.notify(signals, READY);
  };
  worker.onmessage = ({ data: message }) => {
    if (message.packet) {
      memory = message.memory;
      response.resolve(message.packet);
    } else if (message.more) {
      send();
    } else if (message.failure !== undefined) {
      fail(new Error(message.failure));
    } else if (message.closed !== undefined) {
      closed.resolve(message.closed);
    }
  };
  worker.onerror = event => {
    event.preventDefault();
    fail(new Error(`guest worker: ${event.message || 'it did not start'}`));
  };
  worker.onmessageerror = () => fail(new Error('guest worker: a message could not be read'));
  worker.postMessage({ bytes, ownership, channel });
  let description;
  try {
    description = await response.promise;
  } catch (error) {
    worker.terminate();
    throw error;
  }
  return {
    description,
    memoryBytes() { return memory; },
    next(values) {
      if (failure) return Promise.reject(failure);
      response = Promise.withResolvers();
      sending = { values, offset: 0 };
      send();
      return response.promise;
    },
    async close() {
      if (stopping) return;
      stopping = true;
      if (!failure) {
        Atomics.store(signals, STOPPING, 1);
        sending = { values: new Float32Array([0, 0]), offset: 0 };
        send();
        await closed.promise;
      }
      worker.terminate();
    }
  };
}

// The worker's side: runs the guest, blocked at each host call until the
// page has sent the whole next request.
function serveWorker() {
  globalThis.onmessage = async ({ data: { bytes, ownership, channel } }) => {
    globalThis.onmessage = null;
    const signals = new Int32Array(channel, 0, SIGNALS);
    const data = new Float32Array(channel, SIGNALS * 4, CHUNK);
    const capability = Object.freeze({});
    let exports;
    let guest;
    const exchangeValues = (packet, memory) => {
      postMessage({ packet, memory }, [packet.buffer]);
      let values;
      for (;;) {
        Atomics.wait(signals, READY, 0);
        values ??= new Float32Array(signals[TOTAL]);
        const offset = signals[OFFSET];
        const count = signals[COUNT];
        values.set(data.subarray(0, count), offset);
        Atomics.store(signals, READY, 0);
        if (offset + count >= values.length) break;
        postMessage({ more: true });
      }
      return values;
    };
    const exchange = (reference, pointer) => {
      if (reference !== capability) throw new Error('stale guest capability');
      return guest.write(exchangeValues(guest.read(pointer), exports['blot:memory'].buffer.byteLength));
    };
    let outcome = 0;
    let structured = false;
    try {
      structured = await hasTasks(bytes);
      if (structured) {
        await runStructuredSync(bytes, exchangeValues);
      } else {
        exports = await instantiate(bytes, ownership, exchange);
        guest = arrays(exports);
        if (ownership) exports['blot:ownership_enter']();
        exports['blot:reset'](0);
        exports.main(capability);
      }
      if (!Atomics.load(signals, STOPPING)) throw new Error('game main returned before shutdown');
    } catch (error) {
      if (!Atomics.load(signals, STOPPING)) {
        postMessage({ failure: error instanceof Error ? `${error.name}: ${error.message}\n${error.stack ?? ''}` : String(error) });
        return;
      }
      outcome = 1;
    }
    if (structured) {
      postMessage({ closed: outcome });
      return;
    }
    if (ownership) exports['blot:ownership_leave'](outcome);
    exports['blot:reset'](0);
    // Idle teardown, as Blot's adapter disposes of an instance: the module
    // releases its roots now rather than whenever the instance is collected.
    if (ownership) {
      exports['blot:ownership_leave'](2);
      exports['blot:reset'](0);
    }
    postMessage({ closed: outcome });
  };
}

if (typeof WorkerGlobalScope === 'function' && globalThis instanceof WorkerGlobalScope) serveWorker();
