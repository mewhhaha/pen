// The only JS runtime seam: JSPI suspends a guest's Wasm stack at host calls.
// Rust owns game requests, validation (the blot:abi manifest included),
// commits and every platform operation. `ownership` is the manifest's ABI 3
// flag: main runs between blot:ownership_enter and blot:ownership_leave.
export async function openGuest(bytes, ownership) {
  if (!WebAssembly.Suspending || !WebAssembly.promising) {
    throw new Error('This browser needs WebAssembly JS Promise Integration (JSPI).');
  }
  const module = await WebAssembly.compile(bytes);
  const capability = Object.freeze({});
  let exports;
  let incoming;
  let response = Promise.withResolvers();
  let failure;
  let stopping = false;
  const range = (pointer, length) => {
    const address = pointer >>> 0;
    if (address < 4 || address % 4 || address > exports['blot:memory'].buffer.byteLength - 4
      || length > (exports['blot:memory'].buffer.byteLength - address - 4) / 4) throw new Error('array outside guest arena');
    return address;
  };
  const exchange = new WebAssembly.Suspending(async (reference, pointer) => {
    if (reference !== capability) throw new Error('stale guest capability');
    const address = range(pointer, 0);
    const length = new DataView(exports['blot:memory'].buffer).getUint32(address, true);
    range(address, length);
    const packet = new Float32Array(exports['blot:memory'].buffer, address + 4, length).slice();
    incoming = Promise.withResolvers();
    response.resolve(packet);
    const values = await incoming.promise;
    const target = range(exports['blot:allocate']((values.length + 1) * 4), values.length);
    new DataView(exports['blot:memory'].buffer).setUint32(target, values.length, true);
    new Float32Array(exports['blot:memory'].buffer, target + 4, values.length).set(values);
    return target;
  });
  const instance = await WebAssembly.instantiate(module, { 'blot:host/1': { call_array_f32_array_f32: exchange } });
  exports = instance.exports;
  if (!(exports['blot:memory'] instanceof WebAssembly.Memory)
    || typeof exports['blot:allocate'] !== 'function' || typeof exports['blot:reset'] !== 'function') throw new Error('missing guest array arena');
  const enter = exports['blot:ownership_enter'];
  const leave = exports['blot:ownership_leave'];
  if (ownership && (typeof enter !== 'function' || enter.length !== 0
    || typeof leave !== 'function' || leave.length !== 1)) throw new Error('missing guest ownership hooks');
  if (ownership) enter();
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
      if (ownership) leave(outcome);
      exports['blot:reset'](0);
    }
  };
}
