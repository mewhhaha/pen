// The only JS runtime seam: JSPI suspends a guest's Wasm stack at host calls.
// Rust owns game requests, validation, commits and every platform operation.
export async function openGuest(bytes) {
  if (!WebAssembly.Suspending || !WebAssembly.promising) {
    throw new Error('This browser needs WebAssembly JS Promise Integration (JSPI).');
  }
  const module = await WebAssembly.compile(bytes);
  const sections = WebAssembly.Module.customSections(module, 'blot:abi');
  if (sections.length !== 1) throw new Error('game needs one blot:abi manifest');
  const manifest = new Uint8Array(sections[0]);
  let at = 0;
  const byte = () => {
    if (at >= manifest.length) throw new Error('truncated guest ABI');
    return manifest[at++];
  };
  const number = () => {
    let value = 0;
    for (let shift = 0; shift <= 28; shift += 7) {
      const part = byte();
      value += (part & 127) * 2 ** shift;
      if (part < 128) return value;
    }
    throw new Error('invalid guest ABI integer');
  };
  const name = () => {
    const length = number();
    if (at + length > manifest.length) throw new Error('truncated guest ABI name');
    const text = new TextDecoder('utf-8', { fatal: true }).decode(manifest.slice(at, at + length));
    at += length;
    return text;
  };
  if (number() !== 2) throw new Error('unsupported guest ABI version');
  const count = number();
  let main = false;
  const names = new Set();
  for (let i = 0; i < count; i++) {
    const key = name();
    if (names.has(key)) throw new Error('duplicate guest ABI export');
    names.add(key);
    const parameter = byte();
    let input = -1;
    let output = -1;
    if (parameter === 4) { input = byte(); output = byte(); }
    const result = byte();
    if (key === 'main') main = parameter === 4 && input === 6 && output === 6 && result === 0;
  }
  for (let i = 0, constants = number(); i < constants; i++) { name(); byte(); }
  if (!main || at !== manifest.length) throw new Error('game needs main: (Array F32 -> Array F32) -> Unit');
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
      try { await done; } catch { /* Rust reports the failed request. */ }
      exports['blot:reset'](0);
    }
  };
}
