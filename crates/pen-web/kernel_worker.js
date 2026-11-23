// Browser Web Worker for Blot vNext `KernelPool` (`pen-web/kernels.js`, §5.1).
// Runs in a dedicated ES-module Web Worker (zero SharedArrayBuffer required).
//
// Messages (Worker -> Coordinator):
// - { kind: "init_ok", seq, memoryBytes }
// - { kind: "init_fail", seq, oom }
// - { kind: "ctx_ok", seq, memoryBytes }
// - { kind: "ready", seq, resultBytes }
// - { kind: "overflow_ok", seq, memoryBytes }
// - { kind: "overflow_cleanup_fail", seq }
// - { kind: "discard_ok", seq, memoryBytes }
// - { kind: "payload_fixed", seq, outputBuf, memoryBytes }  (transfers [outputBuf])
// - { kind: "error_fixed", seq, outputBuf, resultBytes }    (transfers [outputBuf] in try/finally!)
// - { kind: "payload", seq, buffer, memoryBytes }           (transfers [buffer])
// - { kind: "error", seq, resultBytes }                     (relinquishes buffer before posting!)
// - { kind: "arena_oom", seq }
// - { kind: "run_trap", seq, cleanupOk }

const ARENA_OOM_OFFSET = 640;
const WIRE_U32 = 1;
const WIRE_BOOL = 2;
const WIRE_F32 = 3;

let kernel = null;
let heldPacketPtr = 0;
let heldPacketBytes = 0;
let hasHeldPacket = false;

function readArenaOom(memory) {
  if (!memory || memory.buffer.byteLength < ARENA_OOM_OFFSET + 4) return false;
  return new DataView(memory.buffer).getUint32(ARENA_OOM_OFFSET, true) !== 0;
}

function isMemoryGrowError(err) {
  if (!(err instanceof Error)) return false;
  const msg = err.message.toLowerCase();
  return msg.includes("out of memory") || msg.includes("grow");
}

function writePacketWords(memory, allocate, reset, bytes) {
  reset(0);
  const words = bytes.byteLength >>> 2;
  const ptr = allocate((words + 1) * 4) >>> 0;
  const buf = memory.buffer;
  if (ptr < 4 || (ptr & 3) !== 0 || ptr + 4 + bytes.byteLength > buf.byteLength) {
    throw new Error("packet allocation out of bounds");
  }
  new DataView(buf).setUint32(ptr, words, true);
  if (bytes.byteLength > 0) {
    new Uint8Array(buf, ptr + 4, bytes.byteLength).set(bytes);
  }
  return ptr;
}

function decodeScalarArg(wireTag, bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, 4);
  if (wireTag === WIRE_F32) return dv.getFloat32(0, true);
  const u = dv.getUint32(0, true);
  return wireTag === WIRE_BOOL ? (u !== 0 ? 1 : 0) : u;
}

self.onmessage = ({ data: msg }) => {
  const { kind, seq } = msg;

  if (kind === "init") {
    try {
      const module = msg.module instanceof WebAssembly.Module
        ? msg.module
        : new WebAssembly.Module(msg.wasmBytes);
      const stubImports = {};
      for (const imp of WebAssembly.Module.imports(module)) {
        if (imp.kind === "function") {
          (stubImports[imp.module] ??= {})[imp.name] = () => {
            throw new Error(`unexpected host import ${imp.module}/${imp.name}`);
          };
        }
      }
      const instance = new WebAssembly.Instance(module, stubImports);
      const exports = instance.exports;
      const memory = exports["blot:memory"];
      const allocate = exports["blot:allocate"];
      const reset = exports["blot:reset"];
      const entryFn = exports[msg.entryName];
      if (
        !(memory instanceof WebAssembly.Memory) ||
        typeof allocate !== "function" ||
        typeof reset !== "function" ||
        typeof entryFn !== "function"
      ) {
        self.postMessage({ kind: "init_fail", seq, oom: false });
        return;
      }
      kernel = {
        memory,
        allocate,
        reset,
        finishInvocation: typeof exports["blot:finish_invocation"] === "function"
          ? exports["blot:finish_invocation"]
          : null,
        entryFn,
        ctxFn: msg.ctxExport ? exports[msg.ctxExport] : null,
        clearFn: msg.clearExport ? exports[msg.clearExport] : null,
        variableOutput: Boolean(msg.variableOutput),
        ctxWireTag: msg.ctxWireTag ?? 5,
      };
      hasHeldPacket = false;
      heldPacketPtr = 0;
      heldPacketBytes = 0;
      self.postMessage({
        kind: "init_ok",
        seq,
        memoryBytes: memory.buffer.byteLength,
      });
    } catch (err) {
      self.postMessage({
        kind: "init_fail",
        seq,
        oom: isMemoryGrowError(err),
      });
    }
    return;
  }

  if (!kernel) {
    self.postMessage({ kind: "run_trap", seq, cleanupOk: false });
    return;
  }

  if (kind === "clear_ctx") {
    if (!kernel.clearFn) {
      self.postMessage({
        kind: "ctx_ok",
        seq,
        memoryBytes: kernel.memory.buffer.byteLength,
      });
      return;
    }
    try {
      kernel.reset(0);
      kernel.clearFn(0);
      if (kernel.finishInvocation) kernel.finishInvocation();
      kernel.reset(0);
      self.postMessage({
        kind: "ctx_ok",
        seq,
        memoryBytes: kernel.memory.buffer.byteLength,
      });
    } catch (err) {
      if (readArenaOom(kernel.memory) || isMemoryGrowError(err)) {
        self.postMessage({ kind: "arena_oom", seq });
      } else {
        self.postMessage({ kind: "error", seq, resultBytes: 0 });
      }
    }
    return;
  }

  if (kind === "set_ctx") {
    if (!kernel.ctxFn) {
      self.postMessage({ kind: "run_trap", seq, cleanupOk: false });
      return;
    }
    try {
      const bytes = new Uint8Array(msg.bytes);
      let arg;
      if (
        kernel.ctxWireTag === WIRE_U32 ||
        kernel.ctxWireTag === WIRE_BOOL ||
        kernel.ctxWireTag === WIRE_F32
      ) {
        kernel.reset(0);
        arg = decodeScalarArg(kernel.ctxWireTag, bytes);
      } else {
        arg = writePacketWords(kernel.memory, kernel.allocate, kernel.reset, bytes);
      }
      kernel.ctxFn(arg);
      if (kernel.finishInvocation) kernel.finishInvocation();
      kernel.reset(0);
      self.postMessage({
        kind: "ctx_ok",
        seq,
        memoryBytes: kernel.memory.buffer.byteLength,
      });
    } catch (err) {
      if (readArenaOom(kernel.memory) || isMemoryGrowError(err)) {
        self.postMessage({ kind: "arena_oom", seq });
      } else {
        self.postMessage({ kind: "run_trap", seq, cleanupOk: false });
      }
    }
    return;
  }

  if (kind === "run") {
    hasHeldPacket = false;
    heldPacketPtr = 0;
    heldPacketBytes = 0;
    let ptr = 0;
    try {
      const inBytes = new Uint8Array(msg.bytes);
      const inPtr = writePacketWords(
        kernel.memory,
        kernel.allocate,
        kernel.reset,
        inBytes,
      );
      ptr = kernel.entryFn(inPtr) >>> 0;
    } catch (err) {
      if (readArenaOom(kernel.memory) || isMemoryGrowError(err)) {
        self.postMessage({ kind: "arena_oom", seq });
        return;
      }
      let cleanupOk = true;
      try {
        if (kernel.finishInvocation) kernel.finishInvocation();
        kernel.reset(0);
        if (readArenaOom(kernel.memory)) cleanupOk = false;
      } catch {
        cleanupOk = false;
      }
      self.postMessage({ kind: "run_trap", seq, cleanupOk });
      return;
    }

    const buf = kernel.memory.buffer;
    if (ptr < 4 || (ptr & 3) !== 0 || ptr + 4 > buf.byteLength) {
      self.postMessage({ kind: "run_trap", seq, cleanupOk: false });
      return;
    }
    const lenWords = new DataView(buf).getUint32(ptr, true);
    if (kernel.variableOutput && lenWords === 0xffff_ffff) {
      let cleanupOk = true;
      try {
        if (kernel.finishInvocation) kernel.finishInvocation();
        kernel.reset(0);
        if (readArenaOom(kernel.memory)) cleanupOk = false;
      } catch {
        cleanupOk = false;
      }
      if (cleanupOk) {
        self.postMessage({
          kind: "overflow_ok",
          seq,
          memoryBytes: kernel.memory.buffer.byteLength,
        });
      } else {
        self.postMessage({ kind: "overflow_cleanup_fail", seq });
      }
      return;
    }

    const resultBytes = lenWords * 4;
    if (ptr + 4 + resultBytes > buf.byteLength) {
      self.postMessage({ kind: "run_trap", seq, cleanupOk: false });
      return;
    }
    heldPacketPtr = ptr;
    heldPacketBytes = resultBytes;
    hasHeldPacket = true;
    self.postMessage({ kind: "ready", seq, resultBytes });
    return;
  }

  if (kind === "copy_grant_fixed") {
    const { byteOffset, expectedBytes, outputBuf } = msg;
    const ptr = heldPacketPtr;
    const resultBytes = heldPacketBytes;
    hasHeldPacket = false;
    heldPacketPtr = 0;
    heldPacketBytes = 0;
    let ok = false;
    try {
      if (resultBytes === expectedBytes) {
        if (resultBytes > 0) {
          new Uint8Array(outputBuf, byteOffset, resultBytes).set(
            new Uint8Array(kernel.memory.buffer, ptr + 4, resultBytes),
          );
        }
        if (kernel.finishInvocation) kernel.finishInvocation();
        kernel.reset(0);
        ok = !readArenaOom(kernel.memory);
      }
    } catch {
      ok = false;
    } finally {
      if (ok) {
        self.postMessage(
          {
            kind: "payload_fixed",
            seq,
            outputBuf,
            memoryBytes: kernel.memory.buffer.byteLength,
          },
          [outputBuf],
        );
      } else {
        self.postMessage(
          { kind: "error_fixed", seq, outputBuf, resultBytes },
          [outputBuf],
        );
      }
    }
    return;
  }

  if (kind === "copy_grant") {
    const ptr = heldPacketPtr;
    const resultBytes = heldPacketBytes;
    hasHeldPacket = false;
    heldPacketPtr = 0;
    heldPacketBytes = 0;
    let buffer = new ArrayBuffer(resultBytes);
    let ok = false;
    try {
      if (resultBytes > 0) {
        new Uint8Array(buffer).set(
          new Uint8Array(kernel.memory.buffer, ptr + 4, resultBytes),
        );
      }
      if (kernel.finishInvocation) kernel.finishInvocation();
      kernel.reset(0);
      ok = !readArenaOom(kernel.memory);
    } catch {
      ok = false;
    }
    if (ok) {
      self.postMessage(
        {
          kind: "payload",
          seq,
          buffer,
          memoryBytes: kernel.memory.buffer.byteLength,
        },
        [buffer],
      );
    } else {
      // Relinquish locally allocated variable buffer BEFORE posting error reply!
      buffer = null;
      self.postMessage({ kind: "error", seq, resultBytes });
    }
    return;
  }

  if (kind === "discard") {
    const resultBytes = heldPacketBytes;
    hasHeldPacket = false;
    heldPacketPtr = 0;
    heldPacketBytes = 0;
    let ok = true;
    try {
      if (kernel.finishInvocation) kernel.finishInvocation();
      kernel.reset(0);
      if (readArenaOom(kernel.memory)) ok = false;
    } catch {
      ok = false;
    }
    if (ok) {
      self.postMessage({
        kind: "discard_ok",
        seq,
        memoryBytes: kernel.memory.buffer.byteLength,
      });
    } else {
      self.postMessage({ kind: "error", seq, resultBytes });
    }
  }
};
