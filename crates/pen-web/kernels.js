// Browser Web Worker `KernelPool` for Pen Web (§5.1).
//
// Guarantees:
// - Intrinsic primordial `copyPrivateBytes` + unforgeable `CompiledModuleReceipt`
// - Pre-instantiation `MemoryShape` admission (rejecting unbounded memory when `policy.memory` is set)
// - `deadlineMs` range validation (`1..=2_147_483_647`) + premature-timer monotonic re-arm
// - Mandatory off-thread dedicated Web Worker for ALL `deadlineMs !== undefined` paths
//   (`workers == 0` direct, serial fallback, AND standalone `setContext`/`clearContext`)
// - True 1x single-buffer `runFixed` via serialized zero-copy `[outputBuf]` transfer
// - Variable-output two-step `ready` -> `copy_grant`/`discard` + `PendingFrontierGrant` drain
// - Unified single-owner JS lifecycle (`#state`, `#poolEpoch`, `#activeOp`, `#closePromise`)
// - Platform-honest permanent retirement of unacknowledged transferred buffers on
//   hard Browser `worker.terminate()` (timeout OR mid-transfer `close()`).

const DEFAULT_TRANSPORT_BUDGET = 2_097_152;
const KERNEL_FLAG_HAS_CONTEXT = 1;
const KERNEL_FLAG_VARIABLE_OUTPUT = 2;
const ARENA_OOM_OFFSET = 640;
const MAX_DEADLINE_MS = 2_147_483_647;
const DEFAULT_WORKER_URL = new URL("./kernel_worker.js", import.meta.url);

// Captured intrinsics for side-effect-free private byte copying.
const TypedArrayProto = Object.getPrototypeOf(Uint8Array.prototype);
const TypedArrayPrototypeIndexOf = TypedArrayProto.indexOf;
const TypedArrayPrototypeBuffer = Object.getOwnPropertyDescriptor(
  TypedArrayProto,
  "buffer",
).get;
const TypedArrayPrototypeByteOffset = Object.getOwnPropertyDescriptor(
  TypedArrayProto,
  "byteOffset",
).get;
const TypedArrayPrototypeByteLength = Object.getOwnPropertyDescriptor(
  TypedArrayProto,
  "byteLength",
).get;
const DataViewPrototypeBuffer = Object.getOwnPropertyDescriptor(
  DataView.prototype,
  "buffer",
).get;
const DataViewPrototypeByteOffset = Object.getOwnPropertyDescriptor(
  DataView.prototype,
  "byteOffset",
).get;
const DataViewPrototypeByteLength = Object.getOwnPropertyDescriptor(
  DataView.prototype,
  "byteLength",
).get;
const ArrayBufferPrototypeByteLength = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  "byteLength",
).get;
const Uint8ArrayPrototypeSet = Uint8Array.prototype.set;

export function copyPrivateBytes(view) {
  if (!ArrayBuffer.isView(view)) {
    throw new TypeError(
      "Expected an ArrayBufferView backed by an unshared ArrayBuffer",
    );
  }
  let buf;
  let offset;
  let byteLen;
  if (view instanceof DataView) {
    buf = Reflect.apply(DataViewPrototypeBuffer, view, []);
    offset = Reflect.apply(DataViewPrototypeByteOffset, view, []);
    byteLen = Reflect.apply(DataViewPrototypeByteLength, view, []);
  } else {
    Reflect.apply(TypedArrayPrototypeIndexOf, view, [0, Infinity]);
    buf = Reflect.apply(TypedArrayPrototypeBuffer, view, []);
    offset = Reflect.apply(TypedArrayPrototypeByteOffset, view, []);
    byteLen = Reflect.apply(TypedArrayPrototypeByteLength, view, []);
  }
  const totalBytes = Reflect.apply(ArrayBufferPrototypeByteLength, buf, []);
  if (offset < 0 || byteLen < 0 || offset + byteLen > totalBytes) {
    throw new TypeError("ArrayBufferView lies outside its backing ArrayBuffer");
  }
  const out = new Uint8Array(new ArrayBuffer(byteLen));
  if (byteLen > 0) {
    Reflect.apply(Uint8ArrayPrototypeSet, out, [
      new Uint8Array(buf, offset, byteLen),
    ]);
  }
  return out;
}

export class BrowserKernelError extends Error {
  constructor(code, message, startItem) {
    super(message);
    this.name = "BrowserKernelError";
    this.code = code;
    this.startItem = startItem;
  }
}

function readUleb(bytes, state) {
  let result = 0;
  let shift = 0;
  for (let i = 0; i < 10; i++) {
    if (state.at >= bytes.length) {
      throw new BrowserKernelError("invalid_abi", "Truncated ULEB128");
    }
    const b = bytes[state.at++];
    result += (b & 0x7f) * 2 ** shift;
    if ((b & 0x80) === 0) return result;
    shift += 7;
  }
  throw new BrowserKernelError("invalid_abi", "ULEB128 overflow");
}

export function readMemoryShape(bytes) {
  if (
    bytes.length < 8 ||
    bytes[0] !== 0x00 || bytes[1] !== 0x61 ||
    bytes[2] !== 0x73 || bytes[3] !== 0x6d ||
    bytes[4] !== 0x01 || bytes[5] !== 0x00 ||
    bytes[6] !== 0x00 || bytes[7] !== 0x00
  ) {
    throw new BrowserKernelError("invalid_abi", "Invalid WebAssembly header");
  }
  const state = { at: 8 };
  let memSec = null;
  while (state.at < bytes.length) {
    const id = bytes[state.at++];
    const size = readUleb(bytes, state);
    if (size < 0 || state.at + size > bytes.length) {
      throw new BrowserKernelError("invalid_abi", "Truncated Wasm section");
    }
    if (id === 5) {
      if (memSec !== null) {
        throw new BrowserKernelError("invalid_abi", "Duplicate MemorySection");
      }
      memSec = bytes.subarray(state.at, state.at + size);
    }
    state.at += size;
  }
  if (memSec === null) {
    return Object.freeze({
      sectionPresent: false,
      hasMemory: false,
      memory64: false,
      shared: false,
      pageSizeLog2: 16,
      initialPages: 0,
      maximumPages: undefined,
    });
  }
  const ms = { at: 0 };
  const count = readUleb(memSec, ms);
  if (count === 0) {
    return Object.freeze({
      sectionPresent: true,
      hasMemory: false,
      memory64: false,
      shared: false,
      pageSizeLog2: 16,
      initialPages: 0,
      maximumPages: undefined,
    });
  }
  if (count > 1) {
    throw new BrowserKernelError("invalid_abi", "Multiple Wasm memories unsupported");
  }
  const flags = memSec[ms.at++];
  const hasMax = (flags & 0x01) !== 0;
  const shared = (flags & 0x02) !== 0;
  const memory64 = (flags & 0x04) !== 0;
  const hasPageSize = (flags & 0x08) !== 0;
  const initialPages = readUleb(memSec, ms);
  const maximumPages = hasMax ? readUleb(memSec, ms) : undefined;
  const pageSizeLog2 = hasPageSize ? readUleb(memSec, ms) : 16;
  return Object.freeze({
    sectionPresent: true,
    hasMemory: true,
    memory64,
    shared,
    pageSizeLog2,
    initialPages,
    maximumPages,
  });
}

function parseKernelsSection(module) {
  const secs = WebAssembly.Module.customSections(module, "blot:kernels");
  if (secs.length !== 1) return { transportBudget: DEFAULT_TRANSPORT_BUDGET, kernels: [] };
  const bytes = new Uint8Array(secs[0]);
  const st = { at: 0 };
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const byte = () => {
    if (st.at >= bytes.length) throw new BrowserKernelError("invalid_abi", "Truncated blot:kernels");
    return bytes[st.at++];
  };
  const name = () => {
    const len = readUleb(bytes, st);
    const out = decoder.decode(bytes.subarray(st.at, st.at + len));
    st.at += len;
    return out;
  };
  const u64 = () => {
    const dv = new DataView(bytes.buffer, bytes.byteOffset + st.at, 8);
    st.at += 8;
    return dv.getBigUint64(0, true);
  };
  const parseSchema = () => {
    const kindCode = byte();
    const wireTagCode = byte();
    const stride = readUleb(bytes, st);
    const wordTags = [];
    if (kindCode === 1) {
      for (let i = 0; i < stride; i++) wordTags.push(byte());
    }
    const schemaHash = u64();
    return Object.freeze({
      kind: ["Unit", "FlatFixed", "VarU32", "VarF32"][kindCode],
      wireTagCode,
      stride,
      wordTags: Object.freeze(wordTags),
      schemaHash,
    });
  };
  const version = readUleb(bytes, st);
  if (version !== 1) throw new BrowserKernelError("invalid_abi", "Unsupported blot:kernels version");
  const transportBudget = readUleb(bytes, st);
  const count = readUleb(bytes, st);
  const kernels = [];
  for (let id = 0; id < count; id++) {
    const kName = name();
    const flags = byte();
    const hasContext = (flags & KERNEL_FLAG_HAS_CONTEXT) !== 0;
    const variableOutput = (flags & KERNEL_FLAG_VARIABLE_OUTPUT) !== 0;
    const inSchema = parseSchema();
    const outSchema = parseSchema();
    const ioSchemaHash = u64();
    const implHash = u64();
    const bindingHash = u64();
    let ctxSchema;
    let ctxExport;
    let clearExport;
    if (hasContext) {
      ctxSchema = parseSchema();
      ctxExport = name();
      clearExport = name();
    }
    kernels.push(Object.freeze({
      id,
      name: kName,
      flags,
      hasContext,
      variableOutput,
      inSchema,
      outSchema,
      ioSchemaHash,
      implHash,
      bindingHash,
      ctxSchema,
      ctxExport,
      clearExport,
    }));
  }
  return { transportBudget, kernels: Object.freeze(kernels) };
}

const receiptMetaMap = new WeakMap();

export function compileKernelModule(bytes) {
  const snapshot = copyPrivateBytes(bytes);
  const memoryShape = readMemoryShape(snapshot);
  if (memoryShape.hasMemory) {
    if (memoryShape.shared || memoryShape.memory64 || memoryShape.pageSizeLog2 !== 16) {
      throw new BrowserKernelError("invalid_abi", "Unsupported Wasm memory shape");
    }
  }
  const module = new WebAssembly.Module(snapshot);
  const { transportBudget, kernels } = parseKernelsSection(module);
  const meta = Object.freeze({ module, snapshot, memoryShape, transportBudget, kernels });
  const receipt = Object.freeze({ module, memoryShape, transportBudget, kernels });
  receiptMetaMap.set(receipt, meta);
  return receipt;
}

function resolveValidated(source) {
  if (typeof source === "object" && source !== null) {
    const m = receiptMetaMap.get(source);
    if (m) return m;
  }
  if (source instanceof Uint8Array) {
    const r = compileKernelModule(source);
    return receiptMetaMap.get(r);
  }
  throw new TypeError("Expected raw Wasm Uint8Array or CompiledModuleReceipt");
}

function bytesEqual(a, b) {
  if (a.byteLength !== b.byteLength) return false;
  for (let i = 0; i < a.byteLength; i++) if (a[i] !== b[i]) return false;
  return true;
}

function failurePriority(kind) {
  if (kind === "run_trap" || kind === "single_item_output_overflow" || kind === "out_of_memory") return 3;
  if (kind === "output_budget_exceeded") return 2;
  return 1;
}

class BrowserWorkerSlot {
  constructor(workerIndex, workerFactory, memoryCharge) {
    this.workerIndex = workerIndex;
    this.worker = workerFactory();
    this.memoryCharge = memoryCharge;
    this.terminated = false;
    this.pending = null;
    this.nextSeq = 1;
    this.installedSnapshotId = null;
    this.storeHasPriorHistory = false;
    this.memoryBytes = 0;
    this.ownedTransferredBytes = 0; // unacknowledged transferred outputBuf / copy_grant!
    this.worker.onmessage = ({ data }) => {
      if (this.terminated) return;
      const p = this.pending;
      if (p && data && data.seq === p.seq) {
        this.pending = null;
        p.resolve(data);
      }
    };
    this.worker.onerror = (ev) => {
      if (this.terminated) return;
      const p = this.pending;
      this.pending = null;
      if (p) p.resolve({ kind: "worker_error", message: ev?.message ?? "worker error" });
    };
  }

  request(msg, transfer = []) {
    if (this.terminated) return Promise.resolve({ kind: "fenced" });
    const seq = this.nextSeq++;
    return new Promise((resolve) => {
      this.pending = { seq, resolve };
      this.worker.postMessage({ ...msg, seq }, transfer);
    });
  }

  terminate() {
    if (this.terminated) return;
    this.terminated = true;
    const p = this.pending;
    this.pending = null;
    try {
      this.worker.terminate();
    } catch { /* ignore */ }
    if (p) p.resolve({ kind: "fenced" });
  }
}

export class BrowserKernelPool {
  #meta;
  #workersCount;
  #deadlineMs;
  #memoryLimit;
  #maxTotalOutputBytes;
  #maxPhysicalOutputBytes;
  #retireMemoryThreshold;
  #hooks;
  #workerFactory;

  #state = "open";
  #poolEpoch = 0;
  #activeOp = null;
  #closePromise = null;
  #currentCtx = null;
  #nextSnapshotId = 1;

  #warmSlots = [];
  #directSlot = null;
  #inlineGuest = null;
  #forceFallback = false;

  #peakChargedBytes = 0;
  #inputSnapshotBytes = 0;
  #contextSnapshotBytes = 0;
  #fixedOutputBytes = 0;
  #bufferedChunkBytes = 0;
  #inFlightGrantBytes = 0;
  #concatScratchBytes = 0;
  #workerMemoryBytes = 0;
  #permanentlyRetiredBytes = 0;
  #freshRetries = 0;
  #fallbacks = 0;
  #liveSlots = new Set();

  constructor(meta, manifest, config) {
    this.#meta = meta;
    this.manifest = manifest;
    this.transportBudget = meta.transportBudget;
    this.#workersCount = config.workers ?? 1;
    this.#deadlineMs = config.deadlineMs ?? null;
    this.#memoryLimit = config.memory ?? null;
    this.#maxTotalOutputBytes = config.maxTotalOutputBytes ?? 16 * 1024 * 1024;
    this.#maxPhysicalOutputBytes = config.maxPhysicalOutputBytes ??
      this.#maxTotalOutputBytes * 2;
    this.#retireMemoryThreshold = config.retireMemoryThreshold ?? 64 * 1024 * 1024;
    this.#hooks = config._testHooks;
    this.#forceFallback = Boolean(config._testHooks?.forceFallback);
    this.#workerFactory = config._testHooks?.workerFactory ??
      (() => new Worker(config.workerUrl ?? DEFAULT_WORKER_URL, { type: "module" }));

    const defaultChunk = config.defaultChunkItems ?? 64;
    const maxIn = Math.floor((this.transportBudget - 4) / (manifest.inSchema.stride * 4));
    const maxOut = manifest.variableOutput
      ? Number.POSITIVE_INFINITY
      : Math.floor((this.transportBudget - 4) / (manifest.outSchema.stride * 4));
    this.maxChunkItems = Math.max(1, Math.min(defaultChunk, maxIn, maxOut));
    for (let i = 0; i < this.#workersCount; i++) this.#warmSlots.push(null);
  }

  static async open(source, name, config = {}) {
    const meta = resolveValidated(source);
    const manifest = meta.kernels.find((k) => k.name === name);
    if (!manifest) {
      throw new BrowserKernelError("unknown_kernel", `Unknown kernel ${name}`);
    }
    const workers = config.workers ?? 1;
    if (!Number.isSafeInteger(workers) || workers < 0) {
      throw new BrowserKernelError("invalid_config", "Invalid workers");
    }
    if (config.deadlineMs !== undefined) {
      const d = config.deadlineMs;
      if (!Number.isFinite(d) || d < 1 || d > MAX_DEADLINE_MS) {
        throw new BrowserKernelError(
          "invalid_config",
          `deadlineMs must be finite in 1..=${MAX_DEADLINE_MS}`,
        );
      }
    }
    if (config.memory !== undefined) {
      if (!Number.isSafeInteger(config.memory) || config.memory <= 0) {
        throw new BrowserKernelError("invalid_config", "Invalid memory limit");
      }
      if (meta.memoryShape.hasMemory) {
        if (meta.memoryShape.maximumPages === undefined) {
          throw new BrowserKernelError(
            "memory_limit_unbounded",
            "Browser KernelPool rejects unbounded Wasm memory when policy.memory is set",
          );
        }
        const maxBytes = meta.memoryShape.maximumPages * 65_536;
        if (config.memory < maxBytes) {
          throw new BrowserKernelError(
            "memory_limit_exceeded",
            `policy.memory (${config.memory}) < declared max (${maxBytes})`,
          );
        }
      }
    }
    return await Promise.resolve(new BrowserKernelPool(meta, manifest, config));
  }

  get stats() {
    return Object.freeze({
      peakChargedBytes: this.#peakChargedBytes,
      currentChargedBytes: this.#currentOutputBytes(),
      inputSnapshotBytes: this.#inputSnapshotBytes,
      contextSnapshotBytes: this.#contextSnapshotBytes,
      fixedOutputBytes: this.#fixedOutputBytes,
      bufferedChunkBytes: this.#bufferedChunkBytes,
      inFlightGrantBytes: this.#inFlightGrantBytes,
      workerMemoryBytes: this.#workerMemoryBytes,
      permanentlyRetiredBytes: this.#permanentlyRetiredBytes,
      freshRetries: this.#freshRetries,
      fallbacks: this.#fallbacks,
    });
  }

  forceSerialFallback(enabled = true) {
    this.#forceFallback = enabled;
  }

  #currentOutputBytes() {
    return this.#fixedOutputBytes + this.#bufferedChunkBytes +
      this.#inFlightGrantBytes + this.#concatScratchBytes;
  }

  #recordPeak() {
    const cur = this.#currentOutputBytes();
    if (cur > this.#peakChargedBytes) this.#peakChargedBytes = cur;
  }

  #perWorkerMemoryCharge() {
    if (this.#memoryLimit !== null) return this.#memoryLimit;
    if (this.#meta.memoryShape.maximumPages !== undefined) {
      return this.#meta.memoryShape.maximumPages * 65_536;
    }
    return this.#meta.memoryShape.initialPages * 65_536;
  }

  #armTimer(opDeadlineMs, onExpire) {
    let cleared = false;
    let handle;
    const tick = () => {
      if (cleared) return;
      const remaining = opDeadlineMs - performance.now();
      if (remaining > 0) {
        handle = setTimeout(tick, Math.ceil(remaining));
        return;
      }
      cleared = true;
      onExpire();
    };
    handle = setTimeout(tick, Math.max(1, Math.ceil(opDeadlineMs - performance.now())));
    this.#hooks?.onTimerArmed?.({ deadlineMs: opDeadlineMs, fire: tick });
    return {
      clear: () => {
        cleared = true;
        clearTimeout(handle);
      },
    };
  }

  /**
   * Terminate a Browser Web Worker slot with platform-honest accounting:
   * - Clean idle/cooperative retirement (`hard = false`): releases worker memory cleanly.
   * - Hard termination (`hard = true`, timeout or close() while worker owns unacknowledged
   *   transferred buffer or is actively running): permanently retires its unacknowledged
   *   buffer/memory charge into `#permanentlyRetiredBytes`!
   */
  #retireSlot(slot, hard) {
    if (!slot || slot.terminated) return;
    this.#liveSlots.delete(slot);
    for (let i = 0; i < this.#warmSlots.length; i++) {
      if (this.#warmSlots[i] === slot) this.#warmSlots[i] = null;
    }
    if (this.#directSlot === slot) this.#directSlot = null;
    const ownedTransfer = slot.ownedTransferredBytes;
    slot.ownedTransferredBytes = 0;
    slot.terminate();
    if (hard && (ownedTransfer > 0 || slot.memoryCharge > 0)) {
      // Unacknowledged transferred buffer + hung worker memory permanently retired!
      this.#permanentlyRetiredBytes += ownedTransfer + slot.memoryCharge;
      this.#workerMemoryBytes = Math.max(0, this.#workerMemoryBytes - slot.memoryCharge);
    } else {
      this.#workerMemoryBytes = Math.max(0, this.#workerMemoryBytes - slot.memoryCharge);
    }
  }

  async #spawnSlot(workerIndex) {
    const charge = this.#perWorkerMemoryCharge();
    const slot = new BrowserWorkerSlot(workerIndex, this.#workerFactory, charge);
    this.#workerMemoryBytes += charge;
    this.#liveSlots.add(slot);
    const res = await slot.request({
      kind: "init",
      module: this.#meta.module,
      wasmBytes: this.#meta.snapshot,
      entryName: this.manifest.name,
      variableOutput: this.manifest.variableOutput,
      ctxExport: this.manifest.ctxExport,
      clearExport: this.manifest.clearExport,
      ctxWireTag: this.manifest.ctxSchema?.wireTagCode ?? 5,
    });
    if (res.kind !== "init_ok") {
      this.#retireSlot(slot, res.kind === "fenced");
      return { error: res };
    }
    slot.memoryBytes = res.memoryBytes;
    return { slot };
  }

  async #ensureSlotSnapshot(slotRef, workerIndex, target, opDeadlineMs, opEpoch) {
    let retriedFresh = false;
    for (;;) {
      if (this.#state !== "open" || this.#poolEpoch !== opEpoch) {
        return { ok: false, error: new BrowserKernelError("pool_closed", "KernelPool closed") };
      }
      if (opDeadlineMs !== null && performance.now() >= opDeadlineMs) {
        return { ok: false, error: new BrowserKernelError("timeout", "Timed out") };
      }
      if (!slotRef.slot || slotRef.slot.terminated) {
        const sp = await this.#spawnSlot(workerIndex);
        if (sp.error) {
          if (this.#state !== "open" || this.#poolEpoch !== opEpoch) {
            return { ok: false, error: new BrowserKernelError("pool_closed", "KernelPool closed") };
          }
          if (sp.error.kind === "fenced" || (opDeadlineMs !== null && performance.now() >= opDeadlineMs)) {
            return { ok: false, error: new BrowserKernelError("timeout", "Worker init timed out") };
          }
          return {
            ok: false,
            error: new BrowserKernelError(
              sp.error.oom ? "out_of_memory" : "transport_fault",
              "Worker init failed",
            ),
          };
        }
        slotRef.slot = sp.slot;
      }
      const s = slotRef.slot;
      if (!this.manifest.hasContext || !target || s.installedSnapshotId === target.snapshotId) {
        return { ok: true, slot: s };
      }
      if (s.installedSnapshotId !== null) {
        const eligibleClear = s.storeHasPriorHistory && !retriedFresh;
        const cRes = await s.request({ kind: "clear_ctx" });
        if (this.#state !== "open" || this.#poolEpoch !== opEpoch) {
          this.#retireSlot(s, true);
          slotRef.slot = null;
          return { ok: false, error: new BrowserKernelError("pool_closed", "KernelPool closed") };
        }
        if (cRes.kind === "fenced" || (opDeadlineMs !== null && performance.now() >= opDeadlineMs)) {
          this.#retireSlot(s, true);
          slotRef.slot = null;
          return { ok: false, error: new BrowserKernelError("timeout", "Context clear timed out") };
        }
        if (cRes.kind !== "ctx_ok") {
          this.#retireSlot(s, false);
          slotRef.slot = null;
          if (cRes.kind === "arena_oom" && eligibleClear) {
            retriedFresh = true;
            this.#freshRetries++;
            continue;
          }
          return {
            ok: false,
            error: new BrowserKernelError(
              cRes.kind === "arena_oom" ? "out_of_memory" : "cleanup_trap",
              "Context clear failed",
            ),
          };
        }
        s.installedSnapshotId = null;
        s.storeHasPriorHistory = true;
        s.memoryBytes = cRes.memoryBytes;
      }

      const eligible = s.storeHasPriorHistory && !retriedFresh;
      const res = await s.request({ kind: "set_ctx", bytes: target.bytes });
      if (this.#state !== "open" || this.#poolEpoch !== opEpoch) {
        this.#retireSlot(s, true);
        slotRef.slot = null;
        return { ok: false, error: new BrowserKernelError("pool_closed", "KernelPool closed") };
      }
      if (res.kind === "fenced" || (opDeadlineMs !== null && performance.now() >= opDeadlineMs)) {
        this.#retireSlot(s, true);
        slotRef.slot = null;
        return { ok: false, error: new BrowserKernelError("timeout", "Context prepare timed out") };
      }
      if (res.kind === "ctx_ok") {
        s.installedSnapshotId = target.snapshotId;
        s.memoryBytes = res.memoryBytes;
        return { ok: true, slot: s };
      }
      this.#retireSlot(s, false);
      slotRef.slot = null;
      if (res.kind === "arena_oom" && eligible) {
        retriedFresh = true;
        this.#freshRetries++;
        continue;
      }
      return {
        ok: false,
        error: new BrowserKernelError(
          res.kind === "arena_oom" ? "out_of_memory" : "run_trap",
          "Context prepare failed",
        ),
      };
    }
  }

  setContext(revision, bytes) {
    if (this.#state !== "open") {
      return Promise.reject(new BrowserKernelError("pool_closed", "KernelPool is closed"));
    }
    if (this.#activeOp !== null) {
      return Promise.reject(
        new BrowserKernelError(
          this.#activeOp.kind === "batch" ? "batch_in_progress" : "context_operation_in_progress",
          "Operation in progress",
        ),
      );
    }
    if (!this.manifest.hasContext || !this.manifest.ctxSchema) {
      return Promise.reject(new BrowserKernelError("invalid_config", "Kernel has no context"));
    }
    const rev = typeof revision === "bigint" ? revision : BigInt(revision);
    const ownedBytes = copyPrivateBytes(bytes);
    if (ownedBytes.byteLength % 4 !== 0 || ownedBytes.byteLength > this.transportBudget - 4) {
      return Promise.reject(new BrowserKernelError("invalid_input", "Invalid context byte length"));
    }
    if (
      this.manifest.ctxSchema.kind === "FlatFixed" &&
      ownedBytes.byteLength !== this.manifest.ctxSchema.stride * 4
    ) {
      return Promise.reject(new BrowserKernelError("invalid_input", "Fixed context size mismatch"));
    }
    if (
      this.#currentCtx !== null && this.#currentCtx.revision === rev &&
      !bytesEqual(this.#currentCtx.bytes, ownedBytes)
    ) {
      return Promise.reject(
        new BrowserKernelError("revision_content_mismatch", "Revision content mismatch"),
      );
    }

    const opEpoch = this.#poolEpoch;
    const candidateCharge = ownedBytes.byteLength;
    this.#contextSnapshotBytes += candidateCharge;
    let released = false;
    const releaseCharge = () => {
      if (!released) {
        released = true;
        this.#contextSnapshotBytes = Math.max(0, this.#contextSnapshotBytes - candidateCharge);
      }
    };
    const nextSnapshot = Object.freeze({
      snapshotId: this.#nextSnapshotId++,
      revision: rev,
      bytes: ownedBytes,
    });
    const ctxDeadlineMs = this.#deadlineMs !== null ? performance.now() + this.#deadlineMs : null;
    let candidateSlot = null;
    let timedOut = false;
    let timer = null;

    const cancelForClose = () => {
      timer?.clear();
      if (candidateSlot) this.#retireSlot(candidateSlot, true);
    };

    let resolveCtxBarrier;
    const ctxBarrier = new Promise((r) => { resolveCtxBarrier = r; });
    this.#activeOp = {
      kind: "context",
      epoch: opEpoch,
      cancelForClose,
      done: ctxBarrier,
    };

    const done = (async () => {
      try {
        if (ctxDeadlineMs !== null) {
          timer = this.#armTimer(ctxDeadlineMs, () => {
            timedOut = true;
            if (candidateSlot) this.#retireSlot(candidateSlot, true);
          });
        }
        // Always execute off-thread on a dedicated Web Worker whenever deadlineMs !== null or workers > 0!
        if (this.#deadlineMs !== null || this.#workersCount > 0) {
          const sp = await this.#spawnSlot(-1);
          if (sp.error) {
            if (this.#state !== "open" || this.#poolEpoch !== opEpoch) {
              throw new BrowserKernelError("pool_closed", "KernelPool closed");
            }
            if (timedOut || sp.error.kind === "fenced" || (ctxDeadlineMs !== null && performance.now() >= ctxDeadlineMs)) {
              throw new BrowserKernelError("timeout", "setContext timed out");
            }
            throw new BrowserKernelError(sp.error.oom ? "out_of_memory" : "transport_fault", "Candidate init failed");
          }
          candidateSlot = sp.slot;
          const res = await candidateSlot.request({ kind: "set_ctx", bytes: ownedBytes });
          if (this.#hooks?.onBeforeContextPublish) {
            await this.#hooks.onBeforeContextPublish();
          }
          if (this.#state !== "open" || this.#poolEpoch !== opEpoch) {
            this.#retireSlot(candidateSlot, true);
            candidateSlot = null;
            throw new BrowserKernelError("pool_closed", "KernelPool closed");
          }
          if (timedOut || res.kind === "fenced" || (ctxDeadlineMs !== null && performance.now() >= ctxDeadlineMs)) {
            this.#retireSlot(candidateSlot, true);
            candidateSlot = null;
            throw new BrowserKernelError("timeout", "setContext timed out");
          }
          if (res.kind !== "ctx_ok") {
            this.#retireSlot(candidateSlot, false);
            candidateSlot = null;
            throw new BrowserKernelError(
              res.kind === "arena_oom" ? "out_of_memory" : "run_trap",
              "Context prepare failed",
            );
          }
          candidateSlot.installedSnapshotId = nextSnapshot.snapshotId;
          candidateSlot.memoryBytes = res.memoryBytes;
          if (this.#workersCount === 0 && this.#directSlot === null) {
            this.#directSlot = candidateSlot;
            candidateSlot = null;
          } else if (this.#workersCount > 0 && this.#warmSlots[0] === null) {
            this.#warmSlots[0] = candidateSlot;
            candidateSlot = null;
          } else {
            this.#retireSlot(candidateSlot, false);
            candidateSlot = null;
          }
        } else {
          // Inline unbounded candidate (deadlineMs === null && workers === 0).
          const stubImports = {};
          for (const imp of WebAssembly.Module.imports(this.#meta.module)) {
            if (imp.kind === "function") {
              (stubImports[imp.module] ??= {})[imp.name] = () => {
                throw new Error(`unexpected host import ${imp.module}/${imp.name}`);
              };
            }
          }
          const inst = new WebAssembly.Instance(this.#meta.module, stubImports);
          const exp = inst.exports;
          const mem = exp["blot:memory"];
          const alloc = exp["blot:allocate"];
          const reset = exp["blot:reset"];
          const ctxFn = exp[this.manifest.ctxExport];
          reset(0);
          let arg;
          const wt = this.manifest.ctxSchema.wireTagCode;
          if (wt === 1 || wt === 2 || wt === 3) {
            const dv = new DataView(ownedBytes.buffer, ownedBytes.byteOffset, 4);
            arg = wt === 3 ? dv.getFloat32(0, true) : dv.getUint32(0, true);
          } else {
            const words = ownedBytes.byteLength >>> 2;
            const ptr = alloc((words + 1) * 4) >>> 0;
            new DataView(mem.buffer).setUint32(ptr, words, true);
            new Uint8Array(mem.buffer, ptr + 4, ownedBytes.byteLength).set(ownedBytes);
            arg = ptr;
          }
          ctxFn(arg);
          exp["blot:finish_invocation"]?.();
          reset(0);
          if (this.#hooks?.onBeforeContextPublish) {
            await this.#hooks.onBeforeContextPublish();
          }
          if (this.#state !== "open" || this.#poolEpoch !== opEpoch) {
            throw new BrowserKernelError("pool_closed", "KernelPool closed");
          }
        }
        const prev = this.#currentCtx;
        this.#currentCtx = nextSnapshot;
        if (prev) {
          this.#contextSnapshotBytes = Math.max(
            0,
            this.#contextSnapshotBytes - prev.bytes.byteLength,
          );
        }
        released = true;
      } catch (err) {
        if (candidateSlot) {
          this.#retireSlot(candidateSlot, true);
          candidateSlot = null;
        }
        releaseCharge();
        throw err;
      } finally {
        timer?.clear();
        if (this.#activeOp?.epoch === opEpoch) this.#activeOp = null;
        resolveCtxBarrier();
      }
    })();

    return done;
  }

  clearContext() {
    if (this.#state !== "open") {
      return Promise.reject(new BrowserKernelError("pool_closed", "KernelPool is closed"));
    }
    if (this.#activeOp !== null) {
      return Promise.reject(
        new BrowserKernelError(
          this.#activeOp.kind === "batch" ? "batch_in_progress" : "context_operation_in_progress",
          "Operation in progress",
        ),
      );
    }
    if (!this.manifest.hasContext) {
      return Promise.reject(new BrowserKernelError("invalid_config", "Kernel has no context"));
    }
    const opEpoch = this.#poolEpoch;
    const ctxDeadlineMs = this.#deadlineMs !== null ? performance.now() + this.#deadlineMs : null;
    let timedOut = false;
    let timer = null;
    const clearingSlots = [];

    const cancelForClose = () => {
      timer?.clear();
      for (const s of clearingSlots) this.#retireSlot(s, true);
    };

    let resolveClearB;
    const clearB = new Promise((r) => { resolveClearB = r; });
    this.#activeOp = {
      kind: "context",
      epoch: opEpoch,
      cancelForClose,
      done: clearB,
    };

    const done = (async () => {
      try {
        if (ctxDeadlineMs !== null) {
          timer = this.#armTimer(ctxDeadlineMs, () => {
            timedOut = true;
            for (const s of clearingSlots) this.#retireSlot(s, true);
          });
        }
        const slots = [];
        if (this.#directSlot && this.#directSlot.installedSnapshotId !== null) {
          slots.push(this.#directSlot);
        }
        for (const s of this.#warmSlots) {
          if (s && s.installedSnapshotId !== null) slots.push(s);
        }
        clearingSlots.push(...slots);
        if (slots.length > 0) {
          const results = await Promise.all(
            slots.map((s) => s.request({ kind: "clear_ctx" })),
          );
          if (this.#hooks?.onBeforeContextPublish) {
            await this.#hooks.onBeforeContextPublish();
          }
          if (this.#state !== "open" || this.#poolEpoch !== opEpoch) {
            for (const s of slots) this.#retireSlot(s, true);
            throw new BrowserKernelError("pool_closed", "KernelPool closed");
          }
          if (
            timedOut || results.some((r) => r.kind === "fenced") ||
            (ctxDeadlineMs !== null && performance.now() >= ctxDeadlineMs)
          ) {
            for (const s of slots) this.#retireSlot(s, true);
            throw new BrowserKernelError("timeout", "clearContext timed out");
          }
          for (let i = 0; i < slots.length; i++) {
            if (results[i].kind === "ctx_ok") {
              slots[i].installedSnapshotId = null;
              slots[i].storeHasPriorHistory = true;
              slots[i].memoryBytes = results[i].memoryBytes;
            } else {
              this.#retireSlot(slots[i], false);
            }
          }
        } else {
          if (this.#hooks?.onBeforeContextPublish) {
            await this.#hooks.onBeforeContextPublish();
          }
          if (this.#state !== "open" || this.#poolEpoch !== opEpoch) {
            throw new BrowserKernelError("pool_closed", "KernelPool closed");
          }
          if (timedOut || (ctxDeadlineMs !== null && performance.now() >= ctxDeadlineMs)) {
            throw new BrowserKernelError("timeout", "clearContext timed out");
          }
        }
        const prev = this.#currentCtx;
        this.#currentCtx = null;
        if (prev) {
          this.#contextSnapshotBytes = Math.max(
            0,
            this.#contextSnapshotBytes - prev.bytes.byteLength,
          );
        }
      } finally {
        timer?.clear();
        if (this.#activeOp?.epoch === opEpoch) this.#activeOp = null;
        resolveClearB();
      }
    })();

    return done;
  }

  runFixed(items) {
    if (this.manifest.variableOutput) {
      return Promise.reject(new BrowserKernelError("invalid_config", "Kernel is variable-output"));
    }
    return this.#beginBatch("fixed", items);
  }

  runVariableChunks(items) {
    if (!this.manifest.variableOutput) {
      return Promise.reject(new BrowserKernelError("invalid_config", "Kernel is fixed-output"));
    }
    return this.#beginBatch("variable_chunks", items);
  }

  runVariable(items) {
    if (!this.manifest.variableOutput) {
      return Promise.reject(new BrowserKernelError("invalid_config", "Kernel is fixed-output"));
    }
    return this.#beginBatch("variable_flat", items);
  }

  #beginBatch(mode, items) {
    if (this.#state !== "open") {
      return Promise.reject(new BrowserKernelError("pool_closed", "KernelPool is closed"));
    }
    if (this.#activeOp !== null) {
      return Promise.reject(
        new BrowserKernelError(
          this.#activeOp.kind === "context" ? "context_operation_in_progress" : "batch_in_progress",
          "Operation in progress",
        ),
      );
    }
    if (this.manifest.hasContext && this.#currentCtx === null) {
      return Promise.reject(
        new BrowserKernelError("context_not_prepared", "Context not prepared"),
      );
    }
    const inputSnapshot = copyPrivateBytes(items);
    const inItemBytes = this.manifest.inSchema.stride * 4;
    if (inputSnapshot.byteLength % inItemBytes !== 0) {
      return Promise.reject(new BrowserKernelError("invalid_input", "Input length not a multiple of item stride"));
    }
    const itemCount = inputSnapshot.byteLength / inItemBytes;
    const outItemBytes = this.manifest.outSchema.stride * 4;
    if (mode === "fixed" && itemCount > 0) {
      const total = itemCount * outItemBytes;
      if (total > Math.min(this.#maxTotalOutputBytes, this.#maxPhysicalOutputBytes)) {
        return Promise.reject(new BrowserKernelError("output_budget_exceeded", "Fixed output exceeds budget", 0));
      }
    }

    const batchSnapshot = this.#currentCtx;
    const batchEpoch = this.#poolEpoch;
    this.#inputSnapshotBytes = inputSnapshot.byteLength;
    this.#peakChargedBytes = 0;

    const batchDeadlineMs = this.#deadlineMs !== null ? performance.now() + this.#deadlineMs : null;
    let timedOut = false;
    let closedAbort = false;
    let timer = null;
    const activeSlots = new Set();

    const fenceHard = () => {
      for (const s of activeSlots) {
        // Hard terminate with platform-honest permanent retirement of ownedTransferredBytes!
        this.#retireSlot(s, true);
      }
    };

    const cancelForClose = () => {
      closedAbort = true;
      timer?.clear();
      fenceHard();
    };

    let resolveBatchB;
    const batchB = new Promise((r) => { resolveBatchB = r; });
    this.#activeOp = {
      kind: "batch",
      epoch: batchEpoch,
      cancelForClose,
      done: batchB,
    };

    const done = (async () => {
      try {
        if (batchDeadlineMs !== null) {
          timer = this.#armTimer(batchDeadlineMs, () => {
            timedOut = true;
            fenceHard();
          });
        }

        const checkOpenOrTimeout = () => {
          if (closedAbort || this.#state !== "open" || this.#poolEpoch !== batchEpoch) {
            throw new BrowserKernelError("pool_closed", "KernelPool closed");
          }
          if (timedOut || (batchDeadlineMs !== null && performance.now() >= batchDeadlineMs)) {
            throw new BrowserKernelError("timeout", "Batch timed out");
          }
        };

        if (itemCount === 0) {
          if (this.manifest.hasContext && batchSnapshot !== null) {
            const ref = { slot: this.#workersCount === 0 ? this.#directSlot : this.#warmSlots[0] };
            const r = await this.#ensureSlotSnapshot(ref, 0, batchSnapshot, batchDeadlineMs, batchEpoch);
            if (this.#workersCount === 0) this.#directSlot = ref.slot;
            else this.#warmSlots[0] = ref.slot;
            if (!r.ok) throw r.error;
          }
          checkOpenOrTimeout();
          return mode === "variable_chunks" ? [] : new Uint32Array(0);
        }

        const useFallback = this.#workersCount > 0 && this.#forceFallback;
        if (useFallback) this.#fallbacks++;
        const runSerial = this.#workersCount === 0 || useFallback;
        // Mandatory off-thread Web Worker whenever deadlineMs !== null!
        const workerCount = runSerial ? 1 : this.#workersCount;

        const out = mode === "fixed"
          ? await this.#runWorkerFixed(
            inputSnapshot,
            itemCount,
            batchSnapshot,
            batchEpoch,
            batchDeadlineMs,
            workerCount,
            runSerial,
            activeSlots,
            () => closedAbort || this.#state !== "open" || this.#poolEpoch !== batchEpoch,
            () => timedOut || (batchDeadlineMs !== null && performance.now() >= batchDeadlineMs),
          )
          : await this.#runWorkerVariable(
            mode,
            inputSnapshot,
            itemCount,
            batchSnapshot,
            batchEpoch,
            batchDeadlineMs,
            workerCount,
            runSerial,
            activeSlots,
            () => closedAbort || this.#state !== "open" || this.#poolEpoch !== batchEpoch,
            () => timedOut || (batchDeadlineMs !== null && performance.now() >= batchDeadlineMs),
          );
        checkOpenOrTimeout();
        return out;
      } finally {
        timer?.clear();
        this.#inputSnapshotBytes = 0;
        this.#fixedOutputBytes = 0;
        this.#bufferedChunkBytes = 0;
        this.#inFlightGrantBytes = 0;
        this.#concatScratchBytes = 0;
        if (this.#activeOp?.epoch === batchEpoch) this.#activeOp = null;
        resolveBatchB();
      }
    })();

    return done;
  }

  async #runWorkerFixed(
    inputSnapshot,
    itemCount,
    batchSnapshot,
    batchEpoch,
    batchDeadlineMs,
    workerCount,
    isDirectOrFallback,
    activeSlots,
    isClosed,
    isTimedOut,
  ) {
    const inItemBytes = this.manifest.inSchema.stride * 4;
    const outItemBytes = this.manifest.outSchema.stride * 4;
    const totalFixedBytes = itemCount * outItemBytes;

    let outputBuf = new ArrayBuffer(totalFixedBytes);
    this.#fixedOutputBytes = totalFixedBytes;
    this.#recordPeak();

    const queue = [];
    for (let start = 0; start < itemCount; start += this.maxChunkItems) {
      queue.push({ start, end: Math.min(itemCount, start + this.maxChunkItems), retriedFresh: false });
    }
    let earliestFailed = Number.POSITIVE_INFINITY;
    const failures = new Map();
    let fatalAbort = null;
    const fixedCopyQueue = [];
    let grantInFlight = false;

    const recordFailure = (start, kind, message) => {
      const prev = failures.get(start);
      if (!prev || failurePriority(kind) > failurePriority(prev.kind)) {
        failures.set(start, { kind, message });
      }
      if (start < earliestFailed) {
        earliestFailed = start;
      }
    };

    const pumpQueue = () => {
      if (fatalAbort || earliestFailed < Number.POSITIVE_INFINITY) {
        while (fixedCopyQueue.length > 0) {
          fixedCopyQueue.shift().resolve(fatalAbort ? "abort" : "discard");
        }
        if (!grantInFlight && outputBuf !== null) {
          outputBuf = null;
          this.#fixedOutputBytes = 0;
        }
        return;
      }
      if (grantInFlight || outputBuf === null || fixedCopyQueue.length === 0) return;
      fixedCopyQueue.sort((a, b) => a.range.start - b.range.start);
      const next = fixedCopyQueue.shift();
      grantInFlight = true;
      next.resolve("grant");
    };

    const getSlot = (i) => isDirectOrFallback ? this.#directSlot : this.#warmSlots[i];
    const setSlot = (i, s) => {
      if (isDirectOrFallback) this.#directSlot = s;
      else this.#warmSlots[i] = s;
    };

    const runLoop = async (slotIdx) => {
      for (;;) {
        if (fatalAbort || isClosed() || isTimedOut()) break;
        queue.sort((a, b) => a.start - b.start);
        const range = queue.shift();
        if (!range) break;
        if (range.start >= earliestFailed) continue;

        const ref = { slot: getSlot(slotIdx) };
        const snap = await this.#ensureSlotSnapshot(ref, slotIdx, batchSnapshot, batchDeadlineMs, batchEpoch);
        setSlot(slotIdx, ref.slot);
        if (!snap.ok) {
          if (snap.error.code === "pool_closed" || snap.error.code === "timeout" || snap.error.code === "transport_fault") {
            fatalAbort ??= snap.error;
            break;
          }
          recordFailure(range.start, snap.error.code, snap.error.message);
          pumpQueue();
          continue;
        }
        const s = snap.slot;
        activeSlots.add(s);
        const eligible = s.storeHasPriorHistory && !range.retriedFresh;
        const slice = inputSnapshot.subarray(range.start * inItemBytes, range.end * inItemBytes);
        const runRes = await s.request({ kind: "run", bytes: slice });
        if (isClosed()) {
          fatalAbort ??= new BrowserKernelError("pool_closed", "KernelPool closed");
          break;
        }
        if (runRes.kind === "fenced" || isTimedOut()) {
          fatalAbort ??= new BrowserKernelError("timeout", "Batch timed out");
          break;
        }
        if (runRes.kind === "arena_oom") {
          this.#retireSlot(s, false);
          setSlot(slotIdx, null);
          if (eligible) {
            range.retriedFresh = true;
            this.#freshRetries++;
            queue.push(range);
          } else {
            recordFailure(range.start, "out_of_memory", "Kernel OOM");
            pumpQueue();
          }
          continue;
        }
        if (runRes.kind === "run_trap") {
          if (!runRes.cleanupOk) {
            this.#retireSlot(s, false);
            setSlot(slotIdx, null);
          } else {
            s.storeHasPriorHistory = true;
          }
          recordFailure(range.start, "run_trap", "Kernel trapped");
          pumpQueue();
          continue;
        }
        if (runRes.kind !== "ready") {
          fatalAbort ??= new BrowserKernelError("transport_fault", "Unexpected run response");
          this.#retireSlot(s, true);
          break;
        }

        if (this.#hooks?.onAfterReady) {
          await this.#hooks.onAfterReady({
            start: range.start,
            end: range.end,
            resultBytes: runRes.resultBytes,
            workerIndex: slotIdx,
          });
        }
        if (isClosed()) {
          // Coordinator still owns outputBuf before transfer: release cleanly!
          fatalAbort ??= new BrowserKernelError("pool_closed", "KernelPool closed");
          this.#retireSlot(s, false);
          break;
        }
        if (isTimedOut() || fatalAbort) {
          fatalAbort ??= new BrowserKernelError("timeout", "Batch timed out");
          this.#retireSlot(s, true);
          break;
        }

        const decision = await new Promise((resolve) => {
          if (fatalAbort) return resolve("abort");
          if (earliestFailed < Number.POSITIVE_INFINITY) return resolve("discard");
          fixedCopyQueue.push({ range, resolve });
          pumpQueue();
        });

        if (decision === "abort" || fatalAbort) {
          this.#retireSlot(s, false);
          break;
        }
        if (decision === "discard") {
          const dRes = await s.request({ kind: "discard" });
          if (isClosed() || dRes.kind === "fenced" || isTimedOut()) {
            fatalAbort ??= new BrowserKernelError(isClosed() ? "pool_closed" : "timeout", "Interrupted");
            break;
          }
          if (dRes.kind === "error") {
            this.#retireSlot(s, false);
            setSlot(slotIdx, null);
            recordFailure(range.start, "cleanup_trap", "Discard cleanup trapped");
            pumpQueue();
          } else {
            s.storeHasPriorHistory = true;
          }
          continue;
        }

        // Zero-copy [outputBuf] transfer to ONE ready worker at a time!
        const bufToTransfer = outputBuf;
        outputBuf = null;
        s.ownedTransferredBytes = totalFixedBytes;
        const grantPromise = s.request(
          {
            kind: "copy_grant_fixed",
            byteOffset: range.start * outItemBytes,
            expectedBytes: (range.end - range.start) * outItemBytes,
            outputBuf: bufToTransfer,
          },
          [bufToTransfer],
        );
        if (this.#hooks?.onAfterFixedGrantTransfer) {
          await this.#hooks.onAfterFixedGrantTransfer({
            start: range.start,
            end: range.end,
            workerIndex: slotIdx,
          });
        }
        const gRes = await grantPromise;
        grantInFlight = false;
        if (isClosed()) {
          fatalAbort ??= new BrowserKernelError("pool_closed", "KernelPool closed");
          break;
        }
        if (gRes.kind === "fenced" || isTimedOut()) {
          fatalAbort ??= new BrowserKernelError("timeout", "Batch timed out");
          break;
        }
        // Worker handed outputBuf back! Clear ownedTransferredBytes.
        s.ownedTransferredBytes = 0;
        if (gRes.kind === "error_fixed") {
          outputBuf = gRes.outputBuf;
          this.#retireSlot(s, false);
          setSlot(slotIdx, null);
          recordFailure(range.start, "cleanup_trap", "Cleanup trapped after fixed copy");
          outputBuf = null;
          this.#fixedOutputBytes = 0;
          pumpQueue();
          continue;
        }
        if (gRes.kind === "payload_fixed") {
          outputBuf = gRes.outputBuf;
          s.storeHasPriorHistory = true;
          s.memoryBytes = gRes.memoryBytes;
          if (s.memoryBytes >= this.#retireMemoryThreshold) {
            this.#retireSlot(s, false);
            setSlot(slotIdx, null);
          }
          pumpQueue();
          continue;
        }
        fatalAbort ??= new BrowserKernelError("transport_fault", "Unexpected fixed grant reply");
        this.#retireSlot(s, true);
        break;
      }
    };

    await Promise.all(Array.from({ length: workerCount }, (_, i) => runLoop(i)));
    if (fatalAbort || isClosed() || isTimedOut()) {
      outputBuf = null;
      this.#fixedOutputBytes = 0;
      if (isClosed()) throw new BrowserKernelError("pool_closed", "KernelPool closed");
      if (isTimedOut()) throw new BrowserKernelError("timeout", "Batch timed out");
      throw fatalAbort;
    }
    if (earliestFailed < Number.POSITIVE_INFINITY) {
      outputBuf = null;
      this.#fixedOutputBytes = 0;
      const err = failures.get(earliestFailed);
      throw new BrowserKernelError(err.kind, err.message, earliestFailed);
    }
    return new Uint32Array(outputBuf);
  }

  async #runWorkerVariable(
    mode,
    inputSnapshot,
    itemCount,
    batchSnapshot,
    batchEpoch,
    batchDeadlineMs,
    workerCount,
    isDirectOrFallback,
    activeSlots,
    isClosed,
    isTimedOut,
  ) {
    const inItemBytes = this.manifest.inSchema.stride * 4;
    const queue = [];
    for (let start = 0; start < itemCount; start += this.maxChunkItems) {
      queue.push({ start, end: Math.min(itemCount, start + this.maxChunkItems), retriedFresh: false });
    }
    let frontierItem = 0;
    let settledBytes = 0;
    let higherBytes = 0;
    let earliestFailed = Number.POSITIVE_INFINITY;
    const failures = new Map();
    const cleanupResultBytes = new Map();
    const settledChunks = [];
    const higherChunks = new Map();
    let pendingFrontierGrant = null;
    let fatalAbort = null;
    let activeExecuting = 0;
    const waiters = [];
    const wake = () => { while (waiters.length > 0) waiters.shift()(); };

    const recordFailure = (start, kind, message, resBytes = 0) => {
      if (kind === "cleanup_trap") cleanupResultBytes.set(start, resBytes);
      const prev = failures.get(start);
      if (!prev || failurePriority(kind) > failurePriority(prev.kind)) {
        failures.set(start, { kind, message });
      }
      if (start < earliestFailed) {
        earliestFailed = start;
        for (const [s, c] of [...higherChunks.entries()]) {
          if (s >= earliestFailed) {
            higherChunks.delete(s);
            if (!c.overBudgetHint) {
              higherBytes -= c.byteLength;
              this.#bufferedChunkBytes = settledBytes + higherBytes;
            }
          }
        }
      }
      wake();
    };

    const evictHighest = (minStart) => {
      let maxS = -1;
      for (const [s, c] of higherChunks) {
        if (s > minStart && !c.overBudgetHint && s > maxS) maxS = s;
      }
      if (maxS === -1) return false;
      const c = higherChunks.get(maxS);
      higherChunks.delete(maxS);
      higherBytes -= c.byteLength;
      this.#bufferedChunkBytes = settledBytes + higherBytes;
      if (maxS < earliestFailed) queue.push({ start: maxS, end: c.end, retriedFresh: false });
      return true;
    };

    const tryReleasePending = () => {
      if (!pendingFrontierGrant) return;
      const p = pendingFrontierGrant;
      if (fatalAbort) {
        pendingFrontierGrant = null;
        return p.resolve("abort");
      }
      if (p.range.start >= earliestFailed) {
        pendingFrontierGrant = null;
        return p.resolve("discard");
      }
      if (settledBytes + p.resultBytes > this.#maxTotalOutputBytes) {
        pendingFrontierGrant = null;
        recordFailure(p.range.start, "output_budget_exceeded", "Output budget exceeded");
        return p.resolve("discard");
      }
      while (settledBytes + higherBytes + this.#inFlightGrantBytes + p.resultBytes > this.#maxTotalOutputBytes) {
        if (!evictHighest(frontierItem)) break;
      }
      if (settledBytes + higherBytes + this.#inFlightGrantBytes + p.resultBytes <= this.#maxTotalOutputBytes) {
        pendingFrontierGrant = null;
        this.#inFlightGrantBytes += p.resultBytes;
        this.#recordPeak();
        p.resolve("grant");
      }
    };

    const advanceFrontier = () => {
      for (;;) {
        if (frontierItem === earliestFailed) {
          const cb = cleanupResultBytes.get(frontierItem);
          if (cb !== undefined && settledBytes + cb > this.#maxTotalOutputBytes) {
            recordFailure(frontierItem, "output_budget_exceeded", "Output budget exceeded");
          }
          break;
        }
        const c = higherChunks.get(frontierItem);
        if (!c) break;
        higherChunks.delete(frontierItem);
        if (c.overBudgetHint) {
          if (settledBytes + c.byteLength > this.#maxTotalOutputBytes) {
            recordFailure(frontierItem, "output_budget_exceeded", "Output budget exceeded");
            break;
          }
          queue.push({ start: frontierItem, end: c.end, retriedFresh: false });
          wake();
          break;
        }
        higherBytes -= c.byteLength;
        if (settledBytes + c.byteLength > this.#maxTotalOutputBytes) {
          this.#bufferedChunkBytes = settledBytes + higherBytes;
          recordFailure(frontierItem, "output_budget_exceeded", "Output budget exceeded");
          break;
        }
        settledBytes += c.byteLength;
        this.#bufferedChunkBytes = settledBytes + higherBytes;
        this.#recordPeak();
        settledChunks.push(c.words);
        frontierItem = c.end;
      }
      tryReleasePending();
      wake();
    };

    const getSlot = (i) => isDirectOrFallback ? this.#directSlot : this.#warmSlots[i];
    const setSlot = (i, s) => {
      if (isDirectOrFallback) this.#directSlot = s;
      else this.#warmSlots[i] = s;
    };

    const runLoop = async (slotIdx) => {
      for (;;) {
        if (fatalAbort || isClosed() || isTimedOut()) break;
        if (frontierItem >= itemCount || frontierItem >= earliestFailed) break;
        queue.sort((a, b) => a.start - b.start);
        const range = queue.shift();
        if (!range) {
          if (activeExecuting === 0 && !pendingFrontierGrant) break;
          await new Promise((r) => waiters.push(r));
          continue;
        }
        if (range.start >= earliestFailed) continue;
        activeExecuting++;
        try {
          const ref = { slot: getSlot(slotIdx) };
          const snap = await this.#ensureSlotSnapshot(ref, slotIdx, batchSnapshot, batchDeadlineMs, batchEpoch);
          setSlot(slotIdx, ref.slot);
          if (!snap.ok) {
            if (snap.error.code === "pool_closed" || snap.error.code === "timeout" || snap.error.code === "transport_fault") {
              fatalAbort ??= snap.error;
              break;
            }
            recordFailure(range.start, snap.error.code, snap.error.message);
            advanceFrontier();
            continue;
          }
          const s = snap.slot;
          activeSlots.add(s);
          const eligible = s.storeHasPriorHistory && !range.retriedFresh;
          const slice = inputSnapshot.subarray(range.start * inItemBytes, range.end * inItemBytes);
          const runRes = await s.request({ kind: "run", bytes: slice });
          if (isClosed()) {
            fatalAbort ??= new BrowserKernelError("pool_closed", "KernelPool closed");
            break;
          }
          if (runRes.kind === "fenced" || isTimedOut()) {
            fatalAbort ??= new BrowserKernelError("timeout", "Batch timed out");
            break;
          }
          if (runRes.kind === "arena_oom") {
            this.#retireSlot(s, false);
            setSlot(slotIdx, null);
            if (eligible) {
              range.retriedFresh = true;
              this.#freshRetries++;
              queue.push(range);
              wake();
            } else {
              recordFailure(range.start, "out_of_memory", "Kernel OOM");
              advanceFrontier();
            }
            continue;
          }
          if (runRes.kind === "run_trap") {
            if (!runRes.cleanupOk) { this.#retireSlot(s, false); setSlot(slotIdx, null); }
            else s.storeHasPriorHistory = true;
            recordFailure(range.start, "run_trap", "Kernel trapped");
            advanceFrontier();
            continue;
          }
          if (runRes.kind === "overflow_ok" || runRes.kind === "overflow_cleanup_fail") {
            if (runRes.kind === "overflow_cleanup_fail") {
              this.#retireSlot(s, false);
              setSlot(slotIdx, null);
              recordFailure(
                range.start,
                range.end - range.start === 1 ? "single_item_output_overflow" : "cleanup_trap",
                "Overflow cleanup failed",
              );
              advanceFrontier();
              continue;
            }
            s.storeHasPriorHistory = true;
            if (range.end - range.start === 1) {
              recordFailure(range.start, "single_item_output_overflow", "Single item overflow");
              advanceFrontier();
            } else {
              const mid = range.start + ((range.end - range.start) >> 1);
              queue.push({ start: mid, end: range.end, retriedFresh: false });
              queue.push({ start: range.start, end: mid, retriedFresh: false });
              wake();
            }
            continue;
          }
          if (runRes.kind !== "ready") {
            fatalAbort ??= new BrowserKernelError("transport_fault", "Unexpected run response");
            this.#retireSlot(s, true);
            break;
          }
          const resultBytes = runRes.resultBytes;
          let decision = "discard";
          let requeue = false;
          let overHint = false;
          let frontierOver = false;

          if (range.start >= earliestFailed) {
            decision = "discard";
          } else if (range.start === frontierItem) {
            if (settledBytes + resultBytes > this.#maxTotalOutputBytes) {
              frontierOver = true;
            } else {
              while (settledBytes + higherBytes + this.#inFlightGrantBytes + resultBytes > this.#maxTotalOutputBytes) {
                if (!evictHighest(frontierItem)) break;
              }
              if (settledBytes + higherBytes + this.#inFlightGrantBytes + resultBytes <= this.#maxTotalOutputBytes) {
                this.#inFlightGrantBytes += resultBytes;
                this.#recordPeak();
                decision = "grant";
              } else {
                decision = await new Promise((resolve) => {
                  pendingFrontierGrant = { range, resultBytes, resolve };
                  tryReleasePending();
                });
              }
            }
          } else {
            if (pendingFrontierGrant !== null) {
              requeue = true;
            } else if (settledBytes + resultBytes > this.#maxTotalOutputBytes) {
              overHint = true;
            } else {
              while (settledBytes + higherBytes + this.#inFlightGrantBytes + resultBytes > this.#maxTotalOutputBytes) {
                if (!evictHighest(range.start)) break;
              }
              if (settledBytes + higherBytes + this.#inFlightGrantBytes + resultBytes <= this.#maxTotalOutputBytes) {
                this.#inFlightGrantBytes += resultBytes;
                this.#recordPeak();
                decision = "grant";
              } else {
                requeue = true;
              }
            }
          }

          if (decision === "abort" || fatalAbort) {
            this.#retireSlot(s, false);
            break;
          }
          if (decision === "discard") {
            const dRes = await s.request({ kind: "discard" });
            if (isClosed() || dRes.kind === "fenced" || isTimedOut()) {
              fatalAbort ??= new BrowserKernelError(isClosed() ? "pool_closed" : "timeout", "Interrupted");
              break;
            }
            if (dRes.kind === "error") {
              this.#retireSlot(s, false);
              setSlot(slotIdx, null);
              recordFailure(range.start, frontierOver ? "output_budget_exceeded" : "cleanup_trap", "Cleanup failed", resultBytes);
              advanceFrontier();
              continue;
            }
            s.storeHasPriorHistory = true;
            if (frontierOver) {
              recordFailure(range.start, "output_budget_exceeded", "Output budget exceeded");
              advanceFrontier();
            } else if (overHint && range.start < earliestFailed) {
              higherChunks.set(range.start, { end: range.end, words: null, byteLength: resultBytes, overBudgetHint: true });
              advanceFrontier();
            } else if (requeue && range.start < earliestFailed) {
              queue.push({ start: range.start, end: range.end, retriedFresh: false });
              wake();
            }
            continue;
          }

          s.ownedTransferredBytes = resultBytes;
          const gRes = await s.request({ kind: "copy_grant" });
          if (isClosed() || gRes.kind === "fenced" || isTimedOut()) {
            fatalAbort ??= new BrowserKernelError(isClosed() ? "pool_closed" : "timeout", "Interrupted");
            break;
          }
          s.ownedTransferredBytes = 0;
          this.#inFlightGrantBytes = Math.max(0, this.#inFlightGrantBytes - resultBytes);
          if (gRes.kind === "error") {
            this.#retireSlot(s, false);
            setSlot(slotIdx, null);
            recordFailure(range.start, "cleanup_trap", "Cleanup trapped after copy", resultBytes);
            advanceFrontier();
            continue;
          }
          s.storeHasPriorHistory = true;
          if (range.start >= earliestFailed) {
            tryReleasePending();
            continue;
          }
          if (pendingFrontierGrant !== null && range.start > frontierItem) {
            queue.push({ start: range.start, end: range.end, retriedFresh: false });
            tryReleasePending();
            wake();
            continue;
          }
          higherChunks.set(range.start, {
            end: range.end,
            words: new Uint32Array(gRes.buffer),
            byteLength: resultBytes,
            overBudgetHint: false,
          });
          higherBytes += resultBytes;
          this.#bufferedChunkBytes = settledBytes + higherBytes;
          this.#recordPeak();
          advanceFrontier();
        } finally {
          activeExecuting--;
          wake();
        }
      }
    };

    await Promise.all(Array.from({ length: workerCount }, (_, i) => runLoop(i)));
    if (fatalAbort || isClosed() || isTimedOut()) {
      if (isClosed()) throw new BrowserKernelError("pool_closed", "KernelPool closed");
      if (isTimedOut()) throw new BrowserKernelError("timeout", "Batch timed out");
      throw fatalAbort;
    }
    advanceFrontier();
    if (earliestFailed < Number.POSITIVE_INFINITY) {
      const err = failures.get(earliestFailed);
      throw new BrowserKernelError(err.kind, err.message, earliestFailed);
    }
    if (mode === "variable_chunks") return settledChunks;
    if (settledChunks.length === 1) return settledChunks[0];
    const out = new Uint32Array(settledBytes >>> 2);
    this.#concatScratchBytes = settledBytes;
    this.#recordPeak();
    let off = 0;
    for (let i = 0; i < settledChunks.length; i++) {
      out.set(settledChunks[i], off);
      off += settledChunks[i].length;
      this.#bufferedChunkBytes = Math.max(0, this.#bufferedChunkBytes - settledChunks[i].byteLength);
      settledChunks[i] = new Uint32Array(0);
    }
    return out;
  }

  close() {
    if (this.#closePromise !== null) return this.#closePromise;
    this.#state = "closing";
    this.#poolEpoch++;
    const active = this.#activeOp;
    if (active) active.cancelForClose();
    this.#closePromise = (async () => {
      if (active) await active.done;
      for (const s of [...this.#liveSlots]) {
        // Idle slots (ownedTransferredBytes === 0) retire cleanly; any slot still owning
        // an unacknowledged transferred buffer permanently retires its charge!
        this.#retireSlot(s, s.ownedTransferredBytes > 0);
      }
      this.#warmSlots.fill(null);
      this.#directSlot = null;
      this.#inlineGuest = null;
      this.#currentCtx = null;
      this.#inputSnapshotBytes = 0;
      this.#contextSnapshotBytes = 0;
      this.#fixedOutputBytes = 0;
      this.#bufferedChunkBytes = 0;
      this.#inFlightGrantBytes = 0;
      this.#concatScratchBytes = 0;
      this.#state = "closed";
    })();
    return this.#closePromise;
  }
}
