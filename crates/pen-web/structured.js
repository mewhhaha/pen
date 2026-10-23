// ../blot2/compiler/cleanup_data.ts
var keys = "id,tag,floating,reference,child,row_words,leaf_words,children,fields,variants,name";
var tags = /* @__PURE__ */ new Set([
  "unit",
  "boolean",
  "u32",
  "f32",
  "product",
  "record",
  "array",
  "list",
  "nominal",
  "opaque"
]);
var uint = (value) => typeof value === "number" && Number.isInteger(value) && value >= 0 && value <= 4294967295;
function fail() {
  throw new Error("Invalid closed cleanup data schema");
}
function readDataSchemas(module) {
  const sections = WebAssembly.Module.customSections(module, "blot:cleanup_data");
  if (sections.length === 0) return void 0;
  if (sections.length !== 1 || sections[0].byteLength > 1048576) fail();
  const text = new TextDecoder("utf-8", {
    fatal: true
  }).decode(sections[0]);
  const parsed = JSON.parse(text);
  if (!parsed || Object.keys(parsed).join(",") !== "version,nodes" || parsed.version !== 1 || !Array.isArray(parsed.nodes) || parsed.nodes.length === 0 || parsed.nodes.length > 4096 || JSON.stringify(parsed) !== text) fail();
  const nodes = /* @__PURE__ */ new Map();
  let prior = 0;
  for (const node of parsed.nodes) {
    if (!node || Object.keys(node).join(",") !== keys || !uint(node.id) || node.id <= prior || !tags.has(node.tag) || typeof node.floating !== "boolean" || typeof node.reference !== "boolean" || node.floating && node.reference || !uint(node.child) || !uint(node.row_words) || node.row_words > 16 || !uint(node.leaf_words) || node.tag === "list" && node.leaf_words !== 244 || node.tag !== "list" && node.leaf_words !== 0 || !Array.isArray(node.children) || !node.children.every(uint) || !Array.isArray(node.fields) || !Array.isArray(node.variants) || typeof node.name !== "string") fail();
    prior = node.id;
    const names = /* @__PURE__ */ new Set();
    for (const field of node.fields) {
      if (!field || Object.keys(field).join(",") !== "name,child" || typeof field.name !== "string" || names.has(field.name) || !uint(field.child) || field.child === 0) fail();
      names.add(field.name);
      Object.freeze(field);
    }
    const variants = /* @__PURE__ */ new Set();
    for (const variant of node.variants) {
      if (!variant || Object.keys(variant).join(",") !== "tag,child" || !uint(variant.tag) || variants.has(variant.tag) || !uint(variant.child) || variant.child === 0) fail();
      variants.add(variant.tag);
      Object.freeze(variant);
    }
    const sequence = node.tag === "array" || node.tag === "list";
    if (sequence !== (node.child !== 0) || !sequence && node.row_words !== 0 || node.tag !== "product" && node.tag !== "nominal" && node.tag !== "opaque" && node.children.length !== 0 || node.tag !== "record" && node.fields.length !== 0 || node.tag !== "nominal" && node.variants.length !== 0 || (node.tag === "nominal" || node.tag === "opaque") !== (node.name.length !== 0)) fail();
    if ([
      "unit",
      "boolean",
      "u32",
      "f32"
    ].includes(node.tag) && (node.reference || node.floating !== (node.tag === "f32"))) fail();
    if ([
      "product",
      "record",
      "array",
      "list",
      "nominal"
    ].includes(node.tag) && (!node.reference || node.floating)) fail();
    if (node.tag === "nominal" && node.variants.length === 0) fail();
    Object.freeze(node.children);
    Object.freeze(node.fields);
    Object.freeze(node.variants);
    nodes.set(node.id, Object.freeze(node));
  }
  for (const node of nodes.values()) {
    for (const child of [
      node.child,
      ...node.children,
      ...node.fields.map((field) => field.child),
      ...node.variants.map((variant) => variant.child)
    ]) if (child !== 0 && !nodes.has(child)) fail();
    if (node.row_words !== 0 && packedWords(node.child, /* @__PURE__ */ new Set(), 0) !== node.row_words) fail();
  }
  return nodes;
  function packedWords(id, active, depth) {
    if (depth >= 16 || active.has(id)) fail();
    const node = nodes.get(id);
    if (!node.reference) return 1;
    active.add(id);
    let children;
    if (node.tag === "record") {
      children = node.fields.map((field) => field.child);
    } else if (node.tag === "product") children = node.children;
    else if (node.tag === "nominal" && node.variants.length === 1) {
      children = [
        node.variants[0].child
      ];
    } else fail();
    if (children.length === 0 || children.length > 16) fail();
    let words = 0;
    for (const child of children) {
      words += packedWords(child, active, depth + 1);
      if (words > 16) fail();
    }
    active.delete(id);
    return words;
  }
}
function validateDataRoots(nodes, roots) {
  if (!nodes) {
    if (roots.length !== 0) fail();
    return;
  }
  const seen = /* @__PURE__ */ new Set();
  const path = [];
  let budget = 65536;
  const visit = (id, depth) => {
    if (--budget < 0 || depth >= 128 || !nodes.has(id)) fail();
    const cycle = path.indexOf(id);
    if (cycle >= 0) {
      if (!path.slice(cycle).some((id2) => nodes.get(id2).tag === "nominal")) {
        fail();
      }
      return;
    }
    if (seen.has(id)) return;
    seen.add(id);
    path.push(id);
    const node = nodes.get(id);
    if (node.child !== 0) visit(node.child, depth + 1);
    for (const child of node.children) visit(child, depth + 1);
    for (const field of node.fields) visit(field.child, depth + 1);
    for (const variant of node.variants) visit(variant.child, depth + 1);
    path.pop();
  };
  for (const root of roots) visit(root, 0);
  if (seen.size !== nodes.size) fail();
}
function copyData(memory, nodes, root, word) {
  const view = new DataView(memory.buffer);
  const floatBits = new DataView(new ArrayBuffer(4));
  let budget = 65536;
  const active = /* @__PURE__ */ new Set();
  function range(address, bytes) {
    if (!uint(address) || address % 4 !== 0 || address < 4 || !uint(bytes) || address > view.byteLength || bytes > view.byteLength - address) throw new Error("Cleanup data lies outside guest memory");
  }
  function load(address) {
    range(address, 4);
    return view.getUint32(address, true);
  }
  function block(address, bytes) {
    range(address, bytes);
    if (address < 65536) return;
    range(address - 32, 32);
    const requested = load(address - 28), flags = load(address - 24);
    if (requested < bytes || requested > view.byteLength - address || (flags & 20) !== 4 || load(address - 16) === 0) throw new Error("Cleanup data is not a live typed block");
  }
  function slot(address, index, reference) {
    if (address < 65536) return;
    const flags = load(address - 24);
    let mask = load(address - 12);
    let bit = Math.min(index, 31);
    if ((flags & 1024) !== 0) {
      const words = load(mask);
      if (index >= words) {
        throw new Error("Cleanup data slot exceeds its shape");
      }
      mask = load(mask + 4 + Math.floor(index / 32) * 4);
      bit = index % 32;
    }
    if ((mask >>> bit & 1) !== Number(reference)) {
      throw new Error("Cleanup data slot role differs from its schema");
    }
  }
  function scalar(node, value) {
    if (node.tag === "unit") {
      if (value !== 0) throw new Error("Cleanup Unit is not canonical");
      return null;
    }
    if (node.tag === "boolean") {
      if (value > 1) throw new Error("Cleanup Bool is not canonical");
      return value === 1;
    }
    if (node.tag === "u32") return value;
    if (node.tag === "f32") {
      floatBits.setUint32(0, value, true);
      return floatBits.getFloat32(0, true);
    }
    return Object.freeze({
      kind: "opaque",
      type: node.name
    });
  }
  function aggregate(node, address, depth, inline = false) {
    const fields = node.fields;
    const children = node.children;
    let count = children.length;
    if (node.tag === "record") count = fields.length;
    if (count === 0) {
      if (address !== 0) {
        throw new Error("Cleanup empty data is not canonical");
      }
    } else if (!inline) block(address, count * 4);
    else range(address, count * 4);
    const result = [];
    const record = {};
    for (let index = 0; index < count; index++) {
      let child = children[index];
      if (node.tag === "record") child = fields[index].child;
      if (!inline) slot(address, index, nodes.get(child).reference);
      const value = copy(child, load(address + index * 4), depth + 1);
      if (node.tag === "record") {
        Object.defineProperty(record, fields[index].name, {
          value,
          enumerable: true
        });
      } else result.push(value);
    }
    if (node.tag === "record") return Object.freeze(record);
    return Object.freeze(result);
  }
  function element(node, address, depth, row) {
    if (row) {
      const position = {
        address
      };
      return packed(node, position, depth);
    }
    return copy(node.id, load(address), depth);
  }
  function packed(node, position, depth) {
    if (--budget < 0 || depth >= 128) {
      throw new Error("Cleanup data copy limit");
    }
    if (!node.reference) {
      const value = scalar(node, load(position.address));
      position.address += 4;
      return value;
    }
    if (node.tag === "nominal") {
      const variant = node.variants[0];
      return Object.freeze({
        family: node.name,
        tag: variant.tag,
        value: packed(nodes.get(variant.child), position, depth + 1)
      });
    }
    if (node.tag === "record") {
      const record = {};
      for (const field of node.fields) {
        Object.defineProperty(record, field.name, {
          value: packed(nodes.get(field.child), position, depth + 1),
          enumerable: true
        });
      }
      return Object.freeze(record);
    }
    return Object.freeze(node.children.map((child) => packed(nodes.get(child), position, depth + 1)));
  }
  function copy(id, value, depth) {
    if (--budget < 0 || depth >= 128) {
      throw new Error("Cleanup data copy limit");
    }
    const node = nodes.get(id);
    if (!node) fail();
    value >>>= 0;
    if (!node.reference) return scalar(node, value);
    const key = `${id}:${value}`;
    if (active.has(key)) throw new Error("Cleanup data cycle");
    active.add(key);
    try {
      if (node.tag === "opaque") {
        if (value !== 0) block(value, 4);
        return Object.freeze({
          kind: "opaque",
          type: node.name
        });
      }
      if (node.tag === "product" || node.tag === "record") {
        return aggregate(node, value, depth);
      }
      if (node.tag === "nominal") {
        block(value, 8);
        slot(value, 0, false);
        const tag = load(value);
        let variant;
        for (const candidate of node.variants) {
          if (--budget < 0) throw new Error("Cleanup variant copy limit");
          if (candidate.tag === tag) {
            variant = candidate;
            break;
          }
        }
        if (!variant) {
          throw new Error("Cleanup constructor differs from its schema");
        }
        slot(value, 1, nodes.get(variant.child).reference);
        return Object.freeze({
          family: node.name,
          tag,
          value: copy(variant.child, load(value + 4), depth + 1)
        });
      }
      const child = nodes.get(node.child);
      const words = Math.max(1, node.row_words);
      const values = [];
      if (node.tag === "array") {
        block(value, 4);
        slot(value, 0, false);
        const length2 = load(value);
        if (length2 > budget || length2 > (view.byteLength - value - 4) / (words * 4)) throw new Error("Cleanup array copy limit");
        block(value, 4 + length2 * words * 4);
        for (let index = 0; index < length2; index++) {
          if (words === 1 && node.row_words === 0) {
            slot(value, index + 1, child.reference);
          }
          values.push(element(child, value + 4 + index * words * 4, depth + 1, node.row_words !== 0));
        }
        if (child.tag === "u32") return Uint32Array.from(values);
        if (child.tag === "f32") {
          const floats = new Float32Array(values.length);
          const bits = new DataView(floats.buffer);
          for (let index = 0; index < values.length; index++) {
            bits.setUint32(index * 4, load(value + 4 + index * 4), true);
          }
          return floats;
        }
        return Object.freeze(values);
      }
      if (node.tag !== "list") fail();
      block(value, 32);
      slot(value, 0, false);
      slot(value, 1, true);
      slot(value, 4, false);
      slot(value, 6, false);
      const length = load(value), scalarElements = load(value + 16), leafLimit = load(value + 24);
      let expectedScalar = 0;
      if (node.row_words !== 0) expectedScalar = words;
      else if (!child.reference) expectedScalar = 1;
      if (length % words !== 0 || length / words > budget || scalarElements !== expectedScalar || leafLimit !== node.leaf_words - node.leaf_words % words) throw new Error("Cleanup list representation differs from its schema");
      const root2 = load(value + 4);
      const leafWords = node.leaf_words;
      const address = (node2, index, steps) => {
        if (--budget < 0 || steps >= 128 || node2 === 0) {
          throw new Error("Cleanup list copy limit");
        }
        block(node2, 16);
        const height = load(node2 + 4), length2 = load(node2 + 8), capacity = load(node2 + 12);
        if (index >= length2 || height > 128) {
          throw new Error("Cleanup list node is invalid");
        }
        if (height === 0) {
          if (capacity < length2 || capacity > leafWords || length2 % words !== 0 || index + words > length2) {
            throw new Error("Cleanup list leaf capacity is invalid");
          }
          block(node2, 16 + capacity * 4);
          slot(node2, index + 4, child.reference && words === 1);
          return node2 + 16 + index * 4;
        }
        block(node2, 24);
        slot(node2, 4, true);
        slot(node2, 5, true);
        const left = load(node2 + 16), right = load(node2 + 20);
        block(left, 16);
        const split = load(left + 8);
        if (index < split) return address(left, index, steps + 1);
        return address(right, index - split, steps + 1);
      };
      for (let index = 0; index < length; index += words) {
        values.push(element(child, address(root2, index, 0), depth + 1, node.row_words !== 0));
      }
      return Object.freeze(values);
    } finally {
      active.delete(key);
    }
  }
  return copy(root, word, 0);
}

// ../blot2/compiler/resource_journal.ts
var ResourceTokens = class {
  #next = 1;
  #live = 0;
  reserve() {
    if (this.#live === 4096 || this.#next > 4294967295) {
      throw new Error("Resource journal limit");
    }
    this.#live++;
    return this.#next++;
  }
  release() {
    if (this.#live === 0) throw new Error("Resource token has no owner");
    this.#live--;
  }
};
var ResourceJournal = class {
  adapters;
  invalidate;
  bindings;
  tokens;
  #bindings;
  #entries;
  #count;
  #phase;
  constructor(adapters, invalidate, bindings, tokens = new ResourceTokens()) {
    this.adapters = adapters;
    this.invalidate = invalidate;
    this.bindings = bindings;
    this.tokens = tokens;
    this.#entries = new Uint32Array(4096 * 7);
    this.#count = 0;
    this.#phase = "idle";
    if (bindings) this.#bindings = new Uint32Array(4096);
  }
  begin() {
    if (this.#phase !== "idle") {
      throw new Error("Resource invocation is active");
    }
    this.#phase = "active";
  }
  reserve() {
    if (this.#phase !== "active") {
      throw new Error("No active resource invocation");
    }
    if (this.#count === 4096) {
      throw new Error("Resource journal limit");
    }
    const token = this.tokens.reserve();
    const offset = this.#count++ * 7;
    this.#entries[offset] = token;
    this.#entries[offset + 1] = 0;
    return token;
  }
  register(token, adapter, payload, callable, provider) {
    this.#register(token, adapter, payload, callable, provider, 0);
  }
  registerOwned(token, adapter, payload, callable, provider, owner) {
    if (!this.invalidate || !Number.isInteger(owner) || owner <= 0 || owner > 4294967295) throw new Error("Invalid resource owner registration");
    this.#register(token, adapter, payload, callable, provider, owner);
  }
  #register(token, adapter, payload, callable, provider, owner) {
    const offset = this.#find(token);
    if (this.#phase !== "active" || this.#entries[offset + 1] !== 0 || !Number.isInteger(adapter) || adapter < 0 || adapter >= this.adapters.length) throw new Error("Invalid resource registration");
    const binding = this.bindings?.capture() ?? 0;
    if (binding !== 0) this.bindings.retain(binding);
    if (this.#bindings) this.#bindings[offset / 7] = binding;
    this.#entries[offset + 2] = adapter;
    this.#entries[offset + 3] = payload;
    this.#entries[offset + 4] = callable;
    this.#entries[offset + 5] = provider;
    this.#entries[offset + 6] = owner;
    this.#entries[offset + 1] = 1;
  }
  release(token) {
    if (this.#phase !== "active") {
      throw new Error("No active resource invocation");
    }
    try {
      this.#release(this.#find(token));
    } finally {
      this.#reclaim();
    }
  }
  /** Move an exact cleanup obligation with its typed owner/captures. Capacity
   * and adapter domains are admitted before either journal changes. */
  transfer(token, destination) {
    if (this.#phase !== "active" || destination.#phase !== "active" || destination === this || this.tokens !== destination.tokens || this.adapters !== destination.adapters || this.bindings !== destination.bindings || this.invalidate !== destination.invalidate) throw new Error("Resource transfer crosses an ownership domain");
    const offset = this.#find(token);
    const state = this.#entries[offset + 1];
    if (state !== 1 && state !== 4) {
      throw new Error("Resource is not owned");
    }
    if (destination.#count === 4096) throw new Error("Resource journal limit");
    const target = destination.#count * 7;
    for (let index = 0; index < 7; index++) {
      destination.#entries[target + index] = this.#entries[offset + index];
    }
    if (this.#bindings) {
      destination.#bindings[destination.#count] = this.#bindings[offset / 7];
      this.#bindings[offset / 7] = 0;
    }
    destination.#count++;
    this.#entries[offset + 1] = 2;
    this.#entries[offset + 6] = 0;
    this.#reclaim();
  }
  /** Counted destruction queues an attempt without running user code inside
   * the heap transaction. The next safe point drains these retired owners. */
  retire(token) {
    if (this.#phase === "idle") {
      throw new Error("No active resource invocation");
    }
    const offset = this.#find(token);
    if (this.#entries[offset + 1] !== 1 && this.#entries[offset + 1] !== 4) {
      throw new Error("Resource is not owned");
    }
    this.#entries[offset + 6] = 0;
    this.#entries[offset + 1] = 3;
  }
  drain() {
    if (this.#phase === "draining") return;
    if (this.#phase !== "active") {
      throw new Error("No active resource invocation");
    }
    this.#phase = "draining";
    try {
      this.#drain(false);
    } finally {
      this.#reclaim();
      this.#phase = "active";
    }
  }
  #reclaim() {
    let target = 0;
    for (let index = 0; index < this.#count; index++) {
      const offset = index * 7;
      if (this.#entries[offset + 1] === 2) continue;
      if (target !== index) {
        this.#entries.copyWithin(target * 7, offset, offset + 7);
        if (this.#bindings) this.#bindings[target] = this.#bindings[index];
      }
      target++;
    }
    this.#count = target;
  }
  #find(token) {
    if (!Number.isInteger(token) || token <= 0 || token > 4294967295) {
      throw new Error("Invalid resource token");
    }
    for (let index = this.#count - 1; index >= 0; index--) {
      if (this.#entries[index * 7] === token) return index * 7;
    }
    throw new Error("Stale resource token");
  }
  #release(offset) {
    if (this.#entries[offset + 1] !== 1 && this.#entries[offset + 1] !== 3 && this.#entries[offset + 1] !== 4) {
      throw new Error("Resource is not owned");
    }
    const phase = this.#phase;
    this.#phase = "draining";
    this.#entries[offset + 1] = 2;
    this.tokens.release();
    const adapter = this.adapters[this.#entries[offset + 2]];
    const payload = this.#entries[offset + 3];
    const callable = this.#entries[offset + 4];
    const provider = this.#entries[offset + 5];
    const owner = this.#entries[offset + 6];
    const token = this.#entries[offset];
    this.#entries[offset + 6] = 0;
    const binding = this.#bindings?.[offset / 7] ?? 0;
    if (this.#bindings) this.#bindings[offset / 7] = 0;
    let failed = false;
    let failure;
    try {
      if (owner !== 0) this.invalidate(owner, token);
      adapter.cleanup(payload, callable, provider);
    } catch (cause) {
      failed = true;
      failure = cause;
    }
    try {
      adapter.drop(payload, callable, provider);
    } catch (cause) {
      if (!failed) {
        failed = true;
        failure = cause;
      }
    }
    if (binding !== 0) {
      try {
        this.bindings.release(binding);
      } catch (cause) {
        if (!failed) {
          failed = true;
          failure = cause;
        }
      }
    }
    this.#phase = phase;
    if (failed) throw failure;
  }
  /** Native typed leases have already published a strong owner edge. */
  promote(token) {
    if (this.#phase !== "active") {
      throw new Error("No active resource invocation");
    }
    const offset = this.#find(token);
    const state = this.#entries[offset + 1];
    if (state !== 1 && state !== 4 || this.#entries[offset + 6] === 0) {
      throw new Error("Resource has no live owning lease");
    }
    this.#entries[offset + 1] = 4;
  }
  #drain(all, persistent = false) {
    let failed = false;
    let failure;
    while (true) {
      let selected = -1;
      for (let index = this.#count - 1; index >= 0; index--) {
        const state = this.#entries[index * 7 + 1];
        if (state === 3 || all && state === 1 || persistent && state === 4) {
          selected = index * 7;
          break;
        }
      }
      if (selected === -1) break;
      try {
        this.#release(selected);
      } catch (cause) {
        if (!failed) {
          failed = true;
          failure = cause;
        }
      }
    }
    if (failed) throw failure;
  }
  finish() {
    if (this.#phase === "idle") return;
    if (this.#phase !== "active") throw new Error("Resource cleanup is active");
    this.#phase = "draining";
    try {
      this.#drain(true);
    } finally {
      for (let index = 0; index < this.#count; index++) {
        const offset = index * 7;
        if (this.#entries[offset + 1] === 0) {
          this.#entries[offset + 1] = 2;
          this.tokens.release();
        }
      }
      this.#reclaim();
      this.#phase = "idle";
    }
  }
  dispose() {
    if (this.#phase !== "active") {
      throw new Error("No active resource invocation");
    }
    this.#phase = "draining";
    try {
      this.#drain(true, true);
    } finally {
      for (let index = 0; index < this.#count; index++) {
        if (this.#entries[index * 7 + 1] === 0) this.tokens.release();
      }
      this.#count = 0;
      this.#phase = "active";
    }
  }
};

// ../blot2/compiler/task_runtime.ts
var TaskError = class extends Error {
  code;
  constructor(code, message) {
    super(message), this.code = code;
    this.name = "TaskError";
  }
};
var TaskFailure = class extends Error {
  failures;
  constructor(failures) {
    super("Structured task failed", {
      cause: failures[0]
    }), this.failures = failures;
    this.name = "TaskFailure";
  }
};
var Cancelled = class extends Error {
};
function promise(value) {
  return value instanceof Promise;
}
function after(value, next) {
  if (promise(value)) return value.then(next);
  return next(value);
}
var TaskRuntime = class {
  limit;
  #tasks;
  #groups;
  #next;
  #disposed;
  #root;
  constructor(limit = 4096) {
    this.limit = limit;
    this.#tasks = /* @__PURE__ */ new Map();
    this.#groups = /* @__PURE__ */ new Map();
    this.#next = 1;
    this.#disposed = false;
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 4096) {
      throw new TaskError("task_budget", "Invalid task budget");
    }
    this.#root = {
      id: 0,
      ordinal: 0,
      captures: 0,
      groups: /* @__PURE__ */ new Set(),
      cleanups: [],
      state: "running",
      consumed: false,
      cancel: false,
      failures: []
    };
  }
  #identity() {
    if (this.#disposed) {
      throw new TaskError("task_host_contract", "Disposed task runtime");
    }
    if (this.#next > 4294967295 || this.#tasks.size + this.#groups.size >= this.limit) {
      throw new TaskError("task_budget", "Task identity or live-record budget exceeded");
    }
    return this.#next++;
  }
  #task(id, consume = false) {
    if (this.#disposed || !Number.isInteger(id) || id <= 0) {
      throw new TaskError("task_host_contract", "Invalid task identity");
    }
    const task = this.#tasks.get(id);
    if (!task) {
      throw new TaskError("task_host_contract", "Stale task generation");
    }
    if (consume) {
      if (task.consumed) {
        throw new TaskError("task_handle_consumed", "Task handle was consumed");
      }
      task.consumed = true;
    }
    return task;
  }
  #group(id) {
    if (this.#disposed || !Number.isInteger(id) || id <= 0) {
      throw new TaskError("task_host_contract", "Invalid task group identity");
    }
    const group = this.#groups.get(id);
    if (!group) {
      throw new TaskError("task_escape", "Task group is closed or foreign");
    }
    return group;
  }
  #context(task) {
    return Object.freeze({
      id: task.id,
      checkpoint: () => {
        if (task.cancel) throw new Cancelled();
        for (const group of task.groups) {
          const failures = this.#failures(group);
          if (failures.length !== 0) throw new TaskFailure(failures);
        }
        if (task.state !== "running" && task.state !== "suspended") {
          throw new TaskError("task_host_contract", "Task context has terminated");
        }
      },
      scope: () => {
        if (task.cancel) throw new Cancelled();
        if (task.state !== "running" && task.state !== "suspended") {
          throw new TaskError("task_escape", "Task context has terminated");
        }
        return this.#open(task);
      },
      cleanup: (action) => {
        if (task.state !== "running" && task.state !== "suspended") {
          throw new TaskError("task_escape", "Cleanup registered after task termination");
        }
        if (typeof action !== "function") {
          throw new TaskError("task_host_contract", "Invalid task cleanup");
        }
        if (task.cleanups.length >= this.limit) {
          throw new TaskError("task_budget", "Task cleanup budget exceeded");
        }
        task.cleanups.push(action);
      }
    });
  }
  #open(owner) {
    const id = this.#identity();
    const group = {
      id,
      owner,
      children: [],
      state: "open"
    };
    this.#groups.set(id, group);
    try {
      owner.groups.add(group);
    } catch (cause) {
      this.#groups.delete(id);
      throw cause;
    }
    return id;
  }
  open() {
    return this.#open(this.#root);
  }
  spawn(groupId, adapter, captures, callerId = 0) {
    const group = this.#group(groupId);
    let caller = this.#root;
    if (callerId !== 0) caller = this.#task(callerId);
    if (caller.state !== "running" && caller.state !== "suspended" || caller.cancel || group.owner.cancel) {
      throw new TaskError("task_escape", "Task owner is terminating");
    }
    if (group.state !== "open") {
      let child = caller;
      let descendant = false;
      while (child?.group) {
        if (child.group === group) descendant = true;
        child = child.group.owner;
      }
      if (group.state !== "closing" || !descendant) {
        throw new TaskError("task_escape", "Cannot spawn outside the group's live descendants");
      }
    }
    if (!Number.isInteger(captures) || captures < 0 || captures > 4294967295 || typeof adapter?.run !== "function" || typeof adapter.dropCaptures !== "function" || typeof adapter.dropResult !== "function") {
      throw new TaskError("task_host_contract", "Invalid task adapter");
    }
    const id = this.#identity();
    const task = {
      id,
      group,
      parent: caller,
      ordinal: group.children.length,
      adapter,
      captures,
      groups: /* @__PURE__ */ new Set(),
      cleanups: [],
      state: "runnable",
      consumed: false,
      cancel: false,
      failures: []
    };
    this.#tasks.set(id, task);
    try {
      group.children.push(task);
    } catch (cause) {
      this.#tasks.delete(id);
      throw cause;
    }
    return id;
  }
  /** Undo publication that failed before the compiler exposed its handle.
   * The caller still owns captures; no thunk or cleanup may have started. */
  rollbackSpawn(id) {
    const task = this.#task(id);
    const group = task.group;
    if (task.state !== "runnable" || task.consumed || group.state === "closed" || group.children.at(-1) !== task) {
      throw new TaskError("task_host_contract", "Task publication cannot roll back after execution");
    }
    group.children.pop();
    this.#tasks.delete(id);
  }
  poll(id) {
    return this.#task(id).state;
  }
  #terminal(task) {
    return task.state === "succeeded" || task.state === "failed" || task.state === "cancelled";
  }
  #failure(task, cause) {
    if (cause instanceof Cancelled) task.cancel = true;
    else if (cause instanceof TaskFailure) {
      for (const failure of cause.failures) {
        if (task.group?.children.some((sibling) => sibling !== task && sibling.failures.includes(failure))) continue;
        if (!task.failures.includes(failure)) task.failures.push(failure);
      }
      if (task.failures.length === 0) task.cancel = true;
    } else task.failures.push(cause);
    if (task.failures.length !== 0 && task.group) {
      for (const sibling of task.group.children) {
        if (sibling !== task) this.#request(sibling);
      }
    }
  }
  #request(task) {
    if (this.#terminal(task)) return;
    task.cancel = true;
    for (const group of task.groups) {
      for (const child of group.children) this.#request(child);
    }
  }
  #discard(task) {
    if (task.result === void 0) return;
    const value = task.result;
    task.result = void 0;
    try {
      task.adapter.dropResult(value);
    } catch (cause) {
      this.#failure(task, cause);
    }
  }
  #finish(task) {
    const groups = [
      ...task.groups
    ];
    const closeNext = (index) => {
      if (index === groups.length) return;
      const group = groups[index];
      if (group.state === "closed") return closeNext(index + 1);
      return after(this.#close(group), () => {
        const failures = this.#failures(group);
        for (const cause of failures) this.#failure(task, cause);
        return closeNext(index + 1);
      });
    };
    return after(closeNext(0), () => {
      for (let index = task.cleanups.length - 1; index >= 0; index--) {
        try {
          task.cleanups[index]();
        } catch (cause) {
          this.#failure(task, cause);
        }
      }
      task.cleanups.length = 0;
      try {
        task.adapter.dropCaptures(task.captures);
      } catch (cause) {
        this.#failure(task, cause);
      }
      if (task.cancel || task.failures.length !== 0) this.#discard(task);
      if (task.failures.length !== 0) task.state = "failed";
      else if (task.cancel) task.state = "cancelled";
      else task.state = "succeeded";
    });
  }
  #start(task) {
    if (this.#terminal(task)) return;
    if (task.state !== "runnable") return task.completion;
    task.state = "running";
    let result;
    try {
      if (task.cancel) throw new Cancelled();
      result = task.adapter.run(this.#context(task), task.captures);
      if (!promise(result) && (!Number.isInteger(result) || result < 0 || result > 4294967295)) {
        throw new TaskError("task_host_contract", "Invalid task result envelope");
      }
    } catch (cause) {
      this.#failure(task, cause);
      const completion2 = this.#finish(task);
      if (promise(completion2)) task.completion = completion2;
      return completion2;
    }
    const publish = (value) => {
      if (!Number.isInteger(value) || value < 0 || value > 4294967295) {
        this.#failure(task, new TaskError("task_host_contract", "Invalid task result envelope"));
      } else task.result = value;
      return this.#finish(task);
    };
    if (promise(result)) {
      task.state = "suspended";
      task.completion = result.then(publish, (cause) => {
        this.#failure(task, cause);
        return this.#finish(task);
      });
      return task.completion;
    }
    const completion = publish(result);
    if (promise(completion)) {
      task.state = "suspended";
      task.completion = completion;
    }
    return completion;
  }
  #run(group, caller) {
    const pending = [];
    for (const child of group.children) {
      let ancestor = caller;
      let waitingOnSelf = false;
      while (ancestor) {
        if (child === ancestor) waitingOnSelf = true;
        ancestor = ancestor.parent;
      }
      if (waitingOnSelf) continue;
      const completion = this.#start(child);
      if (promise(completion)) pending.push(completion);
    }
    if (pending.length !== 0) {
      return Promise.all(pending).then(() => {
        if (group.children.some((child) => child.state === "runnable")) return this.#run(group, caller);
      });
    }
  }
  #failures(group) {
    const failures = [];
    for (const child of group.children) failures.push(...child.failures);
    return failures;
  }
  join(id, callerId = 0) {
    let caller = this.#root;
    if (callerId !== 0) caller = this.#task(callerId);
    const task = this.#task(id, true);
    if (task === caller) {
      throw new TaskError("task_host_contract", "A task cannot join itself");
    }
    this.#run(task.group, caller);
    return after(task.completion, () => {
      const failures = this.#failures(task.group);
      if (failures.length !== 0) {
        return after(this.#run(task.group, caller), () => {
          throw new TaskFailure(this.#failures(task.group));
        });
      }
      if (task.state === "cancelled") throw new Cancelled();
      if (task.result === void 0) {
        throw new TaskError("task_host_contract", "Missing terminal result");
      }
      const value = task.result;
      task.result = void 0;
      return value;
    });
  }
  /** Establish the complete dependency set before running any child. Results
   * stay in their task records until all selected tasks succeed, so scope
   * cleanup still owns every partial result if collection fails. */
  collect(ids, callerId = 0) {
    if (ids.length > this.limit) {
      throw new TaskError("task_budget", "Task collection exceeds its budget");
    }
    const caller = callerId === 0 ? this.#root : this.#task(callerId);
    if (caller.cancel || caller.state !== "running" && caller.state !== "suspended") {
      throw new TaskError("task_escape", "Collecting task is terminating");
    }
    const tasks = [];
    const selected = /* @__PURE__ */ new Set();
    const groups = /* @__PURE__ */ new Set();
    for (const id of ids) {
      const task = this.#task(id);
      if (task.consumed || selected.has(task)) {
        throw new TaskError("task_handle_consumed", "Task collection contains a consumed or repeated handle");
      }
      for (let ancestor = caller; ancestor; ancestor = ancestor.parent) {
        if (task === ancestor) {
          throw new TaskError("task_host_contract", "A task cannot collect itself or an ancestor");
        }
      }
      tasks.push(task);
      selected.add(task);
      groups.add(task.group);
    }
    for (const task of tasks) task.consumed = true;
    for (const group of groups) this.#run(group, caller);
    const pending = tasks.flatMap((task) => task.completion ? [
      task.completion
    ] : []);
    const complete = () => {
      const failures = [
        ...groups
      ].flatMap((group) => this.#failures(group));
      if (failures.length !== 0) {
        const draining = [];
        for (const group of groups) {
          const completion = this.#run(group, caller);
          if (promise(completion)) draining.push(completion);
        }
        const failed = () => {
          throw new TaskFailure([
            ...groups
          ].flatMap((group) => this.#failures(group)));
        };
        if (draining.length !== 0) return Promise.all(draining).then(failed);
        return failed();
      }
      for (const task of tasks) {
        if (task.state === "cancelled") throw new Cancelled();
        if (task.state !== "succeeded" || task.result === void 0) {
          throw new TaskError("task_host_contract", "Missing collected task result");
        }
      }
      const values = tasks.map((task) => task.result);
      for (const task of tasks) task.result = void 0;
      return values;
    };
    return pending.length === 0 ? complete() : Promise.all(pending).then(complete);
  }
  cancel(id) {
    const task = this.#task(id, true);
    this.#request(task);
    return after(this.#start(task), () => {
      this.#discard(task);
      if (task.failures.length !== 0) throw new TaskFailure([
        ...task.failures
      ]);
    });
  }
  #close(group) {
    if (group.state === "closed") return;
    group.state = "closing";
    return after(this.#run(group), () => {
      for (const task of group.children) {
        this.#discard(task);
        this.#tasks.delete(task.id);
      }
      group.state = "closed";
      this.#groups.delete(group.id);
      group.owner.groups.delete(group);
    });
  }
  close(id) {
    const group = this.#group(id);
    return after(this.#close(group), () => {
      const failures = this.#failures(group);
      if (failures.length !== 0) throw new TaskFailure(failures);
    });
  }
  /** Complete children before an enclosing lexical resource is retired. The
   * group remains open and its affine handles remain available for joining. */
  drain(owner = 0) {
    const task = owner === 0 ? this.#root : this.#task(owner);
    const groups = [
      ...task.groups
    ];
    const next = (index) => {
      if (index === groups.length) return;
      const group = groups[index];
      return after(this.#run(group), () => {
        const failures = this.#failures(group);
        if (failures.length !== 0) throw new TaskFailure(failures);
        return next(index + 1);
      });
    };
    return next(0);
  }
  finishInvocation(cancel = false) {
    const groups = [
      ...this.#root.groups
    ];
    if (cancel) {
      for (const group of groups) {
        for (const child of group.children) this.#request(child);
      }
    }
    const next = (index) => {
      if (index === groups.length) return;
      return after(this.#close(groups[index]), () => next(index + 1));
    };
    return after(next(0), () => {
      const failures = groups.flatMap((group) => this.#failures(group));
      if (failures.length !== 0) throw new TaskFailure(failures);
    });
  }
  get rootContext() {
    return this.#context(this.#root);
  }
  /** Teardown retains every suspended child until its Promise settles. */
  dispose() {
    if (this.#disposed) return;
    this.#root.cancel = true;
    const groups = [
      ...this.#root.groups
    ];
    for (const group of groups) {
      for (const task of group.children) this.#request(task);
    }
    const drain = (index) => {
      if (index === groups.length) return;
      return after(this.#close(groups[index]), () => drain(index + 1));
    };
    return after(drain(0), () => {
      this.#disposed = true;
      const failures = [];
      for (const group of groups) {
        failures.push(...this.#failures(group));
      }
      if (failures.length !== 0) throw new TaskFailure(failures);
    });
  }
  get liveRecords() {
    return this.#tasks.size + this.#groups.size;
  }
};

// ../blot2/compiler/task_guest.ts
var taskImports = [
  [
    "open",
    0,
    true
  ],
  [
    "close",
    1,
    false
  ],
  [
    "spawn",
    3,
    true
  ],
  [
    "join",
    2,
    true
  ],
  [
    "collect",
    3,
    false
  ],
  [
    "cancel",
    1,
    false
  ],
  [
    "poll",
    2,
    true
  ],
  [
    "drop",
    2,
    false
  ],
  [
    "drain",
    0,
    false
  ],
  [
    "checkpoint",
    0,
    false
  ],
  [
    "current",
    0,
    true
  ],
  [
    "resume",
    1,
    false
  ],
  [
    "transfer",
    1,
    false
  ]
];
function taskExports(count) {
  const names = [
    "blot:task_context/create",
    "blot:task_context/save",
    "blot:task_context/restore",
    "blot:task_context/drop",
    "blot:task_retain",
    "blot:task_release"
  ];
  for (let id = 0; id < count; id++) {
    for (const role of [
      "start",
      "move_captures",
      "move_result"
    ]) {
      names.push(`blot:task_${role}/${id}`);
    }
  }
  return names;
}
function invalid(message) {
  throw new TaskError("task_host_contract", message);
}
function typedTaskFunctions(functions, signatures = taskImports) {
  const uleb = (value) => {
    const bytes = [];
    do {
      let byte = value & 127;
      value >>>= 7;
      if (value !== 0) byte |= 128;
      bytes.push(byte);
    } while (value !== 0);
    return bytes;
  };
  const name = (text) => {
    const bytes = [
      ...new TextEncoder().encode(text)
    ];
    return [
      ...uleb(bytes.length),
      ...bytes
    ];
  };
  const section = (id, bytes) => [
    id,
    ...uleb(bytes.length),
    ...bytes
  ];
  const types = [
    ...uleb(signatures.length)
  ];
  const imports = [
    ...uleb(signatures.length)
  ];
  const exports = [
    ...uleb(signatures.length)
  ];
  for (const [index, [key, parameters, result]] of signatures.entries()) {
    types.push(96, ...uleb(parameters), ...Array(parameters).fill(127));
    types.push(Number(result));
    if (result) types.push(127);
    imports.push(...name("task"), ...name(key), 0, ...uleb(index));
    exports.push(...name(key), 0, ...uleb(index));
  }
  const module = new WebAssembly.Module(new Uint8Array([
    0,
    97,
    115,
    109,
    1,
    0,
    0,
    0,
    ...section(1, types),
    ...section(2, imports),
    ...section(7, exports)
  ]));
  return new WebAssembly.Instance(module, {
    task: functions
  }).exports;
}
var TaskGuest = class {
  descriptors;
  journal;
  select;
  ownership;
  promising;
  #runtime;
  #entries;
  #handles;
  #hooks;
  #memory;
  #root;
  #active;
  #source;
  #destination;
  #moves;
  #asynchronous;
  #cleaning;
  constructor(exports, descriptors, journal, select, ownership, promising) {
    this.descriptors = descriptors;
    this.journal = journal;
    this.select = select;
    this.ownership = ownership;
    this.promising = promising;
    this.#runtime = new TaskRuntime();
    this.#entries = /* @__PURE__ */ new Map();
    this.#handles = /* @__PURE__ */ new Map();
    this.#asynchronous = false;
    this.#cleaning = 0;
    this.imports = {
      open: () => this.#owner().context.scope(),
      close: (group) => {
        const owner = this.#owner();
        const forget = () => {
          for (const [id, entry] of this.#handles) {
            if (entry.group === group >>> 0) this.#handles.delete(id);
          }
        };
        try {
          const completion = this.#restore(owner, this.#runtime.close(group >>> 0));
          if (completion instanceof Promise) return completion.finally(forget);
          forget();
          return completion;
        } catch (cause) {
          forget();
          throw cause;
        }
      },
      spawn: (group, descriptor, captures) => {
        const parent = this.#owner();
        descriptor >>>= 0;
        captures >>>= 0;
        if (descriptor >= this.descriptors) invalid("Unknown task descriptor");
        const resources = this.journal();
        const results = this.journal();
        resources.begin();
        results.begin();
        const native = this.#call("blot:task_context/create", 1);
        const entry = {
          native,
          group: group >>> 0,
          context: parent.context,
          resources,
          results,
          descriptor,
          captures,
          entered: false,
          capturesLive: true,
          closed: false
        };
        this.#call("blot:task_retain", captures);
        let moved = false;
        let published;
        try {
          this.#move(parent.resources, resources, `blot:task_move_captures/${descriptor}`, captures);
          moved = true;
          const id = this.#runtime.spawn(group >>> 0, {
            run: (context) => {
              entry.context = context;
              this.#activate(entry);
              entry.entered = true;
              context.cleanup(() => this.#retire(entry));
              this.ownership.enter();
              const fn = this.#hooks[`blot:task_start/${descriptor}`];
              const publish = (result) => {
                this.#activate(entry);
                try {
                  this.#move(resources, results, `blot:task_move_result/${descriptor}`, result);
                } catch (cause) {
                  this.#call("blot:task_release", result);
                  throw cause;
                }
                return result >>> 0;
              };
              if (this.#asynchronous) {
                return this.promising(fn)(captures).then((value) => {
                  if (typeof value !== "number") {
                    invalid("Invalid task result envelope");
                  }
                  return publish(value);
                });
              }
              return publish(fn(captures));
            },
            dropCaptures: () => this.#retire(entry),
            dropResult: (result) => {
              this.select(results);
              try {
                this.#call("blot:task_release", result);
                this.cleanup(() => results.finish());
              } finally {
                this.select(this.#owner().resources);
              }
            }
          }, captures, parent.context.id);
          published = id;
          entry.id = id;
          this.#entries.set(id, entry);
          this.#handles.set(id, entry);
          return id;
        } catch (cause) {
          if (published !== void 0) this.#runtime.rollbackSpawn(published);
          if (moved) {
            this.#move(resources, parent.resources, `blot:task_move_captures/${descriptor}`, captures);
          }
          this.#retire(entry);
          throw cause;
        }
      },
      join: (handle, descriptor) => {
        const owner = this.#owner();
        const entry = this.#handles.get(handle >>> 0);
        if (!entry || entry.descriptor !== descriptor >>> 0) {
          invalid("Task descriptor changed");
        }
        this.#handles.delete(handle >>> 0);
        const publish = (result2) => {
          this.#activate(owner);
          try {
            this.#move(entry.results, owner.resources, `blot:task_move_result/${entry.descriptor}`, result2);
          } catch (cause) {
            try {
              this.#call("blot:task_release", result2);
            } finally {
              this.cleanup(() => entry.results.finish());
            }
            throw cause;
          }
          entry.results.finish();
          return result2;
        };
        const result = this.#runtime.join(handle >>> 0, owner.context.id);
        if (result instanceof Promise) return result.then(publish);
        return publish(result);
      },
      collect: (input, count, output) => {
        const owner = this.#owner();
        const memory = this.#memory ?? invalid("Task collection has no memory");
        input >>>= 0;
        count >>>= 0;
        output >>>= 0;
        const view = () => new DataView(memory.buffer);
        const range = (pointer, bytes) => {
          if (pointer === 0 || pointer % 4 !== 0 || pointer > memory.buffer.byteLength - bytes) invalid("Invalid task collection storage");
        };
        if (count === 0 || count > 4096) invalid("Invalid task collection size");
        range(input, count * 8);
        range(output, count * 4);
        const entries = [];
        const handles = [];
        const seen = /* @__PURE__ */ new Set();
        for (let index = 0; index < count; index++) {
          const handle = view().getUint32(input + index * 8, true);
          const descriptor = view().getUint32(input + index * 8 + 4, true);
          const entry = this.#handles.get(handle);
          if (!entry || entry.descriptor !== descriptor || seen.has(handle)) {
            invalid("Task collection contains an invalid or repeated handle");
          }
          entries.push(entry);
          handles.push(handle);
          seen.add(handle);
        }
        for (const handle of handles) this.#handles.delete(handle);
        const publish = (results) => {
          this.#activate(owner);
          let moved = 0;
          try {
            for (let index = 0; index < count; index++) {
              const entry = entries[index];
              this.#move(entry.results, owner.resources, `blot:task_move_result/${entry.descriptor}`, results[index]);
              moved++;
            }
            for (const entry of entries) entry.results.finish();
            const target = view();
            for (let index = 0; index < count; index++) {
              target.setUint32(output + index * 4, results[index], true);
            }
          } catch (cause) {
            for (let index = moved - 1; index >= 0; index--) {
              const entry = entries[index];
              try {
                this.#move(owner.resources, entry.results, `blot:task_move_result/${entry.descriptor}`, results[index]);
              } catch {
              }
            }
            for (let index = 0; index < count; index++) {
              try {
                this.#call("blot:task_release", results[index]);
              } catch {
              }
              try {
                this.cleanup(() => entries[index].results.finish());
              } catch {
              }
            }
            throw cause;
          }
        };
        try {
          const results = this.#runtime.collect(handles, owner.context.id);
          if (results instanceof Promise) {
            return results.then(publish, (cause) => {
              this.#activate(owner);
              throw cause;
            });
          }
          return publish(results);
        } catch (cause) {
          this.#activate(owner);
          throw cause;
        }
      },
      cancel: (handle) => {
        const owner = this.#owner();
        this.#handles.delete(handle >>> 0);
        return this.#restore(owner, this.#runtime.cancel(handle >>> 0));
      },
      poll: (handle, descriptor) => {
        const entry = this.#handles.get(handle >>> 0);
        if (!entry || entry.descriptor !== descriptor >>> 0) {
          invalid("Stale task generation or descriptor");
        }
        return [
          "runnable",
          "running",
          "suspended",
          "succeeded",
          "failed",
          "cancelled"
        ].indexOf(this.#runtime.poll(handle >>> 0));
      },
      drop: (handle, descriptor) => {
        const entry = this.#handles.get(handle >>> 0);
        if (!entry || entry.descriptor !== descriptor >>> 0) {
          invalid("Stale task generation or descriptor");
        }
        return this.imports.cancel(handle);
      },
      drain: () => {
        const owner = this.#owner();
        return this.#restore(owner, this.#runtime.drain(owner.context.id));
      },
      checkpoint: () => {
        if (this.#cleaning === 0) this.#owner().context.checkpoint();
      },
      current: () => {
        const id = this.#owner().id;
        if (id === void 0) invalid("Task publication is incomplete");
        return id;
      },
      resume: (generation) => {
        const entry = this.#entries.get(generation >>> 0);
        if (!entry) invalid("Stale task ownership context");
        this.#activate(entry);
      },
      transfer: (token) => {
        if (!this.#destination || !this.#moves) {
          invalid("No checked task transfer");
        }
        token >>>= 0;
        if (this.#moves.includes(token)) return;
        this.#moves.push(token);
        try {
          this.#source.transfer(token >>> 0, this.#destination);
        } catch (cause) {
          this.#moves.pop();
          throw cause;
        }
      }
    };
    const signatures = [
      [
        "blot:task_context/create",
        1,
        true
      ],
      [
        "blot:task_context/save",
        1,
        true
      ],
      [
        "blot:task_context/restore",
        1,
        true
      ],
      [
        "blot:task_context/drop",
        1,
        true
      ],
      [
        "blot:task_retain",
        1,
        false
      ],
      [
        "blot:task_release",
        1,
        false
      ]
    ];
    for (let id = 0; id < descriptors; id++) {
      signatures.push([
        `blot:task_start/${id}`,
        1,
        true
      ]);
      signatures.push([
        `blot:task_move_captures/${id}`,
        1,
        false
      ]);
      signatures.push([
        `blot:task_move_result/${id}`,
        1,
        false
      ]);
    }
    this.#hooks = typedTaskFunctions(exports, signatures);
    this.#memory = exports["blot:memory"] instanceof WebAssembly.Memory ? exports["blot:memory"] : void 0;
  }
  #call(name, value) {
    return this.#hooks[name](value) >>> 0;
  }
  #activate(entry) {
    if (entry.closed) invalid("Task context has terminated");
    this.#call("blot:task_context/restore", entry.native);
    this.#active = entry;
    this.select(entry.resources);
  }
  begin(resources, asynchronous) {
    if (!this.#root) {
      const native = this.#call("blot:task_context/create", 0);
      this.#root = {
        id: 0,
        native,
        group: 0,
        context: this.#runtime.rootContext,
        resources,
        results: resources,
        descriptor: -1,
        captures: 0,
        entered: true,
        capturesLive: false,
        closed: false
      };
      this.#entries.set(0, this.#root);
      this.#call("blot:task_context/save", native);
    }
    this.#active = this.#root;
    this.select(resources);
    this.#asynchronous = asynchronous;
  }
  #owner() {
    return this.#active ?? invalid("No active task invocation");
  }
  #restore(owner, result) {
    if (result instanceof Promise) {
      return result.then((value) => {
        this.#activate(owner);
        return value;
      }, (cause) => {
        this.#activate(owner);
        throw cause;
      });
    }
    this.#activate(owner);
    return result;
  }
  #move(source, destination, adapter, value) {
    if (this.#destination) invalid("Reentrant task transfer");
    const prior = this.#owner();
    const moved = [];
    this.#source = source;
    this.#destination = destination;
    this.#moves = moved;
    this.select(source);
    try {
      this.#call(adapter, value);
    } catch (cause) {
      for (let index = moved.length - 1; index >= 0; index--) {
        destination.transfer(moved[index], source);
      }
      throw cause;
    } finally {
      this.#source = void 0;
      this.#destination = void 0;
      this.#moves = void 0;
      this.select(prior.resources);
    }
  }
  #retire(entry) {
    if (entry.closed) return;
    const prior = this.#owner();
    this.#activate(entry);
    let failed = false;
    let failure;
    const attempt = (action) => {
      try {
        action();
      } catch (cause) {
        if (!failed) {
          failed = true;
          failure = cause;
        }
      }
    };
    try {
      if (!entry.entered) {
        entry.entered = true;
        attempt(() => this.ownership.enter());
      }
      if (entry.capturesLive) {
        entry.capturesLive = false;
        attempt(() => {
          this.#call("blot:task_release", entry.captures);
        });
      }
      attempt(() => this.cleanup(() => entry.resources.finish()));
      attempt(() => this.ownership.leave(Number(failed)));
      entry.entered = false;
      attempt(() => {
        this.#call("blot:task_context/drop", entry.native);
      });
      entry.closed = true;
      if (entry.id !== void 0) this.#entries.delete(entry.id);
    } finally {
      if (!prior.closed) this.#activate(prior);
      else if (this.#root) this.#activate(this.#root);
    }
    if (failed) throw failure;
  }
  imports;
  cleanup(action) {
    this.#cleaning++;
    try {
      return action();
    } finally {
      this.#cleaning--;
    }
  }
  dispose() {
    if (!this.#root) return;
    this.#activate(this.#root);
    this.#call("blot:task_context/drop", this.#root.native);
    this.#root.closed = true;
    this.#entries.clear();
    this.#handles.clear();
  }
  finish(cancel) {
    const root = this.#root;
    const completion = this.#restore(root, this.#runtime.finishInvocation(cancel));
    if (completion instanceof Promise) {
      return completion.finally(() => this.#handles.clear());
    }
    this.#handles.clear();
    return completion;
  }
};

// ../blot2/compiler/guest.ts
var littleEndian = new Uint8Array(new Uint32Array([
  1
]).buffer)[0] === 1;
var scalarTypes = [
  "Unit",
  "U32",
  "Bool",
  "F32"
];
function isArrayType(type) {
  return type === "Array U32" || type === "Array F32";
}
var ownerBrand = Symbol("GuestOwner");
var dataTypeBrand = Symbol("CleanupDataType");
var dataTypeOwners = /* @__PURE__ */ new WeakMap();
var abiDataSchemas = /* @__PURE__ */ new WeakMap();
var capabilityBrand = Symbol("HostCapability");
var GuestError = class extends Error {
  code;
  constructor(code, message, options) {
    super(message, options), this.code = code;
    this.name = "GuestError";
  }
};
var AbiReader = class {
  bytes;
  owners;
  structured;
  #offset;
  tasks;
  constructor(bytes, owners = false, structured = false) {
    this.bytes = bytes;
    this.owners = owners;
    this.structured = structured;
    this.#offset = 0;
    this.tasks = false;
  }
  byte() {
    if (this.#offset === this.bytes.length) {
      throw new GuestError("invalid_abi", "Truncated blot:abi section");
    }
    return this.bytes[this.#offset++];
  }
  count() {
    let value = 0;
    for (let shift = 0; shift <= 28; shift += 7) {
      const byte = this.byte();
      if (shift === 28 && byte > 15) break;
      value += (byte & 127) * 2 ** shift;
      if (byte < 128) {
        if (shift !== 0 && byte === 0) break;
        return value;
      }
    }
    throw new GuestError("invalid_abi", "Non-canonical U32 in blot:abi section");
  }
  name() {
    const length = this.count();
    if (length > this.bytes.length - this.#offset) {
      throw new GuestError("invalid_abi", "Truncated export name in blot:abi");
    }
    const bytes = this.bytes.subarray(this.#offset, this.#offset + length);
    this.#offset += length;
    try {
      return new TextDecoder("utf-8", {
        fatal: true,
        ignoreBOM: true
      }).decode(bytes);
    } catch (cause) {
      throw new GuestError("invalid_abi", "Invalid UTF-8 export name", {
        cause
      });
    }
  }
  scalar(tag = this.byte()) {
    const type = scalarTypes[tag];
    if (type === void 0) {
      throw new GuestError("invalid_abi", `Unknown scalar ABI tag ${tag}`);
    }
    return type;
  }
  value(tag = this.byte()) {
    if (tag === 5) return "Array U32";
    if (tag === 6) return "Array F32";
    return this.scalar(tag);
  }
  publicValue(tag = this.byte()) {
    if (tag !== 8 && tag !== 9) return this.value(tag);
    if (!this.owners) {
      throw new GuestError("invalid_abi", "Owner type requires ABI 6");
    }
    const schema = this.count();
    if (schema === 0) {
      throw new GuestError("invalid_abi", "Invalid owner schema");
    }
    const fresh = this.byte();
    let usage = "unique";
    if (tag === 9) usage = "shared";
    if (fresh === 255) return Object.freeze({
      kind: "owner",
      usage,
      schema
    });
    return Object.freeze({
      kind: "owner",
      usage,
      schema,
      fresh: this.value(fresh)
    });
  }
  parameter() {
    const tag = this.byte();
    if ((tag === 11 || tag === 12 && this.owners) && this.structured) {
      const schema = this.count();
      if (schema === 0) {
        throw new GuestError("invalid_abi", "Empty cleanup data schema");
      }
      if (tag === 12) {
        return Object.freeze({
          kind: "cleanup",
          parameter: "U32",
          result: "Unit",
          lifetime: "owned",
          dataSchema: schema
        });
      }
      return Object.freeze({
        kind: "cleanup",
        parameter: "U32",
        result: "Unit",
        dataSchema: schema
      });
    }
    if (tag === 4 || tag === 7 || tag === 10 && this.owners || tag === 13 && this.tasks) {
      const parameter = this.value();
      const result = this.value();
      if ((tag === 7 || tag === 10) && result !== "Unit") {
        throw new GuestError("invalid_abi", "Cleanup callback must return Unit");
      }
      let kind = "callback";
      if (tag === 7 || tag === 10) kind = "cleanup";
      if (tag === 10) {
        return Object.freeze({
          kind,
          parameter,
          result,
          lifetime: "owned"
        });
      }
      if (tag === 13) {
        return Object.freeze({
          kind,
          parameter,
          result,
          concurrent: true
        });
      }
      return Object.freeze({
        kind,
        parameter,
        result
      });
    }
    return this.publicValue(tag);
  }
  end() {
    if (this.#offset !== this.bytes.length) {
      throw new GuestError("invalid_abi", "Trailing bytes in blot:abi section");
    }
  }
};
function importModule(signature2) {
  if (signature2.dataSchema !== void 0) {
    if (signature2.lifetime === "owned") return "blot:cleanup_data/2";
    return "blot:cleanup_data/1";
  }
  if (signature2.lifetime === "owned") return "blot:cleanup/2";
  if (signature2.kind === "cleanup") return "blot:cleanup/1";
  return "blot:host/1";
}
function importKey(signature2) {
  return `${importModule(signature2)}/${importName(signature2)}`;
}
function importName(signature2) {
  if (signature2.dataSchema !== void 0) return "call_data_unit";
  return `call_${signature2.parameter.toLowerCase().replaceAll(" ", "_")}_${signature2.result.toLowerCase().replaceAll(" ", "_")}`;
}
function readGuestAbi(module) {
  const sections = WebAssembly.Module.customSections(module, "blot:abi");
  if (sections.length !== 1) {
    throw new GuestError("invalid_abi", "Expected exactly one blot:abi section");
  }
  const reader = new AbiReader(new Uint8Array(sections[0]));
  const version = reader.count();
  if (version !== 2 && version !== 3 && version !== 4 && version !== 5 && version !== 6 && version !== 7) {
    throw new GuestError("invalid_abi", `Unsupported guest ABI version ${version}`);
  }
  reader.owners = version === 6;
  reader.structured = version === 7;
  let publicOwners = reader.owners;
  let dataSchemas;
  try {
    dataSchemas = readDataSchemas(module);
  } catch (cause) {
    throw new GuestError("invalid_abi", "Invalid cleanup data schemas", {
      cause
    });
  }
  if (version === 7 !== (dataSchemas !== void 0)) {
    throw new GuestError("invalid_abi", "Cleanup data feature differs from ABI version");
  }
  const functions = [];
  const constants = [];
  const exports = /* @__PURE__ */ new Map();
  const signatures = /* @__PURE__ */ new Set();
  const declare = (name, kind) => {
    if (exports.has(name)) {
      throw new GuestError("invalid_abi", `Duplicate ABI export ${name}`);
    }
    exports.set(name, kind);
  };
  if (version >= 3) {
    declare("blot:ownership_enter", "function");
    declare("blot:ownership_leave", "function");
    declare("blot:rc_roots", "global");
  }
  const taskSections = WebAssembly.Module.customSections(module, "blot:tasks");
  let taskCount;
  if (taskSections.length > 1) {
    throw new GuestError("invalid_abi", "Duplicate task feature section");
  }
  if (taskSections.length === 1) {
    const feature = new AbiReader(new Uint8Array(taskSections[0]));
    if (version < 3 || feature.byte() !== 2) {
      throw new GuestError("invalid_abi", "Unsupported task ownership contract");
    }
    reader.tasks = true;
    taskCount = feature.count();
    if (taskCount > 4096) {
      throw new GuestError("invalid_abi", "Invalid task descriptor count");
    }
    feature.end();
    for (const name of taskExports(taskCount)) declare(name, "function");
  }
  let resources = 0;
  if (version >= 4) {
    resources = reader.count();
    if (resources === 0 && version < 6 || resources > 4096) {
      throw new GuestError("invalid_abi", "Invalid resource adapter count");
    }
  }
  let transfer = version === 5;
  if (version >= 6) {
    const flags = reader.byte();
    let limit = 1;
    if (version === 7) limit = 3;
    if (flags > limit || resources === 0 && (flags & 1) !== 0) {
      throw new GuestError("invalid_abi", "Invalid owner ABI flags");
    }
    transfer = (flags & 1) !== 0;
    if (version === 7) publicOwners = (flags & 2) !== 0;
    reader.owners = publicOwners;
    if (publicOwners) {
      declare("blot:owner_release", "function");
      declare("blot:owner_token", "function");
    }
  }
  const functionCount = reader.count();
  for (let index = 0; index < functionCount; index++) {
    const name = reader.name();
    declare(name, "function");
    const parameter = reader.parameter();
    if (typeof parameter !== "string" && parameter.kind !== "owner") {
      signatures.add(importKey(parameter));
      if (parameter.lifetime === "owned") {
        let binder = "blot:cleanup_bind/1";
        if (parameter.dataSchema !== void 0) {
          binder = "blot:cleanup_data_bind/1";
        }
        signatures.add(`${binder}/${importName(parameter)}`);
      }
    }
    const result2 = reader.publicValue();
    if (typeof result2 !== "string" && result2.fresh !== void 0) {
      throw new GuestError("invalid_abi", "Owner results cannot declare fresh arguments");
    }
    if (typeof parameter !== "string" && parameter.kind === "owner" && parameter.fresh !== void 0) declare(`blot:owner_call/${name}`, "function");
    functions.push(Object.freeze({
      name,
      parameter,
      result: result2
    }));
  }
  const constantCount = reader.count();
  for (let index = 0; index < constantCount; index++) {
    const name = reader.name();
    declare(name, "global");
    constants.push(Object.freeze({
      name,
      type: reader.scalar()
    }));
  }
  reader.end();
  try {
    validateDataRoots(dataSchemas, functions.flatMap((fn) => {
      if (typeof fn.parameter === "string" || fn.parameter.kind === "owner" || fn.parameter.dataSchema === void 0) return [];
      return [
        fn.parameter.dataSchema
      ];
    }));
  } catch (cause) {
    throw new GuestError("invalid_abi", "Invalid cleanup data schema roots", {
      cause
    });
  }
  const multiplicity = WebAssembly.Module.customSections(module, "blot:multiplicity");
  if (multiplicity.length > 1) {
    throw new GuestError("invalid_abi", "Duplicate multiplicity section");
  }
  if (multiplicity.length === 1) {
    const usage = new AbiReader(new Uint8Array(multiplicity[0]));
    if (usage.count() !== 1) {
      throw new GuestError("invalid_abi", "Unsupported multiplicity version");
    }
    const count = usage.count();
    if (count === 0 || count > functions.length) {
      throw new GuestError("invalid_abi", "Invalid multiplicity count");
    }
    const seen = /* @__PURE__ */ new Set();
    const positions = new Map(functions.map((fn, index) => [
      fn.name,
      index
    ]));
    for (let index = 0; index < count; index++) {
      const name = usage.name();
      const position = positions.get(name);
      if (position === void 0 || seen.has(name) || usage.byte() !== 1) {
        throw new GuestError("invalid_abi", "Invalid once function entry");
      }
      seen.add(name);
      functions[position] = Object.freeze({
        ...functions[position],
        multiplicity: "once"
      });
    }
    usage.end();
  }
  if (version === 5 || version >= 6 && resources !== 0 || functions.some((fn) => typeof fn.parameter === "string" && isArrayType(fn.parameter) || typeof fn.result === "string" && isArrayType(fn.result) || typeof fn.parameter !== "string" && fn.parameter.kind === "owner" && fn.parameter.fresh !== void 0 && isArrayType(fn.parameter.fresh) || typeof fn.parameter !== "string" && fn.parameter.kind !== "owner" && (fn.parameter.dataSchema !== void 0 || isArrayType(fn.parameter.parameter) || isArrayType(fn.parameter.result)))) {
    declare("blot:memory", "memory");
    declare("blot:allocate", "function");
    declare("blot:reset", "function");
  }
  if (version >= 3 && (signatures.size !== 0 || taskCount !== void 0)) {
    declare("blot:rc_host_refs", "global");
  }
  if (resources !== 0) {
    for (let index = 0; index < resources; index++) {
      declare(resourceExport(index, false), "function");
      declare(resourceExport(index, true), "function");
    }
  }
  if (taskCount !== void 0 && !exports.has("blot:memory")) {
    declare("blot:memory", "memory");
  }
  const expectedTaskImports = /* @__PURE__ */ new Set();
  if (taskCount !== void 0) {
    for (const [name] of taskImports) expectedTaskImports.add(name);
  }
  const resourceImports = /* @__PURE__ */ new Set();
  if (resources !== 0) {
    resourceImports.add("reserve");
    resourceImports.add("register");
    resourceImports.add("release");
  }
  const transferImports = /* @__PURE__ */ new Set();
  if (transfer) {
    transferImports.add("register_owned");
    transferImports.add("retire");
    transferImports.add("drain");
  }
  const actualExports = WebAssembly.Module.exports(module);
  const ownership = WebAssembly.Module.customSections(module, "blot:ownership");
  if (ownership.length > 1) {
    throw new GuestError("invalid_abi", "Duplicate ownership section");
  }
  if (ownership.length === 1) {
    const state = new Uint8Array(ownership[0]);
    if (state.length !== 3 || state[0] !== 1 && state[0] !== 2 || state[1] > state[0] || state[2] > 7 || state[1] !== 0 !== (state[2] === 0) || version >= 3 !== (state[0] === 2 && state[1] === 2)) throw new GuestError("invalid_abi", "Invalid ownership section");
    if (state[2] !== 1) {
      for (const counter of [
        "leases",
        "retains",
        "releases",
        "destroyed"
      ]) {
        declare(`blot:rc_${counter}`, "global");
      }
    }
  }
  const cycles = WebAssembly.Module.customSections(module, "blot:cycles");
  if (cycles.length > 1) {
    throw new GuestError("invalid_abi", "Duplicate cycle section");
  }
  if (cycles.length === 1) {
    const state = new Uint8Array(cycles[0]);
    if (state.length !== 2 || state[0] !== 1 || state[1] > 1 || ownership.length !== 1 || state[1] !== Number(new Uint8Array(ownership[0])[1] !== 0)) {
      throw new GuestError("invalid_abi", "Invalid cycle section");
    }
    for (const counter of [
      "passes",
      "vertices",
      "edges",
      "bytes"
    ]) {
      declare(`blot:cycle_${counter}`, "global");
    }
  }
  if (version >= 3 && ownership.length !== 1) {
    throw new GuestError("invalid_abi", "Missing root ownership section");
  }
  if (actualExports.length !== exports.size || actualExports.some((entry) => exports.get(entry.name) !== entry.kind)) {
    throw new GuestError("invalid_abi", "Wasm exports differ from blot:abi");
  }
  for (const entry of WebAssembly.Module.imports(module)) {
    if (entry.module === "blot:task/1" && entry.kind === "function" && expectedTaskImports.delete(entry.name)) continue;
    if (entry.module === "blot:resource/1" && entry.kind === "function" && resourceImports.delete(entry.name)) continue;
    if (entry.module === "blot:resource/2" && entry.kind === "function" && transferImports.delete(entry.name)) continue;
    if (entry.module !== "blot:host/1" && entry.module !== "blot:cleanup/1" && entry.module !== "blot:cleanup/2" && entry.module !== "blot:cleanup_bind/1" && entry.module !== "blot:cleanup_data/1" && entry.module !== "blot:cleanup_data/2" && entry.module !== "blot:cleanup_data_bind/1" || entry.kind !== "function" || !signatures.delete(`${entry.module}/${entry.name}`)) {
      throw new GuestError("invalid_abi", `Unexpected Wasm import ${entry.module}/${entry.name}`);
    }
  }
  expectedTaskImports.delete("collect");
  if (expectedTaskImports.size !== 0) {
    throw new GuestError("invalid_abi", "Missing task protocol import");
  }
  if (resourceImports.size !== 0 || transferImports.size !== 0) {
    throw new GuestError("invalid_abi", "Missing resource journal import");
  }
  if (signatures.size !== 0) {
    throw new GuestError("invalid_abi", "Missing generic callback import");
  }
  const result = {
    version,
    functions: Object.freeze(functions),
    constants: Object.freeze(constants)
  };
  if (taskCount !== void 0) {
    Object.assign(result, {
      taskDescriptorCount: taskCount
    });
  }
  if (resources !== 0) {
    Object.assign(result, {
      resourceAdapterCount: resources
    });
  }
  if (version >= 6) {
    Object.assign(result, {
      resourceTransfer: transfer,
      publicOwners
    });
  }
  if (dataSchemas) abiDataSchemas.set(result, dataSchemas);
  return Object.freeze(result);
}
function scalarToWasm(type, value, code, label) {
  switch (type) {
    case "Unit":
      if (value === null) return 0;
      break;
    case "Bool":
      if (typeof value === "boolean") return value ? 1 : 0;
      break;
    case "U32":
      if (typeof value === "number" && Number.isInteger(value) && value >= 0 && value <= 4294967295) return value;
      break;
    case "F32":
      if (typeof value === "number") return Math.fround(value);
      break;
  }
  throw new GuestError(code, `${label} must be ${type}`);
}
function scalarFromWasm(type, value) {
  if (typeof value !== "number") {
    throw new GuestError("invalid_abi", `Wasm ${type} value is not a number`);
  }
  switch (type) {
    case "Unit":
      if (value === 0) return null;
      break;
    case "Bool":
      if (value === 0 || value === 1) return value === 1;
      break;
    case "U32":
      if (Number.isInteger(value) && value >= -2147483648 && value <= 2147483647) {
        return value >>> 0;
      }
      break;
    case "F32":
      return Math.fround(value);
  }
  throw new GuestError("invalid_abi", `Non-canonical Wasm ${type} value`);
}
function arrayRange(arena, pointer, length) {
  if (typeof pointer !== "number" || !Number.isInteger(pointer) || pointer < -2147483648 || pointer > 4294967295) {
    throw new GuestError("invalid_abi", "Array lies outside the guest arena");
  }
  const address = pointer >>> 0;
  if (address < 4 || address % 4 !== 0 || address > arena.memory.buffer.byteLength - 4 || length > (arena.memory.buffer.byteLength - address - 4) / 4) {
    throw new GuestError("invalid_abi", "Array lies outside the guest arena");
  }
  return address;
}
function arrayToWasm(arena, type, argument, name, code = "invalid_argument") {
  if (!(type === "Array U32" ? argument instanceof Uint32Array : argument instanceof Float32Array)) {
    throw new GuestError(code, `${name} argument must be ${type === "Array U32" ? "Uint32Array" : "Float32Array"}`);
  }
  const values = argument;
  const length = values.length;
  const byteLength = length * 4;
  if (length >= 1073741823) {
    throw new GuestError(code, `${name} array byte count exceeds the Wasm32 address range`);
  }
  const pointer = arrayRange(arena, arena.allocate((length + 1) * 4), length);
  const view = new DataView(arena.memory.buffer);
  view.setUint32(pointer, length, true);
  if (littleEndian && length > 32 && values.buffer instanceof ArrayBuffer) {
    new Uint8Array(arena.memory.buffer, pointer + 4, byteLength).set(new Uint8Array(values.buffer, values.byteOffset, byteLength));
  } else {
    const input = type === "Array U32" ? values : new Uint32Array(values.buffer, values.byteOffset, length);
    for (let index = 0; index < length; index++) {
      view.setUint32(pointer + 4 + index * 4, input[index], true);
    }
  }
  return pointer;
}
function arrayFromWasm(arena, type, result) {
  const pointer = arrayRange(arena, result, 0);
  const view = new DataView(arena.memory.buffer);
  const length = view.getUint32(pointer, true);
  arrayRange(arena, pointer, length);
  const values = type === "Array U32" ? new Uint32Array(length) : new Float32Array(length);
  if (littleEndian && length > 32) {
    new Uint8Array(values.buffer).set(new Uint8Array(arena.memory.buffer, pointer + 4, values.byteLength));
  } else {
    const output = type === "Array U32" ? values : new Uint32Array(values.buffer);
    for (let index = 0; index < length; index++) {
      output[index] = view.getUint32(pointer + 4 + index * 4, true);
    }
  }
  return values;
}
var capabilityOwners = /* @__PURE__ */ new WeakMap();
var publicOwnerStates = /* @__PURE__ */ new WeakMap();
function resourceExport(index, drop) {
  if (drop) return `blot:resource_drop/${index}`;
  return `blot:resource_cleanup/${index}`;
}
function resourceU32(value) {
  const bytes = [];
  do {
    let byte = value & 127;
    value >>>= 7;
    if (value !== 0) byte |= 128;
    bytes.push(byte);
  } while (value !== 0);
  return bytes;
}
function resourceName(value) {
  const bytes = [
    ...new TextEncoder().encode(value)
  ];
  return [
    ...resourceU32(bytes.length),
    ...bytes
  ];
}
function resourceModule(sections) {
  return new WebAssembly.Module(new Uint8Array([
    0,
    97,
    115,
    109,
    1,
    0,
    0,
    0,
    ...sections.flatMap((section, index) => {
      if (section.length === 0) return [];
      return [
        index + 1,
        ...resourceU32(section.length),
        ...section
      ];
    })
  ]));
}
function typedCleanupImport(signature2, callback) {
  let parameter = 127;
  if (signature2.parameter === "F32") parameter = 125;
  let handle = 111;
  if (signature2.lifetime === "owned") handle = 127;
  let parameters = [
    2,
    handle,
    parameter
  ];
  if (signature2.dataSchema !== void 0) parameters = [
    3,
    handle,
    127,
    127
  ];
  const module = resourceModule([
    [
      1,
      96,
      ...parameters,
      1,
      127
    ],
    [
      1,
      ...resourceName("h"),
      ...resourceName("c"),
      0,
      0
    ],
    [],
    [],
    [],
    [],
    [
      1,
      ...resourceName("c"),
      0,
      0
    ]
  ]);
  return new WebAssembly.Instance(module, {
    h: {
      c: callback
    }
  }).exports.c;
}
function typedCleanupBind(callback, data = false) {
  let parameters = [
    1,
    111
  ];
  if (data) parameters = [
    2,
    111,
    127
  ];
  const module = resourceModule([
    [
      1,
      96,
      ...parameters,
      1,
      127
    ],
    [
      1,
      ...resourceName("h"),
      ...resourceName("c"),
      0,
      0
    ],
    [],
    [],
    [],
    [],
    [
      1,
      ...resourceName("c"),
      0,
      0
    ]
  ]);
  return new WebAssembly.Instance(module, {
    h: {
      c: callback
    }
  }).exports.c;
}
function typedResourceImports(imports, transfers = false) {
  let names = [
    "reserve",
    "register",
    "release"
  ];
  let parameters = [
    0,
    5,
    1
  ];
  if (transfers) {
    names = [
      "register_owned",
      "retire",
      "drain"
    ];
    parameters = [
      6,
      1,
      0
    ];
  }
  const types = [
    names.length
  ];
  for (let index = 0; index < names.length; index++) {
    types.push(96, parameters[index]);
    for (let count = 0; count < parameters[index]; count++) types.push(127);
    if (!transfers && index === 0) types.push(1, 127);
    else types.push(0);
  }
  const linked = [
    names.length
  ];
  const exports = [
    names.length
  ];
  for (const [index, name] of names.entries()) {
    linked.push(...resourceName("j"), ...resourceName(name), 0, index);
    exports.push(...resourceName(name), 0, index);
  }
  const module = resourceModule([
    types,
    linked,
    [],
    [],
    [],
    [],
    exports
  ]);
  return new WebAssembly.Instance(module, {
    j: imports
  }).exports;
}
function resourceAdapters(instance, count) {
  const adapters = [];
  const imported = {};
  const imports = resourceU32(count * 2);
  for (let index = 0; index < count; index++) {
    const cleanup = instance.exports[resourceExport(index, false)];
    const drop = instance.exports[resourceExport(index, true)];
    if (typeof cleanup !== "function" || typeof drop !== "function") {
      throw new GuestError("invalid_abi", "Missing resource adapter");
    }
    for (const [suffix, fn] of [
      [
        "c",
        cleanup
      ],
      [
        "d",
        drop
      ]
    ]) {
      const name = `${suffix}${index}`;
      imported[name] = fn;
      imports.push(...resourceName("o"), ...resourceName(name), 0, 0);
    }
    adapters.push({
      cleanup,
      drop
    });
  }
  try {
    new WebAssembly.Instance(resourceModule([
      [
        1,
        96,
        3,
        127,
        127,
        127,
        0
      ],
      imports
    ]), {
      o: imported
    });
  } catch (cause) {
    throw new GuestError("invalid_abi", "Invalid resource adapter signature", {
      cause
    });
  }
  return adapters;
}
async function instantiateGuest(bytes, options = {}) {
  const module = bytes instanceof WebAssembly.Module ? bytes : await WebAssembly.compile(bytes);
  const abi = readGuestAbi(module);
  const dataSchemas = abiDataSchemas.get(abi);
  const functions = new Map(abi.functions.map((fn) => [
    fn.name,
    fn
  ]));
  const constants = new Map(abi.constants.map((constant) => [
    constant.name,
    constant
  ]));
  const jspi = WebAssembly;
  function requireAsync() {
    if (!jspi.Suspending || !jspi.promising) {
      throw new GuestError("unsupported_async", "Async guests require WebAssembly.Suspending and WebAssembly.promising");
    }
    return jspi;
  }
  if (options.asynchronous) requireAsync();
  const lifetime = {
    disposed: false
  };
  let callbacks = /* @__PURE__ */ new WeakMap();
  let active;
  let epoch = 0;
  let nextBinding = 1;
  let bindings;
  let running = false;
  function idle() {
    if (lifetime.disposed) {
      throw new GuestError("disposed_guest", "Guest instance is disposed");
    }
    if (running) {
      throw new GuestError("reentrant_call", "Cannot reenter an active guest instance");
    }
  }
  let arena;
  function fromWasm(type, value) {
    if (!isArrayType(type)) return scalarFromWasm(type, value);
    if (!arena) {
      throw new GuestError("invalid_abi", "Array callback requires an arena");
    }
    return arrayFromWasm(arena, type, value);
  }
  function hostResult(type, value, name) {
    const label = `Host callback result during ${name}`;
    if (!isArrayType(type)) {
      return scalarToWasm(type, value, "host_result", label);
    }
    if (!arena) {
      throw new GuestError("invalid_abi", "Array callback requires an arena");
    }
    return arrayToWasm(arena, type, value, label, "host_result");
  }
  async function asynchronousResult(type, result, name, taskContext) {
    let value;
    try {
      value = await result;
    } catch (cause) {
      if (taskContext !== void 0) tasks.imports.resume(taskContext);
      throw new GuestError("host_exception", `Host callback rejected during ${name}`, {
        cause
      });
    }
    if (taskContext !== void 0) tasks.imports.resume(taskContext);
    return hostResult(type, value, name);
  }
  const imports = {};
  const cleanupImports = {};
  const ownedCleanupImports = {};
  const cleanupBindImports = {};
  const dataCleanupImports = {};
  const ownedDataCleanupImports = {};
  const dataBindImports = {};
  for (const fn of abi.functions) {
    if (typeof fn.parameter === "string" || fn.parameter.kind === "owner") {
      continue;
    }
    const signature2 = fn.parameter;
    if (signature2.lifetime === "owned") bindings ??= /* @__PURE__ */ new Map();
    const invokeImport = (reference, value, data) => {
      let invocation = active;
      if (signature2.lifetime === "owned") {
        const binding = bindings.get(Number(reference) >>> 0);
        invocation = binding;
        if (!running || !binding) invocation = void 0;
        if (binding && binding.epoch !== epoch) {
          binding.epoch = epoch;
          binding.calls = 0;
        }
      }
      if (!invocation || signature2.lifetime !== "owned" && reference !== invocation.reference) {
        throw new GuestError("stale_capability", "Host call has no active capability reference");
      }
      if (signature2.dataSchema !== void 0 && value >>> 0 !== invocation.signature.dataSchema || signature2.dataSchema === void 0 && invocation.signature.dataSchema !== void 0) {
        throw new GuestError("capability_signature", "Cleanup data call differs from its exact schema");
      }
      if (signature2.kind !== invocation.signature.kind || signature2.parameter !== invocation.signature.parameter || signature2.result !== invocation.signature.result) {
        throw new GuestError("capability_signature", "Host call signature does not match its capability");
      }
      if (signature2.kind === "cleanup") {
        if (invocation.maxCalls === void 0 || invocation.calls >= invocation.maxCalls) {
          throw new GuestError("host_exception", "Cleanup invocation limit exceeded");
        }
        invocation.calls++;
      }
      let argument;
      if (signature2.dataSchema !== void 0) {
        if (!arena || !dataSchemas || data === void 0) {
          throw new GuestError("invalid_abi", "Missing cleanup data memory");
        }
        try {
          argument = copyData(arena.memory, dataSchemas, value >>> 0, data >>> 0);
        } catch (cause) {
          throw new GuestError("invalid_abi", "Invalid cleanup data value", {
            cause
          });
        }
      } else argument = fromWasm(signature2.parameter, value);
      const taskContext = tasks?.imports.current();
      let result;
      try {
        result = invocation.invoke(argument);
      } catch (cause) {
        throw new GuestError("host_exception", `Host callback threw during ${invocation.exportName}`, {
          cause
        });
      }
      if ((typeof result === "object" && result !== null || typeof result === "function") && "then" in result) {
        if (invocation.asynchronous) {
          return asynchronousResult(signature2.result, result, invocation.exportName, taskContext);
        }
        let message = `Host callback returned a Promise/thenable during ${invocation.exportName}; use capabilityAsync() and callAsync()`;
        if (signature2.kind === "cleanup") {
          message = `Cleanup callback must complete synchronously during ${invocation.exportName}`;
        }
        const failure = new GuestError("async_host_call", message);
        try {
          Promise.prototype.then.call(result, void 0, () => {
          });
        } catch (cause) {
          failure.cause = cause;
        }
        throw failure;
      }
      return hostResult(signature2.result, result, invocation.exportName);
    };
    if (signature2.lifetime === "owned") {
      let invocationImports = ownedCleanupImports;
      let bindingImports = cleanupBindImports;
      if (signature2.dataSchema !== void 0) {
        invocationImports = ownedDataCleanupImports;
        bindingImports = dataBindImports;
      }
      invocationImports[importName(signature2)] = typedCleanupImport(signature2, invokeImport);
      bindingImports[importName(signature2)] = typedCleanupBind((reference, schema) => {
        const invocation = active;
        if (!running || !invocation || reference !== invocation.reference || invocation.signature.kind !== "cleanup" || signature2.dataSchema !== void 0 && invocation.signature.dataSchema !== schema >>> 0 || signature2.dataSchema === void 0 && invocation.signature.dataSchema !== void 0 || invocation.signature.parameter !== signature2.parameter || invocation.signature.result !== "Unit" || invocation.asynchronous || invocation.maxCalls === void 0) {
          throw new GuestError("stale_capability", "Owned cleanup binding requires an active synchronous cleanup capability");
        }
        if (bindings.size === 4096 || nextBinding > 4294967295 || invocation.binding !== void 0) {
          throw new GuestError("host_exception", "Cleanup binding limit");
        }
        const binding = nextBinding++;
        bindings.set(binding, {
          ...invocation,
          maxCalls: invocation.maxCalls,
          refs: 0,
          epoch
        });
        invocation.binding = binding;
        return binding;
      }, signature2.dataSchema !== void 0);
    } else if (signature2.kind === "cleanup") {
      let invocationImports = cleanupImports;
      if (signature2.dataSchema !== void 0) {
        invocationImports = dataCleanupImports;
      }
      invocationImports[importName(signature2)] = typedCleanupImport(signature2, invokeImport);
    } else if (options.asynchronous) {
      imports[importName(signature2)] = new (requireAsync()).Suspending(invokeImport);
    } else {
      imports[importName(signature2)] = invokeImport;
    }
  }
  let resources;
  let rootResources;
  let makeTaskJournal;
  let tasks;
  let initializing = true;
  const modules = {
    "blot:host/1": imports,
    "blot:cleanup/1": cleanupImports,
    "blot:cleanup/2": ownedCleanupImports,
    "blot:cleanup_bind/1": cleanupBindImports,
    "blot:cleanup_data/1": dataCleanupImports,
    "blot:cleanup_data/2": ownedDataCleanupImports,
    "blot:cleanup_data_bind/1": dataBindImports
  };
  if (abi.taskDescriptorCount !== void 0) {
    const callbacks2 = {};
    for (const [name] of taskImports) {
      const invoke = (...args) => {
        if (!tasks) {
          throw new GuestError("invalid_abi", "Task runtime is not active");
        }
        return tasks.imports[name](...args);
      };
      if (options.asynchronous && [
        "close",
        "join",
        "collect",
        "cancel",
        "drop",
        "drain"
      ].includes(name)) callbacks2[name] = new (requireAsync()).Suspending(invoke);
      else callbacks2[name] = invoke;
    }
    modules["blot:task/1"] = typedTaskFunctions(callbacks2);
  }
  const count = abi.resourceAdapterCount ?? 0;
  if (count !== 0) {
    modules["blot:resource/1"] = typedResourceImports({
      reserve: () => {
        if (!resources) {
          throw new GuestError("invalid_abi", "Resource journal is not active");
        }
        return resources.reserve();
      },
      register: (token, adapter, payload, callable, provider) => {
        if (!resources) {
          throw new GuestError("invalid_abi", "Resource journal is not active");
        }
        resources.register(token >>> 0, adapter >>> 0, payload >>> 0, callable >>> 0, provider >>> 0);
      },
      release: (token) => {
        if (!resources) {
          throw new GuestError("invalid_abi", "Resource journal is not active");
        }
        resources.release(token >>> 0);
      }
    });
  }
  if (abi.version === 5 || abi.version >= 6 && abi.resourceTransfer) {
    modules["blot:resource/2"] = typedResourceImports({
      register_owned: (token, adapter, payload, callable, provider, owner) => {
        if (!resources) {
          throw new GuestError("invalid_abi", "Resource journal is not active");
        }
        token >>>= 0;
        owner >>>= 0;
        adapter >>>= 0;
        payload >>>= 0;
        callable >>>= 0;
        provider >>>= 0;
        let words;
        const memory2 = instance.exports["blot:memory"];
        if (!(memory2 instanceof WebAssembly.Memory) || owner < 32 || owner % 4 !== 0 || owner > memory2.buffer.byteLength - 8) throw new GuestError("invalid_abi", "Invalid resource owner address");
        try {
          words = new Uint32Array(memory2.buffer);
        } catch (cause) {
          resources.register(token, adapter, payload, callable, provider);
          try {
            resources.release(token);
          } catch {
          }
          throw cause;
        }
        const slot = owner / 4;
        let layout = 0;
        if (adapter === 2) layout = 1;
        if (adapter > 2 || words[slot - 7] !== 8 || (words[slot - 6] & 3092) !== 4 || words[slot - 4] === 0 || words[slot - 3] !== layout || words[slot] !== payload || words[slot + 1] !== 0) {
          throw new GuestError("invalid_abi", "Resource owner identity changed");
        }
        resources.registerOwned(token, adapter, payload, callable, provider, owner);
        words[slot + 1] = token;
        words[slot - 6] |= 2048;
      },
      retire: (token) => {
        if (!resources) {
          throw new GuestError("invalid_abi", "Resource journal is not active");
        }
        resources.retire(token >>> 0);
      },
      drain: () => {
        if (!resources && initializing) return;
        if (!resources) {
          throw new GuestError("invalid_abi", "Resource journal is not active");
        }
        resources.drain();
      }
    }, true);
  }
  let instance;
  try {
    instance = await WebAssembly.instantiate(module, modules);
    initializing = false;
  } catch (cause) {
    if (count !== 0 || Object.keys(cleanupImports).length !== 0 || Object.keys(ownedCleanupImports).length !== 0 || Object.keys(dataCleanupImports).length !== 0 || Object.keys(ownedDataCleanupImports).length !== 0) {
      throw new GuestError("invalid_abi", "Invalid checked guest import signature", {
        cause
      });
    }
    throw cause;
  }
  if (count !== 0 || abi.taskDescriptorCount !== void 0) {
    let invalidate;
    if (abi.version === 5 || abi.version >= 6 && abi.resourceTransfer) {
      const memory2 = instance.exports["blot:memory"];
      if (!(memory2 instanceof WebAssembly.Memory)) {
        throw new GuestError("invalid_abi", "Resource transfer requires owned memory");
      }
      invalidate = (owner, token) => {
        if (owner < 32 || owner % 4 !== 0 || owner > memory2.buffer.byteLength - 8) throw new GuestError("invalid_abi", "Invalid resource owner address");
        const words = new Uint32Array(memory2.buffer);
        const slot = owner / 4;
        if (words[slot - 7] !== 8 || (words[slot - 6] & 3092) !== 2052 || words[slot - 4] === 0 || words[slot + 1] !== token) {
          throw new GuestError("invalid_abi", "Resource owner identity changed");
        }
        words[slot + 1] = 0;
      };
    }
    let resourceBindings;
    if (bindings) {
      resourceBindings = {
        capture: () => active?.binding ?? 0,
        retain: (id) => {
          const binding = bindings.get(id);
          if (!binding) {
            throw new GuestError("stale_capability", "Missing cleanup binding");
          }
          binding.refs++;
        },
        release: (id) => {
          const binding = bindings.get(id);
          if (!binding || binding.refs === 0) {
            throw new GuestError("stale_capability", "Cleanup binding is not owned");
          }
          binding.refs--;
        }
      };
    }
    const adapters = resourceAdapters(instance, count).map((adapter) => ({
      drop: adapter.drop,
      cleanup: (payload, callable, provider) => {
        const invoke = () => adapter.cleanup(payload, callable, provider);
        if (tasks) tasks.cleanup(invoke);
        else invoke();
      }
    }));
    const tokens = new ResourceTokens();
    makeTaskJournal = () => new ResourceJournal(adapters, invalidate, resourceBindings, tokens);
    rootResources = makeTaskJournal();
    resources = rootResources;
  }
  let ownership;
  if (abi.version >= 3) {
    const enter = instance.exports["blot:ownership_enter"];
    const leave = instance.exports["blot:ownership_leave"];
    if (typeof enter !== "function" || typeof leave !== "function") {
      throw new GuestError("invalid_abi", "Missing ownership exports");
    }
    const validator = new WebAssembly.Module(new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      8,
      2,
      96,
      0,
      0,
      96,
      1,
      127,
      0,
      2,
      13,
      2,
      1,
      111,
      1,
      101,
      0,
      0,
      1,
      111,
      1,
      108,
      0,
      1
    ]));
    try {
      new WebAssembly.Instance(validator, {
        o: {
          e: enter,
          l: leave
        }
      });
    } catch (cause) {
      throw new GuestError("invalid_abi", "Invalid ownership export signatures", {
        cause
      });
    }
    ownership = {
      enter,
      leave
    };
  }
  if (abi.taskDescriptorCount !== void 0) {
    try {
      tasks = new TaskGuest(instance.exports, abi.taskDescriptorCount, makeTaskJournal, (journal) => {
        resources = journal;
      }, ownership, options.asynchronous ? requireAsync().promising : void 0);
    } catch (cause) {
      throw new GuestError("invalid_abi", "Invalid task adapter signatures", {
        cause
      });
    }
  }
  let entered = false;
  const memory = instance.exports["blot:memory"];
  const allocate = instance.exports["blot:allocate"];
  const reset = instance.exports["blot:reset"];
  if (memory !== void 0 && (allocate !== void 0 || reset !== void 0 || abi.taskDescriptorCount === void 0)) {
    if (!(memory instanceof WebAssembly.Memory) || typeof allocate !== "function" || typeof reset !== "function") {
      throw new GuestError("invalid_abi", "Missing array arena exports");
    }
    arena = {
      memory,
      allocate,
      reset
    };
  }
  const valueTypes = [
    ...scalarTypes,
    "Array U32",
    "Array F32"
  ];
  function createCapability(definition, asynchronous, cleanup) {
    idle();
    if (typeof definition !== "object" || definition === null) {
      throw new GuestError("invalid_capability", "Expected a callback definition");
    }
    const { parameter, result, call, concurrent } = definition;
    if (!valueTypes.includes(parameter) || !valueTypes.includes(result) || typeof call !== "function" || concurrent !== void 0 && concurrent !== true) {
      throw new GuestError("invalid_capability", "A capability needs supported value types and a callable");
    }
    let kind = "callback";
    let maxCalls;
    if (cleanup) {
      const maxSteps = cleanup.maxSteps;
      maxCalls = cleanup.maxCalls;
      if (result !== "Unit" || !Number.isSafeInteger(maxSteps) || maxSteps <= 0 || maxSteps > 65536 || !Number.isSafeInteger(maxCalls) || maxCalls <= 0 || maxCalls > 65536) {
        throw new GuestError("invalid_capability", "Cleanup needs finite positive maxSteps and maxCalls up to 65536 and Unit result");
      }
      kind = "cleanup";
    }
    const selected = {
      kind,
      parameter,
      result
    };
    if (concurrent) Object.assign(selected, {
      concurrent: true
    });
    const signature2 = Object.freeze(selected);
    const capability = Object.freeze({
      [capabilityBrand]: true,
      parameter,
      result
    });
    capabilityOwners.set(capability, {
      lifetime,
      signature: signature2,
      asynchronous,
      maxCalls
    });
    callbacks.set(capability, (value) => call(value));
    return capability;
  }
  let ownerHooks;
  let liveOwners;
  let pendingOwner;
  if (abi.publicOwners) {
    const release = instance.exports["blot:owner_release"];
    const token = instance.exports["blot:owner_token"];
    if (typeof release !== "function" || typeof token !== "function") {
      throw new GuestError("invalid_abi", "Missing public owner hooks");
    }
    const types = [
      2,
      96,
      1,
      127,
      0,
      96,
      3,
      127,
      127,
      127,
      1,
      127
    ];
    const imported = {
      r: release,
      t: token
    };
    try {
      new WebAssembly.Instance(resourceModule([
        types,
        [
          2,
          ...resourceName("o"),
          ...resourceName("r"),
          0,
          0,
          ...resourceName("o"),
          ...resourceName("t"),
          0,
          1
        ]
      ]), {
        o: imported
      });
    } catch (cause) {
      throw new GuestError("invalid_abi", "Invalid public owner hook signatures", {
        cause
      });
    }
    for (const fn of abi.functions) {
      const ownerInput = typeof fn.parameter !== "string" && fn.parameter.kind === "owner";
      if (!ownerInput && typeof fn.result === "string") continue;
      let parameter = 127;
      if (typeof fn.parameter === "string") {
        if (fn.parameter === "F32") parameter = 125;
      } else if (fn.parameter.kind === "owner") {
        if (fn.parameter.fresh === "F32") parameter = 125;
      } else parameter = 111;
      let result = 127;
      if (fn.result === "F32") result = 125;
      const entries = [
        [
          fn.name,
          parameter
        ]
      ];
      if (ownerInput && fn.parameter.fresh !== void 0) {
        entries.push([
          `blot:owner_call/${fn.name}`,
          127
        ]);
      }
      for (const [name, argument] of entries) {
        try {
          new WebAssembly.Instance(resourceModule([
            [
              1,
              96,
              1,
              argument,
              1,
              result
            ],
            [
              1,
              ...resourceName("o"),
              ...resourceName("f"),
              0,
              0
            ]
          ]), {
            o: {
              f: instance.exports[name]
            }
          });
        } catch (cause) {
          throw new GuestError("invalid_abi", `Invalid public owner entry signature ${name}`, {
            cause
          });
        }
      }
    }
    ownerHooks = {
      release,
      token
    };
    liveOwners = /* @__PURE__ */ new Map();
  }
  function checkedOwner(value) {
    const state = typeof value === "object" && value !== null ? publicOwnerStates.get(value) : void 0;
    if (!state) {
      throw new GuestError("invalid_owner", "Expected a guest owner handle");
    }
    if (state.lifetime.disposed) {
      throw new GuestError("stale_owner", "Owner belongs to a disposed guest");
    }
    if (state.lifetime !== lifetime) {
      throw new GuestError("foreign_owner", "Owner belongs to another guest");
    }
    if (!state.live) {
      throw new GuestError("use_after_move", "Owner has already been consumed");
    }
    return state;
  }
  function publicResult(type, value) {
    if (typeof type === "string") return fromWasm(type, value);
    if (!ownerHooks || !liveOwners || typeof value !== "number" || !Number.isInteger(value) || value < -2147483648 || value > 4294967295) throw new GuestError("invalid_abi", "Invalid public owner result");
    const token = value >>> 0;
    if (token === 0 || liveOwners.has(token)) {
      throw new GuestError("invalid_owner", "Owner result is not a fresh lease");
    }
    let kind = 1;
    if (type.usage === "shared") kind = 2;
    try {
      const resource = ownerHooks.token(token, type.schema, kind) >>> 0;
      const handle = Object.freeze({
        [ownerBrand]: true,
        usage: type.usage
      });
      const state = {
        lifetime,
        schema: type.schema,
        usage: type.usage,
        token,
        live: true
      };
      if (resource !== 0) {
        if (!resources) {
          throw new GuestError("invalid_abi", "Owner resource has no cleanup journal");
        }
        resources.promote(resource);
      }
      publicOwnerStates.set(handle, state);
      liveOwners.set(token, state);
      pendingOwner = state;
      return handle;
    } catch (cause) {
      try {
        ownerHooks.release(token);
      } catch {
      }
      liveOwners.delete(token);
      throw cause;
    }
  }
  let consumedOnce;
  if (abi.functions.some((fn) => fn.multiplicity === "once")) {
    consumedOnce = /* @__PURE__ */ new Set();
  }
  function begin(name, argument, asynchronous) {
    idle();
    const fn = functions.get(name);
    if (!fn) {
      throw new GuestError("unknown_export", `Unknown function export ${name}`);
    }
    if (fn.multiplicity === "once" && consumedOnce.has(name)) {
      throw new GuestError("once_reused", `${name} has already been invoked`);
    }
    let value;
    let exportedName = name;
    let movedOwner;
    let argumentType;
    if (typeof fn.parameter === "string") {
      value = isArrayType(fn.parameter) ? 0 : scalarToWasm(fn.parameter, argument, "invalid_argument", `${name} argument`);
    } else if (fn.parameter.kind === "owner") {
      const state = typeof argument === "object" && argument !== null ? publicOwnerStates.get(argument) : void 0;
      if (state) {
        const owned = checkedOwner(argument);
        if (owned.schema !== fn.parameter.schema || owned.usage !== fn.parameter.usage) {
          throw new GuestError("owner_type", "Owner does not match the entry's exact type");
        }
        value = owned.token;
        if (fn.parameter.fresh !== void 0) {
          exportedName = `blot:owner_call/${name}`;
        }
        if (owned.usage === "unique") movedOwner = owned;
      } else {
        if (fn.parameter.fresh === void 0) {
          throw new GuestError("invalid_owner", "Entry requires an existing owner handle");
        }
        argumentType = fn.parameter.fresh;
        value = 0;
        if (!isArrayType(argumentType)) {
          value = scalarToWasm(argumentType, argument, "invalid_argument", `${name} argument`);
        }
      }
    } else {
      const owner = typeof argument === "object" && argument !== null ? capabilityOwners.get(argument) : void 0;
      if (!owner) {
        throw new GuestError("invalid_capability", `${name} requires a capability created by guest.capability()`);
      }
      if (owner.lifetime.disposed) {
        throw new GuestError("stale_capability", `${name} received a disposed instance's capability`);
      }
      if (owner.lifetime !== lifetime) {
        throw new GuestError("foreign_capability", `${name} received another instance's capability`);
      }
      if (owner.signature.kind !== fn.parameter.kind || fn.parameter.concurrent === true && owner.signature.concurrent !== true || owner.signature.dataSchema !== fn.parameter.dataSchema || owner.signature.parameter !== fn.parameter.parameter || owner.signature.result !== fn.parameter.result) {
        throw new GuestError("capability_signature", `${name} requires ${fn.parameter.parameter} -> ${fn.parameter.result}`);
      }
      if (owner.asynchronous && !asynchronous) {
        throw new GuestError("async_host_call", `${name} requires callAsync() for this capability`);
      }
      const invoke = callbacks.get(argument);
      if (!invoke) throw new Error("Live capability has no callback binding");
      value = Object.freeze({});
      active = {
        reference: value,
        signature: owner.signature,
        invoke,
        exportName: name,
        asynchronous: asynchronous && owner.asynchronous,
        maxCalls: owner.maxCalls,
        calls: 0
      };
    }
    pendingOwner = void 0;
    epoch++;
    running = true;
    try {
      entered = ownership !== void 0;
      ownership?.enter();
      resources?.begin();
      if (rootResources) tasks?.begin(rootResources, asynchronous);
      arena?.reset(0);
      if (typeof fn.parameter === "string" && isArrayType(fn.parameter)) {
        if (!arena) throw new Error("Validated array export has no arena");
        value = arrayToWasm(arena, fn.parameter, argument, name);
      }
      if (argumentType !== void 0 && isArrayType(argumentType)) {
        if (!arena) throw new Error("Validated owner input has no arena");
        value = arrayToWasm(arena, argumentType, argument, name);
      }
      const exported = instance.exports[exportedName];
      if (typeof exported !== "function") {
        throw new Error("Validated function export disappeared");
      }
      if (fn.multiplicity === "once") consumedOnce.add(name);
      if (movedOwner) {
        movedOwner.live = false;
        liveOwners.delete(movedOwner.token);
      }
      return {
        fn,
        value,
        exported
      };
    } catch (cause) {
      finish(1, true);
      throw cause;
    }
  }
  function finish(outcome, priorFailure = false) {
    if (rootResources) resources = rootResources;
    let failed = false;
    let failure;
    try {
      try {
        resources?.finish();
      } catch (cause) {
        failed = true;
        failure = cause;
      }
      try {
        if (entered) {
          ownership.leave(outcome);
          entered = false;
        }
      } catch (cause) {
        if (!failed) {
          failed = true;
          failure = cause;
        }
      }
      try {
        arena?.reset(0);
      } catch (cause) {
        if (!failed) {
          failed = true;
          failure = cause;
        }
      }
    } finally {
      if (bindings) {
        for (const [id, binding] of bindings) {
          if (binding.refs === 0) bindings.delete(id);
        }
      }
      active = void 0;
      running = false;
    }
    if (failed && pendingOwner) {
      const owner = pendingOwner;
      owner.live = false;
      liveOwners.delete(owner.token);
      pendingOwner = void 0;
      try {
        epoch++;
        running = true;
        entered = ownership !== void 0;
        ownership?.enter();
        resources?.begin();
        ownerHooks.release(owner.token);
      } catch {
      } finally {
        finish(1, true);
      }
    }
    pendingOwner = void 0;
    if (failed && !priorFailure) throw failure;
  }
  return {
    abi,
    memoryBytes: () => {
      if (memory instanceof WebAssembly.Memory) return memory.buffer.byteLength;
      return 0;
    },
    cleanupDataType(name) {
      idle();
      const parameter = functions.get(name)?.parameter;
      if (typeof parameter !== "object" || parameter.kind !== "cleanup" || parameter.dataSchema === void 0 || !dataSchemas) {
        throw new GuestError("capability_signature", "Export has no structured cleanup parameter");
      }
      const root = dataSchemas.get(parameter.dataSchema);
      let description = root.tag;
      if (root.name !== "") description = root.name;
      const type = Object.freeze({
        [dataTypeBrand]: true,
        name: description
      });
      dataTypeOwners.set(type, {
        lifetime,
        schema: parameter.dataSchema
      });
      return type;
    },
    cleanupDataCapability(definition) {
      idle();
      if (typeof definition !== "object" || definition === null) {
        throw new GuestError("invalid_capability", "Expected structured cleanup definition");
      }
      const { type, maxSteps, maxCalls, call } = definition;
      let owner;
      if (typeof type === "object" && type !== null) {
        owner = dataTypeOwners.get(type);
      }
      if (!owner || owner.lifetime !== lifetime || owner.lifetime.disposed || typeof call !== "function") {
        throw new GuestError("invalid_capability", "Cleanup data type belongs to another lifetime");
      }
      const capability = createCapability({
        parameter: "U32",
        result: "Unit",
        call: () => null
      }, false, {
        parameter: "U32",
        result: "Unit",
        call: () => null,
        maxSteps,
        maxCalls
      });
      const state = capabilityOwners.get(capability);
      capabilityOwners.set(capability, {
        ...state,
        signature: Object.freeze({
          kind: "cleanup",
          parameter: "U32",
          result: "Unit",
          dataSchema: owner.schema
        })
      });
      callbacks.set(capability, (value) => call(value));
      return capability;
    },
    capability: (definition) => createCapability(definition, false),
    cleanupCapability: (definition) => createCapability(definition, false, definition),
    capabilityAsync: (definition) => {
      requireAsync();
      if (!options.asynchronous) {
        throw new GuestError("unsupported_async", "Create the guest with { asynchronous: true } to bind async capabilities");
      }
      return createCapability(definition, true);
    },
    call(name, argument) {
      if (options.asynchronous) {
        throw new GuestError("async_host_call", "An asynchronous guest requires callAsync()");
      }
      const { fn, value, exported } = begin(name, argument, false);
      let outcome = 1;
      try {
        const result = publicResult(fn.result, exported(value));
        outcome = 0;
        return result;
      } finally {
        try {
          const completion = tasks?.finish(outcome !== 0);
          if (completion instanceof Promise) {
            throw new GuestError("async_host_call", "Synchronous task cleanup suspended");
          }
        } finally {
          finish(outcome, outcome !== 0);
        }
      }
    },
    async callAsync(name, argument) {
      const integration = requireAsync();
      const { fn, value, exported } = begin(name, argument, true);
      let outcome = 1;
      try {
        const result = publicResult(fn.result, await integration.promising(exported)(value));
        outcome = 0;
        return result;
      } finally {
        try {
          await tasks?.finish(outcome !== 0);
        } finally {
          finish(outcome, outcome !== 0);
        }
      }
    },
    read(name) {
      idle();
      const constant = constants.get(name);
      if (!constant) {
        throw new GuestError("unknown_export", `Unknown constant export ${name}`);
      }
      const exported = instance.exports[name];
      if (!(exported instanceof WebAssembly.Global)) {
        throw new Error("Validated constant export disappeared");
      }
      return scalarFromWasm(constant.type, exported.value);
    },
    releaseOwner(handle) {
      idle();
      const owner = checkedOwner(handle);
      pendingOwner = void 0;
      epoch++;
      running = true;
      let outcome = 1;
      try {
        entered = ownership !== void 0;
        ownership?.enter();
        resources?.begin();
        owner.live = false;
        liveOwners.delete(owner.token);
        ownerHooks.release(owner.token);
        outcome = 0;
      } finally {
        finish(outcome, outcome !== 0);
      }
    },
    dispose() {
      if (lifetime.disposed) return;
      idle();
      let failed = false;
      let failure;
      if (abi.publicOwners) {
        pendingOwner = void 0;
        epoch++;
        running = true;
        try {
          entered = ownership !== void 0;
          ownership?.enter();
          resources?.begin();
          resources?.dispose();
        } catch (cause) {
          failed = true;
          failure = cause;
        } finally {
          try {
            finish(0, failed);
          } catch (cause) {
            if (!failed) {
              failed = true;
              failure = cause;
            }
          }
        }
      }
      try {
        ownership?.leave(2);
        tasks?.dispose();
        if (ownership) arena?.reset(0);
      } catch (cause) {
        if (!failed) {
          failed = true;
          failure = cause;
        }
      } finally {
        lifetime.disposed = true;
        liveOwners?.clear();
        callbacks = /* @__PURE__ */ new WeakMap();
        bindings?.clear();
      }
      if (failed) throw failure;
    }
  };
}

// crates/pen-web/structured-source.ts
var signature = {
  parameter: "Array F32",
  result: "Array F32"
};
async function openStructured(bytes) {
  const guest = await instantiateGuest(bytes, {
    asynchronous: true
  });
  let incoming;
  let response = Promise.withResolvers();
  let stopping = false;
  let failure;
  const capability = guest.capabilityAsync({
    ...signature,
    call: async (values) => {
      if (incoming) throw new Error("Overlapping game frame exchanges");
      const next = Promise.withResolvers();
      incoming = next;
      response.resolve(values);
      return await next.promise;
    }
  });
  const done = guest.callAsync("main", capability).then(() => {
    if (!stopping) throw new Error("game main returned before shutdown");
  });
  void done.catch((error) => {
    failure = error;
    response.reject(error);
  });
  try {
    const description = await response.promise;
    return {
      description,
      memoryBytes: () => guest.memoryBytes(),
      next(values) {
        if (failure) return Promise.reject(failure);
        if (stopping || !incoming) {
          return Promise.reject(new Error("No pending game frame exchange"));
        }
        response = Promise.withResolvers();
        const next = incoming;
        incoming = void 0;
        next.resolve(values);
        return response.promise;
      },
      async close() {
        if (stopping) return;
        stopping = true;
        if (!failure && incoming) {
          const next = incoming;
          incoming = void 0;
          next.resolve(new Float32Array([
            0,
            0
          ]));
        }
        try {
          await done;
        } catch {
        }
        guest.dispose();
      }
    };
  } catch (error) {
    await done.catch(() => {
    });
    guest.dispose();
    throw error;
  }
}
async function runStructuredSync(bytes, exchange) {
  const guest = await instantiateGuest(bytes);
  try {
    const host = guest.capability({
      ...signature,
      call: (values) => exchange(values, guest.memoryBytes())
    });
    guest.call("main", host);
  } finally {
    guest.dispose();
  }
}
export {
  openStructured,
  runStructuredSync
};
