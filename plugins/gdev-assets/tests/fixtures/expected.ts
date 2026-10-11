// What today's TypeScript makes of gdev's asset imports: the fixtures pen's
// gdev-assets plugin is compared with (tests/typescript.rs).
//
// It imports gdev's asset_parsers.ts and blot2's compiler/assets.ts and, for
// each program, captures the `sources` assets.ts passes to the native
// compiler, through `withAssetImports` and a stand-in compiler process whose
// `build` records them, as gdev's blot_runtime.ts opens its compilers.
//
// - programs.json: gdev's two programs, src/main.blot and devtools/main.blot,
//   with a snapshot of the files they read, so the test needs no checkout.
// - cases.json: small made-up checkouts: each parser at work and failing,
//   where the files come from, and WGSL cases for naga against wgsl_reflect
//   (`differs` says why the plugin's answer is not this one).
// - numbers.json: how JavaScript prints numbers, as f64 bits and text.
//
// Regenerate them from pen's root, with gdev's import map and without
// touching its lock (check deno.lock's sha256 before and after):
//
//   deno run --no-lock --config ../gdev/deno.json --allow-read --allow-write \
//     plugins/gdev-assets/tests/fixtures/expected.ts ../gdev ../blot2 \
//     plugins/gdev-assets/tests/fixtures "$(mktemp -d)"
//
// The last argument is where the made-up checkouts are written.
import { dirname, join, resolve } from "node:path";

const [gdev, blot2, out, scratch] = Deno.args.map((arg) => resolve(arg));
if (!gdev || !blot2 || !out || !scratch) {
  throw new Error("usage: expected.ts GDEV BLOT2 FIXTURES SCRATCH_DIRECTORY");
}
const { gameAssets, gameSource } = await import(
  join(gdev, "asset_parsers.ts")
);
const { withAssetImports } = await import(
  join(blot2, "compiler", "assets.ts")
);

interface Outcome {
  modules?: { path: string; source: string }[];
  error?: { file: string; message: string };
  dependencies?: string[];
}

// The modules assets.ts makes for the program at `entry`, as createBlotCompiler
// (gdev's blot_runtime.ts) asks for them.
async function capture(entry: string): Promise<Outcome> {
  entry = resolve(entry);
  const directory = dirname(entry);
  const source = gameSource(entry);
  let sources: Record<string, string | null> | undefined;
  const process = {
    pid: 0,
    compilerIdentity: "",
    build: (options: { sources?: Record<string, string | null> }) => {
      sources = options.sources;
      return Promise.resolve({
        success: true,
        revision: 1,
        attemptStats: null,
        diagnostics: [],
        bytes: new Uint8Array(0),
      });
    },
    exportCheckpoint: () => Promise.resolve(new Uint8Array(0)),
    close: () => Promise.resolve(),
    dispose: () => Promise.resolve(),
  };
  const compiler = await withAssetImports(
    process,
    entry,
    gameAssets(source, directory),
  );
  try {
    const built = await compiler.build();
    if (!built.success) {
      const [diagnostic] = built.diagnostics;
      return {
        error: { file: diagnostic.filename, message: diagnostic.message },
      };
    }
    return {
      modules: Object.entries(sources ?? {}).map(([path, text]) => ({
        path,
        source: text ?? "",
      })),
      dependencies: built.assetDependencies.map((d: { path: string }) =>
        d.path
      ),
    };
  } finally {
    await compiler.dispose();
  }
}

// The files under `root` a program's asset imports can read: each directory's
// catalog, words and tunables and its shaders, relative to `root`.
function snapshot(root: string, directories: string[]): Record<string, string> {
  const files: Record<string, string> = {};
  for (const directory of directories) {
    const candidates = ["assets/catalog.json", "text.json", "tunables.json"];
    try {
      for (const item of Deno.readDirSync(join(directory, "assets"))) {
        if (item.isFile && item.name.endsWith(".wgsl")) {
          candidates.push(`assets/${item.name}`);
        }
      }
    } catch {
      // No assets.
    }
    for (const candidate of candidates) {
      const path = join(directory, candidate);
      try {
        files[path.slice(root.length + 1)] = Deno.readTextFileSync(path);
      } catch {
        // Not there.
      }
    }
  }
  return Object.fromEntries(
    Object.entries(files).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0),
  );
}

// gdev's two programs, and the files they read, relative to its root.
const programs = [];
const read = new Set<string>();
for (const entry of ["src/main.blot", "devtools/main.blot"]) {
  read.add(gameSource(join(gdev, entry)));
  read.add(dirname(join(gdev, entry)));
  programs.push({
    name: entry,
    entry: join(gdev, entry),
    ...(await capture(join(gdev, entry))),
  });
}
const files = snapshot(gdev, [...read]);

// A made-up checkout: `files` relative to its root, the program at
// src/main.blot unless `entry` says otherwise.
interface Case {
  name: string;
  files: Record<string, string>;
  entry?: string;
  /** Why the plugin's answer differs from this one, when it does. */
  differs?: string;
}

const catalog = JSON.stringify({ meshes: [], materials: [], sounds: [] });
const shader = (name: string, code: string, differs?: string): Case => ({
  name: `shader ${name}`,
  files: {
    "src/assets/catalog.json": catalog,
    "src/assets/case.wgsl": code,
  },
  ...(differs === undefined ? {} : { differs }),
});
const tunables = (name: string, list: unknown): Case => ({
  name: `tunables ${name}`,
  files: {
    "src/assets/catalog.json": catalog,
    "src/tunables.json": JSON.stringify(list, null, 2),
  },
});
const text = (name: string, words: string): Case => ({
  name: `text ${name}`,
  files: { "src/assets/catalog.json": catalog, "src/text.json": words },
});
const catalogCase = (name: string, contents: string): Case => ({
  name: `catalog ${name}`,
  files: { "src/assets/catalog.json": contents },
});

// Two tunable names whose ids are the same (gdev's tunableId).
function tunableId(name: string): number {
  let hash = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(name)) {
    hash = Math.imul(hash ^ byte, 0x01000193) >>> 0;
  }
  return ((hash >>> 24) ^ hash) & 0xffffff;
}
const sharedIds = (() => {
  const seen = new Map<number, string>();
  for (let index = 0;; index++) {
    const name = `t${index}`;
    const id = tunableId(name);
    const other = seen.get(id);
    if (other !== undefined) return [other, name];
    seen.set(id, name);
  }
})();

const fragment = `
struct Out { @builtin(position) position: vec4f };
@vertex fn main() -> Out { var out: Out; out.position = vec4f(0.0); return out; }
`;

const cases: Case[] = [
  // Each parser, working.
  catalogCase(
    "fields",
    JSON.stringify({
      meshes: ["rock_menhir.mesh.json", "a-b.c.json", "cube.mesh.json"],
      materials: ["brass.material.json", "brass.png.material.json"],
      sounds: ["blip.wav", "hum.ogg", "x.y.wav", "wind😀.wav"],
      textures: ["ignored.png"],
    }),
  ),
  catalogCase("empty lists", JSON.stringify({ meshes: [] })),
  text(
    "words and lists",
    JSON.stringify({
      title: "Åsa’s café",
      plain: "Hello, world! ~",
      extra: "åäöÅÄÖéüÉ‹›•…×–—",
      lines: ["one", "two", ""],
      empty: "",
      none: [],
    }).replace("’", "'"),
  ),
  text("bom", "﻿" + JSON.stringify({ a: "b" })),
  text("duplicate keys", '{"a": "first", "b": "x", "a": "second"}'),
  tunables("working", {
    about: "ignored",
    tunables: [
      { name: "camera.zoom_speed", default: 0.001, min: 0.0001, max: 0.01 },
      {
        name: "salt.per_strike",
        default: 3,
        min: 1,
        max: 12,
        step: 1,
        doc: "Salt a strike frees ✓ — or so.",
      },
      { name: "a.negative", default: -0.5, min: -1, max: 1e21 },
      { name: "tiny", default: 1e-7, min: 1e-8, max: 1e-6 },
      { name: "big", default: 3e38, min: -3e38, max: 3.4e38 },
      { name: "zero", default: -0, min: -1, max: 1 },
    ],
  }),
  tunables("none", { tunables: [] }),
  // Each parser's errors.
  text("missing glyph", JSON.stringify({ hello: "héllo ✓ ✓ ♥" })),
  text("missing glyph in a list", JSON.stringify({ menu: ["ok", "naïve"] })),
  text("astral glyph", JSON.stringify({ face: "a😀b" })),
  text("field name", JSON.stringify({ Title: "x" })),
  text("keyword field", JSON.stringify({ type: "x" })),
  catalogCase(
    "two files share a name",
    JSON.stringify({ meshes: ["a.mesh.json", "a.json"] }),
  ),
  catalogCase(
    "field name",
    JSON.stringify({ sounds: ["1st.wav"] }),
  ),
  tunables("not a list", { tunable: [] }),
  tunables("not an object", { tunables: [3] }),
  tunables("bad name", { tunables: [{ name: "Camera.zoom" }] }),
  tunables("unknown fields", {
    tunables: [{ name: "a", default: 0, min: 0, max: 1, low: 0, high: 1 }],
  }),
  tunables("missing number", { tunables: [{ name: "a", min: 0, max: 1 }] }),
  tunables("not finite in f32", {
    tunables: [{ name: "a", default: 1e39, min: 0, max: 1 }],
  }),
  tunables("min not below max", {
    tunables: [{ name: "a", default: 1, min: 1, max: 1.00000001 }],
  }),
  tunables("default outside", {
    tunables: [{ name: "a", default: 0.30000000000000004, min: 0.5, max: 1e21 }],
  }),
  tunables("step", {
    tunables: [{ name: "a", default: 0, min: 0, max: 1, step: 0 }],
  }),
  tunables("doc", {
    tunables: [{ name: "a", default: 0, min: 0, max: 1, doc: 3 }],
  }),
  tunables("same field", {
    tunables: [
      { name: "a.b", default: 0, min: 0, max: 1 },
      { name: "a_b", default: 0, min: 0, max: 1 },
    ],
  }),
  tunables("same id", {
    tunables: sharedIds.map((name) => ({ name, default: 0, min: 0, max: 1 })),
  }),
  tunables("keyword field", {
    tunables: [{ name: "type", default: 0, min: 0, max: 1 }],
  }),
  // Shaders: what naga must agree with wgsl_reflect on.
  shader(
    "parameters and scalars",
    `
const a = 1.5;
const b = -2.05;
const c: f32 = 0.1;
const d = 3u;
const e = true;
const f = false;
const g = 3;
const h = -3;
const i = 4i;
const j = vec3f(1.0, 2.0, 3.0);
const k = 1e3;
const l = 2.5f;
const m = 0.0;
const n = -0.0;
const o: bool = true;
const p: u32 = 7u;
const q = 16777217.0;
const r = 1e-7;
const s = 3.4e38;
const t = array<f32, 2>(1.0, 2.0);
const u = 1.0 + 2.0;
const v = 2u * 3u;
struct Parameters { width: f32, height: f32, seed: f32 }
${fragment}`,
  ),
  shader("no parameters", `const scale = 0.25;\n${fragment}`),
  shader(
    "one parameter and another struct",
    `struct Parameters { x: f32 }\nstruct Other { y: u32 }\n${fragment}`,
  ),
  shader(
    "comments and nesting",
    `// const not_this = 1.0;
/* const nor_this = 2.0; /* nested */ const still_not = 3.0; */
const before = 4.0;
fn helper() -> f32 { const inside = 5.0; return inside; }
const after = 6.0;
${fragment}`,
  ),
  shader("five parameters", `
struct Parameters { a: f32, b: f32, c: f32, d: f32, e: f32 }
${fragment}`),
  shader("a vector parameter", `
struct Parameters { a: vec2f, b: f32 }
${fragment}`),
  shader("upper-case constant", `const SCALE = 2.0;\n${fragment}`),
  shader("keyword constant", `const data = 2.0;\n${fragment}`),
  // Where naga and wgsl_reflect part ways.
  shader(
    "a parse error",
    `const a = 1.0;\nfn broken( {\n${fragment}`,
    "the message is naga's, not wgsl_reflect's",
  ),
  shader(
    "typed constants from integers",
    `const a: u32 = 3;\nconst b: f32 = 2;\n${fragment}`,
    "wgsl_reflect leaves out a typed constant whose value is an integer literal without a suffix; naga converts it",
  ),
  shader(
    "a computed float",
    `const a = 16777217.0 - 16777216.0;\n${fragment}`,
    "wgsl_reflect rounds each literal to f32 before it computes; naga computes in f64 and rounds once",
  ),
  shader(
    "a forward reference",
    `const b = a * 2.0;\nconst a = 1.0;\n${fragment}`,
    "wgsl_reflect leaves out a constant that uses one declared after it; naga computes it",
  ),
  shader(
    "a hexadecimal float",
    `const h = 0x1p4;\n${fragment}`,
    "wgsl_reflect reads a hexadecimal float as 0; naga reads it",
  ),
  shader(
    "an aliased parameter",
    `alias Scalar = f32;\nstruct Parameters { a: Scalar }\n${fragment}`,
  ),
  shader(
    "an unknown name in a function",
    `const a = 1.0;\nfn f() -> f32 { return missing; }\n${fragment}`,
    "wgsl_reflect only parses; naga also resolves names as it makes the module",
  ),
  shader(
    "a wrong return type",
    `const a = 1.0;\nfn f() -> f32 { return 1u; }\n${fragment}`,
  ),
  shader("a float too big for f32", `const huge = 1e39;\n${fragment}`),
  shader(
    "an integer too big for i32",
    `const big = 3000000000;\nconst after = 1.0;\n${fragment}`,
  ),
  shader(
    "more scalars",
    `enable f16;
const big = 4294967295u;
const i: i32 = -5;
const half = 1.0h;
const both = true && false;
const either = false || true;
const not = !false;
const quarter = 1.0 / 4.0;
const product = 2u * 3u + 1u;
override tweak: f32 = 1.0;
var<private> state: f32 = 2.0;
const_assert big > 0u;
const last = 7.5;
${fragment}`,
  ),
  catalogCase("not JSON", "{ meshes: [] }"),
  text("a trailing comma", '{\n  "a": "b",\n}\n'),
  {
    name: "tunables an unexpected token",
    files: {
      "src/assets/catalog.json": catalog,
      "src/tunables.json": '{"tunables": [{"name": "a", "default": .5}]}',
    },
  },
  // Where the files come from.
  {
    name: "layout tools beside src",
    entry: "tools/main.blot",
    files: {
      "src/assets/catalog.json": JSON.stringify({ sounds: ["a.wav"] }),
      "src/text.json": JSON.stringify({ game: "Game" }),
      "src/assets/a.wgsl": `const a = 1.0;\n${fragment}`,
      "tools/text.json": JSON.stringify({ tool: "Tool" }),
      "tools/assets/b.wgsl": `const b = 2u;\n${fragment}`,
    },
  },
  {
    name: "layout without a catalog",
    entry: "game/main.blot",
    files: {
      "game/text.json": JSON.stringify({ word: "Word" }),
      "game/assets/c.wgsl": `const c = true;\n${fragment}`,
      "game/assets/notes.txt": "not a shader",
    },
  },
];

const caseOutcomes = [];
for (const item of cases) {
  const directory = join(scratch, "cases", item.name.replaceAll(/\W+/g, "-"));
  await Deno.remove(directory, { recursive: true }).catch(() => {});
  for (const [path, contents] of Object.entries(item.files)) {
    await Deno.mkdir(dirname(join(directory, path)), { recursive: true });
    await Deno.writeTextFile(join(directory, path), contents);
  }
  const outcome = await capture(join(directory, item.entry ?? "src/main.blot"));
  // Paths in the fixture are under /checkout, wherever this ran.
  const json = JSON.stringify(outcome).replaceAll(directory, "/checkout");
  caseOutcomes.push({
    name: item.name,
    root: "/checkout",
    entry: join("/checkout", item.entry ?? "src/main.blot"),
    files: item.files,
    ...(item.differs === undefined ? {} : { differs: item.differs }),
    ...JSON.parse(json),
  });
}

// How JavaScript prints numbers: random doubles and floats, and edge cases.
const view = new DataView(new ArrayBuffer(8));
const numbers: [string, string][] = [];
const add = (value: number) => {
  view.setFloat64(0, value);
  numbers.push([view.getBigUint64(0).toString(16).padStart(16, "0"), String(value)]);
};
for (
  const value of [
    0, -0, 1, -1, 0.1, 1e21, 1e-7, 1e-6, 123456789012345680000, 1.5e300,
    5e-324, Number.MAX_VALUE, 2 ** 53, 0.000001234, 100, 1e20, 4294967295,
    Math.fround(0.1), Math.fround(3.4e38), Math.fround(1e-45), Infinity,
    -Infinity, NaN,
  ]
) add(value);
let seed = 0x2545f491;
const random = () => {
  seed ^= seed << 13;
  seed ^= seed >>> 17;
  seed ^= seed << 5;
  return seed >>> 0;
};
const float = new DataView(new ArrayBuffer(4));
for (let index = 0; index < 600; index++) {
  float.setUint32(0, random());
  const value = float.getFloat32(0);
  if (Number.isFinite(value)) add(value);
  view.setUint32(0, random());
  view.setUint32(4, random());
  const double = view.getFloat64(0);
  if (Number.isFinite(double)) add(double);
  // Small round decimals, as people write them.
  add(Math.fround((random() % 20001 - 10000) / 1000));
}

const write = (name: string, value: unknown) =>
  Deno.writeTextFile(join(out, name), JSON.stringify(value, null, 1) + "\n");
await write("programs.json", { root: gdev, files, programs });
await write("cases.json", caseOutcomes);
await write("numbers.json", numbers);
console.log(
  `${programs.length} programs (${
    programs.map((p) => `${p.name}: ${p.modules?.length ?? p.error?.message}`)
      .join(", ")
  }), ${caseOutcomes.length} cases, ${numbers.length} numbers -> ${out}`,
);
