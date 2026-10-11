// Engine conformance: replays the inputs `pen packets DIR` recorded on
// wasmtime through the same game on V8 (gdev's own program runtime), and
// requires every render packet to match bit for bit.
//
//   pen packets dist/conformance --project ../gdev --script "..."
//   deno run -A --no-lock --config ../gdev/deno.json tools/conformance_v8.ts ../gdev dist/conformance
import { join, resolve, toFileUrl } from "@std/path";

const [checkoutArg = "../gdev", dumpArg = "dist/conformance"] = Deno.args;
const checkout = resolve(checkoutArg);
const dump = resolve(dumpArg);
// What `pen packets` ran: the checkout's compiled program and its scene.
const wasm = join(checkout, ".pen/runtime/game.wasm");
const scenePath = join(checkout, "src/main.scene");
const load = (file: string) => import(toFileUrl(join(checkout, file)).href);
const { startProgram, Operation } = await load("program_runtime.ts");
const { decodeSnapshot } = await load("snapshot.ts");

const lanes = async (path: string) => {
  const bytes = await Deno.readFile(path);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
};
const names = [...Deno.readDirSync(join(dump, "input"))]
  .map((entry) => entry.name)
  .filter((name) => name.endsWith(".f32"))
  .sort();
const schema = Number((await Deno.readTextFile(join(dump, "schema.txt"))).trim());

// The same start as the native run: a new game, then the bundle's scene.
const program = await startProgram(await Deno.readFile(wasm));
if (program.schema !== schema) {
  throw new Error(`schema ${program.schema} on V8, ${schema} in the dump`);
}
await program.request(Operation.create);
program.accept();
const scene = decodeSnapshot(await Deno.readFile(scenePath));
if (scene.schema === program.schema) {
  await program.request(Operation.loadScene, scene.state);
  program.accept();
}

let mismatches = 0;
for (const name of names) {
  const input = await lanes(join(dump, "input", name));
  const expected = await lanes(join(dump, "output", name));
  const actual: Float32Array = await program.request(Operation.frame, input);
  program.accept();
  const a = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const b = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  let first = -1;
  if (a.length === b.length) {
    first = a.findIndex((bits, lane) => bits !== b[lane]);
  } else {
    first = Math.min(a.length, b.length);
  }
  if (first !== -1) {
    mismatches++;
    if (mismatches <= 5) {
      console.error(
        `frame ${name}: ${actual.length} lanes on V8, ${expected.length} on wasmtime; ` +
          `first difference at lane ${first}: ${actual[first]} vs ${expected[first]}`,
      );
    }
  }
}
await program.close();
if (mismatches > 0) {
  console.error(`${mismatches} of ${names.length} frames differ between V8 and wasmtime`);
  Deno.exit(1);
}
console.log(`${names.length} frames: V8 and wasmtime agree bit for bit`);
