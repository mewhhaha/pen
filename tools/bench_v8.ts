// The native runtime's benchmark sequence (`pen bench`) on V8 (Deno), for
// comparison, on the program `pen bench` compiled into the checkout:
//   deno run -A --no-lock --config ../gdev/deno.json tools/bench_v8.ts ../gdev 600
import { join, resolve, toFileUrl } from "@std/path";

const [checkoutArg = "../gdev", framesArg = "600"] = Deno.args;
const checkout = resolve(checkoutArg);
const frames = Number(framesArg);
const load = (file: string) => import(toFileUrl(join(checkout, file)).href);
const { startProgram, Operation } = await load("program_runtime.ts");
const { encodeFrame } = await load("boundary.ts");
const { decodeSnapshot } = await load("snapshot.ts");

const program = await startProgram(await Deno.readFile(join(checkout, ".pen/runtime/game.wasm")));
await program.request(Operation.create);
program.accept();
const scene = decodeSnapshot(await Deno.readFile(join(checkout, "src/main.scene")));
if (scene.schema === program.schema) {
  await program.request(Operation.loadScene, scene.state);
  program.accept();
}
const viewport = { width: 1280n, height: 720n };
const times: number[] = [];
for (let index = 0; index < frames; index++) {
  const events = index === 60
    ? [{ tag: "KeyDown", value: "Tab" }]
    : index === 61
    ? [{ tag: "KeyUp", value: "Tab" }]
    : [];
  const frame = { timestamp: index / 60, viewport, events };
  const started = performance.now();
  await program.request(
    Operation.frame,
    encodeFrame(frame, index === 0 ? 0 : 1 / 60, new Float32Array()),
  );
  times.push(performance.now() - started);
  program.accept();
}
await program.close();
const sorted = times.slice(120).sort((a, b) => a - b);
const at = (q: number) => sorted[Math.floor(q * (sorted.length - 1))].toFixed(2);
console.log(
  `V8: frames ${frames}, after 120 warm-up: median ${at(0.5)} ms, p95 ${at(0.95)} ms`,
);
