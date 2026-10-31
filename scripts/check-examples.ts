// CPU qualification of the scaffold and Bounce; no renderer or desktop window.
// Build pen-codegen first. BLOTC and PEN_CODEGEN may select matching candidates.
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createZigProjectCompiler } from "../../blot2/compiler/zig_project_client.ts";
import { instantiateGuest } from "../../blot2/compiler/guest.ts";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const blot = resolve(root, "../blot2");
await Deno.mkdir(resolve(root, "target"), { recursive: true });
const work = await Deno.makeTempDir({
  dir: resolve(root, "target"),
  prefix: "example-storage-",
});
const scaffold = resolve(work, "new");
async function copy(from: string, to: string) {
  await Deno.mkdir(to, { recursive: true });
  for await (const entry of Deno.readDir(from)) {
    const source = resolve(from, entry.name), target = resolve(to, entry.name);
    if (entry.isDirectory) await copy(source, target);
    else await Deno.copyFile(source, target);
  }
}
await copy(resolve(root, "crates/pen/templates/new"), scaffold);
await Deno.writeTextFile(resolve(scaffold, "pen.json"), "{}");
const generated = await new Deno.Command(
  Deno.env.get("PEN_CODEGEN") ?? resolve(root, "target/release/pen-codegen"),
  { args: ["--project", scaffold], stdout: "inherit", stderr: "inherit" },
).output();
if (!generated.success) throw Error("scaffold code generation failed");
const paths: Record<string, string> = {};
for (
  const [name, project, test] of [[
    "bounce",
    resolve(root, "examples/bounce"),
    "bounce.blot",
  ], ["new", scaffold, "game.blot"]]
) {
  for (const kind of ["tests", "main"]) {
    const fixture = resolve(project, "src/test_fixture.blot");
    if (kind === "tests") {
      await Deno.writeTextFile(
        fixture,
        await Deno.readTextFile(resolve(project, "tests", test)),
        { createNew: true },
      );
    }
    const compiler = await createZigProjectCompiler({
      executable: Deno.env.get("BLOTC") ??
        resolve(blot, "zig-native/zig-out-dev/bin/blotc"),
      entry: kind === "tests" ? fixture : resolve(project, "src/main.blot"),
      prelude: resolve(blot, "std/prelude.blot"),
      stdRoot: resolve(blot, "std"),
      imports: { "pen/": resolve(project, "pen") + "/" },
      nativeAssets: { roots: [resolve(project, "src")] },
      cacheDirectory: false,
    });
    try {
      const result = await compiler.build();
      if (!result.success) throw Error(JSON.stringify(result.diagnostics));
      const path = resolve(work, name + "-" + kind + ".wasm");
      await Deno.writeFile(path, result.bytes);
      paths[name + "-" + kind] = path;
    } finally {
      await compiler.dispose();
      if (kind === "tests") await Deno.remove(fixture);
    }
  }
}
for (const path of [paths["bounce-tests"], paths["new-tests"]]) {
  const guest = await instantiateGuest(await Deno.readFile(path));
  if (guest.abi.version !== 8) throw Error("wrong ABI " + guest.abi.version);
  let count = 0;
  for (
    const fn of guest.abi.functions.filter((x) => x.name.startsWith("test_"))
  ) {
    const report = guest.call(fn.name, null) as Float32Array;
    if (report[1] !== 0) {
      throw Error(fn.name + ": " + JSON.stringify([...report]));
    }
    console.log(fn.name + ": " + report[0] + " checks passed");
    count++;
  }
  if (!count) throw Error("no test exports");
  guest.dispose();
}
for (
  const [name, world] of [["bounce", [3.4, 0, 2, 0, 4, 0]], ["new", [
    0,
    1,
  ]]] as const
) {
  const guest = await instantiateGuest(
    await Deno.readFile(paths[name + "-main"]),
  );
  if (guest.abi.version !== 8) throw Error("wrong ABI " + guest.abi.version);
  const requests = [
    [1, 1],
    [1, 4, ...world],
    [1, 2, 0.1, 640, 480, 0, 0],
    [0, 3],
    [1, 4, ...world],
    [1, 3],
    [1, 0],
  ];
  const responses: number[][] = [];
  const host = guest.capability({
    parameter: "Array F32",
    result: "Array F32",
    call: (v: Float32Array) => {
      responses.push([...v]);
      const next = requests.shift();
      if (!next) throw Error("extra step");
      return new Float32Array(next);
    },
  });
  guest.call("main", host);
  if (requests.length) throw Error("missing responses");
  const [hello, created, restored, rendered, afterRejected, reset, afterReset] =
    responses;
  if (
    hello[0] !== 4670544 || hello[1] !== 2 || created[0] !== 1 ||
    restored[0] !== 4 || rendered[0] !== 2
  ) throw Error("bad framing");
  if (afterReset.some((v, i) => Math.abs(v - [3, ...world][i]) > 0.00001)) {
    throw Error("restore changed");
  }
  if (name === "bounce") {
    const expected = [3, 3.406, 0, -1.94, 0, 5, 0];
    if (afterRejected.some((x, i) => Math.abs(x - expected[i]) > 0.0001)) {
      throw Error("rejected frame rolled back or repeated");
    }
    if (rendered[4] !== 11 || rendered[8] !== 1) {
      throw Error("draw/sound counts " + rendered.slice(0, 15));
    }
    if (
      JSON.stringify(rendered.slice(-6, -2)) !==
        JSON.stringify([5260632, 1, 1, 2])
    ) throw Error("score framing " + rendered.slice(-6));
  } else {
    if (Math.abs(afterRejected[1] - 0.1) > 0.00001 || afterRejected[2] !== 1) {
      throw Error("cube snapshot wrong");
    }
    if (rendered[4] !== 1 || rendered[8] !== 0) {
      throw Error("cube draw/sound counts");
    }
  }
  console.log(
    name + ": ABI8 create/frame/reject/snapshot/restore passed (" +
      rendered.length + " response lanes)",
  );
  guest.dispose();
}

console.log("Artifacts: " + work);
