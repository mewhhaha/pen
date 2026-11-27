// CPU qualification of the scaffold and Bounce; no renderer or desktop window.
// Build pen-codegen first. SQUIDC and PEN_CODEGEN may select matching candidates.
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createZigProjectCompiler } from "../../squid-lang/host/zig_project_client.ts";
import { instantiateGuest } from "../../squid-lang/host/guest.ts";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const squid = resolve(root, "../squid-lang");
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
const testPaths: string[] = [];
const inkLib = resolve(root, "../ink/lib");
for (
  const [name, project, needsInk] of [
    ["bounce", resolve(root, "examples/bounce"), false],
    ["showcase", resolve(root, "examples/showcase"), true],
    ["citadel", resolve(root, "examples/citadel"), true],
    ["new", scaffold, false],
  ] as const
) {
  const tests: string[] = [];
  for await (const entry of Deno.readDir(resolve(project, "tests"))) {
    if (entry.isFile && entry.name.endsWith(".sq")) tests.push(entry.name);
  }
  tests.sort();
  const imports: Record<string, string> = { "pen/": resolve(project, "pen") + "/" };
  if (needsInk) imports["ink/"] = inkLib + "/";
  for (const test of [...tests, null]) {
    const fixture = resolve(project, "src/test_fixture.sq");
    const compiler = await createZigProjectCompiler({
      executable: Deno.env.get("SQUIDC") ??
        resolve(squid, "compiler-goose/build/squidc-goose"),
      entry: test ? fixture : resolve(project, "src/main.sq"),
      prelude: resolve(squid, "std/prelude.sq"),
      stdRoot: resolve(squid, "std"),
      imports,
      nativeAssets: { roots: [resolve(project, "src")] },
      cacheDirectory: false,
    });
    try {
      const result = await compiler.build(
        test
          ? {
            sources: {
              [fixture]: await Deno.readTextFile(
                resolve(project, "tests", test),
              ),
            },
          }
          : undefined,
      );
      if (!result.success) throw Error(JSON.stringify(result.diagnostics));
      const key = name + "-" + (test ? "tests-" + test : "main");
      const path = resolve(work, key + ".wasm");
      await Deno.writeFile(path, result.bytes);
      paths[key] = path;
      if (test) testPaths.push(path);
    } finally {
      await compiler.dispose();
    }
  }
}
for (const path of testPaths) {
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
  if (
    guest.abi.functions.some((fn) => fn.name === "exercise_readonly_session")
  ) {
    const packets: number[][] = [];
    const host = guest.capability({
      parameter: "Array F32",
      result: "Array F32",
      call: (packet: Float32Array) => {
        packets.push([...packet]);
        return new Float32Array();
      },
    });
    guest.call("exercise_readonly_session", host);
    const expected = [[1, 3, 3, 300, 2, 66, 67], [
      1,
      7,
      1,
      65,
      2,
      2,
      3,
      2,
      2,
      3,
    ]];
    if (JSON.stringify(packets) !== JSON.stringify(expected)) {
      throw Error("readonly session packets " + JSON.stringify(packets));
    }
    console.log("readonly selector/equality: copied session packets passed");
  }
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

// Verify showcase ABI 8 create/frame/scene(5)/load_scene(6)/snapshot/restore.
{
  const guest = await instantiateGuest(
    await Deno.readFile(paths["showcase-main"]),
  );
  if (guest.abi.version !== 8) throw Error("wrong ABI " + guest.abi.version);
  let stage = 0;
  let savedScene: number[] = [];
  let savedState: number[] = [];
  const host = guest.capability({
    parameter: "Array F32",
    result: "Array F32",
    call: (v: Float32Array) => {
      const arr = [...v];
      stage++;
      if (stage === 1) {
        if (arr[0] !== 4670544 || arr[1] !== 2 || arr[3] !== 1) {
          throw Error("showcase hello bad framing: " + JSON.stringify(arr));
        }
        return new Float32Array([1, 1]); // Create
      }
      if (stage === 2) {
        if (arr[0] !== 1) throw Error("showcase create bad op");
        savedState = arr.slice(1);
        return new Float32Array([1, 2, 0.016, 1280, 720, 0, 0]); // Frame
      }
      if (stage === 3) {
        if (arr[0] !== 2 || arr.length < 100) throw Error("showcase frame too short");
        return new Float32Array([1, 5]); // Scene (op 5)
      }
      if (stage === 4) {
        if (arr[0] !== 5 || arr.length < 20) throw Error("showcase scene encode failed");
        savedScene = arr.slice(1);
        return new Float32Array([1, 6, ...savedScene]); // LoadScene (op 6)
      }
      if (stage === 5) {
        if (arr[0] !== 6 || arr.length !== savedScene.length + 1) {
          throw Error("showcase load_scene mismatch");
        }
        return new Float32Array([1, 4, ...savedState]); // Restore (op 4)
      }
      if (stage === 6) {
        if (arr[0] !== 4) throw Error("showcase restore failed");
        return new Float32Array([1, 0]); // Stop
      }
      throw Error("showcase extra step");
    },
  });
  guest.call("main", host);
  if (stage !== 6) throw Error("showcase incomplete session: " + stage);
  console.log(
    "showcase: ABI8 create/frame/scene/load_scene/restore passed (scene lanes=" +
      savedScene.length + ")",
  );
  guest.dispose();
}

// Verify citadel ABI 8 create/frame/snapshot/restore.
{
  const guest = await instantiateGuest(
    await Deno.readFile(paths["citadel-main"]),
  );
  if (guest.abi.version !== 8) throw Error("wrong ABI " + guest.abi.version);
  let stage = 0;
  let savedState: number[] = [];
  const host = guest.capability({
    parameter: "Array F32",
    result: "Array F32",
    call: (v: Float32Array) => {
      const arr = [...v];
      stage++;
      if (stage === 1) {
        if (arr[0] !== 4670544 || arr[1] !== 2) {
          throw Error("citadel hello bad framing: " + JSON.stringify(arr));
        }
        return new Float32Array([1, 1]); // Create
      }
      if (stage === 2) {
        if (arr[0] !== 1) throw Error("citadel create bad op");
        savedState = arr.slice(1);
        return new Float32Array([1, 2, 0.016, 1280, 720, 0, 0]); // Frame
      }
      if (stage === 3) {
        if (arr[0] !== 2 || arr.length < 100) throw Error("citadel frame too short");
        return new Float32Array([1, 3]); // Snapshot
      }
      if (stage === 4) {
        if (arr[0] !== 3 || arr.length !== savedState.length + 1) {
          throw Error("citadel snapshot mismatch");
        }
        return new Float32Array([1, 4, ...savedState]); // Restore
      }
      if (stage === 5) {
        if (arr[0] !== 4) throw Error("citadel restore failed");
        return new Float32Array([1, 0]); // Stop
      }
      throw Error("citadel extra step");
    },
  });
  guest.call("main", host);
  if (stage !== 5) throw Error("citadel incomplete session: " + stage);
  console.log(
    "citadel: ABI8 create/frame/snapshot/restore passed (state lanes=" +
      savedState.length + ")",
  );
  guest.dispose();
}

console.log("Artifacts: " + work);
