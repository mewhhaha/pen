// Embedded in the pen binary, which runs it with the project's deno.json:
// compiles a gdev checkout's Blot program into OUT/game.wasm with the
// checkout's own compiler setup (blot_runtime.ts), and with --watch compiles
// again whenever a Blot source, shader or the catalog changes. The binary
// closes stdin to stop it. --entry and --name build another program of the
// checkout's (its dev tools, devtools/main.blot) into OUT/NAME.
//
//   deno run -A --no-lock --config PROJECT/deno.json blot_build.ts PROJECT OUT [--watch]
//     [--entry PATH --name FILE]
import { Buffer } from "node:buffer";
import { join, resolve, toFileUrl } from "@std/path";

const [projectArg, outArg, ...flags] = Deno.args;
const project = resolve(projectArg);
const out = resolve(outArg);
const watch = flags.includes("--watch");
const option = (name: string, otherwise: string) => {
  const at = flags.indexOf(name);
  return at === -1 ? otherwise : flags[at + 1];
};
const entry = option("--entry", "src/main.blot");
const name = option("--name", "game.wasm");
// Another program's lines say which it is, apart from the game's.
const say = (text: string) =>
  console.error(name === "game.wasm" ? text : text.split("\n").map((line) => `${entry}: ${line}`).join("\n"));

const { createBlotCompiler } = await import(
  toFileUrl(join(project, "blot_runtime.ts")).href
);
const compiler = await createBlotCompiler({
  entry: join(project, entry),
  packages: join(project, "packages"),
  onProgress: (progress: { stage: string; message: string; elapsedMs: number }) => {
    if (progress.stage !== "compiling") {
      say(`blot: ${progress.message} (${(progress.elapsedMs / 1000).toFixed(2)} s)`);
    }
  },
});

let previous: Uint8Array | undefined;
// Natively (memcmp): a byte loop took up to 16 ms of every reload.
const same = (a: Uint8Array, b: Uint8Array | undefined) => b !== undefined && Buffer.compare(a, b) === 0;

/** Writes the program in one rename, so the runtime never reads half of
 * it. An unchanged build (a reverted edit, say) leaves it alone. */
async function build(): Promise<boolean> {
  try {
    const { bytes } = await compiler.artifact();
    if (same(bytes, previous)) return true;
    previous = bytes;
    const temporary = join(out, `.${name}.${Deno.pid}.tmp`);
    await Deno.writeFile(temporary, bytes);
    await Deno.rename(temporary, join(out, name));
    return true;
  } catch (error) {
    // The diagnostics are printed; a running build carries on.
    say(error instanceof Error ? error.message : String(error));
    return false;
  }
}

await Deno.mkdir(out, { recursive: true });
const built = await build();
if (!watch) {
  await compiler.dispose();
  Deno.exit(built ? 0 : 1);
}

const watcher = Deno.watchFs(compiler.roots, { recursive: true });
let debounce: ReturnType<typeof setTimeout> | undefined;
let building = Promise.resolve();
// The binary closes stdin when it stops; so does its death.
const closed = (async () => {
  for await (const _ of Deno.stdin.readable) {
    // Nothing is sent; reading only notices the end.
  }
})();
const watching = (async () => {
  for await (const event of watcher) {
    if (!event.paths.some((path) => compiler.builds(path))) continue;
    if (debounce !== undefined) clearTimeout(debounce);
    debounce = setTimeout(() => {
      debounce = undefined;
      building = building.then(async () => {
        await build();
      });
    }, 15);
  }
})();
await closed;
watcher.close();
if (debounce !== undefined) clearTimeout(debounce);
await watching.catch(() => {});
await building;
await compiler.dispose();
Deno.exit(0);
