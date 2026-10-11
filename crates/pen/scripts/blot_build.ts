// Embedded in the pen binary, which runs it with the project's deno.json:
// compiles a gdev checkout's Blot program into OUT/game.wasm with the
// checkout's own compiler setup (blot_runtime.ts), and with --watch compiles
// again whenever a Blot source, shader or the catalog changes. The binary
// closes stdin to stop it.
//
//   deno run -A --no-lock --config PROJECT/deno.json blot_build.ts PROJECT OUT [--watch]
import { Buffer } from "node:buffer";
import { join, resolve, toFileUrl } from "@std/path";

const [projectArg, outArg, ...flags] = Deno.args;
const project = resolve(projectArg);
const out = resolve(outArg);
const watch = flags.includes("--watch");

const { createBlotCompiler } = await import(
  toFileUrl(join(project, "blot_runtime.ts")).href
);
const compiler = await createBlotCompiler({
  entry: join(project, "src/main.blot"),
  packages: join(project, "packages"),
  onProgress: (progress: { stage: string; message: string; elapsedMs: number }) => {
    if (progress.stage !== "compiling") {
      console.error(`blot: ${progress.message} (${(progress.elapsedMs / 1000).toFixed(2)} s)`);
    }
  },
});

let previous: Uint8Array | undefined;
// Natively (memcmp): a byte loop took up to 16 ms of every reload.
const same = (a: Uint8Array, b: Uint8Array | undefined) => b !== undefined && Buffer.compare(a, b) === 0;

/** Writes game.wasm in one rename, so the runtime never reads half of it.
 * An unchanged build (a reverted edit, say) leaves it alone. */
async function build(): Promise<boolean> {
  try {
    const { bytes } = await compiler.artifact();
    if (same(bytes, previous)) return true;
    previous = bytes;
    const temporary = join(out, `.game.wasm.${Deno.pid}.tmp`);
    await Deno.writeFile(temporary, bytes);
    await Deno.rename(temporary, join(out, "game.wasm"));
    return true;
  } catch (error) {
    // The diagnostics are printed; a running build carries on.
    console.error(error instanceof Error ? error.message : error);
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
