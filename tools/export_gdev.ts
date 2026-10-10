// Builds a game bundle for the native runtime from a gdev checkout: the
// compiled Blot program, the asset directory and the starting scene.
//
//   deno run -A --no-lock --config ../gdev/deno.json tools/export_gdev.ts ../gdev dist/gdev
//
// `--no-lock` keeps the checkout's lockfile untouched; the tool imports only
// what gdev already depends on.
import { join, resolve, toFileUrl } from "@std/path";

const [checkoutArg = "../gdev", outArg = "dist/gdev"] = Deno.args;
const checkout = resolve(checkoutArg);
const out = resolve(outArg);

async function copyTree(from: string, to: string): Promise<void> {
  await Deno.mkdir(to, { recursive: true });
  for await (const item of Deno.readDir(from)) {
    const source = join(from, item.name);
    const target = join(to, item.name);
    if (item.isDirectory) await copyTree(source, target);
    else if (item.isFile) await Deno.copyFile(source, target);
  }
}

const { createBlotCompiler } = await import(
  toFileUrl(join(checkout, "blot_runtime.ts")).href
);
const compiler = await createBlotCompiler({
  entry: join(checkout, "src/main.blot"),
  packages: join(checkout, "packages"),
  onProgress: (progress: { message: string }) => console.log(progress.message),
});
try {
  const artifact = await compiler.artifact();
  await Deno.remove(out, { recursive: true }).catch(() => {});
  await Deno.mkdir(out, { recursive: true });
  await Deno.writeFile(join(out, "game.wasm"), artifact.bytes);
  await copyTree(join(checkout, "src/assets"), join(out, "assets"));
  await Deno.copyFile(join(checkout, "src/main.scene"), join(out, "main.scene"));
  console.log(
    `wrote ${out}: game.wasm (${artifact.bytes.length} bytes), assets, main.scene`,
  );
} finally {
  await compiler.dispose();
}
