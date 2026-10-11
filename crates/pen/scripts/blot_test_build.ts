// Embedded in the pen binary, which runs it with the project's deno.json for
// `pen test-blot`: compiles one Blot test module of a gdev checkout into OUT
// with the checkout's test compiler (test_support/compiler.ts). `game`
// compiles it against the game on the tests' level, as test_support/
// compile.ts's gameFixture puts them together; `alone` as if it stood at
// src/test_fixture.blot. The binary chooses from the module's first line.
// The compiler's diagnostics go to stderr, and a module that does not
// compile exits 1.
//
//   deno run -A --no-lock --config PROJECT/deno.json blot_test_build.ts PROJECT MODULE game|alone OUT
import { join, resolve, toFileUrl } from "@std/path";

const [projectArg, moduleArg, mode, outArg] = Deno.args;
if (mode !== "game" && mode !== "alone") {
  console.error(`blot_test_build.ts: the mode is game or alone, not ${mode}`);
  Deno.exit(2);
}
const project = resolve(projectArg);
const { gameFixture } = await import(
  toFileUrl(join(project, "test_support/compile.ts")).href
);
const { createTestCompiler } = await import(
  toFileUrl(join(project, "test_support/compiler.ts")).href
);

const source = await Deno.readTextFile(resolve(moduleArg));
const compiler = await createTestCompiler();
let built = false;
try {
  const { bytes } = await compiler.compile(
    mode === "game" ? await gameFixture(source) : source,
  );
  await Deno.writeFile(resolve(outArg), bytes);
  built = true;
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
} finally {
  await compiler.dispose();
}
Deno.exit(built ? 0 : 1);
