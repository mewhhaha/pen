// Bundle the pinned local Blot adapter; generated output ships without Deno.
const root = new URL("../", import.meta.url);
const source = new URL("crates/pen-web/structured-source.ts", root);
const target = new URL("crates/pen-web/structured.js", root);
// Resolve via the authored import so this tool cannot silently choose another checkout.
const upstream = new URL("../../../blot2/compiler/guest.ts", source);
const input = await Deno.readTextFile(source);
if (!input.includes("../../../blot2/compiler/guest.ts")) {
  throw Error("Unknown adapter source");
}
const temporary = new URL("crates/pen-web/structured.candidate.js", root);
try {
  const result = await new Deno.Command(Deno.execPath(), {
    args: [
      "bundle",
      "--quiet",
      "--output",
      temporary.pathname,
      source.pathname,
    ],
  }).output();
  if (!result.success) throw Error(new TextDecoder().decode(result.stderr));
  const output = await Deno.readFile(temporary);
  if (Deno.args.includes("--check")) {
    const before = await Deno.readFile(target);
    if (
      before.length !== output.length ||
      before.some((byte, i) => byte !== output[i])
    ) {
      throw Error(
        "Stale structured.js; run deno run -A scripts/bundle-structured.ts",
      );
    }
  } else await Deno.writeFile(target, output);
  console.log(`Structured task adapter matches ${upstream.pathname}`);
} finally {
  await Deno.remove(temporary).catch(() => {});
}
