// Embedded in the pen binary for `pen parity`: renders render packets with
// the checkout's own TypeScript renderer (renderer.ts on Deno's WebGPU), the
// reference the native renderer is compared with. Each packet is drawn until
// its assets have loaded and settled, then written as raw RGBA.
//
//   deno run -A --no-lock --config PROJECT/deno.json render_reference.ts \
//     PROJECT ASSETS PACKETS OUT WIDTH HEIGHT NAME...
import { join, toFileUrl } from "@std/path";

const [project, assets, packets, out, widthArg, heightArg, ...names] = Deno.args;
const width = Number(widthArg), height = Number(heightArg);
const load = (file: string) => import(toFileUrl(join(project, file)).href);
const { createRenderer } = await load("renderer.ts");
const { createFileAssetSource } = await load("asset_files.ts");
const { decodeTexture } = await load("texture.ts");
const { decodeFrame, parseCatalog } = await load("boundary.ts");

const catalog = parseCatalog(JSON.parse(await Deno.readTextFile(join(assets, "catalog.json"))));
const schema = Number((await Deno.readTextFile(join(packets, "schema.txt"))).trim());
const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("no WebGPU adapter");
const device = await adapter.requestDevice();
// The first GPU error while drawing a frame: the frame cannot be a reference.
let failure: string | undefined;
device.addEventListener("uncapturederror", (event) => {
  failure ??= (event as GPUUncapturedErrorEvent).error.message;
});
const format = "bgra8unorm";
const texture = device.createTexture({
  size: [width, height],
  format,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
});
let lastEvent = performance.now();
const renderer = createRenderer({
  device,
  format,
  source: createFileAssetSource(assets),
  decodeTexture,
  target: { size: () => [width, height], view: () => texture.createView(), present: () => {} },
  event: (event: { tag: string; value: { kind: string; key: string; message?: string } }) => {
    lastEvent = performance.now();
    if (event.tag === "AssetFailed") {
      console.error(`reference renderer: ${event.value.kind} ${event.value.key}: ${event.value.message}`);
    }
  },
});
renderer.preload(catalog);
for (const name of names) {
  const bytes = await Deno.readFile(join(packets, "output", `${name}.f32`));
  const lanes = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const decoded = decodeFrame(lanes, schema, catalog).render;
  const draw = async (frame: typeof decoded) => {
    failure = undefined;
    const begun = performance.now();
    let submitted = 0;
    for (;;) {
      renderer.submit(frame);
      submitted++;
      await new Promise((resolve) => setTimeout(resolve, 16));
      const settled = renderer.quiet() && performance.now() - lastEvent > 1000;
      if ((settled && submitted >= 3) || performance.now() - begun > 30000) break;
    }
    return failure;
  };
  const failed = await draw(decoded);
  if (failed !== undefined) {
    // Deno's WebGPU refuses a full-screen inset's 4K, 4x MSAA targets; the
    // rest of the frame can still be compared without its insets.
    const again = decoded.insets.length > 0 ? await draw({ ...decoded, insets: [] }) : failed;
    if (again !== undefined) {
      await Deno.writeTextFile(join(out, `${name}.failed`), failed);
      continue;
    }
    await Deno.writeTextFile(join(out, `${name}.stripped`), failed);
  }
  const row = Math.ceil(width * 4 / 256) * 256;
  const buffer = device.createBuffer({
    size: row * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: row }, [width, height]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(buffer.getMappedRange());
  const pixels = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const from = y * row + x * 4, to = (y * width + x) * 4;
      pixels[to] = mapped[from + 2];
      pixels[to + 1] = mapped[from + 1];
      pixels[to + 2] = mapped[from];
      pixels[to + 3] = 255;
    }
  }
  buffer.unmap();
  await Deno.writeFile(join(out, `${name}.rgba`), pixels);
}
await renderer.close();
Deno.exit(0);
