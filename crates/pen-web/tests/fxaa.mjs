// Renders pen-render's emitted production shader witnesses through real WebGPU.
import { mkdir, writeFile } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const output = process.env.PEN_FXAA_BROWSER || 'target/fxaa/browser';
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, args: [
  '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader',
  '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
] });
try {
  const page = await browser.newPage();
  await page.goto(process.env.PEN_FXAA_URL || 'http://127.0.0.1:8043/fxaa/');
  const evidence = await page.evaluate(async () => {
    const manifest = await (await fetch('manifest.json')).json();
    const { width, height } = manifest;
    if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0 || width * 4 % 256) throw new Error('invalid witness dimensions');
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    const source = device.createTexture({ size: [width, height], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const marks = device.createTexture({ size: [width, height], format: 'r8uint', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const glow = device.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: marks }, new Uint8Array(width * height), { bytesPerRow: width }, [width, height]);
    device.queue.writeTexture({ texture: glow }, new Uint8Array(4), { bytesPerRow: 4 }, [1, 1]);
    const uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const target = device.createTexture({ size: [width, height], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const readback = device.createBuffer({ size: width * height * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const quality = await (await fetch('present-quality.wgsl')).text();
    const legacy = await (await fetch('present-legacy.wgsl')).text();
    const results = [];
    for (const scene of ['diagonal', 'fence', 'texture', 'detail']) {
      const input = new Uint8Array(await (await fetch(`${scene}-input.rgba`)).arrayBuffer());
      if (input.length !== width * height * 4) throw new Error(`invalid ${scene} input size`);
      device.queue.writeTexture({ texture: source }, input, { bytesPerRow: width * 4 }, [width, height]);
      for (const version of ['off', 'legacy', 'quality']) {
        device.pushErrorScope('validation');
        let code = quality;
        if (version === 'legacy') code = legacy;
        const module = device.createShaderModule({ code });
        const compilation = await module.getCompilationInfo();
        if (compilation.messages.some(m => m.type === 'error')) throw new Error(JSON.stringify(compilation.messages));
        const pipeline = await device.createRenderPipelineAsync({
          layout: 'auto', vertex: { module, entryPoint: 'vertex' },
          fragment: { module, entryPoint: 'fragment', targets: [{ format: 'rgba8unorm' }] },
          primitive: { topology: 'triangle-list' },
        });
        const lanes = new Float32Array(manifest.uniform);
        lanes[6] = 1;
        if (version === 'off') lanes[6] = 0;
        device.queue.writeBuffer(uniform, 0, lanes);
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: source.createView() },
          { binding: 2, resource: sampler }, { binding: 3, resource: marks.createView() },
          { binding: 4, resource: glow.createView() },
        ] });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.draw(3); pass.end();
        encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow: width * 4 }, [width, height]);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const pixels = Array.from(new Uint8Array(readback.getMappedRange()));
        readback.unmap();
        const error = await device.popErrorScope();
        if (error) throw new Error(error.message);
        const response = await fetch(`${scene}-${version}.rgba`);
        if (!response.ok) throw new Error(`missing native ${scene}-${version} witness`);
        const native = new Uint8Array(await response.arrayBuffer());
        if (native.length !== pixels.length) throw new Error(`invalid native ${scene}-${version} output size`);
        let max = 0;
        let total = 0;
        for (let index = 0; index < native.length; index++) {
          const difference = Math.abs(native[index] - pixels[index]);
          max = Math.max(max, difference);
          total += difference;
        }
        results.push({ scene, version, pixels, max, mean: total / native.length });
      }
    }
    device.destroy();
    return { width, height, results };
  });
  if (evidence.results.length !== 12) throw new Error('expected twelve witnesses');
  for (const { scene, version, pixels } of evidence.results) await writeFile(`${output}/${scene}-${version}.rgba`, new Uint8Array(pixels));
  const comparisons = evidence.results.map(({ scene, version, max, mean }) => ({ scene, version, max, mean }));
  await writeFile(`${output}/comparison.json`, JSON.stringify(comparisons, null, 2));
  await writeFile(`${output}/manifest.json`, JSON.stringify({ width: evidence.width, height: evidence.height, adapter: 'SwiftShader WebGPU', witnesses: evidence.results.length }));
  console.log(`Rendered ${evidence.results.length} browser FXAA witnesses in ${output}`);
  if (comparisons.some(witness => witness.max > 1)) throw new Error(`native/browser FXAA differs by more than one LSB: ${JSON.stringify(comparisons)}`);
  console.log('All twelve native/browser FXAA witnesses agree within one LSB.');
} finally { await browser.close(); }
