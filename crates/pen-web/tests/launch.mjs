// The browser the harnesses run in. PEN_WEB_BROWSER picks it: chromium (the
// default) or firefox. Chromium draws on the GPU through Vulkan, or through
// SwiftShader with PEN_WEB_GPU=software. PEN_WEB_WITHOUT=jspi takes JSPI
// away from pages, as browsers without it are: the host then runs its guests
// in workers.
const playwright = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');

export const browserName = process.env.PEN_WEB_BROWSER || 'chromium';
export const without = new Set((process.env.PEN_WEB_WITHOUT || '').split(',').map(name => name.trim()).filter(Boolean));
for (const name of without) {
  if (name !== 'jspi') throw new Error(`PEN_WEB_WITHOUT: unknown feature ${name} (jspi)`);
}

// `gpu`: whether pages draw (with WebGPU).
export async function launch({ gpu = true } = {}) {
  if (browserName === 'firefox') {
    const firefoxUserPrefs = gpu ? { 'dom.webgpu.enabled': true, 'dom.webgpu.force-enabled': true } : {};
    if (without.has('jspi')) firefoxUserPrefs['javascript.options.wasm_js_promise_integration'] = false;
    return playwright.firefox.launch({ headless: true, firefoxUserPrefs });
  }
  if (browserName !== 'chromium') throw new Error(`PEN_WEB_BROWSER: unknown browser ${browserName} (chromium, firefox)`);
  let args = [];
  if (gpu && process.env.PEN_WEB_GPU === 'software') {
    args = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader',
      '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'];
  } else if (gpu) {
    args = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=native', '--use-angle=vulkan', '--disable-vulkan-surface'];
  }
  return playwright.chromium.launch({ headless: true, args });
}

// Takes what PEN_WEB_WITHOUT names away from `target`'s pages (a page or a
// context) before their own scripts run. Workers are left as they are: the
// host's guest workers do not use JSPI.
export async function hide(target) {
  if (!without.size) return;
  await target.addInitScript(() => {
    delete WebAssembly.Suspending;
    delete WebAssembly.promising;
  });
}
