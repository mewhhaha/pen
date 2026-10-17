// The browser the harnesses run in. PEN_WEB_BROWSER picks it: chromium (the
// default) or firefox. Chromium draws on the GPU through Vulkan, or through
// SwiftShader with PEN_WEB_GPU=software. PEN_WEB_WITHOUT, a comma-separated
// list of `jspi` and `webgpu`, takes those away from pages, as browsers
// without them are: the host then runs its guests in workers, or draws
// through WebGL2.
const playwright = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');

export const browserName = process.env.PEN_WEB_BROWSER || 'chromium';
export const without = new Set((process.env.PEN_WEB_WITHOUT || '').split(',').map(name => name.trim()).filter(Boolean));
for (const name of without) {
  if (name !== 'jspi' && name !== 'webgpu') throw new Error(`PEN_WEB_WITHOUT: unknown feature ${name} (jspi, webgpu)`);
}

// `gpu`: whether pages draw (with WebGPU, unless it is taken away).
export async function launch({ gpu = true } = {}) {
  if (browserName === 'firefox') {
    const firefoxUserPrefs = {};
    if (without.has('jspi')) firefoxUserPrefs['javascript.options.wasm_js_promise_integration'] = false;
    if (without.has('webgpu')) firefoxUserPrefs['dom.webgpu.enabled'] = false;
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
// host's guest workers use neither.
export async function hide(target) {
  if (!without.size) return;
  await target.addInitScript(({ jspi, webgpu }) => {
    if (jspi) {
      delete WebAssembly.Suspending;
      delete WebAssembly.promising;
    }
    if (webgpu) delete Navigator.prototype.gpu;
  }, { jspi: without.has('jspi'), webgpu: without.has('webgpu') });
}
