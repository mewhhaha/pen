#!/usr/bin/env node
// Stages minified deferred WGSL shaders into OUT/shaders/ for the Cargo
// features enabled in a pen-web build (`feature = "deferred-shaders"`).
// Matches crates/pen-render/build.rs's safe WGSL minifier verbatim.
import fs from 'node:fs';
import path from 'node:path';

const FEATURE_SHADERS = {
  'render-ssgi': ['occlusion.wgsl', 'ssgi.wgsl'],
  'render-subsurface': ['subsurface.wgsl'],
  'render-taa': ['taa.wgsl', 'motion_blur.wgsl', 'velocity.wgsl'],
  'render-dof': ['dof.wgsl'],
  'render-volume': ['volume_inject.wgsl', 'volume_integrate.wgsl'],
  'render-decals': ['decals.wgsl'],
  'render-particles-cpu': ['particles.wgsl', 'particle_quads.wgsl', 'ribbons.wgsl'],
  'render-gpu-culling': ['cull.wgsl', 'hiz_base.wgsl', 'hiz.wgsl'],
  'render-atmosphere': ['atmosphere_lut.wgsl', 'clouds.wgsl'],
  'render-env': ['terrain.wgsl', 'water.wgsl', 'foliage.wgsl'],
  'render-particles': ['gpu_particles_sim.wgsl', 'gpu_particles_draw.wgsl'],
  'render-lines': ['lines.wgsl'],
  'render-jfa-outline': ['jfa_outline.wgsl'],
  'render-meshlets': ['meshlet_cull.wgsl', 'meshlet_dag_cut.wgsl'],
  'render-oit': ['oit_composite.wgsl'],
  'render-pbr-decals': ['decal_pbr.wgsl'],
  'render-post-fx': ['upscale_rcas.wgsl', 'post_fx.wgsl'],
  'render-rt': [
    'ddgi.wgsl',
    'tlas_traversal.wgsl',
    'tlas_raytrace.wgsl',
    'restir.wgsl',
    'restir_specular.wgsl',
    'svgf.wgsl',
    'sharc.wgsl',
  ],
};

const PRESETS = {
  default: Object.keys(FEATURE_SHADERS),
  compact: [
    'render-ssgi', 'render-subsurface', 'render-taa', 'render-dof', 'render-volume',
    'render-decals', 'render-particles-cpu', 'render-gpu-culling', 'render-atmosphere',
    'render-env', 'render-particles', 'render-lines', 'render-jfa-outline',
    'render-meshlets', 'render-oit', 'render-pbr-decals', 'render-post-fx',
  ],
  'web-3d': ['render-oit', 'render-atmosphere', 'render-pbr-decals', 'render-post-fx'],
  'web-2d': [],
};

function minifyWgsl(src) {
  let stripped = '';
  let i = 0;
  const len = src.length;
  while (i < len) {
    if (i + 1 < len && src[i] === '/' && src[i + 1] === '*') {
      let depth = 1;
      i += 2;
      while (i < len && depth > 0) {
        if (i + 1 < len && src[i] === '/' && src[i + 1] === '*') {
          depth++;
          i += 2;
        } else if (i + 1 < len && src[i] === '*' && src[i + 1] === '/') {
          depth--;
          i += 2;
        } else {
          i++;
        }
      }
      if (stripped.length > 0 && !' \n\t\r'.includes(stripped[stripped.length - 1])) {
        stripped += ' ';
      }
    } else if (i + 1 < len && src[i] === '/' && src[i + 1] === '/') {
      const nl = src.indexOf('\n', i);
      const lineEnd = nl === -1 ? len : nl;
      const comment = src.slice(i, lineEnd).trimEnd();
      if (comment.startsWith('// __') && comment.endsWith('__')) {
        stripped += comment;
      }
      i = lineEnd;
    } else {
      stripped += src[i++];
    }
  }
  const lines = [];
  for (const raw of stripped.split('\n')) {
    const trimmed = raw.trim();
    if (!trimmed) continue;
    lines.push(trimmed.replace(/[ \t]+/g, ' '));
  }
  return lines.join('\n');
}

const [srcDir, outDir, rawFeatures] = process.argv.slice(2);
if (!srcDir || !outDir) {
  console.error('usage: stage-shaders.mjs SRC_SHADERS_DIR OUT_DIR FEATURE_CSV');
  process.exit(2);
}

const rawList = (rawFeatures ?? '').split(',').map((s) => s.trim()).filter(Boolean);
const expanded = new Set();
let deferredEnabled = false;
for (const feat of rawList) {
  if (feat === 'deferred-shaders') deferredEnabled = true;
  if (Object.hasOwn(PRESETS, feat)) {
    if (feat !== 'web-2d') deferredEnabled = true;
    for (const f of PRESETS[feat]) expanded.add(f);
  } else {
    expanded.add(feat);
  }
}

const targetDir = path.join(outDir, 'shaders');
fs.rmSync(targetDir, { recursive: true, force: true });
if (!deferredEnabled) {
  process.exit(0);
}

const files = new Set();
for (const feat of expanded) {
  for (const name of FEATURE_SHADERS[feat] ?? []) {
    files.add(name);
  }
}
if (files.size === 0) {
  process.exit(0);
}

fs.mkdirSync(targetDir, { recursive: true });
let totalBytes = 0;
for (const name of [...files].sort()) {
  const src = fs.readFileSync(path.join(srcDir, name), 'utf8');
  const min = minifyWgsl(src);
  fs.writeFileSync(path.join(targetDir, name), min);
  totalBytes += Buffer.byteLength(min);
}
console.log(`staged ${files.size} deferred shaders (${totalBytes} bytes minified) in ${targetDir}`);
