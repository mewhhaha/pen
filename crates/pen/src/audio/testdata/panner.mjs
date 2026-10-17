// Renders the panner scenarios in Chromium's WebAudio, offline, for the
// native mixer to match (mixer.rs, `panning_matches_chromium`): writes the
// scenarios to panner.json and Chromium's output to panner.f32 beside this
// file. Each scenario is one looping source of constant samples (so the
// output is the panner's gains) through a GainNode and a PannerNode. Once a
// game frame (800 samples at 48 kHz), at the render quantum the native mixer
// applies the frame at, its position and orientation in the listener's frame
// glide there with setTargetAtTime, and the AudioListener stays where it
// starts, as the browser host has them (pen_runtime::panner).
//
//   PLAYWRIGHT_MODULE=.../playwright-core/index.mjs node crates/pen/src/audio/testdata/panner.mjs
import { writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const here = dirname(fileURLToPath(import.meta.url));

const rate = 48000;
const frame = 800;
const frames = 30;
// Every `keep`th sample is kept: 960 a scenario, at every phase of a quantum.
const keep = 25;
const f = Math.fround;
const vector = (x, y, z) => [f(x), f(y), f(z)];

const scenarios = [
  {
    name: 'a mono source round the listener, nearing and leaving',
    source: [1],
    panner: { distanceModel: 'inverse', refDistance: 1, maxDistance: 10000, rolloffFactor: 1.2, coneInnerAngle: 360, coneOuterAngle: 360, coneOuterGain: 0 },
    frames: Array.from({ length: frames }, (_, index) => {
      const angle = index / frames * 4 * Math.PI;
      const radius = 0.5 + 7.5 * Math.abs(Math.sin(index / 7));
      return {
        position: vector(radius * Math.sin(angle), 0.3 * Math.cos(index / 3), -radius * Math.cos(angle)),
        orientation: vector(0, 0, 0),
        listener: { position: vector(0, 0, 0), forward: vector(0, 0, -1), up: vector(0, 1, 0) },
      };
    }),
  },
  {
    name: 'a stereo source in a cone, past a turning listener',
    source: [1, 0.5],
    panner: { distanceModel: 'linear', refDistance: 2, maxDistance: 12, rolloffFactor: 0.9, coneInnerAngle: 60, coneOuterAngle: 200, coneOuterGain: 0.2 },
    frames: Array.from({ length: frames }, (_, index) => {
      const t = index / (frames - 1);
      const turn = t * Math.PI * 1.5;
      const facing = t * Math.PI * 2;
      return {
        position: vector(-6 + 12 * t, 1, 4 - 9 * t),
        orientation: vector(Math.sin(facing), 0, Math.cos(facing)),
        listener: { position: vector(0.5 * t, 0, 0), forward: vector(Math.sin(turn), 0, -Math.cos(turn)), up: vector(0, 1, 0) },
      };
    }),
  },
  {
    name: 'a mono source above and below a tilted listener',
    source: [1],
    panner: { distanceModel: 'exponential', refDistance: 0.5, maxDistance: 10000, rolloffFactor: 1.5, coneInnerAngle: 360, coneOuterAngle: 360, coneOuterGain: 0 },
    frames: Array.from({ length: frames }, (_, index) => {
      const t = index / (frames - 1);
      const tilt = 0.6 * Math.sin(t * Math.PI * 2);
      return {
        position: vector(3 * Math.cos(t * 5), 4 * Math.sin(t * 7), -2 - 3 * t),
        orientation: vector(0, 0, 0),
        listener: { position: vector(0, 0, 0), forward: vector(0, Math.sin(tilt), -Math.cos(tilt)), up: vector(Math.sin(tilt), Math.cos(tilt), 0) },
      };
    }),
  },
];

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  const rendered = await page.evaluate(async ({ scenarios, rate, frame, frames, keep }) => {
    const out = [];
    for (const scenario of scenarios) {
      const context = new OfflineAudioContext(2, frame * frames, rate);
      const buffer = context.createBuffer(scenario.source.length, rate, rate);
      scenario.source.forEach((value, channel) => buffer.getChannelData(channel).fill(value));
      const source = new AudioBufferSourceNode(context, { buffer, loop: true });
      const gain = new GainNode(context);
      const panner = new PannerNode(context, { panningModel: 'equalpower', ...scenario.panner });
      source.connect(gain).connect(panner).connect(context.destination);
      // The source in the listener's frame: along its right, up and back.
      const sub = (a, b) => a.map((value, index) => value - b[index]);
      const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
      const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
      const normalize = a => { const length = Math.hypot(...a); return length === 0 ? a : a.map(value => value / length); };
      const local = (vector, { forward, up }) => {
        const right = normalize(cross(forward, up));
        const ahead = normalize(forward);
        const upward = cross(right, ahead);
        return [dot(vector, right), dot(vector, upward), -dot(vector, ahead)];
      };
      const params = (state) => {
        const position = local(sub(state.position, state.listener.position), state.listener);
        const orientation = local(state.orientation, state.listener);
        return [
          [panner.positionX, position[0]], [panner.positionY, position[1]], [panner.positionZ, position[2]],
          [panner.orientationX, orientation[0]], [panner.orientationY, orientation[1]], [panner.orientationZ, orientation[2]],
        ];
      };
      scenario.frames.forEach((state, index) => {
        if (index === 0) {
          for (const [param, value] of params(state)) param.value = value;
          return;
        }
        // The native mixer applies a frame at the quantum after the samples
        // rendered before it.
        const start = Math.ceil(index * frame / 128) * 128 / rate;
        for (const [param, value] of params(state)) param.setTargetAtTime(value, start, 0.02);
      });
      source.start(0);
      const result = await context.startRendering();
      const [left, right] = [result.getChannelData(0), result.getChannelData(1)];
      const kept = [];
      for (let at = 0; at < left.length; at += keep) kept.push(left[at], right[at]);
      out.push(kept);
    }
    return out;
  }, { scenarios, rate, frame, frames, keep });
  const samples = new Float32Array(rendered.flat());
  await writeFile(join(here, 'panner.json'), JSON.stringify({ rate, frame, keep, scenarios }) + '\n');
  await writeFile(join(here, 'panner.f32'), Buffer.from(samples.buffer));
  console.log(`panner.f32: ${scenarios.length} scenarios, ${samples.length / 2 / scenarios.length} samples each`);
} finally {
  await browser.close();
}
