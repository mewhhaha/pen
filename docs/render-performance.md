# Renderer measurements

Measured on 2026-10-16 on Linux with an NVIDIA GeForce RTX 2070 SUPER
(driver 615.78, Vulkan), a 16-thread CPU and wgpu 30. Other work was
light (load average under 6). Native and browser builds use the same production WGSL. The
FXAA measurements below date from 2026-10-11.

The load and frame numbers come from pen-render's `render_packet` example.
It replays the 120 packets that `just packets` records from gdev into
`dist/packets`. Frame 60 presses Tab, which opens the editor. `--sequence`
settles on the first packet the way pen settles its first frame: it submits
until the renderer is quiet and nothing is pending. It then submits each
later packet once, 16 ms apart, and times each submit. "Before" is a build
from before the 2026-10-16 changes, with the same mesh parsing. "After" is
the current renderer. The two builds took turns, run by run.

```sh
cargo build --release -p pen-render --example render_packet
# Cold: no NVIDIA shader disk cache and no wgpu pipeline cache.
__GL_SHADER_DISK_CACHE=0 RUST_LOG=1 target/release/examples/render_packet --sequence --veil
# The wgpu pipeline cache alone: run twice with the same directory.
__GL_SHADER_DISK_CACHE=0 target/release/examples/render_packet --sequence --veil --pipelines /var/tmp/pipelines
```

`RUST_LOG` (any value) prints each pipeline compile and how long it took.

## Loading

### Pipelines compile per material need, side by side

A material shader once got all four scene variants (opaque, translucent, and
the overlay's dimmed and full-strength passes), plus an instanced one when it
could. Now it gets only those its loaded materials draw with. Each one is a
job of its own on the loader's threads. A shader's new revision is published
once its pipelines, including a caster's shadow pipelines, are in.

For gdev's 24 mesh shaders, loading now compiles **39 scene pipelines
instead of 98**, plus the same 11 shadow pipelines. The robot shader
dominated cold loading. Its opaque and instanced pipelines took 1.7 s and its
shadow pipelines another 1.5 s, all one after another on one thread. Now
they compile in parallel.

The passes around the scene, the decal pipelines and the inset composite
compile on the loader too, not on the frame's thread. A cold present pass
took 110–810 ms with FXAA, and it used to stall the first frame and every
MSAA change.

Settling, in ms (three runs each, alternating):

| | Before | After |
| --- | ---: | ---: |
| Cold (`__GL_SHADER_DISK_CACHE=0`) | 3,572 / 5,010 / 3,638 | 1,340 / 1,681 / 1,506 |
| Warm (driver cache on) | 705 / 744 / 716 | 696 / 683 / 679 |

Warm settling is bounded by streaming textures in and by the veil's 0.6 s
clear, not by compiling.

### The pipeline cache

On Vulkan, pen keeps a `wgpu::PipelineCache` in `~/.cache/pen/pipelines/`, one
file per adapter (`wgpu_pipeline_cache_vulkan_<vendor>_<device>`, 1.97 MB for
gdev). It is written in the background when a burst of compiles ends. With
NVIDIA's own shader cache off, the first run settled in 3,228 ms and wrote
the file. The next three settled in **731 / 706 / 707 ms**, as fast as with
the driver's cache. Two runs without the wgpu cache took 1,615 and 2,837 ms.
The driver's cache already does this on NVIDIA and Mesa. The wgpu cache helps
where there is no such cache, or where it has been cleared or evicted. The
browser keeps its own cache.

### Meshes

Mesh JSON is now parsed with typed serde straight into `f32`/`u32`. The
`serde_json::Value` tree and the `f64` arrays are gone. `pen_render::pack_mesh`
also makes a compact binary form, which the renderer reads under the same
key. For gdev's 73 meshes on one thread:

| | Bytes | Parse |
| --- | ---: | ---: |
| JSON through `serde_json::Value` (before) | 59.1 MB | 334–359 ms |
| JSON, typed | 59.1 MB | 179 ms |
| Binary (`PMSH`) | 32.7 MB | 8.5 ms |

The loader parses on up to six threads, so the wall-clock saving while
loading is smaller. Reproduce with
`cargo test --release -p pen-render --lib -- --ignored --nocapture mesh_parse_time`.

## The first Tab frame

Tab opens the editor. Its first frame used to take about 30 ms to submit in
earlier measurements inside pen, which were taken on a busy machine. The
cause is in pen-render. The editor draws the arm reaching to the pointer as
a **full-screen inset** (0, 0, 1280×720), at twice the resolution and 4×
MSAA. On its first frame the renderer:

- compiled the inset composite pipeline on the frame's thread, and
- made the inset's targets with 25% room to spare, rounded to 256: 3328×2048,
  that is a 4× multisampled `Rgba16Float` colour target (218 MB), a 4×
  `Depth24PlusStencil8` target (109 MB) and the resolved image (54 MB).
  About 381 MB was allocated in one frame.

Both are fixed. The composite compiles on the loader when the renderer is
made. An inset's first targets are the size it draws (2560×1440, about
206 MB), and only an inset that outgrows them gets targets with room to
spare.

| Frame 60 submit, ms | Before | After |
| --- | ---: | ---: |
| Cold (three runs) | 24.3 / 18.5 / 24.8 | 8.8 / 8.9 / 8.6 |
| Warm (three runs) | 10.3 / 10.2 / 10.1 | 8.8 / 8.7 / 9.3 |

Timing the phases of that submit shows that what remains, about 7.9 ms, is
the driver allocating the 206 MB of inset targets. Two fixes would remove it,
and both are the game's call. The game could size the arm's inset to the
arm, not the screen. Or the renderer could supersample large insets less,
which would change how they look.

## Steady-state frames

gdev's frames draw 16–20 draw calls. A frame's working lists are now index
based and kept from frame to frame:

- placements, batches, items and ready entries, the pass lists, and sprite
  and decal batches;
- interned material render states replace the per-batch
  `format!("{:?}")` key;
- shader and material lookups go by ID, not by scanning slots;
- opaque draws are sorted by stencil mark and material, each with its
  pipeline resolved once, and repeated `set_pipeline` calls are skipped;
- only solids that changed are meshed again, into reused buffers;
- post-pass texture views are made once with their targets;
- fading detail rounds its sampler's level bias to quarters, and drops the
  bind groups it made once the detail settles;
- sprite bind groups are cached by texture.

These cut the CPU cost of a submit. The median over 354 frames went from
0.67 ms to 0.60 ms. With `--robots 40` (1,688 more instances) it went from
1.10 ms to 0.96 ms, while the GPU took 142 ms a frame.

Per-draw uniforms still keep room for 32 extra vectors (688 bytes, 768 with
alignment). gdev's `earth.wgsl` and `salt.wgsl` read `draw.extra` from that
uniform, so moving extras into a storage buffer would change the shader
interface the game's shaders use.

## Culling, levels of detail and the GPU's passes

Measured on 2026-10-17 on the same machine, under other work (load average
15 to 50, so CPU times are noisy; GPU times come from timestamps). "Before"
is 52a55af's renderer with gdev 33643cc's assets; "after" is this renderer
with gdev's shaders declaring their bounds margins and reading the cascaded
shadow maps.

### Timing the GPU's passes

With `TIMESTAMP_QUERY` (most native adapters, and WebGPU's
`timestamp-query`), every pass writes a timestamp as it begins and ends.
They are resolved into a buffer that is mapped a few frames later, so
reading them never waits on the GPU, and summed by part: shadow maps,
culling, scene, the passes after it, and sprites and insets, with the whole
frame from its first pass to its last (`RenderStats::gpu`, the dev tools'
diagnostics at high, `render_packet`'s output).

### The gnorps' shader

Timing the passes showed `render_packet --robots 40` (1,600 gnorps) taking
**149 ms** of GPU time a frame, half in the shadow pass and half in the
scene, though the gnorp has only 382 triangles. `robot3d.wgsl` kept its 32
sampled poses as `const` arrays and indexed them per vertex; naga (and other
WGSL compilers) copy a `const` array into each invocation that indexes it at
run time. Generating them as `var<private>` arrays instead
(`gdev/scripts/build_robot_mesh.py`) gives the same pixels and takes the
frame to **1.7 ms**.

### Culling

A shader that declares `bounds_margin` has its draws tested against the
camera's view and each shadow map's (pen-render's ARCHITECTURE.md, "Culling
and levels of detail"). gdev declares it for its plain shaders, the salt
(0.5) and the gnorps (0.25: their poses reach 0.24 past the rest pose's
sphere, measured over every sampled pose). The six golden scenarios stay
pixel-identical with culling on.

| `--robots 40`, 1280x720 | Before | Culled |
| --- | ---: | ---: |
| Instances drawn by the camera (before Tab, after) | 1,600 | 548, 492 |
| Casters left out of the two shadow maps | 0 | 2,903 |
| GPU, shadow / scene / whole frame (ms) | 74.0 / 74.6 / 149.3 | 0.10 / 0.55 / 1.30 |

(The "before" column has the gnorps' old shader; with the new shader alone,
nothing culled and one shadow map, the frame took 1.72 ms.)

### Batches culled on the GPU

Batches of 256 instances or more are culled by a compute pass into indirect
draws. For the 1,600 gnorps it takes 0.006 ms of GPU time and keeps exactly
what the CPU keeps (548 instances before Tab, 492 after, frame for frame).
At gdev's scale the CPU's culling is as cheap (a few microseconds a
thousand instances), so the two differ by less than the noise; the GPU's
path is for batches of many thousands, whose instances the CPU then writes
once instead of once per view and level.

### Levels of detail

`pen build` packs meshes of 256 triangles or more with up to three simpler
levels (meshopt-rs): 30 of gdev's 73 meshes get them, the assets grow 2%
(32.7 to 33.3 MB of meshes) and packing takes 4.2 s more. Reproduce with
`cargo test --release -p pen-render --features simplify --lib -- --ignored
--nocapture game_mesh_levels`. Some of them:

| Mesh | Triangles at each level (error, mesh units) |
| --- | --- |
| crater | 102,342; 7,357 (0.65); 892 (2.6); 128 (10.4) |
| rock_menhir | 61,464; 2,282 (0.005) |
| phone_handset | 31,906; 1,474 (0.004); 268 (0.016); 56 (0.066) |
| god_hand_left | 13,228; 794 (0.015); 166 (0.059); 20 (0.235) |
| robot3d | 382; 200 (0.006); 134 (0.025) |

A level is drawn where its error spans at most a pixel. With the packaged
assets, `--robots 40` draws 19 calls instead of 17 (two gnorp levels) and
the scene pass takes 0.295 ms instead of 0.451 (shadows 0.040 instead of
0.051). Levels apply only to shaders that declare their bounds, as only
their meshes' positions are where they draw; gdev's god hand, grass and
blueprints keep their whole meshes.

### Cascaded shadows

gdev's lit shaders read pen's nearer shadow map (`shadow_cascades`). In
gdev's play view at 1920x1280 the shadow passes took 1.10 ms with the
game's map alone, 1.99 ms with one nearer map and 2.99 ms with two: gdev's
largest casters (the 102,342-triangle crater and the rocks) reach into every
map, so each costs about as much as the game's. pen fits one. The golden
scenarios change by 0.09 to 0.18% RMSE (under 0.06% of pixels visibly):
shadow edges near the camera are crisper and grazing facets of the rock
shadow themselves.

### Dynamic resolution

`pen run --frame-budget MS` (and `Renderer::frame_budget`) lowers the
scene's render scale while the GPU takes longer than the budget. In gdev's
play view at 1920x1280, `--frame-budget 0.4` (more than this GPU can do) took
the scale to 50% within half a second: the scene pass from 1.91 to 1.30 ms
and the passes after it from 1.16 to 0.39 ms, the shadows unchanged at 3.0
(two nearer maps then).

### Before and after

`render_packet --sequence --veil` over the 119 packets of `just packets`, at
1280x720, three to six alternating rounds each (the range of their
medians), on a machine other work kept busy.
"Before" is 52a55af's `render_packet` with gdev 33643cc's assets; "after" is
this renderer with the export's assets (`pen export`, whole meshes), and
with the package's (`pen build`, levels of detail). Wall times are from a
submit to the GPU finishing it, with pass timestamps off (`--untimed`); GPU
times are the timestamps of separate timed runs.

| | Before | After | After, levels of detail |
| --- | ---: | ---: | ---: |
| Submit (CPU), ms | 0.93-1.14 | 1.01-1.23 | 1.05-1.11 |
| Submit to GPU done, ms | 2.40-2.75 | 2.70-2.94 | 2.68-2.82 |
| GPU: shadow / scene / post / whole, ms | | 0.07 / 0.44 / 0.46 / 1.07 | 0.05 / 0.29 / 0.46 / 0.89 |
| `--robots 40`: submit (CPU), ms | 1.27-1.32 | 1.29-1.53 | 1.29-1.52 |
| `--robots 40`: submit to GPU done, ms | 139.2-139.4 | 3.15-4.65 | 3.18-3.88 |
| `--robots 40`: GPU shadow / cull / scene / whole, ms | 74.0 / - / 74.6 / 149.3 | 0.10 / 0.007 / 0.55 / 1.30 | 0.08 / 0.007 / 0.36 / 1.07 |
| `--robots 40`: gnorps drawn (culled), shadow casters left out | 1,600 (0), 0 | 548 or 492 (1,052 or 1,108), 2,903 | the same |

The same renderer with the old assets (no bounds declared, so nothing
culled, and no nearer shadow map) took 1.04-1.15 ms to submit and 2.64-2.79
to finish: its own work is within the noise of these runs. Pass timestamps,
when on, cost about 0.3 ms of wall time a frame here (consecutive passes no
longer overlap), so only the dev tools' timing (diagnostics at medium and
high), a frame budget or `render_packet` turn them on. The nearer shadow
map adds about 0.04 ms of GPU time to these frames, and reading it in the
lit shaders about 0.09 ms of the scene's.

The browser host's wasm grows from 751,808 to 821,933 bytes (301,579 to
340,599 gzipped): culling, cascades, timestamps, dynamic resolution and the
WebGPU bindings for compute passes, indirect draws and query sets they use.
The simplifier stays out of it (the `simplify` feature is pen's). In
headless Chromium on the GPU, the exported game draws with WebGPU's
`timestamp-query`, the nearer shadow map and the culling pipeline compiled,
with no console or validation errors.

## FXAA Quality

The old directional blur is replaced by the
[NVIDIA FXAA 3.11 Quality algorithm](https://github.com/GameTechDev/CMAA2/blob/master/Projects/CMAA2/FXAA/Fxaa3_11.h),
with preset-12 bounded endpoint search, contrast-based orientation, endpoint
sign rejection, and a measured subpixel coefficient of **0.25**. The existing
boolean selects off or this fixed preset. Luma uses the renderer's tone-mapped
display RGB; HUD sprites and host UI draw afterwards.

Independent analytic diagonal, thin-fence, and frequency-detail scenes use a
16×16 box-supersampled reference. Values below are normalized RGB mean
absolute error; lower is better. They include output quantization.

| Witness | Off | Previous | Quality |
| --- | ---: | ---: | ---: |
| Diagonal | 0.002065 | 0.002639 | 0.000918 |
| Thin fence | 0.035229 | 0.038411 | 0.019225 |
| High-frequency texture | 0.011903 | 0.073406 | 0.014280 |
| Low-contrast detail | 0.000683 | 0.000681 | 0.000683 |
| Moving fence residual | 0.041596 | 0.040477 | 0.031169 |

Quality reduces diagonal error by 56%, fence error by 45%, and the motion
witness's residual by 25% relative to off. High-frequency texture error rises
20% relative to off, while falling 81% relative to the previous filter. Low
contrast detail is byte-for-byte preserved. This is spatial AA; the moving
witness measures residual relative to reference-frame differences, not a
temporal reconstruction method.

Subpixel 0.25/0.5/0.75 gave fence errors 0.019225/0.022876/0.028406 and texture
errors 0.014280/0.016399/0.020184. Coefficient 0.25 retains more detail; 0.5 had
a slightly lower motion residual (0.030115 versus 0.031169).

GPU timestamps measure the complete present pass from an FP16 scene target,
16 warmup draws then 64 samples, with one queue submission per mode. The
high-contrast fence is deliberately demanding. Times are p05 / median / p95
milliseconds; these are GPU pass costs, not whole-frame rates.

| Resolution | Off | Previous | Quality |
| --- | ---: | ---: | ---: |
| 1920×1080 | .0514 / .0527 / .0537 | .1659 / .1663 / .1679 | .2865 / .2870 / .2902 |
| 2560×1440 | .0888 / .0902 / .0922 | .2872 / .2885 / .2908 | .5047 / .5061 / .5468 |

Acceptance also renders actual HUD sprites and host overlay draws and asserts
unchanged pixels at three sizes, including render scale 0.5. HDR witnesses
check tone mapping before the contrast test. Resizes check separate source
texel/output pixel sizes. The test requires an adapter and timestamp queries;
unavailable prerequisites fail instead of silently passing.

```sh
PEN_FXAA_WITNESSES=target/fxaa cargo test --release -p pen-render \
  post::fxaa_tests -- --nocapture --test-threads=1
```

Artifacts include off/previous/quality/reference PNGs, twelve motion frames,
raw inputs, exact emitted production shaders, `manifest.json`, and
`results.json`. Browser qualification uses these same inputs and shaders;
`crates/pen-web/tests/fxaa.mjs` compares browser pixels to native images.
Chromium 1248 WebGPU on SwiftShader agreed within **one 8-bit level for all
twelve final-preset still images**. Off and Quality's low-contrast detail
were bit exact. Quality's mean differences were 0.00342 / 0.01227 / 0.01965
8-bit levels for diagonal / fence / texture. The browser test asserts this
tolerance directly against native raw RGBA witnesses. A small epsilon in
edge comparisons stabilizes otherwise precision-sensitive ties across these
backends.

The synthetic witnesses isolate aliasing and detail loss; they do not prove
quality or performance on every game, GPU, browser, or display transfer
function. Physical external-device checks were unavailable in this Linux
environment.
