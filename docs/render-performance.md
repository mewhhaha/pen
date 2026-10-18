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

Inset meshes honor their material's alpha and blend setting. Blended draws
compose in their supplied order over transparent inset targets, with coverage
preserved for the final screen composite; opaque draws retain depth writes.

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
