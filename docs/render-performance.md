# Renderer measurements

Measured on 2026-10-11, Linux, NVIDIA GeForce RTX 2070 SUPER, Vulkan,
wgpu 30. Native and browser use the same production WGSL.

## Shadow compilation and the first Tab frame

Shadow variants are requested only by loaded opaque, lit materials with
`shadow=true`. Ordinary shader loading and additional shadow/MSAA variants
compile on loader workers. Explicit pending jobs keep initial loading from
settling early; a late caster waits for its pipeline. Reloads retain the
current shader until a replacement succeeds.

The gdev catalog now compiles **11 shadow variants across nine shaders**, down
from the previous unconditional **26**. An actual-GPU regression exercises
a noncaster, a late caster, both instanced variants, a changed sample count,
and reloads with and without a caster.

Two alternating runs with a frozen `game.wasm` measured cold loader settling:

| Run | Installed renderer | Updated renderer |
| --- | ---: | ---: |
| 1 | 4,154 ms | 4,660 ms |
| 2 | 6,160 ms | 4,230 ms |

These timings do **not** establish a startup improvement. Parallel CPU builds
and the expensive remaining robot shader dominate the spread. Both used
`__GL_SHADER_DISK_CACHE=0`; catalog geometry stayed equivalent, but subsequent
asset generation changed vertex packing and tiny atlas antialiasing details.

The reported game frame 3 is the first scripted host frame after loading
settles. Before the final decal preparation change, that frame cost 80–85 ms
cold and 29 ms warm. Its painted-decal pipeline compiled synchronously for
24–27 ms cold (about 1 ms warm). The shared painted-decal variant now prepares
during scene settling, at the active MSAA count. Newly visible material MSAA
variants also queue on workers rather than compiling during submission.

After preparation, the cold first Tab frame cost **50.1 ms total / 46.9 ms
submission**, with the 17.8 ms decal compilation already paid during loading.
Two warm runs measured **36.1–36.4 ms total / 31.4–32.8 ms submission**. Loader
settling was 4,395 ms cold; warm runs settled in 747–761 ms. The shader compile
is removed from Tab, but a roughly 30 ms warm submission cost remains.
These samples do not establish the cause of that remaining cost. Lower
shadow counts alone do not remove GPU warmup, texture allocation, or
unrelated first-frame work.

A synthetic late casting material and both shadow variants became ready in
6.24 ms, including file loading and polling; pipeline compilation took
0.36 ms on its worker. This checks readiness, rather than game-scene first
draw cost. With the frozen guest, the first-gnorp command frame measured
**1.74 ms submission / 2.37 ms total** in the installed renderer and
**0.69 ms submission / 1.54 ms total** in the updated renderer. The updated
command and following three frames stayed below 2.23 ms submission / 3.07 ms
total, with no new shadow compilation after settling. These small samples
show no first-caster compile hitch; they do not establish a general frame
rate improvement.

Reproduce with a compiled, frozen guest and identical assets:

```sh
PEN_FRAME_TIMES=1 PEN_PROFILE_FRAMES=8 PEN_PROFILE_WARMUP=0 \
  __GL_SHADER_DISK_CACHE=0 pen --game /path/to/frozen-game \
  --assets /path/to/assets --scene /path/to/main.scene \
  --headless --mute --frames 8 --script key=Tab
```

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
