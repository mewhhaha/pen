# Assets: the pipeline, its cache, lints, versions and startup

What happens to a checkout's assets between its sources and a player's
screen: the steps that build them (`pipeline.json`), the cache that keeps
what they built, the atlases the pipeline packs, the lints `pen check` runs,
the store of package versions and the patches between them, and the scene
manifests that let a game start with only what it shows first.

## Recipes evaluated at load time

Games can ship seeds and image-processing graphs instead of expanded assets.
Both hosts use the same Rust decoders, then upload ordinary textures and instance
records. Recipes run once during loading; frame packets and shaders keep their
existing interfaces. The formulas and constants belong to the game's asset
generator. Pen supplies bounded operations, with no game-specific terrain,
materials or planting rules.

* **GTEX v3** keeps the usual 16-byte GTEX header, followed by a little-endian
  U32 JSON length, a `nodes` image graph, and an embedded JPEG painting. Operations
  include seamless tiling, powers, per-channel scale/offset, luminance, periodic
  Gaussian blur, difference, absolute value, channel joining and height-to-normal
  conversion. The final image gets color, data or normal-aware mipmaps. Limits
  are 32 nodes, power-of-two output dimensions up to 1M pixels, source images up
  to 16M pixels, and a bounded total filter workload. Color alpha is independent
  data, so zero gloss does not erase RGB. Existing baked GTEX v2 files still load.
* **Instance recipes** occupy `instances.recipe` in a mesh JSON, alongside the
  usual aggregate `min`/`max` bounds. Version 1 contains a U32 `seed`,
  `random_draws`, row `stride`, flattened `inputs`, lookup `tables`, scalar `nodes`
  and 20 `output` node indices. Inputs encode the authored planting mask; arithmetic,
  selection, rounding, lookup and explicit bit-mixing nodes reconstruct records.
  Random nodes address Python-compatible integer-seeded MT19937 samples. References
  must point backward. The decoder bounds records, nodes, random draws, tables and
  total work, and rejects nonfinite results. Recipe meshes have one base LOD.
  Packaging preserves the graph as `PMRP`, a U32 expanded JSON length and one zlib
  stream, rather than baking the generated records back into the package.

`pen check` evaluates recipes as part of asset validation. Runtime compatibility
marks include these decoders, preventing an older shipping player from receiving
new asset formats. Recipes trade smaller files for temporary loading memory and
CPU work; compare cold startup and real GPU output before adopting them.

## Shipping mesh precision

Mesh JSON can opt into smaller shipping coordinates with `"precision": 16`.
Values from 16 through 23 retain that many fraction bits in positions and
normals (23 is full float32 precision). At 16, rounding error is at most
about 0.000763% per non-subnormal component, plus subnormal rounding.
Position extrema remain exact, preserving the authored bounding box. UVs,
shader attributes, triangle indices, skinning and instance records stay exact.
Authoring files and development rendering keep the original coordinates.
Packaging rounds once before the existing byte-plane compressor and keeps the
full-precision version when it is smaller. The packed format and GPU layout
are unchanged; a packed mesh can be repackaged without further rounding.
Choose precision per asset and inspect its shipping render, especially for
geometry that shaders deform or interpret as data.

## The pipeline as a graph

`pipeline.json` lists steps. Each is a Justfile recipe (`just NAME`) or one
of pen's built-in steps, with the files it builds from (`sources`) and the
files it writes (`outputs`), as globs relative to the checkout:

```json
{ "recipe": "crater", "sources": ["models/crater/**"], "outputs": ["src/assets/crater.mesh.json"] }
```

A step that reads a file another writes runs after it, whatever the file's
order (`crates/pen/src/asset_graph.rs`); otherwise steps keep the file's
order. A cycle, or a step that reads its own output, is refused. When two
steps write the same file (gdev's buildings and depot both write the
buildings' palette), the later one's version is the one that lasts, as it
would if both ran.

### The cache

A step's key is the SHA-256 of what it is (the recipe's text as `just
--show` prints it, or the built-in's name, version and options), the
outputs it declares and each source file's path and content. What a step
wrote is kept in a content-addressed store under that key
(`crates/pen/src/store.rs`): `objects/ab/abcdef…` holds each file once, by
its digest, and `records/steps/KEY.json` lists a key's outputs. Keys hold no
absolute paths or times, so a cache made in one checkout or on one machine
serves any other with the same sources.

`pen assets` brings every step's outputs up to date: a step whose outputs
are already what its sources built is left alone, one whose sources the
cache has seen has its outputs copied back without running, and the rest
run and are kept. `--only STEP` limits it to some steps. `--check` builds
nothing: it fails when a step's outputs differ from what the cache holds for
its current sources, and says which steps it could not check. `pen dev` goes
through the same cache: undoing an edit to a model script brings its meshes
back at once instead of starting Blender.

The cache is `PEN_ASSET_CACHE`, else `~/.cache/pen/assets` (`pen clean`
empties it). Records unused for 30 days are pruned, and the oldest beyond
4 GiB. Pointing several checkouts or CI runners at one directory (a CI cache
action's path, a shared volume) shares what each built: a runner without
Blender restores what Blender made on another machine. On CI:

```sh
export PEN_ASSET_CACHE=$RUNNER_TEMP/pen-assets   # restored and saved by the CI's cache step
pen assets --project ../gdev                     # runs only the steps whose sources changed
git -C ../gdev diff --exit-code                  # the committed outputs are what the sources build
```

On 2026-10-17, gdev's 18 steps were keyed in 0.46 s (24 MB of sources). The
cache found that the committed `pointer.mesh.json` is not what `just
pointer` writes today (the same numbers, pretty-printed instead of compact).

### Built-in steps

A step with `builtin` is run by pen itself, in its process for `pen
assets`, and in a `pen assets --only STEP` of its own for `pen dev`. A
built-in is a row of `asset_graph::BUILTINS`: a name, a version (raise it
when the same sources and options would build something else) and a
function given the checkout, the step and its source files. An importer or
an encoder joins the pipeline that way: a glTF importer or a KTX2 encoder
adds its row, reads its `options`, and writes only what the step's
`outputs` name. Its outputs are cached and restored like a recipe's.

### Atlases

The built-in `atlas` (`crates/pen/src/atlas.rs`) packs a step's source
images into pages, so that the sprites of an interface share a texture:

```json
{ "recipe": "ui-atlas", "builtin": "atlas", "sources": ["art/ui/*.png"],
  "outputs": ["src/assets/ui_atlas*", "src/ui_atlas.blot"],
  "options": { "out": "src/assets/ui_atlas", "blot": "src/ui_atlas.blot" } }
```

It writes `ui_atlas.png` (and `ui_atlas_1.png` on when one page of
`maxSize`, 2048 by default, is not enough), `ui_atlas.atlas.json` with each
sprite's page, pixels and texture coordinates, and, asked for, a Blot module
with one `#Frame` constant per sprite (`u0`..`v1` for ink's `ui.image`, its
size, and where its trimmed pixels were; Blot reads a frame bound first,
`let aim = ui_atlas.aim` then `aim.u0`). Packing is MaxRects with the best
short side fit, largest first, so the same sources give the same bytes.
Each image loses its fully transparent border (`trim`), and its edge pixels
are repeated round it (`extrude`, 1) with `spacing` (1) between neighbours:
a GPU test draws four sprites four times their size with linear filtering
and finds no pixel of another's color, where packed edge to edge every one
of them has some.

## Lints in `pen check`

After compiling the game and the dev tools, `pen check` lints `src/assets`
(`crates/pen/src/asset_lint.rs`), reading each file as the runtime would
(`pen_render::inspect_texture`, `inspect_mesh`, `inspect_material`, and
WGSL through naga as wgpu validates it):

| Lint | |
|---|---|
| `format` | An image, mesh, material or shader that does not read, a WAV other than PCM or float, an Ogg other than Opus, JSON that does not parse (errors); a file of a kind nothing loads (a warning) |
| `missing` | A key the catalog or a material names that has no file (an error) |
| `unused` | A file neither the catalog, a material nor the program names, which `pen build` leaves out |
| `oversized` | A texture over `maxTextureSize` (4096) a side, or a file over `maxFileSize` (16 MB); over 8192 a side, which WebGPU need not load, an error |
| `mips` | A texture over 256 a side, drawn on meshes, with one level and none made for it (a KTX2 without its chain); images get theirs on the GPU, and a packed texture's `exact` and `raw` kinds are single on purpose |

Errors fail the check. `pen.json` tunes them:

```json
"assets": { "lints": { "maxTextureSize": 4096, "allow": { "unused": ["gnorp_*"] }, "deny": ["oversized"] } }
```

On gdev (2026-10-17) they find 51 unused files (the 50 `gnorp_*` captures
and atlases of the old impostor renderer, and `walker.wgsl`), the eight
5120 x 3808 `gnorp_walk_*.png` over 4096, and nothing the runtime would
refuse. Linting its 269 files took 0.9–1.2 s on the (busy) development
machine, most of it inflating every packed texture and parsing every mesh;
finding the keys the program names was 3 s of that before it searched with
`memchr::memmem` instead of comparing every window, which `pen build`'s
asset selection gains too.

## Package versions and patches

`pen build` writes `game/files.json` into the package: every file of it by
path, with its size, SHA-256 and whether it is executable, and the
package's version, the digest of that list (`crates/pen/src/patch.rs`). It
keeps the package in a content-addressed store, the checkout's `.pen/store`
unless `--store DIR`: each file once, however many versions hold it, and
each version's list as a record.

`pen patch OLD NEW --out FILE` makes a patch between two versions, package
folders or versions in the store (a digest or its start). Each file of the
new version is nothing when the old one has it at the same path, a copy
when the old one has the same bytes elsewhere, else a binary delta against
the old file at its path when that is smaller (`crates/pen/src/delta.rs`:
64-byte blocks found by a rolling checksum, grown both ways), else the file
itself, each deflated. `pen apply FILE [DIR]` applies it to the package in
DIR, by default the one beside the executable, so a shipped game can patch
itself: it refuses a package that is not the patch's old version or whose
files have changed since, builds the new version beside the old one, checks
every file's digest, and only then swaps it in with two renames.

Measured on 2026-10-17 with gdev: two packages of 120 MB, the second with
one crystal cluster's seed changed (one mesh), a tunable's default changed
(a new `game.wasm` and `game.cwasm`) and a material's color. The store held
both in 128 MB. The patch was 46 KB: 222 files unchanged and 4 changed, the
11.5 MB `game.cwasm` among them as a delta. Applied to a copy of the first
package, the result was the second, file for file.

On Windows the swap renames the package's folder, which a running game
holds open: apply a patch from a launcher or another process there.

## Scene manifests and startup

The game is a pure function of its input, so what its starting scene
draws first is known before it ships. `pen build` (and `pen build --web`,
`pen export`) runs the program for the scene's first second without
rendering and writes `scenes.json` beside the catalog
(`crates/pen/src/scene_assets.rs`): the meshes and materials those frames
draw, each material's shader and textures, in the order first needed, the
sounds they play, and the sample count the first frame asks for. For gdev's
`main.scene` that is 11 of 73 meshes and 9 of 67 materials, with 7 shaders,
10 textures and 2 sounds.

A run that starts from that scene:

- starts reading and decoding those assets on threads of its own as soon as
  it starts (`pen_render::Prefetch`), while the program loads and the GPU
  opens, on two threads; the renderer takes each one's first load from
  there;
- asks the renderer for all of them at once as soon as it has a GPU
  (`Renderer::preload_scene`), before the program's first frame, with their
  pipelines and the passes round the scene compiled for the scene's sample
  count;
- decodes the scene's sounds before the rest of the catalog's;
- starts the rest of the catalog's preload once the scene's assets are in,
  so it never competes with them.

Headless runs still settle on the whole catalog, so every asset event
reaches the game in frame 0 and runs stay deterministic.

`PEN_STARTUP=1` prints when each phase is reached, in milliseconds since the
process started: the program loaded, the GPU adapter found, its device
open, the GPU ready, the game's first frame, the first frame drawn with
everything it asked for (when a window's loading veil starts to clear) and,
headless, settling.

### Measured

Headless (`pen --game dist/measure --headless --frames 1`, gdev's bundle
exported with `scenes.json`), RTX 2070 Super on Vulkan, the pipeline and
driver caches warm, on 2026-10-17 with ten other builds sharing the machine
(load average 30–40 on 16 threads). The builds took turns, eight rounds
inside one hold of the shared GPU lock; medians, in milliseconds since the
process started:

| | Before | After |
|---|---:|---:|
| GPU ready | 357 | 355 |
| The game's first frame | 374 | 366 |
| **First frame drawn whole** (a window's veil starts to clear) | **489** | **408** |
| Settled (the whole catalog in, the veil gone) | 1194 | 1040 |
| From the GPU ready to the first whole frame | 138 | 50 |

The work after the GPU is ready fell by two thirds, and the first whole
frame came 81 ms (17%) sooner. Four reading threads instead of two gave
393 ms, within the noise. A second set, with the final build, an
hour later: the first whole frame at 334 against 395 ms, the work after the
GPU 35 against 105 ms, settled at 969 against 1050 ms. What did not help, measured the same way:

- Loading the scene's list first, without reading it ahead, changed nothing
  (104 against 106 ms from the GPU to the first whole frame): the frame's
  own requests already went first, and the time is in decoding.
- Reading ahead at `SCHED_IDLE`: on the loaded machine the threads were
  starved while holding locks the program's and the GPU's start then waited
  on, and the first whole frame took seconds.
- When each run took the GPU lock on its own, between other lanes' jobs,
  the reading ahead cut the same work (about 175 to 45–65 ms) but the GPU
  opened 80–180 ms later in two of three sets, so the first whole frame
  ranged from 34 ms sooner to 70 ms later. Two threads keep that cost down;
  on a player's otherwise idle machine the cores they use are free.
