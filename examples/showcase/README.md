# Showcase: High-Fidelity Ink + Pen Vertical Slice & Interactive Scene Editor

A runnable 3D showcase and interactive scene editor uniting **Ink** (`ink/`) and
**Pen** (`pen/`) end-to-end:

- **Visual shader graph**: iridescent crystal material authored as `src/assets/crystal.matgraph.json` (`Time`, `Sin`, `Texture2D`, clearcoat + iridescence).
- **High-fidelity GPU passes (live toggles)**:
  - `1` — Physical Bruneton atmosphere sky + drifting volumetric cloud shadows
  - `2` — Ray tracing tier (`0` off, `1` RT shadows + RTAO, `2` + RT reflections & 1-bounce GI)
  - `3` — DDGI 3D octahedral irradiance probe volume
  - `4` — Weighted Blended Order-Independent Transparency (WBOIT) on overlapping stained-glass panes
  - `5` — TAAU temporal upscaling (`0.65x` internal render scale + RCAS sharpening)
  - `6` — Outdoor GPU environment passes (Gerstner water moat via `ENV_WATER_EXT_TAG` + wind-swaying GPU foliage via `ENV_FOLIAGE_EXT_TAG`)
- **Lighting, decals & audio**: AgX filmic tonemapping, LTC rectangular softbox + stained-glass cookie projector, deferred PBR glowing floor decal, courtyard reverb zone, and ambient DSP synth pad.
- **Interactive 3D scene editor (`ink/editor/scene`) + live HUD (`ink/ui/tree`)**:
  - **Left click** — Pick scene nodes, select from the left Hierarchy sidebar, or drag the RGB X/Y/Z gizmo handles
  - **Arrow keys / Mouse wheel** — Orbit and zoom the camera
  - `W` / `E` / `R` — Translate / Rotate / Scale gizmo modes
  - `N` / `D` / `F` / `Delete` — Spawn node, Duplicate selected subtree, Instantiate 2-node Prefab, or Delete selected subtree
  - `Z` / `Y` — Atomic Undo / Redo edit history
  - `S` / `L` — Request Pen Session 2 Scene Save / Load (`editor.encode_scene` / `editor.decode_scene`)
  - `P` — Play-in-Editor snapshot & Hermite sequencer animation (press `P` again to restore the exact pre-play scene)
- **Blot 2 concurrency**: `spark` SIMD noise/foliage/water evaluation plus `#kernel` terrain chunk offload (`ink/kernel_kits` + `ink/kernel_transport`).

## Running

From this directory (`examples/showcase/`):

```sh
pen codegen
pen check
pen test-blot
pen test            # headless GPU golden visual regression suite
pen build --web     # standalone WebGPU + Wasm web bundle
pen run
```

Headless GPU smoke run (8 frames):

```sh
pen run --headless --frames 8
```

Regenerate the procedural meshes, textures, material graph, and chime WAV:

```sh
python3 make_assets.py
```
