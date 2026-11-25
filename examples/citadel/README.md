# Citadel Striker: Playable 3D Action-Adventure Vertical Slice

A self-contained playable 3D third-person arena vertical slice exercising the full **Ink** + **Pen** game stack end-to-end:

- **Third-Person Character Controller & Physics (`ink/physics/character`, `ink/physics/scene`)**: Capsule sweep with pillar wall sliding, elevated dais step-up, gravity, jumping (`Space`), and stamina-gated burst dash (`Shift`).
- **Chase Camera & Game Feel (`ink/rig`, `ink/feel`)**: Spring-damped third-person follow camera (`rig.follow`) with `Q`/`E` orbit, trauma-squared camera shake (`feel.Trauma`), impact freeze hitstop (`feel.Hitstop`), and damped weapon recoil (`feel.Recoil`).
- **Live Skinned Skeletal Animation + IK (`ink/animation`, `ink/animation/ik`)**: 6-bone articulated hero rig driven live in Blot via `animation.world`, `animation.skin_matrices`, `ik.two_bone`, and `render.push_skin_palette`, with the energy blade attached to the `sword_arm` bone via `animation.socket`.
- **Voxel Navmesh, Behavior Tree AI & Blot 2 Concurrency (`ink/navigation/voxel`, `ink/ai/bt`, `ink/kernel_kits`, `ink/kernel_transport`)**: Arena walkable heightfield baked via `voxel.bake`, 4 sentinel guardians driven by a flat Behavior Tree (`bt.tick` patrol/chase/attack) + multi-core `#kernel const steer_crowd` separation.
- **High-Fidelity Rendering & Spatial Audio (`ink/render`, `ink/sound`, `ink/ui/tree`)**: GPU meshlet-culled sentinels, visual `.matgraph.json` arcane shards, GPU sword ribbon trail (`render.RibbonPoint`), impact spark particle bursts (`render.Emitter`), outdoor Gerstner water moat (`ENV_WATER_EXT_TAG`) & swaying perimeter grass (`ENV_FOLIAGE_EXT_TAG`), physical sky, AgX tonemapping, LTC arena torch lights, deferred floor rune decal, 3D spatial sound panners + citadel reverb zone + adaptive combat DSP synth, live HUD, Pause menu, and quicksave/restore.

## Controls

- `W` `A` `S` `D` / Arrow keys — Move hero
- `Q` / `E` — Orbit camera left / right
- `Space` — Jump
- `Shift` — Dash
- `J` or **Left Click** — Energy blade slash
- `Esc` — Pause / Resume
- `R` — Restart arena
- `F6` / `F9` — Quicksave / Quickload

## Running

```sh
python3 make_assets.py
pen codegen
pen check
pen test-blot
pen test            # headless GPU golden visual regression suite
pen build --web     # WebGPU + Wasm web export
pen run
```
