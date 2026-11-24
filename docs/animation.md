# Animation and effects: glTF, skinning, vertex-animation textures, particles

What pen does for moving things, from the asset recipe that brings a model in
to the render packet that poses it, and what is left.

## glTF import (`pen gltf`)

`pen gltf FILE [--out DIR] [--name NAME] [--shader KEY] [--vat FPS]
[--vat-shader KEY]` (crates/pen-gltf) turns a `.gltf` or `.glb` into the
renderer's assets in an asset directory (default `src/assets`), as any asset
recipe does. A checkout gives it a Justfile recipe and lists the file in
`pipeline.json`, so `pen dev` imports the file again whenever an artist
exports it; gdev's `sprout` recipe is the model (`models/sprout/sprout.glb`,
itself made by `sprout-model` from a Blender script).

- **Meshes.** Each triangle primitive the file's scene places becomes a mesh
  JSON: `NAME.mesh.json` for a file of one, `NAME_MESH[_N].mesh.json`
  otherwise. Static ones are baked where the scene's nodes put them (mirrored
  ones turned right side out); smooth normals are worked out where the file
  has none. A primitive whose material has a normal map or anisotropy gets
  `tangents`: the file's `TANGENT` attribute, moved with the placement, or,
  where it has none, MikkTSpace tangents made from its positions, normals and
  texture coordinates (a vertex where mirrored coordinates meet becomes two;
  `pen_gltf::tangents` lists how it differs from the reference code). Skinned
  and static primitives alike. Coordinates stay glTF's: Y up, right-handed, in
  the file's units.
- **Materials.** `NAME_MATERIAL.material.json` for each material they use
  (`NAME_default` for none), naming `--shader` (default `scene.wgsl`): base
  color and its texture, the normal map, `instanceParameters` [1 −
  roughness, metallic, emission, 0] (gdev's `scene.wgsl` reads them as gloss,
  metal and glow), `KHR_materials_unlit`, `blend` for BLEND and
  `alphaCutoff` for MASK. The factors of `KHR_materials_clearcoat`, `sheen`,
  `anisotropy`, `transmission`, `volume`, `ior`, `specular` and
  `emissive_strength` become the material's `surface` block
  ([material shaders](materials.md)); `emissive_strength` also multiplies the
  glow, as it always has, so a shader that multiplies the glow by the
  surface's `emissiveStrength` counts it twice. Their textures are not read.
- **Images.** The PNG and JPEG images those name, as `NAME_IMAGE.png|jpg`.
  Other formats (KTX2, WebP) are left out with a note.
- **What is left out** is reported on standard error, once for each kind: the
  textures of the extensions above, and every other extension in the file's
  `extensionsUsed` (`KHR_texture_transform`, `KHR_materials_iridescence`,
  ...). A file that requires an extension, rather than using it (the
  extensions above are only ever used), is refused, but for
  `KHR_materials_unlit` and `KHR_materials_emissive_strength`.
- **Skins and animations.** A skinned primitive keeps its bind pose and
  carries, in its mesh JSON, its four joints and weights a vertex (weights
  made to add up to 1), its rig (`skeleton`: the joints and the nodes above
  them, parents first, each with its rest translation, rotation and scale;
  `skin`: each joint's node and inverse bind matrix) and every animation of
  the file as a clip (`clips`: name, duration, and channels of translation,
  rotation or scale keys, linear, step or cubic spline). Clips keep the
  file's order, so a clip's number is the same for every skinned mesh of the
  file. Primitive morph targets (`POSITION` and `NORMAL` deltas) are imported
  into the mesh JSON's `morphs` array and packed under flag 64 of `PMSH`;
  more than four joints a vertex and glTF morph-weight animation channels are
  left out with a note.
- **`NAME.import.json`** lists what was written: each mesh with its
  material, joint count and vertex-animation material, and the clips by name
  and duration, for the game (Blot imports JSON) and for people.

A file is written only when its content changes, through a rename, so an
import that changes nothing sets nothing off, and a running game never reads
half a file. The manifest lists every file the import made (`files`); the
next import of the same name removes those it no longer makes (a part
renamed, a material gone), and touches nothing else. Everything a mesh holds is checked as the renderer checks it
(`pen_render::pack_mesh`) before it is written. `pen build` packs skinned
meshes into the binary mesh form with their rig (flag 4 of `PMSH`).

## Skeletal animation

A draw is posed by the render packet's `poses` section: layers, each naming
a draw (its place in the frame's draws), one of its mesh's clips, a time in
seconds (round the clip as often as need be) and a weight. When the draw's
mesh has a skin and its material's shader has a `vertex_skinned` entry
point, the renderer samples each layer's clip, blends the layers by their
weights (translations and scales linearly, rotations by normalized
quaternion sums on one hemisphere), works out each joint's matrix (its node
in the mesh's space times its inverse bind matrix) and draws the draw alone
with the shader's skinned pipeline. Other posed draws draw as they are, in
their bind pose (pen-render's `animation.rs` and `renderer/skinning.rs`).

A skinned pipeline reads three vertex buffers, the mesh's (locations 0, 1, 2
and 8), its tangents (`Float32x4`, location 11, zeros without any) and its
joints (`Uint16x4`, location 9) and weights (`Float32x4`, location 10), and a
third bind group: the frame's joint matrices,
`var<storage, read> joints: array<mat4x4f>` at `@group(3) @binding(0)`,
bound at the draw's offset. gdev's `scene.wgsl` has the entry point:

```wgsl
@vertex fn vertex_skinned(input: SkinnedInput) -> VertexOutput {
  let skin = joints[input.joints.x] * input.weights.x + joints[input.joints.y] * input.weights.y +
    joints[input.joints.z] * input.weights.z + joints[input.joints.w] * input.weights.w;
  ...
}
```

Skinned casters draw into the shadow map skinned too. Sampling runs on the
CPU, a pure function of the packet, so recordings, rewinds and replays pose
alike. Every opaque material's shader that has `vertex_skinned` compiles its
skinned pipeline (and skinned shadow pipeline) with the rest, ahead of need.

Games may also supply explicit joint skinning matrices through the
`bone_matrices` section (e.g. computed by Ink's animation, IK, inertialization,
secondary-bone or ragdoll solvers). Explicit bone matrices override clip poses
for the named draw, or pose it from identity when no clip layers are declared.
Blendshape weights travel in the `morph_weights` section and are evaluated on
the GPU compute pass before skinning and shading. Both `poses`, `bone_matrices`
and `morph_weights` carry an `inset` field (`0` for scene draws, `1 + k` for
`insets[k].draws`) and work across opaque, translucent and inset pipelines.

## Vertex-animation textures

`pen gltf --vat FPS` also bakes each skinned primitive's clips, every frame
at FPS frames a second, into a vertex-animation texture (`*_vat.gtex`, a raw
packed texture), with a material that draws with it (`*_vat.material.json`,
the texture as its `layerTexture`) and, when the asset directory has none, the
shader that reads it (`vat.wgsl`, from pen-gltf's `src/vat.wgsl`; a game may
light it as it likes). The texture holds a small header (the vertex and clip
counts, the box all frames lie in, each clip's first frame, frame count and
duration) and then, for each frame, each vertex's place (16 bits a
coordinate across the box) and octahedral normal (8 bits a coordinate) in two
texels (pen-gltf's `vat.rs`).

A draw with that material and the skinned mesh needs no pose: the shader
reads each vertex by its index, at the clip and time its parameters give
(x: clip, y: seconds), between the two baked frames either side, the clip
looping. Such draws differ only in their parameters, so a crowd of them is
one instanced draw. Bake memory is frames × vertices × 8 bytes; a texture
is at most 2048 × 8192 texels.

## Particles

The render packet's `emitters` section declares particle emitters: a
material (its texture shapes the particles, soft dots without one, and its
color tints them), a transform, a seed, the emitter's own time, how long it
gives birth (0 for as long as frames keep it), a rate, a lifetime, a birth
ball's radius, a velocity in the emitter's space with a random spread, world
gravity, drag, sizes and colors at birth and at death, and whether they add
their light or cover what is behind them.

Nothing is kept from frame to frame. Particle k is born k / rate seconds
into the emitter's life; from the time, rate, lifetime and duration the
renderer works out which are alive (at most rate × lifetime + 1, at most
`MAX_PARTICLES`), and a compute pass places each from its birth number and
the seed alone: a random point of the birth ball and a random added
velocity (integer hashes, the same on every GPU), then gravity and drag in
closed form, `x = v(1 − e^{−kt})/k + g(t − (1 − e^{−kt})/k)/k`. So the same
emitter at the same time draws the same particles, whatever the frame rate
or what came before, and time travel needs no history. They draw as
camera-facing squares after the translucent surfaces and before overlays,
depth-tested, not writing depth (pen-render's `renderer/particles.rs`,
`shaders/particles.wgsl`, `shaders/particle_quads.wgsl`). They draw over the
opaque picture after its sky and mist, and take the mist between the eye and
their own middle: covering particles its transmittance and in-scatter, added
light its transmittance alone (ARCHITECTURE.md, "Translucent surfaces and
particles").

Left out for now: sorting alpha-blended particles back to front, collisions,
particles that stay where they were born while their emitter moves (they
move with it), and lighting.

## Games

A Blot game appends poses and emitters to Ink's named `render.FrameAccess`
pools: `poses` holds `render.Pose` and `emitters` holds `render.Emitter`
(with `matrix` for the transform and `math.Vec4` colors). `render.finish`
borrows these pools in a `render.RenderFrame`; `lib/pen/host.blot` encodes
them before their owners close. gdev's sprout (`src/assets/sprout.import.json`) is imported and
ready to draw, skinned or baked; the game does not draw it yet.
