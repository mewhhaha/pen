# Texture compression (KTX2, BC7): what it would take

An investigation, not implemented. Numbers are for the bundle `pen build`
packaged on 2026-10-11 (game/assets, 108.4 MB), measured with a script that
reads every `.gtex` header and PNG size.

## Where the bytes are

| | On disk | Deflated again (≈ download) | GPU memory |
|---|---|---|---|
| Mesh JSON (`*.mesh.json`, 57 meshes) | 52.3 MB | 9.3 MB | (vertex buffers) |
| `.gtex`, block-compressible kinds (10 files) | 25.3 MB | 25.2 MB | 39.4 MB |
| `.gtex`, `exact`/`raw` kinds (24 files) | 7.5 MB | 7.4 MB | 104.3 MB |
| PNG referenced by a material (14 files) | 10.1 MB | ≈ 9.8 MB | 26.7 MB |
| PNG referenced by nothing (29 files) | 11.2 MB | ≈ 10.8 MB | none (never loaded) |
| Sounds, shaders, materials, scene | ≈ 2 MB | | |

Three things matter more than block compression, or as much:

- **Unused files ship.** `pen build` copies all of `src/assets`. The 29
  `gnorp_*.png` captures (8 of them 5120×3808) are named by no material,
  mesh, scene or source file, and the 21 `gnorp_*.gtex` atlases (exact
  palette indices, oct normals and palettes, 7 MB) by nothing either; the
  README calls their impostor renderer "previous". About 18 MB of the 108 MB
  look like sources for `scripts/prepare_robot_atlases.py`. Packaging only
  what the catalog's meshes and materials reach (and what the program names)
  would drop them; check with the game's author first.
- **Mesh JSON is half the bundle on disk** but deflates 5.6×, and Steam
  compresses depot chunks, so it costs about 9 MB to download and 52 MB of
  disk. pen-render now also reads a binary mesh form, which
  `pen_render::pack_mesh` makes from the JSON and the renderer finds under
  the same key. For gdev's 73 meshes it is 32.7 MB instead of 59.1 MB, and
  it parses in 8.5 ms instead of 179 ms
  ([measurements](render-performance.md#meshes)). Packaging that writes the
  binary form in place of the JSON needs no exporter change. Quantized
  positions or a mesh compressor would shrink it further, but the format
  would then be lossy.
- **The packed textures do not compress further** (deflated per level
  already), so they are what downloads: 32.6 MB.

## What block compression would buy

Block-compressed texels are what the GPU samples directly: BC7 is 1 byte a
texel for RGBA (4× smaller than `rgba8unorm`), BC5 1 byte for two channels
(2× smaller than `rg8unorm`), BC4 half a byte for one (2× smaller than
`r8unorm`).

- **The 10 compressible `.gtex` files** (`color`, `cutout`, `normals`:
  the crater, salt, notice buildings, phone coaster, god hand): GPU memory
  39.4 → 10.6 MB. On disk, BC7 data deflates by perhaps 5–15%, so about
  25.3 → 9–10 MB, which is also the download saving (about 16 MB).
- **The 14 referenced PNGs**: GPU 26.7 → 6.7 MB (with the mip chain the
  renderer makes for them now); on disk 10.1 → about 6 MB, but lossy where
  the PNG was exact. The three 1254×1254 RGB PNGs (blueprint paper and
  details, salt surface, 10 MB together) are line art and fine noise, where
  BC7 artifacts would show first.
- **The `exact` and `raw` kinds cannot be compressed**: palette indices,
  palettes, two-channel oct normals, the font's distance field and packed
  positions must come back bit for bit. They are 104 MB of GPU memory (the
  seven 2560×1904 gnorp poses), and also look unused (above).

So for the shipping game: about 16–20 MB less to download and 50 MB less GPU
memory. Removing the unused files is worth as much on disk and needs no
renderer change.

## What it takes

### Format: extend `.gtex` rather than adopt KTX2

`.gtex` already has what block compression needs: every mip level worked out
offline (a compressed texture cannot be rendered into, so the renderers'
GPU mip generation in `mipmaps.rs` does not apply), stored
smallest first so a texture streams in, each level deflated on its own.
KTX2 also orders levels smallest first, and brings standard tools (`toktx`,
`basisu`) and Basis Universal (UASTC or ETC1S, transcoded at load to BC7 on
desktop, ASTC/ETC2 on mobile). gdev targets desktops, where every GPU has BC
(Vulkan `textureCompressionBC` on NVIDIA, AMD including the Steam Deck,
Intel; D3D12; Metal on Apple silicon), so transcoding buys nothing and costs
a transcoder in both renderers (C++ to wasm in the browser) and a zstd
decoder in the browser. The smaller change:

- `.gtex` version 3: a format byte beside channels and kind, `bc7`, `bc5` or
  `bc4` (or none), with each level's payload the blocks of that level
  (levels under 4×4 still one block), still deflated per level.
- `color` and `cutout` become `bc7-rgba-unorm-srgb`, 4-channel `data` and
  `normals` `bc7-rgba-unorm`, 2-channel `normals`/`data` `bc5-rg-unorm`,
  1-channel `data` `bc4-r-unorm`. `exact` and `raw` stay as they are.
- Base sizes must be multiples of 4 (WebGPU requires it for BC); every
  current compressible texture is.

### Encoding (gdev's `models/gtex.py`)

`gtex.py` computes the levels with numpy and deflates them. It would encode
each level after averaging it, with an encoder Python can call: the
`basisu` or Compressonator command line per level, or (simplest to keep
deterministic) a small Rust tool in this repository using the `intel_tex_2`
crate (ISPC BC7/BC5/BC4 kernels, prebuilt, no ISPC needed) that reads an
uncompressed `.gtex` and writes the compressed one. PNG textures would go
through the same writer instead of shipping as PNG. Encoding is slow at high
BC7 quality (seconds per 1024² texture), so it belongs in the asset export,
not in `pen build`.

### Shared renderer

`pen-render` owns parsing, upload and mip streaming for native and browser
hosts. A compressed `.gtex` format would carry block dimensions, bytes per
block and mip sizes. Adapter format features determine support; retain RGBA
fallbacks for adapters without BC7. Upload and copy row strides must use block
geometry rather than a fixed four bytes per pixel. Native/browser witnesses
should check decoding and reload behavior against independent images.

### Effort

Roughly: the `.gtex` v3 format and a Rust encoder tool, a day; both
the shared renderer's parsing and upload, a day with image witnesses; re-exporting textures and
checking how BC7 looks on the blueprint line art and the normal maps, a day.
Dropping the unused assets from packages is an hour once the game's author
confirms them.
