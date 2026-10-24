# Texture compression (KTX2: BC7, ASTC, ETC2)

pen ships compressed textures as KTX2 files, one per format family a GPU
samples directly, and the renderer reads the family its device has. The
asset pipeline encodes them with `pen texture`; nothing is transcoded at
load time.

## Files

A material names `name.ktx2`. Beside it, `pen texture IMAGE` writes:

| File | Format | Sampled directly by |
| --- | --- | --- |
| `name.bc7.ktx2` | BC7 (`VK_FORMAT_BC7_SRGB_BLOCK`, or `_UNORM` for data) | desktop GPUs: NVIDIA, AMD (the Steam Deck), Intel, Apple silicon; WebGPU's `texture-compression-bc` |
| `name.astc.ktx2` | ASTC 4x4 LDR | Apple GPUs, most phones; `texture-compression-astc` |
| `name.etc2.ktx2` | ETC2 RGBA8 (EAC alpha) | OpenGL ES 3 class phones; `texture-compression-etc2` |
| `name.rgba.ktx2` | RGBA8, uncompressed | every GPU |

Each file is a standard KTX2 container (`pen-texture`'s `ktx2`): the
format's `VkFormat`, a data format descriptor, `KTXwriter`, and every mip level
down to 1 x 1 stored smallest first, each deflated on its own (supercompression
scheme 3, ZLIB). The renderer also reads uncompressed (scheme 0) files, so
`name.ktx2` itself may be any single KTX2 file in one of these formats, written
by any tool. A colour texture is sRGB; `pen texture --linear` writes UNORM
for data such as normal maps (the renderer's linear view reads either).

WebGPU requires a block-compressed texture's size to be a multiple of 4. An
image that is not emits only `name.rgba.ktx2`, which every GPU reads.
Re-encoding removes any old BC7, ASTC and ETC2 variants after writing the
fallback, so changing an image's dimensions cannot leave stale preferred files.

## Choosing at load

`pen_render::optional_features(adapter)` asks for
`TEXTURE_COMPRESSION_BC`, `_ASTC` and `_ETC2` where the adapter has them; both
hosts request them with the device. For a texture key ending in `.ktx2` the
loader tries, in order, the variants the device samples (BC7, then ASTC,
then ETC2), then `name.rgba.ktx2`, then `name.ktx2`, and reads the first one
there (`compressed.rs`). A file whose format the device cannot sample fails
to load, as any broken texture does: it draws as the checker and raises
`AssetFailed`.

Levels then stream in exactly as packed textures (`.gtex`) do: the smallest
first, the texture showing once its levels up to 128 texels across are in,
each finer level fading in. Uploads use the format's block geometry (rows of
4 x 4 blocks, 16 bytes each; a level smaller than a block is one block).
Hot reload of `name.bc7.ktx2` (or any variant) reloads `name.ktx2`.

## Encoding

`pen texture IMAGE [--out name.ktx2] [--linear] [--flat]` reads a PNG,
works out the mip chain (`pen-texture`'s `mips`: box filtered, odd sizes
weighted so nothing shifts, colour averaged in linear light weighted by
alpha) and encodes the supported families with pen-texture's own encoders, pure Rust:

- **BC7**: mode 6 for every block, mode 5 in its four rotations, and the
  best two of the 64 two-subset partitions in mode 1 (opaque) or 7 (alpha),
  each fitted along the principal axis with two least-squares refits and
  every p-bit tried.
- **ETC2 RGBA8**: EAC alpha (all 16 tables, two multipliers and bases);
  colour in individual and differential modes (both flips, all tables, never
  overflowing into another mode), planar mode for gradients, and T and H
  modes for blocks of two unrelated colours.
- **ASTC 4x4**: one-colour blocks as void-extent blocks (exact); otherwise
  one partition, RGBA endpoints with 8 weight levels and 192-level endpoints
  (trits), or for opaque blocks RGB endpoints with 16 weight levels, the
  closer kept. Endpoints are ordered so the decoder never applies blue
  contraction.

Every encoder is checked block by block against an independent decoder
(`texture2ddecoder`) in the tests. On a 1024 x 1024 test image of gradients,
hard shapes, noise and varying alpha, level 0 decodes at 47.6 dB PSNR (BC7),
42.6 dB (ETC2) and 44.5 dB (ASTC). Encoding it with all its levels into all
four files takes 3.7 s on four threads (BC7 1.8 s, ETC2 1.0 s, ASTC 0.24 s,
RGBA's deflate 0.3 s), measured on a busy machine. That belongs in the asset
pipeline: gdev's Justfile has `just ktx2 IMAGE`.

## What it buys gdev, and what is left

Measured on the bundle `pen build` packaged on 2026-10-11 (game/assets,
108.4 MB):

- **The 10 block-compressible `.gtex` files** (`color`, `cutout`,
  `normals`: the crater, salt, notice buildings, phone coaster, god hand):
  GPU memory 39.4 → 10.6 MB with BC7, and about 16 MB less to download.
- **The 14 PNGs materials name**: GPU 26.7 → 6.7 MB.
- **The `exact` and `raw` kinds cannot be compressed**: palette indices,
  palettes, two-channel oct normals, the font's distance field and packed
  positions must come back bit for bit (104 MB of GPU memory, mostly the
  gnorp poses, which look unused).
- **Unused files ship.** `pen build` copies all of `src/assets`: 29
  `gnorp_*.png` captures and 21 `gnorp_*.gtex` atlases are named by nothing
  (about 18 MB); check with the game's author first.

Left, each the game's or the owner's call:

- Re-exporting gdev's textures as KTX2 and pointing its materials at them.
  BC7 is lossy: the three 1254 x 1254 line-art PNGs (blueprint paper and
  details, salt surface) should be looked at first, and as 1254 is not a
  multiple of 4 they need resizing to compress at all. The golden images
  change.
- `models/gtex.py`'s packed `color`/`normals` textures could write KTX2
  through `pen texture` instead; `exact` and `raw` stay `.gtex`.
- BC5 for two-channel normals and BC4 for one-channel data (half of BC7's
  size) need their own encoders and family files.
- Basis Universal (one file transcoded at load) was not chosen: it needs a
  transcoder in both hosts and buys nothing on desktops, which all sample
  BC7.
