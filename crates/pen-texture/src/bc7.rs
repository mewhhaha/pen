// BC7 blocks (BPTC): 16 bytes for 4 x 4 RGBA8 texels, which desktop GPUs
// sample directly. A block tries the modes worth trying for its texels and
// keeps whichever decodes closest:
// - mode 6: one subset, RGBA endpoints of 7 bits and a p-bit each, 16 index
//   levels: the all-rounder;
// - mode 5: colour and alpha on index sets of their own, for alpha that does
//   not follow the colour (or, rotated into alpha's place, one colour channel
//   that does not follow the other two);
// - mode 1 (opaque blocks) or mode 7 (with alpha): two subsets split by one of
//   the 64 partitions, for blocks of two colours apart, like a hard edge. A
//   quick estimate ranks the partitions and the likeliest few encode in full.
// A subset's endpoints come from its texels' principal axis, then least
// squares given the indices they pick, quantized with every choice of
// p-bits. Fields are written least significant bit first. Each subset's
// anchor texel (texel 0 for the first) drops its index's top bit, so a
// subset whose anchor needs that bit swaps its endpoints and flips its
// indices instead.
use std::ops::Range;

use crate::encode::{self, Bits, Block};

/// Index weights, out of 64, for 2-, 3- and 4-bit indices.
const WEIGHTS: [&[i32]; 3] =
    [&[0, 21, 43, 64], &[0, 9, 18, 27, 37, 46, 55, 64], &[0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64]];

/// The two-subset partitions: bit i set puts texel i in the second subset.
const PARTITIONS: [u16; 64] = [
    0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80, 0xC800, 0xFFEC, 0xFE80, 0xE800, 0xFFE8, 0xFF00,
    0xFFF0, 0xF000, 0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310, 0x3100, 0x8CCE, 0x088C, 0x3110, 0x6666, 0x366C,
    0x17E8, 0x0FF0, 0x718E, 0x399C, 0xAAAA, 0xF0F0, 0x5A5A, 0x33CC, 0x3C3C, 0x55AA, 0x9696, 0xA55A, 0x73CE, 0x13C8,
    0x324C, 0x3BDC, 0x6996, 0xC33C, 0x9966, 0x0660, 0x0272, 0x04E4, 0x4E40, 0x2720, 0xC936, 0x936C, 0x39C6, 0x639C,
    0x9336, 0x9CC6, 0x817E, 0xE718, 0xCCF0, 0x0FCC, 0x7744, 0xEE22,
];

/// Each two-subset partition's anchor in the second subset.
const ANCHORS: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
    15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
];

/// Least squares refits of a subset's endpoints to the indices they picked.
const REFITS: usize = 2;

/// Partitions a two-subset mode encodes in full: the likeliest by estimate.
const CANDIDATES: usize = 2;

/// Power iteration steps to a partition's estimate: a rough one will do.
const ESTIMATE_STEPS: usize = 1;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pbits {
    None,
    /// One p-bit for both endpoints of a subset.
    Shared,
    /// A p-bit for each endpoint.
    Each,
}

/// How a mode stores a subset: bits per endpoint channel (p-bit apart),
/// its p-bits, and bits per index.
#[derive(Clone, Copy)]
struct Precision {
    bits: u32,
    pbits: Pbits,
    index_bits: u32,
}

impl Precision {
    fn weights(self) -> &'static [i32] {
        WEIGHTS[(self.index_bits - 2) as usize]
    }

    /// Every choice of p-bits for the two endpoints.
    fn pbit_choices(self) -> &'static [[u8; 2]] {
        match self.pbits {
            Pbits::None => &[[0, 0]],
            Pbits::Shared => &[[0, 0], [1, 1]],
            Pbits::Each => &[[0, 0], [0, 1], [1, 0], [1, 1]],
        }
    }

    /// What stored `code` with p-bit `pbit` decodes to: its bits repeated
    /// to fill 8.
    fn decode(self, code: i32, pbit: u8) -> i32 {
        let (code, bits) = match self.pbits {
            Pbits::None => (code, self.bits),
            _ => ((code << 1) | i32::from(pbit), self.bits + 1),
        };
        (code << (8 - bits)) | (code >> (2 * bits - 8))
    }

    /// The stored value nearest `value` with p-bit `pbit`, and what it
    /// decodes to.
    fn quantize(self, value: f32, pbit: u8) -> (u8, i32) {
        let top = (1 << self.bits) - 1;
        let scaled = match self.pbits {
            Pbits::None => value * top as f32 / 255.0,
            _ => (value * (2 * top + 1) as f32 / 255.0 - f32::from(pbit)) / 2.0,
        };
        let guess = i32::from(encode::to_u8(scaled)).min(top);
        let distance = |code: i32| (self.decode(code, pbit) as f32 - value).abs();
        let code = (guess - 1..=guess + 1)
            .filter(|code| (0..=top).contains(code))
            .min_by(|&a, &b| distance(a).total_cmp(&distance(b)))
            .unwrap_or(guess);
        (u8::try_from(code).unwrap_or(0), self.decode(code, pbit))
    }
}

/// A subset's endpoints and its texels' indices.
#[derive(Clone, Copy)]
struct Fit {
    /// Each endpoint's stored channels, p-bit apart.
    codes: [[u8; 4]; 2],
    pbits: [u8; 2],
    /// Each texel's index, in the order the texels were fitted.
    indices: [u8; 16],
    /// The squared error over the fitted channels.
    error: u32,
}

/// The best endpoints and indices found for `texels` in `channels` at
/// `precision`.
fn fit(texels: &[[u8; 4]], channels: Range<usize>, precision: Precision) -> Fit {
    let mut best = quantized(texels, &channels, precision, encode::line_endpoints(texels, channels.clone()));
    let weights = precision.weights();
    for _ in 0..REFITS {
        if best.error == 0 {
            break;
        }
        let along: [f32; 16] = std::array::from_fn(|i| weights[usize::from(best.indices[i])] as f32 / 64.0);
        let Some(ends) = encode::least_squares(texels, &along[..texels.len()], channels.clone()) else { break };
        let next = quantized(texels, &channels, precision, ends);
        if next.error >= best.error {
            break;
        }
        best = next;
    }
    best
}

/// Endpoints at `precision` nearest `ends` with each choice of p-bits, and
/// `texels`' indices between them: the choice that fits best.
fn quantized(texels: &[[u8; 4]], channels: &Range<usize>, precision: Precision, ends: [[f32; 4]; 2]) -> Fit {
    let weights = precision.weights();
    let mut best: Option<Fit> = None;
    for &pbits in precision.pbit_choices() {
        let (mut codes, mut colors) = ([[0; 4]; 2], [[0; 4]; 2]);
        for end in 0..2 {
            for c in channels.clone() {
                (codes[end][c], colors[end][c]) = precision.quantize(ends[end][c], pbits[end]);
            }
        }
        let palette: [[i32; 4]; 16] = std::array::from_fn(|k| {
            let weight = weights.get(k).copied().unwrap_or(0);
            std::array::from_fn(|c| (colors[0][c] * (64 - weight) + colors[1][c] * weight + 32) >> 6)
        });
        let (indices, error) = encode::nearest(texels, channels.clone(), &palette[..weights.len()]);
        if best.is_none_or(|best| error < best.error) {
            best = Some(Fit { codes, pbits, indices, error });
        }
    }
    best.expect("at least one choice of p-bits")
}

/// Swaps `fit`'s endpoints and flips its indices when the texel at `anchor`
/// has an index with its top bit set, which BC7 does not store.
fn fix_anchor(fit: &mut Fit, anchor: usize, index_bits: u32) {
    let top = (1u8 << index_bits) - 1;
    if fit.indices[anchor] > top >> 1 {
        fit.codes.swap(0, 1);
        fit.pbits.swap(0, 1);
        for index in &mut fit.indices {
            *index = top.saturating_sub(*index);
        }
    }
}

/// Writes each texel's index (`index`), the `anchors`' one bit short.
fn put_indices(bits: &mut Bits, index: impl Fn(usize) -> u8, index_bits: u32, anchors: &[usize]) {
    for texel in 0..16 {
        bits.put(index(texel), index_bits - u32::from(anchors.contains(&texel)));
    }
}

/// A block's bytes and the squared error they decode with.
struct Encoded {
    bytes: [u8; 16],
    error: u32,
}

fn keep_better(best: &mut Encoded, candidate: Encoded) {
    if candidate.error < best.error {
        *best = candidate;
    }
}

/// `rgba` (`width` x `height` RGBA8) as BC7 blocks, row-major; blocks past
/// the image's edge repeat its last column and row.
pub fn encode(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    encode::encode_blocks(rgba, width, height, encode_block, 1)
}

/// The BC7 block of 16 texels, row-major.
pub fn encode_block(texels: &Block) -> [u8; 16] {
    let mut best = mode_6(texels);
    for rotation in 0..4 {
        if best.error == 0 {
            break;
        }
        keep_better(&mut best, mode_5(texels, rotation));
    }
    let layout = if texels.iter().all(|texel| texel[3] == 255) { MODE_1 } else { MODE_7 };
    if best.error > 0 {
        for partition in likely_partitions(texels, layout) {
            keep_better(&mut best, two_subsets(texels, layout, partition));
        }
    }
    best.bytes
}

/// Mode 6: one subset, RGBA endpoints of 7 bits and a p-bit, 4-bit indices.
fn mode_6(texels: &Block) -> Encoded {
    let mut fit = fit(texels, 0..4, Precision { bits: 7, pbits: Pbits::Each, index_bits: 4 });
    fix_anchor(&mut fit, 0, 4);
    let mut bits = Bits::default();
    bits.put(1u32 << 6, 7);
    for c in 0..4 {
        for code in fit.codes {
            bits.put(code[c], 7);
        }
    }
    for pbit in fit.pbits {
        bits.put(pbit, 1);
    }
    put_indices(&mut bits, |texel| fit.indices[texel], 4, &[0]);
    Encoded { bytes: bits.value().to_le_bytes(), error: fit.error }
}

/// Mode 5: RGB endpoints of 7 bits and alpha of 8, each with 2-bit indices
/// of its own. `rotation` 1 to 3 swaps red, green or blue with alpha as the
/// block decodes, so the encoder swaps them first.
fn mode_5(texels: &Block, rotation: u8) -> Encoded {
    let mut rotated = *texels;
    if rotation > 0 {
        for texel in &mut rotated {
            texel.swap(usize::from(rotation) - 1, 3);
        }
    }
    let mut color = fit(&rotated, 0..3, Precision { bits: 7, pbits: Pbits::None, index_bits: 2 });
    let mut alpha = fit(&rotated, 3..4, Precision { bits: 8, pbits: Pbits::None, index_bits: 2 });
    fix_anchor(&mut color, 0, 2);
    fix_anchor(&mut alpha, 0, 2);
    let mut bits = Bits::default();
    bits.put(1u32 << 5, 6);
    bits.put(rotation, 2);
    for c in 0..3 {
        for code in color.codes {
            bits.put(code[c], 7);
        }
    }
    for code in alpha.codes {
        bits.put(code[3], 8);
    }
    put_indices(&mut bits, |texel| color.indices[texel], 2, &[0]);
    put_indices(&mut bits, |texel| alpha.indices[texel], 2, &[0]);
    Encoded { bytes: bits.value().to_le_bytes(), error: color.error + alpha.error }
}

/// A two-subset mode: its number, the channels its endpoints hold (alpha
/// decodes as 255 without one), and how it stores them.
#[derive(Clone, Copy)]
struct TwoSubsets {
    mode: u8,
    channels: usize,
    precision: Precision,
}

/// Mode 1: RGB endpoints of 6 bits and a p-bit each subset shares, 3-bit
/// indices.
const MODE_1: TwoSubsets =
    TwoSubsets { mode: 1, channels: 3, precision: Precision { bits: 6, pbits: Pbits::Shared, index_bits: 3 } };

/// Mode 7: RGBA endpoints of 5 bits and a p-bit each, 2-bit indices.
const MODE_7: TwoSubsets =
    TwoSubsets { mode: 7, channels: 4, precision: Precision { bits: 5, pbits: Pbits::Each, index_bits: 2 } };

/// `texels` as two subsets split by `partition`.
fn two_subsets(texels: &Block, layout: TwoSubsets, partition: usize) -> Encoded {
    let TwoSubsets { mode, channels, precision } = layout;
    let subset = |texel: usize| usize::from(PARTITIONS[partition] >> texel & 1 == 1);
    let anchors = [0, usize::from(ANCHORS[partition])];
    // Each subset's texels, and where each texel sits among its subset's.
    let mut members = [[[0; 4]; 16]; 2];
    let mut counts = [0; 2];
    let mut positions = [0; 16];
    for (texel, value) in texels.iter().enumerate() {
        let s = subset(texel);
        members[s][counts[s]] = *value;
        positions[texel] = counts[s];
        counts[s] += 1;
    }
    let fits = [0, 1].map(|s| {
        let mut found = fit(&members[s][..counts[s]], 0..channels, precision);
        fix_anchor(&mut found, positions[anchors[s]], precision.index_bits);
        found
    });
    let mut bits = Bits::default();
    bits.put(1u32 << mode, u32::from(mode) + 1);
    bits.put(u8::try_from(partition).unwrap_or(0), 6);
    for c in 0..channels {
        for found in &fits {
            for code in found.codes {
                bits.put(code[c], precision.bits);
            }
        }
    }
    for found in &fits {
        let pbits = if precision.pbits == Pbits::Shared { &found.pbits[..1] } else { &found.pbits[..] };
        for &pbit in pbits {
            bits.put(pbit, 1);
        }
    }
    put_indices(&mut bits, |texel| fits[subset(texel)].indices[positions[texel]], precision.index_bits, &anchors);
    let unstored: u32 =
        if channels < 4 { texels.iter().map(|texel| u32::from(255 - texel[3]).pow(2)).sum() } else { 0 };
    Encoded { bytes: bits.value().to_le_bytes(), error: fits[0].error + fits[1].error + unstored }
}

/// Sums over some texels from which their spread about a line follows.
#[derive(Clone, Copy, Default)]
struct Moments {
    count: f32,
    sum: [f32; 4],
    products: [[f32; 4]; 4],
}

impl Moments {
    fn of(texel: &[u8; 4]) -> Self {
        let value = texel.map(f32::from);
        Self { count: 1.0, sum: value, products: value.map(|a| value.map(|b| a * b)) }
    }

    fn add(&mut self, other: &Self) {
        self.count += other.count;
        for c in 0..4 {
            self.sum[c] += other.sum[c];
            for d in 0..4 {
                self.products[c][d] += other.products[c][d];
            }
        }
    }

    fn without(&self, other: &Self) -> Self {
        let mut rest = *self;
        rest.count -= other.count;
        for c in 0..4 {
            rest.sum[c] -= other.sum[c];
            for d in 0..4 {
                rest.products[c][d] -= other.products[c][d];
            }
        }
        rest
    }

    /// The texels' squared distance from their principal line in
    /// `channels`, plus what `steps` index steps along it leave (the texels
    /// spread evenly).
    fn spread(&self, channels: usize, steps: f32) -> f32 {
        if self.count < 1.0 {
            return 0.0;
        }
        let scatter: [[f32; 4]; 4] = std::array::from_fn(|c| {
            std::array::from_fn(|d| {
                if c < channels && d < channels {
                    self.products[c][d] - self.sum[c] * self.sum[d] / self.count
                } else {
                    0.0
                }
            })
        });
        let total: f32 = (0..channels).map(|c| scatter[c][c]).sum();
        let (along, _) = encode::principal_axis(&scatter, ESTIMATE_STEPS);
        total - along + along / (steps * steps)
    }
}

/// The partitions likeliest to fit `texels` well in `layout`: those whose
/// subsets each lie closest to a line, best first.
fn likely_partitions(texels: &Block, layout: TwoSubsets) -> [usize; CANDIDATES] {
    let each = texels.map(|texel| Moments::of(&texel));
    let mut all = Moments::default();
    for moments in &each {
        all.add(moments);
    }
    let steps = ((1 << layout.precision.index_bits) - 1) as f32;
    let mut best = [(f32::MAX, 0); CANDIDATES];
    for (partition, &mask) in PARTITIONS.iter().enumerate() {
        let mut second = Moments::default();
        for (texel, moments) in each.iter().enumerate() {
            if mask >> texel & 1 == 1 {
                second.add(moments);
            }
        }
        let first = all.without(&second);
        let estimate = first.spread(layout.channels, steps) + second.spread(layout.channels, steps);
        if estimate < best[CANDIDATES - 1].0 {
            best[CANDIDATES - 1] = (estimate, partition);
            best.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
    }
    best.map(|(_, partition)| partition)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::tests::{SOLID_COLORS, max_difference, psnr, rgba_of, small_image, test_image};

    fn decode(blocks: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut pixels = vec![0; width * height];
        texture2ddecoder::decode_bc7(blocks, width, height, &mut pixels).unwrap();
        rgba_of(&pixels)
    }

    /// The mode a block is in: the zero bits before its first one.
    fn mode(block: &[u8; 16]) -> u32 {
        block[0].trailing_zeros()
    }

    fn squared_error(texels: &Block, decoded: &[u8]) -> u32 {
        texels.as_flattened().iter().zip(decoded).map(|(&a, &b)| u32::from(a.abs_diff(b)).pow(2)).sum()
    }

    #[test]
    fn decodes_close_to_the_image() {
        let image = test_image(128, 128);
        let decoded = decode(&encode(&image, 128, 128), 128, 128);
        let quality = psnr(&image, &decoded);
        assert!(quality >= 42.0, "{quality} dB");
    }

    #[test]
    fn every_mode_decodes_as_it_reckons() {
        // Each mode's layout, anchors and p-bits, against the reference
        // decoder: a block decodes with just the error its encoder counted.
        let image = test_image(16, 16);
        for (column, row) in (0..4).flat_map(|row| (0..4).map(move |column| (column, row))) {
            let texels = encode::block(&image, 16, 16, column, row);
            let mut candidates = vec![mode_6(&texels)];
            candidates.extend((0..4).map(|rotation| mode_5(&texels, rotation)));
            for partition in (0..64).step_by(5) {
                candidates.push(two_subsets(&texels, MODE_1, partition));
                candidates.push(two_subsets(&texels, MODE_7, partition));
            }
            for candidate in candidates {
                let decoded = decode(&candidate.bytes, 4, 4);
                assert_eq!(squared_error(&texels, &decoded), candidate.error, "mode {}", mode(&candidate.bytes));
            }
        }
    }

    #[test]
    fn solid_colours_round_trip_within_a_level() {
        for color in SOLID_COLORS {
            let block = encode_block(&[color; 16]);
            let decoded = decode(&block, 4, 4);
            assert!(max_difference(&color.repeat(16), &decoded) <= 1, "{color:?} became {:?}", &decoded[..4]);
        }
    }

    #[test]
    fn edge_blocks_repeat_the_last_texels() {
        let image = small_image();
        let blocks = encode(&image, 6, 6);
        assert_eq!(blocks.len(), 4 * 16);
        let quality = psnr(&image, &decode(&blocks, 6, 6));
        assert!(quality >= 42.0, "{quality} dB");
    }

    #[test]
    fn two_lines_take_two_subsets() {
        // Black to red down the left half, green to blue down the right.
        let texels: Block = std::array::from_fn(|texel| {
            let (x, y) = (texel % 4, u8::try_from(texel / 4).unwrap());
            if x < 2 { [y * 80, 0, 0, 255] } else { [0, 255 - y * 80, y * 80, 255] }
        });
        let block = encode_block(&texels);
        assert_eq!(mode(&block), 1);
        assert!(squared_error(&texels, &decode(&block, 4, 4)) < mode_6(&texels).error / 4);
        // The same with alpha falling down the block takes mode 7.
        let faded: Block = std::array::from_fn(|texel| {
            let [r, g, b, _] = texels[texel];
            [r, g, b, 255 - u8::try_from(texel / 4).unwrap() * 60]
        });
        let block = encode_block(&faded);
        assert_eq!(mode(&block), 7);
        assert!(max_difference(faded.as_flattened(), &decode(&block, 4, 4)) <= 8);
    }

    #[test]
    fn alpha_apart_from_colour_takes_its_own_indices() {
        // Colour runs across the block, alpha down it.
        let texels: Block = std::array::from_fn(|texel| {
            let (x, y) = (u8::try_from(texel % 4).unwrap(), u8::try_from(texel / 4).unwrap());
            [40 + x * 60, 200 - x * 50, 90, 255 - y * 85]
        });
        let block = encode_block(&texels);
        assert_eq!(mode(&block), 5);
        assert!(max_difference(texels.as_flattened(), &decode(&block, 4, 4)) <= 3);
    }

    #[test]
    fn anchors_never_need_their_top_bit() {
        // Gradients both ways round, so some fits start from the far end.
        for flip in [false, true] {
            let texels: Block = std::array::from_fn(|texel| {
                let t = u8::try_from(if flip { 15 - texel } else { texel }).unwrap();
                [t * 17, 255 - t * 17, t * 8, 255 - t * 4]
            });
            let block = encode_block(&texels);
            assert!(max_difference(texels.as_flattened(), &decode(&block, 4, 4)) <= 3, "{flip}");
        }
    }
}
