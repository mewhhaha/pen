// ASTC 4 x 4 LDR blocks (16 bytes for 4 x 4 texels), which Apple and most
// mobile GPUs sample directly; the same blocks serve the UNORM and sRGB
// formats. A block of one colour is a void-extent block, exact. Any other
// is one partition with a 4 x 4 grid of weights in one plane, laid out
// whichever of two ways decodes closer:
// - RGBA endpoints (colour endpoint mode 12) and 8 weight levels, which
//   leave room for endpoints of 192 levels;
// - for opaque blocks, RGB endpoints (mode 8) and 16 weight levels, with
//   endpoints of 192 levels still.
// Endpoints of 192 levels take ASTC's integer sequence encoding: a trit and
// 6 bits each, the trits of five values packed into 8 bits. The tables
// that takes are built at compile time by running the spec's decoding
// (`unpack_trits`, `unquantize`) on every input.
// A block's fields run from bit 0 up: block mode (11 bits), partitions less
// one (2), colour endpoint mode (4), then the endpoints; the weights run
// from bit 127 down, every bit reversed. A decoder swaps endpoints whose
// second RGB sums lower than the first's and blue-contracts them, so the
// encoder orders the endpoints the other way round.
use crate::encode::{self, Bits, Block};

/// Least squares refits of the endpoints to the weights they picked.
const REFITS: usize = 2;

/// One way to lay a block out.
#[derive(Clone, Copy)]
struct Layout {
    /// Colour endpoint mode: 12 for RGBA, 8 for RGB (alpha decodes 255).
    endpoint_mode: u32,
    channels: usize,
    /// Its block mode: a 4 x 4 grid of weights in one plane, and their range.
    block_mode: u32,
    weight_bits: u32,
    /// Each weight level, unquantized to 0..=64.
    weights: &'static [u8],
}

const RGBA: Layout =
    Layout { endpoint_mode: 12, channels: 4, block_mode: block_mode(7), weight_bits: 3, weights: &WEIGHTS_8 };

const RGB: Layout =
    Layout { endpoint_mode: 8, channels: 3, block_mode: block_mode(12), weight_bits: 4, weights: &WEIGHTS_16 };

const WEIGHTS_8: [u8; 8] = weight_levels();
const WEIGHTS_16: [u8; 16] = weight_levels();

/// Each endpoint value (a trit times 64 plus 6 bits) unquantized to 8 bits.
const UNQUANTIZED: [u8; 192] = unquantized();

/// Each 8-bit value's nearest endpoint value.
const NEAREST: [u8; 256] = nearest_values();

/// Each five trits' packing (indexed t0 + 3 t1 + 9 t2 + 27 t3 + 81 t4).
const TRIT_PACKING: [u8; 243] = trit_packing();

/// The block mode of a 4 x 4 weight grid in one plane with weights of the
/// spec's `range` (R in its low 3 bits, H above).
const fn block_mode(range: u32) -> u32 {
    let (r0, r1, r2, high) = (range & 1, range >> 1 & 1, range >> 2 & 1, range >> 3 & 1);
    // Bits 2-3 are 0: width B + 4 with B = 0 (bits 7-8), height A + 2 with
    // A = 2 (bits 5-6).
    r1 | (r2 << 1) | (r0 << 4) | (2 << 5) | (high << 9)
}

/// The weight levels of `N` (a power of 2) bits-only levels, unquantized as
/// the spec does: the bits repeated to fill 6, then above 32 one more.
#[allow(clippy::cast_possible_truncation, reason = "levels of 0..=64")]
const fn weight_levels<const N: usize>() -> [u8; N] {
    let bits = N.trailing_zeros();
    let mut levels = [0; N];
    let mut level = 0;
    while level < N {
        let (mut value, mut filled) = (0, 0);
        while filled < 6 {
            value = (value << bits) | level;
            filled += bits;
        }
        value >>= filled - 6;
        levels[level] = (if value > 32 { value + 1 } else { value }) as u8;
        level += 1;
    }
    levels
}

/// An endpoint value of 192 levels (a trit and 6 bits, `fedcba`) to 8 bits,
/// as the spec unquantizes it.
const fn unquantize(value: u32) -> u32 {
    let (trit, bits) = (value >> 6, value & 63);
    let a = if bits & 1 == 1 { 0x1FF } else { 0 };
    // B is fedcb000f.
    let fedcb = bits >> 1;
    let b = (fedcb << 4) | (fedcb >> 4);
    let t = (trit * 5 + b) ^ a;
    (a & 0x80) | (t >> 2)
}

#[allow(clippy::cast_possible_truncation, reason = "unquantized values are 8 bits")]
const fn unquantized() -> [u8; 192] {
    let mut values = [0; 192];
    let mut value = 0;
    while value < 192 {
        values[value] = unquantize(value as u32) as u8;
        value += 1;
    }
    values
}

#[allow(clippy::cast_possible_truncation, reason = "endpoint values are below 192")]
const fn nearest_values() -> [u8; 256] {
    let mut nearest = [0; 256];
    let mut target = 0;
    while target < 256 {
        let mut best = 0;
        let mut value = 1;
        while value < 192 {
            if UNQUANTIZED[value].abs_diff(target as u8) < UNQUANTIZED[best].abs_diff(target as u8) {
                best = value;
            }
            value += 1;
        }
        nearest[target] = best as u8;
        target += 1;
    }
    nearest
}

/// The five trits 8 packed bits hold, decoded as the spec does.
const fn unpack_trits(packed: u32) -> [u32; 5] {
    let (c, t3, t4) = if packed >> 2 & 7 == 7 {
        (((packed >> 5 & 7) << 2) | (packed & 3), 2, 2)
    } else if packed >> 5 & 3 == 3 {
        (packed & 0x1F, packed >> 7 & 1, 2)
    } else {
        (packed & 0x1F, packed >> 5 & 3, packed >> 7 & 1)
    };
    let (t0, t1, t2) = if c & 3 == 3 {
        let (c3, c2) = (c >> 3 & 1, c >> 2 & 1);
        ((c3 << 1) | (c2 & !c3 & 1), c >> 4 & 1, 2)
    } else if c >> 2 & 3 == 3 {
        (c & 3, 2, 2)
    } else {
        let (c1, c0) = (c >> 1 & 1, c & 1);
        ((c1 << 1) | (c0 & !c1 & 1), c >> 2 & 3, c >> 4 & 1)
    };
    [t0, t1, t2, t3, t4]
}

/// The least packing that decodes to each five trits. Being the least, a
/// group's trailing zero trits leave the packing's trailing bits 0, which a
/// short last group relies on.
#[allow(clippy::cast_possible_truncation, reason = "packings are 8 bits")]
const fn trit_packing() -> [u8; 243] {
    let mut packing = [0; 243];
    let mut found = [false; 243];
    let mut packed = 0;
    while packed < 256 {
        let t = unpack_trits(packed);
        let index = (t[0] + 3 * t[1] + 9 * t[2] + 27 * t[3] + 81 * t[4]) as usize;
        if !found[index] {
            (found[index], packing[index]) = (true, packed as u8);
        }
        packed += 1;
    }
    let mut index = 0;
    while index < 243 {
        assert!(found[index], "every five trits have a packing");
        index += 1;
    }
    packing
}

/// What a channel decodes to between endpoints `low` and `high` (8 bits
/// each) at weight `weight` (0..=64): interpolated at 16 bits, read as UNORM8.
fn interpolate(low: i32, high: i32, weight: i32) -> i32 {
    let value = (low * 257 * (64 - weight) + high * 257 * weight + 32) >> 6;
    (value * 255 + 32768) >> 16
}

fn unquantized_value(value: u8) -> i32 {
    i32::from(UNQUANTIZED[usize::from(value)])
}

/// A block's endpoints (endpoint values for each channel) and weights.
#[derive(Clone, Copy)]
struct Fit {
    ends: [[u8; 4]; 2],
    weights: [u8; 16],
    /// The squared error over the layout's channels.
    error: u32,
}

/// `rgba` (`width` x `height` RGBA8) as ASTC 4 x 4 blocks, row-major;
/// blocks past the image's edge repeat its last column and row.
pub fn encode(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    encode::encode_blocks(rgba, width, height, encode_block, 1)
}

/// The ASTC 4 x 4 block of 16 texels, row-major.
pub fn encode_block(texels: &Block) -> [u8; 16] {
    if texels.iter().all(|texel| texel == &texels[0]) {
        return void_extent(texels[0]);
    }
    let (mut best, mut layout) = (fit(texels, RGBA), RGBA);
    if texels.iter().all(|texel| texel[3] == 255) {
        let opaque = fit(texels, RGB);
        if opaque.error < best.error {
            (best, layout) = (opaque, RGB);
        }
    }
    pack(&best, layout)
}

/// A block of one colour: a void-extent block (its extent left unset), the
/// colour in 16 bits a channel.
fn void_extent(color: [u8; 4]) -> [u8; 16] {
    let mut block = [0xFF; 16];
    // Void-extent block mode, LDR, the reserved bits set.
    block[..2].copy_from_slice(&[0xFC, 0xFD]);
    for (c, &value) in color.iter().enumerate() {
        block[8 + 2 * c..10 + 2 * c].copy_from_slice(&[value, value]);
    }
    block
}

/// The best endpoints and weights found for `texels` in `layout`, ordered
/// for the decoder.
fn fit(texels: &Block, layout: Layout) -> Fit {
    let channels = 0..layout.channels;
    let mut best = quantized(texels, layout, encode::line_endpoints(texels, channels.clone()));
    for _ in 0..REFITS {
        if best.error == 0 {
            break;
        }
        let along: [f32; 16] = std::array::from_fn(|i| f32::from(layout.weights[usize::from(best.weights[i])]) / 64.0);
        let Some(ends) = encode::least_squares(texels, &along, channels.clone()) else { break };
        let next = quantized(texels, layout, ends);
        if next.error >= best.error {
            break;
        }
        best = next;
    }
    // The decoder keeps endpoints in order when the second's RGB sums at
    // least the first's; otherwise swap them and flip the weights, which
    // decode the same mirrored.
    let sum = |end: [u8; 4]| (0..3).map(|c| unquantized_value(end[c])).sum::<i32>();
    if sum(best.ends[1]) < sum(best.ends[0]) {
        best.ends.swap(0, 1);
        let top = u8::try_from(layout.weights.len() - 1).unwrap_or(0);
        for weight in &mut best.weights {
            *weight = top - *weight;
        }
    }
    best
}

/// Endpoint values nearest `ends`, and `texels`' weights between them.
fn quantized(texels: &Block, layout: Layout, ends: [[f32; 4]; 2]) -> Fit {
    let mut codes = [[0; 4]; 2];
    for (code, end) in codes.iter_mut().zip(ends) {
        for c in 0..layout.channels {
            code[c] = NEAREST[usize::from(encode::to_u8(end[c]))];
        }
    }
    let palette: [[i32; 4]; 16] = std::array::from_fn(|level| {
        let weight = layout.weights.get(level).map_or(0, |&weight| i32::from(weight));
        std::array::from_fn(|c| interpolate(unquantized_value(codes[0][c]), unquantized_value(codes[1][c]), weight))
    });
    let (weights, error) = encode::nearest(texels, 0..layout.channels, &palette[..layout.weights.len()]);
    Fit { ends: codes, weights, error }
}

/// Writes endpoint `values` (each a trit times 64 plus 6 bits) as ASTC's
/// integer sequence: in groups of five, each value's 6 bits followed by its
/// share of the group's packed trits. A short last group writes only its
/// values' shares; the rest of its packing is 0.
fn put_trits(bits: &mut Bits, values: &[u8]) {
    // The packed bits after each value: 1-0, 3-2, 4, 6-5, 7.
    const SHARES: [(u32, u32); 5] = [(0, 2), (2, 2), (4, 1), (5, 2), (7, 1)];
    for group in values.chunks(5) {
        let index = group.iter().rev().fold(0, |index, &value| index * 3 + usize::from(value >> 6));
        let packed = u32::from(TRIT_PACKING[index]);
        for (&value, (shift, count)) in group.iter().zip(SHARES) {
            bits.put(value & 63, 6);
            bits.put(packed >> shift & ((1 << count) - 1), count);
        }
    }
}

/// The block's 16 bytes.
fn pack(fit: &Fit, layout: Layout) -> [u8; 16] {
    let mut bits = Bits::default();
    bits.put(layout.block_mode, 11);
    // One partition.
    bits.put(0u32, 2);
    bits.put(layout.endpoint_mode, 4);
    // Red, green, blue (and alpha), each low then high.
    let values: [u8; 8] = std::array::from_fn(|i| fit.ends[i % 2][i / 2]);
    put_trits(&mut bits, &values[..2 * layout.channels]);
    let mut weights = Bits::default();
    for &weight in &fit.weights {
        weights.put(weight, layout.weight_bits);
    }
    debug_assert!(bits.len() + weights.len() <= 128);
    (bits.value() | weights.value().reverse_bits()).to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::tests::{SOLID_COLORS, max_difference, psnr, rgba_of, small_image, test_image};

    fn decode(blocks: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut pixels = vec![0; width * height];
        texture2ddecoder::decode_astc(blocks, width, height, 4, 4, &mut pixels).unwrap();
        rgba_of(&pixels)
    }

    fn squared_error(texels: &Block, decoded: &[u8]) -> u32 {
        texels.as_flattened().iter().zip(decoded).map(|(&a, &b)| u32::from(a.abs_diff(b)).pow(2)).sum()
    }

    #[test]
    fn decodes_close_to_the_image() {
        let image = test_image(128, 128);
        let decoded = decode(&encode(&image, 128, 128), 128, 128);
        let quality = psnr(&image, &decoded);
        assert!(quality >= 39.0, "{quality} dB");
    }

    #[test]
    fn both_layouts_decode_as_they_reckon() {
        let image = test_image(16, 16);
        for (column, row) in (0..4).flat_map(|row| (0..4).map(move |column| (column, row))) {
            let texels = encode::block(&image, 16, 16, column, row);
            let opaque = texels.map(|[r, g, b, _]| [r, g, b, 255]);
            for (texels, layout) in [(texels, RGBA), (opaque, RGBA), (opaque, RGB)] {
                let found = fit(&texels, layout);
                assert_eq!(squared_error(&texels, &decode(&pack(&found, layout), 4, 4)), found.error);
            }
        }
    }

    #[test]
    fn solid_colours_are_exact() {
        for color in SOLID_COLORS {
            assert_eq!(decode(&encode_block(&[color; 16]), 4, 4), color.repeat(16));
        }
    }

    #[test]
    fn endpoint_values_unquantize_as_the_decoder_does() {
        for value in 0..192u8 {
            let fit = Fit { ends: [[value; 4]; 2], weights: [0; 16], error: 0 };
            let expected = UNQUANTIZED[usize::from(value)];
            for layout in [RGBA, RGB] {
                let decoded = decode(&pack(&fit, layout), 4, 4);
                assert!(decoded.chunks(4).all(|texel| texel[..3] == [expected; 3]), "{value} to {expected}");
            }
        }
        // Every 8-bit value has an endpoint value within a level or so.
        assert!((0..=255u8).all(|target| UNQUANTIZED[usize::from(NEAREST[usize::from(target)])].abs_diff(target) <= 1));
    }

    #[test]
    fn trits_pack_and_unpack() {
        for index in 0..243 {
            let mut rest = index;
            let trits: [u32; 5] = std::array::from_fn(|_| {
                let trit = rest % 3;
                rest /= 3;
                trit
            });
            let packed = TRIT_PACKING[index as usize];
            assert_eq!(unpack_trits(u32::from(packed)), trits);
            // Trailing zero trits leave the bits a short group omits 0.
            let used = trits.iter().rposition(|&trit| trit != 0).map_or(0, |last| last + 1);
            let kept = [0, 2, 4, 5, 7, 8][used];
            assert_eq!(u32::from(packed) >> kept, 0, "{trits:?}");
        }
    }

    #[test]
    fn weights_mirror() {
        for weights in [&WEIGHTS_8[..], &WEIGHTS_16[..]] {
            let top = weights.len() - 1;
            assert_eq!((weights[0], weights[top]), (0, 64));
            assert!((0..=top).all(|level| weights[level] + weights[top - level] == 64));
        }
        assert_eq!(WEIGHTS_8, [0, 9, 18, 27, 37, 46, 55, 64]);
    }

    #[test]
    fn dark_to_bright_and_back_keep_their_order() {
        // Eight steps along a line, which 8 weight levels hold, run either
        // way: the encoder must order the endpoints so the decoder keeps them.
        for flip in [false, true] {
            let texels: Block = std::array::from_fn(|texel| {
                let step = u8::try_from(if flip { 7 - texel % 8 } else { texel % 8 }).unwrap();
                [step * 35, step * 30, 250 - step * 20, 255 - step * 10]
            });
            let block = encode_block(&texels);
            assert!(max_difference(texels.as_flattened(), &decode(&block, 4, 4)) <= 4, "{flip}");
        }
    }

    #[test]
    fn edge_blocks_repeat_the_last_texels() {
        let image = small_image();
        let blocks = encode(&image, 6, 6);
        assert_eq!(blocks.len(), 4 * 16);
        let quality = psnr(&image, &decode(&blocks, 6, 6));
        assert!(quality >= 38.0, "{quality} dB");
    }
}
