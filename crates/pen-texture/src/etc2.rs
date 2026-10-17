// ETC2 RGBA8 blocks (16 bytes for 4 x 4 texels), which OpenGL ES 3 class
// mobile GPUs sample directly: 8 bytes of EAC alpha, then 8 of ETC2 colour,
// each a 64-bit big-endian word. Texels number down each column first
// (texel x * 4 + y), and that number places a texel's index bits.
//
// Alpha: a base, a multiplier and one of 16 modifier tables, each texel
// taking one of the table's 8 modifiers; every table is tried with the
// multipliers and bases that span the block's alpha.
// Colour, whichever of these decodes closest:
// - two halves, side by side or stacked (the flip bit), each a base colour
//   and one of 8 intensity tables, each texel adding one of the table's 4
//   intensities to all three channels. The bases take 4 bits a channel each
//   (individual mode), or 5 bits and the second a 3-bit difference from the
//   first (differential mode);
// - planar mode: a gradient through three colours, fitted by least squares;
// - T and H modes: four colours from two 4-bit bases and a distance, any
//   texel taking any of them, for blocks of two colours apart in any shape.
// ETC2 tells T, H and planar blocks from differential ones by a difference
// that carries a channel outside 0..31, so the encoder never writes a
// differential block whose difference would: it pulls a base within reach.
use crate::encode::{self, Block};

/// Each intensity table's small and large intensity, added or taken away.
const INTENSITIES: [[i32; 2]; 8] = [[2, 8], [5, 17], [9, 29], [13, 42], [18, 60], [24, 80], [33, 106], [47, 183]];

/// The alpha modifier tables.
const ALPHA_MODIFIERS: [[i32; 8]; 16] = [
    [-3, -6, -9, -15, 2, 5, 8, 14],
    [-3, -7, -10, -13, 2, 6, 9, 12],
    [-2, -5, -8, -13, 1, 4, 7, 12],
    [-2, -4, -6, -13, 1, 3, 5, 12],
    [-3, -6, -8, -12, 2, 5, 7, 11],
    [-3, -7, -9, -11, 2, 6, 8, 10],
    [-4, -7, -8, -11, 3, 6, 7, 10],
    [-3, -5, -8, -11, 2, 4, 7, 10],
    [-2, -6, -8, -10, 1, 5, 7, 9],
    [-2, -5, -8, -10, 1, 4, 7, 9],
    [-2, -4, -8, -10, 1, 3, 7, 9],
    [-2, -5, -7, -10, 1, 4, 6, 9],
    [-3, -4, -7, -10, 2, 3, 6, 9],
    [-1, -2, -3, -10, 0, 1, 2, 9],
    [-4, -6, -8, -9, 3, 5, 7, 8],
    [-3, -5, -7, -9, 2, 4, 6, 8],
];

/// The alpha table with a modifier of 0, and that modifier: a block of one
/// alpha is its base, exactly.
const FLAT_ALPHA: (u8, u8) = (13, 4);

/// Tables whose intensities a half's base moves to centre, the best first.
const RECENTRED: usize = 2;

/// T and H mode's distances.
const DISTANCES: [i32; 8] = [3, 6, 11, 16, 23, 32, 41, 64];

/// The error above which a block tries T and H mode: below it the other
/// modes have done well enough.
const TWO_COLORS_ABOVE: u32 = 192;

/// Times T and H mode move their bases to centre the colours texels took.
const REFINES: usize = 2;

/// Each flip's two halves, row-major texel numbers: side by side (columns
/// 0-1 and 2-3), then stacked (rows 0-1 and 2-3).
const HALVES: [[[usize; 8]; 2]; 2] = [
    [[0, 1, 4, 5, 8, 9, 12, 13], [2, 3, 6, 7, 10, 11, 14, 15]],
    [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
];

/// `rgba` (`width` x `height` RGBA8) as ETC2 RGBA8 blocks, row-major; blocks
/// past the image's edge repeat its last column and row.
pub fn encode(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    encode::encode_blocks(rgba, width, height, encode_block, 1)
}

/// The ETC2 RGBA8 block of 16 texels, row-major.
pub fn encode_block(texels: &Block) -> [u8; 16] {
    let mut block = [0; 16];
    block[..8].copy_from_slice(&alpha(texels).1.to_be_bytes());
    block[8..].copy_from_slice(&color(texels).1.to_be_bytes());
    block
}

/// A row-major texel's number in ETC's column-first order.
fn number(texel: usize) -> usize {
    texel % 4 * 4 + texel / 4
}

fn squared(value: i32) -> u32 {
    value.unsigned_abs().pow(2)
}

/// The alpha word (base, multiplier, table and each texel's modifier) and
/// its squared error.
fn alpha(texels: &Block) -> (u32, u64) {
    let alphas = texels.map(|texel| i32::from(texel[3]));
    let (low, high) = (alphas.iter().copied().min().unwrap_or(0), alphas.iter().copied().max().unwrap_or(0));
    if low == high {
        let (table, modifier) = FLAT_ALPHA;
        return (0, pack_alpha(texels[0][3], 1, table, &[modifier; 16]));
    }
    // (error, base, multiplier, table)
    let mut best = (u32::MAX, 0, 1, 0);
    for (table, modifiers) in (0..).zip(&ALPHA_MODIFIERS) {
        let (least, most) = (modifiers[3], modifiers[7]);
        // The multipliers that span the block's alpha with this table, and
        // the bases that centre it.
        let span = (high - low) / (most - least);
        for multiplier in [span, span + 1].map(|multiplier| multiplier.clamp(1, 15)) {
            let centre = (low + high - multiplier * (least + most)).div_euclid(2);
            for base in [centre, centre + 1].map(|base| base.clamp(0, 255)) {
                let error = alpha_error(&alphas, base, multiplier, modifiers, best.0);
                if error < best.0 {
                    best = (error, base, multiplier, table);
                }
            }
        }
    }
    let (error, base, multiplier, table) = best;
    let modifiers = &ALPHA_MODIFIERS[usize::from(table)];
    let selectors = alphas.map(|alpha| nearest_alpha(alpha, base, multiplier, modifiers).0);
    let byte = |value: i32| u8::try_from(value).unwrap_or(0);
    (error, pack_alpha(byte(base), byte(multiplier), table, &selectors))
}

/// The modifier nearest `alpha` and its squared error.
fn nearest_alpha(alpha: i32, base: i32, multiplier: i32, modifiers: &[i32; 8]) -> (u8, u32) {
    (0..)
        .zip(modifiers)
        .map(|(selector, modifier)| (selector, squared((base + multiplier * modifier).clamp(0, 255) - alpha)))
        .min_by_key(|&(_, error)| error)
        .unwrap_or((0, 0))
}

/// The squared error of the block's alphas with `base`, `multiplier` and
/// `modifiers`; once it reaches `limit` it stops counting.
fn alpha_error(alphas: &[i32; 16], base: i32, multiplier: i32, modifiers: &[i32; 8], limit: u32) -> u32 {
    let mut error = 0;
    for &alpha in alphas {
        error += nearest_alpha(alpha, base, multiplier, modifiers).1;
        if error >= limit {
            break;
        }
    }
    error
}

fn pack_alpha(base: u8, multiplier: u8, table: u8, selectors: &[u8; 16]) -> u64 {
    let mut word = u64::from(base) << 56 | u64::from(multiplier) << 52 | u64::from(table) << 48;
    for (texel, &selector) in selectors.iter().enumerate() {
        word |= u64::from(selector) << (45 - 3 * number(texel));
    }
    word
}

/// One half of the colour block: its base (stored bits per channel),
/// intensity table and each texel's intensity (0 and 1 add the small or
/// large one, 2 and 3 take it away), with the squared error they leave.
#[derive(Clone, Copy)]
struct Half {
    base: [u8; 3],
    table: u8,
    selectors: [u8; 8],
    error: u32,
}

/// What a base channel of `bits` bits decodes to: its bits repeated to 8.
fn expand(code: u8, bits: u32) -> i32 {
    let code = i32::from(code);
    (code << (8 - bits)) | (code >> (2 * bits - 8))
}

/// The `bits`-bit base channel nearest `value`.
fn quantize(value: f32, bits: u32) -> u8 {
    let top = (1u8 << bits) - 1;
    let guess = encode::to_u8(value * f32::from(top) / 255.0).min(top);
    let distance = |code: u8| (expand(code, bits) as f32 - value).abs();
    [guess.saturating_sub(1), guess, (guess + 1).min(top)]
        .into_iter()
        .min_by(|&a, &b| distance(a).total_cmp(&distance(b)))
        .unwrap_or(guess)
}

/// A half's texels with base colour `color` (decoded) and intensity `table`,
/// each texel taking its nearest intensity: their intensities and the
/// squared error.
fn evaluate(texels: &[[i32; 3]; 8], color: [i32; 3], table: usize) -> ([u8; 8], u32) {
    let [small, large] = INTENSITIES[table];
    let shifts = [small, large, -small, -large];
    let shades: [[i32; 4]; 3] = std::array::from_fn(|c| shifts.map(|shift| (color[c] + shift).clamp(0, 255)));
    nearest_shade(texels, &shades)
}

/// Each of `texels`' nearest of four `shades` (given channel by channel, so
/// the four distances compute side by side) and the squared error.
fn nearest_shade<const N: usize>(texels: &[[i32; 3]; N], shades: &[[i32; 4]; 3]) -> ([u8; N], u32) {
    let mut indices = [0; N];
    let mut error = 0;
    for (index, texel) in indices.iter_mut().zip(texels) {
        let mut distances = [0; 4];
        for c in 0..3 {
            for (distance, &shade) in distances.iter_mut().zip(&shades[c]) {
                let difference = shade - texel[c];
                *distance += difference * difference;
            }
        }
        let (best, distance) = (0..).zip(distances).min_by_key(|&(_, distance)| distance).unwrap_or((0, 0));
        *index = best;
        error += distance.unsigned_abs();
    }
    (indices, error)
}

/// A half's texels with base colour `color` (decoded) under every intensity
/// table: each table's intensities and squared error. While no intensity
/// carries a channel past 0..255, a texel `u` off the base errs by
/// |u|^2 + 3 m^2 - 2 m sum(u) at intensity `m`, so it takes an intensity on
/// the side of its sum(u), and the tables score cheaply.
fn tables_at(texels: &[[i32; 3]; 8], color: [i32; 3]) -> [([u8; 8], u32); 8] {
    let offsets = texels.map(|texel| {
        let offset: [i32; 3] = std::array::from_fn(|c| texel[c] - color[c]);
        (offset.iter().map(|value| value * value).sum::<i32>(), offset.iter().sum::<i32>())
    });
    std::array::from_fn(|table| {
        let [small, large] = INTENSITIES[table];
        if color.iter().any(|&channel| channel < large || channel + large > 255) {
            return evaluate(texels, color, table);
        }
        let mut selectors = [0; 8];
        let mut error = 0;
        for (selector, &(distance, sum)) in selectors.iter_mut().zip(&offsets) {
            let cost = |shift: i32| 3 * shift * shift - 2 * shift * sum.abs();
            let (pick, cost) = if cost(large) < cost(small) { (1, cost(large)) } else { (0, cost(small)) };
            *selector = pick + if sum < 0 { 2 } else { 0 };
            error += (distance + cost).unsigned_abs();
        }
        (selectors, error)
    })
}

/// The best base and table found for a half, bases of `bits` bits a channel
/// (4 or 5): from the texels' mean, then where the two best tables'
/// intensities would centre it, then a step either way along grey.
fn best_half(texels: &[[i32; 3]; 8], bits: u32) -> Half {
    let mean: [f32; 3] = std::array::from_fn(|c| texels.iter().map(|texel| texel[c] as f32).sum::<f32>() / 8.0);
    let top = (1u8 << bits) - 1;
    let mut best = Half { base: [0; 3], table: 0, selectors: [0; 8], error: u32::MAX };
    let start = mean.map(|value| quantize(value, bits));
    let tables = consider(texels, start, bits, &mut best);
    let mut tried = vec![start];
    let mut try_base = |base: [u8; 3], best: &mut Half| {
        if !tried.contains(&base) {
            tried.push(base);
            consider(texels, base, bits, best);
        }
    };
    let mut order: [usize; 8] = std::array::from_fn(|table| table);
    order.sort_by_key(|&table| tables[table].1);
    for &table in &order[..RECENTRED] {
        let [small, large] = INTENSITIES[table];
        let shifts = [small, large, -small, -large];
        let shift = tables[table].0.iter().map(|&selector| shifts[usize::from(selector)]).sum::<i32>() as f32 / 8.0;
        try_base(mean.map(|value| quantize(value - shift, bits)), &mut best);
    }
    let centre = best.base;
    for step in [-1, 1] {
        let near = centre.map(|code| code.checked_add_signed(step).filter(|&code| code <= top));
        if let [Some(r), Some(g), Some(b)] = near {
            try_base([r, g, b], &mut best);
        }
    }
    best
}

/// Scores every table at base `base`, keeping the best in `best`; returns
/// each table's intensities and error.
fn consider(texels: &[[i32; 3]; 8], base: [u8; 3], bits: u32, best: &mut Half) -> [([u8; 8], u32); 8] {
    let tables = tables_at(texels, base.map(|code| expand(code, bits)));
    for (table, &(selectors, error)) in (0..).zip(&tables) {
        if error < best.error {
            *best = Half { base, table, selectors, error };
        }
    }
    tables
}

/// The best table for a half with base `base` of `bits` bits a channel.
fn half_at(texels: &[[i32; 3]; 8], base: [u8; 3], bits: u32) -> Half {
    let tables = tables_at(texels, base.map(|code| expand(code, bits)));
    let (table, &(selectors, error)) =
        (0..).zip(&tables).min_by_key(|(_, (_, error))| *error).expect("8 intensity tables");
    Half { base, table, selectors, error }
}

/// The colour word and its squared error: the closest of planar mode, T
/// and H modes, and in both orientations differential mode or (when the
/// halves' bases are too far apart for it) individual mode and differential
/// with a base pulled within reach.
fn color(texels: &Block) -> (u32, u64) {
    let mut best = planar(texels);
    let mut keep = |(error, word): (u32, u64)| {
        if error < best.0 {
            best = (error, word);
        }
    };
    for (flip, members) in HALVES.iter().enumerate() {
        let [first, second]: [[[i32; 3]; 8]; 2] =
            members.map(|half| half.map(|texel| std::array::from_fn(|c| i32::from(texels[texel][c]))));
        let flip = flip == 1;
        let (a, b) = (best_half(&first, 5), best_half(&second, 5));
        if (0..3).all(|c| (-4..=3).contains(&(i32::from(b.base[c]) - i32::from(a.base[c])))) {
            keep((a.error + b.error, pack(flip, true, [a, b], members)));
            continue;
        }
        let near_a = half_at(&second, reach(b.base, a.base, -4, 3), 5);
        keep((a.error + near_a.error, pack(flip, true, [a, near_a], members)));
        let near_b = half_at(&first, reach(a.base, b.base, -3, 4), 5);
        keep((near_b.error + b.error, pack(flip, true, [near_b, b], members)));
        let (a, b) = (best_half(&first, 4), best_half(&second, 4));
        keep((a.error + b.error, pack(flip, false, [a, b], members)));
    }
    if best.0 > TWO_COLORS_ABOVE {
        let two = two_colors(texels);
        if two.0 < best.0 {
            best = two;
        }
    }
    best
}

/// `base` moved, channel by channel, to within `low..=high` of `other`, and
/// within 5 bits.
fn reach(base: [u8; 3], other: [u8; 3], low: i8, high: i8) -> [u8; 3] {
    std::array::from_fn(|c| {
        let (least, most) = (other[c].saturating_add_signed(low), other[c].saturating_add_signed(high).min(31));
        base[c].clamp(least, most)
    })
}

/// The colour word of two halves: individual mode (4-bit bases) or
/// differential (a 5-bit base and the second's 3-bit difference from it).
fn pack(flip: bool, differential: bool, halves: [Half; 2], members: &[[usize; 8]; 2]) -> u64 {
    let [a, b] = halves;
    let mut word = 0;
    for c in 0..3 {
        let at = 56 - 8 * c;
        word |= if differential {
            let difference = (b.base[c].wrapping_sub(a.base[c])) & 7;
            u64::from(a.base[c]) << (at + 3) | u64::from(difference) << at
        } else {
            u64::from(a.base[c]) << (at + 4) | u64::from(b.base[c]) << at
        };
    }
    word |= u64::from(a.table) << 37 | u64::from(b.table) << 34 | u64::from(differential) << 33 | u64::from(flip) << 32;
    for (half, texels) in halves.iter().zip(members) {
        for (&selector, &texel) in half.selectors.iter().zip(texels) {
            let at = number(texel);
            word |= u64::from(selector & 1) << at | u64::from(selector >> 1) << (16 + at);
        }
    }
    word
}

/// Planar mode: the colour at texel (0, 0) and where the gradient reaches
/// 4 texels right and 4 down, red and blue in 6 bits, green in 7. Each
/// channel's plane is fitted by least squares, then each colour steps while
/// a step lowers the error. The squared error and the colour word.
fn planar(texels: &Block) -> (u32, u64) {
    let mut colors = [[0; 3]; 3];
    let mut error = 0;
    for c in 0..3 {
        let bits = if c == 1 { 7 } else { 6 };
        let top = (1u8 << bits) - 1;
        let values = texels.map(|texel| i32::from(texel[c]));
        // Across a 4 x 4 block, x and y less their mean (1.5) square to 20.
        let mean = values.iter().sum::<i32>() as f32 / 16.0;
        let (mut across, mut down) = (0.0, 0.0);
        for (texel, &value) in values.iter().enumerate() {
            across += ((texel % 4) as f32 - 1.5) * value as f32 / 20.0;
            down += ((texel / 4) as f32 - 1.5) * value as f32 / 20.0;
        }
        let origin = mean - 1.5 * (across + down);
        let mut codes = [origin, origin + 4.0 * across, origin + 4.0 * down].map(|value| quantize(value, bits));
        let mut channel_error = planar_error(&values, codes.map(|code| expand(code, bits)));
        let mut stepped = true;
        while stepped {
            stepped = false;
            for i in 0..3 {
                for step in [-1, 1] {
                    let mut near = codes;
                    match near[i].checked_add_signed(step) {
                        Some(value) if value <= top => near[i] = value,
                        _ => continue,
                    }
                    let near_error = planar_error(&values, near.map(|code| expand(code, bits)));
                    if near_error < channel_error {
                        (codes, channel_error, stepped) = (near, near_error, true);
                    }
                }
            }
        }
        for (color, code) in colors.iter_mut().zip(codes) {
            color[c] = code;
        }
        error += channel_error;
    }
    (error, pack_planar(colors))
}

/// The squared error of a channel's `values` against the plane through
/// decoded `origin`, `horizontal` and `vertical`, as the decoder rounds it.
fn planar_error(values: &[i32; 16], [origin, horizontal, vertical]: [i32; 3]) -> u32 {
    (0..4)
        .flat_map(|y| (0..4).map(move |x| (x, y)))
        .zip(values)
        .map(|((x, y), &value)| {
            let decoded = ((x * (horizontal - origin) + y * (vertical - origin) + 4 * origin + 2) >> 2).clamp(0, 255);
            squared(decoded - value)
        })
        .sum()
}

/// The planar colour word.
fn pack_planar(colors: [[u8; 3]; 3]) -> u64 {
    let [[ro, go, bo], [rh, gh, bh], [rv, gv, bv]] = colors;
    let mut bytes = [
        (ro << 1) | (go >> 6),
        ((go & 0x3F) << 1) | (bo >> 5),
        ((bo >> 3 & 3) << 3) | (bo >> 1 & 3),
        ((bo & 1) << 7) | ((rh >> 1) << 2) | 2 | (rh & 1),
        (gh << 1) | (bh >> 5),
        ((bh & 0x1F) << 3) | (rv >> 3),
        ((rv & 7) << 5) | (gv >> 2),
        ((gv & 3) << 6) | bv,
    ];
    within_reach(&mut bytes[0]);
    within_reach(&mut bytes[1]);
    out_of_reach(&mut bytes[2]);
    u64::from_be_bytes(bytes)
}

// T, H and planar blocks keep their colours around the fields where a
// differential block keeps a channel's 5-bit base and 3-bit difference. Which
// channel's difference reaches outside 0..31 marks the mode (red: T, else
// green: H, else blue: planar), and the bits no colour uses see to that.

/// Sets the free top bit of a base and difference byte (bits 6-0 taken)
/// so the difference stays within reach.
fn within_reach(byte: &mut u8) {
    let difference = i32::from(*byte & 3) - i32::from(*byte & 4);
    if i32::from(*byte >> 3) + difference < 0 {
        *byte |= 0x80;
    }
}

/// Sets the free bits 7-5 and 2 of a base and difference byte (bits 4-3 and
/// 1-0 taken) so the difference reaches past 31 or below 0.
fn out_of_reach(byte: &mut u8) {
    let (high, low) = (*byte >> 3 & 3, *byte & 3);
    *byte |= if high + low >= 4 { 0xE0 } else { 0x04 };
}

/// A T or H mode block: four paint colours from two 4-bit base colours and
/// a distance, each texel taking any of them. T mode paints the first base
/// alone and the second plus, without and minus the distance; H mode each
/// base plus and minus it.
#[derive(Clone, Copy)]
struct Paint {
    t: bool,
    bases: [[u8; 3]; 2],
    distance: usize,
    /// Each texel's paint, row-major.
    indices: [u8; 16],
    error: u32,
}

/// The paints of a T or H block, channel by channel.
fn paints(t: bool, bases: [[u8; 3]; 2], distance: usize) -> [[i32; 4]; 3] {
    let [first, second] = bases.map(|base| base.map(|code| expand(code, 4)));
    let d = DISTANCES[distance];
    std::array::from_fn(|c| {
        let shades = if t {
            [first[c], second[c] + d, second[c], second[c] - d]
        } else {
            [first[c] + d, first[c] - d, second[c] + d, second[c] - d]
        };
        shades.map(|shade| shade.clamp(0, 255))
    })
}

/// The best distance for T or H mode with `bases`. In H mode the distance's
/// low bit is whether the first base is at least the second, so equal bases
/// take only odd distances.
fn paint_with(colors: &[[i32; 3]; 16], t: bool, bases: [[u8; 3]; 2]) -> Paint {
    (0..DISTANCES.len())
        .filter(|&distance| t || bases[0] != bases[1] || distance % 2 == 1)
        .map(|distance| {
            let (indices, error) = nearest_shade(colors, &paints(t, bases, distance));
            Paint { t, bases, distance, indices, error }
        })
        .min_by_key(|paint| paint.error)
        .expect("every base pair has an odd distance")
}

/// The bases the texels' paints centre: each the mean of its texels less
/// the distance each took.
fn recentred(colors: &[[i32; 3]; 16], paint: &Paint) -> [[u8; 3]; 2] {
    let d = DISTANCES[paint.distance];
    let mut sums = [[0; 3]; 2];
    let mut counts = [0; 2];
    for (color, &index) in colors.iter().zip(&paint.indices) {
        let (base, shift) = match (paint.t, index) {
            (true, 0) => (0, 0),
            (true, _) => (1, [0, d, 0, -d][usize::from(index)]),
            (false, _) => (usize::from(index >> 1), if index & 1 == 0 { d } else { -d }),
        };
        for c in 0..3 {
            sums[base][c] += color[c] - shift;
        }
        counts[base] += 1;
    }
    std::array::from_fn(|base| {
        if counts[base] == 0 {
            return paint.bases[base];
        }
        std::array::from_fn(|c| quantize(sums[base][c] as f32 / counts[base] as f32, 4))
    })
}

/// The texels split in two (a mask of the second part) along their
/// principal axis, where the parts' spread about their means is least.
fn split(colors: &[[i32; 3]; 16]) -> u16 {
    let mean: [f32; 3] = std::array::from_fn(|c| colors.iter().map(|color| color[c] as f32).sum::<f32>() / 16.0);
    let mut scatter = [[0.0; 4]; 4];
    for color in colors {
        for c in 0..3 {
            for d in 0..3 {
                scatter[c][d] += (color[c] as f32 - mean[c]) * (color[d] as f32 - mean[d]);
            }
        }
    }
    let (_, axis) = encode::principal_axis(&scatter, encode::POWER_STEPS);
    let along = colors.map(|color| (0..3).map(|c| (color[c] as f32 - mean[c]) * axis[c]).sum::<f32>());
    let mut order: [usize; 16] = std::array::from_fn(|texel| texel);
    order.sort_by(|&a, &b| along[a].total_cmp(&along[b]));
    let squares = |color: &[i32; 3]| color.iter().map(|value| value * value).sum::<i32>() as f32;
    let total: [i32; 3] = std::array::from_fn(|c| colors.iter().map(|color| color[c]).sum());
    let total_squares: f32 = colors.iter().map(squares).sum();
    let (mut sum, mut sum_squares, mut mask) = ([0; 3], 0.0, 0u16);
    let mut best = (f32::MAX, 0);
    for (count, &texel) in (1..16).zip(&order) {
        for c in 0..3 {
            sum[c] += colors[texel][c];
        }
        sum_squares += squares(&colors[texel]);
        mask |= 1 << texel;
        let rest: [i32; 3] = std::array::from_fn(|c| total[c] - sum[c]);
        let spread = sum_squares - squares(&sum) / count as f32 + (total_squares - sum_squares)
            - squares(&rest) / (16 - count) as f32;
        if spread < best.0 {
            best = (spread, mask);
        }
    }
    !best.1
}

/// T and H mode's best: the texels split in two, each part's mean a base
/// (in T mode either part may be the lone colour), then the bases moved to
/// centre the paints the texels took. The squared error and colour word.
fn two_colors(texels: &Block) -> (u32, u64) {
    let colors = texels.map(|texel| [texel[0], texel[1], texel[2]].map(i32::from));
    let second = split(&colors);
    let [a, b] = [false, true].map(|part| {
        let members: Vec<&[i32; 3]> =
            (0..16).filter(|&texel| (second >> texel & 1 == 1) == part).map(|texel| &colors[texel]).collect();
        std::array::from_fn(|c| {
            quantize(members.iter().map(|color| color[c] as f32).sum::<f32>() / members.len().max(1) as f32, 4)
        })
    });
    let mut best: Option<Paint> = None;
    for (t, bases) in [(true, [a, b]), (true, [b, a]), (false, [a, b])] {
        let mut found = paint_with(&colors, t, bases);
        for _ in 0..REFINES {
            let next = paint_with(&colors, t, recentred(&colors, &found));
            if next.error >= found.error {
                break;
            }
            found = next;
        }
        if best.is_none_or(|best| found.error < best.error) {
            best = Some(found);
        }
    }
    let best = best.expect("three tries");
    (best.error, pack_paint(best))
}

/// The colour word of a T or H block.
fn pack_paint(mut paint: Paint) -> u64 {
    let distance = u8::try_from(paint.distance).unwrap_or(0);
    let bytes = if paint.t {
        let [[r1, g1, b1], [r2, g2, b2]] = paint.bases;
        let mut bytes = [
            ((r1 >> 2) << 3) | (r1 & 3),
            (g1 << 4) | b1,
            (r2 << 4) | g2,
            (b2 << 4) | ((distance >> 1) << 2) | 2 | (distance & 1),
        ];
        out_of_reach(&mut bytes[0]);
        bytes
    } else {
        // The distance's low bit is whether the first base is at least the
        // second: order the bases to match, their paints swapping with them.
        if (distance & 1 == 1) != (paint.bases[0] >= paint.bases[1]) {
            paint.bases.swap(0, 1);
            for index in &mut paint.indices {
                *index ^= 2;
            }
        }
        let [[r1, g1, b1], [r2, g2, b2]] = paint.bases;
        let mut bytes = [
            (r1 << 3) | (g1 >> 1),
            ((g1 & 1) << 4) | ((b1 >> 3) << 3) | (b1 >> 1 & 3),
            ((b1 & 1) << 7) | (r2 << 3) | (g2 >> 1),
            ((g2 & 1) << 7) | (b2 << 3) | ((distance >> 2) << 2) | 2 | (distance >> 1 & 1),
        ];
        within_reach(&mut bytes[0]);
        out_of_reach(&mut bytes[1]);
        bytes
    };
    let mut word = u64::from(u32::from_be_bytes(bytes)) << 32;
    for (texel, &index) in paint.indices.iter().enumerate() {
        let at = number(texel);
        word |= u64::from(index & 1) << at | u64::from(index >> 1) << (16 + at);
    }
    word
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::tests::{SOLID_COLORS, max_difference, psnr, rgba_of, small_image, test_image};

    fn decode(blocks: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut pixels = vec![0; width * height];
        texture2ddecoder::decode_etc2_rgba8(blocks, width, height, &mut pixels).unwrap();
        rgba_of(&pixels)
    }

    /// Which mode a colour word decodes in, by its difference bit and which
    /// channel a differential reading carries out of 0..31.
    fn mode(word: u64) -> &'static str {
        if word >> 33 & 1 == 0 {
            return "individual";
        }
        let reach = |at: u32| {
            let base = i32::try_from(word >> (at + 3) & 31).unwrap();
            let difference = i32::try_from(word >> at & 3).unwrap() - i32::try_from(word >> at & 4).unwrap();
            (0..32).contains(&(base + difference))
        };
        match (reach(56), reach(48), reach(40)) {
            (false, _, _) => "T",
            (true, false, _) => "H",
            (true, true, false) => "planar",
            (true, true, true) => "differential",
        }
    }

    #[test]
    fn decodes_close_to_the_image() {
        let image = test_image(128, 128);
        let decoded = decode(&encode(&image, 128, 128), 128, 128);
        let quality = psnr(&image, &decoded);
        assert!(quality >= 38.0, "{quality} dB");
    }

    #[test]
    fn every_block_decodes_as_it_reckons() {
        // Each mode's layout, and the reach that tells the modes apart,
        // against the reference decoder: a block decodes with just the error
        // its encoder counted.
        let image = test_image(64, 64);
        let mut seen = std::collections::BTreeSet::new();
        for (column, row) in (0..16).flat_map(|row| (0..16).map(move |column| (column, row))) {
            let texels = encode::block(&image, 64, 64, column, row);
            let (alpha_error, _) = alpha(&texels);
            let (color_error, word) = color(&texels);
            seen.insert(mode(word));
            let decoded = decode(&encode_block(&texels), 4, 4);
            let mut errors = [0; 2];
            for (index, (&a, &b)) in texels.as_flattened().iter().zip(&decoded).enumerate() {
                errors[usize::from(index % 4 == 3)] += u32::from(a.abs_diff(b)).pow(2);
            }
            assert_eq!(errors, [color_error, alpha_error], "block {column}, {row} in {} mode", mode(word));
        }
        assert!(["T", "H", "planar", "differential"].iter().all(|mode| seen.contains(mode)), "{seen:?}");
    }

    #[test]
    fn two_colours_take_t_or_h_mode() {
        // A diagonal edge between two colours far apart.
        let texels: Block =
            std::array::from_fn(|texel| if texel % 4 > texel / 4 { [220, 30, 60, 255] } else { [120, 130, 250, 255] });
        let block = encode_block(&texels);
        let word = u64::from_be_bytes(block[8..].try_into().unwrap());
        assert!(["T", "H"].contains(&mode(word)), "{}", mode(word));
        assert!(max_difference(texels.as_flattened(), &decode(&block, 4, 4)) <= 8);
    }

    #[test]
    fn halves_far_apart_stay_in_their_mode() {
        // Dark beside light, each half four shades of a 4-bit colour: beyond
        // a differential's reach, exact in individual mode.
        let texels: Block = std::array::from_fn(|texel| {
            let shade = [-8, -2, 2, 8][texel / 4];
            let base = if texel % 4 < 2 { [17, 34, 34] } else { [204, 187, 170] };
            let [r, g, b] = base.map(|value: i32| u8::try_from(value + shade).unwrap());
            [r, g, b, 255]
        });
        let block = encode_block(&texels);
        assert_eq!(mode(u64::from_be_bytes(block[8..].try_into().unwrap())), "individual");
        assert_eq!(decode(&block, 4, 4), texels.as_flattened());
        // Near colours beyond a differential's reach in one channel only.
        let texels: Block =
            std::array::from_fn(|texel| if texel / 4 < 2 { [100, 100, 100, 255] } else { [160, 104, 98, 255] });
        let block = encode_block(&texels);
        assert!(max_difference(texels.as_flattened(), &decode(&block, 4, 4)) <= 8);
    }

    #[test]
    fn gradients_take_planar_mode() {
        let texels: Block = std::array::from_fn(|texel| {
            let (x, y) = (u8::try_from(texel % 4).unwrap(), u8::try_from(texel / 4).unwrap());
            [20 + x * 30 + y * 10, 200 - y * 40, 60 + x * 12, 255]
        });
        let block = encode_block(&texels);
        assert_eq!(mode(u64::from_be_bytes(block[8..].try_into().unwrap())), "planar");
        assert!(max_difference(texels.as_flattened(), &decode(&block, 4, 4)) <= 2);
    }

    #[test]
    fn solid_colours_round_trip_within_two_levels() {
        // A colour's channels share one intensity in every mode but planar,
        // whose red and blue keep 6 bits: within 2 levels is ETC2's best.
        for color in SOLID_COLORS {
            let block = encode_block(&[color; 16]);
            let decoded = decode(&block, 4, 4);
            assert!(max_difference(&color.repeat(16), &decoded) <= 2, "{color:?} became {:?}", &decoded[..4]);
            // Alpha of one value is exact.
            assert!(decoded.chunks(4).all(|texel| texel[3] == color[3]));
        }
    }

    #[test]
    fn alpha_spans_its_range() {
        // A cut-out: alpha 0 and 255 only.
        let texels: Block = std::array::from_fn(|texel| [9, 9, 9, if texel % 3 == 0 { 0 } else { 255 }]);
        let decoded = decode(&encode_block(&texels), 4, 4);
        assert!(max_difference(texels.as_flattened(), &decoded) <= 1);
        // A ramp.
        let texels: Block = std::array::from_fn(|texel| [9, 9, 9, u8::try_from(texel * 17).unwrap()]);
        let decoded = decode(&encode_block(&texels), 4, 4);
        assert!(max_difference(texels.as_flattened(), &decoded) <= 20);
    }

    #[test]
    fn edge_blocks_repeat_the_last_texels() {
        let image = small_image();
        let blocks = encode(&image, 6, 6);
        assert_eq!(blocks.len(), 4 * 16);
        let quality = psnr(&image, &decode(&blocks, 6, 6));
        assert!(quality >= 35.0, "{quality} dB");
    }
}
