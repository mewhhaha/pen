#![cfg_attr(not(feature = "render-3d"), allow(dead_code))]
//! Small-footprint sorting helpers that keep `wasm32` code size down by
//! routing hot sorts through a single `[(u64, u32)]` `ipnsort` instantiation
//! and using tiny in-place sorts for short or cold slices.

use std::cmp::Ordering;

/// The single `core::slice::sort_unstable` instantiation shared by hot paths.
#[inline(never)]
pub(crate) fn sort_u64_u32(slice: &mut [(u64, u32)]) {
    slice.sort_unstable();
}

/// Bijection from `f32` to `u32` preserving `f32::total_cmp` order.
#[inline]
pub(crate) fn f32_sortable(x: f32) -> u32 {
    let bits = x.to_bits();
    let mask = (bits.cast_signed() >> 31).cast_unsigned() | 0x8000_0000;
    bits ^ mask
}

/// Bijection from `f64` to `u64` preserving `f64::total_cmp` order.
#[inline]
pub(crate) fn f64_sortable(x: f64) -> u64 {
    let bits = x.to_bits();
    let mask = (bits.cast_signed() >> 63).cast_unsigned() | (1u64 << 63);
    bits ^ mask
}

/// Stably sorts `slice` by a `u64` key using the shared [`sort_u64_u32`].
pub(crate) fn sort_by_u64_key<T>(slice: &mut [T], mut key: impl FnMut(&T) -> u64) {
    if slice.len() <= 1 {
        return;
    }
    assert!(slice.len() < u32::MAX as usize, "slice length exceeds u32 index range");
    #[allow(clippy::cast_possible_truncation, reason = "asserted slice.len() < u32::MAX")]
    let mut keys: Vec<(u64, u32)> = slice.iter().enumerate().map(|(i, item)| (key(item), i as u32)).collect();
    sort_u64_u32(&mut keys);
    apply_permutation(slice, &mut keys);
}

fn apply_permutation<T>(slice: &mut [T], keys: &mut [(u64, u32)]) {
    const DONE: u32 = u32::MAX;
    for start in 0..slice.len() {
        let mut dst = start;
        let mut src = keys[dst].1;
        if src == DONE || src as usize == dst {
            keys[dst].1 = DONE;
            continue;
        }
        loop {
            keys[dst].1 = DONE;
            let next = keys[src as usize].1;
            if next == DONE || src as usize == start {
                break;
            }
            slice.swap(dst, src as usize);
            dst = src as usize;
            src = next;
        }
    }
}

/// Stable, tiny-code-size insertion sort for small slices.
pub(crate) fn insertion_sort_by<T>(slice: &mut [T], mut cmp: impl FnMut(&T, &T) -> Ordering) {
    for i in 1..slice.len() {
        let mut j = i;
        while j > 0 && cmp(&slice[j - 1], &slice[j]) == Ordering::Greater {
            slice.swap(j - 1, j);
            j -= 1;
        }
    }
}

/// Unstable, tiny-code-size Ciura shell sort for medium or cold slices.
pub(crate) fn shell_sort_by<T>(slice: &mut [T], mut cmp: impl FnMut(&T, &T) -> Ordering) {
    const GAPS: [usize; 8] = [701, 301, 132, 57, 23, 10, 4, 1];
    let n = slice.len();
    for &gap in &GAPS {
        if gap >= n {
            continue;
        }
        for i in gap..n {
            let mut j = i;
            while j >= gap && cmp(&slice[j - gap], &slice[j]) == Ordering::Greater {
                slice.swap(j - gap, j);
                j -= gap;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_sortable_matches_total_cmp() {
        let f32s = [
            f32::from_bits(0xffff_ffff),
            -f32::NAN,
            f32::NEG_INFINITY,
            -1e10,
            -1.0,
            -f32::MIN_POSITIVE,
            -f32::from_bits(1),
            -0.0,
            0.0,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            1.0,
            1e10,
            f32::INFINITY,
            f32::NAN,
            f32::from_bits(0x7fff_ffff),
        ];
        for &a in &f32s {
            for &b in &f32s {
                assert_eq!(f32_sortable(a).cmp(&f32_sortable(b)), a.total_cmp(&b));
            }
        }
        let f64s = [
            f64::from_bits(0xffff_ffff_ffff_ffff),
            -f64::NAN,
            f64::NEG_INFINITY,
            -1e100,
            -1.0,
            -f64::from_bits(1),
            -0.0,
            0.0,
            f64::from_bits(1),
            1.0,
            1e100,
            f64::INFINITY,
            f64::NAN,
            f64::from_bits(0x7fff_ffff_ffff_ffff),
        ];
        for &a in &f64s {
            for &b in &f64s {
                assert_eq!(f64_sortable(a).cmp(&f64_sortable(b)), a.total_cmp(&b));
            }
        }
    }

    #[test]
    fn sort_by_u64_key_is_stable_and_exact() {
        for n in [0usize, 1, 2, 3, 7, 32, 200] {
            let mut items: Vec<(u64, usize)> = (0..n).map(|i| (((i * 37 + 13) % 11) as u64, i)).collect();
            let mut expected = items.clone();
            expected.sort_by_key(|&(k, _)| k);
            sort_by_u64_key(&mut items, |&(k, _)| k);
            assert_eq!(items, expected);
        }
    }

    #[test]
    fn insertion_and_shell_sort_match_std() {
        let mut a: Vec<(u32, usize)> = (0..64).map(|i| (((i * 19 + 5) % 9) as u32, i)).collect();
        let mut expected = a.clone();
        expected.sort_by_key(|&(k, _)| k);
        insertion_sort_by(&mut a, |x, y| x.0.cmp(&y.0));
        assert_eq!(a, expected);

        let mut b: Vec<u32> = (0..500).map(|i| (i * 97 + 31) % 137).collect();
        let mut expected_b = b.clone();
        expected_b.sort_unstable();
        shell_sort_by(&mut b, Ord::cmp);
        assert_eq!(b, expected_b);
    }
}
