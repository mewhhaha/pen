//! Binary deltas: a new file as runs copied from an old one and bytes of
//! its own, in the manner of rsync. The old file is indexed in blocks by a
//! rolling checksum; the new one is scanned a byte at a time, and where its
//! window's checksum names an old block whose bytes are the same, the match
//! is grown both ways and copied. What matches nothing is written out.
//!
//! The delta is a list of operations, each a tag byte and LEB128 numbers:
//! 0 copies `length` bytes from `offset` in the old file, 1 inserts the
//! `length` bytes that follow. patch.rs deflates it.
use anyhow::{Result, bail, ensure};

/// The block a match must cover at least. Small enough to find the unchanged
/// stretches between a mesh's edits, large enough that the index stays small.
const BLOCK: usize = 64;
/// The same, as the checksum weighs it.
const BLOCK_WEIGHT: u32 = 64;

const COPY: u8 = 0;
const INSERT: u8 = 1;

/// A checksum of a window that slides a byte at a time (Adler-32's sums,
/// unreduced).
#[derive(Clone, Copy)]
struct Rolling {
    a: u32,
    b: u32,
}

impl Rolling {
    fn of(window: &[u8]) -> Rolling {
        let mut rolling = Rolling { a: 0, b: 0 };
        for (index, &byte) in window.iter().enumerate() {
            rolling.a = rolling.a.wrapping_add(u32::from(byte));
            rolling.b =
                rolling.b.wrapping_add(u32::try_from(window.len() - index).unwrap().wrapping_mul(u32::from(byte)));
        }
        rolling
    }

    /// The window moved on by one byte: `out` left it and `into` joined it.
    fn roll(&mut self, out: u8, into: u8) {
        self.a = self.a.wrapping_sub(u32::from(out)).wrapping_add(u32::from(into));
        self.b = self.b.wrapping_sub(BLOCK_WEIGHT.wrapping_mul(u32::from(out))).wrapping_add(self.a);
    }

    fn value(self) -> u32 {
        (self.b << 16) ^ self.a
    }
}

fn number(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = value.to_le_bytes()[0] & 0x7f;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn read_number(bytes: &[u8], at: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let Some(&byte) = bytes.get(*at) else { bail!("delta ends inside a number") };
        *at += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    bail!("delta has a number too long")
}

/// The delta that makes `new` from `old`.
pub fn diff(old: &[u8], new: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    // Old blocks by checksum; the first of equal ones is kept.
    let mut index: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    if old.len() >= BLOCK {
        for start in (0..=old.len() - BLOCK).step_by(BLOCK) {
            index.entry(Rolling::of(&old[start..start + BLOCK]).value()).or_insert(start);
        }
    }
    let mut literal_start = 0;
    let mut at = 0;
    let mut rolling = (new.len() >= BLOCK).then(|| Rolling::of(&new[..BLOCK]));
    while let Some(sum) = rolling {
        let found = index.get(&sum.value()).copied().filter(|&start| old[start..start + BLOCK] == new[at..at + BLOCK]);
        if let Some(start) = found {
            // Grown back into the pending literal bytes, and forward.
            let mut back = 0;
            while at - back > literal_start && start > back && old[start - back - 1] == new[at - back - 1] {
                back += 1;
            }
            let mut length = BLOCK + back;
            while at - back + length < new.len()
                && start - back + length < old.len()
                && old[start - back + length] == new[at - back + length]
            {
                length += 1;
            }
            insert(&mut out, &new[literal_start..at - back]);
            out.push(COPY);
            number(&mut out, (start - back) as u64);
            number(&mut out, length as u64);
            at = at - back + length;
            literal_start = at;
            rolling = (at + BLOCK <= new.len()).then(|| Rolling::of(&new[at..at + BLOCK]));
            continue;
        }
        if at + BLOCK >= new.len() {
            break;
        }
        let mut next = sum;
        next.roll(new[at], new[at + BLOCK]);
        rolling = Some(next);
        at += 1;
    }
    insert(&mut out, &new[literal_start..]);
    out
}

fn insert(out: &mut Vec<u8>, bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    out.push(INSERT);
    number(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// Makes the new file from `old` and the delta.
pub fn apply(old: &[u8], delta: &[u8]) -> Result<Vec<u8>> {
    let mut new = Vec::new();
    let mut at = 0;
    while at < delta.len() {
        let tag = delta[at];
        at += 1;
        match tag {
            COPY => {
                let offset = usize::try_from(read_number(delta, &mut at)?)?;
                let length = usize::try_from(read_number(delta, &mut at)?)?;
                let end = offset.checked_add(length).filter(|&end| end <= old.len());
                let Some(end) = end else { bail!("delta copies past the old file's end") };
                new.extend_from_slice(&old[offset..end]);
            }
            INSERT => {
                let length = usize::try_from(read_number(delta, &mut at)?)?;
                ensure!(delta.len() - at >= length, "delta ends inside its bytes");
                new.extend_from_slice(&delta[at..at + length]);
                at += length;
            }
            other => bail!("delta has an unknown operation {other}"),
        }
    }
    Ok(new)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic bytes that do not repeat.
    fn noise(length: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..length)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                state.to_be_bytes()[0]
            })
            .collect()
    }

    #[test]
    fn deltas_rebuild_the_new_file_from_little() {
        let old = noise(200_000, 1);
        // An insertion, a deletion, a changed byte and a new tail.
        let mut new = old[..50_000].to_vec();
        new.extend(noise(300, 2));
        new.extend_from_slice(&old[50_000..120_000]);
        new.extend_from_slice(&old[121_000..180_000]);
        new.push(old[180_000] ^ 0xff);
        new.extend_from_slice(&old[180_001..]);
        new.extend(noise(1000, 3));
        let delta = diff(&old, &new);
        assert_eq!(apply(&old, &delta).unwrap(), new);
        // The unchanged stretches are copied: the delta is about the new bytes.
        assert!(delta.len() < 1500 + 200, "{} bytes", delta.len());
        // Nothing alike: the delta is the new file, and a little.
        let other = noise(10_000, 4);
        let delta = diff(&old, &other);
        assert!(delta.len() <= other.len() + 8);
        assert_eq!(apply(&old, &delta).unwrap(), other);
        for (old, new) in [(&b""[..], &b"short"[..]), (b"short", b""), (b"", b""), (&old[..10], &old[..100])] {
            assert_eq!(apply(old, &diff(old, new)).unwrap(), new);
        }
        // A delta that reaches past the old file is refused.
        let mut bad = vec![COPY];
        number(&mut bad, 5);
        number(&mut bad, 10);
        assert!(apply(b"tiny", &bad).is_err());
        assert!(apply(b"tiny", &[7]).is_err());
    }
}
