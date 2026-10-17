//! A guest's linear memory kept as 64 KiB pages, for time travel without
//! the game's own snapshots (docs/instant-rewind.md). An image shares every
//! page that did not change since the image before it, so keeping one a
//! frame costs the pages the frame wrote; restoring one writes back only the
//! pages that differ from the memory as it is.
//!
//! An image alone restores a guest exactly only where its memory (and the
//! globals beside it) is all its state: an instance between calls whose
//! mutable globals the host can read. A long-lived `main(host)` suspended at
//! its host call also keeps state in its Wasm stack, out of any host's
//! reach, so images do not rewind today's games; the host side is ready
//! for a guest ABI that allows it.
use std::sync::Arc;

/// A Wasm page.
pub const PAGE: usize = 65536;

#[derive(Clone, Debug, Default)]
pub struct MemoryImage {
    pages: Vec<Arc<[u8]>>,
}

/// What capturing an image did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Captured {
    /// Pages it had to copy: new, or changed since the image before.
    pub copied: usize,
    /// Pages it shares with the image before.
    pub shared: usize,
}

impl MemoryImage {
    /// An image of `memory` (whole pages), sharing the pages `previous`
    /// has unchanged.
    pub fn capture(memory: &[u8], previous: Option<&MemoryImage>) -> (MemoryImage, Captured) {
        let mut captured = Captured::default();
        let pages = memory
            .chunks(PAGE)
            .enumerate()
            .map(|(index, page)| match previous.and_then(|previous| previous.pages.get(index)) {
                Some(kept) if **kept == *page => {
                    captured.shared += 1;
                    kept.clone()
                }
                _ => {
                    captured.copied += 1;
                    Arc::from(page)
                }
            })
            .collect();
        (MemoryImage { pages }, captured)
    }

    /// Its size in bytes.
    pub fn len(&self) -> usize {
        self.pages.iter().map(|page| page.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Bytes it holds that `other` does not share.
    pub fn unshared(&self, other: &MemoryImage) -> usize {
        let shares =
            |index: usize, page: &Arc<[u8]>| other.pages.get(index).is_some_and(|kept| Arc::ptr_eq(kept, page));
        self.pages.iter().enumerate().filter(|(index, page)| !shares(*index, page)).map(|(_, page)| page.len()).sum()
    }

    /// Writes the image over `memory`, which must be at least as large
    /// (Wasm memory does not shrink): pages that differ are copied, and
    /// memory past the image is zeroed. Returns the pages written.
    pub fn restore(&self, memory: &mut [u8]) -> usize {
        assert!(memory.len() >= self.len(), "memory smaller than the image it restores");
        let mut written = 0;
        let (imaged, rest) = memory.split_at_mut(self.len());
        for (page, kept) in imaged.chunks_mut(PAGE).zip(&self.pages) {
            if *page != **kept {
                page.copy_from_slice(kept);
                written += 1;
            }
        }
        for page in rest.chunks_mut(PAGE) {
            if page.iter().any(|byte| *byte != 0) {
                page.fill(0);
                written += 1;
            }
        }
        written
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn images_share_unchanged_pages_and_restore_what_differs() {
        let mut memory = vec![0u8; 4 * PAGE];
        memory[5] = 1;
        let (first, captured) = MemoryImage::capture(&memory, None);
        assert_eq!(captured, Captured { copied: 4, shared: 0 });
        memory[PAGE * 2 + 9] = 7;
        let (second, captured) = MemoryImage::capture(&memory, Some(&first));
        assert_eq!(captured, Captured { copied: 1, shared: 3 }, "one page written");
        assert_eq!(second.unshared(&first), PAGE);
        assert_eq!(second.len(), 4 * PAGE);
        // The memory grows and changes; the first image comes back.
        memory.resize(6 * PAGE, 0);
        memory[PAGE * 5] = 3;
        memory[0] = 2;
        assert_eq!(first.restore(&mut memory), 3, "pages 0 and 2 differ, and page 5 is past the image");
        assert_eq!(memory[..4 * PAGE], *vec_with(&[(5, 1)], 4 * PAGE));
        assert!(memory[4 * PAGE..].iter().all(|byte| *byte == 0));
        assert_eq!(first.restore(&mut memory), 0, "nothing differs now");
    }

    fn vec_with(bytes: &[(usize, u8)], length: usize) -> Vec<u8> {
        let mut memory = vec![0; length];
        for (at, byte) in bytes {
            memory[*at] = *byte;
        }
        memory
    }
}
