//! Retain GPU instance contents, uploading changed runs instead of the whole
//! scene. Comparisons use bits so signed zero is preserved too. The planner
//! owns reusable scratch; callers invalidate it when replacing the GPU buffer.
use std::ops::Range;

const LANES: usize = 20;
const MAX_WRITES: usize = 32;

#[derive(Default)]
pub(super) struct Uploads {
    previous: Vec<u32>,
    ranges: Vec<Range<usize>>,
}

impl Uploads {
    pub(super) fn invalidate(&mut self) {
        self.previous.clear();
    }

    pub(super) fn changed(&mut self, current: &[f32]) -> &[Range<usize>] {
        self.changed_strided(current, LANES)
    }

    pub(super) fn changed_strided(&mut self, current: &[f32], stride_words: usize) -> &[Range<usize>] {
        debug_assert!(stride_words > 0);
        assert_eq!(current.len() % stride_words, 0);
        let words: &[u32] = bytemuck::cast_slice(current);
        let merge_gap = stride_words * 4;
        self.ranges.clear();
        let common = words.len().min(self.previous.len());
        let mut start = 0;
        while start < common {
            let end = start + stride_words;
            if words[start..end] != self.previous[start..end] {
                if let Some(last) = self.ranges.last_mut()
                    && start - last.end <= merge_gap
                {
                    last.end = end;
                } else {
                    self.ranges.push(start..end);
                }
            }
            start = end;
        }
        if words.len() > common {
            if let Some(last) = self.ranges.last_mut()
                && common - last.end <= merge_gap
            {
                last.end = words.len();
            } else {
                self.ranges.push(common..words.len());
            }
        }
        let changed: usize = self.ranges.iter().map(|range| range.len()).sum();
        if self.ranges.len() > MAX_WRITES || (self.ranges.len() > 1 && changed > words.len() * 3 / 4) {
            self.ranges.clear();
            self.ranges.push(0..words.len());
        }
        if self.ranges.len() == 1 && self.ranges[0] == (0..words.len()) {
            self.previous.clear();
            self.previous.extend_from_slice(words);
        } else {
            if words.len() < self.previous.len() {
                self.previous.truncate(words.len());
            } else if words.len() > self.previous.len() {
                self.previous.resize(words.len(), 0);
            }
            for range in &self.ranges {
                self.previous[range.clone()].copy_from_slice(&words[range.clone()]);
            }
        }
        &self.ranges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(uploads: &mut Uploads, gpu: &mut Vec<u32>, current: &[f32]) -> (usize, usize) {
        let words: &[u32] = bytemuck::cast_slice(current);
        gpu.resize(words.len(), 0xdead_beef);
        let ranges = uploads.changed(current);
        let bytes = ranges.iter().map(|range| range.len() * 4).sum();
        for range in ranges {
            gpu[range.clone()].copy_from_slice(&words[range.clone()]);
        }
        assert_eq!(&gpu[..], words);
        (ranges.len(), bytes)
    }

    #[test]
    fn retained_contents_match_full_upload_through_edits_resize_and_reset() {
        let mut uploads = Uploads::default();
        let mut gpu = Vec::new();
        let mut current: Vec<f32> = (0..LANES * 100).map(|word| word as f32).collect();
        assert_eq!(apply(&mut uploads, &mut gpu, &current), (1, current.len() * 4));
        assert_eq!(apply(&mut uploads, &mut gpu, &current), (0, 0));
        current[LANES * 3] = -0.0;
        current[LANES * 90] += 1.0;
        assert_eq!(apply(&mut uploads, &mut gpu, &current), (2, LANES * 4 * 2));
        current[LANES * 3] = 0.0;
        assert_eq!(apply(&mut uploads, &mut gpu, &current), (1, LANES * 4));
        current.truncate(LANES * 30);
        assert_eq!(apply(&mut uploads, &mut gpu, &current), (0, 0));
        current.resize(LANES * 200, 7.0);
        // A single appended tail preserves the unchanged prefix instead of re-uploading it.
        assert_eq!(apply(&mut uploads, &mut gpu, &current), (1, LANES * 170 * 4));
        uploads.invalidate();
        gpu.fill(0xdead_beef);
        assert_eq!(apply(&mut uploads, &mut gpu, &current), (1, current.len() * 4));
    }

    #[test]
    fn fragmented_changes_use_a_bounded_number_of_writes() {
        let mut uploads = Uploads::default();
        let mut gpu = Vec::new();
        let mut current = vec![0.0; LANES * 400];
        apply(&mut uploads, &mut gpu, &current);
        for index in (0..400).step_by(10) {
            current[index * LANES] = 1.0;
        }
        assert_eq!(apply(&mut uploads, &mut gpu, &current), (1, current.len() * 4));
    }

    #[test]
    fn strided_uniform_uploads_skip_unchanged_slots() {
        const STRIDE: usize = 256;
        let mut uploads = Uploads::default();
        let mut current = vec![0.0f32; STRIDE * 8];
        for slot in 0..8 {
            current[slot * STRIDE] = slot as f32;
        }
        let first = uploads.changed_strided(&current, STRIDE);
        assert_eq!(first, &[0..STRIDE * 8]);
        assert!(uploads.changed_strided(&current, STRIDE).is_empty());
        current[5 * STRIDE + 17] = 42.0;
        let second = uploads.changed_strided(&current, STRIDE);
        assert_eq!(second, &[5 * STRIDE..6 * STRIDE]);
        assert!(uploads.changed_strided(&current, STRIDE).is_empty());
    }
}
