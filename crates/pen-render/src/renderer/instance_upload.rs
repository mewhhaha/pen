//! Retain GPU instance contents, uploading changed runs instead of the whole
//! scene. Comparisons use bits so signed zero is preserved too. The planner
//! owns reusable scratch; callers invalidate it when replacing the GPU buffer.
use std::ops::Range;

const LANES: usize = 20;
const MERGE_GAP: usize = LANES * 4;
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
        assert_eq!(current.len() % LANES, 0);
        let words: &[u32] = bytemuck::cast_slice(current);
        self.ranges.clear();
        if words.len() > self.previous.len() {
            self.ranges.push(0..words.len());
        } else {
            for (instance, values) in words.as_chunks::<LANES>().0.iter().enumerate() {
                let start = instance * LANES;
                let end = start + LANES;
                if values == &self.previous[start..end] {
                    continue;
                }
                if let Some(last) = self.ranges.last_mut()
                    && start - last.end <= MERGE_GAP
                {
                    last.end = end;
                } else {
                    self.ranges.push(start..end);
                }
            }
            let changed: usize = self.ranges.iter().map(|range| range.len()).sum();
            if self.ranges.len() > MAX_WRITES || changed > words.len() / 2 {
                self.ranges.clear();
                self.ranges.push(0..words.len());
            }
        }
        self.previous.clear();
        self.previous.extend_from_slice(words);
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
        assert_eq!(apply(&mut uploads, &mut gpu, &current), (1, current.len() * 4));
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
}
