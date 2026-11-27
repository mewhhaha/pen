//! Persistent GPU instance-slot table for retained scenes (`pen_runtime::RetainedDelta`).
//!
//! Retained draws whose `(mesh, material)` topology is unchanged keep fixed slots in the
//! prefix `[0 .. gpu_reserved)` of `State::instance_data` and `State::instance_buffer`.
//! On steady-state frames:
//! - Zero `place_cached` / asset lookups run for retained draws.
//! - Zero CPU culling or instance-buffer diffing runs for GPU-culled retained draws.
//! - Only the `K` indices in `RetainedHint::dirty_slots` patch `instance_data` and upload
//!   coalesced byte runs via `queue.write_buffer` (0 bytes when unchanged).
use std::ops::Range;

use crate::culling::Sphere;

use super::{Choice, Placement};

/// Maximum gap (in instances) between two dirty slots before coalescing into a single write.
const DIRTY_MERGE_GAP: u32 = 4;
/// Upper bound on separate `write_buffer` calls for dirty retained slots before collapsing to one range.
const MAX_DIRTY_WRITES: usize = 32;

/// Cached descriptor for one instanced batch formed exclusively from retained draws.
pub(super) struct RetainedBatchSpec {
    pub mesh: usize,
    pub material: u32,
    pub pipeline: wgpu::RenderPipeline,
    pub bounds: Option<Sphere>,
    pub casts: bool,
    pub count: u32,
    /// Persistent base slot in `State::instance_buffer` when GPU-culled, or `u32::MAX` when CPU-culled.
    pub persistent_first: u32,
}

/// Persistent renderer-side cache for the active retained scene topology.
#[derive(Default)]
pub(super) struct RetainedGpuTable {
    pub topology_epoch: u64,
    pub last_revision: u64,
    pub samples: u32,
    pub asset_revision: u64,
    pub gpu_culling_ready: bool,
    pub range: Range<usize>,
    pub choices: Vec<Choice>,
    pub placements: Vec<Placement>,
    pub batches: Vec<RetainedBatchSpec>,
    /// Persistent slot in `State::instance_buffer` for each retained draw `r in 0..range.len()`,
    /// or `u32::MAX` if CPU-culled / alone / waiting.
    pub gpu_slot: Vec<u32>,
    /// Relative retained indices `r` whose `gpu_slot[r] == u32::MAX` (need CPU culling or solo draw).
    pub non_gpu_indices: Vec<u32>,
    /// Number of persistent GPU-culled instance slots actively populated at `[0 .. gpu_used)`.
    pub gpu_used: u32,
    /// Aligned reserved prefix `[0 .. gpu_reserved)` ahead of dynamic/immediate instances.
    pub gpu_reserved: u32,
    /// Whether `[0 .. gpu_used)` has been uploaded to the current `State::instance_buffer`.
    pub uploaded: bool,
    /// Reusable scratch for sorting dirty GPU slot indices before coalescing uploads.
    pub dirty_gpu_slots: Vec<u32>,
    /// Reusable scratch for coalesced `(start_slot, end_slot)` upload runs.
    pub upload_runs: Vec<Range<u32>>,
}

impl RetainedGpuTable {
    pub(super) fn invalidate_all(&mut self) {
        self.topology_epoch = 0;
        self.uploaded = false;
        self.gpu_used = 0;
        self.gpu_reserved = 0;
    }

    pub(super) fn matches(
        &self,
        topology_epoch: u64,
        samples: u32,
        asset_revision: u64,
        gpu_culling_ready: bool,
        range: &Range<usize>,
    ) -> bool {
        self.topology_epoch != 0
            && self.topology_epoch == topology_epoch
            && self.samples == samples
            && self.asset_revision == asset_revision
            && self.gpu_culling_ready == gpu_culling_ready
            && self.range == *range
    }

    /// Coalesces `self.dirty_gpu_slots` into bounded contiguous instance-slot runs in `self.upload_runs`.
    pub(super) fn coalesce_dirty_runs(&mut self) -> &[Range<u32>] {
        self.upload_runs.clear();
        if self.dirty_gpu_slots.is_empty() {
            return &self.upload_runs;
        }
        if self.dirty_gpu_slots.len() > 1 {
            crate::sort::sort_by_u64_key(&mut self.dirty_gpu_slots, |&slot| u64::from(slot));
        }
        for &slot in &self.dirty_gpu_slots {
            if let Some(last) = self.upload_runs.last_mut() {
                if slot < last.end {
                    continue;
                }
                if slot - last.end <= DIRTY_MERGE_GAP {
                    last.end = slot + 1;
                    continue;
                }
            }
            self.upload_runs.push(slot..slot + 1);
        }
        if self.upload_runs.len() > MAX_DIRTY_WRITES {
            let start = self.upload_runs.first().unwrap().start;
            let end = self.upload_runs.last().unwrap().end;
            self.upload_runs.clear();
            self.upload_runs.push(start..end);
        }
        &self.upload_runs
    }
}
