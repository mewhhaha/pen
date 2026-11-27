//! Transactional retained draws. Guest IDs are scoped to one program instance.
use anyhow::{Result, bail, ensure};
use pen_protocol::boundary::{MAX_DRAWS, MAX_MOTIONS};
use pen_protocol::{Draw, Matrix, Motion, RenderFrame};
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

/// Globally unique topology epoch counter so restores/reloads never alias a prior scene's epoch.
static NEXT_TOPOLOGY_EPOCH: AtomicU64 = AtomicU64::new(1);

fn next_topology_epoch() -> u64 {
    NEXT_TOPOLOGY_EPOCH.fetch_add(1, Ordering::Relaxed)
}

/// Compact delta metadata describing how the resolved `RenderFrame::draws` relates to
/// the runtime's persistent retained instance table.
#[derive(Clone, Debug)]
pub struct RetainedDelta<'a> {
    /// Nonzero globally-unique epoch bumped on reset, add, remove, reorder, or mesh/material swap.
    pub topology_epoch: u64,
    /// Monotonic revision bumped whenever any retained instance is modified.
    pub revision: u64,
    /// Slice `range` inside the resolved `RenderFrame::draws` occupied by retained draws.
    pub range: Range<usize>,
    /// Relative slot offsets (`0..range.len()`) modified in place on this frame when `!topology_changed`.
    pub dirty_slots: &'a [u32],
    /// True when the retained topology changed (reset, add, remove, reorder, or mesh/material swap).
    pub topology_changed: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct RetainedEntry {
    id: u32,
    order: u32,
}

#[derive(Clone, Default)]
pub(crate) struct Scene {
    started: bool,
    topology_epoch: u64,
    revision: u64,
    last_range: Range<usize>,
    last_topology_changed: bool,
    /// Relative slot indices (`0..ordered.len()`) modified in place on the last frame.
    dirty_slots: Vec<u32>,
    /// Reusable buffer for rollback of `dirty_slots`.
    dirty_scratch: Vec<u32>,
    /// Sorted by `(order, id)`, parallel with `ordered`.
    entries: Vec<RetainedEntry>,
    /// Draws in `(order, id)` order, ready to splice into `RenderFrame::draws`.
    ordered: Vec<Draw>,
    /// Maps retained `id` -> index in `entries` / `ordered`.
    pos_by_id: HashMap<u32, usize>,
    /// Reusable set for duplicate ID validation across removals and updates.
    touched: HashSet<u32>,
    /// Reusable scratch buffer for resolved `(draw_index, previous_transform)` motions.
    moved_buf: Vec<(u32, Matrix)>,
    /// Reusable buffer for in-place rollback records to avoid per-frame allocations.
    rollback_scratch: Vec<(usize, Draw)>,
}

#[derive(Clone)]
pub(crate) struct MetaSnapshot {
    started: bool,
    topology_epoch: u64,
    revision: u64,
    last_range: Range<usize>,
    last_topology_changed: bool,
    dirty_slots: Vec<u32>,
}

pub(crate) enum SceneRollback {
    None(MetaSnapshot),
    InPlace { meta: MetaSnapshot, changes: Vec<(usize, Draw)> },
    Full { meta: MetaSnapshot, entries: Vec<RetainedEntry>, ordered: Vec<Draw>, pos_by_id: HashMap<u32, usize> },
}

impl Scene {
    fn snapshot_meta(&mut self) -> MetaSnapshot {
        let mut dirty = std::mem::take(&mut self.dirty_scratch);
        dirty.clear();
        dirty.extend_from_slice(&self.dirty_slots);
        MetaSnapshot {
            started: self.started,
            topology_epoch: self.topology_epoch,
            revision: self.revision,
            last_range: self.last_range.clone(),
            last_topology_changed: self.last_topology_changed,
            dirty_slots: dirty,
        }
    }

    fn restore_meta(&mut self, meta: MetaSnapshot) {
        self.started = meta.started;
        self.topology_epoch = meta.topology_epoch;
        self.revision = meta.revision;
        self.last_range = meta.last_range;
        self.last_topology_changed = meta.last_topology_changed;
        let old = std::mem::replace(&mut self.dirty_slots, meta.dirty_slots);
        self.dirty_scratch = old;
    }

    /// Returns the delta metadata for the most recently applied frame, when retained draws are active.
    pub fn delta(&self) -> Option<RetainedDelta<'_>> {
        if !self.started || self.topology_epoch == 0 {
            return None;
        }
        Some(RetainedDelta {
            topology_epoch: self.topology_epoch,
            revision: self.revision,
            range: self.last_range.clone(),
            dirty_slots: &self.dirty_slots,
            topology_changed: self.last_topology_changed,
        })
    }

    /// Applies retained updates in place and resolves `frame`. On any error,
    /// `self` is left unmodified. On `Ok(undo)`, if downstream renderer
    /// validation later rejects `frame`, call [`Self::rollback`] with `undo`.
    pub fn apply(&mut self, frame: &mut RenderFrame) -> Result<SceneRollback> {
        if !frame.retained_enabled {
            ensure!(
                !frame.retained_reset && frame.instances.is_empty() && frame.instance_removals.is_empty(),
                "retained updates require retained_enabled"
            );
            let meta = self.snapshot_meta();
            if !self.started && self.entries.is_empty() {
                self.last_range = 0..0;
                self.dirty_slots.clear();
                self.last_topology_changed = false;
                return Ok(SceneRollback::None(meta));
            }
            let undo = SceneRollback::Full {
                meta,
                entries: std::mem::take(&mut self.entries),
                ordered: std::mem::take(&mut self.ordered),
                pos_by_id: std::mem::take(&mut self.pos_by_id),
            };
            self.started = false;
            self.topology_epoch = 0;
            self.revision = self.revision.wrapping_add(1);
            self.last_range = 0..0;
            self.dirty_slots.clear();
            self.last_topology_changed = true;
            return Ok(undo);
        }
        ensure!(self.started || frame.retained_reset, "first retained frame requires a reset");
        let after = frame.retained_after as usize;
        ensure!(after <= frame.draws.len(), "retained insertion exceeds immediate draws");

        let changed = frame.retained_reset || !frame.instances.is_empty() || !frame.instance_removals.is_empty();
        self.moved_buf.clear();

        let undo = if !changed {
            let meta = self.snapshot_meta();
            self.dirty_slots.clear();
            self.last_topology_changed = false;
            self.last_range = after..after + self.ordered.len();
            SceneRollback::None(meta)
        } else if !frame.retained_reset && frame.instance_removals.is_empty() {
            // Check whether every update targets an existing ID with unchanged
            // order so we can patch in place in O(K) with zero sorting or cloning.
            self.touched.clear();
            let mut in_place_ok = true;
            for value in &frame.instances {
                let id = value.id_low | (value.id_high << 16);
                ensure!(self.touched.insert(id), "duplicate retained instance identity {id}");
                if let Some(&pos) = self.pos_by_id.get(&id) {
                    if self.entries[pos].order != value.order {
                        in_place_ok = false;
                    }
                } else {
                    in_place_ok = false;
                }
            }
            if in_place_ok {
                let meta = self.snapshot_meta();
                let mut changes = std::mem::take(&mut self.rollback_scratch);
                changes.clear();
                changes.reserve(frame.instances.len());
                self.dirty_slots.clear();
                self.dirty_slots.reserve(frame.instances.len());
                let mut asset_changed = false;
                for value in &frame.instances {
                    let id = value.id_low | (value.id_high << 16);
                    let pos = self.pos_by_id[&id];
                    let prev_transform = self.ordered[pos].transform;
                    if prev_transform != value.transform {
                        let resolved = u32::try_from(after + pos)?;
                        self.moved_buf.push((resolved, prev_transform));
                    }
                    if self.ordered[pos].mesh != value.mesh || self.ordered[pos].material != value.material {
                        asset_changed = true;
                    }
                    let new_draw = Draw {
                        mesh: value.mesh,
                        material: value.material,
                        transform: value.transform,
                        parameters: value.parameters,
                        extra: None,
                    };
                    let old_draw = std::mem::replace(&mut self.ordered[pos], new_draw);
                    changes.push((pos, old_draw));
                    #[allow(clippy::cast_possible_truncation, reason = "pos < ordered.len() <= MAX_DRAWS")]
                    self.dirty_slots.push(pos as u32);
                }
                if self.moved_buf.len() > 1 {
                    crate::sort::sort_by_u64_key(&mut self.moved_buf, |&(draw, _)| u64::from(draw));
                }
                if !self.started || self.topology_epoch == 0 || asset_changed {
                    self.topology_epoch = next_topology_epoch();
                    self.last_topology_changed = true;
                } else {
                    self.last_topology_changed = false;
                }
                self.started = true;
                self.revision = self.revision.wrapping_add(1);
                self.last_range = after..after + self.ordered.len();
                SceneRollback::InPlace { meta, changes }
            } else {
                let meta = self.snapshot_meta();
                let undo = SceneRollback::Full {
                    meta,
                    entries: self.entries.clone(),
                    ordered: self.ordered.clone(),
                    pos_by_id: self.pos_by_id.clone(),
                };
                if let Err(error) = self.apply_topology_change(frame, after, true) {
                    self.rollback(undo);
                    return Err(error);
                }
                self.topology_epoch = next_topology_epoch();
                self.revision = self.revision.wrapping_add(1);
                self.dirty_slots.clear();
                self.last_topology_changed = true;
                self.last_range = after..after + self.ordered.len();
                undo
            }
        } else {
            let meta = self.snapshot_meta();
            let undo = if frame.retained_reset {
                SceneRollback::Full {
                    meta,
                    entries: std::mem::take(&mut self.entries),
                    ordered: std::mem::take(&mut self.ordered),
                    pos_by_id: std::mem::take(&mut self.pos_by_id),
                }
            } else {
                SceneRollback::Full {
                    meta,
                    entries: self.entries.clone(),
                    ordered: self.ordered.clone(),
                    pos_by_id: self.pos_by_id.clone(),
                }
            };
            if let Err(error) = self.apply_topology_change(frame, after, false) {
                self.rollback(undo);
                return Err(error);
            }
            self.topology_epoch = next_topology_epoch();
            self.revision = self.revision.wrapping_add(1);
            self.dirty_slots.clear();
            self.last_topology_changed = true;
            self.last_range = after..after + self.ordered.len();
            undo
        };

        if let Err(error) = self.finish_frame(frame, after) {
            self.rollback(undo);
            return Err(error);
        }
        Ok(undo)
    }

    /// Recycles rollback scratch memory once a frame has been accepted.
    pub fn commit(&mut self, undo: SceneRollback) {
        match undo {
            SceneRollback::None(mut meta) | SceneRollback::Full { mut meta, .. } => {
                meta.dirty_slots.clear();
                self.dirty_scratch = meta.dirty_slots;
            }
            SceneRollback::InPlace { mut meta, mut changes } => {
                meta.dirty_slots.clear();
                self.dirty_scratch = meta.dirty_slots;
                changes.clear();
                self.rollback_scratch = changes;
            }
        }
    }

    /// Reverts the scene to its state before the matching [`Self::apply`] call.
    pub fn rollback(&mut self, undo: SceneRollback) {
        match undo {
            SceneRollback::None(meta) => {
                self.restore_meta(meta);
            }
            SceneRollback::InPlace { meta, mut changes } => {
                self.restore_meta(meta);
                for (pos, old_draw) in changes.drain(..).rev() {
                    self.ordered[pos] = old_draw;
                }
                self.rollback_scratch = changes;
            }
            SceneRollback::Full { meta, entries, ordered, pos_by_id } => {
                self.restore_meta(meta);
                self.entries = entries;
                self.ordered = ordered;
                self.pos_by_id = pos_by_id;
            }
        }
    }

    fn apply_topology_change(
        &mut self,
        frame: &RenderFrame,
        after: usize,
        instances_already_checked: bool,
    ) -> Result<()> {
        if !instances_already_checked {
            self.touched.clear();
        }
        for value in &frame.instance_removals {
            let id = value.id_low | (value.id_high << 16);
            ensure!(self.touched.insert(id), "duplicate retained instance identity {id}");
            let Some(pos) = self.pos_by_id.remove(&id) else {
                bail!("removing unknown retained instance {id}");
            };
            self.entries.swap_remove(pos);
            self.ordered.swap_remove(pos);
            if let Some(swapped) = self.entries.get(pos) {
                self.pos_by_id.insert(swapped.id, pos);
            }
        }
        let mut moved_ids: Vec<(u32, Matrix)> = Vec::new();
        for value in &frame.instances {
            let id = value.id_low | (value.id_high << 16);
            if !instances_already_checked {
                ensure!(self.touched.insert(id), "duplicate retained instance identity {id}");
            }
            let draw = Draw {
                mesh: value.mesh,
                material: value.material,
                transform: value.transform,
                parameters: value.parameters,
                extra: None,
            };
            if let Some(&pos) = self.pos_by_id.get(&id) {
                if !frame.retained_reset && self.ordered[pos].transform != value.transform {
                    moved_ids.push((id, self.ordered[pos].transform));
                }
                self.entries[pos].order = value.order;
                self.ordered[pos] = draw;
            } else {
                let pos = self.entries.len();
                self.entries.push(RetainedEntry { id, order: value.order });
                self.ordered.push(draw);
                self.pos_by_id.insert(id, pos);
            }
        }
        ensure!(self.entries.len() <= MAX_DRAWS, "too many retained instances");
        let is_sorted = self.entries.windows(2).all(|w| (w[0].order, w[0].id) <= (w[1].order, w[1].id));
        if !is_sorted {
            let count = self.entries.len();
            #[allow(clippy::cast_possible_truncation, reason = "entries.len() <= MAX_DRAWS fits in u32")]
            let mut perm: Vec<(u64, u32)> = self
                .entries
                .iter()
                .enumerate()
                .map(|(i, e)| ((u64::from(e.order) << 32) | u64::from(e.id), i as u32))
                .collect();
            crate::sort::sort_u64_u32(&mut perm);
            let old_entries = std::mem::take(&mut self.entries);
            let mut old_ordered = std::mem::take(&mut self.ordered);
            self.entries.reserve(count);
            self.ordered.reserve(count);
            for (new_pos, (_, old_pos)) in perm.into_iter().enumerate() {
                let old_pos = old_pos as usize;
                let entry = old_entries[old_pos];
                self.entries.push(entry);
                self.ordered.push(std::mem::take(&mut old_ordered[old_pos]));
                self.pos_by_id.insert(entry.id, new_pos);
            }
        }
        if !moved_ids.is_empty() {
            for (id, previous) in moved_ids {
                let pos = self.pos_by_id[&id];
                let resolved = u32::try_from(after + pos)?;
                self.moved_buf.push((resolved, previous));
            }
            if self.moved_buf.len() > 1 {
                crate::sort::sort_by_u64_key(&mut self.moved_buf, |&(draw, _)| u64::from(draw));
            }
        }
        self.started = true;
        Ok(())
    }

    fn finish_frame(&mut self, frame: &mut RenderFrame, after: usize) -> Result<()> {
        ensure!(frame.draws.len() + self.ordered.len() <= MAX_DRAWS, "too many resolved draws");
        // Poses, explicit bone matrices, morph weights and motions targeting scene draws
        // (`inset == 0`) refer to immediate draws in the packet, before insertion.
        let inserted = u32::try_from(self.ordered.len())?;
        for pose in &mut frame.poses {
            if pose.inset == 0 {
                ensure!((pose.draw as usize) < frame.draws.len(), "pose refers to an unknown immediate draw");
                if pose.draw as usize >= after {
                    pose.draw += inserted;
                }
            }
        }
        for bone in &mut frame.bone_matrices {
            if bone.inset == 0 {
                ensure!((bone.draw as usize) < frame.draws.len(), "bone matrix refers to an unknown immediate draw");
                if bone.draw as usize >= after {
                    bone.draw += inserted;
                }
            }
        }
        for morph in &mut frame.morph_weights {
            if morph.inset == 0 {
                ensure!((morph.draw as usize) < frame.draws.len(), "morph weight refers to an unknown immediate draw");
                if morph.draw as usize >= after {
                    morph.draw += inserted;
                }
            }
        }
        for motion in &mut frame.motions {
            ensure!((motion.draw as usize) < frame.draws.len(), "motion refers to an unknown immediate draw");
            if motion.draw as usize >= after {
                motion.draw += inserted;
            }
        }
        for extension in frame.extensions.iter_mut().filter(|ext| ext.tag == 65_007) {
            let values = &mut extension.values;
            if values.len() < 3 || !(1.0..=256.0).contains(&values[0]) {
                continue;
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "bounded style count")]
            let targets_start = 3 + values[0].round() as usize * 11;
            let Some(targets) = values.get_mut(targets_start..) else { continue };
            for pair in targets.as_chunks_mut::<2>().0 {
                let draw = pair[0].round();
                if !draw.is_finite() || pair[0] < 0.0 || draw >= frame.draws.len() as f32 {
                    pair[0] = -1.0;
                } else if draw >= after as f32 {
                    pair[0] = draw + inserted as f32;
                }
            }
        }
        for &(draw, previous) in &self.moved_buf {
            if frame.motions.len() < MAX_MOTIONS {
                frame.motions.push(Motion { draw, previous });
            }
        }
        if after == frame.draws.len() {
            frame.draws.reserve(self.ordered.len());
            frame.draws.extend(self.ordered.iter().cloned());
        } else {
            frame.draws.splice(after..after, self.ordered.iter().cloned());
        }
        // Downstream consumers see a complete ordinary frame, with no deltas
        // to apply a second time (recordings/screenshots also use this form).
        frame.retained_enabled = false;
        frame.retained_reset = false;
        frame.retained_after = 0;
        frame.instances.clear();
        frame.instance_removals.clear();
        Ok(())
    }

    /// Prepare against accepted state. Only publish the returned scene after
    /// both packet and renderer validation succeed and the guest is accepted.
    #[cfg(test)]
    pub fn resolve(&self, frame: &mut RenderFrame) -> Result<Self> {
        let mut next = self.clone();
        next.apply(frame)?;
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_protocol::{Extension, Instance, InstanceRemoval, Pose};
    fn frame(reset: bool, updates: &[(u32, u32, u32)]) -> RenderFrame {
        RenderFrame {
            retained_enabled: true,
            retained_reset: reset,
            instances: updates
                .iter()
                .map(|&(id, order, mesh)| Instance {
                    id_low: id & 65535,
                    id_high: id >> 16,
                    order,
                    mesh,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }
    #[test]
    fn deltas_preserve_order_and_pose_targets() {
        let mut first = frame(true, &[(70000, 2, 2), (5, 1, 1)]);
        first.instances[1].transform[12] = 1.0;
        let scene = Scene::default().resolve(&mut first).unwrap();
        assert_eq!(first.draws.iter().map(|d| d.mesh).collect::<Vec<_>>(), [1, 2]);
        assert!(first.motions.is_empty());
        let mut delta = frame(false, &[(5, 3, 3)]);
        delta.instances[0].transform[12] = 4.0;
        delta.draws = vec![Draw { mesh: 4, ..Default::default() }, Draw { mesh: 5, ..Default::default() }];
        delta.retained_after = 1;
        delta.poses.push(Pose { draw: 1, ..Default::default() });
        delta.motions.push(Motion { draw: 1, previous: [2.0; 16] });
        let mut outline = vec![0.0; 14];
        outline[0] = 1.0;
        outline.extend([0.0, 0.0, 1.0, 0.0, 2.0, 0.0]);
        delta.extensions.push(Extension { tag: 65_007, values: outline });
        let scene = scene.resolve(&mut delta).unwrap();
        assert_eq!(delta.draws.iter().map(|d| d.mesh).collect::<Vec<_>>(), [4, 2, 3, 5]);
        assert_eq!(delta.poses[0].draw, 3);
        assert_eq!(&delta.extensions[0].values[14..], &[0.0, 0.0, 3.0, 0.0, -1.0, 0.0]);
        assert_eq!(delta.motions.len(), 2);
        assert_eq!(delta.motions[0], Motion { draw: 3, previous: [2.0; 16] });
        assert_eq!(delta.motions[1].draw, 2);
        assert_eq!(delta.motions[1].previous[12], 1.0);
        let mut removal = frame(false, &[]);
        removal.instance_removals.push(InstanceRemoval { id_low: 5, id_high: 0 });
        scene.resolve(&mut removal).unwrap();
        assert_eq!(removal.draws.len(), 1);
    }
    #[test]
    fn rejected_or_unaccepted_deltas_do_not_change_the_scene() {
        let scene = Scene::default().resolve(&mut frame(true, &[(1, 0, 1)])).unwrap();
        assert!(scene.resolve(&mut frame(false, &[(1, 0, 2), (1, 0, 3)])).is_err());
        let _unaccepted = scene.resolve(&mut frame(false, &[(1, 0, 4)])).unwrap();
        let mut unchanged = frame(false, &[]);
        scene.resolve(&mut unchanged).unwrap();
        assert_eq!(unchanged.draws[0].mesh, 1);
        let reset = scene.resolve(&mut frame(true, &[])).unwrap();
        assert!(reset.ordered.is_empty());
        let disabled = scene.resolve(&mut RenderFrame::default()).unwrap();
        assert!(disabled.resolve(&mut frame(false, &[])).is_err());
    }
    #[test]
    fn in_place_updates_preserve_motion_order_and_rollback_cleanly() {
        let mut scene = Scene::default();
        let mut init = frame(true, &[(10, 1, 100), (20, 2, 200), (30, 3, 300)]);
        init.instances[0].transform[12] = 1.0;
        init.instances[2].transform[12] = 3.0;
        let _ = scene.apply(&mut init).unwrap();
        // Fast-path in-place update in reverse order: motions must still emit sorted by draw index.
        let mut delta = frame(false, &[(30, 3, 301), (10, 1, 101)]);
        delta.instances[0].transform[12] = 30.0;
        delta.instances[1].transform[12] = 10.0;
        let undo = scene.apply(&mut delta).unwrap();
        assert!(matches!(undo, SceneRollback::InPlace { .. }));
        assert_eq!(delta.draws.iter().map(|d| d.mesh).collect::<Vec<_>>(), [101, 200, 301]);
        assert_eq!(delta.motions.len(), 2);
        assert_eq!(delta.motions[0].draw, 0);
        assert_eq!(delta.motions[0].previous[12], 1.0);
        assert_eq!(delta.motions[1].draw, 2);
        assert_eq!(delta.motions[1].previous[12], 3.0);
        // Rollback reverts every in-place edit.
        scene.rollback(undo);
        let mut check = frame(false, &[]);
        let _ = scene.apply(&mut check).unwrap();
        assert_eq!(check.draws.iter().map(|d| d.mesh).collect::<Vec<_>>(), [100, 200, 300]);
        assert_eq!(check.draws[0].transform[12], 1.0);
        assert_eq!(check.draws[2].transform[12], 3.0);
    }

    #[test]
    fn retained_delta_tracks_epoch_revision_dirty_slots_and_rollback() {
        let mut scene = Scene::default();
        let mut init = frame(true, &[(10, 1, 100), (20, 2, 200), (30, 3, 300)]);
        let undo0 = scene.apply(&mut init).unwrap();
        scene.commit(undo0);
        let d0 = scene.delta().unwrap();
        assert!(d0.topology_changed);
        assert_eq!(d0.range, 0..3);
        let epoch0 = d0.topology_epoch;
        let rev0 = d0.revision;

        // Transform-only in-place update keeps topology_epoch, bumps revision, and lists only dirty slots.
        let mut move_only = frame(false, &[(30, 3, 300)]);
        move_only.instances[0].transform[12] = 9.0;
        let undo1 = scene.apply(&mut move_only).unwrap();
        let d1 = scene.delta().unwrap();
        assert!(!d1.topology_changed);
        assert_eq!(d1.topology_epoch, epoch0);
        assert_eq!(d1.revision, rev0 + 1);
        assert_eq!(d1.dirty_slots, &[2]);
        scene.commit(undo1);

        // Idle frame keeps both topology_epoch and revision with empty dirty_slots.
        let mut idle = frame(false, &[]);
        let undo2 = scene.apply(&mut idle).unwrap();
        let d2 = scene.delta().unwrap();
        assert!(!d2.topology_changed);
        assert_eq!(d2.topology_epoch, epoch0);
        assert_eq!(d2.revision, rev0 + 1);
        assert!(d2.dirty_slots.is_empty());
        scene.commit(undo2);

        // Mesh swap bumps topology_epoch and sets topology_changed; rollback restores prior epoch/revision.
        let mut swap_mesh = frame(false, &[(20, 2, 201)]);
        let undo3 = scene.apply(&mut swap_mesh).unwrap();
        let d3 = scene.delta().unwrap();
        assert!(d3.topology_changed);
        assert_ne!(d3.topology_epoch, epoch0);
        scene.rollback(undo3);
        let d_restored = scene.delta().unwrap();
        assert_eq!(d_restored.topology_epoch, epoch0);
        assert_eq!(d_restored.revision, rev0 + 1);
        assert!(!d_restored.topology_changed);
    }
}
