//! Transactional retained draws. Guest IDs are scoped to one program instance.
use anyhow::{Result, ensure};
use pen_protocol::boundary::{MAX_DRAWS, MAX_MOTIONS};
use pen_protocol::{Draw, Matrix, Motion, RenderFrame};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Clone, Default)]
pub(crate) struct Scene {
    started: bool,
    entries: Arc<BTreeMap<u32, (u32, Draw)>>,
    ordered: Arc<Vec<Draw>>,
}

impl Scene {
    /// Prepare against accepted state. Only publish the returned scene after
    /// both packet and renderer validation succeed and the guest is accepted.
    pub fn resolve(&self, frame: &mut RenderFrame) -> Result<Self> {
        if !frame.retained_enabled {
            ensure!(
                !frame.retained_reset && frame.instances.is_empty() && frame.instance_removals.is_empty(),
                "retained updates require retained_enabled"
            );
            return Ok(Self::default());
        }
        ensure!(self.started || frame.retained_reset, "first retained frame requires a reset");
        let after = frame.retained_after as usize;
        ensure!(after <= frame.draws.len(), "retained insertion exceeds immediate draws");
        let mut next = if frame.retained_reset { Self::default() } else { self.clone() };
        let changed = frame.retained_reset || !frame.instances.is_empty() || !frame.instance_removals.is_empty();
        let mut moved_retained: Vec<(u32, Matrix)> = Vec::new();
        if changed {
            let entries = Arc::make_mut(&mut next.entries);
            let mut touched = BTreeSet::new();
            for value in &frame.instance_removals {
                let id = value.id_low | (value.id_high << 16);
                ensure!(touched.insert(id), "duplicate retained instance identity {id}");
                ensure!(entries.remove(&id).is_some(), "removing unknown retained instance {id}");
            }
            for value in &frame.instances {
                let id = value.id_low | (value.id_high << 16);
                ensure!(touched.insert(id), "duplicate retained instance identity {id}");
                if !frame.retained_reset
                    && let Some((_, prev)) = entries.get(&id)
                    && prev.transform != value.transform
                {
                    moved_retained.push((id, prev.transform));
                }
                entries.insert(
                    id,
                    (
                        value.order,
                        Draw {
                            mesh: value.mesh,
                            material: value.material,
                            transform: value.transform,
                            parameters: value.parameters,
                            extra: None,
                        },
                    ),
                );
            }
            ensure!(entries.len() <= MAX_DRAWS, "too many retained instances");
            let mut ordered: Vec<_> = entries.iter().collect();
            ordered.sort_unstable_by_key(|(id, (order, _))| (*order, **id));
            if !moved_retained.is_empty() {
                let moved_map: BTreeMap<u32, Matrix> = moved_retained.drain(..).collect();
                for (pos, (id, _)) in ordered.iter().enumerate() {
                    if let Some(&previous) = moved_map.get(id) {
                        let resolved = u32::try_from(after + pos)?;
                        moved_retained.push((resolved, previous));
                    }
                }
            }
            next.ordered = Arc::new(ordered.into_iter().map(|(_, (_, draw))| draw.clone()).collect());
        }
        next.started = true;
        ensure!(frame.draws.len() + next.ordered.len() <= MAX_DRAWS, "too many resolved draws");
        // Poses, explicit bone matrices, morph weights and motions targeting scene draws
        // (`inset == 0`) refer to immediate draws in the packet, before insertion.
        let inserted = u32::try_from(next.ordered.len())?;
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
        for (draw, previous) in moved_retained {
            if frame.motions.len() < MAX_MOTIONS {
                frame.motions.push(Motion { draw, previous });
            }
        }
        frame.draws.splice(after..after, next.ordered.iter().cloned());
        // Downstream consumers see a complete ordinary frame, with no deltas
        // to apply a second time (recordings/screenshots also use this form).
        frame.retained_enabled = false;
        frame.retained_reset = false;
        frame.retained_after = 0;
        frame.instances.clear();
        frame.instance_removals.clear();
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
}
