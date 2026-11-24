//! Cinematics & Animation Sequencer (`.seq.json`).
//!
//! Multi-track keyframe timeline supporting Step, Linear, Cubic Hermite, and Cubic Bezier
//! interpolation, Quaternion SLERP for rotations, exact event/audio trigger firing across
//! frame advances (including loop wrap-around), scrubbing, and live application to a [`Scene`].

use serde::{Deserialize, Serialize};

use crate::components::Component;
use crate::id::NodeId;
use crate::scene::Scene;
use crate::transform::normalize_quat;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum CurveInterp {
    Step,
    Linear,
    CubicHermite { out_tangent: f32, next_in_tangent: f32 },
    /// Normalized cubic Bezier ease `(x1, y1, x2, y2)` in `[0, 1]^2`.
    Bezier { x1: f32, y1: f32, x2: f32, y2: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScalarKey {
    pub time: f32,
    pub value: f32,
    #[serde(default = "default_linear")]
    pub interp: CurveInterp,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vec3Key {
    pub time: f32,
    pub value: [f32; 3],
    #[serde(default = "default_linear")]
    pub interp: CurveInterp,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuatKey {
    pub time: f32,
    pub value: [f32; 4],
    #[serde(default = "default_linear")]
    pub interp: CurveInterp,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EventKey {
    pub time: f32,
    pub name: String,
    #[serde(default)]
    pub payload: String,
}

fn default_linear() -> CurveInterp {
    CurveInterp::Linear
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SequenceTrack {
    NodePosition { node: NodeId, keys: Vec<Vec3Key> },
    NodeRotation { node: NodeId, keys: Vec<QuatKey> },
    NodeScale { node: NodeId, keys: Vec<Vec3Key> },
    CameraFov { node: NodeId, keys: Vec<ScalarKey> },
    LightIntensity { node: NodeId, keys: Vec<ScalarKey> },
    LightColor { node: NodeId, keys: Vec<Vec3Key> },
    MorphWeight { node: NodeId, target: u32, keys: Vec<ScalarKey> },
    MaterialParam { node: NodeId, slot: u32, keys: Vec<ScalarKey> },
    AudioTrigger { bus: u32, events: Vec<EventKey> },
    EventMarker { events: Vec<EventKey> },
}

/// Authored multi-track cinematic / animation sequence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sequence {
    pub name: String,
    pub duration: f32,
    #[serde(default = "default_fps")]
    pub fps: f32,
    #[serde(default)]
    pub looping: bool,
    #[serde(default)]
    pub tracks: Vec<SequenceTrack>,
}

fn default_fps() -> f32 {
    30.0
}

#[derive(Clone, Debug, PartialEq)]
pub struct FiredEvent {
    pub is_audio: bool,
    pub bus: u32,
    pub time: f32,
    pub name: String,
    pub payload: String,
}

/// Deterministic playback state for a [`Sequence`].
#[derive(Clone, Debug)]
pub struct SequencePlayer {
    pub time: f32,
    pub playing: bool,
    pub speed: f32,
}

impl Default for SequencePlayer {
    fn default() -> Self {
        Self { time: 0.0, playing: false, speed: 1.0 }
    }
}

/// Solves cubic Bezier X(u) = t via Newton-Raphson (6 iterations) and returns Y(u).
pub fn eval_bezier_ease(t: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    let mut u = t;
    for _ in 0..6 {
        let inv = 1.0 - u;
        let x = 3.0 * inv * inv * u * x1 + 3.0 * inv * u * u * x2 + u * u * u;
        let dx = 3.0 * inv * inv * x1 + 6.0 * inv * u * (x2 - x1) + 3.0 * u * u * (1.0 - x2);
        if dx.abs() < 1e-5 {
            break;
        }
        u = (u - (x - t) / dx).clamp(0.0, 1.0);
    }
    let inv = 1.0 - u;
    3.0 * inv * inv * u * y1 + 3.0 * inv * u * u * y2 + u * u * u
}

fn eval_curve_scalar(v0: f32, v1: f32, dt: f32, u: f32, interp: CurveInterp) -> f32 {
    match interp {
        CurveInterp::Step => v0,
        CurveInterp::Linear => v0 + (v1 - v0) * u,
        CurveInterp::CubicHermite { out_tangent, next_in_tangent } => {
            let u2 = u * u;
            let u3 = u2 * u;
            let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
            let h10 = u3 - 2.0 * u2 + u;
            let h01 = -2.0 * u3 + 3.0 * u2;
            let h11 = u3 - u2;
            h00 * v0 + h10 * dt * out_tangent + h01 * v1 + h11 * dt * next_in_tangent
        }
        CurveInterp::Bezier { x1, y1, x2, y2 } => {
            let e = eval_bezier_ease(u, x1, y1, x2, y2);
            v0 + (v1 - v0) * e
        }
    }
}

pub fn sample_scalar(keys: &[ScalarKey], time: f32) -> Option<f32> {
    if keys.is_empty() {
        return None;
    }
    if time <= keys[0].time || keys.len() == 1 {
        return Some(keys[0].value);
    }
    let last = keys.last().unwrap();
    if time >= last.time {
        return Some(last.value);
    }
    for w in keys.windows(2) {
        if time >= w[0].time && time <= w[1].time {
            let dt = (w[1].time - w[0].time).max(1e-6);
            let u = ((time - w[0].time) / dt).clamp(0.0, 1.0);
            return Some(eval_curve_scalar(w[0].value, w[1].value, dt, u, w[0].interp));
        }
    }
    Some(last.value)
}

pub fn sample_vec3(keys: &[Vec3Key], time: f32) -> Option<[f32; 3]> {
    if keys.is_empty() {
        return None;
    }
    if time <= keys[0].time || keys.len() == 1 {
        return Some(keys[0].value);
    }
    let last = keys.last().unwrap();
    if time >= last.time {
        return Some(last.value);
    }
    for w in keys.windows(2) {
        if time >= w[0].time && time <= w[1].time {
            let dt = (w[1].time - w[0].time).max(1e-6);
            let u = ((time - w[0].time) / dt).clamp(0.0, 1.0);
            return Some([
                eval_curve_scalar(w[0].value[0], w[1].value[0], dt, u, w[0].interp),
                eval_curve_scalar(w[0].value[1], w[1].value[1], dt, u, w[0].interp),
                eval_curve_scalar(w[0].value[2], w[1].value[2], dt, u, w[0].interp),
            ]);
        }
    }
    Some(last.value)
}

fn slerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let a = normalize_quat(a);
    let mut b = normalize_quat(b);
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if dot < 0.0 {
        dot = -dot;
        b = [-b[0], -b[1], -b[2], -b[3]];
    }
    if dot > 0.9995 {
        return normalize_quat([
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        ]);
    }
    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_t = theta.sin();
    let wa = ((1.0 - t) * theta).sin() / sin_t;
    let wb = (t * theta).sin() / sin_t;
    normalize_quat([
        a[0] * wa + b[0] * wb,
        a[1] * wa + b[1] * wb,
        a[2] * wa + b[2] * wb,
        a[3] * wa + b[3] * wb,
    ])
}

pub fn sample_quat(keys: &[QuatKey], time: f32) -> Option<[f32; 4]> {
    if keys.is_empty() {
        return None;
    }
    if time <= keys[0].time || keys.len() == 1 {
        return Some(normalize_quat(keys[0].value));
    }
    let last = keys.last().unwrap();
    if time >= last.time {
        return Some(normalize_quat(last.value));
    }
    for w in keys.windows(2) {
        if time >= w[0].time && time <= w[1].time {
            let dt = (w[1].time - w[0].time).max(1e-6);
            let u = ((time - w[0].time) / dt).clamp(0.0, 1.0);
            let eased = match w[0].interp {
                CurveInterp::Step => return Some(normalize_quat(w[0].value)),
                CurveInterp::Bezier { x1, y1, x2, y2 } => eval_bezier_ease(u, x1, y1, x2, y2),
                _ => u,
            };
            return Some(slerp(w[0].value, w[1].value, eased));
        }
    }
    Some(normalize_quat(last.value))
}

impl SequencePlayer {
    pub fn scrub(&mut self, seq: &Sequence, time: f32) {
        self.time = time.clamp(0.0, seq.duration.max(0.0));
    }

    /// Advances playback by `dt` seconds and returns any event/audio cues crossed in `(prev, next]`.
    pub fn advance(&mut self, seq: &Sequence, dt: f32) -> Vec<FiredEvent> {
        let mut fired = Vec::new();
        if !self.playing || dt <= 0.0 || seq.duration <= 0.0 {
            return fired;
        }
        let prev = self.time;
        let raw_next = prev + dt * self.speed;
        let (next, wrapped) = if seq.looping && raw_next >= seq.duration {
            (raw_next.rem_euclid(seq.duration), true)
        } else {
            let clamped = raw_next.clamp(0.0, seq.duration);
            if !seq.looping && clamped >= seq.duration {
                self.playing = false;
            }
            (clamped, false)
        };
        self.time = next;

        let collect = |events: &[EventKey], is_audio: bool, bus: u32, fired: &mut Vec<FiredEvent>| {
            for ev in events {
                let hit = if wrapped {
                    (ev.time > prev && ev.time <= seq.duration) || (ev.time >= 0.0 && ev.time <= next)
                } else {
                    ev.time > prev && ev.time <= next
                };
                if hit {
                    fired.push(FiredEvent {
                        is_audio,
                        bus,
                        time: ev.time,
                        name: ev.name.clone(),
                        payload: ev.payload.clone(),
                    });
                }
            }
        };

        for track in &seq.tracks {
            match track {
                SequenceTrack::AudioTrigger { bus, events } => collect(events, true, *bus, &mut fired),
                SequenceTrack::EventMarker { events } => collect(events, false, 0, &mut fired),
                _ => {}
            }
        }
        fired
    }

    /// Evaluates every property track at `self.time` and applies it live onto `scene`.
    pub fn apply_to_scene(&self, seq: &Sequence, scene: &mut Scene) {
        for track in &seq.tracks {
            match track {
                SequenceTrack::NodePosition { node, keys } => {
                    if let (Some(v), Some(n)) = (sample_vec3(keys, self.time), scene.find_node_mut(*node)) {
                        n.transform.position = v;
                    }
                }
                SequenceTrack::NodeRotation { node, keys } => {
                    if let (Some(q), Some(n)) = (sample_quat(keys, self.time), scene.find_node_mut(*node)) {
                        n.transform.rotation = q;
                    }
                }
                SequenceTrack::NodeScale { node, keys } => {
                    if let (Some(s), Some(n)) = (sample_vec3(keys, self.time), scene.find_node_mut(*node)) {
                        n.transform.scale = s;
                    }
                }
                SequenceTrack::CameraFov { node, keys } => {
                    if let (Some(fov), Some(n)) = (sample_scalar(keys, self.time), scene.find_node_mut(*node)) {
                        for comp in &mut n.components {
                            if let Component::Camera { fov_deg, .. } = comp {
                                *fov_deg = fov.clamp(5.0, 160.0);
                            }
                        }
                    }
                }
                SequenceTrack::LightIntensity { node, keys } => {
                    if let (Some(int), Some(n)) = (sample_scalar(keys, self.time), scene.find_node_mut(*node)) {
                        for comp in &mut n.components {
                            if let Component::Light { intensity, .. } = comp {
                                *intensity = int.max(0.0);
                            }
                        }
                    }
                }
                SequenceTrack::LightColor { node, keys } => {
                    if let (Some(col), Some(n)) = (sample_vec3(keys, self.time), scene.find_node_mut(*node)) {
                        for comp in &mut n.components {
                            if let Component::Light { color, .. } = comp {
                                *color = [col[0].max(0.0), col[1].max(0.0), col[2].max(0.0)];
                            }
                        }
                    }
                }
                SequenceTrack::MaterialParam { node, slot, keys } => {
                    let s = (*slot as usize).min(3);
                    if let (Some(val), Some(n)) = (sample_scalar(keys, self.time), scene.find_node_mut(*node)) {
                        for comp in &mut n.components {
                            if let Component::MeshRenderer { parameters, .. } = comp {
                                let p = parameters.get_or_insert([0.0; 4]);
                                p[s] = val;
                            }
                        }
                    }
                }
                SequenceTrack::MorphWeight { .. } | SequenceTrack::AudioTrigger { .. } | SequenceTrack::EventMarker { .. } => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::LightKind;
    use crate::scene::SceneNode;

    #[test]
    fn sequencer_interpolates_bezier_hermite_slerp_fires_looping_events_and_mutates_scene() {
        let mut scene = Scene::new("cutscene");
        let mut cam = SceneNode::new("HeroCam", None, 0);
        cam.components.push(Component::Camera {
            fov_deg: 60.0,
            near: 0.1,
            far: 200.0,
            active: true,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            target_material: None,
            viewport: None,
        });
        let cam_id = cam.id;
        let mut lamp = SceneNode::new("KeyLight", None, 1);
        lamp.components.push(Component::Light {
            kind: LightKind::Point,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            range: 10.0,
            inner_cone: 0.0,
            outer_cone: 0.0,
            width: 1.0,
            height: 1.0,
            cast_shadows: false,
        });
        let lamp_id = lamp.id;
        scene.nodes.push(cam);
        scene.nodes.push(lamp);

        let seq = Sequence {
            name: "intro_shot".into(),
            duration: 2.0,
            fps: 30.0,
            looping: true,
            tracks: vec![
                SequenceTrack::NodePosition {
                    node: cam_id,
                    keys: vec![
                        Vec3Key {
                            time: 0.0,
                            value: [0.0, 1.0, 5.0],
                            interp: CurveInterp::Bezier { x1: 0.42, y1: 0.0, x2: 0.58, y2: 1.0 },
                        },
                        Vec3Key { time: 2.0, value: [10.0, 1.0, 5.0], interp: CurveInterp::Linear },
                    ],
                },
                SequenceTrack::CameraFov {
                    node: cam_id,
                    keys: vec![
                        ScalarKey {
                            time: 0.0,
                            value: 60.0,
                            interp: CurveInterp::CubicHermite { out_tangent: 0.0, next_in_tangent: 0.0 },
                        },
                        ScalarKey { time: 2.0, value: 30.0, interp: CurveInterp::Linear },
                    ],
                },
                SequenceTrack::LightIntensity {
                    node: lamp_id,
                    keys: vec![
                        ScalarKey { time: 0.0, value: 1.0, interp: CurveInterp::Step },
                        ScalarKey { time: 1.0, value: 5.0, interp: CurveInterp::Linear },
                    ],
                },
                SequenceTrack::AudioTrigger {
                    bus: 2,
                    events: vec![EventKey { time: 1.8, name: "boom.opus".into(), payload: String::new() }],
                },
                SequenceTrack::EventMarker {
                    events: vec![EventKey { time: 0.1, name: "title_card".into(), payload: "ACT_I".into() }],
                },
            ],
        };

        let mut player = SequencePlayer { time: 0.0, playing: true, speed: 1.0 };
        player.scrub(&seq, 1.0);
        player.apply_to_scene(&seq, &mut scene);
        assert!((scene.find_node(cam_id).unwrap().transform.position[0] - 5.0).abs() < 0.05);
        assert!(matches!(
            scene.find_node(cam_id).unwrap().components[0],
            Component::Camera { fov_deg, .. } if (fov_deg - 45.0).abs() < 0.05
        ));
        assert!(matches!(
            scene.find_node(lamp_id).unwrap().components[0],
            Component::Light { intensity, .. } if (intensity - 5.0).abs() < 1e-4
        ));

        // Advance across loop boundary (1.5 -> 0.2): must fire BOTH boom.opus (at 1.8) AND title_card (at 0.1)!
        player.scrub(&seq, 1.5);
        let fired = player.advance(&seq, 0.7);
        assert_eq!(fired.len(), 2, "must fire both pre-wrap and post-wrap triggers: {fired:?}");
        assert!(fired.iter().any(|e| e.is_audio && e.name == "boom.opus"));
        assert!(fired.iter().any(|e| !e.is_audio && e.name == "title_card"));
    }
}
