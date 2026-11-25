//! Where a voice is heard from, as WebAudio's PannerNode places it with
//! equal-power panning (the Web Audio API, "Spatialization/Panning"): the
//! azimuth of the source around the listener, the gains that pan it by, and
//! how its distance and cone quiet it. The native mixer pans with these, and
//! the browser host gives its PannerNodes the same values ([`Placement`]'s
//! sanitized ones), so both hosts place a sound alike. The arithmetic follows
//! Chrome's (`PannerHandler`, `EqualPowerPanner`, `DistanceEffect`,
//! `ConeEffect`): vectors in f32, angles in f64.
//!
//! Both hosts hear every panner from WebAudio's default listener, at the
//! origin facing -z: a frame's panners are placed in its listener's own
//! frame first ([`Placements::of`]). Where the listener is and faces does
//! not change what a still scene sounds like, but it keeps the browser's
//! AudioListener still, whose automation Chrome runs at twice the rate it
//! should (a glide toward a new listener there is twice as fast as the same
//! glide on a PannerNode), so the hosts glide the same values alike.
#![allow(clippy::cast_possible_truncation, reason = "f64 angles and gains narrowed to f32 samples")]

use std::collections::HashMap;

use pen_protocol::{Listener, Matrix, Panner, RenderFrame, Vector, distance_model};

/// A point or direction, as Chrome keeps one.
pub type Vec3 = [f32; 3];

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn length(a: Vec3) -> f64 {
    let [x, y, z] = a.map(f64::from);
    (x * x + y * y + z * z).sqrt()
}

fn is_zero(a: Vec3) -> bool {
    a == [0.0; 3]
}

/// `a` at length 1, or as it is at length 0.
fn normalize(a: Vec3) -> Vec3 {
    let length = length(a);
    if length == 0.0 { a } else { a.map(|value| (f64::from(value) / length) as f32) }
}

fn degrees(radians: f64) -> f64 {
    radians.to_degrees()
}

/// The listener: where it is, the way it faces, its up, and its velocity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ear {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub velocity: Vec3,
}

/// WebAudio's AudioListener as it starts: at the origin, facing -z, up +y.
pub const DEFAULT_EAR: Ear =
    Ear { position: [0.0; 3], forward: [0.0, 0.0, -1.0], up: [0.0, 1.0, 0.0], velocity: [0.0; 3] };

impl Default for Ear {
    fn default() -> Ear {
        DEFAULT_EAR
    }
}

fn vec3(vector: Vector) -> Vec3 {
    [vector.x, vector.y, vector.z]
}

fn finite_vec3(v: Vec3) -> Vec3 {
    [
        if v[0].is_finite() { v[0] } else { 0.0 },
        if v[1].is_finite() { v[1] } else { 0.0 },
        if v[2].is_finite() { v[2] } else { 0.0 },
    ]
}

/// Speed of sound in air at 20 °C, in meters per second.
pub const SPEED_OF_SOUND: f64 = 343.0;

impl Ear {
    /// The frame's listener, or with none the camera that `view` (a
    /// column-major world-to-camera matrix) looks through: at its eye,
    /// facing where it looks (-z), its up the view's. A view that scales
    /// its axes is undone first.
    pub fn of_frame(listener: Option<&Listener>, view: &Matrix) -> Ear {
        if let Some(listener) = listener {
            return Ear {
                position: vec3(listener.position),
                forward: vec3(listener.forward),
                up: vec3(listener.up),
                velocity: finite_vec3(vec3(listener.velocity)),
            };
        }
        let column = |index: usize| [view[index * 4], view[index * 4 + 1], view[index * 4 + 2]];
        let translation = column(3);
        let eye = [0, 1, 2].map(|axis| {
            let axis = column(axis);
            let scale = dot(axis, axis);
            if scale == 0.0 { 0.0 } else { -dot(axis, translation) / scale }
        });
        // The rows of the view's rotation: the camera's right, up and back.
        let row = |index: usize| [view[index], view[4 + index], view[8 + index]];
        let back = row(2);
        Ear { position: eye, forward: [-back[0], -back[1], -back[2]], up: row(1), velocity: [0.0; 3] }
    }
}

/// A panner's settings, within the ranges WebAudio accepts (its setters
/// throw outside them): distances and rolloff not below 0 (the max
/// distance above it), angles finite, the outer gain within 0 and 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    pub position: Vec3,
    pub orientation: Vec3,
    pub velocity: Vec3,
    pub ear_velocity: Vec3,
    pub distance_model: u32,
    pub ref_distance: f64,
    pub max_distance: f64,
    pub rolloff: f64,
    pub cone_inner: f64,
    pub cone_outer: f64,
    pub cone_outer_gain: f64,
    pub hrtf: bool,
    pub doppler: f64,
    pub occlusion: f64,
    pub portal_occlusion: f64,
    pub zone: u32,
    pub wet_scale: f32,
}

/// A value the browser's setter would take, else `otherwise`.
fn finite(value: f32, otherwise: f64) -> f64 {
    if value.is_finite() { f64::from(value) } else { otherwise }
}

impl Placement {
    /// A panner from the frame, sanitized: what WebAudio would refuse
    /// becomes its default (an inverse model, ref 1, max 10000, rolloff 1,
    /// no cone).
    pub fn new(panner: &Panner) -> Placement {
        let max_distance = finite(panner.max_distance, 10_000.0);
        Placement {
            position: vec3(panner.position),
            orientation: vec3(panner.orientation),
            velocity: finite_vec3(vec3(panner.velocity)),
            ear_velocity: [0.0; 3],
            distance_model: if panner.distance_model <= distance_model::EXPONENTIAL {
                panner.distance_model
            } else {
                distance_model::INVERSE
            },
            ref_distance: finite(panner.ref_distance, 1.0).max(0.0),
            max_distance: if max_distance > 0.0 { max_distance } else { 10_000.0 },
            rolloff: finite(panner.rolloff, 1.0).max(0.0),
            cone_inner: finite(panner.cone_inner, 360.0),
            cone_outer: finite(panner.cone_outer, 360.0),
            cone_outer_gain: finite(panner.cone_outer_gain, 0.0).clamp(0.0, 1.0),
            hrtf: panner.hrtf,
            doppler: finite(panner.doppler, 0.0).clamp(0.0, 4.0),
            occlusion: finite(panner.occlusion, 0.0).clamp(0.0, 1.0),
            portal_occlusion: 0.0,
            zone: panner.zone,
            wet_scale: 1.0,
        }
    }

    /// How much its distance from the listener leaves of it.
    pub fn distance_gain(&self, distance: f64) -> f64 {
        let (reference, rolloff) = (self.ref_distance, self.rolloff);
        match self.distance_model {
            distance_model::LINEAR => {
                let (near, far) = (reference.min(self.max_distance), reference.max(self.max_distance));
                let rolloff = rolloff.clamp(0.0, 1.0);
                if near == far {
                    return 1.0 - rolloff;
                }
                1.0 - rolloff * (distance.clamp(near, far) - near) / (far - near)
            }
            _ if reference == 0.0 => 0.0,
            distance_model::EXPONENTIAL => (distance.max(reference) / reference).powf(-rolloff),
            _ => reference / (reference + rolloff * (distance.max(reference) - reference)),
        }
    }

    /// How much its cone leaves of it, heard from `ear` while it is at
    /// `position` facing `orientation`: whole within the inner angle, the
    /// outer gain beyond the outer one, and between them a line from one to
    /// the other.
    pub fn cone_gain(&self, position: Vec3, orientation: Vec3, ear: Vec3) -> f64 {
        if is_zero(orientation) || (self.cone_inner == 360.0 && self.cone_outer == 360.0) {
            return 1.0;
        }
        let toward = sub(ear, position);
        let lengths = length(toward) * length(orientation);
        if lengths == 0.0 {
            return 1.0;
        }
        let angle = degrees((f64::from(dot(toward, orientation)) / lengths).clamp(-1.0, 1.0).acos()).abs();
        let (inner, outer) = (self.cone_inner.abs() / 2.0, self.cone_outer.abs() / 2.0);
        if angle <= inner {
            1.0
        } else if angle >= outer {
            self.cone_outer_gain
        } else {
            let x = (angle - inner) / (outer - inner);
            (1.0 - x) + self.cone_outer_gain * x
        }
    }

    /// Its position, orientation, and velocities as `ear` has them: relative to
    /// it, along its right, up and back, which is where WebAudio's default
    /// listener hears it as `ear` would.
    pub fn heard_from(self, ear: &Ear) -> Placement {
        let right = normalize(cross(ear.forward, ear.up));
        let forward = normalize(ear.forward);
        let up = cross(right, forward);
        let local = |vector: Vec3| [dot(vector, right), dot(vector, up), -dot(vector, forward)];
        Placement {
            position: local(sub(self.position, ear.position)),
            orientation: local(self.orientation),
            velocity: local(self.velocity),
            ear_velocity: local(ear.velocity),
            ..self
        }
    }

    /// Physical Doppler pitch ratio `(c + v_l) / (c + v_s)` at listener-relative
    /// `position` (where `ear` is at the origin), clamped safely against supersonic
    /// singularities. Returns `1.0` when `doppler <= 0.0` or stationary.
    pub fn doppler_ratio(&self, position: Vec3) -> f32 {
        if self.doppler <= 0.0 || (is_zero(self.velocity) && is_zero(self.ear_velocity)) {
            return 1.0;
        }
        let dist = length(position);
        if dist < 1e-5 {
            return 1.0;
        }
        // Unit vector pointing from listener (origin) toward source.
        let ux = f64::from(position[0]) / dist;
        let uy = f64::from(position[1]) / dist;
        let uz = f64::from(position[2]) / dist;
        let v_l = f64::from(self.ear_velocity[0]) * ux
            + f64::from(self.ear_velocity[1]) * uy
            + f64::from(self.ear_velocity[2]) * uz;
        let v_s = f64::from(self.velocity[0]) * ux
            + f64::from(self.velocity[1]) * uy
            + f64::from(self.velocity[2]) * uz;
        let max_v = 0.9 * SPEED_OF_SOUND;
        let vl_eff = (v_l * self.doppler).clamp(-max_v, max_v);
        let vs_eff = (v_s * self.doppler).clamp(-max_v, max_v);
        ((SPEED_OF_SOUND + vl_eff) / (SPEED_OF_SOUND + vs_eff)).clamp(0.25, 4.0) as f32
    }

    /// Combined direct-path + cross-zone acoustic portal occlusion in `0..=1`.
    pub fn effective_occlusion(&self) -> f64 {
        let a = self.occlusion.clamp(0.0, 1.0);
        let b = self.portal_occlusion.clamp(0.0, 1.0);
        (1.0 - (1.0 - a) * (1.0 - b)).clamp(0.0, 1.0)
    }

    /// Direct-path linear gain multiplier due to occlusion (`1.0` open -> `0.15` fully occluded).
    pub fn occlusion_gain(&self) -> f32 {
        (1.0 - 0.85 * self.effective_occlusion()) as f32
    }

    /// Lowpass cutoff ceiling in Hz due to occlusion (`open_hz` open -> `400 Hz` fully occluded).
    pub fn occlusion_lowpass(&self, open_hz: f32) -> f32 {
        let occ = self.effective_occlusion();
        if occ <= 1e-4 {
            return open_hz;
        }
        (20_000.0 * (400.0 / 20_000.0f64).powf(occ)).min(f64::from(open_hz)) as f32
    }

    /// The distance and cone gains together, for a source at `position`
    /// facing `orientation`, as Chrome multiplies them.
    pub fn gain(&self, position: Vec3, orientation: Vec3, ear: &Ear) -> f32 {
        let distance = length(sub(position, ear.position));
        (self.distance_gain(distance) * self.cone_gain(position, orientation, ear.position)) as f32
    }
}

/// A frame's panners by what they place, in its listener's frame: the first
/// for each voice id, and for each sound by its place in the frame's
/// `sounds`, plus the listener's blended [`crate::reverb::ZoneMix`] and any
/// ray-traced [`crate::reverb::AcousticProfile`].
#[derive(Debug)]
pub struct Placements {
    voices: HashMap<u32, Placement>,
    sounds: Vec<Option<Placement>>,
    zone_mix: crate::reverb::ZoneMix,
    acoustics: Option<crate::reverb::AcousticProfile>,
}

impl Default for Placements {
    fn default() -> Self {
        Self {
            voices: HashMap::new(),
            sounds: Vec::new(),
            zone_mix: crate::reverb::ZoneMix { decay: 0.0, damping: 0.0, wet_scale: 1.0, dominant_zone: 0 },
            acoustics: None,
        }
    }
}

impl Placements {
    pub fn of(frame: &RenderFrame) -> Placements {
        let ear = Ear::of_frame(frame.listener.first(), &frame.view);
        let zone_mix = crate::reverb::blend_zones(
            ear.position,
            frame.reverb.decay,
            frame.reverb.damping,
            &frame.reverb_zones,
        );
        let acoustics = crate::reverb::AcousticProfile::from_extensions(&frame.extensions);
        let placed = |panner: &Panner| {
            let mut p = Placement::new(panner);
            let (portal, wet_scale) = crate::reverb::source_zone_acoustics(
                p.position,
                p.zone,
                ear.position,
                &frame.reverb_zones,
            );
            p.portal_occlusion = f64::from(portal);
            p.wet_scale = wet_scale;
            p.heard_from(&ear)
        };
        let mut placements = Placements {
            voices: HashMap::new(),
            sounds: vec![None; frame.sounds.len()],
            zone_mix,
            acoustics,
        };
        for panner in &frame.panners {
            if panner.voice != 0 {
                placements.voices.entry(panner.voice).or_insert_with(|| placed(panner));
            } else if let Some(slot @ None) = placements.sounds.get_mut(panner.sound as usize) {
                *slot = Some(placed(panner));
            }
        }
        placements
    }

    pub fn zone_mix(&self) -> crate::reverb::ZoneMix {
        self.zone_mix
    }

    pub fn acoustics(&self) -> Option<&crate::reverb::AcousticProfile> {
        self.acoustics.as_ref()
    }

    pub fn voice(&self, id: u32) -> Option<Placement> {
        self.voices.get(&id).copied()
    }

    pub fn sound(&self, index: usize) -> Option<Placement> {
        self.sounds.get(index).copied().flatten()
    }
}

/// The azimuth, in degrees, of a source at `position` around `ear`: 0
/// ahead, 90 to the right, -90 to the left, ±180 behind.
pub fn azimuth(position: Vec3, ear: &Ear) -> f64 {
    let toward = sub(position, ear.position);
    if is_zero(toward) {
        return 0.0;
    }
    let toward = normalize(toward);
    let right = normalize(cross(ear.forward, ear.up));
    let forward = normalize(ear.forward);
    let up = cross(right, forward);
    let lift = dot(toward, up);
    let projected = normalize(sub(toward, up.map(|value| value * lift)));
    let mut azimuth = degrees(f64::from(dot(projected, right).clamp(-1.0, 1.0)).acos());
    if azimuth.is_nan() {
        azimuth = 0.0;
    }
    if dot(projected, forward) < 0.0 {
        azimuth = 360.0 - azimuth;
    }
    if (0.0..=270.0).contains(&azimuth) { 90.0 - azimuth } else { 450.0 - azimuth }
}

/// How a source is panned: the equal-power gains for its azimuth, (left,
/// right) of a mono input, or for a stereo one the share of one side moved
/// into the other, which `right` says, as StereoPannerNode pans.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pan {
    pub left: f32,
    pub right: f32,
    /// A stereo input's left moves into its right (else its right into its
    /// left).
    pub rightward: bool,
}

/// The equal-power pan for `azimuth`; behind folds onto in front.
pub fn equal_power(azimuth: f64, mono: bool) -> Pan {
    let azimuth = azimuth.clamp(-180.0, 180.0);
    let azimuth = if azimuth < -90.0 {
        -180.0 - azimuth
    } else if azimuth > 90.0 {
        180.0 - azimuth
    } else {
        azimuth
    };
    let position = if mono {
        (azimuth + 90.0) / 180.0
    } else if azimuth <= 0.0 {
        (azimuth + 90.0) / 90.0
    } else {
        azimuth / 90.0
    };
    let angle = std::f64::consts::FRAC_PI_2 * position;
    Pan { left: angle.cos() as f32, right: angle.sin() as f32, rightward: azimuth > 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placement(model: u32) -> Placement {
        Placement::new(&Panner {
            distance_model: model,
            ref_distance: 2.0,
            max_distance: 20.0,
            rolloff: 0.5,
            cone_inner: 360.0,
            cone_outer: 360.0,
            ..Panner::default()
        })
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn azimuth_goes_round_the_listener() {
        let ear = DEFAULT_EAR;
        assert!(close(azimuth([0.0, 0.0, -5.0], &ear), 0.0));
        assert!(close(azimuth([5.0, 0.0, 0.0], &ear), 90.0));
        assert!(close(azimuth([-5.0, 0.0, 0.0], &ear), -90.0));
        assert!(close(azimuth([1.0, 0.0, -1.0], &ear), 45.0));
        assert!(close(azimuth([1.0, 0.0, 1.0], &ear), 135.0));
        assert!(close(azimuth([-1.0, 0.0, 1.0], &ear), -135.0));
        // Height does not move it round; at the listener it is ahead.
        assert!(close(azimuth([1.0, 7.0, -1.0], &ear), 45.0));
        assert_eq!(azimuth([0.0; 3], &ear), 0.0);
        // A listener turned to face +x hears +x ahead and -z on its left.
        let turned = Ear { position: [0.0; 3], forward: [1.0, 0.0, 0.0], up: [0.0, 1.0, 0.0], velocity: [0.0; 3] };
        assert!(close(azimuth([3.0, 0.0, 0.0], &turned), 0.0));
        assert!(close(azimuth([0.0, 0.0, -3.0], &turned), -90.0));
    }

    #[test]
    fn equal_power_follows_the_spec() {
        // Mono: -3 dB a side ahead, all of it to one side at ±90, and
        // behind folds onto in front.
        let Pan { left, right, .. } = equal_power(0.0, true);
        assert!((left - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6 && (right - left).abs() < 1e-6);
        let Pan { left, right, .. } = equal_power(90.0, true);
        assert!(left.abs() < 1e-6 && (right - 1.0).abs() < 1e-6);
        assert_eq!(equal_power(150.0, true), equal_power(30.0, true));
        assert_eq!(equal_power(-150.0, true), equal_power(-30.0, true));
        for azimuth in [-90.0, -60.0, -10.0, 0.0, 20.0, 75.0] {
            let Pan { left, right, .. } = equal_power(azimuth, true);
            assert!((left * left + right * right - 1.0).abs() < 1e-6, "{azimuth}");
        }
        // Stereo ahead passes as it is: none of either side moved over. To
        // the right, the left moves over.
        let Pan { left, right, rightward } = equal_power(0.0, false);
        assert!(left.abs() < 1e-6 && (right - 1.0).abs() < 1e-6 && !rightward);
        assert!(equal_power(45.0, false).rightward && !equal_power(-45.0, false).rightward);
    }

    #[test]
    fn distance_models_follow_the_spec() {
        let linear = placement(distance_model::LINEAR);
        assert!(close(linear.distance_gain(1.0), 1.0));
        assert!(close(linear.distance_gain(11.0), 1.0 - 0.5 * 9.0 / 18.0));
        assert!(close(linear.distance_gain(50.0), 0.5));
        let inverse = placement(distance_model::INVERSE);
        assert!(close(inverse.distance_gain(1.0), 1.0));
        assert!(close(inverse.distance_gain(10.0), 2.0 / (2.0 + 0.5 * 8.0)));
        let exponential = placement(distance_model::EXPONENTIAL);
        assert!(close(exponential.distance_gain(8.0), 4f64.powf(-0.5)));
        // WebAudio's own default, and a reference distance of 0.
        let default = Placement::new(&Panner {
            distance_model: distance_model::INVERSE,
            ref_distance: 1.0,
            max_distance: 10_000.0,
            rolloff: 1.0,
            ..Panner::default()
        });
        assert!(close(default.distance_gain(4.0), 0.25));
        assert_eq!(Placement { ref_distance: 0.0, ..default }.distance_gain(4.0), 0.0);
    }

    #[test]
    fn cones_quiet_what_faces_away() {
        let cone = Placement {
            cone_inner: 90.0,
            cone_outer: 180.0,
            cone_outer_gain: 0.25,
            ..placement(distance_model::INVERSE)
        };
        let ear = [0.0, 0.0, 0.0];
        // Facing the listener, within the inner cone; facing away, beyond
        // the outer; 67.5° off, halfway between.
        assert!(close(cone.cone_gain([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], ear), 1.0));
        assert!(close(cone.cone_gain([0.0, 0.0, -1.0], [0.0, 0.0, -1.0], ear), 0.25));
        let angle = 67.5f64.to_radians();
        let facing = [angle.sin() as f32, 0.0, angle.cos() as f32];
        assert!((cone.cone_gain([0.0, 0.0, -1.0], facing, ear) - 0.625).abs() < 1e-4);
        // No orientation, no cone.
        assert!(close(cone.cone_gain([0.0, 0.0, -1.0], [0.0; 3], ear), 1.0));
    }

    /// What WebAudio's setters would refuse becomes its default.
    #[test]
    fn placements_are_what_webaudio_takes() {
        let placed = Placement::new(&Panner {
            distance_model: 9,
            ref_distance: -1.0,
            max_distance: 0.0,
            rolloff: f32::NAN,
            cone_inner: f32::INFINITY,
            cone_outer: 200.0,
            cone_outer_gain: 3.0,
            ..Panner::default()
        });
        assert_eq!(placed.distance_model, distance_model::INVERSE);
        assert_eq!((placed.ref_distance, placed.max_distance, placed.rolloff), (0.0, 10_000.0, 1.0));
        assert_eq!((placed.cone_inner, placed.cone_outer, placed.cone_outer_gain), (360.0, 200.0, 1.0));
    }

    /// Of two panners for one voice or sound the first counts; one that
    /// places nothing in the frame counts for nothing.
    #[test]
    fn placements_go_by_voice_and_sound() {
        let at = |x: f32| Vector { x, y: 0.0, z: 0.0 };
        let frame = RenderFrame {
            sounds: vec![Default::default(); 2],
            listener: vec![Listener {
                position: Vector { x: 0.0, y: 0.0, z: 0.0 },
                forward: Vector { x: 0.0, y: 0.0, z: -1.0 },
                up: Vector { x: 0.0, y: 1.0, z: 0.0 },
                velocity: Vector { x: 0.0, y: 0.0, z: 0.0 },
            }],
            panners: vec![
                Panner { voice: 7, position: at(1.0), ..Panner::default() },
                Panner { voice: 7, position: at(2.0), ..Panner::default() },
                Panner { sound: 1, position: at(3.0), ..Panner::default() },
                Panner { sound: 1, position: at(4.0), ..Panner::default() },
                Panner { sound: 9, position: at(5.0), ..Panner::default() },
            ],
            ..RenderFrame::default()
        };
        let placements = Placements::of(&frame);
        assert_eq!(placements.voice(7).map(|placed| placed.position), Some([1.0, 0.0, 0.0]));
        assert_eq!(placements.voice(8), None);
        assert_eq!(placements.sound(0), None);
        assert_eq!(placements.sound(1).map(|placed| placed.position), Some([3.0, 0.0, 0.0]));
        assert_eq!(placements.sound(9), None);
    }

    /// A source heard from a listener elsewhere, facing elsewhere, is where
    /// the default listener hears it once placed in that listener's frame:
    /// the same azimuth, distance and cone.
    #[test]
    fn placing_in_the_listeners_frame_changes_nothing_heard() {
        let ear = Ear { position: [2.0, -1.0, 5.0], forward: [0.6, 0.0, 0.8], up: [0.0, 1.0, 0.0], velocity: [0.0; 3] };
        let placement = Placement {
            position: [-3.0, 2.0, 1.0],
            orientation: [0.0, 0.0, 1.0],
            cone_inner: 40.0,
            cone_outer: 220.0,
            cone_outer_gain: 0.3,
            ..placement(distance_model::INVERSE)
        };
        let heard = placement.heard_from(&ear);
        assert!(close(azimuth(placement.position, &ear), azimuth(heard.position, &DEFAULT_EAR)));
        let world = placement.gain(placement.position, placement.orientation, &ear);
        let local = heard.gain(heard.position, heard.orientation, &DEFAULT_EAR);
        assert!((world - local).abs() < 1e-6, "{world} {local}");
        // Ahead of the listener is ahead of the default one: along -z.
        let ahead = Placement { position: [2.6, -1.0, 5.8], ..placement }.heard_from(&ear);
        assert!(ahead.position.iter().zip([0.0, 0.0, -1.0]).all(|(a, b)| (a - b).abs() < 1e-6), "{ahead:?}");
    }

    /// With no listener, the frame's camera listens: at its eye, facing
    /// where it looks.
    #[test]
    fn the_camera_listens_by_default() {
        // A camera at (1, 2, 3) turned 90° to look along +x: its view is the
        // inverse of that pose, columns first.
        let view: Matrix = [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -3.0, -2.0, 1.0, 1.0];
        let ear = Ear::of_frame(None, &view);
        assert_eq!(ear.position, [1.0, 2.0, 3.0]);
        assert_eq!(ear.forward, [1.0, 0.0, 0.0]);
        assert_eq!(ear.up, [0.0, 1.0, 0.0]);
        let listener = Listener {
            position: Vector { x: 4.0, y: 5.0, z: 6.0 },
            forward: Vector { x: 0.0, y: 0.0, z: 1.0 },
            up: Vector { x: 0.0, y: 1.0, z: 0.0 },
            velocity: Vector { x: 0.0, y: 0.0, z: 0.0 },
        };
        assert_eq!(Ear::of_frame(Some(&listener), &view).position, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn doppler_and_occlusion_follow_physical_laws() {
        let ear = Ear { position: [0.0; 3], forward: [0.0, 0.0, -1.0], up: [0.0, 1.0, 0.0], velocity: [0.0; 3] };
        let approaching = Placement::new(&Panner {
            position: Vector { x: 10.0, y: 0.0, z: 0.0 },
            velocity: Vector { x: -34.3, y: 0.0, z: 0.0 },
            doppler: 1.0,
            occlusion: 0.5,
            ..Panner::default()
        })
        .heard_from(&ear);
        // Source approaching at 0.1c -> ratio = 1 / (1 - 0.1) = 10/9.
        assert!((approaching.doppler_ratio(approaching.position) - (10.0 / 9.0)).abs() < 1e-4);
        // Receding at 0.1c -> ratio = 1 / 1.1.
        let receding = Placement { velocity: [34.3, 0.0, 0.0], ..approaching };
        assert!((receding.doppler_ratio(receding.position) - (1.0 / 1.1)).abs() < 1e-4);
        // Supersonic velocity stays finite and bounded.
        let supersonic = Placement { velocity: [-10_000.0, 0.0, 0.0], ..approaching };
        let r = supersonic.doppler_ratio(supersonic.position);
        assert!(r.is_finite() && (0.25..=4.0).contains(&r));
        // Occlusion = 0.5 attenuates direct gain and lowers cutoff below 3.5 kHz.
        assert!((approaching.occlusion_gain() - 0.575).abs() < 1e-4);
        let lp = approaching.occlusion_lowpass(24_000.0);
        assert!(lp > 2000.0 && lp < 3500.0, "expected ~2828 Hz, got {lp}");
    }
}
