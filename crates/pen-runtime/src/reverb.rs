#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "audio sample math")]
use std::f64::consts::TAU;

/// The response's length, in seconds.
pub const SECONDS: f64 = 1.6;
/// How long its tail takes to fall 60 dB.
const DECAY: f64 = 1.5;
/// Silence before it starts, and how fast it comes in.
const PREDELAY: f64 = 0.012;
const ONSET: f64 = 0.004;
/// The last of it fades to nothing over this many seconds.
const TAPER: f64 = 0.08;
/// Each channel's response has this much energy (sum of squares).
const ENERGY: f64 = 1.0;
/// ln(0.001): a decay of 60 dB.
const SIXTY_DB: f64 = -6.907_755_278_982_137;
/// The two channels' noise seeds.
const SEEDS: [u32; 2] = [0x2545_f491, 0x9e37_79b9];

/// The impulse response at `rate` samples a second, left and right: noise
/// whose highs die away faster than its lows (a one-pole lowpass closing from
/// 8.2 kHz to 1.2 kHz), decaying 60 dB in `DECAY` seconds after a short
/// predelay, normalized so each channel's energy is `ENERGY`. Generated in
/// f64 with deterministic samples shared by the native and browser hosts.
pub fn impulse_response(rate: f64) -> [Vec<f32>; 2] {
    impulse_response_for(rate, 0.0, 0.0)
}

/// Parameterized room impulse response at `rate` Hz:
/// - `decay`: RT60 decay seconds (0 or non-finite uses the default 1.5 s room; clamped to 0.15..6.0 s).
/// - `damping`: high-frequency damping in 0..1 (0 or non-finite uses the default bright/dark decay curve;
///   higher values lower the tail's lowpass cutoff and shorten HF decay).
pub fn impulse_response_for(rate: f64, decay: f64, damping: f64) -> [Vec<f32>; 2] {
    let custom_decay = decay.is_finite() && decay > 0.0;
    let custom_damping = damping.is_finite() && damping > 0.0;
    let decay = if custom_decay { decay.clamp(0.15, 6.0) } else { DECAY };
    let seconds = if custom_decay { (decay + 0.1).clamp(0.25, 6.2) } else { SECONDS };
    let damp = if custom_damping { damping.clamp(0.0, 1.0) } else { -1.0 };
    let length = (seconds * rate).round().max(1.0) as usize;
    let mut channels = [vec![0.0f64; length], vec![0.0f64; length]];
    let mut energy = 0.0f64;
    for (channel, seed) in channels.iter_mut().zip(SEEDS) {
        let mut state = seed;
        let mut low = 0.0f64;
        for (index, value) in channel.iter_mut().enumerate() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = f64::from(state) / 4_294_967_296.0 * 2.0 - 1.0;
            let t = index as f64 / rate;
            let cutoff = if damp < 0.0 {
                1200.0 + 7000.0 * (-t / 0.3).exp()
            } else {
                // High damping closes down faster toward 400 Hz; low damping stays bright up to 11 kHz.
                let floor_hz = 2400.0 - 2000.0 * damp;
                let span_hz = 9000.0 * (1.0 - 0.7 * damp);
                let tau_hf = 0.55 - 0.42 * damp;
                floor_hz + span_hz * (-t / tau_hf).exp()
            };
            low += (1.0 - (-TAU * cutoff / rate).exp()) * (noise - low);
            let since = t - PREDELAY;
            let envelope = if since < 0.0 {
                0.0
            } else {
                (SIXTY_DB * since / decay).exp() * (since / ONSET).min(1.0) * ((seconds - t) / TAPER).min(1.0)
            };
            *value = low * envelope;
            energy += *value * *value;
        }
    }
    let scale = if energy > 0.0 { (ENERGY / (energy / 2.0)).sqrt() } else { 1.0 };
    channels.map(|channel| channel.into_iter().map(|value| (value * scale) as f32).collect())
}

/// Blended acoustic reverb state for the listener's current world position across
/// any overlapping [`pen_protocol::ReverbZone`] volumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoneMix {
    pub decay: f32,
    pub damping: f32,
    pub wet_scale: f32,
    pub dominant_zone: u32,
}

/// Signed distance from `point` to an axis-aligned box `(center, half_extents)`.
/// Zero inside the box; positive Euclidean distance outside.
fn box_distance(point: [f32; 3], center: [f32; 3], half: [f32; 3]) -> f64 {
    let mut sq = 0.0f64;
    for i in 0..3 {
        let ext = f64::from(half[i]).max(0.0);
        let d = (f64::from(point[i]) - f64::from(center[i])).abs() - ext;
        if d > 0.0 {
            sq += d * d;
        }
    }
    sq.sqrt()
}

/// Smooth weight in `0..=1` for `point` relative to `zone` (1 inside box, smoothstep
/// falloff over `zone.blend` outside).
pub fn zone_weight(point: [f32; 3], zone: &pen_protocol::ReverbZone) -> f64 {
    let pos = [zone.position.x, zone.position.y, zone.position.z];
    let ext = [zone.half_extents.x, zone.half_extents.y, zone.half_extents.z];
    if !pos.iter().chain(&ext).all(|v| v.is_finite()) {
        return 0.0;
    }
    let dist = box_distance(point, pos, ext);
    if dist <= 0.0 {
        return 1.0;
    }
    let blend = if zone.blend.is_finite() { f64::from(zone.blend).max(0.0) } else { 0.0 };
    if blend <= 1e-6 || dist >= blend {
        return 0.0;
    }
    let t = 1.0 - (dist / blend).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Smoothly blends spatial [`pen_protocol::ReverbZone`] rooms around `listener_pos`,
/// falling back cleanly toward `(base_decay, base_damping)` outside all zones.
pub fn blend_zones(
    listener_pos: [f32; 3],
    base_decay: f32,
    base_damping: f32,
    zones: &[pen_protocol::ReverbZone],
) -> ZoneMix {
    let fallback_decay = if base_decay.is_finite() && base_decay > 0.0 { base_decay } else { 0.0 };
    let fallback_damp = if base_damping.is_finite() { base_damping.clamp(0.0, 1.0) } else { 0.0 };
    if zones.is_empty() {
        return ZoneMix { decay: fallback_decay, damping: fallback_damp, wet_scale: 1.0, dominant_zone: 0 };
    }
    let mut sum_w = 0.0f64;
    let mut max_w = 0.0f64;
    let mut dominant_zone = 0u32;
    let mut decay_acc = 0.0f64;
    let mut damp_acc = 0.0f64;
    let mut wet_acc = 0.0f64;
    for zone in zones {
        let w = zone_weight(listener_pos, zone);
        if w <= 0.0 {
            continue;
        }
        let z_decay = if zone.decay.is_finite() && zone.decay > 0.0 {
            f64::from(zone.decay.clamp(0.15, 6.0))
        } else {
            f64::from(if fallback_decay > 0.0 { fallback_decay } else { DECAY as f32 })
        };
        let z_damp = if zone.damping.is_finite() {
            f64::from(zone.damping.clamp(0.0, 1.0))
        } else {
            f64::from(fallback_damp)
        };
        let z_wet = if zone.wet.is_finite() && zone.wet > 0.0 {
            f64::from(zone.wet.clamp(0.0, 2.0))
        } else {
            1.0
        };
        sum_w += w;
        decay_acc += w * z_decay;
        damp_acc += w * z_damp;
        wet_acc += w * z_wet;
        if w > max_w {
            max_w = w;
            dominant_zone = zone.id;
        }
    }
    if sum_w <= 1e-6 {
        return ZoneMix { decay: fallback_decay, damping: fallback_damp, wet_scale: 1.0, dominant_zone: 0 };
    }
    // When partially in a blend margin (sum_w < 1), smoothly mix with the base room.
    let base_w = (1.0 - sum_w).max(0.0);
    let total_w = sum_w + base_w;
    let base_d = f64::from(if fallback_decay > 0.0 { fallback_decay } else { DECAY as f32 });
    let decay = ((decay_acc + base_w * base_d) / total_w) as f32;
    let damping = ((damp_acc + base_w * f64::from(fallback_damp)) / total_w) as f32;
    let wet_scale = ((wet_acc + base_w * 1.0) / total_w) as f32;
    ZoneMix { decay, damping, wet_scale, dominant_zone }
}

/// Computes smooth cross-room portal attenuation in `0..=1` and the **source's**
/// own reverb wet multiplier based on source and listener membership across `zones`.
/// Both vary continuously through zone blend margins without abrupt edge jumps.
pub fn source_zone_acoustics(
    source_pos: [f32; 3],
    explicit_zone: u32,
    listener_pos: [f32; 3],
    zones: &[pen_protocol::ReverbZone],
) -> (f32, f32) {
    if zones.is_empty() {
        return (0.0, 1.0);
    }
    let mut portal = 0.0f64;
    let mut wet_sum = 0.0f64;
    let mut wet_w = 0.0f64;
    for z in zones {
        let wl = zone_weight(listener_pos, z);
        let ws = if explicit_zone != 0 {
            if z.id == explicit_zone { 1.0 } else { 0.0 }
        } else {
            zone_weight(source_pos, z)
        };
        if ws > 0.0 {
            let z_wet = if z.wet.is_finite() && z.wet > 0.0 {
                f64::from(z.wet.clamp(0.0, 2.0))
            } else {
                1.0
            };
            wet_sum += ws * z_wet;
            wet_w += ws;
        }
        if z.id != 0 && z.portal_attenuation.is_finite() {
            // Smooth separation: 0 when source & listener have equal membership in `z`,
            // smoothly ramping to 1 when one is inside (`w=1`) and the other is outside (`w=0`).
            let sep = (wl - ws).abs();
            portal = portal.max(sep * f64::from(z.portal_attenuation.clamp(0.0, 1.0)));
        }
    }
    let base_w = (1.0 - wet_w).max(0.0);
    let wet_scale = ((wet_sum + base_w) / (wet_w + base_w)) as f32;
    (portal.clamp(0.0, 1.0) as f32, wet_scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_decay_and_damping_shape_room_tail() {
        let rate = 24_000.0;
        let short = impulse_response_for(rate, 0.3, 0.2);
        let long = impulse_response_for(rate, 2.5, 0.2);
        assert!(long[0].len() > short[0].len() * 4, "longer decay must produce a longer IR");

        let bright = impulse_response_for(rate, 1.0, 0.05);
        let dark = impulse_response_for(rate, 1.0, 0.95);
        // First-difference RMS measures high-frequency content in the late tail.
        let hf = |ir: &[f32]| -> f64 {
            let tail = &ir[ir.len() / 4..ir.len() * 3 / 4];
            let sum: f64 = tail.windows(2).map(|w| f64::from(w[1] - w[0]).powi(2)).sum();
            (sum / tail.len() as f64).sqrt()
        };
        assert!(hf(&bright[0]) > hf(&dark[0]) * 1.5, "higher damping must attenuate tail highs");
    }

    #[test]
    fn reverb_zones_blend_smoothly_and_apply_continuous_portal_and_source_wet() {
        let v = |x, y, z| pen_protocol::Vector { x, y, z };
        let zones = vec![
            pen_protocol::ReverbZone {
                id: 1,
                position: v(0.0, 0.0, 0.0),
                half_extents: v(5.0, 3.0, 5.0),
                blend: 4.0,
                decay: 0.5,
                damping: 0.7,
                wet: 0.8,
                portal_attenuation: 0.45,
            },
            pen_protocol::ReverbZone {
                id: 2,
                position: v(14.0, 0.0, 0.0),
                half_extents: v(5.0, 3.0, 5.0),
                blend: 4.0,
                decay: 3.5,
                damping: 0.1,
                wet: 1.5,
                portal_attenuation: 0.6,
            },
        ];
        let in_small = blend_zones([0.0, 0.0, 0.0], 1.5, 0.3, &zones);
        assert_eq!(in_small.dominant_zone, 1);
        assert!((in_small.decay - 0.5).abs() < 1e-4);

        // Midpoint x=7 sits 2m outside BOTH boxes (inside both 4m blend margins) -> smooth 50/50 blend.
        let mid = blend_zones([7.0, 0.0, 0.0], 1.5, 0.3, &zones);
        assert!((mid.decay - 2.0).abs() < 0.05, "expected ~2.0s blend, got {}", mid.decay);
        assert!((mid.damping - 0.4).abs() < 0.05, "expected ~0.4 damping, got {}", mid.damping);

        // Same zone -> 0 portal occlusion & source's own wet (0.8).
        let (p_same, w_same) = source_zone_acoustics([1.0, 0.0, 0.0], 0, [0.0, 0.0, 0.0], &zones);
        assert_eq!(p_same, 0.0);
        assert!((w_same - 0.8).abs() < 1e-5);
        // Cross-zone -> portal occlusion 0.6 & source's own zone 2 wet (1.5), NOT listener's (0.8)!
        let (p_cross, w_cross) = source_zone_acoustics([14.0, 0.0, 0.0], 0, [0.0, 0.0, 0.0], &zones);
        assert!((p_cross - 0.6).abs() < 1e-5, "expected 0.6 portal occlusion, got {p_cross}");
        assert!((w_cross - 1.5).abs() < 1e-5, "wet must follow source zone, got {w_cross}");
        // Barely entering blend margin (dist = 3.99 of 4.0) must have tiny portal occlusion, no edge jump!
        let (p_edge, _) = source_zone_acoustics([14.0, 0.0, 0.0], 0, [8.99, 0.0, 0.0], &zones);
        let (p_out, _) = source_zone_acoustics([14.0, 0.0, 0.0], 0, [8.999, 0.0, 0.0], &zones);
        assert!((p_edge - p_out).abs() < 0.01, "portal attenuation must vary continuously at blend edge");
    }
}
