// What a frame must be for the renderer to draw it: finite matrices and
// lighting, invertible transforms, settings in range and field patches inside
// their fields.
use anyhow::{Result, ensure};
use pen_protocol::boundary::{MAX_DRAW_EXTRA, MAX_FIELD};
use pen_protocol::{Exposure, Grade, Matrix, RenderFrame, Vector};

use crate::matrix::narrow;

pub fn validate_frame(frame: &RenderFrame) -> Result<()> {
    validate_matrix(&frame.view)?;
    validate_matrix(&frame.projection)?;
    crate::lights::validate(&frame.lights)?;
    super::cascades::validate(frame)?;
    for patch in &frame.fields {
        let most = u32::try_from(MAX_FIELD)?;
        ensure!(
            patch.width <= most
                && patch.height <= most
                && patch.x <= patch.width
                && patch.z <= patch.height
                && patch.columns <= patch.width - patch.x
                && patch.rows <= patch.height - patch.z
                && patch.values.len() == patch.columns as usize * patch.rows as usize
                && patch.values.iter().all(|value| value.is_finite()),
            "field patch must lie inside its field"
        );
    }
    let finite = |vector: &Vector| [vector.x, vector.y, vector.z].iter().all(|value| value.is_finite());
    for vector in [&frame.ambient, &frame.light_color, &frame.light_direction] {
        ensure!(finite(vector), "render lighting must contain finite float32 values");
    }
    let sky = &frame.sky;
    for vector in [&sky.zenith, &sky.horizon, &sky.ground, &sky.tint]
        .into_iter()
        .chain([&sky.sh0, &sky.sh1, &sky.sh2, &sky.sh3, &sky.sh4, &sky.sh5, &sky.sh6, &sky.sh7, &sky.sh8])
    {
        ensure!(finite(vector), "the sky's colors and coefficients must be finite float32 values");
    }
    let volume = &frame.probe_volume;
    let probes = [volume.nx, volume.ny, volume.nz].iter().map(|&n| n as usize).product::<usize>();
    ensure!(
        frame.probes.is_empty() || (frame.probes.len() == probes && probes <= 16384),
        "probes must fill the probe volume, at most 16384"
    );
    ensure!(
        probes == 0 || (volume.spacing.is_finite() && volume.spacing > 0.0 && finite(&volume.origin)),
        "probe volume needs a positive finite spacing and origin"
    );
    for probe in &frame.probes {
        let terms = [probe.sh0, probe.sh1, probe.sh2, probe.sh3, probe.sh4, probe.sh5, probe.sh6, probe.sh7, probe.sh8];
        ensure!(terms.iter().all(finite) && probe.valid.is_finite(), "probe light must contain finite float32 values");
    }
    for probe in &frame.reflection_probes {
        let vectors = [probe.position, probe.center, probe.extent, probe.parallax_center, probe.parallax_extent];
        ensure!(
            vectors.iter().all(finite) && probe.blend.is_finite() && probe.far.is_finite(),
            "reflection probes must contain finite float32 values"
        );
        ensure!(
            probe.extent.x >= 0.0 && probe.extent.y >= 0.0 && probe.extent.z >= 0.0 && probe.blend >= 0.0,
            "a reflection probe's box and blend distance must not be negative"
        );
        ensure!(probe.far > 0.0, "a reflection probe must see some distance");
    }
    ensure!(hypot(&frame.light_direction) >= 1e-8, "light direction must be nonzero");
    let aberration = &frame.aberration;
    ensure!(
        [aberration.x, aberration.y].iter().all(|value| value.is_finite() && value.abs() <= 0.25),
        "aberration must be finite and within a screen"
    );
    ensure!(frame.blur >= 0.0 && frame.blur <= 0.5, "blur must be within 0..0.5");
    validate_exposure(&frame.exposure)?;
    validate_grade(&frame.grade)?;
    let clear = &frame.clear;
    ensure!(
        [clear.r, clear.g, clear.b, clear.a].iter().all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
        "clear color must have 0..1 channels"
    );
    for draw in &frame.draws {
        validate_matrix(&draw.transform)?;
        if let Some(parameters) = &draw.parameters {
            ensure!(
                parameters.iter().all(|value| value.is_finite()),
                "draw parameters must contain four finite float32 values"
            );
        }
        if let Some(extra) = &draw.extra {
            ensure!(
                extra.len() % 4 == 0
                    && extra.len() <= MAX_DRAW_EXTRA * 4
                    && extra.iter().all(|value| value.is_finite()),
                "draw extra data must be up to 32 vectors of finite float32 values"
            );
        }
        determinant_of(&draw.transform)?;
    }
    for decal in &frame.decals {
        validate_matrix(&decal.transform)?;
        determinant_of(&decal.transform)?;
        ensure!(decal.opacity >= 0.0 && decal.opacity <= 1.0, "decal opacity must be within 0..1");
    }
    for pose in &frame.poses {
        ensure!((pose.draw as usize) < frame.draws.len(), "a pose layer must name one of the frame's draws");
        ensure!(
            pose.time.is_finite() && pose.weight.is_finite() && pose.weight >= 0.0,
            "a pose layer needs a finite time and a finite weight of 0 or more"
        );
    }
    for emitter in &frame.emitters {
        validate_matrix(&emitter.transform)?;
        let colors = [&emitter.color_start, &emitter.color_end]
            .into_iter()
            .flat_map(|color| [color.r, color.g, color.b, color.a]);
        let gravity = [emitter.gravity.x, emitter.gravity.y, emitter.gravity.z];
        let velocity = [emitter.velocity.x, emitter.velocity.y, emitter.velocity.z];
        ensure!(
            [
                emitter.rate,
                emitter.lifetime,
                emitter.radius,
                emitter.spread,
                emitter.drag,
                emitter.size_start,
                emitter.size_end
            ]
            .into_iter()
            .chain(colors)
            .all(|value| value.is_finite() && value >= 0.0)
                && [emitter.time, emitter.duration].into_iter().chain(gravity).chain(velocity).all(f32::is_finite),
            "an emitter needs finite values, and rates, lifetimes, sizes and colors of 0 or more"
        );
    }
    for solid in &frame.solids {
        for face in &solid.faces {
            let normal = &face.normal;
            ensure!(
                [normal.x, normal.y, normal.z, face.distance].iter().all(|value| value.is_finite())
                    && hypot(normal) >= 1e-6,
                "solid faces need a plane and a material"
            );
        }
    }
    Ok(())
}

pub(super) fn hypot(vector: &Vector) -> f64 {
    let [x, y, z] = [vector.x, vector.y, vector.z].map(f64::from);
    (x * x + y * y + z * z).sqrt()
}

pub(super) fn validate_matrix(matrix: &Matrix) -> Result<()> {
    ensure!(matrix.iter().all(|value| value.is_finite()), "render matrices must contain 16 finite float32 values");
    Ok(())
}

pub(super) fn determinant_of(matrix: &Matrix) -> Result<f64> {
    let m = matrix.map(f64::from);
    let (a, b, c) = (m[0], m[1], m[2]);
    let (d, e, f) = (m[4], m[5], m[6]);
    let (g, h, i) = (m[8], m[9], m[10]);
    let determinant = a * (e * i - f * h) - d * (b * i - c * h) + g * (b * f - c * e);
    ensure!(determinant.is_finite() && determinant.abs() >= 1e-12, "draw transform must be invertible");
    Ok(determinant)
}

pub(super) fn normal_matrix(matrix: &Matrix) -> Result<[f32; 16]> {
    let m = matrix.map(f64::from);
    let (a, b, c) = (m[0], m[1], m[2]);
    let (d, e, f) = (m[4], m[5], m[6]);
    let (g, h, i) = (m[8], m[9], m[10]);
    let determinant = determinant_of(matrix)?;
    Ok([
        (e * i - f * h) / determinant,
        (f * g - d * i) / determinant,
        (d * h - e * g) / determinant,
        0.0,
        (c * h - b * i) / determinant,
        (a * i - c * g) / determinant,
        (b * g - a * h) / determinant,
        0.0,
        (b * f - c * e) / determinant,
        (c * d - a * f) / determinant,
        (a * e - b * d) / determinant,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    .map(narrow))
}

/// The exposure's settings: finite, within the ranges the schema documents.
fn validate_exposure(exposure: &Exposure) -> Result<()> {
    let stops = -crate::exposure::MAX_STOPS..=crate::exposure::MAX_STOPS;
    ensure!(
        stops.contains(&exposure.compensation)
            && stops.contains(&exposure.min)
            && stops.contains(&exposure.max)
            && exposure.min <= exposure.max,
        "exposure compensation and limits must be finite stops within -16..16, the least no more than the most"
    );
    ensure!(
        [exposure.brighten, exposure.darken].iter().all(|rate| (0.0..=1000.0).contains(rate))
            && (0.0..=1.0e6).contains(&exposure.dt),
        "exposure speeds must be rates of 0..1000 a second, and its time 0 or more"
    );
    ensure!(
        (0.0..=1.0).contains(&exposure.low)
            && (0.0..=1.0).contains(&exposure.high)
            && exposure.low < exposure.high
            && (0.0..=1.0).contains(&exposure.centre),
        "exposure percentiles must be within 0..1, the low below the high, and its centre weight within 0..1"
    );
    ensure!(
        (-64.0..=64.0).contains(&exposure.range_low)
            && (-64.0..=64.0).contains(&exposure.range_high)
            && exposure.range_high > exposure.range_low
            && exposure.range_high - exposure.range_low <= 32.0,
        "the exposure histogram's range must run up by at most 32 stops, within -64..64"
    );
    ensure!(
        exposure.target.is_finite() && exposure.target > 0.0 && exposure.target <= 1.0e4,
        "the exposure's middle grey must be a positive luminance"
    );
    Ok(())
}

/// The grade's settings: finite, within the ranges the schema documents.
fn validate_grade(grade: &Grade) -> Result<()> {
    let within = |value: f32, low: f32, high: f32| (low..=high).contains(&value);
    ensure!(
        within(grade.temperature, -1.0, 1.0)
            && within(grade.tint, -1.0, 1.0)
            && within(grade.exposure, -8.0, 8.0)
            && within(grade.contrast, 0.0, 4.0)
            && within(grade.pivot, 1.0e-4, 1.0e3)
            && within(grade.saturation, 0.0, 4.0)
            && within(grade.vibrance, -1.0, 1.0),
        "the grade's white balance, exposure, contrast, pivot, saturation and vibrance must be within their ranges"
    );
    let channels = |vector: &Vector, low: f32, high: f32| {
        [vector.x, vector.y, vector.z].iter().all(|&value| within(value, low, high))
    };
    ensure!(
        channels(&grade.lift, -0.5, 0.5) && channels(&grade.gamma, 0.1, 4.0) && channels(&grade.gain, 0.0, 8.0),
        "the grade's lift must be within -0.5..0.5, its gamma within 0.1..4 and its gain within 0..8"
    );
    ensure!(
        within(grade.lut_blend, 0.0, 1.0)
            && within(grade.lut_weight, 0.0, 1.0)
            && within(grade.vignette, 0.0, 1.0)
            && within(grade.vignette_radius, 0.0, 1.0),
        "the grade's table blend and weight and its vignette must be within 0..1"
    );
    Ok(())
}
