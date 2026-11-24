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
    validate_fog(frame)?;
    ensure!(hypot(&frame.light_direction) >= 1e-8, "light direction must be nonzero");
    let aberration = &frame.aberration;
    ensure!(
        [aberration.x, aberration.y].iter().all(|value| value.is_finite() && value.abs() <= 0.25),
        "aberration must be finite and within a screen"
    );
    ensure!(frame.blur >= 0.0 && frame.blur <= 0.5, "blur must be within 0..0.5");
    ensure!(
        frame.motion_blur.strength.is_finite()
            && (0.0..=4.0).contains(&frame.motion_blur.strength)
            && frame.motion_blur.max_radius.is_finite()
            && (0.0..=0.25).contains(&frame.motion_blur.max_radius),
        "motion blur strength must be within 0..4 and max_radius within 0..0.25"
    );
    ensure!(
        frame.grade.sharpness.is_finite()
            && (0.0..=1.0).contains(&frame.grade.sharpness)
            && frame.grade.grain.is_finite()
            && (0.0..=1.0).contains(&frame.grade.grain)
            && frame.grade.flare.is_finite()
            && (0.0..=2.0).contains(&frame.grade.flare)
            && frame.grade.outline_strength.is_finite()
            && (0.0..=1.0).contains(&frame.grade.outline_strength),
        "grade sharpness/grain/outline_strength/flare must be finite and in range"
    );
    for al in &frame.area_lights {
        ensure!(
            finite(&al.position)
                && finite(&al.right)
                && finite(&al.up)
                && finite(&al.color)
                && al.intensity.is_finite()
                && al.intensity >= 0.0
                && al.width.is_finite()
                && al.width >= 0.0
                && al.height.is_finite()
                && al.height >= 0.0
                && al.range.is_finite()
                && al.range >= 0.0,
            "area light parameters must be finite and non-negative"
        );
    }
    for d in &frame.pbr_decals {
        validate_matrix(&d.transform)?;
        ensure!(
            [d.albedo.r, d.albedo.g, d.albedo.b, d.albedo.a, d.normal_strength, d.roughness, d.metallic, d.emissive_strength, d.fade]
                .iter()
                .all(|v| v.is_finite()),
            "PBR decal parameters must be finite"
        );
    }
    for c in &frame.camera_targets {
        validate_matrix(&c.view)?;
        validate_matrix(&c.projection)?;
        ensure!(
            [c.viewport_x, c.viewport_y, c.viewport_w, c.viewport_h].iter().all(|v| v.is_finite()),
            "camera target viewport must be finite"
        );
    }
    ensure!(
        frame.dof.focus_distance.is_finite()
            && frame.dof.focus_distance >= 0.0
            && frame.dof.focus_range.is_finite()
            && frame.dof.focus_range >= 0.0
            && frame.dof.aperture.is_finite()
            && (0.0..=1.0).contains(&frame.dof.aperture)
            && frame.dof.max_blur.is_finite()
            && (0.0..=0.1).contains(&frame.dof.max_blur),
        "depth of field settings must be finite and within range"
    );
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
    let valid_draw_target = |inset: u32, draw: u32| -> bool {
        if inset == 0 {
            (draw as usize) < frame.draws.len()
        } else {
            frame.insets.get((inset - 1) as usize).is_some_and(|ins| (draw as usize) < ins.draws.len())
        }
    };
    for pose in &frame.poses {
        ensure!(valid_draw_target(pose.inset, pose.draw), "a pose layer must name one of the frame's or inset's draws");
        ensure!(
            pose.time.is_finite() && pose.weight.is_finite() && pose.weight >= 0.0,
            "a pose layer needs a finite time and a finite weight of 0 or more"
        );
    }
    for bone in &frame.bone_matrices {
        ensure!(
            valid_draw_target(bone.inset, bone.draw),
            "a bone matrix must name one of the frame's or inset's draws"
        );
        ensure!(
            (bone.joint as usize) < crate::animation::MAX_JOINTS,
            "a bone matrix joint index must lie within MAX_JOINTS"
        );
        validate_matrix(&bone.matrix)?;
    }
    for morph in &frame.morph_weights {
        ensure!(
            valid_draw_target(morph.inset, morph.draw),
            "a morph weight must name one of the frame's or inset's draws"
        );
        ensure!(
            (morph.target as usize) < crate::asset_store::MAX_MORPH_TARGETS && morph.weight.is_finite(),
            "a morph weight needs a valid target index and finite weight"
        );
    }
    for motion in &frame.motions {
        ensure!((motion.draw as usize) < frame.draws.len(), "a motion record must name one of the frame's draws");
        validate_matrix(&motion.previous)?;
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
                emitter.size_end,
                emitter.soft_depth,
                emitter.lit,
                emitter.flipbook_fps
            ]
            .into_iter()
            .chain(colors)
            .all(|value| value.is_finite() && value >= 0.0)
                && [emitter.time, emitter.duration].into_iter().chain(gravity).chain(velocity).all(f32::is_finite),
            "an emitter needs finite values, and rates, lifetimes, sizes and colors of 0 or more"
        );
    }
    for point in &frame.ribbons {
        let color = [point.color.r, point.color.g, point.color.b, point.color.a];
        ensure!(
            finite(&point.position)
                && point.width.is_finite()
                && point.width >= 0.0
                && point.soft_depth.is_finite()
                && point.soft_depth >= 0.0
                && color.iter().all(|v| v.is_finite() && *v >= 0.0),
            "ribbon points need finite position, nonnegative width/soft_depth, and nonnegative color"
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

/// The mist and the local fog volumes: finite, and within what the shaders
/// divide by.
fn validate_fog(frame: &RenderFrame) -> Result<()> {
    let finite = |v: &pen_protocol::Vector| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
    let fog = &frame.fog;
    let scalars = [
        fog.radius,
        fog.density,
        fog.height,
        fog.edge,
        fog.distance,
        fog.anisotropy,
        fog.sun,
        fog.ambient,
        fog.sky,
        fog.noise,
        fog.noise_scale,
    ];
    ensure!(scalars.iter().all(|value| value.is_finite()), "fog must contain finite float32 values");
    ensure!(finite(&fog.albedo) && finite(&fog.drift), "fog must contain finite float32 values");
    ensure!(
        fog.radius >= 0.0
            && fog.density >= 0.0
            && fog.edge >= 0.0
            && fog.distance >= 0.0
            && fog.sun >= 0.0
            && fog.ambient >= 0.0,
        "fog's radius, density, edge, distance, sun and ambient must not be negative"
    );
    ensure!(fog.height > 0.0 && fog.noise_scale > 0.0, "fog needs a positive height and noise scale");
    ensure!(
        (-1.0..=1.0).contains(&fog.anisotropy) && (0.0..=1.0).contains(&fog.sky) && (0.0..=1.0).contains(&fog.noise),
        "fog's anisotropy must be within -1..1, its sky and noise within 0..1"
    );
    ensure!(fog.albedo.x >= 0.0 && fog.albedo.y >= 0.0 && fog.albedo.z >= 0.0, "fog's albedo must not be negative");
    for volume in &frame.fog_volumes {
        let vectors = [volume.center, volume.extent, volume.albedo, volume.emission];
        ensure!(
            vectors.iter().all(finite) && volume.density.is_finite() && volume.falloff.is_finite(),
            "fog volumes must contain finite float32 values"
        );
        ensure!(
            volume.extent.x > 0.0 && volume.extent.y > 0.0 && volume.extent.z > 0.0 && volume.density >= 0.0,
            "a fog volume needs a positive size and a density that is not negative"
        );
        ensure!((0.0..=1.0).contains(&volume.falloff), "a fog volume's falloff must be within 0..1");
        ensure!(
            volume.albedo.x >= 0.0
                && volume.albedo.y >= 0.0
                && volume.albedo.z >= 0.0
                && volume.emission.x >= 0.0
                && volume.emission.y >= 0.0
                && volume.emission.z >= 0.0,
            "a fog volume's albedo and emission must not be negative"
        );
    }
    Ok(())
}
