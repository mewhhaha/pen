// What a frame must be for the renderer to draw it: finite matrices and
// lighting, invertible transforms, settings in range and field patches inside
// their fields.
use anyhow::{Result, ensure};
use pen_protocol::boundary::{MAX_DRAW_EXTRA, MAX_FIELD};
use pen_protocol::{Matrix, RenderFrame, Vector};

use crate::matrix::narrow;

pub fn validate_frame(frame: &RenderFrame) -> Result<()> {
    validate_matrix(&frame.view)?;
    validate_matrix(&frame.projection)?;
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
    ensure!(hypot(&frame.light_direction) >= 1e-8, "light direction must be nonzero");
    let aberration = &frame.aberration;
    ensure!(
        [aberration.x, aberration.y].iter().all(|value| value.is_finite() && value.abs() <= 0.25),
        "aberration must be finite and within a screen"
    );
    ensure!(frame.blur >= 0.0 && frame.blur <= 0.5, "blur must be within 0..0.5");
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
