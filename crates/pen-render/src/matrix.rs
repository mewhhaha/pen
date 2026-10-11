// Column-major 4x4 matrices, as WGSL and the render packet lay them out.
// The arithmetic is in f64, as the TS does it in JavaScript numbers, and only
// the results are narrowed to f32 for the GPU.

pub(crate) const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

fn wide(m: &[f32; 16]) -> [f64; 16] {
    m.map(f64::from)
}

fn narrow(m: [f64; 16]) -> [f32; 16] {
    m.map(|value| value as f32)
}

/// Column-major 4x4 product `a * b`.
pub(crate) fn multiply(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let (a, b) = (wide(a), wide(b));
    let mut out = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[column * 4 + k];
            }
            out[column * 4 + row] = sum;
        }
    }
    narrow(out)
}

/// The inverse of a column-major 4x4 matrix, or the identity if it has none.
pub(crate) fn invert(m: &[f32; 16]) -> [f32; 16] {
    let [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = wide(m);
    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;
    let determinant = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    // `!determinant` in the TS: zero or NaN.
    if determinant == 0.0 || determinant.is_nan() {
        return IDENTITY;
    }
    let d = 1.0 / determinant;
    narrow([
        (a11 * b11 - a12 * b10 + a13 * b09) * d,
        (a02 * b10 - a01 * b11 - a03 * b09) * d,
        (a31 * b05 - a32 * b04 + a33 * b03) * d,
        (a22 * b04 - a21 * b05 - a23 * b03) * d,
        (a12 * b08 - a10 * b11 - a13 * b07) * d,
        (a00 * b11 - a02 * b08 + a03 * b07) * d,
        (a32 * b02 - a30 * b05 - a33 * b01) * d,
        (a20 * b05 - a22 * b02 + a23 * b01) * d,
        (a10 * b10 - a11 * b08 + a13 * b06) * d,
        (a01 * b08 - a00 * b10 - a03 * b06) * d,
        (a30 * b04 - a31 * b02 + a33 * b00) * d,
        (a21 * b02 - a20 * b04 - a23 * b00) * d,
        (a11 * b07 - a10 * b09 - a12 * b06) * d,
        (a00 * b09 - a01 * b07 + a02 * b06) * d,
        (a31 * b01 - a30 * b03 - a32 * b00) * d,
        (a20 * b03 - a21 * b01 + a22 * b00) * d,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_undoes_a_transform() {
        let m = [2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0, 2.0, 3.0, 1.0];
        let product = multiply(&m, &invert(&m));
        for (got, want) in product.iter().zip(IDENTITY) {
            assert!((got - want).abs() < 1e-6);
        }
        assert_eq!(invert(&[0.0; 16]), IDENTITY);
    }
}
