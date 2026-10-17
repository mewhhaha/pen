//! Vertex-animation textures: a skinned mesh's clips baked, frame by frame,
//! into a texture of where each vertex is and which way it faces, which a
//! shader reads by vertex index. Drawing a crowd then costs no joints: every
//! member is an instance of one plain instanced draw, its parameters saying
//! which clip it plays and how far in.
//!
//! The texture is a packed texture of raw RGBA8 texels (`.gtex`, kind
//! "raw"), read with `textureLoad` in a row-major run of `width` texels a
//! row. Its first texels each hold a little-endian u32 or f32:
//!
//! - 0: `VAT1`; 1: the vertex count; 2: the clip count;
//! - 3 to 5: the low corner of the box every frame's vertices lie in; 6 to
//!   8: the box's size;
//! - then three for each clip: its first frame, its frame count and its
//!   duration in seconds (an f32). Frame `f` of a clip of `n` shows it
//!   `f / n` of the way through; the clip loops.
//!
//! Frames follow, each vertex of each in two texels: the high bytes of its
//! place in the box (x, y, z, 16 bits each across the box) and its normal's
//! first octahedral byte, then the low bytes and the second.
use anyhow::{Result, ensure};
use pen_render::animation::{Layer, Rig};

use crate::Geometry;

pub const MAGIC: u32 = u32::from_le_bytes(*b"VAT1");
/// Texels a row: within every WebGPU device's texture limits.
pub const WIDTH: usize = 2048;
const MOST_ROWS: usize = 8192;

pub struct Baked {
    /// The packed texture's bytes.
    pub texture: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Each clip's first frame and frame count.
    pub clips: Vec<(u32, u32)>,
}

/// Bakes every clip of `rig` moving `geometry` at `fps` frames a second.
pub fn bake(geometry: &Geometry, rig: &Rig, fps: f32) -> Result<Baked> {
    ensure!(fps.is_finite() && fps > 0.0, "frames a second must be more than 0");
    let vertices = geometry.positions.len();
    let mut clips = Vec::new();
    let mut frames: Vec<([f32; 3], [f32; 3])> = Vec::new();
    let mut palette = Vec::new();
    for (index, clip) in rig.clips.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a small, finite frame count")]
        let count = (f64::from(clip.duration) * f64::from(fps)).round().max(1.0) as u32;
        let first = u32::try_from(frames.len() / vertices.max(1))?;
        clips.push((first, count));
        for frame in 0..count {
            let time = clip.duration * frame as f32 / count as f32;
            rig.palette(&[Layer { clip: u32::try_from(index)?, time, weight: 1.0 }], &mut palette);
            for vertex in 0..vertices {
                frames.push(skinned(geometry, vertex, &palette));
            }
        }
    }
    ensure!(!clips.is_empty(), "the rig has no clips");
    // The box every frame's vertices lie in.
    let mut low = [f32::INFINITY; 3];
    let mut high = [f32::NEG_INFINITY; 3];
    for (place, _) in &frames {
        for axis in 0..3 {
            low[axis] = low[axis].min(place[axis]);
            high[axis] = high[axis].max(place[axis]);
        }
    }
    let size = [0, 1, 2].map(|axis| (high[axis] - low[axis]).max(1e-6));
    let mut words: Vec<u32> = vec![MAGIC, u32::try_from(vertices)?, u32::try_from(clips.len())?];
    words.extend(low.iter().chain(&size).map(|value| value.to_bits()));
    for ((first, count), clip) in clips.iter().zip(&rig.clips) {
        words.extend([*first, *count, clip.duration.to_bits()]);
    }
    let mut texels: Vec<[u8; 4]> = words.iter().map(|word| word.to_le_bytes()).collect();
    for (place, normal) in &frames {
        let quantized = [0, 1, 2].map(|axis| {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "0..=65535")]
            let value = ((place[axis] - low[axis]) / size[axis] * 65535.0).round().clamp(0.0, 65535.0) as u16;
            value.to_be_bytes()
        });
        let [u, v] = octahedral(*normal);
        texels.push([quantized[0][0], quantized[1][0], quantized[2][0], u]);
        texels.push([quantized[0][1], quantized[1][1], quantized[2][1], v]);
    }
    let width = texels.len().min(WIDTH);
    let height = texels.len().div_ceil(width);
    ensure!(
        height <= MOST_ROWS,
        "{} frames of {vertices} vertices need {height} rows of {WIDTH} texels, more than {MOST_ROWS}: bake fewer frames a second",
        frames.len() / vertices.max(1)
    );
    texels.resize(width * height, [0; 4]);
    let (width, height) = (u32::try_from(width)?, u32::try_from(height)?);
    let texture = crate::gtex::raw(width, height, texels.as_flattened())?;
    Ok(Baked { texture, width, height, clips })
}

/// Vertex `vertex` of `geometry` moved by the joint matrices `palette`:
/// where it is and which way it faces.
pub fn skinned(geometry: &Geometry, vertex: usize, palette: &[[f32; 16]]) -> ([f32; 3], [f32; 3]) {
    let [x, y, z] = geometry.positions[vertex].map(f64::from);
    let [nx, ny, nz] = geometry.normals[vertex].map(f64::from);
    let mut place = [0.0f64; 3];
    let mut normal = [0.0f64; 3];
    for (&joint, &weight) in geometry.joints[vertex].iter().zip(&geometry.weights[vertex]) {
        if weight <= 0.0 {
            continue;
        }
        let m = palette[usize::from(joint)].map(f64::from);
        let weight = f64::from(weight);
        for row in 0..3 {
            place[row] += weight * (m[row] * x + m[4 + row] * y + m[8 + row] * z + m[12 + row]);
            normal[row] += weight * (m[row] * nx + m[4 + row] * ny + m[8 + row] * nz);
        }
    }
    let length = normal.iter().map(|value| value * value).sum::<f64>().sqrt();
    let normal = if length > 1e-12 { normal.map(|value| value / length) } else { [0.0, 1.0, 0.0] };
    #[allow(clippy::cast_possible_truncation, reason = "the f32 a texture holds")]
    (place.map(|value| value as f32), normal.map(|value| value as f32))
}

/// A unit vector folded onto the octahedron and its two coordinates as
/// bytes, -1..1 as 0..255.
pub fn octahedral(normal: [f32; 3]) -> [u8; 2] {
    let [x, y, z] = normal.map(f64::from);
    let sum = x.abs() + y.abs() + z.abs();
    let (mut u, mut v) = if sum > 1e-12 { (x / sum, y / sum) } else { (0.0, 0.0) };
    if z < 0.0 {
        (u, v) = ((1.0 - v.abs()) * u.signum(), (1.0 - u.abs()) * v.signum());
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "0..=255")]
    [u, v].map(|value| ((value * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_render::animation::{Channel, Clip, Interpolation, Joint, Node, Path, Transform};

    /// A stick of two joints bending at its middle over one second.
    fn bending() -> (Geometry, Rig) {
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let inverse = |y: f32| [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -y, 0.0, 1.0];
        let rig = Rig {
            nodes: vec![
                Node { name: "root".into(), parent: None, rest: Transform::IDENTITY },
                Node {
                    name: "middle".into(),
                    parent: Some(0),
                    rest: Transform { translation: [0.0, 1.0, 0.0], ..Transform::IDENTITY },
                },
            ],
            joints: vec![Joint { node: 0, inverse_bind: inverse(0.0) }, Joint { node: 1, inverse_bind: inverse(1.0) }],
            clips: vec![Clip {
                name: "bend".into(),
                duration: 1.0,
                channels: vec![Channel {
                    node: 1,
                    path: Path::Rotation,
                    interpolation: Interpolation::Linear,
                    times: vec![0.0, 1.0],
                    values: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, half, half],
                }],
            }],
        };
        let geometry = Geometry {
            positions: vec![[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.5, 2.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0; 2]; 3],
            indices: vec![0, 1, 2],
            joints: vec![[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            weights: vec![[1.0, 0.0, 0.0, 0.0]; 3],
        };
        (geometry, rig)
    }

    /// The texels of a packed raw texture.
    fn texels(texture: &[u8]) -> Vec<[u8; 4]> {
        use std::io::Read;
        let mut pixels = Vec::new();
        flate2::read::ZlibDecoder::new(&texture[20..]).read_to_end(&mut pixels).unwrap();
        pixels.as_chunks::<4>().0.to_vec()
    }

    #[test]
    fn frames_hold_each_vertex_where_the_clip_puts_it() {
        let (geometry, rig) = bending();
        let baked = bake(&geometry, &rig, 4.0).unwrap();
        assert_eq!(baked.clips, [(0, 4)]);
        assert!(baked.texture.starts_with(b"GTEX") && baked.texture[6] == 5, "a raw packed texture");
        let texels = texels(&baked.texture);
        assert_eq!(texels.len(), (baked.width * baked.height) as usize);
        let word = |index: usize| u32::from_le_bytes(texels[index]);
        assert_eq!([word(0), word(1), word(2)], [MAGIC, 3, 1]);
        let low = [3, 4, 5].map(|index| f32::from_bits(word(index)));
        let size = [6, 7, 8].map(|index| f32::from_bits(word(index)));
        assert_eq!([word(9), word(10), f32::from_bits(word(11)).to_bits()], [0, 4, 1.0f32.to_bits()]);
        let place = |frame: usize, vertex: usize| {
            let at = 12 + (frame * 3 + vertex) * 2;
            [0, 1, 2].map(|axis| {
                let quantized = u16::from_be_bytes([texels[at][axis], texels[at + 1][axis]]);
                low[axis] + f32::from(quantized) / 65535.0 * size[axis]
            })
        };
        // Frame 2 of 4 is halfway: the top has turned 45 degrees about the
        // middle, toward -x; the foot never moves.
        let angle = std::f32::consts::FRAC_PI_4;
        let top = place(2, 1);
        assert!((top[0] + angle.sin()).abs() < 1e-3 && (top[1] - 1.0 - angle.cos()).abs() < 1e-3, "{top:?}");
        assert!(place(2, 0).iter().zip([0.0, 0.0, 0.0]).all(|(a, b): (&f32, f32)| (a - b).abs() < 1e-3));
        // Frame 0 is the rest pose.
        let rest = place(0, 2);
        assert!((rest[0] - 0.5).abs() < 1e-3 && (rest[1] - 2.0).abs() < 1e-3, "{rest:?}");
    }

    #[test]
    fn octahedral_normals_cover_the_sphere() {
        let decode = |[u, v]: [u8; 2]| {
            let (mut x, mut y) = (f64::from(u) / 255.0 * 2.0 - 1.0, f64::from(v) / 255.0 * 2.0 - 1.0);
            let z = 1.0 - x.abs() - y.abs();
            if z < 0.0 {
                (x, y) = ((1.0 - y.abs()) * x.signum(), (1.0 - x.abs()) * y.signum());
            }
            let length = (x * x + y * y + z * z).sqrt();
            [x / length, y / length, z / length]
        };
        for normal in [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.6, -0.48, -0.64]] {
            let back = decode(octahedral(normal));
            let cosine: f64 = back.iter().zip(normal).map(|(a, b)| a * f64::from(b)).sum();
            assert!(cosine > 0.999, "{normal:?} came back as {back:?}");
        }
    }
}
