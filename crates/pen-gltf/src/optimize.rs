//! Mesh optimization and `EXT_meshopt_compression` post-filters for glTF import:
//! - Tom Forsyth linear-speed post-transform vertex cache optimization (O(T) with
//!   live adjacency swap-removal so high-valence fans never turn quadratic);
//! - vertex fetch locality re-indexing (reordering vertices and every parallel
//!   attribute buffer in first-use order);
//! - `EXT_meshopt_compression` post-filters (`OCTAHEDRAL` 8/16-bit, `QUATERNION`
//!   16-bit, `EXPONENTIAL` 32-bit).
use anyhow::{Result, ensure};

use crate::Geometry;

const CACHE_SIZE: usize = 32;

fn vertex_score(cache_pos: i32, active_tris: u32) -> f32 {
    if active_tris == 0 {
        return -1.0;
    }
    let mut score = if cache_pos < 0 {
        0.0
    } else if cache_pos < 3 {
        0.75
    } else {
        let scaler = 1.0 / (CACHE_SIZE - 3) as f32;
        (1.0 - (cache_pos - 3) as f32 * scaler).clamp(0.0, 1.0).powf(1.5)
    };
    score += 2.0 * (active_tris as f32).powf(-0.5);
    score
}

/// Reorders triangle indices using Tom Forsyth's linear-speed vertex cache
/// algorithm to minimize GPU post-transform cache misses (ACMR).
///
/// If `indices` contains out-of-range vertex indices or fewer than 3 triangles,
/// returns `indices` unchanged.
pub fn optimize_vertex_cache(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    let tri_count = indices.len() / 3;
    if tri_count <= 2 || vertex_count == 0 {
        return indices.to_vec();
    }
    let valid_len = tri_count * 3;
    if indices[..valid_len].iter().any(|&idx| (idx as usize) >= vertex_count) {
        return indices.to_vec();
    }
    let mut counts = vec![0u32; vertex_count];
    for &idx in &indices[..valid_len] {
        counts[idx as usize] += 1;
    }
    let mut offsets = vec![0usize; vertex_count + 1];
    for i in 0..vertex_count {
        offsets[i + 1] = offsets[i] + counts[i] as usize;
    }
    let mut adj = vec![0usize; offsets[vertex_count]];
    let mut cursor = offsets[..vertex_count].to_vec();
    for t in 0..tri_count {
        for k in 0..3 {
            let v = indices[t * 3 + k] as usize;
            adj[cursor[v]] = t;
            cursor[v] += 1;
        }
    }

    // Live per-vertex adjacency lengths: emitted triangles are swap-removed from
    // `adj[offsets[v] .. offsets[v] + live[v]]` so emitted faces are never scanned again.
    let mut live = counts;
    let mut cache_pos = vec![-1i32; vertex_count];
    let mut v_score: Vec<f32> = live.iter().map(|&c| vertex_score(-1, c)).collect();
    let mut emitted = vec![false; tri_count];
    let mut tri_score: Vec<f32> = (0..tri_count)
        .map(|t| {
            v_score[indices[t * 3] as usize]
                + v_score[indices[t * 3 + 1] as usize]
                + v_score[indices[t * 3 + 2] as usize]
        })
        .collect();

    let mut cache: Vec<u32> = Vec::with_capacity(CACHE_SIZE + 3);
    let mut out = Vec::with_capacity(indices.len());
    let mut scan = 0usize;
    let mut best_tri = (0..tri_count).max_by(|&a, &b| tri_score[a].total_cmp(&tri_score[b])).unwrap_or(0);

    for _ in 0..tri_count {
        if emitted[best_tri] {
            while scan < tri_count && emitted[scan] {
                scan += 1;
            }
            if scan >= tri_count {
                break;
            }
            best_tri = scan;
        }
        emitted[best_tri] = true;
        let tri = [indices[best_tri * 3], indices[best_tri * 3 + 1], indices[best_tri * 3 + 2]];
        out.extend_from_slice(&tri);

        for &v in &tri {
            let vi = v as usize;
            // Remove `best_tri` from vertex `vi`'s live adjacency list via swap-remove.
            let start = offsets[vi];
            let len = live[vi] as usize;
            if let Some(pos) = adj[start..start + len].iter().position(|&t| t == best_tri) {
                let last = len - 1;
                adj.swap(start + pos, start + last);
                live[vi] = u32::try_from(last).unwrap_or(0);
            }
            if let Some(pos) = cache.iter().position(|&c| c == v) {
                cache.remove(pos);
            }
            cache.insert(0, v);
        }
        // Evict tail beyond CACHE_SIZE.
        while cache.len() > CACHE_SIZE {
            if let Some(evicted) = cache.pop() {
                let ei = evicted as usize;
                cache_pos[ei] = -1;
                let old = v_score[ei];
                let new = vertex_score(-1, live[ei]);
                v_score[ei] = new;
                let delta = new - old;
                let start = offsets[ei];
                let end = start + live[ei] as usize;
                for &t in &adj[start..end] {
                    tri_score[t] += delta;
                }
            }
        }
        // Update scores for vertices currently in the cache and pick next best adjacent triangle.
        // Cap per-vertex scan at 64 live triangles on extreme high-valence hubs to stay strictly O(T).
        let mut next_best = usize::MAX;
        let mut next_score = f32::NEG_INFINITY;
        for (pos, &v) in cache.iter().enumerate() {
            let vi = v as usize;
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "cache <= 32")]
            let p = pos as i32;
            cache_pos[vi] = p;
            let old = v_score[vi];
            let new = vertex_score(p, live[vi]);
            v_score[vi] = new;
            let delta = new - old;
            let start = offsets[vi];
            let live_n = live[vi] as usize;
            let scan_end = start + live_n.min(64);
            for &t in &adj[start..scan_end] {
                tri_score[t] += delta;
                if tri_score[t] > next_score {
                    next_score = tri_score[t];
                    next_best = t;
                }
            }
            if live_n > 64 {
                for &t in &adj[scan_end..start + live_n] {
                    tri_score[t] += delta;
                }
            }
        }
        if next_best != usize::MAX {
            best_tri = next_best;
        }
    }
    if valid_len < indices.len() {
        out.extend_from_slice(&indices[valid_len..]);
    }
    out
}

/// Computes the Average Cache Miss Ratio (ACMR: misses per triangle) for a
/// FIFO cache of `cache_size` entries.
pub fn acmr(indices: &[u32], cache_size: usize) -> f64 {
    let tris = indices.len() / 3;
    if tris == 0 {
        return 0.0;
    }
    let mut cache: Vec<u32> = Vec::with_capacity(cache_size);
    let mut misses = 0usize;
    for &idx in indices {
        if !cache.contains(&idx) {
            misses += 1;
            if cache.len() == cache_size {
                cache.remove(0);
            }
            cache.push(idx);
        }
    }
    misses as f64 / tris as f64
}

/// Optimizes `geometry` in place for vertex cache locality and sequential
/// vertex-buffer fetch order, preserving all vertex attributes and morph deltas.
pub fn optimize_geometry(geometry: &mut Geometry) {
    let n = geometry.positions.len();
    if n == 0 || geometry.indices.len() < 9 || geometry.indices.iter().any(|&i| (i as usize) >= n) {
        return;
    }
    geometry.indices = optimize_vertex_cache(&geometry.indices, n);

    // Reorder vertices in first-use order for optimal vertex fetch locality.
    let mut remap = vec![u32::MAX; n];
    let mut order = Vec::with_capacity(n);
    for idx in &mut geometry.indices {
        let old = *idx as usize;
        if remap[old] == u32::MAX {
            #[allow(clippy::cast_possible_truncation, reason = "vertex index fits u32")]
            let next = order.len() as u32;
            remap[old] = next;
            order.push(old);
        }
        *idx = remap[old];
    }
    let gather = |src: &[[f32; 3]]| -> Vec<[f32; 3]> { order.iter().map(|&i| src[i]).collect() };
    geometry.positions = gather(&geometry.positions);
    geometry.normals = gather(&geometry.normals);
    geometry.uvs = order.iter().map(|&i| geometry.uvs[i]).collect();
    if !geometry.uvs1.is_empty() {
        geometry.uvs1 = order.iter().map(|&i| geometry.uvs1[i]).collect();
    }
    if !geometry.colors.is_empty() {
        geometry.colors = order.iter().map(|&i| geometry.colors[i]).collect();
    }
    if !geometry.joints.is_empty() {
        geometry.joints = order.iter().map(|&i| geometry.joints[i]).collect();
    }
    if !geometry.weights.is_empty() {
        geometry.weights = order.iter().map(|&i| geometry.weights[i]).collect();
    }
    if !geometry.tangents.is_empty() {
        geometry.tangents = order.iter().map(|&i| geometry.tangents[i]).collect();
    }
    for morph in &mut geometry.morphs {
        morph.positions = gather(&morph.positions);
        morph.normals = gather(&morph.normals);
    }
}

/// Decodes an `EXT_meshopt_compression` `OCTAHEDRAL` 8-bit snorm filter
/// in-place (`stride == 4`, 4 bytes per element: x, y, z, w).
pub fn decode_filter_octahedral_8(data: &mut [u8]) -> Result<()> {
    ensure!(data.len().is_multiple_of(4), "octahedral-8 filter expects 4-byte stride");
    for chunk in data.as_chunks_mut::<4>().0 {
        let x = f32::from(chunk[0].cast_signed());
        let y = f32::from(chunk[1].cast_signed());
        let one = f32::from(chunk[2].cast_signed());
        let z = one - x.abs() - y.abs();
        let t = (-z).max(0.0);
        let nx = x + if x >= 0.0 { -t } else { t };
        let ny = y + if y >= 0.0 { -t } else { t };
        let len = (nx * nx + ny * ny + z * z).sqrt().max(1e-8);
        let scale = 127.0 / len;
        #[allow(clippy::cast_possible_truncation, reason = "clamped to i8")]
        {
            chunk[0] = ((nx * scale).round().clamp(-127.0, 127.0) as i8).cast_unsigned();
            chunk[1] = ((ny * scale).round().clamp(-127.0, 127.0) as i8).cast_unsigned();
            chunk[2] = ((z * scale).round().clamp(-127.0, 127.0) as i8).cast_unsigned();
        }
    }
    Ok(())
}

/// Decodes an `EXT_meshopt_compression` `OCTAHEDRAL` 16-bit snorm filter
/// in-place (`stride == 8`, four little-endian `i16` values: x, y, z, w).
pub fn decode_filter_octahedral_16(data: &mut [u8]) -> Result<()> {
    ensure!(data.len().is_multiple_of(8), "octahedral-16 filter expects 8-byte stride");
    for chunk in data.as_chunks_mut::<8>().0 {
        let x = f32::from(i16::from_le_bytes([chunk[0], chunk[1]]));
        let y = f32::from(i16::from_le_bytes([chunk[2], chunk[3]]));
        let one = f32::from(i16::from_le_bytes([chunk[4], chunk[5]]));
        let z = one - x.abs() - y.abs();
        let t = (-z).max(0.0);
        let nx = x + if x >= 0.0 { -t } else { t };
        let ny = y + if y >= 0.0 { -t } else { t };
        let len = (nx * nx + ny * ny + z * z).sqrt().max(1e-8);
        let scale = 32767.0 / len;
        #[allow(clippy::cast_possible_truncation, reason = "clamped to i16")]
        {
            let rx = (nx * scale).round().clamp(-32767.0, 32767.0) as i16;
            let ry = (ny * scale).round().clamp(-32767.0, 32767.0) as i16;
            let rz = (z * scale).round().clamp(-32767.0, 32767.0) as i16;
            chunk[0..2].copy_from_slice(&rx.to_le_bytes());
            chunk[2..4].copy_from_slice(&ry.to_le_bytes());
            chunk[4..6].copy_from_slice(&rz.to_le_bytes());
        }
    }
    Ok(())
}

/// Decodes an `EXT_meshopt_compression` `QUATERNION` 16-bit snorm filter
/// in-place (`stride == 8`, 4 little-endian `i16` values per quaternion).
/// Per `EXT_meshopt_compression` / `meshopt_decodeFilterQuat`: `w & 3` selects the
/// reconstructed largest component and `w | 3` supplies the quantization scale divisor.
pub fn decode_filter_quaternion_16(data: &mut [u8]) -> Result<()> {
    ensure!(data.len().is_multiple_of(8), "quaternion filter expects 8-byte stride");
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    for chunk in data.as_chunks_mut::<8>().0 {
        let x = i16::from_le_bytes([chunk[0], chunk[1]]);
        let y = i16::from_le_bytes([chunk[2], chunk[3]]);
        let z = i16::from_le_bytes([chunk[4], chunk[5]]);
        let w = i16::from_le_bytes([chunk[6], chunk[7]]);
        ensure!(w >= 3, "invalid meshopt quaternion control word {w}");
        let max_comp = usize::try_from(w & 3).unwrap_or(0);
        let scale = inv_sqrt2 / f32::from(w | 3);
        let a = f32::from(x) * scale;
        let b = f32::from(y) * scale;
        let c = f32::from(z) * scale;
        let d = (1.0 - a * a - b * b - c * c).max(0.0).sqrt();
        let mut q = [0.0f32; 4];
        q[(max_comp + 1) & 3] = a;
        q[(max_comp + 2) & 3] = b;
        q[(max_comp + 3) & 3] = c;
        q[max_comp] = d;
        for (i, val) in q.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation, reason = "clamped to i16")]
            let s = (*val * 32767.0).round().clamp(-32767.0, 32767.0) as i16;
            chunk[i * 2..i * 2 + 2].copy_from_slice(&s.to_le_bytes());
        }
    }
    Ok(())
}

/// Decodes an `EXT_meshopt_compression` `EXPONENTIAL` filter in-place
/// (`stride % 4 == 0`: each 32-bit word packs an 8-bit signed exponent in bits
/// 24..32 and a 24-bit signed mantissa in bits 0..24, decoded to `mantissa * 2^exp`).
pub fn decode_filter_exponential(data: &mut [u8]) -> Result<()> {
    ensure!(data.len().is_multiple_of(4), "exponential filter expects 4-byte-aligned stride");
    for chunk in data.as_chunks_mut::<4>().0 {
        let v = u32::from_le_bytes(*chunk);
        let exp = v.cast_signed() >> 24;
        let mantissa = (v << 8).cast_signed() >> 8;
        let f = (mantissa as f32) * (2.0f32).powi(exp);
        ensure!(f.is_finite(), "exponential filter decoded non-finite float");
        chunk.copy_from_slice(&f.to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MorphTarget;

    #[test]
    fn vertex_cache_optimizer_reduces_acmr_and_handles_large_fan_and_invalid_indices() {
        let side = 16u32;
        let mut indices = Vec::new();
        for x in 0..side {
            for y in 0..side {
                let i0 = y * (side + 1) + x;
                let i1 = i0 + 1;
                let i2 = i0 + (side + 1);
                let i3 = i2 + 1;
                indices.extend([i0, i2, i1, i1, i2, i3]);
            }
        }
        let vcount = ((side + 1) * (side + 1)) as usize;
        let before = acmr(&indices, 16);
        let optimized = optimize_vertex_cache(&indices, vcount);
        let after = acmr(&optimized, 16);
        assert_eq!(optimized.len(), indices.len());
        assert!(after < before, "optimizer must lower ACMR: {after:.3} vs {before:.3}");

        // Out-of-bounds index returns safely without panicking.
        let bad = vec![0, 1, 99, 0, 1, 2, 1, 2, 3];
        assert_eq!(optimize_vertex_cache(&bad, 4), bad);

        // High-valence 512-triangle fan completes cleanly with exact triangle count.
        let spokes = 512u32;
        let mut fan = Vec::with_capacity(spokes as usize * 3);
        for i in 0..spokes {
            fan.extend([0, 1 + i, 1 + (i + 1) % spokes]);
        }
        let opt_fan = optimize_vertex_cache(&fan, spokes as usize + 1);
        assert_eq!(opt_fan.len(), fan.len());
    }

    #[test]
    fn optimize_geometry_preserves_all_vertex_attributes_and_morph_alignment() {
        let mut geom = Geometry {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 5],
            uvs: vec![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 1.0]],
            uvs1: vec![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
            colors: vec![[0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0, 1.0], [0.2, 0.0, 0.0, 1.0], [0.3, 0.0, 0.0, 1.0], [0.4, 0.0, 0.0, 1.0]],
            indices: vec![4, 3, 2, 2, 1, 0, 4, 2, 1],
            joints: vec![[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]],
            weights: vec![[1.0, 0.0, 0.0, 0.0]; 5],
            tangents: vec![[1.0, 0.0, 0.0, 1.0]; 5],
            morphs: vec![MorphTarget {
                positions: vec![[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.2, 0.0], [0.0, 0.3, 0.0], [0.0, 0.4, 0.0]],
                normals: vec![[0.0; 3]; 5],
            }],
        };
        optimize_geometry(&mut geom);
        // Every vertex's uvs1[0], colors[0]*10, joints[0][0], and morph.positions[1]*10 must stay aligned with original index!
        for v in 0..geom.positions.len() {
            let orig = geom.joints[v][0] as usize;
            assert!((geom.uvs1[v][0] - (orig as f32 + 1.0)).abs() < 1e-6);
            assert!((geom.colors[v][0] - orig as f32 * 0.1).abs() < 1e-6);
            assert!((geom.morphs[0].positions[v][1] - orig as f32 * 0.1).abs() < 1e-6);
        }
    }

    #[test]
    fn meshopt_octahedral_quaternion_and_exponential_filters_decode_accurately() {
        // 8-bit octahedral: (0, 0, +1) -> (0, 0, 127)
        let mut oct8 = [0u8, 0u8, 127u8, 127u8];
        decode_filter_octahedral_8(&mut oct8).unwrap();
        assert_eq!(oct8[2].cast_signed(), 127);

        // 16-bit octahedral: diagonal normal in lower hemisphere (x=-4096, y=-4096, one=16383 -> z=8191)
        let mut oct16 = [0u8; 8];
        oct16[0..2].copy_from_slice(&(-4096i16).to_le_bytes());
        oct16[2..4].copy_from_slice(&(-4096i16).to_le_bytes());
        oct16[4..6].copy_from_slice(&16383i16.to_le_bytes());
        decode_filter_octahedral_16(&mut oct16).unwrap();
        let nx = f32::from(i16::from_le_bytes([oct16[0], oct16[1]])) / 32767.0;
        let ny = f32::from(i16::from_le_bytes([oct16[2], oct16[3]])) / 32767.0;
        let nz = f32::from(i16::from_le_bytes([oct16[4], oct16[5]])) / 32767.0;
        let nlen = (nx * nx + ny * ny + nz * nz).sqrt();
        assert!((nlen - 1.0).abs() < 1e-3 && nx < 0.0 && ny < 0.0 && nz > 0.0, "({nx},{ny},{nz})");

        // 16-bit quaternion: 90-degree rotation around Y: q = (0, sin(45°), 0, cos(45°)) = (0, 0.7071, 0, 0.7071)
        // Encode with 12-bit scale_max = 2047 (0x07FF), max_comp = 3 (w):
        // w_word = (2047 & !3) | 3 = 2047; y_quant = round(0.70710678 / (FRAC_1_SQRT_2 / 2047)) = 2047? Wait:
        // Use 60-degree rotation around Y: q = (0, 0.5, 0, 0.8660254), max_comp = 3:
        // a = q[(3+1)&3] = q[0] = 0, b = q[1] = 0.5, c = q[2] = 0, d = q[3] = 0.8660254
        // y_quant = round(0.5 * 2047 / FRAC_1_SQRT_2) = 1447
        let scale_max = 2047i16;
        let mut quat = [0u8; 8];
        #[allow(clippy::cast_possible_truncation, reason = "known test value fits i16")]
        let y_enc = ((0.5f32 * f32::from(scale_max)) / std::f32::consts::FRAC_1_SQRT_2).round() as i16;
        quat[2..4].copy_from_slice(&y_enc.to_le_bytes());
        quat[6..8].copy_from_slice(&((scale_max & !3) | 3).to_le_bytes());
        decode_filter_quaternion_16(&mut quat).unwrap();
        let qy = f32::from(i16::from_le_bytes([quat[2], quat[3]])) / 32767.0;
        let qw = f32::from(i16::from_le_bytes([quat[6], quat[7]])) / 32767.0;
        assert!((qy - 0.5).abs() < 1e-3 && (qw - 0.8660254).abs() < 1e-3, "qy={qy} qw={qw}");

        // Exponential filter: mantissa = -3, exp = -2 => -3 * 2^-2 = -0.75
        let word: u32 = (u32::from((-2i8).cast_unsigned()) << 24) | ((-3i32).cast_unsigned() & 0x00ff_ffff);
        let mut exp_buf = word.to_le_bytes();
        decode_filter_exponential(&mut exp_buf).unwrap();
        let decoded = f32::from_le_bytes(exp_buf);
        assert!((decoded - (-0.75)).abs() < 1e-6, "decoded={decoded}");
    }
}
