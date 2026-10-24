//! Shipping storage for precompiled guests: one standard Zstd frame, with
//! its expanded size and checksum. Wasmtime receives the original bytes;
//! development caches keep their own format and no expanded copy is saved.
use anyhow::{Context, Result, ensure};

const MAX_BYTES: usize = 256 * 1024 * 1024;

pub fn encode(bytes: &[u8]) -> Result<Vec<u8>> {
    ensure!((1..=MAX_BYTES).contains(&bytes.len()), "precompiled game exceeds its size limit");
    let mut compressor = zstd::bulk::Compressor::new(19)?;
    compressor.set_parameter(zstd::zstd_safe::CParameter::ChecksumFlag(true))?;
    Ok(compressor.compress(bytes)?)
}

pub fn decode(bytes: &[u8]) -> Result<Vec<u8>> {
    let size = zstd::zstd_safe::get_frame_content_size(bytes)
        .map_err(|_| anyhow::anyhow!("invalid compressed game header"))?
        .context("compressed game is missing its expanded size")?;
    ensure!((1..=MAX_BYTES as u64).contains(&size), "compressed game expands beyond its size limit");
    let frame_size = zstd::zstd_safe::find_frame_compressed_size(bytes)
        .map_err(|code| anyhow::anyhow!("invalid compressed game: {}", zstd::zstd_safe::get_error_name(code)))?;
    ensure!(frame_size == bytes.len(), "compressed game has trailing bytes or extra frames");
    let decoded =
        zstd::bulk::decompress(bytes, usize::try_from(size)?).context("decompressing the precompiled game")?;
    ensure!(decoded.len() as u64 == size, "compressed game does not match its expanded size");
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_is_lossless_and_readable_by_standard_zstd() {
        let bytes: Vec<u8> = (0..=255).cycle().take(16 * 1024).collect();
        let packed = encode(&bytes).unwrap();
        assert!(packed.len() < bytes.len() / 10);
        assert_eq!(decode(&packed).unwrap(), bytes);
        assert_eq!(zstd::stream::decode_all(packed.as_slice()).unwrap(), bytes);
    }

    #[test]
    fn rejects_truncation_corruption_extra_frames_and_excessive_expansion() {
        let packed = encode(&vec![42; 4096]).unwrap();
        for end in 0..packed.len() {
            assert!(decode(&packed[..end]).is_err(), "truncated at {end}");
        }
        let mut corrupt = packed.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(decode(&corrupt).is_err(), "checksum corruption");
        let mut extra = packed.clone();
        extra.push(0);
        assert!(decode(&extra).is_err(), "trailing byte");
        assert!(decode(&[packed.clone(), packed].concat()).is_err(), "concatenated frames");
        // A standard single-segment frame with an eight-byte content size.
        // Reject its size before allocating or trying to read its blocks.
        let mut oversized = b"\x28\xb5\x2f\xfd\xe0".to_vec();
        oversized.extend((MAX_BYTES as u64 + 1).to_le_bytes());
        assert!(decode(&oversized).unwrap_err().to_string().contains("size limit"));
        assert!(encode(&[]).is_err());
    }
}
