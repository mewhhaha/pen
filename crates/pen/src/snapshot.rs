//! Saves on disk: "GDEVSAVE", version 1, schema and length (u32s), then the
//! state's float32 lanes, all little-endian (gdev's `snapshot.ts`).
use std::path::Path;

use anyhow::{Context, Result, ensure};
use pen_protocol::boundary::{MAX_STATE_FLOATS, validate_schema, validate_state};

const MAGIC: &[u8; 8] = b"GDEVSAVE";
pub const HEADER_BYTES: usize = 20;

#[derive(Clone, Debug, PartialEq)]
pub struct SavedState {
    pub schema: u32,
    pub state: Vec<f32>,
}

pub fn encode(saved: &SavedState) -> Result<Vec<u8>> {
    validate_schema(saved.schema as f32)?;
    validate_state(&saved.state)?;
    let mut bytes = Vec::with_capacity(HEADER_BYTES + saved.state.len() * 4);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&saved.schema.to_le_bytes());
    bytes.extend_from_slice(&(saved.state.len() as u32).to_le_bytes());
    for value in &saved.state {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

pub fn decode(bytes: &[u8]) -> Result<SavedState> {
    ensure!(
        bytes.len() >= HEADER_BYTES
            && bytes.len() <= HEADER_BYTES + MAX_STATE_FLOATS * 4
            && bytes.starts_with(MAGIC),
        "file is not a bounded gdev quicksave"
    );
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let schema = word(12);
    let length = word(16) as usize;
    ensure!(
        word(8) == 1 && bytes.len() == HEADER_BYTES + length * 4,
        "invalid quicksave version or length"
    );
    validate_schema(schema as f32)?;
    let state: Vec<f32> = bytes[HEADER_BYTES..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    validate_state(&state)?;
    Ok(SavedState { schema, state })
}

pub fn read(path: &Path) -> Result<SavedState> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    decode(&bytes).with_context(|| format!("reading {}", path.display()))
}

/// Writes through a temporary file, so a crash never leaves half a save.
pub fn write(path: &Path, saved: &SavedState) -> Result<()> {
    let bytes = encode(saved)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let result = std::fs::write(&temporary, &bytes).and_then(|()| std::fs::rename(&temporary, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result.with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let saved = SavedState { schema: 7, state: vec![1.0, -2.5, 3.25] };
        assert_eq!(decode(&encode(&saved).unwrap()).unwrap(), saved);
        assert!(decode(b"GDEVSAVE").is_err());
    }
}
