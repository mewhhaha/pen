//! Saves on disk, in `pen_runtime::snapshot`'s format (which the browser
//! host keeps in IndexedDB too): "GDEVSAVE", version 1, schema and length
//! (u32s), then the state's float32 lanes, all little-endian.
use std::path::Path;

use anyhow::{Context, Result};

pub use pen_runtime::snapshot::*;

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
