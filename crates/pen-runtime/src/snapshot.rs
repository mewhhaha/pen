//! Saves on disk: "GDEVSAVE", version 1, schema and length (u32s), then the
//! state's float32 lanes, all little-endian. Version 2 puts a [`Stamp`]
//! between the length and the lanes: when the save was written (a u64 of
//! milliseconds since the Unix epoch), then the count (u32) and the
//! [`identity`]s (u64s) of the saves it replaced, newest first. Both
//! versions read; [`encode`] writes version 1 (scenes, the browser's saves)
//! and [`encode_stamped`] version 2 (the player's saves on disk).
use anyhow::{Result, ensure};
use pen_protocol::boundary::{MAX_STATE_FLOATS, validate_schema, validate_state};

/// The save format's identifier, whatever the game: it keeps the name of the
/// first game that saved in it, and changing it would orphan every save.
const MAGIC: &[u8; 8] = b"GDEVSAVE";
/// Version 1's header; version 2's adds its stamp.
pub const HEADER_BYTES: usize = 20;
/// The most identities a [`Stamp`] keeps of the saves before it.
pub const LINEAGE: usize = 8;
/// A stamp's bytes before its identities: the time and their count.
const STAMP_BYTES: usize = 12;

#[derive(Clone, Debug, PartialEq)]
pub struct SavedState {
    pub schema: u32,
    pub state: Vec<f32>,
}

/// Where a stamped save comes from: when it was written, and the saves it
/// replaced where it was written, as their [`identity`]s, newest first. With
/// them a host tells a save another device wrote (Steam Cloud brought it)
/// from its own, and whether that save continued from its own last one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stamp {
    /// Milliseconds since the Unix epoch.
    pub written: u64,
    /// At most [`LINEAGE`].
    pub lineage: Vec<u64>,
}

/// A save file's identity: FNV-1a over its bytes. It names one version of a
/// file; it is no defence against anyone.
pub fn identity(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3))
}

pub fn encode(saved: &SavedState) -> Result<Vec<u8>> {
    encode_version(saved, None)
}

/// Version 2: `saved`, with where it comes from.
pub fn encode_stamped(saved: &SavedState, stamp: &Stamp) -> Result<Vec<u8>> {
    encode_version(saved, Some(stamp))
}

fn encode_version(saved: &SavedState, stamp: Option<&Stamp>) -> Result<Vec<u8>> {
    validate_schema(saved.schema as f32)?;
    validate_state(&saved.state)?;
    let stamp_bytes = stamp.map_or(0, |stamp| STAMP_BYTES + stamp.lineage.len() * 8);
    let mut bytes = Vec::with_capacity(HEADER_BYTES + stamp_bytes + saved.state.len() * 4);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(if stamp.is_some() { 2u32 } else { 1 }).to_le_bytes());
    bytes.extend_from_slice(&saved.schema.to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(saved.state.len())?.to_le_bytes());
    if let Some(stamp) = stamp {
        ensure!(stamp.lineage.len() <= LINEAGE, "a save's stamp keeps at most {LINEAGE} saves before it");
        bytes.extend_from_slice(&stamp.written.to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(stamp.lineage.len())?.to_le_bytes());
        for identity in &stamp.lineage {
            bytes.extend_from_slice(&identity.to_le_bytes());
        }
    }
    for value in &saved.state {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

pub fn decode(bytes: &[u8]) -> Result<SavedState> {
    decode_stamped(bytes).map(|(saved, _)| saved)
}

/// A save of either version, and its stamp if it has one (version 2).
pub fn decode_stamped(bytes: &[u8]) -> Result<(SavedState, Option<Stamp>)> {
    ensure!(
        bytes.len() >= HEADER_BYTES
            && bytes.len() <= HEADER_BYTES + STAMP_BYTES + LINEAGE * 8 + MAX_STATE_FLOATS * 4
            && bytes.starts_with(MAGIC),
        "file is not a bounded gdev quicksave"
    );
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let long = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
    let schema = word(12);
    let length = word(16) as usize;
    let (stamp, start) = match word(8) {
        1 => (None, HEADER_BYTES),
        2 => {
            ensure!(bytes.len() >= HEADER_BYTES + STAMP_BYTES, "invalid quicksave stamp");
            let count = word(HEADER_BYTES + 8) as usize;
            let start = HEADER_BYTES + STAMP_BYTES + count * 8;
            ensure!(count <= LINEAGE && bytes.len() >= start, "invalid quicksave stamp");
            let lineage = (0..count).map(|index| long(HEADER_BYTES + STAMP_BYTES + index * 8)).collect();
            (Some(Stamp { written: long(HEADER_BYTES), lineage }), start)
        }
        _ => (None, 0),
    };
    ensure!(start > 0 && bytes.len() == start + length * 4, "invalid quicksave version or length");
    validate_schema(schema as f32)?;
    let state: Vec<f32> = bytes[start..].as_chunks::<4>().0.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect();
    validate_state(&state)?;
    Ok((SavedState { schema, state }, stamp))
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

    #[test]
    fn stamped_saves_round_trip_and_read_as_saves() {
        let saved = SavedState { schema: 7, state: vec![1.0, -2.5, 3.25] };
        let stamp = Stamp { written: 1_760_000_000_000, lineage: vec![3, u64::MAX, 1] };
        let bytes = encode_stamped(&saved, &stamp).unwrap();
        assert_eq!(bytes.len(), HEADER_BYTES + STAMP_BYTES + 3 * 8 + 3 * 4);
        assert_eq!(decode_stamped(&bytes).unwrap(), (saved.clone(), Some(stamp)));
        assert_eq!(decode(&bytes).unwrap(), saved);
        // Version 1 has no stamp; an empty lineage is a stamp still.
        assert_eq!(decode_stamped(&encode(&saved).unwrap()).unwrap().1, None);
        let first = Stamp { written: 5, lineage: Vec::new() };
        assert_eq!(decode_stamped(&encode_stamped(&saved, &first).unwrap()).unwrap().1, Some(first));
        // Cut short, a lineage longer than kept or the file, or another
        // version: refused.
        assert!(decode(&bytes[..bytes.len() - 4]).is_err());
        assert!(decode(&bytes[..HEADER_BYTES + 4]).is_err());
        let mut long = bytes.clone();
        long[HEADER_BYTES + 8..HEADER_BYTES + 12].copy_from_slice(&9u32.to_le_bytes());
        assert!(decode(&long).is_err());
        let mut third = bytes;
        third[8] = 3;
        assert!(decode(&third).is_err());
        assert!(encode_stamped(&saved, &Stamp { written: 0, lineage: vec![0; LINEAGE + 1] }).is_err());
    }

    #[test]
    fn identities_tell_files_apart() {
        assert_eq!(identity(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(identity(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_ne!(identity(b"ab"), identity(b"ba"));
    }
}
