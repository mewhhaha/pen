use crate::saved::STATE_MAGIC;
use pen_protocol::boundary::whole;
use pen_runtime::state::Section;

/// A complete Ink state, or no view if any section is malformed.
pub fn sections(state: &[f32]) -> Option<Vec<Section<'_>>> {
    let mut found = Vec::new();
    if state.len() < 3 || state[0] != STATE_MAGIC {
        return None;
    }
    let count = whole(state[2])?;
    let mut offset = 3;
    for _ in 0..count {
        let &[key, version, width, length] =
            state.get(offset..offset + 4).and_then(|header| <&[f32; 4]>::try_from(header).ok())?;
        let start = offset + 4;
        let (Some(key), Some(version), Some(width), Some(length)) =
            (whole(key), whole(version), whole(width), whole(length))
        else {
            return None;
        };
        let lanes = start.checked_add(length as usize).and_then(|end| state.get(start..end))?;
        found.push(Section { key, version, width, lanes });
        offset = start + lanes.len();
    }
    (offset == state.len()).then_some(found)
}
