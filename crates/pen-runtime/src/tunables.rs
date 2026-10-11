use std::collections::HashMap;
const STATE_MAGIC: f32 = 5678.0;
const TUNED_SECTION: f32 = 11.0;

pub fn tuned_values(state: &[f32]) -> HashMap<u32, f32> {
    let mut tuned = HashMap::new();
    if state.len() < 3 || state[0] != STATE_MAGIC {
        return tuned;
    }
    let mut offset = 3;
    for _ in 0..state[2] as usize {
        let Some(&[key, _, _, length]) = state.get(offset..offset + 4).and_then(|header| <&[f32; 4]>::try_from(header).ok())
        else {
            break;
        };
        let start = offset + 4;
        let end = start + length as usize;
        if key == TUNED_SECTION
            && let Some(lanes) = state.get(start..end)
        {
            for &[id, value] in lanes.as_chunks::<2>().0 {
                tuned.insert(id as u32, value);
            }
        }
        offset = end;
    }
    tuned
}
