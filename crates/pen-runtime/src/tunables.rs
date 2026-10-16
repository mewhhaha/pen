use std::collections::HashMap;

use pen_protocol::boundary::whole;

const STATE_MAGIC: f32 = 5678.0;
const TUNED_SECTION: f32 = 11.0;

pub fn tuned_values(state: &[f32]) -> HashMap<u32, f32> {
    let mut tuned = HashMap::new();
    if state.len() < 3 || state[0] != STATE_MAGIC {
        return tuned;
    }
    let mut offset = 3;
    let Some(sections) = whole(state[2]) else { return tuned };
    for _ in 0..sections {
        let Some(&[key, _, _, length]) =
            state.get(offset..offset + 4).and_then(|header| <&[f32; 4]>::try_from(header).ok())
        else {
            break;
        };
        let start = offset + 4;
        let end = whole(length).and_then(|length| start.checked_add(length as usize));
        let Some(lanes) = end.and_then(|end| state.get(start..end)) else { break };
        if key == TUNED_SECTION {
            for &[id, value] in lanes.as_chunks::<2>().0 {
                if let Some(id) = whole(id) {
                    tuned.insert(id, value);
                }
            }
        }
        offset = start + lanes.len();
    }
    tuned
}
