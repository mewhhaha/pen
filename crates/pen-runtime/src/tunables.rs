use std::collections::HashMap;

use pen_protocol::boundary::whole;

use crate::inspect::sections;

/// The section the game keeps the values the host tuned in.
const TUNED_SECTION: u32 = 11;

/// The values the host tuned that a saved state holds, by tunable id.
pub fn tuned_values(state: &[f32]) -> HashMap<u32, f32> {
    let mut tuned = HashMap::new();
    for section in sections(state).into_iter().filter(|section| section.key == TUNED_SECTION) {
        for &[id, value] in section.lanes.as_chunks::<2>().0 {
            if let Some(id) = whole(id) {
                tuned.insert(id, value);
            }
        }
    }
    tuned
}
