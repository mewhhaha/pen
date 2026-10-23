use pen_protocol::boundary::whole;

/// What Ink reports for one saved section (persist.blot Fate): 0 kept as saved, 1 migrated from
/// another version, 2 reset for want of a migration, 3 started afresh because
/// the state lacked it, 4 dropped because the build no longer saves it; and
/// the host's own, [`DECLARED`]: migrated by the game's declared saved
/// types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SectionOutcome {
    /// The section's key, its saved type's from build to build.
    pub key: u32,
    pub fate: u32,
    /// The version the state saved it with; 0 when the state lacked it.
    pub saved: u32,
    /// The build's version; 0 for a section the build does not read.
    pub version: u32,
}

const REPORT_MAGIC: f32 = 5680.0;
/// A section's fate when the game reset it for want of a migration.
pub const RESET: u32 = 2;
/// A section's fate when the host migrated it as the game's saved types
/// declare (saved.rs), after the game could not.
pub const DECLARED: u32 = 5;

/// The report a restore or scene load answers with: `5680, count`, then
/// `key, fate, saved, version` per section. None for any other answer, as
/// from a game whose restore answers with its state.
pub fn restore_report(response: &[f32]) -> Option<Vec<SectionOutcome>> {
    let [magic, count, rest @ ..] = response else {
        return None;
    };
    if *magic != REPORT_MAGIC || whole(*count).and_then(|count| (count as usize).checked_mul(4)) != Some(rest.len()) {
        return None;
    }
    rest.as_chunks::<4>()
        .0
        .iter()
        .map(|&[key, fate, saved, version]| {
            Some(SectionOutcome {
                key: whole(key)?,
                fate: whole(fate)?,
                saved: whole(saved)?,
                version: whole(version)?,
            })
        })
        .collect()
}

/// What a restore did other than keep each saved section as it was, for the
/// log, or None if it kept them all or did not say.
pub fn describe_restore(outcomes: Option<&[SectionOutcome]>) -> Option<String> {
    let outcomes = outcomes?;
    let changes: Vec<String> = outcomes
        .iter()
        .filter_map(|&SectionOutcome { key, fate, saved, version }| match fate {
            0 => None,
            1 => Some(format!("migrated section {key} from version {saved} to {version}")),
            2 => Some(format!("reset section {key} (version {saved} has no migration to {version})")),
            3 => Some(format!("started section {key} afresh (the state has none)")),
            4 => Some(format!("dropped section {key} (no longer saved)")),
            DECLARED => {
                Some(format!("migrated section {key} from version {saved} to {version} as saved.json declares"))
            }
            _ => Some(format!("section {key}: unknown fate {fate}")),
        })
        .collect();
    if changes.is_empty() {
        return None;
    }
    let read = outcomes.iter().filter(|outcome| outcome.fate != 4).count();
    let kept = outcomes.iter().filter(|outcome| outcome.fate == 0).count();
    let mut parts = vec![format!("kept {kept} of {read} saved sections")];
    parts.extend(changes);
    Some(parts.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reports_parse_and_describe_what_was_not_kept() {
        assert_eq!(restore_report(&[5678.0, 26.0, 0.0]), None, "a state is no report");
        assert_eq!(restore_report(&[5680.0, 1.0, 100.0]), None, "a cut-off report is none");
        assert_eq!(restore_report(&[5680.0, 1e30, 100.0]), None, "a count past any report is none");
        assert_eq!(restore_report(&[5680.0, 1.0, 100.0, -1.0, 1.0, 1.0]), None, "so is a fate that is no number");
        let report =
            restore_report(&[5680.0, 3.0, 0.0, 0.0, 1.0, 1.0, 100.0, 1.0, 1.0, 2.0, 104.0, 4.0, 1.0, 0.0]).unwrap();
        assert_eq!(report[1], SectionOutcome { key: 100, fate: 1, saved: 1, version: 2 });
        assert_eq!(
            describe_restore(Some(&report)).as_deref(),
            Some(
                "kept 1 of 2 saved sections; migrated section 100 from version 1 to 2; dropped section 104 (no longer saved)"
            )
        );
        assert_eq!(describe_restore(Some(&report[..1])), None, "all kept");
    }
}
