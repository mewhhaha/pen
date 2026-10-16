use super::*;
use crate::game::Edits;

fn moment(value: f32) -> Moment {
    Moment { saved: SavedState { schema: 1, state: vec![value] }, edits: Edits::default() }
}

fn frame(elapsed: f32) -> Recorded {
    Recorded { input: vec![elapsed, 64.0, 64.0, 0.0, 0.0], ..Recorded::default() }
}

/// Runs `count` frames as a session does: a moment when wanted, then the frame.
fn run(timeline: &mut Timeline, count: u64) {
    for _ in 0..count {
        if timeline.wants_moment() {
            let position = timeline.position() as f32;
            timeline.keep_moment(moment(position));
        }
        timeline.ran(frame(1.0 / 60.0));
    }
}

#[test]
fn keeps_moments_every_interval_and_trims_to_the_window() {
    let mut timeline = Timeline::default();
    timeline.keep_history(Some(2.0));
    run(&mut timeline, 600);
    assert_eq!(timeline.position(), 600);
    assert_eq!(timeline.head(), 600);
    // Two seconds are 120 frames: the newest moment at least that far back.
    let earliest = timeline.earliest().unwrap();
    assert!(earliest.is_multiple_of(MOMENT_INTERVAL) && (450..=480).contains(&earliest), "{earliest}");
    assert!(timeline.seconds() >= 2.0 && timeline.seconds() < 2.6);
    assert_eq!(timeline.seconds_back(1.0), 540);
}

#[test]
fn without_history_nothing_is_kept() {
    let mut timeline = Timeline::default();
    run(&mut timeline, 100);
    assert_eq!((timeline.position(), timeline.head(), timeline.earliest()), (100, 100, None));
    assert!(timeline.next().is_none());

    // A recording's frames play, then play goes on live with nothing kept.
    timeline.load(Recording { start: moment(0.0), frames: vec![frame(0.5); 3] });
    assert_eq!((timeline.position(), timeline.head(), timeline.earliest()), (0, 3, Some(0)));
    run(&mut timeline, 5);
    assert_eq!((timeline.position(), timeline.head(), timeline.earliest()), (5, 5, None));
}

#[test]
fn going_live_in_the_past_drops_the_future() {
    let mut timeline = Timeline::default();
    timeline.keep_history(None);
    run(&mut timeline, 100);
    let moment = timeline.seek(60);
    timeline.restored();
    assert_eq!(moment.saved.state, [60.0]);
    assert!(timeline.restored_here());
    timeline.ran(frame(1.0 / 60.0));
    assert!(timeline.recorded(60).unwrap().restored);
    timeline.go_live();
    assert_eq!((timeline.position(), timeline.head()), (61, 61));
    assert_eq!(timeline.moments.keys().copied().collect::<Vec<_>>(), [0, 30, 60]);

    // Restored at the cursor, then live: the next frame recorded follows a restore.
    timeline.seek(60);
    timeline.restored();
    timeline.go_live();
    assert_eq!(timeline.head(), 60);
    timeline.ran(frame(1.0 / 60.0));
    timeline.ran(frame(1.0 / 60.0));
    assert!(timeline.recorded(60).unwrap().restored && !timeline.recorded(61).unwrap().restored);
}

#[test]
fn a_new_build_replaces_the_moments_after_a_restart() {
    let mut timeline = Timeline::default();
    timeline.keep_history(None);
    run(&mut timeline, 100);
    timeline.rebuilt();
    // The same build: later moments still follow.
    timeline.seek(90);
    assert_eq!(timeline.moments.len(), 4);
    // An older build's moment: those after it are stale.
    timeline.seek(30);
    assert_eq!(timeline.moments.keys().copied().collect::<Vec<_>>(), [0, 30]);
    assert!(timeline.wants_moment());
    run(&mut timeline, 70);
    assert_eq!(timeline.moments.keys().copied().collect::<Vec<_>>(), [0, 30, 60, 90]);
    assert!(timeline.moments.values().skip(1).all(|kept| kept.build == 1));
}

#[test]
fn a_history_is_a_recording_and_counts_back_in_game_time() {
    let mut timeline = Timeline::default();
    timeline.keep_history(Some(1.0));
    run(&mut timeline, 120);
    let earliest = timeline.earliest().unwrap();
    assert_eq!(timeline.moments().next(), Some(earliest));
    assert_eq!(timeline.moments().last(), Some(90));
    let recording = timeline.recording().unwrap();
    assert_eq!(recording.start, moment(earliest as f32));
    assert_eq!(recording.frames.len() as u64, 120 - earliest);
    // Half a second of 60 Hz frames, then steps' worth before the history.
    assert_eq!(timeline.frames_before(120, 0.5), 90);
    assert_eq!(timeline.frames_before(120, 100.0), earliest);
    assert_eq!(Timeline::default().recording(), None);
}
