//! What the dev tools look into, worked out alike in every host: the game's
//! packets frame by frame and how two frames differ (`packets`,
//! `packet_diff`), the sections of its saved state and how two states
//! differ (`state`, `state_diff`), and a watch that pauses the game when a
//! lane of its state comes to hold a value (`watch`), and a guest's samples
//! as a flame graph (`flame`). The answers' layouts are DevtoolsOperation's
//! in `protocol/schema.json`.
use std::collections::VecDeque;

use pen_protocol::boundary::whole;
use pen_protocol::{RenderFrame, devtools_compare, devtools_packet};

use crate::services::{Answer, code_points};

/// The frames the packet inspector keeps at most.
pub const KEPT_FRAMES: usize = 120;
/// The most lanes of a section one state answer carries.
pub const STATE_LANES: usize = 256;

const STATE_MAGIC: f32 = 5678.0;
/// Lanes before an input packet's events: elapsed time, viewport width and
/// height, the event count and the state's length.
const INPUT_HEADER: usize = 5;
const EVENT_LANES: usize = 5;

/// A frame's packets as the inspector keeps them: its input's lanes, its
/// render packet's length and the frame the packet decoded to.
#[derive(Clone, Debug)]
pub struct KeptFrame {
    pub position: u64,
    pub input: Vec<f32>,
    pub packet: usize,
    pub frame: RenderFrame,
}

/// The input's event records, five lanes each.
fn events(input: &[f32]) -> &[f32] {
    let count = input.get(3).copied().and_then(whole).map_or(0, |count| count as usize);
    let end = (INPUT_HEADER + count * EVENT_LANES).min(input.len());
    input.get(INPUT_HEADER..end).unwrap_or(&[])
}

/// The frame's requests to the host, a bit each in DevtoolsPacket's order.
fn requests(frame: &RenderFrame) -> u32 {
    [
        frame.save_requested,
        frame.load_requested,
        frame.scene_save_requested,
        frame.scene_load_requested,
        frame.checkpoint_requested,
        frame.undo_requested,
        frame.redo_requested,
        frame.exit_requested,
    ]
    .iter()
    .enumerate()
    .fold(0, |bits, (bit, &asked)| bits | (u32::from(asked) << bit))
}

/// How many of the frame's settings (everything but its lists and
/// requests) differ between `a` and `b`.
fn settings_changed(a: &RenderFrame, b: &RenderFrame) -> usize {
    [
        a.view != b.view,
        a.projection != b.projection,
        a.clear != b.clear,
        a.ambient != b.ambient,
        a.light_direction != b.light_direction,
        a.light_color != b.light_color,
        a.aberration != b.aberration,
        a.quality != b.quality,
        a.sky != b.sky,
        a.fog != b.fog,
        a.shadow != b.shadow,
        a.blur != b.blur,
        a.clouds != b.clouds,
        a.hide_cursor != b.hide_cursor,
        a.preferences != b.preferences,
    ]
    .into_iter()
    .filter(|changed| *changed)
    .count()
}

/// How many elements of two lists differ: those at the same place that are
/// not equal, and those only one has.
fn changed<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.iter().zip(b).filter(|(a, b)| a != b).count() + a.len().abs_diff(b.len())
}

/// The lists of a frame in DevtoolsPacket's order from `draws` to `voices`,
/// each as its length and how many elements differ from `other`'s.
fn lists(frame: &RenderFrame, other: &RenderFrame) -> [(usize, usize); 11] {
    [
        (frame.draws.len(), changed(&frame.draws, &other.draws)),
        (frame.solids.len(), changed(&frame.solids, &other.solids)),
        (frame.decals.len(), changed(&frame.decals, &other.decals)),
        (frame.sounds.len(), changed(&frame.sounds, &other.sounds)),
        (frame.sprites.len(), changed(&frame.sprites, &other.sprites)),
        (frame.insets.len(), changed(&frame.insets, &other.insets)),
        (frame.fields.len(), changed(&frame.fields, &other.fields)),
        (frame.ui.len(), changed(&frame.ui, &other.ui)),
        (frame.services.len(), changed(&frame.services, &other.services)),
        (frame.buses.len(), changed(&frame.buses, &other.buses)),
        (frame.voices.len(), changed(&frame.voices, &other.voices)),
    ]
}

impl KeptFrame {
    /// Its numbers in DevtoolsPacket's order.
    pub fn summary(&self) -> Vec<f32> {
        let mut values = vec![0.0; devtools_packet::NAMES.len()];
        values[devtools_packet::POSITION as usize] = self.position as f32;
        values[devtools_packet::EVENTS as usize] = (events(&self.input).len() / EVENT_LANES) as f32;
        values[devtools_packet::INPUT as usize] = self.input.len() as f32;
        values[devtools_packet::PACKET as usize] = self.packet as f32;
        values[devtools_packet::SETTINGS as usize] = 1.0;
        for (at, (length, _)) in lists(&self.frame, &self.frame).into_iter().enumerate() {
            values[devtools_packet::DRAWS as usize + at] = length as f32;
        }
        values[devtools_packet::REQUESTS as usize] = requests(&self.frame) as f32;
        values
    }
}

/// How frame `b` differs from frame `a`: for each of DevtoolsPacket's
/// numbers after `position`, its count in `a`, its count in `b` and how
/// many of its elements differ (events by record, input lanes one by one,
/// settings by field, requests by bit).
pub fn packet_diff(a: &KeptFrame, b: &KeptFrame) -> Vec<f32> {
    let (events_a, events_b) = (events(&a.input), events(&b.input));
    let records = |lanes: &[f32]| lanes.chunks(EVENT_LANES).map(<[f32]>::to_vec).collect::<Vec<_>>();
    let mut triples = vec![
        (events_a.len() / EVENT_LANES, events_b.len() / EVENT_LANES, changed(&records(events_a), &records(events_b))),
        (a.input.len(), b.input.len(), changed(&a.input, &b.input)),
        (a.packet, b.packet, usize::from(a.packet != b.packet)),
        (1, 1, settings_changed(&a.frame, &b.frame)),
    ];
    for ((length_a, changes), (length_b, _)) in lists(&a.frame, &b.frame).into_iter().zip(lists(&b.frame, &a.frame)) {
        triples.push((length_a, length_b, changes));
    }
    let (bits_a, bits_b) = (requests(&a.frame), requests(&b.frame));
    triples.push((bits_a as usize, bits_b as usize, (bits_a ^ bits_b).count_ones() as usize));
    triples.into_iter().flat_map(|(a, b, changes)| [a as f32, b as f32, changes as f32]).collect()
}

/// The last frames' packets, kept while the dev tools look at them.
#[derive(Debug, Default)]
pub struct Packets {
    keeping: bool,
    frames: VecDeque<KeptFrame>,
}

impl Packets {
    /// Keeps each frame from now on, or stops and forgets them.
    pub fn keep(&mut self, keeping: bool) {
        self.keeping = keeping;
        if !keeping {
            self.frames.clear();
        }
    }

    pub fn keeping(&self) -> bool {
        self.keeping
    }

    /// A frame ran: its position, its input and the packet it gave (its
    /// length in lanes, and decoded). Running a frame again (after a
    /// rewind) replaces the frames kept from there on.
    pub fn ran(&mut self, position: u64, input: &[f32], packet: usize, frame: &RenderFrame) {
        if !self.keeping {
            return;
        }
        while self.frames.back().is_some_and(|kept| kept.position >= position) {
            self.frames.pop_back();
        }
        if self.frames.len() == KEPT_FRAMES {
            self.frames.pop_front();
        }
        self.frames.push_back(KeptFrame { position, input: input.to_vec(), packet, frame: frame.clone() });
    }

    pub fn frame(&self, position: u64) -> Option<&KeptFrame> {
        self.frames.iter().find(|kept| kept.position == position)
    }

    /// The answer to a packets request: the frames kept, oldest first.
    pub fn answer(&self) -> Answer {
        let values = self.frames.iter().flat_map(KeptFrame::summary).collect();
        Answer { values, ..Answer::ok(self.frames.len() as f32) }
    }

    /// The answer to a packet_diff request; unavailable unless both frames
    /// are kept.
    pub fn diff(&self, a: u64, b: u64) -> Answer {
        match (self.frame(a), self.frame(b)) {
            (Some(a), Some(b)) => Answer { values: packet_diff(a, b), ..Answer::ok(1.0) },
            _ => Answer::unavailable(),
        }
    }
}

/// A section of a saved state (Ink's persist.blot framing: `magic, schema,
/// count`, then each section's `key, version, width, length` and lanes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Section<'a> {
    pub key: u32,
    pub version: u32,
    pub width: u32,
    pub lanes: &'a [f32],
}

/// The sections of a saved state, as far as they are whole.
pub fn sections(state: &[f32]) -> Vec<Section<'_>> {
    let mut found = Vec::new();
    if state.len() < 3 || state[0] != STATE_MAGIC {
        return found;
    }
    let Some(count) = whole(state[2]) else { return found };
    let mut offset = 3;
    for _ in 0..count {
        let Some(&[key, version, width, length]) =
            state.get(offset..offset + 4).and_then(|header| <&[f32; 4]>::try_from(header).ok())
        else {
            break;
        };
        let start = offset + 4;
        let (Some(key), Some(version), Some(width), Some(length)) =
            (whole(key), whole(version), whole(width), whole(length))
        else {
            break;
        };
        let Some(lanes) = start.checked_add(length as usize).and_then(|end| state.get(start..end)) else { break };
        found.push(Section { key, version, width, lanes });
        offset = start + lanes.len();
    }
    found
}

/// The answer listing a state's sections: each its key, version, width and
/// length; the value is the state's schema.
pub fn sections_answer(schema: u32, state: &[f32]) -> Answer {
    let values = sections(state)
        .iter()
        .flat_map(|section| {
            [section.key as f32, section.version as f32, section.width as f32, section.lanes.len() as f32]
        })
        .collect();
    Answer { values, ..Answer::ok(schema as f32) }
}

/// The answer with section `key`'s lanes from `from` on, as many as one
/// carries; the value is its length. Unavailable for a section the state
/// lacks.
pub fn lanes_answer(state: &[f32], key: u32, from: usize) -> Answer {
    match sections(state).into_iter().find(|section| section.key == key) {
        Some(section) => {
            let values = section.lanes.iter().skip(from).take(STATE_LANES).copied().collect();
            Answer { values, ..Answer::ok(section.lanes.len() as f32) }
        }
        None => Answer::unavailable(),
    }
}

/// The lanes of section `key` among `sections`; none when it is not there.
fn lanes_of<'a>(sections: &[Section<'a>], key: u32) -> &'a [f32] {
    sections.iter().find(|section| section.key == key).map_or(&[], |section| section.lanes)
}

/// How state `now` differs from state `then`: for each section of either,
/// in `now`'s order and then `then`'s, its key, its length then and now,
/// how many lanes differ and the first that does (-1 for none).
pub fn state_diff(then: &[f32], now: &[f32]) -> Vec<f32> {
    let (then, now) = (sections(then), sections(now));
    let keys = now.iter().chain(&then).map(|section| section.key);
    let mut seen = Vec::new();
    let mut values = Vec::new();
    for key in keys {
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        let (before, after) = (lanes_of(&then, key), lanes_of(&now, key));
        // Bit for bit: NaN is not in a state, and -0 differs from 0 there.
        let differs = |at: usize| before.get(at).map(|lane| lane.to_bits()) != after.get(at).map(|lane| lane.to_bits());
        let longest = before.len().max(after.len());
        let count = (0..longest).filter(|&at| differs(at)).count();
        let first = (0..longest).find(|&at| differs(at)).map_or(-1.0, |at| at as f32);
        values.extend([key as f32, before.len() as f32, after.len() as f32, count as f32, first]);
    }
    values
}

/// The lane `lane` of section `key`, if the state has it.
pub fn lane(state: &[f32], key: u32, lane: u32) -> Option<f32> {
    lanes_of(&sections(state), key).get(lane as usize).copied()
}

/// A watch on a lane of the game's state: it pauses the game once the lane
/// compares with `value` as `compare` (DevtoolsCompare) says, having not
/// before.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Watch {
    pub key: u32,
    pub lane: u32,
    pub compare: u32,
    pub value: f32,
    /// The lane when the watch started, for `changes`.
    pub start: Option<f32>,
    /// The lane as last seen.
    pub now: Option<f32>,
    /// Whether it held when last looked at: it pauses when it starts to.
    pub held: bool,
    /// The frame it paused the game at.
    pub paused: Option<u64>,
}

impl Watch {
    pub fn new(key: u32, lane: u32, compare: u32, value: f32) -> Watch {
        Watch { key, lane, compare, value, start: None, now: None, held: false, paused: None }
    }

    /// Whether `lane` (the lane now) is what the watch waits for.
    fn holds(&self, lane: f32) -> bool {
        match self.compare {
            devtools_compare::EQUAL => lane == self.value,
            devtools_compare::UNEQUAL => lane != self.value,
            devtools_compare::BELOW => lane < self.value,
            devtools_compare::ABOVE => lane > self.value,
            devtools_compare::CHANGES => self.start.is_some_and(|start| start.to_bits() != lane.to_bits()),
            _ => false,
        }
    }

    /// Looks at the state after the frame before `position`: whether the
    /// game should pause there, the watch having just come to hold.
    pub fn look(&mut self, state: &[f32], position: u64) -> bool {
        let Some(now) = lane(state, self.key, self.lane) else {
            self.now = None;
            self.held = false;
            return false;
        };
        self.start.get_or_insert(now);
        self.now = Some(now);
        let holds = self.holds(now);
        let started = holds && !self.held;
        self.held = holds;
        if started {
            self.paused = Some(position);
        }
        started
    }

    /// The answer describing it: key, lane, comparison, value, the lane now
    /// (or -1) and the frame it paused at (or -1).
    pub fn values(&self) -> Vec<f32> {
        vec![
            self.key as f32,
            self.lane as f32,
            self.compare as f32,
            self.value,
            self.now.unwrap_or(-1.0),
            self.paused.map_or(-1.0, |frame| frame as f32),
        ]
    }
}

/// A count in two lanes, its high and low 24 bits, each exact as a float32.
pub fn halves(count: u64) -> [f32; 2] {
    [(count >> 24) as f32, (count & 0x00ff_ffff) as f32]
}

/// The answer to a memory request: the guest's committed bytes and the most
/// it committed, then each counter's name (its length and code points) and
/// value in two 24-bit halves; the value is the number of counters.
pub fn memory_answer(memory: usize, peak: usize, counters: &[(String, u64)]) -> Answer {
    let mut values = vec![memory as f32, peak as f32];
    for (name, count) in counters {
        let name = code_points(name);
        values.push(name.len() as f32);
        values.extend(name);
        values.extend(halves(*count));
    }
    Answer { values, ..Answer::ok(counters.len() as f32) }
}

/// The most nodes a flame answer carries, the deepest, and the least share
/// of the samples a node needs to be in it.
const FLAME_NODES: usize = 300;
const FLAME_DEPTH: usize = 128;
const FLAME_LEAST: u64 = 400;
/// The longest function name a flame answer carries.
const NAME_LENGTH: usize = 64;

/// A call in a flame graph: its function, its samples (its callees'
/// included) and those in it alone, and its callees.
#[derive(Debug, Default)]
struct Call {
    function: u32,
    samples: u64,
    alone: u64,
    callees: Vec<Call>,
}

impl Call {
    fn add(&mut self, stack: &[u32], samples: u64) {
        self.samples += samples;
        let Some((&function, rest)) = stack.split_first() else {
            self.alone += samples;
            return;
        };
        let at = match self.callees.iter().position(|callee| callee.function == function) {
            Some(at) => at,
            None => {
                self.callees.push(Call { function, ..Call::default() });
                self.callees.len() - 1
            }
        };
        self.callees[at].add(rest, samples);
    }
}

/// The functions a flame answer names as found running most.
const HOTTEST: usize = 8;

/// The answer to a flame request from a profiler's samples: each stack
/// (function indices from the outermost call in) and how many samples
/// found it, and the function names. First the functions found running
/// most (at the top of their stacks): their count, then each its index,
/// samples, name's length and code points. Then each node, a caller before
/// its callees and those by samples: its depth, function index, first
/// sample, samples, samples in it alone, its name's length and code
/// points. The value is the samples. Nodes with less than 1/400 of the
/// samples are left out, and past 300 nodes or 128 calls deep.
pub fn flame<'a>(stacks: impl IntoIterator<Item = (&'a [u32], u64)>, name: impl Fn(u32) -> String) -> Answer {
    let mut root = Call::default();
    let mut running: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    for (stack, samples) in stacks {
        root.add(stack, samples);
        if let Some(&top) = stack.last() {
            *running.entry(top).or_default() += samples;
        }
    }
    let mut hottest: Vec<(u32, u64)> = running.into_iter().collect();
    hottest.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    hottest.truncate(HOTTEST);
    let named = |function: u32| -> Vec<f32> { code_points(&name(function)).into_iter().take(NAME_LENGTH).collect() };
    let mut values = vec![hottest.len() as f32];
    for &(function, samples) in &hottest {
        let name = named(function);
        values.extend([function as f32, samples as f32, name.len() as f32]);
        values.extend(name);
    }
    let least = root.samples.div_ceil(FLAME_LEAST).max(1);
    let mut nodes = 0;
    // Depth first, callers before callees: (call, depth, first sample).
    let mut pending = vec![(&root, 0usize, 0u64)];
    while let Some((call, depth, first)) = pending.pop() {
        if depth > 0 {
            let name = named(call.function);
            values.extend([depth as f32 - 1.0, call.function as f32, first as f32, call.samples as f32]);
            values.extend([call.alone as f32, name.len() as f32]);
            values.extend(name);
            nodes += 1;
            if nodes == FLAME_NODES {
                break;
            }
        }
        if depth == FLAME_DEPTH {
            continue;
        }
        let mut callees: Vec<&Call> = call.callees.iter().filter(|callee| callee.samples >= least).collect();
        callees.sort_by(|a, b| b.samples.cmp(&a.samples).then(a.function.cmp(&b.function)));
        // Children from the first on, each starting where the one before ended.
        let mut starts = Vec::with_capacity(callees.len());
        let mut at = first;
        for callee in &callees {
            starts.push(at);
            at += callee.samples;
        }
        for (callee, start) in callees.into_iter().zip(starts).rev() {
            pending.push((callee, depth + 1, start));
        }
    }
    Answer { values, ..Answer::ok(root.samples as f32) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_protocol::Draw;

    /// A state of sections given as (key, lanes), version 1, width 0.
    fn state(sections: &[(u32, &[f32])]) -> Vec<f32> {
        let mut state = vec![STATE_MAGIC, 3.0, sections.len() as f32];
        for (key, lanes) in sections {
            state.extend([*key as f32, 1.0, 0.0, lanes.len() as f32]);
            state.extend_from_slice(lanes);
        }
        state
    }

    #[test]
    fn reads_a_states_sections_and_their_lanes() {
        let saved = state(&[(0, &[1.0, 1.0]), (11, &[4.0, 2.0, 9.0, 0.5])]);
        let found = sections(&saved);
        assert_eq!(found.len(), 2);
        assert_eq!((found[1].key, found[1].lanes), (11, &[4.0, 2.0, 9.0, 0.5][..]));
        assert_eq!(sections_answer(3, &saved).values, [0.0, 1.0, 0.0, 2.0, 11.0, 1.0, 0.0, 4.0]);
        assert_eq!(lanes_answer(&saved, 11, 1), Answer { values: vec![2.0, 9.0, 0.5], ..Answer::ok(4.0) });
        assert_eq!(lanes_answer(&saved, 7, 0), Answer::unavailable());
        // A section cut short, or another magic, reads as far as it is whole.
        assert_eq!(sections(&saved[..saved.len() - 1]).len(), 1);
        assert!(sections(&[5679.0, 3.0, 0.0]).is_empty());
    }

    #[test]
    fn says_which_sections_changed_and_where() {
        let then = state(&[(0, &[1.0, 1.0]), (5, &[3.0]), (11, &[4.0, 2.0])]);
        let now = state(&[(0, &[1.0, 1.0]), (11, &[4.0, 3.0, 7.0]), (12, &[1.0])]);
        assert_eq!(
            state_diff(&then, &now),
            [
                0.0, 2.0, 2.0, 0.0, -1.0, // the same
                11.0, 2.0, 3.0, 2.0, 1.0, // a lane changed and one more
                12.0, 0.0, 1.0, 1.0, 0.0, // new
                5.0, 1.0, 0.0, 1.0, 0.0, // gone
            ]
        );
    }

    #[test]
    fn a_watch_pauses_once_as_its_lane_comes_to_hold() {
        let at = |value: f32| state(&[(20, &[0.0, value])]);
        let mut watch = Watch::new(20, 1, devtools_compare::ABOVE, 5.0);
        assert!(!watch.look(&at(1.0), 10));
        assert!(watch.look(&at(6.0), 11), "it came to hold");
        assert!(!watch.look(&at(7.0), 12), "it held already");
        assert!(!watch.look(&at(2.0), 13));
        assert!(watch.look(&at(9.0), 14), "and again");
        assert_eq!(watch.values(), [20.0, 1.0, 3.0, 5.0, 9.0, 14.0]);
        let mut changes = Watch::new(20, 1, devtools_compare::CHANGES, 0.0);
        assert!(!changes.look(&at(1.0), 1) && !changes.look(&at(1.0), 2));
        assert!(changes.look(&at(2.0), 3), "it changed from where it started");
        let mut missing = Watch::new(99, 0, devtools_compare::EQUAL, 0.0);
        assert!(!missing.look(&at(0.0), 1));
        assert_eq!(missing.values()[4], -1.0);
    }

    #[test]
    fn lays_samples_out_as_a_flame_graph() {
        let stacks: [(&[u32], u64); 4] = [(&[1, 2, 3], 6), (&[1, 2], 2), (&[1, 4], 3), (&[5], 1)];
        let answer = flame(stacks, |function| format!("f{function}"));
        assert_eq!(answer.value, 12.0);
        // Found running most: f3 (6), f4 (3), f2 (2), f5 (1).
        assert_eq!(answer.values[..4], [4.0, 3.0, 6.0, 2.0]);
        let mut at = 1;
        let mut hottest = Vec::new();
        for _ in 0..4 {
            hottest.push(answer.values[at]);
            at += 3 + whole(answer.values[at + 2]).unwrap() as usize;
        }
        assert_eq!(hottest, [3.0, 4.0, 2.0, 5.0]);
        let mut nodes = Vec::new();
        while at < answer.values.len() {
            let length = whole(answer.values[at + 5]).unwrap() as usize;
            nodes.push(answer.values[at..at + 5].to_vec());
            assert_eq!(answer.values[at + 6], 'f' as u32 as f32);
            at += 6 + length;
        }
        assert_eq!(
            nodes,
            [
                [0.0, 1.0, 0.0, 11.0, 0.0],
                [1.0, 2.0, 0.0, 8.0, 2.0],
                [2.0, 3.0, 0.0, 6.0, 6.0],
                [1.0, 4.0, 8.0, 3.0, 3.0],
                [0.0, 5.0, 11.0, 1.0, 1.0],
            ]
        );
        // A call with too few samples is left out, its time still its caller's.
        let many: Vec<(Vec<u32>, u64)> = vec![(vec![1], 1000), (vec![1, 9], 1)];
        let answer = flame(many.iter().map(|(stack, samples)| (stack.as_slice(), *samples)), |_| String::new());
        assert_eq!(answer.values, [2.0, 1.0, 1000.0, 0.0, 9.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1001.0, 1000.0, 0.0]);
    }

    #[test]
    fn counts_cross_in_two_halves() {
        assert_eq!(halves(5), [0.0, 5.0]);
        assert_eq!(halves((3 << 24) + 7), [3.0, 7.0]);
        let answer = memory_answer(1 << 20, 2 << 20, &[("rc_retains".into(), 1 << 25)]);
        assert_eq!(answer.value, 1.0);
        assert_eq!(answer.values[..3], [1048576.0, 2097152.0, 10.0]);
        assert_eq!(answer.values[13..], [2.0, 0.0]);
    }

    fn kept(position: u64, events: &[[f32; 5]], draws: usize) -> KeptFrame {
        let mut input = vec![0.016, 960.0, 640.0, events.len() as f32, 0.0];
        input.extend(events.iter().flatten());
        let frame =
            RenderFrame { draws: vec![Draw::default(); draws], save_requested: position == 2, ..Default::default() };
        KeptFrame { position, input, packet: 100 + draws * 20, frame }
    }

    #[test]
    fn keeps_the_last_frames_and_runs_again_over_them() {
        let mut packets = Packets::default();
        packets.ran(0, &[], 0, &RenderFrame::default());
        assert_eq!(packets.answer().value, 0.0, "nothing kept until asked");
        packets.keep(true);
        for position in 0..KEPT_FRAMES as u64 + 5 {
            let frame = kept(position, &[], 1);
            packets.ran(position, &frame.input, frame.packet, &frame.frame);
        }
        assert_eq!(packets.answer().value, KEPT_FRAMES as f32);
        assert!(packets.frame(4).is_none() && packets.frame(5).is_some());
        // A rewind to frame 100 runs it again: the frames after it go.
        packets.ran(100, &[], 0, &RenderFrame::default());
        assert_eq!(packets.frames.back().map(|kept| kept.position), Some(100));
        assert_eq!(packets.frames.len(), 96);
        packets.keep(false);
        assert_eq!(packets.answer().value, 0.0);
    }

    #[test]
    fn summarizes_and_diffs_frames() {
        let key = [1.0, 65.0, 0.0, 0.0, 0.0];
        let pointer = [5.0, 0.0, 10.0, 20.0, 0.0];
        let (a, b) = (kept(1, &[key, pointer], 3), kept(2, &[key], 5));
        let summary = b.summary();
        assert_eq!(summary.len(), devtools_packet::NAMES.len());
        assert_eq!(summary[devtools_packet::POSITION as usize], 2.0);
        assert_eq!(summary[devtools_packet::EVENTS as usize], 1.0);
        assert_eq!(summary[devtools_packet::INPUT as usize], 10.0);
        assert_eq!(summary[devtools_packet::PACKET as usize], 200.0);
        assert_eq!(summary[devtools_packet::DRAWS as usize], 5.0);
        assert_eq!(summary[devtools_packet::REQUESTS as usize], 1.0, "a save");
        let diff = packet_diff(&a, &b);
        assert_eq!(diff.len(), (devtools_packet::NAMES.len() - 1) * 3);
        let triple = |field: u32| &diff[(field as usize - 1) * 3..field as usize * 3];
        assert_eq!(triple(devtools_packet::EVENTS), [2.0, 1.0, 1.0]);
        assert_eq!(triple(devtools_packet::INPUT), [15.0, 10.0, 6.0], "the count lane and five lanes gone");
        assert_eq!(triple(devtools_packet::SETTINGS), [1.0, 1.0, 0.0]);
        assert_eq!(triple(devtools_packet::DRAWS), [3.0, 5.0, 2.0]);
        assert_eq!(triple(devtools_packet::REQUESTS), [0.0, 1.0, 1.0]);
        let mut packets = Packets::default();
        packets.keep(true);
        for frame in [&a, &b] {
            packets.ran(frame.position, &frame.input, frame.packet, &frame.frame);
        }
        assert_eq!(packets.diff(1, 2).values, diff);
        assert_eq!(packets.diff(1, 3), Answer::unavailable());
    }
}
