//! The Rust and Blot packet codecs, checked against each other on random
//! frames. Each side's encoder and decoder mirror each other, so a round
//! trip within one side cannot show the two sides laying a packet out
//! differently. Instead each side walks the values it holds in one order that
//! lays nothing out (pen-codegen's canonical.rs), and the walks must agree to
//! the bit:
//!
//! - input: Rust encodes a random frame (`encode_frame`), the game's Blot
//!   decodes it into a named event pool (`decode_input_into`) and walks what it got, which must be Rust's
//!   walk of the frame (`canonical_input`);
//! - render: Rust walks a random frame (`canonical_render`), Blot reads the
//!   walk directly into named packet, child-draw and extra-lane pools
//!   (`render_packet_from_canonical`), then projects those flat lanes through
//!   the actual production `encode_render_views`. Both packets must agree to
//!   the bit, and Rust decodes the production packet
//!   (`decode_frame`) into the frame it walked, to the bit.
//!
//! The Blot side (codec_round_trip.blot) compiles through the sibling
//! checkout's compiler plugin, with the bindings this pen's schema generates
//! standing in for the checkout's `pen/`: it checks this pen's codegen,
//! whatever the checkout holds. Without a checkout it skips, as a missing
//! fixture does. PEN_CODEC_CASES says how many frames each property tries
//! (default 2000), PEN_CODEC_SEED the first one's seed (decimal or 0x hex); a
//! failure names its frame's seed, which PEN_CODEC_CASES=1 tries alone.
#![allow(clippy::cast_possible_truncation, reason = "random numbers below small bounds, picked as u64")]
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Result;
use blot_host::{Guest, GuestModule, Value};
use pen_plugins::{Builder, Expectation, Level, Overlay, Plugins, TestModule};
use pen_protocol::boundary::{
    MAX_BUSES, MAX_DRAW_EXTRA, MAX_DRAWS, MAX_FIELD, MAX_INSETS, MAX_SERVICE_DATA, MAX_SERVICE_ID,
    MAX_SERVICE_OPERATION, MAX_SOLID_FACES, MAX_UI_ID, MAX_UI_NODES, MAX_UI_TEXT, MAX_VOICE_ID, PREFERENCE_FLOATS,
    canonical_input, canonical_render, decode_frame, encode_frame, whole,
};
use pen_protocol::{
    Aberration, Bus, Catalog, Clouds, Color, Decal, Draw, Drift, Emitter, FieldPatch, Fog, Frame, GameEvent, Inset,
    Instance, InstanceRemoval, Light, Listener, Panner, Pose, Quality, RenderFrame, ServiceRequest, Shadow, Sky, Solid,
    SolidFace, Sound, Sprite, UiNode, Vector, Viewport, Voice, service, ui_live, ui_role, ui_state,
};

const HARNESS: &str = include_str!("codec_round_trip.blot");
/// The largest whole number a lane holds exactly (Blot's `lanes.max_integer`).
const MAX_INTEGER: u32 = 16_777_215;
const CASES: u64 = 2000;
const SEED: u64 = 0x5eed_c0de_ca5e_0000;
/// GameEvent's variants, which `variant` numbers.
const VARIANTS: usize = 30;
/// The first of them, which pen's own bindings decode: all but Game, the
/// game's own events, which only a checkout that declares them decodes (its
/// pen/extensions.blot), so the random frames leave it out.
const CORE_VARIANTS: usize = 29;

#[test]
fn input_packets_mean_the_same_to_rust_and_blot() {
    let Some(wasm) = harness() else { return };
    let (cases, first) = cases();
    let started = Instant::now();
    let seen = with_guest(wasm, |guest| {
        let mut seen = [0; VARIANTS];
        for case in 0..cases {
            let seed = first.wrapping_add(case);
            let failed = |what: String| -> ! { panic!("input {}\n{what}", describe(case, cases, seed)) };
            let (frame, elapsed, state) = random_input(&mut Rng(seed));
            for event in &frame.events {
                seen[variant(event)] += 1;
            }
            let packet = encode_frame(&frame, elapsed, &state)
                .unwrap_or_else(|error| failed(format!("pen-protocol refused the random frame: {error:#}")));
            let expected = canonical_input(&frame, elapsed, &state);
            let actual = call(guest, "input_round_trip", packet)
                .unwrap_or_else(|error| failed(format!("the game's decode_input_into failed: {error:#}\n{frame:#?}")));
            if let Some(lane) = first_difference(&expected, &actual) {
                failed(format!(
                    "the game's walk of the packet differs from Rust's walk of the frame at lane {lane}, {}:\n{}\n{frame:#?}",
                    input_place(&frame, lane),
                    around(("Rust", &expected), ("Blot", &actual), lane)
                ));
            }
        }
        seen
    });
    let events: usize = seen.iter().sum();
    eprintln!(
        "input: {cases} frames ({events} events) from seed {first:#x} agree, in {:.1} s; events by variant: {seen:?}",
        started.elapsed().as_secs_f64()
    );
    if cases >= 100 {
        assert!(seen[..CORE_VARIANTS].iter().all(|&count| count > 0), "a GameEvent variant never came up: {seen:?}");
    }
}

#[test]
fn render_packets_mean_the_same_to_rust_and_blot() {
    let Some(wasm) = harness() else { return };
    let (cases, first) = cases();
    let started = Instant::now();
    let seen = with_guest(wasm, |guest| {
        let mut seen = [0; SECTIONS.len()];
        for case in 0..cases {
            let seed = first.wrapping_add(case);
            let failed = |what: String| -> ! { panic!("render {}\n{what}", describe(case, cases, seed)) };
            let mut rng = Rng(seed);
            let catalog = random_catalog(&mut rng);
            let (frame, schema, state) = random_render(&mut rng, &catalog);
            for (count, held) in seen.iter_mut().zip(sections(&frame)) {
                *count += usize::from(held);
            }
            let walk = canonical_render(&frame, schema, &state);
            let packet = call(guest, "render_round_trip", walk.clone()).unwrap_or_else(|error| {
                failed(format!("the game's canonical packet writer failed: {error:#}\n{frame:#?}"))
            });
            let streamed = call(guest, "render_stream_round_trip", walk.clone())
                .unwrap_or_else(|error| failed(format!("the canonical stream writer failed: {error:#}\n{frame:#?}")));
            if let Some(lane) = first_difference(&streamed, &packet) {
                failed(format!(
                    "production encode_render_views differs from the canonical stream at lane {lane}:\n{}",
                    around(("stream", &streamed), ("production", &packet), lane)
                ));
            }
            let decoded = decode_frame(&packet, schema, &catalog).unwrap_or_else(|error| {
                failed(format!("pen-protocol refused the game's packet: {error:#}\n{catalog:?}\n{frame:#?}"))
            });
            let again = canonical_render(&decoded.render, schema, &decoded.state);
            let lane = first_difference(&walk, &again);
            if lane.is_some() || decoded.render != frame || decoded.state != state {
                let field = first_field_difference(&format!("{frame:#?}"), &format!("{:#?}", decoded.render));
                let state = first_difference(&state, &decoded.state).map(|lane| format!("state lane {lane}"));
                let lane = lane.unwrap_or(0);
                failed(format!(
                    "the frame Rust decoded from the game's packet is not the one walked: {}\nat lane {lane} of the walk:\n{}",
                    field.or(state).unwrap_or_else(|| "a value Debug does not show".to_owned()),
                    around(("walked", &walk), ("decoded", &again), lane)
                ));
            }
        }
        seen
    });
    let held: Vec<String> = SECTIONS.iter().zip(&seen).map(|(name, count)| format!("{name} {count}")).collect();
    eprintln!(
        "render: {cases} frames from seed {first:#x} agree, in {:.1} s; frames holding {}",
        started.elapsed().as_secs_f64(),
        held.join(", ")
    );
    if cases >= 100 {
        assert!(seen.iter().all(|&count| count > 0), "a section never came up: {}", held.join(", "));
    }
}

/// Without a checkout: the random frames are ones pen-protocol takes, and
/// the walks hold the values they are given where they say.
#[test]
fn random_frames_are_valid_and_walked_whole() {
    for case in 0..200 {
        let mut rng = Rng(SEED.wrapping_add(case));
        let (frame, elapsed, state) = random_input(&mut rng);
        let packet = encode_frame(&frame, elapsed, &state).unwrap();
        let walk = canonical_input(&frame, elapsed, &state);
        assert_eq!(walk[..4], packet[..4]);
        let (count, lanes) = walk[walk.len() - state.len() - 1..].split_first().unwrap();
        assert_eq!((whole(*count), lanes), (u32::try_from(state.len()).ok(), &state[..]));
        // Each event walks to its tag and at most the four lanes it travels in.
        let events = walk.len() - 5 - state.len();
        assert!((frame.events.len()..=frame.events.len() * 5).contains(&events), "{events} lanes");

        let catalog = random_catalog(&mut rng);
        let (render, schema, state) = random_render(&mut rng, &catalog);
        let walk = canonical_render(&render, schema, &state);
        assert_eq!(whole(walk[0]), Some(schema));
        let mut changed = render.clone();
        changed.hide_cursor = !changed.hide_cursor;
        assert_ne!(walk, canonical_render(&changed, schema, &state));
    }
}

/// The harness, compiled once for both properties; None (having said why)
/// when there is no checkout to compile it with.
fn harness() -> Option<&'static [u8]> {
    static WASM: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    WASM.get_or_init(compile).as_deref()
}

fn compile() -> Option<Vec<u8>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../gdev");
    let log: pen_plugins::Logger = Arc::new(|level, message| {
        if level == Level::Error {
            eprintln!("{message}");
        }
    });
    let Some(plugins) = Plugins::load(&root, log).ok().flatten().filter(|plugins| plugins.compiles()) else {
        crate::fixtures::missing("no ../gdev checkout whose pen.json chooses a compiler");
        return None;
    };
    let checkout = plugins.root().to_owned();
    // Where the plugin lays a checkout's test modules out (gdev's tests/README.md).
    let entry = checkout.join("src/test_fixture.blot").display().to_string();
    let bindings = pen_codegen::blot_bindings().expect("pen's schema generates the game's bindings");
    let mut overlays: Vec<Overlay> = bindings
        .into_iter()
        .map(|(name, source)| Overlay {
            path: checkout.join("pen").join(name).display().to_string(),
            contents: Some(source),
        })
        .collect();
    overlays.push(Overlay { path: entry.clone(), contents: Some(HARNESS.to_owned()) });
    let harness = TestModule {
        path: Path::new(env!("CARGO_MANIFEST_DIR")).join("src/codec_round_trip.blot").display().to_string(),
        entry: entry.clone(),
        overlays,
        prefix: String::new(),
        expect: Expectation::Checks,
    };
    let started = Instant::now();
    let mut builder = Builder::new(plugins, Path::new(&entry));
    let wasm = crate::blot_test::build_test_module(&mut builder, &harness)
        .unwrap_or_else(|error| panic!("the codec harness does not compile in {}:\n{error:#}", checkout.display()));
    let _ = builder.close();
    eprintln!(
        "compiled the codec harness with {}'s compiler in {:.1} s",
        checkout.display(),
        started.elapsed().as_secs_f64()
    );
    Some(wasm)
}

/// How many frames each property tries, and the first one's seed.
fn cases() -> (u64, u64) {
    let read = |name: &str| {
        let text = std::env::var(name).ok()?;
        let text = text.trim();
        let number = match text.strip_prefix("0x") {
            Some(hex) => u64::from_str_radix(hex, 16),
            None => text.parse(),
        };
        Some(number.unwrap_or_else(|_| panic!("{name} is {text}, not a number")))
    };
    (read("PEN_CODEC_CASES").unwrap_or(CASES), read("PEN_CODEC_SEED").unwrap_or(SEED))
}

/// A failing case, and how to try it alone.
fn describe(case: u64, cases: u64, seed: u64) -> String {
    format!("frame {case} of {cases} (seed {seed:#x}; PEN_CODEC_SEED={seed:#x} PEN_CODEC_CASES=1 tries it alone):")
}

/// Runs `work` with an instance of the harness, on a thread with the stack a
/// guest needs.
fn with_guest<T: Send>(wasm: &[u8], work: impl FnOnce(&mut Guest) -> T + Send) -> T {
    std::thread::scope(|scope| {
        let thread = std::thread::Builder::new()
            .name("pen-codec".into())
            .stack_size(blot_host::GUEST_STACK_BYTES + (8 << 20))
            .spawn_scoped(scope, || {
                let engine = blot_host::dev_engine().unwrap();
                let module = GuestModule::new(&engine, wasm, None).unwrap();
                let mut guest = Guest::new(&engine, &module).unwrap();
                // Room for a loaded machine; each call is a few milliseconds.
                guest.set_deadline(Duration::from_secs(60));
                work(&mut guest)
            })
            .unwrap();
        thread.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    })
}

fn call(guest: &mut Guest, entry: &str, lanes: Vec<f32>) -> Result<Vec<f32>> {
    guest.call(entry, Value::ArrayF32(lanes))?.into_f32_array()
}

/// Where two walks first differ to the bit, or None.
fn first_difference(expected: &[f32], actual: &[f32]) -> Option<usize> {
    let differs = expected.iter().zip(actual).position(|(expected, actual)| expected.to_bits() != actual.to_bits());
    differs.or_else(|| (expected.len() != actual.len()).then(|| expected.len().min(actual.len())))
}

/// Two walks' lanes about `lane`.
fn around(expected: (&str, &[f32]), actual: (&str, &[f32]), lane: usize) -> String {
    let start = lane.saturating_sub(4);
    let shown = |(name, lanes): (&str, &[f32])| {
        let end = (lane + 5).min(lanes.len());
        let near = lanes.get(start..end).unwrap_or_default();
        format!("  {name:<8} lanes {start}..{end} of {}: {near:?}", lanes.len())
    };
    format!("{}\n{}", shown(expected), shown(actual))
}

/// What lane `lane` of `frame`'s input walk holds.
fn input_place(frame: &Frame, lane: usize) -> String {
    const HEADER: [&str; 4] = ["elapsed", "the viewport's width", "the viewport's height", "the events' count"];
    if let Some(name) = HEADER.get(lane) {
        return (*name).to_owned();
    }
    let mut start = HEADER.len();
    for (index, event) in frame.events.iter().enumerate() {
        let alone = Frame { timestamp: 0.0, viewport: frame.viewport, events: vec![event.clone()] };
        let lanes = canonical_input(&alone, 0.0, &[]).len() - HEADER.len() - 1;
        if lane < start + lanes {
            return format!("event {index}'s lane {} ({event:?})", lane - start);
        }
        start += lanes;
    }
    if lane == start { "the state's count".to_owned() } else { format!("state lane {}", lane - start - 1) }
}

/// Where two values' pretty Debug output first differs: the path to it
/// (`.insets[0].draws[1].mesh`) and the two lines.
fn first_field_difference(expected: &str, actual: &str) -> Option<String> {
    let expected: Vec<&str> = expected.lines().collect();
    let actual: Vec<&str> = actual.lines().collect();
    let at = expected.iter().zip(&actual).position(|(expected, actual)| expected != actual)?;
    // The fields and elements open at that line: each one's label, whether it
    // is a list, and how many of its elements have gone by.
    let mut open: Vec<(String, bool, usize)> = Vec::new();
    let label = |line: &str, open: &[(String, bool, usize)]| match line.split_once(": ") {
        Some((field, _)) => format!(".{field}"),
        None => match open.last() {
            Some((_, true, index)) => format!("[{index}]"),
            _ => String::new(),
        },
    };
    for line in &expected[..at] {
        let line = line.trim();
        if line.starts_with(['}', ']', ')']) {
            open.pop();
            if let Some(parent) = open.last_mut() {
                parent.2 += 1;
            }
        } else if line.ends_with(['{', '[', '(']) {
            open.push((label(line, &open), line.ends_with('['), 0));
        } else if let Some(parent) = open.last_mut() {
            parent.2 += 1;
        }
    }
    let path: String =
        open.iter().map(|(label, ..)| label.as_str()).collect::<String>() + &label(expected[at].trim(), &open);
    Some(format!("{path}: walked `{}`, decoded `{}`", expected[at].trim(), actual[at].trim()))
}

/// SplitMix64: small, fast, and good enough to pick test values with.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number from 0 to `most`.
    fn upto(&mut self, most: u32) -> u32 {
        (self.next() % (u64::from(most) + 1)) as u32
    }

    /// An index into a list of `count`.
    fn pick(&mut self, count: usize) -> u32 {
        (self.next() % count as u64) as u32
    }

    fn flag(&mut self) -> bool {
        self.next() & 1 == 1
    }

    /// True one time in `odds`.
    fn one_in(&mut self, odds: u64) -> bool {
        self.next().is_multiple_of(odds)
    }

    /// A whole number from 0 to `most`, often one of its ends.
    fn whole(&mut self, most: u32) -> u32 {
        match self.next() % 5 {
            0 => 0,
            1 => most,
            2 => self.upto(most.min(9)),
            _ => self.upto(most),
        }
    }

    /// How many of something up to `most`: often none or a few, now and
    /// then a few dozen.
    fn count(&mut self, most: usize) -> usize {
        let most = u32::try_from(most).unwrap_or(u32::MAX);
        let count = match self.next() % 8 {
            0 | 1 => 0,
            2 => self.upto(most.min(40)),
            _ => self.upto(most.min(3)),
        };
        count as usize
    }

    /// A finite f32: zero of either sign, a small whole number, one of the
    /// format's extremes, a subnormal, a fraction, or any finite bits.
    fn number(&mut self) -> f32 {
        match self.next() % 10 {
            0 => 0.0,
            1 => -0.0,
            2 => self.upto(200) as f32 - 100.0,
            3 => {
                let tiny = f32::from_bits(1);
                let extremes = [f32::MAX, f32::MIN, f32::MIN_POSITIVE, -f32::MIN_POSITIVE, f32::EPSILON, tiny, -tiny];
                extremes[self.pick(extremes.len()) as usize]
            }
            4 => f32::from_bits(self.next() as u32 & 0x807f_ffff),
            5 | 6 => (self.next() >> 40) as f32 / 16_777_216.0 * 2.0 - 1.0,
            _ => loop {
                let value = f32::from_bits(self.next() as u32);
                if value.is_finite() {
                    break value;
                }
            },
        }
    }

    fn numbers(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.number()).collect()
    }

    fn lanes<const N: usize>(&mut self) -> [f32; N] {
        std::array::from_fn(|_| self.number())
    }

    fn vector(&mut self) -> Vector {
        Vector { x: self.number(), y: self.number(), z: self.number() }
    }

    fn list<T>(&mut self, count: usize, mut item: impl FnMut(&mut Rng) -> T) -> Vec<T> {
        (0..count).map(|_| item(self)).collect()
    }
}

/// Which of GameEvent's variants `event` is. A new one fails to compile here
/// until `random_event` makes it.
fn variant(event: &GameEvent) -> usize {
    match event {
        GameEvent::KeyDown(_) => 0,
        GameEvent::KeyUp(_) => 1,
        GameEvent::PointerDown { .. } => 2,
        GameEvent::PointerUp { .. } => 3,
        GameEvent::PointerMove { .. } => 4,
        GameEvent::Wheel { .. } => 5,
        GameEvent::FocusLost => 6,
        GameEvent::AssetReady { .. } => 7,
        GameEvent::AssetFailed { .. } => 8,
        GameEvent::Console { .. } => 9,
        GameEvent::Preference { .. } => 10,
        GameEvent::Tune { .. } => 11,
        GameEvent::UiAction { .. } => 12,
        GameEvent::ServiceAnswer { .. } => 13,
        GameEvent::ServiceData { .. } => 14,
        GameEvent::GamepadConnected { .. } => 15,
        GameEvent::GamepadDisconnected { .. } => 16,
        GameEvent::GamepadButton { .. } => 17,
        GameEvent::GamepadAxis { .. } => 18,
        GameEvent::Action { .. } => 19,
        GameEvent::TextInput { .. } => 20,
        GameEvent::Composition { .. } => 21,
        GameEvent::MouseMotion { .. } => 22,
        GameEvent::Touch { .. } => 23,
        GameEvent::Pan { .. } => 24,
        GameEvent::Pinch { .. } => 25,
        GameEvent::Player { .. } => 26,
        GameEvent::SessionStarted { .. } => 27,
        GameEvent::RendererReady => 28,
        GameEvent::Game { .. } => 29,
    }
}

/// A key's name: named keys, letters (whose capitals travel), digits,
/// others the host sends as 0, and any one UTF-16 unit.
fn key(rng: &mut Rng) -> String {
    const KEYS: [&str; 14] =
        ["a", "z", "A", "7", " ", "ArrowLeft", "Shift", "Enter", "Escape", "F12", "é", "😀", "Unidentified", ""];
    if rng.one_in(4) {
        return char::from_u32(rng.upto(0xffff)).unwrap_or('x').to_string();
    }
    KEYS[rng.pick(KEYS.len()) as usize].to_owned()
}

fn random_event(rng: &mut Rng) -> GameEvent {
    // Whole numbers the game reads from a lane: up to the lane's exact range,
    // or the schema's bound.
    let integer = |rng: &mut Rng| rng.whole(MAX_INTEGER);
    match rng.pick(CORE_VARIANTS) {
        0 => GameEvent::KeyDown(key(rng)),
        1 => GameEvent::KeyUp(key(rng)),
        2 => GameEvent::PointerDown { button: rng.whole(7), x: rng.number(), y: rng.number() },
        3 => GameEvent::PointerUp { button: rng.whole(7), x: rng.number(), y: rng.number() },
        4 => GameEvent::PointerMove { x: rng.number(), y: rng.number() },
        5 => GameEvent::Wheel { x: rng.number(), y: rng.number(), delta: rng.number() },
        6 => GameEvent::FocusLost,
        7 => GameEvent::AssetReady { key: key(rng), kind: "mesh".into(), revision: rng.next() },
        8 => GameEvent::AssetFailed { key: key(rng), kind: "sound".into(), revision: rng.next(), message: key(rng) },
        9 => GameEvent::Console { command: integer(rng) as f32, argument: integer(rng) as f32 },
        10 => GameEvent::Preference { index: integer(rng) as f32, value: rng.number() },
        11 => GameEvent::Tune { index: integer(rng), value: rng.number() },
        12 => GameEvent::UiAction { node: integer(rng), action: rng.whole(7), value: rng.number() },
        13 => GameEvent::ServiceAnswer {
            request: integer(rng),
            status: rng.whole(3),
            value: rng.number(),
            length: integer(rng),
        },
        14 => GameEvent::ServiceData { request: integer(rng), a: rng.number(), b: rng.number(), c: rng.number() },
        15 => GameEvent::GamepadConnected { pad: rng.whole(15), standard: integer(rng) },
        16 => GameEvent::GamepadDisconnected { pad: rng.whole(15) },
        17 => GameEvent::GamepadButton {
            button: rng.whole(31),
            pad: integer(rng),
            value: rng.number(),
            pressed: integer(rng),
        },
        18 => GameEvent::GamepadAxis { axis: rng.whole(15), pad: integer(rng), value: rng.number() },
        19 => GameEvent::Action { action: rng.whole(255), value: rng.number(), device: integer(rng) },
        20 => GameEvent::TextInput { character: rng.whole(0x10_ffff), composing: integer(rng) },
        21 => GameEvent::Composition { length: integer(rng), cursor: integer(rng) },
        22 => GameEvent::MouseMotion { dx: rng.number(), dy: rng.number() },
        23 => GameEvent::Touch { id: rng.whole(31), x: rng.number(), y: rng.number(), phase: integer(rng) },
        24 => GameEvent::Pan { touches: rng.whole(31), dx: rng.number(), dy: rng.number() },
        25 => GameEvent::Pinch { x: rng.number(), y: rng.number(), scale: rng.number() },
        26 => GameEvent::Player { index: rng.whole(255) },
        27 => GameEvent::SessionStarted { restored: rng.whole(1) },
        _ => GameEvent::RendererReady,
    }
}

/// A frame, the seconds it covers and the state, as `encode_frame` takes
/// them.
fn random_input(rng: &mut Rng) -> (Frame, f64, Vec<f32>) {
    let events = if rng.one_in(10) { 1 + rng.upto(63) as usize } else { rng.count(8) };
    let frame = Frame {
        timestamp: f64::from(rng.number()),
        viewport: Viewport { width: 1 + rng.whole(MAX_INTEGER - 1), height: 1 + rng.whole(MAX_INTEGER - 1) },
        events: rng.list(events, random_event),
    };
    let elapsed = match rng.next() % 4 {
        0 => 0.0,
        1 => -0.0,
        // An f64 the f32 lane rounds.
        2 => f64::from(rng.upto(1000)) * 0.001,
        _ => f64::from(rng.number().abs()),
    };
    let count = rng.count(300);
    (frame, elapsed, rng.numbers(count))
}

/// A few assets of each kind, sometimes no sounds.
fn random_catalog(rng: &mut Rng) -> Catalog {
    let names = |count: u32, kind: &str| (0..count).map(|index| format!("{kind}/{index}")).collect();
    Catalog {
        meshes: names(1 + rng.upto(3), "mesh"),
        materials: names(1 + rng.upto(5), "material"),
        sounds: names(rng.upto(3), "sound"),
    }
}

fn random_draw(rng: &mut Rng, catalog: &Catalog) -> Draw {
    Draw {
        mesh: rng.pick(catalog.meshes.len()),
        material: rng.pick(catalog.materials.len()),
        transform: rng.lanes(),
        parameters: rng.flag().then(|| rng.lanes()),
        extra: rng.one_in(3).then(|| {
            let vec4s = if rng.one_in(8) { MAX_DRAW_EXTRA } else { 1 + rng.count(MAX_DRAW_EXTRA - 1) };
            rng.numbers(vec4s * 4)
        }),
    }
}

/// A frame valid for `catalog` and the decoder's limits, its schema and its
/// state, as `canonical_render` takes them. The bounds the schema writes
/// inline (a bus, filtering, shadow texels) are written here too.
fn random_render(rng: &mut Rng, catalog: &Catalog) -> (RenderFrame, u32, Vec<f32>) {
    let material = |rng: &mut Rng| rng.pick(catalog.materials.len());
    let clip = |rng: &mut Rng| rng.pick(catalog.sounds.len());
    let bus = |rng: &mut Rng| rng.whole(u32::try_from(MAX_BUSES).unwrap() - 1);
    let schema = 1 + rng.whole(MAX_INTEGER - 1);
    let count = rng.count(300);
    let state = rng.numbers(count);
    let sounds = if catalog.sounds.is_empty() { 0 } else { rng.count(12) };
    let voices = if catalog.sounds.is_empty() { 0 } else { rng.count(12) };
    let mut frame = RenderFrame {
        save_requested: rng.flag(),
        load_requested: rng.flag(),
        scene_save_requested: rng.flag(),
        scene_load_requested: rng.flag(),
        checkpoint_requested: rng.flag(),
        undo_requested: rng.flag(),
        redo_requested: rng.flag(),
        exit_requested: rng.flag(),
        view: rng.lanes(),
        projection: rng.lanes(),
        clear: Color { r: rng.number(), g: rng.number(), b: rng.number(), a: rng.number() },
        ambient: rng.vector(),
        light_direction: rng.vector(),
        light_color: rng.vector(),
        aberration: Aberration { x: rng.number(), y: rng.number() },
        quality: Quality {
            // The decoder reads anything but 4 as 1.
            msaa: if rng.flag() { 4 } else { 1 },
            fxaa: rng.flag(),
            scale: rng.number(),
            filtering: rng.whole(16),
            occlusion: rng.flag(),
            sky: rng.flag(),
            effects: rng.flag(),
            bloom: rng.flag(),
            taa: rng.flag(),
        },
        sky: Sky {
            zenith: rng.vector(),
            horizon: rng.vector(),
            ground: rng.vector(),
            sun: rng.vector(),
            night: rng.number(),
        },
        fog: Fog { radius: rng.number(), density: rng.number(), height: rng.number() },
        shadow: Shadow { view: rng.lanes(), projection: rng.lanes(), texel: rng.number(), texels: rng.whole(8192) },
        blur: rng.number(),
        clouds: Clouds { cover: rng.number(), drift: Drift { x: rng.number(), z: rng.number() } },
        hide_cursor: rng.flag(),
        loading: rng.flag(),
        show_captions: rng.flag(),
        draws: {
            let count = rng.count(12);
            rng.list(count, |rng| random_draw(rng, catalog))
        },
        solids: {
            let count = rng.count(6);
            rng.list(count, |rng| {
                let faces = if rng.one_in(8) { MAX_SOLID_FACES } else { 4 + rng.count(MAX_SOLID_FACES - 4) };
                Solid {
                    faces: rng.list(faces, |rng| SolidFace {
                        normal: rng.vector(),
                        distance: rng.number(),
                        material: material(rng),
                    }),
                }
            })
        },
        decals: {
            let count = rng.count(6);
            rng.list(count, |rng| Decal { material: material(rng), transform: rng.lanes(), opacity: rng.number() })
        },
        sounds: rng.list(sounds, |rng| Sound {
            clip: clip(rng),
            bus: bus(rng),
            gain: rng.number(),
            pan: rng.number(),
            pitch: rng.number(),
            lowpass: rng.number(),
            reverb: rng.number(),
        }),
        sprites: {
            let count = rng.count(12);
            rng.list(count, |rng| Sprite {
                x: rng.number(),
                y: rng.number(),
                width: rng.number(),
                height: rng.number(),
                u0: rng.number(),
                v0: rng.number(),
                u1: rng.number(),
                v1: rng.number(),
                color: rng.lanes(),
                material: material(rng),
                parameters: rng.lanes(),
            })
        },
        insets: {
            let count = if rng.one_in(20) { MAX_INSETS } else { rng.count(3) };
            rng.list(count, |rng| Inset {
                x: rng.number(),
                y: rng.number(),
                width: rng.number(),
                height: rng.number(),
                view: rng.lanes(),
                projection: rng.lanes(),
                ambient: rng.vector(),
                light_direction: rng.vector(),
                light_color: rng.vector(),
                draws: {
                    let count = rng.count(4);
                    rng.list(count, |rng| random_draw(rng, catalog))
                },
                pixel_size: rng.number(),
                after_sprites: 0,
            })
        },
        // As many as the game keeps; none at all reads back as none.
        preferences: rng.flag().then(|| {
            let count = if rng.one_in(10) { PREFERENCE_FLOATS } else { 1 + rng.count(PREFERENCE_FLOATS - 1) };
            rng.numbers(count)
        }),
        fields: {
            let count = rng.count(4);
            rng.list(count, |rng| {
                let field = u32::try_from(MAX_FIELD).unwrap();
                let (width, height) = (rng.whole(field), rng.whole(field));
                let (x, z) = (rng.whole(width), rng.whole(height));
                let most = if rng.one_in(30) { 64 } else { 6 };
                let (columns, rows) = (rng.whole((width - x).min(most)), rng.whole((height - z).min(most)));
                let values = rng.numbers(columns as usize * rows as usize);
                FieldPatch { material: material(rng), width, height, x, z, columns, rows, values }
            })
        },
        ui: {
            let count = rng.count(8);
            rng.list(count, |rng| {
                let most = if rng.one_in(20) { u32::try_from(MAX_UI_TEXT).unwrap() } else { 12 };
                let text_length = rng.whole(most);
                let name_length = rng.whole(text_length);
                let description_length = rng.whole(text_length - name_length);
                UiNode {
                    id: rng.whole(u32::try_from(MAX_UI_ID).unwrap()),
                    parent: rng.whole(u32::try_from(MAX_UI_NODES).unwrap()),
                    role: rng.whole(u32::try_from(ui_role::NAMES.len()).unwrap() - 1),
                    x: rng.number(),
                    y: rng.number(),
                    width: rng.number(),
                    height: rng.number(),
                    focusable: rng.flag(),
                    focused: rng.flag(),
                    disabled: rng.flag(),
                    hidden: rng.flag(),
                    modal: rng.flag(),
                    checked: rng.whole(u32::try_from(ui_state::NAMES.len()).unwrap() - 1),
                    // Expanded and selected are never mixed.
                    expanded: rng.whole(ui_state::TRUE),
                    selected: rng.whole(ui_state::TRUE),
                    live: rng.whole(u32::try_from(ui_live::NAMES.len()).unwrap() - 1),
                    value: rng.number(),
                    minimum: rng.number(),
                    maximum: rng.number(),
                    step: rng.number(),
                    actions: rng.whole(255),
                    text_length,
                    name_length,
                    description_length,
                    text: (0..text_length).map(|_| rng.upto(0x10_ffff) as f32).collect(),
                }
            })
        },
        services: {
            let count = rng.count(6);
            rng.list(count, |rng| {
                let most = if rng.one_in(50) { u32::try_from(MAX_SERVICE_DATA).unwrap() } else { 8 };
                let value_count = rng.whole(most);
                ServiceRequest {
                    request: rng.whole(u32::try_from(MAX_SERVICE_ID).unwrap()),
                    service: rng.whole(u32::try_from(service::NAMES.len()).unwrap() - 1),
                    operation: rng.whole(u32::try_from(MAX_SERVICE_OPERATION).unwrap()),
                    a: rng.number(),
                    b: rng.number(),
                    c: rng.number(),
                    d: rng.number(),
                    value_count,
                    values: rng.numbers(value_count as usize),
                }
            })
        },
        buses: {
            let count = if rng.one_in(10) { MAX_BUSES } else { rng.count(4) };
            rng.list(count, |rng| Bus { gain: rng.number(), lowpass: rng.number(), reverb: rng.number() })
        },
        voices: rng.list(voices, |rng| Voice {
            id: rng.whole(u32::try_from(MAX_VOICE_ID).unwrap()),
            clip: clip(rng),
            bus: bus(rng),
            gain: rng.number(),
            pan: rng.number(),
            pitch: rng.number(),
            lowpass: rng.number(),
            reverb: rng.number(),
            looping: rng.flag(),
            start: rng.number(),
            fade_in: rng.number(),
            fade_out: rng.number(),
        }),
        lights: {
            let count = rng.count(8);
            rng.list(count, |rng| Light {
                position: rng.vector(),
                radius: rng.number(),
                color: rng.vector(),
                direction: rng.vector(),
                inner: rng.number(),
                outer: rng.number(),
            })
        },
        listener: {
            let count = rng.count(1);
            rng.list(count, |rng| Listener { position: rng.vector(), forward: rng.vector(), up: rng.vector() })
        },
        panners: {
            let count = rng.count(6);
            rng.list(count, |rng| Panner {
                voice: rng.whole(u32::try_from(MAX_VOICE_ID).unwrap()),
                sound: rng.whole(255),
                position: rng.vector(),
                orientation: rng.vector(),
                distance_model: rng.whole(2),
                ref_distance: rng.number(),
                max_distance: rng.number(),
                rolloff: rng.number(),
                cone_inner: rng.number(),
                cone_outer: rng.number(),
                cone_outer_gain: rng.number(),
            })
        },
        // The game's own sections follow the packet, and only a checkout that
        // declares them reads or writes them; pen's own bindings carry none.
        extensions: Vec::new(),
        poses: {
            let count = rng.count(8);
            rng.list(count, |rng| Pose {
                draw: rng.whole(u32::try_from(MAX_DRAWS).unwrap()),
                clip: rng.whole(65535),
                time: rng.number(),
                weight: rng.number(),
            })
        },
        emitters: {
            let count = rng.count(4);
            rng.list(count, |rng| Emitter {
                material: material(rng),
                transform: rng.lanes(),
                seed: rng.whole(MAX_INTEGER),
                time: rng.number(),
                duration: rng.number(),
                rate: rng.number(),
                lifetime: rng.number(),
                radius: rng.number(),
                velocity: rng.vector(),
                spread: rng.number(),
                gravity: rng.vector(),
                drag: rng.number(),
                size_start: rng.number(),
                size_end: rng.number(),
                color_start: Color { r: rng.number(), g: rng.number(), b: rng.number(), a: rng.number() },
                color_end: Color { r: rng.number(), g: rng.number(), b: rng.number(), a: rng.number() },
                additive: rng.flag(),
            })
        },
        retained_enabled: rng.flag(),
        retained_reset: rng.flag(),
        retained_after: 0,
        instances: {
            let count = rng.count(12);
            rng.list(count, |rng| Instance {
                id_low: rng.whole(65535),
                id_high: rng.whole(65535),
                order: rng.whole(MAX_INTEGER),
                mesh: rng.pick(catalog.meshes.len()),
                material: material(rng),
                transform: rng.lanes(),
                parameters: rng.flag().then(|| rng.lanes()),
            })
        },
        instance_removals: {
            let count = rng.count(12);
            rng.list(count, |rng| InstanceRemoval { id_low: rng.whole(65535), id_high: rng.whole(65535) })
        },
    };
    for inset in &mut frame.insets {
        inset.after_sprites = rng.whole(u32::try_from(frame.sprites.len()).unwrap());
    }
    frame.retained_after = rng.whole(u32::try_from(frame.draws.len()).unwrap());
    (frame, schema, state)
}

/// What `sections` reports of a frame, in its order.
const SECTIONS: [&str; 23] = [
    "draws",
    "draw parameters",
    "draw extras",
    "solids",
    "decals",
    "sounds",
    "sprites",
    "insets",
    "inset draws",
    "preferences",
    "field values",
    "interface nodes",
    "interface text",
    "service values",
    "buses",
    "voices",
    "poses",
    "emitters",
    "lights",
    "listener",
    "panners",
    "retained instances",
    "instance removals",
];

/// Which of `SECTIONS` a frame holds any of.
fn sections(frame: &RenderFrame) -> [bool; SECTIONS.len()] {
    let draws = || frame.draws.iter().chain(frame.insets.iter().flat_map(|inset| &inset.draws));
    [
        !frame.draws.is_empty(),
        draws().any(|draw| draw.parameters.is_some()),
        draws().any(|draw| draw.extra.is_some()),
        !frame.solids.is_empty(),
        !frame.decals.is_empty(),
        !frame.sounds.is_empty(),
        !frame.sprites.is_empty(),
        !frame.insets.is_empty(),
        frame.insets.iter().any(|inset| !inset.draws.is_empty()),
        frame.preferences.is_some(),
        frame.fields.iter().any(|field| !field.values.is_empty()),
        !frame.ui.is_empty(),
        frame.ui.iter().any(|node| !node.text.is_empty()),
        frame.services.iter().any(|request| !request.values.is_empty()),
        !frame.buses.is_empty(),
        !frame.voices.is_empty(),
        !frame.poses.is_empty(),
        !frame.emitters.is_empty(),
        !frame.lights.is_empty(),
        !frame.listener.is_empty(),
        !frame.panners.is_empty(),
        !frame.instances.is_empty(),
        !frame.instance_removals.is_empty(),
    ]
}
