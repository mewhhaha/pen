#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "hand-built Wasm's small sizes and LEB128 bytes"
)]
use super::*;

fn leb(mut value: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            bytes.push(byte);
            return bytes;
        }
        bytes.push(byte | 0x80);
    }
}

pub(crate) fn signed_leb(mut value: i32) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        let done = (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0);
        bytes.push(if done { byte } else { byte | 0x80 });
        if done {
            return bytes;
        }
    }
}

fn vector(items: Vec<Vec<u8>>) -> Vec<u8> {
    let mut bytes = leb(items.len() as u32);
    items.into_iter().for_each(|item| bytes.extend(item));
    bytes
}

fn name(text: &str) -> Vec<u8> {
    let mut bytes = leb(text.len() as u32);
    bytes.extend(text.as_bytes());
    bytes
}

fn section(id: u8, payload: Vec<u8>) -> Vec<u8> {
    let mut bytes = vec![id];
    bytes.extend(leb(payload.len() as u32));
    bytes.extend(payload);
    bytes
}

/// `i32.const at`, as a function body or an offset.
fn constant(at: i32) -> Vec<u8> {
    [vec![0x41], signed_leb(at), vec![0x0b]].concat()
}

/// A Blot guest as the compiler lays one out, with these entries
/// (`test_…` among them) and a `main(host)` that imports its adapter:
/// each entry is a name, its manifest tags and its Wasm type and body.
/// The arrays it returns lie at fixed places in its memory.
fn guest(entries: &[(&str, &[u8], usize, Vec<u8>)], arrays: &[(i32, &[f32])]) -> Vec<u8> {
    guest_with_main(entries, arrays, constant(0))
}

/// A long-lived program fixture, without a compiler or plugin dependency.
pub(crate) fn program(main: Vec<u8>, arrays: &[(i32, &[f32])]) -> Vec<u8> {
    guest_with_main(&[], arrays, main)
}

fn guest_with_main(entries: &[(&str, &[u8], usize, Vec<u8>)], arrays: &[(i32, &[f32])], main: Vec<u8>) -> Vec<u8> {
    let function_type = |parameters: &[u8], results: &[u8]| {
        [
            vec![0x60],
            vector(parameters.iter().map(|&p| vec![p]).collect()),
            vector(results.iter().map(|&r| vec![r]).collect()),
        ]
        .concat()
    };
    // 0: Unit, U32 or an array to one (and the arena's functions); 1: F32 to F32;
    // 2: the adapter; 3: main.
    let types = vector(vec![
        function_type(&[I32], &[I32]),
        function_type(&[F32], &[F32]),
        function_type(&[EXTERNREF, I32], &[I32]),
        function_type(&[EXTERNREF], &[I32]),
    ]);
    let imports =
        vector(vec![[name(blot_host::HOST_MODULE), name("call_array_f32_array_f32"), vec![0x00, 2]].concat()]);
    // Functions after the import: blot:allocate, blot:reset, main, then the entries.
    let mut bodies = vec![constant(1024), constant(0), main];
    let mut type_indices = vec![0, 0, 3];
    let mut exports = vec![
        [name("blot:memory"), vec![0x02, 0]].concat(),
        [name("blot:allocate"), vec![0x00, 1]].concat(),
        [name("blot:reset"), vec![0x00, 2]].concat(),
        [name("main"), vec![0x00, 3]].concat(),
    ];
    // main takes (Array F32 -> Array F32 ! {Foreign}) and returns Unit.
    let mut manifest = vec![[name("main"), vec![4, 6, 6, 0]].concat()];
    for (index, (entry, tags, type_index, body)) in entries.iter().enumerate() {
        bodies.push(body.clone());
        type_indices.push(*type_index as u32);
        exports.push([name(entry), vec![0x00], leb(4 + index as u32)].concat());
        manifest.push([name(entry), tags.to_vec()].concat());
    }
    let code = vector(bodies.into_iter().map(|body| [leb(body.len() as u32 + 1), vec![0], body].concat()).collect());
    let data = vector(
        arrays
            .iter()
            .map(|(at, lanes)| {
                let mut bytes = (lanes.len() as u32).to_le_bytes().to_vec();
                lanes.iter().for_each(|lane| bytes.extend(lane.to_le_bytes()));
                [vec![0x00], constant(*at), vector(bytes.into_iter().map(|byte| vec![byte]).collect())].concat()
            })
            .collect(),
    );
    let abi = [name("blot:abi"), leb(2), vector(manifest), leb(0)].concat();
    [
        b"\0asm\x01\0\0\0".to_vec(),
        section(1, types),
        section(2, imports),
        section(3, vector(type_indices.into_iter().map(leb).collect())),
        section(5, vector(vec![vec![0x00, 1]])),
        section(7, vector(exports)),
        section(10, code),
        section(11, data),
        section(0, abi),
    ]
    .concat()
}

const I32: u8 = 0x7f;
const F32: u8 = 0x7d;
const EXTERNREF: u8 = 0x6f;
/// Manifest tags: Unit to Array F32, and F32 to F32.
const UNIT_TO_ARRAY: &[u8] = &[0, 6];
const F32_TO_F32: &[u8] = &[3, 3];

fn sample() -> Vec<u8> {
    guest(
        &[
            ("test_passes", UNIT_TO_ARRAY, 0, constant(8)),
            ("test_fails", UNIT_TO_ARRAY, 0, constant(24)),
            ("test_traps", UNIT_TO_ARRAY, 0, vec![0x00, 0x0b]),
            ("test_takes_f32", F32_TO_F32, 1, vec![0x20, 0, 0x0b]),
            ("failing_on_purpose", UNIT_TO_ARRAY, 0, constant(24)),
            ("test_malformed", UNIT_TO_ARRAY, 0, constant(64)),
        ],
        &[(8, &[2.0, 0.0]), (24, &[2.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 4.0]), (64, &[1.0, 1.0, 1.0, 5.0])],
    )
}

/// Entries for a module that must trap: one stopping at `unreachable`, as
/// Blot's `@panic` does; two returning, a report and lanes that are
/// none; and three trapping otherwise.
fn trapping_sample() -> Vec<u8> {
    guest(
        &[
            ("test_panics", UNIT_TO_ARRAY, 0, vec![0x00, 0x0b]),
            ("test_returns_a_report", UNIT_TO_ARRAY, 0, constant(24)),
            ("test_returns_lanes", UNIT_TO_ARRAY, 0, constant(64)),
            // i32.load at the address space's last byte.
            (
                "test_reads_out_of_bounds",
                UNIT_TO_ARRAY,
                0,
                [vec![0x41], signed_leb(-1), vec![0x28, 2, 0, 0x0b]].concat(),
            ),
            // 1 / 0 (i32.div_s).
            ("test_divides_by_zero", UNIT_TO_ARRAY, 0, vec![0x41, 1, 0x41, 0, 0x6d, 0x0b]),
            // Calls itself, function 9: the import, blot:allocate,
            // blot:reset, main, then these entries.
            ("test_recurses", UNIT_TO_ARRAY, 0, vec![0x20, 0, 0x10, 9, 0x0b]),
        ],
        &[(24, &[2.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 4.0]), (64, &[1.0, 1.0, 1.0, 5.0])],
    )
}

#[test]
fn reads_reports() {
    assert_eq!(decode(&[0.0, 0.0]).unwrap(), Report { checks: 0, failures: vec![] });
    assert_eq!(decode(&[5.0, 0.0]).unwrap(), Report { checks: 5, failures: vec![] });
    // gdev's tests/hand_salt.blot's `failing_on_purpose`.
    assert_eq!(
        decode(&[2.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 4.0]).unwrap(),
        Report { checks: 2, failures: vec![Failure { place: 2, expected: vec![2.0, 3.0], actual: vec![2.0, 4.0] }] }
    );
    let two = decode(&[3.0, 2.0, 1.0, 0.0, 1.0, 0.5, 3.0, 1.0, 1.0, 1.0, 0.0]).unwrap();
    assert_eq!(two.failures[0], Failure { place: 1, expected: vec![], actual: vec![0.5] });
    assert_eq!(two.failures[1], Failure { place: 3, expected: vec![1.0], actual: vec![0.0] });
    for malformed in [
        &[][..],
        &[1.0],
        &[1.0, 2.0],
        &[1.0, 1.0],
        &[1.0, 1.0, 1.0, 2.0, 1.0],
        &[1.0, 1.0, 2.0, 0.0, 0.0],
        &[1.0, 0.0, 7.0],
        &[1.5, 0.0],
        &[-1.0, 0.0],
        &[f32::NAN, 0.0],
        &[f32::INFINITY, 0.0],
    ] {
        assert!(decode(malformed).is_err(), "{malformed:?}");
    }
}

#[test]
fn describes_outcomes() {
    let passed = Outcome::Reported(Report { checks: 2, failures: vec![] });
    assert_eq!(describe("test_a", &passed), ["  ok      test_a"]);
    let failed = Outcome::Reported(decode(&[2.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 4.0]).unwrap());
    assert_eq!(
        describe("test_b", &failed),
        ["  FAILED  test_b (1 of 2 checks)", "    check 2: expected [2, 3] got [2, 4]"]
    );
    let near = Failure { place: 1, expected: vec![0.39, -0.0], actual: vec![1e-7, f32::NAN] };
    let near = Outcome::Reported(Report { checks: 1, failures: vec![near] });
    assert_eq!(describe("test_c", &near)[1], "    check 1: expected [0.39, -0] got [0.0000001, NaN]");
    let broken = Outcome::Broken("wasm trap\nunreachable".into());
    assert_eq!(describe("test_d", &broken), ["  FAILED  test_d", "    wasm trap\n    unreachable"]);
}

#[test]
fn runs_tests_that_must_trap() {
    let engine = blot_host::engine().unwrap();
    let tests = run_tests(&engine, &trapping_sample(), Kind::Traps, "test_", None, &Slots::new(4)).unwrap();
    assert_eq!(tests.tests.len(), 6);
    let outcome = |name: &str| &tests.tests.iter().find(|(test, _)| test == name).unwrap().1;
    // Trapping at `unreachable`, as `@panic` does, passes.
    assert_eq!(*outcome("test_panics"), Outcome::Panicked);
    assert!(outcome("test_panics").passed());
    assert_eq!(describe("test_panics", outcome("test_panics")), ["  ok      test_panics"]);
    // Returning fails, showing the report it returned, or its lanes.
    assert!(!outcome("test_returns_a_report").passed());
    assert_eq!(
        describe("test_returns_a_report", outcome("test_returns_a_report")),
        [
            "  FAILED  test_returns_a_report (returned instead of trapping)",
            "    its report: 1 of 2 checks failed",
            "    check 2: expected [2, 3] got [2, 4]",
        ]
    );
    assert_eq!(*outcome("test_returns_lanes"), Outcome::Returned(vec![1.0, 1.0, 1.0, 5.0]));
    assert_eq!(
        describe("test_returns_lanes", outcome("test_returns_lanes")),
        ["  FAILED  test_returns_lanes (returned instead of trapping)", "    it returned [1, 1, 1, 5]"]
    );
    // Any other trap fails, showing it.
    for (name, trap) in [
        ("test_reads_out_of_bounds", "out of bounds memory access"),
        ("test_divides_by_zero", "integer divide by zero"),
        ("test_recurses", "call stack exhausted"),
    ] {
        let Outcome::Mistrapped(message) = outcome(name) else { panic!("{name}: {:?}", outcome(name)) };
        assert!(message.starts_with(&format!("wasm trap: {trap}")), "{message}");
        let lines = describe(name, outcome(name));
        assert_eq!(lines[0], format!("  FAILED  {name} (trapped, but not at `unreachable`)"));
        assert!(lines[1].starts_with(&format!("    wasm trap: {trap}")), "{}", lines[1]);
    }
    // Where a test must not trap, `unreachable` fails it.
    let checks = run_tests(&engine, &trapping_sample(), Kind::Checks, "test_", Some("panics"), &Slots::new(4)).unwrap();
    let Outcome::Broken(trap) = &checks.tests[0].1 else { panic!("{:?}", checks.tests) };
    assert!(trap.starts_with("wasm trap: wasm `unreachable`"), "{trap}");
}

#[test]
fn judges_modules_that_must_not_compile() {
    let rejects = "component type already registered";
    // Not compiling, with diagnostics that contain the text, passes.
    let diagnostics = "/gdev/packages/ecs.blot:0-0: const_panic: component type already registered";
    let rejected = rejection(rejects, Err(anyhow!("{diagnostics}")));
    assert_eq!(rejected.tests, [(rejects.to_owned(), Outcome::Rejected)]);
    assert!(rejected.tests[0].1.passed());
    assert_eq!(describe(rejects, &Outcome::Rejected), ["  ok      rejected: component type already registered"]);
    // Compiling fails.
    let compiled = rejection(rejects, Ok(sample()));
    assert_eq!(compiled.tests, [(rejects.to_owned(), Outcome::Compiled)]);
    assert!(!compiled.tests[0].1.passed());
    assert_eq!(
        describe(rejects, &Outcome::Compiled),
        ["  FAILED  compiled, but should be rejected: component type already registered"]
    );
    // Not compiling with other diagnostics fails, showing them.
    for other in [
        "/gdev/src/test_fixture.blot:9532-9551: type_mismatch: Types do not match",
        "/gdev/packages/ecs.blot:0-0: const_panic: component type already initialized",
    ] {
        let misrejected = rejection(rejects, Err(anyhow!("{other}")));
        assert_eq!(misrejected.tests, [(rejects.to_owned(), Outcome::Misrejected(other.to_owned()))]);
        assert!(!misrejected.tests[0].1.passed());
        assert_eq!(
            describe(rejects, &misrejected.tests[0].1),
            [
                "  FAILED  rejected, but the diagnostics lack: component type already registered".to_owned(),
                format!("    {other}")
            ]
        );
    }
}

#[test]
fn tallies_modules_by_what_the_plugin_expects() {
    let expected = [Expectation::Checks, Expectation::Traps, Expectation::Rejected("type_mismatch".into())];
    assert_eq!(expected.map(|expect| Kind::of(&expect)), Kind::ALL);
    let test = TestModule {
        path: "/gdev/tests/traps/codec.blot".into(),
        entry: "/gdev/src/test_fixture.blot".into(),
        overlays: Vec::new(),
        prefix: "test_".into(),
        expect: Expectation::Traps,
    };
    let module = Module::new(Path::new("/gdev"), test);
    let shown = Path::new("tests/traps/codec.blot").to_str().unwrap();
    assert_eq!((module.shown.as_str(), module.kind), (shown, Kind::Traps));
}

#[test]
fn slots_bound_how_many_tests_run_at_once() {
    let slots = Slots::new(2);
    let (running, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
    std::thread::scope(|scope| {
        for _ in 0..6 {
            scope.spawn(|| {
                slots.hold(|| {
                    most.fetch_max(running.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(20));
                    running.fetch_sub(1, Ordering::SeqCst);
                });
            });
        }
    });
    assert_eq!(most.load(Ordering::SeqCst), 2);
    // A slot comes back when its work panics.
    let one = Slots::new(1);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| one.hold(|| panic!("a test")))).is_err());
    assert_eq!(one.hold(|| 7), 7);
}

#[test]
fn hands_results_over_in_order() {
    let items: Vec<u64> = (0..12).collect();
    let (running, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let mut handed = Vec::new();
    in_order(
        &items,
        4,
        || (),
        |(), &item| {
            most.fetch_max(running.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
            // Later items finish sooner.
            std::thread::sleep(Duration::from_millis(3 * (12 - item)));
            running.fetch_sub(1, Ordering::SeqCst);
            item * 10
        },
        |&item, result| handed.push((item, result)),
    );
    assert_eq!(handed, items.iter().map(|&item| (item, item * 10)).collect::<Vec<_>>());
    let most = most.load(Ordering::SeqCst);
    assert!((2..=4).contains(&most), "{most} at once");
    // Nothing to do, and one at a time.
    in_order(&[] as &[u64], 4, || (), |(), _| unreachable!(), |_, ()| unreachable!());
    let mut one = Vec::new();
    in_order(&items, 1, || (), |(), &item| item, |_, result| one.push(result));
    assert_eq!(one, items);
}

#[test]
fn runs_a_modules_tests() {
    let engine = blot_host::engine().unwrap();
    let tests = run_tests(&engine, &sample(), Kind::Checks, "test_", None, &Slots::new(4)).unwrap();
    let names: Vec<&str> = tests.tests.iter().map(|(name, _)| name.as_str()).collect();
    // Only `test_…` entries, Unit to Array F32, in the module's order.
    assert_eq!(names, ["test_passes", "test_fails", "test_traps", "test_malformed"]);
    assert_eq!(tests.skipped, ["test_takes_f32: it is F32 -> F32, not Unit -> Array F32"]);
    let outcome = |name: &str| &tests.tests.iter().find(|(test, _)| test == name).unwrap().1;
    assert_eq!(*outcome("test_passes"), Outcome::Reported(Report { checks: 2, failures: vec![] }));
    assert!(outcome("test_passes").passed());
    assert_eq!(describe("test_fails", outcome("test_fails"))[1], "    check 2: expected [2, 3] got [2, 4]");
    let Outcome::Broken(trap) = outcome("test_traps") else { panic!("test_traps did not trap") };
    assert!(trap.starts_with("wasm trap: wasm `unreachable`"), "{trap}");
    let Outcome::Broken(malformed) = outcome("test_malformed") else { panic!("test_malformed passed") };
    assert!(malformed.starts_with("its report is malformed (it ends within failure 1's expected"), "{malformed}");
    assert!(malformed.ends_with("[1, 1, 1, 5]"), "{malformed}");

    let filtered = run_tests(&engine, &sample(), Kind::Checks, "test_", Some("fail"), &Slots::new(4)).unwrap();
    assert_eq!(filtered.tests.len(), 1);
    assert_eq!(filtered.tests[0].0, "test_fails");
    assert!(run_tests(&engine, b"\0asm\x01\0\0\0", Kind::Checks, "test_", None, &Slots::new(4)).is_err());
}

#[test]
fn runs_the_checkouts_pilot() {
    // The sibling gdev checkout's pilot module, when this repo has one
    // and its plugins are there to compile it.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../gdev");
    let pilot = root.join("tests/hand_salt.blot");
    let plugins = pilot.is_file().then(|| test_plugins(&root).ok()).flatten();
    let Some(plugins) = plugins else {
        eprintln!("skipped: no ../gdev/tests/hand_salt.blot, or no plugins to compile it");
        return;
    };
    // As the plugin lays it out.
    let pilot = pilot.canonicalize().unwrap();
    let mut found = plugins.tests(&plugins.root().join(ENTRY), std::slice::from_ref(&pilot), None).unwrap();
    assert_eq!(found.len(), 1);
    let test = found.remove(0);
    assert!(Path::new(&test.path) == pilot && matches!(test.expect, Expectation::Checks), "{}", test.path);
    let mut builder = Builder::new(plugins, Path::new(&test.entry));
    let wasm = build_test_module(&mut builder, &test).unwrap();
    let engine = blot_host::dev_engine().unwrap();
    let tests = run_tests(&engine, &wasm, Kind::Checks, &test.prefix, None, &Slots::new(4)).unwrap();
    assert!(!tests.tests.is_empty());
    for (name, outcome) in &tests.tests {
        assert!(outcome.passed(), "{}", describe(name, outcome).join("\n"));
    }
    // A test that fails on purpose (and is no `test_…`, so it does not run).
    let module = GuestModule::new(&engine, &wasm, None).unwrap();
    if module.abi().function("failing_on_purpose").is_some() {
        let lanes = std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(blot_host::GUEST_STACK_BYTES + (8 << 20))
                .spawn_scoped(scope, || {
                    Guest::new(&engine, &module).unwrap().call("failing_on_purpose", Value::Unit).unwrap()
                })
                .unwrap()
                .join()
                .unwrap()
        });
        let outcome = Outcome::Reported(decode(&lanes.into_f32_array().unwrap()).unwrap());
        assert!(!outcome.passed());
        let lines = describe("failing_on_purpose", &outcome);
        eprintln!("{}", lines.join("\n"));
        assert!(lines[1].starts_with("    check "), "{lines:?}");
    }
    // A module that does not compile, in the same session, fails with the
    // compiler's diagnostics, placed in the module.
    let source = "entry const test_broken = fn () => missing_name\n";
    let broken = TestModule {
        path: root.join("tests/broken.blot").display().to_string(),
        overlays: vec![pen_plugins::Overlay { path: test.entry.clone(), contents: Some(source.into()) }],
        ..test
    };
    let error = format!("{:#}", build_test_module(&mut builder, &broken).unwrap_err());
    assert!(error.contains("missing_name"), "{error}");
    assert!(error.starts_with(&format!("{}:1:", broken.path)), "{error}");
}
