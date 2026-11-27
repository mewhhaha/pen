#![cfg(feature = "compiler")]
use anyhow::{Result, bail, ensure};
use blot_host::{
    Argument, Guest, GuestModule, Limits, Value,
    processes::{
        AdmitResult, ProcessEvent, ProcessHandle, ProcessOutcome, ProcessPolicy, ProcessStatus, ProcessValue,
        Processes, SendResult,
    },
};
use std::{
    collections::VecDeque,
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};
use wasmtime::Engine;

const MEMORY: Limits = Limits { memory: 64 * 1024 * 1024 };

fn fixture() -> Option<&'static (Engine, GuestModule)> {
    static FIXTURE: OnceLock<Option<(Engine, GuestModule)>> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let blot = root.join("../../../blot2");
            let compiler = std::env::var_os("BLOTC")
                .map(PathBuf::from)
                .unwrap_or_else(|| blot.join("compiler-goose/build/blotc-goose"));
            let source = blot.join("benchmarks/language/process_replay.blot");
            let prelude = blot.join("std/prelude.blot");
            if !compiler.is_file() || !source.is_file() || !prelude.is_file() {
                let required = std::env::var_os("PEN_REQUIRE_FIXTURES").is_some_and(|v| !v.is_empty() && v != "0");
                assert!(
                    !required,
                    "processes tests require blotc-goose and sibling blot2 checkout, and PEN_REQUIRE_FIXTURES is set"
                );
                eprintln!("skipping: processes tests require blotc-goose and sibling blot2 checkout");
                return None;
            }
            let output = root.join("../../target/returned-processes.wasm");
            let built = Command::new(compiler)
                .arg("build")
                .arg(&source)
                .arg(&output)
                .arg("--prelude")
                .arg(&prelude)
                .output()
                .unwrap();
            assert!(
                built.status.success(),
                "{}{}",
                String::from_utf8_lossy(&built.stdout),
                String::from_utf8_lossy(&built.stderr)
            );
            let engine = blot_host::engine().unwrap();
            let module = GuestModule::new(&engine, &std::fs::read(output).unwrap(), None).unwrap();
            Some((engine, module))
        })
        .as_ref()
}

fn policy(name: &str, step_work: u32, slots: usize, mailbox: usize, memory: Limits) -> ProcessPolicy {
    ProcessPolicy {
        name: name.into(),
        init_work: 1,
        step_work,
        rust_wasmtime_fuel: None,
        deadline: None,
        memory,
        mailbox,
        mailbox_bytes: 1024,
        max_reply_bytes: 16,
        slots,
    }
}

fn word(value: Value) -> Result<u32> {
    if let Value::U32(word) = value {
        return Ok(word);
    }
    bail!("expected U32")
}

fn words(value: Value) -> Result<Vec<u32>> {
    if let Value::ArrayU32(words) = value {
        return Ok(words);
    }
    bail!("expected Array U32")
}

fn accept(send: SendResult) -> u64 {
    let SendResult::Accepted(ticket) = send else { panic!("expected an accepted ticket: {send:?}") };
    ticket
}

fn admitted(res: AdmitResult) -> ProcessHandle {
    let AdmitResult::Admitted(h) = res else { panic!("expected admitted handle: {res:?}") };
    h
}

fn apply(guest: &mut Guest, world: u32, values: &[u32]) -> Result<u32> {
    let packet = std::iter::once(world).chain(values.iter().copied()).collect();
    word(guest.call("apply_tick", Value::ArrayU32(packet))?)
}

#[test]
fn native_turns_replay_same_spawning_caller_and_world_at_every_tick() -> Result<()> {
    let Some((engine, module)) = fixture() else {
        return Ok(());
    };
    let host = Arc::new(Mutex::new(Processes::new(engine, module, vec![policy("planner", 1, 2, 1, MEMORY)])?));
    let handles: Arc<Mutex<Vec<ProcessHandle>>> = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::new(Mutex::new(VecDeque::new()));
    let coordinator = host.clone();
    let handles_cap = handles.clone();
    let admissions = log.clone();
    let mut caller = Guest::new(engine, module)?;
    assert_eq!(word(caller.call("local", Value::U32(4))?)?, 4);
    assert_eq!(word(caller.call("local", Value::U32(5))?)?, 0);
    let live = Argument::Callback(Box::new(move |packet| {
        let packet = words(packet)?;
        let name: String = packet[1..].iter().map(|&b| char::from(u8::try_from(b).unwrap())).collect();
        let answer = match coordinator.lock().unwrap().admit(&name, ProcessValue::U32(packet[0])) {
            AdmitResult::Admitted(h) => {
                handles_cap.lock().unwrap().push(h);
                h.endpoint
            }
            _ => 0,
        };
        admissions.lock().unwrap().push_back((packet, answer));
        Ok(Value::U32(answer))
    }));
    let endpoints = words(caller.call("live", live)?)?;
    let [a_ep, b_ep, refused] = endpoints.as_slice() else { bail!("expected two endpoints and a refusal") };
    assert_ne!(*a_ep, *b_ep);
    assert_eq!(*refused, 0);
    let captured = handles.lock().unwrap().clone();
    let (a, b) = (captured[0], captured[1]);
    assert_eq!((a.endpoint, b.endpoint), (*a_ep, *b_ep));

    let mut world = 1000;
    let mut frames = Vec::new();
    let mut host = host.lock().unwrap();
    assert_eq!(host.status(a), ProcessStatus::Starting);
    assert_eq!(host.send(a, ProcessValue::U32(99)), SendResult::Starting);
    assert_eq!(host.memory_bytes(), 0);
    world = apply(&mut caller, world, &endpoints)?;
    frames.push((endpoints.clone(), world));
    assert_eq!(host.advance(2)?, 2);
    assert_eq!(host.status(a), ProcessStatus::Ready);
    assert_eq!(host.status(b), ProcessStatus::Ready);
    let memory = host.memory_bytes();
    assert!(memory > 0 && memory <= 2 * MEMORY.memory);
    let one = accept(host.send(a, ProcessValue::U32(1)));
    let two = accept(host.send(b, ProcessValue::U32(7)));
    assert_eq!(host.send(a, ProcessValue::U32(9)), SendResult::Full);
    assert_eq!(host.advance(2)?, 2);
    assert_eq!(host.send(a, ProcessValue::U32(9)), SendResult::Full);
    assert_eq!(host.poll(a), Some(ProcessEvent { ticket: one, outcome: ProcessOutcome::Done(ProcessValue::U32(11)) }));
    assert_eq!(host.poll(b), Some(ProcessEvent { ticket: two, outcome: ProcessOutcome::Done(ProcessValue::U32(107)) }));
    let values = vec![0, 107, 11];
    world = apply(&mut caller, world, &values)?;
    frames.push((values, world));
    assert_eq!(host.advance(2)?, 0);
    let wake = accept(host.send(a, ProcessValue::U32(2)));
    assert_eq!(host.advance(1)?, 1);
    assert_eq!(host.poll(a), Some(ProcessEvent { ticket: wake, outcome: ProcessOutcome::Done(ProcessValue::U32(13)) }));
    world = apply(&mut caller, world, &[13])?;
    frames.push((vec![13], world));

    let mut durations = Vec::new();
    for i in 1..=120u32 {
        let now = Instant::now();
        let first = accept(host.send(a, ProcessValue::U32(1)));
        let second = accept(host.send(b, ProcessValue::U32(1)));
        assert_eq!(host.advance(2)?, 2);
        assert_eq!(
            host.poll(a),
            Some(ProcessEvent { ticket: first, outcome: ProcessOutcome::Done(ProcessValue::U32(13 + i)) })
        );
        assert_eq!(
            host.poll(b),
            Some(ProcessEvent { ticket: second, outcome: ProcessOutcome::Done(ProcessValue::U32(107 + i)) })
        );
        durations.push(now.elapsed().as_nanos());
        assert_eq!(host.memory_bytes(), memory);
    }
    durations.sort_unstable();
    println!(
        "120 one-thread frames; two ABI 8 turns per frame; committed bytes={memory}; p50/p95/p99 ns={}/{}/{}",
        durations[59], durations[113], durations[118]
    );
    assert!(host.restore().is_empty());
    assert_eq!(host.status(a), ProcessStatus::Stale);
    assert_eq!(host.memory_bytes(), 0);
    drop(host);
    let replay_log = log.clone();
    let replay = Argument::Callback(Box::new(move |packet| {
        let (expected, outcome) = replay_log.lock().unwrap().pop_front().unwrap();
        ensure!(words(packet)? == expected, "replay admission order changed");
        Ok(Value::U32(outcome))
    }));
    assert_eq!(words(caller.call("live", replay)?)?, endpoints);
    assert!(log.lock().unwrap().is_empty());
    let mut replay_world = 1000;
    for (values, expected) in frames {
        replay_world = apply(&mut caller, replay_world, &values)?;
        assert_eq!(replay_world, expected);
    }
    Ok(())
}

#[test]
fn native_lifecycle_preserves_the_first_terminal_outcome_with_one_slot() -> Result<()> {
    let Some((engine, module)) = fixture() else {
        return Ok(());
    };
    let mut host = Processes::new(
        engine,
        module,
        vec![
            policy("planner", 1, 2, 1, MEMORY),
            policy("bad_init", 1, 1, 1, MEMORY),
            policy("bad_step", 1, 1, 1, MEMORY),
        ],
    )?;
    let a = admitted(host.admit("planner", ProcessValue::U32(10)));
    let bad = admitted(host.admit("bad_init", ProcessValue::U32(10)));
    let fault = admitted(host.admit("bad_step", ProcessValue::U32(10)));
    assert_eq!(host.advance(3)?, 3);
    assert_eq!(host.status(bad), ProcessStatus::InitializationFailed);
    let cancelled = accept(host.send(a, ProcessValue::U32(1)));
    assert!(host.cancel(a, cancelled));
    assert!(!host.cancel(a, cancelled));
    assert_eq!(host.send(a, ProcessValue::U32(2)), SendResult::Full);
    assert_eq!(host.poll(a), Some(ProcessEvent { ticket: cancelled, outcome: ProcessOutcome::Cancelled }));
    let done = accept(host.send(a, ProcessValue::U32(2)));
    let failed = accept(host.send(fault, ProcessValue::U32(4)));
    assert_eq!(host.advance(2)?, 2);
    assert!(!host.cancel(a, done));
    assert_eq!(host.status(fault), ProcessStatus::StepFailed);
    assert_eq!(host.send(fault, ProcessValue::U32(5)), SendResult::Closed);
    assert!(host.release(bad));
    assert!(!host.release(fault));
    let old = host.restore();
    assert_eq!(
        old[&a].front(),
        Some(&ProcessEvent { ticket: done, outcome: ProcessOutcome::Done(ProcessValue::U32(12)) })
    );
    assert_eq!(old[&fault].front(), Some(&ProcessEvent { ticket: failed, outcome: ProcessOutcome::Failed }));
    assert!(host.restore().is_empty());
    assert_eq!(host.send(a, ProcessValue::U32(10)), SendResult::Stale);
    let fresh = admitted(host.admit("planner", ProcessValue::U32(20)));
    assert_ne!(fresh, a);

    let mut tight = Processes::new(engine, module, vec![policy("planner", 1, 1, 1, Limits { memory: 1 })])?;
    let impossible = admitted(tight.admit("planner", ProcessValue::U32(0)));
    assert_eq!(tight.advance(1)?, 1);
    assert_eq!(tight.status(impossible), ProcessStatus::InitializationFailed);
    assert_eq!(tight.memory_bytes(), 0);
    assert_eq!(tight.reserved_memory_bytes(), 0);
    Ok(())
}

#[test]
fn native_deferred_turn_is_first_on_its_next_frame_serial_and_parallel() -> Result<()> {
    let Some((engine, module)) = fixture() else {
        return Ok(());
    };
    for parallel in [false, true] {
        let mut host = Processes::new(
            engine,
            module,
            vec![policy("planner", 1, 2, 1, MEMORY), policy("costly", 2, 1, 1, MEMORY)],
        )?;
        let a = admitted(host.admit("planner", ProcessValue::U32(10)));
        let b = admitted(host.admit("costly", ProcessValue::U32(10)));
        let c = admitted(host.admit("planner", ProcessValue::U32(10)));
        let step = |h: &mut Processes, budget: u32| -> Result<u32> {
            Ok(if parallel { h.advance_all_parallel(budget)? } else { h.advance(budget)? })
        };
        assert_eq!(step(&mut host, 3)?, 3);
        for id in [a, b, c] {
            accept(host.send(id, ProcessValue::U32(1)));
        }
        assert_eq!(step(&mut host, 2)?, 2);
        assert!(host.poll(a).is_some());
        assert!(host.poll(c).is_some());
        assert!(host.poll(b).is_none());
        accept(host.send(a, ProcessValue::U32(1)));
        accept(host.send(c, ProcessValue::U32(1)));
        assert_eq!(step(&mut host, 2)?, 2);
        assert!(host.poll(a).is_none());
        assert!(host.poll(c).is_none());
        assert!(host.poll(b).is_some());
        assert_eq!(step(&mut host, 2)?, 2);
        assert!(host.poll(a).is_some());
        assert!(host.poll(c).is_some());
    }
    Ok(())
}
