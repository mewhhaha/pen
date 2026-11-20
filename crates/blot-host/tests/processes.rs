use anyhow::{Result, bail, ensure};
use blot_host::{
    Argument, Guest, GuestModule, Limits, Value,
    processes::{Event, Outcome, Processes, Program, Send, Status},
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

fn fixture() -> &'static (Engine, GuestModule) {
    static FIXTURE: OnceLock<(Engine, GuestModule)> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let blot = root.join("../../../blot2");
        let compiler = std::env::var_os("BLOTC")
            .map(PathBuf::from)
            .unwrap_or_else(|| blot.join("compiler-goose/build/blotc-goose"));
        let output = root.join("../../target/returned-processes.wasm");
        let built = Command::new(compiler)
            .arg("build")
            .arg(blot.join("benchmarks/language/process_replay.blot"))
            .arg(&output)
            .arg("--prelude")
            .arg(blot.join("std/prelude.blot"))
            .output()
            .unwrap();
        assert!(built.status.success(), "{}", String::from_utf8_lossy(&built.stdout));
        let engine = blot_host::engine().unwrap();
        let module = GuestModule::new(&engine, &std::fs::read(output).unwrap(), None).unwrap();
        (engine, module)
    })
}

fn program(entry: &str, step_work: u32) -> Program {
    Program { entry: entry.into(), init_work: 1, step_work }
}

fn start_packet(entry: &str, config: u32) -> Vec<u32> {
    std::iter::once(config).chain(entry.bytes().map(u32::from)).collect()
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

fn accept(send: Send) -> u64 {
    let Send::Accepted(ticket) = send else { panic!("expected an accepted ticket: {send:?}") };
    ticket
}

fn apply(guest: &mut Guest, world: u32, values: &[u32]) -> Result<u32> {
    let packet = std::iter::once(world).chain(values.iter().copied()).collect();
    word(guest.call("apply_tick", Value::ArrayU32(packet))?)
}

#[test]
fn native_turns_replay_same_spawning_caller_and_world_at_every_tick() -> Result<()> {
    let (engine, module) = fixture();
    let host = Arc::new(Mutex::new(Processes::new(engine, module, vec![program("planner", 1)], 2, 1, MEMORY)?));
    let log = Arc::new(Mutex::new(VecDeque::new()));
    let coordinator = host.clone();
    let admissions = log.clone();
    let mut caller = Guest::new(engine, module)?;
    assert_eq!(word(caller.call("local", Value::U32(4))?)?, 4);
    assert_eq!(word(caller.call("local", Value::U32(5))?)?, 0);
    let live = Argument::Callback(Box::new(move |packet| {
        let packet = words(packet)?;
        let answer = coordinator.lock().unwrap().admit(&packet);
        admissions.lock().unwrap().push_back((packet, answer));
        Ok(Value::U32(answer))
    }));
    let endpoints = words(caller.call("live", live)?)?;
    let [a, b, refused] = endpoints.as_slice() else { bail!("expected two endpoints and a refusal") };
    let (a, b) = (*a, *b);
    assert_ne!(a, b);
    assert_eq!(*refused, 0);

    let mut world = 1000;
    let mut frames = Vec::new();
    let mut host = host.lock().unwrap();
    assert_eq!(host.status(a), Status::Starting);
    assert_eq!(host.send(a, 99), Send::Starting);
    assert_eq!(host.memory_bytes(), 0);
    world = apply(&mut caller, world, &endpoints)?;
    frames.push((endpoints.clone(), world));
    assert_eq!(host.advance(2), 2);
    assert_eq!(host.status(a), Status::Ready);
    assert_eq!(host.status(b), Status::Ready);
    let memory = host.memory_bytes();
    assert!(memory > 0 && memory <= 2 * MEMORY.memory);
    let one = accept(host.send(a, 1));
    let two = accept(host.send(b, 7));
    assert_eq!(host.send(a, 9), Send::Full);
    assert_eq!(host.advance(2), 2);
    assert_eq!(host.send(a, 9), Send::Full);
    assert_eq!(host.poll(a), Some(Event { ticket: one, outcome: Outcome::Done(11) }));
    assert_eq!(host.poll(b), Some(Event { ticket: two, outcome: Outcome::Done(107) }));
    let values = vec![0, 107, 11];
    world = apply(&mut caller, world, &values)?;
    frames.push((values, world));
    assert_eq!(host.advance(2), 0);
    let wake = accept(host.send(a, 2));
    assert_eq!(host.advance(1), 1);
    assert_eq!(host.poll(a), Some(Event { ticket: wake, outcome: Outcome::Done(13) }));
    world = apply(&mut caller, world, &[13])?;
    frames.push((vec![13], world));

    // Repeated native turns are timed together with the scheduler and mailboxes.
    // This fixture measures no renderer or production game frame.
    let mut durations = Vec::new();
    for i in 1..=120 {
        let now = Instant::now();
        let first = accept(host.send(a, 1));
        let second = accept(host.send(b, 1));
        assert_eq!(host.advance(2), 2);
        assert_eq!(host.poll(a), Some(Event { ticket: first, outcome: Outcome::Done(13 + i) }));
        assert_eq!(host.poll(b), Some(Event { ticket: second, outcome: Outcome::Done(107 + i) }));
        durations.push(now.elapsed().as_nanos());
        assert_eq!(host.memory_bytes(), memory);
    }
    durations.sort_unstable();
    println!(
        "120 one-thread frames; two ABI 8 turns per frame; committed bytes={memory}; p50/p95/p99 ns={}/{}/{}",
        durations[59], durations[113], durations[118]
    );
    assert!(host.restore().is_empty());
    assert_eq!(host.status(a), Status::Closed);
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
    let (engine, module) = fixture();
    let mut host = Processes::new(
        engine,
        module,
        vec![program("planner", 1), program("bad_init", 1), program("bad_step", 1)],
        3,
        1,
        MEMORY,
    )?;
    let a = host.admit(&start_packet("planner", 10));
    let bad = host.admit(&start_packet("bad_init", 10));
    let fault = host.admit(&start_packet("bad_step", 10));
    assert_eq!(host.advance(3), 3);
    assert_eq!(host.status(bad), Status::InitializationFailed);
    let cancelled = accept(host.send(a, 1));
    assert!(host.cancel(a, cancelled));
    assert!(!host.cancel(a, cancelled));
    assert_eq!(host.send(a, 2), Send::Full);
    assert_eq!(host.poll(a), Some(Event { ticket: cancelled, outcome: Outcome::Cancelled }));
    let done = accept(host.send(a, 2));
    let failed = accept(host.send(fault, 4));
    assert_eq!(host.advance(2), 2);
    assert!(!host.cancel(a, done));
    assert_eq!(host.status(fault), Status::StepFailed);
    assert_eq!(host.send(fault, 5), Send::Closed);
    assert!(host.release(bad));
    assert!(!host.release(fault));
    let old = host.restore();
    assert_eq!(old[&a].front(), Some(&Event { ticket: done, outcome: Outcome::Done(12) }));
    assert_eq!(old[&fault].front(), Some(&Event { ticket: failed, outcome: Outcome::Failed }));
    assert!(host.restore().is_empty());
    assert_eq!(host.send(a, 10), Send::Closed);
    assert_ne!(host.admit(&start_packet("planner", 20)), a);

    let mut tight = Processes::new(engine, module, vec![program("planner", 1)], 1, 1, Limits { memory: 1 })?;
    let impossible = tight.admit(&start_packet("planner", 0));
    assert_ne!(impossible, 0);
    assert_eq!(tight.advance(1), 1);
    assert_eq!(tight.status(impossible), Status::InitializationFailed);
    assert_eq!(tight.memory_bytes(), 0);
    Ok(())
}

#[test]
fn native_deferred_turn_is_first_on_its_next_frame() -> Result<()> {
    let (engine, module) = fixture();
    let mut host = Processes::new(engine, module, vec![program("planner", 1), program("costly", 2)], 3, 1, MEMORY)?;
    let a = host.admit(&start_packet("planner", 10));
    let b = host.admit(&start_packet("costly", 10));
    let c = host.admit(&start_packet("planner", 10));
    assert_eq!(host.advance(3), 3);
    for id in [a, b, c] {
        accept(host.send(id, 1));
    }
    assert_eq!(host.advance(2), 2);
    assert!(host.poll(a).is_some());
    assert!(host.poll(c).is_some());
    assert!(host.poll(b).is_none());
    accept(host.send(a, 1));
    accept(host.send(c, 1));
    assert_eq!(host.advance(2), 2);
    assert!(host.poll(a).is_none());
    assert!(host.poll(c).is_none());
    assert!(host.poll(b).is_some());
    assert_eq!(host.advance(2), 2);
    assert!(host.poll(a).is_some());
    assert!(host.poll(c).is_some());
    Ok(())
}
