//! Run against a compiled Ink/Pen game with main and main_trace entries:
//! PEN_TRACE_FIXTURE=game.wasm cargo test -p blot-host --test system_trace -- --ignored
#![cfg(feature = "compiler")]
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use blot_host::{Argument, Guest, GuestModule, SystemTrace, Value};
use pen_protocol::boundary::encode_frame;
use pen_protocol::{Frame, GameEvent, Viewport};
use pen_runtime::program::{Operation, request};

fn run(engine: &wasmtime::Engine, module: &GuestModule) -> Vec<Vec<f32>> {
    let output = Arc::new(Mutex::new(Vec::new()));
    let found = output.clone();
    let mut requests = VecDeque::from([request(true, Operation::Create, &[])]);
    for index in 0..12 {
        let events = if index == 1 { vec![GameEvent::KeyDown("Enter".into())] } else { vec![] };
        let frame =
            Frame { timestamp: f64::from(index) / 60.0, viewport: Viewport { width: 640, height: 480 }, events };
        requests.push_back(request(true, Operation::Frame, &encode_frame(&frame, 1.0 / 60.0, &[]).unwrap()));
    }
    requests.push_back(request(true, Operation::Snapshot, &[]));
    let mut guest = Guest::new(engine, module).unwrap();
    guest
        .call(
            module.main_entry(),
            Argument::Callback(Box::new(move |packet| {
                found.lock().unwrap().push(packet.into_f32_array()?);
                Ok(Value::ArrayF32(requests.pop_front().unwrap_or_else(|| request(true, Operation::Stop, &[]))))
            })),
        )
        .unwrap();
    guest.dispose().unwrap();
    Arc::try_unwrap(output).unwrap().into_inner().unwrap()
}

#[test]
#[ignore = "requires PEN_TRACE_FIXTURE, a compiled real game"]
fn tracing_preserves_every_packet_and_snapshot() {
    let wasm = std::fs::read(std::env::var_os("PEN_TRACE_FIXTURE").expect("PEN_TRACE_FIXTURE")).unwrap();
    let engine = blot_host::engine().unwrap();
    let plain = GuestModule::new(&engine, &wasm, None).unwrap();
    let expected = run(&engine, &plain);
    let path = std::env::temp_dir().join(format!("pen-game-trace-{}.json", std::process::id()));
    let trace = SystemTrace::create(&path, 100_000).unwrap();
    let traced = GuestModule::traced(&engine, &wasm, None, trace.clone()).unwrap();
    trace.systems(true, 0);
    let actual = run(&engine, &traced);
    assert_eq!(actual, expected, "profiling must preserve the wire packets and final saved world");
    let (total, mut values) = trace.systems(true, 0);
    let mut observed = 0;
    let mut measured = 0;
    let mut allocated = 0.0;
    loop {
        let mut at = 0;
        while at < values.len() {
            assert!(at + 9 <= values.len());
            let length = usize::try_from(pen_protocol::boundary::whole(values[at + 8]).unwrap()).unwrap();
            assert!(at + 9 + length <= values.len());
            if values[at + 2] > 0.0 {
                measured += 1;
                assert!(values[at + 3] >= 0.0 && values[at + 6] >= 0.0 && values[at + 7] >= 0.0);
                allocated += values[at + 6];
            }
            observed += 1;
            at += 9 + length;
        }
        if observed == total {
            break;
        }
        values = trace.systems(true, observed).1;
        assert!(!values.is_empty(), "live timing pagination stopped before all systems");
    }
    assert!(measured > 10 && allocated > 0.0, "actual game must populate live timing and allocation rows");
    drop(traced);
    drop(trace);
    let capture: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let events = capture["traceEvents"].as_array().unwrap();
    assert!(events.iter().any(|event| event["cat"] == "Render"));
    assert!(events.iter().any(|event| event["cat"] == "Update"));
    assert!(events.iter().any(|event| event["args"]["allocated_bytes"].as_u64().unwrap() > 0));
    assert!(events.iter().all(|event| event["args"]["aborted"] == false));
    eprintln!("{} matching packets, {} spans; {}", actual.len(), events.len(), path.display());
}
