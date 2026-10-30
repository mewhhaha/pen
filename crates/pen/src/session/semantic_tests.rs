//! Identical deterministic guest bytes run by native and browser sessions.
use super::*;
use crate::blot_test::tests::{program, signed_leb};
use pen_protocol::boundary::{EXTENSIONS_MARKER, PACKET_HEADER_FLOATS, PACKET_MAGIC, lanes};
use pen_runtime::semantic::{self, Observation, Request as SemanticRequest};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

const HANDSHAKE: i32 = 2048;
const CREATE: i32 = 2080;
const SNAPSHOT: i32 = 2160;
const RESTORE: i32 = 2240;

fn integer(value: i32) -> Vec<u8> {
    [vec![0x41], signed_leb(value)].concat()
}
fn load(at: i32) -> Vec<u8> {
    [integer(at), vec![0x28, 2, 0]].concat()
}
fn store(at: i32, value: Vec<u8>) -> Vec<u8> {
    [integer(at), value, vec![0x36, 2, 0]].concat()
}
fn request(offset: u8) -> Vec<u8> {
    [load(0), vec![0x2a, 2, offset, 0xa8]].concat()
}
fn exchange() -> Vec<u8> {
    [integer(0), vec![0x20, 0], load(20), vec![0x10, 0, 0x36, 2, 0]].concat()
}

fn identity(generation: u32) -> Value {
    json!({"kind":"Hauler","words":[7,u32::MAX,2,generation]})
}

fn observation(tick: u32) -> Observation {
    serde_json::from_value(json!({
        "version":1,"tick":tick.to_string(),
        "schema":{"id":"hauler-fixture-v1","types":[{"id":1,"label":"Hauler","descriptor":{"version":1,"variants":[
            {"tag":0,"label":"Hauling","fields":[
                {"id":17,"label":"destination","type_name":"U32","kind":"unsigned"},
                {"id":31,"label":"capacity","type_name":"F32","kind":"float"}
            ]}
        ]}}]},
        "objects":[{"id":identity(u32::from(tick >= 3)),"type_id":1,"tag":0,"fields":[
            {"id":17,"value":{"kind":"unsigned","value":if tick >= 2 {u32::MAX} else {0}}},
            {"id":31,"value":{"kind":"float","value":2.5}}
        ]}],
        "decisions":if tick == 1 || tick == 2 {json!([{"tick":tick.to_string(),"object":identity(0),"system":"routing",
            "source":{"file":"src/żuraw.blot","start":17,"end":u32::MAX},
            "reason":if tick == 1 {"destination full"} else {"destination selected"}}])} else {json!([])},
        "omitted_objects":0,"omitted_decisions":0
    }))
    .unwrap()
}

fn packet(tick: u32, observed: bool) -> Vec<f32> {
    let mut packet = vec![0.0; PACKET_HEADER_FLOATS];
    packet[0] = PACKET_MAGIC;
    packet[1] = 1.0;
    for start in [20, 36, 90, 106] {
        for diagonal in [0, 5, 10, 15] {
            packet[start + diagonal] = 1.0;
        }
    }
    for at in [55, 60, 67, 69] {
        packet[at] = 1.0;
    }
    packet[52] = tick as f32 / 3.0;
    packet[lanes::header::SERVICE_COUNT] = 1.0;
    packet[lanes::header::SERVICE_LANES] = 8.0;
    packet.extend([
        1.0,
        pen_protocol::service::PLATFORM as f32,
        pen_protocol::platform_op::OS as f32,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]);
    if observed {
        let extension = observation(tick).extension(42).unwrap();
        packet.extend([EXTENSIONS_MARKER, 1.0, 42.0, extension.values.len() as f32]);
        packet.extend(extension.values);
    }
    packet
}

fn guest(observed: bool) -> Vec<u8> {
    // Tick and service answers are simulation state. Observation payloads are
    // projections only; the stripped guest executes the same state machine.
    let mut main = store(20, integer(HANDSHAKE));
    main.extend(exchange());
    main.extend([0x03, 0x40]);
    main.extend([request(4), vec![0x04, 0x40], store(4, load(8)), store(12, load(16)), vec![0x0b]].concat());
    main.extend(store(8, load(4)));
    main.extend(store(16, load(12)));
    main.extend([request(8), vec![0x45, 0x04, 0x40], integer(0), vec![0x0f, 0x0b]].concat());
    for (op, reply, action) in [
        (1, CREATE, [store(8, integer(0)), store(16, integer(0))].concat()),
        (
            3,
            SNAPSHOT,
            [
                integer(SNAPSHOT + 36),
                load(4),
                vec![0xb2, 0x38, 2, 0],
                integer(SNAPSHOT + 40),
                load(12),
                vec![0xb2, 0x38, 2, 0],
            ]
            .concat(),
        ),
        (4, RESTORE, [store(8, request(40)), store(16, request(44))].concat()),
    ] {
        main.extend(
            [
                request(8),
                integer(op),
                vec![0x46, 0x04, 0x40],
                action,
                store(20, integer(reply)),
                exchange(),
                vec![0x0c, 1, 0x0b],
            ]
            .concat(),
        );
    }
    main.extend([request(8), integer(2), vec![0x47, 0x04, 0x40, 0x00, 0x0b]].concat());
    // A validation/restore frame has elapsed=0 and does not advance time.
    main.extend(
        [
            load(0),
            vec![0x2a, 2, 12, 0x43, 0, 0, 0, 0, 0x5e, 0x04, 0x40],
            store(8, [load(4), integer(1), vec![0x6a]].concat()),
            vec![0x0b],
        ]
        .concat(),
    );
    main.extend(store(24, integer(0)));
    main.extend(store(28, request(24)));
    main.extend([0x02, 0x40, 0x03, 0x40]);
    main.extend([load(24), load(28), vec![0x4f, 0x0d, 1]].concat());
    main.extend(
        [
            load(0),
            integer(32),
            vec![0x6a],
            load(24),
            integer(20),
            vec![0x6c, 0x6a, 0x2a, 2, 0, 0xa8],
            integer(13),
            vec![0x46, 0x04, 0x40],
            store(16, [load(16), integer(1), vec![0x6a]].concat()),
            vec![0x0b],
        ]
        .concat(),
    );
    main.extend(store(24, [load(24), integer(1), vec![0x6a]].concat()));
    main.extend([0x0c, 0, 0x0b, 0x0b]);
    main.extend(store(20, integer(32768)));
    for tick in 0..3 {
        main.extend(
            [load(8), integer(tick), vec![0x46, 0x04, 0x40], store(20, integer(8192 + tick * 8192)), vec![0x0b]]
                .concat(),
        );
    }
    main.extend(exchange());
    main.extend([0x0c, 0, 0x0b]);
    main.extend(integer(0));
    main.push(0x0b);
    // Saved state section 99: tick and received external-service answers.
    let state = [5678.0, 1.0, 1.0, 99.0, 1.0, 2.0, 2.0, 0.0, 0.0];
    let create = [vec![1.0], state.to_vec()].concat();
    let snapshot = [vec![3.0], state.to_vec()].concat();
    let packets: Vec<_> = (0..4).map(|tick| [vec![2.0], packet(tick, observed)].concat()).collect();
    for packet in &packets {
        assert!(packet.len() * 4 + 4 < 8192);
    }
    program(
        main,
        &[
            (HANDSHAKE, &[4670544.0, 2.0, 1.0, 0.0]),
            (CREATE, &create),
            (SNAPSHOT, &snapshot),
            (RESTORE, &[4.0, 0.0]),
            (8192, &packets[0]),
            (16384, &packets[1]),
            (24576, &packets[2]),
            (32768, &packets[3]),
        ],
    )
}

fn query(session: &mut Session, request: Value) -> Value {
    let answer = session.observations().unwrap().answer(serde_json::from_value(request).unwrap());
    serde_json::from_str(&pen_runtime::services::text(&answer.values)).unwrap()
}

struct Counted(Arc<AtomicUsize>);
impl crate::services::Service<crate::input::Input> for Counted {
    fn code(&self) -> u32 {
        pen_protocol::service::PLATFORM
    }
    fn call(&mut self, _: &crate::services::Request) -> Option<crate::services::Answer> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Some(crate::services::Answer::ok(123.0))
    }
}

fn start(engine: &wasmtime::Engine, gpu: Gpu, directory: &Path, observed: bool) -> Session {
    Session::start(
        Options {
            frames: 0,
            screenshot: None,
            record: None,
            fps: Some(60.0),
            mute: true,
            audio_buffer: None,
            dev: false,
            script: Vec::new(),
            headless: true,
            settle: false,
        },
        Paths::new(directory.to_owned(), Some(directory.join("assets")), None, None, Some(directory.join("data"))),
        engine.clone(),
        GuestModule::new(engine, &guest(observed), None).unwrap(),
        Catalog { meshes: vec![], materials: vec![], sounds: vec![] },
        gpu,
        None,
    )
    .unwrap()
}

#[test]
fn semantic_history_preserves_simulation_and_external_service_replay() {
    let directory = PathBuf::from("/var/tmp").join(format!("pen-semantic-{}", std::process::id()));
    std::fs::create_dir_all(directory.join("assets")).unwrap();
    std::fs::write(directory.join("assets/saved.json"), r#"{"format":"ink-persist-v1"}"#).unwrap();
    let engine = blot_host::engine().unwrap();
    let mut gpu = Gpu::headless((32, 24), &directory.join("assets"), false).unwrap();
    let mut witnesses = Vec::new();
    let mut expected = Vec::new();
    for observed in [false, true] {
        let mut session = start(&engine, gpu, &directory, observed);
        let calls = Arc::new(AtomicUsize::new(0));
        session.services.register(Box::new(Counted(calls.clone())));
        session.playback(Playback { history: true, ..Playback::default() }, None).unwrap();
        session.pause();
        if observed {
            session.observations().unwrap().answer(SemanticRequest::Capture { tag: Some(42) });
        }
        let mut pass = Vec::new();
        for _ in 0..4 {
            session.step_frames(1);
            assert!(session.step(None).unwrap());
            let saved = session.runtime.snapshot().unwrap();
            let mut shown = session.last.as_ref().unwrap().clone();
            shown.extensions.clear();
            pass.push((saved, shown));
        }
        assert_eq!(session.timeline.position(), 4);
        assert_eq!(calls.load(Ordering::Relaxed), 4);
        let state = &pass[3].0.state;
        assert_eq!(state[state.len() - 2..], [3.0, 3.0]);
        let calls_before = calls.load(Ordering::Relaxed);
        session.rewind_to(2);
        assert!(session.step(None).unwrap());
        assert_eq!(session.timeline.position(), 2);
        assert_eq!(session.runtime.snapshot().unwrap(), pass[1].0);
        if observed {
            session.observations().unwrap().answer(SemanticRequest::Watch {
                watch: semantic::Watch {
                    object: serde_json::from_value(identity(0)).unwrap(),
                    field: Some(17),
                    equals: Some(semantic::Value::Unsigned(u32::MAX)),
                },
            });
        }
        session.step_frames(2);
        assert!(session.step(None).unwrap());
        assert_eq!(session.timeline.position(), 3);
        if observed {
            assert!(session.transport.paused());
            assert!(session.step(None).unwrap());
            assert_eq!(session.timeline.position(), 3, "semantic watch cancels remaining steps");
            for request in [
                json!({"op":"values","position":"2","object":identity(0)}),
                json!({"op":"values","position":"3","object":identity(0)}),
                json!({"op":"diff","before":"2","after":"3"}),
                json!({"op":"diff","before":"3","after":"4"}),
                json!({"op":"decisions","from":"1","through":"4"}),
                json!({"op":"schema","position":"3"}),
            ] {
                let answer = query(&mut session, request.clone());
                assert_eq!(answer["status"], "ok", "{answer}");
                expected.push(json!({"request":request,"answer":answer}));
            }
        }
        assert_eq!(
            calls.load(Ordering::Relaxed),
            calls_before,
            "rewind and recorded steps never call external services"
        );
        assert_eq!(session.runtime.snapshot().unwrap(), pass[2].0);
        if observed {
            // A real save replacement publishes its final rendered state and
            // cannot fire a simulation-transition watch across the restore.
            session.rewind_to(4);
            assert!(session.step(None).unwrap());
            session.observations().unwrap().answer(SemanticRequest::Watch {
                watch: semantic::Watch {
                    object: serde_json::from_value(identity(0)).unwrap(),
                    field: None,
                    equals: None,
                },
            });
            let saved = directory.join("restore.bin");
            crate::snapshot::write(&saved, &pass[1].0).unwrap();
            session.script = Some(Script::new(vec![Step::Load(saved), Step::Wait(100)]));
            session.step_frames(1);
            assert!(session.step(None).unwrap());
            assert_eq!(session.runtime.snapshot().unwrap(), pass[1].0);
            let objects = query(&mut session, json!({"op":"objects","position":"5"}));
            assert_eq!(objects["restored"], true);
            assert_eq!(objects["rows"][0]["id"], identity(0));
            assert!(session.observations().unwrap().watching());
            session.rewind_to(4);
            assert!(session.step(None).unwrap());
            session.rewind_to(5);
            assert!(session.step(None).unwrap());
            assert_eq!(query(&mut session, json!({"op":"objects","position":"5"}))["restored"], true);
            assert_eq!(session.runtime.snapshot().unwrap(), pass[1].0);
            session.reload(
                None,
                GuestModule::new(&engine, &guest(true), None).unwrap(),
                Catalog { meshes: vec![], materials: vec![], sounds: vec![] },
            );
            let status = query(&mut session, json!({"op":"status"}));
            assert_eq!(status["invalidated"], "build_changed");
            assert_eq!(status["retained"], 0);
            assert!(status["watch"].is_null());
            assert_eq!(session.runtime.snapshot().unwrap(), pass[1].0);
        }
        witnesses.push(pass);
        gpu = session.finish().unwrap();
    }
    assert_eq!(witnesses[0], witnesses[1], "observation collection cannot change simulation or rendered output");
    drop(gpu);
    if let Some(path) = std::env::var_os("PEN_SEMANTIC_FIXTURES") {
        let path = PathBuf::from(path);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("observed.wasm"), guest(true)).unwrap();
        std::fs::write(path.join("plain.wasm"), guest(false)).unwrap();
        std::fs::write(path.join("native.json"), serde_json::to_vec_pretty(&expected).unwrap()).unwrap();
        crate::snapshot::write(&path.join("restore.bin"), &witnesses[0][1].0).unwrap();
        std::fs::write(
            path.join("operations.json"),
            serde_json::to_vec(&json!({
                "pause":pen_protocol::devtools_op::PAUSE,"step":pen_protocol::devtools_op::STEP,
                "history":pen_protocol::devtools_op::HISTORY,"state":pen_protocol::devtools_op::STATE,
                "tuned":pen_protocol::devtools_op::TUNED,"watch":pen_protocol::devtools_op::WATCH,
                "export":pen_protocol::devtools_op::EXPORT,
                "semantic":pen_protocol::devtools_op::SEMANTIC
            }))
            .unwrap(),
        )
        .unwrap();
    }
    std::fs::remove_dir_all(directory).unwrap();
}
