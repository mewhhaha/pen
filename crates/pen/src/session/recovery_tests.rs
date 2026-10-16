use super::transport::Run;
use super::*;
use crate::blot_test::tests::{program, signed_leb};
use pen_protocol::boundary::{PACKET_HEADER_FLOATS, PACKET_MAGIC, decode_frame};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const HANDSHAKE: i32 = 2048;
const CREATE: i32 = 2080;
const SNAPSHOT: i32 = 2112;
const RESTORE: i32 = 2144;
const GOOD: i32 = 4096;
const BAD: i32 = 8192;
/// Paused by a rejected frame, to play on after a valid reload.
const REJECTED: Run = Run::Paused { steps: 0, resume_after_reload: true };
const DECODER_ERROR: &str = "game returned a truncated, oversized or nonfinite frame";

struct TickingOverlay(Arc<AtomicUsize>);

impl Overlay for TickingOverlay {
    fn window_event(&mut self, _: &Window, _: &WindowEvent) -> bool {
        true
    }

    fn update(&mut self, _: &mut Session, _: Option<&Window>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    fn paint(&mut self, _: &wgpu::Device, _: &wgpu::Queue, _: &wgpu::Texture) {}

    fn wants_pointer(&self) -> bool {
        false
    }
}

fn integer(value: i32) -> Vec<u8> {
    [vec![0x41], signed_leb(value)].concat()
}

fn load(address: i32) -> Vec<u8> {
    [integer(address), vec![0x28, 2, 0]].concat()
}

fn store(address: i32, value: Vec<u8>) -> Vec<u8> {
    [integer(address), value, vec![0x36, 2, 0]].concat()
}

fn request(offset: u8) -> Vec<u8> {
    [load(0), vec![0x2a, 2, offset, 0xa8]].concat()
}

fn exchange() -> Vec<u8> {
    // main's host callback, with its returned request pointer kept at 0.
    [integer(0), vec![0x20, 0], load(20), vec![0x10, 0, 0x36, 2, 0]].concat()
}

fn packet(red: f32) -> Vec<f32> {
    let mut packet = vec![0.0; PACKET_HEADER_FLOATS];
    packet[0] = PACKET_MAGIC;
    packet[1] = 1.0;
    for start in [20, 36, 90, 106] {
        for diagonal in [0, 5, 10, 15] {
            packet[start + diagonal] = 1.0;
        }
    }
    packet[52] = red;
    packet[55] = 1.0;
    packet[60] = 1.0;
    packet[67] = 1.0;
    packet[69] = 1.0;
    packet
}

/// State and candidate are separate, just like a compiled Blot program.
/// Every host request checks the actual commit lane. A rejected frame
/// being accepted traps before Snapshot can return the preceding state.
fn guest(engine: &wasmtime::Engine, bad: &[f32], reject_after: i32, red: f32) -> GuestModule {
    GuestModule::new(engine, &guest_bytes(bad, reject_after, red, false), None).unwrap()
}

fn guest_bytes(bad: &[f32], reject_after: i32, red: f32, trap: bool) -> Vec<u8> {
    // Memory: request, frame count, rejected candidate, state, candidate, reply.
    let mut main = store(20, integer(HANDSHAKE));
    main.extend(exchange());
    main.extend([0x03, 0x40]); // loop
    main.extend([load(8), request(4), vec![0x71, 0x04, 0x40, 0x00, 0x0b]].concat());
    main.extend([request(4), vec![0x04, 0x40], store(12, load(16)), vec![0x0b]].concat());
    main.extend(store(8, integer(0)));
    main.extend(store(16, load(12)));
    main.extend([request(8), vec![0x45, 0x04, 0x40], integer(0), vec![0x0f, 0x0b]].concat());
    for (operation, reply, action) in [
        (1, CREATE, store(16, integer(40))),
        (3, SNAPSHOT, [integer(SNAPSHOT + 8), load(12), vec![0xb2, 0x38, 2, 0]].concat()),
        // A reload must receive precisely the last accepted live state.
        (4, RESTORE, [request(12), integer(42), vec![0x47, 0x04, 0x40, 0x00, 0x0b], store(16, integer(42))].concat()),
    ] {
        main.extend(
            [
                request(8),
                integer(operation),
                vec![0x46, 0x04, 0x40],
                action,
                store(20, integer(reply)),
                exchange(),
                vec![0x0c, 1, 0x0b],
            ]
            .concat(),
        );
    }
    // Only Frame remains. Its third response fails, after startup and one live frame.
    main.extend([request(8), integer(2), vec![0x47, 0x04, 0x40, 0x00, 0x0b]].concat());
    main.extend(store(4, [load(4), integer(1), vec![0x6a]].concat()));
    main.extend(store(16, [load(12), integer(1), vec![0x6a]].concat()));
    main.extend(store(20, integer(GOOD)));
    let mut failure = Vec::new();
    if trap {
        failure.push(0x00); // unreachable: a terminated guest is a fatal error.
    }
    main.extend(
        [
            load(4),
            integer(reject_after),
            vec![0x4e, 0x04, 0x40],
            failure,
            store(8, integer(1)),
            store(20, integer(BAD)),
            vec![0x0b],
        ]
        .concat(),
    );
    main.extend(exchange());
    main.extend([0x0c, 0, 0x0b]);
    main.extend(integer(0));
    main.push(0x0b);
    let good = [vec![2.0], packet(red)].concat();
    let bad = [vec![2.0], bad.to_vec()].concat();
    program(
        main,
        &[
            (HANDSHAKE, &[4670544.0, 1.0, 1.0, 0.0]),
            (CREATE, &[1.0, 40.0]),
            (SNAPSHOT, &[3.0, 0.0]),
            (RESTORE, &[4.0, 42.0]),
            (GOOD, &good),
            (BAD, &bad),
        ],
    )
}

fn catalog() -> Catalog {
    Catalog { meshes: Vec::new(), materials: Vec::new(), sounds: Vec::new() }
}

fn start(engine: &wasmtime::Engine, gpu: Gpu, directory: &Path, module: GuestModule, dev: bool) -> Session {
    Session::start(
        Options {
            frames: 0,
            screenshot: None,
            record: None,
            fps: Some(60.0),
            mute: true,
            dev,
            script: Vec::new(),
            headless: true,
            settle: false,
        },
        Paths::new(
            directory.to_owned(),
            Some(directory.join("assets")),
            None,
            Some(directory.join("shots")),
            Some(directory.join("data")),
        ),
        engine.clone(),
        module,
        catalog(),
        gpu,
        None,
    )
    .unwrap()
}

#[test]
fn malformed_native_frames_pause_recover_and_never_commit() {
    let directory = PathBuf::from("/var/tmp").join(format!(
        "pen-frame-recovery-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir_all(directory.join("assets")).unwrap();
    let engine = blot_host::engine().unwrap();
    // A missing GPU is a failure, never a successful skipped regression.
    let mut gpu = Gpu::headless((32, 24), &directory.join("assets"), false).unwrap();
    let mut truncated = packet(1.0);
    truncated[3] = 1.0; // Claims a draw but supplies no draw lanes.
    let mut oversized = packet(1.0);
    oversized.push(0.0);
    let mut nonfinite = packet(1.0);
    nonfinite[52] = f32::NAN;
    for bad in [&truncated, &oversized, &nonfinite] {
        assert_eq!(decode_frame(bad, 1, &catalog()).err().unwrap().to_string(), DECODER_ERROR);
        for overlap in [false, true] {
            let shot = directory.join("shots/rejected.png");
            if shot.exists() {
                std::fs::remove_file(&shot).unwrap();
            }
            let mut session = start(&engine, gpu, &directory, guest(&engine, bad, 3, 1.0), true);
            let updates = Arc::new(AtomicUsize::new(0));
            session.overlay = Some(Box::new(TickingOverlay(updates.clone())));
            let trace = directory.join("frames.trace");
            let recording = directory.join("frames.penrec");
            session
                .playback(
                    Playback {
                        overlap,
                        trace: Some(trace.clone()),
                        record_input: Some(recording.clone()),
                        ..Playback::default()
                    },
                    None,
                )
                .unwrap();
            session.script = Some(Script::new(vec![Step::Wait(1), Step::Shot("rejected".into()), Step::Wait(100)]));
            assert!(session.step(None).unwrap());
            assert_eq!(session.overlap.flight.is_some(), overlap);
            assert_eq!(session.last.as_ref().unwrap().clear.r, 1.0);
            session.input.enqueue(GameEvent::KeyDown("a".into()));
            assert!(session.step(None).unwrap());
            assert_eq!(session.transport.run(), REJECTED);
            assert!(session.overlap.flight.is_none());
            assert_eq!(session.timeline.position(), 1);
            assert_eq!(session.timeline.head(), 1);
            assert_eq!(session.last.as_ref().unwrap().clear.r, 1.0);
            assert_eq!(session.runtime.snapshot().unwrap().state, [42.0]);
            assert!(shot.exists());
            let captured = image::open(&shot).unwrap().into_rgba8();
            assert_eq!(captured.dimensions(), (32, 24));
            assert!(captured.pixels().all(|pixel| pixel[1] == 0 && pixel[2] == 0 && pixel[3] == 255));
            let events = session.input.next_frame(0.0, Viewport { width: 32, height: 24 }).events;
            assert!(matches!(events.as_slice(), [GameEvent::FocusLost]));
            for _ in 0..3 {
                assert!(session.step(None).unwrap());
                assert!(session.transport.paused());
                assert_eq!(session.timeline.position(), 1);
                assert_eq!(session.last.as_ref().unwrap().clear.r, 1.0);
            }
            assert_eq!(updates.load(Ordering::Relaxed), 5);
            session.trace.as_mut().unwrap().flush().unwrap();
            session.stop_recording().unwrap();
            assert_eq!(std::fs::read_to_string(&trace).unwrap().lines().count(), 1);
            assert_eq!(recording::read(&recording).unwrap().frames.len(), 1);

            // An invalid replacement is delivered while paused and stays rejected.
            session.build = Some((guest(&engine, bad, 1, 0.0), catalog()));
            assert!(session.step(None).unwrap());
            assert_eq!(session.transport.run(), REJECTED);
            assert_eq!(session.runtime.snapshot().unwrap().state, [42.0]);
            assert_eq!(session.last.as_ref().unwrap().clear.r, 1.0);
            session.build = Some((guest(&engine, bad, i32::MAX, 0.0), catalog()));
            assert!(session.step(None).unwrap());
            assert_eq!(session.transport.run(), Run::Playing);
            assert_eq!(session.last.as_ref().unwrap().clear.r, 0.0);
            assert_eq!(session.timeline.position(), 2);
            // Drain the overlapped frame before a Snapshot request.
            session.overlap.enabled = false;
            if session.overlap.flight.is_some() {
                assert!(session.step(None).unwrap());
            }
            assert!(session.runtime.snapshot().unwrap().state[0] > 42.0);
            gpu = session.finish().unwrap();
        }

        // Production runs retain the precise decoder error as the cause.
        let mut session = start(&engine, gpu, &directory, guest(&engine, bad, 3, 1.0), false);
        assert!(session.step(None).unwrap());
        let error = session.step(None).unwrap_err();
        assert_eq!(error.root_cause().to_string(), DECODER_ERROR);
        let diagnostic = format!("{error:#}");
        if bad[52].is_nan() {
            assert!(diagnostic.contains("nonfinite value NaN at frame lane 52"));
        } else {
            assert!(diagnostic.contains(&format!("decoding a {}-lane game frame", bad.len())));
        }
        assert!(!session.transport.paused());
        assert_eq!(session.runtime.snapshot().unwrap().state, [42.0]);
        gpu = session.finish().unwrap();
    }

    // A deliberate single step stays paused when the corrected build arrives.
    let mut session = start(&engine, gpu, &directory, guest(&engine, &truncated, 3, 1.0), true);
    assert!(session.step(None).unwrap());
    session.pause();
    session.step_frames(1);
    assert!(session.step(None).unwrap());
    assert_eq!(session.transport.run(), Run::Paused { steps: 0, resume_after_reload: false });
    assert_eq!(session.runtime.snapshot().unwrap().state, [42.0]);
    // Exercise the real development watcher while redraws keep the app alive.
    std::fs::write(
        directory.join("assets/catalog.json"),
        r#"{"meshes":["unused.mesh"],"materials":["unused.material"],"sounds":[]}"#,
    )
    .unwrap();
    std::fs::write(directory.join("game.wasm"), guest_bytes(&truncated, i32::MAX, 0.0, false)).unwrap();
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    while session.last.as_ref().unwrap().clear.r == 1.0 {
        assert!(Instant::now() < deadline, "development watcher did not deliver the corrected build");
        assert!(session.step(None).unwrap());
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(session.transport.paused());
    assert_eq!(session.timeline.position(), 1);
    assert_eq!(session.last.as_ref().unwrap().clear.r, 0.0);
    gpu = session.finish().unwrap();
    let trapped = GuestModule::new(&engine, &guest_bytes(&truncated, 3, 1.0, true), None).unwrap();
    let mut session = start(&engine, gpu, &directory, trapped, true);
    assert!(session.step(None).unwrap());
    let error = session.step(None).unwrap_err();
    assert!(!error.is::<crate::game::InvalidFrame>());
    assert!(format!("{error:#}").contains("unreachable"));
    assert!(!session.transport.paused());
    drop(session.finish().unwrap());
    std::fs::remove_dir_all(directory).unwrap();
}
