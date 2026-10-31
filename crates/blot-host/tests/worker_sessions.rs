use anyhow::Result;
use blot_host::{GuestModule, Value, worker_sessions::Workers};
use std::{path::PathBuf, process::Command, sync::OnceLock};
fn engine() -> &'static wasmtime::Engine {
    static ENGINE: OnceLock<wasmtime::Engine> = OnceLock::new();
    ENGINE.get_or_init(|| blot_host::engine().unwrap())
}
fn module() -> GuestModule {
    static MODULE: OnceLock<GuestModule> = OnceLock::new();
    MODULE
        .get_or_init(|| {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let compiler = std::env::var_os("BLOTC")
                .map(PathBuf::from)
                .unwrap_or_else(|| root.join("../../../blot2/zig-native/zig-out/bin/blotc"));
            let output = root.join("../../target/worker-sessions.wasm");
            let result = Command::new(compiler)
                .arg("build")
                .arg(root.join("tests/fixtures/worker_sessions.blot"))
                .arg(&output)
                .arg("--prelude")
                .arg(root.join("../../../blot2/std/prelude.blot"))
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            GuestModule::new(engine(), &std::fs::read(output).unwrap(), None).unwrap()
        })
        .clone()
}
#[test]
fn sessions_prepare_once_preserve_order_and_recover_from_each_failure() -> Result<()> {
    let module = module();
    let engine = engine().clone();
    for workers in [1, 4] {
        let mut pool = Workers::new(&engine, &module, "service", workers)?;
        pool.configure(u32::MAX, &[100.])?;
        let inputs: Vec<_> = (0..32).map(|i| vec![f32::from(u16::try_from(i).unwrap())]).collect();
        let mut memory = vec![];
        for iteration in 0..1000 {
            let results = pool.run_prepared(inputs.clone())?;
            for (i, answer) in results.iter().enumerate() {
                assert_eq!(answer.as_ref().unwrap(), &[100. + f32::from(u16::try_from(i)?)]);
            }
            if iteration == 10 {
                memory = pool.stats().memory_bytes;
            }
            if iteration > 10 {
                assert_eq!(pool.stats().memory_bytes, memory);
            }
        }
        assert_eq!(pool.stats().starts, workers);
        assert_eq!(pool.stats().contexts, workers);
        assert_eq!(pool.stats().context_lanes, workers);
        assert!(pool.configure(u32::MAX, &[101.]).is_err());
        for failure in [-1., -2., -3.] {
            let answers = pool.run_prepared(vec![vec![1.], vec![failure], vec![2.]])?;
            assert_eq!(answers[0].as_ref().unwrap(), &[101.]);
            assert!(answers[1].is_err());
            assert_eq!(answers[2].as_ref().unwrap(), &[102.]);
            assert!(pool.run_prepared(inputs.clone())?.iter().all(Result::is_ok));
        }
        pool.reset();
        assert!(pool.run_prepared(vec![vec![1.]]).is_err());
        pool.configure(u32::MAX, &[200.])?;
        assert_eq!(pool.run_prepared(vec![vec![1.]])?[0].as_ref().unwrap(), &[201.]);
        let bad = pool.run(0, &[], inputs.clone())?;
        assert!(bad.iter().all(Result::is_err));
        assert!(pool.run(1, &[100.], inputs)?.iter().all(Result::is_ok));
        pool.close()?;
        assert_eq!(pool.stats().memory_bytes, vec![0; workers]);
        pool.close()?;
        assert!(pool.run_prepared(vec![]).is_err());
    }
    Ok(())
}
#[test]
fn sessions_reject_bad_framing_before_submitting_jobs() -> Result<()> {
    let module = module();
    let engine = engine().clone();
    assert!(Workers::new(&engine, &module, "wrong_abi", 1).is_err());
    assert!(Workers::new(&engine, &module, "missing", 1).is_err());
    assert!(Workers::new(&engine, &module, "service", 0).is_err());
    let mut bad = Workers::new(&engine, &module, "bad_handshake", 1)?;
    assert!(bad.run(0, &[1.], vec![vec![0.]])?[0].is_err());
    bad.close()?;
    let mut pool = Workers::new(&engine, &module, "service", 2)?;
    const C: f32 = -4670547.;
    const B: f32 = -4670545.;
    for packet in [
        vec![],
        vec![C],
        vec![C, 9.],
        vec![C, 0., 0.],
        vec![C, 1., 65536., 0., 0.],
        vec![C, 1., 0.5, 0., 0.],
        vec![C, 1., 0., 0., 1.],
        vec![C, 2., B, 1., 2., 0.],
        vec![C, 2., B, 0., 0.],
        vec![C, 2., B, 33.],
        vec![C, 2., B, f32::NAN],
    ] {
        assert!(pool.call(Value::ArrayF32(packet)).is_err());
    }
    assert_eq!(pool.stats().jobs, 0);
    assert!(pool.configure(0, &[f32::NAN]).is_err());
    assert!(pool.configure(0, &vec![0.; 16385]).is_err());
    pool.configure(0, &[0.])?;
    assert!(pool.configure(0, &[-0.]).is_err());
    assert!(pool.run_prepared(vec![vec![0.]; 33]).is_err());
    assert!(pool.run_prepared(vec![vec![0.; 16385]]).is_err());
    assert!(pool.run_prepared(vec![vec![f32::INFINITY]]).is_err());
    assert_eq!(pool.stats().jobs, 0);
    assert_eq!(pool.call(Value::ArrayF32(vec![C, 1., 65535., 65535., 1., 7.]))?, Value::ArrayF32(vec![C, 1.]));
    assert_eq!(
        pool.call(Value::ArrayF32(vec![C, 2., B, 2., 1., 3., 1., 4.]))?,
        Value::ArrayF32(vec![B, 2., 1., 10., 1., 11.])
    );
    assert!(pool.call(Value::ArrayF32(vec![C, 2., B, 2., 1., 3., 1., -1.])).is_err());
    assert_eq!(pool.run_prepared(vec![vec![1.]])?[0].as_ref().unwrap(), &[8.]);
    Ok(())
}
