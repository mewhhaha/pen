use anyhow::{Context, Result, ensure};
use blot_host::{Guest, GuestModule, Value};
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

fn fixture() -> Result<Vec<u8>> {
    static BYTES: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    BYTES
        .get_or_init(|| compile_fixture().map_err(|error| format!("{error:#}")))
        .clone()
        .map_err(|error| anyhow::anyhow!("{error}"))
}
fn compile_fixture() -> Result<Vec<u8>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let compiler = std::env::var_os("BLOTC")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("../../../blot2/zig-native/zig-out-dev/bin/blotc"));
    let output = root.join("../../target/owned-storage.wasm");
    let result = Command::new(compiler)
        .args(["build"])
        .arg(root.join("tests/fixtures/owned_storage.blot"))
        .arg(&output)
        .arg("--prelude")
        .arg(root.join("../../../blot2/std/prelude.blot"))
        .output()
        .context("compile owned-storage conformance fixture (BLOTC or sibling blot2)")?;
    ensure!(
        result.status.success(),
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(std::fs::read(output)?)
}
#[test]
fn owned_calls_preserve_copies_bounds_and_reuse_after_failure() -> Result<()> {
    let engine = blot_host::engine()?;
    let module = GuestModule::new(&engine, &fixture()?, None)?;
    let mut guest = Guest::new(&engine, &module)?;
    for _ in 0..20 {
        assert_eq!(guest.call("collected", Value::U32(3))?, Value::U32(48));
        let counters = guest.ownership();
        assert_eq!(counters.roots, None);
        assert_eq!(counters.leases, None);
        assert_eq!(counters.host_refs, None);
        assert_eq!(guest.call("incremented", Value::U32(41))?, Value::U32(42));
        let values = Value::ArrayU32(vec![0, 0xffff_ffff, 0x8000_0000, 17]);
        assert_eq!(guest.call("captured", values.clone())?, values);
        assert_eq!(guest.call("partitioned", Value::ArrayU32(vec![20, 99, 19, 88]))?, Value::U32(42));
        assert!(guest.call("partitioned", Value::ArrayU32(vec![20, 99])).is_err());
        assert!(guest.call("failing", Value::U32(7)).is_err());
        assert_eq!(guest.call("collected", Value::U32(3))?, Value::U32(48));
    }
    assert!(guest.capture(None).is_err());
    guest.dispose()?;
    guest.dispose()?;
    let counters = guest.ownership();
    assert_eq!(counters.roots, None);
    assert_eq!(counters.leases, None);
    Ok(())
}

#[test]
fn copied_worker_batches_are_bounded_ordered_and_reusable() -> Result<()> {
    let engine = blot_host::engine()?;
    let module = GuestModule::new(&engine, &fixture()?, None)?;
    let mut pool = blot_host::workers::Workers::new(&engine, &module, "collected", 4)?;
    for _ in 0..3 {
        let results = pool.run((0..32).map(Value::U32).collect())?;
        assert!(pool.timings().is_empty());
        for (index, value) in results.into_iter().enumerate() {
            assert_eq!(value.unwrap(), Value::U32(u32::try_from(index)? * 2 + 42));
        }
    }
    assert!(pool.run(vec![Value::U32(0); 33]).is_err());
    assert!(pool.run(vec![Value::ArrayU32(vec![0; 16_385])]).is_err());
    pool.set_tracing(true);
    assert!(pool.run(vec![Value::ArrayU32(vec![0])])?[0].is_err());
    assert_eq!(pool.timings().len(), 1);
    assert!(!pool.timings()[0].succeeded);
    let results = pool.run((0..32).map(Value::U32).collect())?;
    assert_eq!(pool.timings().len(), 32);
    for (index, timing) in pool.timings().iter().enumerate() {
        assert_eq!(timing.job, index);
        assert!(timing.worker < 4);
        assert!(timing.submitted <= timing.started && timing.started <= timing.finished);
        assert!(timing.succeeded);
        assert_eq!(results[index].as_ref().unwrap(), &Value::U32(u32::try_from(index)? * 2 + 42));
    }
    assert_eq!(pool.run(vec![Value::U32(1)])?[0].as_ref().unwrap(), &Value::U32(44));
    assert_eq!(pool.timings().len(), 1);
    pool.set_tracing(false);
    assert!(pool.timings().is_empty());
    assert!(pool.run(vec![])?.is_empty());
    pool.close()?;
    pool.close()?;
    assert!(pool.run(vec![Value::U32(0)]).is_err());
    Ok(())
}

#[test]
fn owned_capacity_failure_leaves_the_guest_reusable() -> Result<()> {
    let engine = blot_host::engine()?;
    let module = GuestModule::new(&engine, &fixture()?, None)?;
    // Planned storage exceeds the old 1 MiB sandbox before invocation.
    assert!(Guest::sandboxed(&engine, &module, blot_host::Limits { memory: 1 << 20 }).is_err());
    let mut guest = Guest::new(&engine, &module)?;
    assert_eq!(guest.call("repeated", Value::U32(128))?, Value::U32(8256));
    assert!(guest.call("large_result", Value::U32(1_000_000)).is_err());
    assert_eq!(guest.call("collected", Value::U32(1))?, Value::U32(44));
    assert_eq!(guest.call("repeated", Value::U32(5000))?, Value::U32(12_502_500));
    assert_eq!(guest.call("incremented", Value::U32(41))?, Value::U32(42));
    assert_eq!(guest.ownership().roots, None);
    assert_eq!(guest.ownership().leases, None);
    guest.dispose()?;
    Ok(())
}
