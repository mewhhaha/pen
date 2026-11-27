use super::cache::*;
use super::toolchain::*;
use super::*;
use std::sync::atomic::AtomicU64;

fn test_temp_dir(name: &str) -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "squid-host-wtime-aot-{name}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test temp dir");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).expect("chmod 0700");
    }
    dir
}

fn default_x64_cfg() -> AotCompileConfig {
    AotCompileConfig {
        backend: "x64".to_string(),
        c_opt: "auto".to_string(),
        toolchain: "stable".to_string(),
        target_cpu: "x86-64-v3".to_string(),
        explicit_backend: false,
    }
}

#[test]
fn default_aot_backend_matches_host_arch() {
    #[cfg(all(target_os = "linux", target_arch = "aarch64", target_pointer_width = "64"))]
    assert_eq!(DEFAULT_AOT_BACKEND, "aarch64");
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64", target_pointer_width = "64")))]
    assert_eq!(DEFAULT_AOT_BACKEND, "x64");
}

#[test]
fn aarch64_backend_fallback_and_classification() {
    let mut cfg = AotCompileConfig {
        backend: "aarch64".to_string(),
        c_opt: "auto".to_string(),
        toolchain: "stable".to_string(),
        target_cpu: "native".to_string(),
        explicit_backend: false,
    };
    let mut seen = Vec::new();
    let res = load_verified_so_with_fallback(b"wasm", None, &cfg, |_w, _c, cur| {
        seen.push(cur.backend.clone());
        if cur.backend == "aarch64" {
            Err(classify_build_failure("aarch64", Some(X64_ADMISSION_REJECTED_EXIT_CODE), "exit status: 4", "rej"))
        } else {
            Ok(cur.backend.clone())
        }
    })
    .expect("implicit aarch64 admission rejection should fall back to c");
    assert_eq!(res, "c");
    assert_eq!(seen, vec!["aarch64", "c"]);

    cfg.explicit_backend = true;
    let err = load_verified_so_with_fallback(b"wasm", None, &cfg, |_w, _c, _cur| -> Result<()> {
        Err(classify_build_failure("aarch64", Some(X64_ADMISSION_REJECTED_EXIT_CODE), "exit status: 4", "rej"))
    })
    .expect_err("explicit aarch64 backend must propagate fallback error");
    assert!(err.downcast_ref::<X64FallbackError>().is_some());
    assert!(parse_aarch64_target_cpu_syntax("neoverse-v2").is_ok());
    assert!(parse_aarch64_target_cpu_syntax("bogus-cpu").is_err());
}

#[test]
fn default_x64_fallback_triggers_on_all_three_fallback_reasons() {
    let reasons = [
        X64FallbackReason::UnsupportedPlatform("not linux x86_64".into()),
        X64FallbackReason::UnsupportedIsa("missing sse4.1".into()),
        X64FallbackReason::AdmissionRejected("unsupported proposal".into()),
    ];
    for reason in reasons {
        let cfg = default_x64_cfg();
        let mut seen = Vec::new();
        let result = load_verified_so_with_fallback(b"wasm", None, &cfg, |_w, _c, cur| {
            seen.push(cur.clone());
            if cur.backend == "x64" {
                Err(anyhow::Error::new(X64FallbackError(reason.clone())))
            } else {
                Ok(cur.backend.clone())
            }
        })
        .expect("default x64 fallback should recover with c backend");
        assert_eq!(result, "c");
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].backend, "x64");
        assert_eq!(seen[1].backend, "c");
        assert_eq!(seen[1].c_opt, "auto");
        assert_eq!(seen[1].target_cpu, "native");
        assert!(seen[1].explicit_backend);
    }
}

#[test]
fn explicit_x64_backend_propagates_fallback_errors_without_fallback() {
    let reasons = [
        X64FallbackReason::UnsupportedPlatform("not linux x86_64".into()),
        X64FallbackReason::UnsupportedIsa("cet shstk enabled".into()),
        X64FallbackReason::AdmissionRejected("memory64 unsupported".into()),
    ];
    for reason in reasons {
        let mut cfg = default_x64_cfg();
        cfg.explicit_backend = true;
        let mut calls = 0usize;
        let err = load_verified_so_with_fallback(b"wasm", None, &cfg, |_w, _c, _cur| -> Result<()> {
            calls += 1;
            Err(anyhow::Error::new(X64FallbackError(reason.clone())))
        })
        .expect_err("explicit x64 backend must propagate X64FallbackError");
        assert_eq!(calls, 1);
        let typed = err.downcast_ref::<X64FallbackError>().expect("typed X64FallbackError");
        assert_eq!(typed.0, reason);
    }
}

#[test]
fn non_fallback_error_and_exit_code_1_propagate_even_on_default_x64() {
    let exit4 = classify_build_failure("x64", Some(X64_ADMISSION_REJECTED_EXIT_CODE), "exit status: 4", "bad op");
    assert!(matches!(
        exit4.downcast_ref::<X64FallbackError>(),
        Some(X64FallbackError(X64FallbackReason::AdmissionRejected(msg))) if msg == "bad op"
    ));

    let exit1 = classify_build_failure("x64", Some(1), "exit status: 1", "internal compiler error");
    assert!(exit1.downcast_ref::<X64FallbackError>().is_none());

    let c_exit4 = classify_build_failure("c", Some(X64_ADMISSION_REJECTED_EXIT_CODE), "exit status: 4", "c err");
    assert!(c_exit4.downcast_ref::<X64FallbackError>().is_none());

    let cfg = default_x64_cfg();
    let mut calls = 0usize;
    let err = load_verified_so_with_fallback(b"wasm", None, &cfg, |_w, _c, _cur| -> Result<()> {
        calls += 1;
        Err(classify_build_failure("x64", Some(1), "exit status: 1", "compiler io failure"))
    })
    .expect_err("exit code 1 must not trigger fallback");
    assert_eq!(calls, 1);
    assert!(err.downcast_ref::<X64FallbackError>().is_none());
    assert!(err.to_string().contains("compiler io failure"));
}

#[test]
fn bin_digest_caches_invalidates_on_change_and_ignores_tampered_entries() {
    let root = test_temp_dir("bin-digest");
    let cache = ensure_private_cache_dir(&root.join("cache")).expect("private cache");
    let bin = root.join("fake-wtime-aot");
    std::fs::write(&bin, b"binary-version-1").expect("write v1");
    let (canon1, d1) = bin_digest(&bin, &cache).expect("digest v1");
    assert_eq!(d1, wasm_sha256_hex(b"binary-version-1"));

    let bin_cache_dir = cache.join("bin-digests");
    let path_hash = wasm_sha256_hex(canon1.as_os_str().as_encoded_bytes());
    let entry = bin_cache_dir.join(format!("{path_hash}.txt"));
    assert!(entry.is_file(), "bin-digest entry should be persisted");

    // Second lookup hits cache with exact digest.
    let (_, d1_again) = bin_digest(&bin, &cache).expect("digest v1 cached");
    assert_eq!(d1_again, d1);

    // Overwriting the binary with a different length/content must change BinStamp and recompute.
    std::fs::write(&bin, b"binary-version-2-longer").expect("write v2");
    let (_, d2) = bin_digest(&bin, &cache).expect("digest v2");
    assert_eq!(d2, wasm_sha256_hex(b"binary-version-2-longer"));
    assert_ne!(d2, d1);

    // Corrupting the on-disk entry (oversized or invalid mode) must fail closed and recompute cleanly.
    let bin3 = root.join("fake-wtime-aot-3");
    std::fs::write(&bin3, b"binary-version-3").expect("write v3");
    let canon3 = std::fs::canonicalize(&bin3).expect("canon v3");
    let entry3 = bin_cache_dir.join(format!("{}.txt", wasm_sha256_hex(canon3.as_os_str().as_encoded_bytes())));
    std::fs::write(&entry3, vec![b'x'; 16_384]).expect("write oversized entry");
    let (_, d3) = bin_digest(&bin3, &cache).expect("digest v3 ignores oversized entry");
    assert_eq!(d3, wasm_sha256_hex(b"binary-version-3"));

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let bin4 = root.join("fake-wtime-aot-4");
        std::fs::write(&bin4, b"binary-version-4").expect("write v4");
        let canon4 = std::fs::canonicalize(&bin4).expect("canon v4");
        let entry4 = bin_cache_dir.join(format!("{}.txt", wasm_sha256_hex(canon4.as_os_str().as_encoded_bytes())));
        let st4 = bin_stamp(&canon4).expect("stamp v4");
        let forged_sha = "0".repeat(64);
        std::fs::write(&entry4, format!("v1|{}|{st4:?}\n{forged_sha}\n", canon4.display())).expect("write forged");
        std::fs::set_permissions(&entry4, std::fs::Permissions::from_mode(0o666)).expect("chmod 0666");
        let (_, d4) = bin_digest(&bin4, &cache).expect("digest v4 ignores world-writable entry");
        assert_eq!(d4, wasm_sha256_hex(b"binary-version-4"));
        assert_ne!(d4, forged_sha);
    }

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn validate_aot_wasm_rejects_huge_code_section_size_without_overflow_and_matches_imaged_globals() {
    use wasm_encoder::{
        CodeSection, ConstExpr, ExportKind, ExportSection, Function, FunctionSection, GlobalSection, GlobalType,
        Instruction, Module, TypeSection, ValType as EncValType,
    };
    let abi = GuestAbi::default();

    // Crafted Wasm with Code section (id=10) claiming size = u32::MAX (0xffff_ffff LEB128).
    let crafted = [
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic + version
        0x0a, // Code section id
        0xff, 0xff, 0xff, 0xff, 0x0f, // size = u32::MAX
        0x00, // truncated body
    ];
    assert!(validate_aot_wasm(&crafted, &abi).is_err());
    assert!(crate::imaged_globals(&crafted).is_err());

    // Valid module with 2 exported mutable i32 globals (one counter), 1 immutable global, and 1 function body:
    // verify validate_aot_wasm().imaged matches crate::imaged_globals() verbatim.
    let mut m = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([], []);
    m.section(&types);
    let mut funcs = FunctionSection::new();
    funcs.function(0);
    m.section(&funcs);
    let mut globals = GlobalSection::new();
    globals.global(GlobalType { val_type: EncValType::I32, mutable: true, shared: false }, &ConstExpr::i32_const(7));
    globals.global(GlobalType { val_type: EncValType::I32, mutable: true, shared: false }, &ConstExpr::i32_const(0));
    globals.global(GlobalType { val_type: EncValType::I32, mutable: false, shared: false }, &ConstExpr::i32_const(42));
    m.section(&globals);
    let mut exports = ExportSection::new();
    exports.export("g_mut", ExportKind::Global, 0);
    exports.export("squid:ctr", ExportKind::Global, 1);
    m.section(&exports);
    let mut code = CodeSection::new();
    let mut body = Function::new([]);
    body.instruction(&Instruction::Nop);
    body.instruction(&Instruction::End);
    code.function(&body);
    m.section(&code);
    let valid = m.finish();

    let validated = validate_aot_wasm(&valid, &abi).expect("validate valid module");
    assert_eq!(validated.imaged, crate::imaged_globals(&valid));
    assert_eq!(validated.imaged, Ok(vec!["g_mut".to_string(), "squid:ctr".to_string()]));
    assert_eq!(validated.counter_names, vec!["squid:ctr".to_string()]);

    // Hidden mutable global produces the exact same error string in both paths.
    let mut hm = Module::new();
    let mut hglobals = GlobalSection::new();
    hglobals.global(GlobalType { val_type: EncValType::I32, mutable: true, shared: false }, &ConstExpr::i32_const(1));
    hm.section(&hglobals);
    let hidden = hm.finish();
    let v_hidden = validate_aot_wasm(&hidden, &abi).expect("validate hidden");
    assert_eq!(v_hidden.imaged, crate::imaged_globals(&hidden));
    assert!(v_hidden.imaged.is_err());
}

#[cfg(feature = "compiler")]
#[test]
fn verify_artifact_sidecar_digest_accepts_plain_sha256_and_merkle256k_over_256kib() {
    let small = b"small-so-artifact";
    let small_sha = wasm_sha256_hex(small);
    assert!(verify_artifact_sidecar_digest(small, &small_sha, &small_sha));
    assert!(!verify_artifact_sidecar_digest(small, &small_sha, &"0".repeat(64)));
    assert!(!verify_artifact_sidecar_digest(small, &small_sha, "short"));

    // > 256 KiB payload (600 KiB = 3 chunks: 256 KiB + 256 KiB + 88 KiB) matching wtime-aot's merkle256k format.
    let mut large: Vec<u8> = (0u32..(600 << 10))
        .map(|i| u8::try_from((i.wrapping_mul(31) ^ (i >> 7)) & 0xff).expect("masked byte"))
        .collect();
    let large_sha = wasm_sha256_hex(&large);
    let merkle_line = format!("merkle256k:{}", merkle256k_sha256_hex(&large));
    assert!(verify_artifact_sidecar_digest(&large, &large_sha, &merkle_line));
    assert!(verify_artifact_sidecar_digest(&large, &large_sha, &large_sha));
    assert!(!verify_artifact_sidecar_digest(&large, &large_sha, &format!("merkle256k:{}", "0".repeat(64))));

    // Flip a byte in the final partial chunk and confirm Merkle verification fails closed.
    let last = large.len() - 1;
    large[last] ^= 0x01;
    let mutated_sha = wasm_sha256_hex(&large);
    assert!(!verify_artifact_sidecar_digest(&large, &mutated_sha, &merkle_line));
}

#[cfg(feature = "compiler")]
#[test]
fn parallel_merkle256k_matches_serial_and_verified_so_digest_roundtrips() {
    use sha2::{Digest, Sha256};
    // 1.1 MiB payload across 5 chunks (4 full 256 KiB + 1 partial) exercises multi-thread path.
    let payload: Vec<u8> =
        (0u32..(1120 << 10)).map(|i| u8::try_from((i.wrapping_mul(17) ^ (i >> 9)) & 0xff).expect("byte")).collect();
    let mut serial_root = Sha256::new();
    serial_root.update(WTIME_AOT_MERKLE256K_DOMAIN);
    serial_root.update((payload.len() as u64).to_le_bytes());
    for chunk in payload.chunks(WTIME_AOT_MERKLE256K_CHUNK_BYTES) {
        serial_root.update(Sha256::digest(chunk));
    }
    let expected_hex = format!("{:x}", serial_root.finalize());
    assert_eq!(merkle256k_sha256_hex(&payload), expected_hex);

    let line = format!("{WTIME_AOT_MERKLE256K_PREFIX}{expected_hex}");
    let parsed = VerifiedSoDigest::parse(&line).expect("parse merkle256k");
    assert_eq!(parsed.line_text(), line);
    assert_eq!(parsed.filename(), format!("m-{expected_hex}.so"));
    assert!(parsed.matches_bytes(&payload));

    let plain_hex = wasm_sha256_hex(b"tiny");
    let parsed_plain = VerifiedSoDigest::parse(&plain_hex).expect("parse plain sha256");
    assert_eq!(parsed_plain.line_text(), plain_hex);
    assert_eq!(parsed_plain.filename(), format!("{plain_hex}.so"));
    assert!(parsed_plain.matches_bytes(b"tiny"));
    assert!(VerifiedSoDigest::parse("merkle256k:short").is_none());
}
