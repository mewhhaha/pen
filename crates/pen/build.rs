use std::path::{Path, PathBuf};

fn main() {
    // Packaged builds ship Steam's library beside the executable.
    if std::env::var_os("CARGO_FEATURE_STEAM").is_some() {
        match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
            Ok("linux") => println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN"),
            Ok("macos") => println!("cargo:rustc-link-arg-bins=-Wl,-rpath,@executable_path"),
            _ => {}
        }
    }
    println!("cargo:rustc-env=PEN_RUNTIME_COMPATIBILITY={}", compatibility());
}

/// What a game.cwasm must have been precompiled by for this runtime to load
/// it: wasmtime's version (which its serialized modules must match) and a
/// hash of what configures its engine, blot-host and the workspace's
/// wasmtime features. Any change to blot-host or the ABI manifest parser
/// counts, which is cautious: a player built before it may run a guest
/// differently too, or refuse its ABI version.
fn compatibility() -> String {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let workspace = manifest.join("../..");
    let host = manifest.join("../blot-host");
    let abi = manifest.join("../pen-runtime/src/abi.rs");
    for path in [workspace.join("Cargo.lock"), workspace.join("Cargo.toml"), host.clone(), abi.clone()] {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-changed=build.rs");
    let lock = std::fs::read_to_string(workspace.join("Cargo.lock")).unwrap_or_default();
    let wasmtime = lock
        .split("[[package]]")
        .find(|package| package.lines().any(|line| line.trim() == "name = \"wasmtime\""))
        .and_then(|package| package.lines().find_map(|line| line.trim().strip_prefix("version = ")))
        .map_or_else(|| "unknown".to_owned(), |version| version.trim_matches('"').to_owned());
    let mut hash = Fnv::default();
    let members = std::fs::read_to_string(workspace.join("Cargo.toml")).unwrap_or_default();
    for line in members.lines().filter(|line| line.trim_start().starts_with("wasmtime")) {
        hash.write(line.as_bytes());
    }
    let mut files = Vec::new();
    collect(&host, &mut files);
    files.sort();
    for file in files {
        hash.write(file.strip_prefix(&host).unwrap_or(&file).to_string_lossy().as_bytes());
        hash.write(&std::fs::read(&file).unwrap_or_default());
    }
    hash.write(&std::fs::read(&abi).unwrap_or_default());
    format!("wasmtime {wasmtime}, engine {:016x}", hash.0)
}

/// Every file under `directory` but build output.
fn collect(directory: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if entry.file_name() != "target" {
                collect(&path, files);
            }
        } else {
            files.push(path);
        }
    }
}

/// FNV-1a, 64 bits: build scripts have no hashing crate to hand.
struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv {
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
        // A separator, so that moving bytes between writes changes the hash.
        self.0 = (self.0 ^ 0xff).wrapping_mul(0x0100_0000_01b3);
    }
}
