//! Metadata for a native-only package. Instructions and initialized memory
//! already live in Wasmtime's artifact; retain only the host's ABI, symbols
//! and snapshot-global names. Source-map offsets still refer to the original
//! Wasm and match the offsets recorded in that artifact.
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    version: u32,
    fingerprint: u64,
    sections: Vec<u8>,
    globals: Result<Vec<String>, String>,
}

/// Extracts the host information needed beside a precompiled artifact.
#[cfg(feature = "compiler")]
pub fn native_metadata(wasm: &[u8]) -> Result<Vec<u8>> {
    read_abi(wasm)?;
    let mut sections = b"\0asm\x01\0\0\0".to_vec();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        let payload = payload?;
        let keep = match &payload {
            wasmparser::Payload::ExportSection(_) => true,
            wasmparser::Payload::CustomSection(section) => {
                matches!(section.name(), "blot:abi" | "blot:tasks" | "name" | "sourceMappingURL")
            }
            _ => false,
        };
        if keep {
            let (id, range) = payload.as_section().context("metadata section has no range")?;
            let range = usize::try_from(range.start)?..usize::try_from(range.end)?;
            sections.push(id);
            let mut length = u32::try_from(range.len())?;
            loop {
                let byte = u8::try_from(length & 127)?;
                length >>= 7;
                sections.push(byte | if length == 0 { 0 } else { 128 });
                if length == 0 {
                    break;
                }
            }
            sections.extend_from_slice(&wasm[range]);
        }
    }
    Ok(serde_json::to_vec(&Metadata {
        version: 1,
        fingerprint: pen_runtime::net::wire::hash_bytes(wasm),
        sections,
        globals: imaged_globals(wasm),
    })?)
}

/// The original Wasm's identity, shared by native and browser net sessions.
pub fn native_fingerprint(metadata: &[u8]) -> Result<u64> {
    let metadata: Metadata = serde_json::from_slice(metadata)?;
    ensure!(metadata.version == 1, "unsupported native metadata version");
    Ok(metadata.fingerprint)
}

impl GuestModule {
    /// Loads a native package without an executable Wasm copy or a compiler
    /// fallback. Incompatible artifacts require repackaging.
    ///
    /// # Safety
    /// `precompiled` must be trusted output from [`precompile`], like the
    /// executable itself. Metadata must come from the same original module.
    pub unsafe fn from_native(engine: &Engine, metadata: &[u8], precompiled: &[u8]) -> Result<Self> {
        ensure!(metadata.len() <= 32 * 1024 * 1024, "native metadata exceeds its size limit");
        let metadata: Metadata = serde_json::from_slice(metadata).context("reading native game metadata")?;
        ensure!(metadata.version == 1, "unsupported native metadata version");
        let abi = Arc::new(read_abi(&metadata.sections)?);
        // SAFETY: the caller vouches for the artifact; Wasmtime checks compatibility.
        let module = unsafe { Module::deserialize(engine, precompiled) }
            .map_err(anyhow::Error::from)
            .context("the precompiled game does not suit this runtime; package it again with `pen build`")?;
        validate_module(&module, &abi)?;
        Ok(Self {
            module,
            trace: None,
            abi,
            symbols: Arc::new(Symbols::read(&metadata.sections)),
            imaged: Arc::new(metadata.globals),
        })
    }
}
