//! Shipping-only optimization; development builds and exported test bundles
//! retain the compiler's output. Names and embedded source maps travel with
//! the optimized instructions, and guest ABI/custom metadata is checked.
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use pen_runtime::abi::{custom_sections, read_abi, sections};
use pen_runtime::symbols::{SOURCE_MAP_SECTION, Symbols};
use sha2::{Digest, Sha256};

const MARK: &str = "pen:optimization";

fn instruction() -> &'static str {
    "install Binaryen's wasm-opt (version 133 is tested), or use pen build --no-optimize"
}

/// Optimize a staged file, publishing only after successful validation.
pub fn file(path: &Path) -> Result<()> {
    let source = std::fs::read(path)?;
    let version = Command::new("wasm-opt").arg("--version").output().with_context(instruction)?;
    ensure!(version.status.success(), "wasm-opt --version failed: {}", instruction());
    let version = String::from_utf8(version.stdout)?.trim().to_owned();
    let settings = format!("{version}; -O3 --all-features --debuginfo");
    let custom = custom_sections(&source)?;
    if let Some((_, marker)) = custom.iter().find(|(name, _)| name == MARK) {
        let without = replace_custom(&source, MARK, None)?;
        if *marker == stamp(&settings, &without).as_bytes() {
            return Ok(());
        }
    }
    let scratch = path.with_extension("optimization");
    std::fs::create_dir(&scratch)?;
    let result = optimize(path, &source, &custom, &scratch, &settings);
    let _ = std::fs::remove_dir_all(&scratch);
    result
}

fn optimize(path: &Path, source: &[u8], custom: &[(String, &[u8])], scratch: &Path, settings: &str) -> Result<()> {
    let output = scratch.join("output.wasm");
    let map_out = scratch.join("output.map");
    let mut command = Command::new("wasm-opt");
    command.args(["-O3", "--all-features", "--debuginfo"]).arg(path).arg("-o").arg(&output);
    let mapped = if let Some((_, payload)) = custom.iter().find(|(name, _)| name == SOURCE_MAP_SECTION) {
        let url = wasm_string(payload)?;
        let map = url.strip_prefix("data:application/json;base64,").context(
            "shipping optimization needs an embedded base64 JSON source map; use --no-optimize for an external map",
        )?;
        let map_in = scratch.join("input.map");
        // Binaryen 133 treats file 0, line 1, column 0 as its initial
        // "no location" value. Bias lines while it runs so the first source
        // line survives too, then restore the standard zero-based map.
        std::fs::write(&map_in, shift_lines(&STANDARD.decode(map)?, 1)?)?;
        command.arg("--input-source-map").arg(map_in).arg("--output-source-map").arg(&map_out);
        command.args(["--output-source-map-url", "pen-optimized.map"]);
        true
    } else {
        false
    };
    let run = command.output().with_context(instruction)?;
    ensure!(run.status.success(), "wasm-opt failed: {}", String::from_utf8_lossy(&run.stderr));
    let mut optimized = std::fs::read(&output)?;
    if mapped {
        // The map's URL must follow code: changing its length must not move
        // the instruction offsets Binaryen just mapped.
        let mut after_map = false;
        for (id, payload) in sections(&optimized)? {
            ensure!(!(after_map && id == 10), "wasm-opt put the source map before code");
            if id == 0 && custom_name(payload)? == SOURCE_MAP_SECTION {
                after_map = true;
            }
        }
        let map = shift_lines(&std::fs::read(&map_out)?, -1)?;
        let url = format!("data:application/json;base64,{}", STANDARD.encode(map));
        let mut payload = leb(url.len())?;
        payload.extend(url.as_bytes());
        optimized = replace_custom(&optimized, SOURCE_MAP_SECTION, Some(&payload))?;
    }
    optimized = replace_custom(&optimized, MARK, None)?;
    ensure!(read_abi(source)? == read_abi(&optimized)?, "wasm-opt changed the guest ABI");
    let after = custom_sections(&optimized)?;
    for (name, payload) in custom.iter().filter(|(name, _)| name.starts_with("blot:")) {
        ensure!(after.iter().any(|(n, p)| n == name && p == payload), "wasm-opt changed {name}");
    }
    let symbols = Symbols::read(&optimized);
    ensure!(symbols.problems.is_empty(), "optimized debug metadata is invalid: {:?}", symbols.problems);
    ensure!(!mapped || symbols.has_source_map(), "wasm-opt lost the source map");
    let marker = stamp(settings, &optimized);
    optimized = replace_custom(&optimized, MARK, Some(marker.as_bytes()))?;
    // Validate the final stamped bytes through the same cache used by the
    // startup-scene analysis, so it reuses this compilation. No guest code
    // runs during validation; target-specific precompilation stays separate.
    let engine = blot_host::engine()?;
    let cache = crate::cache_dir().map(|directory| directory.join("wasm"));
    blot_host::GuestModule::new(&engine, &optimized, cache.as_deref())?;
    std::fs::write(&output, &optimized)?;
    std::fs::rename(&output, path)?;
    eprintln!(
        "{}: wasm-opt -O3 {:.2} -> {:.2} MB ({settings})",
        path.file_name().unwrap_or_default().to_string_lossy(),
        source.len() as f64 / 1e6,
        optimized.len() as f64 / 1e6
    );
    Ok(())
}

fn stamp(settings: &str, bytes: &[u8]) -> String {
    format!("{settings}\n{:x}", Sha256::digest(bytes))
}

/// Source lines are delta-coded across segments. Changing just the first
/// mapped segment biases every line without touching instruction offsets.
fn shift_lines(map: &[u8], delta: i64) -> Result<Vec<u8>> {
    const DIGITS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut json: serde_json::Value = serde_json::from_slice(map)?;
    let mappings = json["mappings"].as_str().context("source map has no mappings")?;
    ensure!(!mappings.contains(';'), "Wasm source map must have one generated line");
    let mut start = 0;
    let mut replacement = None;
    for segment in mappings.split(',') {
        let mut fields = Vec::new();
        let (mut value, mut shift, mut at) = (0u64, 0u32, 0);
        for (i, byte) in segment.bytes().enumerate() {
            let digit = DIGITS.iter().position(|&d| d == byte).context("invalid source-map VLQ")? as u64;
            ensure!(shift < 60, "source-map VLQ overflow");
            value |= (digit & 31) << shift;
            shift += 5;
            if digit < 32 {
                let number = i64::try_from(value >> 1)? * if value & 1 == 0 { 1 } else { -1 };
                fields.push((at, i + 1, number));
                (value, shift, at) = (0, 0, i + 1);
            }
        }
        ensure!(shift == 0, "truncated source-map VLQ");
        if fields.len() >= 4 {
            let (from, to, line) = fields[2];
            let line = line.checked_add(delta).context("source-map line overflow")?;
            ensure!(line >= 0, "negative source-map line");
            let mut encoded = String::new();
            let mut value = u64::try_from(line)? << 1;
            while value >= 32 {
                encoded.push(DIGITS[usize::try_from((value & 31) | 32)?] as char);
                value >>= 5;
            }
            encoded.push(DIGITS[usize::try_from(value)?] as char);
            replacement = Some((start + from, start + to, encoded));
            break;
        }
        start += segment.len() + 1;
    }
    if let Some((from, to, encoded)) = replacement {
        json["mappings"] = format!("{}{}{}", &mappings[..from], encoded, &mappings[to..]).into();
    }
    Ok(serde_json::to_vec(&json)?)
}

fn leb(value: usize) -> Result<Vec<u8>> {
    let mut value = u32::try_from(value)?;
    let mut out = Vec::new();
    while value >= 128 {
        out.push(u8::try_from(value & 127)? | 128);
        value >>= 7;
    }
    out.push(u8::try_from(value)?);
    Ok(out)
}

fn string_prefix(bytes: &[u8]) -> Result<(&str, usize)> {
    let (mut length, mut at) = (0u32, 0usize);
    loop {
        let byte = *bytes.get(at).context("truncated Wasm string")?;
        ensure!(at < 5 && (at != 4 || byte < 16), "invalid Wasm string length");
        length |= u32::from(byte & 127) << (at * 7);
        at += 1;
        if byte < 128 {
            break;
        }
    }
    let end = at.checked_add(length as usize).context("Wasm string length overflow")?;
    let text = std::str::from_utf8(bytes.get(at..end).context("truncated Wasm string")?)?;
    Ok((text, end))
}

fn wasm_string(bytes: &[u8]) -> Result<&str> {
    let (text, end) = string_prefix(bytes)?;
    ensure!(end == bytes.len(), "trailing bytes in Wasm string");
    Ok(text)
}

fn custom_name(payload: &[u8]) -> Result<&str> {
    Ok(string_prefix(payload)?.0)
}

/// Preserve section framing and offsets when removing/replacing trailing
/// debug sections. Noncanonical input lengths are deliberately retained.
fn replace_custom(wasm: &[u8], name: &str, replacement: Option<&[u8]>) -> Result<Vec<u8>> {
    let parsed = sections(wasm)?;
    let mut out = wasm[..8].to_vec();
    let base = wasm.as_ptr() as usize;
    let mut start = 8;
    for (id, payload) in parsed {
        let end = payload.as_ptr() as usize - base + payload.len();
        if id != 0 || custom_name(payload)? != name {
            out.extend_from_slice(&wasm[start..end]);
        }
        start = end;
    }
    if let Some(payload) = replacement {
        let mut contents = leb(name.len())?;
        contents.extend(name.as_bytes());
        contents.extend(payload);
        out.push(0);
        out.extend(leb(contents.len())?);
        out.extend(contents);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Binaryen wasm-opt; run with --include-ignored"]
    fn shipping_preserves_execution_and_embedded_debug_info() {
        let root = crate::project::temporary_dir().unwrap().join(format!("optimize-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        // An unused function precedes main(i32) -> i32 (argument plus one).
        // DCE must change both main's function index and instruction offsets.
        let mut wasm = b"\0asm\x01\0\0\0\x01\x06\x01\x60\x01\x7f\x01\x7f\x03\x03\x02\0\0\x07\x08\x01\x04main\0\x01\x0a\x0e\x02\x04\0\x41\0\x0b\x07\0\x20\0\x41\x01\x6a\x0b".to_vec();
        wasm = replace_custom(&wasm, "blot:abi", Some(b"\x02\x01\x04main\x01\x01\0")).unwrap();
        wasm = replace_custom(&wasm, "name", Some(b"\x01\x0d\x01\x01\x0agame.plus1")).unwrap();
        // Byte 41 is local.get, mapped to the first source line and column.
        let map = br#"{"version":3,"sources":["src/fixture.blot"],"names":[],"mappings":"yCAAA"}"#;
        let url = format!("data:application/json;base64,{}", STANDARD.encode(map));
        let mut payload = leb(url.len()).unwrap();
        payload.extend(url.as_bytes());
        wasm = replace_custom(&wasm, SOURCE_MAP_SECTION, Some(&payload)).unwrap();
        let path = root.join("game.wasm");
        std::fs::write(&path, &wasm).unwrap();
        file(&path).unwrap();
        let optimized = std::fs::read(&path).unwrap();
        let engine = blot_host::engine().unwrap();
        let module = blot_host::GuestModule::new(&engine, &optimized, None).unwrap();
        let mut guest = blot_host::Guest::new(&engine, &module).unwrap();
        assert!(matches!(guest.call("main", blot_host::Value::U32(41)).unwrap(), blot_host::Value::U32(42)));
        let symbols = Symbols::read(&optimized);
        assert_eq!(symbols.function(0), Some("game.plus1"));
        assert!(symbols.has_source_map());
        let code_size = |bytes: &[u8]| sections(bytes).unwrap().into_iter().find(|(id, _)| *id == 10).unwrap().1.len();
        assert!(code_size(&optimized) < code_size(&wasm));
        assert!(
            (0..optimized.len())
                .any(|offset| symbols.location(offset).is_some_and(|l| l.file == "src/fixture.blot" && l.line == 1))
        );
        file(&path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), optimized, "repackaging is byte-idempotent");
        std::fs::remove_dir_all(root).unwrap();
    }
}
