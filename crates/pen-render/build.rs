use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let is_wasm32 = env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32");
    let minify = is_wasm32 || env::var("PEN_MINIFY_WGSL").as_deref() == Ok("1");

    println!("cargo:rerun-if-env-changed=PEN_MINIFY_WGSL");

    let shaders_src = manifest_dir.join("src/shaders");
    let shaders_out = out_dir.join("shaders");
    fs::create_dir_all(&shaders_out).unwrap();
    println!("cargo:rerun-if-changed={}", shaders_src.display());

    let mut entries: Vec<PathBuf> = fs::read_dir(&shaders_src)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("wgsl"))
        .collect();
    entries.sort();

    for path in entries {
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path.file_name().unwrap();
        let src = fs::read_to_string(&path).unwrap();
        let out = if minify { minify_wgsl(&src, !is_wasm32) } else { src };
        write_if_changed(&shaders_out.join(name), out.as_bytes());
    }

    let baseline_src = manifest_dir.join("src/surface/baseline.wgsl");
    if baseline_src.is_file() {
        println!("cargo:rerun-if-changed={}", baseline_src.display());
        let baseline_out = out_dir.join("surface");
        fs::create_dir_all(&baseline_out).unwrap();
        let src = fs::read_to_string(&baseline_src).unwrap();
        let out = if minify { minify_wgsl(&src, !is_wasm32) } else { src };
        write_if_changed(&baseline_out.join("baseline.wgsl"), out.as_bytes());
    }
}

fn write_if_changed(path: &Path, bytes: &[u8]) {
    if fs::read(path).ok().as_deref() == Some(bytes) {
        return;
    }
    fs::write(path, bytes).unwrap();
}

/// Minifies WGSL source safely for browser builds:
/// - Strips nested `/* ... */` block comments and `// ...` line comments
///   (preserving `// __...__` template/section markers used at runtime or const-eval).
/// - Trims leading/trailing whitespace, drops blank lines, and collapses runs of
///   horizontal whitespace to a single space (preserving single spaces so runtime
///   `.replace(...)` patterns match identically).
fn minify_wgsl(src: &str, keep_test_markers: bool) -> String {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut stripped = String::with_capacity(len);
    let mut i = 0;
    while i < len {
        if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            let mut depth = 1usize;
            i += 2;
            while i < len && depth > 0 {
                if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if i + 1 < len && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if !stripped.ends_with([' ', '\n', '\t', '\r']) {
                stripped.push(' ');
            }
        } else if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            // Keep `// __...__` marker comments verbatim.
            let line_end = src[i..].find('\n').map_or(len, |off| i + off);
            let comment = src[i..line_end].trim_end();
            if (comment.starts_with("// __") && comment.ends_with("__"))
                || (keep_test_markers && comment.starts_with("// SHEEN TABLE "))
            {
                stripped.push_str(comment);
            }
            i = line_end;
        } else {
            let ch = src[i..].chars().next().unwrap();
            stripped.push(ch);
            i += ch.len_utf8();
        }
    }

    let mut out = String::with_capacity(stripped.len());
    for raw_line in stripped.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        let mut prev_space = false;
        for ch in trimmed.chars() {
            if ch == ' ' || ch == '\t' {
                if !prev_space {
                    out.push(' ');
                    prev_space = true;
                }
            } else {
                out.push(ch);
                prev_space = false;
            }
        }
    }
    out
}
