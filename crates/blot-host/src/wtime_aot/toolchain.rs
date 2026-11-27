//! C/Rust toolchain probing, ELF SHSTK/CET detection, and host ISA fingerprinting
//! for `wtime-aot` cache invalidation and backend admission.

use std::io::{Read as _, Seek as _, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};

use super::cache::*;
use super::{
    X64FallbackError, X64FallbackReason, find_in_path, is_bare_prog_name, is_executable_file, sanitized_path_env,
};

/// Inspects an ELF64 Linux x86-64 shared library for `NT_GNU_PROPERTY_TYPE_0` carrying
/// `GNU_PROPERTY_X86_FEATURE_1_SHSTK`.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub(super) fn elf64_has_x86_shstk_property(elf: &[u8]) -> bool {
    if elf.len() < 64 || &elf[0..4] != b"\x7fELF" || elf[4] != 2 || elf[5] != 1 {
        return false;
    }
    let e_machine = u16::from_le_bytes(elf[18..20].try_into().unwrap());
    if e_machine != 62 {
        return false;
    }
    let Ok(e_phoff) = usize::try_from(u64::from_le_bytes(elf[32..40].try_into().unwrap())) else {
        return false;
    };
    let e_phentsize = usize::from(u16::from_le_bytes(elf[54..56].try_into().unwrap()));
    if e_phentsize < 56 {
        return false;
    }
    let e_phnum = usize::from(u16::from_le_bytes(elf[56..58].try_into().unwrap()));
    const PT_NOTE: u32 = 4;
    const PT_GNU_PROPERTY: u32 = 0x6474_e553;
    const NT_GNU_PROPERTY_TYPE_0: u32 = 5;
    const GNU_PROPERTY_X86_FEATURE_1_AND: u32 = 0xc000_0002;
    const GNU_PROPERTY_X86_FEATURE_1_SHSTK: u32 = 1 << 1;
    let align_up = |v: usize, a: usize| v.checked_add(a - 1).map(|x| x & !(a - 1));
    for i in 0..e_phnum {
        let Some(off) = i.checked_mul(e_phentsize).and_then(|d| e_phoff.checked_add(d)) else {
            break;
        };
        let Some(ph_end) = off.checked_add(56) else {
            break;
        };
        if ph_end > elf.len() {
            break;
        }
        let p_type = u32::from_le_bytes(elf[off..off + 4].try_into().unwrap());
        if p_type != PT_NOTE && p_type != PT_GNU_PROPERTY {
            continue;
        }
        let Ok(p_offset) = usize::try_from(u64::from_le_bytes(elf[off + 8..off + 16].try_into().unwrap())) else {
            continue;
        };
        let Ok(p_filesz) = usize::try_from(u64::from_le_bytes(elf[off + 32..off + 40].try_into().unwrap())) else {
            continue;
        };
        let Ok(p_align) = usize::try_from(u64::from_le_bytes(elf[off + 48..off + 56].try_into().unwrap())) else {
            continue;
        };
        let note_align = if p_type == PT_GNU_PROPERTY || p_align >= 8 { 8 } else { 4 };
        let mut cur = p_offset;
        let end = p_offset.saturating_add(p_filesz).min(elf.len());
        while let Some(name_start) = cur.checked_add(12) {
            if name_start > end {
                break;
            }
            let Ok(namesz) = usize::try_from(u32::from_le_bytes(elf[cur..cur + 4].try_into().unwrap())) else {
                break;
            };
            let Ok(descsz) = usize::try_from(u32::from_le_bytes(elf[cur + 4..cur + 8].try_into().unwrap())) else {
                break;
            };
            let ntype = u32::from_le_bytes(elf[cur + 8..cur + 12].try_into().unwrap());
            let Some(name_end) = name_start.checked_add(namesz) else {
                break;
            };
            let Some(desc_start) = align_up(name_end, note_align) else {
                break;
            };
            let Some(desc_end) = desc_start.checked_add(descsz) else {
                break;
            };
            if desc_end > end {
                break;
            }
            if ntype == NT_GNU_PROPERTY_TYPE_0 && &elf[name_start..name_end] == b"GNU\0" {
                let mut p = desc_start;
                while let Some(data_start) = p.checked_add(8) {
                    if data_start > desc_end {
                        break;
                    }
                    let pr_type = u32::from_le_bytes(elf[p..p + 4].try_into().unwrap());
                    let Ok(pr_datasz) = usize::try_from(u32::from_le_bytes(elf[p + 4..p + 8].try_into().unwrap()))
                    else {
                        break;
                    };
                    let Some(data_end) = data_start.checked_add(pr_datasz) else {
                        break;
                    };
                    if data_end > desc_end {
                        break;
                    }
                    if pr_type == GNU_PROPERTY_X86_FEATURE_1_AND && pr_datasz == 4 {
                        let bits = u32::from_le_bytes(elf[data_start..data_start + 4].try_into().unwrap());
                        if (bits & GNU_PROPERTY_X86_FEATURE_1_SHSTK) != 0 {
                            return true;
                        }
                    }
                    let Some(next_p) = align_up(data_end, 8) else {
                        break;
                    };
                    p = next_p;
                }
            }
            let Some(next_cur) = align_up(desc_end, note_align) else {
                break;
            };
            cur = next_cur;
        }
    }
    false
}

pub(super) type WatchedFile = (PathBuf, PathBuf, BinStamp);

pub(super) const SYS_HEADER_PROBE_C: &[u8] = b"#if defined(__linux__) && !defined(_GNU_SOURCE)\n\
#define _GNU_SOURCE\n\
#endif\n\
#include <stdint.h>\n\
#include <stddef.h>\n\
#include <stdlib.h>\n\
#include <string.h>\n\
#include <math.h>\n\
#if defined(__linux__)\n\
#include <pthread.h>\n\
#endif\n";

pub(super) fn clean_cc_cmd(program: impl AsRef<std::ffi::OsStr>) -> Command {
    let mut cmd = Command::new(program);
    cmd.env_clear();
    if let Some(path) = sanitized_path_env() {
        cmd.env("PATH", path);
    }
    for var in ["HOME", "TMPDIR"] {
        if let Some(v) = std::env::var_os(var) {
            cmd.env(var, v);
        }
    }
    cmd
}

pub(super) const LINK_PROBE_C: &[u8] = b"#if defined(__linux__) && !defined(_GNU_SOURCE)\n\
#define _GNU_SOURCE\n\
#endif\n\
#include <math.h>\n\
#include <stdlib.h>\n\
#include <string.h>\n\
#if defined(__linux__)\n\
#include <pthread.h>\n\
#endif\n\
int wt_cet_probe(size_t n) {\n\
    void *p = malloc(n ? n : 1);\n\
    if (!p) return 0;\n\
    memset(p, 0, 1);\n\
    int r = memcmp(p, p, 1);\n\
    free(p);\n\
    volatile float x = (float)n + 0.5f;\n\
    r += (int)truncf(x);\n\
#if defined(__linux__)\n\
    pthread_attr_t a;\n\
    if (pthread_getattr_np(pthread_self(), &a) == 0) pthread_attr_destroy(&a);\n\
#endif\n\
    return r;\n\
}\n";

pub(super) fn stamp_len(stamp: BinStamp) -> u64 {
    #[cfg(unix)]
    {
        stamp.2
    }
    #[cfg(not(unix))]
    {
        stamp.0
    }
}

pub(super) fn hash_stamp(h: &mut sha2::Sha256, stamp: BinStamp) {
    use sha2::Digest;
    #[cfg(unix)]
    {
        h.update(stamp.0.to_le_bytes());
        h.update(stamp.1.to_le_bytes());
        h.update(stamp.2.to_le_bytes());
        h.update(stamp.3.to_le_bytes());
        h.update(stamp.4.to_le_bytes());
        h.update(stamp.5.to_le_bytes());
        h.update(stamp.6.to_le_bytes());
    }
    #[cfg(not(unix))]
    {
        h.update(stamp.0.to_le_bytes());
        h.update(stamp.1.to_le_bytes());
        h.update(stamp.2.to_le_bytes());
    }
}

/// Resolves an auxiliary toolchain program via `cc -print-prog-name=<prog>` (with `PATH` fallback
/// when `allow_path_fallback` is true). Fails closed on non-zero exit or relative non-bare paths.
pub(super) fn resolve_driver_prog(
    cc_path: &Path,
    prog: &str,
    allow_path_fallback: bool,
    scratch_dir: &Path,
    deadline: Instant,
) -> Result<Option<WatchedFile>> {
    let mut cmd = clean_cc_cmd(cc_path);
    cmd.arg(format!("-print-prog-name={prog}"));
    let out = run_cmd_output_with_deadline(
        cmd,
        scratch_dir,
        &format!("{} -print-prog-name={prog}", cc_path.display()),
        deadline,
    )?;
    ensure!(
        out.status.success(),
        "{} -print-prog-name={prog} failed ({}): {}",
        cc_path.display(),
        out.status,
        String::from_utf8_lossy(&out.stderr).trim()
    );
    let printed = String::from_utf8_lossy(&out.stdout).trim().to_string();
    ensure!(!printed.is_empty(), "{} -print-prog-name={prog} returned empty output", cc_path.display());
    let candidate = if is_bare_prog_name(&printed) {
        if allow_path_fallback {
            find_in_path(&printed).or_else(|| find_in_path(prog))
        } else {
            return Ok(None);
        }
    } else {
        let p = PathBuf::from(&printed);
        ensure!(
            p.is_absolute() && is_executable_file(&p),
            "{} -print-prog-name={prog} returned non-absolute or non-executable path {printed:?}",
            cc_path.display()
        );
        Some(p)
    };
    let Some(cand) = candidate else { return Ok(None) };
    let canon = std::fs::canonicalize(&cand).with_context(|| format!("canonicalizing {prog} {}", cand.display()))?;
    let stamp = bin_stamp(&canon)?;
    Ok(Some((cand, canon, stamp)))
}

pub(super) fn resolve_driver_linker(
    cc_path: &Path,
    has_lld: bool,
    scratch_dir: &Path,
    deadline: Instant,
) -> Result<WatchedFile> {
    let prog = if has_lld { "ld.lld" } else { "ld" };
    resolve_driver_prog(cc_path, prog, true, scratch_dir, deadline)?
        .ok_or_else(|| anyhow!("could not resolve driver linker {prog} for C compiler {}", cc_path.display()))
}

#[cfg(target_os = "linux")]
pub(super) fn elf64_pt_interp(elf_path: &Path) -> Result<Option<PathBuf>> {
    let mut f = std::fs::File::open(elf_path).with_context(|| format!("opening {}", elf_path.display()))?;
    let file_len = f.metadata().with_context(|| format!("stat {}", elf_path.display()))?.len();
    if file_len < 4 {
        return Ok(None);
    }
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).with_context(|| format!("reading {}", elf_path.display()))?;
    if &magic != b"\x7fELF" {
        return Ok(None);
    }
    ensure!(file_len >= 64, "{}: truncated ELF header ({file_len} bytes)", elf_path.display());
    f.seek(SeekFrom::Start(0)).with_context(|| format!("seeking {}", elf_path.display()))?;
    let mut ehdr = [0u8; 64];
    f.read_exact(&mut ehdr).with_context(|| format!("reading ELF header {}", elf_path.display()))?;
    ensure!(
        ehdr[4] == 2 && ehdr[5] == 1,
        "{}: unsupported ELF class/endianness ({}, {}) for shared-library tracking",
        elf_path.display(),
        ehdr[4],
        ehdr[5]
    );
    let e_phoff = u64::from_le_bytes(ehdr[32..40].try_into().unwrap());
    let e_phentsize = u64::from(u16::from_le_bytes(ehdr[54..56].try_into().unwrap()));
    let e_phnum = u64::from(u16::from_le_bytes(ehdr[56..58].try_into().unwrap()));
    if e_phnum == 0 {
        return Ok(None);
    }
    ensure!(
        e_phentsize >= 56 && e_phnum <= 4096,
        "{}: invalid ELF program header table (phentsize={e_phentsize}, phnum={e_phnum})",
        elf_path.display()
    );
    let ph_table_bytes = e_phentsize
        .checked_mul(e_phnum)
        .ok_or_else(|| anyhow!("{}: ELF program header size overflow", elf_path.display()))?;
    let ph_end = e_phoff
        .checked_add(ph_table_bytes)
        .ok_or_else(|| anyhow!("{}: ELF program header offset overflow", elf_path.display()))?;
    ensure!(ph_end <= file_len, "{}: ELF program headers exceed file size", elf_path.display());
    f.seek(SeekFrom::Start(e_phoff)).with_context(|| format!("seeking phdrs in {}", elf_path.display()))?;
    let mut phdrs = vec![0u8; usize::try_from(ph_table_bytes)?];
    f.read_exact(&mut phdrs).with_context(|| format!("reading phdrs in {}", elf_path.display()))?;
    const PT_DYNAMIC: u32 = 2;
    const PT_INTERP: u32 = 3;
    let stride = usize::try_from(e_phentsize)?;
    let mut has_dynamic = false;
    let mut interp_span: Option<(u64, u64)> = None;
    for i in 0..usize::try_from(e_phnum)? {
        let off = i * stride;
        let p_type = u32::from_le_bytes(phdrs[off..off + 4].try_into().unwrap());
        if p_type == PT_DYNAMIC {
            has_dynamic = true;
        } else if p_type == PT_INTERP {
            let p_offset = u64::from_le_bytes(phdrs[off + 8..off + 16].try_into().unwrap());
            let p_filesz = u64::from_le_bytes(phdrs[off + 32..off + 40].try_into().unwrap());
            interp_span = Some((p_offset, p_filesz));
        }
    }
    let Some((p_offset, p_filesz)) = interp_span else {
        ensure!(!has_dynamic, "{}: dynamic ELF has PT_DYNAMIC but no PT_INTERP", elf_path.display());
        return Ok(None);
    };
    ensure!(
        p_filesz > 0 && p_filesz <= 4096 && p_offset.checked_add(p_filesz).is_some_and(|e| e <= file_len),
        "{}: invalid PT_INTERP span ({p_offset}, {p_filesz})",
        elf_path.display()
    );
    f.seek(SeekFrom::Start(p_offset)).with_context(|| format!("seeking PT_INTERP in {}", elf_path.display()))?;
    let mut raw = vec![0u8; usize::try_from(p_filesz)?];
    f.read_exact(&mut raw).with_context(|| format!("reading PT_INTERP in {}", elf_path.display()))?;
    let trimmed = raw.split(|&b| b == 0).next().unwrap_or(&raw);
    let s = std::str::from_utf8(trimmed).map_err(|_| anyhow!("{}: non-UTF-8 PT_INTERP path", elf_path.display()))?;
    let p = PathBuf::from(s);
    ensure!(
        p.is_absolute() && is_executable_file(&p),
        "{}: PT_INTERP {s:?} is not an absolute executable",
        elf_path.display()
    );
    Ok(Some(p))
}

pub(super) fn probe_dynamic_shared_libs(
    elf_path: &Path,
    scratch_dir: &Path,
    deadline: Instant,
) -> Result<Vec<WatchedFile>> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (elf_path, scratch_dir, deadline);
        Ok(Vec::new())
    }
    #[cfg(target_os = "linux")]
    {
        let Some(interp) = elf64_pt_interp(elf_path)? else {
            return Ok(Vec::new());
        };
        let mut cmd = clean_cc_cmd(&interp);
        cmd.arg("--list").arg(elf_path);
        let out = run_cmd_output_with_deadline(
            cmd,
            scratch_dir,
            &format!("{} --list {}", interp.display(), elf_path.display()),
            deadline,
        )?;
        ensure!(
            out.status.success(),
            "{} --list {} failed: {}",
            interp.display(),
            elf_path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
        let mut libs = Vec::new();
        let interp_canon = std::fs::canonicalize(&interp)
            .with_context(|| format!("canonicalizing dynamic linker {}", interp.display()))?;
        let interp_stamp = bin_stamp(&interp_canon)?;
        libs.push((interp, interp_canon, interp_stamp));
        for raw_line in String::from_utf8_lossy(&out.stdout).lines() {
            let line = raw_line.trim();
            if line.is_empty()
                || line.starts_with("linux-vdso.")
                || line.starts_with("linux-gate.")
                || line == "statically linked"
            {
                continue;
            }
            let rhs = if let Some((_, after)) = line.split_once(" => ") { after.trim() } else { line };
            let path_part = match rhs.rfind(" (0x") {
                Some(idx) => rhs[..idx].trim(),
                None => rhs.trim(),
            };
            let p = Path::new(path_part);
            ensure!(
                path_part != "not found" && p.is_absolute() && p.is_file(),
                "unresolved or non-absolute shared library for {}: {line}",
                elf_path.display()
            );
            let canon =
                std::fs::canonicalize(p).with_context(|| format!("canonicalizing shared library {}", p.display()))?;
            let stamp = bin_stamp(&canon)?;
            libs.push((p.to_path_buf(), canon, stamp));
        }
        libs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        libs.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
        Ok(libs)
    }
}

pub(super) fn hash_shared_lib_sample(h: &mut sha2::Sha256, canon: &Path, stamp: BinStamp) -> Result<()> {
    use sha2::Digest;
    let path_bytes = canon.as_os_str().as_encoded_bytes();
    h.update((path_bytes.len() as u64).to_le_bytes());
    h.update(path_bytes);
    hash_stamp(h, stamp);
    let mut f = std::fs::File::open(canon).with_context(|| format!("opening {}", canon.display()))?;
    let total_len = f.metadata().with_context(|| format!("stat {}", canon.display()))?.len();
    ensure!(
        total_len == stamp_len(stamp),
        "shared library {} changed size during fingerprint ({} -> {total_len})",
        canon.display(),
        stamp_len(stamp)
    );
    let mut head = Vec::new();
    (&mut f).take(65536).read_to_end(&mut head).with_context(|| format!("reading head of {}", canon.display()))?;
    h.update((head.len() as u64).to_le_bytes());
    h.update(&head);
    if total_len > 65536 {
        f.seek(SeekFrom::Start(total_len - 65536)).with_context(|| format!("seeking tail of {}", canon.display()))?;
        let mut tail = Vec::new();
        (&mut f).take(65536).read_to_end(&mut tail).with_context(|| format!("reading tail of {}", canon.display()))?;
        h.update((tail.len() as u64).to_le_bytes());
        h.update(&tail);
    } else {
        h.update(0u64.to_le_bytes());
    }
    Ok(())
}

pub(super) fn hash_shared_lib_set(h: &mut sha2::Sha256, libs: &[WatchedFile]) -> Result<()> {
    use sha2::Digest;
    let mut unique: Vec<(&Path, BinStamp)> = Vec::new();
    for (_, canon, stamp) in libs {
        if unique.last().is_none_or(|(prev, _)| *prev != canon.as_path()) {
            unique.push((canon.as_path(), *stamp));
        }
    }
    h.update((unique.len() as u64).to_le_bytes());
    for (canon, stamp) in unique {
        hash_shared_lib_sample(h, canon, stamp)?;
    }
    Ok(())
}

pub(super) fn hash_compiler_binaries(
    cc_path: &Path,
    canon_cc: &Path,
    canon_cc1: Option<&Path>,
    canon_as: Option<&Path>,
    shared_libs: &[WatchedFile],
) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"cc-bin-v2\0");
    let argv0 = cc_path.file_name().map_or(&[][..], |n| n.as_encoded_bytes());
    h.update((argv0.len() as u64).to_le_bytes());
    h.update(argv0);
    let cc_bytes = std::fs::read(canon_cc).with_context(|| format!("reading {}", canon_cc.display()))?;
    h.update((cc_bytes.len() as u64).to_le_bytes());
    h.update(&cc_bytes);
    for opt_bin in [canon_cc1, canon_as] {
        if let Some(p) = opt_bin {
            let bytes = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
            h.update([1u8]);
            h.update((bytes.len() as u64).to_le_bytes());
            h.update(&bytes);
        } else {
            h.update([0u8]);
        }
    }
    hash_shared_lib_set(&mut h, shared_libs)?;
    Ok(format!("{:x}", h.finalize()))
}

pub(super) fn hash_linker_binaries(
    linker_path: &Path,
    canon_linker: &Path,
    canon_collect2: Option<&Path>,
    shared_libs: &[WatchedFile],
) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"ld-bin-v2\0");
    let argv0 = linker_path.file_name().map_or(&[][..], |n| n.as_encoded_bytes());
    h.update((argv0.len() as u64).to_le_bytes());
    h.update(argv0);
    let ld_bytes = std::fs::read(canon_linker).with_context(|| format!("reading {}", canon_linker.display()))?;
    h.update((ld_bytes.len() as u64).to_le_bytes());
    h.update(&ld_bytes);
    if let Some(p) = canon_collect2 {
        let bytes = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
        h.update([1u8]);
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(&bytes);
    } else {
        h.update([0u8]);
    }
    hash_shared_lib_set(&mut h, shared_libs)?;
    Ok(format!("{:x}", h.finalize()))
}

pub(super) fn parse_ld_trace_files(stdout: &[u8], stderr: &[u8]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for buf in [stdout, stderr] {
        for raw_line in String::from_utf8_lossy(buf).lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let cand = if let Some(rest) = line.strip_prefix('(')
                && let Some((inner, _)) = rest.split_once(')')
            {
                inner.trim()
            } else if let Some((_, rest)) = line.rsplit_once(" (")
                && let Some(inner) = rest.strip_suffix(')')
            {
                inner.trim()
            } else if line.starts_with('/')
                && line.ends_with(')')
                && let Some((arch, _)) = line.split_once('(')
            {
                arch.trim()
            } else {
                line
            };
            let p = Path::new(cand);
            if p.is_absolute() && p.is_file() {
                out.push(p.to_path_buf());
            }
        }
    }
    out
}

pub(super) fn hash_watched_files(domain: &[u8], files: &mut Vec<WatchedFile>) -> Result<String> {
    use sha2::{Digest, Sha256};
    files.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    files.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    let mut h = Sha256::new();
    h.update(domain);
    let mut last_canon: Option<&Path> = None;
    for (_, canon, _) in files.iter() {
        if last_canon == Some(canon.as_path()) {
            continue;
        }
        last_canon = Some(canon.as_path());
        let path_bytes = canon.as_os_str().as_encoded_bytes();
        h.update((path_bytes.len() as u64).to_le_bytes());
        h.update(path_bytes);
        let bytes = std::fs::read(canon).with_context(|| format!("reading {}", canon.display()))?;
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(&bytes);
    }
    Ok(format!("{:x}", h.finalize()))
}

pub(super) fn verify_watched_files(watched: &[WatchedFile]) -> bool {
    watched.iter().all(|(orig, canon, stamp)| {
        std::fs::canonicalize(orig).ok().as_ref() == Some(canon) && bin_stamp(canon).is_ok_and(|s| s == *stamp)
    })
}

pub(super) fn extract_linemarker_path(line: &str) -> Option<String> {
    let rest = line.strip_prefix("# ")?;
    let (num, after_num) = rest.split_once(' ')?;
    if num.is_empty() || !num.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut chars = after_num.strip_prefix('"')?.chars();
    let mut path = String::new();
    while let Some(c) = chars.next() {
        if c == '"' {
            return Some(path);
        }
        if c == '\\' {
            let esc = chars.next()?;
            path.push(esc);
        } else {
            path.push(c);
        }
    }
    None
}

pub(super) fn probe_sys_headers(
    cc_path: &Path,
    target_cpu: &str,
    probe_dir: &Path,
    deadline: Instant,
) -> Result<(String, Vec<WatchedFile>)> {
    let (mut hdr_file, hdr_path) = create_private_temp_file(probe_dir, "hdrs", ".c")?;
    let _rm_hdr = RemoveOnDrop(hdr_path.clone());
    hdr_file.write_all(SYS_HEADER_PROBE_C)?;
    drop(hdr_file);
    let canon_src = std::fs::canonicalize(&hdr_path)?;
    let mut watched = Vec::new();
    let mut record_header = |path_str: &str, opt_flag: &str| -> Result<()> {
        let dep = PathBuf::from(path_str);
        ensure!(
            dep.is_absolute() && dep.is_file(),
            "{} {opt_flag} -H/-E reported non-absolute or missing header: {path_str:?}",
            cc_path.display()
        );
        let canon =
            std::fs::canonicalize(&dep).with_context(|| format!("canonicalizing system header {}", dep.display()))?;
        if canon == canon_src {
            return Ok(());
        }
        let stamp = bin_stamp(&canon)?;
        watched.push((dep, canon, stamp));
        Ok(())
    };
    for opt_flag in ["-O2", "-O0"] {
        let mut cmd = clean_cc_cmd(cc_path);
        cmd.args([opt_flag, "-fPIC", &format!("-march={target_cpu}"), "-H", "-E", "-x", "c"]).arg(&hdr_path);
        let out = run_cmd_output_with_deadline(
            cmd,
            probe_dir,
            &format!("{} {opt_flag} -H -E system headers", cc_path.display()),
            deadline,
        )?;
        ensure!(
            out.status.success(),
            "{} {opt_flag} -H -E system header probe failed: {}",
            cc_path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
        for raw_line in String::from_utf8_lossy(&out.stderr).lines() {
            let trimmed = raw_line.trim_end_matches('\r');
            let rest = trimmed.trim_start_matches('.');
            if rest.len() < trimmed.len()
                && let Some(path_str) = rest.strip_prefix(' ')
            {
                record_header(path_str, opt_flag)?;
            }
        }
        for raw_line in String::from_utf8_lossy(&out.stdout).lines() {
            if let Some(path_str) = extract_linemarker_path(raw_line.trim_end_matches('\r'))
                && path_str.starts_with('/')
            {
                record_header(&path_str, opt_flag)?;
            }
        }
    }
    #[cfg(unix)]
    ensure!(!watched.is_empty(), "{} -H -E returned no system headers", cc_path.display());
    let digest = hash_watched_files(b"sys-headers-v3\0", &mut watched)?;
    Ok((digest, watched))
}

pub(super) fn probe_link_env_and_cet(
    cc_path: &Path,
    has_lld: bool,
    probe_dir: &Path,
    deadline: Instant,
) -> Result<(bool, String, Vec<WatchedFile>)> {
    let (mut c_file, c_path) = create_private_temp_file(probe_dir, "cet", ".c")?;
    let _rm_c = RemoveOnDrop(c_path.clone());
    c_file.write_all(LINK_PROBE_C)?;
    drop(c_file);

    struct LinkProbeOut {
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        so_bytes: Vec<u8>,
        canon_obj: PathBuf,
    }
    let compile_and_link = |with_cet: bool| -> Result<Option<LinkProbeOut>> {
        let (o_file, o_path) = create_private_temp_file(probe_dir, "cet", ".o")?;
        drop(o_file);
        let _rm_o = RemoveOnDrop(o_path.clone());
        let canon_o = std::fs::canonicalize(&o_path)?;
        let mut c_cmd = clean_cc_cmd(cc_path);
        c_cmd.args(["-c", "-O0", "-fPIC"]);
        if with_cet {
            c_cmd.arg("-fcf-protection=return");
        }
        c_cmd.arg(&c_path).arg("-o").arg(&o_path);
        let c_out =
            run_cmd_output_with_deadline(c_cmd, probe_dir, &format!("{} probe compile", cc_path.display()), deadline)?;
        if !c_out.status.success() {
            return Ok(None);
        }
        let (so_file, so_path) = create_private_temp_file(probe_dir, "cet", ".so")?;
        drop(so_file);
        let _rm_so = RemoveOnDrop(so_path.clone());
        let mut l_cmd = clean_cc_cmd(cc_path);
        l_cmd.args(["-shared", "-fPIC", "-Wl,-t"]);
        if with_cet {
            l_cmd.arg("-fcf-protection=return");
        }
        if has_lld {
            l_cmd.arg("-fuse-ld=lld");
        }
        l_cmd.arg(&o_path).arg("-o").arg(&so_path);
        #[cfg(target_os = "linux")]
        l_cmd.arg("-pthread");
        #[cfg(unix)]
        l_cmd.arg("-lm");
        let l_out =
            run_cmd_output_with_deadline(l_cmd, probe_dir, &format!("{} probe link", cc_path.display()), deadline)?;
        if !l_out.status.success() {
            return Ok(None);
        }
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        let so_bytes = std::fs::read(&so_path)?;
        Ok(Some(LinkProbeOut {
            stdout: l_out.stdout,
            stderr: l_out.stderr,
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            so_bytes,
            canon_obj: canon_o,
        }))
    };

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    let (has_cet_return, probe_out) = if let Some(out) = compile_and_link(true)?
        && elf64_has_x86_shstk_property(&out.so_bytes)
    {
        (true, out)
    } else {
        let out = compile_and_link(false)?.ok_or_else(|| anyhow!("{} -shared probe link failed", cc_path.display()))?;
        (false, out)
    };
    #[cfg(all(not(target_os = "linux"), target_arch = "x86_64"))]
    let (has_cet_return, probe_out) = if let Some(out) = compile_and_link(true)? {
        (true, out)
    } else {
        let out = compile_and_link(false)?.ok_or_else(|| anyhow!("{} -shared probe link failed", cc_path.display()))?;
        (false, out)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let (has_cet_return, probe_out) = {
        let out = compile_and_link(false)?.ok_or_else(|| anyhow!("{} -shared probe link failed", cc_path.display()))?;
        (false, out)
    };
    let LinkProbeOut { stdout, stderr, canon_obj: canon_o, .. } = probe_out;

    let canon_probe_dir = std::fs::canonicalize(probe_dir).ok();
    let mut watched = Vec::new();
    for path in parse_ld_trace_files(&stdout, &stderr) {
        let canon =
            std::fs::canonicalize(&path).with_context(|| format!("canonicalizing link input {}", path.display()))?;
        if canon == canon_o {
            continue;
        }
        if let Some(ref pd) = canon_probe_dir
            && canon.starts_with(pd)
        {
            continue;
        }
        let stamp = bin_stamp(&canon)?;
        watched.push((path, canon, stamp));
    }
    #[cfg(target_os = "linux")]
    ensure!(!watched.is_empty(), "{} -Wl,-t returned no link inputs", cc_path.display());
    let digest = hash_watched_files(b"link-env-v3\0", &mut watched)?;
    Ok((has_cet_return, digest, watched))
}

/// Computes the exact same C compiler, `cc1`, `as`, dynamic shared-library, driver-selected linker
/// and `collect2`, transitive system-header (`-O2` and `-O0`), and CRT/link-environment identity
/// as `wtime-aot`'s `cgen.rs` and `build.rs`.
pub(super) type CcIdentityKey = (
    &'static str,
    PathBuf,
    PathBuf,
    BinStamp,
    Option<WatchedFile>,
    Option<WatchedFile>,
    Option<WatchedFile>,
    String,
    String,
);
#[derive(Clone)]
pub(super) struct ResolvedCcIdentity {
    pub(super) cc_name: &'static str,
    pub(super) cc_path: PathBuf,
    pub(super) canon_cc: PathBuf,
    pub(super) stamp: BinStamp,
    pub(super) cc1_file: Option<WatchedFile>,
    pub(super) as_file: Option<WatchedFile>,
    pub(super) collect2_file: Option<WatchedFile>,
    pub(super) lld_key: Option<WatchedFile>,
    pub(super) has_lld: bool,
    pub(super) linker_path: PathBuf,
    pub(super) canon_linker: PathBuf,
    pub(super) linker_stamp: BinStamp,
    pub(super) cc_sha256: String,
    pub(super) linker_sha256: String,
    pub(super) sys_headers_sha256: String,
    pub(super) link_env_sha256: String,
    pub(super) watched_files: Vec<WatchedFile>,
    pub(super) cc_ver: String,
    pub(super) cc_target: String,
    pub(super) has_cet_return: bool,
    pub(super) has_stack_clash: bool,
}

pub(super) fn c_compiler_identity(
    target_cpu: &str,
    canon_cache: &Path,
    deadline: Instant,
) -> Result<ResolvedCcIdentity> {
    use sha2::{Digest, Sha256};
    static CACHE: OnceLock<Mutex<std::collections::HashMap<CcIdentityKey, ResolvedCcIdentity>>> = OnceLock::new();
    let host_fp = host_cpu_fingerprint().to_string();
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let probe_dir = ensure_private_cache_dir(&canon_cache.join("cc-probe"))?;
    let lld_key = find_in_path("ld.lld")
        .and_then(|p| std::fs::canonicalize(&p).ok().map(|c| (p, c)))
        .and_then(|(p, c)| bin_stamp(&c).ok().map(|s| (p, c, s)));
    let ld_key = find_in_path("ld")
        .and_then(|p| std::fs::canonicalize(&p).ok().map(|c| (p, c)))
        .and_then(|(p, c)| bin_stamp(&c).ok().map(|s| (p, c, s)));
    let as_key = find_in_path("as")
        .and_then(|p| std::fs::canonicalize(&p).ok().map(|c| (p, c)))
        .and_then(|(p, c)| bin_stamp(&c).ok().map(|s| (p, c, s)));
    let is_sha64 = |s: &str| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit());
    for cc in ["clang", "cc", "gcc"] {
        let Some(p) = find_in_path(cc) else { continue };
        let Ok(canon_cc) = std::fs::canonicalize(&p) else { continue };
        let Ok(stamp) = bin_stamp(&canon_cc) else { continue };
        let mem_key = (
            cc,
            p.clone(),
            canon_cc.clone(),
            stamp,
            lld_key.clone(),
            ld_key.clone(),
            as_key.clone(),
            host_fp.clone(),
            target_cpu.to_string(),
        );
        if let Some(hit) = cache.lock().unwrap_or_else(|p| p.into_inner()).get(&mem_key).cloned()
            && std::fs::canonicalize(&hit.linker_path).ok().as_ref() == Some(&hit.canon_linker)
            && bin_stamp(&hit.canon_linker).is_ok_and(|s| s == hit.linker_stamp)
            && verify_watched_files(&hit.watched_files)
        {
            return Ok(hit);
        }

        let probe_key_line = format!(
            "v7|{cc}|{}|{}|{:?}|{:?}|{:?}|{:?}|{}|{}",
            p.display(),
            canon_cc.display(),
            stamp,
            lld_key,
            ld_key,
            as_key,
            host_fp,
            target_cpu
        );
        let probe_file = probe_dir.join(format!("{}.txt", wasm_sha256_hex(probe_key_line.as_bytes())));
        if let Ok(meta) = std::fs::symlink_metadata(&probe_file)
            && !meta.file_type().is_symlink()
            && meta.file_type().is_file()
        {
            let owner_and_mode_ok = {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::{MetadataExt, PermissionsExt};
                    (meta.permissions().mode() & 0o022) == 0
                        && current_euid().ok().is_some_and(|euid| meta.uid() == euid)
                }
                #[cfg(not(unix))]
                {
                    true
                }
            };
            if owner_and_mode_ok && let Ok(text) = std::fs::read_to_string(&probe_file) {
                let mut lines = text.lines();
                if let (
                    Some(k),
                    Some(ver),
                    Some(digest),
                    Some(cet),
                    Some(sc),
                    Some(cc_sha),
                    Some(linker_orig_str),
                    Some(linker_canon_str),
                    Some(linker_stamp_str),
                    Some(linker_sha),
                    Some(has_lld_str),
                    Some(sys_hdrs_sha),
                    Some(link_env_sha),
                    Some(cc1_line),
                    Some(as_line),
                    Some(collect2_line),
                    Some(n_watched_str),
                ) = (
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                    lines.next(),
                ) && k == probe_key_line
                    && is_sha64(digest)
                    && is_sha64(cc_sha)
                    && is_sha64(linker_sha)
                    && is_sha64(sys_hdrs_sha)
                    && is_sha64(link_env_sha)
                    && (cet == "0" || cet == "1")
                    && (sc == "0" || sc == "1")
                    && (has_lld_str == "0" || has_lld_str == "1")
                    && !linker_orig_str.is_empty()
                    && !linker_canon_str.is_empty()
                    && let Ok(n_watched) = n_watched_str.parse::<usize>()
                    && n_watched <= 4096
                    && (!cfg!(unix) || n_watched > 0)
                {
                    let linker_path = PathBuf::from(linker_orig_str);
                    let canon_linker = PathBuf::from(linker_canon_str);
                    let linker_ok = (std::fs::canonicalize(&linker_path).ok().as_ref() == Some(&canon_linker))
                        .then(|| bin_stamp(&canon_linker).ok().filter(|s| format!("{s:?}") == linker_stamp_str))
                        .flatten();
                    let parse_watched_line = |line: &str| -> Option<WatchedFile> {
                        let mut parts = line.split('\t');
                        let (orig_s, canon_s, stamp_s) = (parts.next()?, parts.next()?, parts.next()?);
                        if parts.next().is_some() || orig_s.is_empty() || canon_s.is_empty() {
                            return None;
                        }
                        let orig = PathBuf::from(orig_s);
                        let canon = PathBuf::from(canon_s);
                        if std::fs::canonicalize(&orig).ok().as_ref() != Some(&canon) {
                            return None;
                        }
                        let cur_stamp = bin_stamp(&canon).ok()?;
                        if format!("{cur_stamp:?}") != stamp_s {
                            return None;
                        }
                        Some((orig, canon, cur_stamp))
                    };
                    let parse_opt = |line: &str| -> Option<Option<WatchedFile>> {
                        if line == "-" { Some(None) } else { parse_watched_line(line).map(Some) }
                    };
                    if let (Some(cur_linker_stamp), Some(cc1_file), Some(as_file), Some(collect2_file)) =
                        (linker_ok, parse_opt(cc1_line), parse_opt(as_line), parse_opt(collect2_line))
                    {
                        let mut watched_files = Vec::with_capacity(n_watched);
                        let mut all_ok = true;
                        for _ in 0..n_watched {
                            if let Some(w) = lines.next().and_then(parse_watched_line) {
                                watched_files.push(w);
                            } else {
                                all_ok = false;
                                break;
                            }
                        }
                        if all_ok && lines.next().is_none() {
                            let resolved = ResolvedCcIdentity {
                                cc_name: cc,
                                cc_path: p,
                                canon_cc,
                                stamp,
                                cc1_file,
                                as_file,
                                collect2_file,
                                lld_key: lld_key.clone(),
                                has_lld: has_lld_str == "1",
                                linker_path,
                                canon_linker,
                                linker_stamp: cur_linker_stamp,
                                cc_sha256: cc_sha.to_string(),
                                linker_sha256: linker_sha.to_string(),
                                sys_headers_sha256: sys_hdrs_sha.to_string(),
                                link_env_sha256: link_env_sha.to_string(),
                                watched_files,
                                cc_ver: ver.to_string(),
                                cc_target: digest.to_string(),
                                has_cet_return: cet == "1",
                                has_stack_clash: sc == "1",
                            };
                            cache.lock().unwrap_or_else(|p| p.into_inner()).insert(mem_key, resolved.clone());
                            return Ok(resolved);
                        }
                    }
                }
            }
        }

        let mut ver_cmd = clean_cc_cmd(&p);
        ver_cmd.arg("--version");
        let ver_out = run_cmd_output_with_deadline(ver_cmd, &probe_dir, "C compiler --version probe", deadline)?;
        if !ver_out.status.success() {
            continue;
        }
        let cc_ver = String::from_utf8_lossy(&ver_out.stdout).lines().next().unwrap_or(cc).trim().to_string();

        let mut target_cmd = clean_cc_cmd(&p);
        target_cmd.args([&format!("-march={target_cpu}"), "-dM", "-E", "-x", "c", "-"]);
        let out = run_cmd_output_with_deadline(
            target_cmd,
            &probe_dir,
            &format!("probing {} -march={target_cpu}", p.display()),
            deadline,
        )?;
        ensure!(
            out.status.success(),
            "{} -march={target_cpu} probe failed: {}",
            p.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
        let mut lines: Vec<&str> =
            std::str::from_utf8(&out.stdout).unwrap_or("").lines().filter(|l| l.starts_with("#define __")).collect();
        lines.sort_unstable();
        let mut h = Sha256::new();
        h.update(b"cc-target-v1");
        h.update(target_cpu.as_bytes());
        for l in lines {
            h.update(l.as_bytes());
            h.update(b"\n");
        }
        let cc_target = format!("{:x}", h.finalize());

        let has_lld = if let Some((ref lld_prog, _, _)) = lld_key {
            let mut lld_cmd = clean_cc_cmd(lld_prog);
            lld_cmd.arg("--version");
            run_cmd_output_with_deadline(lld_cmd, &probe_dir, "probing ld.lld --version", deadline)?.status.success()
        } else {
            false
        };
        let (linker_path, canon_linker, linker_stamp) = resolve_driver_linker(&p, has_lld, &probe_dir, deadline)?;
        let cc1_file = resolve_driver_prog(&p, "cc1", false, &probe_dir, deadline)?;
        let as_file = resolve_driver_prog(&p, "as", true, &probe_dir, deadline)?;
        let collect2_file = resolve_driver_prog(&p, "collect2", false, &probe_dir, deadline)?;

        let mut cc_libs = probe_dynamic_shared_libs(&canon_cc, &probe_dir, deadline)?;
        if let Some((_, ref c1, _)) = cc1_file {
            cc_libs.extend(probe_dynamic_shared_libs(c1, &probe_dir, deadline)?);
        }
        if let Some((_, ref a, _)) = as_file {
            cc_libs.extend(probe_dynamic_shared_libs(a, &probe_dir, deadline)?);
        }
        cc_libs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        cc_libs.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        let mut ld_libs = probe_dynamic_shared_libs(&canon_linker, &probe_dir, deadline)?;
        if let Some((_, ref c2, _)) = collect2_file {
            ld_libs.extend(probe_dynamic_shared_libs(c2, &probe_dir, deadline)?);
        }
        ld_libs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        ld_libs.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        let (sys_headers_sha256, header_files) = probe_sys_headers(&p, target_cpu, &probe_dir, deadline)?;
        let (has_cet_return, link_env_sha256, link_files) = probe_link_env_and_cet(&p, has_lld, &probe_dir, deadline)?;
        let cc_sha256 = hash_compiler_binaries(
            &p,
            &canon_cc,
            cc1_file.as_ref().map(|(_, c, _)| c.as_path()),
            as_file.as_ref().map(|(_, c, _)| c.as_path()),
            &cc_libs,
        )?;
        let linker_sha256 = hash_linker_binaries(
            &linker_path,
            &canon_linker,
            collect2_file.as_ref().map(|(_, c, _)| c.as_path()),
            &ld_libs,
        )?;

        let has_stack_clash = {
            let mut sc_cmd = clean_cc_cmd(&p);
            sc_cmd.args(["-x", "c", "-c", "-fstack-clash-protection", "-o", "/dev/null", "-"]);
            run_cmd_output_with_deadline(sc_cmd, &probe_dir, "probing -fstack-clash-protection", deadline)?
                .status
                .success()
        };

        let mut watched_files = header_files;
        watched_files.extend(link_files);
        watched_files.extend(cc_libs);
        watched_files.extend(ld_libs);
        #[cfg(target_os = "linux")]
        {
            let ld_cache = Path::new("/etc/ld.so.cache");
            if ld_cache.is_file()
                && let Ok(canon) = std::fs::canonicalize(ld_cache)
                && let Ok(st) = bin_stamp(&canon)
            {
                watched_files.push((ld_cache.to_path_buf(), canon, st));
            }
        }
        if let Some(ref f) = cc1_file {
            watched_files.push(f.clone());
        }
        if let Some(ref f) = as_file {
            watched_files.push(f.clone());
        }
        if let Some(ref f) = collect2_file {
            watched_files.push(f.clone());
        }
        watched_files.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        watched_files.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        fn clean_path_str(path: &Path) -> Option<&str> {
            let s = path.to_str()?;
            (!s.is_empty() && !s.contains('|') && !s.contains('\t') && !s.contains('\n') && !s.contains('\r'))
                .then_some(s)
        }
        let clean_opt_watched = |opt: &Option<WatchedFile>| -> bool {
            opt.as_ref().is_none_or(|(o, c, _)| clean_path_str(o).is_some() && clean_path_str(c).is_some())
        };
        let format_watched = |(orig, canon, st): &WatchedFile| -> Option<String> {
            Some(format!("{}\t{}\t{st:?}", clean_path_str(orig)?, clean_path_str(canon)?))
        };
        let format_opt = |opt: &Option<WatchedFile>| -> Option<String> {
            match opt {
                None => Some("-".to_string()),
                Some(w) => format_watched(w),
            }
        };
        if watched_files.len() <= 4096
            && (!cfg!(unix) || !watched_files.is_empty())
            && clean_path_str(&p).is_some()
            && clean_path_str(&canon_cc).is_some()
            && clean_opt_watched(&lld_key)
            && clean_opt_watched(&ld_key)
            && clean_opt_watched(&as_key)
            && let (Some(l_orig), Some(l_canon)) = (clean_path_str(&linker_path), clean_path_str(&canon_linker))
            && let (Some(cc1_str), Some(as_str), Some(collect2_str)) =
                (format_opt(&cc1_file), format_opt(&as_file), format_opt(&collect2_file))
            && let Some(watched_lines) = watched_files.iter().map(format_watched).collect::<Option<Vec<String>>>()
        {
            let mut body = format!(
                "{probe_key_line}\n{cc_ver}\n{cc_target}\n{}\n{}\n{cc_sha256}\n{l_orig}\n{l_canon}\n{linker_stamp:?}\n{linker_sha256}\n{}\n{sys_headers_sha256}\n{link_env_sha256}\n{cc1_str}\n{as_str}\n{collect2_str}\n{}\n",
                u8::from(has_cet_return),
                u8::from(has_stack_clash),
                u8::from(has_lld),
                watched_lines.len(),
            );
            for line in watched_lines {
                body.push_str(&line);
                body.push('\n');
            }
            if let Ok((mut tmp_file, tmp)) = create_private_temp_file(&probe_dir, "probe", ".tmp") {
                let ok = tmp_file.write_all(body.as_bytes()).is_ok();
                drop(tmp_file);
                if !ok || std::fs::rename(&tmp, &probe_file).is_err() {
                    let _ = std::fs::remove_file(&tmp);
                }
            }
        }

        let resolved = ResolvedCcIdentity {
            cc_name: cc,
            cc_path: p,
            canon_cc,
            stamp,
            cc1_file,
            as_file,
            collect2_file,
            lld_key: lld_key.clone(),
            has_lld,
            linker_path,
            canon_linker,
            linker_stamp,
            cc_sha256,
            linker_sha256,
            sys_headers_sha256,
            link_env_sha256,
            watched_files,
            cc_ver,
            cc_target,
            has_cet_return,
            has_stack_clash,
        };
        cache.lock().unwrap_or_else(|p| p.into_inner()).insert(mem_key, resolved.clone());
        return Ok(resolved);
    }
    bail!("no C compiler found (install clang or gcc)")
}

pub(super) fn sanitize_key_part(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' }).collect()
}

pub(super) struct RemoveOnDrop(pub(super) PathBuf);
impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub(super) type RustcProxyKey = (PathBuf, BinStamp, String);
#[derive(Clone)]
pub(super) struct ResolvedRustcIdentity {
    pub(super) proxy_path: PathBuf,
    pub(super) canon_real: PathBuf,
    pub(super) real_stamp: BinStamp,
    pub(super) fp: String,
}

/// Probes the selected Rust toolchain preserving proxy `argv[0]` (so `rustup` proxies like
/// `/usr/bin/rustc -> /usr/bin/rustup` execute `rustc` rather than `rustup`), locates and stamps
/// the actual `<sysroot>/bin/rustc` executable, and fails loudly if `RUSTUP_TOOLCHAIN` fails
/// (matching `wtime-aot/src/build.rs`).
pub(super) fn rustc_identity(
    toolchain: &str,
    scratch_dir: &Path,
    deadline: Instant,
    force_refresh: bool,
) -> Result<ResolvedRustcIdentity> {
    static CACHE: OnceLock<Mutex<std::collections::HashMap<RustcProxyKey, ResolvedRustcIdentity>>> = OnceLock::new();
    let proxy_path = find_in_path("rustc").ok_or_else(|| anyhow!("no rustc found on PATH for --backend rust"))?;
    let canon_proxy = std::fs::canonicalize(&proxy_path)
        .with_context(|| format!("canonicalizing rustc proxy {}", proxy_path.display()))?;
    let proxy_stamp = bin_stamp(&canon_proxy)?;
    let key = (proxy_path.clone(), proxy_stamp, toolchain.to_string());
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if !force_refresh
        && let Some(hit) = cache.lock().unwrap_or_else(|p| p.into_inner()).get(&key).cloned()
        && bin_stamp(&hit.canon_real).is_ok_and(|s| s == hit.real_stamp)
    {
        return Ok(hit);
    }

    let clean_rustc = || {
        // Execute `proxy_path` directly (NOT canonicalized) so rustup preserves `argv[0] = "rustc"`.
        let mut cmd = Command::new(&proxy_path);
        cmd.env_clear();
        if let Some(path) = sanitized_path_env() {
            cmd.env("PATH", path);
        }
        for var in ["HOME", "RUSTUP_HOME", "CARGO_HOME"] {
            if let Some(v) = std::env::var_os(var) {
                cmd.env(var, v);
            }
        }
        cmd.env("RUSTUP_TOOLCHAIN", toolchain);
        cmd
    };

    let mut sysroot_cmd = clean_rustc();
    sysroot_cmd.args(["--print", "sysroot"]);
    let sysroot_out = run_cmd_output_with_deadline(sysroot_cmd, scratch_dir, "rustc --print sysroot", deadline)?;
    ensure!(
        sysroot_out.status.success(),
        "rustc --print sysroot failed for toolchain {toolchain}: {}",
        String::from_utf8_lossy(&sysroot_out.stderr).trim()
    );
    let sysroot = PathBuf::from(String::from_utf8_lossy(&sysroot_out.stdout).trim());
    let real_rustc = sysroot.join("bin").join(if cfg!(windows) { "rustc.exe" } else { "rustc" });
    let canon_real = if is_executable_file(&real_rustc) {
        std::fs::canonicalize(&real_rustc)
            .with_context(|| format!("canonicalizing toolchain rustc {}", real_rustc.display()))?
    } else {
        canon_proxy
    };
    let real_stamp = bin_stamp(&canon_real)?;

    let mut ver_cmd = clean_rustc();
    ver_cmd.arg("-vV");
    let ver_out = run_cmd_output_with_deadline(ver_cmd, scratch_dir, "rustc -vV probe", deadline)?;
    ensure!(
        ver_out.status.success(),
        "rustc -vV failed for toolchain {toolchain}: {}",
        String::from_utf8_lossy(&ver_out.stderr).trim()
    );
    let fp = wasm_sha256_hex(&ver_out.stdout)[..16].to_string();
    let id = ResolvedRustcIdentity { proxy_path, canon_real, real_stamp, fp };
    cache.lock().unwrap_or_else(|p| p.into_inner()).insert(key, id.clone());
    Ok(id)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", target_pointer_width = "64"))]
pub(super) enum ShstkStatus {
    ConfirmedOff,
    Enabled,
    Unknown(String),
}

/// Inspects the calling Linux thread's CET hardware shadow-stack (`shstk`) state via
/// `/proc/thread-self/status` `x86_Thread_features:` (zero unsafe).
/// Distinguishes confirmed-off from unavailable/indeterminate status so strict CET loaders never
/// fail opaquely behind a false "off" assumption.
#[cfg(all(target_os = "linux", target_arch = "x86_64", target_pointer_width = "64"))]
pub(super) fn linux_shstk_status() -> ShstkStatus {
    let status = match read_thread_proc_status() {
        Ok(s) => s,
        Err(e) => return ShstkStatus::Unknown(format!("cannot read /proc/thread-self/status: {e}")),
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("x86_Thread_features:") {
            return if rest.split_whitespace().any(|tok| tok == "shstk") {
                ShstkStatus::Enabled
            } else {
                ShstkStatus::ConfirmedOff
            };
        }
    }
    // Pre-6.6 kernels omit `x86_Thread_features:`; confirm via `/proc/cpuinfo` that the kernel/CPU
    // does not expose user CET shadow stacks (`user_shstk` / `shstk`), otherwise fail closed.
    let cpuinfo = match std::fs::read_to_string("/proc/cpuinfo") {
        Ok(s) => s,
        Err(e) => {
            return ShstkStatus::Unknown(format!(
                "missing x86_Thread_features in proc status and cannot read /proc/cpuinfo: {e}"
            ));
        }
    };
    let mut saw_flags = false;
    for line in cpuinfo.lines() {
        if let Some(rest) = line.strip_prefix("flags")
            && let Some((_, flags)) = rest.split_once(':')
        {
            saw_flags = true;
            if flags.split_whitespace().any(|f| f == "user_shstk" || f == "shstk") {
                return ShstkStatus::Unknown(
                    "CPU/kernel advertises CET shstk but /proc/thread-self/status lacks x86_Thread_features".into(),
                );
            }
        }
    }
    if saw_flags {
        ShstkStatus::ConfirmedOff
    } else {
        ShstkStatus::Unknown("unable to confirm CET shstk status from procfs".into())
    }
}

/// Validates `--target-cpu` syntax independently of the host OS/architecture so an invalid CPU
/// name always fails loudly as a configuration error (never triggering platform/ISA fallback).
pub(super) fn parse_x64_target_cpu_syntax(target_cpu: &str) -> Result<Option<&'static str>> {
    match target_cpu {
        "native" => Ok(None),
        "x86-64-v2" | "generic" => Ok(Some("x64-v2")),
        "x86-64-v3" | "x86-64-v4" | "haswell" | "broadwell" | "skylake" | "skylake-avx512" | "cascadelake"
        | "cooperlake" | "cannonlake" | "icelake-client" | "icelake-server" | "tigerlake" | "rocketlake"
        | "alderlake" | "raptorlake" | "sapphirerapids" | "emeraldrapids" | "graniterapids" => Ok(Some("x64-v3")),
        other
            if other.strip_prefix("znver").is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())) =>
        {
            Ok(Some("x64-v3"))
        }
        other => bail!(
            "unsupported PEN_WTIME_AOT_TARGET_CPU={other:?} (expected native, x86-64-v2, generic, x86-64-v3, x86-64-v4, znver<N>, or haswell+)"
        ),
    }
}

/// Canonical internal cache profile tag matching `wtime-aot --backend x64 --target-cpu`,
/// validating `target_cpu` syntax FIRST (unconditionally across all platforms), then checking host
/// platform (`Linux x86-64 LP64`), calling-thread CET `SHSTK` status (failing closed when enabled
/// or indeterminate), and CPU capability (`SSE4.1 + POPCNT`, plus `ABM/LZCNT + BMI1` for `x64-v3`)
/// BEFORE cache lookup so unsupported hosts fail with a typed `X64FallbackError` while `wasm` &
/// `cache` are still available for default-path fallback (and bad CPU names always fail loudly).
pub(super) fn resolved_x64_isa_tag(target_cpu: &str) -> Result<&'static str> {
    let explicit_tag = parse_x64_target_cpu_syntax(target_cpu)?;
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64", target_pointer_width = "64")))]
    {
        let _ = explicit_tag;
        Err(anyhow::Error::new(X64FallbackError(X64FallbackReason::UnsupportedPlatform(
            "wtime-aot --backend x64 is only supported on Linux x86_64 LP64 hosts".into(),
        ))))
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64", target_pointer_width = "64"))]
    {
        match linux_shstk_status() {
            ShstkStatus::ConfirmedOff => {}
            ShstkStatus::Enabled => {
                return Err(anyhow::Error::new(X64FallbackError(X64FallbackReason::UnsupportedIsa(
                    "calling thread has Linux CET hardware shadow stack (SHSTK) enabled".into(),
                ))));
            }
            ShstkStatus::Unknown(why) => {
                return Err(anyhow::Error::new(X64FallbackError(X64FallbackReason::UnsupportedIsa(why))));
            }
        }
        if !std::is_x86_feature_detected!("sse4.1") || !std::is_x86_feature_detected!("popcnt") {
            return Err(anyhow::Error::new(X64FallbackError(X64FallbackReason::UnsupportedIsa(
                "host CPU lacks baseline SSE4.1 + POPCNT required by wtime-aot --backend x64".into(),
            ))));
        }
        let tag = match explicit_tag {
            Some(t) => t,
            None => {
                if std::is_x86_feature_detected!("lzcnt") && std::is_x86_feature_detected!("bmi1") {
                    "x64-v3"
                } else {
                    "x64-v2"
                }
            }
        };
        if tag == "x64-v3" && (!std::is_x86_feature_detected!("lzcnt") || !std::is_x86_feature_detected!("bmi1")) {
            return Err(anyhow::Error::new(X64FallbackError(X64FallbackReason::UnsupportedIsa(format!(
                "host CPU lacks ABM/LZCNT + BMI1 required by PEN_WTIME_AOT_TARGET_CPU={target_cpu:?}"
            )))));
        }
        Ok(tag)
    }
}

pub(super) fn parse_aarch64_target_cpu_syntax(target_cpu: &str) -> Result<&'static str> {
    match target_cpu {
        "native" | "generic" | "armv8-a" | "armv8.1-a" | "armv8.2-a" | "armv8.3-a" | "armv8.4-a" | "armv8.5-a"
        | "armv8.6-a" | "armv9-a" => Ok("aarch64-v8a"),
        other
            if other.starts_with("neoverse-")
                || other.starts_with("cortex-")
                || other.starts_with("apple-")
                || other.starts_with("graviton") =>
        {
            Ok("aarch64-v8a")
        }
        other => bail!(
            "unsupported PEN_WTIME_AOT_TARGET_CPU={other:?} for aarch64 (expected native, generic, armv8-a, neoverse-*, cortex-*, apple-*, or graviton*)"
        ),
    }
}

pub(super) fn resolved_aarch64_isa_tag(target_cpu: &str) -> Result<&'static str> {
    let tag = parse_aarch64_target_cpu_syntax(target_cpu)?;
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64", target_pointer_width = "64")))]
    {
        let _ = tag;
        Err(anyhow::Error::new(X64FallbackError(X64FallbackReason::UnsupportedPlatform(
            "wtime-aot --backend aarch64 is only supported on Linux aarch64 LP64 hosts".into(),
        ))))
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64", target_pointer_width = "64"))]
    {
        Ok(tag)
    }
}
