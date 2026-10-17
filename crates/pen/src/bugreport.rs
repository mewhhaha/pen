//! Bug reports (`.penbug`): what a player saw go wrong, in one file. Ctrl+Shift+B
//! in a window (or a script's `report[=FILE]` step) writes one: the last
//! [`SECONDS`] of input as a recording that replays them, the game's state
//! now, the log the session kept and what crash reports say of the build,
//! the system and the GPU. Windows keep a history that long for it.
//!
//! A report is a ustar archive (`tar -xf` opens it) of `report.txt`,
//! `log.txt`, `recording.penrec` and `state.bin` (a quicksave). `pen report
//! FILE` says what it holds (`--out DIR` unpacks it), and `pen --replay FILE`
//! (`pen run --replay FILE` in a checkout) plays its recording. Reports go to
//! `reports/` in the data directory unless a file is named; it keeps the
//! latest [`KEEP`].
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

use crate::recording;
use crate::snapshot::{self, SavedState};
use crate::timeline::{Recorded, Recording};

/// How much game time of input a report keeps.
pub const SECONDS: f64 = 30.0;
/// Reports `reports/` keeps: the latest, older ones go.
const KEEP: usize = 20;

const BLOCK: usize = 512;
/// The archive's entries, in the order written.
const REPORT: &str = "report.txt";
const LOG: &str = "log.txt";
const RECORDING: &str = "recording.penrec";
const STATE: &str = "state.bin";

/// A number in a ustar header field `width` bytes wide: octal digits, then NUL.
fn octal(field: &mut [u8], value: u64) {
    let digits = format!("{value:0width$o}", width = field.len() - 1);
    field[..digits.len()].copy_from_slice(digits.as_bytes());
}

/// `entries` as a ustar archive.
fn archive(entries: &[(&str, Vec<u8>)], mtime: u64) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, data) in entries {
        let mut header = [0u8; BLOCK];
        header[..name.len()].copy_from_slice(name.as_bytes());
        octal(&mut header[100..108], 0o644);
        octal(&mut header[108..116], 0);
        octal(&mut header[116..124], 0);
        octal(&mut header[124..136], data.len() as u64);
        octal(&mut header[136..148], mtime);
        header[156] = b'0';
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        // The checksum counts its own field as spaces.
        header[148..156].fill(b' ');
        let sum: u64 = header.iter().map(|&byte| u64::from(byte)).sum();
        octal(&mut header[148..155], sum);
        out.extend_from_slice(&header);
        out.extend_from_slice(data);
        out.resize(out.len().next_multiple_of(BLOCK), 0);
    }
    out.resize(out.len() + 2 * BLOCK, 0);
    out
}

/// The entries of a ustar archive, as [`archive`] writes them.
fn entries(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
    let mut found = Vec::new();
    let mut at = 0;
    while let Some(header) = bytes.get(at..at + BLOCK) {
        if header.iter().all(|&byte| byte == 0) {
            return Ok(found);
        }
        ensure!(&header[257..262] == b"ustar", "not a pen bug report (a ustar archive)");
        let name = String::from_utf8_lossy(header[..100].split(|&byte| byte == 0).next().unwrap_or_default());
        let size = std::str::from_utf8(&header[124..136])?.trim_matches(['\0', ' ']);
        let size = usize::from_str_radix(size, 8).context("a bug report entry's size")?;
        let data = bytes.get(at + BLOCK..at + BLOCK + size).context("the bug report ends early")?;
        found.push((name.into_owned(), data.to_vec()));
        at += BLOCK + size.next_multiple_of(BLOCK);
    }
    bail!("the bug report ends early")
}

/// Whether `bytes` are a bug report: an archive whose first entry is its
/// report.
pub fn is_report(bytes: &[u8]) -> bool {
    bytes.len() >= BLOCK && &bytes[257..262] == b"ustar" && bytes.starts_with(REPORT.as_bytes())
}

/// The recording a bug report holds, as `.penrec` bytes.
pub fn recording_bytes(bytes: &[u8]) -> Result<Vec<u8>> {
    entries(bytes)?
        .into_iter()
        .find_map(|(name, data)| (name == RECORDING).then_some(data))
        .context("this bug report holds no recording (its session kept no history)")
}

/// What a session tells a report.
pub struct Capture<'a> {
    /// The history: its earliest moment and every frame since.
    pub recording: Option<Recording>,
    pub state: SavedState,
    /// The directory of the game's `game.wasm`.
    pub game: &'a Path,
    /// The frame the session is at.
    pub position: u64,
}

/// The report's words: when, the game and its build, the recording, then
/// what crash reports say of the build, the system and the GPU.
fn summary(capture: &Capture, log_lines: usize) -> String {
    let mut text = String::new();
    let identity = crate::identity();
    let _ = writeln!(text, "{} ({}) bug report, {}", identity.name, identity.title, crate::crash::now());
    let game = std::fs::read(capture.game.join("game.wasm"))
        .map_or_else(|_| "unknown".to_owned(), |wasm| crate::bench::fingerprint(&wasm));
    let _ = writeln!(text, "game.wasm {game}, state schema {}, at frame {}", capture.state.schema, capture.position);
    match &capture.recording {
        Some(recording) => {
            let seconds: f64 = recording.frames.iter().map(Recorded::elapsed).sum();
            let _ = writeln!(
                text,
                "{RECORDING}: the last {} frames ({seconds:.1} s of game time); replay with `pen --replay FILE`",
                recording.frames.len()
            );
        }
        None => {
            let _ = writeln!(text, "no recording: the session kept no history");
        }
    }
    let _ = writeln!(text, "{STATE}: the state at frame {}; {LOG}: {log_lines} lines", capture.position);
    text.push_str(&crate::crash::about());
    text
}

/// Writes a report to `path`, or to `reports/` in `data` when None; returns
/// where.
pub fn write(capture: &Capture, path: Option<PathBuf>, data: &Path) -> Result<PathBuf> {
    let directory = data.join("reports");
    let path = match path {
        Some(path) => path,
        None => {
            let stamp = crate::crash::now().replace([':', '-'], "");
            directory.join(format!("report-{stamp}-{}.penbug", std::process::id()))
        }
    };
    let log = crate::devlog::recent();
    let mut contents =
        vec![(REPORT, summary(capture, log.len()).into_bytes()), (LOG, (log.join("\n") + "\n").into_bytes())];
    if let Some(recording) = &capture.recording {
        contents.push((RECORDING, recording::encode(recording)));
    }
    contents.push((STATE, snapshot::encode(&capture.state)?));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mtime = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    std::fs::write(&path, archive(&contents, mtime)).with_context(|| format!("writing {}", path.display()))?;
    prune(&directory);
    Ok(path)
}

/// Deletes all but the latest [`KEEP`] reports in `directory`: their names
/// sort by when they were written.
fn prune(directory: &Path) {
    let mut reports: Vec<PathBuf> = std::fs::read_dir(directory)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "penbug"))
        .collect();
    reports.sort();
    for old in &reports[..reports.len().saturating_sub(KEEP)] {
        let _ = std::fs::remove_file(old);
    }
}

/// `pen report FILE`: what a report holds; with `out`, unpacks it there.
pub fn open(path: &Path, out: Option<&Path>) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(is_report(&bytes), "{} is not a pen bug report", path.display());
    let entries = entries(&bytes)?;
    let text = |name: &str| {
        entries.iter().find(|(entry, _)| entry == name).map(|(_, data)| String::from_utf8_lossy(data).into_owned())
    };
    println!("{}", text(REPORT).unwrap_or_default().trim_end());
    let log = text(LOG).unwrap_or_default();
    let lines: Vec<&str> = log.lines().collect();
    println!("\nthe log's last {} of {} lines:", lines.len().min(40), lines.len());
    for line in &lines[lines.len().saturating_sub(40)..] {
        println!("  {line}");
    }
    if let Some((_, data)) = entries.iter().find(|(name, _)| name == RECORDING) {
        let recording = recording::decode(data)?;
        println!(
            "\nits recording: {} frames from a moment of state schema {}: `pen --replay {}` plays it (`pen run \
             --replay` in the game's checkout)",
            recording.frames.len(),
            recording.start.saved.schema,
            path.display()
        );
    }
    if let Some(out) = out {
        std::fs::create_dir_all(out)?;
        for (name, data) in &entries {
            std::fs::write(out.join(name), data)?;
        }
        println!("unpacked into {}", out.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Moment;
    use pen_runtime::game::Edits;

    #[test]
    fn archives_are_ustar_and_read_back() {
        let entries_in = [("report.txt", b"words".to_vec()), ("state.bin", vec![7; 1500]), ("empty", Vec::new())];
        let bytes = archive(&entries_in, 1_791_675_099);
        assert_eq!(bytes.len() % BLOCK, 0);
        let read = entries(&bytes).unwrap();
        assert_eq!(read.len(), 3);
        for ((name, data), (read_name, read_data)) in entries_in.iter().zip(&read) {
            assert_eq!((*name, data), (read_name.as_str(), read_data));
        }
        assert!(is_report(&bytes));
        // The checksum tar computes: the header's bytes, its own field as spaces.
        let mut header = bytes[..BLOCK].to_vec();
        let stored = usize::from_str_radix(std::str::from_utf8(&header[148..154]).unwrap(), 8).unwrap();
        header[148..156].fill(b' ');
        assert_eq!(stored, header.iter().map(|&byte| usize::from(byte)).sum::<usize>());
        assert!(entries(&bytes[..BLOCK + 10]).is_err());
        assert!(!is_report(b"PENREC\0\0"));
    }

    #[test]
    fn reports_hold_the_recording_the_state_and_the_log() {
        let directory = std::env::temp_dir().join(format!("pen-bugreport-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let saved = SavedState { schema: 26, state: vec![1.0, 2.0] };
        let recording = Recording {
            start: Moment { saved: saved.clone(), edits: Edits::default() },
            frames: vec![Recorded { input: vec![0.5, 960.0, 640.0, 0.0, 0.0], ..Recorded::default() }],
        };
        let capture =
            Capture { recording: Some(recording.clone()), state: saved.clone(), game: &directory, position: 9 };
        let path = write(&capture, None, &directory).unwrap();
        assert!(path.starts_with(directory.join("reports")) && path.extension().unwrap() == "penbug");
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(recording::decode(&recording_bytes(&bytes).unwrap()).unwrap(), recording);
        let read = entries(&bytes).unwrap();
        let names: Vec<&str> = read.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, [REPORT, LOG, RECORDING, STATE]);
        let report = String::from_utf8(read[0].1.clone()).unwrap();
        assert!(report.contains("state schema 26, at frame 9"), "{report}");
        assert!(report.contains("the last 1 frames (0.5 s of game time)"), "{report}");
        assert_eq!(snapshot::decode(&read[3].1).unwrap(), saved);
        // `--replay` reads a report as its recording.
        assert_eq!(recording::read(&path).unwrap(), recording);
        let out = directory.join("unpacked");
        open(&path, Some(&out)).unwrap();
        assert_eq!(std::fs::read(out.join(STATE)).unwrap(), read[3].1);
        // Without a history, there is no recording to replay.
        let bare =
            write(&Capture { recording: None, ..capture }, Some(directory.join("bare.penbug")), &directory).unwrap();
        assert!(format!("{:#}", recording::read(&bare).unwrap_err()).contains("no recording"));
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
