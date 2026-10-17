//! Crash reports. A panic on any thread writes a report to `crashes/` in the
//! data directory, beside the saves and preferences, and says where it is:
//! what panicked where, the backtrace, this build, the system, the GPU the
//! game drew with and how long its last frames took, for a player to send.
//! So does an error that stops the player runtime, which players start
//! without a terminal (and on Windows it shows in a message box too). Only
//! the latest [`KEEP`] reports stay.
//!
//! Shipped runtimes are stripped copies of a build that keeps its symbols
//! (`just player`, `just windows`), so a backtrace has addresses rather than
//! names. Against that unstripped build they resolve with the executable's
//! load address (Linux), `addr2line -fiCe pen 0x<address - load address>`,
//! or on any system with the anchor, where `crash::frame` was: an address
//! minus the anchor plus `crash::frame`'s address in the unstripped build.
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How many of the latest frame times a report shows.
const FRAMES: usize = 30;
/// Reports one run writes at most (a panic can bring others with it).
const REPORTS: usize = 3;
/// Reports `crashes/` keeps: the latest, older ones go.
const KEEP: usize = 20;

static FRAME_MS: [AtomicU32; FRAMES] = [const { AtomicU32::new(0) }; FRAMES];
static FRAME_COUNT: AtomicUsize = AtomicUsize::new(0);
static WRITTEN: AtomicUsize = AtomicUsize::new(0);
/// PEN_CRASH_TEST=N: the frame to panic at, to try reports (0: never).
static CRASH_AT: AtomicUsize = AtomicUsize::new(0);
static GPU: Mutex<Option<String>> = Mutex::new(None);
/// Whether nobody is there to dismiss a message box: a headless run, as on
/// a CI machine, where one would wait forever.
static UNATTENDED: AtomicBool = AtomicBool::new(false);
/// Where reports go, once `install` knows.
static DIRECTORY: OnceLock<PathBuf> = OnceLock::new();

/// Where reports go: `crashes/` in the data directory.
#[cfg(not(feature = "compiler"))]
fn directory() -> PathBuf {
    DIRECTORY.get().cloned().unwrap_or_else(|| crate::data_dir().join("crashes"))
}

/// Writes a report for every panic from now on, into `crashes/` of `data`
/// (`crate::data_dir` when None, as for saves).
/// PEN_CRASH_TEST=N panics at frame N, to try it on a platform.
pub fn install(data: Option<PathBuf>) {
    let crash_at = std::env::var("PEN_CRASH_TEST").ok().and_then(|frame| frame.parse().ok());
    CRASH_AT.store(crash_at.unwrap_or(0), Ordering::Relaxed);
    let data = data.unwrap_or_else(crate::data_dir);
    let directory = DIRECTORY.get_or_init(|| data.join("crashes")).clone();
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        previous(info);
        if WRITTEN.fetch_add(1, Ordering::Relaxed) >= REPORTS {
            return;
        }
        let backtrace = std::backtrace::Backtrace::force_capture();
        let thread = std::thread::current();
        let panic = Panic {
            what: "crashed",
            message: info.payload_as_str().unwrap_or("(not a string)"),
            location: info.location().map(ToString::to_string),
            thread: thread.name().unwrap_or("unnamed"),
            details: ("backtrace", &format!("{backtrace:#}")),
        };
        let name = &crate::identity().name;
        match write(&directory, &panic) {
            Ok(path) => eprintln!("{name} crashed. A report is in {}", path.display()),
            Err(error) => {
                eprintln!("{name} crashed, and its report could not be written in {}: {error}", directory.display())
            }
        }
    }));
}

/// Says that this run is unattended (headless, or a command such as
/// `packets`): a stopping error goes to stderr and its report, not to a
/// message box.
pub fn unattended() {
    UNATTENDED.store(true, Ordering::Relaxed);
}

/// Says why pen stopped with an error: on stderr, as ever. The player
/// runtime, which players start without a terminal, writes it to a report
/// too, and on Windows, where it has no console, shows it in a message box.
pub fn stopped(error: &anyhow::Error) {
    eprintln!("Error: {error:?}");
    #[cfg(not(feature = "compiler"))]
    {
        let directory = directory();
        let said = match write(&directory, &stopping(&format!("{error:?}"))) {
            Ok(path) => format!("A report is in {}", path.display()),
            Err(problem) => format!("Its report could not be written in {}: {problem}", directory.display()),
        };
        eprintln!("{said}");
        #[cfg(windows)]
        if !UNATTENDED.load(Ordering::Relaxed) {
            message_box(&crate::identity().title, &format!("{} stopped: {error:#}\n\n{said}", crate::identity().name));
        }
    }
}

/// A report of an error, as `{error:?}` has it: its words on the first
/// line, then its causes (and a backtrace, when RUST_BACKTRACE asks).
#[cfg(any(test, not(feature = "compiler")))]
fn stopping(error: &str) -> Panic<'_> {
    let message = error.lines().next().unwrap_or_default();
    Panic { what: "stopped", message, location: None, thread: "main", details: ("error", error) }
}

/// Shows `text` in a message box over every window, until it is dismissed.
#[cfg(all(windows, not(feature = "compiler")))]
fn message_box(title: &str, text: &str) {
    use windows_sys::Win32::UI::WindowsAndMessaging::{MB_ICONERROR, MB_OK, MB_SETFOREGROUND, MessageBoxW};
    let wide = |text: &str| text.encode_utf16().chain([0]).collect::<Vec<u16>>();
    let (text, title) = (wide(text), wide(title));
    // SAFETY: both are NUL-terminated UTF-16 strings that outlive the call,
    // and no owner window is fine.
    unsafe {
        MessageBoxW(std::ptr::null_mut(), text.as_ptr(), title.as_ptr(), MB_OK | MB_ICONERROR | MB_SETFOREGROUND)
    };
}

/// The adapter frames are drawn with, for reports.
pub fn gpu(info: &wgpu::AdapterInfo) {
    let description = format!(
        "{} ({:?}, {:?}; driver {} {}; vendor {:#06x}, device {:#06x})",
        info.name, info.backend, info.device_type, info.driver, info.driver_info, info.vendor, info.device
    );
    if let Ok(mut gpu) = GPU.lock() {
        *gpu = Some(description);
    }
}

/// The adapter frames are drawn with, as reports say it, once one is open.
pub fn gpu_description() -> Option<String> {
    GPU.try_lock().ok().and_then(|gpu| gpu.clone())
}

/// Notes how long a frame took; cheap enough for every frame.
pub fn frame(duration: Duration) {
    let index = FRAME_COUNT.fetch_add(1, Ordering::Relaxed);
    FRAME_MS[index % FRAMES].store((duration.as_secs_f32() * 1000.0).to_bits(), Ordering::Relaxed);
    crate::reports::frame(duration);
    if index + 1 == CRASH_AT.load(Ordering::Relaxed) {
        panic!("PEN_CRASH_TEST: a crash at frame {}, as asked", index + 1);
    }
}

/// What a report is of: a panic, or an error the runtime stopped with.
struct Panic<'a> {
    /// What the game did: "crashed" or "stopped".
    what: &'a str,
    message: &'a str,
    location: Option<String>,
    thread: &'a str,
    /// The backtrace, or the error's causes, and what to call them.
    details: (&'a str, &'a str),
}

fn write(directory: &Path, panic: &Panic) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(directory)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let stamp = timestamp(now.as_secs()).replace([':', '-'], "");
    let path = directory.join(format!("crash-{stamp}-{}.txt", std::process::id()));
    std::fs::write(&path, report(panic, now.as_secs()))?;
    prune(directory, KEEP);
    crate::reports::crashed(&path);
    Ok(path)
}

/// Deletes all but the latest `keep` reports in `directory`: their names
/// sort by when they were written.
fn prune(directory: &Path, keep: usize) {
    let mut reports: Vec<PathBuf> = std::fs::read_dir(directory)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("crash-") && name.ends_with(".txt"))
        })
        .collect();
    reports.sort();
    for old in &reports[..reports.len().saturating_sub(keep)] {
        let _ = std::fs::remove_file(old);
    }
}

fn report(panic: &Panic, now: u64) -> String {
    let mut text = String::new();
    let _ = writeln!(text, "{} {}: {}", crate::identity().name, panic.what, panic.message);
    let at = panic.location.as_ref().map(|location| format!("at {location} ")).unwrap_or_default();
    let _ = writeln!(text, "{at}on thread {}, {}", panic.thread, timestamp(now));
    text.push_str(&about());
    if let Some(base) = load_address() {
        let _ = writeln!(text, "executable loaded at {base:#x}");
    }
    let _ = writeln!(text, "anchor: crash::frame at {:#x}", frame as fn(Duration) as usize);
    let _ = writeln!(text, "\n{}:\n{}", panic.details.0, panic.details.1);
    text
}

/// What a report says of the build, the system and the run: pen's version,
/// the system, the command line, the GPU and the last frames' times, a
/// line each (bug reports share it).
pub fn about() -> String {
    let mut text = String::new();
    let features: Vec<&str> = [("compiler", cfg!(feature = "compiler")), ("steam", cfg!(feature = "steam"))]
        .into_iter()
        .filter_map(|(name, on)| on.then_some(name))
        .collect();
    let _ = writeln!(
        text,
        "pen {}{} ({}, {}-{}, features: {})",
        env!("CARGO_PKG_VERSION"),
        option_env!("PEN_BUILD").map(|build| format!(" {build}")).unwrap_or_default(),
        if cfg!(debug_assertions) { "debug" } else { "optimized" },
        std::env::consts::ARCH,
        std::env::consts::OS,
        if features.is_empty() { "none".into() } else { features.join(", ") },
    );
    let _ = writeln!(text, "system: {}", system());
    let _ = writeln!(text, "command: {}", std::env::args().collect::<Vec<_>>().join(" "));
    let gpu = GPU.try_lock().ok().and_then(|gpu| gpu.clone());
    let _ = writeln!(text, "GPU: {}", gpu.as_deref().unwrap_or("none opened"));
    let count = FRAME_COUNT.load(Ordering::Relaxed);
    let recent: Vec<String> = (count.saturating_sub(FRAMES)..count)
        .map(|index| format!("{:.1}", f32::from_bits(FRAME_MS[index % FRAMES].load(Ordering::Relaxed))))
        .collect();
    let _ = writeln!(text, "frames: {count}; the last {} took (ms, oldest first): {}", recent.len(), recent.join(" "));
    text
}

/// Now, as `2026-10-10T23:31:39Z`.
pub fn now() -> String {
    timestamp(SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs())
}

/// The OS and its version, as far as is cheap to tell.
pub(crate) fn system() -> String {
    let release = std::fs::read_to_string("/etc/os-release").ok().and_then(|text| {
        text.lines().find_map(|line| line.strip_prefix("PRETTY_NAME=")).map(|name| name.trim_matches('"').to_owned())
    });
    let kernel = std::fs::read_to_string("/proc/sys/kernel/osrelease").ok();
    let mut system = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
    for part in [release, kernel.map(|kernel| kernel.trim().to_owned())].into_iter().flatten() {
        system = format!("{system}, {part}");
    }
    if let Ok(app) = std::env::var("SteamAppId") {
        system = format!("{system}, under Steam (app {app})");
    }
    system
}

/// Where the executable is mapped (Linux), which turns a stripped
/// backtrace's addresses into offsets in the executable.
fn load_address() -> Option<u64> {
    let exe = std::env::current_exe().ok()?;
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    maps.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        let range = fields.next()?;
        let offset = fields.nth(1)?;
        let path = fields.nth(2)?;
        (Path::new(path) == exe && u64::from_str_radix(offset, 16).ok()? == 0)
            .then(|| u64::from_str_radix(range.split('-').next()?, 16).ok())
            .flatten()
    })
}

/// `seconds` since 1970 as an ISO 8601 UTC time.
pub(crate) fn timestamp(seconds: u64) -> String {
    let days = i64::try_from(seconds / 86_400).expect("u64 seconds are fewer than i64::MAX days");
    let time = seconds % 86_400;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 { month_index + 3 } else { month_index - 9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z", time / 3600, time / 60 % 60, time % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps() {
        assert_eq!(timestamp(0), "1970-01-01T00:00:00Z");
        assert_eq!(timestamp(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(timestamp(1_791_675_099), "2026-10-10T23:31:39Z");
    }

    #[test]
    fn reports() {
        for _ in 0..40 {
            frame(Duration::from_micros(8_333));
        }
        let panic = Panic {
            what: "crashed",
            message: "boom",
            location: Some("src/x.rs:1:2".into()),
            thread: "main",
            details: ("backtrace", "0: here"),
        };
        let text = report(&panic, 0);
        let name = &crate::identity().name;
        assert!(
            text.starts_with(&format!("{name} crashed: boom\nat src/x.rs:1:2 on thread main, 1970-01-01T00:00:00Z\n"))
        );
        assert!(text.contains("the last 30 took (ms, oldest first): 8.3 8.3"));
        assert!(text.ends_with("backtrace:\n0: here\n"));
        let directory = std::env::temp_dir().join(format!("pen-crash-test-{}", std::process::id()));
        let path = write(&directory, &panic).unwrap();
        assert!(std::fs::read_to_string(&path).unwrap().contains("boom"));
        std::fs::remove_dir_all(directory).unwrap();

        // An error the runtime stopped with: its words, and its causes.
        let error = anyhow::anyhow!("no adapter").context("opening the GPU");
        let details = format!("{error:?}");
        let text = report(&stopping(&details), 0);
        assert!(text.starts_with(&format!("{name} stopped: opening the GPU\non thread main, ")), "{text}");
        assert!(text.contains("error:\nopening the GPU\n\nCaused by:\n    no adapter"), "{text}");
    }

    #[test]
    fn only_the_latest_reports_stay() {
        let directory = std::env::temp_dir().join(format!("pen-crash-prune-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        for day in 10..15 {
            std::fs::write(directory.join(format!("crash-202610{day}T000000Z-1.txt")), "").unwrap();
        }
        std::fs::write(directory.join("notes.txt"), "the player's own").unwrap();
        prune(&directory, 2);
        let mut left: Vec<String> = std::fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        assert_eq!(left, ["crash-20261013T000000Z-1.txt", "crash-20261014T000000Z-1.txt", "notes.txt"]);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
