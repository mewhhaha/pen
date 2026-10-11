//! Crash reports. A panic on any thread writes a report to `crashes/` in the
//! data directory, beside the saves and preferences, and says where it is:
//! what panicked where, the backtrace, this build, the system, the GPU the
//! game drew with and how long its last frames took, for a player to send.
//!
//! Shipped runtimes are stripped copies of a build that keeps its symbols
//! (`just player`, `just windows`), so a backtrace has addresses rather than
//! names. Against that unstripped build they resolve with the executable's
//! load address (Linux), `addr2line -fiCe pen 0x<address - load address>`,
//! or on any system with the anchor, where `crash::frame` was: an address
//! minus the anchor plus `crash::frame`'s address in the unstripped build.
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How many of the latest frame times a report shows.
const FRAMES: usize = 30;
/// Reports one run writes at most (a panic can bring others with it).
const REPORTS: usize = 3;

static FRAME_MS: [AtomicU32; FRAMES] = [const { AtomicU32::new(0) }; FRAMES];
static FRAME_COUNT: AtomicUsize = AtomicUsize::new(0);
static WRITTEN: AtomicUsize = AtomicUsize::new(0);
/// PEN_CRASH_TEST=N: the frame to panic at, to try reports (0: never).
static CRASH_AT: AtomicUsize = AtomicUsize::new(0);
static GPU: Mutex<Option<String>> = Mutex::new(None);

/// Writes a report for every panic from now on, into `crashes/` of `data`
/// (`crate::data_dir` when None, as for saves).
/// PEN_CRASH_TEST=N panics at frame N, to try it on a platform.
pub fn install(data: Option<PathBuf>) {
    let crash_at = std::env::var("PEN_CRASH_TEST").ok().and_then(|frame| frame.parse().ok());
    CRASH_AT.store(crash_at.unwrap_or(0), Ordering::Relaxed);
    let data = data.unwrap_or_else(crate::data_dir);
    let directory = data.join("crashes");
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        previous(info);
        if WRITTEN.fetch_add(1, Ordering::Relaxed) >= REPORTS {
            return;
        }
        let backtrace = std::backtrace::Backtrace::force_capture();
        let thread = std::thread::current();
        let panic = Panic {
            message: info.payload_as_str().unwrap_or("(not a string)"),
            location: info.location().map(ToString::to_string).unwrap_or_default(),
            thread: thread.name().unwrap_or("unnamed"),
            backtrace: &format!("{backtrace:#}"),
        };
        match write(&directory, &panic) {
            Ok(path) => eprintln!("{} crashed. A report is in {}", crate::GAME, path.display()),
            Err(error) => {
                eprintln!("{} crashed, and its report could not be written in {}: {error}", crate::GAME, directory.display())
            }
        }
    }));
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

/// Notes how long a frame took; cheap enough for every frame.
pub fn frame(duration: Duration) {
    let index = FRAME_COUNT.fetch_add(1, Ordering::Relaxed);
    FRAME_MS[index % FRAMES].store((duration.as_secs_f32() * 1000.0).to_bits(), Ordering::Relaxed);
    if index + 1 == CRASH_AT.load(Ordering::Relaxed) {
        panic!("PEN_CRASH_TEST: a crash at frame {}, as asked", index + 1);
    }
}

struct Panic<'a> {
    message: &'a str,
    location: String,
    thread: &'a str,
    backtrace: &'a str,
}

fn write(directory: &Path, panic: &Panic) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(directory)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let stamp = timestamp(now.as_secs()).replace([':', '-'], "");
    let path = directory.join(format!("crash-{stamp}-{}.txt", std::process::id()));
    std::fs::write(&path, report(panic, now.as_secs()))?;
    Ok(path)
}

fn report(panic: &Panic, now: u64) -> String {
    let mut text = String::new();
    let features: Vec<&str> =
        [("compiler", cfg!(feature = "compiler")), ("steam", cfg!(feature = "steam"))]
            .into_iter()
            .filter_map(|(name, on)| on.then_some(name))
            .collect();
    let _ = writeln!(text, "{} crashed: {}", crate::GAME, panic.message);
    let _ = writeln!(text, "at {} on thread {}, {}", panic.location, panic.thread, timestamp(now));
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
    if let Some(base) = load_address() {
        let _ = writeln!(text, "executable loaded at {base:#x}");
    }
    let _ = writeln!(text, "anchor: crash::frame at {:#x}", frame as fn(Duration) as usize);
    let _ = writeln!(text, "\nbacktrace:\n{}", panic.backtrace);
    text
}

/// The OS and its version, as far as is cheap to tell.
fn system() -> String {
    let release = std::fs::read_to_string("/etc/os-release").ok().and_then(|text| {
        text.lines()
            .find_map(|line| line.strip_prefix("PRETTY_NAME="))
            .map(|name| name.trim_matches('"').to_owned())
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
fn timestamp(seconds: u64) -> String {
    let days = (seconds / 86_400) as i64;
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
        let panic = Panic { message: "boom", location: "src/x.rs:1:2".into(), thread: "main", backtrace: "0: here" };
        let text = report(&panic, 0);
        assert!(text.starts_with("gdev crashed: boom\nat src/x.rs:1:2 on thread main, 1970-01-01T00:00:00Z\n"));
        assert!(text.contains("the last 30 took (ms, oldest first): 8.3 8.3"));
        assert!(text.ends_with("backtrace:\n0: here\n"));
        let directory = std::env::temp_dir().join(format!("pen-crash-test-{}", std::process::id()));
        let path = write(&directory, &panic).unwrap();
        assert!(std::fs::read_to_string(&path).unwrap().contains("boom"));
        std::fs::remove_dir_all(directory).unwrap();
    }
}
