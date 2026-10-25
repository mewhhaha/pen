//! Crash and performance reports for the game's developer, which the
//! player opts into. Nothing is sent unless both of these hold:
//!
//! - The developer gave an address for reports: `pen.json`'s
//!   `"reports": {"endpoint": "https://..."}`, which `pen build` ships as
//!   `game/reports.json`. A checkout's runs, `pen export`'s bundles and the
//!   browser have none, and the reports service answers them unavailable,
//!   so that a game offers no choice with nothing behind it.
//! - The player agreed, to crash reports or to performance reports or to
//!   both, through the game (the reports service, `REPORTS_OP_*`), which
//!   keeps their choice in `reports/consent.json` in the data directory.
//!   Until they choose, the answer is neither.
//!
//! With consent, a crash report (`crash.rs`) is copied into the local queue,
//! `reports/queue/` in the data directory, with the player's home directory
//! taken out of its paths; and as a packaged game ends, a performance report
//! joins it: its frame times (median, 95th and 99th percentiles, the worst),
//! the GPU, the system and pen's version, and nothing else. When a packaged
//! game starts, a thread of its own sends the queue's reports of the kinds
//! the player agreed to, through the system's curl (Windows 10 and later,
//! macOS, SteamOS and desktop Linux have it; pen links no HTTP client), one
//! POST each; a report that went is deleted, and one that did not waits for
//! the next start. The queue keeps the latest [`QUEUE`]. PEN_REPORTS=off
//! sends nothing, whatever was agreed; tests send nothing.
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail, ensure};
use pen_protocol::{reports_op, service};
use serde::{Deserialize, Serialize};

use crate::services::{Answer, Request, Service};

/// Reports the queue keeps: the latest; older ones go.
pub const QUEUE: usize = 20;

/// What the player agreed to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize, Serialize)]
pub struct Consent {
    #[serde(default)]
    pub crashes: bool,
    #[serde(default)]
    pub performance: bool,
}

impl Consent {
    /// As the reports service has it: 1 for crash reports, plus 2 for
    /// performance reports.
    pub fn mask(self) -> u32 {
        u32::from(self.crashes) | u32::from(self.performance) << 1
    }

    pub fn from_mask(mask: u32) -> Option<Consent> {
        (mask <= 3).then_some(Consent { crashes: mask & 1 != 0, performance: mask & 2 != 0 })
    }

    fn path(data: &Path) -> PathBuf {
        data.join("reports/consent.json")
    }

    /// The player's choice; neither when they have made none (or it
    /// cannot be read).
    pub fn read(data: &Path) -> Consent {
        std::fs::read_to_string(Consent::path(data))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    pub fn write(self, data: &Path) -> Result<()> {
        let path = Consent::path(data);
        crate::snapshot::replace(&path, (serde_json::to_string_pretty(&self)? + "\n").as_bytes())
    }
}

/// `game/reports.json`: where the developer takes reports.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub endpoint: String,
}

impl Config {
    /// An https address, or http to this machine (a developer's own test
    /// server).
    pub fn check(&self) -> Result<()> {
        let address = &self.endpoint;
        let local = ["http://127.0.0.1", "http://localhost", "http://[::1]"].iter().any(|start| {
            address.strip_prefix(start).is_some_and(|rest| rest.is_empty() || rest.starts_with([':', '/']))
        });
        ensure!(
            (address.starts_with("https://") || local)
                && address.len() <= 2048
                && address.chars().all(|character| character.is_ascii_graphic() && character != '"'),
            "reports.endpoint {address:?} is not an https address"
        );
        Ok(())
    }

    /// The bundle `game`'s, if it has one.
    pub fn read(game: &Path) -> Result<Option<Config>> {
        let path = game.join("reports.json");
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
        };
        let config: Config = serde_json::from_str(&text).with_context(|| format!("reading {}", path.display()))?;
        config.check()?;
        Ok(Some(config))
    }
}

/// `pen build`'s copy of the checkout's `"reports"` section, as
/// `game/reports.json`, when it has one.
#[cfg(feature = "compiler")]
pub fn ship_config(root: &Path, game: &Path) -> Result<()> {
    let file = root.join("pen.json");
    let text = std::fs::read_to_string(&file).with_context(|| format!("reading {}", file.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("reading {}", file.display()))?;
    let Some(section) = config.get("reports") else { return Ok(()) };
    let reports: Config =
        serde_json::from_value(section.clone()).with_context(|| format!("{}: reports", file.display()))?;
    reports.check()?;
    std::fs::write(game.join("reports.json"), serde_json::to_string_pretty(section)? + "\n")?;
    Ok(())
}

/// A report's kind, which starts its file's name.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Crash,
    Performance,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Crash => "crash",
            Kind::Performance => "performance",
        }
    }

    fn of(path: &Path) -> Option<Kind> {
        let name = path.file_name()?.to_str()?;
        [Kind::Crash, Kind::Performance].into_iter().find(|kind| name.starts_with(&format!("{}-", kind.name())))
    }

    fn agreed(self, consent: Consent) -> bool {
        match self {
            Kind::Crash => consent.crashes,
            Kind::Performance => consent.performance,
        }
    }

    fn content_type(self) -> &'static str {
        match self {
            Kind::Crash => "text/plain; charset=utf-8",
            Kind::Performance => "application/json",
        }
    }
}

fn queue(data: &Path) -> PathBuf {
    data.join("reports/queue")
}

/// The queue's reports, oldest first (their names start with their kind,
/// then when they were written).
fn queued(data: &Path) -> Vec<PathBuf> {
    let mut reports: Vec<PathBuf> = std::fs::read_dir(queue(data))
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| Kind::of(path).is_some())
        .collect();
    let written = |path: &PathBuf| {
        path.file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.split_once('-'))
            .map(|(_, rest)| rest.to_owned())
    };
    reports.sort_by_key(written);
    reports
}

/// The player's home directory taken out of `text`'s paths.
fn redacted(text: &str) -> String {
    match dirs::home_dir().map(|home| home.to_string_lossy().into_owned()) {
        Some(home) if home.len() > 1 => text.replace(&home, "~"),
        _ => text.to_owned(),
    }
}

/// Queues a report, keeping the latest [`QUEUE`].
fn enqueue(data: &Path, kind: Kind, text: &str) -> Result<PathBuf> {
    let directory = queue(data);
    std::fs::create_dir_all(&directory)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let extension = if kind == Kind::Performance { "json" } else { "txt" };
    let path = directory.join(format!("{}-{:020}-{}.{extension}", kind.name(), now.as_nanos(), std::process::id()));
    std::fs::write(&path, redacted(text))?;
    let reports = queued(data);
    for old in &reports[..reports.len().saturating_sub(QUEUE)] {
        let _ = std::fs::remove_file(old);
    }
    Ok(path)
}

/// Where a packaged game keeps reports, and whether its developer takes
/// them: set once, as it starts.
struct Installed {
    data: PathBuf,
    config: Option<Config>,
}

static INSTALLED: OnceLock<Installed> = OnceLock::new();

/// For a packaged game (its bundle `game`, its data directory `data`):
/// reads its developer's address for reports, and sends what is queued of
/// what the player agreed to, on a thread of its own.
pub fn install(game: &Path, data: PathBuf) {
    let config = Config::read(game).unwrap_or_else(|error| {
        eprintln!("reports: {error:#}");
        None
    });
    let installed = INSTALLED.get_or_init(|| Installed { data, config });
    let Some(config) = installed.config.clone() else { return };
    let consent = Consent::read(&installed.data);
    if consent == Consent::default() || std::env::var("PEN_REPORTS").is_ok_and(|value| value == "off") {
        return;
    }
    let data = installed.data.clone();
    let _ = std::thread::Builder::new().name("reports".into()).spawn(move || {
        if let Err(error) = send(&data, &config, consent, &Curl) {
            eprintln!("reports: {error:#}; they wait for the next start");
        }
    });
}

/// After a crash report is written at `report`: a copy for the queue, when
/// the player agreed to crash reports.
pub fn crashed(report: &Path) {
    let Some(installed) = INSTALLED.get().filter(|installed| installed.config.is_some()) else { return };
    if !Consent::read(&installed.data).crashes {
        return;
    }
    if let Err(error) = std::fs::read_to_string(report)
        .map_err(anyhow::Error::from)
        .and_then(|text| enqueue(&installed.data, Kind::Crash, &text))
    {
        eprintln!("reports: queueing {}: {error:#}", report.display());
    }
}

/// Frame times by quarter millisecond, up to a quarter second: what a
/// performance report's percentiles come from.
const BUCKETS: usize = 1000;
static FRAMES: [AtomicU32; BUCKETS + 1] = [const { AtomicU32::new(0) }; BUCKETS + 1];
static WORST_US: AtomicU64 = AtomicU64::new(0);

/// Notes how long a frame took (`crash::frame`): an atomic add.
pub fn frame(duration: Duration) {
    let quarters = usize::try_from(duration.as_micros() / 250).unwrap_or(usize::MAX).min(BUCKETS);
    FRAMES[quarters].fetch_add(1, Ordering::Relaxed);
    WORST_US.fetch_max(u64::try_from(duration.as_micros()).unwrap_or(u64::MAX), Ordering::Relaxed);
}

/// The frame times noted: their count, and the median, 95th and 99th
/// percentiles in milliseconds (each its bucket's middle).
fn percentiles(counts: &[u32]) -> (u64, [f64; 3]) {
    let total: u64 = counts.iter().map(|count| u64::from(*count)).sum();
    let at = |fraction: f64| {
        let wanted = (total as f64 * fraction).ceil().max(1.0);
        let mut seen = 0.0;
        for (bucket, count) in counts.iter().enumerate() {
            seen += f64::from(*count);
            if seen >= wanted {
                return (bucket as f64 + 0.5) / 4.0;
            }
        }
        0.0
    };
    (total, [at(0.5), at(0.95), at(0.99)])
}

/// The performance report of the frames noted so far.
fn performance() -> Option<String> {
    let counts: Vec<u32> = FRAMES.iter().map(|count| count.load(Ordering::Relaxed)).collect();
    let (frames, [median, p95, p99]) = percentiles(&counts);
    if frames == 0 {
        return None;
    }
    // A bucket's middle may lie past the slowest frame in it.
    let worst = WORST_US.load(Ordering::Relaxed) as f64 / 1000.0;
    let [median, p95, p99] = [median, p95, p99].map(|time| time.min(worst));
    let report = serde_json::json!({
        "game": crate::identity().name,
        "pen": env!("CARGO_PKG_VERSION"),
        "system": crate::crash::system(),
        "gpu": crate::crash::gpu_description(),
        "frames": frames,
        "frame_ms": {
            "median": median,
            "p95": p95,
            "p99": p99,
            "worst": worst,
        },
    });
    serde_json::to_string_pretty(&report).ok()
}

/// As a packaged game ends: its performance report for the queue, when the
/// player agreed to performance reports.
pub fn finish() {
    let Some(installed) = INSTALLED.get().filter(|installed| installed.config.is_some()) else { return };
    if !Consent::read(&installed.data).performance {
        return;
    }
    if let Some(report) = performance()
        && let Err(error) = enqueue(&installed.data, Kind::Performance, &report)
    {
        eprintln!("reports: queueing the performance report: {error:#}");
    }
}

/// How a report goes to the developer.
pub trait Transport {
    fn post(&self, endpoint: &str, content_type: &str, kind: &str, body: &Path) -> Result<()>;
}

/// The system's curl: a POST of the report's file, which must be answered
/// with success within 30 seconds.
pub struct Curl;

impl Transport for Curl {
    fn post(&self, endpoint: &str, content_type: &str, kind: &str, body: &Path) -> Result<()> {
        if cfg!(test) {
            bail!("tests send nothing");
        }
        let mut command = std::process::Command::new("curl");
        command
            .args(["--silent", "--show-error", "--fail", "--max-time", "30", "--proto", "=https,http"])
            .args(["--header", &format!("Content-Type: {content_type}")])
            .args(["--header", &format!("X-Pen-Report: {kind}")])
            .args(["--header", &format!("X-Pen-Game: {}", crate::identity().name)])
            .arg("--data-binary")
            .arg(format!("@{}", body.display()))
            .arg(endpoint)
            .stdout(std::process::Stdio::null());
        // No console window flashing up beside the game.
        #[cfg(windows)]
        std::os::windows::process::CommandExt::creation_flags(&mut command, 0x0800_0000);
        let output = command.output().context("running curl")?;
        ensure!(output.status.success(), "curl: {}", String::from_utf8_lossy(&output.stderr).trim());
        Ok(())
    }
}

/// Sends the queue's reports of the kinds `consent` allows, oldest first,
/// deleting each that went; stops at the first that does not. Returns how
/// many went.
fn send(data: &Path, config: &Config, consent: Consent, transport: &dyn Transport) -> Result<usize> {
    config.check()?;
    let mut sent = 0;
    for report in queued(data) {
        let Some(kind) = Kind::of(&report).filter(|kind| kind.agreed(consent)) else { continue };
        transport.post(&config.endpoint, kind.content_type(), kind.name(), &report)?;
        std::fs::remove_file(&report)?;
        sent += 1;
    }
    Ok(sent)
}

/// The reports service: the player's consent, which the game asks for and
/// sets, and the queue's length. `data` is a packaged game's data
/// directory when its developer takes reports; without it, every request
/// is answered unavailable.
pub struct Reports {
    data: Option<PathBuf>,
}

impl Reports {
    /// This run's: available when it plays a package whose developer takes
    /// reports.
    pub fn new() -> Reports {
        let data =
            INSTALLED.get().filter(|installed| installed.config.is_some()).map(|installed| installed.data.clone());
        Reports { data }
    }
}

impl Service<crate::input::Input> for Reports {
    fn code(&self) -> u32 {
        service::REPORTS
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        let Some(data) = &self.data else { return Some(Answer::unavailable()) };
        Some(match request.operation {
            reports_op::CONSENT => Answer::ok(Consent::read(data).mask() as f32),
            reports_op::SET_CONSENT => {
                let consent = pen_protocol::boundary::whole(request.numbers[0]).and_then(Consent::from_mask);
                match consent.map(|consent| (consent, consent.write(data))) {
                    Some((consent, Ok(()))) => Answer::ok(consent.mask() as f32),
                    Some((_, Err(error))) => {
                        eprintln!("reports: {error:#}");
                        Answer::failed()
                    }
                    None => Answer::failed(),
                }
            }
            reports_op::QUEUED => Answer::ok(queued(data).len() as f32),
            _ => Answer::unknown(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use pen_protocol::service_status;

    use super::*;

    fn directory(name: &str) -> PathBuf {
        let directory = crate::project::temporary_dir().unwrap().join(format!("reports-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    /// Notes what it would have sent; fails from the `fail_from`th on.
    struct Fake {
        posted: RefCell<Vec<(String, String)>>,
        fail_from: usize,
    }

    impl Transport for Fake {
        fn post(&self, endpoint: &str, _content_type: &str, kind: &str, body: &Path) -> Result<()> {
            let mut posted = self.posted.borrow_mut();
            ensure!(posted.len() < self.fail_from, "offline");
            posted.push((format!("{kind} to {endpoint}"), std::fs::read_to_string(body)?));
            Ok(())
        }
    }

    #[test]
    fn consent_is_neither_until_the_player_chooses() {
        let data = directory("consent");
        assert_eq!(Consent::read(&data), Consent::default());
        let both = Consent { crashes: true, performance: true };
        assert_eq!((both.mask(), Consent::from_mask(3)), (3, Some(both)));
        assert_eq!(Consent::from_mask(2), Some(Consent { crashes: false, performance: true }));
        assert_eq!(Consent::from_mask(4), None);
        both.write(&data).unwrap();
        assert_eq!(Consent::read(&data), both);
        std::fs::write(data.join("reports/consent.json"), "nonsense").unwrap();
        assert_eq!(Consent::read(&data), Consent::default());
        std::fs::remove_dir_all(data).unwrap();
    }

    #[test]
    fn only_https_endpoints_or_this_machine() {
        let check = |endpoint: &str| Config { endpoint: endpoint.into() }.check().is_ok();
        assert!(check("https://reports.example.com/gdev"));
        assert!(check("http://127.0.0.1:8047/reports") && check("http://localhost/") && check("http://[::1]:9"));
        assert!(!check("http://reports.example.com/") && !check("http://localhost.example.com/"));
        assert!(!check("ftp://example.com/") && !check("https://a b") && !check(""));
        let game = directory("config");
        assert!(Config::read(&game).unwrap().is_none());
        std::fs::write(game.join("reports.json"), r#"{"endpoint": "http://example.com"}"#).unwrap();
        assert!(Config::read(&game).is_err());
        std::fs::remove_dir_all(game).unwrap();
    }

    #[test]
    fn the_queue_sends_what_was_agreed_and_keeps_what_did_not_go() {
        let data = directory("queue");
        let home = dirs::home_dir().unwrap().to_string_lossy().into_owned();
        let crash = enqueue(&data, Kind::Crash, &format!("gdev crashed\ncommand: {home}/games/gdev --data {home}/x\n"))
            .unwrap();
        assert_eq!(std::fs::read_to_string(&crash).unwrap(), "gdev crashed\ncommand: ~/games/gdev --data ~/x\n");
        enqueue(&data, Kind::Performance, "{\"frames\": 60}").unwrap();
        enqueue(&data, Kind::Crash, "second").unwrap();
        std::fs::write(queue(&data).join("notes.txt"), "not a report").unwrap();
        assert_eq!(queued(&data).len(), 3);
        let config = Config { endpoint: "https://reports.example.com/gdev".into() };

        // Crash reports only: the performance report stays.
        let fake = Fake { posted: RefCell::new(Vec::new()), fail_from: usize::MAX };
        let crashes = Consent { crashes: true, performance: false };
        assert_eq!(send(&data, &config, crashes, &fake).unwrap(), 2);
        let posted = fake.posted.borrow();
        assert_eq!(posted[0].0, "crash to https://reports.example.com/gdev");
        assert!(posted[0].1.starts_with("gdev crashed") && posted[1].1 == "second");
        assert_eq!(queued(&data).len(), 1);

        // Offline: nothing goes, and it waits.
        let offline = Fake { posted: RefCell::new(Vec::new()), fail_from: 0 };
        let both = Consent { crashes: true, performance: true };
        assert!(send(&data, &config, both, &offline).is_err());
        assert_eq!(queued(&data).len(), 1);
        // The real transport refuses in tests.
        assert!(send(&data, &config, both, &Curl).is_err());
        assert_eq!(queued(&data).len(), 1);

        // The queue keeps the latest QUEUE.
        for index in 0..QUEUE + 5 {
            enqueue(&data, Kind::Crash, &index.to_string()).unwrap();
        }
        let left = queued(&data);
        assert_eq!(left.len(), QUEUE);
        assert_eq!(std::fs::read_to_string(left.last().unwrap()).unwrap(), (QUEUE + 4).to_string());
        assert!(queue(&data).join("notes.txt").exists());
        std::fs::remove_dir_all(data).unwrap();
    }

    #[test]
    fn the_service_answers_consent_where_reports_can_go() {
        let request = |operation| Request { id: 1, operation, numbers: [3.0, 0.0, 0.0, 0.0], values: &[] };
        for operation in [reports_op::CONSENT, reports_op::SET_CONSENT, reports_op::QUEUED] {
            let answer = Reports { data: None }.call(&request(operation)).unwrap();
            assert_eq!(answer.status, service_status::UNAVAILABLE);
        }

        let data = directory("service");
        let mut reports = Reports { data: Some(data.clone()) };
        let mut ask = |operation, a: f32| {
            let request = Request { id: 1, operation, numbers: [a, 0.0, 0.0, 0.0], values: &[] };
            let answer = reports.call(&request).unwrap();
            (answer.status, answer.value)
        };
        assert_eq!(ask(reports_op::CONSENT, 0.0), (service_status::OK, 0.0));
        assert_eq!(ask(reports_op::SET_CONSENT, 3.0), (service_status::OK, 3.0));
        assert_eq!(Consent::read(&data), Consent { crashes: true, performance: true });
        assert_eq!(ask(reports_op::SET_CONSENT, 1.5).0, service_status::FAILED);
        assert_eq!(ask(reports_op::SET_CONSENT, 8.0).0, service_status::FAILED);
        assert_eq!(ask(reports_op::SET_CONSENT, 2.0), (service_status::OK, 2.0));
        assert_eq!(ask(reports_op::CONSENT, 0.0), (service_status::OK, 2.0));
        assert_eq!(ask(reports_op::QUEUED, 0.0), (service_status::OK, 0.0));
        assert_eq!(ask(99, 0.0).0, service_status::UNKNOWN);
        std::fs::remove_dir_all(data).unwrap();
    }

    #[test]
    fn frame_times_give_percentiles() {
        let mut counts = vec![0u32; BUCKETS + 1];
        // 90 frames of 4 ms, 9 of 10 ms and one of 50 ms.
        counts[16] = 90;
        counts[40] = 9;
        counts[200] = 1;
        assert_eq!(percentiles(&counts), (100, [4.125, 10.125, 10.125]));
        counts[200] = 2;
        assert_eq!(percentiles(&counts).1[2], 50.125);
        assert_eq!(percentiles(&vec![0; BUCKETS + 1]).0, 0);
        // Other tests note frames too (crash::frame): what these add.
        let before = (FRAMES[12].load(Ordering::Relaxed), FRAMES[BUCKETS].load(Ordering::Relaxed));
        frame(Duration::from_millis(3));
        frame(Duration::from_secs(9));
        let after = (FRAMES[12].load(Ordering::Relaxed), FRAMES[BUCKETS].load(Ordering::Relaxed));
        assert_eq!((after.0 - before.0, after.1 - before.1), (1, 1));
        let report: serde_json::Value = serde_json::from_str(&performance().unwrap()).unwrap();
        assert!(report["frames"].as_u64().unwrap() >= 2 && report["frame_ms"]["worst"].as_f64().unwrap() >= 9000.0);
        assert!(report["system"].as_str().unwrap().starts_with(std::env::consts::OS), "{report}");
    }
}
