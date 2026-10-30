//! The golden-image tests' review page: every shot of a `pen test` run
//! beside its golden image and their difference, with how far apart they
//! are. Each run writes it as `review.html` among its shots (the golden
//! images copied beside them, so the directory stands alone: CI keeps it
//! with the shots), and `pen test --review` serves it on the loopback
//! address with an Accept button for each shot, which makes the shot the
//! golden image. Only shots of the run are accepted, only from the page's
//! own origin, as JSON. `pen test --review --last` serves the last run's
//! page again without running anything (nor needing the GPU).
use std::fmt::Write as _;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::web::{Reply, Request, decode_path, read_request, respond};

/// How a shot compared with its golden image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    Passed,
    Failed,
    /// There is no golden image yet.
    Missing,
    /// The golden image has another size.
    Resized,
    /// Written as the golden image (`--update`).
    Updated,
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Status::Passed => "passed",
            Status::Failed => "failed",
            Status::Missing => "no golden image",
            Status::Resized => "another size",
            Status::Updated => "updated",
        }
    }
}

/// A shot of the run, and how it compared.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Shot {
    pub scenario: String,
    pub about: String,
    /// Its file name, `NAME.png`.
    pub file: String,
    pub status: Status,
    /// How far apart they are, or why they were not compared.
    pub summary: String,
}

impl Shot {
    /// `scenario/file`: its place under the shots and the golden images.
    pub fn id(&self) -> String {
        format!("{}/{}", self.scenario, self.file)
    }

    fn stem(&self) -> &str {
        self.file.strip_suffix(".png").unwrap_or(&self.file)
    }
}

/// The page's name among the shots.
pub const PAGE: &str = "review.html";
/// The run's shots and threshold beside it, for `--last` to serve again.
const RESULTS: &str = "review.json";

#[derive(Serialize, Deserialize)]
struct Results {
    threshold: f64,
    shots: Vec<Shot>,
}

/// `text` for HTML text and attribute values.
fn escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&#39;")
}

/// The page, its images at their paths among the shots: `SCENARIO/NAME.png`,
/// `SCENARIO/NAME.golden.png` and `SCENARIO/NAME.diff.png`.
pub fn page(shots: &[Shot], threshold: f64) -> String {
    let failing = shots.iter().filter(|shot| !matches!(shot.status, Status::Passed | Status::Updated)).count();
    let mut html = String::from(STYLE);
    let _ = write!(
        html,
        "<h1>Golden images</h1>\n<p>{} shots, {failing} not passing (RMSE threshold {:.2}%). \\n         <label><input type=checkbox id=failing{}> only those not passing</label></p>\n<p id=offline hidden>Opened \\n         as a file: <code>pen test --review</code> serves this page with buttons that accept a shot as its golden \\n         image, or <code>pen test --update --only SCENARIO</code> accepts a scenario's.</p>\n",
        shots.len(),
        threshold * 100.0,
        if failing > 0 { " checked" } else { "" }
    );
    for shot in shots {
        let (id, stem, scenario) = (escape(&shot.id()), escape(shot.stem()), escape(&shot.scenario));
        let golden = if matches!(shot.status, Status::Missing) {
            "<div class=none>no golden image</div>".to_owned()
        } else {
            format!(
                "<a href=\"{scenario}/{stem}.golden.png\"><img src=\"{scenario}/{stem}.golden.png\" alt=\"golden\"></a>"
            )
        };
        let diff = if matches!(shot.status, Status::Failed | Status::Passed) && !shot.summary.contains("largest 0.0%") {
            format!(
                "<a href=\"{scenario}/{stem}.diff.png\"><img src=\"{scenario}/{stem}.diff.png\" alt=\"difference\"></a>"
            )
        } else {
            "<div class=none>no difference image</div>".to_owned()
        };
        let _ = write!(
            html,
            "<section class=\"shot {status}\" data-id=\"{id}\">\n<h2>{id} <span class=status>{label}</span></h2>\n\\n             <p>{about}<br>{summary}</p>\n<div class=images>\n<figure>{golden}<figcaption>golden</figcaption></figure>\n\\n             <figure><a href=\"{scenario}/{stem}.png\"><img src=\"{scenario}/{stem}.png\" alt=\"shot\"></a>\\n             <figcaption>this run</figcaption></figure>\n<figure>{diff}<figcaption>difference, 4x</figcaption></figure>\n\\n             </div>\n<button class=accept>Accept as the golden image</button>\n</section>\n",
            status = if matches!(shot.status, Status::Passed | Status::Updated) { "passing" } else { "failing" },
            label = shot.status.label(),
            about = escape(&shot.about),
            summary = escape(&shot.summary),
        );
    }
    html.push_str(SCRIPT);
    html
}

const STYLE: &str = r#"<!doctype html>
<html lang=en>
<meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Golden images</title>
<style>
:root { color-scheme: light dark; --page: #fafafa; --text: #1d1d1f; --muted: #5f6368; --card: #fff; --line: #d9d9de; --bad: #b3261e; --good: #1e7d3a; }
@media (prefers-color-scheme: dark) { :root { --page: #161618; --text: #e8e8ea; --muted: #a0a0a8; --card: #202024; --line: #3a3a40; --bad: #f28b82; --good: #81c995; } }
body { background: var(--page); color: var(--text); font: 15px/1.45 system-ui, sans-serif; margin: 0 auto; max-width: 1400px; padding: 16px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.shot { background: var(--card); border: 1px solid var(--line); border-radius: 8px; margin: 16px 0; padding: 12px 16px; }
.shot h2 { font-size: 16px; margin: 0; }
.shot p { color: var(--muted); margin: 4px 0 8px; }
.status { font-weight: normal; margin-left: 8px; }
.failing .status { color: var(--bad); }
.passing .status, .accepted .status { color: var(--good); }
.images { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
figure { margin: 0; }
figure img { border: 1px solid var(--line); display: block; height: auto; width: 100%; }
figcaption { color: var(--muted); font-size: 13px; margin-top: 2px; }
.none { align-items: center; aspect-ratio: 3 / 2; border: 1px dashed var(--line); color: var(--muted); display: flex; justify-content: center; }
button { font: inherit; margin-top: 10px; padding: 4px 12px; }
body.only-failing .passing { display: none; }
</style>
"#;

const SCRIPT: &str = r#"<script>
const offline = location.protocol === "file:";
document.getElementById("offline").hidden = !offline;
const failing = document.getElementById("failing");
const filter = () => document.body.classList.toggle("only-failing", failing.checked);
failing.addEventListener("change", filter);
filter();
for (const section of document.querySelectorAll(".shot")) {
  const button = section.querySelector(".accept");
  if (offline) { button.remove(); continue; }
  button.addEventListener("click", async () => {
    button.disabled = true;
    const response = await fetch("accept", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ shots: [section.dataset.id] }) });
    const status = section.querySelector(".status");
    if (response.ok) {
      status.textContent = "accepted";
      // Still shown under the filter, until the page is loaded again.
      section.classList.replace("failing", "accepted");
      for (const image of section.querySelectorAll("img")) image.src = image.src.split("?")[0] + "?" + Date.now();
    } else {
      status.textContent = "not accepted: " + await response.text();
      button.disabled = false;
    }
  });
}
</script>
"#;

/// Writes the page among the shots in `out` (and the run's results beside
/// it, for [`last`]), and returns its path.
pub fn write(out: &Path, shots: &[Shot], threshold: f64) -> Result<PathBuf> {
    std::fs::create_dir_all(out)?;
    let path = out.join(PAGE);
    std::fs::write(&path, page(shots, threshold)).with_context(|| format!("writing {}", path.display()))?;
    let results = serde_json::to_string_pretty(&Results { threshold, shots: shots.to_vec() })?;
    std::fs::write(out.join(RESULTS), results)?;
    Ok(path)
}

/// The shots of the run that last wrote its page in `out`, its page written
/// again (by this pen) for them.
pub fn last(out: &Path) -> Result<Vec<Shot>> {
    let path = out.join(RESULTS);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {} (`pen test` writes it with each run)", path.display()))?;
    let results: Results = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    write(out, &results.shots, results.threshold)?;
    Ok(results.shots)
}

/// What the review server serves and may change.
struct Review {
    /// The shots, the page and the golden copies beside them.
    out: PathBuf,
    golden: PathBuf,
    shots: Vec<Shot>,
    port: u16,
}

impl Review {
    fn is_own(&self, host: &str) -> bool {
        ["127.0.0.1", "localhost", "[::1]"]
            .iter()
            .any(|name| host.eq_ignore_ascii_case(&format!("{name}:{}", self.port)))
    }
}

#[derive(Deserialize)]
struct Accepting {
    shots: Vec<String>,
}

/// Makes each of the shots named in `body` its golden image (and the copy
/// beside it); returns them.
fn accept(review: &Review, body: &[u8]) -> Result<Vec<String>> {
    let accepting: Accepting = serde_json::from_slice(body).context("expected {\"shots\": [\"SCENARIO/NAME.png\"]}")?;
    for id in &accepting.shots {
        let shot = review
            .shots
            .iter()
            .find(|shot| shot.id() == *id)
            .with_context(|| format!("{id} is no shot of this run"))?;
        let actual = review.out.join(&shot.scenario).join(&shot.file);
        let golden = review.golden.join(&shot.scenario).join(&shot.file);
        std::fs::create_dir_all(golden.parent().expect("a scenario's directory"))?;
        std::fs::copy(&actual, &golden)
            .with_context(|| format!("copying {} to {}", actual.display(), golden.display()))?;
        std::fs::copy(&actual, review.out.join(&shot.scenario).join(format!("{}.golden.png", shot.stem())))?;
        eprintln!("review: accepted {id} as {}", golden.display());
    }
    Ok(accepting.shots)
}

fn route(review: &Review, request: &Request) -> Reply {
    if !request.header("host").is_some_and(|host| review.is_own(host)) {
        return Reply::text("403 Forbidden", "this server answers to its loopback address and port only");
    }
    if request.path == "/accept" {
        if request.method != "POST" {
            return Reply::text("405 Method Not Allowed", "accept takes a POST");
        }
        let own = request
            .header("origin")
            .is_none_or(|origin| origin.strip_prefix("http://").is_some_and(|host| review.is_own(host)));
        let json = request
            .header("content-type")
            .and_then(|kind| kind.split(';').next())
            .is_some_and(|kind| kind.trim().eq_ignore_ascii_case("application/json"));
        if !own || !json {
            return Reply::text("403 Forbidden", "accepting takes JSON from this server's own page");
        }
        return match accept(review, &request.body) {
            Ok(accepted) => Reply::text("200 OK", format!("accepted {}", accepted.join(", "))),
            Err(error) => Reply::text("400 Bad Request", format!("{error:#}")),
        };
    }
    if request.method != "GET" {
        return Reply::text("405 Method Not Allowed", "GET, or POST to /accept");
    }
    let Ok(relative) = decode_path(&request.path) else {
        return Reply::text("400 Bad Request", "invalid path");
    };
    let relative = if relative.as_os_str().is_empty() { PathBuf::from(PAGE) } else { relative };
    let page = relative == Path::new(PAGE);
    let png = relative.extension().is_some_and(|extension| extension == "png");
    let file =
        review.out.join(&relative).canonicalize().ok().filter(|file| (page || png) && file.starts_with(&review.out));
    match file.and_then(|file| std::fs::read(file).ok()) {
        Some(body) => {
            Reply { status: "200 OK", content_type: if page { "text/html; charset=utf-8" } else { "image/png" }, body }
        }
        None => Reply::text("404 Not Found", "file unavailable"),
    }
}

/// Serves the page on the loopback address (`port`, or the first free one
/// from 8000) until stopped.
pub fn serve(out: &Path, golden: &Path, shots: Vec<Shot>, port: Option<u16>) -> Result<()> {
    let listener = match port {
        Some(port) => TcpListener::bind(("127.0.0.1", port))?,
        None => (8000..8016)
            .find_map(|port| TcpListener::bind(("127.0.0.1", port)).ok())
            .context("no free port in 8000..8015")?,
    };
    ensure!(!shots.is_empty(), "the run took no shots to review");
    let review = Arc::new(Review {
        out: out.canonicalize()?,
        golden: golden.to_owned(),
        shots,
        port: listener.local_addr()?.port(),
    });
    eprintln!("review: http://{}/ (Ctrl+C to stop)", listener.local_addr()?);
    // A thread a connection: browsers open some they send nothing on.
    for socket in listener.incoming() {
        let mut socket = socket?;
        let review = review.clone();
        std::thread::spawn(move || {
            let answered = (|| {
                socket.set_read_timeout(Some(Duration::from_secs(5)))?;
                socket.set_write_timeout(Some(Duration::from_secs(5)))?;
                let request = read_request(&mut socket)?;
                respond(&mut socket, &route(&review, &request))
            })();
            if let Err(error) = answered {
                eprintln!("review: {error:#}");
            }
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shot(scenario: &str, status: Status) -> Shot {
        Shot {
            scenario: scenario.into(),
            about: "the <rock> & salt".into(),
            file: "play.png".into(),
            status,
            summary: "RMSE 2.0000%, largest 9.0%, 1.000% of pixels visibly changed".into(),
        }
    }

    #[test]
    fn the_page_shows_each_shot_beside_its_golden_image() {
        let html =
            page(&[shot("play", Status::Failed), shot("menu", Status::Passed), shot("new", Status::Missing)], 0.01);
        assert!(html.contains("3 shots, 2 not passing (RMSE threshold 1.00%)"), "{html}");
        assert!(html.contains("data-id=\"play/play.png\""));
        assert!(html.contains("src=\"play/play.golden.png\"") && html.contains("src=\"play/play.diff.png\""));
        assert!(html.contains("the &lt;rock&gt; &amp; salt"));
        assert!(html.contains("<section class=\"shot passing\" data-id=\"menu/play.png\">"));
        assert!(!html.contains("src=\"new/play.golden.png\""));
        assert!(html.contains("id=failing checked"));
    }

    /// A run's shots and golden images in a fresh directory.
    fn review(name: &str) -> Review {
        let root = std::env::temp_dir().join(format!("pen-review-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (out, golden) = (root.join("out"), root.join("golden"));
        std::fs::create_dir_all(out.join("play")).unwrap();
        std::fs::create_dir_all(golden.join("play")).unwrap();
        std::fs::write(out.join("play/play.png"), "new").unwrap();
        std::fs::write(out.join("play/play.golden.png"), "old").unwrap();
        std::fs::write(golden.join("play/play.png"), "old").unwrap();
        std::fs::write(root.join("secret.png"), "secret").unwrap();
        write(&out, &[shot("play", Status::Failed)], 0.01).unwrap();
        assert_eq!(last(&out).unwrap(), [shot("play", Status::Failed)]);
        Review { out: out.canonicalize().unwrap(), golden, shots: vec![shot("play", Status::Failed)], port: 8123 }
    }

    fn request(method: &str, path: &str, headers: &[(&str, &str)], body: &str) -> Request {
        let mut all = vec![("Host".to_owned(), "127.0.0.1:8123".to_owned())];
        all.extend(headers.iter().map(|(key, value)| ((*key).to_owned(), (*value).to_owned())));
        Request { method: method.into(), path: path.into(), headers: all, body: body.as_bytes().to_vec() }
    }

    #[test]
    fn the_server_serves_the_page_and_its_images_only() {
        let review = review("serve");
        let answer = |request: &Request| {
            let reply = route(&review, request);
            (reply.status, String::from_utf8_lossy(&reply.body).into_owned())
        };
        assert!(answer(&request("GET", "/", &[], "")).1.contains("<h1>Golden images</h1>"));
        assert_eq!(answer(&request("GET", "/play/play.png", &[], "")), ("200 OK", "new".into()));
        assert_eq!(answer(&request("GET", "/../secret.png", &[], "")).0, "400 Bad Request");
        assert_eq!(answer(&request("GET", "/play/missing.png", &[], "")).0, "404 Not Found");
        let mut elsewhere = request("GET", "/", &[], "");
        elsewhere.headers = vec![("Host".into(), "evil.example:8123".into())];
        assert_eq!(answer(&elsewhere).0, "403 Forbidden");
        std::fs::remove_dir_all(review.out.parent().unwrap()).unwrap();
    }

    #[test]
    fn accepting_makes_a_shot_its_golden_image() {
        let review = review("accept");
        let json = ("Content-Type", "application/json");
        let body = r#"{"shots": ["play/play.png"]}"#;
        let golden = review.golden.join("play/play.png");
        for refused in [
            request("POST", "/accept", &[("Origin", "http://evil.example"), json], body),
            request("POST", "/accept", &[("Content-Type", "text/plain")], body),
            request("GET", "/accept", &[json], body),
        ] {
            assert_ne!(route(&review, &refused).status, "200 OK");
            assert_eq!(std::fs::read_to_string(&golden).unwrap(), "old");
        }
        let unknown = route(&review, &request("POST", "/accept", &[json], r#"{"shots": ["../../etc/passwd"]}"#));
        assert_eq!(unknown.status, "400 Bad Request");
        let origin = ("Origin", "http://localhost:8123");
        let reply = route(&review, &request("POST", "/accept", &[origin, json], body));
        assert_eq!((reply.status, reply.body.as_slice()), ("200 OK", b"accepted play/play.png".as_slice()));
        assert_eq!(std::fs::read_to_string(&golden).unwrap(), "new");
        assert_eq!(std::fs::read_to_string(review.out.join("play/play.golden.png")).unwrap(), "new");
        std::fs::remove_dir_all(review.out.parent().unwrap()).unwrap();
    }
}
