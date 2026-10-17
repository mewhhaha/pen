//! Loopback development server for pen's browser front end: the browser
//! host and the programs' builds from `.pen/web`, the checkout's assets and
//! scene as they are, and a revision that moves on whenever one of those
//! changes, for the page to reload.
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use notify::{RecursiveMode, Watcher};

use crate::project::Project;

/// What the server serves, all canonical paths.
struct Site {
    /// The checkout.
    root: PathBuf,
    /// The browser host and the programs' builds.
    out: PathBuf,
    /// The port it listens on, which requests' Host and Origin must name.
    port: u16,
}

impl Site {
    fn assets(&self) -> PathBuf {
        self.root.join("src/assets")
    }

    fn scene(&self) -> PathBuf {
        self.root.join("src/main.scene")
    }

    /// Whether `host` (a Host header, or an Origin without its scheme) is
    /// this server: a loopback name and its port, never a name a page's
    /// own domain could be made to resolve to.
    fn is_own(&self, host: &str) -> bool {
        ["127.0.0.1", "localhost", "[::1]"]
            .iter()
            .any(|name| host.eq_ignore_ascii_case(&format!("{name}:{}", self.port)))
    }

    /// Whether a change to `path` reloads the page: a program's new build,
    /// or an asset or the scene, but not an editor's or Blender's temporary
    /// files beside them.
    fn reloads(&self, path: &Path) -> bool {
        if path == self.out.join("game.wasm") || path == self.out.join("devtools.wasm") {
            return true;
        }
        (path == self.scene() || path.starts_with(self.assets()))
            && path.file_name().is_some_and(|name| !temporary(&name.to_string_lossy()))
    }
}

/// Editors' swap, backup and lock files, and Blender's files, backups and
/// saves in progress: never what the page loads.
fn temporary(name: &str) -> bool {
    let blender =
        name.rsplit_once(".blend").is_some_and(|(_, rest)| rest == "@" || rest.chars().all(|c| c.is_ascii_digit()));
    blender
        || name.starts_with(['.', '#'])
        || name.ends_with(['~', '#'])
        || [".tmp", ".swp", ".swo", ".swx", ".bak", ".orig", ".part"].iter().any(|suffix| name.ends_with(suffix))
        // Vim's test that it may write in the directory.
        || name == "4913"
}

pub fn serve(project: Project, port: Option<u16>) -> Result<()> {
    let out = project.root.join(".pen/web");
    // A fresh browser host; the programs build into it below, and the
    // assets and scene are served from the checkout.
    let staging = crate::staging::Staging::new(&project.root, &out)?;
    crate::project::web_host(&staging.directory)?;
    crate::project::Manifest { game: project.identity.clone(), runtime: None }.write(&staging.directory)?;
    staging.publish()?;
    let _game = project.watch(&out)?;
    let mut devtools = project.build_devtools(&out)?;
    // The page's first load has both programs.
    if let Some(build) = &mut devtools
        && !build.wait()?
    {
        eprintln!("devtools: its program did not compile");
    }
    let listener = if let Some(port) = port {
        TcpListener::bind(("127.0.0.1", port))?
    } else {
        (8000..8016)
            .find_map(|port| TcpListener::bind(("127.0.0.1", port)).ok())
            .context("no free port in 8000..8015")?
    };
    let site =
        Arc::new(Site { root: project.root.clone(), out: out.canonicalize()?, port: listener.local_addr()?.port() });
    let revision = Arc::new(AtomicU64::new(1));
    let (changed, watched) = (revision.clone(), site.clone());
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
        if let Ok(event) = event
            && !matches!(event.kind, notify::EventKind::Access(_))
            && event.paths.iter().any(|path| watched.reloads(path))
        {
            changed.fetch_add(1, Ordering::Relaxed);
        }
    })?;
    watcher.watch(&site.out, RecursiveMode::NonRecursive)?;
    watcher.watch(&site.assets(), RecursiveMode::Recursive)?;
    // The scene, through its directory: a save replaces the file.
    watcher.watch(&site.root.join("src"), RecursiveMode::NonRecursive)?;
    eprintln!("pen web: http://{}/?dev (Ctrl+C to stop)", listener.local_addr()?);
    for socket in listener.incoming() {
        let mut socket = socket?;
        let (site, revision) = (site.clone(), revision.clone());
        std::thread::spawn(move || {
            let answered = (|| {
                socket.set_read_timeout(Some(Duration::from_secs(5)))?;
                socket.set_write_timeout(Some(Duration::from_secs(5)))?;
                let request = read_request(&mut socket)?;
                respond(&mut socket, &route(&site, &request, revision.load(Ordering::Relaxed)))
            })();
            if let Err(error) = answered {
                eprintln!("pen web: {error:#}");
            }
        });
    }
    Ok(())
}

/// An HTTP request: its method, its path without the query, its headers
/// and its body.
#[derive(Debug, Default)]
pub(crate) struct Request {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) headers: Vec<(String, String)>,
    pub(crate) body: Vec<u8>,
}

impl Request {
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(key, _)| key.eq_ignore_ascii_case(name)).map(|(_, value)| value.as_str())
    }
}

pub(crate) fn read_request(socket: &mut impl Read) -> Result<Request> {
    let mut bytes = Vec::new();
    let end = loop {
        if let Some(at) = bytes.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
            break at + 4;
        }
        ensure!(bytes.len() < 16 * 1024, "HTTP headers exceed 16 KiB");
        let mut chunk = [0; 4096];
        let count = socket.read(&mut chunk)?;
        ensure!(count != 0, "incomplete request");
        bytes.extend_from_slice(&chunk[..count]);
    };
    let text = std::str::from_utf8(&bytes[..end])?;
    let mut lines = text.lines();
    let mut first = lines.next().context("missing request line")?.split_whitespace();
    let method = first.next().context("missing method")?.to_owned();
    let path = first.next().context("missing URL")?.split('?').next().unwrap_or("/").to_owned();
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim().to_owned(), value.trim().to_owned()))
        .collect();
    let mut request = Request { method, path, headers, body: Vec::new() };
    let length = request.header("content-length").map(str::parse::<usize>).transpose()?.unwrap_or(0);
    ensure!(length <= 64 * 1024, "request body exceeds 64 KiB");
    while bytes.len() < end + length {
        let mut chunk = [0; 4096];
        let count = socket.read(&mut chunk)?;
        ensure!(count != 0, "incomplete request body");
        bytes.extend_from_slice(&chunk[..count]);
    }
    request.body = bytes[end..end + length].to_vec();
    Ok(request)
}

/// A response: its status line's code and reason, its type and its body.
#[derive(Debug)]
pub(crate) struct Reply {
    pub(crate) status: &'static str,
    pub(crate) content_type: &'static str,
    pub(crate) body: Vec<u8>,
}

impl Reply {
    pub(crate) fn text(status: &'static str, text: impl Into<String>) -> Reply {
        Reply { status, content_type: "text/plain; charset=utf-8", body: text.into().into_bytes() }
    }
}

pub(crate) fn respond(socket: &mut TcpStream, reply: &Reply) -> Result<()> {
    write!(
        socket,
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nCross-Origin-Opener-Policy: same-origin\r\nCross-Origin-Embedder-Policy: require-corp\r\nConnection: close\r\n\r\n",
        reply.status,
        reply.content_type,
        reply.body.len()
    )?;
    socket.write_all(&reply.body)?;
    Ok(())
}

pub(crate) fn decode_path(path: &str) -> Result<PathBuf> {
    let mut bytes = Vec::new();
    let mut input = path.bytes();
    while let Some(byte) = input.next() {
        if byte == b'%' {
            let a = input.next().context("incomplete URL escape")?;
            let b = input.next().context("incomplete URL escape")?;
            let hex = |byte: u8| char::from(byte).to_digit(16).context("invalid URL escape");
            bytes.push(u8::try_from(hex(a)? * 16 + hex(b)?)?);
        } else {
            bytes.push(byte);
        }
    }
    let text = String::from_utf8(bytes)?;
    ensure!(!text.contains(['\0', '\\']), "invalid URL path");
    let path = PathBuf::from(text.trim_start_matches('/'));
    ensure!(path.components().all(|part| matches!(part, Component::Normal(_) | Component::CurDir)), "invalid URL path");
    Ok(path)
}

/// What the server answers `request` with, the page's revision being
/// `revision`.
fn route(site: &Site, request: &Request, revision: u64) -> Reply {
    // A page elsewhere can have its own domain resolve to this machine, and
    // would then be served as this server: its Host gives it away.
    if !request.header("host").is_some_and(|host| site.is_own(host)) {
        return Reply::text("403 Forbidden", "this server answers to its loopback address and port only");
    }
    if request.path == "/__pen/bake" {
        if request.method != "POST" {
            return Reply::text("405 Method Not Allowed", "bake takes a POST");
        }
        if let Err(refusal) = bakeable(site, request) {
            return Reply::text("403 Forbidden", format!("{refusal:#}"));
        }
        return match bake(&site.root, &request.body) {
            Ok(text) => Reply::text("200 OK", text),
            Err(error) => Reply::text("400 Bad Request", format!("{error:#}")),
        };
    }
    if request.method != "GET" {
        return Reply::text("405 Method Not Allowed", "GET, or POST to /__pen/bake");
    }
    if request.path == "/__pen/revision" {
        return Reply::text("200 OK", revision.to_string());
    }
    let Ok(relative) = decode_path(&request.path) else {
        return Reply::text("400 Bad Request", "invalid path");
    };
    let file = if let Ok(asset) = relative.strip_prefix("assets") {
        site.assets().join(asset)
    } else if relative == Path::new("catalog.json") {
        site.assets().join("catalog.json")
    } else if relative == Path::new("scene.bin") {
        site.scene()
    } else if relative.as_os_str().is_empty() {
        site.out.join("index.html")
    } else {
        site.out.join(&relative)
    };
    // Symlinked files must stay inside the served roots.
    let served =
        file.canonicalize().ok().filter(|file| file.starts_with(&site.out) || file.starts_with(site.root.join("src")));
    let Some(body) = served.as_ref().and_then(|file| std::fs::read(file).ok()) else {
        // A game that declares no saved types, as the page asks whatever.
        if relative == Path::new("assets").join(pen_runtime::saved::SAVED_TYPES_FILE) {
            return Reply {
                status: "200 OK",
                content_type: "application/json",
                body: pen_runtime::saved::NO_SAVED_TYPES.into(),
            };
        }
        return Reply::text("404 Not Found", "file unavailable");
    };
    let content_type = match served.as_ref().and_then(|file| file.extension()).and_then(|extension| extension.to_str())
    {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript",
        Some("wasm") => "application/wasm",
        Some("json") => "application/json",
        Some("png") => "image/png",
        Some("ogg") => "audio/ogg",
        Some("wav") => "audio/wav",
        _ => "application/octet-stream",
    };
    Reply { status: "200 OK", content_type, body }
}

/// Whether a bake may write the checkout's tunables: it comes from this
/// server's own pages (an Origin, which browsers send with a POST, naming
/// it), as JSON, which no other site's page can send here without asking
/// first, and being refused.
fn bakeable(site: &Site, request: &Request) -> Result<()> {
    if let Some(origin) = request.header("origin") {
        let own = origin.strip_prefix("http://").is_some_and(|host| site.is_own(host));
        ensure!(own, "a bake must come from this server's own page, not {origin}");
    }
    let json = request
        .header("content-type")
        .and_then(|kind| kind.split(';').next())
        .is_some_and(|kind| kind.trim().eq_ignore_ascii_case("application/json"));
    ensure!(json, "a bake must be JSON (Content-Type: application/json)");
    Ok(())
}

fn bake(root: &Path, body: &[u8]) -> Result<String> {
    let values: Vec<(u32, f32)> = serde_json::from_slice(body)?;
    let declarations = crate::tunables::load(root).map_err(anyhow::Error::msg)?;
    let mut named = Vec::new();
    for (id, value) in values {
        let Some(declaration) = declarations.iter().find(|declaration| declaration.id == id) else {
            bail!("unknown tunable {id}");
        };
        named.push((declaration.name.clone(), value));
    }
    crate::tunables::bake(root, &named)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_paths_cannot_escape_the_served_roots() {
        assert_eq!(decode_path("/assets/a%20b.png").unwrap(), Path::new("assets/a b.png"));
        for path in ["/../secret", "/%2e%2e/secret", "/a%00b", "/a%5cb", "/a%GG"] {
            assert!(decode_path(path).is_err(), "{path}");
        }
    }

    /// A checkout with an asset, a scene, tunables and a built page.
    fn site(name: &str) -> Site {
        let root = crate::project::temporary_dir().unwrap().join(format!("web-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src/assets")).unwrap();
        std::fs::create_dir_all(root.join(".pen/web")).unwrap();
        std::fs::write(root.join("src/assets/catalog.json"), r#"{"meshes": []}"#).unwrap();
        std::fs::write(root.join("src/assets/a b.png"), "png").unwrap();
        std::fs::write(root.join("src/main.scene"), "scene").unwrap();
        std::fs::write(root.join("src/main.blot"), "source").unwrap();
        let tunables =
            r#"{ "tunables": [{ "name": "camera.zoom_speed", "default": 0.001, "min": 0.0002, "max": 0.005 }] }"#;
        std::fs::write(root.join("src/tunables.json"), tunables).unwrap();
        std::fs::write(root.join(".pen/web/index.html"), "<html>").unwrap();
        std::fs::write(root.join(".pen/web/game.wasm"), "wasm").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(root.join("pen.json"), root.join(".pen/web/escape.json")).unwrap();
        std::fs::write(root.join("pen.json"), "{}").unwrap();
        let root = root.canonicalize().unwrap();
        Site { out: root.join(".pen/web"), root, port: 8123 }
    }

    fn get(path: &str) -> Request {
        Request {
            method: "GET".into(),
            path: path.into(),
            headers: vec![("Host".into(), "127.0.0.1:8123".into())],
            body: Vec::new(),
        }
    }

    fn bake_request(headers: &[(&str, &str)], body: &str) -> Request {
        let mut request = get("/__pen/bake");
        request.method = "POST".into();
        request.headers.extend(headers.iter().map(|(key, value)| (key.to_string(), value.to_string())));
        request.body = body.as_bytes().to_vec();
        request
    }

    #[test]
    fn requests_route_to_the_build_the_assets_and_the_scene() {
        let site = site("route");
        let answer = |request: &Request| {
            let reply = route(&site, request, 7);
            (reply.status, reply.content_type, String::from_utf8_lossy(&reply.body).into_owned())
        };
        assert_eq!(answer(&get("/")), ("200 OK", "text/html; charset=utf-8", "<html>".into()));
        assert_eq!(answer(&get("/game.wasm")).2, "wasm");
        assert_eq!(answer(&get("/assets/a%20b.png")), ("200 OK", "image/png", "png".into()));
        assert_eq!(answer(&get("/catalog.json")).2, r#"{"meshes": []}"#);
        assert_eq!(answer(&get("/scene.bin")).2, "scene");
        assert_eq!(answer(&get("/__pen/revision")).2, "7");
        assert_eq!(answer(&get("/missing.js")).0, "404 Not Found");
        // A game without saved types: the page asks whatever.
        let saved = ("200 OK", "application/json", pen_runtime::saved::NO_SAVED_TYPES.into());
        assert_eq!(answer(&get("/assets/saved.json")), saved);
        assert_eq!(answer(&get("/../src/main.blot")).0, "400 Bad Request");
        // The checkout's sources are not served, not even through a link.
        assert_eq!(answer(&get("/escape.json")).0, "404 Not Found");
        let mut post = get("/game.wasm");
        post.method = "POST".into();
        assert_eq!(answer(&post).0, "405 Method Not Allowed");
        assert_eq!(answer(&bake_request(&[], "[]")).0, "403 Forbidden");
        // A page whose own domain resolves here is not this server.
        let mut rebound = get("/");
        rebound.headers = vec![("Host".into(), "attacker.example:8123".into())];
        assert_eq!(answer(&rebound).0, "403 Forbidden");
        rebound.headers.clear();
        assert_eq!(answer(&rebound).0, "403 Forbidden");
        let mut localhost = get("/");
        localhost.headers = vec![("host".into(), "localhost:8123".into())];
        assert_eq!(answer(&localhost).0, "200 OK");
        std::fs::remove_dir_all(&site.root).unwrap();
    }

    #[test]
    fn bakes_come_from_the_servers_own_page_as_json() {
        let site = site("bake");
        let tunables = site.root.join("src/tunables.json");
        let original = std::fs::read_to_string(&tunables).unwrap();
        let values = "[[11175412, 0.002]]";
        let json = ("Content-Type", "application/json");
        for (headers, refusal) in [
            (vec![("Origin", "http://evil.example"), json], "own page"),
            (vec![("Origin", "http://127.0.0.1:9999"), json], "own page"),
            (vec![("Origin", "null"), json], "own page"),
            (vec![("Origin", "http://127.0.0.1:8123"), ("Content-Type", "text/plain")], "JSON"),
            (vec![("Origin", "http://127.0.0.1:8123")], "JSON"),
        ] {
            let reply = route(&site, &bake_request(&headers, values), 1);
            let body = String::from_utf8(reply.body).unwrap();
            assert_eq!(reply.status, "403 Forbidden", "{headers:?}");
            assert!(body.contains(refusal), "{body}");
            assert_eq!(std::fs::read_to_string(&tunables).unwrap(), original);
        }
        let unknown = route(&site, &bake_request(&[json], "[[1, 2.0]]"), 1);
        assert_eq!((unknown.status, unknown.body.as_slice()), ("400 Bad Request", b"unknown tunable 1".as_slice()));
        let headers = [("Origin", "http://localhost:8123"), ("Content-Type", "application/json; charset=utf-8")];
        let reply = route(&site, &bake_request(&headers, values), 1);
        assert_eq!(reply.status, "200 OK", "{}", String::from_utf8_lossy(&reply.body));
        assert!(std::fs::read_to_string(&tunables).unwrap().contains("0.002"));
        std::fs::remove_dir_all(&site.root).unwrap();
    }

    #[test]
    fn requests_are_read_with_their_bodies() {
        let raw = b"POST /__pen/bake?x=1 HTTP/1.1\r\nHost: 127.0.0.1:8000\r\nContent-Length: 4\r\nContent-Type: application/json\r\n\r\n[[]]";
        let request = read_request(&mut &raw[..]).unwrap();
        assert_eq!((request.method.as_str(), request.path.as_str()), ("POST", "/__pen/bake"));
        assert_eq!(
            (request.header("content-type"), request.body.as_slice()),
            (Some("application/json"), b"[[]]".as_slice())
        );
        assert!(read_request(&mut &b"GET / HTTP/1.1\r\nContent-Length: 99999999\r\n\r\n"[..]).is_err());
        assert!(read_request(&mut &b"GET / HTTP/1.1\r\n"[..]).is_err());
    }

    #[test]
    fn the_page_reloads_for_builds_assets_and_the_scene_only() {
        let site = Site { root: PathBuf::from("/c"), out: PathBuf::from("/c/.pen/web"), port: 8000 };
        for path in [
            "/c/.pen/web/game.wasm",
            "/c/.pen/web/devtools.wasm",
            "/c/src/assets/rock.png",
            "/c/src/assets/a/b.json",
            "/c/src/main.scene",
        ] {
            assert!(site.reloads(Path::new(path)), "{path}");
        }
        for path in [
            "/c/.pen/web/.game.wasm.1.tmp",
            "/c/target/debug/pen",
            "/c/node_modules/x/index.js",
            "/c/src/game.blot",
            "/c/models/rock.blend1",
            "/c/src/assets/rock.blend",
            "/c/src/assets/rock.blend1",
            "/c/src/assets/rock.blend@",
            "/c/src/assets/.rock.png.swp",
            "/c/src/assets/rock.png~",
            "/c/src/assets/#rock.json#",
            "/c/src/assets/4913",
            "/c/src/assets/.tunables.1-2.tmp",
        ] {
            assert!(!site.reloads(Path::new(path)), "{path}");
        }
    }
}
