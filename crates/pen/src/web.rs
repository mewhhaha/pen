//! Loopback development server for pen's browser front end.
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use notify::Watcher;

use crate::project::Project;

pub fn serve(project: Project, port: Option<u16>) -> Result<()> {
    let out = project.root.join(".pen/web");
    crate::project::export_web(&project, &out)?;
    let _game = project.watch(&out)?;
    let _devtools = project.build_devtools(&out, true)?;
    let listener = if let Some(port) = port {
        TcpListener::bind(("127.0.0.1", port))?
    } else {
        (8000..8016)
            .find_map(|port| TcpListener::bind(("127.0.0.1", port)).ok())
            .context("no free port in 8000..8015")?
    };
    let revision = Arc::new(AtomicU64::new(1));
    let changed = revision.clone();
    let root = project.root.clone();
    let build = out.clone();
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
        if let Ok(event) = event
            && !matches!(event.kind, notify::EventKind::Access(_))
            && event.paths.iter().any(|path| {
                path == &build.join("game.wasm")
                    || path == &build.join("devtools.wasm")
                    || (!path.starts_with(root.join(".pen"))
                        && !path.starts_with(root.join(".git")))
            })
        {
            changed.fetch_add(1, Ordering::Relaxed);
        }
    })?;
    watcher.watch(&project.root, notify::RecursiveMode::Recursive)?;
    eprintln!(
        "pen web: http://{}/?dev (Ctrl+C to stop)",
        listener.local_addr()?
    );
    let root = Arc::new(project.root);
    let out = Arc::new(out);
    for socket in listener.incoming() {
        let socket = socket?;
        let root = root.clone();
        let out = out.clone();
        let revision = revision.clone();
        std::thread::spawn(move || {
            if let Err(error) = request(socket, &root, &out, revision.load(Ordering::Relaxed)) {
                eprintln!("pen web: {error:#}");
            }
        });
    }
    Ok(())
}

fn decode_path(path: &str) -> Result<PathBuf> {
    let mut bytes = Vec::new();
    let mut input = path.bytes();
    while let Some(byte) = input.next() {
        if byte == b'%' {
            let a = input.next().context("incomplete URL escape")?;
            let b = input.next().context("incomplete URL escape")?;
            let hex = |byte: u8| char::from(byte).to_digit(16).context("invalid URL escape");
            bytes.push((hex(a)? * 16 + hex(b)?) as u8);
        } else {
            bytes.push(byte);
        }
    }
    let text = String::from_utf8(bytes)?;
    ensure!(!text.contains(['\0', '\\']), "invalid URL path");
    let path = PathBuf::from(text.trim_start_matches('/'));
    ensure!(
        path.components()
            .all(|part| matches!(part, Component::Normal(_) | Component::CurDir)),
        "invalid URL path"
    );
    Ok(path)
}

fn response(socket: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) -> Result<()> {
    write!(
        socket,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nCross-Origin-Opener-Policy: same-origin\r\nCross-Origin-Embedder-Policy: require-corp\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    socket.write_all(body)?;
    Ok(())
}

fn request(mut socket: TcpStream, root: &Path, out: &Path, revision: u64) -> Result<()> {
    socket.set_read_timeout(Some(Duration::from_secs(5)))?;
    socket.set_write_timeout(Some(Duration::from_secs(5)))?;
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
    let headers = std::str::from_utf8(&bytes[..end])?;
    let mut lines = headers.lines();
    let mut first = lines
        .next()
        .context("missing request line")?
        .split_whitespace();
    let method = first.next().context("missing method")?.to_owned();
    let url = first
        .next()
        .context("missing URL")?
        .split('?')
        .next()
        .unwrap_or("/")
        .to_owned();
    let length = lines
        .filter_map(|line| line.split_once(':'))
        .find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
        .map(|(_, value)| value.trim().parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    ensure!(length <= 64 * 1024, "request body exceeds 64 KiB");
    while bytes.len() < end + length {
        let mut chunk = [0; 4096];
        let count = socket.read(&mut chunk)?;
        ensure!(count != 0, "incomplete request body");
        bytes.extend_from_slice(&chunk[..count]);
    }
    if method == "POST" && url == "/__pen/bake" {
        let result = bake(root, &bytes[end..end + length]);
        return match result {
            Ok(text) => response(
                &mut socket,
                "200 OK",
                "text/plain; charset=utf-8",
                text.as_bytes(),
            ),
            Err(error) => response(
                &mut socket,
                "400 Bad Request",
                "text/plain; charset=utf-8",
                format!("{error:#}").as_bytes(),
            ),
        };
    }
    if method != "GET" {
        return response(
            &mut socket,
            "405 Method Not Allowed",
            "text/plain",
            b"GET or POST bake required",
        );
    }
    if url == "/__pen/revision" {
        return response(
            &mut socket,
            "200 OK",
            "text/plain",
            revision.to_string().as_bytes(),
        );
    }
    let relative = match decode_path(&url) {
        Ok(path) => path,
        Err(_) => {
            return response(
                &mut socket,
                "400 Bad Request",
                "text/plain",
                b"invalid path",
            );
        }
    };
    let file = if let Ok(asset) = relative.strip_prefix("assets") {
        root.join("src/assets").join(asset)
    } else if relative == Path::new("catalog.json") {
        root.join("src/assets/catalog.json")
    } else if relative == Path::new("scene.bin") {
        root.join("src/main.scene")
    } else if relative.as_os_str().is_empty() {
        out.join("index.html")
    } else {
        out.join(relative)
    };
    // Symlinked files must stay inside the served roots.
    let canonical = match file.canonicalize() {
        Ok(file) if file.starts_with(out) || file.starts_with(root.join("src")) => file,
        _ => {
            return response(
                &mut socket,
                "404 Not Found",
                "text/plain",
                b"file unavailable",
            );
        }
    };
    let body = match std::fs::read(&canonical) {
        Ok(body) => body,
        Err(_) => {
            return response(
                &mut socket,
                "404 Not Found",
                "text/plain",
                b"file unavailable",
            );
        }
    };
    let content_type = match canonical
        .extension()
        .and_then(|extension| extension.to_str())
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
    response(&mut socket, "200 OK", content_type, &body)
}

fn bake(root: &Path, body: &[u8]) -> Result<String> {
    let values: Vec<(u32, f32)> = serde_json::from_slice(body)?;
    let declarations = crate::overlay::tunables::load(root).map_err(anyhow::Error::msg)?;
    let mut named = Vec::new();
    for (id, value) in values {
        let Some(declaration) = declarations.iter().find(|declaration| declaration.id == id) else {
            bail!("unknown tunable {id}");
        };
        named.push((declaration.name.clone(), value));
    }
    crate::overlay::tunables::bake(root, &named)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn static_paths_cannot_escape_the_served_roots() {
        assert_eq!(
            decode_path("/assets/a%20b.png").unwrap(),
            Path::new("assets/a b.png")
        );
        for path in ["/../secret", "/%2e%2e/secret", "/a%00b", "/a%5cb", "/a%GG"] {
            assert!(decode_path(path).is_err(), "{path}");
        }
    }
}
