//! A session with `blotc serve-project`: one compiler process, opened on one
//! program and asked to build it again and again (blot2's
//! `zig-native/PROJECT_PROTOCOL.md`, version 1, whose reference client is
//! `compiler/zig_project_client.ts`).
use std::collections::HashSet;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use crate::settings::{self, Settings};
use crate::{Build, Diagnostic, Overlay, Program};

/// The protocol's name and version, as the compiler's hello gives them.
const PROTOCOL: &str = "blot-zig-project";
const VERSION: u64 = 1;
/// The largest frame either side sends, and the largest metadata in one.
const MAX_FRAME_BYTES: u64 = 64 * 1024 * 1024;
const MAX_METADATA_BYTES: u64 = 1024 * 1024;
/// What a compiler must be able to do for the plugin to use it.
const REQUIRED: [&str; 3] = ["project-build", "utf8_bytes", "abort-disposes-process"];
/// The most overlays one build carries, and the most bytes one of them has.
const MAX_OVERLAYS: usize = 4096;
const MAX_OVERLAY_BYTES: usize = 16 * 1024 * 1024;
/// The largest revision there is.
const MAX_REVISION: u64 = (1 << 53) - 1;
/// How long the hello and the open may take together.
const STARTUP: Duration = Duration::from_secs(30);
/// How long a closing compiler may take to save its cache and say so.
const CLOSING: Duration = Duration::from_secs(10);
/// The most bytes asked of the child's output at once.
const READ_CHUNK: usize = 1024 * 1024;
/// A Wasm module's first bytes: its magic and version 1.
const WASM_HEADER: [u8; 8] = *b"\0asm\x01\0\0\0";

/// Why reading a child's output gave nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadError {
    /// It closed its output.
    Closed,
    /// Nothing came within the time asked for.
    Timeout,
    Other(String),
}

/// A running compiler process, with piped standard input and output: what
/// pen's `process.child` gives the component, and what tests make natively.
pub trait Child {
    /// Writes all of `bytes` to its standard input.
    fn write(&mut self, bytes: &[u8]) -> Result<(), String>;
    /// Closes its standard input.
    fn close_input(&mut self);
    /// Waits for its standard output, up to `timeout_ms` (forever when
    /// none), and returns what came: at least one byte, at most `max`.
    fn read(&mut self, max: u32, timeout_ms: Option<u32>) -> Result<Vec<u8>, ReadError>;
    /// The last 64 KiB of its standard error, as text.
    fn error_tail(&mut self) -> String;
    /// Waits up to `timeout_ms` for it to exit; its exit code, or none while
    /// it still runs (or when a signal stopped it).
    fn wait(&mut self, timeout_ms: Option<u32>) -> Option<i32>;
    fn kill(&mut self);
}

/// A request ready to send: its kind and id, which its reply must have,
/// and its frame.
struct Request {
    kind: Value,
    id: u64,
    frame: Vec<u8>,
}

/// A reply: its metadata, a JSON object, and its payload.
struct Frame {
    metadata: Map<String, Value>,
    payload: Vec<u8>,
}

/// A compiler process with a program open in it. Any failure of the
/// protocol stops it for good: it is killed, the build that found the
/// failure returns it (with the end of the compiler's standard error), and
/// every later one says it stopped. Dropping a session kills its process;
/// [`Session::close`] lets it save its restart cache first.
pub struct Session<C: Child> {
    /// None once it stopped.
    child: Option<C>,
    /// Why it stopped, when it failed.
    failure: Option<String>,
    root: String,
    entry: String,
    watched: Vec<String>,
    identity: String,
    overlays: bool,
    epoch: String,
    revision: u64,
    next_id: u64,
    /// Output read past the frame that was wanted.
    pending: Vec<u8>,
}

impl<C: Child> Session<C> {
    /// Opens `program`: starts its compiler with `spawn` (given the name it
    /// is granted under, its arguments and its environment, which holds only
    /// `BLOT_CACHE_DIR` = `cache_directory`), checks its hello and opens the
    /// program in it.
    pub fn open(
        program: &Program,
        cache_directory: &str,
        spawn: impl FnOnce(&str, &[String], &[(String, String)]) -> Result<C, String>,
    ) -> Result<Session<C>, String> {
        let settings = Settings::parse(&program.settings, &program.root)?;
        let entry = settings::resolve(&program.root, &program.entry);
        let environment = [("BLOT_CACHE_DIR".to_string(), cache_directory.to_string())];
        let child = spawn(&settings.program, &settings.args(), &environment)
            .map_err(|error| format!("starting the Blot compiler `{}`: {error}", settings.program))?;
        let mut session = Session {
            child: Some(child),
            failure: None,
            root: program.root.clone(),
            watched: settings.watched(&entry),
            entry,
            identity: String::new(),
            overlays: false,
            epoch: String::new(),
            revision: 0,
            next_id: 1,
            pending: Vec::new(),
        };
        match session.start(&settings) {
            Ok(()) => Ok(session),
            Err(error) => Err(session.stop(error)),
        }
    }

    /// Reads the hello and opens the program, within [`STARTUP`].
    fn start(&mut self, settings: &Settings) -> Result<(), String> {
        let deadline = Some(Instant::now() + STARTUP);
        let hello = self.receive(deadline)?;
        let metadata = &hello.metadata;
        let identity = metadata.get("compilerIdentity").and_then(Value::as_str).unwrap_or("");
        if !hello.payload.is_empty()
            || metadata.get("kind") != Some(&json!("hello"))
            || metadata.get("protocol") != Some(&json!(PROTOCOL))
            || metadata.get("version").and_then(Value::as_u64) != Some(VERSION)
            || metadata.get("maxFrameBytes").and_then(Value::as_u64) != Some(MAX_FRAME_BYTES)
            || !settings::is_identity(identity)
        {
            return Err(format!(
                "its hello is not one of {PROTOCOL} version {VERSION}: {}",
                Value::Object(hello.metadata)
            ));
        }
        let capabilities: Vec<&str> = match metadata.get("capabilities") {
            Some(Value::Array(capabilities)) => capabilities.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        if let Some(missing) = REQUIRED.iter().find(|required| !capabilities.contains(required)) {
            return Err(format!("it cannot do {missing} (it can do {})", capabilities.join(", ")));
        }
        if let Some(expected) = settings.expected_identity.as_deref().filter(|expected| *expected != identity) {
            return Err(format!("it is compiler {identity}, not the expected {expected}"));
        }
        self.identity = identity.to_string();
        self.overlays = capabilities.contains(&"source-overlays");
        let imports: Vec<Value> =
            settings.imports.iter().map(|(prefix, root)| json!({ "prefix": prefix, "root": root })).collect();
        let mut open = json!({
            "kind": "open",
            "entry": self.entry,
            "prelude": settings.prelude,
            "stdRoot": settings.std,
            "imports": imports,
            "dependencies": settings.dependencies,
        });
        if let Some(tier) = settings.codegen_tier {
            open["codegenTier"] = json!(tier.name());
        }
        let open = self.request(open, &[])?;
        let reply = self.exchange(open, deadline)?;
        let epoch = reply.metadata.get("epoch").and_then(Value::as_str).unwrap_or("");
        if !reply.payload.is_empty() || epoch.is_empty() || reply.metadata.get("revision") != Some(&json!(0)) {
            return Err(format!("its open reply is not one: {}", Value::Object(reply.metadata)));
        }
        self.epoch = epoch.to_string();
        Ok(())
    }

    /// Builds the program from the files on disk, and `overlays` over them
    /// for this build only. An error is the compiler's failure (it has
    /// stopped), or overlays it could not be sent.
    pub fn build(&mut self, overlays: &[Overlay]) -> Result<Build, String> {
        if let Some(failure) = &self.failure {
            return Err(format!("the Blot compiler stopped after an earlier failure: {failure}"));
        }
        if self.child.is_none() {
            return Err("the Blot compiler is closed".into());
        }
        let (sources, payload) = self.sources(overlays)?;
        let mut request = json!({ "kind": "build", "epoch": self.epoch, "revision": self.revision });
        if !sources.is_empty() {
            request["sources"] = Value::Array(sources);
        }
        let request = self.request(request, &payload)?;
        let built = self.exchange(request, None).and_then(|reply| self.built(reply));
        built.map_err(|error| self.stop(error))
    }

    /// `overlays` as a build's `sources` and payload, or why they cannot be
    /// sent (which leaves the compiler running).
    fn sources(&self, overlays: &[Overlay]) -> Result<(Vec<Value>, Vec<u8>), String> {
        if overlays.len() > MAX_OVERLAYS {
            return Err(format!("{} overlays are more than the {MAX_OVERLAYS} a build takes", overlays.len()));
        }
        if !overlays.is_empty() && !self.overlays {
            return Err("this Blot compiler takes no overlays".into());
        }
        let mut sources = Vec::with_capacity(overlays.len());
        let mut payload = Vec::new();
        let mut paths = HashSet::new();
        for overlay in overlays {
            if overlay.path.is_empty() || overlay.path.contains('\0') {
                return Err(format!("the overlay {:?} has no path", overlay.path));
            }
            let path = settings::resolve(&self.root, &overlay.path);
            if !paths.insert(path.clone()) {
                return Err(format!("{path} is overlaid twice"));
            }
            let contents = overlay.contents.as_deref().unwrap_or("").as_bytes();
            if contents.len() > MAX_OVERLAY_BYTES {
                return Err(format!("the overlay of {path} is over 16 MiB"));
            }
            sources.push(json!({
                "path": path,
                "start": payload.len(),
                "length": contents.len(),
                "deleted": overlay.contents.is_none(),
            }));
            payload.extend_from_slice(contents);
        }
        Ok((sources, payload))
    }

    /// What a build's reply says.
    fn built(&mut self, reply: Frame) -> Result<Build, String> {
        let metadata = &reply.metadata;
        let revision = metadata.get("revision").and_then(Value::as_u64);
        let success = metadata.get("success").and_then(Value::as_bool);
        if metadata.get("epoch").and_then(Value::as_str) != Some(&self.epoch) || success.is_none() {
            return Err(format!("its build reply is not one: {}", Value::Object(reply.metadata)));
        }
        if success == Some(true) {
            if self.revision == MAX_REVISION || revision != Some(self.revision + 1) {
                return Err(format!("it built revision {revision:?} after {}", self.revision));
            }
            if !reply.payload.starts_with(&WASM_HEADER) {
                return Err(format!("what it built ({} bytes) is not a Wasm module", reply.payload.len()));
            }
            self.revision += 1;
            return Ok(Build::Built(reply.payload));
        }
        if revision != Some(self.revision) || !reply.payload.is_empty() {
            return Err(format!(
                "it rejected revision {revision:?} at {}, with {} bytes",
                self.revision,
                reply.payload.len()
            ));
        }
        let diagnostics = match metadata.get("diagnostics") {
            Some(Value::Array(diagnostics)) if !diagnostics.is_empty() => diagnostics,
            _ => return Err(format!("it rejected the build without diagnostics: {}", Value::Object(reply.metadata))),
        };
        diagnostics.iter().map(diagnostic).collect::<Result<_, _>>().map(Build::Rejected)
    }

    /// The paths whose changes need a new build: the entry's directory,
    /// every import's directory, the standard library's and the prelude.
    pub fn watched(&self) -> &[String] {
        &self.watched
    }

    /// The compiler's identity, from its hello.
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The successful builds so far.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The program's entry, absolute.
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// Asks the compiler to close, which saves its restart cache, and waits
    /// for it to exit. It is killed when it does not within [`CLOSING`], and
    /// when it had failed.
    pub fn close(&mut self) -> Result<(), String> {
        if self.failure.is_some() || self.child.is_none() {
            self.kill();
            return Ok(());
        }
        let deadline = Instant::now() + CLOSING;
        let request = json!({ "kind": "close", "epoch": self.epoch, "revision": self.revision });
        let request = self.request(request, &[]);
        let closed = request.and_then(|request| self.exchange(request, Some(deadline))).and_then(|reply| {
            let metadata = &reply.metadata;
            if !reply.payload.is_empty()
                || metadata.get("epoch").and_then(Value::as_str) != Some(&self.epoch)
                || metadata.get("revision").and_then(Value::as_u64) != Some(self.revision)
            {
                return Err(format!("its close reply is not one: {}", Value::Object(reply.metadata)));
            }
            let child = self.child.as_mut().expect("a running compiler");
            child.close_input();
            match child.wait(Some(deadline.saturating_duration_since(Instant::now()).as_millis() as u32)) {
                Some(0) => Ok(()),
                Some(code) => Err(format!("it exited with {code} on closing")),
                None => Err("it did not exit on closing".into()),
            }
        });
        match closed {
            Ok(()) => {
                self.child = None;
                Ok(())
            }
            Err(error) => Err(self.stop(error)),
        }
    }

    /// `request`, given the next id, as a frame with `payload`; or why it is
    /// past the protocol's limits (which leaves the compiler running).
    fn request(&mut self, mut request: Value, payload: &[u8]) -> Result<Request, String> {
        let id = self.next_id;
        if id > u32::MAX as u64 {
            return Err("the Blot compiler has run out of request ids".into());
        }
        self.next_id += 1;
        request["id"] = json!(id);
        let metadata = request.to_string().into_bytes();
        let length = 8 + metadata.len() as u64 + payload.len() as u64;
        if metadata.len() as u64 > MAX_METADATA_BYTES || length > MAX_FRAME_BYTES {
            return Err(format!("a {} request of {length} bytes is past the protocol's limits", request["kind"]));
        }
        let mut frame = Vec::with_capacity(length as usize);
        frame.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        frame.extend_from_slice(&metadata);
        frame.extend_from_slice(payload);
        Ok(Request { kind: request["kind"].take(), id, frame })
    }

    /// Sends `request` and reads its reply, which must have its kind and id.
    fn exchange(&mut self, request: Request, deadline: Option<Instant>) -> Result<Frame, String> {
        let child = self.child.as_mut().expect("a running compiler");
        child.write(&request.frame).map_err(|error| format!("writing to it: {error}"))?;
        let reply = self.receive(deadline)?;
        if reply.metadata.get("kind") != Some(&request.kind) || reply.metadata.get("id") != Some(&json!(request.id)) {
            let (kind, id) = (request.kind, request.id);
            return Err(format!("it answered {kind} {id} with {}", Value::Object(reply.metadata)));
        }
        Ok(reply)
    }

    /// Reads one frame, before `deadline` if there is one.
    fn receive(&mut self, deadline: Option<Instant>) -> Result<Frame, String> {
        let header = self.read_exact(8, deadline)?;
        let metadata_length = u32::from_le_bytes(header[..4].try_into().unwrap()) as u64;
        let payload_length = u32::from_le_bytes(header[4..].try_into().unwrap()) as u64;
        if metadata_length == 0
            || metadata_length > MAX_METADATA_BYTES
            || 8 + metadata_length + payload_length > MAX_FRAME_BYTES
        {
            return Err(format!("it sent a frame of {metadata_length} and {payload_length} bytes, past the limits"));
        }
        let metadata = self.read_exact(metadata_length as usize, deadline)?;
        let metadata = match serde_json::from_slice(&metadata) {
            Ok(Value::Object(metadata)) => metadata,
            _ => return Err(format!("its metadata is no JSON object: {}", String::from_utf8_lossy(&metadata))),
        };
        Ok(Frame { metadata, payload: self.read_exact(payload_length as usize, deadline)? })
    }

    /// The next `length` bytes of the compiler's output.
    fn read_exact(&mut self, length: usize, deadline: Option<Instant>) -> Result<Vec<u8>, String> {
        let child = self.child.as_mut().expect("a running compiler");
        while self.pending.len() < length {
            let timeout = match deadline {
                None => None,
                Some(deadline) => match deadline.saturating_duration_since(Instant::now()).as_millis() {
                    0 => return Err("it did not answer in time".into()),
                    left => Some(left.min(u32::MAX as u128) as u32),
                },
            };
            let wanted = (length - self.pending.len()).min(READ_CHUNK) as u32;
            match child.read(wanted, timeout) {
                Ok(bytes) if bytes.is_empty() => return Err("reading from it gave nothing".into()),
                Ok(bytes) => self.pending.extend_from_slice(&bytes),
                Err(ReadError::Closed) => return Err("it closed its output".into()),
                Err(ReadError::Timeout) => return Err("it did not answer in time".into()),
                Err(ReadError::Other(error)) => return Err(format!("reading from it: {error}")),
            }
        }
        let rest = self.pending.split_off(length);
        Ok(std::mem::replace(&mut self.pending, rest))
    }

    /// Stops the compiler after `error`: kills it, and returns (and keeps,
    /// for later builds) the error with its exit code and the end of its
    /// standard error.
    fn stop(&mut self, error: String) -> String {
        let (mut code, mut tail) = (None, String::new());
        if let Some(mut child) = self.child.take() {
            child.kill();
            code = child.wait(Some(1000));
            tail = child.error_tail();
        }
        let exit = code.map(|code| format!(" (exit {code})")).unwrap_or_default();
        let tail = tail.trim_end();
        let message = format!("the Blot compiler{exit}: {error}{}{tail}", if tail.is_empty() { "" } else { "\n" });
        self.failure.get_or_insert(error);
        message
    }

    /// Kills the compiler, if it runs.
    fn kill(&mut self) {
        if let Some(mut child) = self.child.take() {
            child.kill();
            child.wait(Some(1000));
        }
    }
}

impl<C: Child> Drop for Session<C> {
    fn drop(&mut self) {
        self.kill();
    }
}

/// A diagnostic of a rejected build, as pen has it.
fn diagnostic(value: &Value) -> Result<Diagnostic, String> {
    let invalid = || format!("its diagnostic is not one: {value}");
    let text = |name: &str| value.get(name).and_then(Value::as_str).map(str::to_string).ok_or_else(invalid);
    let offset = |name: &str| value.get(name).and_then(Value::as_u64).and_then(|offset| u32::try_from(offset).ok());
    let (Some(start), Some(end)) = (offset("start"), offset("end")) else {
        return Err(invalid());
    };
    if start > end || value.get("offset_encoding").and_then(Value::as_str) != Some("utf8_bytes") {
        return Err(invalid());
    }
    Ok(Diagnostic {
        file: text("filename")?,
        stage: text("stage")?,
        code: text("code")?,
        start,
        end,
        message: text("message")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A child that answers from a script: each request it is written gets
    /// the next of its replies.
    struct Scripted {
        output: Vec<u8>,
        replies: Vec<Vec<u8>>,
        written: Vec<Value>,
        killed: bool,
    }

    impl Child for Scripted {
        fn write(&mut self, bytes: &[u8]) -> Result<(), String> {
            let length = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
            self.written.push(serde_json::from_slice(&bytes[8..8 + length]).unwrap());
            if !self.replies.is_empty() {
                let reply = self.replies.remove(0);
                self.output.extend(reply);
            }
            Ok(())
        }
        fn close_input(&mut self) {}
        fn read(&mut self, max: u32, _: Option<u32>) -> Result<Vec<u8>, ReadError> {
            if self.output.is_empty() {
                return Err(ReadError::Closed);
            }
            // A byte at a time, to read frames across reads.
            let _ = max;
            Ok(vec![self.output.remove(0)])
        }
        fn error_tail(&mut self) -> String {
            "blotc: Scripted\n".into()
        }
        fn wait(&mut self, _: Option<u32>) -> Option<i32> {
            self.killed.then_some(1)
        }
        fn kill(&mut self) {
            self.killed = true;
        }
    }

    fn frame(metadata: Value, payload: &[u8]) -> Vec<u8> {
        let metadata = metadata.to_string().into_bytes();
        let mut frame = (metadata.len() as u32).to_le_bytes().to_vec();
        frame.extend((payload.len() as u32).to_le_bytes());
        frame.extend(metadata);
        frame.extend(payload);
        frame
    }

    fn hello() -> Value {
        json!({
            "kind": "hello",
            "protocol": PROTOCOL,
            "version": 1,
            "compilerIdentity": "ab".repeat(32),
            "capabilities": ["project-build", "utf8_bytes", "abort-disposes-process", "source-overlays"],
            "maxFrameBytes": MAX_FRAME_BYTES,
        })
    }

    fn program(settings: &str) -> Program {
        Program { root: "/p".into(), entry: "/p/src/main.blot".into(), settings: settings.into() }
    }

    /// Opens a session on `hello` and the replies, and what it wrote.
    fn open(hello: Value, replies: Vec<Vec<u8>>, settings: &str) -> (Result<Session<Scripted>, String>, Vec<String>) {
        let mut spawned = Vec::new();
        let session = Session::open(&program(settings), "/cache", |name, args, env| {
            spawned.push(name.to_string());
            spawned.extend(args.iter().cloned());
            spawned.extend(env.iter().map(|(name, value)| format!("{name}={value}")));
            Ok(Scripted { output: frame(hello, &[]), replies, written: Vec::new(), killed: false })
        });
        (session, spawned)
    }

    fn opened() -> Vec<u8> {
        frame(json!({ "kind": "open", "id": 1, "epoch": "1", "revision": 0 }), &[])
    }

    #[test]
    fn opens_with_absolute_paths() {
        let settings = r#"{"std": "../std", "imports": {"gdev/": "packages"}, "semanticWorkers": 2}"#;
        let (session, spawned) = open(hello(), vec![opened()], settings);
        assert_eq!(spawned, ["blotc", "serve-project", "--semantic-workers", "2", "BLOT_CACHE_DIR=/cache"]);
        let session = session.unwrap();
        assert_eq!(session.identity(), "ab".repeat(32));
        assert_eq!(
            session.child.as_ref().unwrap().written[0],
            json!({
                "kind": "open",
                "id": 1,
                "entry": "/p/src/main.blot",
                "prelude": "/std/prelude.blot",
                "stdRoot": "/std",
                "imports": [{ "prefix": "gdev/", "root": "/p/packages" }],
                "dependencies": null,
            })
        );
    }

    #[test]
    fn refuses_a_hello_it_does_not_know() {
        let mut old = hello();
        old["version"] = json!(2);
        let mut unable = hello();
        unable["capabilities"] = json!(["project-build"]);
        let other = format!(r#"{{"expectedIdentity": "{}"}}"#, "cd".repeat(32));
        for (hello, settings, refusal) in [
            (old, "{}", "is not one of blot-zig-project version 1"),
            (unable, "{}", "cannot do utf8_bytes"),
            (hello(), other.as_str(), "not the expected"),
        ] {
            let error = open(hello, vec![opened()], settings).0.err().unwrap();
            assert!(error.contains(refusal) && error.ends_with("blotc: Scripted"), "{error}");
        }
    }

    #[test]
    fn builds_rejects_and_stops_on_a_bad_reply() {
        let built =
            frame(json!({ "kind": "build", "id": 2, "epoch": "1", "revision": 1, "success": true }), &WASM_HEADER);
        let rejected = frame(
            json!({ "kind": "build", "id": 3, "epoch": "1", "revision": 1, "success": false, "diagnostics": [{
                "filename": "/p/src/main.blot", "stage": "check", "code": "unknown_name", "start": 3, "end": 7,
                "message": "unknown name: x", "offset_encoding": "utf8_bytes", "utf16": null, "details": null,
            }], "attemptStats": null }),
            &[],
        );
        let wrong = frame(json!({ "kind": "build", "id": 9, "epoch": "1", "revision": 1, "success": false }), &[]);
        let (session, _) = open(hello(), vec![opened(), built, rejected, wrong], "{}");
        let mut session = session.unwrap();
        assert_eq!(session.build(&[]).unwrap(), Build::Built(WASM_HEADER.to_vec()));
        assert_eq!(session.revision(), 1);
        let overlays = [
            Overlay { path: "src/main.blot".into(), contents: Some("é".into()) },
            Overlay { path: "/p/src/gone.blot".into(), contents: None },
        ];
        let Build::Rejected(diagnostics) = session.build(&overlays).unwrap() else { panic!("built") };
        assert_eq!(
            (diagnostics[0].file.as_str(), diagnostics[0].start, diagnostics[0].end),
            ("/p/src/main.blot", 3, 7)
        );
        assert_eq!(
            session.child.as_ref().unwrap().written[2]["sources"],
            json!([
                { "path": "/p/src/main.blot", "start": 0, "length": 2, "deleted": false },
                { "path": "/p/src/gone.blot", "start": 2, "length": 0, "deleted": true },
            ])
        );
        let error = session.build(&[]).unwrap_err();
        assert!(error.starts_with("the Blot compiler (exit 1): it answered \"build\" 4"), "{error}");
        assert!(session.child.is_none());
        let again = session.build(&[]).unwrap_err();
        assert!(again.contains("stopped after an earlier failure"), "{again}");
    }

    #[test]
    fn refuses_overlays_it_cannot_send_and_keeps_running() {
        let (session, _) = open(hello(), vec![opened()], "{}");
        let mut session = session.unwrap();
        let twice = [
            Overlay { path: "/p/a.blot".into(), contents: None },
            Overlay { path: "/p/x/../a.blot".into(), contents: None },
        ];
        assert!(session.build(&twice).unwrap_err().contains("overlaid twice"));
        let large = Some("x".repeat(MAX_OVERLAY_BYTES));
        let too_many: Vec<Overlay> =
            (0..4).map(|n| Overlay { path: format!("/p/{n}.blot"), contents: large.clone() }).collect();
        assert!(session.build(&too_many).unwrap_err().contains("past the protocol's limits"));
        assert!(session.failure.is_none() && session.child.is_some());
    }
}
