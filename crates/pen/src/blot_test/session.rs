//! Native session tests. The test guest and game guest are separate instances;
//! only this driver owns the GPU, files and game input. GPU tests run serially.
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail, ensure};
use blot_host::{Argument, Guest, GuestModule, Value};
use pen_protocol::boundary::{read_catalog, validate_schema, validate_state, whole};
use pen_protocol::{Catalog, GameEvent, UiNode, test_driver_op as op};
use serde_json::{Value as Json, json};
use wasmtime::Engine;

use crate::session::{Gpu, Options as GameOptions, Paths, Playback, Session};
use crate::snapshot::{self, SavedState};

pub struct Options {
    pub out: Option<PathBuf>,
    pub golden: Option<PathBuf>,
    pub update: bool,
    pub software: bool,
}

pub struct Prepared {
    root: PathBuf,
    out: PathBuf,
    golden: PathBuf,
    bundle: PathBuf,
    engine: Engine,
    module: GuestModule,
    catalog: Catalog,
    update: bool,
    software: bool,
}

impl Prepared {
    pub fn new(root: &Path, options: Options) -> Result<Self> {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let out = options
            .out
            .unwrap_or_else(|| root.join(".pen/session-tests"))
            .join(format!("run-{}-{stamp}", std::process::id()));
        let out = std::path::absolute(out)?;
        let bundle = out.join("game");
        let project = crate::project::Project::find(Some(root.to_owned()))?;
        crate::project::export_bundle(&project, &bundle)?;
        let engine = blot_host::engine()?;
        let module = GuestModule::new(&engine, &std::fs::read(bundle.join("game.wasm"))?, None)?;
        module.abi().check_game()?;
        let catalog = read_catalog(&bundle.join("assets"))?;
        eprintln!("session artifacts: {}", out.display());
        Ok(Self {
            root: root.to_owned(),
            out,
            bundle,
            engine,
            module,
            catalog,
            golden: options.golden.unwrap_or_else(|| root.join("tests/scenarios/golden")),
            update: options.update,
            software: options.software,
        })
    }

    pub fn module(&self, path: &Path) -> Run<'_> {
        use sha2::{Digest, Sha256};
        let relative = path.strip_prefix(&self.root).unwrap_or(path).to_string_lossy();
        let hash = format!("{:x}", Sha256::digest(relative.as_bytes()));
        let stem: String = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '-' })
            .collect();
        Run { prepared: self, directory: format!("{stem}-{}", &hash[..8]) }
    }
}

pub struct Run<'a> {
    prepared: &'a Prepared,
    directory: String,
}

impl Run<'_> {
    pub fn call(&self, engine: &Engine, module: &GuestModule, name: &str, deadline: Duration) -> Result<Value> {
        safe_name(name)?;
        let out = self.prepared.out.join(&self.directory).join(name);
        std::fs::create_dir_all(&out)?;
        let mut driver = Driver {
            bundle: self.prepared.bundle.clone(),
            engine: self.prepared.engine.clone(),
            module: self.prepared.module.clone(),
            catalog: self.prepared.catalog.clone(),
            golden: self.prepared.golden.join(&self.directory).join(name),
            update: self.prepared.update,
            software: self.prepared.software,
            out: out.clone(),
            session: None,
            trace: vec![],
            assertions: 0,
            ticks: 0,
            finished: false,
            started: Instant::now(),
            deadline,
        };
        // Session's UI/services stay on their owning thread. The test guest
        // blocks on copied command/answer values, never on the game's stack.
        let (send, receive) = mpsc::sync_channel::<(Vec<f32>, mpsc::SyncSender<Result<Value>>)>(0);
        let called = std::thread::scope(|scope| -> Result<Value> {
            let test = std::thread::Builder::new()
                .name("pen-scenario".into())
                .stack_size(blot_host::GUEST_STACK_BYTES + (8 << 20))
                .spawn_scoped(scope, move || -> Result<Value> {
                    let mut guest = Guest::new(engine, module)?;
                    guest.set_deadline(deadline);
                    let result = guest.call(
                        name,
                        Argument::Callback(Box::new(move |value| {
                            let (answer, result) = mpsc::sync_channel(0);
                            send.send((value.into_f32_array()?, answer)).context("session driver stopped")?;
                            result.recv().context("session driver stopped before answering")?
                        })),
                    );
                    let disposed = guest.dispose();
                    let value = result?;
                    disposed?;
                    Ok(value)
                })?;
            while let Ok((values, answer)) = receive.recv() {
                let _ = answer.send(driver.command(&values).map(Value::ArrayF32));
            }
            let result = test.join().map_err(|_| anyhow::anyhow!("session test thread panicked"))??;
            ensure!(driver.finished, "session test must finish through pen/test.run");
            Ok(result)
        });
        let completed = driver.complete(called.as_ref().err());
        match (called, completed) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), cleanup) => {
                let suffix = cleanup.err().map(|e| format!("; writing artifacts: {e:#}")).unwrap_or_default();
                Err(error.context(format!("session artifacts: {}{suffix}", out.display())))
            }
            (Ok(_), Err(error)) => Err(error.context(format!("session artifacts: {}", out.display()))),
        }
    }
}

struct Driver {
    bundle: PathBuf,
    engine: Engine,
    module: GuestModule,
    catalog: Catalog,
    out: PathBuf,
    golden: PathBuf,
    update: bool,
    software: bool,
    session: Option<Session>,
    trace: Vec<Json>,
    assertions: u32,
    ticks: u32,
    finished: bool,
    started: Instant,
    deadline: Duration,
}

impl Driver {
    fn check_deadline(&self) -> Result<()> {
        ensure!(
            self.started.elapsed() <= self.deadline,
            "session exceeded its {:?} wall-clock deadline",
            self.deadline
        );
        Ok(())
    }

    fn game(&mut self) -> Result<&mut Session> {
        self.session.as_mut().context("launch a game before issuing session commands")
    }

    fn command(&mut self, values: &[f32]) -> Result<Vec<f32>> {
        ensure!(!self.finished, "session commands after finish");
        ensure!(self.trace.len() < 10_000, "session command limit exceeded");
        self.check_deadline()?;
        let mut input = Input { values, at: 0 };
        ensure!(input.integer()? == 1, "unsupported session driver version");
        let code = input.integer()?;
        let name = op::NAMES.get(usize::try_from(code)?).context("unknown session command")?;
        self.trace.push(json!({"action": name, "before_tick": self.ticks}));
        let result = self.dispatch(code, &mut input).and_then(|value| {
            input.end()?;
            self.check_deadline()?;
            Ok(value)
        });
        let row = self.trace.last_mut().unwrap();
        row["after_tick"] = json!(self.ticks);
        row["status"] = json!(if result.is_ok() { "ok" } else { "failed" });
        if let Err(error) = &result {
            row["error"] = json!(format!("{error:#}"));
        }
        result.with_context(|| format!("{name} at tick {}", self.ticks))
    }

    fn dispatch(&mut self, code: u32, input: &mut Input<'_>) -> Result<Vec<f32>> {
        match code {
            op::LAUNCH => {
                ensure!(self.session.is_none(), "each test launches exactly one isolated session");
                let (width, height, schema) = (input.integer()?, input.integer()?, input.integer()?);
                ensure!(
                    (64..=2048).contains(&width) && (64..=2048).contains(&height),
                    "viewport must be 64..2048 pixels per side"
                );
                validate_schema(schema as f32)?;
                let saved =
                    SavedState { schema, state: input.array(pen_protocol::boundary::MAX_STATE_FLOATS)?.to_vec() };
                validate_state(&saved.state)?;
                input.end()?;
                self.trace.last_mut().unwrap()["viewport"] = json!([width, height]);
                self.trace.last_mut().unwrap()["initial_lanes"] = json!(saved.state.len());
                let data = self.out.join("data");
                std::fs::create_dir_all(&data)?;
                snapshot::write(&data.join("quicksave.bin"), &saved)?;
                snapshot::write(&self.out.join("initial.bin"), &saved)?;
                let scene = data.join("start.scene");
                if self.bundle.join("main.scene").is_file() {
                    std::fs::copy(self.bundle.join("main.scene"), &scene)?;
                }
                let paths = Paths::new(self.bundle.clone(), None, Some(scene), Some(self.out.clone()), Some(data));
                let options = GameOptions {
                    frames: 0,
                    screenshot: None,
                    record: None,
                    fps: Some(60.0),
                    mute: true,
                    audio_buffer: None,
                    dev: false,
                    script: vec![],
                    headless: true,
                    settle: true,
                };
                let gpu = Gpu::headless((width, height), &paths.assets, self.software)?;
                let mut session = Session::testing(
                    options,
                    paths,
                    self.engine.clone(),
                    self.module.clone(),
                    self.catalog.clone(),
                    gpu,
                )?;
                session.playback(
                    Playback {
                        record_input: Some(self.out.join("input.penrec")),
                        history: true,
                        ..Playback::default()
                    },
                    None,
                )?;
                self.session = Some(session);
            }
            op::STEP => {
                let ticks = input.ticks()?;
                input.end()?;
                for _ in 0..ticks {
                    self.advance(&[])?;
                }
            }
            op::CLICK | op::VISIBLE | op::HIDDEN => {
                let (role, ticks, name) = (input.integer()?, input.ticks()?, input.text()?);
                ensure!(usize::try_from(role)? < pen_protocol::ui_role::NAMES.len(), "unknown UI role");
                input.end()?;
                self.trace.last_mut().unwrap()["selector"] =
                    json!({"role": pen_protocol::ui_role::NAMES[usize::try_from(role)?], "name": name});
                let node = self.wait_for(role, &name, ticks, code)?;
                if code == op::CLICK {
                    let node = node.context("click target disappeared")?;
                    let (x, y) = center(&node, self.game()?.gpu.size)?;
                    self.advance(&[GameEvent::PointerMove { x, y }])?;
                    let current = selected(self.game()?.interface(), role, &name)?
                        .context("click target disappeared after pointer movement")?;
                    ensure!(current.id == node.id && !current.disabled, "click target changed after pointer movement");
                    let (x, y) = center(&current, self.game()?.gpu.size)?;
                    self.trace.last_mut().unwrap()["pointer"] = json!([x, y]);
                    self.advance(&[GameEvent::PointerDown { button: 0, x, y }])?;
                    self.advance(&[GameEvent::PointerUp { button: 0, x, y }])?;
                } else {
                    self.assertions += 1;
                }
            }
            op::PRESS => {
                let key = input.text()?;
                input.end()?;
                crate::script::parse(&[format!("key={key}")])?;
                self.trace.last_mut().unwrap()["key"] = json!(key);
                self.advance(&[GameEvent::KeyDown(key.clone())])?;
                self.advance(&[GameEvent::KeyUp(key)])?;
            }
            op::SNAPSHOT => {
                input.end()?;
                let saved = self.game()?.test_snapshot()?;
                let mut result = vec![saved.schema as f32];
                result.extend(saved.state);
                return Ok(result);
            }
            op::EQUAL => {
                self.game()?;
                let label = input.text()?;
                let expected = input.array(4096)?;
                let actual = input.array(4096)?;
                input.end()?;
                self.trace.last_mut().unwrap()["assertion"] =
                    json!({"label":label,"expected":expected,"actual":actual});
                ensure!(expected == actual, "{label}: expected {expected:?}, received {actual:?}");
                self.assertions += 1;
            }
            op::SCREENSHOT => {
                let compare = input.integer()?;
                ensure!(compare <= 1, "screenshot comparison flag must be 0 or 1");
                let tolerance = input.number()?;
                ensure!((0.0..=1.0).contains(&tolerance), "screenshot tolerance must be 0..1");
                let name = input.text()?;
                safe_name(&name)?;
                input.end()?;
                self.trace.last_mut().unwrap()["screenshot"] = json!(format!("{name}.png"));
                let path = self.out.join(format!("{name}.png"));
                self.game()?.test_screenshot(path.clone())?;
                if compare == 1 {
                    self.compare(&name, &path, f64::from(tolerance))?;
                    self.assertions += 1;
                }
            }
            op::FINISH => {
                self.game()?;
                input.end()?;
                ensure!(self.assertions > 0, "session test made no assertions");
                self.finished = true;
                return Ok(vec![self.assertions as f32, 0.0]);
            }
            _ => bail!("unknown session command {code}"),
        }
        Ok(vec![])
    }

    fn advance(&mut self, events: &[GameEvent]) -> Result<()> {
        self.check_deadline()?;
        let session = self.game()?;
        session.test_settle()?;
        for event in events {
            session.input.enqueue(event.clone());
        }
        ensure!(session.step(None)?, "the game exited before the scenario finished");
        self.ticks = self.ticks.checked_add(1).context("session tick overflow")?;
        self.check_deadline()
    }

    fn wait_for(&mut self, role: u32, name: &str, ticks: u32, action: u32) -> Result<Option<UiNode>> {
        let mut previous: Option<UiNode> = None;
        for waited in 0..=ticks {
            self.check_deadline()?;
            let session = self.game()?;
            let node = selected(session.interface(), role, name)?;
            match (action, &node) {
                (op::HIDDEN, None) => return Ok(None),
                (op::VISIBLE, Some(_)) => return Ok(node),
                (op::CLICK, Some(node))
                    if !node.disabled
                        && center(node, session.gpu.size).is_ok()
                        && previous.as_ref().is_some_and(|old| stable(old, node)) =>
                {
                    return Ok(Some(node.clone()));
                }
                _ => {}
            }
            previous = node;
            if waited < ticks {
                self.advance(&[])?;
            }
        }
        bail!(
            "{} {:?} was not {} within {ticks} ticks",
            pen_protocol::ui_role::NAMES[usize::try_from(role)?],
            name,
            match action {
                op::HIDDEN => "hidden",
                op::CLICK => "enabled, onscreen and stable",
                _ => "visible",
            }
        )
    }

    fn compare(&mut self, name: &str, actual: &Path, tolerance: f64) -> Result<()> {
        let golden = self.golden.join(format!("{name}.png"));
        if self.update {
            std::fs::create_dir_all(&self.golden)?;
            std::fs::copy(actual, &golden)?;
            self.trace.last_mut().unwrap()["golden_updated"] = json!(golden);
            return Ok(());
        }
        ensure!(golden.is_file(), "missing golden {}; inspect the capture, then run with --update", golden.display());
        std::fs::copy(&golden, self.out.join(format!("{name}.golden.png")))?;
        let (actual, expected) = (image::open(actual)?.to_rgba8(), image::open(golden)?.to_rgba8());
        ensure!(
            actual.dimensions() == expected.dimensions(),
            "screenshot dimensions differ: {:?} vs {:?}",
            actual.dimensions(),
            expected.dimensions()
        );
        let difference = crate::visual::compare(&actual, &expected, &self.out.join(format!("{name}.diff.png")))?;
        self.trace.last_mut().unwrap()["rmse"] = json!(difference.rmse);
        ensure!(difference.rmse <= tolerance, "screenshot {name}: RMSE {:.6} exceeds {tolerance:.6}", difference.rmse);
        Ok(())
    }

    fn complete(&mut self, failure: Option<&anyhow::Error>) -> Result<()> {
        let passed = failure.is_none();
        let mut errors = Vec::new();
        if let Some(mut session) = self.session.take() {
            if !passed && let Err(e) = session.test_screenshot(self.out.join("failure.png")) {
                errors.push(format!("screenshot: {e:#}"));
            }
            match session.test_snapshot().and_then(|saved| snapshot::write(&self.out.join("state.bin"), &saved)) {
                Ok(()) => {}
                Err(e) => errors.push(format!("snapshot: {e:#}")),
            }
            let nodes: Vec<_> = session
                .interface()
                .iter()
                .map(|node| {
                    json!({
                        "id":node.id,"role":node.role,"name":crate::accessibility::texts(node)[0],
                        "bounds":[node.x,node.y,node.width,node.height],"hidden":node.hidden,"disabled":node.disabled,
                    })
                })
                .collect();
            if let Err(e) = std::fs::write(self.out.join("ui.json"), serde_json::to_vec_pretty(&nodes)?) {
                errors.push(e.to_string());
            }
            if let Err(e) = session.finish() {
                errors.push(format!("shutdown: {e:#}"));
            }
        }
        std::fs::write(
            self.out.join("trace.json"),
            serde_json::to_vec_pretty(&json!({
                "version":1,"passed":passed && errors.is_empty(),"error":failure.map(|error| format!("{error:#}")),
                "ticks":self.ticks,"actions":self.trace,"artifact_errors":errors,
            }))?,
        )?;
        ensure!(errors.is_empty(), "{}", errors.join("; "));
        Ok(())
    }
}

fn safe_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && name.len() <= 128
            && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-'),
        "artifact names use 1..128 ASCII letters, digits, underscores or hyphens"
    );
    Ok(())
}

fn selected(ui: &[UiNode], role: u32, name: &str) -> Result<Option<UiNode>> {
    let included = crate::accessibility::included(ui);
    let mut matches = ui
        .iter()
        .zip(included)
        .filter(|(node, kept)| *kept && node.role == role && crate::accessibility::texts(node)[0] == name);
    let first = matches.next().map(|(node, _)| node.clone());
    ensure!(matches.next().is_none(), "ambiguous selector: more than one visible node named {name:?} with role {role}");
    Ok(first)
}

fn center(node: &UiNode, size: (u32, u32)) -> Result<(f32, f32)> {
    let (left, top) = (node.x.max(0.0), node.y.max(0.0));
    let (right, bottom) = ((node.x + node.width).min(size.0 as f32), (node.y + node.height).min(size.1 as f32));
    ensure!(right > left && bottom > top, "target has no visible area in the viewport");
    Ok(((left + right) * 0.5, (top + bottom) * 0.5))
}

fn stable(a: &UiNode, b: &UiNode) -> bool {
    a.id == b.id && [a.x - b.x, a.y - b.y, a.width - b.width, a.height - b.height].iter().all(|v| v.abs() <= 0.25)
}

struct Input<'a> {
    values: &'a [f32],
    at: usize,
}
impl<'a> Input<'a> {
    fn number(&mut self) -> Result<f32> {
        let value = *self.values.get(self.at).context("truncated session command")?;
        self.at += 1;
        ensure!(value.is_finite(), "nonfinite session command value");
        Ok(value)
    }
    fn integer(&mut self) -> Result<u32> {
        whole(self.number()?).context("expected an exact nonnegative integer")
    }
    fn ticks(&mut self) -> Result<u32> {
        let ticks = self.integer()?;
        ensure!(ticks <= 3600, "a step or wait is bounded to 3600 ticks");
        Ok(ticks)
    }
    fn array(&mut self, maximum: usize) -> Result<&'a [f32]> {
        let count = usize::try_from(self.integer()?)?;
        ensure!(count <= maximum, "session array exceeds {maximum} lanes");
        let end = self.at.checked_add(count).context("session array size overflow")?;
        let values = self.values.get(self.at..end).context("truncated session array")?;
        ensure!(values.iter().all(|v| v.is_finite()), "nonfinite session array");
        self.at = end;
        Ok(values)
    }
    fn text(&mut self) -> Result<String> {
        self.array(256)?
            .iter()
            .map(|v| {
                whole(*v)
                    .and_then(char::from_u32)
                    .filter(|c| *c != '\0')
                    .context("invalid Unicode scalar in session text")
            })
            .collect()
    }
    fn end(&self) -> Result<()> {
        ensure!(self.at == self.values.len(), "trailing session command values");
        Ok(())
    }
}

#[cfg(test)]
mod tests;
