//! The asset pipeline while developing: the checkout's `pipeline.json` maps
//! source globs (model scripts, .blend files, fonts, and other recipes'
//! outputs) to the Justfile recipes that build assets from them, and this
//! reruns a recipe when its sources change. The recipe writes into
//! `src/assets`, which dev.rs then reloads in the running game.
//!
//! Recipes run one at a time, in the order the file lists them, so a recipe
//! that reads another's output is listed after it and runs after it when
//! both are due. Several write the same files (the buildings' palette), and
//! Blender runs are heavy, so nothing runs side by side. A change that
//! arrives while a recipe runs queues it again. A file counts as changed only
//! when its content does, so rewriting an identical output does not set off
//! the recipes that read it, and saving without an edit does nothing. A
//! failed recipe is reported and the game keeps the assets it has.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{DefaultHasher, Hasher};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use notify::EventKind;
use serde::Deserialize;

use crate::dev::watch;
use crate::devlog::{self, Source};

/// How long sources must be quiet before a recipe runs: an editor's save or
/// a checkout of many files arrives as one change, and starting Blender
/// twice costs far more than the wait.
const DEBOUNCE: Duration = Duration::from_millis(250);
/// How often a running recipe is checked on.
const TICK: Duration = Duration::from_millis(50);

#[derive(Deserialize)]
struct Config {
    recipes: Vec<Recipe>,
}

/// One of the Justfile's recipes, `just NAME`, and the files it builds from.
#[derive(Deserialize)]
struct Recipe {
    recipe: String,
    /// Globs relative to the checkout: `*` and `?` within a name, `**` for
    /// any number of directories.
    sources: Vec<String>,
    /// Programs the recipe runs that may not be installed, such as Blender.
    #[serde(default)]
    requires: Vec<String>,
}

enum Message {
    Changed(watch::Event),
    Stop,
}

/// The pipeline watching the checkout; stops (and stops a running recipe)
/// when dropped.
pub struct Pipeline {
    messages: Sender<Message>,
    thread: Option<JoinHandle<()>>,
    _watch: watch::Subscription,
}

impl Pipeline {
    /// Starts the checkout's pipeline, or reports why it cannot and returns
    /// None: a checkout without `pipeline.json` has none.
    pub fn start(root: &Path) -> Option<Pipeline> {
        let file = root.join("pipeline.json");
        if !file.is_file() {
            return None;
        }
        match Pipeline::new(root, &file) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                eprintln!("pipeline: not rebuilding assets: {error:#}");
                None
            }
        }
    }

    fn new(root: &Path, file: &Path) -> Result<Option<Pipeline>> {
        let text = std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
        let config: Config = serde_json::from_str(&text).with_context(|| format!("reading {}", file.display()))?;
        if !installed("just") {
            eprintln!("pipeline: just is not installed, so assets do not rebuild as their sources change");
            return Ok(None);
        }
        let root = std::fs::canonicalize(root)?;
        warn_unknown(&root, &config.recipes);
        let unavailable = unavailable(&config.recipes);

        let (messages, received) = channel();
        let changes = messages.clone();
        let patterns: Vec<&str> =
            config.recipes.iter().flat_map(|recipe| &recipe.sources).map(String::as_str).collect();
        let directories = watched(&patterns);
        let mut roots = Vec::new();
        for (directory, recursive) in &directories {
            let path = root.join(directory);
            if !path.is_dir() {
                eprintln!("pipeline: {} does not exist; pipeline.json's sources there are not watched", path.display());
                continue;
            }
            roots.push((path, *recursive));
        }
        let roots: Vec<(&Path, bool)> = roots.iter().map(|(path, recursive)| (path.as_path(), *recursive)).collect();
        let watch = watch::subscribe(&roots, move |event| {
            let _ = changes.send(Message::Changed(event));
        })?;
        let available = config.recipes.len() - unavailable.iter().filter(|skip| **skip).count();
        eprintln!("pipeline: rerunning {available} recipes as their sources change ({})", file.display());

        let scheduler = Scheduler {
            root,
            recipes: config.recipes,
            unavailable,
            seen: HashMap::new(),
            touched: BTreeSet::new(),
            touched_at: None,
            pending: BTreeMap::new(),
            running: None,
        };
        let thread = std::thread::Builder::new()
            .name("pen-pipeline".into())
            .spawn(move || scheduler.run(&directories, received))?;
        Ok(Some(Pipeline { messages, thread: Some(thread), _watch: watch }))
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        let _ = self.messages.send(Message::Stop);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Says which of pipeline.json's recipes the Justfile does not have.
fn warn_unknown(root: &Path, recipes: &[Recipe]) {
    let Ok(output) = Command::new("just").arg("--summary").current_dir(root).stderr(Stdio::null()).output() else {
        return;
    };
    if !output.status.success() {
        return;
    }
    let summary = String::from_utf8_lossy(&output.stdout);
    let known: BTreeSet<&str> = summary.split_whitespace().collect();
    for recipe in recipes.iter().filter(|recipe| !known.contains(recipe.recipe.as_str())) {
        eprintln!("pipeline: the Justfile has no recipe {}", recipe.recipe);
    }
}

/// Which recipes cannot run because a program they need is not installed,
/// said once for each such program.
fn unavailable(recipes: &[Recipe]) -> Vec<bool> {
    let tools: BTreeSet<&str> = recipes.iter().flat_map(|recipe| &recipe.requires).map(String::as_str).collect();
    let missing: BTreeSet<&str> = tools.into_iter().filter(|tool| !installed(tool)).collect();
    for tool in &missing {
        let names: Vec<&str> = recipes
            .iter()
            .filter(|recipe| recipe.requires.iter().any(|needed| needed == tool))
            .map(|recipe| recipe.recipe.as_str())
            .collect();
        eprintln!("pipeline: {tool} is not installed, so these recipes do not rerun: {}", names.join(", "));
    }
    recipes.iter().map(|recipe| recipe.requires.iter().any(|tool| missing.contains(tool.as_str()))).collect()
}

/// Whether `program` is on the PATH, as `which` finds it.
fn installed(program: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else { return false };
    std::env::split_paths(&path).any(|directory| {
        let candidate = directory.join(program);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            candidate.metadata().is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
        }
        #[cfg(not(unix))]
        {
            candidate.is_file() || candidate.with_extension("exe").is_file()
        }
    })
}

/// Whether `path` (relative, `/`-separated) matches `pattern`.
fn matches(pattern: &str, path: &str) -> bool {
    let pattern: Vec<&str> = pattern.split('/').collect();
    let path: Vec<&str> = path.split('/').collect();
    matches_parts(&pattern, &path)
}

fn matches_parts(pattern: &[&str], path: &[&str]) -> bool {
    match pattern.split_first() {
        None => path.is_empty(),
        Some((&"**", rest)) => (0..=path.len()).any(|skip| matches_parts(rest, &path[skip..])),
        Some((first, rest)) => path.split_first().is_some_and(|(name, remaining)| {
            matches_name(first.as_bytes(), name.as_bytes()) && matches_parts(rest, remaining)
        }),
    }
}

/// `*` and `?` within one name.
fn matches_name(pattern: &[u8], name: &[u8]) -> bool {
    match pattern.split_first() {
        None => name.is_empty(),
        Some((b'*', rest)) => (0..=name.len()).any(|skip| matches_name(rest, &name[skip..])),
        Some((b'?', rest)) => !name.is_empty() && matches_name(rest, &name[1..]),
        Some((byte, rest)) => name.first() == Some(byte) && matches_name(rest, &name[1..]),
    }
}

/// The directories to watch for these globs, relative to the checkout, and
/// whether to watch below them: each glob's directory before its first
/// wildcard, leaving out any that a recursive watch already covers.
fn watched(patterns: &[&str]) -> Vec<(String, bool)> {
    let mut directories: BTreeMap<String, bool> = BTreeMap::new();
    for pattern in patterns {
        let parts: Vec<&str> = pattern.split('/').collect();
        let wild = parts.iter().position(|part| part.contains(['*', '?'])).unwrap_or(parts.len() - 1);
        // A wildcard only in the last name needs that directory alone.
        let recursive = wild + 1 < parts.len() || parts[wild] == "**";
        let directory = parts[..wild].join("/");
        let entry = directories.entry(directory).or_insert(false);
        *entry |= recursive;
    }
    let covered = |directory: &str| {
        directories.iter().any(|(other, recursive)| {
            *recursive
                && other != directory
                && (other.is_empty()
                    || directory.strip_prefix(other.as_str()).is_some_and(|rest| rest.starts_with('/')))
        })
    };
    directories
        .iter()
        .filter(|(directory, _)| !covered(directory))
        .map(|(directory, recursive)| (directory.clone(), *recursive))
        .collect()
}

/// Editors' and tools' temporaries and caches, which are never sources:
/// hidden files, backups, Python's bytecode and Blender's `.blend1`.
fn temporary(path: &str) -> bool {
    path.split('/').any(|part| part.starts_with('.') || part == "__pycache__")
        || [".tmp", ".swp", ".pyc", ".blend1", ".part", "~"].iter().any(|suffix| path.ends_with(suffix))
}

/// A file's content, or None when it is not there.
fn fingerprint(path: &Path) -> Option<u64> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = DefaultHasher::new();
    let mut buffer = vec![0; 1 << 16];
    loop {
        match file.read(&mut buffer) {
            Ok(0) => return Some(hasher.finish()),
            Ok(read) => hasher.write(&buffer[..read]),
            Err(_) => return None,
        }
    }
}

/// Prints a recipe's output as it comes, each line after the recipe's name.
fn forward(name: &str, pipe: impl Read + Send + 'static) -> Option<JoinHandle<()>> {
    let name = name.to_owned();
    std::thread::Builder::new()
        .name("pen-pipeline-output".into())
        .spawn(move || {
            let mut reader = BufReader::new(pipe);
            let mut line = Vec::new();
            while reader.read_until(b'\n', &mut line).is_ok_and(|read| read > 0) {
                eprintln!("[{name}] {}", String::from_utf8_lossy(&line).trim_end());
                line.clear();
            }
        })
        .ok()
}

struct Running {
    index: usize,
    child: Child,
    started: Instant,
    output: Vec<JoinHandle<()>>,
}

struct Scheduler {
    root: PathBuf,
    recipes: Vec<Recipe>,
    /// Recipes skipped because a program they need is not installed.
    unavailable: Vec<bool>,
    /// The content of each source file as last seen, by path relative to
    /// the checkout.
    seen: HashMap<String, u64>,
    /// Sources with events since they were last looked at.
    touched: BTreeSet<String>,
    touched_at: Option<Instant>,
    /// Recipes due, by their place in the file, with the change that made
    /// each due.
    pending: BTreeMap<usize, String>,
    running: Option<Running>,
}

impl Scheduler {
    fn run(mut self, directories: &[(String, bool)], messages: Receiver<Message>) {
        self.remember(directories);
        loop {
            match messages.recv_timeout(TICK) {
                Ok(Message::Changed(Ok(event))) => self.note(&event, Instant::now()),
                Ok(Message::Changed(Err(error))) => eprintln!("pipeline: watching files: {error}"),
                Ok(Message::Stop) | Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {}
            }
            if self.touched_at.is_some_and(|at| at.elapsed() >= DEBOUNCE) {
                self.settle();
            }
            self.check();
            self.next();
        }
        self.stop();
    }

    fn source(&self, path: &str) -> bool {
        !temporary(path) && self.recipes.iter().any(|recipe| recipe.sources.iter().any(|glob| matches(glob, path)))
    }

    /// The content of every source there is now, to tell later edits from
    /// rewrites that change nothing.
    fn remember(&mut self, directories: &[(String, bool)]) {
        fn walk(directory: &Path, recursive: bool, files: &mut Vec<PathBuf>) {
            let Ok(entries) = std::fs::read_dir(directory) else { return };
            for entry in entries.flatten() {
                let path = entry.path();
                match entry.file_type() {
                    Ok(kind) if kind.is_dir() && recursive => walk(&path, true, files),
                    Ok(kind) if kind.is_file() => files.push(path),
                    _ => {}
                }
            }
        }
        let mut files = Vec::new();
        for (directory, recursive) in directories {
            walk(&self.root.join(directory), *recursive, &mut files);
        }
        for file in files {
            let Some(path) = self.relative(&file) else { continue };
            if self.source(&path)
                && let Some(content) = fingerprint(&file)
            {
                self.seen.insert(path, content);
            }
        }
    }

    fn relative(&self, path: &Path) -> Option<String> {
        let relative = path.strip_prefix(&self.root).ok()?;
        let parts: Vec<String> =
            relative.components().map(|part| part.as_os_str().to_string_lossy().into_owned()).collect();
        Some(parts.join("/"))
    }

    fn note(&mut self, event: &notify::Event, now: Instant) {
        // Reading a source (as recipes do) changes nothing.
        if matches!(event.kind, EventKind::Access(_)) {
            return;
        }
        for path in &event.paths {
            if let Some(path) = self.relative(path)
                && self.source(&path)
            {
                self.touched.insert(path);
                self.touched_at = Some(now);
            }
        }
    }

    /// Looks at the sources touched since the last time: those whose content
    /// changed make the recipes that read them due.
    fn settle(&mut self) {
        self.touched_at = None;
        for path in std::mem::take(&mut self.touched) {
            let content = fingerprint(&self.root.join(&path));
            let before = match content {
                Some(content) => self.seen.insert(path.clone(), content),
                None => self.seen.remove(&path),
            };
            if before == content {
                continue;
            }
            for (index, recipe) in self.recipes.iter().enumerate() {
                if !self.unavailable[index] && recipe.sources.iter().any(|glob| matches(glob, &path)) {
                    self.pending.entry(index).or_insert_with(|| path.clone());
                }
            }
        }
    }

    /// Reports a recipe that has finished.
    fn check(&mut self) {
        let Some(running) = &mut self.running else { return };
        let status = match running.child.try_wait() {
            Ok(None) => return,
            Ok(Some(status)) => Ok(status),
            Err(error) => Err(error),
        };
        let running = self.running.take().unwrap();
        // Its last lines first; a process it left behind may hold the pipes,
        // so not for long.
        let deadline = Instant::now() + Duration::from_millis(500);
        while Instant::now() < deadline && !running.output.iter().all(JoinHandle::is_finished) {
            std::thread::sleep(Duration::from_millis(5));
        }
        let name = &self.recipes[running.index].recipe;
        let seconds = running.started.elapsed().as_secs_f64();
        match status {
            Ok(status) if status.success() => {
                devlog::info(Source::Pipeline, format!("pipeline: {name} done in {seconds:.1} s"))
            }
            Ok(status) => devlog::error(
                Source::Pipeline,
                format!("pipeline: {name} failed ({status}) after {seconds:.1} s; the game keeps the assets it has"),
            ),
            Err(error) => devlog::error(Source::Pipeline, format!("pipeline: {name}: {error}")),
        }
    }

    /// Starts the first recipe due, once nothing runs and the latest changes
    /// have been looked at (a recipe that just finished may have made a
    /// later one due).
    fn next(&mut self) {
        if self.running.is_some() || !self.touched.is_empty() {
            return;
        }
        let Some((index, cause)) = self.pending.pop_first() else { return };
        let name = &self.recipes[index].recipe;
        devlog::info(Source::Pipeline, format!("pipeline: {cause} changed; running just {name}"));
        let spawned = Command::new("just")
            .arg(name)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        match spawned {
            Ok(mut child) => {
                let output = [
                    child.stdout.take().and_then(|pipe| forward(name, pipe)),
                    child.stderr.take().and_then(|pipe| forward(name, pipe)),
                ];
                self.running = Some(Running {
                    index,
                    child,
                    started: Instant::now(),
                    output: output.into_iter().flatten().collect(),
                });
            }
            Err(error) => eprintln!("pipeline: cannot run just {name}: {error}"),
        }
    }

    /// Stops a running recipe: just passes SIGTERM on to what it runs, which
    /// gets a moment to finish before it is killed.
    fn stop(&mut self) {
        let Some(mut running) = self.running.take() else { return };
        eprintln!("pipeline: stopping {}", self.recipes[running.index].recipe);
        if terminate(&running.child) {
            let deadline = Instant::now() + Duration::from_secs(3);
            while Instant::now() < deadline {
                if let Ok(Some(_)) = running.child.try_wait() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
        let _ = running.child.kill();
        let _ = running.child.wait();
    }
}

/// Asks `child` to stop (SIGTERM); whether it was asked.
#[cfg(unix)]
fn terminate(child: &Child) -> bool {
    let Ok(pid) = libc::pid_t::try_from(child.id()) else { return false };
    // SAFETY: kill only sends a signal; the child is ours and not yet
    // waited for, so its pid is still its own.
    unsafe { libc::kill(pid, libc::SIGTERM) == 0 }
}

/// Elsewhere there is no asking: the child is killed.
#[cfg(not(unix))]
fn terminate(_child: &Child) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs_match_names_and_directories() {
        assert!(matches("models/rock/**", "models/rock/rock.py"));
        assert!(matches("models/rock/**", "models/rock/a/b.py"));
        assert!(!matches("models/rock/**", "models/rocks/rock.py"));
        assert!(matches("models/textures/*.py", "models/textures/salt.py"));
        assert!(!matches("models/textures/*.py", "models/textures/sub/salt.py"));
        assert!(!matches("models/textures/*.py", "models/textures/salt.jpg"));
        assert!(matches("src/assets/building_*.mesh.json", "src/assets/building_hut.mesh.json"));
        assert!(!matches("src/assets/building_*.mesh.json", "src/assets/buildings.png"));
        assert!(matches("models/export.py", "models/export.py"));
        assert!(!matches("models/export.py", "models/export.pyc"));
        assert!(matches("**/*.py", "a.py"));
        assert!(matches("**/*.py", "models/a/b.py"));
        assert!(matches("models/?im/*", "models/aim/aim.py"));
    }

    #[test]
    fn watches_each_glob_from_its_last_plain_directory() {
        let directories = watched(&[
            "models/rock/**",
            "models/export.py",
            "models/textures/*.py",
            "src/assets/crater.mesh.json",
            "models/rock/rock.py",
            "models/**/*.blend",
        ]);
        assert_eq!(directories, vec![("models".to_owned(), true), ("src/assets".to_owned(), false),]);
        assert_eq!(
            watched(&["models/rock/**", "models/aim/aim.py"]),
            vec![("models/aim".to_owned(), false), ("models/rock".to_owned(), true),]
        );
    }

    #[test]
    fn temporaries_are_never_sources() {
        assert!(temporary("models/rock/.rock.py.swp"));
        assert!(temporary("models/rock/rock.py~"));
        assert!(temporary("models/__pycache__/export.cpython-314.pyc"));
        assert!(temporary("models/phone/rotary.blend1"));
        assert!(!temporary("models/phone/rotary.blend"));
        assert!(!temporary("models/rock/rock.py"));
    }

    fn scheduler(root: &Path, recipes: &[(&str, &[&str])]) -> Scheduler {
        let recipes: Vec<Recipe> = recipes
            .iter()
            .map(|(name, sources)| Recipe {
                recipe: name.to_string(),
                sources: sources.iter().map(|glob| glob.to_string()).collect(),
                requires: Vec::new(),
            })
            .collect();
        Scheduler {
            root: root.to_owned(),
            unavailable: vec![false; recipes.len()],
            recipes,
            seen: HashMap::new(),
            touched: BTreeSet::new(),
            touched_at: None,
            pending: BTreeMap::new(),
            running: None,
        }
    }

    fn touch(scheduler: &mut Scheduler, path: &Path) {
        let event = notify::Event::new(EventKind::Modify(notify::event::ModifyKind::Any)).add_path(path.to_owned());
        scheduler.note(&event, Instant::now());
        scheduler.settle();
    }

    #[test]
    fn stopping_asks_a_running_recipe_to_end() {
        let mut pipeline = scheduler(Path::new("/"), &[("slow", &["models/**"])]);
        let child = Command::new("sleep").arg("30").spawn().unwrap();
        pipeline.running = Some(Running { index: 0, child, started: Instant::now(), output: Vec::new() });
        let started = Instant::now();
        pipeline.stop();
        assert!(pipeline.running.is_none());
        // Asked, it ends at once rather than when killed after a wait.
        assert!(started.elapsed() < Duration::from_secs(2), "{:?}", started.elapsed());
    }

    #[test]
    fn content_changes_make_recipes_due_in_order() {
        let root = std::env::temp_dir().join(format!("pen-pipeline-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("models/crater")).unwrap();
        std::fs::create_dir_all(root.join("src/assets")).unwrap();
        let script = root.join("models/crater/crater.py");
        let mesh = root.join("src/assets/crater.mesh.json");
        std::fs::write(&script, "a").unwrap();
        std::fs::write(&mesh, "{}").unwrap();
        let mut pipeline = scheduler(
            &root,
            &[("crater", &["models/crater/**"]), ("depot", &["models/depot/**", "src/assets/crater.mesh.json"])],
        );
        pipeline.remember(&watched(&["models/crater/**", "models/depot/**", "src/assets/crater.mesh.json"]));
        assert_eq!(pipeline.seen.len(), 2);

        // Rewriting the same content, or a file no recipe reads, does nothing.
        std::fs::write(&mesh, "{}").unwrap();
        touch(&mut pipeline, &mesh);
        std::fs::write(root.join("src/assets/other.json"), "1").unwrap();
        touch(&mut pipeline, &root.join("src/assets/other.json"));
        touch(&mut pipeline, &root.join("models/crater/.crater.py.swp"));
        assert!(pipeline.pending.is_empty());

        // The output changing makes its reader due; the source, its recipe,
        // which runs first.
        std::fs::write(&mesh, "{\"positions\":[]}").unwrap();
        touch(&mut pipeline, &mesh);
        std::fs::write(&script, "b").unwrap();
        touch(&mut pipeline, &script);
        assert_eq!(pipeline.pending.keys().copied().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(pipeline.pending[&0], "models/crater/crater.py");
        assert_eq!(pipeline.pending[&1], "src/assets/crater.mesh.json");

        // A file that comes and goes between looks was never there.
        pipeline.pending.clear();
        let temporary = root.join("models/crater/4913");
        std::fs::write(&temporary, "").unwrap();
        std::fs::remove_file(&temporary).unwrap();
        touch(&mut pipeline, &temporary);
        assert!(pipeline.pending.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }
}
