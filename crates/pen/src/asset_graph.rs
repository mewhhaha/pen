//! The asset pipeline as a graph of steps, with a cache of what each built.
//!
//! A checkout's `pipeline.json` lists its steps: each a Justfile recipe
//! (`just NAME`) or one of pen's own (`builtin`, see [`BUILTINS`]), the files
//! it builds from (`sources`) and the files it writes (`outputs`), as globs
//! relative to the checkout. A step that reads a file another writes runs
//! after it; otherwise steps keep the file's order, and a cycle is refused.
//!
//! A step's key is the SHA-256 of what it is (its recipe's text, or the
//! built-in's name, version and options), the outputs it declares, and each
//! source file's path and content. What it wrote goes into a
//! content-addressed store (store.rs) under that key, so when the same
//! sources come round again, in this checkout or another, here or on a CI
//! runner sharing the directory, the outputs are copied back instead of the
//! step running again. Keys hold no absolute paths and no times, so a cache
//! made anywhere serves everywhere, and a runner without Blender can restore
//! what Blender made elsewhere.
//!
//! `pen assets` runs the steps through the cache, and `pen dev` (pipeline.rs)
//! runs each step whose sources change through it.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::pipeline::{installed, matches, temporary, watched};
use crate::store::{Digest, Store, digest_file};

/// The cache's records of what steps built, in its store.
const RECORDS: &str = "steps";
/// Changes when keys are made differently, so older records are not reused.
const KEY_VERSION: u32 = 1;
/// Records unused this long are pruned, and the oldest beyond this size.
const KEEP_FOR: Duration = Duration::from_secs(30 * 86_400);
const KEEP_BYTES: u64 = 4 << 30;

#[derive(Deserialize)]
struct Config {
    recipes: Vec<Step>,
}

/// One step of the pipeline.
#[derive(Clone, Debug, Deserialize)]
pub struct Step {
    /// Its name: the Justfile recipe it runs (`just NAME`), unless built in.
    pub recipe: String,
    /// Globs relative to the checkout: `*` and `?` within a name, `**` for
    /// any number of directories.
    pub sources: Vec<String>,
    /// What it writes, as globs like its sources. A step without outputs is
    /// not cached: it runs whenever it is asked to.
    #[serde(default)]
    pub outputs: Vec<String>,
    /// Programs it runs that may not be installed, such as Blender.
    #[serde(default)]
    pub requires: Vec<String>,
    /// One of [`BUILTINS`], which pen runs itself in place of a recipe.
    #[serde(default)]
    pub builtin: Option<String>,
    /// The built-in's options.
    #[serde(default)]
    pub options: serde_json::Value,
}

/// What a built-in step is given.
pub struct Context<'a> {
    /// The checkout.
    pub root: &'a Path,
    pub step: &'a Step,
    /// Its source files, relative to the checkout (`/` between names), sorted.
    pub sources: Vec<String>,
}

/// A step pen runs itself, which a step names with `builtin`.
pub struct Builtin {
    pub name: &'static str,
    /// Raised whenever it writes something else for the same sources and
    /// options, so that what an older one cached is not reused.
    pub version: u32,
    pub run: fn(&Context) -> Result<()>,
}

/// pen's built-in steps. An importer or encoder (a model format, a texture
/// compressor) joins the pipeline by adding itself here.
pub const BUILTINS: &[Builtin] = &[Builtin { name: "atlas", version: 1, run: crate::atlas::step }];

fn builtin(name: &str) -> Result<&'static Builtin> {
    BUILTINS.iter().find(|builtin| builtin.name == name).with_context(|| {
        let names: Vec<&str> = BUILTINS.iter().map(|builtin| builtin.name).collect();
        format!("pipeline.json: no built-in step is called {name} (there are {})", names.join(", "))
    })
}

/// The checkout's steps, in the order they run.
pub struct Graph {
    pub root: PathBuf,
    pub steps: Vec<Step>,
    /// For each step, the steps whose outputs it reads.
    pub inputs: Vec<Vec<usize>>,
}

impl Graph {
    /// The checkout's pipeline, or None when it has no `pipeline.json`.
    pub fn load(root: &Path) -> Result<Option<Graph>> {
        let file = root.join("pipeline.json");
        let text = match std::fs::read_to_string(&file) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).with_context(|| format!("reading {}", file.display())),
        };
        Graph::parse(root, &text).with_context(|| format!("reading {}", file.display())).map(Some)
    }

    pub fn parse(root: &Path, text: &str) -> Result<Graph> {
        let config: Config = serde_json::from_str(text)?;
        let mut graph = Graph { root: root.to_owned(), steps: config.recipes, inputs: Vec::new() };
        let mut names = BTreeSet::new();
        for step in &graph.steps {
            ensure!(names.insert(step.recipe.as_str()), "two steps are called {}", step.recipe);
            if let Some(name) = &step.builtin {
                builtin(name)?;
            }
        }
        graph.order()?;
        Ok(graph)
    }

    /// Every file under the checkout that one of `globs` matches, relative
    /// and sorted, leaving out editors' and tools' temporaries.
    pub fn files(&self, globs: &[String]) -> Vec<String> {
        fn walk(root: &Path, directory: &Path, recursive: bool, files: &mut BTreeSet<String>) {
            let Ok(entries) = std::fs::read_dir(directory) else { return };
            for entry in entries.flatten() {
                let path = entry.path();
                match entry.file_type() {
                    Ok(kind) if kind.is_dir() && recursive => walk(root, &path, true, files),
                    Ok(kind) if kind.is_file() => {
                        if let Ok(relative) = path.strip_prefix(root) {
                            let parts: Vec<String> = relative
                                .components()
                                .map(|part| part.as_os_str().to_string_lossy().into_owned())
                                .collect();
                            files.insert(parts.join("/"));
                        }
                    }
                    _ => {}
                }
            }
        }
        let patterns: Vec<&str> = globs.iter().map(String::as_str).collect();
        let mut files = BTreeSet::new();
        for (directory, recursive) in watched(&patterns) {
            walk(&self.root, &self.root.join(directory), recursive, &mut files);
        }
        files.into_iter().filter(|path| !temporary(path) && globs.iter().any(|glob| matches(glob, path))).collect()
    }

    /// Sorts the steps so that each comes after the steps whose outputs it
    /// reads, keeping the file's order otherwise.
    fn order(&mut self) -> Result<()> {
        let count = self.steps.len();
        // The files that could join two steps: those on disk that some step
        // reads, and the sources named without wildcards.
        let sources: Vec<String> = self.steps.iter().flat_map(|step| step.sources.clone()).collect();
        let mut candidates: BTreeSet<String> = self.files(&sources).into_iter().collect();
        candidates.extend(sources.iter().filter(|glob| !glob.contains(['*', '?'])).cloned());
        let reads = |reader: &Step, writer: &Step| {
            candidates.iter().any(|file| {
                writer.outputs.iter().any(|glob| matches(glob, file))
                    && reader.sources.iter().any(|glob| matches(glob, file))
            }) || writer
                .outputs
                .iter()
                .any(|output| !output.contains(['*', '?']) && reader.sources.iter().any(|glob| matches(glob, output)))
        };
        let mut inputs = vec![Vec::new(); count];
        for (reader, step) in self.steps.iter().enumerate() {
            for (writer, other) in self.steps.iter().enumerate() {
                if reader == writer {
                    if let Some(file) = candidates.iter().find(|file| {
                        step.outputs.iter().any(|glob| matches(glob, file))
                            && step.sources.iter().any(|glob| matches(glob, file))
                    }) {
                        bail!("step {} reads {file}, which it writes", step.recipe);
                    }
                } else if reads(step, other) {
                    inputs[reader].push(writer);
                }
            }
        }
        // Kahn's algorithm, taking the first step in the file's order that
        // is ready.
        let mut waiting: Vec<usize> = inputs.iter().map(Vec::len).collect();
        let mut ready: BTreeSet<usize> = (0..count).filter(|&index| waiting[index] == 0).collect();
        let mut order = Vec::with_capacity(count);
        while let Some(index) = ready.pop_first() {
            order.push(index);
            for (reader, read) in inputs.iter().enumerate() {
                if read.contains(&index) {
                    waiting[reader] -= 1;
                    if waiting[reader] == 0 {
                        ready.insert(reader);
                    }
                }
            }
        }
        if order.len() < count {
            let stuck: Vec<&str> = (0..count)
                .filter(|index| !order.contains(index))
                .map(|index| self.steps[index].recipe.as_str())
                .collect();
            bail!("these steps read each other's outputs in a cycle: {}", stuck.join(", "));
        }
        let place: BTreeMap<usize, usize> = order.iter().enumerate().map(|(new, &old)| (old, new)).collect();
        let steps = order.iter().map(|&old| self.steps[old].clone()).collect();
        self.inputs = order
            .iter()
            .map(|&old| {
                let mut read: Vec<usize> = inputs[old].iter().map(|writer| place[writer]).collect();
                read.sort_unstable();
                read
            })
            .collect();
        self.steps = steps;
        Ok(())
    }

    /// Whether a step after this one writes `path` too, so that what this
    /// one wrote there does not last when all of them run.
    pub fn overwritten(&self, index: usize, path: &str) -> bool {
        self.steps[index + 1..].iter().any(|later| later.outputs.iter().any(|glob| matches(glob, path)))
    }

    /// The step called `name`.
    pub fn find(&self, name: &str) -> Option<usize> {
        self.steps.iter().position(|step| step.recipe == name)
    }

    /// The step's key: what it is, what it declares it writes, and each of
    /// its sources' path and content as they are now.
    pub fn key(&self, index: usize) -> Result<Digest> {
        let step = &self.steps[index];
        let mut text = format!("pen asset step {KEY_VERSION}\nname {}\n", step.recipe);
        match &step.builtin {
            Some(name) => {
                let builtin = builtin(name)?;
                text +=
                    &format!("builtin {name} {}\noptions {}\n", builtin.version, serde_json::to_string(&step.options)?);
            }
            None => text += &format!("recipe {}\n", recipe_text(&self.root, &step.recipe)?),
        }
        for output in &step.outputs {
            text += &format!("output {output}\n");
        }
        for path in self.files(&step.sources) {
            let (digest, _) = digest_file(&self.root.join(&path))?;
            text += &format!("source {path} {digest}\n");
        }
        Ok(Digest::of(text.as_bytes()))
    }

    /// Runs the step itself, its output going to this process's.
    fn run(&self, index: usize) -> Result<()> {
        let step = &self.steps[index];
        if let Some(name) = &step.builtin {
            let context = Context { root: &self.root, step, sources: self.files(&step.sources) };
            return (builtin(name)?.run)(&context).with_context(|| format!("step {}", step.recipe));
        }
        let status = Command::new("just")
            .arg(&step.recipe)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .status()
            .context("running just")?;
        ensure!(status.success(), "just {} failed ({status})", step.recipe);
        Ok(())
    }
}

/// A recipe as `just --show` prints it: what running it does.
fn recipe_text(root: &Path, recipe: &str) -> Result<String> {
    let output = Command::new("just")
        .args(["--show", recipe])
        .current_dir(root)
        .stderr(Stdio::piped())
        .output()
        .context("running just --show")?;
    ensure!(
        output.status.success(),
        "the Justfile has no recipe {recipe}: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Where the pipeline's cache is: `PEN_ASSET_CACHE`, else `assets` in pen's
/// cache directory, which `pen clean` empties. Pointing several checkouts
/// or CI runs at one directory shares what each built.
pub fn default_cache() -> Option<PathBuf> {
    match std::env::var_os("PEN_ASSET_CACHE") {
        Some(directory) if !directory.is_empty() => Some(PathBuf::from(directory)),
        _ => dirs::cache_dir().map(|directory| directory.join("pen/assets")),
    }
}

/// What a step built from one key's sources.
#[derive(Debug, Deserialize, Serialize)]
pub struct Record {
    pub step: String,
    /// How long building took.
    pub seconds: f64,
    pub outputs: Vec<Output>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Output {
    pub path: String,
    pub digest: Digest,
    pub size: u64,
}

/// The pipeline's cache: a store of what steps built, by key.
pub struct Cache {
    store: Store,
}

impl Cache {
    pub fn open(directory: &Path) -> Result<Cache> {
        Ok(Cache { store: Store::open(directory)? })
    }

    pub fn lookup(&self, key: &Digest) -> Result<Option<Record>> {
        self.store.read_record(RECORDS, &key.to_string())
    }

    /// Keeps the step's outputs as they are now as what `key` builds.
    pub fn record(&self, graph: &Graph, index: usize, key: &Digest, seconds: f64) -> Result<Record> {
        let step = &graph.steps[index];
        let mut outputs = Vec::new();
        for path in graph.files(&step.outputs) {
            let (digest, size) = self.store.put_file(&graph.root.join(&path))?;
            outputs.push(Output { path, digest, size });
        }
        let record = Record { step: step.recipe.clone(), seconds, outputs };
        self.store.write_record(RECORDS, &key.to_string(), &record)?;
        Ok(record)
    }

    /// The outputs of step `index` whose files differ from what its record
    /// holds, leaving out those a later step writes too.
    pub fn differing<'a>(&self, graph: &Graph, index: usize, record: &'a Record) -> Vec<&'a Output> {
        record
            .outputs
            .iter()
            .filter(|output| !graph.overwritten(index, &output.path))
            .filter(|output| {
                digest_file(&graph.root.join(&output.path)).map(|(digest, _)| digest).ok() != Some(output.digest)
            })
            .collect()
    }

    /// Writes back the outputs of step `index` that differ from its record;
    /// how many.
    pub fn restore(&self, graph: &Graph, index: usize, record: &Record) -> Result<usize> {
        let root = &graph.root;
        let differing = self.differing(graph, index, record);
        for output in &differing {
            self.store.copy_out(&output.digest, &root.join(&output.path))?;
        }
        Ok(differing.len())
    }

    pub fn prune(&self) -> Result<u64> {
        self.store.prune(RECORDS, KEEP_FOR, KEEP_BYTES)
    }
}

/// What `pen assets` does.
pub struct Options {
    /// The cache's directory.
    pub cache: PathBuf,
    /// Only report whether each step's outputs are what the cache holds for
    /// its sources; build nothing.
    pub check: bool,
    /// Only these steps (all, when empty).
    pub only: Vec<String>,
}

/// How a step came out.
#[derive(Debug, PartialEq)]
pub enum Outcome {
    /// Its outputs were already what its sources build.
    Current,
    /// Its outputs were copied back from the cache.
    Restored(usize),
    /// It ran.
    Built,
    /// It could not run: a program it needs is not installed.
    Skipped(Vec<String>),
    /// (Checking.) The cache has no build of its sources.
    Unknown,
    /// (Checking.) Its outputs differ from the cache's build of its sources.
    Stale(Vec<String>),
}

/// Brings each step's outputs up to date with its sources, through the
/// cache: copied back when it has them, else built and kept. Checking,
/// says which differ instead. Returns each step's name and outcome.
pub fn run(root: &Path, options: &Options) -> Result<Vec<(String, Outcome)>> {
    let graph = Graph::load(root)?.context("this checkout has no pipeline.json")?;
    for name in &options.only {
        ensure!(graph.find(name).is_some(), "pipeline.json has no step {name}");
    }
    let cache = Cache::open(&options.cache)?;
    let mut outcomes = Vec::new();
    for index in 0..graph.steps.len() {
        let step = &graph.steps[index];
        if !options.only.is_empty() && !options.only.contains(&step.recipe) {
            continue;
        }
        let outcome = advance(&graph, &cache, index, options.check).with_context(|| format!("step {}", step.recipe))?;
        let said = match &outcome {
            Outcome::Current => "up to date".to_owned(),
            Outcome::Restored(count) => format!("{count} outputs restored from the cache"),
            Outcome::Built => "built".to_owned(),
            Outcome::Skipped(missing) => format!("skipped: {} not installed", missing.join(", ")),
            Outcome::Unknown => "not checked: the cache has no build of its sources".to_owned(),
            Outcome::Stale(paths) => format!("out of date: {}", paths.join(", ")),
        };
        eprintln!("assets: {}: {said}", step.recipe);
        outcomes.push((step.recipe.clone(), outcome));
    }
    if !options.check {
        let freed = cache.prune()?;
        if freed > 0 {
            eprintln!("assets: pruned {:.1} MB from the cache", freed as f64 / 1e6);
        }
    }
    Ok(outcomes)
}

/// Says what `run` did in a line; checking, fails when a step's outputs are
/// not what its sources build.
pub fn summarize(outcomes: &[(String, Outcome)], check: bool) -> Result<()> {
    let count = |matching: fn(&Outcome) -> bool| outcomes.iter().filter(|(_, outcome)| matching(outcome)).count();
    let stale: Vec<&str> = outcomes
        .iter()
        .filter(|(_, outcome)| matches!(outcome, Outcome::Stale(_)))
        .map(|(name, _)| name.as_str())
        .collect();
    let mut parts = vec![format!("{} up to date", count(|outcome| *outcome == Outcome::Current))];
    for (number, what) in [
        (count(|outcome| *outcome == Outcome::Built), "built"),
        (count(|outcome| matches!(outcome, Outcome::Restored(_))), "restored from the cache"),
        (count(|outcome| matches!(outcome, Outcome::Skipped(_))), "skipped"),
        (count(|outcome| *outcome == Outcome::Unknown), "not in the cache"),
        (stale.len(), "out of date"),
    ] {
        if number > 0 {
            parts.push(format!("{number} {what}"));
        }
    }
    eprintln!("assets: {} steps: {}", outcomes.len(), parts.join(", "));
    ensure!(!check || stale.is_empty(), "these steps' outputs are not what their sources build: {}", stale.join(", "));
    Ok(())
}

fn advance(graph: &Graph, cache: &Cache, index: usize, check: bool) -> Result<Outcome> {
    let step = &graph.steps[index];
    let cached = !step.outputs.is_empty();
    let key = graph.key(index)?;
    if cached && let Some(record) = cache.lookup(&key)? {
        let differing = cache.differing(graph, index, &record);
        return Ok(match (differing.is_empty(), check) {
            (true, _) => Outcome::Current,
            (false, true) => Outcome::Stale(differing.iter().map(|output| output.path.clone()).collect()),
            (false, false) => Outcome::Restored(cache.restore(graph, index, &record)?),
        });
    }
    if check {
        return Ok(Outcome::Unknown);
    }
    let missing: Vec<String> = step.requires.iter().filter(|tool| !installed(tool)).cloned().collect();
    if !missing.is_empty() {
        return Ok(Outcome::Skipped(missing));
    }
    let started = Instant::now();
    graph.run(index)?;
    if cached {
        // Keyed by the sources it ran with: a step that rewrote one of
        // them is keyed afresh.
        let record = cache.record(graph, index, &graph.key(index)?, started.elapsed().as_secs_f64())?;
        ensure!(!record.outputs.is_empty(), "it wrote nothing its outputs ({}) name", step.outputs.join(", "));
    }
    Ok(Outcome::Built)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkout(name: &str, pipeline: &str, justfile: &str) -> PathBuf {
        let root = crate::store::scratch(name);
        std::fs::write(root.join("pipeline.json"), pipeline).unwrap();
        std::fs::write(root.join("Justfile"), justfile).unwrap();
        root
    }

    fn names(graph: &Graph) -> Vec<&str> {
        graph.steps.iter().map(|step| step.recipe.as_str()).collect()
    }

    #[test]
    fn steps_run_after_the_steps_whose_outputs_they_read() {
        let root = crate::store::scratch("graph-order");
        std::fs::create_dir_all(root.join("src/assets")).unwrap();
        std::fs::write(root.join("src/assets/crater.mesh.json"), "{}").unwrap();
        // Listed reader first: the graph puts the writer before it.
        let text = r#"{"recipes": [
            {"recipe": "depot", "sources": ["models/depot/**", "src/assets/crater.mesh.json"], "outputs": ["src/assets/depot_*.mesh.json"]},
            {"recipe": "aim", "sources": ["models/aim/**"], "outputs": ["src/assets/aim.png"]},
            {"recipe": "ground", "sources": ["models/ground/**", "src/assets/depot_house.mesh.json"], "outputs": ["src/assets/tufts_*.mesh.json"]},
            {"recipe": "crater", "sources": ["models/crater/**"], "outputs": ["src/assets/crater.mesh.json"]}
        ]}"#;
        let graph = Graph::parse(&root, text).unwrap();
        assert_eq!(names(&graph), ["aim", "crater", "depot", "ground"]);
        assert_eq!(graph.inputs, [vec![], vec![], vec![1], vec![2]]);
        let cycle = r#"{"recipes": [
            {"recipe": "a", "sources": ["b.txt"], "outputs": ["a.txt"]},
            {"recipe": "b", "sources": ["a.txt"], "outputs": ["b.txt"]}
        ]}"#;
        let refused = format!("{:#}", Graph::parse(&root, cycle).err().unwrap());
        assert!(refused.contains("cycle: a, b"), "{refused}");
        let own = r#"{"recipes": [{"recipe": "a", "sources": ["src/**"], "outputs": ["src/assets/*.json"]}]}"#;
        assert!(format!("{:#}", Graph::parse(&root, own).err().unwrap()).contains("reads src/assets/crater.mesh.json"));
        let unknown = r#"{"recipes": [{"recipe": "a", "sources": [], "builtin": "teleport"}]}"#;
        assert!(format!("{:#}", Graph::parse(&root, unknown).err().unwrap()).contains("no built-in step"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_file_two_steps_write_is_the_last_ones() {
        if !installed("just") {
            eprintln!("skipped: just is not installed");
            return;
        }
        // As gdev's buildings and depot both write the buildings' palette.
        let root = checkout(
            "graph-shared",
            r#"{"recipes": [
                {"recipe": "first", "sources": ["a.txt"], "outputs": ["out/palette.txt", "out/first.txt"]},
                {"recipe": "second", "sources": ["b.txt"], "outputs": ["out/palette.txt"]}
            ]}"#,
            "first:\n  mkdir -p out\n  echo first > out/palette.txt\n  cp a.txt out/first.txt\n\n\\n             second:\n  mkdir -p out\n  echo second > out/palette.txt\n",
        );
        std::fs::write(root.join("a.txt"), "a").unwrap();
        std::fs::write(root.join("b.txt"), "b").unwrap();
        let options = Options { cache: root.join("cache"), check: false, only: Vec::new() };
        let outcomes = |list: Vec<(String, Outcome)>| list.into_iter().map(|(_, outcome)| outcome).collect::<Vec<_>>();
        assert_eq!(outcomes(run(&root, &options).unwrap()), [Outcome::Built, Outcome::Built]);
        assert_eq!(std::fs::read_to_string(root.join("out/palette.txt")).unwrap(), "second\n");
        // The first step's palette does not last, so it is neither stale nor restored.
        assert_eq!(outcomes(run(&root, &options).unwrap()), [Outcome::Current, Outcome::Current]);
        let check = Options { check: true, ..options };
        assert_eq!(outcomes(run(&root, &check).unwrap()), [Outcome::Current, Outcome::Current]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn steps_are_cached_by_their_sources_and_restored() {
        if !installed("just") {
            eprintln!("skipped: just is not installed");
            return;
        }
        let root = checkout(
            "graph-cache",
            r#"{"recipes": [
                {"recipe": "shout", "sources": ["models/*.txt"], "outputs": ["out/shout.txt"]},
                {"recipe": "twice", "sources": ["out/shout.txt"], "outputs": ["out/twice.txt"]}
            ]}"#,
            "shout:\n  mkdir -p out\n  tr a-z A-Z < models/word.txt > out/shout.txt\n  echo ran >> runs.log\n\n\\n             twice:\n  cat out/shout.txt out/shout.txt > out/twice.txt\n  echo ran >> runs.log\n",
        );
        std::fs::create_dir_all(root.join("models")).unwrap();
        std::fs::write(root.join("models/word.txt"), "salt\n").unwrap();
        let options = Options { cache: root.join("cache"), check: false, only: Vec::new() };
        let runs = || std::fs::read_to_string(root.join("runs.log")).unwrap_or_default().lines().count();
        let outcomes = |list: Vec<(String, Outcome)>| list.into_iter().map(|(_, outcome)| outcome).collect::<Vec<_>>();

        assert_eq!(outcomes(run(&root, &options).unwrap()), [Outcome::Built, Outcome::Built]);
        assert_eq!(std::fs::read_to_string(root.join("out/twice.txt")).unwrap(), "SALT\nSALT\n");
        assert_eq!(runs(), 2);
        // Nothing changed: nothing runs.
        assert_eq!(outcomes(run(&root, &options).unwrap()), [Outcome::Current, Outcome::Current]);
        // An edit runs both, the second because the first's output changed.
        std::fs::write(root.join("models/word.txt"), "rock\n").unwrap();
        assert_eq!(outcomes(run(&root, &options).unwrap()), [Outcome::Built, Outcome::Built]);
        assert_eq!(runs(), 4);
        // Undoing it runs nothing: both come back from the cache.
        std::fs::write(root.join("models/word.txt"), "salt\n").unwrap();
        assert_eq!(outcomes(run(&root, &options).unwrap()), [Outcome::Restored(1), Outcome::Restored(1)]);
        assert_eq!(std::fs::read_to_string(root.join("out/twice.txt")).unwrap(), "SALT\nSALT\n");
        assert_eq!(runs(), 4);
        // Another checkout of the same sources shares the cache, without running.
        let other = crate::store::scratch("graph-cache-other");
        for file in ["pipeline.json", "Justfile", "models/word.txt"] {
            std::fs::create_dir_all(other.join(file).parent().unwrap()).unwrap();
            std::fs::copy(root.join(file), other.join(file)).unwrap();
        }
        assert_eq!(outcomes(run(&other, &options).unwrap()), [Outcome::Restored(1), Outcome::Restored(1)]);
        assert!(!other.join("runs.log").exists());
        // Checking reports an output edited by hand, and builds nothing.
        std::fs::write(other.join("out/twice.txt"), "pepper\n").unwrap();
        let check = Options { check: true, ..options };
        assert_eq!(
            outcomes(run(&other, &check).unwrap()),
            [Outcome::Current, Outcome::Stale(vec!["out/twice.txt".into()])]
        );
        std::fs::write(other.join("out/twice.txt"), "SALT\nSALT\n").unwrap();
        std::fs::write(other.join("models/word.txt"), "brick\n").unwrap();
        assert_eq!(outcomes(run(&other, &check).unwrap()), [Outcome::Unknown, Outcome::Current]);
        // Editing the recipe is a new key too.
        let justfile = std::fs::read_to_string(root.join("Justfile")).unwrap().replace("tr a-z A-Z", "tr a-z a-z");
        std::fs::write(root.join("Justfile"), justfile).unwrap();
        let options = Options { check: false, only: vec!["shout".into()], ..check };
        assert_eq!(outcomes(run(&root, &options).unwrap()), [Outcome::Built]);
        assert_eq!(std::fs::read_to_string(root.join("out/shout.txt")).unwrap(), "salt\n");
        std::fs::remove_dir_all(root).unwrap();
        std::fs::remove_dir_all(other).unwrap();
    }
}
