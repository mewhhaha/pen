//! `pen bench`'s results, and comparing them from build to build. A bench
//! runs the game's program without rendering (`packets::run`) and measures
//! the guest: each frame's time, a snapshot every moment's interval (what a
//! history costs) and loading the module. `--runs N` runs it N times and
//! keeps each metric's best, which shrugs off a machine's other work.
//!
//! `--json` prints the results on stdout as a JSON document ([`Results`]);
//! `--compare BASE.json` compares them with another build's and fails when
//! a metric is slower by more than its threshold. CI benches a commit and
//! its parent this way, on the same runner, and fails the job on a
//! regression.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::packets::{Measured, warmup};

/// The version of [`Results`]' layout.
const FORMAT: u32 = 1;

/// One bench, as `--json` prints it and `--compare` reads it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Results {
    pub format: u32,
    /// pen's version, and the build label it was built with (`PEN_BUILD`).
    pub pen: String,
    pub build: Option<String>,
    /// The game runtime's compatibility mark (wasmtime and blot-host).
    pub runtime: String,
    /// FNV-1a of the game.wasm benched, so results of different games are
    /// not mistaken for a regression.
    pub game: String,
    pub frames: usize,
    pub warmup: usize,
    pub runs: usize,
    /// Each metric's value; all are milliseconds, lower is better.
    pub metrics: BTreeMap<String, f64>,
}

/// The metrics, and by how much each may grow (percent) before a comparison
/// calls it a regression: medians and means hold steady, tails and one-off
/// costs less so.
const METRICS: [(&str, f64); 5] = [
    ("guest_median_ms", 10.0),
    ("guest_mean_ms", 10.0),
    ("guest_p95_ms", 20.0),
    ("snapshot_median_ms", 20.0),
    ("load_ms", 25.0),
];

/// Changes smaller than this (ms) are the clock's and the machine's, not
/// the build's.
const NOISE_MS: f64 = 0.005;

/// FNV-1a over bytes, as a hex string.
pub fn fingerprint(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// The value at `percent` of sorted `values`.
fn percentile(sorted: &[f64], percent: usize) -> f64 {
    sorted[(sorted.len() - 1) * percent / 100]
}

/// One run's metrics: its frames past the warm-up, and its snapshots.
fn metrics(measured: &Measured) -> Result<BTreeMap<String, f64>> {
    let warm = warmup(measured.frames.len());
    let mut frames = measured.frames[warm..].to_vec();
    ensure!(!frames.is_empty(), "a bench needs more frames than its {warm} warm-up frames (--frames)");
    frames.sort_by(f64::total_cmp);
    let mut metrics = BTreeMap::new();
    metrics.insert("guest_median_ms".to_owned(), percentile(&frames, 50));
    metrics.insert("guest_p95_ms".to_owned(), percentile(&frames, 95));
    metrics.insert("guest_mean_ms".to_owned(), frames.iter().sum::<f64>() / frames.len() as f64);
    let mut snapshots = measured.snapshots.clone();
    if !snapshots.is_empty() {
        snapshots.sort_by(f64::total_cmp);
        metrics.insert("snapshot_median_ms".to_owned(), percentile(&snapshots, 50));
    }
    Ok(metrics)
}

/// The results of `runs`, each metric its best (lowest) over them; `load`
/// is how long loading the module took.
pub fn results(wasm: &[u8], load_ms: f64, runs: &[Measured]) -> Result<Results> {
    let mut best: BTreeMap<String, f64> = BTreeMap::new();
    for run in runs {
        for (name, value) in metrics(run)? {
            best.entry(name).and_modify(|kept| *kept = kept.min(value)).or_insert(value);
        }
    }
    best.insert("load_ms".to_owned(), load_ms);
    let frames = runs.first().map_or(0, |run| run.frames.len());
    Ok(Results {
        format: FORMAT,
        pen: env!("CARGO_PKG_VERSION").to_owned(),
        build: option_env!("PEN_BUILD").map(str::to_owned),
        runtime: crate::project::RUNTIME.to_owned(),
        game: fingerprint(wasm),
        frames,
        warmup: warmup(frames),
        runs: runs.len(),
        metrics: best,
    })
}

pub fn read(path: &Path) -> Result<Results> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let results: Results = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    ensure!(
        results.format == FORMAT,
        "{} is a bench of format {}, and this pen reads format {FORMAT}",
        path.display(),
        results.format
    );
    Ok(results)
}

/// How one metric changed from a base to a new bench.
#[derive(Debug, PartialEq)]
pub struct Change {
    pub metric: String,
    pub base: f64,
    pub new: f64,
    /// The threshold it was held to, in percent.
    pub threshold: f64,
}

impl Change {
    /// The change in percent of the base.
    pub fn percent(&self) -> f64 {
        if self.base > 0.0 { (self.new / self.base - 1.0) * 100.0 } else { 0.0 }
    }

    /// Slower by more than the threshold, and by more than noise.
    pub fn regressed(&self) -> bool {
        self.new - self.base > NOISE_MS && self.percent() > self.threshold
    }
}

/// Each metric both benches have, held to `threshold` percent, or to its own
/// when None.
pub fn compare(base: &Results, new: &Results, threshold: Option<f64>) -> Vec<Change> {
    new.metrics
        .iter()
        .filter_map(|(metric, &value)| {
            let &before = base.metrics.get(metric)?;
            let own = METRICS.iter().find(|(name, _)| name == metric).map_or(20.0, |(_, threshold)| *threshold);
            Some(Change { metric: metric.clone(), base: before, new: value, threshold: threshold.unwrap_or(own) })
        })
        .collect()
}

/// The bench at each of `paths`, as one: each metric its best over them
/// (CI benches the base build several times).
pub fn read_best(paths: &[PathBuf]) -> Result<Results> {
    let (first, rest) = paths.split_first().context("no bench to compare with")?;
    let mut best = read(first)?;
    for path in rest {
        for (name, value) in read(path)?.metrics {
            best.metrics.entry(name).and_modify(|kept| *kept = kept.min(value)).or_insert(value);
        }
    }
    Ok(best)
}

/// Says how `new` compares with the bench at `paths` (the best of them),
/// and fails when a metric regressed.
pub fn check(paths: &[PathBuf], new: &Results, threshold: Option<f64>) -> Result<()> {
    let base = read_best(paths)?;
    let shown: Vec<String> = paths.iter().map(|path| path.display().to_string()).collect();
    eprintln!("compared with {}:", shown.join(", "));
    if base.game != new.game {
        eprintln!("  warning: it benched another game.wasm ({} here, {} there)", new.game, base.game);
    }
    if (base.frames, base.warmup) != (new.frames, new.warmup) {
        eprintln!(
            "  warning: it ran {} frames after {} warm-up, and this {} after {}",
            base.frames - base.warmup,
            base.warmup,
            new.frames - new.warmup,
            new.warmup
        );
    }
    let changes = compare(&base, new, threshold);
    for change in &changes {
        eprintln!(
            "  {:<20}{:>10.4} -> {:<10.4}{:>+7.1}%  (threshold {}%){}",
            change.metric,
            change.base,
            change.new,
            change.percent(),
            change.threshold,
            if change.regressed() { "  REGRESSED" } else { "" }
        );
    }
    let regressed: Vec<String> = changes
        .iter()
        .filter(|change| change.regressed())
        .map(|change| {
            format!("{} {:+.1}% ({:.4} -> {:.4} ms)", change.metric, change.percent(), change.base, change.new)
        })
        .collect();
    if !regressed.is_empty() {
        bail!("bench regressed beyond its thresholds: {}", regressed.join(", "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured(frames: &[f64], snapshots: &[f64]) -> Measured {
        Measured { frames: frames.to_vec(), snapshots: snapshots.to_vec() }
    }

    #[test]
    fn metrics_skip_the_warm_up_and_keep_each_best() {
        // 120 slow warm-up frames, then 0.1 to 1.0 ms.
        let mut frames = vec![50.0; 120];
        frames.extend((1..=10).map(|tenth| f64::from(tenth) / 10.0));
        let first = measured(&frames, &[0.4, 0.2, 0.3]);
        frames.iter_mut().skip(120).for_each(|frame| *frame *= 2.0);
        let second = measured(&frames, &[0.1, 0.9, 0.9]);
        let results = results(b"game", 12.5, &[first, second]).unwrap();
        assert_eq!((results.frames, results.warmup, results.runs), (130, 120, 2));
        assert_eq!(results.metrics["guest_median_ms"], 0.5);
        assert_eq!(results.metrics["guest_p95_ms"], 0.9);
        assert!((results.metrics["guest_mean_ms"] - 0.55).abs() < 1e-9);
        // The second run's snapshots were better.
        assert_eq!(results.metrics["snapshot_median_ms"], 0.3);
        assert_eq!(results.metrics["load_ms"], 12.5);
        assert_eq!(results.game, fingerprint(b"game"));
        assert_ne!(results.game, fingerprint(b"other game"));
        assert!(super::results(b"game", 1.0, &[measured(&[1.0; 100], &[])]).is_err());
    }

    fn bench(values: &[(&str, f64)]) -> Results {
        Results {
            format: FORMAT,
            pen: "0".into(),
            build: None,
            runtime: "r".into(),
            game: "g".into(),
            frames: 600,
            warmup: 120,
            runs: 1,
            metrics: values.iter().map(|(name, value)| ((*name).to_owned(), *value)).collect(),
        }
    }

    #[test]
    fn comparisons_flag_what_slowed_past_its_threshold() {
        let base = bench(&[("guest_median_ms", 0.30), ("guest_p95_ms", 0.40), ("load_ms", 100.0), ("gone", 1.0)]);
        let new = bench(&[("guest_median_ms", 0.34), ("guest_p95_ms", 0.47), ("load_ms", 80.0), ("new", 1.0)]);
        let changes = compare(&base, &new, None);
        let names: Vec<&str> = changes.iter().map(|change| change.metric.as_str()).collect();
        assert_eq!(names, ["guest_median_ms", "guest_p95_ms", "load_ms"]);
        let regressed: Vec<&str> =
            changes.iter().filter(|change| change.regressed()).map(|change| change.metric.as_str()).collect();
        // +13% against 10%; +17.5% against 20%; faster.
        assert_eq!(regressed, ["guest_median_ms"]);
        assert!((changes[0].percent() - 13.333).abs() < 0.01);
        // One threshold for all, when given.
        let strict = compare(&base, &new, Some(5.0));
        assert_eq!(strict.iter().filter(|change| change.regressed()).count(), 2);
        // Within noise, however large in percent.
        let tiny = compare(&bench(&[("guest_median_ms", 0.001)]), &bench(&[("guest_median_ms", 0.004)]), None);
        assert!(!tiny[0].regressed());
    }

    #[test]
    fn results_read_back_from_json_and_checks_fail_on_regressions() {
        let directory = std::env::temp_dir().join(format!("pen-bench-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let base = directory.join("base.json");
        std::fs::write(&base, serde_json::to_string_pretty(&bench(&[("guest_median_ms", 0.30)])).unwrap()).unwrap();
        assert_eq!(read(&base).unwrap(), bench(&[("guest_median_ms", 0.30)]));
        let bases = [base.clone()];
        assert!(check(&bases, &bench(&[("guest_median_ms", 0.31)]), None).is_ok());
        let error = check(&bases, &bench(&[("guest_median_ms", 0.36)]), None).unwrap_err().to_string();
        assert_eq!(error, "bench regressed beyond its thresholds: guest_median_ms +20.0% (0.3000 -> 0.3600 ms)");
        // Against the best of several, metric by metric.
        let other = directory.join("other.json");
        let slow = bench(&[("guest_median_ms", 0.50), ("guest_p95_ms", 0.20)]);
        std::fs::write(&other, serde_json::to_string(&slow).unwrap()).unwrap();
        let best = read_best(&[base.clone(), other]).unwrap();
        assert_eq!(best.metrics, bench(&[("guest_median_ms", 0.30), ("guest_p95_ms", 0.20)]).metrics);
        let mut future = bench(&[]);
        future.format = 9;
        std::fs::write(&base, serde_json::to_string(&future).unwrap()).unwrap();
        assert!(read(&base).unwrap_err().to_string().contains("format 9"));
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
