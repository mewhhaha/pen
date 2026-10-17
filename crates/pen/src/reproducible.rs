//! Reproducible packages: `pen build` gives the same bytes from the same
//! checkout, runtime and options (`scripts/reproducible.sh` builds twice
//! and compares). The files hold nothing of when or where they were built;
//! their modification times do, so with SOURCE_DATE_EPOCH set (the
//! reproducible-builds.org convention: seconds since 1970, a release's
//! commit time, say) every file and directory of a package takes that time
//! too, and an archive of the package is the same bytes as well.
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

/// SOURCE_DATE_EPOCH's time, if it is set.
fn source_date() -> Result<Option<SystemTime>> {
    match std::env::var("SOURCE_DATE_EPOCH") {
        Ok(text) => {
            let seconds: u64 = text.trim().parse().with_context(|| format!("SOURCE_DATE_EPOCH={text}: not seconds"))?;
            Ok(Some(UNIX_EPOCH + Duration::from_secs(seconds)))
        }
        Err(_) => Ok(None),
    }
}

/// Gives everything under `directory`, and itself, SOURCE_DATE_EPOCH's
/// modification time, when it is set.
pub fn settle_times(directory: &Path) -> Result<()> {
    match source_date()? {
        Some(time) => set_times(directory, time),
        None => Ok(()),
    }
}

fn set_times(path: &Path, time: SystemTime) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.is_dir() {
        for entry in std::fs::read_dir(path)? {
            set_times(&entry?.path(), time)?;
        }
        // Windows opens no directory as a file; its packages' archives
        // take the files' times.
        if cfg!(windows) {
            return Ok(());
        }
    } else if !metadata.is_file() {
        return Ok(());
    }
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    file.set_modified(time).with_context(|| format!("setting the time of {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_file_and_directory_takes_the_time() {
        let root = crate::project::temporary_dir().unwrap().join(format!("reproducible-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("game/assets")).unwrap();
        std::fs::write(root.join("game/assets/a.wgsl"), "a").unwrap();
        std::fs::write(root.join("gdev"), "b").unwrap();
        let time = UNIX_EPOCH + Duration::from_secs(1_760_000_000);
        set_times(&root, time).unwrap();
        let modified = |path: &str| std::fs::metadata(root.join(path)).unwrap().modified().unwrap();
        for path in ["game/assets/a.wgsl", "gdev"] {
            assert_eq!(modified(path), time, "{path}");
        }
        #[cfg(unix)]
        for path in ["", "game", "game/assets"] {
            assert_eq!(modified(path), time, "{path}");
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
